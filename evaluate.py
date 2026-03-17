#!/usr/bin/env python3
"""evaluate.py - Three-level generalization evaluation for RoboBrain-3DGS.

Evaluation levels:

  Level 1 - Seen Task (episode split)
    Train episodes 0-79, test episodes 80-99 of the SAME tasks.
    Tests spatial generalization (same task, different object placement).

  Level 2 - Unseen Task (task holdout)
    Entire tasks held out from training.
    Tests semantic transfer (novel manipulation instructions).

  Level 3 - Cross Camera (view transfer)
    Train on front camera, test on left_shoulder / right_shoulder / wrist.
    Tests viewpoint invariance - the 3D branch's core advantage.

Model configurations compared at each level:

  #  Config                 2D ViT  3D Branch  LoRA
  1  Baseline (RGB+text)      Y        -        -
  2  Baseline (text-only)     -        -        -
  3  Trained  (no 3D)         -        -        Y
  4  Trained  + 3D            -        Y        Y

Metrics:
  affordance_l2      L2 on [u,v] coordinates              (lower better)
  gripper_width_mae  MAE on gripper width                  (lower better)
  approach_cos_sim   Cosine similarity of approach vector  (higher better)
  valid_format_pct   % outputs matching expected format    (higher better)
  lm_loss / ppl      Cross-entropy / perplexity on GT      (lower better)

Usage:
    # Full 3-level evaluation
    python evaluate.py \\
        --eval_config config/eval_split.yaml \\
        --checkpoint  outputs/lora/best \\
        --mode lora

    # Quick: only level 1 (seen-task), 50 samples
    python evaluate.py \\
        --eval_config config/eval_split.yaml \\
        --checkpoint  outputs/lora/best \\
        --only_level 1 --num_samples 50

    # Backward-compatible (no split, single dataset like before)
    python evaluate.py \\
        --data_root data/rlbench_sample \\
        --checkpoint outputs/lora/best
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.robobrain_vlm import RoboBrain3DGS_VLM
from data.rlbench_loader import RLBenchDataset
from train import collate_fn
from utils.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TASK_TYPE,
    build_chat_inputs,
    build_messages,
    format_inference_prompt,
    parse_affordance_output,
)


# ---------------------------------------------------------------------------
# Output parsing (delegates to shared utility)
# ---------------------------------------------------------------------------

def parse_output(text: str) -> dict:
    """Extract structured fields from model output string."""
    parsed = parse_affordance_output(text)
    # Convert approach list to ndarray for metric computation
    if parsed["approach"] is not None:
        parsed["approach"] = np.array(parsed["approach"])
    return parsed


def aggregate_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    """Compute spatial-accuracy metrics from prediction/GT string lists."""
    aff_l2, wid_mae, app_cos = [], [], []
    valid = 0
    for pred, gt in zip(predictions, ground_truths):
        p = parse_output(pred)
        g = parse_output(gt)
        if p["u"] is not None:
            valid += 1
            if g["u"] is not None:
                aff_l2.append(math.sqrt((p["u"] - g["u"])**2 + (p["v"] - g["v"])**2))
        if p["gripper_width"] is not None and g["gripper_width"] is not None:
            wid_mae.append(abs(p["gripper_width"] - g["gripper_width"]))
        if p["approach"] is not None and g["approach"] is not None:
            np_p = np.linalg.norm(p["approach"])
            np_g = np.linalg.norm(g["approach"])
            if np_p > 1e-8 and np_g > 1e-8:
                app_cos.append(float(np.dot(p["approach"], g["approach"]) / (np_p * np_g)))
    N = len(predictions)
    return {
        "n_samples": N,
        "valid_format_pct": 100.0 * valid / max(N, 1),
        "affordance_l2": float(np.mean(aff_l2)) if aff_l2 else float("nan"),
        "gripper_width_mae": float(np.mean(wid_mae)) if wid_mae else float("nan"),
        "approach_cos_sim": float(np.mean(app_cos)) if app_cos else float("nan"),
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def tensor_to_pil(rgb_tensor: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] float [0,1] tensor to PIL Image."""
    arr = (rgb_tensor.cpu().float().clamp(0, 1) * 255).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, checkpoint_path, mode, model_cfg):
    """Load RoboBrain3DGS_VLM, optionally restore checkpoint weights."""
    print(f"\n  Loading base model from {model_path} ...")
    model = RoboBrain3DGS_VLM.from_pretrained(
        model_path=model_path,
        num_gaussians=model_cfg["num_gaussians"],
        sh_degree=model_cfg["sh_degree"],
        num_gs_tokens=model_cfg["num_gs_tokens"],
        gs_encoder_dim=model_cfg["gs_encoder_dim"],
        freeze_vision_encoder=True,
        freeze_llm=(mode == "lora"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if checkpoint_path is None:
        print("    No checkpoint - original pretrained weights.")
        return model

    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.exists():
        print(f"    WARNING: checkpoint not found at {checkpoint_path}")
        return model
    print(f"    Checkpoint: {ckpt_dir}")
    meta_path = ckpt_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"    step={meta['step']}, loss={meta['loss']:.4f}, mode={meta['mode']}")

    # 3D branch
    gs_path = ckpt_dir / "3d_branch.pt"
    if gs_path.exists():
        gs_state = torch.load(gs_path, map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        loaded = sum(1 for n, t in gs_state.items()
                     if n in params and params[n].data.copy_(t.to(dtype=params[n].dtype)) is not None)
        print(f"    3D branch: {loaded}/{len(gs_state)} tensors")

    # VLM weights
    if mode == "lora":
        adapter_dir = ckpt_dir / "lora_adapter"
        if adapter_dir.exists():
            from peft import PeftModel
            model.vlm = PeftModel.from_pretrained(model.vlm, str(adapter_dir), is_trainable=False)
            print(f"    LoRA adapter loaded")
    else:
        vlm_path = ckpt_dir / "vlm_trainable.pt"
        if vlm_path.exists():
            vlm_state = torch.load(vlm_path, map_location="cpu", weights_only=True)
            params = dict(model.named_parameters())
            loaded = sum(1 for n, t in vlm_state.items()
                         if n in params and params[n].data.copy_(t.to(dtype=params[n].dtype)) is not None)
            print(f"    VLM params: {loaded}/{len(vlm_state)} tensors")
    return model


# ---------------------------------------------------------------------------
# Inference - native VLM (baseline with RGB via ViT)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_native_vlm(model, batch, processor, device, max_new_tokens=64):
    """Generate using native Qwen3-VL pipeline (RGB via ViT) with chat template."""
    inner = model.module if hasattr(model, "module") else model
    vlm = inner.vlm
    task_types = batch.get("task_types", [DEFAULT_TASK_TYPE] * len(batch["prompts"]))
    results = []
    for i, prompt in enumerate(batch["prompts"]):
        pil_img = tensor_to_pil(batch["rgb"][i])
        messages = build_messages(
            prompt, None, DEFAULT_SYSTEM_PROMPT, task_types[i], image=pil_img,
        )
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        output_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        results.append(processor.decode(new_tokens, skip_special_tokens=True).strip())
    return results


@torch.no_grad()
def lm_loss_native_vlm(model, batch, processor, device):
    """LM loss using native VLM forward with RGB via ViT and chat template."""
    inner = model.module if hasattr(model, "module") else model
    vlm = inner.vlm
    task_types = batch.get("task_types", [DEFAULT_TASK_TYPE] * len(batch["prompts"]))
    total_loss, count = 0.0, 0
    for i in range(len(batch["prompts"])):
        pil_img = tensor_to_pil(batch["rgb"][i])
        prompt, target = batch["prompts"][i], batch["targets"][i]
        msgs_full = build_messages(
            prompt, target, DEFAULT_SYSTEM_PROMPT, task_types[i], image=pil_img,
        )
        text_full = processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        inputs_full = processor(text=[text_full], images=[pil_img], return_tensors="pt", padding=True)
        msgs_prompt = build_messages(
            prompt, None, DEFAULT_SYSTEM_PROMPT, task_types[i], image=pil_img,
        )
        text_prompt = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        inputs_prompt = processor(text=[text_prompt], images=[pil_img], return_tensors="pt", padding=True)
        full_ids = inputs_full["input_ids"].to(device)
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100
        out = vlm(
            input_ids=full_ids,
            attention_mask=inputs_full["attention_mask"].to(device),
            pixel_values=inputs_full["pixel_values"].to(device),
            image_grid_thw=inputs_full["image_grid_thw"].to(device),
            labels=labels,
        )
        if out.loss is not None:
            total_loss += out.loss.item()
            count += 1
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Inference - text-only / 3D token pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch(model, batch, tokenizer, device, use_3d, max_new_tokens=64):
    """Generate with generate_with_3d using chat template (text-only or with 3D tokens)."""
    inner = model.module if hasattr(model, "module") else model
    dtype = next(inner.parameters()).dtype
    task_types = batch.get("task_types", [DEFAULT_TASK_TYPE] * len(batch["prompts"]))
    # Build chat-formatted prompts — batch tokenize
    prompt_texts = []
    for prompt, task_type in zip(batch["prompts"], task_types):
        text, _ = format_inference_prompt(prompt, tokenizer, DEFAULT_SYSTEM_PROMPT, task_type)
        prompt_texts.append(text)
    enc = tokenizer(
        prompt_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=512,
    )
    padded_ids = enc.input_ids.to(device)
    padded_masks = enc.attention_mask.to(device)
    if use_3d:
        rgb = batch["rgb"].to(device=device, dtype=dtype)
        depth = batch["depth"].to(device=device, dtype=dtype)
        intrinsics = batch["intrinsics"].to(device=device, dtype=dtype)
    else:
        rgb = depth = intrinsics = None
    generated_ids = inner.generate_with_3d(
        input_ids=padded_ids, attention_mask=padded_masks,
        depth=depth, intrinsics=intrinsics, rgb_for_3d=rgb,
        max_new_tokens=max_new_tokens, do_sample=False,
    )
    prompt_len = padded_ids.shape[1]
    return [tokenizer.decode(generated_ids[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(len(batch["prompts"]))]


@torch.no_grad()
def lm_loss_batch(model, batch, tokenizer, device, use_3d):
    """LM loss using 3DGS wrapper forward with chat template."""
    inner = model.module if hasattr(model, "module") else model
    dtype = next(inner.parameters()).dtype
    rgb = batch["rgb"].to(device=device, dtype=dtype)
    depth = batch["depth"].to(device=device, dtype=dtype) if use_3d else None
    intrinsics = batch["intrinsics"].to(device=device, dtype=dtype) if use_3d else None
    task_types = batch.get("task_types")
    input_ids, attention_mask, labels = build_chat_inputs(
        batch["prompts"], batch["targets"], tokenizer, device,
        task_types=task_types)
    out = inner(input_ids=input_ids, attention_mask=attention_mask,
                depth=depth, intrinsics=intrinsics,
                rgb_for_3d=rgb if use_3d else None, labels=labels)
    loss = out.get("loss")
    return float(loss.item()) if loss is not None else float("nan")


# ---------------------------------------------------------------------------
# Generic evaluation loop
# ---------------------------------------------------------------------------

def _run_loop(model, dataloader, max_samples, label, gen_fn, loss_fn):
    """Evaluate one (model, config) on a dataloader."""
    model.train(False)
    all_preds, all_gts, all_losses = [], [], []
    n = 0
    for batch in dataloader:
        if 0 < max_samples <= n:
            break
        preds = gen_fn(batch)
        loss = loss_fn(batch)
        all_preds.extend(preds)
        all_gts.extend(batch["targets"])
        all_losses.append(loss)
        n += len(preds)
        short = preds[-1][:60] if preds else ""
        print(f"      {n:>4} samples  loss={loss:.4f}  {short!r}", end="\r", flush=True)
    print(f"      {n:>4} samples done{' ' * 50}")
    metrics = aggregate_metrics(all_preds, all_gts)
    valid_losses = [x for x in all_losses if not math.isnan(x)]
    mean_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
    metrics["lm_loss"] = mean_loss
    metrics["perplexity"] = float(math.exp(mean_loss)) if not math.isnan(mean_loss) else float("nan")
    return {"label": label, "metrics": metrics, "predictions": all_preds, "ground_truths": all_gts}


# ---------------------------------------------------------------------------
# Run all 4 model configs on one dataset
# ---------------------------------------------------------------------------

def evaluate_all_configs(
    model_path, checkpoint, mode, model_cfg,
    dataloader, tokenizer, processor, device,
    max_samples, max_new_tokens,
    level_label: str,
    skip_rgb_baseline: bool = False,
):
    """Run all 4 model configs on a single evaluation dataset.

    Returns list of result dicts (one per config).
    """
    print(f"\n{'=' * 60}")
    print(f"  {level_label}")
    print(f"{'=' * 60}")

    results = []

    # --- Baseline (load original model) ---
    baseline = load_model(model_path, None, mode, model_cfg)

    if not skip_rgb_baseline:
        r = _run_loop(baseline, dataloader, max_samples,
                       "Baseline (RGB+text)",
                       lambda b: generate_native_vlm(baseline, b, processor, device, max_new_tokens),
                       lambda b: lm_loss_native_vlm(baseline, b, processor, device))
        results.append(r)

    r = _run_loop(baseline, dataloader, max_samples,
                   "Baseline (text-only)",
                   lambda b: generate_batch(baseline, b, tokenizer, device, False, max_new_tokens),
                   lambda b: lm_loss_batch(baseline, b, tokenizer, device, False))
    results.append(r)
    del baseline
    torch.cuda.empty_cache()

    # --- Trained model (from checkpoint) ---
    if checkpoint is not None:
        trained = load_model(model_path, checkpoint, mode, model_cfg)

        r = _run_loop(trained, dataloader, max_samples,
                       "Trained (no 3D)",
                       lambda b: generate_batch(trained, b, tokenizer, device, False, max_new_tokens),
                       lambda b: lm_loss_batch(trained, b, tokenizer, device, False))
        results.append(r)

        r = _run_loop(trained, dataloader, max_samples,
                       "Trained + 3D (ours)",
                       lambda b: generate_batch(trained, b, tokenizer, device, True, max_new_tokens),
                       lambda b: lm_loss_batch(trained, b, tokenizer, device, True))
        results.append(r)
        del trained
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_DEFS = [
    ("Model",          None,                32, "s"),
    ("Valid%",         "valid_format_pct",   7, ".1f"),
    ("Aff L2",         "affordance_l2",      9, ".4f"),
    ("Width MAE",      "gripper_width_mae", 10, ".4f"),
    ("Approach cos",   "approach_cos_sim",  13, ".4f"),
    ("PPL",            "perplexity",        10, ".2f"),
    ("LM Loss",        "lm_loss",            9, ".4f"),
    ("N",              "n_samples",          5, "d"),
]

def _fmt(v, fmt):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return format(v, fmt)

def print_table(title: str, results: list[dict]):
    """Print a formatted comparison table."""
    if not results:
        return
    widths = [max(len(h), w) for h, _, w, _ in _COL_DEFS]
    header = "  ".join(f"{h:>{widths[i]}}" for i, (h, _, _, _) in enumerate(_COL_DEFS))
    sep = "  ".join("-" * w for w in widths)
    bar = "=" * len(header)

    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(header)
    print(sep)
    for r in results:
        m = r["metrics"]
        vals = []
        for i, (_, key, _, fmt) in enumerate(_COL_DEFS):
            if key is None:
                vals.append(f"{r['label']:>{widths[i]}}")
            else:
                vals.append(f"{_fmt(m.get(key), fmt):>{widths[i]}}")
        print("  ".join(vals))
    print(bar)

    # Delta vs first result
    if len(results) > 1:
        bm = results[0]["metrics"]
        for r in results[1:]:
            m = r["metrics"]
            parts = []
            for _, key, _, _ in _COL_DEFS[1:-1]:  # skip label and N
                bv, mv = bm.get(key, float("nan")), m.get(key, float("nan"))
                if isinstance(bv, float) and isinstance(mv, float) and not (math.isnan(bv) or math.isnan(mv)):
                    d = mv - bv
                    pct = f"({d / abs(bv) * 100:+.0f}%)" if abs(bv) > 1e-9 else ""
                    parts.append(f"{key.split('_')[0][:7]}={d:+.3f}{pct}")
            if parts:
                print(f"    vs baseline: {r['label']:<28} " + "  ".join(parts))
    print()


def print_qualitative(results: list[dict], n: int = 3):
    if not results or not results[0]["ground_truths"]:
        return
    gts = results[0]["ground_truths"]
    for i in range(min(n, len(gts))):
        print(f"  Sample {i+1}:")
        print(f"    GT: {gts[i]}")
        for r in results:
            pred = r["predictions"][i] if i < len(r["predictions"]) else ""
            p, g = parse_output(pred), parse_output(gts[i])
            err = ""
            if p["u"] is not None and g["u"] is not None:
                err = f"  [L2={math.sqrt((p['u']-g['u'])**2+(p['v']-g['v'])**2):.4f}]"
            print(f"    {r['label'][:28]:<28}: {pred[:80]}{err}")
    print()


# ---------------------------------------------------------------------------
# Dataset builders for each level
# ---------------------------------------------------------------------------

def _build_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=0)


def build_level1_dataset(eval_cfg, image_size):
    """Level 1: Seen-task, unseen episodes (test split of training tasks)."""
    held_out = set(eval_cfg.get("held_out_tasks", []))
    ds = RLBenchDataset(
        root_dir=eval_cfg["data_root"],
        camera=eval_cfg.get("train_camera", "front"),
        image_size=image_size,
        split="test",
        train_ratio=eval_cfg.get("train_episode_ratio", 0.8),
        seed=eval_cfg.get("seed", 42),
        task_exclude=list(held_out) if held_out else None,
        max_frames_per_episode=eval_cfg.get("max_frames_per_episode", -1),
    )
    return ds


def build_level2_dataset(eval_cfg, image_size):
    """Level 2: Unseen tasks (all episodes of held-out tasks)."""
    held_out = eval_cfg.get("held_out_tasks", [])
    if not held_out:
        return None
    ds = RLBenchDataset(
        root_dir=eval_cfg["data_root"],
        camera=eval_cfg.get("train_camera", "front"),
        image_size=image_size,
        task_filter=held_out,
        max_frames_per_episode=eval_cfg.get("max_frames_per_episode", -1),
    )
    return ds


def build_level3_datasets(eval_cfg, image_size):
    """Level 3: Cross-camera. Returns {camera_name: dataset}."""
    held_out = set(eval_cfg.get("held_out_tasks", []))
    test_cameras = eval_cfg.get("test_cameras", [])
    train_camera = eval_cfg.get("train_camera", "front")
    datasets = {}
    for cam in test_cameras:
        if cam == train_camera:
            continue  # same as Level 1, skip
        ds = RLBenchDataset(
            root_dir=eval_cfg["data_root"],
            camera=cam,
            image_size=image_size,
            split="test",
            train_ratio=eval_cfg.get("train_episode_ratio", 0.8),
            seed=eval_cfg.get("seed", 42),
            task_exclude=list(held_out) if held_out else None,
            max_frames_per_episode=eval_cfg.get("max_frames_per_episode", -1),
        )
        if len(ds) > 0:
            datasets[cam] = ds
    return datasets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-level evaluation: RoboBrain-3DGS vs baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_config", default=None,
                   help="Path to eval_split.yaml (enables 3-level evaluation)")
    p.add_argument("--data_root", default="data/rlbench_sample",
                   help="Fallback data root when --eval_config not given")
    p.add_argument("--model_path", default="/home/w50037733/models/RoboBrain2.5-8B-NV")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mode", default="lora", choices=["lora", "full"])
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=-1,
                   help="Max samples per level (-1 = all)")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_results", default=None)
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--skip_rgb_baseline", action="store_true")
    p.add_argument("--only_level", type=int, default=None, choices=[1, 2, 3],
                   help="Run only this evaluation level")
    p.add_argument("--num_gaussians", type=int, default=1024)
    p.add_argument("--sh_degree", type=int, default=2)
    p.add_argument("--num_gs_tokens", type=int, default=64)
    p.add_argument("--gs_encoder_dim", type=int, default=512)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    model_cfg = {
        "num_gaussians": args.num_gaussians, "sh_degree": args.sh_degree,
        "num_gs_tokens": args.num_gs_tokens, "gs_encoder_dim": args.gs_encoder_dim,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(args.model_path)

    ckpt = args.checkpoint
    if ckpt and not Path(ckpt).is_absolute():
        ckpt = str(Path(__file__).parent / ckpt)

    all_level_results = {}

    # =======================================================================
    # Mode A: Three-level evaluation (--eval_config provided)
    # =======================================================================
    if args.eval_config:
        with open(args.eval_config) as f:
            eval_cfg = yaml.safe_load(f)
        print(f"\nEval config: {args.eval_config}")
        print(f"  data_root:      {eval_cfg['data_root']}")
        print(f"  held_out_tasks: {eval_cfg.get('held_out_tasks', [])}")
        print(f"  train_camera:   {eval_cfg.get('train_camera', 'front')}")
        print(f"  test_cameras:   {eval_cfg.get('test_cameras', [])}")

        run1 = eval_cfg.get("run_level1_seen_task", True)
        run2 = eval_cfg.get("run_level2_unseen_task", True)
        run3 = eval_cfg.get("run_level3_cross_camera", True)
        if args.only_level:
            run1 = args.only_level == 1
            run2 = args.only_level == 2
            run3 = args.only_level == 3

        # --- Level 1: Seen Task, Unseen Episode ---
        if run1:
            ds = build_level1_dataset(eval_cfg, args.image_size)
            print(f"\n  Level 1 dataset: {len(ds)} samples (seen tasks, test episodes)")
            if len(ds) > 0:
                dl = _build_dataloader(ds, args.batch_size)
                results = evaluate_all_configs(
                    args.model_path, ckpt, args.mode, model_cfg,
                    dl, tokenizer, processor, args.device,
                    args.num_samples, args.max_new_tokens,
                    "LEVEL 1: Seen Task / Unseen Episode",
                    args.skip_rgb_baseline,
                )
                all_level_results["level1_seen_task"] = results
                print_table("LEVEL 1: Seen Task / Unseen Episode", results)
                print_qualitative(results, args.n_examples)

        # --- Level 2: Unseen Task ---
        if run2:
            ds = build_level2_dataset(eval_cfg, args.image_size)
            if ds and len(ds) > 0:
                print(f"\n  Level 2 dataset: {len(ds)} samples (unseen tasks)")
                dl = _build_dataloader(ds, args.batch_size)
                results = evaluate_all_configs(
                    args.model_path, ckpt, args.mode, model_cfg,
                    dl, tokenizer, processor, args.device,
                    args.num_samples, args.max_new_tokens,
                    "LEVEL 2: Unseen Task",
                    args.skip_rgb_baseline,
                )
                all_level_results["level2_unseen_task"] = results
                print_table("LEVEL 2: Unseen Task", results)
                print_qualitative(results, args.n_examples)
            else:
                print("\n  Level 2: skipped (no held_out_tasks or no data found)")

        # --- Level 3: Cross Camera ---
        if run3:
            cam_datasets = build_level3_datasets(eval_cfg, args.image_size)
            for cam, ds in cam_datasets.items():
                print(f"\n  Level 3 dataset ({cam}): {len(ds)} samples")
                dl = _build_dataloader(ds, args.batch_size)
                results = evaluate_all_configs(
                    args.model_path, ckpt, args.mode, model_cfg,
                    dl, tokenizer, processor, args.device,
                    args.num_samples, args.max_new_tokens,
                    f"LEVEL 3: Cross Camera ({cam})",
                    args.skip_rgb_baseline,
                )
                all_level_results[f"level3_{cam}"] = results
                print_table(f"LEVEL 3: Cross Camera ({cam})", results)

    # =======================================================================
    # Mode B: Simple single-dataset evaluation (backward-compatible)
    # =======================================================================
    else:
        data_root = args.data_root
        if not Path(data_root).is_absolute():
            data_root = str(Path(__file__).parent / data_root)
        print(f"\nDataset: {data_root} (no split - backward-compatible mode)")
        ds = RLBenchDataset(root_dir=data_root, camera="front",
                            image_size=args.image_size, max_frames=args.num_samples)
        print(f"  {len(ds)} samples")
        if len(ds) == 0:
            print("ERROR: no samples found.")
            return 1
        dl = _build_dataloader(ds, args.batch_size)
        results = evaluate_all_configs(
            args.model_path, ckpt, args.mode, model_cfg,
            dl, tokenizer, processor, args.device,
            args.num_samples, args.max_new_tokens,
            "Single Dataset (no split)",
            args.skip_rgb_baseline,
        )
        all_level_results["no_split"] = results
        print_table("Single Dataset (no split)", results)
        print_qualitative(results, args.n_examples)

    # =======================================================================
    # Save
    # =======================================================================
    if args.save_results:
        save_data = {}
        for level_key, results in all_level_results.items():
            save_data[level_key] = [
                {"label": r["label"], "metrics": r["metrics"],
                 "samples": [{"prediction": p, "ground_truth": g}
                             for p, g in zip(r["predictions"], r["ground_truths"])]}
                for r in results
            ]
        with open(args.save_results, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.save_results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
