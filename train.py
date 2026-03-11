"""RoboBrain-3DGS Training Script.

Unified training interface supporting both LoRA and full fine-tuning.

Modes:
    LoRA:  Freeze LLM, inject low-rank adapters (q/k/v/o), train 3D branch + LoRA.
           Memory: ~20GB per GPU with ZeRO-2. Fast convergence.
    Full:  Unfreeze LLM (all or last N layers), train everything.
           Memory: ~30GB per GPU with ZeRO-3 + optimizer offload. Maximum capacity.

In both modes, ViT (2D vision encoder) stays frozen.

Losses:
    L_total = L_lm (language modeling) + lambda * L_render (3DGS reconstruction)

Usage:
    # LoRA fine-tuning (single GPU)
    python train.py --config config/train_lora.yaml

    # LoRA fine-tuning (multi-GPU)
    deepspeed --num_gpus=4 train.py --config config/train_lora.yaml

    # Full fine-tuning (multi-GPU, requires ZeRO-3)
    deepspeed --num_gpus=4 train.py --config config/train_full.yaml

    # Override mode via CLI
    python train.py --config config/train_lora.yaml --finetune_mode full

    # Dry run (1 step, validates pipeline)
    python train.py --config config/train_lora.yaml --dry_run

    # Resume from checkpoint
    python train.py --config config/train_lora.yaml --resume outputs/lora/checkpoint-500
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.robobrain_vlm import RoboBrain3DGS_VLM
from models.gs_renderer import GaussianRenderingLoss
from data.rlbench_loader import RLBenchDataset
from data.droid_loader import DROIDDataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RoboBrain-3DGS Training")
    parser.add_argument("--config", type=str, default="config/train_lora.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--finetune_mode", type=str, default=None,
                        choices=["lora", "full"],
                        help="Override finetune mode from config")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config JSON (overrides config file)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by launcher)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 1 batch only (pipeline validation)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    return local_rank


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    return {
        "rgb": torch.stack([s["rgb"] for s in batch]),
        "depth": torch.stack([s["depth"] for s in batch]),
        "intrinsics": torch.stack([s["intrinsics"] for s in batch]),
        "prompts": [s["prompt"] for s in batch],
    }


def build_datasets(cfg: dict):
    data_cfg = cfg["data"]
    datasets = []

    if data_cfg.get("rlbench_root") and os.path.exists(data_cfg["rlbench_root"]):
        ds = RLBenchDataset(
            root_dir=data_cfg["rlbench_root"],
            camera=data_cfg.get("camera", "front"),
            image_size=data_cfg.get("image_size", 256),
            max_frames=data_cfg.get("max_frames", -1),
        )
        print(f"  RLBench: {len(ds)} samples")
        datasets.append(ds)

    if data_cfg.get("droid_root") and os.path.exists(data_cfg["droid_root"]):
        ds = DROIDDataset(
            root_dir=data_cfg["droid_root"],
            image_size=data_cfg.get("image_size", 256),
            max_frames=data_cfg.get("max_frames", -1),
        )
        print(f"  DROID: {len(ds)} samples")
        datasets.append(ds)

    if not datasets:
        raise ValueError("No datasets found. Check data paths in config.")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total: {len(combined)} samples")
    return combined


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _is_3d_branch_param(name: str) -> bool:
    """Check if a parameter belongs to the 3D Gaussian branch."""
    return any(k in name for k in [
        "depth_to_gaussian", "gs_encoder", "gs_projector", "gs_type_embedding",
    ])


def _apply_gradient_checkpointing(model: RoboBrain3DGS_VLM):
    """Enable gradient checkpointing on the LLM to save memory."""
    lang_model = model._get_language_model()
    if hasattr(lang_model, "gradient_checkpointing_enable"):
        lang_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")
    else:
        print("  Gradient checkpointing: not supported by this model")


def _freeze_vision_encoder(model: RoboBrain3DGS_VLM):
    """Freeze the 2D ViT encoder."""
    for param in model._get_visual().parameters():
        param.requires_grad = False


def build_model_lora(cfg: dict, local_rank: int):
    """Build model with LoRA adapters on the LLM."""
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))

    # DeepSpeed: load to CPU first, let deepspeed.initialize() handle device placement.
    # Single-GPU: use device_map="auto" for automatic GPU placement.
    use_deepspeed = local_rank >= 0
    device_map = None if use_deepspeed else "auto"

    model = RoboBrain3DGS_VLM.from_pretrained(
        model_path=model_cfg["base_model"],
        num_gaussians=model_cfg.get("num_gaussians", 1024),
        sh_degree=model_cfg.get("sh_degree", 2),
        num_gs_tokens=model_cfg.get("num_gs_tokens", 64),
        gs_encoder_dim=model_cfg.get("gs_encoder_dim", 512),
        freeze_vision_encoder=model_cfg.get("freeze_vision_encoder", True),
        freeze_llm=True,  # Freeze LLM first, then inject LoRA
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Apply LoRA (on CPU if DeepSpeed, on GPU otherwise)
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model.vlm = get_peft_model(model.vlm, lora_config)
    model.vlm.print_trainable_parameters()

    return model


def build_model_full(cfg: dict, local_rank: int):
    """Build model for full fine-tuning (no LoRA)."""
    model_cfg = cfg["model"]
    full_cfg = cfg.get("full_finetune", {})
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))

    use_deepspeed = local_rank >= 0
    device_map = None if use_deepspeed else "auto"

    model = RoboBrain3DGS_VLM.from_pretrained(
        model_path=model_cfg["base_model"],
        num_gaussians=model_cfg.get("num_gaussians", 1024),
        sh_degree=model_cfg.get("sh_degree", 2),
        num_gs_tokens=model_cfg.get("num_gs_tokens", 64),
        gs_encoder_dim=model_cfg.get("gs_encoder_dim", 512),
        freeze_vision_encoder=model_cfg.get("freeze_vision_encoder", True),
        freeze_llm=False,  # LLM unfrozen for full fine-tuning
        torch_dtype=dtype,
        device_map=device_map,
    )

    # Partial layer unfreezing (optional: freeze first N layers to save memory)
    unfreeze_layers = full_cfg.get("unfreeze_llm_layers", -1)
    if unfreeze_layers > 0:
        lang_model = model._get_language_model()
        layers = lang_model.layers if hasattr(lang_model, "layers") else lang_model.model.layers
        total_layers = len(layers)
        freeze_count = total_layers - unfreeze_layers

        for i, layer in enumerate(layers):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False

        print(f"  LLM layers: {total_layers} total, {freeze_count} frozen, {unfreeze_layers} trainable")

    # LM head
    if not full_cfg.get("unfreeze_lm_head", True):
        for param in model._get_lm_head().parameters():
            param.requires_grad = False
        print("  LM head: frozen")

    # Gradient checkpointing
    if full_cfg.get("gradient_checkpointing", True):
        _apply_gradient_checkpointing(model)

    return model


def _needs_cpu_adam(cfg: dict) -> bool:
    """Check if DeepSpeed CPU offload is enabled, requiring DeepSpeedCPUAdam."""
    ds_cfg_path = cfg.get("deepspeed", {}).get("config")
    if not ds_cfg_path or not isinstance(ds_cfg_path, str):
        return False
    try:
        import json
        with open(ds_cfg_path) as f:
            ds_cfg = json.load(f)
        offload = ds_cfg.get("zero_optimization", {}).get("offload_optimizer", {})
        return offload.get("device", "none") == "cpu"
    except Exception:
        return False


def build_optimizer(model: nn.Module, cfg: dict, mode: str):
    """Build optimizer with per-group learning rates.

    Groups:
        3d_branch: DepthToGaussian + GS Encoder + Projector (always highest LR)
        lora:      LoRA adapters (mode=lora only)
        llm:       Full LLM parameters (mode=full only)

    Uses DeepSpeedCPUAdam when ZeRO-3 CPU offload is enabled
    (required by DeepSpeed for efficient CPU-side optimizer steps).
    """
    train_cfg = cfg["training"]

    # Collect trainable parameters by group
    params_3d = []
    params_lora = []
    params_llm = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_3d_branch_param(name):
            params_3d.append(param)
        elif "lora" in name.lower():
            params_lora.append(param)
        else:
            params_llm.append(param)

    groups = []
    if params_3d:
        groups.append({
            "params": params_3d,
            "lr": train_cfg.get("lr_3d_branch", 1e-3),
            "name": "3d_branch",
        })
    if params_lora:
        groups.append({
            "params": params_lora,
            "lr": train_cfg.get("learning_rate", 2e-4),
            "name": "lora",
        })
    if params_llm:
        groups.append({
            "params": params_llm,
            "lr": train_cfg.get("learning_rate", 5e-5),
            "name": "llm",
        })

    # Summary
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  [{g['name']}] {n / 1e6:.1f}M params, lr={g['lr']}")

    total = sum(p.numel() for g in groups for p in g["params"])
    total_all = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {total / 1e6:.1f}M / {total_all / 1e6:.1f}M ({total / total_all * 100:.2f}%)")

    # Select optimizer based on hardware and offload config
    use_cpu_adam = _needs_cpu_adam(cfg)
    use_fused = train_cfg.get("use_fused_adam", False)
    wd = train_cfg.get("weight_decay", 0.01)

    if use_cpu_adam:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(groups, weight_decay=wd)
        print(f"  Optimizer: DeepSpeedCPUAdam (CPU offload)")
    elif use_fused:
        try:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(groups, weight_decay=wd)
            print(f"  Optimizer: FusedAdam (GPU fused kernel)")
        except ImportError:
            optimizer = torch.optim.AdamW(groups, weight_decay=wd)
            print(f"  Optimizer: AdamW (FusedAdam not available, fallback)")
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=wd)
        print(f"  Optimizer: AdamW")
    return optimizer


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, train_cfg: dict, total_steps: int):
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    batch: dict,
    tokenizer,
    render_loss_fn: GaussianRenderingLoss | None,
    render_weight: float,
    device: str,
) -> dict[str, torch.Tensor]:
    """One forward + loss computation. Does NOT call backward."""
    inner = model.module if hasattr(model, "module") else model
    dtype = next(inner.parameters()).dtype

    rgb = batch["rgb"].to(device=device, dtype=dtype)
    depth = batch["depth"].to(device=device, dtype=dtype)
    intrinsics = batch["intrinsics"].to(device=device, dtype=dtype)
    prompts = batch["prompts"]

    # Tokenize
    texts = [f"Task: {p} Affordance:" for p in prompts]
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)

    # Labels (mask prompt prefix)
    labels = input_ids.clone()
    mask_len = int(input_ids.shape[1] * 0.7)
    labels[:, :mask_len] = -100

    # Forward
    outputs = inner(
        input_ids=input_ids,
        attention_mask=attention_mask,
        depth=depth,
        intrinsics=intrinsics,
        rgb_for_3d=rgb,
        labels=labels,
    )
    lm_loss = outputs["loss"]

    # Rendering loss
    render_metrics = {}
    if render_loss_fn is not None:
        gaussians = inner.depth_to_gaussian(rgb, depth, intrinsics)
        rl = render_loss_fn(gaussians, intrinsics, rgb, depth)
        render_loss = rl["loss"].to(lm_loss.device)
        render_metrics = {f"render/{k}": v.item() for k, v in rl.items() if k != "loss"}
    else:
        render_loss = torch.tensor(0.0, device=lm_loss.device)

    total_loss = lm_loss + render_weight * render_loss

    return {
        "loss": total_loss,
        "lm_loss": lm_loss.item(),
        "render_loss": render_loss.item(),
        **render_metrics,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer, scheduler, step: int,
                    save_dir: str, mode: str):
    """Save checkpoint. Strategy depends on finetune mode.

    LoRA:  Save adapter weights (~60MB) + 3D branch
    Full:  Save full model state_dict or DeepSpeed engine state
    """
    os.makedirs(save_dir, exist_ok=True)
    inner = model.module if hasattr(model, "module") else model

    # 1. Save 3D branch (always)
    gs_state = {}
    for name, param in inner.named_parameters():
        if _is_3d_branch_param(name):
            gs_state[name] = param.data.cpu()
    torch.save(gs_state, os.path.join(save_dir, "3d_branch.pt"))

    # 2. Save VLM weights (mode-dependent)
    if mode == "lora":
        # PEFT save_pretrained saves only adapter weights
        if hasattr(inner, "vlm") and hasattr(inner.vlm, "save_pretrained"):
            inner.vlm.save_pretrained(os.path.join(save_dir, "lora_adapter"))
    else:
        # Full: save all trainable VLM parameters
        vlm_state = {}
        for name, param in inner.named_parameters():
            if param.requires_grad and not _is_3d_branch_param(name):
                vlm_state[name] = param.data.cpu()
        torch.save(vlm_state, os.path.join(save_dir, "vlm_trainable.pt"))

    # 3. Training state
    train_state = {
        "step": step,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "mode": mode,
    }
    torch.save(train_state, os.path.join(save_dir, "training_state.pt"))
    print(f"  Checkpoint saved: {save_dir} (mode={mode})")


def load_checkpoint(model: nn.Module, optimizer, scheduler,
                    ckpt_dir: str, mode: str, device: str):
    """Load checkpoint and restore training state.

    Returns:
        step: the global step to resume from
    """
    inner = model.module if hasattr(model, "module") else model

    # 1. Load 3D branch
    gs_path = os.path.join(ckpt_dir, "3d_branch.pt")
    if os.path.exists(gs_path):
        gs_state = torch.load(gs_path, map_location=device, weights_only=True)
        missing, unexpected = [], []
        model_state = dict(inner.named_parameters())
        for name, tensor in gs_state.items():
            if name in model_state:
                model_state[name].data.copy_(tensor)
            else:
                unexpected.append(name)
        print(f"  3D branch loaded ({len(gs_state)} params, {len(unexpected)} unexpected)")

    # 2. Load VLM weights
    if mode == "lora":
        adapter_dir = os.path.join(ckpt_dir, "lora_adapter")
        if os.path.exists(adapter_dir) and hasattr(inner.vlm, "load_adapter"):
            inner.vlm.load_adapter(adapter_dir, adapter_name="default")
            print(f"  LoRA adapter loaded from {adapter_dir}")
    else:
        vlm_path = os.path.join(ckpt_dir, "vlm_trainable.pt")
        if os.path.exists(vlm_path):
            vlm_state = torch.load(vlm_path, map_location=device, weights_only=True)
            model_state = dict(inner.named_parameters())
            loaded = 0
            for name, tensor in vlm_state.items():
                if name in model_state:
                    model_state[name].data.copy_(tensor)
                    loaded += 1
            print(f"  VLM trainable params loaded ({loaded}/{len(vlm_state)})")

    # 3. Training state
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        step = state.get("step", 0)
        if optimizer is not None and state.get("optimizer_state"):
            try:
                optimizer.load_state_dict(state["optimizer_state"])
            except Exception as e:
                print(f"  Warning: could not load optimizer state: {e}")
        if scheduler is not None and state.get("scheduler_state"):
            try:
                scheduler.load_state_dict(state["scheduler_state"])
            except Exception as e:
                print(f"  Warning: could not load scheduler state: {e}")
        print(f"  Resuming from step {step}")

    return step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, args):
    local_rank = setup_distributed()
    is_main = local_rank <= 0

    # Resolve finetune mode (CLI overrides config)
    mode = args.finetune_mode or cfg.get("finetune_mode", "lora")
    train_cfg = cfg["training"]
    render_cfg = cfg.get("rendering_loss", {})

    if is_main:
        print("=" * 60)
        print(f"  RoboBrain-3DGS Training [{mode.upper()} fine-tuning]")
        print("=" * 60)

    # -- Data --
    if is_main:
        print("\n[Data]")
    dataset = build_datasets(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("per_device_batch_size", 1),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # -- Model --
    if is_main:
        print(f"\n[Model] mode={mode}")
    if mode == "lora":
        model = build_model_lora(cfg, local_rank)
    elif mode == "full":
        model = build_model_full(cfg, local_rank)
    else:
        raise ValueError(f"Unknown finetune_mode: {mode!r}. Use 'lora' or 'full'.")

    # -- Optimizer --
    if is_main:
        print("\n[Optimizer]")
    optimizer = build_optimizer(model, cfg, mode)

    # -- Tokenizer --
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg["model"]["base_model"])
    tokenizer = processor.tokenizer

    # -- Training config (needed before DeepSpeed init) --
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)

    # -- Scheduler (must be created before DeepSpeed wraps the optimizer) --
    total_steps = len(dataloader) * train_cfg.get("num_epochs", 3)
    scheduler = build_scheduler(optimizer, train_cfg, total_steps)

    # -- DeepSpeed --
    ds_config_path = args.deepspeed or cfg.get("deepspeed", {}).get("config")
    ds_config = None
    if ds_config_path and local_rank >= 0:
        import deepspeed
        import json

        # Load DS config and inject batch size parameters
        # ('auto' values only work with HF Trainer, not deepspeed.initialize)
        if isinstance(ds_config_path, str):
            with open(ds_config_path) as f:
                ds_config = json.load(f)
        else:
            ds_config = ds_config_path

        micro_bs = train_cfg.get("per_device_batch_size", 1)
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        ds_config["train_micro_batch_size_per_gpu"] = micro_bs
        ds_config["gradient_accumulation_steps"] = grad_accum
        ds_config["train_batch_size"] = micro_bs * world_size * grad_accum

        # Pass scheduler to DS so it manages stepping at accumulation boundaries
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
        )
        device = model.device
        if is_main:
            print(f"\n[DeepSpeed] Initialized on {world_size} GPUs")
            print(f"  ZeRO stage: {ds_config.get('zero_optimization', {}).get('stage', 0)}")
            print(f"  Micro batch: {micro_bs}, Grad accum: {grad_accum}, Global batch: {ds_config['train_batch_size']}")
    else:
        inner = model.module if hasattr(model, "module") else model
        device = next(inner.parameters()).device

    # -- Rendering loss (after DeepSpeed init so model is on correct device) --
    render_loss_fn = None
    if render_cfg.get("enabled", False):
        render_size = render_cfg.get("render_size", 64)
        render_loss_fn = GaussianRenderingLoss(
            lambda_l1=render_cfg.get("lambda_l1", 0.8),
            lambda_ssim=render_cfg.get("lambda_ssim", 0.2),
            lambda_depth=render_cfg.get("lambda_depth", 0.5),
            lambda_opacity=render_cfg.get("lambda_opacity", 0.01),
            image_size=(render_size, render_size),
        )
        render_loss_fn = render_loss_fn.to(device)
        if is_main:
            print(f"\n[Rendering Loss] {render_size}x{render_size}, weight={render_cfg.get('weight', 0.1)}")
    render_weight = render_cfg.get("weight", 0.1) if render_loss_fn else 0.0

    # -- Resume --
    start_step = 0
    if args.resume:
        if is_main:
            print(f"\n[Resume] {args.resume}")
        start_step = load_checkpoint(model, optimizer, scheduler, args.resume, mode, str(device))

    # -- Training config (continued) --
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    logging_steps = train_cfg.get("logging_steps", 10)
    save_steps = train_cfg.get("save_steps", 500)
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    if is_main:
        print(f"\n[Training]")
        print(f"  Mode:       {mode}")
        print(f"  Epochs:     {train_cfg.get('num_epochs', 3)}")
        print(f"  Batch:      {train_cfg.get('per_device_batch_size', 1)} x {grad_accum} accum")
        print(f"  Steps:      {total_steps} (resume from {start_step})")
        print(f"  Output:     {output_dir}")
        print(f"  DeepSpeed:  {'enabled' if (ds_config and local_rank >= 0) else 'disabled'}")

    # -- Loop --
    global_step = start_step
    best_loss = float("inf")
    model.train()

    for epoch in range(train_cfg.get("num_epochs", 3)):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for step, batch in enumerate(dataloader):
            # Skip already-processed steps when resuming
            abs_step = epoch * len(dataloader) + step
            if abs_step < start_step:
                continue

            t_step = time.time()

            metrics = train_step(
                model, batch, tokenizer,
                render_loss_fn, render_weight, str(device),
            )
            loss = metrics["loss"]

            # Backward + optimize
            if ds_config and local_rank >= 0:
                model.backward(loss)
                model.step()
                # Scheduler is managed by DeepSpeed (passed via lr_scheduler=)
                # It auto-steps at gradient accumulation boundaries
            else:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            global_step += 1
            epoch_loss += metrics["lm_loss"]
            epoch_steps += 1

            # Log
            if is_main and global_step % logging_steps == 0:
                dt = time.time() - t_step
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{mode}] Epoch {epoch + 1} | Step {global_step}/{total_steps} | "
                    f"lm={metrics['lm_loss']:.4f} render={metrics['render_loss']:.4f} "
                    f"lr={lr:.2e} time={dt:.1f}s"
                )

            # Save
            if is_main and save_steps > 0 and global_step % save_steps == 0:
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                save_checkpoint(model, optimizer, scheduler, global_step, ckpt_dir, mode)

            # Dry run
            if args.dry_run:
                if is_main:
                    print(f"\n  [Dry run] 1 step OK. lm_loss={metrics['lm_loss']:.4f}")
                return metrics

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        if is_main:
            dt = time.time() - t_epoch
            print(f"\n  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, time={dt:.1f}s")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, scheduler, global_step,
                                os.path.join(output_dir, "best"), mode)
                print(f"  Best model saved (loss={best_loss:.4f})")

    # Final
    if is_main:
        save_checkpoint(model, optimizer, scheduler, global_step,
                        os.path.join(output_dir, "final"), mode)
        print(f"\nTraining complete [{mode}]. Best loss: {best_loss:.4f}")
        print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, args)
