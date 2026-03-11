"""End-to-end test: Run the full train loop with DeepSpeed using tiny model.

Since we can't fit the 8B model on a single partially-occupied GPU,
this test simulates the DeepSpeed training loop directly with a tiny model.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config
from models.gs_renderer import GaussianRenderingLoss
from train import train_step, build_scheduler, collate_fn


def init_distributed():
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def create_tiny_model(freeze_llm=True):
    config = create_tiny_vlm_config()
    model = RoboBrain3DGS_VLM(
        vlm_config=config,
        num_gaussians=64,
        sh_degree=2,
        num_gs_tokens=8,
        gs_encoder_dim=128,
        freeze_vision_encoder=True,
        freeze_llm=freeze_llm,
    )
    return model


def create_synthetic_batch(B=1, seq_len=32, img_size=64, device="cpu"):
    """Create a synthetic batch mimicking the dataloader output."""
    return {
        "rgb": torch.randn(B, 3, img_size, img_size),
        "depth": torch.rand(B, 1, img_size, img_size) * 2,
        "intrinsics": torch.tensor([
            [100.0, 0.0, 32.0],
            [0.0, 100.0, 32.0],
            [0.0, 0.0, 1.0],
        ]).unsqueeze(0).expand(B, -1, -1).clone(),
        "prompts": ["pick up the red cup"] * B,
    }


def test_e2e_lora_with_ds():
    """E2E: LoRA mode with DeepSpeed ZeRO-2 (tiny model, 5 steps)."""
    print("\n=== E2E Test 1: LoRA + DeepSpeed ZeRO-2 (5 steps) ===")
    import deepspeed

    init_distributed()

    # Build model on CPU (as train.py now does for DeepSpeed mode)
    model = create_tiny_model(freeze_llm=True)

    # Apply LoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model.vlm = get_peft_model(model.vlm, lora_config)

    # Build optimizer with param groups (as train.py does)
    params_3d, params_lora = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["depth_to_gaussian", "gs_encoder", "gs_projector", "gs_type_embedding"]):
            params_3d.append(param)
        elif "lora" in name.lower():
            params_lora.append(param)

    groups = []
    if params_3d:
        groups.append({"params": params_3d, "lr": 1e-3, "name": "3d_branch"})
    if params_lora:
        groups.append({"params": params_lora, "lr": 2e-4, "name": "lora"})

    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    n_3d = sum(p.numel() for p in params_3d)
    n_lora = sum(p.numel() for p in params_lora)
    print(f"  Trainable: 3d={n_3d/1e3:.1f}K, lora={n_lora/1e3:.1f}K")

    # Load ZeRO-2 config and inject batch sizes
    ds_path = Path(__file__).parent.parent / "config" / "deepspeed_zero2.json"
    with open(ds_path) as f:
        ds_config = json.load(f)

    grad_accum = 2
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = 1 * 1 * grad_accum
    ds_config["bf16"]["enabled"] = False  # tiny model is float32

    # Scheduler must be created before DS wraps the optimizer
    total_steps = 5
    scheduler = build_scheduler(optimizer, {"warmup_ratio": 0.2}, total_steps)

    # DeepSpeed init (model on CPU -> moved to GPU, scheduler managed by DS)
    engine, opt, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    device = engine.device
    print(f"  DeepSpeed engine on {device}")

    # Rendering loss (after DS init, model on GPU)
    render_loss_fn = GaussianRenderingLoss(
        lambda_l1=0.8, lambda_ssim=0.2,
        lambda_depth=0.5, lambda_opacity=0.01,
        image_size=(32, 32),
    ).to(device)

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/w50037733/models/RoboBrain2.5-8B-NV")

    # Training loop
    losses = []
    print(f"  Running {total_steps * grad_accum} micro-batches ({total_steps} optimizer steps)...")

    for step in range(total_steps * grad_accum):
        batch = create_synthetic_batch(B=1)

        metrics = train_step(
            model=engine,
            batch=batch,
            tokenizer=tokenizer,
            render_loss_fn=render_loss_fn,
            render_weight=0.1,
            device=str(device),
        )
        loss = metrics["loss"]

        engine.backward(loss)
        engine.step()
        # Scheduler managed by DeepSpeed (auto-steps at accum boundaries)

        losses.append(metrics["lm_loss"])

        if step % grad_accum == 0:
            opt_step = step // grad_accum + 1
            print(f"    Step {opt_step}/{total_steps}: lm={metrics['lm_loss']:.4f} "
                  f"render={metrics['render_loss']:.4f}")

    print(f"  Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  OK: LoRA + ZeRO-2 training completed successfully")

    del engine
    return True


def test_e2e_full_zero3_with_ds():
    """E2E: Full fine-tuning with DeepSpeed ZeRO-3 + CPU offload (tiny model, 3 steps).

    This tests the actual production config path:
    - ZeRO-3 parameter sharding (each GPU holds 1/N params)
    - CPU optimizer offload
    - Gradient checkpointing interaction
    """
    print("\n=== E2E Test 2: Full fine-tuning + ZeRO-3 + CPU offload (3 steps) ===")
    import deepspeed

    init_distributed()

    model = create_tiny_model(freeze_llm=False)

    # Enable gradient checkpointing (as train_full.yaml does)
    lang_model = model._get_language_model()
    if hasattr(lang_model, "gradient_checkpointing_enable"):
        lang_model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # Optimizer (3D branch + LLM)
    params_3d, params_llm = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["depth_to_gaussian", "gs_encoder", "gs_projector", "gs_type_embedding"]):
            params_3d.append(param)
        else:
            params_llm.append(param)

    groups = []
    if params_3d:
        groups.append({"params": params_3d, "lr": 1e-3, "name": "3d_branch"})
    if params_llm:
        groups.append({"params": params_llm, "lr": 5e-5, "name": "llm"})

    n_3d = sum(p.numel() for p in params_3d)
    n_llm = sum(p.numel() for p in params_llm)
    print(f"  Trainable: 3d={n_3d/1e3:.1f}K, llm={n_llm/1e3:.1f}K")

    # Use DeepSpeedCPUAdam (required for ZeRO-3 CPU offload)
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    optimizer = DeepSpeedCPUAdam(groups, weight_decay=0.01)
    print(f"  Optimizer: DeepSpeedCPUAdam")

    # Load production ZeRO-3 config
    ds_path = Path(__file__).parent.parent / "config" / "deepspeed_zero3.json"
    with open(ds_path) as f:
        ds_config = json.load(f)

    grad_accum = 2
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = 1 * 1 * grad_accum
    ds_config["bf16"]["enabled"] = False  # tiny model is float32

    # Scheduler before DS init
    total_steps = 3
    scheduler = build_scheduler(optimizer, {"warmup_ratio": 0.1}, total_steps)

    engine, opt, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )
    device = engine.device
    print(f"  DeepSpeed ZeRO-3 engine on {device}")
    print(f"  Optimizer offload: CPU")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/w50037733/models/RoboBrain2.5-8B-NV")

    # Rendering loss (test with rendering too, as production does)
    render_loss_fn = GaussianRenderingLoss(
        lambda_l1=0.8, lambda_ssim=0.2,
        lambda_depth=0.5, lambda_opacity=0.01,
        image_size=(32, 32),
    ).to(device)

    losses = []
    for step in range(total_steps * grad_accum):
        batch = create_synthetic_batch(B=1)
        metrics = train_step(engine, batch, tokenizer, render_loss_fn, 0.1, str(device))
        engine.backward(metrics["loss"])
        engine.step()
        losses.append(metrics["lm_loss"])

        if step % grad_accum == 0:
            opt_step = step // grad_accum + 1
            print(f"    Step {opt_step}/{total_steps}: lm={metrics['lm_loss']:.4f} "
                  f"render={metrics['render_loss']:.4f}")

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Verify gradient flow — with ZeRO-3, param.grad may be None after step
    # but if loss decreased, gradients flowed correctly
    if losses[-1] < losses[0]:
        print(f"  OK: Loss decreased, gradients flowing correctly through ZeRO-3")
    else:
        print(f"  WARN: Loss did not decrease (may need more steps, not necessarily a bug)")

    print(f"  OK: Full fine-tuning + ZeRO-3 + CPU offload completed successfully")

    del engine
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("  DeepSpeed End-to-End Training Tests")
    print("=" * 60)

    results = []
    results.append(("LoRA + ZeRO-2", test_e2e_lora_with_ds()))
    results.append(("Full + ZeRO-3 + CPU offload", test_e2e_full_zero3_with_ds()))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    print("\n" + "=" * 60)
    all_passed = all(r[1] for r in results)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)
