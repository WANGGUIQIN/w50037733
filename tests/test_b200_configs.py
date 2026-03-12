"""Validate B200-optimized configs with tiny model.

Tests ZeRO-1 + FusedAdam (LoRA) and ZeRO-2 + FusedAdam (Full) paths.
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config
from models.gs_renderer import GaussianRenderingLoss
from train import train_step, build_scheduler


def init_distributed():
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29504")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def create_tiny_model(freeze_llm=True):
    config = create_tiny_vlm_config()
    return RoboBrain3DGS_VLM(
        vlm_config=config,
        num_gaussians=64, sh_degree=2,
        num_gs_tokens=8, gs_encoder_dim=128,
        freeze_vision_encoder=True, freeze_llm=freeze_llm,
    )


def create_batch(B=1):
    K = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    K[:, 0, 0] = K[:, 1, 1] = 100.0
    K[:, 0, 2] = K[:, 1, 2] = 32.0
    return {
        "rgb": torch.randn(B, 3, 64, 64),
        "depth": torch.rand(B, 1, 64, 64) * 2,
        "intrinsics": K,
        "prompts": ["pick up the red cup"] * B,
        "targets": [
            "affordance: [0.50, 0.45]. constraint: gripper_width=0.08, approach=[0.00, 0.00, -1.00]."
        ] * B,
    }


def test_lora_zero1_fused():
    """LoRA + ZeRO-1 + FusedAdam (B200 LoRA config path)."""
    print("\n=== Test 1: LoRA + ZeRO-1 + FusedAdam (3 steps) ===")
    import deepspeed

    init_distributed()

    model = create_tiny_model(freeze_llm=True)

    from peft import LoraConfig, get_peft_model
    model.vlm = get_peft_model(model.vlm, LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_train / 1e3:.1f}K params")

    # FusedAdam
    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(trainable, lr=2e-4, weight_decay=0.01)
    print(f"  Optimizer: FusedAdam")

    # Load B200 ZeRO-1 config
    ds_path = Path(__file__).parent.parent / "config" / "deepspeed_zero1_b200.json"
    with open(ds_path) as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["train_batch_size"] = 1
    ds_config["bf16"]["enabled"] = False
    # Reduce bucket sizes for test env (B200 uses 1e9 but test GPU has less memory)
    ds_config["zero_optimization"]["reduce_bucket_size"] = 5e7
    ds_config["zero_optimization"]["allgather_bucket_size"] = 5e7

    scheduler = build_scheduler(optimizer, {"warmup_ratio": 0.1}, 3)

    engine, opt, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=ds_config, model_parameters=trainable,
    )
    print(f"  ZeRO-1 engine on {engine.device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/w50037733/models/RoboBrain2.5-8B-NV")

    losses = []
    for step in range(3):
        metrics = train_step(engine, create_batch(), tokenizer, None, 0.0, str(engine.device))
        engine.backward(metrics["loss"])
        engine.step()
        losses.append(metrics["lm_loss"])
        print(f"    Step {step+1}: lm={metrics['lm_loss']:.4f}")

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    del engine
    return True


def test_full_zero2_fused():
    """Full fine-tuning + ZeRO-2 + FusedAdam (B200 Full config path)."""
    print("\n=== Test 2: Full + ZeRO-2 + FusedAdam (3 steps) ===")
    import deepspeed

    init_distributed()

    model = create_tiny_model(freeze_llm=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_train / 1e3:.1f}K params")

    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(trainable, lr=5e-5, weight_decay=0.01)

    ds_path = Path(__file__).parent.parent / "config" / "deepspeed_zero2_b200.json"
    with open(ds_path) as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = 1
    ds_config["gradient_accumulation_steps"] = 1
    ds_config["train_batch_size"] = 1
    ds_config["bf16"]["enabled"] = False
    # Reduce bucket sizes for test env (B200 uses 1e9 but test GPU has less memory)
    ds_config["zero_optimization"]["reduce_bucket_size"] = 5e7
    ds_config["zero_optimization"]["allgather_bucket_size"] = 5e7

    scheduler = build_scheduler(optimizer, {"warmup_ratio": 0.1}, 3)

    engine, opt, _, scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=ds_config, model_parameters=trainable,
    )
    print(f"  ZeRO-2 engine on {engine.device}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/w50037733/models/RoboBrain2.5-8B-NV")

    # Include rendering loss (as B200 config does)
    render_loss_fn = GaussianRenderingLoss(
        lambda_l1=0.8, lambda_ssim=0.2,
        lambda_depth=0.5, lambda_opacity=0.01,
        image_size=(32, 32),
    ).to(engine.device)

    losses = []
    for step in range(3):
        metrics = train_step(engine, create_batch(), tokenizer, render_loss_fn, 0.1, str(engine.device))
        engine.backward(metrics["loss"])
        engine.step()
        losses.append(metrics["lm_loss"])
        print(f"    Step {step+1}: lm={metrics['lm_loss']:.4f} render={metrics['render_loss']:.4f}")

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    del engine
    return True


def test_fused_adam_param_groups():
    """FusedAdam with multi-group LR (3d_branch vs lora/llm)."""
    print("\n=== Test 3: FusedAdam multi-group LR ===")
    import deepspeed
    from deepspeed.ops.adam import FusedAdam

    init_distributed()

    model = create_tiny_model(freeze_llm=False)

    params_3d, params_llm = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["depth_to_gaussian", "gs_encoder", "gs_projector", "gs_type_embedding"]):
            params_3d.append(param)
        else:
            params_llm.append(param)

    groups = [
        {"params": params_3d, "lr": 1e-3, "name": "3d_branch"},
        {"params": params_llm, "lr": 5e-5, "name": "llm"},
    ]
    optimizer = FusedAdam(groups, weight_decay=0.01)
    print(f"  3d_branch: {sum(p.numel() for p in params_3d)/1e3:.1f}K @ lr=1e-3")
    print(f"  llm: {sum(p.numel() for p in params_llm)/1e3:.1f}K @ lr=5e-5")

    ds_config = {
        "bf16": {"enabled": False}, "fp16": {"enabled": False},
        "zero_optimization": {"stage": 1, "reduce_bucket_size": 1e9},
        "gradient_accumulation_steps": 1, "gradient_clipping": 1.0,
        "train_batch_size": 1, "train_micro_batch_size_per_gpu": 1,
    }

    engine, opt, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
    )

    # Verify LR groups preserved
    lrs = [g["lr"] for g in opt.param_groups]
    assert 1e-3 in lrs, f"3d_branch lr=1e-3 not found in {lrs}"
    assert 5e-5 in lrs, f"llm lr=5e-5 not found in {lrs}"
    print(f"  OK: Multi-group LRs preserved: {lrs}")

    del engine
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("  B200 Configuration Validation Tests")
    print("=" * 60)

    results = []
    results.append(("LoRA + ZeRO-1 + FusedAdam", test_lora_zero1_fused()))
    results.append(("Full + ZeRO-2 + FusedAdam", test_full_zero2_fused()))
    results.append(("FusedAdam multi-group LR", test_fused_adam_param_groups()))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    print("\n" + "=" * 60)
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 60)

    sys.exit(0 if all(r[1] for r in results) else 1)
