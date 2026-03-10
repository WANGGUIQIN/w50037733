"""VLM Integration Validation for RoboBrain-3DGS.

Tests the full pipeline with a real Qwen2.5-VL architecture:
    RGBD -> 3D Gaussians -> GS Encoder -> Token Injection -> Qwen2.5-VL LLM -> Text

Uses a tiny randomly-initialized Qwen2.5-VL for fast validation.
The integration code is identical to what runs with the full model.
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/w50037733/robobrain_3dgs")

from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config
from data.synthetic import create_synthetic_sample


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def validate_vlm_integration():
    """Run full VLM integration validation."""
    print_section("RoboBrain-3DGS VLM Integration Validation")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ======================================
    # Step 1: Create Model
    # ======================================
    print_section("1. Model Creation")

    print("Creating tiny Qwen2.5-VL config for validation...")
    vlm_config = create_tiny_vlm_config()
    print(f"  LLM hidden_dim: {vlm_config.text_config.hidden_size}")
    print(f"  LLM layers: {vlm_config.text_config.num_hidden_layers}")
    vit_depth = vlm_config.vision_config.depth if hasattr(vlm_config.vision_config, 'depth') else vlm_config.vision_config.get('depth', 'N/A')
    print(f"  ViT depth: {vit_depth}")
    print(f"  Vocab size: {vlm_config.text_config.vocab_size}")

    print("\nCreating RoboBrain3DGS_VLM model...")
    model = RoboBrain3DGS_VLM(
        vlm_config=vlm_config,
        num_gaussians=512,       # Small for validation
        sh_degree=2,
        num_gs_tokens=32,        # 32 3D tokens
        gs_encoder_dim=256,
        freeze_vision_encoder=True,
        freeze_llm=False,
    ).to(device)

    # Parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {frozen:,}")

    modules = {
        "Qwen2.5-VL ViT (frozen)": model.vlm.model.visual,
        "Qwen2.5-VL LLM": model.vlm.model.language_model,
        "Qwen2.5-VL LM Head": model.vlm.lm_head,
        "DepthToGaussian": model.depth_to_gaussian,
        "GS Encoder": model.gs_encoder,
        "GS Projector": model.gs_projector,
    }
    for name, module in modules.items():
        n_params = sum(p.numel() for p in module.parameters())
        n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status = "TRAIN" if n_trainable > 0 else "FROZEN"
        print(f"    [{status}] {name}: {n_params:,} params ({n_trainable:,} trainable)")

    # ======================================
    # Step 2: Create Synthetic Data
    # ======================================
    print_section("2. Synthetic Data")

    sample = create_synthetic_sample(image_size=256, device=device)
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Depth: {sample['depth'].shape} range=[{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")
    print(f"  Intrinsics: {sample['intrinsics'].shape}")

    # Create text input (simulating tokenized instruction)
    B = 1
    seq_len = 20
    input_ids = torch.randint(100, 5000, (B, seq_len), device=device)
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

    # Create labels (shifted input_ids for next-token prediction)
    labels = input_ids.clone()
    # Mask the first few tokens as context (no loss)
    labels[:, :5] = -100

    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Labels: {labels.shape} (first 5 masked)")

    # ======================================
    # Step 3: Forward Pass (text only, no image)
    # ======================================
    print_section("3. Forward Pass - Text Only")

    model.train(False)
    with torch.no_grad():
        t0 = time.time()
        out_text_only = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        t1 = time.time()

    print(f"  Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Logits: {out_text_only['logits'].shape}")
    print(f"  Loss: {out_text_only['loss'].item():.4f}")

    # ======================================
    # Step 4: Forward Pass (text + 3D Gaussian)
    # ======================================
    print_section("4. Forward Pass - Text + 3D Gaussian")

    with torch.no_grad():
        t0 = time.time()
        out_with_3d = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth=sample["depth"],
            intrinsics=sample["intrinsics"],
            rgb_for_3d=sample["rgb"],
            labels=labels,
        )
        t1 = time.time()

    print(f"  Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Logits: {out_with_3d['logits'].shape}")
    print(f"  Hidden states: {out_with_3d['hidden_states'].shape}")
    print(f"  Loss: {out_with_3d['loss'].item():.4f}")

    expected_seq_len = seq_len + model.num_gs_tokens
    actual_seq_len = out_with_3d["logits"].shape[1]
    print(f"\n  Sequence length check:")
    print(f"    Text tokens: {seq_len}")
    print(f"    3D GS tokens: {model.num_gs_tokens}")
    print(f"    Expected total: {expected_seq_len}")
    print(f"    Actual total: {actual_seq_len}")
    seq_check = actual_seq_len == expected_seq_len
    print(f"    Match: {'YES' if seq_check else 'NO'}")

    # ======================================
    # Step 5: Training Loop (10 steps)
    # ======================================
    print_section("5. Training Loop (10 steps)")

    model.train()
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    losses = []
    has_3d_grad = False
    vit_has_grad = False

    for step in range(10):
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth=sample["depth"],
            intrinsics=sample["intrinsics"],
            rgb_for_3d=sample["rgb"],
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()

        if step == 0:
            # Verify gradient flow
            print("\n  Gradient flow check (step 0):")
            grad_checks = {
                "depth_to_gaussian": False,
                "gs_encoder": False,
                "gs_projector": False,
                "vlm.model": False,
                "vlm.lm_head": False,
            }
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.norm() > 0:
                    for key in grad_checks:
                        if key in name:
                            grad_checks[key] = True

            for key, has_grad in grad_checks.items():
                status = "PASS" if has_grad else "FAIL"
                print(f"    [{status}] {key}: gradient received = {has_grad}")

            has_3d_grad = grad_checks["depth_to_gaussian"] and grad_checks["gs_encoder"]

            # Verify ViT is frozen
            vit_has_grad = any(
                p.grad is not None and p.grad.norm() > 0
                for p in model.vlm.model.visual.parameters()
            )
            print(f"    [{'PASS' if not vit_has_grad else 'FAIL'}] ViT frozen: no gradients = {not vit_has_grad}")

        optimizer.step()
        losses.append(loss.item())

        if step % 3 == 0 or step == 9:
            print(f"  Step {step:2d}: loss = {loss.item():.4f}")

    # ======================================
    # Step 6: Text Generation
    # ======================================
    print_section("6. Text Generation (autoregressive)")

    model.train(False)
    # Short prompt
    prompt_ids = torch.randint(100, 5000, (1, 5), device=device)
    prompt_mask = torch.ones(1, 5, dtype=torch.long, device=device)

    t0 = time.time()
    generated = model.generate_with_3d(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        depth=sample["depth"],
        intrinsics=sample["intrinsics"],
        rgb_for_3d=sample["rgb"],
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True,
    )
    t1 = time.time()

    num_generated = generated.shape[1] - prompt_ids.shape[1]
    print(f"  Prompt tokens: {prompt_ids.shape[1]}")
    print(f"  Generated tokens: {num_generated}")
    print(f"  Total output: {generated.shape[1]}")
    print(f"  Generation time: {(t1-t0)*1000:.1f}ms")
    print(f"  Generated IDs: {generated[0, prompt_ids.shape[1]:].tolist()}")

    # ======================================
    # Summary
    # ======================================
    print_section("Validation Summary")

    print(f"  Model: RoboBrain3DGS + Qwen2.5-VL (tiny)")
    print(f"  Pipeline: RGBD -> 3D Gaussians (512) -> {model.num_gs_tokens} tokens -> Qwen2.5-VL LLM -> text")
    print(f"  LLM hidden dim: {model.llm_hidden_dim}")

    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Training: loss {losses[0]:.4f} -> {losses[-1]:.4f} ({loss_reduction:.1f}% reduction)")

    checks = [
        ("Forward pass (text only)", out_text_only["loss"] is not None),
        ("Forward pass (text + 3D)", out_with_3d["loss"] is not None),
        ("Sequence length correct", seq_check),
        ("Gradients flow to 3D branch", has_3d_grad),
        ("ViT encoder frozen", not vit_has_grad),
        ("Loss decreases in training", losses[-1] < losses[0]),
        ("Text generation works", num_generated > 0),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # ======================================
    # Production Usage Guide
    # ======================================
    print_section("Next Steps for Production")
    print("""
  To use with the real RoboBrain 2.5 (8B) model:

  1. Load pretrained VLM:
     from transformers import AutoConfig
     config = AutoConfig.from_pretrained("BAAI/RoboBrain2.5-8B-NV")
     model = RoboBrain3DGS_VLM(vlm_config=config, ...)
     # Then load pretrained weights into model.vlm

  2. Apply LoRA for efficient fine-tuning:
     from peft import LoraConfig, get_peft_model
     lora_config = LoraConfig(r=16, target_modules=["q_proj","v_proj"])
     model.vlm = get_peft_model(model.vlm, lora_config)

  3. Output format (autoregressive text):
     Affordance: "grasp region [x1, y1, x2, y2]"
     Constraint: "approach (0, 0, -1), gripper 0.06m, force 5N"
     Trajectory: "[(x1,y1,d1), (x2,y2,d2), ...]"
""")

    return all_pass


if __name__ == "__main__":
    success = validate_vlm_integration()
    sys.exit(0 if success else 1)
