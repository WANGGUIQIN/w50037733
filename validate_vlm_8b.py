"""Production validation with real RoboBrain2.5-8B-NV model.

Tests the full RGBD -> 3D Gaussian -> Qwen3-VL -> text pipeline
using the actual 8B pretrained weights.
"""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/w50037733/robobrain_3dgs")
os.environ["HF_TOKEN"] = ""

from transformers import AutoProcessor
from models.robobrain_vlm import RoboBrain3DGS_VLM
from data.synthetic import create_synthetic_sample

MODEL_PATH = "/home/w50037733/models/RoboBrain2.5-8B-NV"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def validate_8b():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_section("RoboBrain2.5-8B-NV + 3DGS Validation")
    print(f"Device: {device}")

    # ======================================
    # Step 1: Load Real Model
    # ======================================
    print_section("1. Loading Real Model (8B)")

    t0 = time.time()
    model = RoboBrain3DGS_VLM.from_pretrained(
        model_path=MODEL_PATH,
        num_gaussians=1024,
        sh_degree=2,
        num_gs_tokens=64,
        gs_encoder_dim=512,
        freeze_vision_encoder=True,
        freeze_llm=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    t1 = time.time()
    print(f"  Load time: {t1-t0:.1f}s")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total/1e9:.2f}B")
    print(f"  Trainable parameters: {trainable/1e6:.1f}M")

    modules = {
        "Qwen3-VL ViT (frozen)": model.vlm.model.visual,
        "Qwen3-VL LLM": model.vlm.model.language_model,
        "Qwen3-VL LM Head": model.vlm.lm_head,
        "DepthToGaussian (new)": model.depth_to_gaussian,
        "GS Encoder (new)": model.gs_encoder,
        "GS Projector (new)": model.gs_projector,
    }
    for name, module in modules.items():
        n_params = sum(p.numel() for p in module.parameters())
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status = "TRAIN" if n_train > 0 else "FROZEN"
        print(f"    [{status}] {name}: {n_params/1e6:.1f}M ({n_train/1e6:.1f}M trainable)")

    # ======================================
    # Step 2: Load Processor and Create Inputs
    # ======================================
    print_section("2. Processor & Synthetic Data")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f"  Processor vocab size: {processor.tokenizer.vocab_size}")

    # Create synthetic RGBD data
    sample = create_synthetic_sample(image_size=256, device=device)
    print(f"  RGB: {sample['rgb'].shape}, Depth: {sample['depth'].shape}")
    print(f"  Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")

    # Create tokenized instruction
    instruction = "You are a robot. The task is 'pick up the blue box'. Please predict a possible affordance area of the end effector."
    tokens = processor.tokenizer(instruction, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    print(f"  Instruction: {instruction[:60]}...")
    print(f"  Token count: {input_ids.shape[1]}")

    # Labels: predict the output (mask prompt as -100)
    labels = input_ids.clone()
    labels[:, :] = -100  # For generation validation, mask all

    # ======================================
    # Step 3: Forward Pass (text only)
    # ======================================
    print_section("3. Forward Pass - Text Only")

    model.train(False)
    with torch.no_grad():
        t0 = time.time()
        out_text = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        t1 = time.time()

    print(f"  Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Logits: {out_text['logits'].shape}")
    print(f"  Hidden states: {out_text['hidden_states'].shape}")

    # ======================================
    # Step 4: Forward Pass (text + 3D Gaussian)
    # ======================================
    print_section("4. Forward Pass - Text + 3D Gaussian")

    with torch.no_grad():
        t0 = time.time()
        out_3d = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth=sample["depth"].to(torch.bfloat16),
            intrinsics=sample["intrinsics"].to(torch.bfloat16),
            rgb_for_3d=sample["rgb"].to(torch.bfloat16),
        )
        t1 = time.time()

    print(f"  Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Logits: {out_3d['logits'].shape}")
    print(f"  Hidden states: {out_3d['hidden_states'].shape}")

    expected_len = input_ids.shape[1] + model.num_gs_tokens
    actual_len = out_3d["logits"].shape[1]
    seq_ok = actual_len == expected_len
    print(f"  Sequence: {input_ids.shape[1]} text + {model.num_gs_tokens} 3D = {expected_len} expected, {actual_len} actual -> {'OK' if seq_ok else 'MISMATCH'}")

    # ======================================
    # Step 5: Text Generation with 3D Context
    # ======================================
    print_section("5. Text Generation with 3D Gaussian Context")

    # Short prompt for affordance
    prompt = "pick up the blue box. affordance region:"
    prompt_tokens = processor.tokenizer(prompt, return_tensors="pt")
    prompt_ids = prompt_tokens.input_ids.to(device)
    prompt_mask = prompt_tokens.attention_mask.to(device)

    print(f"  Prompt: '{prompt}'")
    print(f"  Generating with 3D Gaussian context...")

    t0 = time.time()
    generated = model.generate_with_3d(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        depth=sample["depth"].to(torch.bfloat16),
        intrinsics=sample["intrinsics"].to(torch.bfloat16),
        rgb_for_3d=sample["rgb"].to(torch.bfloat16),
        max_new_tokens=50,
        temperature=0.7,
        do_sample=False,  # Greedy for reproducibility
    )
    t1 = time.time()

    new_tokens = generated[0, prompt_ids.shape[1]:]
    generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"  Generation time: {(t1-t0)*1000:.1f}ms")
    print(f"  Generated tokens: {len(new_tokens)}")
    print(f"  Generated text: '{generated_text}'")

    # ======================================
    # Step 6: One Training Step
    # ======================================
    print_section("6. One Training Step (gradient check)")

    model.train()
    # Only train 3D branch + projector (freeze LLM for this check)
    trainable_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(k in n for k in ["depth_to_gaussian", "gs_encoder", "gs_projector", "gs_type_embedding"])
    ]
    print(f"  Training 3D branch only: {sum(p.numel() for p in trainable_params)/1e6:.1f}M params")
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    # Create labels for LM loss
    labels_for_loss = input_ids.clone()
    labels_for_loss[:, :-1] = -100  # only predict last token

    optimizer.zero_grad()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        depth=sample["depth"].to(torch.bfloat16),
        intrinsics=sample["intrinsics"].to(torch.bfloat16),
        rgb_for_3d=sample["rgb"].to(torch.bfloat16),
        labels=labels_for_loss,
    )
    loss = out["loss"]
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()

    has_3d_grad = any(
        p.grad is not None and p.grad.norm() > 0
        for n, p in model.named_parameters()
        if "depth_to_gaussian" in n or "gs_encoder" in n
    )
    print(f"  3D branch has gradients: {'YES' if has_3d_grad else 'NO'}")
    optimizer.step()

    # ======================================
    # Summary
    # ======================================
    print_section("Validation Summary")

    checks = [
        ("Text-only forward pass", out_text["logits"] is not None),
        ("3D Gaussian forward pass", out_3d["logits"] is not None),
        ("Sequence length correct", seq_ok),
        ("Text generation works", len(generated_text) > 0),
        ("3D branch gradients flow", has_3d_grad),
    ]
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  Model: RoboBrain2.5-8B-NV + 3DGS")
    print(f"  Pipeline: RGBD -> {model.num_gs_tokens} 3D tokens prepended to Qwen3-VL")
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    return all_pass


if __name__ == "__main__":
    success = validate_8b()
    sys.exit(0 if success else 1)
