"""RoboBrain-3DGS LoRA Deployment Demo.

Minimal, self-contained script showing real inference with a trained
LoRA checkpoint. Copy this as a starting point for your deployment.

Usage:
    python demo_inference.py

    # Custom checkpoint + data
    python demo_inference.py \
        --checkpoint outputs/lora/best \
        --rgb  data/rlbench_sample/close_jar/all_variations/episodes/episode0/front_rgb/0.png \
        --depth data/rlbench_sample/close_jar/all_variations/episodes/episode0/front_depth/0.png \
        --prompt "close the jar"
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from inference_3dgs import UnifiedInference3DGS


def main():
    parser = argparse.ArgumentParser(description="RoboBrain-3DGS LoRA Deployment Demo")
    parser.add_argument("--base_model", default="/home/w50037733/models/RoboBrain2.5-8B-NV")
    parser.add_argument("--checkpoint", default="outputs/lora/best")
    parser.add_argument("--rgb", default="data/rlbench_sample/close_jar/all_variations/episodes/episode0/front_rgb/0.png")
    parser.add_argument("--depth", default="data/rlbench_sample/close_jar/all_variations/episodes/episode0/front_depth/0.png")
    parser.add_argument("--prompt", default="close the jar")
    args = parser.parse_args()

    # =========================================================
    # 1. Load model + LoRA checkpoint (one-time, ~7s)
    # =========================================================
    print("=" * 60)
    print("  RoboBrain-3DGS LoRA Deployment Demo")
    print("=" * 60)

    t0 = time.time()
    model = UnifiedInference3DGS(
        model_id=args.base_model,
        checkpoint=args.checkpoint,
        mode="lora",
    )
    print(f"\nModel loaded in {time.time() - t0:.1f}s")

    # =========================================================
    # 2. Affordance prediction (RGBD + prompt)
    # =========================================================
    print("\n" + "-" * 60)
    print("  Test 1: Affordance (RGBD) — primary use case")
    print("-" * 60)

    t1 = time.time()
    result = model.inference(
        text=args.prompt,
        image=args.rgb,
        depth=args.depth,
        task="affordance",
        do_sample=False,
        max_new_tokens=128,
        plot=True,
    )
    dt1 = time.time() - t1

    print(f"\n  Time: {dt1:.1f}s")
    print(f"  Answer: {result['answer']}")
    print(f"  Parsed: {result['parsed']}")

    # =========================================================
    # 3. Affordance without depth (2D fallback)
    # =========================================================
    print("\n" + "-" * 60)
    print("  Test 2: Affordance (RGB only) — 2D fallback")
    print("-" * 60)

    t2 = time.time()
    result_2d = model.inference(
        text=args.prompt,
        image=args.rgb,
        depth=None,           # no depth -> no 3D tokens
        task="affordance",
        do_sample=False,
        max_new_tokens=128,
    )
    dt2 = time.time() - t2

    print(f"\n  Time: {dt2:.1f}s")
    print(f"  Answer: {result_2d['answer']}")

    # =========================================================
    # 4. General VQA (like RoboBrain 2.5)
    # =========================================================
    print("\n" + "-" * 60)
    print("  Test 3: General VQA — RoboBrain 2.5 compatible")
    print("-" * 60)

    t3 = time.time()
    result_vqa = model.inference(
        text="What objects are on the table?",
        image=args.rgb,
        depth=None,
        task="general",
        do_sample=False,
        max_new_tokens=128,
    )
    dt3 = time.time() - t3

    print(f"\n  Time: {dt3:.1f}s")
    print(f"  Answer: {result_vqa['answer']}")

    # =========================================================
    # 5. Programmatic usage example
    # =========================================================
    print("\n" + "-" * 60)
    print("  Programmatic Usage (copy-paste ready)")
    print("-" * 60)
    print("""
    from inference_3dgs import UnifiedInference3DGS

    # Init once
    model = UnifiedInference3DGS(
        model_id="/path/to/RoboBrain2.5-8B-NV",
        checkpoint="outputs/lora/best",
        mode="lora",
    )

    # Predict affordance
    result = model.inference(
        text="pick up the red cup",
        image="rgb.png",
        depth="depth.npy",
        task="affordance",
        do_sample=False,
    )

    # Use structured output
    u, v = result["parsed"]["u"], result["parsed"]["v"]
    width = result["parsed"]["gripper_width"]
    approach = result["parsed"]["approach"]
    print(f"Grasp at ({u:.2f}, {v:.2f}), width={width}, approach={approach}")
    """)

    print("=" * 60)
    print("  Demo complete. Annotated image saved in result/")
    print("=" * 60)


if __name__ == "__main__":
    main()
