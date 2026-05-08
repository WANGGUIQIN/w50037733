#!/usr/bin/env python3
"""End-to-end inference: trained RoboBrain-3DGS + Lang-SAM affordance refinement.

CLI mirrors run_inference.py. Adds --refine-langsam to apply GroundingDINO+SAM
post-processing on each step's affordance after the model emits the plan.

Usage:
    # Single scene with trained LoRA checkpoint
    python scripts/e2e_inference.py \
        --checkpoint outputs/lora/best \
        --image data/processed/rlbench/episode_000300/rgb_0.png \
        --depth data/processed/rlbench/episode_000300/depth_0.npy \
        --text "take the steak off the grill" \
        --task planning \
        --refine-langsam --visualize \
        --output_dir output_e2e/single

    # Batch over a dataset
    python scripts/e2e_inference.py \
        --checkpoint outputs/lora/best \
        --dataset rlbench --num_episodes 20 \
        --task planning --refine-langsam \
        --output_dir output_e2e/rlbench

    # Use base model only (no checkpoint)
    python scripts/e2e_inference.py \
        --image scene.png --text "pick up the cup" \
        --task planning --refine-langsam --visualize
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Project root + scripts/ on path so we can import both
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image

from utils.prompt_utils import parse_affordance_output
from utils.visualize_inference import render as render_inference

# Reuse plan-format parsing helpers from run_inference.py
sys.path.insert(0, str(ROOT))
from run_inference import format_plan_as_json  # type: ignore

from postprocess_affordance import GroundingSAM, refine_plan, visualize as viz_refined


# ---------------------------------------------------------------------------
# Lang-SAM refinement helper
# ---------------------------------------------------------------------------

def refine_with_langsam(
    structured: dict,
    image_path: str,
    depth_path: str | None,
    grounder: "GroundingSAM",
    strategy: str = "inscribed",
    intrinsics: np.ndarray | None = None,
) -> dict:
    """Apply Lang-SAM post-processor to a structured plan dict.

    The original LoRA-emitted affordance is preserved under
    'affordance_lora' (mirroring --refine-affordance) for A/B comparison.
    """
    # Snapshot the lora-emitted coords before they get refined
    snapshot = copy.deepcopy(structured)
    for s in snapshot.get("steps", []):
        if "affordance" in s:
            s["affordance_lora"] = list(s["affordance"])

    rgb = Image.open(image_path).convert("RGB")
    depth = np.load(depth_path) if depth_path and os.path.exists(depth_path) else None

    if intrinsics is None:
        W, H = rgb.size
        intrinsics = np.array([[W / 2, 0, W / 2],
                               [0, H / 2, H / 2],
                               [0, 0, 1]], dtype=np.float32)

    refined = refine_plan(rgb, depth, intrinsics, snapshot, grounder, strategy=strategy)
    return refined


def _load_intrinsics_from_meta(image_path: str) -> np.ndarray | None:
    """If image lives in a processed-data episode dir, pull intrinsics."""
    img = Path(image_path)
    meta_path = img.parent / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            return np.array(meta["intrinsics"], dtype=np.float32)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Per-scene driver (mirrors run_inference.run_single)
# ---------------------------------------------------------------------------

def run_single(model, image: str, depth: str | None, text: str, task: str,
               temperature: float, max_tokens: int) -> dict:
    """Run inference on a single scene, return structured result."""
    result = model.inference(
        text=text,
        image=image,
        depth=depth,
        task=task,
        temperature=temperature,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
    )

    raw = result["answer"]

    if task == "planning":
        structured = format_plan_as_json(raw, task=text)
    elif task == "affordance":
        structured = parse_affordance_output(raw)
        structured["task"] = text
    else:
        structured = {"task": text, "answer": raw}

    return {"raw": raw, "structured": structured}


# ---------------------------------------------------------------------------
# CLI (aligned with run_inference.py)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RoboBrain-3DGS E2E Inference + Lang-SAM Affordance Refinement"
    )
    # --- alignment with run_inference.py ---
    parser.add_argument("--model", default="/home/edge/RoboBrain/models/RoboBrain2.5-8B-NV")
    parser.add_argument("--checkpoint", default=None,
                        help="Trained checkpoint (e.g. outputs/lora/best)")
    parser.add_argument("--mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--task", default="planning",
                        choices=["planning", "affordance", "pointing",
                                 "trajectory", "grounding", "general"])
    parser.add_argument("--dataset", default=None, help="Batch: dataset name")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--output_dir", default=None, help="Save JSON results to dir")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--visualize", action="store_true",
                        help="Save affordance-overlay PNG next to JSON output. "
                             "Single-image mode -> result_viz.png in --output_dir "
                             "(or alongside --image if no output_dir). "
                             "Batch mode -> <episode>_viz.png per episode.")
    parser.add_argument("--refine-affordance", dest="refine_affordance",
                        action="store_true",
                        help="Two-stage inference using base RoboBrain pointing. "
                             "(Existing path from run_inference.py.)")

    # --- new flags for Lang-SAM postprocessing ---
    parser.add_argument("--refine-langsam", dest="refine_langsam",
                        action="store_true",
                        help="Apply GroundingDINO+SAM post-processor on the "
                             "model-emitted affordances. The original LoRA "
                             "coordinates are preserved as 'affordance_lora' "
                             "for A/B comparison. Only applies when --task=planning.")
    parser.add_argument("--strategy", default="inscribed",
                        choices=["centroid", "inscribed", "pca"],
                        help="Lang-SAM point-selection strategy.")
    parser.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam", default="facebook/sam-vit-base")

    args = parser.parse_args()

    # ---- model ----
    from inference_3dgs import UnifiedInference3DGS
    model = UnifiedInference3DGS(
        model_id=args.model,
        checkpoint=args.checkpoint,
        mode=args.mode,
    )

    # ---- Lang-SAM grounder (lazy-loaded on demand) ----
    grounder = None
    if args.refine_langsam:
        if args.task != "planning":
            print("WARN: --refine-langsam only applies when --task=planning. "
                  "Disabling.")
            args.refine_langsam = False
        else:
            grounder = GroundingSAM(gdino_id=args.gdino, sam_id=args.sam)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # =================================================================
    # Batch mode (--dataset)
    # =================================================================
    if args.dataset:
        data_dir = Path(f"data/processed/{args.dataset}")
        episodes = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith("episode_")])
        episodes = episodes[:args.num_episodes]

        for ep_dir in episodes:
            with open(ep_dir / "meta.json") as f:
                meta = json.load(f)

            text = meta.get("task", "manipulation")
            image = str(ep_dir / "rgb_0.png")
            depth = str(ep_dir / "depth_0.npy")

            print(f"\n{'='*60}")
            print(f"Episode: {ep_dir.name} | Task: {text}")

            out = run_single(model, image, depth, text, args.task,
                             args.temperature, args.max_tokens)
            if args.refine_affordance and args.task == "planning":
                out["structured"] = model.refine_affordance_via_base(
                    out["structured"], image,
                )
            if args.refine_langsam and args.task == "planning":
                K = _load_intrinsics_from_meta(image)
                out["structured"] = refine_with_langsam(
                    out["structured"], image, depth, grounder,
                    strategy=args.strategy, intrinsics=K,
                )
            out["episode"] = ep_dir.name
            results.append(out)

            print(json.dumps(out["structured"], indent=2, ensure_ascii=False))

            if args.output_dir:
                save_path = Path(args.output_dir) / f"{ep_dir.name}.json"
                with open(save_path, "w") as f:
                    json.dump(out["structured"], f, indent=2, ensure_ascii=False)

            if args.visualize:
                viz_dir = Path(args.output_dir) if args.output_dir else ep_dir
                viz_path = viz_dir / f"{ep_dir.name}_viz.png"
                try:
                    render_inference(image, out["structured"], args.task, viz_path)
                    print(f"  viz -> {viz_path}")
                except Exception as e:
                    print(f"  WARN render_inference failed: {e}")

                # Extra: side-by-side viz that also shows orig vs refined points
                if args.refine_langsam:
                    try:
                        viz_refined(ep_dir, out["structured"],
                                    viz_dir / f"{ep_dir.name}_refined_viz.png")
                    except Exception as e:
                        print(f"  WARN refined viz failed: {e}")

    # =================================================================
    # Single-scene mode (--image)
    # =================================================================
    elif args.image:
        out = run_single(model, args.image, args.depth,
                         args.text or "describe the scene", args.task,
                         args.temperature, args.max_tokens)
        if args.refine_affordance and args.task == "planning":
            out["structured"] = model.refine_affordance_via_base(
                out["structured"], args.image,
            )
        if args.refine_langsam and args.task == "planning":
            K = _load_intrinsics_from_meta(args.image)
            out["structured"] = refine_with_langsam(
                out["structured"], args.image, args.depth, grounder,
                strategy=args.strategy, intrinsics=K,
            )
        results.append(out)
        print("\n" + json.dumps(out["structured"], indent=2, ensure_ascii=False))

        if args.output_dir:
            with open(Path(args.output_dir) / "result.json", "w") as f:
                json.dump(out["structured"], f, indent=2, ensure_ascii=False)

        if args.visualize:
            if args.output_dir:
                viz_path = Path(args.output_dir) / "result_viz.png"
            else:
                viz_path = Path(args.image).with_name(
                    Path(args.image).stem + "_viz.png"
                )
            try:
                render_inference(args.image, out["structured"], args.task, viz_path)
                print(f"viz -> {viz_path}")
            except Exception as e:
                print(f"WARN render_inference failed: {e}")

            if args.refine_langsam:
                # Build a temporary "episode dir" with the rgb so viz_refined works
                episode_dir = (
                    Path(args.image).parent
                    if (Path(args.image).parent / f"rgb_0.png").exists()
                    else Path(args.output_dir or ".")
                )
                refined_viz = (
                    Path(args.output_dir) / "result_refined_viz.png"
                    if args.output_dir
                    else Path(args.image).with_name(
                        Path(args.image).stem + "_refined_viz.png"
                    )
                )
                try:
                    viz_refined(episode_dir, out["structured"], refined_viz)
                except Exception as e:
                    print(f"WARN refined viz failed: {e}")
    else:
        parser.print_help()
        sys.exit(1)

    # ---- aggregate ----
    if args.output_dir and results:
        with open(Path(args.output_dir) / "all_results.json", "w") as f:
            json.dump([r["structured"] for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
