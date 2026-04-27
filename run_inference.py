#!/usr/bin/env python3
"""Run inference with trained RoboBrain-3DGS model, output structured JSON.

Usage:
    # Single scene planning
    python run_inference.py \
        --checkpoint outputs/lora/best \
        --image data/processed/rlbench/episode_000300/rgb_0.png \
        --depth data/processed/rlbench/episode_000300/depth_0.npy \
        --text "take the steak off the grill" \
        --task planning

    # Batch inference, save JSON per episode
    python run_inference.py \
        --checkpoint outputs/lora/best \
        --dataset rlbench --num_episodes 10 \
        --task planning --output_dir results/

    # Affordance (RoboBrain2.5 native)
    python run_inference.py \
        --image scene.png --text "pick up the cup" --task affordance
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.prompt_utils import parse_planning_output, parse_affordance_output
from utils.visualize_inference import render as render_inference


def _parse_markdown_plan(text: str) -> dict:
    """Fallback parser for Markdown-formatted plan output.

    Handles output like:
        #### Step 1: Reach
        * **Operation Primitive:** reach
        * **Target Object:** yellow lid
        * **Affordance Point:** [u=840, v=750]
        * **Done_when:** ...
    """
    import re
    result = {"scene_objects": [], "steps": []}

    # Extract steps from markdown headers
    step_blocks = re.split(r'#{2,4}\s*Step\s+(\d+)', text)
    for i in range(1, len(step_blocks), 2):
        step_num = int(step_blocks[i])
        block = step_blocks[i + 1] if i + 1 < len(step_blocks) else ""

        step = {"step": step_num}

        # Extract operation primitive / action
        m = re.search(r'\*?\*?Operation Primitive:?\*?\*?\s*(\w+)', block, re.I)
        if m:
            step["action"] = m.group(1).lower()

        # Extract target (stop at first * or newline)
        m = re.search(r'\*?\*?Target Object:?\*?\*?\s*([^*\n]+)', block, re.I)
        if m:
            step["target"] = m.group(1).strip().rstrip('.')

        # Extract affordance - handle [u=840, v=750] or [0.84, 0.75]
        m = re.search(r'Affordance.*?\[(?:u\s*=\s*)?([0-9.]+)[,\s]+(?:v\s*=\s*)?([0-9.]+)\]', block, re.I)
        if m:
            u, v = float(m.group(1)), float(m.group(2))
            # Normalize if pixel coords (>1.0)
            if u > 1.0:
                u /= 1000.0
            if v > 1.0:
                v /= 1000.0
            step["affordance"] = [round(u, 2), round(v, 2)]

        # Extract approach
        m = re.search(r'Approach.*?\[(?:x\s*=\s*)?([0-9.-]+)[,\s]+(?:y\s*=\s*)?([0-9.-]+)[,\s]+(?:z\s*=\s*)?([0-9.-]+)\]', block, re.I)
        if m:
            step["approach"] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

        # Extract done_when
        m = re.search(r'Done.?when:?\*?\*?\s*(.+?)(?:\n|$)', block, re.I)
        if m:
            step["done_when"] = m.group(1).strip().strip('*').strip()

        if "action" in step:
            result["steps"].append(step)

    return result


def format_plan_as_json(raw_text: str, task: str) -> dict:
    """Parse raw model output into structured plan JSON.

    Tries compact format first (Step N: action(target)), falls back to
    Markdown parser if compact parse returns empty steps.
    """
    parsed = parse_planning_output(raw_text)

    # Fallback: if compact parser found no steps, try Markdown parser
    if not parsed.get("steps"):
        parsed = _parse_markdown_plan(raw_text)

    plan = {
        "task": task,
        "scene_objects": parsed.get("scene_objects", []),
        "num_steps": len(parsed.get("steps", [])),
        "steps": [],
    }

    for step in parsed.get("steps", []):
        s = {
            "step": step.get("step"),
            "action": step.get("action"),
            "target": step.get("target"),
        }
        if step.get("destination"):
            s["destination"] = step["destination"]
        if step.get("affordance"):
            s["affordance"] = step["affordance"]
        if step.get("approach"):
            s["approach"] = step["approach"]
        if step.get("constraints"):
            s["constraints"] = step["constraints"]
        if step.get("done_when"):
            s["done_when"] = step["done_when"]
        plan["steps"].append(s)

    return plan


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


def main():
    parser = argparse.ArgumentParser(description="RoboBrain-3DGS Structured Inference")
    parser.add_argument("--model", default="/home/edge/Embodied/models/RoboBrain2.5-8B-NV")
    parser.add_argument("--checkpoint", default=None,
                        help="Trained checkpoint (e.g. outputs/lora/best)")
    parser.add_argument("--mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--task", default="planning",
                        choices=["planning", "affordance", "pointing", "trajectory", "grounding", "general"])
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
    args = parser.parse_args()

    from inference_3dgs import UnifiedInference3DGS

    model = UnifiedInference3DGS(
        model_id=args.model,
        checkpoint=args.checkpoint,
        mode=args.mode,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []

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
            out["episode"] = ep_dir.name
            results.append(out)

            # Print structured output
            print(json.dumps(out["structured"], indent=2, ensure_ascii=False))

            # Save per-episode JSON
            if args.output_dir:
                save_path = Path(args.output_dir) / f"{ep_dir.name}.json"
                with open(save_path, "w") as f:
                    json.dump(out["structured"], f, indent=2, ensure_ascii=False)

            # Optional affordance visualization
            if args.visualize:
                viz_dir = Path(args.output_dir) if args.output_dir else ep_dir
                viz_path = viz_dir / f"{ep_dir.name}_viz.png"
                try:
                    render_inference(image, out["structured"], args.task, viz_path)
                    print(f"  viz -> {viz_path}")
                except Exception as e:
                    print(f"  WARN visualize failed for {ep_dir.name}: {e}")

    elif args.image:
        out = run_single(model, args.image, args.depth,
                         args.text or "describe the scene", args.task,
                         args.temperature, args.max_tokens)
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
                print(f"WARN visualize failed: {e}")
    else:
        parser.print_help()
        sys.exit(1)

    # Save all results
    if args.output_dir and results:
        with open(Path(args.output_dir) / "all_results.json", "w") as f:
            json.dump([r["structured"] for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
