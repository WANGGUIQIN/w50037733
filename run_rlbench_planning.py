#!/usr/bin/env python3
"""Run RoboBrain2.5 task planning on an RLBench scene.

Loads a local RLBench episode, sends the scene image + prompt-template
planning prompt to RoboBrain2.5, and saves the JSON output and scene
images to a new output folder.

Usage:
    CUDA_VISIBLE_DEVICES=3 python run_rlbench_planning.py \
        --episode episode_000300 \
        --output_dir ./output_rlbench_planning
"""

import argparse
import json
import os
import re
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# Prompt template (from docs/plans/prompt-template.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are RoboBrain, an embodied AI assistant with 3D spatial understanding "
    "specialized in robotic manipulation planning.\n\n"
    "Your job: given a scene image and a task instruction, output a structured "
    "task decomposition with operation primitives and per-stage geometric "
    "constraints.\n\n"
    "You must follow a 3-step reasoning process:\n"
    "1. SCENE ANALYSIS: Identify all task-relevant objects, their spatial "
    "relations, and physical properties (fragile, liquid-containing, "
    "articulated, etc.)\n"
    "2. TASK DECOMPOSITION: Break the task into 2-5 sequential operation "
    "primitives at manipulation-semantic granularity (not low-level motor "
    "commands, not high-level goals)\n"
    "3. CONSTRAINT SPECIFICATION: For each stage, specify geometric constraints "
    "organized by physical category (contact, spatial, pose, direction, safety) "
    "with runtime role labels (completion, safety, progress)\n\n"
    "Granularity principle: each stage should correspond to ONE nameable "
    "manipulation verb (reach, grasp, transport, place, etc.) where the scene "
    "undergoes an observable state change upon completion.\n\n"
    "Available operation primitives: reach, grasp, transport, place, push, pull, "
    "insert, pour, rotate, release, flip, wipe.\n\n"
    "Output format: structured JSON with fields: task, scene_objects, num_steps, "
    "steps (each with step, action, target, destination, affordance [u,v], "
    "approach [x,y,z], constraints by category, done_when)."
)

USER_PROMPT_TEMPLATE = (
    'Task: "{task_description}"\n\n'
    "Analyze the scene and plan the manipulation steps. For each step, specify:\n"
    "- Operation primitive and target object\n"
    "- Affordance point [u, v] (normalized image coordinates, 0-1)\n"
    "- Approach direction [x, y, z] (unit vector in camera frame)\n"
    "- Constraints organized by category with role labels\n"
    "- Completion condition (done_when)\n\n"
    "Output as structured JSON."
)


def load_episode(rlbench_root: str, episode_id: str):
    """Load an RLBench episode: meta.json + RGB images."""
    ep_dir = os.path.join(rlbench_root, episode_id)
    if not os.path.isdir(ep_dir):
        raise FileNotFoundError(f"Episode not found: {ep_dir}")

    with open(os.path.join(ep_dir, "meta.json")) as f:
        meta = json.load(f)

    # Collect all RGB images
    rgb_files = sorted(
        [os.path.join(ep_dir, f) for f in os.listdir(ep_dir) if f.startswith("rgb_") and f.endswith(".png")]
    )
    return meta, rgb_files


def run_planning(model, processor, image_path: str, task_description: str):
    """Run the planning prompt through RoboBrain2.5."""
    user_text = USER_PROMPT_TEMPLATE.format(task_description=task_description)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Tokenize
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Generate
    print("Running inference ...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def extract_json(text: str) -> dict | None:
    """Try to extract JSON from model output."""
    # Try to find JSON block in markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="RoboBrain2.5 RLBench Planning")
    parser.add_argument(
        "--episode", default="episode_000300",
        help="Episode ID (default: episode_000300 — 'take the steak off the grill')",
    )
    parser.add_argument(
        "--rlbench_root",
        default=os.path.join(os.path.dirname(__file__), "data/processed/rlbench"),
        help="Path to processed RLBench data",
    )
    parser.add_argument(
        "--model_path",
        default="/home/edge/Embodied/models/RoboBrain2.5-8B-NV",
        help="Path to local RoboBrain2.5 model",
    )
    parser.add_argument(
        "--output_dir", default="./output_rlbench_planning",
        help="Output directory for JSON and images",
    )
    parser.add_argument(
        "--keyframe", type=int, default=0,
        help="Which keyframe image to use (default: 0 = initial scene)",
    )
    args = parser.parse_args()

    # 1. Load episode
    print(f"Loading episode: {args.episode}")
    meta, rgb_files = load_episode(args.rlbench_root, args.episode)
    task = meta["task"]
    print(f"Task: {task}")
    print(f"Found {len(rgb_files)} RGB images")

    image_path = rgb_files[args.keyframe]
    print(f"Using keyframe {args.keyframe}: {image_path}")

    # 2. Load model
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded.")

    # 3. Run planning inference
    raw_output = run_planning(model, processor, image_path, task)
    print(f"\n{'='*50}")
    print("RAW MODEL OUTPUT:")
    print(f"{'='*50}")
    print(raw_output)
    print(f"{'='*50}\n")

    # 4. Create output directory
    out_dir = os.path.join(args.output_dir, args.episode)
    os.makedirs(out_dir, exist_ok=True)

    # 5. Save scene images
    for rgb_file in rgb_files:
        dst = os.path.join(out_dir, os.path.basename(rgb_file))
        shutil.copy2(rgb_file, dst)
        print(f"Saved image: {dst}")

    # 6. Save raw output
    raw_path = os.path.join(out_dir, "raw_output.txt")
    with open(raw_path, "w") as f:
        f.write(raw_output)
    print(f"Saved raw output: {raw_path}")

    # 7. Try to parse and save structured JSON
    parsed = extract_json(raw_output)
    if parsed:
        plan_path = os.path.join(out_dir, "plan.json")
        with open(plan_path, "w") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)
        print(f"Saved structured plan: {plan_path}")
    else:
        print("Warning: Could not extract structured JSON from model output.")
        print("Raw output saved — you may need to manually format it.")

    # 8. Save meta info
    run_meta = {
        "episode_id": args.episode,
        "task": task,
        "model": args.model_path,
        "keyframe_used": args.keyframe,
        "source_image": image_path,
        "episode_meta": meta,
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved run meta: {meta_path}")

    print(f"\nDone! All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
