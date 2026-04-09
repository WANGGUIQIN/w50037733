#!/usr/bin/env python3
"""Batch-generate plan.json for episodes using GPT vision API.

Sends scene image + task description + few-shot prompt template to GPT,
parses structured JSON output, saves as plan.json in each episode directory.

Usage:
    # Generate for all RLBench episodes
    python scripts/generate_plans.py --dataset rlbench

    # Generate for specific episodes
    python scripts/generate_plans.py --dataset rlbench --start 0 --end 100

    # Dry run (1 episode)
    python scripts/generate_plans.py --dataset rlbench --end 1

    # Resume (skip episodes that already have plan.json)
    python scripts/generate_plans.py --dataset rlbench --resume

    # Use multiple workers
    python scripts/generate_plans.py --dataset rlbench --workers 8
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "sk-RcVDkxbG6OBXrK1FlLC6fDdXsFz1LBTeSXnRLSjeSuL91MW5",
)
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
MODEL = os.environ.get("PLAN_MODEL", "gpt-4o-mini")
DATA_ROOT = Path(__file__).parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Prompt template (from docs/plans/prompt-template.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are RoboBrain, an embodied AI assistant with 3D spatial understanding specialized in robotic manipulation planning.

Your job: given a scene image and a task instruction, output a structured task decomposition with operation primitives and per-stage geometric constraints.

You must follow a 3-step reasoning process:
1. SCENE ANALYSIS: Identify all task-relevant objects, their spatial relations, and physical properties (fragile, liquid-containing, articulated, etc.)
2. TASK DECOMPOSITION: Break the task into 2-5 sequential operation primitives at manipulation-semantic granularity (not low-level motor commands, not high-level goals)
3. CONSTRAINT SPECIFICATION: For each stage, specify geometric constraints organized by physical category (contact, spatial, pose, direction, safety) with runtime role labels (completion, safety, progress)

Granularity principle: each stage should correspond to ONE nameable manipulation verb (reach, grasp, transport, place, push, pull, insert, pour, rotate, release, flip, wipe) where the scene undergoes an observable state change upon completion.

Available operation primitives: reach, grasp, transport, place, push, pull, insert, pour, rotate, release, flip, wipe.

Constraint categories and roles:
- contact: gripper_contact, gripper_state, gripper_width, holding, released, surface_contact, inserted
- spatial: distance, above, below, inside, aligned_xy, aligned_z, height, on_surface, clear_path, near
- pose: upright, tilt, tilted, stable, level, orientation_match
- direction: approach_dir, grasp_axis, motion_dir, insert_dir, retreat_dir
- safety: no_collision, within_workspace, no_spill, no_drop, force_limit, support_stable

Roles: completion (advance when satisfied), safety (correct when violated), progress (replan when stagnant).

Output ONLY valid JSON. No markdown, no explanation, no code fences."""

# Two few-shot examples (condensed to save tokens)
FEW_SHOT_EXAMPLE_1 = """\
Task: "Pick up the red block and place it on the blue plate"

{
  "task": "Pick up the red block and place it on the blue plate",
  "scene_objects": ["red_block", "blue_plate", "table"],
  "num_steps": 4,
  "steps": [
    {
      "step": 1, "action": "reach", "target": "red_block",
      "affordance": [0.35, 0.48], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "gripper_state", "args": ["open"], "role": "progress"}],
        "spatial": [{"pred": "distance", "args": ["gripper", "red_block", "<", 0.03], "role": "completion"}],
        "safety": [{"pred": "no_collision", "args": ["gripper", "blue_plate"]}]
      },
      "done_when": "distance(gripper, red_block) < 0.03 AND gripper_state(open)"
    },
    {
      "step": 2, "action": "grasp", "target": "red_block",
      "affordance": [0.35, 0.48], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "gripper_contact", "args": ["red_block"], "role": "completion"}, {"pred": "holding", "args": ["red_block"], "role": "completion"}],
        "direction": [{"pred": "grasp_axis", "args": [0, 0, -1], "role": "safety"}]
      },
      "done_when": "holding(red_block)"
    },
    {
      "step": 3, "action": "transport", "target": "red_block", "destination": "blue_plate",
      "affordance": [0.62, 0.55], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "holding", "args": ["red_block"], "role": "safety"}],
        "spatial": [{"pred": "above", "args": ["red_block", "blue_plate", 0.05], "role": "completion"}, {"pred": "aligned_xy", "args": ["red_block", "blue_plate", 0.03], "role": "completion"}],
        "safety": [{"pred": "no_collision", "args": ["red_block", "blue_plate"]}, {"pred": "no_drop", "args": ["red_block"]}]
      },
      "done_when": "above(red_block, blue_plate, 0.05) AND aligned_xy(red_block, blue_plate, 0.03)"
    },
    {
      "step": 4, "action": "place", "target": "red_block", "destination": "blue_plate",
      "affordance": [0.62, 0.55], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "surface_contact", "args": ["red_block", "blue_plate"], "role": "completion"}, {"pred": "released", "args": ["red_block"], "role": "completion"}],
        "pose": [{"pred": "stable", "args": ["red_block"], "role": "completion"}],
        "safety": [{"pred": "support_stable", "args": ["red_block"]}]
      },
      "done_when": "surface_contact(red_block, blue_plate) AND released(red_block) AND stable(red_block)"
    }
  ]
}"""

FEW_SHOT_EXAMPLE_2 = """\
Task: "Open the top drawer"

{
  "task": "Open the top drawer",
  "scene_objects": ["drawer_handle", "drawer", "cabinet"],
  "num_steps": 3,
  "steps": [
    {
      "step": 1, "action": "reach", "target": "drawer_handle",
      "affordance": [0.50, 0.30], "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [{"pred": "gripper_state", "args": ["open"], "role": "progress"}],
        "spatial": [{"pred": "distance", "args": ["gripper", "drawer_handle", "<", 0.03], "role": "completion"}],
        "direction": [{"pred": "approach_dir", "args": [0, 1, 0], "role": "safety"}]
      },
      "done_when": "distance(gripper, drawer_handle) < 0.03"
    },
    {
      "step": 2, "action": "grasp", "target": "drawer_handle",
      "affordance": [0.50, 0.30], "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [{"pred": "gripper_contact", "args": ["drawer_handle"], "role": "completion"}, {"pred": "holding", "args": ["drawer_handle"], "role": "completion"}],
        "direction": [{"pred": "grasp_axis", "args": [1, 0, 0], "role": "safety"}]
      },
      "done_when": "holding(drawer_handle)"
    },
    {
      "step": 3, "action": "pull", "target": "drawer",
      "affordance": [0.50, 0.30], "approach": [0.0, 1.0, 0.0],
      "constraints": {
        "contact": [{"pred": "holding", "args": ["drawer_handle"], "role": "safety"}],
        "spatial": [{"pred": "distance", "args": ["drawer", "cabinet", ">", 0.20], "role": "completion"}],
        "direction": [{"pred": "motion_dir", "args": [0, 1, 0], "role": "progress"}],
        "safety": [{"pred": "within_workspace", "args": ["gripper"]}]
      },
      "done_when": "distance(drawer, cabinet) > 0.20"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """\
Here are two examples of the expected output format:

Example 1:
{example1}

Example 2:
{example2}

Now analyze the scene image and generate a plan for:
Task: "{task}"

Output ONLY the JSON object. No markdown code fences, no explanation."""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json(text: str) -> dict | None:
    """Extract JSON from model output, handling code fences."""
    # Strip markdown fences if present
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)

    # Try to find JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_plan(plan: dict) -> bool:
    """Check that plan has the required structure."""
    if not isinstance(plan, dict):
        return False
    if "task" not in plan or "steps" not in plan:
        return False
    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        return False
    # Check scene_objects is a string list (not dict list)
    if "scene_objects" in plan:
        if isinstance(plan["scene_objects"], list) and plan["scene_objects"]:
            if isinstance(plan["scene_objects"][0], dict):
                # Fix: extract names from dict list
                plan["scene_objects"] = [
                    obj.get("name", str(obj)) if isinstance(obj, dict) else str(obj)
                    for obj in plan["scene_objects"]
                ]
    for step in plan["steps"]:
        if "action" not in step or "target" not in step:
            return False
    return True


def generate_plan(client: OpenAI, image_path: str, task: str, retries: int = 2) -> dict | None:
    """Call GPT to generate a plan for one episode."""
    img_b64 = encode_image(image_path)

    user_text = USER_PROMPT_TEMPLATE.format(
        example1=FEW_SHOT_EXAMPLE_1,
        example2=FEW_SHOT_EXAMPLE_2,
        task=task,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=2048,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content
            plan = extract_json(raw)

            if plan and validate_plan(plan):
                # Ensure task field matches
                plan["task"] = task
                if "num_steps" not in plan:
                    plan["num_steps"] = len(plan["steps"])
                return plan

            if attempt < retries:
                time.sleep(1)
                continue
            return None

        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            print(f"    ERROR: {e}")
            return None


def process_episode(client: OpenAI, ep_dir: Path) -> tuple[str, bool]:
    """Generate plan.json for one episode. Returns (episode_id, success)."""
    meta_path = ep_dir / "meta.json"
    plan_path = ep_dir / "plan.json"

    with open(meta_path) as f:
        meta = json.load(f)

    task = meta.get("task", "manipulation")
    image_path = ep_dir / "rgb_0.png"

    if not image_path.exists():
        return (ep_dir.name, False)

    plan = generate_plan(client, str(image_path), task)
    if plan is None:
        return (ep_dir.name, False)

    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return (ep_dir.name, True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch generate plan.json via GPT API")
    parser.add_argument("--dataset", default="rlbench", help="Dataset name under data/processed/")
    parser.add_argument("--start", type=int, default=0, help="Start episode index")
    parser.add_argument("--end", type=int, default=-1, help="End episode index (-1 = all)")
    parser.add_argument("--resume", action="store_true", help="Skip episodes with existing plan.json")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API workers")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    global MODEL
    if args.model:
        MODEL = args.model

    ds_dir = DATA_ROOT / args.dataset
    if not ds_dir.exists():
        print(f"ERROR: {ds_dir} not found")
        sys.exit(1)

    # Collect episodes
    episodes = sorted([d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if args.end > 0:
        episodes = episodes[args.start:args.end]
    else:
        episodes = episodes[args.start:]

    if args.resume:
        episodes = [ep for ep in episodes if not (ep / "plan.json").exists()]

    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {len(episodes)} (start={args.start}, end={args.end})")
    print(f"Model: {MODEL}")
    print(f"Workers: {args.workers}")
    print(f"Resume: {args.resume}")
    print()

    if not episodes:
        print("Nothing to do.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    success = 0
    failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_episode, client, ep): ep for ep in episodes}

        for i, future in enumerate(as_completed(futures), 1):
            ep_id, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  FAILED: {ep_id}")

            if i % 20 == 0 or i == len(episodes):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(episodes) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i}/{len(episodes)}] "
                    f"ok={success} fail={failed} "
                    f"rate={rate:.1f}/s "
                    f"ETA={eta/60:.0f}min"
                )

    elapsed = time.time() - t0
    print(f"\nDone: {success} success, {failed} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
