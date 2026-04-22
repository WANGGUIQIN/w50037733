#!/usr/bin/env python3
"""Regenerate plan.json using keyframe sequence for grounded affordance & semantics.

Key improvements over generate_plans.py:
  1. Sends ALL keyframe images (rgb_0..rgb_N) — GPT sees the full temporal process
  2. Affordance coordinates are grounded in actual object positions in the images
  3. Temporal context lets GPT better understand WHAT the task involves
  4. Step count is decided by GPT's own reasoning (NOT forced to keyframe count)
  5. Backs up original plan.json before overwriting

Usage:
    # Regenerate all episodes
    python scripts/regenerate_plans_keyframe.py --dataset rlbench

    # Regenerate only episodes with bad affordance coordinates
    python scripts/regenerate_plans_keyframe.py --dataset rlbench --filter-bad

    # Dry run: audit quality without regenerating
    python scripts/regenerate_plans_keyframe.py --dataset rlbench --audit-only

    # Resume (skip already-regenerated episodes)
    python scripts/regenerate_plans_keyframe.py --dataset rlbench --resume

    # Specify range
    python scripts/regenerate_plans_keyframe.py --dataset rlbench --start 0 --end 100 --workers 8
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not set. "
        "Export it before running this script: export OPENAI_API_KEY=sk-..."
    )
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://yunwu.ai/v1")
MODEL = os.environ.get("PLAN_MODEL", "gpt-4o-mini")
DATA_ROOT = Path(__file__).parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Keyframe-aware prompt  (affordance grounding + semantic enhancement)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are RoboBrain, an embodied AI assistant specialized in robotic manipulation planning.

You will receive a SEQUENCE of keyframe images from a robot manipulation episode, \
numbered Frame 0 (initial state) through Frame N (task complete). These keyframes \
capture the critical moments of the task execution so you can observe the full \
temporal process.

Your job:
1. SCENE ANALYSIS — Use Frame 0 to identify all task-relevant objects and their \
positions.  Use later frames to understand how the scene evolves.
2. TASK DECOMPOSITION — Break the task into 2-6 sequential operation primitives. \
Decide the number of steps based on the TASK SEMANTICS, not the number of frames.
3. AFFORDANCE GROUNDING — For EVERY step, look at the images and report the \
normalised image coordinate [u, v] where the robot should act:
   * u = horizontal (0 = left edge, 1 = right edge)
   * v = vertical   (0 = top edge,  1 = bottom edge)
   * Locate the TARGET OBJECT in Frame 0 (or the frame where the action starts) \
and set the affordance point on that object.
   * Different steps that act on DIFFERENT objects MUST have DIFFERENT affordance \
coordinates.
   * Even steps that act on the SAME object may need slightly different coordinates \
(e.g. grasp the lid ≠ place the lid on the jar).
4. APPROACH DIRECTION — Infer from the images how the gripper approaches.  \
Default top-down is [0, 0, -1]; lateral approach could be [0, 1, 0], etc.
5. CONSTRAINT SPECIFICATION — For each step, list constraints by category.

Available operation primitives: reach, grasp, transport, place, push, pull, \
insert, pour, rotate, release, flip, wipe.

Constraint categories:
- contact: gripper_contact, gripper_state, gripper_width, holding, released, \
surface_contact, inserted
- spatial: distance, above, below, inside, aligned_xy, aligned_z, height, \
on_surface, clear_path, near
- pose: upright, tilt, tilted, stable, level, orientation_match
- direction: approach_dir, grasp_axis, motion_dir, insert_dir, retreat_dir
- safety: no_collision, within_workspace, no_spill, no_drop, force_limit, \
support_stable

Roles: completion (advance when satisfied), safety (correct when violated), \
progress (replan when stagnant).

Output ONLY valid JSON. No markdown, no explanation, no code fences."""

# Few-shot: note different affordance coords for different targets
FEW_SHOT_EXAMPLE = """\
Task: "Pick up the red block and place it on the blue plate"
(5 keyframes provided — Frame 0 initial, Frame 4 done)

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
      "affordance": [0.35, 0.50], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "gripper_contact", "args": ["red_block"], "role": "completion"}, {"pred": "holding", "args": ["red_block"], "role": "completion"}],
        "direction": [{"pred": "grasp_axis", "args": [0, 0, -1], "role": "safety"}]
      },
      "done_when": "holding(red_block)"
    },
    {
      "step": 3, "action": "transport", "target": "red_block", "destination": "blue_plate",
      "affordance": [0.62, 0.40], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "holding", "args": ["red_block"], "role": "safety"}],
        "spatial": [{"pred": "above", "args": ["red_block", "blue_plate", 0.05], "role": "completion"}],
        "safety": [{"pred": "no_collision", "args": ["red_block", "blue_plate"]}, {"pred": "no_drop", "args": ["red_block"]}]
      },
      "done_when": "above(red_block, blue_plate, 0.05) AND aligned_xy(red_block, blue_plate, 0.03)"
    },
    {
      "step": 4, "action": "place", "target": "red_block", "destination": "blue_plate",
      "affordance": [0.62, 0.45], "approach": [0.0, 0.0, -1.0],
      "constraints": {
        "contact": [{"pred": "surface_contact", "args": ["red_block", "blue_plate"], "role": "completion"}, {"pred": "released", "args": ["red_block"], "role": "completion"}],
        "pose": [{"pred": "stable", "args": ["red_block"], "role": "completion"}],
        "safety": [{"pred": "support_stable", "args": ["red_block"]}]
      },
      "done_when": "surface_contact(red_block, blue_plate) AND released(red_block) AND stable(red_block)"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """\
Example of the expected output format:

{example}

Now look at the keyframe sequence below and generate a manipulation plan.

Task: "{task}"

The images show the full execution process: Frame 0 is the initial state, \
Frame {last_frame} is the completed state. Use these images to:
- Identify where each object is located in the image (for affordance coordinates)
- Understand what physical actions occur during the task
- Decide how many steps are needed based on the TASK itself (not the frame count)

CRITICAL RULES for affordance coordinates:
- Look at Frame 0 and FIND the target object visually — report its [u, v] position
- If a step targets a different location (e.g. place destination), use THAT location's coords
- NEVER use the same [u, v] for all steps unless every step truly acts on the same pixel
- u,v must be in [0, 1] range

Output ONLY the JSON object. No markdown code fences, no explanation."""


# ---------------------------------------------------------------------------
# Quality audit  (focuses on affordance quality, not step-count alignment)
# ---------------------------------------------------------------------------

DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}


def _affordance_key(step: dict) -> str:
    """Which object the affordance should point at: destination for placing-type actions."""
    action = step.get("action", "")
    dest = step.get("destination")
    if action in DESTINATION_ACTIONS and dest:
        return dest
    return step.get("target", "")


def audit_plan(plan: dict, meta: dict) -> list[str]:
    """Check plan quality, return list of issues found."""
    issues = []

    steps = plan.get("steps", [])
    actual_steps = len(steps)

    # 1. Cross-target identical affordances — different target objects sharing the
    #    same coordinate (genuine bug). Same target across multiple steps sharing
    #    a coordinate is correct (same physical object → same pixel).
    aff_to_targets: dict[tuple, set[str]] = {}
    for step in steps:
        aff = step.get("affordance")
        if not aff:
            continue
        key = _affordance_key(step)
        aff_t = tuple(round(v, 3) for v in aff)
        aff_to_targets.setdefault(aff_t, set()).add(key)

    cross_target = [
        (aff, targets)
        for aff, targets in aff_to_targets.items()
        if len(targets) > 1
    ]
    if cross_target:
        sample = cross_target[0]
        issues.append(
            f"cross_target_identical_affordances: {len(cross_target)} "
            f"coordinate(s) shared across different targets "
            f"(e.g. {sample[0]} used by {sorted(sample[1])})"
        )

    # 2. num_steps field vs actual step list
    declared = plan.get("num_steps", 0)
    if declared != actual_steps:
        issues.append(f"num_steps_field_mismatch: declared {declared}, actual {actual_steps}")

    # 3. Missing required fields per step
    for step in steps:
        sn = step.get("step", "?")
        if "action" not in step:
            issues.append(f"step_{sn}_missing_action")
        if "target" not in step:
            issues.append(f"step_{sn}_missing_target")
        if "affordance" not in step:
            issues.append(f"step_{sn}_missing_affordance")

    # 4. Affordance out of [0, 1]
    for step in steps:
        aff = step.get("affordance", [])
        if aff and any(v < 0 or v > 1 for v in aff):
            issues.append(f"step_{step.get('step','?')}_affordance_out_of_range: {aff}")

    # 5. Zero or too many steps
    if actual_steps == 0:
        issues.append("no_steps")
    elif actual_steps > 8:
        issues.append(f"too_many_steps: {actual_steps}")

    return issues


def run_audit(ds_dir: Path) -> dict:
    """Audit all episodes, return summary."""
    episodes = sorted([d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])

    stats = {
        "total": len(episodes),
        "has_plan": 0,
        "clean": 0,
        "cross_target_identical_affordances": 0,
        "missing_affordance": 0,
        "other_issues": 0,
        "bad_episodes": [],
    }

    for ep_dir in episodes:
        plan_path = ep_dir / "plan.json"
        meta_path = ep_dir / "meta.json"

        if not plan_path.exists() or not meta_path.exists():
            continue

        stats["has_plan"] += 1

        with open(plan_path) as f:
            plan = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        issues = audit_plan(plan, meta)

        if not issues:
            stats["clean"] += 1
        else:
            for issue in issues:
                if "cross_target_identical_affordances" in issue:
                    stats["cross_target_identical_affordances"] += 1
                elif "missing_affordance" in issue:
                    stats["missing_affordance"] += 1
                else:
                    stats["other_issues"] += 1
            stats["bad_episodes"].append((ep_dir.name, issues))

    return stats


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json(text: str) -> dict | None:
    """Extract JSON from model output."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_plan(plan: dict) -> bool:
    """Validate basic plan structure (no step-count constraint)."""
    if not isinstance(plan, dict):
        return False
    if "task" not in plan or "steps" not in plan:
        return False
    if not isinstance(plan["steps"], list) or len(plan["steps"]) == 0:
        return False

    # Normalise scene_objects
    if "scene_objects" in plan:
        if isinstance(plan["scene_objects"], list) and plan["scene_objects"]:
            if isinstance(plan["scene_objects"][0], dict):
                plan["scene_objects"] = [
                    obj.get("name", str(obj)) if isinstance(obj, dict) else str(obj)
                    for obj in plan["scene_objects"]
                ]

    for step in plan["steps"]:
        if "action" not in step or "target" not in step:
            return False
    return True


def normalise_plan(plan: dict) -> dict:
    """Fix step numbering and num_steps field."""
    for i, step in enumerate(plan["steps"]):
        step["step"] = i + 1
    plan["num_steps"] = len(plan["steps"])
    return plan


def generate_plan_keyframe(
    client: OpenAI,
    ep_dir: Path,
    meta: dict,
    retries: int = 2,
) -> dict | None:
    """Generate plan using ALL keyframe images for grounded affordance."""
    task = meta.get("task", "manipulation")
    num_kf = meta.get("num_keyframes", 5)

    # Build multimodal content: Frame 0 … Frame N
    content_parts = []
    for i in range(num_kf):
        img_path = ep_dir / f"rgb_{i}.png"
        if not img_path.exists():
            break
        img_b64 = encode_image(str(img_path))
        content_parts.append({"type": "text", "text": f"Frame {i}:"})
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    user_text = USER_PROMPT_TEMPLATE.format(
        example=FEW_SHOT_EXAMPLE,
        task=task,
        last_frame=num_kf - 1,
    )
    content_parts.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=2048,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content
            plan = extract_json(raw)

            if plan and validate_plan(plan):
                plan["task"] = task
                plan = normalise_plan(plan)
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


def process_episode(
    client: OpenAI,
    ep_dir: Path,
    backup: bool = True,
) -> tuple[str, bool, list[str]]:
    """Regenerate plan.json for one episode.

    Returns (episode_id, success, old_issues).
    """
    meta_path = ep_dir / "meta.json"
    plan_path = ep_dir / "plan.json"

    with open(meta_path) as f:
        meta = json.load(f)

    # Audit old plan
    old_issues = []
    if plan_path.exists():
        with open(plan_path) as f:
            old_plan = json.load(f)
        old_issues = audit_plan(old_plan, meta)

        if backup:
            backup_path = ep_dir / "plan_original.json"
            if not backup_path.exists():
                shutil.copy2(plan_path, backup_path)

    plan = generate_plan_keyframe(client, ep_dir, meta)
    if plan is None:
        return (ep_dir.name, False, old_issues)

    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return (ep_dir.name, True, old_issues)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="Regenerate plan.json — keyframe-grounded affordance & semantics"
    )
    parser.add_argument("--dataset", default="rlbench",
                        help="Single-dataset mode (ignored when --splits is set)")
    parser.add_argument("--splits", default=None,
                        help="Path to splits JSON (e.g. data/splits/train.json)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-backup", action="store_true",
                        help="Don't backup original plan.json")
    parser.add_argument("--filter-bad", action="store_true",
                        help="Only regenerate episodes with identical affordances")
    parser.add_argument("--audit-only", action="store_true",
                        help="Just audit quality, don't regenerate")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes that already have plan.json (fresh-generation mode)")
    args = parser.parse_args()

    if args.model:
        MODEL = args.model

    # Determine episode source
    if args.splits:
        split_path = Path(args.splits)
        if not split_path.exists():
            print(f"ERROR: splits file not found: {split_path}")
            sys.exit(1)
        with open(split_path) as f:
            split_data = json.load(f)
        episodes = [DATA_ROOT / e["dataset"] / e["episode_id"] for e in split_data["episodes"]]
        episodes = [e for e in episodes if e.exists()]
        ds_dir = None  # audit mode requires single ds; disabled with --splits
        source_desc = f"splits={split_path.name}"
    else:
        ds_dir = DATA_ROOT / args.dataset
        if not ds_dir.exists():
            print(f"ERROR: {ds_dir} not found")
            sys.exit(1)
        source_desc = f"dataset={args.dataset}"

    # --- Audit-only mode ---
    if args.audit_only:
        if ds_dir is None:
            print("ERROR: --audit-only requires --dataset (not --splits)")
            sys.exit(1)
        print(f"Auditing {ds_dir} ...")
        stats = run_audit(ds_dir)
        print(f"\n{'='*60}")
        print(f"Quality Audit Report")
        print(f"{'='*60}")
        print(f"Total episodes:          {stats['total']}")
        print(f"With plan.json:          {stats['has_plan']}")
        print(f"Clean (no issues):       {stats['clean']}  ({stats['clean']/max(stats['has_plan'],1)*100:.1f}%)")
        print(f"Cross-target identical aff: {stats['cross_target_identical_affordances']}  ({stats['cross_target_identical_affordances']/max(stats['has_plan'],1)*100:.1f}%)")
        print(f"Missing affordance:      {stats['missing_affordance']}")
        print(f"Other issues:            {stats['other_issues']}")
        print()
        if stats["bad_episodes"]:
            print("Sample bad episodes:")
            for ep_name, issues in stats["bad_episodes"][:20]:
                print(f"  {ep_name}: {'; '.join(issues)}")
        return

    # --- Collect episodes ---
    if ds_dir is not None:
        episodes = sorted([
            d for d in ds_dir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ])
    # else: episodes came from --splits above

    if args.end > 0:
        episodes = episodes[args.start:args.end]
    else:
        episodes = episodes[args.start:]

    if args.resume:
        # Skip episodes that already have plan.json (fresh-generation semantics)
        episodes = [ep for ep in episodes if not (ep / "plan.json").exists()]

    if args.filter_bad:
        bad_eps = []
        for ep_dir in episodes:
            pp = ep_dir / "plan.json"
            mp = ep_dir / "meta.json"
            if not pp.exists() or not mp.exists():
                continue
            with open(pp) as f:
                plan = json.load(f)
            with open(mp) as f:
                meta = json.load(f)
            if audit_plan(plan, meta):
                bad_eps.append(ep_dir)
        print(f"Filtered: {len(bad_eps)} bad out of {len(episodes)} total")
        episodes = bad_eps

    print(f"Source: {source_desc}")
    print(f"Episodes to process: {len(episodes)}")
    print(f"Model: {MODEL}")
    print(f"Workers: {args.workers}")
    print(f"Backup: {not args.no_backup}")
    print()

    if not episodes:
        print("Nothing to do.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    success = 0
    failed = 0
    issues_fixed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_episode, client, ep, not args.no_backup): ep
            for ep in episodes
        }
        for i, future in enumerate(as_completed(futures), 1):
            ep_id, ok, old_issues = future.result()
            if ok:
                success += 1
                if old_issues:
                    issues_fixed += len(old_issues)
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
                    f"issues_fixed={issues_fixed} "
                    f"rate={rate:.1f}/s "
                    f"ETA={eta/60:.0f}min"
                )

    elapsed = time.time() - t0
    print(f"\nDone: {success} ok, {failed} fail, "
          f"{issues_fixed} issues addressed, {elapsed:.0f}s")

    # Post audit
    print("\nPost-regeneration audit...")
    stats = run_audit(ds_dir)
    print(f"Clean: {stats['clean']}/{stats['has_plan']} "
          f"({stats['clean']/max(stats['has_plan'],1)*100:.1f}%)")
    if stats["cross_target_identical_affordances"]:
        print(f"Remaining cross-target identical affordances: {stats['cross_target_identical_affordances']}")


if __name__ == "__main__":
    main()
