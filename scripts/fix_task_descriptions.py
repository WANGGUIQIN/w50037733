#!/usr/bin/env python3
"""Fix task descriptions in processed dataset meta.json files.

Fixes:
  1. DROID (34,404 ep): backfill from droid_episode_tasks.json
     ~48% episodes have fallback "robotic manipulation" — replace with real descriptions
  2. ALOHA (20 ep): expand short task names to full manipulation descriptions
     e.g. "battery" → "Pick up the battery and insert it into the remote controller"
  3. Incomplete episodes: detect and optionally remove rgb/depth mismatches

Usage:
    conda run -n robobrain_3dgs python scripts/fix_task_descriptions.py --all
    conda run -n robobrain_3dgs python scripts/fix_task_descriptions.py --droid
    conda run -n robobrain_3dgs python scripts/fix_task_descriptions.py --clean
    conda run -n robobrain_3dgs python scripts/fix_task_descriptions.py --clean --dry-run
"""
import argparse
import json
import os
import shutil
from pathlib import Path

PROCESSED_DIR = Path("/home/frontier/Embodied/robobrain_3dgs/data/processed")
DATA_DIR = Path("/home/frontier/Embodied/robobrain_3dgs/data")


# ── DROID ────────────────────────────────────────────────────────────────────

def fix_droid_tasks():
    """Backfill DROID task descriptions from droid_episode_tasks.json."""
    print("=" * 60)
    print("  DROID: Backfilling task descriptions")
    print("=" * 60)

    mapping_path = DATA_DIR / "droid_episode_tasks.json"
    if not mapping_path.exists():
        print("  ERROR: droid_episode_tasks.json not found")
        return {"fixed": 0, "already_good": 0, "no_mapping": 0, "total": 0}

    with open(mapping_path) as f:
        task_map = json.load(f)
    print(f"  Loaded {len(task_map)} task mappings")

    droid_dir = PROCESSED_DIR / "droid"
    if not droid_dir.exists():
        print("  ERROR: droid directory not found")
        return {"fixed": 0, "already_good": 0, "no_mapping": 0, "total": 0}

    episodes = sorted([e for e in droid_dir.iterdir()
                        if e.is_dir() and e.name.startswith("episode_")])
    fixed = 0
    already_good = 0
    no_mapping = 0

    for ep_dir in episodes:
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        current_task = meta.get("task", "")

        # Only fix fallback entries
        if current_task not in ("robotic manipulation", "manipulation", ""):
            already_good += 1
            continue

        # episode_000006 -> "6"
        ep_idx = str(int(ep_dir.name.split("_")[1]))

        if ep_idx in task_map and task_map[ep_idx].strip():
            meta["task"] = task_map[ep_idx].strip()
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            fixed += 1
        else:
            no_mapping += 1

    result = {"fixed": fixed, "already_good": already_good,
              "no_mapping": no_mapping, "total": len(episodes)}
    print(f"  Fixed:      {fixed}")
    print(f"  Already OK: {already_good}")
    print(f"  No mapping: {no_mapping}")
    print(f"  Total:      {len(episodes)}")
    return result


# ── ALOHA ────────────────────────────────────────────────────────────────────

ALOHA_TASK_MAP = {
    "battery":        "Pick up the battery and insert it into the remote controller",
    "candy":          "Pick up the candy and place it into the wrapper",
    "coffee":         "Pick up the coffee pod and insert it into the coffee machine",
    "coffee new":     "Insert the coffee capsule into the machine and close the lid",
    "coffee_new":     "Insert the coffee capsule into the machine and close the lid",
    "cups open":      "Grasp the cup and remove its lid",
    "cups_open":      "Grasp the cup and remove its lid",
    "fork pick up":   "Pick up the fork from the table",
    "fork_pick_up":   "Pick up the fork from the table",
    "screw driver":   "Pick up the screwdriver and use it to tighten the screw",
    "screw_driver":   "Pick up the screwdriver and use it to tighten the screw",
    "tape":           "Pick up the tape dispenser and apply tape to the surface",
    "thread velcro":  "Thread the velcro strap through the slot and fasten it",
    "thread_velcro":  "Thread the velcro strap through the slot and fasten it",
    "towel":          "Pick up the towel and fold it on the table",
    "vinh cup":       "Pick up the cup and place it on the coaster",
    "vinh_cup":       "Pick up the cup and place it on the coaster",
    "vinh cup left":  "Pick up the cup with the left hand and place it on the coaster",
    "vinh_cup_left":  "Pick up the cup with the left hand and place it on the coaster",
    "ziploc slide":   "Slide the ziploc bag seal to close it",
    "ziploc_slide":   "Slide the ziploc bag seal to close it",
    "pingpong test":  "Pick up the ping pong ball and place it into the cup",
    "pingpong_test":  "Pick up the ping pong ball and place it into the cup",
    "pro pencil":     "Pick up the pencil and place it into the pencil holder",
    "pro_pencil":     "Pick up the pencil and place it into the pencil holder",
}


def fix_aloha_tasks():
    """Expand ALOHA short task names to full manipulation descriptions."""
    print("\n" + "=" * 60)
    print("  ALOHA: Expanding task descriptions")
    print("=" * 60)

    aloha_dir = PROCESSED_DIR / "aloha"
    if not aloha_dir.exists():
        print("  ERROR: aloha directory not found")
        return {"fixed": 0, "total": 0}

    episodes = sorted([e for e in aloha_dir.iterdir()
                        if e.is_dir() and e.name.startswith("episode_")])
    fixed = 0

    for ep_dir in episodes:
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        old_task = meta.get("task", "")

        if old_task in ALOHA_TASK_MAP:
            meta["task_original"] = old_task
            meta["task"] = ALOHA_TASK_MAP[old_task]
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"  {ep_dir.name}: \"{old_task}\" -> \"{meta['task']}\"")
            fixed += 1
        else:
            print(f"  {ep_dir.name}: \"{old_task}\" (no mapping, kept)")

    result = {"fixed": fixed, "total": len(episodes)}
    print(f"  Fixed: {fixed}/{len(episodes)}")
    return result


# ── Incomplete episodes ──────────────────────────────────────────────────────

def detect_and_clean_incomplete(dry_run=True):
    """Detect and optionally remove episodes with rgb/depth mismatch or <3 frames."""
    print("\n" + "=" * 60)
    print(f"  Incomplete episodes {'(DRY RUN)' if dry_run else '(REMOVING)'}")
    print("=" * 60)

    total_found = 0
    total_removed = 0

    for ds_dir in sorted(PROCESSED_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue

        ds_name = ds_dir.name
        incomplete = []

        for ep_dir in sorted(ds_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue

            filenames = [f.name for f in ep_dir.iterdir()]
            rgb_count = sum(1 for f in filenames if f.startswith("rgb_"))
            depth_count = sum(1 for f in filenames if f.startswith("depth_"))
            has_meta = "meta.json" in filenames

            issues = []
            if rgb_count != depth_count:
                issues.append(f"rgb={rgb_count} depth={depth_count}")
            if rgb_count < 3:
                issues.append(f"only {rgb_count} frames")
            if not has_meta:
                issues.append("no meta.json")

            if issues:
                incomplete.append((ep_dir, issues))

        if incomplete:
            print(f"\n  {ds_name}: {len(incomplete)} incomplete")
            total_found += len(incomplete)

            for ep_dir, issues in incomplete:
                issue_str = ", ".join(issues)
                if dry_run:
                    print(f"    [skip] {ep_dir.name}: {issue_str}")
                else:
                    shutil.rmtree(ep_dir)
                    print(f"    [del]  {ep_dir.name}: {issue_str}")
                    total_removed += 1

    if total_found == 0:
        print("  All episodes complete!")
    else:
        print(f"\n  Found: {total_found} incomplete episodes")
        if dry_run:
            print("  (Use --clean without --dry-run to remove)")
        else:
            print(f"  Removed: {total_removed}")

    return total_found


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fix task descriptions in processed data")
    parser.add_argument("--droid", action="store_true", help="Fix DROID task descriptions")
    parser.add_argument("--aloha", action="store_true", help="Fix ALOHA task descriptions")
    parser.add_argument("--clean", action="store_true", help="Detect/remove incomplete episodes")
    parser.add_argument("--all", action="store_true", help="Run all fixes")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't modify files")
    args = parser.parse_args()

    if not any([args.droid, args.aloha, args.clean, args.all]):
        args.all = True

    if args.droid or args.all:
        fix_droid_tasks()

    if args.aloha or args.all:
        fix_aloha_tasks()

    if args.clean or args.all:
        detect_and_clean_incomplete(dry_run=args.dry_run if args.clean else True)


if __name__ == "__main__":
    main()
