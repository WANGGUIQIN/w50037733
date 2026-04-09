#!/usr/bin/env python3
"""Fix data quality issues in robotics datasets.

Task 1.1: Fix Bridge task descriptions (ALL CAPS, typos, mojibake)
Task 1.2: Backfill num_keyframes field across multiple datasets
"""

import json
import os
import re
from pathlib import Path

BASE = Path("/home/edge/Embodied/robobrain_3dgs/data/processed")

# ── Task 1.1: Bridge task description fixes ──────────────────────────

TYPO_FIXES = {
    "cardboardfence": "cardboard fence",
    "CARDBOARDFENCE": "CARDBOARD FENCE",  # will be lowered later if ALL CAPS
    "ANDPUT": "AND PUT",
    "andput": "and put",
    "ANITHING": "ANYTHING",
    "anithing": "anything",
}

MOJIBAKE_FIXES = {
    "\u201a\u00c4\u00f4": "'",
    "\u201a\u00c4\u00fa": '"',
    "\u201a\u00c4\u00f9": '"',
}


def fix_task(task: str) -> tuple[str, list[str]]:
    """Fix a task string. Returns (fixed_task, list_of_fix_types_applied)."""
    fixes = []
    original = task

    # 1. Mojibake (do first, before case changes)
    for bad, good in MOJIBAKE_FIXES.items():
        if bad in task:
            task = task.replace(bad, good)
            if "mojibake" not in fixes:
                fixes.append("mojibake")

    # 2. Typo / compound word fixes (case-sensitive patterns)
    for bad, good in TYPO_FIXES.items():
        if bad in task:
            task = task.replace(bad, good)
            if "typo" not in fixes:
                fixes.append("typo")

    # 3. ALL CAPS → lowercase
    if task == task.upper() and len(task) > 3:
        task = task.lower()
        if "allcaps" not in fixes:
            fixes.append("allcaps")

    # After lowercasing, re-apply lowercase typo fixes in case CAPS conversion
    # created new matches (e.g., "ANDPUT" → "and put" after lowering "AND PUT")
    for bad, good in [("andput", "and put"), ("anithing", "anything"),
                       ("cardboardfence", "cardboard fence")]:
        if bad in task:
            task = task.replace(bad, good)
            if "typo" not in fixes:
                fixes.append("typo")

    return task, fixes


def process_bridge():
    """Fix task descriptions in bridge dataset."""
    bridge_dir = BASE / "bridge"
    episodes = sorted(bridge_dir.iterdir())

    total = 0
    caps_fixed = 0
    typo_fixed = 0
    mojibake_fixed = 0

    for ep_dir in episodes:
        meta_path = ep_dir / "meta.json"
        if not meta_path.exists():
            continue
        total += 1

        with open(meta_path, "r") as f:
            meta = json.load(f)

        task = meta.get("task", "")
        new_task, fixes = fix_task(task)

        if new_task != task:
            meta["task"] = new_task
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                f.write("\n")

            if "allcaps" in fixes:
                caps_fixed += 1
            if "typo" in fixes:
                typo_fixed += 1
            if "mojibake" in fixes:
                mojibake_fixed += 1

    print("=== Task 1.1: Bridge Task Description Fixes ===")
    print(f"  Total episodes scanned: {total}")
    print(f"  ALL_CAPS fixed:         {caps_fixed}")
    print(f"  Typos fixed:            {typo_fixed}")
    print(f"  Mojibake fixed:         {mojibake_fixed}")
    print(f"  Total modified:         {caps_fixed + typo_fixed + mojibake_fixed - len(set())}")
    # Some episodes may have multiple fix types; count unique
    print()


# ── Task 1.2: Backfill num_keyframes ─────────────────────────────────

BACKFILL_DATASETS = [
    "bridge",
    "fractal20220817_data",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "jaco_play",
    "rlbench",
]


def backfill_num_keyframes():
    """Add num_keyframes field where missing."""
    print("=== Task 1.2: Backfill num_keyframes ===")

    grand_total = 0
    grand_fixed = 0

    for ds_name in BACKFILL_DATASETS:
        ds_dir = BASE / ds_name
        if not ds_dir.exists():
            print(f"  [SKIP] {ds_name} not found")
            continue

        episodes = sorted(ds_dir.iterdir())
        total = 0
        fixed = 0

        for ep_dir in episodes:
            meta_path = ep_dir / "meta.json"
            if not meta_path.exists():
                continue
            total += 1

            with open(meta_path, "r") as f:
                meta = json.load(f)

            if "num_keyframes" not in meta:
                kf = meta.get("keyframe_indices", [])
                meta["num_keyframes"] = len(kf)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                    f.write("\n")
                fixed += 1

        print(f"  {ds_name}: scanned={total}, backfilled={fixed}")
        grand_total += total
        grand_fixed += fixed

    print(f"  TOTAL: scanned={grand_total}, backfilled={grand_fixed}")
    print()


if __name__ == "__main__":
    process_bridge()
    backfill_num_keyframes()
    print("Done. Script is idempotent — safe to re-run.")
