#!/usr/bin/env python3
"""Clean schema violations in plan.json files produced by GPT-4o.

The GPT plan generator occasionally hallucinates placeholder strings instead
of the structured dict/list schema the loader expects, which crashes training
on a small number of episodes. Known hallucination patterns:

    "constraints": "similar to above steps"       # str, loader expects dict
    "approach":    "dynamic"                      # str, loader expects list[float, 3]
    "action":      "repeat" / "finalize"          # placeholder step with no real data

This script does a one-pass cleanup:

    1. Drop steps whose `action` is a known placeholder (repeat / finalize).
    2. For any remaining step, coerce malformed numeric fields to loader
       defaults (affordance -> [0.5, 0.5], approach -> [0, 0, -1]) and drop
       non-dict `constraints`.
    3. Rewrite plan.json in place (only when changes were made).
    4. Print a summary grouped by split and dataset.

Idempotent: running twice is a no-op on the second pass.

Usage:
    python scripts/clean_plan_schema.py                       # scan + fix all splits
    python scripts/clean_plan_schema.py --splits train val    # subset of splits
    python scripts/clean_plan_schema.py --dry-run             # report, don't write
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

PLACEHOLDER_ACTIONS = {"repeat", "finalize"}
DEFAULT_AFFORDANCE = [0.5, 0.5]
DEFAULT_APPROACH = [0.0, 0.0, -1.0]


def _is_numeric_list(x, min_len: int) -> bool:
    return (
        isinstance(x, list)
        and len(x) >= min_len
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x[:min_len])
    )


def clean_plan(plan: dict) -> tuple[dict, list[str]]:
    """Return (cleaned_plan, list_of_applied_fixes). Modifies nothing in place."""
    fixes: list[str] = []
    steps = plan.get("steps", [])

    # 1. Drop placeholder steps
    kept_steps = []
    for step in steps:
        if step.get("action") in PLACEHOLDER_ACTIONS:
            fixes.append(f"drop_step(action={step.get('action')})")
            continue
        kept_steps.append(step)

    # 2. Coerce malformed fields in surviving steps
    for step in kept_steps:
        if not _is_numeric_list(step.get("affordance"), 2):
            fixes.append(f"coerce_affordance(step={step.get('step')})")
            step["affordance"] = list(DEFAULT_AFFORDANCE)
        if not _is_numeric_list(step.get("approach"), 3):
            fixes.append(f"coerce_approach(step={step.get('step')})")
            step["approach"] = list(DEFAULT_APPROACH)
        c = step.get("constraints")
        if c is not None and not isinstance(c, dict):
            fixes.append(f"drop_constraints(step={step.get('step')}, type={type(c).__name__})")
            step["constraints"] = {}

    cleaned = dict(plan)
    cleaned["steps"] = kept_steps
    cleaned["num_steps"] = len(kept_steps)
    return cleaned, fixes


def load_split_episodes(split_name: str) -> list[Path]:
    """Return plan.json paths for episodes listed in data/splits/<split_name>.json."""
    split_path = SPLITS_DIR / f"{split_name}.json"
    if not split_path.exists():
        return []
    with open(split_path) as f:
        data = json.load(f)
    paths = []
    for e in data["episodes"]:
        p = DATA_ROOT / e["dataset"] / e["episode_id"] / "plan.json"
        if p.exists():
            paths.append(p)
    return paths


def scan_filesystem() -> list[Path]:
    """Fallback when split files aren't available (e.g. on a training server
    that only has data/processed/ extracted from a tarball)."""
    if not DATA_ROOT.exists():
        return []
    return sorted(DATA_ROOT.glob("*/episode_*/plan.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test_id", "test_ood_embo", "test_ood_task"],
        help="Which splits to process (default: all five). Ignored if --all is set "
             "or if no split files exist on disk.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ignore split files and walk data/processed/ directly "
             "(use on training servers where split JSONs weren't transferred).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned fixes without writing any files.",
    )
    args = parser.parse_args()

    # Decide which mode to run: split-whitelisted or filesystem-walk.
    # Auto-fallback if the user passed --splits but no split files exist.
    use_filesystem = args.all or not SPLITS_DIR.exists() or not any(SPLITS_DIR.glob("*.json"))
    if use_filesystem and not args.all:
        print(f"[info] {SPLITS_DIR} is empty or missing — falling back to filesystem scan")

    total_scanned = 0
    total_dirty = 0
    fix_counter: Counter = Counter()
    per_split: dict[str, dict] = {}

    if use_filesystem:
        groups = {"all": scan_filesystem()}
    else:
        groups = {name: load_split_episodes(name) for name in args.splits}

    for split, paths in groups.items():
        if not paths:
            print(f"[{split}] no episodes on disk (or split file missing), skipping")
            continue

        dirty_files = 0
        per_dataset = defaultdict(int)
        for p in paths:
            total_scanned += 1
            try:
                plan = json.loads(p.read_text())
            except json.JSONDecodeError as ex:
                fix_counter["json_decode_error"] += 1
                print(f"  BAD-JSON: {p} ({ex})")
                continue
            cleaned, fixes = clean_plan(plan)
            if not fixes:
                continue
            dirty_files += 1
            total_dirty += 1
            per_dataset[p.parent.parent.name] += 1
            for fx in fixes:
                fix_counter[fx.split("(")[0]] += 1
            if not args.dry_run:
                p.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))

        per_split[split] = {
            "scanned": len(paths),
            "dirty": dirty_files,
            "by_dataset": dict(per_dataset),
        }

    print("\n=== summary ===")
    for split, stats in per_split.items():
        line = f"{split:18s} scanned={stats['scanned']:5d}  dirty={stats['dirty']:4d}"
        if stats["by_dataset"]:
            line += f"  datasets={stats['by_dataset']}"
        print(line)
    print(f"\nTotal scanned: {total_scanned}   dirty: {total_dirty}")
    print(f"Fix operations: {dict(fix_counter)}")
    if args.dry_run:
        print("\n(dry-run, no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
