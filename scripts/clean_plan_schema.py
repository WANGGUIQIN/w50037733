#!/usr/bin/env python3
"""Clean schema violations in plan.json files produced by GPT-4o.

The GPT plan generator occasionally hallucinates placeholder strings instead
of the structured dict/list schema the loader expects, which crashes training
on a small number of episodes. Known hallucination patterns:

    "constraints": "similar to above steps"       # str, loader expects dict
    "approach":    "dynamic"                      # str, loader expects list[float, 3]
    "action":      "repeat" / "finalize"          # placeholder step with no real data

This script walks <data_root>/<dataset>/episode_*/plan.json and:

    1. Drops steps whose `action` is a known placeholder (repeat / finalize).
    2. For surviving steps, coerces malformed numeric fields to loader
       defaults (affordance -> [0.5, 0.5], approach -> [0, 0, -1]) and
       drops non-dict `constraints`.
    3. Rewrites plan.json in place only when changes were needed.
    4. Prints a summary grouped by dataset.

Idempotent: running twice is a no-op on the second pass.

Usage:
    # Data lives under the repo default (data/processed/):
    python scripts/clean_plan_schema.py

    # Data was extracted somewhere else (e.g. a training server):
    python scripts/clean_plan_schema.py --data-root /workspace/data/processed

    # Report only, don't write:
    python scripts/clean_plan_schema.py --dry-run
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed"

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

    kept_steps = []
    for step in steps:
        if step.get("action") in PLACEHOLDER_ACTIONS:
            fixes.append(f"drop_step(action={step.get('action')})")
            continue
        kept_steps.append(step)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory holding <dataset>/episode_*/plan.json subtrees "
             f"(default: {DEFAULT_DATA_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned fixes without writing any files.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    print(f"[info] data_root = {data_root}")
    if not data_root.exists():
        print(f"ERROR: data_root does not exist: {data_root}")
        print("Pass --data-root <path> pointing at the directory that contains "
              "<dataset>/episode_*/ subdirectories.")
        return 2

    paths = sorted(data_root.glob("*/episode_*/plan.json"))
    if not paths:
        print(f"ERROR: no plan.json found under {data_root}")
        return 2

    total_scanned = 0
    total_dirty = 0
    fix_counter: Counter = Counter()
    per_dataset: Counter = Counter()

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

        total_dirty += 1
        per_dataset[p.parent.parent.name] += 1
        for fx in fixes:
            fix_counter[fx.split("(")[0]] += 1

        if not args.dry_run:
            p.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    print(f"scanned: {total_scanned}")
    print(f"dirty:   {total_dirty}")
    if per_dataset:
        print(f"by dataset: {dict(per_dataset)}")
    print(f"fix ops: {dict(fix_counter)}")
    if args.dry_run:
        print("\n(dry-run, no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
