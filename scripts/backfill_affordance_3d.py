#!/usr/bin/env python3
"""Back-fill missing affordance_3d in plan.json by re-projecting existing 2D
affordances through depth_0.npy + meta.json intrinsics.

Use case: when refine_affordance_robobrain.py was originally run before depth
maps existed, plan.json got refined 2D affordances but no 3D back-projection.
This script adds the missing affordance_3d field without re-running RoboBrain.

Per-step rule: only fills affordance_3d if it's missing AND depth at the
affordance pixel is valid (in [0.01, 10.0] range, not NaN/Inf).

Usage:
    python scripts/backfill_affordance_3d.py --dataset rlbench
    python scripts/backfill_affordance_3d.py --dataset rlbench --dry-run
    python scripts/backfill_affordance_3d.py --all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse pixel_to_3d / load_depth_and_intrinsics directly without importing
# refine_affordance_robobrain (which pulls in the RoboBrain model module).

def pixel_to_3d(u: float, v: float, depth_map: np.ndarray, intrinsics: list):
    h, w = depth_map.shape
    px = min(max(int(round(u * w)), 0), w - 1)
    py = min(max(int(round(v * h)), 0), h - 1)
    z = float(depth_map[py, px])
    if np.isnan(z) or np.isinf(z) or z <= 0.01 or z > 10.0:
        return None
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    x_cam = (px - cx) / fx * z
    y_cam = (py - cy) / fy * z
    return [round(x_cam, 4), round(y_cam, 4), round(z, 4)]


def load_depth_and_intrinsics(ep_dir: Path):
    depth_path = ep_dir / "depth_0.npy"
    meta_path = ep_dir / "meta.json"
    depth_map, intrinsics = None, None
    if depth_path.exists():
        try:
            depth_map = np.load(depth_path)
        except Exception:
            pass
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            intrinsics = meta.get("intrinsics")
        except Exception:
            pass
    return depth_map, intrinsics


def process_episode(ep_dir: Path, dry_run: bool) -> dict:
    """Returns counters for this episode."""
    plan_path = ep_dir / "plan.json"
    if not plan_path.exists():
        return {"skipped_no_plan": 1}

    plan = json.loads(plan_path.read_text())
    steps = plan.get("steps", [])
    if not steps:
        return {"skipped_no_steps": 1}

    # Quick check: any step missing aff_3d?
    needs_fill = [s for s in steps if "affordance_3d" not in s and s.get("affordance")]
    if not needs_fill:
        return {"already_complete": 1}

    depth_map, intrinsics = load_depth_and_intrinsics(ep_dir)
    if depth_map is None or intrinsics is None:
        return {"skipped_no_depth_or_intr": 1}

    filled = 0
    failed = 0
    for s in needs_fill:
        u, v = s["affordance"]
        a3d = pixel_to_3d(u, v, depth_map, intrinsics)
        if a3d is not None:
            s["affordance_3d"] = a3d
            filled += 1
        else:
            failed += 1

    if not dry_run and filled > 0:
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))

    return {
        "episodes_modified": 1 if filled else 0,
        "steps_filled": filled,
        "steps_invalid_depth": failed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=None, help="Single dataset name (e.g. rlbench)")
    ap.add_argument("--all", action="store_true", help="Process all datasets")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "processed"))
    args = ap.parse_args()

    if not args.dataset and not args.all:
        ap.error("must specify --dataset NAME or --all")

    data_root = Path(args.data_root)
    if args.all:
        datasets = sorted(d.name for d in data_root.iterdir()
                          if d.is_dir() and any(d.glob("episode_*/plan.json")))
    else:
        datasets = [args.dataset]

    print(f"Datasets: {datasets}")
    print(f"Dry run: {args.dry_run}")
    grand_total = {
        "episodes_modified": 0,
        "steps_filled": 0,
        "steps_invalid_depth": 0,
        "already_complete": 0,
        "skipped_no_depth_or_intr": 0,
        "skipped_no_plan": 0,
        "skipped_no_steps": 0,
    }

    for ds_name in datasets:
        ds_dir = data_root / ds_name
        if not ds_dir.exists():
            print(f"  {ds_name}: directory missing, skipping")
            continue
        eps = sorted(ds_dir.glob("episode_*"))
        eps = [e for e in eps if (e / "plan.json").exists()]
        ds_totals = dict.fromkeys(grand_total, 0)
        t0 = time.time()
        for i, ep in enumerate(eps, 1):
            result = process_episode(ep, args.dry_run)
            for k, v in result.items():
                ds_totals[k] = ds_totals.get(k, 0) + v
            if i % 500 == 0:
                print(f"  {ds_name}: {i}/{len(eps)} processed ({time.time()-t0:.1f}s)")

        elapsed = time.time() - t0
        print(f"\n  [{ds_name}] {len(eps)} episodes in {elapsed:.1f}s:")
        print(f"    episodes_modified:        {ds_totals['episodes_modified']}")
        print(f"    steps_filled:             {ds_totals['steps_filled']}")
        print(f"    steps_invalid_depth:      {ds_totals['steps_invalid_depth']}")
        print(f"    already_complete:         {ds_totals['already_complete']}")
        print(f"    skipped_no_depth_or_intr: {ds_totals['skipped_no_depth_or_intr']}")
        for k in grand_total:
            grand_total[k] += ds_totals[k]

    print("\n=== GRAND TOTAL ===")
    for k, v in grand_total.items():
        print(f"  {k:30s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
