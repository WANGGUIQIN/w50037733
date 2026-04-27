#!/usr/bin/env python3
"""Diagnose RoboBrain pointing-task failure rate on rh20t plan.json queries.

Samples N episodes, collects all per-step queries (same `resolve_query` logic
as refine_affordance_robobrain.py), runs RoboBrain pointing, and classifies
each result into four buckets:

    parse_fail  — no <point> tag / coords outside 0-1000
    bad_rgb     — point lands on near-uniform region (likely background)
    bad_depth   — depth at point is 0 / NaN / out of [0.01, 10] range
    ok          — passed all checks

Intent: answer the question "can RoboBrain pointing be used stand-alone
instead of GPT+RoboBrain hybrid" by quantifying how often RoboBrain's
single-point output is actually usable.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n robobrain_3dgs \\
        python scripts/diagnose_robobrain_pointing.py --n 100

    # dataset other than rh20t:
    CUDA_VISIBLE_DEVICES=0 conda run -n robobrain_3dgs \\
        python scripts/diagnose_robobrain_pointing.py --dataset rlbench --n 50
"""
import argparse
import contextlib
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "RoboBrain2.5"))

from inference import UnifiedInference  # noqa: E402

DEFAULT_MODEL_PATH = "/home/edge/Embodied/models/RoboBrain2.5-8B-NV"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}
POINT_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

# RGB-region check: 11x11 patch around the point
PATCH_HALF = 5
# Threshold below which a patch is considered "flat" (likely background).
# Measured as stddev across all 3 channels in 0-255 scale.
FLAT_STDDEV_THRESHOLD = 8.0


def extract_point(answer: str) -> tuple[int, int] | None:
    """Return (x, y) in 0-1000 range, or None on parse failure."""
    m = POINT_RE.search(answer)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return (x, y)


def resolve_query(step: dict) -> str:
    action = step.get("action", "")
    dest = step.get("destination")
    raw = dest if (action in DESTINATION_ACTIONS and dest) else step.get("target", "")
    if isinstance(raw, list):
        raw = " and ".join(str(x) for x in raw)
    return raw or ""


def check_rgb_flat(rgb: np.ndarray, u: float, v: float) -> bool:
    """True iff the 11x11 patch around (u, v) is near-uniform (likely background)."""
    h, w, _ = rgb.shape
    px = min(max(int(round(u * w)), PATCH_HALF), w - 1 - PATCH_HALF)
    py = min(max(int(round(v * h)), PATCH_HALF), h - 1 - PATCH_HALF)
    patch = rgb[py - PATCH_HALF : py + PATCH_HALF + 1,
                px - PATCH_HALF : px + PATCH_HALF + 1]
    return float(patch.std()) < FLAT_STDDEV_THRESHOLD


def check_depth(depth: np.ndarray | None, u: float, v: float) -> bool:
    """True iff depth at (u, v) is invalid (0 / NaN / out-of-range)."""
    if depth is None:
        return False  # no depth map -> don't count as bad
    h, w = depth.shape
    px = min(max(int(round(u * w)), 0), w - 1)
    py = min(max(int(round(v * h)), 0), h - 1)
    z = float(depth[py, px])
    if np.isnan(z) or np.isinf(z):
        return True
    return z <= 0.01 or z > 10.0


def diagnose_query(
    model: UnifiedInference,
    rgb_arr: np.ndarray,
    depth: np.ndarray | None,
    image_path: str,
    query: str,
) -> str:
    """Run one pointing call, classify outcome. Returns bucket name."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.inference(
                text=query.replace("_", " "),
                image=image_path,
                task="pointing",
                do_sample=False,
                temperature=0.0,
            )
        answer = result.get("answer", "")
    except Exception:
        return "parse_fail"

    pt = extract_point(answer)
    if pt is None:
        return "parse_fail"
    u, v = pt[0] / 1000.0, pt[1] / 1000.0

    if check_rgb_flat(rgb_arr, u, v):
        return "bad_rgb"
    if check_depth(depth, u, v):
        return "bad_depth"
    return "ok"


def collect_episode_queries(ep_dir: Path) -> list[tuple[str, dict]]:
    """Return list of (query, step_meta) from plan.json, deduped within episode."""
    plan_path = ep_dir / "plan.json"
    if not plan_path.exists():
        return []
    try:
        plan = json.loads(plan_path.read_text())
    except json.JSONDecodeError:
        return []
    seen = set()
    out = []
    for step in plan.get("steps", []):
        q = resolve_query(step)
        if not q or q in seen:
            continue
        seen.add(q)
        out.append((q, {"action": step.get("action"), "query": q}))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="rh20t")
    parser.add_argument("--n", type=int, default=100, help="Number of episodes to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    ds_dir = DATA_ROOT / args.dataset
    if not ds_dir.exists():
        print(f"ERROR: dataset dir not found: {ds_dir}")
        return 2

    episodes = sorted(
        p for p in ds_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and (p / "plan.json").exists()
    )
    if not episodes:
        print(f"ERROR: no episodes with plan.json under {ds_dir}")
        return 2

    rng = random.Random(args.seed)
    sample = rng.sample(episodes, min(args.n, len(episodes)))
    print(f"Dataset: {args.dataset}")
    print(f"Sampled: {len(sample)} / {len(episodes)} episodes (seed={args.seed})")

    # Build query list: (episode_name, query, rgb_path, depth_path)
    queries: list[tuple[str, str, Path, Path]] = []
    per_ep_queries: dict[str, int] = {}
    for ep in sample:
        qs = collect_episode_queries(ep)
        per_ep_queries[ep.name] = len(qs)
        rgb_path = ep / "rgb_0.png"
        depth_path = ep / "depth_0.npy"
        if not rgb_path.exists():
            continue
        for q, _ in qs:
            queries.append((ep.name, q, rgb_path, depth_path))

    print(f"Total queries: {len(queries)} "
          f"(avg {len(queries) / len(sample):.1f} per episode)")
    print(f"Loading model from {args.model_path} on {args.device} ...")
    model = UnifiedInference(args.model_path, device_map=args.device)

    buckets: Counter = Counter()
    per_bucket_examples: dict[str, list] = defaultdict(list)
    # Cache by (image_path, query) since we might revisit within an episode
    # (already deduped within-episode, but across episodes some queries like
    # "block_1" may recur with different images — cache key includes image)
    seen_cache: dict[tuple[str, str], str] = {}

    t0 = time.time()
    for i, (ep_name, query, rgb_path, depth_path) in enumerate(queries, 1):
        cache_key = (str(rgb_path), query)
        if cache_key in seen_cache:
            bucket = seen_cache[cache_key]
        else:
            rgb_arr = np.array(Image.open(rgb_path).convert("RGB"))
            depth = np.load(depth_path) if depth_path.exists() else None
            bucket = diagnose_query(model, rgb_arr, depth, str(rgb_path), query)
            seen_cache[cache_key] = bucket

        buckets[bucket] += 1
        if len(per_bucket_examples[bucket]) < 5:
            per_bucket_examples[bucket].append({"episode": ep_name, "query": query})

        if i % 20 == 0 or i == len(queries):
            rate = i / (time.time() - t0)
            print(f"  [{i}/{len(queries)}] {dict(buckets)} ({rate:.1f} q/s)")

    elapsed = time.time() - t0
    total = sum(buckets.values())
    print("\n=== RoboBrain pointing diagnostic summary ===")
    print(f"dataset:  {args.dataset}")
    print(f"episodes: {len(sample)}")
    print(f"queries:  {total} ({elapsed/60:.1f} min, {total/elapsed:.1f} q/s)")
    print()
    for bucket in ("ok", "parse_fail", "bad_rgb", "bad_depth"):
        n = buckets.get(bucket, 0)
        pct = 100.0 * n / total if total else 0
        print(f"  {bucket:<12s} {n:>5d}  ({pct:5.1f}%)")
    print()
    print("Sample failures:")
    for bucket in ("parse_fail", "bad_rgb", "bad_depth"):
        ex = per_bucket_examples.get(bucket, [])
        if ex:
            print(f"  [{bucket}]")
            for e in ex:
                print(f"    {e['episode']}: {e['query']!r}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dataset": args.dataset,
            "n_episodes": len(sample),
            "n_queries": total,
            "buckets": dict(buckets),
            "examples": {k: v for k, v in per_bucket_examples.items()},
            "elapsed_sec": elapsed,
            "seed": args.seed,
        }, indent=2))
        print(f"\nWrote report to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
