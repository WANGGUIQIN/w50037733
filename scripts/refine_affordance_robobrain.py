#!/usr/bin/env python3
"""Refine plan.json affordance coordinates using RoboBrain2.5-8B base model.

Hybrid strategy: keep GPT-generated semantics (action, target, destination,
constraints, done_when), but replace the 2D affordance [u, v] with RoboBrain's
grounded prediction. RoboBrain is domain-aligned (trained on robotic scenes)
and produces more stable pixel-level localization than gpt-4o-mini.

Per-step target resolution rule:
    transport / place / insert / pour  -> locate DESTINATION
    others (reach / grasp / push / ...) -> locate TARGET

Frame choice: rgb_0.png — all objects are visible in the initial state, and
object positions are deterministic in RLBench scenes.

Usage:
    # Dry run on 3 episodes (no disk write, print comparison)
    conda run -n robobrain_3dgs python scripts/refine_affordance_robobrain.py --dry-run --limit 3

    # Full run with resume
    conda run -n robobrain_3dgs python scripts/refine_affordance_robobrain.py --resume

    # Range
    conda run -n robobrain_3dgs python scripts/refine_affordance_robobrain.py --start 0 --end 100
"""
import argparse
import contextlib
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "RoboBrain2.5"))

from inference import UnifiedInference  # noqa: E402


DEFAULT_MODEL_PATH = "/home/edge/Embodied/models/RoboBrain2.5-8B-NV"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
DATASET_DIR = DATA_ROOT / "rlbench"
BACKUP_SUFFIX = "plan_before_robobrain.json"

DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}

POINT_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

# Idea B: action -> semantic part template. RoboBrain pointing on the templated
# query lands on the functional part (e.g., handle) instead of object center.
# Validated on 5 cases: 4/5 wins (only butter_dairy fails because it has no handle).
ACTION_PART = {
    "reach":     "the {obj}",
    "grasp":     "the handle of the {obj}",
    "pick":      "the handle of the {obj}",
    "lift":      "the handle of the {obj}",
    "rotate":    "the body of the {obj}",
    "press":     "the center of the {obj}",
    "push":      "the center of the {obj}",
    "pull":      "the handle of the {obj}",
    "place":     "an empty area on the {obj}",
    "transport": "an empty area on the {obj}",
    "insert":    "the top opening of the {obj}",
    "pour":      "the inside of the {obj}",
    "flip":      "the body of the {obj}",
    "wipe":      "the surface of the {obj}",
    "release":   "the {obj}",
    "open":      "the handle of the {obj}",
    "close":     "the handle of the {obj}",
}

# Objects that have no natural "handle" / functional part — fall back to bare
# query to avoid RoboBrain getting confused by a non-existent affordance hint.
# Triggered when the lowercase object name CONTAINS any of these tokens.
NO_HANDLE_TOKENS = {
    "butter", "dairy", "package", "box", "block", "cube", "ball", "sphere",
    "rag", "cloth", "towel", "paper", "card", "coin", "battery", "marker",
    "sponge", "candy", "carrot", "potato", "onion", "tomato", "lemon",
    "apple", "banana", "orange", "egg", "bread", "cheese",
}


def pixel_to_3d(u: float, v: float, depth_map: np.ndarray,
                intrinsics: list) -> list | None:
    """Back-project (u, v) normalized in [0,1] to 3D camera coords using depth.

    Returns None if depth is invalid (0 or out of range).
    """
    h, w = depth_map.shape
    px = min(max(int(round(u * w)), 0), w - 1)
    py = min(max(int(round(v * h)), 0), h - 1)
    z = float(depth_map[py, px])
    if z <= 0.01 or z > 10.0:
        return None
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    x_cam = (px - cx) / fx * z
    y_cam = (py - cy) / fy * z
    return [round(x_cam, 4), round(y_cam, 4), round(z, 4)]


def load_depth_and_intrinsics(ep_dir: Path) -> tuple[np.ndarray | None, list | None]:
    depth_path = ep_dir / "depth_0.npy"
    meta_path = ep_dir / "meta.json"
    depth_map = None
    intrinsics = None
    if depth_path.exists():
        try:
            depth_map = np.load(depth_path)
        except Exception:
            depth_map = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            intrinsics = meta.get("intrinsics")
        except Exception:
            intrinsics = None
    return depth_map, intrinsics


def extract_point(answer: str) -> tuple[float, float] | None:
    """Parse RoboBrain pointing output. Coords are in 0-1000 range, normalize to [0, 1]."""
    m = POINT_RE.search(answer)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return (round(x / 1000.0, 3), round(y / 1000.0, 3))


def resolve_query(step: dict) -> str:
    """Pick the object to localize: destination for placing-type actions, target otherwise."""
    action = step.get("action", "")
    dest = step.get("destination")
    raw = dest if (action in DESTINATION_ACTIONS and dest) else step.get("target", "")
    if isinstance(raw, list):
        raw = " and ".join(str(x) for x in raw)
    return raw or ""


def _has_no_handle(obj: str) -> bool:
    """True if obj name contains a token that suggests no functional handle."""
    obj_lower = obj.lower()
    return any(tok in obj_lower for tok in NO_HANDLE_TOKENS)


def build_query(step: dict, mode: str = "template") -> str:
    """Construct RoboBrain pointing query.

    mode='bare':     "red_jar" (legacy)
    mode='template': "the handle of the red jar" (idea B, default)

    Falls back to bare query when:
      - action not in ACTION_PART table (unknown verb)
      - obj name suggests no functional handle (butter, package, ...)
    Returns the query string with underscores replaced by spaces.
    """
    raw_obj = resolve_query(step)
    if not raw_obj:
        return ""
    obj_disp = raw_obj.replace("_", " ")
    if mode == "bare":
        return obj_disp

    action = step.get("action", "").lower()
    template = ACTION_PART.get(action)
    if template is None:
        return obj_disp
    # Skip handle-mentioning templates for handle-less objects (butter, towel,
    # block, ...). Place/insert/pour templates ("an empty area on the {obj}",
    # "the top opening of the {obj}") still make sense and are kept.
    if "handle of" in template and _has_no_handle(raw_obj):
        return obj_disp
    # For "the {obj}" template, equivalent to bare; skip to keep cache consistent.
    if template == "the {obj}":
        return f"the {obj_disp}"
    return template.format(obj=obj_disp)


def refine_plan(
    model: UnifiedInference,
    plan: dict,
    image_path: str,
    cache: dict,
    depth_map: np.ndarray | None = None,
    intrinsics: list | None = None,
    query_mode: str = "template",
) -> tuple[dict, int, int]:
    """Refine affordance per step. Adds affordance_3d if depth+intrinsics given.

    Cache key is (action, query) so the same target with different actions
    (e.g. grasp vs place on the same object) gets distinct affordance points.

    Each refined step also gets an `affordance_hint` field with the templated
    query string — this is the supervision signal for part-aware training and
    the human-readable explanation of what the (u, v) coord means.

    Returns (new_plan, hit, miss).
    """
    hit = 0
    miss = 0
    for step in plan["steps"]:
        query = build_query(step, mode=query_mode)
        if not query:
            miss += 1
            continue

        action = step.get("action", "")
        cache_key = (action, query)
        if cache_key in cache:
            new_aff = cache[cache_key]
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = model.inference(
                        text=query,
                        image=image_path,
                        task="pointing",
                        do_sample=False,
                        temperature=0.0,
                    )
                new_aff = extract_point(result["answer"])
            except Exception as e:
                print(f"    WARN: inference failed for '{query}': {e}")
                new_aff = None
            cache[cache_key] = new_aff

        # Always record the hint so downstream training has the part name even
        # if pointing failed (training still benefits from the hint as input).
        step["affordance_hint"] = query

        if new_aff is not None:
            step["affordance"] = list(new_aff)
            # Layer 2: 2D -> 3D back-projection
            if depth_map is not None and intrinsics is not None:
                aff_3d = pixel_to_3d(new_aff[0], new_aff[1], depth_map, intrinsics)
                if aff_3d is not None:
                    step["affordance_3d"] = aff_3d
            hit += 1
        else:
            miss += 1
    return plan, hit, miss


def process_episode(
    model: UnifiedInference,
    ep_dir: Path,
    dry_run: bool,
    query_mode: str = "template",
) -> tuple[bool, int, int]:
    plan_path = ep_dir / "plan.json"
    rgb_path = ep_dir / "rgb_0.png"
    if not plan_path.exists() or not rgb_path.exists():
        return (False, 0, 0)

    with open(plan_path) as f:
        plan = json.load(f)

    old_affordances = [tuple(s.get("affordance", [])) for s in plan["steps"]]
    cache: dict[tuple[str, str], tuple[float, float] | None] = {}

    depth_map, intrinsics = load_depth_and_intrinsics(ep_dir)
    plan, hit, miss = refine_plan(model, plan, str(rgb_path), cache,
                                  depth_map=depth_map, intrinsics=intrinsics,
                                  query_mode=query_mode)

    if dry_run:
        new_affordances = [tuple(s.get("affordance", [])) for s in plan["steps"]]
        print(f"  {ep_dir.name} task={plan['task']!r}")
        for i, (old, new, step) in enumerate(
            zip(old_affordances, new_affordances, plan["steps"]), 1
        ):
            hint = step.get("affordance_hint", "")
            changed = "*" if old != new else " "
            print(f"    {changed} step {i} {step['action']} -> "
                  f"hint={hint!r}: {old} -> {new}")
        return (True, hit, miss)

    backup_path = ep_dir / BACKUP_SUFFIX
    if not backup_path.exists():
        shutil.copy2(plan_path, backup_path)

    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    return (True, hit, miss)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--splits", default=None,
                        help="Path to splits JSON (e.g. data/splits/train.json). "
                             "Overrides --dataset-based iteration.")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR),
                        help="Single-dataset directory when --splits is not set.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=-1, help="Process only first N episodes (for dry-run).")
    parser.add_argument("--resume", action="store_true", help="Skip episodes whose backup already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print comparisons, don't write.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-mode", choices=["bare", "template"], default="template",
                        help="bare: query RoboBrain with raw object name (legacy). "
                             "template: action-aware query (idea B, default), e.g. "
                             "'the handle of the red jar' for grasp.")
    args = parser.parse_args()

    if args.splits:
        split_path = Path(args.splits)
        if not split_path.exists():
            print(f"ERROR: splits file not found: {split_path}")
            sys.exit(1)
        with open(split_path) as f:
            split_data = json.load(f)
        episodes = [DATA_ROOT / e["dataset"] / e["episode_id"] for e in split_data["episodes"]]
        episodes = [e for e in episodes if e.exists()]
        print(f"Loaded splits: {split_path.name}, {len(episodes)} episodes on disk")
    else:
        ds_dir = Path(args.dataset_dir)
        episodes = sorted(p for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("episode_"))

    if args.end > 0:
        episodes = episodes[args.start:args.end]
    else:
        episodes = episodes[args.start:]
    if args.limit > 0:
        episodes = episodes[:args.limit]
    if args.resume:
        episodes = [e for e in episodes if not (e / BACKUP_SUFFIX).exists()]

    print(f"Model: {args.model_path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Query mode: {args.query_mode}")
    print(f"Dry run: {args.dry_run}")
    if not episodes:
        print("Nothing to do.")
        return

    print("Loading RoboBrain2.5 ...")
    t0 = time.time()
    model = UnifiedInference(args.model_path, device_map=args.device)
    print(f"Loaded in {time.time() - t0:.1f}s")

    total_hit = 0
    total_miss = 0
    ok = 0
    t_start = time.time()
    for i, ep in enumerate(episodes, 1):
        processed, hit, miss = process_episode(model, ep, args.dry_run,
                                               query_mode=args.query_mode)
        if processed:
            ok += 1
            total_hit += hit
            total_miss += miss
        if i % 10 == 0 or i == len(episodes):
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(episodes) - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{len(episodes)}] ok={ok} hit={total_hit} miss={total_miss} "
                f"rate={rate:.2f} ep/s ETA={eta / 60:.1f}min"
            )

    elapsed = time.time() - t_start
    print(f"\nDone. {ok}/{len(episodes)} episodes, {total_hit} affordances refined, "
          f"{total_miss} kept original (parse/query fail), {elapsed / 60:.1f}min")


if __name__ == "__main__":
    main()
