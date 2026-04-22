#!/usr/bin/env python3
"""Visual comparison of GPT vs RoboBrain affordance predictions.

For each sampled episode:
  - Read rgb_0.png and current plan.json (GPT affordances)
  - Run RoboBrain to predict new affordances
  - Draw both points on the image (blue = GPT, red = RoboBrain)
  - Save comparison to result/affordance_compare/

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n robobrain_3dgs python scripts/verify_affordance_robobrain.py --n 10
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "RoboBrain2.5"))

from inference import UnifiedInference  # noqa: E402

DEFAULT_MODEL = "/home/edge/Embodied/models/RoboBrain2.5-8B-NV"
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "rlbench"
OUTPUT_DIR = PROJECT_ROOT / "result" / "affordance_compare"
DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}
POINT_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def extract_point(answer: str) -> tuple[float, float] | None:
    m = POINT_RE.search(answer)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return (x / 1000.0, y / 1000.0)


def resolve_query(step: dict) -> str:
    action = step.get("action", "")
    dest = step.get("destination")
    if action in DESTINATION_ACTIONS and dest:
        return dest
    return step.get("target", "")


def draw_comparison(
    image_path: Path,
    steps: list[dict],
    gpt_points: list[tuple[float, float]],
    rb_points: list[tuple[float, float] | None],
    out_path: Path,
    upsample: int = 3,
):
    """Draw GPT (blue) and RoboBrain (red) points with step labels."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * upsample, h * upsample), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]

    for i, (step, gpt, rb) in enumerate(zip(steps, gpt_points, rb_points), 1):
        q = resolve_query(step)
        label = f"{i}.{step['action']}({q})"

        if gpt:
            gx, gy = int(gpt[0] * w), int(gpt[1] * h)
            cv2.circle(img, (gx, gy), 8, (255, 80, 0), -1)  # blue (BGR)
            cv2.circle(img, (gx, gy), 10, (255, 255, 255), 2)
            cv2.putText(img, f"G{i}", (gx + 10, gy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

        if rb:
            rx, ry = int(rb[0] * w), int(rb[1] * h)
            cv2.circle(img, (rx, ry), 8, (0, 80, 255), -1)  # red (BGR)
            cv2.circle(img, (rx, ry), 10, (255, 255, 255), 2)
            cv2.putText(img, f"R{i}", (rx + 10, ry + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    # Legend
    cv2.rectangle(img, (5, 5), (max(400, 18 * len(steps) + 160), 30 + 22 * len(steps)),
                  (40, 40, 40), -1)
    cv2.putText(img, "Blue=GPT  Red=RoboBrain", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    for i, step in enumerate(steps, 1):
        q = resolve_query(step)
        cv2.putText(img, f"{i}. {step['action']}({q})",
                    (10, 25 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def sample_episodes(n: int, seed: int = 42) -> list[Path]:
    """Sample N episodes uniformly across the 1800 range to span all RLBench tasks."""
    all_eps = sorted(p for p in DATASET_DIR.iterdir()
                     if p.is_dir() and p.name.startswith("episode_"))
    total = len(all_eps)
    if n >= total:
        return all_eps
    # Stratified: one per 100 to span 18 tasks (RLBench groups 100 ep per task)
    step = total // n
    indices = [i * step for i in range(n)]
    return [all_eps[i] for i in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of episodes to verify")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--episodes", nargs="*", help="Specific episode names (e.g. episode_000000)")
    args = parser.parse_args()

    if args.episodes:
        eps = [DATASET_DIR / name for name in args.episodes]
    else:
        eps = sample_episodes(args.n)

    print(f"Sampled {len(eps)} episodes:")
    for ep in eps:
        print(f"  {ep.name}")

    print("\nLoading RoboBrain2.5 ...")
    model = UnifiedInference(args.model_path, device_map="auto")
    print("Loaded.\n")

    summary = []
    for ep in eps:
        plan_path = ep / "plan.json"
        rgb_path = ep / "rgb_0.png"
        meta_path = ep / "meta.json"
        if not (plan_path.exists() and rgb_path.exists()):
            continue

        with open(plan_path) as f:
            plan = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        gpt_points = [tuple(s.get("affordance", [0.5, 0.5])) for s in plan["steps"]]
        rb_points: list[tuple[float, float] | None] = []
        cache: dict[str, tuple[float, float] | None] = {}

        for step in plan["steps"]:
            q = resolve_query(step)
            if not q:
                rb_points.append(None)
                continue
            if q in cache:
                rb_points.append(cache[q])
                continue
            try:
                result = model.inference(
                    text=q.replace("_", " "),
                    image=str(rgb_path),
                    task="pointing",
                    do_sample=False,
                    temperature=0.0,
                )
                pt = extract_point(result["answer"])
            except Exception as e:
                print(f"    WARN: {ep.name} '{q}': {e}")
                pt = None
            cache[q] = pt
            rb_points.append(pt)

        out_path = OUTPUT_DIR / f"{ep.name}.png"
        draw_comparison(rgb_path, plan["steps"], gpt_points, rb_points, out_path)
        summary.append((ep.name, meta.get("task", "?"), plan["steps"], gpt_points, rb_points))
        print(f"  {ep.name} [{meta.get('task','?')}] -> {out_path.name}")

    print(f"\n{'=' * 70}")
    print(f"Saved {len(summary)} comparisons to {OUTPUT_DIR}")
    print(f"{'=' * 70}")
    for name, task, steps, gpt, rb in summary:
        print(f"\n{name}  task: {task!r}")
        for i, (s, g, r) in enumerate(zip(steps, gpt, rb), 1):
            q = resolve_query(s)
            delta = "?" if r is None else f"Δ={((g[0]-r[0])**2+(g[1]-r[1])**2)**0.5:.2f}"
            print(f"  step {i} {s['action']}({q}): GPT={tuple(round(x,3) for x in g)} "
                  f"RB={r if r is None else tuple(round(x,3) for x in r)} {delta}")


if __name__ == "__main__":
    main()
