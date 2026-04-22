#!/usr/bin/env python3
"""Analyze task string distribution across all processed datasets.

Output:
  - Per dataset: total episodes, unique tasks, top-N task coverage
  - Cross dataset summary
  - Insights for selection quota tuning
"""
import json
from collections import Counter
from pathlib import Path

DATA_ROOT = Path("/home/edge/Embodied/robobrain_3dgs/data/processed")


def analyze_dataset(ds_dir: Path) -> dict:
    task_counter: Counter[str] = Counter()
    robot_counter: Counter[str] = Counter()
    depth_counter: Counter[str] = Counter()

    for ep in ds_dir.iterdir():
        if not (ep.is_dir() and ep.name.startswith("episode_")):
            continue
        meta_path = ep / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                m = json.load(f)
        except Exception:
            continue
        task = (m.get("task") or "").strip().lower()
        task_counter[task] += 1
        robot_counter[m.get("robot", "unknown")] += 1
        depth_counter[m.get("depth_type", "unknown")] += 1

    total = sum(task_counter.values())
    n_unique = len(task_counter)
    top5 = task_counter.most_common(5)
    top5_coverage = sum(c for _, c in top5) / total if total else 0
    top1_coverage = top5[0][1] / total if top5 else 0

    # Long-tail: how many tasks cover 50% of episodes
    sorted_counts = sorted(task_counter.values(), reverse=True)
    cum = 0
    tasks_for_50pct = 0
    for c in sorted_counts:
        cum += c
        tasks_for_50pct += 1
        if cum >= total * 0.5:
            break

    # Rare task count: tasks appearing exactly once
    singletons = sum(1 for c in task_counter.values() if c == 1)

    return {
        "name": ds_dir.name,
        "total": total,
        "n_unique_tasks": n_unique,
        "top1_task": top5[0][0] if top5 else "",
        "top1_count": top5[0][1] if top5 else 0,
        "top1_pct": top1_coverage,
        "top5_pct": top5_coverage,
        "tasks_for_50pct": tasks_for_50pct,
        "singletons": singletons,
        "top5_sample": top5,
        "robots": dict(robot_counter.most_common(3)),
        "depth_types": dict(depth_counter),
    }


def main():
    results = []
    for ds_dir in sorted(DATA_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        r = analyze_dataset(ds_dir)
        if r["total"] > 0:
            results.append(r)

    # Sort by total episodes descending
    results.sort(key=lambda x: x["total"], reverse=True)

    # === Summary table ===
    print(f"\n{'=' * 100}")
    print(f"  TASK DISTRIBUTION ANALYSIS")
    print(f"{'=' * 100}")
    print(f"{'dataset':<45} {'eps':>7} {'uniq':>6} {'top1%':>7} {'top5%':>7} "
          f"{'50%@':>6} {'solo':>6}  {'main_depth':>10}")
    print("-" * 100)

    grand_total = 0
    grand_unique = 0
    for r in results:
        depth_str = max(r["depth_types"], key=r["depth_types"].get) if r["depth_types"] else "?"
        print(f"{r['name']:<45} {r['total']:>7,} {r['n_unique_tasks']:>6,} "
              f"{r['top1_pct'] * 100:>6.1f}% {r['top5_pct'] * 100:>6.1f}% "
              f"{r['tasks_for_50pct']:>6,} {r['singletons']:>6,}  {depth_str:>10}")
        grand_total += r["total"]
        grand_unique += r["n_unique_tasks"]

    print("-" * 100)
    print(f"{'TOTAL':<45} {grand_total:>7,} {grand_unique:>6,}  "
          f"(note: unique counts may overlap across datasets)")

    # === Per dataset top tasks ===
    print(f"\n{'=' * 100}")
    print(f"  TOP TASKS PER DATASET (top-5 by frequency)")
    print(f"{'=' * 100}")
    for r in results:
        print(f"\n{r['name']}  (total={r['total']:,}, unique={r['n_unique_tasks']:,})")
        for task, count in r["top5_sample"]:
            pct = count / r["total"] * 100
            t_display = task[:80] + "..." if len(task) > 80 else task
            print(f"  {count:>6,} ({pct:5.1f}%)  {t_display!r}")

    # === Selection insights ===
    print(f"\n{'=' * 100}")
    print(f"  SELECTION INSIGHTS")
    print(f"{'=' * 100}")

    print("\n[Task diversity rank] — 'effective unique tasks per 1000 episodes'")
    diversity_ranked = sorted(results, key=lambda x: x["n_unique_tasks"] / max(x["total"], 1),
                              reverse=True)
    for r in diversity_ranked:
        density = r["n_unique_tasks"] / max(r["total"], 1) * 1000
        print(f"  {r['name']:<45} {density:7.1f} uniq/1k")

    print("\n[Redundancy rank] — 'top-5 tasks占比', high means same tasks repeat a lot")
    redundancy_ranked = sorted(results, key=lambda x: x["top5_pct"], reverse=True)
    for r in redundancy_ranked:
        marker = "❌ 高度重复" if r["top5_pct"] > 0.6 else ("⚠️" if r["top5_pct"] > 0.3 else "✅")
        print(f"  {r['name']:<45} top5={r['top5_pct'] * 100:5.1f}%  {marker}")

    print("\n[Quota suggestion per 30K budget] — 粗略配比")
    # Base heuristic: quota ∝ sqrt(n_unique_tasks) * base_weight
    # Adjust for depth rarity and special datasets
    SPECIAL_FULL = {
        "rlbench", "taco_play", "nyu_franka_play_dataset_converted_externally_to_rlds",
        "rh20t", "jaco_play", "berkeley_cable_routing",
    }
    OOD_RESERVED = {"aloha", "utokyo_xarm_bimanual_converted_externally_to_rlds"}

    fixed_alloc = 0
    flexible_pool = []
    for r in results:
        if r["name"] in OOD_RESERVED:
            continue
        if r["name"] in SPECIAL_FULL:
            take = r["total"]
            fixed_alloc += take
            print(f"  {r['name']:<45} FULL  {take:>6,}")
        else:
            flexible_pool.append(r)

    remaining = 30_000 - fixed_alloc
    # Diversity-weighted allocation among flexible
    import math
    weights = {r["name"]: math.sqrt(r["n_unique_tasks"]) for r in flexible_pool}
    w_sum = sum(weights.values())
    for r in flexible_pool:
        quota = int(remaining * weights[r["name"]] / w_sum)
        quota = min(quota, r["total"])
        print(f"  {r['name']:<45} SAMP  {quota:>6,} / {r['total']:,} "
              f"({quota / max(r['total'], 1) * 100:.1f}% sampling rate)")

    for name in OOD_RESERVED:
        r = next((x for x in results if x["name"] == name), None)
        if r:
            print(f"  {r['name']:<45} OOD   {0:>6}  (reserved: {r['total']:,} ep for embodiment OOD)")


if __name__ == "__main__":
    main()
