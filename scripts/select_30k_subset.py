#!/usr/bin/env python3
"""Select a 30K-episode subset of processed data for first-version training.

Strategy:
  1. Skip 4 placeholder-task datasets (furniture_bench, berkeley_cable, nyu_franka, utokyo_xarm).
     Except utokyo_xarm + aloha are reserved as OOD-embodiment.
  2. Full-take rare high-quality datasets: rlbench, taco_play, rh20t, jaco_play.
  3. Stratified sampling on large datasets: bridge (7K), droid (10K), fractal (3K).
     Within each, apply semantic dedup with sentence-transformer (cosine > 0.88 merged).
  4. Hold out 3 RLBench task families as OOD-task.
  5. Within-task split: val 5% + test_id 5% + train 90% per task cluster.

Outputs 5 JSON files to data/splits/:
    train.json         ~27,000 ep
    val.json           ~1,500 ep    (for eval during training)
    test_id.json       ~1,500 ep    (in-distribution final test)
    test_ood_task.json ~270 ep      (RLBench unseen task families)
    test_ood_embo.json ~83 ep       (aloha + utokyo_xarm, bimanual OOD)
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/home/edge/Embodied/robobrain_3dgs/data/processed")
OUT_DIR = Path("/home/edge/Embodied/robobrain_3dgs/data/splits")

SKIP_DATASETS = {
    "furniture_bench_dataset_converted_externally_to_rlds",
    "berkeley_cable_routing",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
}
OOD_EMBODIMENT_DATASETS = {
    "aloha",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
}
FULL_TAKE = {"rlbench", "taco_play", "rh20t", "jaco_play"}
SAMPLING_QUOTAS = {"bridge": 7000, "droid": 10000, "fractal20220817_data": 3000}

# RLBench task OOD — 3 task families that fully stay out of training.
# These strings are regex patterns matched against the task field.
OOD_RLBENCH_PATTERNS = [
    r"insert .* onto the .* spoke",      # insert family: ring onto spoke
    r"slide (the )?block to .*target",   # sliding family
    r"stack (\d+ )?.*(block|cup)",       # stacking family
]

SIM_THRESHOLD = 0.88
VAL_RATIO = 0.05
TEST_ID_RATIO = 0.05


def load_episode_meta(ep_dir: Path) -> dict | None:
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    return meta


def iter_dataset(ds_name: str):
    ds_dir = DATA_ROOT / ds_name
    if not ds_dir.exists():
        return
    for ep in sorted(ds_dir.iterdir()):
        if not (ep.is_dir() and ep.name.startswith("episode_")):
            continue
        meta = load_episode_meta(ep)
        if meta is None:
            continue
        yield {
            "dataset": ds_name,
            "episode_id": ep.name,
            "task": (meta.get("task") or "").strip().lower(),
            "robot": meta.get("robot", "unknown"),
            "depth_type": meta.get("depth_type", "unknown"),
            "num_keyframes": meta.get("num_keyframes", 0),
        }


def semantic_dedup(task_strs: list[str], model) -> dict[str, int]:
    """Greedy dedup. Returns task -> cluster_id map."""
    if not task_strs:
        return {}
    embs = model.encode(task_strs, batch_size=64, convert_to_numpy=True,
                        show_progress_bar=False, normalize_embeddings=True)

    cluster_id_of: dict[str, int] = {}
    cluster_centroids: list[np.ndarray] = []

    for task, emb in zip(task_strs, embs):
        if not cluster_centroids:
            cluster_id_of[task] = 0
            cluster_centroids.append(emb)
            continue
        sims = np.array([np.dot(emb, c) for c in cluster_centroids])
        best = int(sims.argmax())
        if sims[best] >= SIM_THRESHOLD:
            cluster_id_of[task] = best
        else:
            cluster_id_of[task] = len(cluster_centroids)
            cluster_centroids.append(emb)
    return cluster_id_of


def stratified_sample(
    episodes: list[dict],
    quota: int,
    cluster_of: dict[str, int],
    min_per_cluster: int = 1,
    max_per_cluster: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample quota episodes stratified by semantic cluster."""
    rng = random.Random(seed)
    # Group by cluster
    buckets: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes:
        cid = cluster_of.get(ep["task"], -1)
        buckets[cid].append(ep)

    n_clusters = len(buckets)
    total = len(episodes)
    if total <= quota:
        return episodes[:]

    # If there are more clusters than the budget, grab the top-quota largest
    # clusters (one episode each). This happens on droid where 53% of tasks
    # are singletons and dedup still leaves ~15K clusters.
    if n_clusters >= quota:
        sorted_buckets = sorted(buckets.items(), key=lambda x: len(x[1]), reverse=True)
        return [rng.choice(eps) for _, eps in sorted_buckets[:quota]]

    # Initial quota: proportional with floor
    cluster_quota: dict[int, int] = {}
    for cid, eps in buckets.items():
        q = max(min_per_cluster, int(quota * len(eps) / total))
        if max_per_cluster is not None:
            q = min(q, max_per_cluster)
        q = min(q, len(eps))
        cluster_quota[cid] = q

    # Rebalance to hit quota
    while sum(cluster_quota.values()) > quota:
        over = sum(cluster_quota.values()) - quota
        sorted_cids = sorted(cluster_quota, key=cluster_quota.get, reverse=True)
        for cid in sorted_cids[:over]:
            if cluster_quota[cid] > min_per_cluster:
                cluster_quota[cid] -= 1
        if all(cluster_quota[cid] <= min_per_cluster for cid in cluster_quota):
            break

    while sum(cluster_quota.values()) < quota:
        under = quota - sum(cluster_quota.values())
        growable = [cid for cid, q in cluster_quota.items() if q < len(buckets[cid])]
        if not growable:
            break
        growable.sort(key=lambda cid: cluster_quota[cid])  # smallest first
        for cid in growable[:under]:
            if cluster_quota[cid] < len(buckets[cid]):
                cluster_quota[cid] += 1

    selected = []
    for cid, q in cluster_quota.items():
        pool = buckets[cid]
        if q >= len(pool):
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, q))
    return selected


def split_train_val_test_id(
    episodes: list[dict],
    cluster_of: dict[str, int] | None = None,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_ID_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Within-task split. Fine-grained clusters (e.g. droid) fall back to random split."""
    rng = random.Random(seed)
    group_key = (lambda t: cluster_of.get(t, -1)) if cluster_of else (lambda t: t)
    groups: dict = defaultdict(list)
    for ep in episodes:
        groups[group_key(ep["task"])].append(ep)

    # If clusters are too fine-grained (average < 10 ep/cluster), fall back to
    # a single random pool split — per-cluster 5% truncates to 0 otherwise.
    avg_size = len(episodes) / max(len(groups), 1)
    if avg_size < 10:
        shuffled = list(episodes)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        return shuffled[n_val + n_test:], shuffled[:n_val], shuffled[n_val:n_val + n_test]

    train, val, test = [], [], []
    for g, eps in groups.items():
        n = len(eps)
        rng.shuffle(eps)
        n_val = max(0, int(round(n * val_ratio)))
        n_test = max(0, int(round(n * test_ratio)))
        if n - n_val - n_test < 1:
            n_val = min(n_val, max(0, n - 1 - n_test))
        val.extend(eps[:n_val])
        test.extend(eps[n_val:n_val + n_test])
        train.extend(eps[n_val + n_test:])
    return train, val, test


def write_split(path: Path, name: str, episodes: list[dict], meta_extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_ds = defaultdict(int)
    for ep in episodes:
        by_ds[ep["dataset"]] += 1
    out = {
        "split": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(episodes),
        "by_dataset": dict(by_ds),
        "episodes": [{"dataset": e["dataset"], "episode_id": e["episode_id"]} for e in episodes],
    }
    if meta_extra:
        out.update(meta_extra)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path.name}  ({len(episodes):,} ep across {len(by_ds)} datasets)")


def main():
    global SIM_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--similarity", type=float, default=SIM_THRESHOLD)
    args = parser.parse_args()
    SIM_THRESHOLD = args.similarity
    out_dir = Path(args.out)
    random.seed(args.seed)

    print("=" * 70)
    print("  30K Subset Selection")
    print("=" * 70)

    # --- Scan all candidate datasets ---
    all_episodes_by_ds: dict[str, list[dict]] = {}
    for ds_name in sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()):
        if ds_name in SKIP_DATASETS:
            continue
        eps = list(iter_dataset(ds_name))
        if eps:
            all_episodes_by_ds[ds_name] = eps
            print(f"  scanned {ds_name}: {len(eps):,} ep")

    # --- Pull out OOD-embodiment ---
    ood_embo = []
    for ds in OOD_EMBODIMENT_DATASETS:
        if ds in all_episodes_by_ds:
            ood_embo.extend(all_episodes_by_ds.pop(ds))
    print(f"\nOOD-embodiment reserved: {len(ood_embo):,} ep")

    # --- Pull out OOD-task from RLBench ---
    print("\nExtracting OOD-task (RLBench unseen families) ...")
    rl_eps = all_episodes_by_ds.get("rlbench", [])
    ood_task = []
    kept_rl = []
    patterns = [re.compile(p, re.IGNORECASE) for p in OOD_RLBENCH_PATTERNS]
    ood_tasks_set = set()
    for ep in rl_eps:
        if any(pat.search(ep["task"]) for pat in patterns):
            ood_task.append(ep)
            ood_tasks_set.add(ep["task"])
        else:
            kept_rl.append(ep)
    all_episodes_by_ds["rlbench"] = kept_rl
    print(f"  OOD-task: {len(ood_task):,} ep, {len(ood_tasks_set)} unique task strings")
    print(f"  Sample OOD tasks:")
    for t in sorted(ood_tasks_set)[:5]:
        print(f"    - {t!r}")
    print(f"  rlbench remaining for train: {len(kept_rl):,} ep")

    # --- Load sentence transformer ---
    print("\nLoading sentence-transformer model (all-MiniLM-L6-v2) ...")
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # --- Full-take datasets ---
    taken_total = []
    print("\n[Full-take datasets]")
    for ds in sorted(FULL_TAKE):
        if ds in all_episodes_by_ds:
            eps = all_episodes_by_ds[ds]
            taken_total.extend(eps)
            print(f"  {ds}: {len(eps):,} ep (FULL)")

    # --- Sampling datasets (with semantic dedup) ---
    print("\n[Sampling datasets with semantic dedup]")
    for ds, quota in SAMPLING_QUOTAS.items():
        if ds not in all_episodes_by_ds:
            continue
        eps = all_episodes_by_ds[ds]
        unique_tasks = sorted({ep["task"] for ep in eps if ep["task"]})
        print(f"\n  {ds}: {len(eps):,} ep, {len(unique_tasks):,} unique task strings")
        t0 = time.time()
        cluster_of = semantic_dedup(unique_tasks, model)
        n_clusters = len(set(cluster_of.values())) if cluster_of else 0
        print(f"    dedup: {len(unique_tasks):,} tasks -> {n_clusters:,} semantic clusters "
              f"({time.time() - t0:.1f}s)")
        # Max per cluster: avoid one cluster saturating
        max_per = max(2, quota // max(n_clusters, 1) * 3)
        picked = stratified_sample(eps, quota, cluster_of,
                                   min_per_cluster=1, max_per_cluster=max_per, seed=args.seed)
        print(f"    picked: {len(picked):,} / {quota:,} target")
        # Attach cluster_of for later in-task split
        for ep in picked:
            ep["_cluster"] = cluster_of.get(ep["task"], -1)
        taken_total.extend(picked)

    # --- For full-take datasets, compute clusters too (for within-task split) ---
    print("\n[Clustering full-take datasets for in-task split]")
    for ds in FULL_TAKE:
        ds_eps = [e for e in taken_total if e["dataset"] == ds]
        unique_tasks = sorted({ep["task"] for ep in ds_eps if ep["task"]})
        cluster_of = semantic_dedup(unique_tasks, model)
        for ep in ds_eps:
            ep["_cluster"] = cluster_of.get(ep["task"], -1)
        print(f"  {ds}: {len(unique_tasks)} tasks -> {len(set(cluster_of.values()))} clusters")

    # --- Per-dataset within-task split ---
    print("\n[Within-task split (val 5%, test_id 5%, train 90%)]")
    train_all, val_all, test_id_all = [], [], []
    by_dataset = defaultdict(list)
    for ep in taken_total:
        by_dataset[ep["dataset"]].append(ep)

    for ds, eps in by_dataset.items():
        # Build cluster_of from episode metadata
        cluster_of = {ep["task"]: ep["_cluster"] for ep in eps if "_cluster" in ep}
        tr, vl, te = split_train_val_test_id(
            eps, cluster_of=cluster_of, seed=args.seed,
        )
        train_all.extend(tr)
        val_all.extend(vl)
        test_id_all.extend(te)
        print(f"  {ds}: train={len(tr):,} val={len(vl):,} test_id={len(te):,}")

    # Strip _cluster field before saving
    for ep in train_all + val_all + test_id_all:
        ep.pop("_cluster", None)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Final splits")
    print("=" * 70)
    print(f"  train          : {len(train_all):,} ep")
    print(f"  val            : {len(val_all):,} ep")
    print(f"  test_id        : {len(test_id_all):,} ep")
    print(f"  test_ood_task  : {len(ood_task):,} ep  (RLBench held-out families)")
    print(f"  test_ood_embo  : {len(ood_embo):,} ep  (aloha + utokyo_xarm bimanual)")
    print(f"  TOTAL          : {len(train_all) + len(val_all) + len(test_id_all) + len(ood_task) + len(ood_embo):,} ep")

    # --- Write files ---
    print(f"\nWriting to {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(out_dir / "train.json", "train", train_all,
                meta_extra={"seed": args.seed, "similarity_threshold": SIM_THRESHOLD})
    write_split(out_dir / "val.json", "val", val_all)
    write_split(out_dir / "test_id.json", "test_id", test_id_all)
    write_split(out_dir / "test_ood_task.json", "test_ood_task", ood_task,
                meta_extra={"ood_patterns": OOD_RLBENCH_PATTERNS})
    write_split(out_dir / "test_ood_embo.json", "test_ood_embo", ood_embo)

    print("\nDone.")


if __name__ == "__main__":
    main()
