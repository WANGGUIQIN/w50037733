#!/usr/bin/env python3
"""Deep data quality analysis for robotics datasets."""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

random.seed(42)
np.random.seed(42)

DATA_ROOT = Path("/home/edge/Embodied/robobrain_3dgs/data/processed")
DATASETS = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])

print("=" * 80)
print("DEEP DATA QUALITY ANALYSIS REPORT")
print("=" * 80)
print(f"\nDatasets found: {len(DATASETS)}")
for ds in DATASETS:
    print(f"  - {ds}")

# ============================================================
# Helper: collect all episodes
# ============================================================
def get_episodes(dataset, max_count=None):
    ds_path = DATA_ROOT / dataset
    eps = sorted([d for d in os.listdir(ds_path) if d.startswith("episode_")])
    if max_count:
        eps = eps[:max_count]
    return eps

def load_meta(dataset, episode):
    meta_path = DATA_ROOT / dataset / episode / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None

# ============================================================
# 1. TASK DESCRIPTION QUALITY
# ============================================================
print("\n" + "=" * 80)
print("1. TASK DESCRIPTION QUALITY")
print("=" * 80)

all_tasks_by_dataset = {}
all_issues = []

for ds in DATASETS:
    episodes = get_episodes(ds)
    tasks = []
    for ep in episodes:
        meta = load_meta(ds, ep)
        if meta and "task" in meta:
            tasks.append((ep, meta["task"]))
    all_tasks_by_dataset[ds] = tasks

for ds in DATASETS:
    tasks = all_tasks_by_dataset[ds]
    if not tasks:
        print(f"\n  [{ds}] NO TASKS FOUND - CRITICAL")
        continue

    task_texts = [t[1] for t in tasks]
    unique_tasks = set(task_texts)
    unique_lower = set(t.lower().strip() for t in task_texts)

    print(f"\n  [{ds}] {len(tasks)} episodes, {len(unique_tasks)} unique tasks, {len(unique_lower)} case-insensitive unique")

    # Case-insensitive duplicates
    if len(unique_tasks) != len(unique_lower):
        lower_map = defaultdict(set)
        for t in task_texts:
            lower_map[t.lower().strip()].add(t)
        case_dupes = {k: v for k, v in lower_map.items() if len(v) > 1}
        if case_dupes:
            print(f"    WARNING: {len(case_dupes)} case-insensitive duplicate groups:")
            for k, variants in list(case_dupes.items())[:5]:
                print(f"      '{k}' has variants: {variants}")

    # Task distribution
    task_counts = Counter(task_texts)
    print(f"    Top 5 most common tasks:")
    for task, count in task_counts.most_common(5):
        print(f"      [{count:4d}x] {task[:80]}{'...' if len(task) > 80 else ''}")

    # ALL CAPS check
    all_caps = [(ep, t) for ep, t in tasks if t == t.upper() and len(t) > 3]
    if all_caps:
        print(f"    WARNING: {len(all_caps)} ALL CAPS tasks")
        for ep, t in all_caps[:3]:
            print(f"      {ep}: '{t}'")

    # Very short tasks
    short_tasks = [(ep, t) for ep, t in tasks if len(t) < 5]
    if short_tasks:
        print(f"    WARNING: {len(short_tasks)} very short tasks (<5 chars)")
        for ep, t in short_tasks[:5]:
            print(f"      {ep}: '{t}'")

    # Very long tasks
    long_tasks = [(ep, t) for ep, t in tasks if len(t) > 200]
    if long_tasks:
        print(f"    WARNING: {len(long_tasks)} very long tasks (>200 chars)")
        for ep, t in long_tasks[:3]:
            print(f"      {ep}: '{t[:80]}...' (len={len(t)})")

    # No-space compound words (camelCase or smashed words)
    no_space = []
    for ep, t in tasks:
        # Check for words longer than 20 chars without spaces that look like compounds
        words = t.split()
        for w in words:
            if len(w) > 20 and not w.startswith("http") and not w.startswith("/"):
                no_space.append((ep, t, w))
                break
    if no_space:
        print(f"    WARNING: {len(no_space)} tasks with potential compound words (no spaces):")
        for ep, t, w in no_space[:3]:
            print(f"      {ep}: word='{w[:50]}'")

    # Non-ASCII characters
    non_ascii = [(ep, t) for ep, t in tasks if not all(ord(c) < 128 for c in t)]
    if non_ascii:
        print(f"    WARNING: {len(non_ascii)} tasks with non-ASCII characters")
        for ep, t in non_ascii[:3]:
            bad_chars = [c for c in t if ord(c) >= 128]
            print(f"      {ep}: '{t[:60]}' non-ASCII: {bad_chars[:5]}")

    # Empty or whitespace-only
    empty_tasks = [(ep, t) for ep, t in tasks if not t.strip()]
    if empty_tasks:
        print(f"    CRITICAL: {len(empty_tasks)} empty/whitespace-only tasks")

# ============================================================
# 2. IMAGE QUALITY SAMPLING
# ============================================================
print("\n" + "=" * 80)
print("2. IMAGE QUALITY SAMPLING (50 random episodes)")
print("=" * 80)

# Collect all episodes across datasets
all_episodes = []
for ds in DATASETS:
    episodes = get_episodes(ds)
    for ep in episodes:
        all_episodes.append((ds, ep))

sample_size = min(50, len(all_episodes))
sampled = random.sample(all_episodes, sample_size)

resolution_by_dataset = defaultdict(list)
corrupt_images = []
duplicate_frames = []
total_images_checked = 0

for ds, ep in sampled:
    ep_path = DATA_ROOT / ds / ep
    rgb_files = sorted([f for f in os.listdir(ep_path) if f.startswith("rgb_") and f.endswith(".png")])

    prev_pixels = None
    for rgb_f in rgb_files:
        rgb_path = ep_path / rgb_f
        try:
            img = Image.open(rgb_path)
            w, h = img.size
            resolution_by_dataset[ds].append((w, h))
            total_images_checked += 1

            arr = np.array(img)
            # Check all-black
            if arr.max() == 0:
                corrupt_images.append((ds, ep, rgb_f, "ALL BLACK"))
            # Check all-white
            elif arr.min() == 255:
                corrupt_images.append((ds, ep, rgb_f, "ALL WHITE"))
            # Check very low variance (near-uniform)
            elif arr.std() < 1.0:
                corrupt_images.append((ds, ep, rgb_f, f"NEAR-UNIFORM (std={arr.std():.2f})"))

            # Check consecutive duplicate
            flat = arr.flatten()
            if prev_pixels is not None and flat.shape == prev_pixels.shape and np.array_equal(flat, prev_pixels):
                duplicate_frames.append((ds, ep, rgb_f, "IDENTICAL to previous frame"))
            prev_pixels = flat
        except Exception as e:
            corrupt_images.append((ds, ep, rgb_f, f"LOAD ERROR: {e}"))

print(f"\n  Total images checked: {total_images_checked}")

print(f"\n  Resolution distribution per dataset:")
for ds in sorted(resolution_by_dataset.keys()):
    res_counts = Counter(resolution_by_dataset[ds])
    res_str = ", ".join(f"{w}x{h} ({c})" for (w, h), c in res_counts.most_common())
    print(f"    [{ds}] {res_str}")

if corrupt_images:
    print(f"\n  CORRUPT/SUSPICIOUS IMAGES: {len(corrupt_images)}")
    for ds, ep, f, reason in corrupt_images:
        print(f"    {ds}/{ep}/{f}: {reason}")
else:
    print(f"\n  No corrupt images found in sample.")

if duplicate_frames:
    print(f"\n  DUPLICATE CONSECUTIVE FRAMES: {len(duplicate_frames)}")
    for ds, ep, f, reason in duplicate_frames:
        print(f"    {ds}/{ep}/{f}: {reason}")
else:
    print(f"\n  No duplicate consecutive frames found in sample.")

# ============================================================
# 3. DEPTH DATA QUALITY
# ============================================================
print("\n" + "=" * 80)
print("3. DEPTH DATA QUALITY (30 sampled episodes)")
print("=" * 80)

# Find datasets with depth
depth_episodes = []
for ds, ep in all_episodes:
    ep_path = DATA_ROOT / ds / ep
    depth_files = [f for f in os.listdir(ep_path) if f.startswith("depth_") and f.endswith(".npy")]
    if depth_files:
        depth_episodes.append((ds, ep))

depth_sample_size = min(30, len(depth_episodes))
if depth_episodes:
    depth_sampled = random.sample(depth_episodes, depth_sample_size)
else:
    depth_sampled = []

print(f"\n  Episodes with depth data: {len(depth_episodes)} / {len(all_episodes)}")
print(f"  Sampling {depth_sample_size} for analysis")

depth_stats_by_dataset = defaultdict(lambda: {"mins": [], "maxs": [], "means": [], "zeros": 0, "negative": 0, "huge": 0, "total": 0, "res_mismatches": 0})
depth_issues = []

for ds, ep in depth_sampled:
    ep_path = DATA_ROOT / ds / ep
    depth_files = sorted([f for f in os.listdir(ep_path) if f.startswith("depth_") and f.endswith(".npy")])
    rgb_files = sorted([f for f in os.listdir(ep_path) if f.startswith("rgb_") and f.endswith(".png")])

    for df in depth_files:
        try:
            depth = np.load(ep_path / df)
            stats = depth_stats_by_dataset[ds]
            stats["total"] += 1

            d_min = float(depth.min())
            d_max = float(depth.max())
            d_mean = float(depth.mean())
            stats["mins"].append(d_min)
            stats["maxs"].append(d_max)
            stats["means"].append(d_mean)

            # All-zero check
            if d_max == 0:
                stats["zeros"] += 1
                depth_issues.append((ds, ep, df, "ALL ZERO"))

            # Negative values
            if d_min < 0:
                stats["negative"] += 1
                depth_issues.append((ds, ep, df, f"NEGATIVE depth: min={d_min:.4f}"))

            # Unrealistically large (>100m)
            if d_max > 100:
                stats["huge"] += 1
                depth_issues.append((ds, ep, df, f"HUGE depth: max={d_max:.2f}"))

            # Resolution match check
            idx = df.replace("depth_", "").replace(".npy", "")
            rgb_match = f"rgb_{idx}.png"
            if rgb_match in rgb_files:
                try:
                    img = Image.open(ep_path / rgb_match)
                    rw, rh = img.size
                    dh, dw = depth.shape[:2]
                    if (dw, dh) != (rw, rh):
                        stats["res_mismatches"] += 1
                        if stats["res_mismatches"] <= 2:
                            depth_issues.append((ds, ep, df, f"RES MISMATCH: depth={dw}x{dh} vs rgb={rw}x{rh}"))
                except:
                    pass
        except Exception as e:
            depth_issues.append((ds, ep, df, f"LOAD ERROR: {e}"))

print(f"\n  Depth statistics per dataset:")
for ds in sorted(depth_stats_by_dataset.keys()):
    s = depth_stats_by_dataset[ds]
    if s["total"] == 0:
        continue
    print(f"\n    [{ds}] ({s['total']} depth maps sampled)")
    print(f"      Range: min={np.min(s['mins']):.4f}, max={np.max(s['maxs']):.4f}, mean_of_means={np.mean(s['means']):.4f}")
    if s["zeros"]:
        print(f"      WARNING: {s['zeros']} all-zero depth maps")
    if s["negative"]:
        print(f"      WARNING: {s['negative']} with negative values")
    if s["huge"]:
        print(f"      WARNING: {s['huge']} with values >100m")
    if s["res_mismatches"]:
        print(f"      WARNING: {s['res_mismatches']} resolution mismatches vs RGB")

if depth_issues:
    print(f"\n  Depth issues found ({len(depth_issues)} total):")
    for ds, ep, f, reason in depth_issues[:20]:
        print(f"    {ds}/{ep}/{f}: {reason}")
    if len(depth_issues) > 20:
        print(f"    ... and {len(depth_issues) - 20} more")

# ============================================================
# 4. KEYFRAME DISTRIBUTION
# ============================================================
print("\n" + "=" * 80)
print("4. KEYFRAME DISTRIBUTION")
print("=" * 80)

keyframe_issues = []
kf_dist_by_dataset = defaultdict(lambda: Counter())

for ds in DATASETS:
    episodes = get_episodes(ds)
    non_monotonic = 0
    out_of_bounds = 0

    for ep in episodes:
        meta = load_meta(ds, ep)
        if not meta:
            continue

        nk = meta.get("num_keyframes", "?")
        kf_dist_by_dataset[ds][nk] += 1

        kf_indices = meta.get("keyframe_indices", [])
        num_total = meta.get("num_frames_total", None)

        # Monotonically increasing check
        if kf_indices and not all(kf_indices[i] < kf_indices[i+1] for i in range(len(kf_indices)-1)):
            non_monotonic += 1
            if non_monotonic <= 2:
                keyframe_issues.append((ds, ep, f"NON-MONOTONIC indices: {kf_indices}"))

        # Bounds check
        if kf_indices and num_total is not None:
            if max(kf_indices) >= num_total:
                out_of_bounds += 1
                if out_of_bounds <= 2:
                    keyframe_issues.append((ds, ep, f"OUT OF BOUNDS: max_idx={max(kf_indices)}, total={num_total}"))

        # Consistency: num_keyframes vs actual files
        actual_rgb = len([f for f in os.listdir(DATA_ROOT / ds / ep) if f.startswith("rgb_") and f.endswith(".png")])
        if actual_rgb != nk:
            if len(keyframe_issues) < 50:
                keyframe_issues.append((ds, ep, f"KEYFRAME COUNT MISMATCH: meta says {nk}, found {actual_rgb} RGB files"))

    print(f"\n  [{ds}]")
    print(f"    Keyframe count distribution: {dict(kf_dist_by_dataset[ds].most_common())}")
    if non_monotonic:
        print(f"    WARNING: {non_monotonic} episodes with non-monotonic keyframe indices")
    if out_of_bounds:
        print(f"    WARNING: {out_of_bounds} episodes with keyframe indices out of bounds")

if keyframe_issues:
    print(f"\n  Keyframe issues ({len(keyframe_issues)} total):")
    for ds, ep, reason in keyframe_issues[:20]:
        print(f"    {ds}/{ep}: {reason}")
    if len(keyframe_issues) > 20:
        print(f"    ... and {len(keyframe_issues) - 20} more")

# ============================================================
# 5. CROSS-DATASET STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("5. CROSS-DATASET STATISTICS")
print("=" * 80)

robot_counter = Counter()
depth_type_counter = Counter()
dataset_sizes = {}

for ds in DATASETS:
    # Disk size
    ds_path = DATA_ROOT / ds
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(ds_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except:
                pass
    dataset_sizes[ds] = total_size

    # Sample first few episodes for robot/depth_type
    episodes = get_episodes(ds)
    robots_in_ds = set()
    dtypes_in_ds = set()
    for ep in episodes[:20]:  # sample 20 to check consistency
        meta = load_meta(ds, ep)
        if meta:
            r = meta.get("robot", "MISSING")
            d = meta.get("depth_type", "MISSING")
            robots_in_ds.add(r)
            dtypes_in_ds.add(d)
            robot_counter[r] += 1
            depth_type_counter[d] += 1

    print(f"\n  [{ds}]")
    print(f"    Size: {total_size / (1024**3):.2f} GB")
    print(f"    Episodes: {len(episodes)}")
    print(f"    Robots: {robots_in_ds}")
    print(f"    Depth types: {dtypes_in_ds}")

print(f"\n  Robot type distribution (sampled):")
for robot, count in robot_counter.most_common():
    print(f"    {robot}: {count}")

print(f"\n  Depth type distribution (sampled):")
for dt, count in depth_type_counter.most_common():
    print(f"    {dt}: {count}")

print(f"\n  Dataset sizes sorted:")
for ds, size in sorted(dataset_sizes.items(), key=lambda x: -x[1]):
    print(f"    {ds}: {size / (1024**3):.2f} GB")

# ============================================================
# 6. META FIELD COMPLETENESS
# ============================================================
print("\n" + "=" * 80)
print("6. META FIELD COMPLETENESS")
print("=" * 80)

required_fields = ["task", "intrinsics", "keyframe_indices", "source", "episode_id",
                   "depth_type", "robot", "num_frames_total", "num_keyframes"]

for ds in DATASETS:
    episodes = get_episodes(ds)
    field_presence = defaultdict(int)
    all_fields = set()
    missing_critical = defaultdict(list)

    for ep in episodes:
        meta = load_meta(ds, ep)
        if meta is None:
            missing_critical["meta.json"].append(ep)
            continue
        for k in meta:
            field_presence[k] += 1
            all_fields.add(k)
        for req in required_fields:
            if req not in meta:
                missing_critical[req].append(ep)

    print(f"\n  [{ds}] ({len(episodes)} episodes)")
    print(f"    Fields present: {sorted(all_fields)}")
    for field in required_fields:
        count = field_presence.get(field, 0)
        pct = 100.0 * count / len(episodes) if episodes else 0
        status = "OK" if pct == 100 else f"MISSING in {len(episodes) - count} eps"
        if pct < 100:
            print(f"    WARNING: '{field}' present in {count}/{len(episodes)} ({pct:.1f}%) - {status}")
            # Show first few missing
            if missing_critical[field]:
                print(f"             First missing: {missing_critical[field][:3]}")

    # Extra fields not in required
    extra = all_fields - set(required_fields)
    if extra:
        print(f"    Extra fields: {sorted(extra)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("ACTIONABLE SUMMARY")
print("=" * 80)

total_episodes = sum(len(get_episodes(ds)) for ds in DATASETS)
total_size = sum(dataset_sizes.values())
print(f"\n  Total: {len(DATASETS)} datasets, {total_episodes} episodes, {total_size / (1024**3):.2f} GB")

print(f"\n  Key findings:")
print(f"  - Corrupt/suspicious images in sample: {len(corrupt_images)}")
print(f"  - Duplicate consecutive frames in sample: {len(duplicate_frames)}")
print(f"  - Depth issues in sample: {len(depth_issues)}")
print(f"  - Keyframe issues: {len(keyframe_issues)}")

print("\nDone.")
