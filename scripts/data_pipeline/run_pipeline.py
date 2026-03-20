#!/usr/bin/env python3
"""Main data pipeline orchestrator.

Downloads datasets in priority order, extracts action-change keyframes,
generates pseudo-depth via Depth Anything V2, saves unified format.

Usage:
    python scripts/data_pipeline/run_pipeline.py
    python scripts/data_pipeline/run_pipeline.py --datasets bridge,droid
    python scripts/data_pipeline/run_pipeline.py --resume
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from keyframe_extractor import extract_keyframes, uniform_keyframes
from depth_generator import DepthGenerator
from episode_saver import save_episode
from disk_monitor import check_disk, get_free_gb

OUTPUT_DIR = "/home/w50037733/robobrain_3dgs/data/processed"
CACHE_DIR = "/tmp/data_pipeline_cache"
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
IMAGE_SIZE = 256
DEFAULT_INTRINSICS = [[222.7, 0, 128.0], [0, 222.7, 128.0], [0, 0, 1]]

# Processing order: small/local first, large last
DATASET_ORDER = [
    "rlbench",
    "aloha",
    "utokyo_xarm_bimanual_converted_externally_to_rlds",
    "berkeley_cable_routing",
    "taco_play",
    "jaco_play",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "bridge",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "fractal20220817_data",
    "rh20t",
    "droid",
]


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_datasets": [], "episode_counts": {}}


def save_progress(progress):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def get_episode_stream(dataset_name):
    """Return an episode generator for the given dataset."""
    if dataset_name == "rlbench":
        from downloaders.rlbench_local import stream_rlbench_local
        return stream_rlbench_local()
    elif dataset_name == "aloha":
        from downloaders.aloha import stream_aloha_datasets
        return stream_aloha_datasets(cache_dir=CACHE_DIR)
    elif dataset_name == "droid":
        from downloaders.droid import stream_droid_dataset
        return stream_droid_dataset(cache_dir=CACHE_DIR)
    elif dataset_name == "rh20t":
        from downloaders.rh20t import stream_rh20t_dataset
        return stream_rh20t_dataset(cache_dir=CACHE_DIR)
    else:
        from downloaders.oxe_tar import stream_oxe_dataset
        return stream_oxe_dataset(dataset_name, cache_dir=CACHE_DIR)


def process_dataset(dataset_name, depth_gen, progress):
    """Process one dataset: stream, extract keyframes, generate depth, save."""
    print(f"\n{'='*60}")
    print(f"  Processing: {dataset_name}")
    print(f"  Disk free: {get_free_gb():.0f} GB")
    print(f"{'='*60}")

    ep_count = 0
    t0 = time.time()

    for ep_data in get_episode_stream(dataset_name):
        # Disk check every 100 episodes
        if ep_count > 0 and ep_count % 100 == 0:
            status = check_disk()
            if status == "critical":
                print(f"  CRITICAL: disk low! Stopping at {ep_count} episodes.")
                break
            if status == "warn":
                print(f"  WARNING: disk getting low ({get_free_gb():.0f} GB free)")

        rgb_frames = ep_data["rgb_frames"]
        actions = ep_data.get("actions")

        # Extract keyframes
        if actions is not None and len(actions) >= 5:
            kf_indices = extract_keyframes(actions, num_keyframes=5)
        else:
            kf_indices = uniform_keyframes(len(rgb_frames), num_keyframes=5)

        # Clamp to valid range
        kf_indices = sorted(set(
            min(max(0, i), len(rgb_frames) - 1) for i in kf_indices
        ))

        # Select keyframe RGB
        kf_rgb = [rgb_frames[i] for i in kf_indices]

        # Depth: use native if available, else generate pseudo
        has_native = ep_data.get("depth_frames") is not None
        if has_native:
            native_depths = ep_data["depth_frames"]
            kf_depth = [
                native_depths[i] if i < len(native_depths)
                else native_depths[-1]
                for i in kf_indices
            ]
            depth_type = ep_data.get("depth_type", "native")
        else:
            pil_imgs = [
                Image.fromarray(f).resize((IMAGE_SIZE, IMAGE_SIZE))
                for f in kf_rgb
            ]
            kf_depth = depth_gen.estimate_batch(pil_imgs)
            depth_type = "pseudo"

        # Intrinsics
        intrinsics = ep_data.get("intrinsics", DEFAULT_INTRINSICS)

        save_episode(
            output_dir=OUTPUT_DIR,
            dataset_name=dataset_name,
            episode_id=ep_data["episode_id"],
            rgb_frames=kf_rgb,
            depth_frames=kf_depth,
            task=ep_data.get("task", "manipulation"),
            intrinsics=intrinsics,
            keyframe_indices=kf_indices,
            depth_type=depth_type,
            robot=ep_data.get("robot", "unknown"),
            num_frames_total=len(rgb_frames),
            image_size=IMAGE_SIZE,
        )

        ep_count += 1
        if ep_count % 500 == 0:
            elapsed = time.time() - t0
            rate = ep_count / elapsed
            print(
                f"  {dataset_name}: {ep_count} ep, "
                f"{rate:.1f} ep/s, "
                f"disk={get_free_gb():.0f}GB"
            )

    elapsed = time.time() - t0
    print(f"  Done: {dataset_name} — {ep_count} episodes in {elapsed:.0f}s")
    return ep_count


def main():
    parser = argparse.ArgumentParser(description="RoboBrain-3DGS Data Pipeline")
    parser.add_argument(
        "--datasets", type=str, default=None,
        help="Comma-separated datasets (default: all)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already completed datasets",
    )
    parser.add_argument(
        "--depth_device", type=str, default="cuda:1",
        help="GPU for Depth Anything V2",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()

    datasets = DATASET_ORDER
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    if args.resume:
        datasets = [d for d in datasets if d not in progress["completed_datasets"]]

    print("=" * 60)
    print("  RoboBrain-3DGS Data Pipeline")
    print("=" * 60)
    print(f"  Datasets: {datasets}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Disk free: {get_free_gb():.0f} GB")

    # Init depth generator
    print("\nLoading Depth Anything V2 vitl ...")
    depth_gen = DepthGenerator(device=args.depth_device, image_size=IMAGE_SIZE)
    print("  Ready.\n")

    total_episodes = 0
    t_start = time.time()

    for dataset_name in datasets:
        if check_disk() == "critical":
            print(f"\nCRITICAL: stopping, disk too low ({get_free_gb():.0f} GB)")
            break

        try:
            count = process_dataset(dataset_name, depth_gen, progress)
            progress["completed_datasets"].append(dataset_name)
            progress["episode_counts"][dataset_name] = count
            total_episodes += count
            save_progress(progress)
        except Exception as e:
            print(f"\nERROR: {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            save_progress(progress)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    print(f"  Total: {total_episodes} episodes in {total_time/3600:.1f}h")
    print(f"  Disk free: {get_free_gb():.0f} GB")
    for ds, count in progress.get("episode_counts", {}).items():
        print(f"    {ds}: {count}")


if __name__ == "__main__":
    main()
