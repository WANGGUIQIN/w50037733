"""Stream episodes from lerobot/aloha_static_* bimanual datasets.

Lerobot v3 format: parquet (state+action+episode_index) + mp4 videos.
Images are stored as video files, not inline in parquet.
"""
import json
import os
import shutil

import numpy as np
from huggingface_hub import hf_hub_download

ALOHA_TASKS = [
    "battery", "candy", "coffee", "coffee_new", "cups_open",
    "fork_pick_up", "screw_driver", "tape", "thread_velcro",
    "towel", "vinh_cup", "vinh_cup_left", "ziploc_slide",
    "pingpong_test", "pro_pencil",
]

# Primary camera for ALOHA (overhead view, best for manipulation understanding)
CAMERA_KEY = "observation.images.cam_high"


def stream_aloha_datasets(cache_dir: str = "/tmp/aloha_cache"):
    """Yield episodes from all ALOHA bimanual tasks."""
    global_ep = 0

    for task_name in ALOHA_TASKS:
        repo = f"lerobot/aloha_static_{task_name}"
        print(f"    ALOHA: {task_name}")

        try:
            # Get episode info
            meta_path = hf_hub_download(repo, "meta/info.json", repo_type="dataset", cache_dir=cache_dir)
            with open(meta_path) as f:
                info = json.load(f)

            total_eps = info.get("total_episodes", 0)
            fps = info.get("fps", 50)
            video_path_template = info.get("video_path", "")
            chunks_size = info.get("chunks_size", 1000)

            # Get tasks
            tasks_path = hf_hub_download(repo, "meta/tasks.jsonl", repo_type="dataset", cache_dir=cache_dir)
            task_desc = task_name.replace("_", " ")
            try:
                with open(tasks_path) as f:
                    for line in f:
                        t = json.loads(line)
                        if "task" in t:
                            task_desc = t["task"]
                            break
            except Exception:
                pass

            # Get actions from parquet
            ep_actions = _load_all_actions(repo, info, cache_dir)

            # Process each episode
            task_ep_count = 0
            for ep_idx in range(total_eps):
                chunk_idx = ep_idx // chunks_size
                # Download video for this episode
                video_filename = f"videos/{CAMERA_KEY}/chunk-{chunk_idx:03d}/file-{ep_idx:03d}.mp4"

                try:
                    vid_path = hf_hub_download(repo, video_filename, repo_type="dataset", cache_dir=cache_dir)
                except Exception:
                    # Try alternative naming
                    try:
                        video_filename = f"videos/{CAMERA_KEY}/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.mp4"
                        vid_path = hf_hub_download(repo, video_filename, repo_type="dataset", cache_dir=cache_dir)
                    except Exception:
                        continue

                # Decode video
                rgb_frames = _decode_video(vid_path)
                _cleanup(vid_path)

                if len(rgb_frames) < 2:
                    continue

                actions = ep_actions.get(ep_idx)

                yield {
                    "episode_id": f"episode_{global_ep:06d}",
                    "rgb_frames": rgb_frames,
                    "actions": actions,
                    "task": task_desc,
                    "robot": "aloha_dual_viper",
                }
                global_ep += 1
                task_ep_count += 1

            print(f"      {task_ep_count} episodes")

        except Exception as e:
            print(f"      failed: {e}")

        # Cleanup
        dl_dir = os.path.join(cache_dir, "downloads")
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir, ignore_errors=True)


def _load_all_actions(repo, info, cache_dir):
    """Load actions from parquet files, grouped by episode."""
    import pyarrow.parquet as pq

    ep_actions = {}
    chunks_size = info.get("chunks_size", 1000)
    total_eps = info.get("total_episodes", 0)
    num_chunks = (total_eps + chunks_size - 1) // chunks_size

    for chunk_idx in range(num_chunks):
        # Try different parquet naming conventions
        for pq_name in [
            f"data/chunk-{chunk_idx:03d}/file-000.parquet",
            f"data/chunk-{chunk_idx:03d}/episode_{chunk_idx * chunks_size:06d}.parquet",
        ]:
            try:
                pq_path = hf_hub_download(repo, pq_name, repo_type="dataset", cache_dir=cache_dir)
                table = pq.read_table(pq_path)
                df = table.to_pandas()

                if "episode_index" in df.columns and "action" in df.columns:
                    for ep_idx, group in df.groupby("episode_index"):
                        acts = np.stack(group["action"].values).astype(np.float32)
                        ep_actions[int(ep_idx)] = acts
                elif "episode_index" in df.columns:
                    # Actions might be in separate columns
                    action_cols = [c for c in df.columns if c.startswith("action")]
                    if action_cols:
                        for ep_idx, group in df.groupby("episode_index"):
                            acts = group[action_cols].values.astype(np.float32)
                            ep_actions[int(ep_idx)] = acts

                _cleanup(pq_path)
                break
            except Exception:
                continue

    return ep_actions


def _decode_video(video_path: str, max_frames: int = 1000) -> list[np.ndarray]:
    import av

    frames = []
    try:
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break
    except Exception:
        pass
    return frames


def _cleanup(file_path: str):
    try:
        real = os.path.realpath(file_path)
        if os.path.exists(real) and real != file_path:
            os.remove(real)
        if os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass
