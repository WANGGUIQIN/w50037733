"""Stream episodes from cadene/droid (lerobot format: parquet + mp4).

Only yields episodes that have real task descriptions (from tasks.jsonl or
the episode-level task mapping). Episodes without annotations are skipped.
"""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

DROID_REPO = "cadene/droid"
NUM_CHUNKS = 93
EPISODES_PER_CHUNK = 1000
CAMERA = "observation.images.exterior_image_1_left"
DROID_INTRINSICS = [[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1]]

# Global caches
_TASK_MAP = None
_EPISODE_TASK_MAP = None


def _load_task_map(cache_dir):
    """Load task_index -> task description mapping from meta/tasks.jsonl."""
    global _TASK_MAP
    if _TASK_MAP is not None:
        return _TASK_MAP

    _TASK_MAP = {}
    try:
        tasks_path = hf_hub_download(
            DROID_REPO, "meta/tasks.jsonl", repo_type="dataset", cache_dir=cache_dir
        )
        with open(tasks_path) as f:
            for line in f:
                task_data = json.loads(line)
                task_idx = task_data.get("task_index", len(_TASK_MAP))
                task_desc = task_data.get("task", "")
                _TASK_MAP[task_idx] = task_desc
        print(f"    Loaded {len(_TASK_MAP)} task descriptions from tasks.jsonl")
    except Exception as e:
        print(f"    Warning: Could not load tasks.jsonl: {e}")

    return _TASK_MAP


def _load_episode_task_map(data_dir):
    """Load episode_index -> task description from droid_episode_tasks.json.

    This provides a pre-computed mapping of which episodes have real task
    annotations. Episodes not in this map are skipped during processing.
    """
    global _EPISODE_TASK_MAP
    if _EPISODE_TASK_MAP is not None:
        return _EPISODE_TASK_MAP

    path = os.path.join(data_dir, "droid_episode_tasks.json")
    if os.path.exists(path):
        with open(path) as f:
            _EPISODE_TASK_MAP = json.load(f)
        print(f"    Loaded {len(_EPISODE_TASK_MAP)} episode task annotations (filter)")
    else:
        _EPISODE_TASK_MAP = {}
        print(f"    Warning: {path} not found, no episode filter applied")

    return _EPISODE_TASK_MAP


def stream_droid_dataset(
    max_chunks: int = 93,
    cache_dir: str = "/tmp/droid_cache",
    start_chunk: int = 0,
    data_dir: str = None,
):
    """Yield episodes from DROID, one chunk at a time.

    Only episodes with real task descriptions are yielded. Episodes without
    annotations in either tasks.jsonl or droid_episode_tasks.json are skipped.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Load task mappings
    task_map = _load_task_map(cache_dir)
    ep_task_map = _load_episode_task_map(data_dir) if data_dir else {}

    for chunk_idx in range(start_chunk, max_chunks):
        chunk_name = f"chunk-{chunk_idx:03d}"
        start_ep = chunk_idx * EPISODES_PER_CHUNK
        ep_count = 0
        skipped = 0

        print(f"    DROID {chunk_name} ({chunk_idx+1}/{max_chunks})")

        for ep_offset in range(EPISODES_PER_CHUNK):
            ep_idx = start_ep + ep_offset
            ep_name = f"episode_{ep_idx:06d}"

            # Skip episodes without task annotations early (before downloading)
            if ep_task_map and str(ep_idx) not in ep_task_map:
                skipped += 1
                continue

            try:
                ep = _process_episode(chunk_name, ep_name, cache_dir, task_map)
            except Exception:
                continue

            if ep is not None:
                # Final check: reject fallback descriptions
                if ep.get("task") in ("robotic manipulation", "manipulation", ""):
                    skipped += 1
                    continue
                yield ep
                ep_count += 1

        print(f"      {ep_count} episodes yielded, {skipped} skipped (no annotation)")


def _process_episode(chunk_name, ep_name, cache_dir, task_map):
    """Download and process a single DROID episode."""
    import pyarrow.parquet as pq

    # Download parquet
    pq_path = _download(f"data/{chunk_name}/{ep_name}.parquet", cache_dir)
    if pq_path is None:
        return None

    # Read metadata
    try:
        table = pq.read_table(pq_path)
        df = table.to_pandas()
    except Exception:
        _cleanup(pq_path)
        return None

    # Extract actions (each cell is a list, need to stack)
    action_cols = sorted([c for c in df.columns if c.startswith("action")])
    if action_cols:
        try:
            # Each column contains arrays, stack them into a matrix
            action_arrays = [np.stack(df[col].values) for col in action_cols]
            actions = np.concatenate(action_arrays, axis=1).astype(np.float32)
        except Exception:
            actions = None
    else:
        actions = None

    # Extract task description from task_index -> tasks.jsonl mapping
    task = "robotic manipulation"
    if "task_index" in df.columns:
        task_idx = int(df["task_index"].iloc[0])
        task = task_map.get(task_idx, task)
        if not task:  # Empty string fallback
            task = "robotic manipulation"

    _cleanup(pq_path)

    # Download and decode video
    vid_path = _download(f"videos/{chunk_name}/{CAMERA}/{ep_name}.mp4", cache_dir)
    if vid_path is None:
        return None

    rgb_frames = _decode_video(vid_path)
    _cleanup(vid_path)

    if len(rgb_frames) < 2:
        return None

    return {
        "episode_id": ep_name,
        "rgb_frames": rgb_frames,
        "actions": actions,
        "task": task,
        "robot": "franka_panda",
        "depth_type": "pseudo",
        "intrinsics": DROID_INTRINSICS,
    }


def _download(filename, cache_dir):
    try:
        return hf_hub_download(
            repo_id=DROID_REPO,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
    except Exception:
        return None


def _decode_video(video_path: str, max_frames: int = 300) -> list[np.ndarray]:
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
