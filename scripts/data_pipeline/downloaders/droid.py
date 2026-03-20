"""Stream episodes from cadene/droid (lerobot format: parquet + mp4)."""
import os

import numpy as np
from huggingface_hub import hf_hub_download

DROID_REPO = "cadene/droid"
NUM_CHUNKS = 93
EPISODES_PER_CHUNK = 1000
CAMERA = "observation.images.exterior_image_1_left"
DROID_INTRINSICS = [[200.0, 0, 160.0], [0, 200.0, 120.0], [0, 0, 1]]


def stream_droid_dataset(
    max_chunks: int = 93,
    cache_dir: str = "/tmp/droid_cache",
):
    """Yield episodes from DROID, one chunk at a time."""
    os.makedirs(cache_dir, exist_ok=True)

    for chunk_idx in range(max_chunks):
        chunk_name = f"chunk-{chunk_idx:03d}"
        start_ep = chunk_idx * EPISODES_PER_CHUNK
        ep_count = 0

        print(f"    DROID {chunk_name} ({chunk_idx+1}/{max_chunks})")

        for ep_offset in range(EPISODES_PER_CHUNK):
            ep_idx = start_ep + ep_offset
            ep_name = f"episode_{ep_idx:06d}"

            try:
                ep = _process_episode(chunk_name, ep_name, cache_dir)
            except Exception:
                continue

            if ep is not None:
                yield ep
                ep_count += 1

        print(f"      {ep_count} episodes from {chunk_name}")


def _process_episode(chunk_name, ep_name, cache_dir):
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

    # Extract actions
    action_cols = sorted([c for c in df.columns if c.startswith("action")])
    actions = df[action_cols].values.astype(np.float32) if action_cols else None

    # Extract language
    task = "robotic manipulation"
    if "language_instruction" in df.columns:
        instructions = df["language_instruction"].dropna().unique()
        if len(instructions) > 0:
            task = str(instructions[0])

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
