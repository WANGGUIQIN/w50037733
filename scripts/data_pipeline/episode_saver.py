"""Save processed episodes in unified format."""
import json
import os

import numpy as np
from PIL import Image


def save_episode(
    output_dir: str,
    dataset_name: str,
    episode_id: str,
    rgb_frames: list[np.ndarray],
    depth_frames: list[np.ndarray],
    task: str,
    intrinsics: list[list[float]],
    keyframe_indices: list[int],
    depth_type: str = "pseudo",
    robot: str = "unknown",
    num_frames_total: int = 0,
    image_size: int = 256,
):
    """Save one episode to disk in unified format."""
    ep_dir = os.path.join(output_dir, dataset_name, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    for i, (rgb, depth) in enumerate(zip(rgb_frames, depth_frames)):
        # RGB
        img = Image.fromarray(rgb)
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        img.save(os.path.join(ep_dir, f"rgb_{i}.png"))

        # Depth
        if depth.shape != (image_size, image_size):
            d_pil = Image.fromarray(depth, mode="F")
            d_pil = d_pil.resize((image_size, image_size), Image.NEAREST)
            depth = np.array(d_pil)
        np.save(os.path.join(ep_dir, f"depth_{i}.npy"), depth.astype(np.float32))

    meta = {
        "task": task,
        "intrinsics": intrinsics,
        "keyframe_indices": keyframe_indices,
        "source": dataset_name,
        "episode_id": episode_id,
        "depth_type": depth_type,
        "robot": robot,
        "num_frames_total": num_frames_total,
    }
    with open(os.path.join(ep_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
