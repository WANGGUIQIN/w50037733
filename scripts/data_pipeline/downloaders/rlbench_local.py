"""Load episodes from existing local RLBench data.

Note: Uses pickle to load RLBench .pkl files (variation_descriptions, low_dim_obs).
These are trusted local files from the RLBench simulator, not untrusted external data.
"""
import os
import pickle

import numpy as np
from PIL import Image

RLBENCH_INTRINSICS = [[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1]]


def stream_rlbench_local(
    root_dir: str = "/home/w50037733/robobrain_3dgs/data/rlbench_sample",
    camera: str = "front",
):
    """Yield episodes from local RLBench sample data."""
    global_ep = 0

    for task_name in sorted(os.listdir(root_dir)):
        task_dir = os.path.join(root_dir, task_name)
        eps_dir = os.path.join(task_dir, "all_variations", "episodes")
        if not os.path.isdir(eps_dir):
            continue

        for ep_name in sorted(os.listdir(eps_dir)):
            ep_dir = os.path.join(eps_dir, ep_name)
            rgb_dir = os.path.join(ep_dir, f"{camera}_rgb")
            depth_dir = os.path.join(ep_dir, f"{camera}_depth")
            if not os.path.isdir(rgb_dir):
                continue

            frame_files = sorted(
                [f for f in os.listdir(rgb_dir) if f.endswith(".png")],
                key=lambda x: int(x.split(".")[0]),
            )

            rgb_frames, depth_frames = [], []
            for fname in frame_files:
                rgb = np.array(
                    Image.open(os.path.join(rgb_dir, fname)).convert("RGB")
                )
                rgb_frames.append(rgb)

                dpath = os.path.join(depth_dir, fname)
                if os.path.exists(dpath):
                    depth_frames.append(_decode_rlbench_depth(dpath))

            if len(rgb_frames) < 2:
                continue

            task = task_name.replace("_", " ")
            vdesc = os.path.join(ep_dir, "variation_descriptions.pkl")
            if os.path.exists(vdesc):
                try:
                    with open(vdesc, "rb") as f:
                        descs = pickle.load(f)  # trusted RLBench local file
                    if isinstance(descs, list) and descs:
                        task = str(descs[0])
                except Exception:
                    pass

            actions = _load_actions(os.path.join(ep_dir, "low_dim_obs.pkl"))

            yield {
                "episode_id": f"episode_{global_ep:06d}",
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames if depth_frames else None,
                "actions": actions,
                "task": task,
                "robot": "franka_panda",
                "depth_type": "native",
                "intrinsics": RLBENCH_INTRINSICS,
            }
            global_ep += 1


def _decode_rlbench_depth(path: str) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
    near, far = 0.01, 5.0
    encoded = rgb[:, :, 0] + rgb[:, :, 1] * 256.0 + rgb[:, :, 2] * 65536.0
    return (near + (far - near) * encoded / (256.0**3 - 1.0)).astype(np.float32)


def _load_actions(pkl_path: str) -> np.ndarray | None:
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            obs = pickle.load(f)  # trusted RLBench local file
        if isinstance(obs, list):
            acts = []
            for o in obs:
                if hasattr(o, "gripper_pose"):
                    acts.append(np.array(o.gripper_pose, dtype=np.float32))
                elif hasattr(o, "joint_velocities"):
                    acts.append(np.array(o.joint_velocities, dtype=np.float32))
            return np.stack(acts) if acts else None
        return None
    except Exception:
        return None
