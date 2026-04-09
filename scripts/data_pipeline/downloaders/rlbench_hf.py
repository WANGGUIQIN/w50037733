"""Download and stream RLBench episodes from HuggingFace.

Dataset: hqfang/rlbench-18-tasks
Contains 18 manipulation tasks with native depth maps.

Structure:
    data/{train,val,test}/{task_name}.zip
    Each zip contains:
        {task}/all_variations/episodes/episode{N}/
            front_rgb/
            front_depth/
            low_dim_obs.pkl
            variation_descriptions.pkl

Security Note: This module uses pickle.loads() to load RLBench data files
(low_dim_obs.pkl, variation_descriptions.pkl). This is required because the
RLBench dataset format uses pickle to serialize Python objects. The data
comes from the trusted HuggingFace repository hqfang/rlbench-18-tasks.
"""
import io
import os
import pickle  # nosec B403 - required for RLBench data format
import zipfile

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

RLBENCH_TASKS = [
    "close_jar",
    "insert_onto_square_peg",
    "light_bulb_in",
    "meat_off_grill",
    "open_drawer",
    "place_cups",
    "place_shape_in_shape_sorter",
    "place_wine_at_rack_location",
    "push_buttons",
    "put_groceries_in_cupboard",
    "put_item_in_drawer",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "stack_blocks",
    "stack_cups",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

RLBENCH_INTRINSICS = [[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1]]

# Use train split by default (most data)
DEFAULT_SPLITS = ["train"]


def stream_rlbench_from_hf(
    cache_dir: str = "/tmp/rlbench_hf_cache",
    splits: list[str] = None,
    tasks: list[str] = None,
):
    """Yield episodes from RLBench dataset on HuggingFace.

    Args:
        cache_dir: Directory to store downloaded zip files temporarily.
        splits: Which splits to use ["train", "val", "test"]. Default: ["train"]
        tasks: Which tasks to process. Default: all 18 tasks.

    Yields:
        Episode dicts compatible with the processing pipeline.
    """
    if splits is None:
        splits = DEFAULT_SPLITS
    if tasks is None:
        tasks = RLBENCH_TASKS

    os.makedirs(cache_dir, exist_ok=True)
    global_ep = 0

    for task_name in tasks:
        for split in splits:
            zip_filename = f"data/{split}/{task_name}.zip"
            print(f"    Downloading {task_name}/{split}...")

            try:
                zip_path = hf_hub_download(
                    repo_id="hqfang/rlbench-18-tasks",
                    filename=zip_filename,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
            except Exception as e:
                print(f"      Failed to download {zip_filename}: {e}")
                continue

            task_ep_count = 0
            try:
                for ep_data in _parse_rlbench_zip(zip_path, task_name, global_ep):
                    yield ep_data
                    global_ep += 1
                    task_ep_count += 1
            except Exception as e:
                print(f"      Failed to parse {zip_filename}: {e}")

            print(f"      {task_ep_count} episodes from {task_name}/{split}")

            # Clean up zip file to save space
            _cleanup_file(zip_path)


def _parse_rlbench_zip(zip_path: str, task_name: str, ep_offset: int):
    """Parse RLBench zip file and yield episodes."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find all episode directories
        all_files = zf.namelist()

        # Build episode directory list
        episode_dirs = set()
        prefix = f"{task_name}/all_variations/episodes/"
        for f in all_files:
            if f.startswith(prefix) and f != prefix:
                # Extract episode directory name
                rest = f[len(prefix):]
                if "/" in rest:
                    ep_dir = rest.split("/")[0]
                    episode_dirs.add(ep_dir)

        for ep_dir in sorted(episode_dirs):
            ep_path = f"{prefix}{ep_dir}/"
            rgb_dir = f"{ep_path}front_rgb/"
            depth_dir = f"{ep_path}front_depth/"

            # Get RGB frame files
            rgb_files = [
                f for f in all_files
                if f.startswith(rgb_dir) and f.endswith(".png")
            ]
            if len(rgb_files) < 2:
                continue

            # Sort by frame number
            rgb_files = sorted(
                rgb_files,
                key=lambda x: int(x.split("/")[-1].split(".")[0])
            )

            rgb_frames = []
            depth_frames = []

            for rgb_file in rgb_files:
                # Load RGB
                try:
                    with zf.open(rgb_file) as f:
                        img = Image.open(io.BytesIO(f.read())).convert("RGB")
                        rgb_frames.append(np.array(img))
                except Exception:
                    continue

                # Load corresponding depth
                frame_name = rgb_file.split("/")[-1]
                depth_file = f"{depth_dir}{frame_name}"
                if depth_file in all_files:
                    try:
                        with zf.open(depth_file) as f:
                            depth_frames.append(_decode_rlbench_depth(f.read()))
                    except Exception:
                        pass

            if len(rgb_frames) < 2:
                continue

            # Load actions from low_dim_obs.pkl
            low_dim_path = f"{ep_path}low_dim_obs.pkl"
            actions = _load_actions_from_zip(zf, low_dim_path)

            # Load task description
            task_desc = task_name.replace("_", " ")
            var_desc_path = f"{ep_path}variation_descriptions.pkl"
            try:
                with zf.open(var_desc_path) as f:
                    # Trusted RLBench dataset from HuggingFace
                    descs = pickle.loads(f.read())  # nosec B301
                if isinstance(descs, list) and descs:
                    task_desc = str(descs[0])
            except Exception:
                pass

            yield {
                "episode_id": f"episode_{ep_offset:06d}",
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames if depth_frames else None,
                "actions": actions,
                "task": task_desc,
                "robot": "franka_panda",
                "depth_type": "native",
                "intrinsics": RLBENCH_INTRINSICS,
            }
            ep_offset += 1


def _decode_rlbench_depth(data: bytes) -> np.ndarray:
    """Decode RLBench depth from RGB-encoded PNG.

    RLBench encodes depth in 24-bit RGB:
    depth = (R + G*256 + B*65536) / (256^3 - 1) * (far - near) + near
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(img, dtype=np.float64)

    near, far = 0.01, 5.0
    encoded = rgb[:, :, 0] + rgb[:, :, 1] * 256.0 + rgb[:, :, 2] * 65536.0
    depth = near + (far - near) * encoded / (256.0**3 - 1.0)
    return depth.astype(np.float32)


def _load_actions_from_zip(zf: zipfile.ZipFile, pkl_path: str) -> np.ndarray | None:
    """Load actions from low_dim_obs.pkl inside a zip."""
    try:
        with zf.open(pkl_path) as f:
            # Trusted RLBench dataset from HuggingFace
            obs = pickle.loads(f.read())  # nosec B301
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


def _cleanup_file(path: str):
    """Remove downloaded file and any symlinks."""
    try:
        real = os.path.realpath(path)
        if os.path.exists(real) and real != path:
            os.remove(real)
        if os.path.islink(path):
            os.remove(path)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
