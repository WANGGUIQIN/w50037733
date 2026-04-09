"""Stream episodes from jxu124/OpenX-Embodiment tar files.

OXE format: each tar contains pickle files, one per episode.
Each pickle is a dict with 'steps' (list of timestep dicts).
Each step has observation.image (JPEG bytes), action, language instruction.

Note: pickle.loads is required here because the OXE dataset stores episodes as
pickle files. These are downloaded from the trusted jxu124/OpenX-Embodiment
HuggingFace repository (Google DeepMind's official dataset release).
"""
import io
import os
import pickle  # required: OXE stores episodes as pickle files from trusted HF repo
import tarfile

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

OXE_DATASETS = {
    "bridge": {
        "robot": "widowx", "num_tars": 49,
        "tar_prefix": "bridge/bridge",
        "default_task": "tabletop manipulation",
        "image_key": "image",
    },
    "fractal20220817_data": {
        "robot": "everyday_robots", "num_tars": 78,
        "tar_prefix": "fractal20220817_data/fractal20220817_data",
        "default_task": "office manipulation",
        "image_key": "image",
    },
    "taco_play": {
        "robot": "kuka", "num_tars": 11,
        "tar_prefix": "taco_play/taco_play",
        "default_task": "free play manipulation",
        "image_key": "rgb_static",
        "depth_key": "depth_static",
    },
    "jaco_play": {
        "robot": "kinova_jaco", "num_tars": 2,
        "tar_prefix": "jaco_play/jaco_play",
        "default_task": "tabletop play",
        "image_key": "image",
    },
    "berkeley_cable_routing": {
        "robot": "franka_panda", "num_tars": 1,
        "tar_prefix": "berkeley_cable_routing/berkeley_cable_routing",
        "default_task": "cable routing",
        "image_key": "image",
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "robot": "franka_panda", "num_tars": 79,
        "tar_prefix": "furniture_bench_dataset_converted_externally_to_rlds/"
                      "furniture_bench_dataset_converted_externally_to_rlds",
        "default_task": "furniture assembly",
        "image_key": "image",
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "robot": "franka_panda", "num_tars": 2,
        "tar_prefix": "nyu_franka_play_dataset_converted_externally_to_rlds/"
                      "nyu_franka_play_dataset_converted_externally_to_rlds",
        "default_task": "free play manipulation",
        "image_key": "image",
        "depth_key": "depth",
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "robot": "dual_xarm", "num_tars": 1,
        "tar_prefix": "utokyo_xarm_bimanual_converted_externally_to_rlds/"
                      "utokyo_xarm_bimanual_converted_externally_to_rlds",
        "default_task": "bimanual manipulation",
        "image_key": "image",
    },
}


def stream_oxe_dataset(dataset_name: str, cache_dir: str = "/tmp/oxe_cache"):
    """Yield episodes from an OXE tar dataset, one tar at a time."""
    meta = OXE_DATASETS[dataset_name]
    os.makedirs(cache_dir, exist_ok=True)
    global_ep = 0

    for tar_idx in range(meta["num_tars"]):
        tar_name = f"{meta['tar_prefix']}_{tar_idx:05d}.tar"
        print(f"    tar {tar_idx+1}/{meta['num_tars']}: {tar_name}")

        try:
            tar_path = hf_hub_download(
                repo_id="jxu124/OpenX-Embodiment",
                filename=tar_name,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"      download failed: {e}")
            continue

        tar_ep_count = 0
        try:
            for ep in _parse_tar_pickles(tar_path, meta, global_ep):
                yield ep
                global_ep += 1
                tar_ep_count += 1
        except Exception as e:
            print(f"      parse failed: {e}")

        print(f"      {tar_ep_count} episodes")
        _cleanup_hf_cache(tar_path)


def _parse_tar_pickles(tar_path: str, meta: dict, ep_offset: int):
    """Parse tar of pickle files. Each pickle is one episode."""
    image_key = meta.get("image_key", "image")
    depth_key = meta.get("depth_key")

    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".pickle"):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue

            try:
                # OXE episode pickle from trusted HuggingFace dataset
                data = pickle.loads(f.read())  # noqa: S301
            except Exception:
                continue

            steps = data.get("steps", [])
            if len(steps) < 2:
                continue

            rgb_frames = []
            depth_frames = []
            actions = []
            task = meta["default_task"]

            for step in steps:
                obs = step.get("observation", {})

                # Extract image (JPEG bytes -> numpy)
                img_bytes = obs.get(image_key) or obs.get("image")
                if img_bytes and isinstance(img_bytes, bytes):
                    try:
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        rgb_frames.append(np.array(img))
                    except Exception:
                        pass

                # Extract native depth (16-bit PNG, uint16 millimeters -> float32 meters)
                if depth_key:
                    d_bytes = obs.get(depth_key)
                    if d_bytes and isinstance(d_bytes, bytes):
                        try:
                            d_img = Image.open(io.BytesIO(d_bytes))
                            d_arr = np.array(d_img, dtype=np.float32) / 1000.0
                            d_arr = np.clip(d_arr, 0.01, 5.0)
                            depth_frames.append(d_arr)
                        except Exception:
                            pass

                # Extract action
                act_data = step.get("action", {})
                if isinstance(act_data, dict):
                    act_parts = []
                    for k in sorted(act_data.keys()):
                        v = act_data[k]
                        if hasattr(v, "shape"):
                            act_parts.append(v.flatten().astype(np.float32))
                        elif isinstance(v, (int, float)):
                            act_parts.append(np.array([v], dtype=np.float32))
                    if act_parts:
                        actions.append(np.concatenate(act_parts))
                elif hasattr(act_data, "shape"):
                    actions.append(np.array(act_data, dtype=np.float32).flatten())

                # Extract language
                lang = obs.get("natural_language_instruction")
                if lang:
                    if isinstance(lang, bytes):
                        lang = lang.decode("utf-8", errors="ignore")
                    if isinstance(lang, str) and lang.strip():
                        task = lang.strip()

            if len(rgb_frames) < 2:
                continue

            has_depth = (
                depth_key
                and len(depth_frames) == len(rgb_frames)
            )

            ep = {
                "episode_id": f"episode_{ep_offset:06d}",
                "rgb_frames": rgb_frames,
                "actions": np.stack(actions) if actions else None,
                "task": task,
                "robot": meta["robot"],
            }
            if has_depth:
                ep["depth_frames"] = depth_frames
                ep["depth_type"] = "native"

            yield ep
            ep_offset += 1


def _cleanup_hf_cache(file_path: str):
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
