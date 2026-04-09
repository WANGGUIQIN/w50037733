"""Stream episodes from RH20T dataset with native depth.

RH20T distributes RGB and depth as separate tar.gz archives per config.
Each config has: color (MP4 videos), depth (PNG frames), lowdim, calib.

Download priority:
1. Google Drive (official) for both RGB + depth tars
2. HF lerobot fallback (hainh22/rh20t) — 30 episodes, no depth
"""
import os
import shutil
import tarfile

import numpy as np
from PIL import Image

# Google Drive file IDs (from rh20t.github.io)
# Each config needs BOTH color and depth downloads
RH20T_CONFIGS = {
    "cfg3": {
        "color_id": "1aekLEcX1ruS9f2z6900ys5t_U_OJnEzQ".replace("depth", ""),
        "depth_id": "1aekLEcX1ruS9f2z6900ys5t_U_OJnEzQ",
        "color_size": "26GB",
        "depth_size": "26GB",
    },
    "cfg2": {
        "color_id": "1A0qvT4JV1c9AvGVGxdCwHFHF6nR7CkAH",
        "depth_id": "1Z-g-A7Smlxi4AI6h9nJ04bcdHmGiSWQK",
        "color_size": "80GB",
        "depth_size": "108GB",
    },
    "cfg1": {
        "color_id": "1MRFHfBhsLQ1_dRqPHsEXfV0JMF84v24A",
        "depth_id": "1RcglSD0_S10xsIwQVmJhC0flAQ9WZtIM",
        "color_size": "178GB",
        "depth_size": "228GB",
    },
}

# Small HuggingFace version (fallback)
RH20T_HF_REPO = "hainh22/rh20t"


def stream_rh20t_dataset(
    configs: list[str] | None = None,
    cache_dir: str = "/tmp/rh20t_cache",
    use_hf_fallback: bool = True,
):
    """Yield episodes from RH20T with native depth.

    Downloads color + depth tars from Google Drive, extracts both,
    pairs RGB video frames with depth PNG frames per camera.
    """
    if configs is None:
        configs = ["cfg2"]

    os.makedirs(cache_dir, exist_ok=True)

    success = False
    for cfg in configs:
        cfg_info = RH20T_CONFIGS.get(cfg)
        if not cfg_info:
            print(f"    RH20T {cfg}: unknown config, skipping")
            continue

        # Try color-only (pseudo depth) from cached/downloaded tar
        try:
            yield from _stream_cfg_color_only(cfg, cfg_info, cache_dir)
            success = True
            continue
        except Exception as e:
            print(f"      Color-only failed: {e}")

        # Try full RGB+depth from GDrive
        try:
            yield from _stream_cfg_with_depth(cfg, cfg_info, cache_dir)
            success = True
        except Exception as e:
            print(f"      GDrive failed: {e}")

    # Fallback: small HF lerobot version (no depth)
    if not success and use_hf_fallback:
        print(f"      Falling back to HF lerobot version (no depth)")
        yield from _stream_from_hf(cache_dir)


def _stream_cfg_color_only(cfg: str, cfg_info: dict, cache_dir: str):
    """Stream episodes from color tar only (no depth, will use pseudo depth)."""
    import av
    from huggingface_hub import hf_hub_download

    color_tar = os.path.join(cache_dir, f"RH20T_{cfg}.tar.gz")

    # Try to get color tar: from cache, HF mirror, or GDrive
    if not os.path.exists(color_tar):
        # Try HF mirror first
        hf_parts = {
            "cfg2": ["RH20T_cfg2.part.aa", "RH20T_cfg2.part.ab"],
        }
        if cfg in hf_parts:
            print(f"      Downloading {cfg} color from HF mirror ...")
            part_paths = []
            for part_name in hf_parts[cfg]:
                p = hf_hub_download(
                    repo_id="Malak-Mansour/RH20T",
                    filename=part_name,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                part_paths.append(p)
            print(f"      Concatenating parts ...")
            with open(color_tar, "wb") as out:
                for p in part_paths:
                    with open(p, "rb") as inp:
                        shutil.copyfileobj(inp, out, length=64 * 1024 * 1024)
        else:
            # GDrive fallback
            _download_gdrive(cfg_info["color_id"], color_tar, f"{cfg} color")

    if not os.path.exists(color_tar):
        raise FileNotFoundError(f"Color tar not found: {color_tar}")

    # Extract
    extract_dir = os.path.join(cache_dir, f"RH20T_{cfg}_color")
    if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
        print(f"      Extracting color ...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(color_tar, "r:gz") as tf:
            tf.extractall(extract_dir)

    # Find cfg subdir
    cfg_subdir = None
    for d in os.listdir(extract_dir):
        if d.startswith(f"RH20T_{cfg}"):
            cfg_subdir = d
            break
    if cfg_subdir is None:
        raise FileNotFoundError(f"No RH20T_{cfg}* in {extract_dir}")

    color_base = os.path.join(extract_dir, cfg_subdir)
    sessions = sorted([
        d for d in os.listdir(color_base)
        if os.path.isdir(os.path.join(color_base, d)) and d.startswith("task_")
    ])

    print(f"      Found {len(sessions)} sessions")
    global_ep = 0

    for session in sessions:
        session_dir = os.path.join(color_base, session)

        # Find cameras
        cam_dirs = sorted([
            d for d in os.listdir(session_dir)
            if os.path.isdir(os.path.join(session_dir, d)) and d.startswith("cam_")
        ])

        # Use first camera with a color.mp4
        for cam in cam_dirs:
            color_mp4 = os.path.join(session_dir, cam, "color.mp4")
            if not os.path.exists(color_mp4):
                continue

            rgb_frames = []
            try:
                with av.open(color_mp4) as container:
                    for frame in container.decode(video=0):
                        rgb_frames.append(frame.to_ndarray(format="rgb24"))
                        if len(rgb_frames) >= 500:
                            break
            except Exception:
                continue

            if len(rgb_frames) < 2:
                continue

            task_part = session.split("_user_")[0]
            task = task_part.replace("task_", "").replace("_", " ") if task_part else "manipulation"

            yield {
                "episode_id": f"episode_{global_ep:06d}",
                "rgb_frames": rgb_frames,
                "actions": None,
                "task": task,
                "robot": f"rh20t_{cfg}",
                "depth_type": "pseudo",
                "intrinsics": [[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]],
            }
            global_ep += 1
            break  # One camera per session

    print(f"      {cfg} color-only: {global_ep} episodes total")

    # Cleanup extracted dir
    shutil.rmtree(extract_dir, ignore_errors=True)


def _download_gdrive(gdrive_id: str, output_path: str, label: str):
    """Download from Google Drive using gdown."""
    import gdown

    if os.path.exists(output_path):
        print(f"      {label}: already cached")
        return

    print(f"      Downloading {label} from Google Drive ...")
    gdown.download(id=gdrive_id, output=output_path, quiet=False)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed: {output_path}")


def _stream_cfg_with_depth(cfg: str, cfg_info: dict, cache_dir: str):
    """Download RGB + depth tars, extract, pair frames, yield episodes."""
    color_tar = os.path.join(cache_dir, f"RH20T_{cfg}.tar.gz")
    depth_tar = os.path.join(cache_dir, f"RH20T_{cfg}_depth.tar.gz")
    color_dir = os.path.join(cache_dir, f"RH20T_{cfg}_color")
    depth_dir = os.path.join(cache_dir, f"RH20T_{cfg}_depth")

    # Download both tars
    _download_gdrive(cfg_info["color_id"], color_tar, f"{cfg} color")
    _download_gdrive(cfg_info["depth_id"], depth_tar, f"{cfg} depth")

    # Extract color
    print(f"      Extracting color ...")
    os.makedirs(color_dir, exist_ok=True)
    with tarfile.open(color_tar, "r:gz") as tf:
        tf.extractall(color_dir)

    # Extract depth
    print(f"      Extracting depth ...")
    os.makedirs(depth_dir, exist_ok=True)
    with tarfile.open(depth_tar, "r:gz") as tf:
        tf.extractall(depth_dir)

    # Walk and pair episodes
    ep_count = 0
    for ep in _walk_paired_episodes(color_dir, depth_dir, cfg):
        yield ep
        ep_count += 1
        if ep_count % 200 == 0:
            print(f"      {cfg}: {ep_count} episodes processed")

    print(f"      {cfg}: {ep_count} episodes total")

    # Cleanup
    shutil.rmtree(color_dir, ignore_errors=True)
    shutil.rmtree(depth_dir, ignore_errors=True)
    try:
        os.remove(color_tar)
        os.remove(depth_tar)
    except Exception:
        pass


def _walk_paired_episodes(color_root: str, depth_root: str, cfg: str):
    """Walk color directory, find matching depth, yield episodes.

    RH20T cfg structure (color tar):
        RH20T_cfgX/task_XXXX_user_XXXX_scene_XXXX_cfg_XXXX/
            cam_SERIAL/color.mp4
            cam_SERIAL/timestamps.npy
            metadata.json

    RH20T cfg structure (depth tar):
        RH20T_cfgX_depth/task_XXXX_user_XXXX_scene_XXXX_cfg_XXXX/
            cam_SERIAL/depth/NNNNNN.png
    """
    import av

    global_ep = 0

    # Find all session directories in color root
    cfg_subdir = None
    for d in os.listdir(color_root):
        if d.startswith(f"RH20T_{cfg}"):
            cfg_subdir = d
            break

    if cfg_subdir is None:
        print(f"      Could not find RH20T_{cfg}* in {color_root}")
        return

    color_base = os.path.join(color_root, cfg_subdir)
    # Depth dir might have _depth suffix
    depth_base = None
    for d in os.listdir(depth_root):
        if d.startswith(f"RH20T_{cfg}"):
            depth_base = os.path.join(depth_root, d)
            break

    if depth_base is None:
        print(f"      Could not find depth directory for {cfg}")
        return

    # Iterate sessions
    sessions = sorted([
        d for d in os.listdir(color_base)
        if os.path.isdir(os.path.join(color_base, d)) and d.startswith("task_")
    ])

    for session in sessions:
        session_color = os.path.join(color_base, session)
        session_depth = os.path.join(depth_base, session)

        if not os.path.isdir(session_depth):
            continue

        # Find cameras with both color.mp4 and depth/
        cam_dirs = [
            d for d in os.listdir(session_color)
            if os.path.isdir(os.path.join(session_color, d)) and d.startswith("cam_")
        ]

        # Pick first camera that has both color video and depth frames
        for cam in sorted(cam_dirs):
            color_mp4 = os.path.join(session_color, cam, "color.mp4")
            cam_depth_dir = os.path.join(session_depth, cam, "depth")

            if not os.path.exists(color_mp4) or not os.path.isdir(cam_depth_dir):
                continue

            # Decode video frames
            rgb_frames = []
            try:
                with av.open(color_mp4) as container:
                    for frame in container.decode(video=0):
                        rgb_frames.append(frame.to_ndarray(format="rgb24"))
                        if len(rgb_frames) >= 500:
                            break
            except Exception:
                continue

            if len(rgb_frames) < 2:
                continue

            # Load depth frames (sorted PNG files)
            depth_files = sorted([
                f for f in os.listdir(cam_depth_dir)
                if f.endswith(".png")
            ])

            if len(depth_files) < 2:
                continue

            depth_frames = []
            for df in depth_files[:len(rgb_frames)]:
                try:
                    d = np.array(Image.open(os.path.join(cam_depth_dir, df)), dtype=np.float32)
                    if d.ndim == 3:
                        d = d[:, :, 0]
                    # RH20T depth is in mm, convert to meters
                    d = d / 1000.0
                    depth_frames.append(d)
                except Exception:
                    break

            # Only use if we can pair RGB and depth
            num_paired = min(len(rgb_frames), len(depth_frames))
            if num_paired < 2:
                continue

            rgb_frames = rgb_frames[:num_paired]
            depth_frames = depth_frames[:num_paired]

            # Extract task name from session path
            task_part = session.split("_user_")[0]
            task = task_part.replace("task_", "").replace("_", " ") if task_part else "manipulation"

            yield {
                "episode_id": f"episode_{global_ep:06d}",
                "rgb_frames": rgb_frames,
                "depth_frames": depth_frames,
                "actions": None,
                "task": task,
                "robot": f"rh20t_{cfg}",
                "depth_type": "native",
                "intrinsics": [[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]],
            }
            global_ep += 1
            break  # One camera per session


def _stream_from_hf(cache_dir: str):
    """Fallback: load from small HuggingFace version (lerobot format with videos)."""
    from huggingface_hub import hf_hub_download
    import cv2
    import pyarrow.parquet as pq

    print("    RH20T HF fallback (30 episodes from videos)")

    camera_priority = [
        "observation.images.cam_front_view",
        "observation.images.cam_side_view",
        "observation.images.cam_eye_in_hand",
    ]

    for ep_idx in range(30):
        try:
            pq_path = hf_hub_download(
                RH20T_HF_REPO,
                f"data/chunk-000/episode_{ep_idx:06d}.parquet",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            df = pq.read_table(pq_path).to_pandas()

            actions = None
            if "observation.action" in df.columns:
                actions = np.stack(df["observation.action"].values)

            rgb_frames = None
            for cam in camera_priority:
                video_path = f"videos/chunk-000/{cam}/episode_{ep_idx:06d}.mp4"
                try:
                    mp4_path = hf_hub_download(
                        RH20T_HF_REPO,
                        video_path,
                        repo_type="dataset",
                        cache_dir=cache_dir,
                    )
                    cap = cv2.VideoCapture(mp4_path)
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if frame.shape[0] != 256 or frame.shape[1] != 256:
                            frame = cv2.resize(frame, (256, 256))
                        frames.append(frame)
                    cap.release()

                    if len(frames) >= 2:
                        rgb_frames = frames
                        break
                except Exception:
                    continue

            if rgb_frames is None or len(rgb_frames) < 2:
                continue

            yield {
                "episode_id": f"episode_{ep_idx:06d}",
                "rgb_frames": rgb_frames,
                "depth_frames": None,
                "actions": actions,
                "task": "manipulation",
                "robot": "ur5",
                "depth_type": "pseudo",
                "intrinsics": [[320.0, 0, 320.0], [0, 320.0, 180.0], [0, 0, 1]],
            }

        except Exception as e:
            print(f"      Episode {ep_idx}: {e}")
            continue
