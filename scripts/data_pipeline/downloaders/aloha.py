"""Stream episodes from lerobot/aloha_static_* bimanual datasets."""
import os
import shutil

import numpy as np
from PIL import Image

ALOHA_TASKS = [
    "battery", "candy", "coffee", "coffee_new", "cups_open",
    "fork_pick_up", "screw_driver", "tape", "thread_velcro",
    "towel", "vinh_cup", "vinh_cup_left", "ziploc_slide",
    "pingpong_test", "pro_pencil",
]


def stream_aloha_datasets(cache_dir: str = "/tmp/aloha_cache"):
    """Yield episodes from all ALOHA bimanual tasks."""
    from datasets import load_dataset

    global_ep = 0

    for task_name in ALOHA_TASKS:
        repo = f"lerobot/aloha_static_{task_name}"
        print(f"    ALOHA: {task_name}")

        try:
            ds = load_dataset(repo, split="train", cache_dir=cache_dir)
        except Exception as e:
            print(f"      failed: {e}")
            continue

        # Group by episode_index
        episodes = {}
        for row in ds:
            ep_idx = row.get("episode_index", 0)
            if ep_idx not in episodes:
                episodes[ep_idx] = {"frames": [], "actions": []}

            # Extract image (try multiple possible keys)
            img = None
            for key in [
                "observation.images.top",
                "observation.images.cam_high",
                "observation.image",
                "image",
            ]:
                if key in row and row[key] is not None:
                    img = row[key]
                    break

            if img is not None:
                if isinstance(img, Image.Image):
                    episodes[ep_idx]["frames"].append(np.array(img.convert("RGB")))
                elif isinstance(img, np.ndarray):
                    episodes[ep_idx]["frames"].append(img)

            # Extract action
            if "action" in row and row["action"] is not None:
                act = row["action"]
                if isinstance(act, (list, np.ndarray)):
                    episodes[ep_idx]["actions"].append(
                        np.array(act, dtype=np.float32)
                    )

        task_ep_count = 0
        for ep_idx in sorted(episodes.keys()):
            ep_data = episodes[ep_idx]
            if len(ep_data["frames"]) < 2:
                continue

            actions = None
            if ep_data["actions"]:
                try:
                    actions = np.stack(ep_data["actions"])
                except ValueError:
                    pass

            yield {
                "episode_id": f"episode_{global_ep:06d}",
                "rgb_frames": ep_data["frames"],
                "actions": actions,
                "task": task_name.replace("_", " "),
                "robot": "aloha_dual_viper",
            }
            global_ep += 1
            task_ep_count += 1

        print(f"      {task_ep_count} episodes")

        # Cleanup cache
        dl_dir = os.path.join(cache_dir, "downloads")
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir, ignore_errors=True)
