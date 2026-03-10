"""RLBench dataset loader for RoboBrain-3DGS training.

Loads RGBD frames from RLBench episodes (simulation data with native depth).
Each sample: RGB image + depth map + camera intrinsics + task instruction.

RLBench depth format: RGB-encoded PNG where
    depth_meters = near + (far - near) * (R + G*256 + B*256^2) / (256^3 - 1)
with near=0.01, far=5.0 (CoppeliaSim defaults).
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# CoppeliaSim default camera intrinsics for RLBench (128x128 images)
# These are approximate; actual values depend on the camera FOV setting.
# RLBench uses FOV ~60 degrees -> fx = fy = W / (2 * tan(FOV/2))
RLBENCH_INTRINSICS_128 = {
    "fx": 128 / (2 * np.tan(np.radians(30))),  # ~110.85
    "fy": 128 / (2 * np.tan(np.radians(30))),
    "cx": 64.0,
    "cy": 64.0,
}

# Task description templates for RLBench tasks
TASK_PROMPTS = {
    "close_jar": "Close the jar by placing the lid on top of it.",
    "open_drawer": "Open the drawer by pulling the handle.",
    "slide_block": "Slide the block to the target location.",
    "pick_up_cup": "Pick up the cup from the table.",
    "default": "Complete the manipulation task shown in the image.",
}


def decode_rlbench_depth(depth_rgb: np.ndarray, near: float = 0.01, far: float = 5.0) -> np.ndarray:
    """Decode RLBench RGB-encoded depth to meters.

    Args:
        depth_rgb: [H, W, 3] uint8 array from depth PNG
        near: near plane in meters
        far: far plane in meters

    Returns:
        depth_m: [H, W] float32 depth in meters
    """
    r = depth_rgb[:, :, 0].astype(np.float64)
    g = depth_rgb[:, :, 1].astype(np.float64)
    b = depth_rgb[:, :, 2].astype(np.float64)
    normalized = (r + g * 256 + b * 65536) / (256**3 - 1)
    depth_m = near + (far - near) * normalized
    return depth_m.astype(np.float32)


class RLBenchDataset(Dataset):
    """Load RLBench episodes as RGBD samples for 3DGS training.

    Directory structure expected:
        root/
          task_name/
            all_variations/
              episodes/
                episode0/
                  front_rgb/0.png, 1.png, ...
                  front_depth/0.png, 1.png, ...
                  ...
    """

    def __init__(
        self,
        root_dir: str,
        camera: str = "front",
        image_size: int = 256,
        max_frames: int = -1,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.camera = camera
        self.image_size = image_size

        # Discover all frames
        self.samples = []
        for task_dir in sorted(self.root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            episodes_dir = task_dir / "all_variations" / "episodes"
            if not episodes_dir.exists():
                continue
            for ep_dir in sorted(episodes_dir.iterdir()):
                rgb_dir = ep_dir / f"{camera}_rgb"
                depth_dir = ep_dir / f"{camera}_depth"
                if not rgb_dir.exists() or not depth_dir.exists():
                    continue
                for rgb_file in sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem)):
                    depth_file = depth_dir / rgb_file.name
                    if depth_file.exists():
                        self.samples.append({
                            "rgb_path": str(rgb_file),
                            "depth_path": str(depth_file),
                            "task": task_name,
                            "episode": ep_dir.name,
                            "frame": int(rgb_file.stem),
                        })

        if max_frames > 0:
            self.samples = self.samples[:max_frames]

        # Build intrinsics
        intr = RLBENCH_INTRINSICS_128
        self.base_intrinsics = np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1],
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]

        # Load RGB
        rgb_img = Image.open(info["rgb_path"]).convert("RGB")
        orig_w, orig_h = rgb_img.size
        rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb = np.array(rgb_img).astype(np.float32) / 255.0  # [H, W, 3]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]

        # Load depth
        depth_rgb = np.array(Image.open(info["depth_path"]).convert("RGB"))
        depth_m = decode_rlbench_depth(depth_rgb)  # [h, w]
        # Resize depth (nearest to avoid interpolation artifacts)
        depth_pil = Image.fromarray(depth_m, mode="F")
        depth_pil = depth_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        depth = torch.from_numpy(np.array(depth_pil)).unsqueeze(0)  # [1, H, W]

        # Scale intrinsics for resized image
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        intrinsics = self.base_intrinsics.copy()
        intrinsics[0, 0] *= scale_x  # fx
        intrinsics[0, 2] *= scale_x  # cx
        intrinsics[1, 1] *= scale_y  # fy
        intrinsics[1, 2] *= scale_y  # cy
        intrinsics = torch.from_numpy(intrinsics)

        # Task prompt
        prompt = TASK_PROMPTS.get(info["task"], TASK_PROMPTS["default"])

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "prompt": prompt,
            "task": info["task"],
            "episode": info["episode"],
            "frame": info["frame"],
        }


def validate_rlbench_loader():
    """Quick validation of the RLBench data loader."""
    data_root = "/home/w50037733/robobrain_3dgs/data/rlbench_sample"
    ds = RLBenchDataset(data_root, camera="front", image_size=256, max_frames=5)
    print(f"RLBench dataset: {len(ds)} samples")

    sample = ds[0]
    print(f"  RGB: {sample['rgb'].shape}, range=[{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
    print(f"  Depth: {sample['depth'].shape}, range=[{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")
    print(f"  Intrinsics: {sample['intrinsics'].shape}")
    print(f"  Prompt: {sample['prompt']}")
    print(f"  Task: {sample['task']}, Episode: {sample['episode']}, Frame: {sample['frame']}")
    return True


if __name__ == "__main__":
    validate_rlbench_loader()
