"""DROID dataset loader for RoboBrain-3DGS training.

DROID is an RGB-only real-world manipulation dataset. Since it lacks native depth,
we use Depth Anything V2 to generate pseudo-depth maps during preprocessing.

For now, this loader handles:
1. Extracting frames from DROID MP4 videos via ffmpeg
2. Loading pre-extracted frames as RGB + pseudo-depth pairs

The pipeline: DROID RGB -> Depth Anything V2 -> pseudo-depth -> 3DGS training
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Approximate camera intrinsics for DROID (320x180 exterior camera)
# DROID uses varied cameras; these are reasonable defaults.
DROID_INTRINSICS_320x180 = {
    "fx": 200.0,
    "fy": 200.0,
    "cx": 160.0,
    "cy": 90.0,
}

# Task templates (DROID tasks are described in metadata)
DEFAULT_PROMPT = "Complete the robotic manipulation task shown in the image."


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    max_frames: int = 10,
    fps: float = 1.0,
) -> list[str]:
    """Extract frames from an MP4 video using ffmpeg.

    Args:
        video_path: path to the MP4 file
        output_dir: directory to save extracted PNGs
        max_frames: maximum number of frames to extract
        fps: frames per second to extract (1.0 = one per second)

    Returns:
        list of extracted frame file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = os.path.join(output_dir, "frame_%04d.png")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(max_frames),
        "-update", "0",
        pattern,
        "-y",
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(Path(output_dir).glob("frame_*.png"))
    return [str(f) for f in frames]


def generate_pseudo_depth(rgb_path: str, output_path: str) -> np.ndarray:
    """Generate pseudo-depth using a simple disparity estimation.

    In production, replace this with Depth Anything V2:
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vitl', ...)
        depth = model.infer_image(rgb_image)

    For validation, we use a grayscale-based heuristic (darker = farther).
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)
    # Simple heuristic: use luminance as rough depth proxy
    # Brighter pixels tend to be closer in tabletop scenes
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    # Normalize to 0.3-2.0m range (typical tabletop)
    depth = 2.0 - (luminance / 255.0) * 1.7 + 0.3
    # Add slight noise
    depth += np.random.randn(*depth.shape).astype(np.float32) * 0.01
    depth = np.clip(depth, 0.1, 5.0).astype(np.float32)

    # Save for caching
    np.save(output_path, depth)
    return depth


class DROIDDataset(Dataset):
    """Load DROID frames as RGBD samples for 3DGS training.

    Expects pre-extracted frames in:
        root_dir/
          rgb/frame_0001.png, frame_0002.png, ...
          depth/frame_0001.npy, frame_0002.npy, ...  (pseudo-depth from Depth Anything V2)
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        max_frames: int = -1,
        auto_generate_depth: bool = True,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.image_size = image_size

        rgb_dir = self.root / "rgb" if (self.root / "rgb").exists() else self.root
        depth_dir = self.root / "depth"
        depth_dir.mkdir(exist_ok=True)

        # Discover RGB frames
        self.samples = []
        rgb_files = sorted(rgb_dir.glob("*.png"))
        if not rgb_files:
            rgb_files = sorted(rgb_dir.glob("*.jpg"))

        for rgb_file in rgb_files:
            depth_file = depth_dir / (rgb_file.stem + ".npy")

            # Auto-generate pseudo-depth if missing
            if auto_generate_depth and not depth_file.exists():
                generate_pseudo_depth(str(rgb_file), str(depth_file))

            self.samples.append({
                "rgb_path": str(rgb_file),
                "depth_path": str(depth_file),
            })

        if max_frames > 0:
            self.samples = self.samples[:max_frames]

        # Build intrinsics
        intr = DROID_INTRINSICS_320x180
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
        rgb = np.array(rgb_img).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # [3, H, W]

        # Load pseudo-depth
        depth_path = info["depth_path"]
        if os.path.exists(depth_path):
            depth_np = np.load(depth_path)
        else:
            # Fallback: generate on the fly
            depth_np = generate_pseudo_depth(info["rgb_path"], depth_path)

        # Resize depth
        depth_pil = Image.fromarray(depth_np, mode="F")
        depth_pil = depth_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        depth = torch.from_numpy(np.array(depth_pil)).unsqueeze(0)  # [1, H, W]

        # Scale intrinsics
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        intrinsics = self.base_intrinsics.copy()
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] *= scale_y
        intrinsics = torch.from_numpy(intrinsics)

        # DROID has no ground-truth affordance annotations.
        # Use a neutral placeholder so the LLM at least learns the output format.
        target = (
            "affordance: [0.50, 0.50]. "
            "constraint: gripper_width=0.08, approach=[0.00, 0.00, -1.00]."
        )

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "prompt": DEFAULT_PROMPT,
            "target": target,
        }


def validate_droid_loader():
    """Quick validation of the DROID data loader."""
    data_root = "/home/w50037733/robobrain_3dgs/data/droid_sample"
    ds = DROIDDataset(data_root, image_size=256, max_frames=5)
    print(f"DROID dataset: {len(ds)} samples")

    if len(ds) > 0:
        sample = ds[0]
        print(f"  RGB: {sample['rgb'].shape}, range=[{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
        print(f"  Depth: {sample['depth'].shape}, range=[{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")
        print(f"  Intrinsics: {sample['intrinsics'].shape}")
        print(f"  Prompt: {sample['prompt']}")
    return True


if __name__ == "__main__":
    validate_droid_loader()
