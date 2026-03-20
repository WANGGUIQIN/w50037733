"""Batch depth estimation using Depth Anything V2 vitl."""
import numpy as np
import torch
from PIL import Image


class DepthGenerator:
    """Generate pseudo-depth maps using Depth Anything V2 Large."""

    def __init__(self, device: str = "cuda:1", image_size: int = 256):
        from transformers import pipeline

        self.pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=device,
        )
        self.image_size = image_size

    def estimate(self, rgb: Image.Image) -> np.ndarray:
        """Estimate depth from a single RGB PIL image.

        Returns:
            depth: [H, W] float32 array. Normalized to [0.01, 5.0] for
            compatibility with backproject_depth.
        """
        result = self.pipe(rgb)
        depth_pil = result["depth"]
        depth_pil = depth_pil.resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        depth = np.array(depth_pil, dtype=np.float32)
        # Normalize to approximate metric range [0.01, 5.0]
        dmin, dmax = depth.min(), depth.max()
        if dmax > dmin:
            depth = 0.01 + 4.99 * (depth - dmin) / (dmax - dmin)
        else:
            depth = np.full_like(depth, 1.0)
        return depth

    def estimate_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Estimate depth for a batch of images."""
        return [self.estimate(img) for img in images]
