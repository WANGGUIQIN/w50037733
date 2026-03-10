"""Generate synthetic RGBD data for pipeline validation.

Creates a simple tabletop manipulation scene with:
    - A table surface
    - A target object (box) on the table
    - Realistic camera intrinsics (RealSense D435 style)
    - Ground truth affordance bbox and constraints
"""

import torch
import numpy as np


def create_synthetic_sample(
    image_size: int = 256,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create one synthetic RGBD sample with ground truth labels.

    Simulates a top-down view of a table with an object to grasp.

    Returns:
        dict with rgb, depth, intrinsics, text_ids, gt_affordance, gt_constraints
    """
    H = W = image_size

    # ---- Camera Intrinsics (RealSense D435 style) ----
    fx, fy = 385.0, 385.0
    cx, cy = H / 2.0, W / 2.0
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]

    # ---- Depth Map ----
    # Table at ~0.7m, object at ~0.55m (object is 15cm tall on table)
    depth = torch.full((1, 1, H, W), 0.7, dtype=torch.float32, device=device)

    # Place a rectangular object in the center
    obj_x1, obj_y1 = int(H * 0.35), int(W * 0.4)
    obj_x2, obj_y2 = int(H * 0.65), int(W * 0.6)
    depth[:, :, obj_x1:obj_x2, obj_y1:obj_y2] = 0.55

    # Add some noise to make it realistic
    depth = depth + torch.randn_like(depth) * 0.005

    # ---- RGB Image ----
    # Brown table, blue object
    rgb = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)
    # Table color (wooden brown)
    rgb[:, 0, :, :] = 0.55  # R
    rgb[:, 1, :, :] = 0.35  # G
    rgb[:, 2, :, :] = 0.20  # B
    # Object color (blue)
    rgb[:, 0, obj_x1:obj_x2, obj_y1:obj_y2] = 0.2
    rgb[:, 1, obj_x1:obj_x2, obj_y1:obj_y2] = 0.4
    rgb[:, 2, obj_x1:obj_x2, obj_y1:obj_y2] = 0.8

    # Add noise
    rgb = rgb + torch.randn_like(rgb) * 0.02
    rgb = rgb.clamp(0, 1)

    # ---- Text (mock token ids for "pick up the blue box") ----
    text_ids = torch.randint(100, 5000, (1, 10), device=device)

    # ---- Ground Truth Affordance ----
    # Normalized bounding box of the graspable object [x1, y1, x2, y2]
    gt_affordance = torch.tensor([[
        [obj_y1 / W, obj_x1 / H, obj_y2 / W, obj_x2 / H],  # object bbox
        [obj_y1 / W + 0.02, obj_x1 / H + 0.02,
         obj_y2 / W - 0.02, obj_x2 / H - 0.02],  # grasp region (slightly inside)
        [0.0, 0.0, 1.0, 1.0],  # workspace bbox (full image)
        [0.0, 0.0, 0.0, 0.0],  # unused
    ]], dtype=torch.float32, device=device)  # [1, 4, 4]

    # ---- Ground Truth Constraints ----
    gt_constraints = {
        "approach_direction": torch.tensor([[0.0, 0.0, -1.0]], device=device),  # top-down
        "contact_normal": torch.tensor([[0.0, 0.0, 1.0]], device=device),       # upward normal
        "gripper_width": torch.tensor([[0.6]], device=device),                   # 60% open
        "force_limit": torch.tensor([[5.0]], device=device),                     # 5N
    }

    return {
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "text_ids": text_ids,
        "gt_affordance": gt_affordance,
        "gt_constraints": gt_constraints,
    }
