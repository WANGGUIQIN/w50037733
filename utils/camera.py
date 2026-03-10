"""Camera utilities: depth backprojection and point cloud processing."""

import torch
import torch.nn.functional as F


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Backproject depth map to 3D point cloud using camera intrinsics.

    Args:
        depth: [B, 1, H, W] depth map in meters
        intrinsics: [B, 3, 3] camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]

    Returns:
        points: [B, H, W, 3] 3D coordinates in camera frame
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth.dtype),
        torch.arange(W, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(u)
    # [3, H, W]
    pixel_coords = torch.stack([u, v, ones], dim=0)

    # Invert intrinsics: K^{-1} @ [u, v, 1]^T * d
    # torch.inverse requires float32; cast up then back
    orig_dtype = intrinsics.dtype
    K_inv = torch.inverse(intrinsics.float()).to(orig_dtype)  # [B, 3, 3]

    # [B, 3, H*W]
    pixel_flat = pixel_coords.reshape(3, -1).unsqueeze(0).expand(B, -1, -1)
    cam_coords = K_inv @ pixel_flat  # [B, 3, H*W]

    # Scale by depth
    depth_flat = depth.reshape(B, 1, -1)  # [B, 1, H*W]
    points = cam_coords * depth_flat  # [B, 3, H*W]

    # Reshape to [B, H, W, 3]
    points = points.reshape(B, 3, H, W).permute(0, 2, 3, 1)
    return points


def normalize_points(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize point cloud to unit cube centered at origin.

    Args:
        points: [B, N, 3]

    Returns:
        normalized: [B, N, 3] normalized points
        centroid: [B, 1, 3] centroid used
        scale: [B, 1, 1] scale factor used
    """
    centroid = points.mean(dim=1, keepdim=True)  # [B, 1, 3]
    points_centered = points - centroid
    scale = points_centered.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    scale = scale.clamp(min=1e-6)
    normalized = points_centered / scale
    return normalized, centroid, scale
