"""Differentiable 3D Gaussian Splatting renderer for auxiliary supervision.

Implements a simplified differentiable renderer that projects 3D Gaussians back
to 2D image space. Used as auxiliary training loss for the DepthToGaussian module:

    L_render = L1(rendered_rgb, target_rgb) + lambda_ssim * (1 - SSIM(rendered, target))
              + lambda_depth * L1(rendered_depth, target_depth)

This is a pure-PyTorch implementation (no CUDA kernels) for portability.
For production, consider using the official diff-gaussian-rasterization package.

Reference: 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix.

    Args:
        q: [..., 4] quaternion (w, x, y, z)

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def compute_cov2d(
    means_3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D covariance matrices for Gaussians projected to image plane.

    Follows the EWA splatting formulation:
        Sigma_2d = J @ R @ S @ S^T @ R^T @ J^T
    where J is the Jacobian of the perspective projection.

    Args:
        means_3d: [B, N, 3] Gaussian centers in camera frame
        scales: [B, N, 3] Gaussian scales
        rotations: [B, N, 4] quaternions (w,x,y,z)
        intrinsics: [B, 3, 3] camera intrinsics

    Returns:
        cov2d: [B, N, 2, 2] 2D covariance matrices
    """
    B, N, _ = means_3d.shape

    # Rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(rotations)  # [B, N, 3, 3]

    # Scale matrix: S = diag(sx, sy, sz)
    S = torch.diag_embed(scales)  # [B, N, 3, 3]

    # 3D covariance: Sigma = R @ S @ S^T @ R^T
    RS = R @ S  # [B, N, 3, 3]
    cov3d = RS @ RS.transpose(-1, -2)  # [B, N, 3, 3]

    # Jacobian of perspective projection
    fx = intrinsics[:, 0, 0].unsqueeze(1)  # [B, 1]
    fy = intrinsics[:, 1, 1].unsqueeze(1)  # [B, 1]
    z = means_3d[:, :, 2].clamp(min=1e-6)  # [B, N]
    z_sq = z * z

    # J = [[fx/z, 0, -fx*x/z^2],
    #      [0, fy/z, -fy*y/z^2]]
    # We only need the 2x3 top-left block for 2D projection
    J = torch.zeros(B, N, 2, 3, device=means_3d.device, dtype=means_3d.dtype)
    J[:, :, 0, 0] = fx / z
    J[:, :, 0, 2] = -fx * means_3d[:, :, 0] / z_sq
    J[:, :, 1, 1] = fy / z
    J[:, :, 1, 2] = -fy * means_3d[:, :, 1] / z_sq

    # 2D covariance: J @ Sigma_3d @ J^T
    cov2d = J @ cov3d @ J.transpose(-1, -2)  # [B, N, 2, 2]

    # Add small diagonal for numerical stability
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device, dtype=cov2d.dtype)

    return cov2d


def render_gaussians(
    means_3d: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coeffs: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    sh_degree: int = 2,
    uncertainty: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Render 3D Gaussians to a 2D image using differentiable splatting.

    This is a simplified alpha-compositing renderer (not tile-based).
    For each pixel, we evaluate all Gaussians and composite front-to-back.

    For efficiency, we only evaluate Gaussians within a bounding box
    around each pixel. With num_gaussians < 2048, this is tractable.

    Args:
        means_3d: [B, N, 3] Gaussian centers in camera frame
        scales: [B, N, 3] Gaussian scales
        rotations: [B, N, 4] unit quaternions
        opacities: [B, N, 1] opacities in [0, 1]
        sh_coeffs: [B, N, K] spherical harmonics coefficients
        intrinsics: [B, 3, 3] camera intrinsics
        image_size: (H, W) output image size
        sh_degree: SH degree (0=DC only for simplified rendering)
        uncertainty: [B, N, 1] optional per-Gaussian uncertainty values

    Returns:
        dict with:
            rendered_rgb: [B, 3, H, W] rendered color image
            rendered_depth: [B, 1, H, W] rendered depth map
            rendered_alpha: [B, 1, H, W] alpha/occupancy map
            rendered_uncertainty: [B, 1, H, W] uncertainty map
    """
    B, N, _ = means_3d.shape
    H, W = image_size
    device = means_3d.device
    dtype = means_3d.dtype

    # 1. Project Gaussian centers to 2D
    fx = intrinsics[:, 0, 0]  # [B]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    z = means_3d[:, :, 2].clamp(min=1e-4)  # [B, N]
    px = fx.unsqueeze(1) * means_3d[:, :, 0] / z + cx.unsqueeze(1)  # [B, N]
    py = fy.unsqueeze(1) * means_3d[:, :, 1] / z + cy.unsqueeze(1)  # [B, N]

    # 2. Compute 2D covariance
    cov2d = compute_cov2d(means_3d, scales, rotations, intrinsics)  # [B, N, 2, 2]

    # 3. Extract SH DC component as base color (degree 0)
    # SH coefficients: first 3 values are DC (R, G, B)
    # C0 = 0.28209479177387814 (SH basis for l=0, m=0)
    C0 = 0.28209479177387814
    colors = sh_coeffs[:, :, :3] * C0 + 0.5  # [B, N, 3] - DC color
    colors = colors.clamp(0, 1)

    # 4. Compute inverse covariance for Gaussian evaluation
    det = cov2d[:, :, 0, 0] * cov2d[:, :, 1, 1] - cov2d[:, :, 0, 1] * cov2d[:, :, 1, 0]
    det = det.clamp(min=1e-8)  # [B, N]

    cov_inv = torch.zeros_like(cov2d)
    cov_inv[:, :, 0, 0] = cov2d[:, :, 1, 1] / det.unsqueeze(-1).unsqueeze(-1).squeeze(-1).squeeze(-1)
    cov_inv[:, :, 1, 1] = cov2d[:, :, 0, 0] / det.unsqueeze(-1).unsqueeze(-1).squeeze(-1).squeeze(-1)
    cov_inv[:, :, 0, 1] = -cov2d[:, :, 0, 1] / det.unsqueeze(-1).unsqueeze(-1).squeeze(-1).squeeze(-1)
    cov_inv[:, :, 1, 0] = -cov2d[:, :, 1, 0] / det.unsqueeze(-1).unsqueeze(-1).squeeze(-1).squeeze(-1)

    # Simplify: use scalar approach for efficiency
    inv_00 = cov2d[:, :, 1, 1] / det  # [B, N]
    inv_11 = cov2d[:, :, 0, 0] / det
    inv_01 = -cov2d[:, :, 0, 1] / det

    # 5. Render via splatting
    # For efficiency with small N, we use a vectorized approach:
    # evaluate all Gaussians at all pixels simultaneously

    # Create pixel grid
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]

    # Compute distance from each pixel to each Gaussian center
    # [B, N, 1, 1] - [1, 1, H, W] -> [B, N, H, W]
    dx = grid_x.unsqueeze(0).unsqueeze(0) - px.unsqueeze(-1).unsqueeze(-1)  # [B, N, H, W]
    dy = grid_y.unsqueeze(0).unsqueeze(0) - py.unsqueeze(-1).unsqueeze(-1)

    # Gaussian evaluation: exp(-0.5 * (dx, dy) @ cov_inv @ (dx, dy)^T)
    # = exp(-0.5 * (inv_00*dx^2 + 2*inv_01*dx*dy + inv_11*dy^2))
    power = -0.5 * (
        inv_00.unsqueeze(-1).unsqueeze(-1) * dx * dx
        + 2 * inv_01.unsqueeze(-1).unsqueeze(-1) * dx * dy
        + inv_11.unsqueeze(-1).unsqueeze(-1) * dy * dy
    )  # [B, N, H, W]

    # Clamp for numerical stability
    alpha_per_gaussian = torch.exp(power.clamp(max=0, min=-10))  # [B, N, H, W]
    alpha_per_gaussian = alpha_per_gaussian * opacities.squeeze(-1).unsqueeze(-1).unsqueeze(-1)
    alpha_per_gaussian = alpha_per_gaussian.clamp(max=0.99)

    # 6. Sort by depth for front-to-back compositing
    depth_order = z.argsort(dim=1)  # [B, N] indices sorted by depth
    depth_order_expanded = depth_order.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    alpha_sorted = torch.gather(alpha_per_gaussian, 1, depth_order_expanded)  # [B, N, H, W]

    color_order = depth_order.unsqueeze(-1).expand(-1, -1, 3)
    colors_sorted = torch.gather(colors, 1, color_order)  # [B, N, 3]

    depth_vals_sorted = torch.gather(z, 1, depth_order)  # [B, N]

    # 7. Alpha compositing (front-to-back)
    # T_i = prod(1 - alpha_j) for j < i
    one_minus_alpha = 1 - alpha_sorted  # [B, N, H, W]
    # Cumulative product gives transmittance at each Gaussian
    # T_i = cumprod of (1-alpha) for indices 0..i-1
    # We shift by 1: T_0 = 1, T_i = prod(1-alpha_j for j<i)
    T = torch.ones(B, 1, H, W, device=device, dtype=dtype)
    transmittance = torch.cumprod(one_minus_alpha, dim=1)
    # Shift: prepend 1, remove last
    transmittance = torch.cat([T, transmittance[:, :-1]], dim=1)  # [B, N, H, W]

    # Weight = T_i * alpha_i
    weights = transmittance * alpha_sorted  # [B, N, H, W]

    # Weighted sum of colors: [B, 3, H, W]
    rendered_rgb = torch.einsum("bnhw,bnc->bchw", weights, colors_sorted)

    # Weighted sum of depths: [B, 1, H, W]
    rendered_depth = (weights * depth_vals_sorted.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)

    # Total alpha
    rendered_alpha = weights.sum(dim=1, keepdim=True)

    # Uncertainty rendering
    if uncertainty is not None:
        # Sort per-Gaussian uncertainty by depth (same order as colors/depths)
        unc_sorted = torch.gather(uncertainty.squeeze(-1), 1, depth_order)  # [B, N]
        rendered_uncertainty = (weights * unc_sorted.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
    else:
        # Geometric uncertainty: depth variance
        depth_diff_sq = (depth_vals_sorted.unsqueeze(-1).unsqueeze(-1) - rendered_depth) ** 2
        rendered_uncertainty = (weights * depth_diff_sq).sum(dim=1, keepdim=True)

    return {
        "rendered_rgb": rendered_rgb.clamp(0, 1),
        "rendered_depth": rendered_depth,
        "rendered_alpha": rendered_alpha.clamp(0, 1),
        "rendered_uncertainty": rendered_uncertainty,
    }


class SSIMLoss(nn.Module):
    """Structural Similarity (SSIM) loss for image quality assessment."""

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)  # [ws, ws]
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
        self.register_buffer("window", window)
        self.channels = channels

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM between two images.

        Args:
            img1, img2: [B, C, H, W] images in [0, 1]

        Returns:
            ssim: scalar SSIM value (higher is better, max 1.0)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = self.window_size // 2

        window = self.window.to(img1.device, img1.dtype)

        mu1 = F.conv2d(img1, window, padding=pad, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=self.channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=self.channels) - mu12

        ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim.mean()


class GaussianRenderingLoss(nn.Module):
    """Combined rendering loss for 3DGS supervision.

    L = lambda_l1 * L1(rgb) + lambda_ssim * (1 - SSIM(rgb))
        + lambda_depth * L1(depth)
        + lambda_opacity * entropy(opacity)  # encourages binary opacity
    """

    def __init__(
        self,
        lambda_l1: float = 0.8,
        lambda_ssim: float = 0.2,
        lambda_depth: float = 0.5,
        lambda_opacity: float = 0.01,
        image_size: tuple[int, int] = (64, 64),  # Render at low resolution for speed
        sh_degree: int = 2,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.lambda_opacity = lambda_opacity
        self.image_size = image_size
        self.sh_degree = sh_degree
        self.ssim = SSIMLoss(window_size=7, channels=3)

    def forward(
        self,
        gaussians: torch.Tensor,
        intrinsics: torch.Tensor,
        target_rgb: torch.Tensor,
        target_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute rendering loss.

        Args:
            gaussians: [B, N, D] predicted Gaussians (xyz+scale+rot+opacity+sh)
            intrinsics: [B, 3, 3] camera intrinsics
            target_rgb: [B, 3, H, W] ground truth RGB
            target_depth: [B, 1, H, W] ground truth depth

        Returns:
            dict with 'loss' (total), 'l1_rgb', 'ssim', 'l1_depth', 'opacity_reg'
        """
        H, W = self.image_size

        # Parse Gaussian parameters
        means_3d = gaussians[:, :, :3]
        scales = gaussians[:, :, 3:6]
        rotations = gaussians[:, :, 6:10]
        opacities = gaussians[:, :, 10:11]
        sh_coeffs = gaussians[:, :, 11:]

        # Scale intrinsics to render resolution
        B = intrinsics.shape[0]
        orig_h, orig_w = target_rgb.shape[2], target_rgb.shape[3]
        scaled_intrinsics = intrinsics.clone()
        scaled_intrinsics[:, 0, :] *= W / orig_w
        scaled_intrinsics[:, 1, :] *= H / orig_h

        # Render
        rendered = render_gaussians(
            means_3d, scales, rotations, opacities, sh_coeffs,
            scaled_intrinsics, (H, W), self.sh_degree,
        )

        # Downsample targets to render resolution
        target_rgb_ds = F.interpolate(target_rgb, size=(H, W), mode="bilinear", align_corners=False)
        target_depth_ds = F.interpolate(target_depth, size=(H, W), mode="nearest")

        # L1 RGB loss
        l1_rgb = F.l1_loss(rendered["rendered_rgb"], target_rgb_ds)

        # SSIM loss
        ssim_val = self.ssim(rendered["rendered_rgb"], target_rgb_ds)
        ssim_loss = 1 - ssim_val

        # L1 Depth loss (only where alpha > 0.5)
        alpha_mask = (rendered["rendered_alpha"] > 0.5).float()
        if alpha_mask.sum() > 0:
            l1_depth = (
                (rendered["rendered_depth"] - target_depth_ds).abs() * alpha_mask
            ).sum() / alpha_mask.sum().clamp(min=1)
        else:
            l1_depth = torch.tensor(0.0, device=gaussians.device, dtype=gaussians.dtype)

        # Opacity regularization (encourage binary: 0 or 1)
        opacity_reg = (-opacities * torch.log(opacities.clamp(min=1e-6))
                       - (1 - opacities) * torch.log((1 - opacities).clamp(min=1e-6))).mean()

        # Total loss
        loss = (
            self.lambda_l1 * l1_rgb
            + self.lambda_ssim * ssim_loss
            + self.lambda_depth * l1_depth
            + self.lambda_opacity * opacity_reg
        )

        return {
            "loss": loss,
            "l1_rgb": l1_rgb,
            "ssim": ssim_val,
            "l1_depth": l1_depth,
            "opacity_reg": opacity_reg,
        }
