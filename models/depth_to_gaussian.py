"""DepthToGaussian: Convert single-frame RGBD to 3D Gaussian parameters.

The key insight is that with RGBD input, Gaussian center positions (xyz) are
deterministically computed via depth backprojection. The network only needs to
learn per-Gaussian shape (scale, rotation) and appearance (SH coefficients).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.camera import backproject_depth


class ResBlock(nn.Module):
    """Simple residual block for feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class DepthToGaussian(nn.Module):
    """Generate 3D Gaussian parameters from a single RGBD frame.

    Input:  RGB [B,3,H,W], Depth [B,1,H,W], Intrinsics [B,3,3]
    Output: Gaussians [B, N, D] where D = 3(xyz) + 3(scale) + 4(rotation) + 1(opacity) + K(SH)

    The xyz comes from depth backprojection (deterministic).
    Scale, rotation, opacity, SH are predicted by the network.
    """

    def __init__(
        self,
        num_gaussians: int = 2048,
        sh_degree: int = 2,
        feat_dim: int = 128,
        num_res_blocks: int = 4,
        predict_uncertainty: bool = False,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.sh_coeffs = (sh_degree + 1) ** 2 * 3  # RGB SH coefficients
        # 3(scale) + 4(rotation quaternion) + 1(opacity) + sh_coeffs
        self.param_dim = 3 + 4 + 1 + self.sh_coeffs
        self.predict_uncertainty = predict_uncertainty

        # Feature extraction from RGBD (4 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, feat_dim, 7, stride=2, padding=3),  # H/2
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1),  # H/4
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            *[ResBlock(feat_dim) for _ in range(num_res_blocks)],
        )

        # Per-pixel Gaussian parameter prediction head
        self.param_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1),
            nn.GELU(),
            nn.Conv2d(feat_dim, self.param_dim, 1),
        )

        # Per-pixel uncertainty prediction head
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim // 2, 1),
                nn.GELU(),
                nn.Conv2d(feat_dim // 2, 1, 1),
            )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
            intrinsics: [B, 3, 3]

        Returns:
            gaussians: [B, num_gaussians, 3+3+4+1+sh_coeffs]
        """
        B, _, H, W = rgb.shape

        # 1. Backproject depth to 3D points
        points_3d = backproject_depth(depth, intrinsics)  # [B, H, W, 3]

        # 2. Extract features from RGBD
        rgbd = torch.cat([rgb, depth], dim=1)  # [B, 4, H, W]
        features = self.encoder(rgbd)  # [B, feat_dim, H/4, W/4]

        # 3. Predict per-pixel Gaussian parameters
        params = self.param_head(features)  # [B, param_dim, H/4, W/4]

        # 4. Downsample 3D points to match feature map resolution
        h_feat, w_feat = features.shape[2], features.shape[3]
        points_down = F.adaptive_avg_pool2d(
            points_3d.permute(0, 3, 1, 2),  # [B, 3, H, W]
            (h_feat, w_feat),
        )  # [B, 3, h, w]

        # 5. Reshape to point set: [B, h*w, D]
        points_flat = points_down.permute(0, 2, 3, 1).reshape(B, -1, 3)  # [B, N_all, 3]
        params_flat = params.permute(0, 2, 3, 1).reshape(B, -1, self.param_dim)  # [B, N_all, param_dim]

        # 5b. If predicting uncertainty, compute and concatenate with params
        # so that _fps_select uses the same indices for both
        if self.predict_uncertainty:
            uncertainty_map = self.uncertainty_head(features)  # [B, 1, h, w]
            uncertainty_flat = uncertainty_map.permute(0, 2, 3, 1).reshape(B, -1, 1)  # [B, N_all, 1]
            params_flat = torch.cat([params_flat, uncertainty_flat], dim=-1)  # [B, N_all, param_dim+1]

        # 6. Farthest Point Sampling to select num_gaussians points
        gaussians_xyz, gaussians_params = self._fps_select(
            points_flat, params_flat, self.num_gaussians
        )

        # 6b. Split out uncertainty after FPS selection
        if self.predict_uncertainty:
            raw_uncertainty = gaussians_params[..., -1:]  # [B, N, 1]
            gaussians_params = gaussians_params[..., :-1]  # [B, N, param_dim]

        # 7. Activate parameters
        scale = F.softplus(gaussians_params[..., :3])  # positive scale
        rotation = F.normalize(gaussians_params[..., 3:7], dim=-1)  # unit quaternion
        opacity = torch.sigmoid(gaussians_params[..., 7:8])  # [0, 1]
        sh = gaussians_params[..., 8:]  # SH coefficients (raw)

        # 8. Concatenate: [B, N, 3+3+4+1+sh(+1 uncertainty)]
        parts = [gaussians_xyz, scale, rotation, opacity, sh]
        if self.predict_uncertainty:
            uncertainty = F.softplus(raw_uncertainty)  # always positive
            parts.append(uncertainty)
        gaussians = torch.cat(parts, dim=-1)
        return gaussians

    def _fps_select(
        self,
        points: torch.Tensor,
        params: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Farthest Point Sampling for selecting representative Gaussians.

        Args:
            points: [B, N, 3]
            params: [B, N, D]
            num_samples: number of points to select

        Returns:
            selected_points: [B, num_samples, 3]
            selected_params: [B, num_samples, D]
        """
        B, N, _ = points.shape
        device = points.device

        if N <= num_samples:
            # Pad if fewer points than requested
            pad_n = num_samples - N
            points = F.pad(points, (0, 0, 0, pad_n))
            params = F.pad(params, (0, 0, 0, pad_n))
            return points, params

        # Simplified FPS
        indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distances = torch.full((B, N), float("inf"), device=device)

        # Start from random point
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_samples):
            indices[:, i] = farthest
            centroid = points[torch.arange(B, device=device), farthest].unsqueeze(1)  # [B, 1, 3]
            dist = torch.sum((points - centroid) ** 2, dim=-1)  # [B, N]
            distances = torch.min(distances, dist)
            farthest = distances.argmax(dim=-1)  # [B]

        # Gather selected points and params
        indices_expanded = indices.unsqueeze(-1)
        selected_points = torch.gather(
            points, 1, indices_expanded.expand(-1, -1, 3)
        )
        selected_params = torch.gather(
            params, 1, indices_expanded.expand(-1, -1, params.shape[-1])
        )
        return selected_points, selected_params
