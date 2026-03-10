"""GaussianEncoder: Encode 3D Gaussian parameters into tokens for LLM.

Uses a PointNet++-inspired architecture to hierarchically aggregate
3D Gaussian features into a fixed number of tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction layer.

    Samples centroids via FPS, groups neighbors by ball query,
    and applies shared MLPs to aggregate local features.
    """

    def __init__(
        self,
        num_centroids: int,
        radius: float,
        max_neighbors: int,
        in_channels: int,
        mlp_channels: list[int],
    ):
        super().__init__()
        self.num_centroids = num_centroids
        self.radius = radius
        self.max_neighbors = max_neighbors

        layers = []
        prev_ch = in_channels + 3  # +3 for relative xyz
        for out_ch in mlp_channels:
            layers.extend([
                nn.Conv1d(prev_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
            ])
            prev_ch = out_ch
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: [B, N, 3] point positions
            features: [B, N, C] point features

        Returns:
            new_xyz: [B, num_centroids, 3]
            new_features: [B, num_centroids, C']
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # 1. FPS to select centroids
        centroid_idx = self._fps(xyz, self.num_centroids)  # [B, num_centroids]

        new_xyz = torch.gather(
            xyz, 1, centroid_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, num_centroids, 3]

        # 2. Ball query: find neighbors within radius
        # [B, num_centroids, max_neighbors]
        group_idx = self._ball_query(xyz, new_xyz)

        # 3. Group features
        grouped_xyz = torch.gather(
            xyz, 1, group_idx.reshape(B, -1, 1).expand(-1, -1, 3)
        ).reshape(B, self.num_centroids, self.max_neighbors, 3)

        # Relative coordinates
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

        grouped_features = torch.gather(
            features, 1,
            group_idx.reshape(B, -1, 1).expand(-1, -1, features.shape[-1])
        ).reshape(B, self.num_centroids, self.max_neighbors, -1)

        # Concatenate relative xyz with features
        grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
        # [B, num_centroids, max_neighbors, C+3]

        # 4. Apply shared MLP
        # Reshape for Conv1d: [B*num_centroids, C+3, max_neighbors]
        grouped = rearrange(grouped, "b n k c -> (b n) c k")
        grouped = self.mlp(grouped)  # [(B*n), C', k]

        # 5. Max pooling over neighbors
        new_features = grouped.max(dim=-1)[0]  # [(B*n), C']
        new_features = rearrange(
            new_features, "(b n) c -> b n c", b=B, n=self.num_centroids
        )

        return new_xyz, new_features

    def _fps(self, xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Farthest point sampling."""
        B, N, _ = xyz.shape
        device = xyz.device
        indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
        distances = torch.full((B, N), float("inf"), device=device)
        farthest = torch.randint(0, N, (B,), device=device)

        for i in range(num_samples):
            indices[:, i] = farthest
            centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            distances = torch.min(distances, dist)
            farthest = distances.argmax(dim=-1)

        return indices

    def _ball_query(
        self,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """Ball query: find K nearest neighbors within radius."""
        B, N, _ = xyz.shape
        M = new_xyz.shape[1]
        device = xyz.device

        # Pairwise distances: [B, M, N]
        dist = torch.cdist(new_xyz, xyz)

        # Mask points outside radius
        dist[dist > self.radius] = float("inf")

        # Get top-k nearest
        _, idx = dist.topk(self.max_neighbors, dim=-1, largest=False)  # [B, M, K]

        # Handle case where fewer than K neighbors within radius
        # Fill with first neighbor
        first_idx = idx[:, :, 0:1].expand_as(idx)
        mask = (dist.gather(2, idx) == float("inf"))
        idx[mask] = first_idx[mask]

        return idx


class GaussianEncoder(nn.Module):
    """Encode 3D Gaussians into a fixed number of tokens for LLM consumption.

    Architecture: 3-level PointNet++ hierarchy
        2048 Gaussians → 512 → 128 → num_tokens (e.g., 64)

    Each Gaussian has D dimensions:
        3(xyz) + 3(scale) + 4(rotation) + 1(opacity) + K(SH)
    """

    def __init__(
        self,
        gaussian_dim: int = 38,  # 3+3+4+1+27 (SH degree 2)
        num_tokens: int = 64,
        token_dim: int = 1024,  # Match LLM hidden dim
    ):
        super().__init__()
        self.num_tokens = num_tokens
        feat_dim = gaussian_dim - 3  # Features without xyz

        # Input projection for Gaussian parameters (excluding xyz)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # PointNet++ hierarchy
        self.sa1 = PointNetSetAbstraction(
            num_centroids=512, radius=0.2, max_neighbors=32,
            in_channels=128, mlp_channels=[128, 128, 256],
        )
        self.sa2 = PointNetSetAbstraction(
            num_centroids=128, radius=0.4, max_neighbors=32,
            in_channels=256, mlp_channels=[256, 256, 512],
        )
        self.sa3 = PointNetSetAbstraction(
            num_centroids=num_tokens, radius=0.8, max_neighbors=32,
            in_channels=512, mlp_channels=[512, 512, token_dim],
        )

        # Final layer norm
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, gaussians: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gaussians: [B, N, D] where D = 3(xyz) + rest

        Returns:
            tokens: [B, num_tokens, token_dim] ready for LLM
        """
        xyz = gaussians[..., :3]  # [B, N, 3]
        params = gaussians[..., 3:]  # [B, N, D-3]

        # Project parameters to feature space
        features = self.input_proj(params)  # [B, N, 128]

        # Hierarchical abstraction
        xyz1, feat1 = self.sa1(xyz, features)     # [B, 512, 256]
        xyz2, feat2 = self.sa2(xyz1, feat1)        # [B, 128, 512]
        xyz3, feat3 = self.sa3(xyz2, feat2)        # [B, num_tokens, token_dim]

        tokens = self.norm(feat3)
        return tokens
