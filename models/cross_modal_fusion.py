"""Cross-Modal Fusion: Align 3D Gaussian tokens with 2D ViT tokens.

Architecture per layer (standard Transformer decoder):
    Self-Attn(3D, 3D)  ->  Cross-Attn(3D queries, 2D keys/values)  ->  FFN

The 2D tokens serve as a read-only memory bank. Only 3D tokens are updated.
This enriches geometric representations with pretrained visual semantics,
so the LLM can process 3D tokens similarly to how it processes ViT tokens.

References:
    - 3D-LLM (Hong et al., NeurIPS 2023) - 3D tokens attending to 2D features
    - LEO (Huang et al., ICML 2024) - embodied 3D-language grounding
    - LLaVA (Liu et al., NeurIPS 2023) - 2-stage alignment protocol
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """One layer of: Self-Attn(3D) -> Cross-Attn(3D, 2D) -> FFN.

    Uses pre-norm residual connections (modern Transformer convention).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, N_3d, D]  3D tokens (updated)
            kv: [B, N_2d, D]  2D tokens (read-only)
        Returns:
            q:  [B, N_3d, D]  updated 3D tokens
        """
        # Self-attention among 3D tokens
        q2 = self.norm1(q)
        q = q + self.self_attn(q2, q2, q2, need_weights=False)[0]

        # Cross-attention: 3D queries attend to 2D keys/values
        q2 = self.norm2(q)
        q = q + self.cross_attn(q2, kv, kv, need_weights=False)[0]

        # FFN
        q2 = self.norm3(q)
        q = q + self.ffn(q2)

        return q


class CrossModalFusion(nn.Module):
    """Fuse 3D Gaussian tokens with 2D ViT tokens via cross-attention.

    Input dimensions:
        3D tokens: [B, N_3d, d_3d]  (from PointNet++, e.g. 512d)
        2D tokens: [B, N_2d, d_2d]  (from ViT projector, e.g. 4096d)
    Output:
        fused:     [B, N_3d, d_out]  (ready for LLM, e.g. 4096d)
    """

    def __init__(
        self,
        d_3d: int = 512,
        d_2d: int = 4096,
        d_model: int = 4096,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_3d = nn.Linear(d_3d, d_model)
        # 2D tokens are already d_model if d_2d == d_model; use LN for refinement
        self.proj_2d = (
            nn.LayerNorm(d_model) if d_2d == d_model
            else nn.Sequential(nn.Linear(d_2d, d_model), nn.LayerNorm(d_model))
        )
        self.layers = nn.ModuleList(
            [FusionLayer(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens_3d: torch.Tensor,
        tokens_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens_3d: [B, N_3d, d_3d]  raw 3D tokens from PointNet++
            tokens_2d: [B, N_2d, d_2d]  2D tokens from ViT (post-projector)
        Returns:
            fused: [B, N_3d, d_model]  fused 3D tokens for LLM injection
        """
        q = self.proj_3d(tokens_3d)
        kv = self.proj_2d(tokens_2d)

        for layer in self.layers:
            q = layer(q, kv)

        return self.out_norm(q)


class AlignmentLoss(nn.Module):
    """Feature alignment loss between fused 3D tokens and 2D ViT tokens.

    Encourages the fused 3D representation to stay within the ViT feature
    manifold, so the LLM can process them using its pretrained visual
    understanding pathways.

    L_align = MSE(pool(fused_3d), pool(2d)) + lambda_cos * (1 - cos_sim)
    """

    def __init__(self, lambda_cos: float = 0.5):
        super().__init__()
        self.lambda_cos = lambda_cos

    def forward(
        self,
        fused_3d: torch.Tensor,
        tokens_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_3d: [B, N_3d, D]
            tokens_2d: [B, N_2d, D]
        Returns:
            scalar loss
        """
        # Global average pool to [B, D]
        pool_3d = fused_3d.mean(dim=1)
        pool_2d = tokens_2d.mean(dim=1).detach()  # stop gradient on teacher

        # MSE alignment
        mse = F.mse_loss(pool_3d, pool_2d)

        # Cosine similarity alignment
        cos_sim = F.cosine_similarity(pool_3d, pool_2d, dim=-1).mean()
        cos_loss = 1.0 - cos_sim

        return mse + self.lambda_cos * cos_loss
