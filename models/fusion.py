"""DualStreamFusion: Fuse 2D visual tokens and 3D Gaussian tokens.

Supports two modes:
  - 'concat': Simple concatenation (recommended for first iteration)
  - 'cross_attn': Bidirectional cross-attention with gated fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Single cross-attention: query attends to key-value."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D]
            kv: [B, N_kv, D]
        Returns:
            out: [B, N_q, D]
        """
        # Cross attention with residual
        attn_out, _ = self.attn(query, kv, kv)
        query = self.norm(query + attn_out)
        # FFN with residual
        query = self.norm2(query + self.ffn(query))
        return query


class DualStreamFusion(nn.Module):
    """Fuse 2D visual tokens from SigLIP and 3D tokens from GS Encoder.

    Args:
        d_model: Token dimension (must match LLM hidden size)
        mode: 'concat' or 'cross_attn'
        num_heads: Number of attention heads (for cross_attn mode)
    """

    def __init__(
        self,
        d_model: int = 1024,
        mode: str = "concat",
        num_heads: int = 8,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == "cross_attn":
            # Bidirectional cross-attention
            self.cross_3d_to_2d = CrossAttentionBlock(d_model, num_heads)
            self.cross_2d_to_3d = CrossAttentionBlock(d_model, num_heads)
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        tokens_2d: torch.Tensor,
        tokens_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens_2d: [B, N_2d, D] from 2D visual encoder
            tokens_3d: [B, N_3d, D] from GS Encoder

        Returns:
            fused: [B, N_fused, D] fused tokens for LLM
        """
        if self.mode == "concat":
            # Simple concatenation — let LLM's self-attention handle fusion
            return torch.cat([tokens_2d, tokens_3d], dim=1)

        elif self.mode == "cross_attn":
            # Bidirectional cross-attention
            enhanced_3d = self.cross_3d_to_2d(tokens_3d, tokens_2d)  # 3D queries 2D
            enhanced_2d = self.cross_2d_to_3d(tokens_2d, tokens_3d)  # 2D queries 3D

            # Gated fusion per token position
            # Pad to same length and fuse
            max_len = max(enhanced_2d.shape[1], enhanced_3d.shape[1])
            if enhanced_2d.shape[1] < max_len:
                enhanced_2d = F.pad(enhanced_2d, (0, 0, 0, max_len - enhanced_2d.shape[1]))
            if enhanced_3d.shape[1] < max_len:
                enhanced_3d = F.pad(enhanced_3d, (0, 0, 0, max_len - enhanced_3d.shape[1]))

            gate = self.gate(torch.cat([enhanced_2d, enhanced_3d], dim=-1))
            fused = gate * enhanced_2d + (1 - gate) * enhanced_3d
            fused = self.output_proj(fused)
            return fused

        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
