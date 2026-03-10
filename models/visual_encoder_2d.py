"""2D Visual Encoder wrapper.

Wraps a pretrained vision model (SigLIP / CLIP / DINOv2) to extract
2D visual tokens from RGB images. In the full RoboBrain pipeline,
this would be the frozen SigLIP encoder.

For validation, we provide a lightweight stand-in encoder.
"""

import torch
import torch.nn as nn


class LightweightVisualEncoder(nn.Module):
    """Lightweight 2D visual encoder for validation.

    Replaces SigLIP during development. Uses a simple CNN + projection
    to generate visual tokens matching the LLM hidden dimension.
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        token_dim: int = 1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            3, token_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(token_dim)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, token_dim) * 0.02
        )

        # Simple transformer layers for feature refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=token_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, 3, H, W]

        Returns:
            tokens: [B, num_patches, token_dim]
        """
        # [B, token_dim, H/P, W/P]
        x = self.patch_embed(rgb)
        # [B, num_patches, token_dim]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x
