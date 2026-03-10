"""RoboBrain3DGS: Main model integrating RGBD → 3D Gaussian → VLM.

Architecture:
    RGB ──→ 2D Visual Encoder (SigLIP) ──→ 2D Tokens ─┐
                                                        ├─→ Fusion ──→ LLM ──→ Affordance/Constraint
    RGBD ──→ DepthToGaussian ──→ GS Encoder ──→ 3D Tokens ─┘
    Text ──→ Tokenizer ──────────────────────────────────→ LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_to_gaussian import DepthToGaussian
from .gs_encoder import GaussianEncoder
from .fusion import DualStreamFusion
from .visual_encoder_2d import LightweightVisualEncoder


class SimpleLLMBackbone(nn.Module):
    """Simplified LLM backbone for validation.

    Replaces Qwen2-VL during development. Demonstrates the interface
    that the full model would use with a real LLM.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_tokens: [B, N_vis, D] fused 2D+3D tokens
            text_ids: [B, N_text] tokenized text (optional)

        Returns:
            hidden_states: [B, N_total, D]
        """
        tokens = visual_tokens

        if text_ids is not None:
            text_emb = self.token_embed(text_ids)
            tokens = torch.cat([visual_tokens, text_emb], dim=1)

        seq_len = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :seq_len, :]
        hidden = self.transformer(tokens)
        hidden = self.norm(hidden)
        return hidden


class AffordanceHead(nn.Module):
    """Predict affordance bounding boxes from LLM hidden states.

    Outputs normalized bbox coordinates [x1, y1, x2, y2] in [0, 1].
    """

    def __init__(self, hidden_dim: int = 1024, num_affordances: int = 4):
        super().__init__()
        self.num_affordances = num_affordances
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_affordances * 4),  # 4 coords per bbox
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, N, D]

        Returns:
            affordances: [B, num_affordances, 4] normalized bboxes
        """
        # Global average pooling over sequence
        pooled = hidden_states.mean(dim=1)  # [B, D]
        out = self.head(pooled)  # [B, num_affordances * 4]
        out = out.reshape(-1, self.num_affordances, 4)
        return torch.sigmoid(out)  # Normalize to [0, 1]


class ConstraintHead(nn.Module):
    """Predict manipulation constraints from LLM hidden states.

    Outputs constraint parameters:
        - approach_direction: [B, 3] unit vector
        - contact_normal: [B, 3] unit vector
        - gripper_width: [B, 1] scalar
        - force_limit: [B, 1] scalar
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.normal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, N, D]

        Returns:
            dict with constraint parameters
        """
        pooled = hidden_states.mean(dim=1)  # [B, D]
        return {
            "approach_direction": F.normalize(self.direction_head(pooled), dim=-1),
            "contact_normal": F.normalize(self.normal_head(pooled), dim=-1),
            "gripper_width": self.gripper_head(pooled),
            "force_limit": self.force_head(pooled),
        }


class RoboBrain3DGS(nn.Module):
    """Full pipeline: RGBD → 3D Gaussian → VLM → Affordance + Constraint.

    This is a modular framework. For production use:
        - Replace LightweightVisualEncoder with frozen SigLIP
        - Replace SimpleLLMBackbone with Qwen2-VL / LLaMA
        - Load RoboBrain pretrained weights for the 2D pathway
    """

    def __init__(
        self,
        image_size: int = 256,
        num_gaussians: int = 2048,
        sh_degree: int = 2,
        num_gs_tokens: int = 64,
        hidden_dim: int = 1024,
        fusion_mode: str = "concat",
        num_affordances: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        # ====== 2D Stream (would be frozen SigLIP in production) ======
        self.visual_encoder_2d = LightweightVisualEncoder(
            image_size=image_size,
            patch_size=16,
            token_dim=hidden_dim,
        )

        # ====== 3D Stream (new) ======
        gaussian_dim = 3 + 3 + 4 + 1 + (sh_degree + 1) ** 2 * 3
        self.depth_to_gaussian = DepthToGaussian(
            num_gaussians=num_gaussians,
            sh_degree=sh_degree,
            feat_dim=128,
        )
        self.gs_encoder = GaussianEncoder(
            gaussian_dim=gaussian_dim,
            num_tokens=num_gs_tokens,
            token_dim=hidden_dim,
        )

        # ====== Fusion ======
        self.fusion = DualStreamFusion(
            d_model=hidden_dim,
            mode=fusion_mode,
        )

        # ====== LLM Backbone (simplified for validation) ======
        self.llm = SimpleLLMBackbone(
            hidden_dim=hidden_dim,
            num_layers=4,
            num_heads=8,
        )

        # ====== Task Heads ======
        self.affordance_head = AffordanceHead(hidden_dim, num_affordances)
        self.constraint_head = ConstraintHead(hidden_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        text_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            rgb: [B, 3, H, W] RGB image
            depth: [B, 1, H, W] depth map in meters
            intrinsics: [B, 3, 3] camera intrinsics
            text_ids: [B, N_text] tokenized instruction (optional)

        Returns:
            dict with:
                - 'affordances': [B, num_affordances, 4] bounding boxes
                - 'constraints': dict of constraint tensors
                - 'gaussians': [B, N, D] intermediate 3D Gaussians
                - 'hidden_states': [B, N_total, D] LLM hidden states
        """
        # ---- 2D Stream ----
        tokens_2d = self.visual_encoder_2d(rgb)  # [B, N_patches, D]

        # ---- 3D Stream ----
        gaussians = self.depth_to_gaussian(rgb, depth, intrinsics)  # [B, N_gs, D_gs]
        tokens_3d = self.gs_encoder(gaussians)  # [B, num_gs_tokens, D]

        # ---- Fusion ----
        fused_tokens = self.fusion(tokens_2d, tokens_3d)  # [B, N_2d+N_3d, D]

        # ---- LLM ----
        hidden_states = self.llm(fused_tokens, text_ids)  # [B, N_total, D]

        # ---- Task Heads ----
        affordances = self.affordance_head(hidden_states)
        constraints = self.constraint_head(hidden_states)

        return {
            "affordances": affordances,
            "constraints": constraints,
            "gaussians": gaussians,
            "hidden_states": hidden_states,
        }
