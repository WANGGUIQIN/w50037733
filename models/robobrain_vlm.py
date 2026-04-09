"""RoboBrain3DGS-VLM: Integration with Qwen2.5-VL backbone.

This module wraps Qwen2.5-VL (the backbone used by RoboBrain 2.0/2.5)
and injects 3D Gaussian tokens into the VLM's processing pipeline.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Qwen2.5-VL Backbone                  │
    │                                                         │
    │  RGB ──> ViT Encoder ──> Visual Tokens (2D)             │
    │                              │                          │
    │  RGBD ──> DepthToGaussian ──> GS Encoder ──> 3D Tokens  │
    │                              │                          │
    │          [2D Tokens] + [3D Tokens] + [Text Tokens]      │
    │                              │                          │
    │                      LLM Decoder                        │
    │                              │                          │
    │                    Text Output (Affordance/Constraint)   │
    └─────────────────────────────────────────────────────────┘

Integration strategy:
    - Hook into Qwen2.5-VL's forward pass AFTER visual encoding
    - Inject 3D tokens as additional "visual" tokens before the LLM
    - Use Qwen2.5-VL's native text generation for output
    - Keep 2D visual encoder frozen, train 3D branch + LoRA on LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from .depth_to_gaussian import DepthToGaussian
from .gs_encoder import GaussianEncoder
from .cross_modal_fusion import CrossModalFusion

# Supported VLM types: maps model_type string -> (config_class, model_class)
SUPPORTED_VLM = {
    "qwen2_5_vl": (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration),
    "qwen3_vl": (Qwen3VLConfig, Qwen3VLForConditionalGeneration),
}


class GaussianTokenProjector(nn.Module):
    """Project 3D Gaussian tokens to match VLM's hidden dimension.

    Bridges the GS Encoder output dim to the LLM's hidden dim with
    learnable projection + LayerNorm (same pattern as Qwen2.5-VL's
    visual token projector).
    """

    def __init__(self, gs_token_dim: int, llm_hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(gs_token_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
        )
        self.norm = nn.LayerNorm(llm_hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(tokens))


class RoboBrain3DGS_VLM(nn.Module):
    """Full integration of 3D Gaussian branch with Qwen VL backbone.

    Supports Qwen2.5-VL (RoboBrain 2.0) and Qwen3-VL (RoboBrain 2.5).

    This model:
    1. Uses Qwen VL's native vision encoder for 2D visual tokens
    2. Adds DepthToGaussian + GS Encoder for 3D tokens
    3. Injects 3D tokens into the LLM's input sequence
    4. Uses Qwen VL's native text generation for output

    For training:
    - Freeze 2D vision encoder (ViT weights are valuable)
    - Freeze most of LLM, apply LoRA
    - Train: DepthToGaussian + GS Encoder + Projector + LoRA
    """

    def __init__(
        self,
        vlm_config,
        num_gaussians: int = 2048,
        sh_degree: int = 2,
        num_gs_tokens: int = 64,
        gs_encoder_dim: int = 512,
        freeze_vision_encoder: bool = True,
        freeze_llm: bool = False,
    ):
        super().__init__()
        self.num_gs_tokens = num_gs_tokens

        # ====== Load VLM backbone (Qwen2.5-VL or Qwen3-VL) ======
        model_type = getattr(vlm_config, "model_type", "qwen2_5_vl")
        if model_type not in SUPPORTED_VLM:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported: {list(SUPPORTED_VLM)}")
        _, model_cls = SUPPORTED_VLM[model_type]
        self.vlm = model_cls(vlm_config)

        # Get LLM hidden dim from config
        self.llm_hidden_dim = vlm_config.text_config.hidden_size

        # ====== 3D Gaussian Branch (NEW) ======
        gaussian_dim = 3 + 3 + 4 + 1 + (sh_degree + 1) ** 2 * 3

        self.depth_to_gaussian = DepthToGaussian(
            num_gaussians=num_gaussians,
            sh_degree=sh_degree,
            feat_dim=128,
            predict_uncertainty=True,
        )

        self.gs_encoder = GaussianEncoder(
            gaussian_dim=gaussian_dim,
            num_tokens=num_gs_tokens,
            token_dim=gs_encoder_dim,
        )

        self.gs_projector = GaussianTokenProjector(
            gs_token_dim=gs_encoder_dim,
            llm_hidden_dim=self.llm_hidden_dim,
        )

        # ====== Cross-Modal Fusion (optional, enabled by from_pretrained) ======
        # When active, 3D tokens attend to 2D ViT tokens via cross-attention,
        # aligning them with the ViT feature manifold the LLM understands.
        self.fusion: CrossModalFusion | None = None

        # ====== Learnable 3D position embedding ======
        # Tells the LLM "these tokens come from 3D space"
        self.gs_type_embedding = nn.Parameter(
            torch.randn(1, 1, self.llm_hidden_dim) * 0.02
        )

        # ====== Freeze strategies ======
        # Qwen2.5-VL structure: vlm.model.visual, vlm.model.language_model, vlm.lm_head
        if freeze_vision_encoder:
            for param in self.vlm.model.visual.parameters():
                param.requires_grad = False

        if freeze_llm:
            for param in self.vlm.model.language_model.parameters():
                param.requires_grad = False
            for param in self.vlm.lm_head.parameters():
                param.requires_grad = False

    @property
    def _is_peft_wrapped(self) -> bool:
        """Check if the VLM is wrapped by PEFT (has peft_type attribute)."""
        return hasattr(self.vlm, "peft_type")

    def _get_vlm_inner(self):
        """Get the inner VLM model, handling PEFT wrapping.

        After get_peft_model(), the path changes:
            Without PEFT: self.vlm.model  -> Qwen3VLModel (has .language_model, .visual)
            With PEFT:    self.vlm.model.model -> Qwen3VLModel
        """
        if self._is_peft_wrapped:
            return self.vlm.model.model
        return self.vlm.model

    def _get_embed_tokens(self):
        return self._get_vlm_inner().language_model.embed_tokens

    def _get_language_model(self):
        return self._get_vlm_inner().language_model

    def _get_visual(self):
        return self._get_vlm_inner().visual

    def _get_lm_head(self):
        if self._is_peft_wrapped:
            return self.vlm.model.lm_head
        return self.vlm.lm_head

    @torch.no_grad()
    def extract_vit_tokens(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen ViT to extract per-sample 2D visual tokens.

        Args:
            pixel_values:   preprocessed by AutoProcessor
            image_grid_thw: [B, 3] grid info from AutoProcessor

        Returns:
            vit_tokens: [B, N_2d, llm_hidden_dim]
        """
        B = image_grid_thw.shape[0]
        visual_out = self._get_visual()(pixel_values, grid_thw=image_grid_thw)
        # visual_out is [total_tokens, D] (flat across batch)
        N_2d = visual_out.shape[0] // B
        return visual_out.reshape(B, N_2d, -1)

    def encode_3d(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        vit_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode RGBD input into 3D Gaussian tokens, optionally fused with 2D.

        When ``vit_tokens`` is provided and ``self.fusion`` is active, the raw
        PointNet++ tokens attend to ViT tokens via cross-attention before
        being projected to LLM dim.  Otherwise falls back to the simple
        linear projector (backward-compatible).

        Args:
            rgb: [B, 3, H, W]
            depth: [B, 1, H, W]
            intrinsics: [B, 3, 3]
            vit_tokens: [B, N_2d, llm_hidden_dim]  (optional, from frozen ViT)

        Returns:
            gs_tokens: [B, num_gs_tokens, llm_hidden_dim]
            gaussians: [B, num_gaussians, gaussian_dim]  (for rendering loss)
        """
        # RGBD -> 3D Gaussians
        gaussians = self.depth_to_gaussian(rgb, depth, intrinsics)

        # Strip uncertainty before encoding for LLM
        if self.depth_to_gaussian.predict_uncertainty:
            gaussians_for_encoder = gaussians[..., :-1]
        else:
            gaussians_for_encoder = gaussians

        # 3D Gaussians -> raw tokens
        raw_tokens = self.gs_encoder(gaussians_for_encoder)  # [B, N, gs_encoder_dim]

        # Fuse with 2D if available, else simple projection
        if vit_tokens is not None and self.fusion is not None:
            gs_tokens = self.fusion(raw_tokens, vit_tokens)
        else:
            gs_tokens = self.gs_projector(raw_tokens)

        # Add type embedding
        gs_tokens = gs_tokens + self.gs_type_embedding
        return gs_tokens, gaussians  # return FULL gaussians (with unc) for rendering loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        rgb_for_3d: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Forward pass with dual-stream visual encoding.

        Dual-stream pipeline (when both pixel_values and depth are provided):
          1. ViT (frozen) encodes RGB -> 2D tokens, injected at <image_pad>
          2. 3D branch encodes RGBD -> raw 3D tokens
          3. CrossModalFusion: 3D queries attend to 2D keys/values -> fused tokens
          4. Fused 3D tokens prepended before the full sequence
          5. LLM sees: [fused_3d | 2d_visual + text] -> output

        Backward-compatible: works with pixel_values=None (text-only) or
        depth=None (2D-only) as before.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Step 1: Get text embeddings
        inputs_embeds = self._get_embed_tokens()(input_ids)

        # Step 2: Run ViT if pixel_values provided (native 2D path)
        vit_tokens = None
        if pixel_values is not None and image_grid_thw is not None:
            visual_outputs = self._get_visual()(
                pixel_values, grid_thw=image_grid_thw,
            )
            # Inject 2D tokens at <image_pad> positions (standard VLM path)
            image_token_id = self.vlm.config.image_token_id
            image_mask = input_ids == image_token_id
            if image_mask.any():
                inputs_embeds[image_mask] = visual_outputs.to(inputs_embeds.dtype)

            # Keep per-sample 2D tokens for fusion
            N_2d = visual_outputs.shape[0] // B
            vit_tokens = visual_outputs.reshape(B, N_2d, -1).detach()

        # Step 3: Encode 3D Gaussian tokens with optional fusion
        gaussians = None
        if depth is not None and intrinsics is not None and rgb_for_3d is not None:
            gs_tokens, gaussians = self.encode_3d(
                rgb_for_3d, depth, intrinsics, vit_tokens=vit_tokens,
            )

            # Prepend fused 3D tokens before the full sequence
            inputs_embeds = torch.cat([gs_tokens, inputs_embeds], dim=1)

            gs_attention = torch.ones(
                B, self.num_gs_tokens, dtype=attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([gs_attention, attention_mask], dim=1)

            if labels is not None:
                gs_labels = torch.full(
                    (B, self.num_gs_tokens), -100, dtype=labels.dtype, device=device,
                )
                labels = torch.cat([gs_labels, labels], dim=1)

        # Step 4: Run LLM decoder
        outputs = self._get_language_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state

        # Step 5: Compute LM loss if labels provided
        logits = self._get_lm_head()(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Ensure labels are on same device as logits (multi-GPU)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
            "gaussians": gaussians,
            "vit_tokens": vit_tokens,
        }

    @torch.no_grad()
    def generate_with_3d(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        rgb_for_3d: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate text with 3D Gaussian-enhanced understanding.

        Handles 3D token injection then delegates to the VLM's
        autoregressive generation.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Get base embeddings
        inputs_embeds = self._get_embed_tokens()(input_ids)

        # Visual encoding (2D ViT path)
        vit_tokens = None
        if pixel_values is not None and image_grid_thw is not None:
            visual_outputs = self._get_visual()(
                pixel_values, grid_thw=image_grid_thw,
            )
            image_token_id = self.vlm.config.image_token_id
            image_mask = input_ids == image_token_id
            if image_mask.any():
                inputs_embeds[image_mask] = visual_outputs.to(inputs_embeds.dtype)
            # Keep 2D tokens for cross-modal fusion
            N_2d = visual_outputs.shape[0] // B
            vit_tokens = visual_outputs.reshape(B, N_2d, -1).detach()

        # 3D token injection (with optional cross-modal fusion)
        if depth is not None and intrinsics is not None and rgb_for_3d is not None:
            gs_tokens, _ = self.encode_3d(
                rgb_for_3d, depth, intrinsics, vit_tokens=vit_tokens,
            )
            inputs_embeds = torch.cat([gs_tokens, inputs_embeds], dim=1)

            gs_attention = torch.ones(
                B, self.num_gs_tokens, dtype=attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([gs_attention, attention_mask], dim=1)

        # Hoist EOS config outside the generation loop
        eos_id = self.vlm.config.text_config.eos_token_id
        if isinstance(eos_id, int):
            eos_id = [eos_id]
        eos_set = set(eos_id)

        # Pre-allocate output buffer instead of quadratic torch.cat
        initial_len = input_ids.shape[1]
        generated = torch.zeros(
            B, initial_len + max_new_tokens, dtype=input_ids.dtype, device=device,
        )
        generated[:, :initial_len] = input_ids
        gen_len = initial_len

        # Pre-allocate attention mask buffer
        total_attn_len = attention_mask.shape[1] + max_new_tokens
        full_attn = torch.ones(B, total_attn_len, dtype=attention_mask.dtype, device=device)
        attn_len = attention_mask.shape[1]
        full_attn[:, :attn_len] = attention_mask

        # Autoregressive generation
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                out = self._get_language_model()(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attn[:, :attn_len],
                )
            else:
                out = self._get_language_model()(
                    inputs_embeds=next_embeds,
                    attention_mask=full_attn[:, :attn_len],
                    past_key_values=past_key_values,
                )

            past_key_values = out.past_key_values
            logits = self._get_lm_head()(out.last_hidden_state[:, -1:, :])

            if do_sample and temperature > 0:
                probs = F.softmax(logits[:, 0, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)

            # Move to input device (lm_head may be on a different GPU)
            next_token = next_token.to(device)
            generated[:, gen_len] = next_token[:, 0]
            gen_len += 1
            attn_len += 1

            # Check for EOS (B=1 fast path)
            if B == 1:
                if next_token.item() in eos_set:
                    break
            else:
                if all(generated[b, gen_len - 1].item() in eos_set for b in range(B)):
                    break

            # Prepare next step
            next_embeds = self._get_embed_tokens()(next_token)

        return generated[:, :gen_len]

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_gaussians: int = 2048,
        sh_degree: int = 2,
        num_gs_tokens: int = 64,
        gs_encoder_dim: int = 512,
        freeze_vision_encoder: bool = True,
        freeze_llm: bool = False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ) -> "RoboBrain3DGS_VLM":
        """Load a pretrained RoboBrain VLM and attach 3D Gaussian modules.

        Args:
            model_path: Path to pretrained model (local dir or HF repo id)
            num_gaussians: Number of 3D Gaussians to generate per frame
            sh_degree: Spherical harmonics degree for appearance
            num_gs_tokens: Number of tokens output by GS Encoder
            gs_encoder_dim: Internal dim of GS Encoder
            freeze_vision_encoder: Whether to freeze the 2D ViT
            freeze_llm: Whether to freeze the LLM decoder
            torch_dtype: Model dtype (bfloat16 recommended for 8B)
            device_map: Device mapping strategy. Use {"": 0} for single-GPU
                training, None for DeepSpeed (CPU load), "auto" for inference.

        Returns:
            RoboBrain3DGS_VLM with pretrained VLM weights loaded
        """
        print(f"Loading pretrained VLM from {model_path} ...")

        # Load pretrained VLM directly with proper device placement
        print("Loading pretrained VLM backbone...")
        pretrained_vlm = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
        )
        config = pretrained_vlm.config

        # Build the wrapper model (3D branch only, skip creating a new VLM)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.num_gs_tokens = num_gs_tokens

        # Use the pretrained VLM directly (already on GPU)
        model.vlm = pretrained_vlm

        # Get LLM hidden dim
        model.llm_hidden_dim = config.text_config.hidden_size

        # Build 3D branch
        gaussian_dim = 3 + 3 + 4 + 1 + (sh_degree + 1) ** 2 * 3

        model.depth_to_gaussian = DepthToGaussian(
            num_gaussians=num_gaussians,
            sh_degree=sh_degree,
            feat_dim=128,
            predict_uncertainty=True,
        )
        model.gs_encoder = GaussianEncoder(
            gaussian_dim=gaussian_dim,
            num_tokens=num_gs_tokens,
            token_dim=gs_encoder_dim,
        )
        model.gs_projector = GaussianTokenProjector(
            gs_token_dim=gs_encoder_dim,
            llm_hidden_dim=model.llm_hidden_dim,
        )
        model.gs_type_embedding = nn.Parameter(
            torch.randn(1, 1, model.llm_hidden_dim) * 0.02
        )

        # Cross-modal fusion (not enabled by default in from_pretrained)
        model.fusion = None

        # Move 3D branch to same device as embed_tokens
        embed_device = model.vlm.model.language_model.embed_tokens.weight.device
        model.depth_to_gaussian = model.depth_to_gaussian.to(device=embed_device, dtype=torch_dtype)
        model.gs_encoder = model.gs_encoder.to(device=embed_device, dtype=torch_dtype)
        model.gs_projector = model.gs_projector.to(device=embed_device, dtype=torch_dtype)
        model.gs_type_embedding = nn.Parameter(
            model.gs_type_embedding.to(device=embed_device, dtype=torch_dtype)
        )

        # Freeze strategies
        if freeze_vision_encoder:
            for param in model.vlm.model.visual.parameters():
                param.requires_grad = False
        if freeze_llm:
            for param in model.vlm.model.language_model.parameters():
                param.requires_grad = False
            for param in model.vlm.lm_head.parameters():
                param.requires_grad = False

        print(f"Model ready. 3D branch on {embed_device}, LLM hidden_dim={model.llm_hidden_dim}")
        return model


def create_tiny_vlm_config() -> Qwen2_5_VLConfig:
    """Create a tiny Qwen2.5-VL config for validation.

    This produces a ~15M parameter model (instead of 2B-72B)
    that has the exact same architecture for integration testing.
    """
    config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 2,           # 2 ViT layers (vs 32)
            "hidden_size": 256,   # (vs 1280-3584)
            "hidden_act": "silu",
            "intermediate_size": 512,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 112,
            "fullatt_block_indexes": [1],
            "out_hidden_size": 256,
        },
        text_config={
            "vocab_size": 152064,
            "hidden_size": 256,      # (vs 1536-8192)
            "intermediate_size": 512,
            "num_hidden_layers": 2,  # (vs 28-80)
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-5,
            "use_cache": True,
            "use_sliding_window": False,
            "max_window_layers": 2,
            "attention_dropout": 0.0,
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "rope_type": "default",
                "mrope_section": [8, 12, 12],
            },
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    return config
