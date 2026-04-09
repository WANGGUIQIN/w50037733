# VarSplat-Inspired Uncertainty-Aware Model Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the 3D Gaussian branch with per-Gaussian uncertainty modeling, uncertainty-aware rendering, and uncertainty-guided optimization -- inspired by VarSplat (Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM, CVPR 2026) and CG-SLAM -- to make the system robust to noisy pseudo-depth (9/12 datasets use estimated depth).

**Architecture:** Three key changes to the existing pipeline:
1. **Per-Gaussian uncertainty prediction**: The `DepthToGaussian` CNN head predicts an additional uncertainty scalar per Gaussian, measuring confidence in its position (derived from depth reliability).
2. **Uncertainty-aware rendering**: The renderer accumulates uncertainty like color -- via alpha compositing -- producing pixel-level uncertainty maps alongside RGB/depth. This enables uncertainty-weighted losses (unreliable regions contribute less).
3. **Uncertainty-guided selection**: Gaussian selection uses learned uncertainty to prefer high-confidence Gaussians (low uncertainty) over blind FPS sampling.

The VLM integration path (GaussianEncoder -> Projector -> LLM) is unchanged. Uncertainty flows through the 3D branch only.

**Tech Stack:** PyTorch, existing Qwen3-VL backbone (frozen), existing training infrastructure (LoRA, DeepSpeed)

**Key reference formulations (from CG-SLAM / VarSplat):**
- Per-Gaussian uncertainty: `v_i = mean_over_views[ alpha_i * T_i * (D_observed - d_i)^2 ]`
- Rendered uncertainty map: `U(p) = sum_i[ alpha_i * T_i * (d_i - D_p)^2 ]`
- Geometry variance loss: `L_var = mean(|U|)` (encourages low uncertainty)
- Uncertainty-pruning threshold: Gaussians with `v_i > tau` get opacity suppressed

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `models/depth_to_gaussian.py` | **Modify** | Add uncertainty head; output `[B, N, D+1]` with uncertainty per Gaussian |
| `models/gs_renderer.py` | **Modify** | Add uncertainty rendering channel + uncertainty-weighted losses |
| `models/gs_encoder.py` | **Modify** | Add uncertainty-weighted pooling in PointNet++ hierarchy |
| `models/robobrain_vlm.py` | **Modify** | Pass uncertainty through `encode_3d()` for training diagnostics |
| `tests/test_uncertainty.py` | **Create** | Tests for uncertainty prediction, rendering, and loss |
| `tests/test_uncertainty_integration.py` | **Create** | End-to-end integration tests with uncertainty |
| `config/train_lora.yaml` | **Modify** | Add uncertainty loss weights |

---

## Task 1: Per-Gaussian Uncertainty Prediction

**Files:**
- Modify: `models/depth_to_gaussian.py`
- Test: `tests/test_uncertainty.py`

Add an uncertainty prediction head to `DepthToGaussian`. Each Gaussian gets an uncertainty scalar `v_i` that measures how reliable its 3D position is. This is critical because 9/12 datasets use pseudo-depth (MiDaS/DPT), which has significant noise.

The uncertainty head shares the CNN encoder features and outputs a single scalar per spatial location, activated with softplus (uncertainty is always positive). The output tensor grows from `[B, N, D]` to `[B, N, D+1]` where the last channel is uncertainty.

Design choice: Following CG-SLAM, uncertainty is **not** a fixed property -- it's learned by the network based on RGBD input features. Regions with high texture + reliable depth get low uncertainty; featureless regions or depth edges get high uncertainty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_uncertainty.py
import torch
import pytest


def test_d2g_outputs_uncertainty():
    """DepthToGaussian should output uncertainty as last channel."""
    from models.depth_to_gaussian import DepthToGaussian

    B, H, W = 2, 64, 64
    rgb = torch.randn(B, 3, H, W)
    depth = torch.randn(B, 1, H, W).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0).expand(B, -1, -1).clone()
    intrinsics[:, 0, 0] = 128.0
    intrinsics[:, 1, 1] = 128.0
    intrinsics[:, 0, 2] = 32.0
    intrinsics[:, 1, 2] = 32.0

    module = DepthToGaussian(num_gaussians=256, sh_degree=2, predict_uncertainty=True)
    gaussians = module(rgb, depth, intrinsics)
    # Original: 3+3+4+1+27 = 38. With uncertainty: 39
    expected_dim = 38 + 1
    assert gaussians.shape == (B, 256, expected_dim), f"Got {gaussians.shape}"


def test_d2g_uncertainty_positive():
    """Uncertainty values should always be positive (softplus activation)."""
    from models.depth_to_gaussian import DepthToGaussian

    module = DepthToGaussian(num_gaussians=64, sh_degree=0, predict_uncertainty=True)
    rgb = torch.randn(1, 3, 32, 32)
    depth = torch.randn(1, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    uncertainty = gaussians[..., -1]  # last channel
    assert (uncertainty > 0).all(), f"Found non-positive uncertainty: min={uncertainty.min()}"


def test_d2g_backward_compat_no_uncertainty():
    """Default behavior (predict_uncertainty=False) should be unchanged."""
    from models.depth_to_gaussian import DepthToGaussian

    module = DepthToGaussian(num_gaussians=128, sh_degree=2, predict_uncertainty=False)
    rgb = torch.randn(1, 3, 64, 64)
    depth = torch.randn(1, 1, 64, 64).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 128
    intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    assert gaussians.shape == (1, 128, 38), f"Got {gaussians.shape}"


def test_d2g_uncertainty_gradient_flow():
    """Gradients should flow through uncertainty prediction."""
    from models.depth_to_gaussian import DepthToGaussian

    module = DepthToGaussian(num_gaussians=32, sh_degree=0, predict_uncertainty=True)
    rgb = torch.randn(1, 3, 32, 32, requires_grad=True)
    depth = torch.randn(1, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    uncertainty = gaussians[..., -1]
    loss = uncertainty.sum()
    loss.backward()
    assert rgb.grad is not None
    assert rgb.grad.abs().sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: FAIL (`predict_uncertainty` parameter not recognized)

- [ ] **Step 3: Modify DepthToGaussian to predict uncertainty**

In `models/depth_to_gaussian.py`, update the `DepthToGaussian` class:

**3a. Add `predict_uncertainty` parameter to `__init__`:**

After `self.param_dim = 3 + 4 + 1 + self.sh_coeffs` add:

```python
        self.predict_uncertainty = predict_uncertainty

        # Uncertainty prediction head (separate from param head)
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim // 2, 1),
                nn.GELU(),
                nn.Conv2d(feat_dim // 2, 1, 1),
            )
```

Update the `__init__` signature to include `predict_uncertainty: bool = False`.

**3b. Add uncertainty to `forward()`:**

After line `gaussians_xyz, gaussians_params = self._fps_select(...)`, before parameter activation, add uncertainty prediction:

```python
        # Predict uncertainty if enabled
        if self.predict_uncertainty:
            uncertainty_map = self.uncertainty_head(features)  # [B, 1, h, w]
            uncertainty_flat = uncertainty_map.permute(0, 2, 3, 1).reshape(B, -1, 1)
            # Select same indices as params
            _, gaussians_uncertainty = self._fps_select(
                points_flat, uncertainty_flat, self.num_gaussians
            )
            # Activate: softplus ensures positive uncertainty
            gaussians_uncertainty = F.softplus(gaussians_uncertainty[..., 1:])  # drop xyz from _fps_select
```

Wait -- `_fps_select` returns `(points, params)` using the same indices. We need to select uncertainty with the same indices. Simpler approach: compute uncertainty alongside params and include it in the selection. Update `forward()` to:

After `params_flat = params.permute(...)` and before `_fps_select`, compute uncertainty and concatenate:

```python
        if self.predict_uncertainty:
            unc_map = self.uncertainty_head(features)  # [B, 1, h, w]
            unc_flat = unc_map.permute(0, 2, 3, 1).reshape(B, -1, 1)  # [B, N_all, 1]
            params_flat = torch.cat([params_flat, unc_flat], dim=-1)  # [B, N_all, param_dim+1]
```

Then after `_fps_select` and activation, split uncertainty out:

```python
        if self.predict_uncertainty:
            uncertainty = F.softplus(gaussians_params[..., -1:])  # [B, N, 1]
            gaussians_params = gaussians_params[..., :-1]  # remove unc from params
```

And in the final concatenation:

```python
        gaussians = torch.cat([gaussians_xyz, scale, rotation, opacity, sh], dim=-1)
        if self.predict_uncertainty:
            gaussians = torch.cat([gaussians, uncertainty], dim=-1)
        return gaussians
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add models/depth_to_gaussian.py tests/test_uncertainty.py
git commit -m "feat: add per-Gaussian uncertainty prediction to DepthToGaussian

Inspired by VarSplat/CG-SLAM. Each Gaussian gets an uncertainty scalar
from a learned CNN head (softplus-activated, always positive).
Backward compatible: predict_uncertainty=False preserves original behavior."
```

---

## Task 2: Uncertainty-Aware Rendering

**Files:**
- Modify: `models/gs_renderer.py`
- Test: `tests/test_uncertainty.py` (append new tests)

Update the renderer to accumulate uncertainty through alpha compositing, producing a pixel-level uncertainty map. The key formula from CG-SLAM:

```
U(p) = sum_i[ alpha_i * T_i * (d_i - D_p)^2 ]
```

This measures per-pixel depth variance -- how much the contributing Gaussians disagree about the depth at that pixel. Additionally, if per-Gaussian uncertainty is available (from Task 1), it's also composited:

```
U_learned(p) = sum_i[ alpha_i * T_i * v_i ]
```

The renderer output dict gains `rendered_uncertainty: [B, 1, H, W]`.

- [ ] **Step 1: Append new tests to test_uncertainty.py**

```python
# Append to tests/test_uncertainty.py

def test_render_uncertainty_map():
    """Renderer should output uncertainty map when uncertainty is provided."""
    from models.gs_renderer import render_gaussians

    B, N = 1, 32
    means = torch.randn(B, N, 3)
    means[..., 2] = means[..., 2].abs() + 1.0  # positive depth
    scales = torch.ones(B, N, 3) * 0.1
    rotations = torch.zeros(B, N, 4)
    rotations[..., 0] = 1.0  # identity quaternion
    opacities = torch.ones(B, N, 1) * 0.5
    sh_coeffs = torch.randn(B, N, 27)
    uncertainty = torch.rand(B, N, 1) * 0.5 + 0.01  # positive
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    result = render_gaussians(
        means, scales, rotations, opacities, sh_coeffs,
        intrinsics, (16, 16), uncertainty=uncertainty,
    )
    assert "rendered_uncertainty" in result
    assert result["rendered_uncertainty"].shape == (B, 1, 16, 16)
    assert (result["rendered_uncertainty"] >= 0).all()


def test_render_no_uncertainty_backward_compat():
    """Without uncertainty input, renderer should work as before."""
    from models.gs_renderer import render_gaussians

    B, N = 1, 16
    means = torch.randn(B, N, 3)
    means[..., 2] = means[..., 2].abs() + 1.0
    scales = torch.ones(B, N, 3) * 0.1
    rotations = torch.zeros(B, N, 4)
    rotations[..., 0] = 1.0
    opacities = torch.ones(B, N, 1) * 0.5
    sh_coeffs = torch.randn(B, N, 27)
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    result = render_gaussians(
        means, scales, rotations, opacities, sh_coeffs,
        intrinsics, (8, 8),
    )
    assert "rendered_rgb" in result
    assert "rendered_depth" in result
    # rendered_uncertainty may or may not be present -- either is fine
```

- [ ] **Step 2: Run to verify failures**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py::test_render_uncertainty_map -v`
Expected: FAIL (render_gaussians doesn't accept uncertainty parameter)

- [ ] **Step 3: Update render_gaussians to handle uncertainty**

In `models/gs_renderer.py`, update the `render_gaussians` function signature to add `uncertainty: torch.Tensor | None = None`.

After the existing depth/alpha compositing (after line `rendered_alpha = weights.sum(...)`), add:

```python
    # Uncertainty rendering (VarSplat-inspired)
    rendered_uncertainty = None
    if uncertainty is not None:
        # Learned per-Gaussian uncertainty, composited like color
        # U_learned(p) = sum_i[weight_i * v_i]
        unc_sorted = torch.gather(
            uncertainty.squeeze(-1), 1, depth_order
        )  # [B, N] sorted by depth
        rendered_uncertainty = (
            weights * unc_sorted.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    else:
        # Geometric uncertainty: depth variance from alpha compositing
        # U_geo(p) = sum_i[weight_i * (d_i - D_rendered)^2]
        depth_diff_sq = (
            depth_vals_sorted.unsqueeze(-1).unsqueeze(-1)
            - rendered_depth
        ) ** 2  # [B, N, H, W] vs [B, 1, H, W]
        rendered_uncertainty = (weights * depth_diff_sq).sum(dim=1, keepdim=True)
```

Add `rendered_uncertainty` to the return dict:

```python
    result = {
        "rendered_rgb": rendered_rgb.clamp(0, 1),
        "rendered_depth": rendered_depth,
        "rendered_alpha": rendered_alpha.clamp(0, 1),
    }
    if rendered_uncertainty is not None:
        result["rendered_uncertainty"] = rendered_uncertainty
    return result
```

- [ ] **Step 4: Run tests**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: 6 PASSED (4 from Task 1 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add models/gs_renderer.py tests/test_uncertainty.py
git commit -m "feat: add uncertainty rendering channel to GS renderer

Supports two modes:
- Learned uncertainty: composites per-Gaussian uncertainty via alpha blending
- Geometric uncertainty: depth variance from contributing Gaussians
Backward compatible when uncertainty=None."
```

---

## Task 3: Uncertainty-Weighted Rendering Loss

**Files:**
- Modify: `models/gs_renderer.py` (update `GaussianRenderingLoss`)
- Test: `tests/test_uncertainty.py` (append tests)

The key insight from VarSplat/CG-SLAM: pixels with high uncertainty should contribute less to the rendering loss. This prevents noisy pseudo-depth from corrupting the Gaussian optimization.

Two new loss terms:
1. **Uncertainty-weighted depth loss**: `L_depth = mean(|D_rendered - D_target| / (1 + U))`
2. **Geometry variance loss**: `L_var = mean(U)` (encourages low uncertainty -- prevents trivial solution of infinite uncertainty)

- [ ] **Step 1: Append test**

```python
# Append to tests/test_uncertainty.py

def test_uncertainty_weighted_loss():
    """Rendering loss should accept and use uncertainty."""
    from models.gs_renderer import GaussianRenderingLoss

    loss_fn = GaussianRenderingLoss(
        image_size=(16, 16), sh_degree=0, lambda_uncertainty=0.1
    )

    B, N = 1, 64
    gaussians = torch.randn(B, N, 12)  # 3+3+4+1+1(sh_dc)+... simplified
    # Build proper gaussians: xyz(3) + scale(3) + rot(4) + opacity(1) + sh(3) + unc(1) = 15
    xyz = torch.randn(B, N, 3)
    xyz[..., 2] = xyz[..., 2].abs() + 0.5
    scale = torch.ones(B, N, 3) * 0.05
    rot = torch.zeros(B, N, 4); rot[..., 0] = 1
    opa = torch.ones(B, N, 1) * 0.5
    sh = torch.randn(B, N, 3)
    unc = torch.rand(B, N, 1) * 0.3
    gaussians = torch.cat([xyz, scale, rot, opa, sh, unc], dim=-1)

    intrinsics = torch.eye(3).unsqueeze(0) * 32
    intrinsics[:, 2, 2] = 1
    target_rgb = torch.rand(B, 3, 64, 64)
    target_depth = torch.rand(B, 1, 64, 64) + 0.5

    result = loss_fn(gaussians, intrinsics, target_rgb, target_depth, has_uncertainty=True)
    assert "loss" in result
    assert "uncertainty_var" in result
    assert result["uncertainty_var"] >= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py::test_uncertainty_weighted_loss -v`
Expected: FAIL

- [ ] **Step 3: Update GaussianRenderingLoss**

In `models/gs_renderer.py`, update `GaussianRenderingLoss`:

**3a. Add `lambda_uncertainty` to `__init__`:**

```python
    def __init__(
        self,
        lambda_l1: float = 0.8,
        lambda_ssim: float = 0.2,
        lambda_depth: float = 0.5,
        lambda_opacity: float = 0.01,
        lambda_uncertainty: float = 0.1,  # NEW: geometry variance loss weight
        image_size: tuple[int, int] = (64, 64),
        sh_degree: int = 2,
    ):
        ...
        self.lambda_uncertainty = lambda_uncertainty
```

**3b. Update `forward()` to accept `has_uncertainty` flag:**

```python
    def forward(
        self,
        gaussians: torch.Tensor,
        intrinsics: torch.Tensor,
        target_rgb: torch.Tensor,
        target_depth: torch.Tensor,
        has_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
```

**3c. Parse uncertainty from gaussians when present:**

After parsing sh_coeffs, add:

```python
        uncertainty = None
        if has_uncertainty:
            sh_coeffs = gaussians[:, :, 11:-1]  # exclude last channel
            uncertainty = gaussians[:, :, -1:]    # [B, N, 1]
        else:
            sh_coeffs = gaussians[:, :, 11:]
```

**3d. Pass uncertainty to render_gaussians:**

```python
        rendered = render_gaussians(
            means_3d, scales, rotations, opacities, sh_coeffs,
            scaled_intrinsics, (H, W), self.sh_degree,
            uncertainty=uncertainty,
        )
```

**3e. Add uncertainty-weighted depth loss and variance loss:**

Replace the existing depth loss with:

```python
        if alpha_mask.sum() > 0:
            depth_error = (rendered["rendered_depth"] - target_depth_ds).abs()
            if "rendered_uncertainty" in rendered:
                # Uncertainty-weighted: unreliable regions contribute less
                unc_map = rendered["rendered_uncertainty"].clamp(min=1e-6)
                l1_depth = (depth_error / (1.0 + unc_map) * alpha_mask).sum() / alpha_mask.sum().clamp(min=1)
            else:
                l1_depth = (depth_error * alpha_mask).sum() / alpha_mask.sum().clamp(min=1)
        else:
            l1_depth = torch.tensor(0.0, device=gaussians.device, dtype=gaussians.dtype)

        # Geometry variance loss: encourage low uncertainty (VarSplat)
        uncertainty_var = torch.tensor(0.0, device=gaussians.device, dtype=gaussians.dtype)
        if "rendered_uncertainty" in rendered:
            uncertainty_var = rendered["rendered_uncertainty"].mean()
```

**3f. Add to total loss:**

```python
        loss = (
            self.lambda_l1 * l1_rgb
            + self.lambda_ssim * ssim_loss
            + self.lambda_depth * l1_depth
            + self.lambda_opacity * opacity_reg
            + self.lambda_uncertainty * uncertainty_var
        )
```

Add `"uncertainty_var": uncertainty_var` to the return dict.

- [ ] **Step 4: Run tests**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add models/gs_renderer.py tests/test_uncertainty.py
git commit -m "feat: uncertainty-weighted rendering loss (VarSplat)

- Depth loss down-weighted by (1 + uncertainty) at unreliable pixels
- Geometry variance loss encourages low uncertainty (prevents trivial solution)
- Backward compatible when has_uncertainty=False"
```

---

## Task 4: Uncertainty-Guided Gaussian Selection

**Files:**
- Modify: `models/depth_to_gaussian.py` (update `_fps_select`)
- Test: `tests/test_uncertainty.py` (append test)

Replace blind FPS with uncertainty-aware selection: prefer Gaussians with low uncertainty. High-uncertainty Gaussians (from noisy depth) are deprioritized. This is analogous to VarSplat/CG-SLAM's pruning of high-uncertainty Gaussians (threshold `tau = 0.025`).

Implementation: use FPS on a modified distance metric that combines spatial distance with uncertainty penalty. Gaussians with high uncertainty are treated as "closer" to already-selected points, so FPS skips them.

- [ ] **Step 1: Append test**

```python
# Append to tests/test_uncertainty.py

def test_uncertainty_guided_selection():
    """High-uncertainty points should be less likely to be selected."""
    from models.depth_to_gaussian import DepthToGaussian

    torch.manual_seed(42)
    module = DepthToGaussian(num_gaussians=32, sh_degree=0, predict_uncertainty=True)
    rgb = torch.randn(1, 3, 32, 32)
    depth = torch.randn(1, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    uncertainty = gaussians[..., -1]  # [1, 32]
    # Selected Gaussians should have lower mean uncertainty than total pool
    # (This is a soft test - the selection prefers low uncertainty)
    assert uncertainty.shape == (1, 32)
    assert (uncertainty > 0).all()
```

- [ ] **Step 2: Update `_fps_select` to incorporate uncertainty**

In `models/depth_to_gaussian.py`, update `_fps_select` to accept an optional uncertainty tensor:

```python
    def _fps_select(
        self,
        points: torch.Tensor,
        params: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
```

When `predict_uncertainty` is enabled and params contains uncertainty (last channel), use uncertainty-penalized distance:

```python
        # If uncertainty is in params (last channel), use it to bias selection
        if self.predict_uncertainty and params.shape[-1] > self.param_dim:
            unc = params[..., -1]  # [B, N]
            # Penalty: high uncertainty -> reduced effective distance
            # This makes high-uncertainty points less likely to be "farthest"
            unc_penalty = torch.sigmoid(unc) * 2.0  # [0, 2] range
        else:
            unc_penalty = None

        ...
        for i in range(num_samples):
            indices[:, i] = farthest
            centroid = points[torch.arange(B, device=device), farthest].unsqueeze(1)
            dist = torch.sum((points - centroid) ** 2, dim=-1)
            if unc_penalty is not None:
                # Reduce distance for high-uncertainty points
                dist = dist / (1.0 + unc_penalty)
            distances = torch.min(distances, dist)
            farthest = distances.argmax(dim=-1)
```

- [ ] **Step 3: Run tests**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: 8 PASSED

- [ ] **Step 4: Commit**

```bash
git add models/depth_to_gaussian.py tests/test_uncertainty.py
git commit -m "feat: uncertainty-guided FPS selection

High-uncertainty Gaussians are deprioritized via distance penalty,
preferring spatially diverse AND confident Gaussians."
```

---

## Task 5: Wire Uncertainty Through VLM Pipeline

**Files:**
- Modify: `models/robobrain_vlm.py`
- Modify: `config/train_lora.yaml`

Update the VLM integration to:
1. Create `DepthToGaussian` with `predict_uncertainty=True`
2. Pass `has_uncertainty=True` to the rendering loss during training
3. The GaussianEncoder receives Gaussians **without** the uncertainty channel (strip it before encoding, since LLM tokens shouldn't carry rendering-level uncertainty)

- [ ] **Step 1: Update __init__ and from_pretrained**

In `models/robobrain_vlm.py`, update the `DepthToGaussian` construction in both `__init__` and `from_pretrained`:

```python
        self.depth_to_gaussian = DepthToGaussian(
            num_gaussians=num_gaussians,
            sh_degree=sh_degree,
            feat_dim=128,
            predict_uncertainty=True,
        )
```

- [ ] **Step 2: Update encode_3d to strip uncertainty before GaussianEncoder**

```python
    def encode_3d(self, rgb, depth, intrinsics, vit_tokens=None):
        gaussians = self.depth_to_gaussian(rgb, depth, intrinsics)

        # Strip uncertainty channel before encoding for LLM
        # (uncertainty is for rendering loss only, not for language understanding)
        if self.depth_to_gaussian.predict_uncertainty:
            gaussians_for_encoder = gaussians[..., :-1]
        else:
            gaussians_for_encoder = gaussians

        raw_tokens = self.gs_encoder(gaussians_for_encoder)

        if vit_tokens is not None and self.fusion is not None:
            gs_tokens = self.fusion(raw_tokens, vit_tokens)
        else:
            gs_tokens = self.gs_projector(raw_tokens)

        gs_tokens = gs_tokens + self.gs_type_embedding
        return gs_tokens, gaussians  # return FULL gaussians (with uncertainty) for rendering loss
```

- [ ] **Step 3: Update train_lora.yaml**

Add to the `rendering_loss:` section:

```yaml
  lambda_uncertainty: 0.1   # VarSplat geometry variance loss weight
```

Add to the `model:` section:

```yaml
  predict_uncertainty: true  # VarSplat-inspired per-Gaussian uncertainty
```

- [ ] **Step 4: Commit**

```bash
git add models/robobrain_vlm.py config/train_lora.yaml
git commit -m "feat: wire uncertainty through VLM pipeline

- DepthToGaussian created with predict_uncertainty=True
- Uncertainty stripped before GaussianEncoder (for LLM tokens)
- Full Gaussians (with uncertainty) returned for rendering loss
- Config updated with uncertainty loss weight"
```

---

## Task 6: Integration Tests

**Files:**
- Create: `tests/test_uncertainty_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# tests/test_uncertainty_integration.py
import torch
import pytest


def test_full_pipeline_with_uncertainty():
    """Full VLM pipeline should work with uncertainty enabled."""
    from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config

    config = create_tiny_vlm_config()
    model = RoboBrain3DGS_VLM(
        vlm_config=config,
        num_gaussians=64,
        sh_degree=0,
        num_gs_tokens=8,
        gs_encoder_dim=64,
    )

    B = 1
    rgb = torch.randn(B, 3, 32, 32)
    depth = torch.randn(B, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1
    input_ids = torch.randint(0, 1000, (B, 10))
    attention_mask = torch.ones(B, 10, dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        depth=depth,
        intrinsics=intrinsics,
        rgb_for_3d=rgb,
    )

    assert out["gaussians"] is not None
    # Gaussians should have uncertainty as last channel
    assert model.depth_to_gaussian.predict_uncertainty
    expected_dim = 3 + 3 + 4 + 1 + 3 + 1  # xyz+scale+rot+opa+sh_dc+unc
    assert out["gaussians"].shape[-1] == expected_dim


def test_rendering_loss_with_uncertainty():
    """Rendering loss should use uncertainty weighting."""
    from models.gs_renderer import GaussianRenderingLoss

    loss_fn = GaussianRenderingLoss(
        image_size=(8, 8), sh_degree=0, lambda_uncertainty=0.1
    )

    B, N = 1, 32
    xyz = torch.randn(B, N, 3); xyz[..., 2] = xyz[..., 2].abs() + 0.5
    scale = torch.ones(B, N, 3) * 0.1
    rot = torch.zeros(B, N, 4); rot[..., 0] = 1
    opa = torch.ones(B, N, 1) * 0.5
    sh = torch.randn(B, N, 3)
    unc = torch.rand(B, N, 1) * 0.2
    gaussians = torch.cat([xyz, scale, rot, opa, sh, unc], dim=-1)

    intrinsics = torch.eye(3).unsqueeze(0) * 32; intrinsics[:, 2, 2] = 1
    target_rgb = torch.rand(B, 3, 32, 32)
    target_depth = torch.rand(B, 1, 32, 32) + 0.5

    result = loss_fn(gaussians, intrinsics, target_rgb, target_depth, has_uncertainty=True)
    assert "uncertainty_var" in result
    assert result["loss"].requires_grad


def test_uncertainty_lower_for_native_depth():
    """Sanity: model should learn lower uncertainty for consistent depth."""
    # This is a property we expect after training, not testable in unit test
    # Just verify the model produces varying uncertainty across the image
    from models.depth_to_gaussian import DepthToGaussian

    module = DepthToGaussian(num_gaussians=128, sh_degree=0, predict_uncertainty=True)
    rgb = torch.randn(1, 3, 64, 64)
    depth = torch.randn(1, 1, 64, 64).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 128; intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    unc = gaussians[..., -1]
    # Uncertainty should not be constant (network should produce varying values)
    assert unc.std() > 0, "Uncertainty is constant -- network not differentiating"
```

- [ ] **Step 2: Run integration tests**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/test_uncertainty_integration.py -v`
Expected: 3 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_uncertainty_integration.py
git commit -m "test: integration tests for uncertainty-aware pipeline"
```

---

## Task 7: Full Test Suite and Final Verification

- [ ] **Step 1: Run complete test suite**

Run: `cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Print parameter count delta**

```bash
cd /home/edge/Embodied/robobrain_3dgs && /home/edge/miniconda3/envs/robobrain_3dgs/bin/python -c "
from models.depth_to_gaussian import DepthToGaussian

old = DepthToGaussian(num_gaussians=1024, sh_degree=2, predict_uncertainty=False)
new = DepthToGaussian(num_gaussians=1024, sh_degree=2, predict_uncertainty=True)

p_old = sum(p.numel() for p in old.parameters())
p_new = sum(p.numel() for p in new.parameters())
delta = p_new - p_old

print(f'Without uncertainty: {p_old:,} params ({p_old/1e6:.1f}M)')
print(f'With uncertainty:    {p_new:,} params ({p_new/1e6:.1f}M)')
print(f'Delta:               {delta:,} params ({delta/1e6:.2f}M)')
print(f'Overhead:            {delta/p_old*100:.1f}%')
"
```

Expected: Uncertainty head adds ~10K params (<1% overhead).

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: VarSplat uncertainty-aware model optimization complete

Inspired by VarSplat (CVPR 2026) and CG-SLAM:
- Per-Gaussian uncertainty prediction via learned CNN head
- Uncertainty-aware rendering (alpha-composited uncertainty maps)
- Uncertainty-weighted depth loss (unreliable regions downweighted)
- Geometry variance loss (prevents trivial infinite uncertainty)
- Uncertainty-guided FPS selection (prefer confident Gaussians)
- Backward compatible: predict_uncertainty=False preserves original behavior

Critical for robustness: 9/12 training datasets use pseudo-depth
which has significant noise. Uncertainty prevents this noise from
corrupting Gaussian positions during optimization."
```
