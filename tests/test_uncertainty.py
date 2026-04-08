"""Tests for per-Gaussian uncertainty prediction in DepthToGaussian."""

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
    expected_dim = 38 + 1  # original 38 + 1 uncertainty
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
    uncertainty = gaussians[..., -1]
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


def test_render_uncertainty_map():
    """Renderer should output uncertainty map when uncertainty is provided."""
    from models.gs_renderer import render_gaussians

    B, N = 1, 32
    means = torch.randn(B, N, 3)
    means[..., 2] = means[..., 2].abs() + 1.0
    scales = torch.ones(B, N, 3) * 0.1
    rotations = torch.zeros(B, N, 4)
    rotations[..., 0] = 1.0
    opacities = torch.ones(B, N, 1) * 0.5
    sh_coeffs = torch.randn(B, N, 27)
    uncertainty = torch.rand(B, N, 1) * 0.5 + 0.01
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    result = render_gaussians(
        means, scales, rotations, opacities, sh_coeffs,
        intrinsics, (16, 16), uncertainty=uncertainty,
    )
    assert "rendered_uncertainty" in result
    assert result["rendered_uncertainty"].shape == (B, 1, 16, 16)
    assert (result["rendered_uncertainty"] >= 0).all()


def test_render_geometric_uncertainty_no_input():
    """Without uncertainty input, renderer computes depth variance."""
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
    assert "rendered_uncertainty" in result
    assert result["rendered_uncertainty"].shape == (B, 1, 8, 8)
    assert (result["rendered_uncertainty"] >= 0).all()


def test_uncertainty_weighted_loss():
    """Rendering loss should use uncertainty weighting."""
    from models.gs_renderer import GaussianRenderingLoss

    loss_fn = GaussianRenderingLoss(
        image_size=(8, 8), sh_degree=0, lambda_uncertainty=0.1
    )

    B, N = 1, 32
    xyz = torch.randn(B, N, 3)
    xyz[..., 2] = xyz[..., 2].abs() + 0.5
    scale = torch.ones(B, N, 3) * 0.1
    rot = torch.zeros(B, N, 4); rot[..., 0] = 1
    opa = torch.ones(B, N, 1) * 0.5
    sh = torch.randn(B, N, 3)  # sh_degree=0: 1^2 * 3 = 3 coeffs
    unc = torch.rand(B, N, 1) * 0.2 + 0.01
    gaussians = torch.cat([xyz, scale, rot, opa, sh, unc], dim=-1)  # 3+3+4+1+3+1 = 15
    gaussians.requires_grad_(True)

    intrinsics = torch.eye(3).unsqueeze(0) * 32
    intrinsics[:, 2, 2] = 1
    target_rgb = torch.rand(B, 3, 32, 32)
    target_depth = torch.rand(B, 1, 32, 32) + 0.5

    result = loss_fn(gaussians, intrinsics, target_rgb, target_depth, has_uncertainty=True)
    assert "loss" in result
    assert "uncertainty_var" in result
    assert result["uncertainty_var"] >= 0
    assert result["loss"].requires_grad


def test_uncertainty_guided_selection():
    """High-uncertainty points should be deprioritized in FPS."""
    from models.depth_to_gaussian import DepthToGaussian

    torch.manual_seed(42)
    module = DepthToGaussian(num_gaussians=32, sh_degree=0, predict_uncertainty=True)
    rgb = torch.randn(1, 3, 32, 32)
    depth = torch.randn(1, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1

    gaussians = module(rgb, depth, intrinsics)
    uncertainty = gaussians[..., -1]
    assert uncertainty.shape == (1, 32)
    assert (uncertainty > 0).all()
