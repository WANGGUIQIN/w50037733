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
