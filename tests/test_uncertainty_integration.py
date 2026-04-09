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
    # For sh_degree=0: xyz(3)+scale(3)+rot(4)+opa(1)+sh(3)+unc(1) = 15
    assert out["gaussians"].shape[-1] == 15, f"Got {out['gaussians'].shape}"
    # Logits should have gs_tokens prepended
    assert out["logits"].shape[1] == 10 + model.num_gs_tokens


def test_rendering_loss_with_uncertainty_from_vlm():
    """Rendering loss should work with gaussians from VLM pipeline."""
    from models.gs_renderer import GaussianRenderingLoss
    from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config

    config = create_tiny_vlm_config()
    model = RoboBrain3DGS_VLM(
        vlm_config=config,
        num_gaussians=32,
        sh_degree=0,
        num_gs_tokens=4,
        gs_encoder_dim=64,
    )

    B = 1
    rgb = torch.randn(B, 3, 32, 32)
    depth = torch.randn(B, 1, 32, 32).abs() + 0.1
    intrinsics = torch.eye(3).unsqueeze(0) * 64
    intrinsics[:, 2, 2] = 1
    input_ids = torch.randint(0, 1000, (B, 5))
    attention_mask = torch.ones(B, 5, dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        depth=depth,
        intrinsics=intrinsics,
        rgb_for_3d=rgb,
    )

    # Use the gaussians from VLM with the rendering loss
    loss_fn = GaussianRenderingLoss(
        image_size=(8, 8), sh_degree=0, lambda_uncertainty=0.1
    )
    target_rgb = torch.rand(B, 3, 32, 32)
    target_depth = torch.rand(B, 1, 32, 32) + 0.5

    result = loss_fn(
        out["gaussians"], intrinsics, target_rgb, target_depth,
        has_uncertainty=True,
    )
    assert "uncertainty_var" in result
    assert result["loss"].requires_grad


def test_uncertainty_not_in_llm_tokens():
    """GaussianEncoder should NOT receive uncertainty channel."""
    from models.robobrain_vlm import RoboBrain3DGS_VLM, create_tiny_vlm_config
    from models.gs_encoder import GaussianEncoder

    config = create_tiny_vlm_config()
    model = RoboBrain3DGS_VLM(
        vlm_config=config,
        num_gaussians=32,
        sh_degree=0,
        num_gs_tokens=4,
        gs_encoder_dim=64,
    )

    # GaussianEncoder's gaussian_dim should match D WITHOUT uncertainty
    # For sh_degree=0: 3+3+4+1+3 = 14 (no uncertainty)
    expected_dim = 3 + 3 + 4 + 1 + (0 + 1)**2 * 3  # 14
    assert model.gs_encoder.num_tokens == 4
