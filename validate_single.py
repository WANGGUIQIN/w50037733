"""Single-sample validation script for RoboBrain-3DGS pipeline.

Validates that the full pipeline runs end-to-end:
    RGBD -> 3D Gaussian -> GS Encoder -> Fusion -> LLM -> Affordance + Constraint

Also performs a short training loop (20 steps) to verify gradients flow correctly
through all modules including the 3D Gaussian branch.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/w50037733/robobrain_3dgs")

from models.robobrain_3dgs import RoboBrain3DGS
from data.synthetic import create_synthetic_sample


def compute_loss(
    outputs: dict[str, torch.Tensor],
    gt_affordance: torch.Tensor,
    gt_constraints: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss for affordance and constraint prediction.

    Args:
        outputs: model outputs
        gt_affordance: [B, num_affordances, 4] ground truth bboxes
        gt_constraints: dict of ground truth constraint tensors

    Returns:
        total_loss: scalar
        loss_dict: individual loss components for logging
    """
    # Affordance loss: L1 + IoU-like loss
    pred_aff = outputs["affordances"]
    loss_affordance = F.l1_loss(pred_aff, gt_affordance)

    # Constraint losses
    pred_con = outputs["constraints"]

    loss_direction = 1.0 - F.cosine_similarity(
        pred_con["approach_direction"],
        gt_constraints["approach_direction"],
        dim=-1,
    ).mean()

    loss_normal = 1.0 - F.cosine_similarity(
        pred_con["contact_normal"],
        gt_constraints["contact_normal"],
        dim=-1,
    ).mean()

    loss_gripper = F.mse_loss(
        pred_con["gripper_width"],
        gt_constraints["gripper_width"],
    )

    loss_force = F.mse_loss(
        pred_con["force_limit"],
        gt_constraints["force_limit"],
    )

    # Weighted sum
    total_loss = (
        loss_affordance * 10.0
        + loss_direction * 5.0
        + loss_normal * 5.0
        + loss_gripper * 2.0
        + loss_force * 1.0
    )

    loss_dict = {
        "affordance": loss_affordance.item(),
        "direction": loss_direction.item(),
        "normal": loss_normal.item(),
        "gripper": loss_gripper.item(),
        "force": loss_force.item(),
        "total": total_loss.item(),
    }
    return total_loss, loss_dict


def validate_forward_pass():
    """Validate a single forward pass through the entire pipeline."""
    print("=" * 60)
    print("RoboBrain-3DGS Pipeline Validation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ---- Create Model ----
    print("\n[1/5] Creating model...")
    model = RoboBrain3DGS(
        image_size=256,
        num_gaussians=1024,       # Reduced for validation speed
        sh_degree=2,
        num_gs_tokens=64,
        hidden_dim=512,           # Smaller for validation
        fusion_mode="concat",
        num_affordances=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Module-level param counts
    module_params = {
        "2D Visual Encoder": sum(p.numel() for p in model.visual_encoder_2d.parameters()),
        "DepthToGaussian": sum(p.numel() for p in model.depth_to_gaussian.parameters()),
        "GS Encoder": sum(p.numel() for p in model.gs_encoder.parameters()),
        "Fusion": sum(p.numel() for p in model.fusion.parameters()),
        "LLM Backbone": sum(p.numel() for p in model.llm.parameters()),
        "Affordance Head": sum(p.numel() for p in model.affordance_head.parameters()),
        "Constraint Head": sum(p.numel() for p in model.constraint_head.parameters()),
    }
    for name, count in module_params.items():
        print(f"    {name}: {count:,}")

    # ---- Create Synthetic Data ----
    print("\n[2/5] Generating synthetic RGBD data...")
    sample = create_synthetic_sample(image_size=256, device=device)
    print(f"  RGB shape: {sample['rgb'].shape}")
    print(f"  Depth shape: {sample['depth'].shape}")
    print(f"  Depth range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}] m")
    print(f"  Intrinsics shape: {sample['intrinsics'].shape}")
    print(f"  Text IDs shape: {sample['text_ids'].shape}")

    # ---- Forward Pass ----
    print("\n[3/5] Running forward pass...")
    model_for_inference = model
    model_for_inference.train(False)
    with torch.no_grad():
        t0 = time.time()
        outputs = model_for_inference(
            rgb=sample["rgb"],
            depth=sample["depth"],
            intrinsics=sample["intrinsics"],
            text_ids=sample["text_ids"],
        )
        t1 = time.time()

    print(f"  Forward pass time: {(t1-t0)*1000:.1f} ms")
    print(f"  Output shapes:")
    print(f"    Gaussians: {outputs['gaussians'].shape}")
    print(f"    Hidden states: {outputs['hidden_states'].shape}")
    print(f"    Affordances: {outputs['affordances'].shape}")
    print(f"    Constraints:")
    for k, v in outputs["constraints"].items():
        print(f"      {k}: {v.shape} = {v[0].tolist()}")

    # ---- Verify Gaussian Properties ----
    print("\n[4/5] Verifying 3D Gaussian properties...")
    gs = outputs["gaussians"]
    xyz = gs[0, :, :3]
    scale = gs[0, :, 3:6]
    rotation = gs[0, :, 6:10]
    opacity = gs[0, :, 10:11]

    print(f"  XYZ range: x=[{xyz[:,0].min():.3f}, {xyz[:,0].max():.3f}], "
          f"y=[{xyz[:,1].min():.3f}, {xyz[:,1].max():.3f}], "
          f"z=[{xyz[:,2].min():.3f}, {xyz[:,2].max():.3f}]")
    print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}]")
    quat_norms = rotation.norm(dim=-1)
    print(f"  Rotation quaternion norms: [{quat_norms.min():.4f}, {quat_norms.max():.4f}] (should be ~1.0)")
    print(f"  Opacity range: [{opacity.min():.4f}, {opacity.max():.4f}]")

    # ---- Training Validation (20 steps) ----
    print("\n[5/5] Training validation (20 steps)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    has_3d_grad = False
    losses = []
    for step in range(20):
        optimizer.zero_grad()
        outputs = model(
            rgb=sample["rgb"],
            depth=sample["depth"],
            intrinsics=sample["intrinsics"],
            text_ids=sample["text_ids"],
        )
        loss, loss_dict = compute_loss(
            outputs, sample["gt_affordance"], sample["gt_constraints"]
        )
        loss.backward()

        # Check gradient flow
        if step == 0:
            print("\n  Gradient flow check:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if "depth_to_gaussian" in name and "encoder.0" in name:
                        print(f"    DepthToGaussian (first conv): grad_norm = {grad_norm:.6f}")
                    elif "gs_encoder.sa1" in name and "mlp.0" in name:
                        print(f"    GS Encoder (SA1 first layer): grad_norm = {grad_norm:.6f}")
                    elif "affordance_head" in name and "head.0" in name:
                        print(f"    Affordance Head (first layer): grad_norm = {grad_norm:.6f}")

            # Verify 3D branch receives gradients
            has_3d_grad = any(
                p.grad is not None and p.grad.norm() > 0
                for n, p in model.named_parameters()
                if "depth_to_gaussian" in n or "gs_encoder" in n
            )
            print(f"\n  3D branch receives gradients: {'YES' if has_3d_grad else 'NO'}")

        optimizer.step()
        losses.append(loss_dict["total"])

        if step % 5 == 0 or step == 19:
            print(f"  Step {step:2d}: total={loss_dict['total']:.4f}  "
                  f"aff={loss_dict['affordance']:.4f}  "
                  f"dir={loss_dict['direction']:.4f}  "
                  f"norm={loss_dict['normal']:.4f}  "
                  f"grip={loss_dict['gripper']:.4f}  "
                  f"force={loss_dict['force']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f} ({loss_reduction:.1f}%)")
    print(f"  Pipeline: RGBD -> 3D Gaussian ({gs.shape[1]} points) "
          f"-> {outputs['hidden_states'].shape[1]} tokens -> Affordance + Constraint")

    checks = [
        ("Forward pass completes", True),
        ("Gradients flow to 3D branch", has_3d_grad),
        ("Loss decreases", losses[-1] < losses[0]),
        ("Gaussian quaternions normalized", bool(quat_norms.min() > 0.99)),
        ("Opacity in [0,1]", bool(opacity.min() >= 0 and opacity.max() <= 1)),
    ]
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = validate_forward_pass()
    sys.exit(0 if success else 1)
