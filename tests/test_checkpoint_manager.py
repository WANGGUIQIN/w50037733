"""Tests for CheckpointManager: rotation, atomicity, best tracking, load."""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import CheckpointManager, _is_3d_branch_param


# ---------------------------------------------------------------------------
# Minimal model for testing (no heavy VLM)
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_to_gaussian = nn.Linear(4, 8)   # 3d branch param
        self.gs_encoder        = nn.Linear(8, 16)  # 3d branch param
        self.llm_layer         = nn.Linear(16, 32) # non-3d param

    def named_parameters(self, *a, **kw):
        return super().named_parameters(*a, **kw)


def make_manager(tmpdir, keep_last_n=3, save_optimizer_state=True):
    return CheckpointManager(str(tmpdir), keep_last_n=keep_last_n,
                             save_optimizer_state=save_optimizer_state)


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def make_scheduler(opt):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_save_creates_files():
    """A saved checkpoint must contain all expected files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model  = TinyModel()
        opt    = make_optimizer(model)
        sched  = make_scheduler(opt)
        mgr    = make_manager(tmpdir)

        ckpt_dir = mgr.save(model, opt, sched, step=100, epoch=0, loss=2.5, mode="lora")

        assert (ckpt_dir / "3d_branch.pt").exists(),  "3d_branch.pt missing"
        assert (ckpt_dir / "metadata.json").exists(), "metadata.json missing"
        assert (ckpt_dir / "optimizer.pt").exists(),  "optimizer.pt missing"
        assert (ckpt_dir / "scheduler.pt").exists(),  "scheduler.pt missing"
        assert (Path(tmpdir) / "checkpoints.json").exists(), "registry missing"

        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["step"] == 100
        assert meta["loss"] == 2.5
        assert meta["mode"] == "lora"

        print("PASS test_save_creates_files")


def test_no_optimizer_state_when_disabled():
    """save_optimizer_state=False must not write optimizer.pt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, save_optimizer_state=False)

        ckpt_dir = mgr.save(model, opt, None, step=1, epoch=0, loss=1.0, mode="full")
        assert not (ckpt_dir / "optimizer.pt").exists(), \
            "optimizer.pt should not exist when save_optimizer_state=False"

        print("PASS test_no_optimizer_state_when_disabled")


def test_best_symlink_points_to_lowest_loss():
    """best/ must always point to the checkpoint with the lowest loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, keep_last_n=5)

        mgr.save(model, opt, None, step=100, epoch=0, loss=3.0, mode="lora")
        mgr.save(model, opt, None, step=200, epoch=0, loss=2.0, mode="lora")  # best so far
        mgr.save(model, opt, None, step=300, epoch=0, loss=2.5, mode="lora")

        best_link = Path(tmpdir) / "best"
        assert best_link.exists(), "best/ link missing"

        # best/ should resolve to checkpoint-200
        resolved = best_link.resolve() if best_link.is_symlink() else best_link
        assert "checkpoint-200" in str(resolved), \
            f"best/ should point to checkpoint-200, got {resolved}"

        # Registry should record step=200 as best
        assert mgr._registry["best"]["step"] == 200

        print("PASS test_best_symlink_points_to_lowest_loss")


def test_rotation_keeps_last_n():
    """After N+1 saves, only keep_last_n + (protected best) dirs should exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, keep_last_n=3)

        losses = [3.0, 2.8, 2.6, 2.4, 2.2]  # monotone decreasing → each is new best
        for i, loss in enumerate(losses):
            mgr.save(model, opt, None, step=(i+1)*100, epoch=i, loss=loss, mode="lora")

        # All 5 are "best" at some point; final best is checkpoint-500
        # But only 3 entries should be in the rotation list
        assert len(mgr._registry["checkpoints"]) <= 3 + 1, \
            f"Too many checkpoints in registry: {len(mgr._registry['checkpoints'])}"

        # Oldest non-best checkpoint dirs should be gone
        ckpt_dirs = [p for p in Path(tmpdir).iterdir()
                     if p.is_dir() and p.name.startswith("checkpoint-")]
        assert len(ckpt_dirs) <= 4, f"Too many checkpoint dirs on disk: {[d.name for d in ckpt_dirs]}"

        print(f"  Dirs on disk: {sorted(d.name for d in ckpt_dirs)}")
        print("PASS test_rotation_keeps_last_n")


def test_best_protected_from_rotation():
    """The best checkpoint must NOT be deleted even if it falls outside keep_last_n."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, keep_last_n=2)

        # Best at step 100, then loss gets worse
        mgr.save(model, opt, None, step=100, epoch=0, loss=1.0, mode="lora")  # best
        mgr.save(model, opt, None, step=200, epoch=1, loss=2.0, mode="lora")
        mgr.save(model, opt, None, step=300, epoch=2, loss=2.5, mode="lora")
        mgr.save(model, opt, None, step=400, epoch=3, loss=3.0, mode="lora")

        # checkpoint-100 is best and must still exist on disk
        best_dir = Path(tmpdir) / "checkpoint-100"
        assert best_dir.exists(), "Best checkpoint was incorrectly rotated out!"

        best_link = Path(tmpdir) / "best"
        assert best_link.exists(), "best/ link missing"
        resolved = best_link.resolve() if best_link.is_symlink() else best_link
        assert "checkpoint-100" in str(resolved)

        print("PASS test_best_protected_from_rotation")


def test_atomic_write_no_partial_on_tmp_cleanup():
    """No .tmp directory should remain after a successful save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir)

        mgr.save(model, opt, None, step=50, epoch=0, loss=1.5, mode="lora")

        tmp_dirs = list(Path(tmpdir).glob("*.tmp"))
        assert tmp_dirs == [], f"Leftover .tmp dirs: {tmp_dirs}"

        print("PASS test_atomic_write_no_partial_on_tmp_cleanup")


def test_load_restores_step():
    """load() must return the correct step number from metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        sched = make_scheduler(opt)
        mgr   = make_manager(tmpdir)

        mgr.save(model, opt, sched, step=250, epoch=1, loss=1.8, mode="full")

        # Load using explicit path
        model2 = TinyModel()
        opt2   = make_optimizer(model2)
        sched2 = make_scheduler(opt2)
        ckpt_path = str(Path(tmpdir) / "checkpoint-250")
        step = mgr.load(ckpt_path, model2, opt2, sched2, mode="full", device="cpu")

        assert step == 250, f"Expected step=250, got {step}"
        print("PASS test_load_restores_step")


def test_load_best_keyword():
    """load('best', ...) must load the best checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, keep_last_n=5)

        mgr.save(model, opt, None, step=100, epoch=0, loss=3.0, mode="lora")
        mgr.save(model, opt, None, step=200, epoch=1, loss=1.0, mode="lora")  # best
        mgr.save(model, opt, None, step=300, epoch=2, loss=2.0, mode="lora")

        model2 = TinyModel()
        step = mgr.load("best", model2, None, None, mode="lora", device="cpu")
        assert step == 200, f"Expected step=200 from best, got {step}"

        print("PASS test_load_best_keyword")


def test_load_latest_keyword():
    """load('latest', ...) must load the most recently saved checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt   = make_optimizer(model)
        mgr   = make_manager(tmpdir, keep_last_n=5)

        mgr.save(model, opt, None, step=100, epoch=0, loss=3.0, mode="lora")
        mgr.save(model, opt, None, step=200, epoch=1, loss=2.0, mode="lora")
        mgr.save(model, opt, None, step=300, epoch=2, loss=2.5, mode="lora")

        model2 = TinyModel()
        step = mgr.load("latest", model2, None, None, mode="lora", device="cpu")
        assert step == 300, f"Expected step=300 from latest, got {step}"

        print("PASS test_load_latest_keyword")


def test_from_config():
    """CheckpointManager.from_config() must read keep_last_n and save_optimizer_state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "training":   {"output_dir": tmpdir},
            "checkpoint": {"keep_last_n": 7, "save_optimizer_state": False},
        }
        mgr = CheckpointManager.from_config(cfg)
        assert mgr.keep_last_n == 7
        assert mgr.save_optimizer_state is False

        print("PASS test_from_config")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_save_creates_files,
        test_no_optimizer_state_when_disabled,
        test_best_symlink_points_to_lowest_loss,
        test_rotation_keeps_last_n,
        test_best_protected_from_rotation,
        test_atomic_write_no_partial_on_tmp_cleanup,
        test_load_restores_step,
        test_load_best_keyword,
        test_load_latest_keyword,
        test_from_config,
    ]

    passed = failed = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
