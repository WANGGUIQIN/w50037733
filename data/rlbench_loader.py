"""RLBench dataset loader for RoboBrain-3DGS training.

Loads RGBD frames from RLBench episodes (simulation data with native depth).
Each sample: RGB image + depth map + camera intrinsics + task instruction.

RLBench depth format: RGB-encoded PNG where
    depth_meters = near + (far - near) * (R + G*256 + B*256^2) / (256^3 - 1)
with near=0.01, far=5.0 (CoppeliaSim defaults).
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# -------------------------------------------------------------------------
# Stub unpickler: lets us load RLBench pkl files without installing rlbench
# -------------------------------------------------------------------------

class _StubMeta(type):
    def __new__(mcs, name, bases, ns):
        return super().__new__(mcs, name, (object,), ns)


def _make_stub(module, name):
    return _StubMeta(name, (object,), {
        "__module__": module,
        "__init__": lambda self, *a, **kw: self.__dict__.update({"_args": a, **kw}),
        "__setstate__": lambda self, s: self.__dict__.update(
            s if isinstance(s, dict) else {"_state": s}
        ),
    })


class _RLBenchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _make_stub(module, name)


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

# Native RLBench image resolution (used for pixel-coordinate normalisation)
RLBENCH_NATIVE_SIZE = 128

# Task description templates (fallback when variation_descriptions.pkl absent)
TASK_PROMPTS = {
    "close_jar": "close the jar",
    "open_drawer": "open the drawer",
    "slide_block": "slide the block to the target location",
    "pick_up_cup": "pick up the cup from the table",
    "default": "complete the manipulation task shown in the image",
}

# Fallback target when episode observations are unavailable
_FALLBACK_TARGET = (
    "affordance: [0.50, 0.50]. "
    "constraint: gripper_width=0.08, approach=[0.00, 0.00, -1.00]."
)


def decode_rlbench_depth(depth_rgb: np.ndarray, near: float = 0.01, far: float = 5.0) -> np.ndarray:
    """Decode RLBench RGB-encoded depth to meters.

    Args:
        depth_rgb: [H, W, 3] uint8 array from depth PNG
        near: near plane in meters
        far: far plane in meters

    Returns:
        depth_m: [H, W] float32 depth in meters
    """
    r = depth_rgb[:, :, 0].astype(np.float64)
    g = depth_rgb[:, :, 1].astype(np.float64)
    b = depth_rgb[:, :, 2].astype(np.float64)
    normalized = (r + g * 256 + b * 65536) / (256**3 - 1)
    depth_m = near + (far - near) * normalized
    return depth_m.astype(np.float32)


def _split_episodes(
    all_eps: list[Path],
    split: str,
    train_ratio: float,
    seed: int,
) -> set[str]:
    """Deterministically split episode directories into train/test sets.

    Episodes are sorted by name, shuffled with a fixed seed, then the
    first ``train_ratio`` fraction goes to "train" and the rest to "test".
    Returns a set of episode *names* belonging to the requested split.
    """
    import random as _random
    names = [ep.name for ep in sorted(all_eps, key=lambda p: p.name)]
    rng = _random.Random(seed)
    rng.shuffle(names)
    n_train = max(1, int(len(names) * train_ratio))
    if split == "train":
        return set(names[:n_train])
    elif split == "test":
        return set(names[n_train:])
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")


class RLBenchDataset(Dataset):
    """Load RLBench episodes as RGBD samples for 3DGS training.

    Directory structure expected:
        root/
          task_name/
            all_variations/
              episodes/
                episode0/
                  front_rgb/0.png, 1.png, ...
                  front_depth/0.png, 1.png, ...
                  ...

    Split modes (controlled by ``split`` and related params):

        split=None   Load everything (backward-compatible default).
        split="train" / "test"
            Episode-level split within each task.  Episodes are sorted by
            name, deterministically shuffled with ``seed``, then the first
            ``train_ratio`` fraction goes to train and the rest to test.
        task_filter   If given, only load these task names.
        task_exclude  If given, skip these task names entirely.
        max_frames_per_episode
            Cap the number of frames sampled from each episode (useful
            for large-scale eval so runtime stays bounded).
    """

    def __init__(
        self,
        root_dir: str,
        camera: str = "front",
        image_size: int = 256,
        max_frames: int = -1,
        # --- split params (all optional, backward-compatible) ---
        split: str | None = None,
        train_ratio: float = 0.8,
        seed: int = 42,
        task_filter: list[str] | None = None,
        task_exclude: list[str] | None = None,
        max_frames_per_episode: int = -1,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.camera = camera
        self.image_size = image_size

        _task_filter = set(task_filter) if task_filter else None
        _task_exclude = set(task_exclude) if task_exclude else set()

        # Discover all frames, grouped by (task, episode)
        self.samples = []
        for task_dir in sorted(self.root.iterdir()):
            if not task_dir.is_dir():
                continue
            task_name = task_dir.name
            if _task_filter is not None and task_name not in _task_filter:
                continue
            if task_name in _task_exclude:
                continue
            episodes_dir = task_dir / "all_variations" / "episodes"
            if not episodes_dir.exists():
                continue

            # Determine which episodes belong to this split
            all_eps = sorted(
                [d for d in episodes_dir.iterdir() if d.is_dir()],
                key=lambda p: p.name,
            )
            if split is not None and all_eps:
                selected = _split_episodes(all_eps, split, train_ratio, seed)
            else:
                selected = set(ep.name for ep in all_eps)

            for ep_dir in sorted(episodes_dir.iterdir()):
                if ep_dir.name not in selected:
                    continue
                rgb_dir = ep_dir / f"{camera}_rgb"
                depth_dir = ep_dir / f"{camera}_depth"
                if not rgb_dir.exists() or not depth_dir.exists():
                    continue
                ep_frames = []
                for rgb_file in sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem)):
                    depth_file = depth_dir / rgb_file.name
                    if depth_file.exists():
                        ep_frames.append({
                            "rgb_path": str(rgb_file),
                            "depth_path": str(depth_file),
                            "task": task_name,
                            "ep_dir": str(ep_dir),  # full path for pkl loading
                            "episode": ep_dir.name,
                            "frame": int(rgb_file.stem),
                        })
                # Cap frames per episode
                if max_frames_per_episode > 0 and len(ep_frames) > max_frames_per_episode:
                    # Deterministic uniform subsample
                    step = len(ep_frames) / max_frames_per_episode
                    ep_frames = [ep_frames[int(i * step)] for i in range(max_frames_per_episode)]
                self.samples.extend(ep_frames)

        if max_frames > 0:
            self.samples = self.samples[:max_frames]

        # LRU-style caches keyed by episode path
        self._obs_cache: dict = {}
        self._desc_cache: dict = {}

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_obs(self, ep_dir: str):
        """Load and cache episode observations from low_dim_obs.pkl."""
        if ep_dir in self._obs_cache:
            return self._obs_cache[ep_dir]
        pkl_path = Path(ep_dir) / "low_dim_obs.pkl"
        if not pkl_path.exists():
            self._obs_cache[ep_dir] = None
            return None
        try:
            with open(pkl_path, "rb") as f:
                demo = _RLBenchUnpickler(f).load()
            self._obs_cache[ep_dir] = demo._observations
            return demo._observations
        except Exception:
            self._obs_cache[ep_dir] = None
            return None

    def _load_variation_desc(self, ep_dir: str):
        """Load and cache the first variation description string."""
        if ep_dir in self._desc_cache:
            return self._desc_cache[ep_dir]
        pkl_path = Path(ep_dir) / "variation_descriptions.pkl"
        if not pkl_path.exists():
            self._desc_cache[ep_dir] = None
            return None
        try:
            with open(pkl_path, "rb") as f:
                descs = pickle.load(f)
            desc = descs[0] if isinstance(descs, list) and descs else None
            self._desc_cache[ep_dir] = desc
            return desc
        except Exception:
            self._desc_cache[ep_dir] = None
            return None

    def _find_next_waypoint(self, obs_list: list, frame_idx: int):
        """Return the next observation where the gripper state changes.

        Falls back to the last observation if no state change occurs.
        """
        cur_open = obs_list[frame_idx].gripper_open
        for i in range(frame_idx + 1, len(obs_list)):
            if abs(obs_list[i].gripper_open - cur_open) > 0.3:
                return obs_list[i]
        return obs_list[-1]

    def _project_world_to_image(
        self,
        xyz_world: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics_raw: np.ndarray,
    ) -> tuple:
        """Project a world-frame 3-D point to normalised image coordinates.

        CoppeliaSim stores intrinsics with negated fx/fy; we take absolute
        values and apply the standard OpenCV projection formula.

        Returns:
            (u, v) each clamped to [0, 1].
        """
        p_cam = extrinsics @ np.array([*xyz_world, 1.0])
        z = p_cam[2]
        if z < 1e-3:
            return 0.5, 0.5
        fx = abs(float(intrinsics_raw[0, 0]))
        fy = abs(float(intrinsics_raw[1, 1]))
        cx = float(intrinsics_raw[0, 2])
        cy = float(intrinsics_raw[1, 2])
        u = float(np.clip((fx * p_cam[0] / z + cx) / RLBENCH_NATIVE_SIZE, 0.0, 1.0))
        v = float(np.clip((fy * p_cam[1] / z + cy) / RLBENCH_NATIVE_SIZE, 0.0, 1.0))
        return u, v

    def _make_target(self, obs_list: list, frame_idx: int) -> str:
        """Generate the affordance + constraint target string for one frame."""
        wp_obs  = self._find_next_waypoint(obs_list, frame_idx)
        cur_obs = obs_list[frame_idx]
        cam = self.camera
        extrinsics   = cur_obs.misc.get(f"{cam}_camera_extrinsics")
        intrinsics_raw = cur_obs.misc.get(f"{cam}_camera_intrinsics")
        if extrinsics is None or intrinsics_raw is None:
            return _FALLBACK_TARGET

        gx, gy, gz = float(wp_obs.gripper_pose[0]), float(wp_obs.gripper_pose[1]), float(wp_obs.gripper_pose[2])
        u, v = self._project_world_to_image(np.array([gx, gy, gz]), extrinsics, intrinsics_raw)

        width = 0.08 if float(wp_obs.gripper_open) > 0.5 else 0.00

        approach = wp_obs.gripper_matrix[:3, 2].astype(np.float64)
        approach = approach / (np.linalg.norm(approach) + 1e-8)

        return (
            f"affordance: [{u:.2f}, {v:.2f}]. "
            f"constraint: gripper_width={width:.2f}, "
            f"approach=[{approach[0]:.2f}, {approach[1]:.2f}, {approach[2]:.2f}]."
        )

    def __getitem__(self, idx: int) -> dict:
        info = self.samples[idx]

        # --- RGB ---
        rgb_img = Image.open(info["rgb_path"]).convert("RGB")
        rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb = torch.from_numpy(
            np.array(rgb_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]

        # --- Depth ---
        depth_rgb = np.array(Image.open(info["depth_path"]).convert("RGB"))
        depth_m = decode_rlbench_depth(depth_rgb)
        depth_pil = Image.fromarray(depth_m, mode="F")
        depth_pil = depth_pil.resize((self.image_size, self.image_size), Image.NEAREST)
        depth = torch.from_numpy(np.array(depth_pil)).unsqueeze(0)  # [1, H, W]

        # --- Intrinsics: use actual values from misc when available ---
        obs_list = self._load_obs(info["ep_dir"])
        frame_idx = info["frame"]
        raw_intr = None
        if obs_list is not None and frame_idx < len(obs_list):
            raw_intr = obs_list[frame_idx].misc.get(f"{self.camera}_camera_intrinsics")

        scale = self.image_size / RLBENCH_NATIVE_SIZE
        if raw_intr is not None:
            intr = np.array([
                [abs(raw_intr[0, 0]) * scale, 0, raw_intr[0, 2] * scale],
                [0, abs(raw_intr[1, 1]) * scale, raw_intr[1, 2] * scale],
                [0, 0, 1],
            ], dtype=np.float32)
        else:
            approx_fx = (RLBENCH_NATIVE_SIZE / (2 * np.tan(np.radians(30)))) * scale
            intr = np.array([
                [approx_fx, 0, self.image_size / 2],
                [0, approx_fx, self.image_size / 2],
                [0, 0, 1],
            ], dtype=np.float32)
        intrinsics = torch.from_numpy(intr)

        # --- Prompt (prefer variation_descriptions.pkl) ---
        desc = self._load_variation_desc(info["ep_dir"])
        prompt = desc or TASK_PROMPTS.get(info["task"], TASK_PROMPTS["default"])

        # --- Target (affordance + constraint from episode obs) ---
        if obs_list is not None and frame_idx < len(obs_list):
            target = self._make_target(obs_list, frame_idx)
        else:
            target = _FALLBACK_TARGET

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "prompt": prompt,
            "target": target,
            "task": info["task"],
            "task_type": "affordance",
            "episode": info["episode"],
            "frame": info["frame"],
        }


def validate_rlbench_loader():
    """Quick validation of the RLBench data loader."""
    data_root = "/home/w50037733/robobrain_3dgs/data/rlbench_sample"
    ds = RLBenchDataset(data_root, camera="front", image_size=256, max_frames=5)
    print(f"RLBench dataset: {len(ds)} samples")

    sample = ds[0]
    print(f"  RGB:       {sample['rgb'].shape}, range=[{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
    print(f"  Depth:     {sample['depth'].shape}, range=[{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]m")
    print(f"  Intrinsics:{sample['intrinsics']}")
    print(f"  Prompt:    {sample['prompt']!r}")
    print(f"  Target:    {sample['target']!r}")
    print(f"  Task/Ep/Frame: {sample['task']}/{sample['episode']}/{sample['frame']}")
    return True


if __name__ == "__main__":
    validate_rlbench_loader()
