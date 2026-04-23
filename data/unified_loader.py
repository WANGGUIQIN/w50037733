"""Unified dataset loader for RoboBrain-3DGS training.

Loads episodes from the processed data pipeline output (data/processed/).
Supports two training modes:
  - "affordance": Single-frame affordance prediction (original task)
  - "planning": Task decomposition + per-stage constraint generation (new task)

Each episode directory contains:
    rgb_0.png .. rgb_4.png     (256x256 keyframes)
    depth_0.npy .. depth_4.npy (256x256 float32 meters)
    meta.json                  (task, intrinsics, keyframe_indices, ...)

For planning mode, also requires:
    plan.json                  (GPT-generated task decomposition + constraints)
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UnifiedDataset(Dataset):
    """Load processed episodes for affordance or planning training.

    Args:
        root_dir: Path to data/processed/
        datasets: List of dataset names to include (None = all)
        image_size: Target image size (should match processed data, 256)
        task_type: "affordance" or "planning"
        frame_index: Which keyframe to use as the scene observation (0 = first)
        max_episodes: Cap total episodes (for debugging)
    """

    def __init__(
        self,
        root_dir: str,
        datasets: list[str] | None = None,
        image_size: int = 256,
        task_type: str = "affordance",
        frame_index: int = 0,
        max_episodes: int = -1,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.image_size = image_size
        self.task_type = task_type
        self.frame_index = frame_index

        # Discover episodes
        self.episodes = []
        dataset_dirs = sorted(self.root.iterdir())
        for ds_dir in dataset_dirs:
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            if datasets is not None and ds_name not in datasets:
                continue
            for ep_dir in sorted(ds_dir.iterdir()):
                if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                    continue
                meta_path = ep_dir / "meta.json"
                if not meta_path.exists():
                    continue
                # For planning mode, require plan.json
                if task_type == "planning" and not (ep_dir / "plan.json").exists():
                    continue
                self.episodes.append({
                    "dir": str(ep_dir),
                    "dataset": ds_name,
                })

        if max_episodes > 0:
            self.episodes = self.episodes[:max_episodes]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        ep_dir = Path(ep["dir"])

        # Load meta
        with open(ep_dir / "meta.json") as f:
            meta = json.load(f)

        fi = self.frame_index

        # --- RGB ---
        rgb_path = ep_dir / f"rgb_{fi}.png"
        rgb_img = Image.open(rgb_path).convert("RGB")
        if rgb_img.size != (self.image_size, self.image_size):
            rgb_img = rgb_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        rgb = torch.from_numpy(
            np.array(rgb_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]

        # --- Depth ---
        depth_path = ep_dir / f"depth_{fi}.npy"
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != (self.image_size, self.image_size):
            depth_pil = Image.fromarray(depth, mode="F")
            depth_pil = depth_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            depth = np.array(depth_pil)
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        # --- Intrinsics ---
        intr = meta.get("intrinsics", [[222.7, 0, 128.0], [0, 222.7, 128.0], [0, 0, 1]])
        intrinsics = torch.tensor(intr, dtype=torch.float32)

        # --- Prompt and Target ---
        task_desc = meta.get("task", "manipulation")

        if self.task_type == "planning":
            prompt = task_desc
            target = self._load_plan_target(ep_dir)
        else:
            prompt = task_desc
            target = self._make_affordance_target(meta)

        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": intrinsics,
            "prompt": prompt,
            "target": target,
            "task": task_desc,
            "task_type": self.task_type,
            "episode": meta.get("episode_id", ep_dir.name),
            "frame": fi,
            "dataset": ep["dataset"],
        }

    def _make_affordance_target(self, meta: dict) -> str:
        """Generate affordance target from meta (simple fallback)."""
        # Use center of image as default affordance point
        return (
            "affordance: [0.50, 0.50]. "
            "constraint: gripper_width=0.08, approach=[0.00, 0.00, -1.00]."
        )

    def _load_plan_target(self, ep_dir: Path) -> str:
        """Load GPT-generated plan as the training target.

        Handles both the new format (with constraint categories) and legacy
        format (flat gripper field). Output is the structured text format
        that the LLM learns to generate.
        """
        plan_path = ep_dir / "plan.json"
        with open(plan_path) as f:
            plan = json.load(f)

        lines = []

        # Scene objects line
        scene_objects = plan.get("scene_objects", [])
        if scene_objects:
            lines.append(f"Scene: {', '.join(scene_objects)}")

        for step in plan.get("steps", []):
            step_num = step.get("step", "?")
            action = step.get("action", "manipulate")
            target_obj = step.get("target", "object")
            dest = step.get("destination", "")
            aff = step.get("affordance", [0.5, 0.5])
            approach = step.get("approach", [0.0, 0.0, -1.0])
            done = step.get("done_when", "")

            # Coerce malformed numeric fields to defaults — GPT sometimes
            # emits placeholder strings (e.g. "dynamic", "similar to above")
            # instead of lists of floats, which would crash the f-string
            # formatting with "unknown format code 'f' for object of type str".
            if not (isinstance(aff, list) and len(aff) >= 2 and
                    all(isinstance(v, (int, float)) for v in aff[:2])):
                aff = [0.5, 0.5]
            if not (isinstance(approach, list) and len(approach) >= 3 and
                    all(isinstance(v, (int, float)) for v in approach[:3])):
                approach = [0.0, 0.0, -1.0]

            # Step header: "action(target)" or "action(target -> dest)"
            if dest:
                header = f"Step {step_num}: {action}({target_obj} -> {dest})"
            else:
                header = f"Step {step_num}: {action}({target_obj})"

            lines.append(header)
            # .3f precision so models learn 1000-bin resolution instead of 100;
            # u=/v= prefixes give explicit field-name context for the decoder.
            lines.append(
                f"  affordance: [u={aff[0]:.3f}, v={aff[1]:.3f}], "
                f"approach: [x={approach[0]:.3f}, y={approach[1]:.3f}, z={approach[2]:.3f}]"
            )

            # Constraint categories (new format).
            # GPT sometimes hallucinates a placeholder string instead of the
            # structured dict — coerce anything non-dict into empty dict so
            # the loader falls back to legacy-gripper rendering instead of
            # crashing with AttributeError on .get().
            constraints = step.get("constraints", {})
            if not isinstance(constraints, dict):
                constraints = {}
            if constraints:
                for cat in ("contact", "spatial", "pose", "direction", "safety"):
                    cat_constraints = constraints.get(cat, [])
                    if not cat_constraints:
                        continue
                    parts = []
                    for c in cat_constraints:
                        pred = c["pred"]
                        args = ", ".join(str(a) for a in c.get("args", []))
                        role = c.get("role", "safety" if cat == "safety" else "")
                        entry = f"{pred}({args})"
                        if role and not (cat == "safety" and role == "safety"):
                            entry += f" [{role}]"
                        parts.append(entry)
                    lines.append(f"  {cat}: {'; '.join(parts)}")
            else:
                # Legacy format fallback
                gripper = step.get("gripper", "")
                if gripper:
                    lines.append(f"  gripper: {gripper}")

            if done:
                lines.append(f"  done_when: {done}")

        # End-of-plan signal: lets the model learn when to stop generating.
        # Without this, greedy/sampled decoding often loops through extra
        # synthetic steps (observed in first-version student model output).
        lines.append("<END_OF_PLAN>")
        return "\n".join(lines)
