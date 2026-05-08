"""Lang-SAM + clustering post-processor for affordance refinement.

Pipeline: model plan.json (with affordance_hint) + RGB-D
  -> GroundingDINO (text -> bbox)
  -> SAM (bbox -> mask)
  -> mask disambiguation (confidence + coarse prior + depth)
  -> point selection (centroid / inscribed_circle / pca_axis)
  -> depth back-projection -> (u, v, x, y, z, approach)

Usage:
  # Single episode test
  python scripts/postprocess_affordance.py \
    --episode data/processed/rlbench/episode_000000 \
    --strategy pca \
    --visualize out.png

  # Batch over a dataset split
  python scripts/postprocess_affordance.py \
    --root data/processed/rlbench \
    --strategy inscribed \
    --output refined_plans.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from sklearn.decomposition import PCA
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


# ============================================================
# Grounding + SAM
# ============================================================

class GroundingSAM:
    """Wrap GroundingDINO + SAM for text-conditioned segmentation."""

    def __init__(
        self,
        gdino_id: str = "IDEA-Research/grounding-dino-tiny",
        sam_id: str = "facebook/sam-vit-base",
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print(f"[GroundingSAM] loading {gdino_id} ...")
        self.gdino_proc = AutoProcessor.from_pretrained(gdino_id)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device).eval()

        print(f"[GroundingSAM] loading {sam_id} ...")
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, prompt: str):
        """Return (masks: [N, H, W] bool, scores: [N], boxes: [N, 4] xyxy)."""
        # GroundingDINO expects period-terminated lowercase prompt
        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."

        gd_inputs = self.gdino_proc(images=image, text=text, return_tensors="pt").to(self.device)
        gd_out = self.gdino(**gd_inputs)
        results = self.gdino_proc.post_process_grounded_object_detection(
            gd_out,
            input_ids=gd_inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"].cpu()                        # [N, 4] xyxy
        scores = results["scores"].cpu()                      # [N]
        if len(boxes) == 0:
            return np.zeros((0, *image.size[::-1]), bool), np.zeros(0), np.zeros((0, 4))

        # SAM with each box as prompt
        sam_inputs = self.sam_proc(
            image, input_boxes=[boxes.tolist()], return_tensors="pt"
        ).to(self.device)
        sam_out = self.sam(**sam_inputs, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu(),
        )[0]                                                  # [N, 1, H, W]
        masks = masks.squeeze(1).numpy().astype(bool)         # [N, H, W]
        return masks, scores.numpy(), boxes.numpy()


# ============================================================
# Affordance point selection strategies
# ============================================================

@dataclass
class AffordanceResult:
    u: float        # normalized [0, 1]
    v: float
    x: float        # camera frame meters
    y: float
    z: float
    approach: tuple[float, float, float]
    strategy: str
    confidence: float


def _resize_depth_to(depth: np.ndarray | None, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Resize depth to target (H, W) using nearest-neighbor (preserves zeros)."""
    if depth is None:
        return None
    if depth.shape[:2] == target_hw:
        return depth
    th, tw = target_hw
    # use cv2 if available (faster); fall back to PIL nearest
    try:
        import cv2
        return cv2.resize(depth.astype(np.float32), (tw, th),
                          interpolation=cv2.INTER_NEAREST)
    except Exception:
        from PIL import Image as _PIL
        return np.array(
            _PIL.fromarray(depth.astype(np.float32)).resize((tw, th), _PIL.NEAREST)
        )


def _select_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    coarse_uv: tuple[float, float] | None = None,
    depth: np.ndarray | None = None,
    score_thresh: float = 0.30,
) -> tuple[np.ndarray, float] | None:
    """Disambiguate multiple masks. Returns (mask, score) or None."""
    if len(masks) == 0:
        return None
    H, W = masks.shape[1:]

    # Align depth to mask resolution if provided
    depth_aligned = _resize_depth_to(depth, (H, W))

    # 1. confidence filter
    keep = scores > score_thresh
    if not keep.any():
        keep = np.argsort(scores)[-1:]      # always keep top-1 fallback
        keep_mask = np.zeros(len(scores), bool)
        keep_mask[keep] = True
        keep = keep_mask
    masks, scores = masks[keep], scores[keep]

    # 2. prefer mask containing coarse model prediction
    if coarse_uv is not None:
        u_px = int(np.clip(coarse_uv[0] * W, 0, W - 1))
        v_px = int(np.clip(coarse_uv[1] * H, 0, H - 1))
        contains = masks[:, v_px, u_px]
        if contains.any():
            masks, scores = masks[contains], scores[contains]

    # 3. multiple remain: prefer foreground (smaller median depth)
    if len(masks) > 1 and depth_aligned is not None:
        depths = []
        for m in masks:
            d = depth_aligned[m & (depth_aligned > 0)]
            depths.append(np.median(d) if len(d) > 0 else np.inf)
        idx = int(np.argmin(depths))
        return masks[idx], float(scores[idx])

    # 4. fallback: highest score
    idx = int(np.argmax(scores))
    return masks[idx], float(scores[idx])


def point_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    return int(xs.mean()), int(ys.mean())


def point_inscribed_circle(mask: np.ndarray) -> tuple[int, int]:
    """Largest inscribed circle center — robust 'safe grasp' point."""
    dist = distance_transform_edt(mask)
    v, u = np.unravel_index(int(dist.argmax()), dist.shape)
    return int(u), int(v)


def point_pca(mask: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """PCA on mask: returns (center_uv, approach_2d_unit_vector).

    approach_2d is the SHORT axis (perpendicular to elongation),
    which is the natural grasp axis for elongated objects.
    """
    pts = np.column_stack(np.where(mask))               # [N, 2] (y, x)
    if len(pts) < 10:
        u, v = point_centroid(mask)
        return (u, v), np.array([0.0, -1.0])
    pca = PCA(n_components=2).fit(pts)
    cy, cx = pca.mean_
    short_axis = pca.components_[1]                     # (dy, dx)
    approach_2d = np.array([short_axis[1], short_axis[0]])
    approach_2d = approach_2d / (np.linalg.norm(approach_2d) + 1e-8)
    return (int(cx), int(cy)), approach_2d


def _depth_in_mask(mask: np.ndarray, depth: np.ndarray) -> float:
    valid = depth[mask & (depth > 0)]
    if len(valid) == 0:
        return 1.0
    return float(np.median(valid))


def _backproject(u_px: int, v_px: int, z: float, K: np.ndarray) -> tuple[float, float, float]:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_px - cx) * z / fx
    y = (v_px - cy) * z / fy
    return float(x), float(y), float(z)


def extract_affordance(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    strategy: str = "inscribed",
    confidence: float = 1.0,
) -> AffordanceResult:
    """mask: [H, W] bool, depth: [Hd, Wd] meters (any size; auto-aligned),
    intrinsics: [3, 3] for the depth's original resolution."""
    H, W = mask.shape
    approach_3d = (0.0, 0.0, -1.0)

    if strategy == "centroid":
        u_px, v_px = point_centroid(mask)
    elif strategy == "inscribed":
        u_px, v_px = point_inscribed_circle(mask)
    elif strategy == "pca":
        (u_px, v_px), approach_2d = point_pca(mask)
        # lift 2D approach to 3D (z component pointing into the scene)
        approach_3d = (float(approach_2d[0]), float(approach_2d[1]), -1.0)
        norm = np.linalg.norm(approach_3d)
        approach_3d = tuple(c / norm for c in approach_3d)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Align depth + intrinsics to the mask's resolution before sampling
    if depth is not None and depth.shape[:2] != (H, W):
        Hd, Wd = depth.shape[:2]
        depth = _resize_depth_to(depth, (H, W))
        # Scale intrinsics so back-projection uses (u_px, v_px) at mask res
        sx, sy = W / Wd, H / Hd
        K = intrinsics.copy().astype(np.float32)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
    else:
        K = intrinsics

    z = _depth_in_mask(mask, depth)
    x, y, z = _backproject(u_px, v_px, z, K)

    return AffordanceResult(
        u=u_px / W,
        v=v_px / H,
        x=x, y=y, z=z,
        approach=approach_3d,
        strategy=strategy,
        confidence=confidence,
    )


# ============================================================
# Episode-level driver
# ============================================================

def _normalize_uv(uv, img_w: int, img_h: int) -> tuple[float, float] | None:
    """Auto-detect pixel vs normalized coords. Returns normalized [0,1] tuple."""
    if uv is None or len(uv) < 2:
        return None
    u, v = float(uv[0]), float(uv[1])
    # heuristic: max(|u|, |v|) > 2 => assumed pixel coords
    if max(abs(u), abs(v)) > 2.0:
        u, v = u / img_w, v / img_h
    # clip to plausible range (some inference outputs leak >1 even after norm)
    return (max(0.0, min(1.0, u)), max(0.0, min(1.0, v)))


def _fix_approach(approach) -> tuple[float, float, float]:
    """Coerce approach to a unit vector; default top-down if degenerate."""
    if approach is None or len(approach) < 3:
        return (0.0, 0.0, -1.0)
    a = np.array(approach, dtype=np.float32)
    n = np.linalg.norm(a)
    if n < 1e-3:
        return (0.0, 0.0, -1.0)
    a = a / n
    return (float(a[0]), float(a[1]), float(a[2]))


def _build_prompt(step: dict, scene_objects: list, plan_task: str) -> str:
    """Construct grounding prompt for Lang-SAM.

    Priority:
      1. affordance_hint — already a part-aware noun phrase like
         "the handle of the red jar"; this is exactly what Lang-SAM needs
         and using it alone gives the most precise mask.
      2. scene_objects enrichment — only when hint is absent (e.g. the
         inference output of base RoboBrain has scene_objects but no hint),
         combine target with affordance.type for a coarser prompt.
      3. step.target alone — last resort before plan.task.
      4. plan.task — fallback.
    """
    hint = (step.get("affordance_hint") or "").strip()
    if hint:
        return hint

    target = (step.get("target") or "").strip().replace("_", " ")
    parts: list[str] = []
    if target:
        parts.append(target)

    if scene_objects and target:
        for obj in scene_objects:
            if not isinstance(obj, dict):
                continue
            if (obj.get("name") or "").replace("_", " ") == target:
                affs = obj.get("affordances") or []
                for a in affs:
                    t = (a.get("type") or "").strip().replace("_", " ")
                    if t and t.lower() not in target.lower():
                        parts.append(t)
                        break
                break

    if not parts:
        parts.append(plan_task or "object")

    return ", ".join(parts)


def _scene_obj_3d(scene_objects: list, target: str) -> tuple[float, float, float] | None:
    """Pull model's predicted 3D position for cross-checking."""
    if not scene_objects or not target:
        return None
    for obj in scene_objects:
        if isinstance(obj, dict) and obj.get("name") == target:
            pos = obj.get("spatial_relations", {}).get("position")
            if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
    return None


def refine_plan(
    rgb: Image.Image,
    depth: np.ndarray | None,
    intrinsics: np.ndarray,
    plan: dict,
    grounder: GroundingSAM,
    strategy: str = "inscribed",
    consistency_thresh: float = 0.5,
) -> dict:
    """Refine a plan dict in-place. Works for both training and inference formats.

    Inference-format support:
      - Auto-normalizes pixel coords to [0,1].
      - Uses scene_objects[*].affordances[*].type as grounding prompt enrichment.
      - Cross-checks model 3D position against SAM-derived 3D for confidence gating.
      - Fixes non-unit approach vectors.
      - Caches mask per target across steps for consistency + speed.
    """
    W, H = rgb.size
    scene_objects = plan.get("scene_objects") or []
    plan_task = plan.get("task", "")
    refined_steps = []
    mask_cache: dict[str, tuple[np.ndarray, float, str]] = {}  # target -> (mask, score, prompt)

    for step in plan.get("steps", []):
        target = (step.get("target") or "").strip()
        prompt = _build_prompt(step, scene_objects, plan_task)

        # Normalize coarse uv (handle pixel-format inference outputs)
        coarse_uv = _normalize_uv(step.get("affordance"), W, H)

        # Cache by target+prompt to reuse across steps
        cache_key = f"{target}::{prompt}"
        if cache_key in mask_cache:
            mask, score, _ = mask_cache[cache_key]
            cache_hit = True
        else:
            masks, scores, _ = grounder.predict(rgb, prompt)
            sel = _select_mask(masks, scores, coarse_uv=coarse_uv, depth=depth)
            cache_hit = False
            if sel is None:
                mask, score = None, 0.0
            else:
                mask, score = sel
                mask_cache[cache_key] = (mask, score, prompt)

        new_step = dict(step)
        new_step["refine_prompt"] = prompt
        new_step["refine_cache_hit"] = cache_hit

        if mask is None:
            # Fallback: keep model prediction (normalized) so downstream isn't None
            new_step["affordance_refined"] = list(coarse_uv) if coarse_uv else None
            new_step["affordance_3d_refined"] = None
            new_step["approach_refined"] = list(_fix_approach(step.get("approach")))
            new_step["refine_status"] = "no_mask_fallback" if coarse_uv else "no_mask"
            new_step["refine_confidence"] = 0.0
            refined_steps.append(new_step)
            continue

        if depth is None:
            depth_dummy = np.ones_like(mask, dtype=np.float32)
            res = extract_affordance(mask, depth_dummy, intrinsics, strategy, score)
        else:
            res = extract_affordance(mask, depth, intrinsics, strategy, score)

        # 3D consistency cross-check with model's scene_objects.position
        status = "ok"
        model_3d = _scene_obj_3d(scene_objects, target)
        if model_3d is not None and depth is not None:
            d3 = float(np.linalg.norm(np.array(model_3d) - np.array([res.x, res.y, res.z])))
            if d3 > consistency_thresh:
                status = "3d_inconsistent"

        new_step["affordance_refined"] = [round(res.u, 4), round(res.v, 4)]
        new_step["affordance_3d_refined"] = [round(res.x, 4), round(res.y, 4), round(res.z, 4)]
        new_step["approach_refined"] = [round(c, 4) for c in res.approach]
        new_step["refine_status"] = status
        new_step["refine_strategy"] = strategy
        new_step["refine_confidence"] = round(res.confidence, 4)
        refined_steps.append(new_step)

    plan["steps"] = refined_steps
    return plan


def refine_episode(
    episode_dir: Path,
    grounder: GroundingSAM,
    strategy: str = "inscribed",
    frame_idx: int = 0,
    plan_path_override: Path | None = None,
) -> dict:
    """Refine affordance for one episode. Reads RGB/depth/meta from data dir,
    plan from either the data dir (training format) or override path
    (inference output format)."""
    rgb_path = episode_dir / f"rgb_{frame_idx}.png"
    depth_path = episode_dir / f"depth_{frame_idx}.npy"
    meta_path = episode_dir / "meta.json"
    plan_path = plan_path_override or (episode_dir / "plan.json")

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb in {episode_dir}")
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan: {plan_path}")

    rgb = Image.open(rgb_path).convert("RGB")
    depth = np.load(depth_path) if depth_path.exists() else None
    meta = json.loads(meta_path.read_text())
    plan = json.loads(plan_path.read_text())
    K = np.array(meta["intrinsics"], dtype=np.float32)

    return refine_plan(rgb, depth, K, plan, grounder, strategy=strategy)


def visualize(episode_dir: Path, refined_plan: dict, out_path: Path, frame_idx: int = 0):
    import matplotlib.pyplot as plt
    rgb = np.array(Image.open(episode_dir / f"rgb_{frame_idx}.png").convert("RGB"))
    H, W = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    # Group orig + refined points by pixel coordinate so repeated steps on
    # the same target aren't drawn on top of each other. Concatenate step
    # numbers in the annotation instead.
    orig_groups: dict[tuple[int, int], list[int]] = {}
    refined_groups: dict[tuple[int, int], tuple[list[int], bool]] = {}
    for i, step in enumerate(refined_plan["steps"]):
        step_num = step.get("step", i + 1)
        orig = step.get("affordance_lora") or step.get("affordance")
        if orig is not None and len(orig) >= 2:
            uv = _normalize_uv(orig, W, H)
            if uv is not None:
                key = (int(round(uv[0] * W)), int(round(uv[1] * H)))
                orig_groups.setdefault(key, []).append(step_num)
        if step.get("affordance_refined"):
            u, v = step["affordance_refined"]
            key = (int(round(u * W)), int(round(v * H)))
            ok = step.get("refine_status") == "ok"
            existing = refined_groups.get(key, ([], True))
            refined_groups[key] = (existing[0] + [step_num], existing[1] and ok)

    legend_done = {"orig": False, "refined": False}
    for (x, y), nums in orig_groups.items():
        ax.scatter(x, y, c="red", s=140, marker="x", linewidths=3,
                   label=None if legend_done["orig"] else "orig (model)")
        legend_done["orig"] = True
        ax.annotate(",".join(str(n) for n in sorted(nums)),
                    (x + 8, y - 8), color="red", fontsize=8, fontweight="bold")
    for (x, y), (nums, ok) in refined_groups.items():
        color = "lime" if ok else "orange"
        ax.scatter(x, y, c=color, s=140, marker="o",
                   edgecolors="black", linewidths=2,
                   label=None if legend_done["refined"] else "refined")
        legend_done["refined"] = True
        ax.annotate(",".join(str(n) for n in sorted(nums)),
                    (x, y), color="white", fontsize=9,
                    ha="center", va="center", fontweight="bold")
    ax.legend(loc="upper right")
    title = f"{episode_dir.name}: {refined_plan.get('task', '')}"
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


# ============================================================
# CLI
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path, help="Episode dir with rgb/depth/meta")
    p.add_argument("--plan", type=Path, help="Plan json path (inference output). "
                   "If omitted, uses <episode>/plan.json (training format).")
    p.add_argument("--inference_root", type=Path,
                   help="Root of inference outputs (e.g. output_rlbench_planning/). "
                   "Each subdir holds plan.json from model. Pair with --data_root.")
    p.add_argument("--data_root", type=Path,
                   help="Root of processed data (for rgb/depth/meta lookup) "
                   "when --inference_root is used.")
    p.add_argument("--root", type=Path, help="Dataset root for batch mode (training plans)")
    p.add_argument("--strategy", choices=["centroid", "inscribed", "pca"], default="inscribed")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--output", type=Path, help="Output jsonl for batch mode, or json for single")
    p.add_argument("--visualize", type=Path, help="Save viz png (single-episode only)")
    p.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam", default="facebook/sam-vit-base")
    p.add_argument("--limit", type=int, default=0, help="Limit episodes in batch mode")
    args = p.parse_args()

    if not (args.episode or args.root or args.inference_root):
        p.error("one of --episode / --root / --inference_root required")

    grounder = GroundingSAM(gdino_id=args.gdino, sam_id=args.sam)

    if args.episode:
        refined = refine_episode(args.episode, grounder, args.strategy, args.frame,
                                 plan_path_override=args.plan)
        out_str = json.dumps(refined, indent=2, ensure_ascii=False)
        print(out_str)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(out_str)
            print(f"[saved] {args.output}")
        if args.visualize:
            visualize(args.episode, refined, args.visualize, args.frame)
    elif args.inference_root:
        if not args.data_root:
            p.error("--inference_root requires --data_root")
        episodes = sorted([d for d in args.inference_root.iterdir()
                           if d.is_dir() and (d / "plan.json").exists()])
        if args.limit:
            episodes = episodes[:args.limit]
        out = args.output or (args.inference_root / "refined_plans.jsonl")
        with open(out, "w") as f:
            for i, ep_inf in enumerate(episodes):
                ep_data = args.data_root / ep_inf.name
                if not ep_data.exists():
                    print(f"[skip] no data for {ep_inf.name}")
                    continue
                try:
                    refined = refine_episode(ep_data, grounder, args.strategy,
                                             args.frame,
                                             plan_path_override=ep_inf / "plan.json")
                    f.write(json.dumps({"episode": ep_inf.name, "plan": refined},
                                       ensure_ascii=False) + "\n")
                    if i % 10 == 0:
                        print(f"[{i+1}/{len(episodes)}] {ep_inf.name}")
                except Exception as e:
                    print(f"[err] {ep_inf.name}: {e}")
        print(f"[done] -> {out}")
    else:
        episodes = sorted([d for d in args.root.iterdir() if d.is_dir() and (d / "plan.json").exists()])
        if args.limit:
            episodes = episodes[:args.limit]
        out = args.output or args.root / "refined_plans.jsonl"
        with open(out, "w") as f:
            for i, ep in enumerate(episodes):
                try:
                    refined = refine_episode(ep, grounder, args.strategy, args.frame)
                    f.write(json.dumps({"episode": ep.name, "plan": refined}, ensure_ascii=False) + "\n")
                    if i % 20 == 0:
                        print(f"[{i+1}/{len(episodes)}] {ep.name}")
                except Exception as e:
                    print(f"[err] {ep.name}: {e}")
        print(f"[done] -> {out}")


if __name__ == "__main__":
    sys.exit(main())
