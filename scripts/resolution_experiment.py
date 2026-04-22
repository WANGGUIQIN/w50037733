#!/usr/bin/env python3
"""Resolution experiment: does upsampling 256->512->1024 improve GPT grounding?

For each sample episode, query gpt-5.4 with the same target at three scales and
overlay all predictions on the original 256 image for comparison.

Usage:
    conda run -n robobrain_3dgs python scripts/resolution_experiment.py --n-per-ds 2
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "result" / "resolution_experiment"

sys.path.insert(0, str(Path(__file__).parent))
from regenerate_plans_keyframe import API_KEY, BASE_URL  # noqa: E402
from refine_affordance_robobrain import resolve_query  # noqa: E402
from openai import OpenAI  # noqa: E402

MODEL = "gpt-5.4"
SCALES = [256, 512, 1024]
COLORS = {256: (255, 80, 0), 512: (0, 200, 100), 1024: (0, 80, 255)}  # BGR

PROMPT_TPL = (
    "Provide the 2D image pixel coordinate of the center of {target}. "
    "Output ONLY a tuple of normalized coordinates in the format "
    "[(u, v)] where 0.0 <= u, v <= 1.0 (u is horizontal left-to-right, "
    "v is vertical top-to-bottom). No other text."
)
POINT_RE = re.compile(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)")


def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_gpt(client: OpenAI, img: Image.Image, target: str, model: str = MODEL) -> tuple | None:
    img_b64 = encode_pil(img)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT_TPL.format(target=target.replace('_', ' '))},
            ],
        }],
        max_tokens=60,
    )
    text = resp.choices[0].message.content
    m = POINT_RE.search(text)
    if not m:
        return None
    u, v = float(m.group(1)), float(m.group(2))
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return (u, v)


def upscale(img: Image.Image, size: int) -> Image.Image:
    if img.size == (size, size):
        return img
    return img.resize((size, size), Image.LANCZOS)


def draw_overlay(rgb256_path: Path, predictions: dict, target: str, out_path: Path):
    """predictions: {scale: (u, v) or None}"""
    img = cv2.imread(str(rgb256_path))
    h, w = img.shape[:2]
    # upscale x3 for display
    img = cv2.resize(img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]

    cv2.rectangle(img, (0, 0), (w, 28), (30, 30, 30), -1)
    cv2.putText(img, f"target: {target}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    legend_y = 50
    for scale, color in COLORS.items():
        pt = predictions.get(scale)
        if pt is None:
            cv2.putText(img, f"{scale}x{scale}: FAIL", (5, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            x, y = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(img, (x, y), 14, color, -1)
            cv2.circle(img, (x, y), 16, (255, 255, 255), 2)
            cv2.putText(img, f"{scale}", (x + 18, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(img, f"{scale}x{scale}: ({pt[0]:.3f}, {pt[1]:.3f})",
                        (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        legend_y += 22

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def pick_samples(n_per_ds: int) -> list[dict]:
    """Pick episodes with existing plan.json (from prior pilot) so we have a target."""
    samples = []
    datasets = [
        "rlbench", "jaco_play", "taco_play", "fractal20220817_data",
        "bridge", "droid", "rh20t",
    ]
    for ds in datasets:
        ds_dir = DATA_ROOT / ds
        if not ds_dir.exists():
            continue
        # Prefer episodes under pilot_gpt5_4 (already have good plan)
        pilot_dir = PROJECT_ROOT / "result" / "pilot_gpt5_4" / ds
        candidate_names = []
        if pilot_dir.exists():
            candidate_names = [p.stem for p in sorted(pilot_dir.glob("*.png"))]

        taken = 0
        for ep_name in candidate_names:
            if taken >= n_per_ds:
                break
            ep_dir = ds_dir / ep_name
            plan_path = ep_dir / "plan.json"
            if not plan_path.exists():
                continue
            with open(plan_path) as f:
                plan = json.load(f)
            if not plan.get("steps"):
                continue
            # Use the first step that has a concrete target
            for step in plan["steps"]:
                q = resolve_query(step)
                if q:
                    samples.append({
                        "dataset": ds, "episode_id": ep_name,
                        "target": q, "action": step.get("action", ""),
                    })
                    taken += 1
                    break

        if taken == 0 and ds == "rlbench":
            # Fallback for RLBench if pilot dir empty
            for ep_dir in sorted(ds_dir.iterdir())[:n_per_ds]:
                plan_path = ep_dir / "plan.json"
                if not plan_path.exists():
                    continue
                with open(plan_path) as f:
                    plan = json.load(f)
                for step in plan["steps"]:
                    q = resolve_query(step)
                    if q:
                        samples.append({
                            "dataset": ds, "episode_id": ep_dir.name,
                            "target": q, "action": step.get("action", ""),
                        })
                        break
    return samples


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-ds", type=int, default=2)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    MODEL = args.model

    samples = pick_samples(args.n_per_ds)
    print(f"Picked {len(samples)} samples:")
    for s in samples:
        print(f"  {s['dataset']}/{s['episode_id']}  action={s['action']} target={s['target']!r}")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results = []
    t0 = time.time()

    for i, s in enumerate(samples, 1):
        ds, ep_id, target = s["dataset"], s["episode_id"], s["target"]
        rgb_path = DATA_ROOT / ds / ep_id / "rgb_0.png"
        img_orig = Image.open(rgb_path).convert("RGB")

        predictions = {}
        for scale in SCALES:
            img = upscale(img_orig, scale)
            try:
                pt = query_gpt(client, img, target, model=args.model)
            except Exception as e:
                print(f"    [{scale}] ERROR: {e}")
                pt = None
            predictions[scale] = pt

        out_path = OUT_DIR / ds / f"{ep_id}.png"
        draw_overlay(rgb_path, predictions, target, out_path)

        # Compute pairwise diffs
        if predictions[256] and predictions[1024]:
            d = ((predictions[256][0] - predictions[1024][0]) ** 2 +
                 (predictions[256][1] - predictions[1024][1]) ** 2) ** 0.5
        else:
            d = None
        print(f"  [{i}/{len(samples)}] {ds}/{ep_id} {target}")
        for scale, pt in predictions.items():
            pt_str = f"({pt[0]:.3f}, {pt[1]:.3f})" if pt else "FAIL"
            print(f"    {scale}x{scale}: {pt_str}")
        if d is not None:
            print(f"    Δ(256 vs 1024) = {d:.3f}")
        results.append({**s, "predictions": predictions, "delta_256_1024": d})

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  {len(samples)} samples, {elapsed:.0f}s, saved to {OUT_DIR}/")
    print(f"{'=' * 70}")

    # Summary: if delta is consistently large, upscaling matters
    deltas = [r["delta_256_1024"] for r in results if r["delta_256_1024"] is not None]
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        print(f"\n  Mean Δ(256↔1024) across {len(deltas)} samples: {mean_d:.3f}")
        print(f"  Max Δ: {max(deltas):.3f}, Min Δ: {min(deltas):.3f}")
        if mean_d > 0.1:
            print("  → Resolution change meaningfully affects GPT predictions.")
            print("    Inspect images to see if upscaling is BETTER or just DIFFERENT.")
        else:
            print("  → Resolution change has minimal effect on predictions.")
            print("    Bottleneck is not resolution; it's VLM grounding capability itself.")


if __name__ == "__main__":
    main()
