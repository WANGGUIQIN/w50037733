#!/usr/bin/env python3
"""Annotate and visualize RoboBrain2.5-8B affordance/pointing predictions.

Three modes (auto-detected from CLI args):

    1. Single query:
        python scripts/affordance_robobrain_viz.py \\
            --image data/processed/bridge/episode_000001/rgb_0.png \\
            --query "the handle of the silver pot"

    2. Multiple queries (each gets a distinct color on the same image):
        python scripts/affordance_robobrain_viz.py \\
            --image scene.png \\
            --queries "the silver pot" "the handle of the silver pot" \\
                      "an empty area on the blue towel"

    3. Episode mode (loads meta.json + optional plan.json):
        # Single query = task description from meta.json
        python scripts/affordance_robobrain_viz.py \\
            --episode data/processed/bridge/episode_000001

        # Multi query = idea-B templated queries derived from plan.json steps
        python scripts/affordance_robobrain_viz.py \\
            --episode data/processed/bridge/episode_000001 --use-plan

Output: annotated PNG + JSON sidecar with raw (u, v) predictions in [0, 1].
"""
import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent / "RoboBrain2.5"))

# Heavy ML imports (torch / transformers / qwen_vl_utils) are deferred to the
# inference call path so that --help and arg validation work in any env.

DEFAULT_MODEL_PATH = "/home/edge/Embodied/models/RoboBrain2.5-8B-NV"
POINT_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")

# Idea-B action -> functional-part templates. Mirrors
# scripts/refine_affordance_robobrain.py so this tool produces the same query
# strings as the offline refine pipeline.
ACTION_PART = {
    "reach":     "the {obj}",
    "grasp":     "the handle of the {obj}",
    "pick":      "the handle of the {obj}",
    "lift":      "the handle of the {obj}",
    "rotate":    "the body of the {obj}",
    "press":     "the center of the {obj}",
    "push":      "the center of the {obj}",
    "pull":      "the handle of the {obj}",
    "place":     "an empty area on the {obj}",
    "transport": "an empty area on the {obj}",
    "insert":    "the top opening of the {obj}",
    "pour":      "the inside of the {obj}",
    "flip":      "the body of the {obj}",
    "wipe":      "the surface of the {obj}",
    "release":   "the {obj}",
    "open":      "the handle of the {obj}",
    "close":     "the handle of the {obj}",
}

NO_HANDLE_TOKENS = {
    "butter", "dairy", "package", "box", "block", "cube", "ball", "sphere",
    "rag", "cloth", "towel", "paper", "card", "coin", "battery", "marker",
    "sponge", "candy", "carrot", "potato", "onion", "tomato", "lemon",
    "apple", "banana", "orange", "egg", "bread", "cheese",
}

DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}

# Distinct hue-spaced colors. Up to 10 queries before cycling.
COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6",
    "#1abc9c", "#e67e22", "#d35400", "#16a085", "#8e44ad",
]


def _load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_point(answer: str):
    """Parse RoboBrain pointing output. Coords are 0-1000; normalize to [0, 1]."""
    m = POINT_RE.search(answer)
    if not m:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return (round(x / 1000.0, 4), round(y / 1000.0, 4))


def build_query_for_step(step: dict) -> str:
    """Idea-B query for a plan step. transport/place/insert/pour locate the
    destination; everything else locates the target."""
    action = step.get("action", "").lower()
    dest = step.get("destination")
    raw = dest if (action in DESTINATION_ACTIONS and dest) else step.get("target", "")
    if isinstance(raw, list):
        raw = " and ".join(str(x) for x in raw)
    obj = (raw or "").replace("_", " ")
    if not obj:
        return ""
    template = ACTION_PART.get(action)
    if template is None:
        return obj
    obj_lower = obj.lower()
    if "handle of" in template and any(t in obj_lower for t in NO_HANDLE_TOKENS):
        return obj
    if template == "the {obj}":
        return f"the {obj}"
    return template.format(obj=obj)


def predict_points(model, image_path: str, queries: list) -> list:
    """Run RoboBrain pointing for each query. Suppress the verbose per-call
    prints from inference.py so the script's own output stays readable."""
    results = []
    for q in queries:
        with contextlib.redirect_stdout(io.StringIO()):
            r = model.inference(
                text=q, image=image_path, task="pointing",
                do_sample=False, temperature=0.0,
            )
        pt = extract_point(r["answer"])
        results.append({
            "query": q,
            "answer": r["answer"].strip(),
            "point": list(pt) if pt else None,
        })
    return results


def render(image_path: str, results: list, output_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    font_size = max(12, w // 32)
    font = _load_font(font_size)

    drawn = 0
    for i, r in enumerate(results):
        pt = r.get("point")
        if pt is None:
            continue
        u, v = pt
        x = max(0, min(int(round(u * w)), w - 1))
        y = max(0, min(int(round(v * h)), h - 1))
        color = COLORS[i % len(COLORS)]
        radius = max(7, w // 50)

        # Hollow circle + crosshair for sub-pixel readability
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=color, width=3)
        draw.line([x - radius - 2, y, x - 2, y], fill=color, width=1)
        draw.line([x + 2, y, x + radius + 2, y], fill=color, width=1)
        draw.line([x, y - radius - 2, x, y - 2], fill=color, width=1)
        draw.line([x, y + 2, x, y + radius + 2], fill=color, width=1)

        # Numbered label, anchored right of the marker; flip to left if it
        # would overflow the right edge.
        label = f"{i+1}. {r['query']}"
        tx, ty = x + radius + 4, y - radius - 2
        bbox = draw.textbbox((tx, ty), label, font=font)
        if bbox[2] >= w:
            tx = max(0, x - radius - 4 - (bbox[2] - bbox[0]))
            bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill="#000000bb")
        draw.text((tx, ty), label, fill=color, font=font)
        drawn += 1

    banner = (f"RoboBrain2.5-8B pointing  queries={len(results)}  "
              f"drawn={drawn}  source={Path(image_path).name}")
    bbox = draw.textbbox((4, 4), banner, font=font)
    draw.rectangle(bbox, fill="#000000bb")
    draw.text((4, 4), banner, fill="#ffffff", font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def resolve_episode(ep_dir: Path, use_plan: bool):
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing in {ep_dir}")
    meta = json.load(open(meta_path))
    image_path = str(ep_dir / "rgb_0.png")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"rgb_0.png missing in {ep_dir}")

    if use_plan:
        plan_path = ep_dir / "plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"--use-plan: {plan_path} missing")
        plan = json.load(open(plan_path))
        queries = []
        for s in plan.get("steps", []):
            q = build_query_for_step(s)
            if q:
                queries.append(q)
        if not queries:
            raise ValueError(f"No queryable steps in {plan_path}")
        return image_path, queries

    return image_path, [meta.get("task", "the target object")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1] if "\n" in __doc__ else "",
    )
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--device", default="cuda:0")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="RGB image path (single or multi-query mode).")
    src.add_argument("--episode", help="Episode dir (uses meta.json + optional plan.json).")

    ap.add_argument("--query", help="Single query string.")
    ap.add_argument("--queries", nargs="+",
                    help="Multiple query strings, each gets a distinct color.")
    ap.add_argument("--use-plan", action="store_true",
                    help="Episode mode: derive queries from plan.json steps "
                         "via idea-B action-aware templates.")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: alongside the input image "
                         "with suffix '_robobrain_viz.png'.")
    args = ap.parse_args()

    if args.image:
        image_path = args.image
        if not Path(image_path).exists():
            ap.error(f"Image not found: {image_path}")
        if args.queries:
            queries = args.queries
        elif args.query:
            queries = [args.query]
        else:
            ap.error("--image requires --query or --queries")
    else:
        ep = Path(args.episode)
        if not ep.is_dir():
            ap.error(f"Episode dir not found: {ep}")
        image_path, queries = resolve_episode(ep, args.use_plan)

    out_path = args.output or str(
        Path(image_path).with_name(Path(image_path).stem + "_robobrain_viz.png")
    )

    print(f"Image:   {image_path}")
    print(f"Queries: ({len(queries)})")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print(f"\nLoading RoboBrain2.5 from {args.model_path} ...")
    from inference import UnifiedInference  # deferred heavy import
    model = UnifiedInference(args.model_path, device_map=args.device)

    print("\nRunning pointing inference ...")
    results = predict_points(model, image_path, queries)

    print()
    for i, r in enumerate(results, 1):
        pt = r["point"]
        pt_str = f"[{pt[0]:.3f}, {pt[1]:.3f}]" if pt else "FAIL (parse)"
        print(f"  {i}. {r['query']:55s} -> {pt_str}")

    render(image_path, results, out_path)
    print(f"\nSaved viz:  {out_path}")
    json_path = Path(out_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(
            {"image": image_path, "model": args.model_path, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved json: {json_path}")


if __name__ == "__main__":
    main()
