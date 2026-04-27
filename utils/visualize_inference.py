"""Render affordance/planning inference output on top of the input image.

Used by run_inference.py when --visualize is set. Standalone — only depends on
PIL, so it works in any inference environment.

Two task types are supported:
  - "affordance": single point from {"u", "v"} (with optional approach arrow).
  - "planning":   multiple points, one per step, with step number + action label.

Other task types ("pointing", "grounding", "trajectory") fall back to a generic
heuristic that scans the structured dict for any [u, v] in [0, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

# Distinct colors for up to 12 steps; cycles after.
_STEP_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6",
    "#1abc9c", "#d35400", "#c0392b", "#16a085", "#8e44ad", "#2c3e50",
]


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    """Try DejaVuSans (common on Linux), fall back to PIL default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_point(draw: ImageDraw.ImageDraw, x: int, y: int, color: str,
                radius: int = 7, label: str | None = None,
                font: ImageFont.ImageFont | None = None) -> None:
    """Draw a hollow circle marker with optional label to the right."""
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=color, width=3,
    )
    # Crosshair for sub-pixel precision when point is small
    draw.line([x - radius - 2, y, x - 2, y], fill=color, width=1)
    draw.line([x + 2, y, x + radius + 2, y], fill=color, width=1)
    draw.line([x, y - radius - 2, x, y - 2], fill=color, width=1)
    draw.line([x, y + 2, x, y + radius + 2], fill=color, width=1)
    if label:
        # Background pill behind text for readability on white/cluttered scenes
        tx, ty = x + radius + 4, y - radius - 2
        if font is not None:
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle(bbox, fill="#000000aa")
            draw.text((tx, ty), label, fill=color, font=font)
        else:
            draw.text((tx, ty), label, fill=color)


def _norm_to_pixel(u: float, v: float, w: int, h: int) -> tuple[int, int]:
    """Convert normalized [0,1] coords to pixel ints, clamped to image bounds."""
    return (
        max(0, min(int(round(u * w)), w - 1)),
        max(0, min(int(round(v * h)), h - 1)),
    )


def _iter_planning_points(structured: dict) -> Iterable[tuple[int, str, str, list[float]]]:
    """Yield (step_num, action, target, [u, v]) for each step that has affordance."""
    for s in structured.get("steps", []):
        aff = s.get("affordance")
        if not aff or len(aff) < 2:
            continue
        yield (
            s.get("step", 0),
            s.get("action", "?"),
            s.get("target", "?"),
            aff,
        )


def render(image_path: str | Path, structured: dict, task: str,
           output_path: str | Path) -> Path:
    """Render structured inference output as an overlay PNG.

    Args:
        image_path: Source RGB image (PNG/JPG).
        structured: Parsed inference dict from run_single().
        task: "planning" | "affordance" | "pointing" | other.
        output_path: Where to save the annotated PNG.

    Returns:
        Path to the saved image. Caller can rely on this even if no points were
        found — the source image is copied through with a text banner so the
        artifact still exists for batch pipelines.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(12, w // 28))

    drawn = 0

    if task == "planning":
        for step_num, action, target, (u, v) in _iter_planning_points(structured):
            color = _STEP_COLORS[(step_num - 1) % len(_STEP_COLORS)]
            x, y = _norm_to_pixel(u, v, w, h)
            label = f"{step_num}.{action}({target})"
            _draw_point(draw, x, y, color, label=label, font=font)
            drawn += 1

    elif task == "affordance":
        u, v = structured.get("u"), structured.get("v")
        if u is not None and v is not None:
            x, y = _norm_to_pixel(float(u), float(v), w, h)
            label = "affordance"
            ga = structured.get("gripper_width")
            if ga is not None:
                label += f" w={ga:.2f}"
            _draw_point(draw, x, y, _STEP_COLORS[0], label=label, font=font)
            drawn += 1

    else:
        # Generic fallback: any (u, v) with values in [0, 1].
        u, v = structured.get("u"), structured.get("v")
        if isinstance(u, (int, float)) and isinstance(v, (int, float)) \
                and 0 <= u <= 1 and 0 <= v <= 1:
            x, y = _norm_to_pixel(float(u), float(v), w, h)
            _draw_point(draw, x, y, _STEP_COLORS[0], label=task, font=font)
            drawn += 1

    banner = f"task={task}  points={drawn}"
    bbox = draw.textbbox((4, 4), banner, font=font)
    draw.rectangle(bbox, fill="#000000bb")
    draw.text((4, 4), banner, fill="#ffffff", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
