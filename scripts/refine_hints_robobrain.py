#!/usr/bin/env python3
"""Refine plan.json affordance_hint fields using base RoboBrain (no LoRA).

The current training labels are templated ("the block", "an empty area on
the designated location") and occasionally hallucinated ("the handle of the
napkin"). This script uses the base RoboBrain-3DGS VLM (without the trained
LoRA adapter) as a teacher: it sees the actual scene image and rewrites each
step's hint into a richer, visually grounded noun phrase that mirrors the
inference-time PLANNING_SYSTEM_PROMPT rules.

For each episode under data/processed/<dataset>/episode_*/:
  - reads plan.json + rgb_0.png
  - for each step, queries the base model with task="general" using a
    prompt that asks for a scene-grounded, modifier-rich hint
  - writes plan_v2.json next to plan.json (does NOT modify the original)

Sharding for two-GPU parallel run (recommended):
    # Terminal 1 — GPU 0, datasets A
    CUDA_VISIBLE_DEVICES=0 python scripts/refine_hints_robobrain.py \\
        --datasets bridge,droid,fractal20220817_data --resume

    # Terminal 2 — GPU 1, datasets B
    CUDA_VISIBLE_DEVICES=1 python scripts/refine_hints_robobrain.py \\
        --datasets rh20t,rlbench,taco_play,jaco_play --resume

Quick checks:
    # Smoke test on 3 episodes, no disk write
    python scripts/refine_hints_robobrain.py \\
        --datasets rlbench --limit 3 --dry-run --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference_3dgs import UnifiedInference3DGS  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt — same 6 rules as the inference-time PLANNING_SYSTEM_PROMPT.
# Phrased as a self-contained instruction for a single VQA-style call.
# ---------------------------------------------------------------------------

INSTRUCTION_HEADER = """You are looking at a robotic manipulation scene. The robot is about to execute one step of a plan. Your job is to describe in a short noun phrase the SPECIFIC PART of the relevant object that the gripper should interact with.

Rules:
1. Look at the image and name a part that ACTUALLY EXISTS on the visible object. Do NOT invent parts. A bowl has no handle (say "the rim of the bowl"). A ball has no grip (say "the body of the ball"). A napkin has no handle (say "the corner of the napkin").
2. Add at least one discriminative modifier when possible: color, material, position (left/right/front/back/top), size, or relation to another visible object.
3. Pick a part that matches the action:
   - reach / release: the object itself or its body
   - grasp / pick / lift / pull / open / close: a grip-able feature (handle / knob / lid / neck / rim / edge)
   - place / transport: an empty area on the destination surface
   - insert: the top opening or slot of the destination
   - pour: the inside or mouth of the destination container
   - push / press: the center of the contact face
   - rotate / flip: the body of the object
   - wipe: the surface being wiped
4. For transport / place / insert / pour, describe the DESTINATION, not the held object.
5. Output a single noun phrase, 4-15 words, starting with "the". No punctuation except hyphens. No sentences. No JSON. No explanation.

Example outputs:
  the curved metal handle on the left side of the steel kettle
  the rim of the red bowl
  the lid of the green jar
  the wooden grip of the chef knife
  the empty area on the wooden tray between the two plates
  the top opening of the smaller glass jar
"""

DESTINATION_ACTIONS = {"transport", "place", "insert", "pour"}


def to_str(x) -> str:
    """Normalize plan fields that may be str, list[str], or None to a single string."""
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join(str(v) for v in x if v)
    return str(x)


def build_query(task: str, step: dict, scene_objects: list[str]) -> str:
    """Build the full prompt fed to the base model (task="general")."""
    action = step.get("action", "")
    target = to_str(step.get("target", ""))
    destination = to_str(step.get("destination"))
    old_hint = step.get("affordance_hint", "")

    ctx = [
        f"Task: {task}",
        f"Action: {action}",
        f"Target object: {target}",
    ]
    if destination:
        ctx.append(f"Destination: {destination}")
    if scene_objects:
        ctx.append(f"Other objects in scene: {', '.join(scene_objects)}")
    ctx.append(f"Original (templated) hint: \"{old_hint}\"")
    ctx.append("")
    ctx.append("Rewrite the hint following the rules. Output the noun phrase only.")
    return INSTRUCTION_HEADER + "\n" + "\n".join(ctx)


# ---------------------------------------------------------------------------
# Output cleanup + validation
# ---------------------------------------------------------------------------

def clean_output(raw: str) -> str:
    """Extract the noun phrase from a possibly noisy generation."""
    if not raw:
        return ""
    s = raw.strip()
    # Take only the first line — strip trailing reasoning if any
    s = s.split("\n", 1)[0].strip()
    # Strip surrounding quotes
    for q in ('"', "'", "`"):
        if s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    # Strip leading bullets / numbers
    while s and s[0] in "-*•1234567890.) ":
        s = s[1:].strip()
        if len(s) < 2:
            break
    # Strip trailing punctuation
    s = s.rstrip(".,;:!?")
    return s.strip()


def validate_hint(hint: str, target: str, destination: str | None) -> tuple[bool, str]:
    """Return (ok, reason). Same rules as the GPT-4o variant."""
    if not hint:
        return False, "empty"
    h = hint.lower()
    n_words = len(h.split())
    if n_words < 2:
        return False, "too_short"
    if n_words > 25:
        return False, "too_long"
    relevant = (destination or target).lower().replace("_", " ")
    relevant_tokens = [t for t in relevant.split() if len(t) > 2]
    if relevant_tokens and not any(tok in h for tok in relevant_tokens):
        return False, f"missing_token:{relevant}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(model: UnifiedInference3DGS, ep_dir: Path,
                    dry_run: bool, verbose: bool) -> dict:
    plan_path = ep_dir / "plan.json"
    out_path = ep_dir / "plan_v2.json"
    image_path = ep_dir / "rgb_0.png"

    if out_path.exists():
        return {"status": "skipped_exists", "refined": 0, "rejected": 0}
    if not plan_path.exists() or not image_path.exists():
        return {"status": "skipped_missing", "refined": 0, "rejected": 0}

    with open(plan_path) as f:
        plan = json.load(f)

    task = plan.get("task", "manipulation")
    scene_objects = plan.get("scene_objects", [])
    new_plan = copy.deepcopy(plan)
    refined = rejected = 0

    for step in new_plan.get("steps", []):
        old_hint = step.get("affordance_hint", "")
        prompt = build_query(task, step, scene_objects)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = model.inference(
                    text=prompt,
                    image=str(image_path),
                    task="general",
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=64,
                )
            raw = result.get("answer", "")
        except Exception as e:
            raw = ""
            reason = f"infer_error:{type(e).__name__}"
            new_hint = old_hint
            confidence = reason
            rejected += 1
            step["affordance_hint_original"] = old_hint
            step["affordance_hint"] = new_hint
            step["affordance_hint_confidence"] = confidence
            if verbose:
                print(f"    [{step.get('action','?'):10s}] ERROR: {e}")
            continue

        new_hint = clean_output(raw)
        target = to_str(step.get("target", ""))
        destination = to_str(step.get("destination")) if step.get("action") in DESTINATION_ACTIONS else None
        ok, reason = validate_hint(new_hint, target, destination)
        if ok:
            confidence = "ok"
            refined += 1
        else:
            new_hint = old_hint  # fall back
            confidence = f"rejected:{reason}"
            rejected += 1

        step["affordance_hint_original"] = old_hint
        step["affordance_hint"] = new_hint
        step["affordance_hint_confidence"] = confidence

        if verbose:
            tag = "OK" if confidence == "ok" else confidence.upper()
            print(f"    [{step.get('action','?'):10s}] {old_hint!r:40s} -> "
                  f"{new_hint!r}  [{tag}]")

    if not dry_run:
        # Write atomically so a crash mid-write can't corrupt the file
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(new_plan, f, indent=2, ensure_ascii=False)
        tmp.replace(out_path)

    return {"status": "ok", "refined": refined, "rejected": rejected}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def gather_episodes(data_root: Path, datasets: list[str],
                    start: int | None, end: int | None,
                    limit: int | None) -> list[Path]:
    episodes: list[Path] = []
    for ds in datasets:
        ds_dir = data_root / ds
        if not ds_dir.exists():
            print(f"  skipping {ds}: not found", file=sys.stderr)
            continue
        eps = sorted(p for p in ds_dir.iterdir()
                     if p.is_dir() and p.name.startswith("episode_"))
        episodes.extend(eps)
    if start is not None:
        episodes = episodes[start:]
    if end is not None:
        episodes = episodes[: end - (start or 0)]
    if limit:
        episodes = episodes[:limit]
    return episodes


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-root", default="data/processed")
    p.add_argument("--datasets", default="rlbench",
                   help="Comma-separated list of dataset folder names. "
                        "Use 'all' for bridge,droid,fractal20220817_data,"
                        "jaco_play,rh20t,rlbench,taco_play.")
    p.add_argument("--model-path",
                   default="/home/edge/RoboBrain/models/RoboBrain2.5-8B-NV")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Run inference but do not write plan_v2.json")
    p.add_argument("--resume", action="store_true",
                   help="Default behavior: existing plan_v2.json files are "
                        "skipped. Flag is for explicitness; always on.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.datasets == "all":
        datasets = ["bridge", "droid", "fractal20220817_data",
                    "jaco_play", "rh20t", "rlbench", "taco_play"]
    else:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    episodes = gather_episodes(
        Path(args.data_root), datasets,
        args.start, args.end, args.limit,
    )
    print(f"Datasets: {datasets}")
    print(f"Episodes to consider: {len(episodes)}")
    if not episodes:
        return

    # Pre-count how many already have plan_v2.json so user knows true workload
    todo = [ep for ep in episodes if not (ep / "plan_v2.json").exists()]
    print(f"Episodes already refined (will skip): {len(episodes) - len(todo)}")
    print(f"Episodes remaining to process: {len(todo)}")

    print("\nLoading base RoboBrain (no LoRA, no trained 3D branch) ...")
    model = UnifiedInference3DGS(
        model_id=args.model_path,
        checkpoint=None,
        mode="lora",
    )
    print()

    t0 = time.time()
    totals = {"ok": 0, "skipped": 0, "refined": 0, "rejected": 0}

    for i, ep in enumerate(episodes):
        stats = process_episode(model, ep, args.dry_run, args.verbose)
        if stats["status"] == "ok":
            totals["ok"] += 1
            totals["refined"] += stats["refined"]
            totals["rejected"] += stats["rejected"]
        else:
            totals["skipped"] += 1

        # Progress every 10 episodes (or per-episode in verbose mode)
        if args.verbose or (i + 1) % 10 == 0 or (i + 1) == len(episodes):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            remaining = (len(episodes) - i - 1) / max(rate, 1e-6)
            print(f"[{i+1}/{len(episodes)}] {ep.parent.name}/{ep.name} "
                  f"{stats['status']} refined={stats['refined']} "
                  f"rejected={stats['rejected']} | "
                  f"{rate:.2f} eps/s, ETA {remaining/60:.1f} min")

    print("\n=== Totals ===")
    print(json.dumps(totals, indent=2))
    print(f"Wall time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
