#!/usr/bin/env python3
"""Pilot verification for Stage 0.

Samples N episodes per non-RLBench dataset from train.json, runs:
  1. GPT V2 plan generation (keyframe-grounded)
  2. RoboBrain 2.5 affordance refinement
  3. Visual overlay (GPT blue vs RoboBrain red)
  4. Per-dataset quality report

Writes outputs to:
  - data/processed/{dataset}/{ep}/plan.json  (final, refined)
  - data/processed/{dataset}/{ep}/plan_before_robobrain.json  (GPT backup)
  - result/pilot/{dataset}/{episode}.png
  - result/pilot/pilot_report.md

Usage:
    CUDA_VISIBLE_DEVICES=1 HF_HOME=~/.cache/huggingface \
      conda run -n robobrain_3dgs python scripts/pilot_verify_datasets.py --n 15
"""
import argparse
import contextlib
import copy
import io
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
DEFAULT_ROBOBRAIN_PATH = "/home/edge/Embodied/models/RoboBrain2.5-8B-NV"


def model_to_tag(model_name: str) -> str:
    """Convert 'gpt-5.4' -> 'gpt5_4', 'gpt-4o-mini' -> 'gpt4omini' for use in paths."""
    return model_name.replace("-", "").replace(".", "_")

TARGET_DATASETS = [
    "bridge", "droid", "fractal20220817_data",
    "taco_play", "rh20t", "jaco_play",
]

# Path wiring for imports
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT.parent / "RoboBrain2.5"))

from regenerate_plans_keyframe import generate_plan_keyframe, API_KEY, BASE_URL  # noqa: E402
from refine_affordance_robobrain import extract_point, resolve_query  # noqa: E402
from verify_affordance_robobrain import draw_comparison  # noqa: E402
from inference import UnifiedInference  # noqa: E402
from openai import OpenAI  # noqa: E402


def sample_episodes(n_per_dataset: int, seed: int = 42) -> dict[str, list[dict]]:
    """Sample episodes from train.json, skipping those already with plan.json."""
    rng = random.Random(seed)
    with open(SPLITS_DIR / "train.json") as f:
        train = json.load(f)

    by_ds: dict[str, list[dict]] = {}
    for ep in train["episodes"]:
        by_ds.setdefault(ep["dataset"], []).append(ep)

    selected = {}
    for ds in TARGET_DATASETS:
        if ds not in by_ds:
            print(f"  WARN: {ds} not in train.json")
            continue
        candidates = [
            e for e in by_ds[ds]
            if not (DATA_ROOT / ds / e["episode_id"] / "plan.json").exists()
        ]
        if not candidates:
            print(f"  WARN: {ds} has no un-annotated episodes in train.json")
            selected[ds] = []
            continue
        selected[ds] = rng.sample(candidates, min(n_per_dataset, len(candidates)))
    return selected


def gpt_worker(ds: str, ep_info: dict, gpt_client: OpenAI) -> tuple[str, str, dict | None]:
    ep_dir = DATA_ROOT / ds / ep_info["episode_id"]
    meta_path = ep_dir / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    plan = generate_plan_keyframe(gpt_client, ep_dir, meta)
    return (ds, ep_info["episode_id"], plan)


def rb_refine_plan(rb_model: UnifiedInference, ep_dir: Path, plan: dict) -> dict:
    rgb_path = ep_dir / "rgb_0.png"
    cache: dict[str, tuple[float, float] | None] = {}
    for step in plan["steps"]:
        query = resolve_query(step)
        if not query:
            continue
        if query in cache:
            new_aff = cache[query]
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = rb_model.inference(
                        text=query.replace("_", " "),
                        image=str(rgb_path),
                        task="pointing",
                        do_sample=False,
                        temperature=0.0,
                    )
                new_aff = extract_point(result["answer"])
            except Exception:
                new_aff = None
            cache[query] = new_aff
        if new_aff is not None:
            step["affordance"] = list(new_aff)
    return plan


def write_report(out_path: Path, stats: dict, n_per_ds: int):
    lines = [
        "# Pilot Verification Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Sample size: {n_per_ds} episodes per dataset\n",
        "\n## Per-dataset results\n",
        "| Dataset | GPT success | Avg steps | Mean Δ (GPT vs RB) | Visual dir |",
        "|---------|------------:|----------:|-------------------:|:-----------|",
    ]
    for ds, s in sorted(stats.items()):
        success = f"{s['ok']}/{s['ok'] + s['fail']}"
        avg_steps = f"{sum(s['n_steps']) / len(s['n_steps']):.1f}" if s["n_steps"] else "—"
        avg_delta = f"{sum(s['aff_delta']) / len(s['aff_delta']):.3f}" if s["aff_delta"] else "—"
        lines.append(f"| {ds} | {success} | {avg_steps} | {avg_delta} | `result/pilot/{ds}/` |")

    lines.extend([
        "\n## Interpretation\n",
        "- **GPT success**: V2 prompt JSON parse + structural validation pass rate.",
        "- **Avg steps**: typical plan complexity (2-6 is healthy).",
        "- **Mean Δ**: L2 distance between GPT and RoboBrain affordance in normalized coords.",
        "  - Δ < 0.10: RB and GPT roughly agree (both confident or both wrong)",
        "  - 0.10 ≤ Δ < 0.30: meaningful disagreement, **visual review critical**",
        "  - Δ ≥ 0.30: one of them is probably hallucinating",
        "\n## Next decisions\n",
        "- If any dataset has GPT success < 90%: investigate prompt/image compatibility",
        "- If a dataset's mean Δ is small AND visual shows GPT correct: skip RB refine for it",
        "- If a dataset's mean Δ is small AND visual shows RB correct: OK, RB matches GPT locally",
        "- If a dataset's mean Δ is large AND RB correct: RB refine clearly beneficial",
        "- If a dataset's mean Δ is large AND RB wrong: **exclude RB refine for this dataset**",
    ])
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=15, help="Episodes per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6, help="GPT concurrent workers")
    parser.add_argument("--model-path", default=DEFAULT_ROBOBRAIN_PATH,
                        help="RoboBrain model path")
    parser.add_argument("--gpt-model", default="gpt-4o-mini",
                        help="GPT model name (e.g. gpt-4o-mini, gpt-5.4)")
    parser.add_argument("--tag", default=None,
                        help="Output tag. Defaults to derived from --gpt-model.")
    args = parser.parse_args()

    tag = args.tag or model_to_tag(args.gpt_model)
    out_dir = PROJECT_ROOT / "result" / f"pilot_{tag}"
    backup_name = f"plan_{tag}_before_rb.json"

    # Monkey-patch the imported MODEL constant so generate_plan_keyframe uses it
    import regenerate_plans_keyframe as rpk
    rpk.MODEL = args.gpt_model

    print("=" * 70)
    print(f"  Pilot: {args.n} ep × {len(TARGET_DATASETS)} datasets")
    print(f"  GPT model: {args.gpt_model}")
    print(f"  Tag: {tag}    Output: {out_dir}")
    print(f"  GPT backup filename: {backup_name}")
    print("=" * 70)

    # --- Stage A: Sample ---
    selected = sample_episodes(args.n, args.seed)
    all_eps = [(ds, e) for ds, eps in selected.items() for e in eps]
    total = len(all_eps)
    print(f"\nSampled {total} episodes:")
    for ds, eps in selected.items():
        print(f"  {ds:<40} {len(eps)}")

    if total == 0:
        print("Nothing to do.")
        return

    # --- Stage B: GPT V2 generation ---
    print(f"\n[Stage B] GPT V2 generation (workers={args.workers}) ...")
    gpt_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    gpt_results: dict[tuple[str, str], dict | None] = {}
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(gpt_worker, ds, ep, gpt_client) for ds, ep in all_eps]
        for i, fut in enumerate(as_completed(futures), 1):
            ds, ep_id, plan = fut.result()
            gpt_results[(ds, ep_id)] = plan
            if plan:
                with open(DATA_ROOT / ds / ep_id / "plan.json", "w") as f:
                    json.dump(plan, f, indent=2, ensure_ascii=False)
                ok += 1
            else:
                fail += 1
            if i % 10 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  [{i}/{total}] ok={ok} fail={fail}  {elapsed:.0f}s  "
                      f"rate={i / elapsed:.1f} ep/s")
    print(f"  Stage B done: {ok}/{total} ok, {time.time() - t0:.0f}s")

    if ok == 0:
        print("All GPT calls failed; aborting before RoboBrain.")
        return

    # --- Stage C: RoboBrain load ---
    print(f"\n[Stage C] Loading RoboBrain2.5 ...")
    rb_model = UnifiedInference(args.model_path, device_map="auto")
    print("  Loaded.")

    # --- Stage D: RoboBrain refine ---
    print(f"\n[Stage D] RoboBrain affordance refine ...")
    refined_results: dict[tuple[str, str], tuple[dict, dict]] = {}
    t0 = time.time()
    succ = 0
    for i, ((ds, ep_id), gpt_plan) in enumerate(gpt_results.items(), 1):
        if not gpt_plan:
            continue
        ep_dir = DATA_ROOT / ds / ep_id
        refined = rb_refine_plan(rb_model, ep_dir, copy.deepcopy(gpt_plan))
        refined_results[(ds, ep_id)] = (gpt_plan, refined)
        succ += 1
        # Backup with tag so multiple model runs can coexist
        backup = ep_dir / backup_name
        with open(backup, "w") as f:
            json.dump(gpt_plan, f, indent=2, ensure_ascii=False)
        with open(ep_dir / "plan.json", "w") as f:
            json.dump(refined, f, indent=2, ensure_ascii=False)
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{ok}] refined  {elapsed:.0f}s")
    print(f"  Stage D done: {succ} refined, {time.time() - t0:.0f}s")

    # --- Stage E: Visualize + Stats ---
    print(f"\n[Stage E] Visualizing & computing stats ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for (ds, ep_id), plan in gpt_results.items():
        s = stats.setdefault(ds, {"ok": 0, "fail": 0, "n_steps": [], "aff_delta": []})
        if not plan:
            s["fail"] += 1
            continue
        s["ok"] += 1
        s["n_steps"].append(len(plan["steps"]))

        if (ds, ep_id) not in refined_results:
            continue
        gpt_plan, refined = refined_results[(ds, ep_id)]
        gpt_points = [tuple(st.get("affordance", [0.5, 0.5])) for st in gpt_plan["steps"]]
        rb_points = [tuple(st.get("affordance", [0.5, 0.5])) for st in refined["steps"]]
        for gp, rp in zip(gpt_points, rb_points):
            s["aff_delta"].append(((gp[0] - rp[0]) ** 2 + (gp[1] - rp[1]) ** 2) ** 0.5)

        ds_out = out_dir / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        draw_comparison(
            DATA_ROOT / ds / ep_id / "rgb_0.png",
            refined["steps"], gpt_points, rb_points,
            ds_out / f"{ep_id}.png",
        )
    print(f"  Images saved to {out_dir}/")

    # --- Stage F: Report ---
    report_path = out_dir / "pilot_report.md"
    write_report(report_path, stats, args.n)
    print(f"\n[Report] {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<40} {'GPT ok':>8} {'avg steps':>10} {'mean Δ':>10}")
    for ds, s in sorted(stats.items()):
        total_ds = s["ok"] + s["fail"]
        avg_st = sum(s["n_steps"]) / len(s["n_steps"]) if s["n_steps"] else 0
        avg_d = sum(s["aff_delta"]) / len(s["aff_delta"]) if s["aff_delta"] else 0
        print(f"{ds:<40} {s['ok']:>4}/{total_ds:<4} {avg_st:>10.1f} {avg_d:>10.3f}")


if __name__ == "__main__":
    main()
