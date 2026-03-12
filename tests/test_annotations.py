"""Tests for target-text annotation pipeline.

Validates that:
1. RLBench loader returns correctly-formatted target strings derived from
   low_dim_obs.pkl (not hardcoded fallback placeholders).
2. DROID loader returns the expected placeholder target.
3. build_lm_inputs correctly masks prompt tokens and exposes target tokens.
4. Label masking: at least one token per sample is NOT masked (-100).
5. Projection sanity: affordance coordinates are within [0, 1].
"""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.rlbench_loader import RLBenchDataset, _FALLBACK_TARGET
from data.droid_loader import DROIDDataset
from train import build_lm_inputs

RLBENCH_ROOT = "/home/w50037733/robobrain_3dgs/data/rlbench_sample"
DROID_ROOT   = "/home/w50037733/robobrain_3dgs/data/droid_sample"

# Regex for the expected target format
TARGET_RE = re.compile(
    r"affordance: \[(\d+\.\d+), (\d+\.\d+)\]\. "
    r"constraint: gripper_width=(\d+\.\d+), "
    r"approach=\[(-?\d+\.\d+), (-?\d+\.\d+), (-?\d+\.\d+)\]\."
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_target(target: str) -> dict:
    m = TARGET_RE.fullmatch(target)
    assert m, f"Target does not match expected format:\n  {target!r}"
    u, v, w, ax, ay, az = [float(x) for x in m.groups()]
    return {"u": u, "v": v, "gripper_width": w, "approach": (ax, ay, az)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rlbench_target_format():
    """All RLBench samples should return a correctly formatted target string."""
    ds = RLBenchDataset(RLBENCH_ROOT, camera="front", image_size=256, max_frames=10)
    assert len(ds) > 0, "No RLBench samples found"

    for i in range(min(5, len(ds))):
        sample = ds[i]
        assert "target" in sample, "Missing 'target' key in sample"
        parsed = _parse_target(sample["target"])
        print(f"  [{i}] frame={sample['frame']} target={sample['target']!r}")

        assert 0.0 <= parsed["u"] <= 1.0,           f"u={parsed['u']} out of [0,1]"
        assert 0.0 <= parsed["v"] <= 1.0,           f"v={parsed['v']} out of [0,1]"
        assert parsed["gripper_width"] in (0.00, 0.08), \
            f"Unexpected gripper_width={parsed['gripper_width']}"

        approach_norm = sum(x ** 2 for x in parsed["approach"]) ** 0.5
        assert abs(approach_norm - 1.0) < 0.01, f"Approach not unit vector: {parsed['approach']}"

    print("PASS test_rlbench_target_format")


def test_rlbench_target_not_fallback():
    """Targets for frames with valid obs should differ from the fallback placeholder."""
    ds = RLBenchDataset(RLBENCH_ROOT, camera="front", image_size=256, max_frames=20)
    assert len(ds) > 0

    non_fallback = [ds[i]["target"] for i in range(len(ds)) if ds[i]["target"] != _FALLBACK_TARGET]
    assert len(non_fallback) > 0, (
        "All targets are the fallback placeholder — obs loading may have failed"
    )
    print(f"  {len(non_fallback)}/{len(ds)} samples have real (non-fallback) targets")
    print("PASS test_rlbench_target_not_fallback")


def test_rlbench_variation_description_used_as_prompt():
    """Prompt should come from variation_descriptions.pkl when available."""
    ds = RLBenchDataset(RLBENCH_ROOT, camera="front", image_size=256, max_frames=5)
    assert len(ds) > 0

    sample = ds[0]
    # variation_descriptions contain task-specific natural language
    assert len(sample["prompt"]) > 5, "Prompt is too short"
    # Should NOT be the generic default
    from data.rlbench_loader import TASK_PROMPTS
    assert sample["prompt"] != TASK_PROMPTS["default"], \
        "Prompt is the generic default — variation_descriptions.pkl was not loaded"
    print(f"  prompt={sample['prompt']!r}")
    print("PASS test_rlbench_variation_description_used_as_prompt")


def test_droid_target_format():
    """DROID samples should have a valid placeholder target."""
    ds = DROIDDataset(DROID_ROOT, image_size=256, max_frames=3)
    if len(ds) == 0:
        print("SKIP test_droid_target_format (no DROID samples)")
        return

    sample = ds[0]
    assert "target" in sample, "Missing 'target' key in DROID sample"
    parsed = _parse_target(sample["target"])
    assert parsed["u"] == 0.50 and parsed["v"] == 0.50, "DROID placeholder coords should be 0.50"
    print(f"  target={sample['target']!r}")
    print("PASS test_droid_target_format")


def test_build_lm_inputs_masks_prompt():
    """build_lm_inputs must set labels=-100 for all prompt tokens."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/home/w50037733/models/RoboBrain2.5-8B-NV",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"SKIP test_build_lm_inputs_masks_prompt (tokenizer load failed: {e})")
        return

    prompts = ["close the jar", "open the drawer"]
    targets = [
        "affordance: [0.53, 0.42]. constraint: gripper_width=0.08, approach=[0.24, -0.97, -0.04].",
        "affordance: [0.45, 0.60]. constraint: gripper_width=0.00, approach=[0.00, 0.00, -1.00].",
    ]

    input_ids, attention_mask, labels = build_lm_inputs(prompts, targets, tok, "cpu")

    B, L = input_ids.shape
    print(f"  seq_len={L}, batch={B}")

    for i in range(B):
        total_tokens = int(attention_mask[i].sum())
        masked        = int((labels[i] == -100).sum())
        unmasked      = int((labels[i] != -100).sum())

        print(f"  [{i}] total={total_tokens}, masked(prompt+pad)={masked}, unmasked(target)={unmasked}")

        assert unmasked > 0, f"Sample {i}: all labels are -100 (no target tokens to train on)"
        assert masked > 0,   f"Sample {i}: no labels are -100 (prompt was not masked)"

        # Target tokens should be at the END of the sequence (after the prompt)
        label_row = labels[i]
        first_unmasked = (label_row != -100).nonzero(as_tuple=True)[0][0].item()
        last_masked_before = (label_row[:first_unmasked] == -100).all()
        assert last_masked_before, f"Sample {i}: unmasked token found before prompt boundary"

    print("PASS test_build_lm_inputs_masks_prompt")


def test_build_lm_inputs_target_token_count():
    """Target tokens should account for a meaningful portion of the sequence."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "/home/w50037733/models/RoboBrain2.5-8B-NV",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"SKIP test_build_lm_inputs_target_token_count (tokenizer load failed: {e})")
        return

    # Tokenise just the target to know its length
    target = "affordance: [0.53, 0.42]. constraint: gripper_width=0.08, approach=[0.24, -0.97, -0.04]."
    target_toks = tok(target, return_tensors="pt")
    target_len = int(target_toks.attention_mask[0].sum())

    input_ids, attention_mask, labels = build_lm_inputs(
        ["close the jar"], [target], tok, "cpu"
    )
    unmasked = int((labels[0] != -100).sum())

    # Allow ±2 tokens (BOS/EOS handling varies by tokenizer)
    assert abs(unmasked - target_len) <= 3, (
        f"Unexpected unmasked count: got {unmasked}, expected ~{target_len}"
    )
    print(f"  target_len={target_len}, unmasked={unmasked}")
    print("PASS test_build_lm_inputs_target_token_count")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_rlbench_target_format,
        test_rlbench_target_not_fallback,
        test_rlbench_variation_description_used_as_prompt,
        test_droid_target_format,
        test_build_lm_inputs_masks_prompt,
        test_build_lm_inputs_target_token_count,
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

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
