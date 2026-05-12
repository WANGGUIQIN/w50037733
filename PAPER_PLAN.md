# RoboBrain-3DGS — ICLR Paper Plan

> Status: planning document. No code changes implied; this file maps existing
> training/inference pipelines onto an ICLR submission narrative, with the
> remaining work being **experiments + writing only**.

---

## 1. Training Pipeline (data → model → loss)

### 1.1 Data preparation (three offline stages, already done)

```
Raw data (12 robotics datasets: RLBench, DROID, Bridge, taco_play, ALOHA,
          rh20t, jaco_play, nyu_franka, berkeley_cable, furniture_bench,
          fractal20220817, utokyo_xarm_bimanual)
     │
     ▼
─── Stage 1: scripts/data_pipeline/run_pipeline.py ───────────────────────
  - downloaders/*.py       per-episode streaming (no raw retained on disk)
  - keyframe_extractor.py  action-delta Gaussian smoothing + greedy peak pick
                           5 keyframes per episode, min_gap = T/6
  - depth_generator.py     native depth where available, otherwise
                           Depth Anything V2 Large pseudo-depth normalized
                           to [0.01, 5.0]
  - episode_saver.py       unified 256×256 layout, RGB + depth + meta.json
     │
     ▼
data/processed/{ds}/episode_*/{rgb_0..4.png, depth_0..4.npy, meta.json}
     │
─── Stage 2: scripts/generate_plans.py / regenerate_plans_keyframe.py ────
  - GPT-4o-mini (default) / GPT-4o, temperature 0.2–0.3, 2× retry
  - few-shot system prompt enforces JSON-only output
  - 12 manipulation primitives:
      reach, grasp, transport, place, push, pull,
      insert, pour, rotate, release, flip, wipe
  - 5 constraint categories × 3 role tags
  - multi-frame variant injects per-step affordance disambiguation
     │
     ▼ plan_original.json
─── Stage 3: scripts/refine_affordance_robobrain.py ──────────────────────
  - RoboBrain2.5-8B-NV pointing task → (u, v) ∈ [0, 1000]²
  - per-episode query cache (one call per unique object phrase)
  - pixel_to_3d using depth_0 + intrinsics → affordance_3d
     │
     ▼ plan.json   (= training target)
─── Stage 3.5: scripts/backfill_affordance_3d.py ─────────────────────────
  - pure NumPy back-projection to recover missing affordance_3d fields
```

**Scale:** 27,061 episodes / 101,205 (RGB, 3D-point) supervision pairs
across 12 datasets (6 native depth, 6 pseudo).

### 1.2 Training entry: `train.py` (unified LoRA / Full interface)

```
─── build_datasets() ──────────────────────────
  UnifiedDataset (data/unified_loader.py)
    Reads plan.json → text target format
        "Step N: action(target)
           affordance_hint: <part-aware noun phrase>
           affordance: [u=..., v=...], approach: [...]
           contact|spatial|pose|direction|safety: pred(args) [role]
           done_when: <condition>
         <END_OF_PLAN>"
    Output dict: {rgb [3,256,256], depth [1,256,256], intrinsics [3,3],
                   prompt, target, task_type, ...}

─── build_model_{lora|full}() ─────────────────
  RoboBrain3DGS_VLM.from_pretrained(BAAI/RoboBrain2.5-8B-NV)
    ├─ vlm.model.visual            frozen Qwen3-VL ViT
    ├─ vlm.model.language_model    LoRA on q/k/v/o (r=16, α=32, dropout=0.05)
    ├─ vlm.lm_head                 LoRA-wrapped
    └─ 3D Branch (fully trainable, ~37M params)
        ├─ DepthToGaussian         RGBD CNN → 2048 Gaussians
        │     xyz = depth back-projection (deterministic)
        │     scale / rotation / SH / α / σ predicted
        │     uncertainty-aware FPS subsample
        ├─ GaussianEncoder         PointNet++ 3-level hierarchy
        │     2048 → 512 → 128 → 64 tokens
        ├─ GaussianTokenProjector  Linear → GELU → Linear → LN → 4096-d
        ├─ gs_type_embedding       1×1×4096 learnable parameter
        └─ CrossModalFusion (opt)  Self-Attn(3D) → Cross-Attn(3D, ViT-KV) ×2

─── build_optimizer() ─────────────────────────
  Two parameter groups (critical):
    - 3D branch params      lr_3d_branch = 1e-3
    - LoRA / LLM params     lr = 2e-4 (LoRA) | 5e-5 (full)
  AdamW + Cosine + linear warmup (warmup_ratio = 0.05)
  weight_decay = 0.01, max_grad_norm = 1.0

─── train_step() ──────────────────────────────
  Qwen3-VL chat template:
    messages = [system, user(image + text), assistant(target)]
  AutoProcessor → input_ids (with <image_pad>), pixel_values, image_grid_thw
  labels = input_ids; mask prompt + <image_pad> positions to -100

  Forward (models/robobrain_vlm.py):
    inputs_embeds = embed(input_ids)
    ViT(pixel_values, grid_thw) → 2D visual tokens, injected at <image_pad>
    encode_3d(rgb, depth, intrinsics, vit_tokens) → 64 fused 3D tokens
    [gs_tokens (+ gs_type_embedding)]  prepend  [text + image embeds]
    gs_labels = -100  (3D tokens contribute no LM loss)
    LLM(inputs_embeds, attention_mask) → hidden → lm_head → logits → CE

  Auxiliary loss (gs_renderer.GaussianRenderingLoss):
    L_render = 0.8 · L1(rendered_rgb, rgb)
             + 0.2 · (1 − SSIM)
             + 0.5 · L1(rendered_depth, depth)
             + 0.01 · |α|                     (opacity sparsity)
             + 0.1  · VarSplat(σ)              (uncertainty regularization)

  Total objective:
    L = L_lm + 0.3 · L_render

─── CheckpointManager ─────────────────────────
  Atomic writes, keep_last_n = 3, best/ symlink, per-checkpoint metadata.json
  checkpoint-N/
    3d_branch.pt          DepthToGaussian + GS Encoder + Projector + type_emb
    lora_adapter/         PEFT adapter ~60 MB                    [LoRA mode]
    vlm_trainable.pt      LLM trainable weights                  [Full mode]
    training_state.pt     optimizer + scheduler + step
```

### 1.3 Training recipes matrix

| Recipe | Config | Hardware | Trainable | VRAM/GPU |
|--------|--------|----------|-----------|----------|
| LoRA (single GPU) | `train_lora_5090.yaml` | 1× RTX 5090 | 36.8 M / 8.8 B (0.42 %) | ~20 GB |
| LoRA (multi-GPU) | `train_lora.yaml` / `train_lora_b200.yaml` | 4× B200 192 GB | 36.8 M | ~30 GB |
| LoRA Planning | `train_lora_planning.yaml` / `train_lora_b200_planning.yaml` | 4× B200 | 36.8 M | ~30 GB |
| Full (last 8 layers) | `train_full.yaml` (`unfreeze_llm_layers: 8`) | 4× A100 80 GB | 2.8 B (32 %) | ~60 GB |
| Full (all layers) | `train_full.yaml` (`-1`) | 4× A100 80 GB | 8.2 B (93 %) | ~28 GB (ZeRO-3 + CPU offload) |

DeepSpeed configs: `zero2_b200.json` (LoRA), `zero3.json` (full), `zero1_b200.json` (planning).

---

## 2. Inference Pipeline (three paths)

### 2.1 Path A — Core 3D-VLM inference

`inference_3dgs.py::UnifiedInference3DGS`

```
UnifiedInference3DGS(model_id, checkpoint, mode="lora"|"full")
   │   load base VLM → from_pretrained
   │   load 3d_branch.pt + lora_adapter (LoRA) or vlm_trainable.pt (full)
   ▼
.inference(image, depth=None, text, task=...)
   task ∈ {
     "general"      → generic VQA
     "pointing"     → RoboBrain native grounding, (u, v) ∈ [0, 1000]
     "affordance"   → "affordance: [u, v]. constraint: ..."
     "planning"     → multi-step plan text (affordance_hint + 5 constraints)
     "trajectory"   → multi-step trajectory
   }
   ▼
generate_with_3d():
   - 2D path injects ViT features at <image_pad> positions
   - depth provided → encode_3d → prepend 64 gs_tokens
   - autoregressive generation with KV-cache, EOS early-stop, pre-allocated buffers
```

### 2.2 Path B — RLBench closed-loop planning

`run_rlbench_planning.py`

```
RLBench episode (front_rgb / front_depth + variation descriptions)
   │
   ▼ load_episode()           decode RLBench native depth PNG
   ▼ run_planning(model, RGB path, task_desc)
        → UnifiedInference3DGS.inference(task="planning")
   ▼ extract_json()           parse generated text into structured plan
```

### 2.3 Path C — End-to-end with LangSAM refinement

`scripts/e2e_inference.py` (post-processing, currently inference-time only)

```
UnifiedInference3DGS.inference(..., task="planning") → structured plan
   │
   ▼   --refine-langsam flag (planning task only)
   ▼
refine_with_langsam(structured, image, depth, intrinsics, grounder, strategy)
   ├─ snapshot original affordance → affordance_lora (A/B retention)
   ├─ for each step:
   │     prompt = _build_prompt(step)
   │              prioritizes affordance_hint ("the handle of the red jar")
   │     GroundingSAM.predict(image, prompt)
   │         GroundingDINO (IDEA-Research/grounding-dino-tiny) → boxes + scores
   │         SAM (facebook/sam-vit-base) → masks
   │     _select_mask: GDino conf > 0.30 → coarse-uv ∈ mask → median-depth foreground
   │     extract_affordance(mask, depth, intr, strategy):
   │         centroid | inscribed_circle (distance_transform_edt argmax) | pca (short axis)
   │     back-projection → (u, v, x, y, z, approach)
   │     refine_status ∈ {ok, no_mask, fallback}
   └─ overwrite step["affordance"], step["approach"]; preserve _refined / _lora
```

### 2.4 Evaluation entry

`evaluate.py` with `config/eval_split.yaml`

```
Three-level generalization split:
  L1 Seen Task      train ep 0–79 / test ep 80–99, same tasks, different poses
  L2 Unseen Task    entire tasks held out from training
  L3 Cross Camera   train on front camera, test on left_shoulder / right_shoulder / wrist

4 model configurations × 3 levels:
  #1 Baseline RGB + text (zero-shot Qwen3-VL)
  #2 Baseline text-only
  #3 Trained, no 3D branch
  #4 Trained + 3D branch ← ours

Metrics:
  affordance_l2        L2 on [u, v]                    (lower better)
  gripper_width_mae    MAE on gripper width            (lower better)
  approach_cos_sim     cosine similarity, approach     (higher better)
  valid_format_pct     % outputs matching schema       (higher better)
  lm_loss / ppl        cross-entropy / perplexity      (lower better)
```

---

## 3. Innovation Summary (claims supportable by current code)

| # | Claim | Implementation | Strength | Nearest related work |
|---|-------|----------------|----------|-----------------------|
| 1 | Render-supervised 3D Gaussian tokens as a pluggable visual modality for pretrained VLMs | `robobrain_vlm.py` + `gs_renderer.py` | ★★★ | 3D-LLM, LEO, ManipLLM use point clouds / voxels; combining GS with rendering loss feedback to a VLM is novel |
| 2 | RGBD-deterministic Gaussian factorization: xyz from depth back-projection, the network only learns scale / rotation / SH / opacity / uncertainty | `depth_to_gaussian.py` | ★★ | pixelSplat / Splatter-Image; novelty is embedding into a VLM |
| 3 | VarSplat-style per-Gaussian uncertainty + uncertainty-aware FPS — high-σ Gaussians are down-weighted during token sampling | `depth_to_gaussian.py::_fps_select` | ★★ | VarSplat (3DGS reconstruction); first application to VLM token budgeting |
| 4 | Cross-modal fusion as alignment: 3D tokens query a ViT KV bank, landing 3D representations on the ViT feature manifold | `cross_modal_fusion.py` | ★ | LLaVA stage-1 / Q-Former; weak novelty in isolation |
| 5 | Structured constraint taxonomy as supervised LM target: 5 categories × 3 roles, LLM learns to emit **verifiable geometric predicates** | `unified_loader.py::_load_plan_target` + plan schema | ★★★ | More structured than RoboPoint's pure pointing; lighter than ReKep's program synthesis |
| 6 | affordance_hint + part-aware language conditioning: LLM learns to emit "the handle of …" before (u, v), grounding the visual head on language | `unified_loader.py:194-203` + plan schema | ★★ | Chain-of-pointing flavor (RoboPoint, PIVOT) |
| 7 | Two-LR co-training: 3D branch 1e-3, LoRA 2e-4, LLM full 5e-5 — fast convergence of new parameters without catastrophic forgetting | `train.py::build_optimizer` | ★ | Engineering practice |
| 8 | Joint optimization with differentiable rendering: L_lm + 0.3 · L_render flows geometry-aware gradients back into the VLM | `train.py::train_step` + `gs_renderer.py::GaussianRenderingLoss` | ★★★ | Pair with claim #1 — cleanest story line |
| 9 | 12-dataset unified pipeline with pseudo-depth completion: 27K episodes mixing sim / real / bimanual / long-horizon | `scripts/data_pipeline/*` + `data/unified_loader.py` | engineering | Data-card appendix |
| 10 | Three-level generalization evaluation protocol: seen-task / unseen-task / cross-camera (the third level isolates the 3D-branch contribution) | `evaluate.py` + `eval_split.yaml` | ★★ | Methodological contribution |

> **LangSAM post-processing (Path C) is not a standalone contribution.** It is a
> well-trodden recipe (MOKA, RoboPoint, CoPa, PIVOT). However, paired with
> claims #5 and #6, it supports a clean A/B story: *"VLM training-target design
> produces better downstream LangSAM grounding."* Report it as a minor result
> in §5.3 of the paper.

---

## 4. Paper Narrative (single-shot draft)

### Title (candidate)

> **GaussianBrain: Render-Supervised 3D Gaussian Tokens for Verifiable Embodied Planning in Vision-Language Models**

### Three contributions (mapping back to existing code)

**C1 — Representation:**
We inject RGBD-deterministic 3D Gaussian tokens into a pretrained Qwen3-VL,
supervised by a differentiable rasterization auxiliary loss that propagates
geometry-aware gradients into a frozen ViT and LoRA-tuned LLM.
*Covers claims #1, #2, #3, #8.*

**C2 — Supervision:**
We design a 5-category × 3-role manipulation constraint taxonomy as LM
training targets, paired with part-aware affordance hints that the LLM learns
to emit jointly with pixel coordinates.
*Covers claims #5, #6.*

**C3 — Evaluation:**
We release a 27K-episode multi-dataset benchmark with three-level
generalization splits (seen / unseen / cross-camera), where the cross-camera
split isolates the contribution of the 3D branch.
*Covers claims #9, #10.*

### Section-to-code map

| Section | Source | What's left |
|---------|--------|-------------|
| §1 Intro | `docs/plans/2026-04-01-task-planning-design.md` | writing |
| §2 Related Work | — | writing (3D-LLM, LEO, MOKA, RoboPoint, π0, OpenVLA) |
| §3 Method | `models/*.py` + `train.py::train_step` | writing + architecture figure |
| §3.1 Gaussian Branch | `depth_to_gaussian.py` + `gs_encoder.py` | writing |
| §3.2 Render Loss | `gs_renderer.py` | writing |
| §3.3 Constraint Schema | `docs/plans/2026-04-01-constraint-taxonomy.md` + `unified_loader.py` | writing |
| §4 Data Pipeline | `docs/data-pipeline.md` + `scripts/data_pipeline/` | writing |
| §5 Experiments | `evaluate.py` + `eval_split.yaml` | **run experiments** |
| §5.1 Main Table | `evaluate.py` 4 × 3 grid | run |
| §5.2 Ablations | YAML flag flips only: `rendering_loss.enabled`, `predict_uncertainty`, `lambda_uncertainty`, `num_gs_tokens`, fusion on/off | run |
| §5.3 LangSAM Refinement | `scripts/e2e_inference.py` with `--refine-langsam` | run A/B |
| §5.4 Qualitative | `output_e2e/`, `output_postprocess/` already have visualizations | curate |

---

## 5. Four-Week Sprint (no code changes, experiments + writing only)

| Week | Work | Deliverables |
|------|------|--------------|
| **W1** | LoRA training × 3 epochs (4× B200, `train_lora.yaml`) + Full last-8-layer training (`train_full.yaml`) | 2 best checkpoints |
| **W2** | Full `evaluate.py` 4 × 3 grid + 5 ablations: render on/off, fusion on/off, σ on/off, `num_gs_tokens` ∈ {16, 64, 256}, `num_gaussians` ∈ {512, 2048} | Main table + ablation table |
| **W3** | `run_rlbench_planning.py` closed-loop success rate + `e2e_inference.py --refine-langsam` A/B + figure curation from `output_postprocess/` | Table 3 + Figures 4–6 |
| **W4** | Writing (abstract / intro / method / experiments / related work) + revision rounds | ICLR submission |

---

## 6. One-Line Summary

The current codebase already supports an ICLR-grade submission whose core
claim is: **render-supervised 3D Gaussian tokens + a structured constraint
taxonomy give an 8 B VLM geometry priors for cross-view manipulation
planning.** The remaining work is experiments and writing — no code changes
required.
