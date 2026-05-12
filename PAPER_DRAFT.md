# GaussianBrain — ICLR Submission Draft (Abstract + Introduction)

> Draft v0.1.  Numbers are placeholders pending the W2 experiment sweep.
> Companion document to `PAPER_PLAN.md`.

---

## Title (candidate)

**GaussianBrain: Render-Supervised 3D Gaussian Tokens for Verifiable Embodied Planning in Vision-Language Models**

## Abstract

Vision-language models (VLMs) trained on internet image-text pairs lack the
geometric grounding needed for embodied manipulation: a model that scores well
on visual question answering can still misplace a grasp by tens of
centimeters, and its competence collapses as soon as the camera viewpoint
shifts.  We argue that the bottleneck is *representational*, not data: a
flat 2D token stream cannot encode the metric scene structure that
manipulation reasoning requires.

We propose **GaussianBrain**, a plug-in 3D representation that injects a small
set of 3D Gaussian tokens into a pretrained 8B VLM (Qwen3-VL / RoboBrain 2.5)
alongside its native visual tokens.  Three design choices make the integration
work without retraining the backbone:
(i) **RGBD-deterministic factorization** — Gaussian centers are obtained by
analytic depth back-projection, so the network only learns scale, rotation,
spherical-harmonic appearance, opacity and a per-Gaussian uncertainty;
(ii) **differentiable-rendering supervision** — an auxiliary photometric
+ depth + opacity loss routes geometry-aware gradients back into the VLM,
turning rendering quality into a representation regularizer;
(iii) **uncertainty-aware token budgeting** — farthest-point sampling is
re-weighted by the predicted variance so that the 64-token budget given to
the LLM concentrates on confident geometry.

To make the model's outputs *verifiable*, we further train the LLM on a
structured plan target: each manipulation step emits a part-aware affordance
hint, normalized pixel coordinates, and a typed list of geometric constraints
drawn from a five-category × three-role taxonomy (contact, spatial, pose,
direction, safety × completion, safety, progress).  This converts free-form
plan strings into machine-checkable predicates that a downstream executor or
runtime monitor can verify against the same 3D Gaussian field that produced
them.

We curate a 27 K-episode benchmark unifying twelve robotics datasets (RLBench,
DROID, Bridge V2, taco_play, ALOHA, rh20t, and seven Open X-Embodiment
subsets) and evaluate under a three-level generalization protocol that
isolates spatial, semantic, and *cross-camera* transfer.  Across the cross-
camera level — where conventional 2D-only fine-tuning degrades sharply — we
observe that the 3D branch closes a substantial fraction of the gap to the
seen-camera setting, while affordance L2 on the seen split improves over a
LoRA-only baseline and over a point-cloud variant without rendering
supervision.  We release the data pipeline, training code, and checkpoints.

---

## 1. Introduction

### 1.1 Motivation

Robotic manipulation requires a model to ground a natural-language goal in
the metric geometry of a physical scene: where exactly to grasp, in what
direction to approach, under what contact and pose constraints.  Recent
vision-language-action models — OpenVLA, RT-2, π₀, RoboBrain — have made
striking progress on language conditioning, but their visual stack is still
fundamentally **two-dimensional**: a frozen ViT encodes pixels, the language
model decodes actions or affordances, and any notion of 3D structure must be
re-learned from monocular cues at the cost of view invariance and metric
accuracy.

The cost of this 2D bias surfaces in two failure modes.  First, *spatial
imprecision*: even when an 8B VLM names the correct object, the (u, v)
coordinate it emits is biased toward visual centroids and is unreliable for
contact-rich grasps such as handles, rims and lids.  Second, *view
brittleness*: a model trained from a front camera loses substantial accuracy
when evaluated from a shoulder or wrist view of the same scene, because the
2D feature manifold does not separate geometry from appearance.  Both failure
modes are aggravated by the fact that VLM pretraining cannot include
calibrated RGB-D observations at scale, so any 3D inductive bias must be
introduced at fine-tuning time.

### 1.2 Why existing 3D-VLM approaches are not enough

A growing family of works injects 3D information into language models: 3D-LLM
and LEO consume point clouds, ManipLLM and ShapeLLM operate on object-level
3D primitives, and recent papers (LangSplat, Gaussian Grouping, LERF) attach
language fields to 3D-Gaussian-Splatting reconstructions.  Three limitations
prevent these designs from being used as a drop-in upgrade to a manipulation
VLM:

1. **Optimization-heavy 3D representations.** 3DGS-based scene fields
   typically require per-scene optimization or large feed-forward
   reconstructors; both are at odds with the single-frame, real-time regime
   of a manipulation policy.
2. **Lack of cross-modal supervision back into the VLM.**  Existing systems
   either keep the 3D branch frozen or treat it as a pre-encoded input; few
   provide a gradient signal that *teaches the VLM* to use the 3D modality.
3. **Unstructured outputs.**  Affordance predictions are usually pixel points
   or free-form plans, neither of which a downstream executor can verify.
   When the model is wrong, there is no way to detect it short of running
   the action.

### 1.3 Our approach

GaussianBrain attacks all three limitations by making the 3D branch
*lightweight, jointly supervised, and structurally-output-typed*.

**A deterministic-Gaussian feed-forward 3D head.**  Given one RGB-D frame and
intrinsics, we back-project depth to obtain Gaussian centers analytically
and train a small CNN to predict the remaining parameters (scale,
quaternion rotation, spherical-harmonic colour, opacity, and a per-Gaussian
variance).  This reduces 3D representation learning to per-pixel regression
and runs at a few milliseconds per frame.

**A PointNet++ tokenizer with uncertainty-aware FPS.**  A three-level
hierarchical aggregator compresses 2 048 Gaussians into 64 tokens; the
farthest-point-sampling step is biased so that high-variance Gaussians are
sampled less often, concentrating the LLM's budget on confident geometry.
Tokens are projected to the LLM's hidden dimension, tagged with a learnable
type embedding, and prepended to the standard image+text sequence.

**Differentiable rendering as a representation regularizer.**  We train the
3D branch jointly with the LLM under a combined objective
`L = L_lm + 0.3 · L_render`, where `L_render` is a photometric + depth +
opacity + variance loss evaluated on a tile-free differentiable rasterizer.
This is the key mechanism by which geometric supervision flows back through
the GS tokens into the LLM's gradient path, encouraging the language head to
make use of geometrically meaningful features rather than overfit to 2D
patterns.

**A typed plan output for verification.**  The LLM is supervised to emit
each manipulation step as a quadruple of *(action, target, part-aware
affordance hint, structured constraints)* with constraints drawn from a
five-category × three-role taxonomy.  Because every predicate is grounded in
either 3D distances or pose primitives, downstream verifiers — including a
LangSAM-based test-time refinement we report as an ablation — can be wired
directly to the same Gaussian field the model emitted.

### 1.4 Empirical results (preview)

We train two configurations: a parameter-efficient LoRA variant
(36.8 M trainable, 0.42 % of the backbone) and a partial full fine-tune that
unfreezes the last eight LLM layers.  Across the three-level evaluation
protocol — same-task / unseen-task / cross-camera — the 3D branch yields its
largest gains on the **cross-camera** split, the level that most directly
tests view invariance and on which a 2D-only LoRA fine-tune is brittle.
Ablations show that (a) removing the rendering loss collapses cross-camera
performance to the 2D-only baseline, (b) replacing learned Gaussians with a
raw point cloud loses a substantial fraction of the gain at matched token
budget, and (c) the uncertainty-weighted FPS contributes a measurable margin
that grows with scene complexity.  A test-time LangSAM refinement, conditioned
on the model's emitted part-aware hint rather than the bare object name,
improves grasp accuracy on multi-instance scenes without retraining.

### 1.5 Contributions

* **A render-supervised 3D Gaussian token branch** that plugs into a
  pretrained 8B VLM, with deterministic Gaussian centers and an
  uncertainty-aware token budget.  The branch and its training objective
  together transform geometric reconstruction quality into a regularizer
  on the LLM's visual reasoning.

* **A structured manipulation-plan supervision target** organized around a
  five-category × three-role constraint taxonomy and a part-aware
  affordance-hint convention, making model outputs *verifiable predicates*
  rather than free-form text.

* **A 27 K-episode unified data pipeline** spanning twelve heterogeneous
  robotics datasets with mixed native and pseudo depth, together with a
  three-level generalization evaluation protocol (seen / unseen / cross-
  camera) that isolates the contribution of the 3D branch.

* **Open release** of the data pipeline, training and inference code, and
  trained checkpoints, including the LangSAM-based test-time refinement
  used in our ablations.

The remainder of the paper is organized as follows.  §2 reviews 3D-enhanced
VLMs and embodied planning literatures.  §3 details the GaussianBrain
architecture and the joint training objective.  §4 describes the data
pipeline and plan supervision schema.  §5 reports the three-level
benchmark, ablations, and qualitative analyses.  §6 discusses limitations
(single-frame inputs, dependence on depth quality) and outlines future work
on temporal Gaussian fields and learned constraint verification.
