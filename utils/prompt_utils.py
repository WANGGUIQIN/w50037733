"""Prompt utilities for RoboBrain-3DGS training and inference.

Aligns with RoboBrain2.5's Qwen3-VL chat template protocol:
  - <|im_start|>system\n{system_prompt}<|im_end|>
  - <|im_start|>user\n{task_prompt}<|im_end|>
  - <|im_start|>assistant\n{target}<|im_end|>

Key design decisions:
  1. Use apply_chat_template() to match the pretrained model's distribution
  2. Task-specific prompt augmentation matching RoboBrain2.5's inference.py
  3. Label masking: only assistant response tokens contribute to loss
  4. System prompt is optional (the pretrained model supports it)

Reference: /home/w50037733/RoboBrain2.5/inference.py (official inference patterns)
"""

import re

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are RoboBrain, an embodied AI assistant specialized in robotic "
    "manipulation. Given an image of a scene with 3D spatial understanding, "
    "provide precise affordance predictions and manipulation constraints."
)

DEFAULT_TASK_TYPE = "affordance"

# Marker that separates prompt from assistant response in the chat template
_ASSISTANT_MARKER = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Task-specific prompt templates (matching RoboBrain2.5's inference.py)
# ---------------------------------------------------------------------------

TASK_TEMPLATES = {
    # Affordance prediction: our primary training task.
    "affordance": (
        "{text}. Please predict the affordance point and manipulation "
        "constraints for completing this task. Your answer should include: "
        "the affordance coordinates as [u, v] in normalized image space, "
        "gripper_width, and approach vector as [x, y, z]."
    ),
    # Pointing: RoboBrain2.5's pointing task (2D coordinate output)
    "pointing": (
        "{text}. Please provide its 2D coordinates. Your answer should be "
        "formatted as a tuple, i.e. [(x, y)], where the tuple contains the "
        "x and y coordinates of a point satisfying the conditions above."
    ),
    # Trajectory: RoboBrain2.5's 3D trajectory prediction
    "trajectory": (
        "Please predict 3D end-effector-centric waypoints to complete the "
        'task successfully. The task is "{text}". Your answer should be '
        "formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), "
        "...], where each tuple contains the x and y coordinates and the "
        "depth of the point."
    ),
    # Grounding: bounding box prediction
    "grounding": (
        "Please provide the bounding box coordinate of the region this "
        "sentence describes: {text}."
    ),
    # General VQA: pass through as-is
    "general": "{text}",
}


# ---------------------------------------------------------------------------
# Output parsing (shared by inference and evaluation)
# ---------------------------------------------------------------------------

_AFF_RE = re.compile(
    r"affordance[:\s]*\[([0-9.]+)[,\s]+([0-9.]+)\]", re.I,
)
_WID_RE = re.compile(r"gripper_width\s*=\s*([0-9.]+)", re.I)
_APP_RE = re.compile(
    r"approach\s*=\s*\[([-0-9.e+]+)[,\s]+([-0-9.e+]+)[,\s]+([-0-9.e+]+)\]",
    re.I,
)


def parse_affordance_output(text: str) -> dict:
    """Parse affordance output into structured fields.

    Expected format::

        affordance: [u, v]. constraint: gripper_width=X, approach=[x, y, z].

    Returns:
        dict with keys: u, v, gripper_width, approach (each None if not found).
    """
    out: dict = {"u": None, "v": None, "gripper_width": None, "approach": None}
    m = _AFF_RE.search(text)
    if m:
        out["u"] = float(m.group(1))
        out["v"] = float(m.group(2))
    m = _WID_RE.search(text)
    if m:
        out["gripper_width"] = float(m.group(1))
    m = _APP_RE.search(text)
    if m:
        out["approach"] = [float(m.group(i)) for i in range(1, 4)]
    return out


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def augment_prompt(text: str, task_type: str = DEFAULT_TASK_TYPE) -> str:
    """Apply task-specific prompt augmentation.

    Matches RoboBrain2.5's inference.py prompt patterns so the model
    sees the same style of instructions it was pretrained on.
    """
    template = TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["general"])
    return template.format(text=text)


def build_messages(
    user_text: str,
    assistant_text: str | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    task_type: str = DEFAULT_TASK_TYPE,
    image=None,
) -> list[dict]:
    """Build chat messages in Qwen3-VL format.

    Args:
        user_text: Raw user prompt (will be augmented by task_type).
        assistant_text: Target response (None for inference).
        system_prompt: System message (None to omit).
        task_type: Task type for prompt augmentation.
        image: Optional PIL Image to include in the user message
               (for native VLM ViT path).

    Returns:
        List of message dicts ready for apply_chat_template().
    """
    augmented = augment_prompt(user_text, task_type)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": augmented})
    messages.append({"role": "user", "content": user_content})

    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_chat_inputs(
    prompts: list[str],
    targets: list[str],
    tokenizer,
    device: str,
    max_length: int = 512,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    task_types: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build tokenized inputs with proper chat template and label masking.

    Uses Qwen3-VL's native chat template so the model sees the same
    format it was pretrained on.  Only assistant response tokens
    contribute to the LM loss; system + user tokens are masked with -100.

    Optimized: one ``apply_chat_template()`` call per sample, then
    batched tokenization (2 batch calls total instead of 2*B individual).

    Args:
        prompts: List of task descriptions.
        targets: List of target responses.
        tokenizer: HuggingFace tokenizer (from AutoProcessor.tokenizer).
        device: Target device string.
        max_length: Maximum sequence length.
        system_prompt: Optional system message (None to omit).
        task_types: Per-sample task types (default: all "affordance").

    Returns:
        input_ids:      [B, L]
        attention_mask: [B, L]
        labels:         [B, L] with -100 for non-target positions
    """
    if task_types is None:
        task_types = [DEFAULT_TASK_TYPE] * len(prompts)

    # Step 1: Build all text strings (one apply_chat_template per sample)
    all_full_texts = []
    all_prompt_texts = []

    for prompt, target, task_type in zip(prompts, targets, task_types):
        messages_full = build_messages(prompt, target, system_prompt, task_type)
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False,
        )
        # Extract prompt prefix by finding the assistant marker in the full text
        # This avoids a second apply_chat_template() call per sample
        marker_pos = full_text.find(_ASSISTANT_MARKER)
        prompt_text = full_text[:marker_pos + len(_ASSISTANT_MARKER)]

        all_full_texts.append(full_text)
        all_prompt_texts.append(prompt_text)

    # Step 2: Batch tokenize (2 calls total, not 2*B)
    full_enc = tokenizer(
        all_full_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length,
    )
    prompt_enc = tokenizer(
        all_prompt_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_length,
    )

    input_ids = full_enc.input_ids.to(device)
    attention_mask = full_enc.attention_mask.to(device)

    # Step 3: Build labels — mask prompt tokens and padding
    prompt_lens = prompt_enc.attention_mask.sum(dim=1)  # [B]
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels


def format_inference_prompt(
    text: str,
    tokenizer,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    task_type: str = DEFAULT_TASK_TYPE,
) -> tuple[str, torch.Tensor]:
    """Format a single prompt for inference (no target).

    Returns the full prompt string (with generation prefix) and its token IDs,
    ready for model.generate() or manual autoregressive generation.

    Returns:
        prompt_text: The formatted prompt string.
        input_ids: [1, L] token ID tensor.
    """
    messages = build_messages(text, None, system_prompt, task_type)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    return prompt_text, enc.input_ids
