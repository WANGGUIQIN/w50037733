"""RoboBrain-3DGS Inference: 3D-enhanced robotic manipulation prediction.

Mirrors RoboBrain2.5's UnifiedInference interface, adding 3D Gaussian
understanding from RGBD input.

Architecture flow:
    RGBD -> DepthToGaussian -> GS Encoder -> 3D Tokens
    Text prompt -> Chat template (Qwen3-VL) -> Token IDs
    [3D Tokens | Text Tokens] -> LLM -> Affordance/Constraint Output

Usage:
    from inference_3dgs import UnifiedInference3DGS

    model = UnifiedInference3DGS(
        model_id="/path/to/RoboBrain2.5-8B-NV",
        checkpoint="outputs/lora/best",
    )

    # Affordance with 3D (RGBD)
    result = model.inference(
        text="close the jar",
        image="scene.png",
        depth="depth.npy",
        task="affordance",
        plot=True,
    )
    print(result["answer"])
    print(result["parsed"])   # structured affordance output

    # Pointing (2D, same as RoboBrain 2.5)
    result = model.inference(
        text="the red cup",
        image="scene.png",
        task="pointing",
        plot=True,
    )

    # Compatible with RoboBrain 2.5's interface (no depth = 2D-only)
    result = model.inference(
        text="What is in this scene?",
        image="scene.png",
        task="general",
    )
"""

import os
import re
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))

from models.robobrain_vlm import RoboBrain3DGS_VLM
from utils.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    format_inference_prompt,
    parse_affordance_output,
)


# ---------------------------------------------------------------------------
# Task-specific prompt templates (matching RoboBrain 2.5)
# ---------------------------------------------------------------------------

_TASK_PROMPTS = {
    "pointing": (
        "{text}. Please provide its 2D coordinates. Your answer should be "
        "formatted as a tuple, i.e. [(x, y)], where the tuple contains the "
        "x and y coordinates of a point satisfying the conditions above."
    ),
    "trajectory": (
        'Please predict 3D end-effector-centric waypoints to complete the '
        'task successfully. The task is "{text}". Your answer should be '
        "formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), "
        "...], where each tuple contains the x and y coordinates and the "
        "depth of the point."
    ),
    "grounding": (
        "Please provide the bounding box coordinate of the region this "
        "sentence describes: {text}."
    ),
    "affordance": (
        "{text}. Please predict the affordance point and manipulation "
        "constraints for completing this task. Your answer should include: "
        "the affordance coordinates as [u, v] in normalized image space, "
        "gripper_width, and approach vector as [x, y, z]."
    ),
    "general": "{text}",
}


class UnifiedInference3DGS:
    """Unified inference for RoboBrain-3DGS.

    Mirrors RoboBrain2.5's UnifiedInference API with additional 3D Gaussian
    understanding from RGBD input.

    Supports:
        - 3D-enhanced inference (RGBD input via depth parameter)
        - 2D-only inference (RGB only, compatible with RoboBrain 2.5)
        - All RoboBrain 2.5 task types + affordance
        - Result visualization with plot=True
        - LoRA and full checkpoint loading
    """

    def __init__(
        self,
        model_id: str = "/home/w50037733/models/RoboBrain2.5-8B-NV",
        checkpoint: str | None = None,
        mode: str = "lora",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ):
        """Initialize model and processor.

        Args:
            model_id: Path or HuggingFace model identifier.
            checkpoint: Path to trained checkpoint directory (or None for base model).
            mode: Checkpoint mode ("lora" or "full").
            device_map: Device mapping ("auto" for inference, {"": 0} for training).
            torch_dtype: Model dtype (bfloat16 recommended).
            system_prompt: System prompt for chat template (None to disable).
        """
        print("Loading RoboBrain-3DGS model ...")
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.dtype = torch_dtype

        # Load processor / tokenizer (same as RoboBrain 2.5)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        # Load model with 3D branch
        self.model = RoboBrain3DGS_VLM.from_pretrained(
            model_path=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # Load checkpoint if provided
        if checkpoint:
            self._load_checkpoint(checkpoint, mode)

        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"Model ready on {self.device}")

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_path: str, mode: str):
        """Load trained weights from checkpoint directory.

        Checkpoint layout:
            checkpoint-N/
            +-- 3d_branch.pt          # DepthToGaussian + GS Encoder + Projector
            +-- lora_adapter/         # (mode=lora) PEFT adapter weights
            +-- vlm_trainable.pt      # (mode=full) full VLM trainable weights
            +-- metadata.json
        """
        ckpt_dir = Path(ckpt_path)
        if not ckpt_dir.exists():
            print(f"  Warning: checkpoint not found at {ckpt_path}")
            return

        # 3D branch
        gs_path = ckpt_dir / "3d_branch.pt"
        if gs_path.exists():
            gs_state = torch.load(gs_path, map_location="cpu", weights_only=True)
            param_map = dict(self.model.named_parameters())
            loaded = 0
            for n, t in gs_state.items():
                if n in param_map:
                    param_map[n].data.copy_(t)
                    loaded += 1
            print(f"  3D branch: {loaded}/{len(gs_state)} tensors loaded")

        # VLM weights
        if mode == "lora":
            adapter_dir = ckpt_dir / "lora_adapter"
            if adapter_dir.exists():
                from peft import PeftModel
                self.model.vlm = PeftModel.from_pretrained(
                    self.model.vlm, str(adapter_dir),
                )
                print("  LoRA adapter loaded")
        else:
            vlm_path = ckpt_dir / "vlm_trainable.pt"
            if vlm_path.exists():
                vlm_state = torch.load(vlm_path, map_location="cpu", weights_only=True)
                param_map = dict(self.model.named_parameters())
                loaded = 0
                for n, t in vlm_state.items():
                    if n in param_map:
                        param_map[n].data.copy_(t)
                        loaded += 1
                print(f"  VLM params: {loaded}/{len(vlm_state)} tensors loaded")

    # ------------------------------------------------------------------
    # RGBD preprocessing
    # ------------------------------------------------------------------

    def _load_depth(self, depth_path: str, image_size: int) -> torch.Tensor:
        """Load depth map from various formats.

        Supported formats:
            .npy  -- numpy array (meters)
            .png  -- RLBench RGB-encoded depth (auto-detected)
            .exr  -- OpenEXR float depth
        """
        p = Path(depth_path)

        if p.suffix == ".npy":
            depth_np = np.load(depth_path).astype(np.float32)
        elif p.suffix == ".png":
            # RLBench RGB-encoded depth: (R + G*256 + B*65536) / (256^3 - 1)
            depth_rgb = np.array(Image.open(depth_path).convert("RGB")).astype(np.float64)
            near, far = 0.01, 5.0
            encoded = (
                depth_rgb[:, :, 0]
                + depth_rgb[:, :, 1] * 256.0
                + depth_rgb[:, :, 2] * 65536.0
            )
            depth_np = (near + (far - near) * encoded / (256.0**3 - 1.0)).astype(np.float32)
        elif p.suffix == ".exr":
            depth_np = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported depth format: {p.suffix}. "
                "Supported: .npy, .png (RLBench), .exr"
            )

        # Ensure 2D
        if depth_np.ndim == 3:
            depth_np = depth_np[:, :, 0]

        # Resize to match image
        depth_pil = Image.fromarray(depth_np, mode="F")
        depth_pil = depth_pil.resize((image_size, image_size), Image.NEAREST)
        depth_tensor = torch.from_numpy(np.array(depth_pil))
        return depth_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def _make_intrinsics(
        self,
        image_size: int,
        intrinsics: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build or validate camera intrinsics matrix [1, 3, 3]."""
        if intrinsics is not None:
            K = torch.as_tensor(intrinsics, dtype=torch.float32)
            if K.dim() == 2:
                K = K.unsqueeze(0)
            return K

        # Default: approximate 60deg FOV
        fx = image_size * 0.87
        return torch.tensor([
            [fx, 0, image_size / 2],
            [0, fx, image_size / 2],
            [0,  0, 1],
        ], dtype=torch.float32).unsqueeze(0)

    def _prepare_rgbd(
        self,
        image_path: str,
        depth_path: str | None,
        intrinsics: np.ndarray | torch.Tensor | None,
        image_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Load and preprocess RGBD data for the 3D branch."""
        # RGB
        rgb_img = Image.open(image_path).convert("RGB")
        rgb_img = rgb_img.resize((image_size, image_size), Image.BILINEAR)
        rgb = torch.from_numpy(
            np.array(rgb_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Depth
        depth = None
        if depth_path is not None:
            depth = self._load_depth(depth_path, image_size)

        # Intrinsics
        K = self._make_intrinsics(image_size, intrinsics)

        return rgb, depth, K

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, text: str, task: str) -> str:
        """Apply task-specific prompt template (matching RoboBrain 2.5)."""
        template = _TASK_PROMPTS.get(task, _TASK_PROMPTS["general"])
        return template.format(text=text)

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        text: str,
        image: Union[list, str],
        depth: str | None = None,
        task: str = "affordance",
        intrinsics: np.ndarray | torch.Tensor | None = None,
        plot: bool = False,
        do_sample: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 768,
        image_size: int = 256,
    ) -> dict:
        """Perform inference with text, image, and optional depth input.

        Args:
            text: Input text prompt.
            image: Path to RGB image (str) or list of paths.
            depth: Path to depth map (.npy, .png, .exr) or None for 2D-only.
            task: Task type -- "affordance", "pointing", "trajectory",
                  "grounding", or "general".
            intrinsics: Camera intrinsics [3,3] matrix (optional, estimated if None).
            plot: Whether to save annotated result image to ./result/.
            do_sample: Whether to use sampling during generation.
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
            image_size: Resize target for 3D branch input.

        Returns:
            dict with keys:
                "answer": Raw model output text.
                "parsed": Structured output (affordance task only, else None).
        """
        # Normalize image input
        if isinstance(image, str):
            image = [image]

        valid_tasks = ["general", "pointing", "trajectory", "grounding", "affordance"]
        assert task in valid_tasks, (
            f"Invalid task: {task!r}. Supported: {valid_tasks}"
        )
        assert task == "general" or len(image) == 1, (
            "Pointing, grounding, trajectory, and affordance tasks require exactly one image."
        )

        # Apply task-specific prompt
        prompt_text = self._format_prompt(text, task)
        print(f"\n{'='*20} INPUT {'='*20}")
        print(f"Task: {task}")
        print(f"Prompt: {prompt_text}")
        print(f"Image: {image[0]}")
        print(f"Depth: {depth or 'None (2D-only)'}")
        print(f"{'='*47}\n")

        # Prepare 3D branch inputs (RGBD)
        rgb, depth_tensor, K = None, None, None
        if len(image) >= 1:
            rgb, depth_tensor, K = self._prepare_rgbd(
                image[0], depth, intrinsics, image_size,
            )
            rgb = rgb.to(device=self.device, dtype=self.dtype)
            K = K.to(device=self.device, dtype=self.dtype)
            if depth_tensor is not None:
                depth_tensor = depth_tensor.to(device=self.device, dtype=self.dtype)

        # Tokenize with chat template + image (enables ViT)
        pil_img = Image.open(image[0]).convert("RGB")
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt_text},
        ]})
        text_for_proc = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        proc_inputs = self.processor(
            text=[text_for_proc], images=[pil_img],
            return_tensors="pt", padding=True,
        )
        input_ids = proc_inputs["input_ids"].to(self.device)
        attention_mask = proc_inputs["attention_mask"].to(self.device)
        pixel_values = proc_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_grid_thw = proc_inputs["image_grid_thw"].to(self.device)

        # Generate with ViT (2D) + 3D branch
        print("Running inference ...")
        generated = self.model.generate_with_3d(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            depth=depth_tensor,
            intrinsics=K,
            rgb_for_3d=rgb if depth_tensor is not None else None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Decode
        generated_trimmed = generated[0, input_ids.shape[1]:]
        answer = self.tokenizer.decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # Parse structured output for affordance
        parsed = None
        if task == "affordance":
            parsed = parse_affordance_output(answer)

        # Visualization
        if plot and task in ["pointing", "trajectory", "grounding", "affordance"]:
            self._plot_result(image[0], answer, task)

        result = {"answer": answer, "parsed": parsed}
        print(f"\nAnswer: {answer}")
        if parsed:
            print(f"Parsed: {parsed}")
        return result

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference_batch(
        self,
        texts: list[str],
        images: list[str],
        depths: list[str | None] | None = None,
        task: str = "affordance",
        intrinsics: np.ndarray | torch.Tensor | None = None,
        max_new_tokens: int = 768,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Sequential batch inference for evaluation."""
        if depths is None:
            depths = [None] * len(texts)

        results = []
        for i, (txt, img, dep) in enumerate(zip(texts, images, depths)):
            print(f"\n--- Sample {i+1}/{len(texts)} ---")
            result = self.inference(
                text=txt,
                image=img,
                depth=dep,
                task=task,
                intrinsics=intrinsics,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Visualization (matching RoboBrain 2.5's draw_on_image)
    # ------------------------------------------------------------------

    def _plot_result(self, image_path: str, answer: str, task: str):
        """Parse answer and draw annotations on the image."""
        print("Plotting results ...")
        plot_points, plot_boxes, plot_trajectories = None, None, None
        affordance_point = None

        if task == "trajectory":
            pattern = r'(\d+),\s*(\d+),\s*([+-]?\d+\.?\d*)'
            matches = re.findall(pattern, answer)
            plot_trajectories = [
                [(int(x), int(y), float(d)) for x, y, d in matches]
            ]
            suffix = "trajectory"

        elif task == "pointing":
            pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
            matches = re.findall(pattern, answer)
            plot_points = [(int(x), int(y)) for x, y in matches]
            suffix = "pointing"

        elif task == "grounding":
            pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
            matches = re.findall(pattern, answer)
            plot_boxes = [
                [int(x1), int(y1), int(x2), int(y2)]
                for x1, y1, x2, y2 in matches
            ]
            suffix = "grounding"

        elif task == "affordance":
            parsed = parse_affordance_output(answer)
            if parsed["u"] is not None:
                affordance_point = (parsed["u"], parsed["v"])
            suffix = "affordance"

        # Save annotated image
        os.makedirs("result", exist_ok=True)
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        output_path = os.path.join("result", f"{name}_{suffix}_annotated{ext}")

        self.draw_on_image(
            image_path,
            points=plot_points,
            boxes=plot_boxes,
            trajectories=plot_trajectories,
            affordance=affordance_point,
            output_path=output_path,
        )

    def draw_on_image(
        self,
        image_path: str,
        points: list[tuple] | None = None,
        boxes: list[list] | None = None,
        trajectories: list[list] | None = None,
        affordance: tuple | None = None,
        output_path: str | None = None,
    ) -> str | None:
        """Draw annotations on image (compatible with RoboBrain 2.5).

        Args:
            image_path: Path to input image.
            points: List of (x, y) points in relative coords (0~1000).
            boxes: List of [x1, y1, x2, y2] boxes in relative coords.
            trajectories: List of [(x, y, d), ...] trajectories in relative coords.
            affordance: (u, v) in normalized [0, 1] image space.
            output_path: Path to save annotated image.

        Returns:
            Path to saved image, or None on error.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                """Convert relative (0~1000) to absolute pixel coords."""
                x = max(0, min(w - 1, int(round(x_rel / 1000.0 * w))))
                y = max(0, min(h - 1, int(round(y_rel / 1000.0 * h))))
                return x, y

            # Points (red circles)
            if points:
                for x_rel, y_rel in points:
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            # Bounding boxes (green rectangles)
            if boxes:
                for x1r, y1r, x2r, y2r in boxes:
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Trajectories (blue lines with start/end markers)
            if trajectories:
                for traj in trajectories:
                    if not traj or len(traj) < 2:
                        continue
                    abs_pts = [rel_to_abs(p[0], p[1]) for p in traj]
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (255, 0, 0), 2)
                    cv2.circle(image, abs_pts[0], 7, (0, 255, 0), -1)   # green start
                    cv2.circle(image, abs_pts[-1], 7, (255, 0, 0), -1)  # blue end

            # Affordance point (red crosshair, normalized 0~1)
            if affordance is not None:
                u, v = affordance
                ax = max(0, min(w - 1, int(round(u * w))))
                ay = max(0, min(h - 1, int(round(v * h))))
                # Crosshair
                size = 15
                cv2.line(image, (ax - size, ay), (ax + size, ay), (0, 0, 255), 2)
                cv2.line(image, (ax, ay - size), (ax, ay + size), (0, 0, 255), 2)
                cv2.circle(image, (ax, ay), 8, (0, 0, 255), 2)
                # Label
                cv2.putText(
                    image, f"({u:.2f}, {v:.2f})", (ax + 12, ay - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                )

            # Save
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            cv2.imwrite(output_path, image)
            print(f"  Annotated image saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"  Error drawing on image: {e}")
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RoboBrain-3DGS Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Affordance prediction with 3D (RGBD)\n"
            "  python inference_3dgs.py \\\n"
            "      --image scene.png --depth depth.npy \\\n"
            "      --prompt 'close the jar' --task affordance --plot\n"
            "\n"
            "  # Pointing (2D only, like RoboBrain 2.5)\n"
            "  python inference_3dgs.py \\\n"
            "      --image scene.png \\\n"
            "      --prompt 'the red cup' --task pointing --plot\n"
            "\n"
            "  # With trained checkpoint\n"
            "  python inference_3dgs.py \\\n"
            "      --checkpoint outputs/lora/best --mode lora \\\n"
            "      --image scene.png --depth depth.npy \\\n"
            "      --prompt 'pick up the cup' --task affordance\n"
        ),
    )
    parser.add_argument("--base_model", default="/home/w50037733/models/RoboBrain2.5-8B-NV",
                        help="Path to pretrained model")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--mode", default="lora", choices=["lora", "full"],
                        help="Checkpoint mode")
    parser.add_argument("--image", required=True,
                        help="Path to RGB image")
    parser.add_argument("--depth", default=None,
                        help="Path to depth map (.npy, .png, .exr)")
    parser.add_argument("--prompt", required=True,
                        help="Task description text")
    parser.add_argument("--task", default="affordance",
                        choices=["affordance", "pointing", "trajectory",
                                 "grounding", "general"],
                        help="Task type")
    parser.add_argument("--plot", action="store_true",
                        help="Save annotated result image to ./result/")
    parser.add_argument("--no_sample", action="store_true",
                        help="Disable sampling (use greedy decoding)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=768,
                        help="Maximum tokens to generate")
    args = parser.parse_args()

    model = UnifiedInference3DGS(
        model_id=args.base_model,
        checkpoint=args.checkpoint,
        mode=args.mode,
    )

    result = model.inference(
        text=args.prompt,
        image=args.image,
        depth=args.depth,
        task=args.task,
        plot=args.plot,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
