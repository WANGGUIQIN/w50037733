"""RoboBrain-3DGS Inference: 3D-enhanced robotic manipulation prediction.

Mirrors RoboBrain2.5's inference.py interface but adds 3D Gaussian
understanding from RGBD input.

Architecture flow:
    RGBD -> DepthToGaussian -> GS Encoder -> 3D Tokens
    Text prompt -> Chat template (Qwen3-VL) -> Token IDs
    [3D Tokens | Text Tokens] -> LLM -> Affordance/Constraint Output

Usage:
    from inference_3dgs import RoboBrain3DGS_Inference

    model = RoboBrain3DGS_Inference(
        checkpoint="outputs/lora/best",
        base_model="/home/w50037733/models/RoboBrain2.5-8B-NV",
    )
    result = model.inference(
        prompt="close the jar",
        rgb_path="image.png",
        depth_path="depth.npy",  # or None for 2D-only
        task="affordance",
    )
    print(result["answer"])
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from models.robobrain_vlm import RoboBrain3DGS_VLM
from utils.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    format_inference_prompt,
    parse_affordance_output,
)


class RoboBrain3DGS_Inference:
    """Unified inference for RoboBrain-3DGS.

    Supports both 3D-enhanced (RGBD) and 2D-only (RGB) inference.
    Uses the same Qwen3-VL chat template as the official RoboBrain2.5.
    """

    def __init__(
        self,
        base_model: str = "/home/w50037733/models/RoboBrain2.5-8B-NV",
        checkpoint: str | None = None,
        mode: str = "lora",
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    ):
        print("Loading RoboBrain-3DGS model...")

        # Load processor/tokenizer
        self.processor = AutoProcessor.from_pretrained(base_model)
        self.tokenizer = self.processor.tokenizer
        self.system_prompt = system_prompt

        # Load model
        self.model = RoboBrain3DGS_VLM.from_pretrained(
            model_path=base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        # Load checkpoint if provided
        if checkpoint:
            self._load_checkpoint(checkpoint, mode)

        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.dtype = torch_dtype

        print(f"Model ready on {self.device}")

    def _load_checkpoint(self, ckpt_path: str, mode: str):
        """Load trained weights from checkpoint."""
        ckpt_dir = Path(ckpt_path)
        if not ckpt_dir.exists():
            print(f"Warning: checkpoint not found at {ckpt_path}")
            return

        # Load 3D branch
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

        # Load VLM weights
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

    def _prepare_rgbd(
        self,
        rgb_path: str,
        depth_path: str | None = None,
        image_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Load and preprocess RGBD data."""
        # RGB
        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb_img = rgb_img.resize((image_size, image_size), Image.BILINEAR)
        rgb = torch.from_numpy(
            np.array(rgb_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # Depth
        depth = None
        if depth_path:
            p = Path(depth_path)
            if p.suffix == ".npy":
                depth_np = np.load(depth_path)
            elif p.suffix == ".png":
                # RLBench RGB-encoded depth
                from data.rlbench_loader import decode_rlbench_depth
                depth_rgb = np.array(Image.open(depth_path).convert("RGB"))
                depth_np = decode_rlbench_depth(depth_rgb)
            else:
                raise ValueError(f"Unsupported depth format: {p.suffix}")

            depth_pil = Image.fromarray(depth_np, mode="F")
            depth_pil = depth_pil.resize((image_size, image_size), Image.NEAREST)
            depth = torch.from_numpy(np.array(depth_pil)).unsqueeze(0).unsqueeze(0)

        # Default intrinsics (approximate)
        fx = image_size * 0.87  # ~60deg FOV
        intrinsics = torch.tensor([
            [fx, 0, image_size / 2],
            [0, fx, image_size / 2],
            [0, 0, 1],
        ], dtype=torch.float32).unsqueeze(0)  # [1, 3, 3]

        return rgb, depth, intrinsics

    @torch.no_grad()
    def inference(
        self,
        prompt: str,
        rgb_path: str | None = None,
        depth_path: str | None = None,
        task: str = "affordance",
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        image_size: int = 256,
    ) -> dict:
        """Run inference with the RoboBrain-3DGS model.

        Args:
            prompt: Task description text.
            rgb_path: Path to RGB image (optional for text-only).
            depth_path: Path to depth map (optional for 2D-only).
            task: Task type (affordance/pointing/trajectory/grounding/general).
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            image_size: Image resize target.

        Returns:
            dict with "answer" key containing the model's response.
        """
        # Format prompt with chat template
        _, input_ids = format_inference_prompt(
            prompt, self.tokenizer, self.system_prompt, task,
        )
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        # Prepare RGBD if provided
        rgb, depth, intrinsics = None, None, None
        if rgb_path:
            rgb, depth, intrinsics = self._prepare_rgbd(
                rgb_path, depth_path, image_size,
            )
            rgb = rgb.to(device=self.device, dtype=self.dtype)
            intrinsics = intrinsics.to(device=self.device, dtype=self.dtype)
            if depth is not None:
                depth = depth.to(device=self.device, dtype=self.dtype)

        # Generate
        generated = self.model.generate_with_3d(
            input_ids=input_ids,
            attention_mask=attention_mask,
            depth=depth,
            intrinsics=intrinsics,
            rgb_for_3d=rgb,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        # Decode: trim input tokens, decode only generated part
        generated_trimmed = generated[0, input_ids.shape[1]:]
        answer = self.tokenizer.decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {"answer": answer.strip()}

    @torch.no_grad()
    def inference_batch(
        self,
        prompts: list[str],
        rgb_paths: list[str],
        depth_paths: list[str | None] | None = None,
        task: str = "affordance",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Batch inference (sequential, for evaluation)."""
        if depth_paths is None:
            depth_paths = [None] * len(prompts)

        results = []
        for prompt, rgb_path, depth_path in zip(prompts, rgb_paths, depth_paths):
            result = self.inference(
                prompt=prompt,
                rgb_path=rgb_path,
                depth_path=depth_path,
                task=task,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            results.append(result)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RoboBrain-3DGS Inference")
    parser.add_argument("--base_model", default="/home/w50037733/models/RoboBrain2.5-8B-NV")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--prompt", default="close the jar")
    parser.add_argument("--rgb", default=None, help="Path to RGB image")
    parser.add_argument("--depth", default=None, help="Path to depth map")
    parser.add_argument("--task", default="affordance",
                        choices=["affordance", "pointing", "trajectory", "grounding", "general"])
    parser.add_argument("--no_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model = RoboBrain3DGS_Inference(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        mode=args.mode,
    )

    result = model.inference(
        prompt=args.prompt,
        rgb_path=args.rgb,
        depth_path=args.depth,
        task=args.task,
        do_sample=not args.no_sample,
        temperature=args.temperature,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Task:   {args.task}")
    print(f"Answer: {result['answer']}")

    parsed = parse_affordance_output(result["answer"])
    if parsed:
        print(f"Parsed: {parsed}")
