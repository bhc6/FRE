"""Independent local SmolVLM2 reader for persisted RGB memory and questions."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


VERIFIED_TRANSFORMERS = "4.57.1"
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
DEFAULT_MAX_PIXELS = 512 * 512
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"


class PreparedBatch(dict):
    """Tensors and separate audit metadata; no ground truth is accepted."""

    def __init__(self, tensors: dict[str, Any], metadata: dict[str, Any]):
        super().__init__(tensors)
        self.metadata = metadata


class SmolVLMBackend:
    def __init__(self, model: Any, processor: Any, metadata: dict[str, Any], device: str):
        self.model = model
        self.processor = processor
        self.metadata = metadata
        self.device = device
        self.image_size = int(model.config.vision_config.image_size)
        self.image_token_id = int(model.config.image_token_id)
        self.image_seq_len = int(processor.image_seq_len)
        expected = (self.image_size // model.config.vision_config.patch_size // model.config.scale_factor) ** 2
        if self.image_seq_len != expected:
            raise ValueError("Processor image tokens disagree with the model's spatial reduction")
        if processor.tokenizer.convert_tokens_to_ids("<image>") != self.image_token_id:
            raise ValueError("Processor and model image-token IDs disagree")
        if processor.video_processor.max_image_size != {"longest_edge": self.image_size}:
            raise ValueError("Video processor and model spatial sizes disagree")

    def _question(self, question: str) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        controls = ("<image>", "<video>", "<fake_token_around_image>", "<global-img>",
                    "<|im_start|>", "<|im_end|>", "<end_of_utterance>", "<row_")
        if any(token in question for token in controls):
            raise ValueError("Question text must not inject model control tokens")

    def _batch(self, tensors: dict[str, Any], metadata: dict[str, Any]) -> PreparedBatch:
        maximum_tokens = min(8192, int(self.model.config.text_config.max_position_embeddings))
        if tensors["input_ids"].shape[0] != 1 or tensors["input_ids"].shape[1] > maximum_tokens:
            raise ValueError(f"SmolVLM reader requires batch=1 and at most {maximum_tokens} input tokens")
        converted = {
            name: value.to(device=self.device, dtype=self.model.dtype)
            if name == "pixel_values" else value.to(self.device)
            for name, value in tensors.items()
        }
        metadata = {**self.metadata, **metadata, "input_tokens": int(tensors["input_ids"].shape[1])}
        metadata["prompt_sha256"] = hashlib.sha256(metadata["rendered_prompt"].encode("utf-8")).hexdigest()
        return PreparedBatch(converted, metadata)

    def prepare_text(self, question: str) -> PreparedBatch:
        self._question(question)
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tensors = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return self._batch(tensors, {"rendered_prompt": prompt, "visual_tokens": 0, "modality": "text"})

    def prepare_video(
        self, frames: list[Any], timestamps_s: list[float], question: str,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> PreparedBatch:
        self._question(question)
        timestamps = np.asarray(timestamps_s, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) != len(frames) or not 1 <= len(frames) <= 64:
            raise ValueError("Provide 1..64 frames with exactly one timestamp per frame")
        if not np.isfinite(timestamps).all() or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels < self.image_size**2:
            raise ValueError(f"Official SmolVLM video preprocessing requires max_pixels >= {self.image_size**2}")
        arrays = [np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame) for frame in frames]
        if any(array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8 for array in arrays):
            raise ValueError("Frames must be RGB uint8 arrays or PIL images")
        if len({array.shape for array in arrays}) != 1 or min(arrays[0].shape[:2]) < 1:
            raise ValueError("Frames must have identical nonempty spatial dimensions")
        visual = self.processor.video_processor(
            videos=[np.stack(arrays)], do_sample_frames=False, return_metadata=False, return_tensors="pt",
        )
        expected_shape = (1, len(frames), 3, self.image_size, self.image_size)
        if tuple(visual["pixel_values"].shape) != expected_shape:
            raise ValueError("Official video preprocessing unexpectedly changed frame count or spatial size")
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt.count(self.processor.video_token) != 1:
            raise ValueError("Expected exactly one SmolVLM video placeholder")
        explicit_times = SimpleNamespace(fps=1.0, timestamps=timestamps.tolist(), duration=float(timestamps[-1]))
        prompt = self.processor.expand_text_with_video_tokens(
            [prompt], {"pixel_values": visual["pixel_values"], "video_metadata": [explicit_times]},
        )[0]
        tensors = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        visual_count = int((tensors["input_ids"] == self.image_token_id).sum())
        if visual_count != self.image_seq_len * len(frames):
            raise ValueError("Expanded visual-token count does not match the retained frame count")
        rendered_times = [f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}" for timestamp in timestamps]
        metadata = {
            "rendered_prompt": prompt, "modality": "video", "visual_tokens": visual_count,
            "source_timestamps_s": timestamps.tolist(), "frame_count": len(frames),
            "frame_groups": [[index] for index in range(len(frames))],
            "rendered_frame_timestamps": rendered_times,
            "timestamp_rendering": "official SmolVLM per-frame MM:SS; fractional seconds truncated by upstream formatter",
            "timestamp_source": "explicit stored timestamps, no inferred video FPS",
            "timestamp_prompt_precision_decimals": 0,
            "duration_definition": "last retained timestamp relative to source video start",
            "processor_resamples_frames": False, "image_tokens_per_frame": self.image_seq_len,
            "max_pixels_per_frame": max_pixels, "resized_pixels_per_frame": self.image_size**2,
            "processor_spatial_policy": "official video processor: aspect resize followed by fixed square resize; no splitting",
            "processor_interpolation": "official Transformers 4.57.1 fast-video BICUBIC fallback for configured LANCZOS",
            "source_rgb_shape": list(arrays[0].shape), "processed_rgb_shape": list(expected_shape[2:]),
            "preprocessing_compute_included_in_forward_timing": False,
        }
        return self._batch({**tensors, **visual}, metadata)

    def _start_measurement(self) -> float:
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return time.perf_counter()

    def _finish_measurement(self, started: float) -> dict[str, Any]:
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(self.device)) if self.device.startswith("cuda") else None,
            "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(self.device)) if self.device.startswith("cuda") else None,
        }

    def choose_batch(self, batch: PreparedBatch, option_count: int) -> dict[str, Any]:
        """Select a unique allowed option letter without receiving an answer label."""
        import torch

        if isinstance(option_count, bool) or not isinstance(option_count, int) or not 2 <= option_count <= 26:
            raise ValueError("option_count must be within 2..26")
        letters = [chr(ord("A") + index) for index in range(option_count)]
        tokenized = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if any(len(tokens) != 1 for tokens in tokenized) or len({tokens[0] for tokens in tokenized}) != option_count:
            raise ValueError("Constrained choice requires unique single-token option letters")
        started = self._start_measurement()
        with torch.inference_mode():
            output = self.model(**batch, use_cache=False, output_hidden_states=False, return_dict=True, logits_to_keep=1)
            logits = output.logits[0, -1].float()
            allowed = logits[[tokens[0] for tokens in tokenized]]
            if not torch.isfinite(allowed).all():
                raise RuntimeError("Nonfinite option logits")
            chosen = int(allowed.argmax().item())
            probabilities = allowed.softmax(dim=-1).cpu().tolist()
            values = allowed.cpu().tolist()
            unconstrained = int(logits.argmax().item())
        return {
            "raw_text": letters[chosen], "answer_index": chosen, "option_logits": values,
            "option_probabilities": probabilities, "option_token_ids": [tokens[0] for tokens in tokenized],
            "unconstrained_top_token_id": unconstrained,
            "unconstrained_top_token": self.processor.tokenizer.decode([unconstrained]),
            **self._finish_measurement(started),
            "input_tokens": int(batch["input_ids"].shape[1]), "output_tokens": 1,
            "visual_tokens": batch.metadata["visual_tokens"], "device": self.device,
            "model_id": self.metadata["model_id"], "model_revision": self.metadata["model_revision"],
            "decoding": "argmax_single_token_allowed_option_letters_v1", "do_sample": False,
            "timing_scope": "model forward and constrained choice; excludes input preparation",
            "rendered_prompt": batch.metadata["rendered_prompt"], "provenance": batch.metadata,
        }


def load_backend(
    model_path: str = DEFAULT_MODEL, device: str = "cuda", revision: str | None = DEFAULT_REVISION,
) -> SmolVLMBackend:
    """Load the pinned checkpoint locally; acquisition is a separate audited step."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, SmolVLMConfig, SmolVLMForConditionalGeneration

    version = importlib.metadata.version("transformers")
    if version != VERIFIED_TRANSFORMERS:
        raise RuntimeError(f"Verified backend requires transformers=={VERIFIED_TRANSFORMERS}, installed {version}")
    if not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter")
    requested_revision = revision or DEFAULT_REVISION
    model_directory = Path(model_path).expanduser()
    if not model_directory.is_dir():
        model_directory = Path(snapshot_download(
            repo_id=model_path, revision=requested_revision, cache_dir=str(DEFAULT_CACHE),
            local_files_only=True, token=False,
        ))
    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(str(model_directory), **options)
    if not isinstance(config, SmolVLMConfig):
        raise ValueError("This reader requires a SmolVLM checkpoint")
    processor = AutoProcessor.from_pretrained(str(model_directory), **options)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    if device.startswith("cuda") and dtype == torch.float32:
        dtype = torch.float16
    model = SmolVLMForConditionalGeneration.from_pretrained(
        str(model_directory), dtype=dtype, attn_implementation="sdpa", **options,
    ).to(device).eval()
    model.requires_grad_(False)
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision is None and len(model_directory.name) == 40 and all(
        character in "0123456789abcdefABCDEF" for character in model_directory.name
    ):
        resolved_revision = model_directory.name
    if model_path == DEFAULT_MODEL and resolved_revision != requested_revision:
        raise ValueError("Resolved checkpoint does not match the requested immutable revision")
    metadata = {
        "model_id": model_path, "model_revision": resolved_revision or "unresolved-local",
        "processor_revision": resolved_revision or "unresolved-local", "requested_revision": requested_revision,
        "local_snapshot_path": str(model_directory.resolve()), "transformers": version,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "dtype": str(dtype), "device": device, "attention_implementation": "sdpa",
        "local_files_only": True, "trust_remote_code": False, "max_input_tokens": 8192,
        "model_class": type(model).__name__, "reader_role": "independent reader of persisted RGB memory",
        "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.startswith("cuda") else None,
    }
    return SmolVLMBackend(model, processor, metadata, device)
