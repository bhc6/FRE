"""Local Qwen3-VL inference with explicit timestamped video-token alignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path
import time
from typing import Any

import numpy as np


VERIFIED_TRANSFORMERS = "4.57.1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_MAX_PIXELS = 64 * 32 * 32


class PreparedBatch(dict):
    """Model tensors plus metadata that is never forwarded as model kwargs."""

    def __init__(self, tensors: dict[str, Any], metadata: dict[str, Any], positions: np.ndarray | None = None):
        super().__init__(tensors)
        self.metadata = metadata
        self.video_positions = positions


@dataclass
class ExtractionResult:
    hidden_states: dict[int, np.ndarray]
    timestamps_s: np.ndarray
    metadata: dict[str, Any]


def temporal_grid(timestamps_s: list[float], temporal_patch_size: int = 2) -> np.ndarray:
    """Preserve real timestamps; irregular spacing is valid for question answering."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be a finite one-dimensional sequence")
    if temporal_patch_size != 2 or len(timestamps) < 2 or len(timestamps) % temporal_patch_size:
        raise ValueError("This verified adapter requires complete pairs of frames and temporal_patch_size=2")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Frame timestamps must be strictly increasing")
    return timestamps.reshape(-1, temporal_patch_size).mean(axis=1)


def video_token_groups(
    input_ids: np.ndarray,
    video_grid_thw: np.ndarray,
    video_token_id: int,
    vision_start_id: int,
    vision_end_id: int,
    merge_size: int,
    expected_grid_times: int,
) -> np.ndarray:
    """Return [temporal_grid, spatial_token] positions separated by timestamp text."""
    identifiers = np.asarray(input_ids)
    grid = np.asarray(video_grid_thw)
    if identifiers.ndim != 1 or grid.shape != (1, 3) or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Expected one token sequence and one integer [T,H,W] video grid")
    temporal_bins, grid_height, grid_width = map(int, grid[0])
    if min(temporal_bins, grid_height, grid_width, merge_size) < 1:
        raise ValueError("Grid dimensions and merge_size must be positive")
    if temporal_bins != expected_grid_times or grid_height % merge_size or grid_width % merge_size:
        raise ValueError("Video grid changed temporal sampling or has invalid spatial merge dimensions")
    spatial_tokens = (grid_height // merge_size) * (grid_width // merge_size)
    positions = np.flatnonzero(identifiers == video_token_id)
    if len(positions) != temporal_bins * spatial_tokens:
        raise ValueError("Video token count does not match the processor grid")
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    if len(groups) != temporal_bins or any(len(group) != spatial_tokens for group in groups):
        raise ValueError("Qwen3-VL requires one separate video-token block per temporal grid")
    for group in groups:
        if group[0] == 0 or group[-1] + 1 >= len(identifiers):
            raise ValueError("Video-token block lacks vision delimiters")
        if identifiers[group[0] - 1] != vision_start_id or identifiers[group[-1] + 1] != vision_end_id:
            raise ValueError("Unexpected timestamp/vision delimiter layout")
    if np.count_nonzero(identifiers == vision_start_id) != temporal_bins:
        raise ValueError("Unexpected extra visual blocks in the prompt")
    return np.stack(groups)


class Qwen3Backend:
    def __init__(self, model: Any, processor: Any, metadata: dict[str, Any], device: str):
        self.model = model
        self.processor = processor
        self.metadata = metadata
        self.device = device
        self.last_generation_stats: dict[str, Any] = {}
        self.layer_count = int(model.config.text_config.num_hidden_layers)
        vision_config = model.config.vision_config
        video_processor = processor.video_processor
        self.temporal_patch_size = int(video_processor.temporal_patch_size)
        self.merge_size = int(video_processor.merge_size)
        self.patch_size = int(video_processor.patch_size)
        if (
            self.temporal_patch_size != 2
            or self.temporal_patch_size != vision_config.temporal_patch_size
            or self.merge_size != vision_config.spatial_merge_size
            or self.patch_size != vision_config.patch_size
        ):
            raise ValueError("Qwen3 processor/model patch configurations disagree")
        for token, identifier in [
            ("<|video_pad|>", model.config.video_token_id),
            ("<|vision_start|>", model.config.vision_start_token_id),
            ("<|vision_end|>", model.config.vision_end_token_id),
        ]:
            if processor.tokenizer.convert_tokens_to_ids(token) != identifier:
                raise ValueError("Tokenizer visual-token IDs disagree with model configuration")

    def _question(self, question: str) -> None:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if any(token in question for token in ["<|video_pad|>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>"]):
            raise ValueError("Question text must not inject model control tokens")

    def _batch(self, tensors: dict[str, Any], metadata: dict[str, Any], positions: np.ndarray | None = None) -> PreparedBatch:
        if tensors["input_ids"].shape[0] != 1 or tensors["input_ids"].shape[1] > 4096:
            raise ValueError("This small-model backend requires batch=1 and at most 4096 input tokens")
        converted = {}
        for name, value in tensors.items():
            converted[name] = value.to(device=self.device, dtype=self.model.dtype) if name.startswith("pixel_values") else value.to(self.device)
        metadata = {**self.metadata, **metadata}
        metadata["input_tokens"] = int(tensors["input_ids"].shape[1])
        metadata["prompt_sha256"] = hashlib.sha256(metadata["rendered_prompt"].encode("utf-8")).hexdigest()
        return PreparedBatch(converted, metadata, positions)

    def prepare_text(self, question: str) -> PreparedBatch:
        self._question(question)
        messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return self._batch(inputs, {"rendered_prompt": prompt, "visual_tokens": 0, "modality": "text"})

    def prepare_video(
        self,
        frames: list[Any],
        timestamps_s: list[float],
        question: str,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> PreparedBatch:
        self._question(question)
        grid_times = temporal_grid(timestamps_s, self.temporal_patch_size)
        if len(frames) != len(timestamps_s):
            raise ValueError("frames and timestamps_s must have identical lengths")
        factor = self.patch_size * self.merge_size
        if not isinstance(max_pixels, int) or max_pixels < factor**2:
            raise ValueError("max_pixels must accommodate at least one merged spatial token")
        arrays = [np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame) for frame in frames]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("All frames must have identical spatial dimensions")
        if any(array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8 for array in arrays):
            raise ValueError("Frames must be RGB uint8 arrays or PIL images")
        if min(arrays[0].shape[:2]) < factor:
            raise ValueError(f"Frames must have height and width at least {factor} pixels")
        visual = self.processor.video_processor(
            videos=[np.stack(arrays)],
            do_sample_frames=False,
            return_metadata=False,
            return_tensors="pt",
            size={"shortest_edge": len(frames) * factor**2, "longest_edge": len(frames) * max_pixels},
        )
        grid = visual["video_grid_thw"].cpu().numpy()
        if grid.shape != (1, 3) or int(grid[0, 0]) != len(grid_times):
            raise ValueError("Video preprocessing padded, discarded, or resampled input frames")
        resized_pixels = int(grid[0, 1] * grid[0, 2]) * self.patch_size**2
        if resized_pixels > max_pixels:
            raise ValueError("Rounded resize exceeds max_pixels; increase budget or adjust source aspect ratio")
        spatial_tokens = int(grid[0, 1] * grid[0, 2]) // self.merge_size**2
        expanded_video = "".join(
            f"<{timestamp:.1f} seconds><|vision_start|>" + "<|video_pad|>" * spatial_tokens + "<|vision_end|>"
            for timestamp in grid_times
        )
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        if prompt.count(placeholder) != 1:
            raise ValueError("Unexpected Qwen3 chat-template video placeholder")
        prompt = prompt.replace(placeholder, expanded_video, 1)
        text = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        positions = video_token_groups(
            text["input_ids"][0].cpu().numpy(), grid, self.model.config.video_token_id,
            self.model.config.vision_start_token_id, self.model.config.vision_end_token_id,
            self.merge_size, len(grid_times),
        )
        intervals = np.diff(grid_times)
        uniform = bool(len(intervals) < 2 or np.allclose(intervals, np.median(intervals), rtol=1e-5, atol=1e-7))
        metadata = {
            "rendered_prompt": prompt, "modality": "video", "visual_tokens": int(positions.size),
            "source_timestamps_s": list(map(float, timestamps_s)), "hidden_timestamps_s": grid_times.tolist(),
            "frame_groups": np.arange(len(frames)).reshape(-1, 2).tolist(), "video_grid_thw": grid[0].tolist(),
            "timestamp_prompt_precision_decimals": 1, "uniform_hidden_sampling": uniform,
            "hidden_sampling_hz": float(1.0 / np.median(intervals)) if len(intervals) and uniform else None,
            "temporal_patch_size": self.temporal_patch_size, "patch_size": self.patch_size,
            "spatial_merge_size": self.merge_size, "max_pixels_per_frame": max_pixels,
            "resized_pixels_per_frame": resized_pixels, "processor_resamples_frames": False,
            "timestamp_rendering": "official Qwen3 per-grid format, actual adjacent-frame centers",
        }
        return self._batch({**text, **visual}, metadata, positions)

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

    def generate(self, batch: PreparedBatch, max_new_tokens: int = 8, seed: int = 0) -> str:
        import torch

        if max_new_tokens < 1 or max_new_tokens > 256:
            raise ValueError("max_new_tokens must be within 1..256")
        torch.manual_seed(seed)
        started = self._start_measurement()
        with torch.inference_mode():
            output = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                use_cache=True, output_hidden_states=False, return_dict_in_generate=False,
                logits_to_keep=1,
            )
        new_tokens = output[0, batch["input_ids"].shape[1]:]
        answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.last_generation_stats = {
            **self._finish_measurement(started), "input_tokens": int(batch["input_ids"].shape[1]),
            "output_tokens": int(new_tokens.numel()), "visual_tokens": batch.metadata["visual_tokens"],
            "model_id": self.metadata["model_id"], "model_revision": self.metadata["model_revision"],
            "device": self.device, "seed": seed, "do_sample": False,
            "timing_scope": "model generation and decoding; excludes image I/O and input preprocessing",
            "rendered_prompt": batch.metadata["rendered_prompt"], "provenance": batch.metadata,
        }
        return answer

    def generate_response(
        self, question: str, frame_paths: list[str], timestamps_s: list[float],
        max_new_tokens: int = 8, seed: int = 0, max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> dict[str, Any]:
        from PIL import Image

        if not frame_paths:
            if timestamps_s:
                raise ValueError("Text-only inputs cannot include video timestamps")
            batch = self.prepare_text(question)
        else:
            frames = []
            for path in frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("RGB").copy())
            batch = self.prepare_video(frames, timestamps_s, question, max_pixels=max_pixels)
        raw_text = self.generate(batch, max_new_tokens=max_new_tokens, seed=seed)
        return {"raw_text": raw_text, **self.last_generation_stats}

    def choose_response(
        self, question: str, frame_paths: list[str], timestamps_s: list[float],
        option_count: int, max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> dict[str, Any]:
        """Choose the highest-logit allowed letter; ground truth is never an input."""
        from PIL import Image

        if not frame_paths:
            if timestamps_s:
                raise ValueError("Text-only inputs cannot include timestamps")
            batch = self.prepare_text(question)
        else:
            frames = []
            for path in frame_paths:
                with Image.open(path) as image:
                    frames.append(image.convert("RGB").copy())
            batch = self.prepare_video(frames, timestamps_s, question, max_pixels=max_pixels)
        return self.choose_batch(batch, option_count)

    def choose_batch(self, batch: PreparedBatch, option_count: int) -> dict[str, Any]:
        """Read only the supplied memory batch using a fixed allowed-letter decision."""
        import torch

        if isinstance(option_count, bool) or not isinstance(option_count, int) or not 2 <= option_count <= 26:
            raise ValueError("option_count must be within 2..26")
        letters = [chr(ord("A") + index) for index in range(option_count)]
        tokenized = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
        if any(len(tokens) != 1 for tokens in tokenized):
            raise ValueError("Constrained choice requires single-token option letters")
        if len({tokens[0] for tokens in tokenized}) != option_count:
            raise ValueError("Option letters do not have unique tokens")
        started = self._start_measurement()
        with torch.inference_mode():
            output = self.model(**batch, use_cache=False, output_hidden_states=False,
                                return_dict=True, logits_to_keep=1)
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
            "visual_tokens": batch.metadata["visual_tokens"],
            "model_id": self.metadata["model_id"], "model_revision": self.metadata["model_revision"],
            "decoding": "argmax_single_token_allowed_option_letters_v1", "do_sample": False,
            "timing_scope": "model forward and constrained choice; excludes input preparation",
            "rendered_prompt": batch.metadata["rendered_prompt"], "provenance": batch.metadata,
        }

    def extract(self, batch: PreparedBatch, layers: tuple[int, ...]) -> ExtractionResult:
        """Capture selected decoder inputs after preceding DeepStack additions; N is final norm."""
        import torch

        if batch.video_positions is None:
            raise ValueError("Hidden-state extraction requires a prepared video batch")
        if (
            not layers
            or any(isinstance(layer, bool) or not isinstance(layer, (int, np.integer)) for layer in layers)
            or len(set(layers)) != len(layers)
            or any(layer < 0 or layer > self.layer_count for layer in layers)
        ):
            raise ValueError(f"Layer indices must be unique integers in 0..{self.layer_count}")
        positions = torch.as_tensor(batch.video_positions, device=self.device)
        collected: dict[int, np.ndarray] = {}
        handles = []

        def capture(layer_index: int, states: Any) -> None:
            pooled = states[0, positions].float().mean(dim=1).cpu().numpy()
            if not np.isfinite(pooled).all():
                raise RuntimeError(f"Nonfinite hidden states at layer {layer_index}")
            collected[layer_index] = pooled

        def make_hook(layer_index: int):
            def hook(module: Any, inputs: tuple[Any, ...]) -> None:
                capture(layer_index, inputs[0])
            return hook

        started = self._start_measurement()
        try:
            for layer in layers:
                if layer < self.layer_count:
                    handles.append(self.model.model.language_model.layers[layer].register_forward_pre_hook(make_hook(layer)))
            with torch.inference_mode():
                outputs = self.model.model(**batch, output_hidden_states=False, use_cache=False, return_dict=True)
                if self.layer_count in layers:
                    capture(self.layer_count, outputs.last_hidden_state)
        finally:
            for handle in handles:
                handle.remove()
        if set(collected) != set(layers):
            raise RuntimeError("Not all requested decoder stages were captured")
        metadata = {
            **batch.metadata, **self._finish_measurement(started), "layers": list(layers), "pooling": "spatial_mean",
            "layer_indexing": "0=decoder input visual projection; k<N=decoder layer k input after prior DeepStack; N=final RMSNorm",
            "capture_method": "selected forward pre-hooks; no full hidden-state tuple; no LM logits",
            "deepstack_visual_indexes": list(self.model.config.vision_config.deepstack_visual_indexes),
        }
        return ExtractionResult(collected, np.asarray(batch.metadata["hidden_timestamps_s"], dtype=np.float64), metadata)


def load_backend(model_path: str = DEFAULT_MODEL, device: str = "cuda", revision: str | None = None) -> Qwen3Backend:
    """Load once from local files/cache only; never download model weights."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    version = importlib.metadata.version("transformers")
    if version != VERIFIED_TRANSFORMERS:
        raise RuntimeError(f"Verified backend requires transformers=={VERIFIED_TRANSFORMERS}, installed {version}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter; use the isolated CUDA environment")
    if not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError("Only CPU and CUDA devices are supported")
    requested_revision = revision or (DEFAULT_REVISION if model_path == DEFAULT_MODEL else None)
    model_directory = Path(model_path).expanduser()
    if not model_directory.is_dir():
        model_directory = Path(snapshot_download(
            repo_id=model_path, revision=requested_revision, local_files_only=True, token=False,
        ))
    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(str(model_directory), **options)
    if not isinstance(config, Qwen3VLConfig):
        raise ValueError("This backend requires a Qwen3-VL checkpoint")
    processor = AutoProcessor.from_pretrained(str(model_directory), **options)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    if device.startswith("cuda") and dtype == torch.float32:
        dtype = torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_directory), dtype=dtype, attn_implementation="sdpa", **options
    ).to(device).eval()
    model.requires_grad_(False)
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision is None and len(model_directory.name) == 40 and all(character in "0123456789abcdefABCDEF" for character in model_directory.name):
        resolved_revision = model_directory.name
    metadata = {
        "model_id": model_path, "model_revision": resolved_revision or "unresolved-local",
        "processor_revision": resolved_revision or "unresolved-local", "requested_revision": requested_revision,
        "local_snapshot_path": str(model_directory.resolve()),
        "transformers": version, "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "dtype": str(dtype), "device": device, "attention_implementation": "sdpa", "local_files_only": True,
        "max_input_tokens": 4096, "model_class": type(model).__name__,
        "device_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "CPU",
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.startswith("cuda") else None,
    }
    return Qwen3Backend(model, processor, metadata, device)
