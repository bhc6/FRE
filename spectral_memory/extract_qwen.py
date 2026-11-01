"""Extract timestamp-aligned Qwen2.5-VL video-token representations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np


VERIFIED_TRANSFORMERS = "4.57.1"
FIXED_INSTRUCTION = "Describe the visible events."
VIDEO_PLACEHOLDER = "<|video_pad|>"
PROMPT_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|>{video_tokens}<|vision_end|>"
    + FIXED_INSTRUCTION
    + "<|im_end|>\n<|im_start|>assistant\n"
)


def validate_timestamps(timestamps_s: np.ndarray, temporal_patch_size: int) -> tuple[np.ndarray, float]:
    """Return patch-center timestamps and sampled-frame FPS; never resample."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError("timestamps_s must be a finite one-dimensional array")
    if temporal_patch_size < 1 or len(timestamps) < 8 * temporal_patch_size:
        raise ValueError("At least eight complete temporal patches are required")
    if len(timestamps) % temporal_patch_size:
        raise ValueError("Frame count must be divisible by temporal_patch_size; padding is forbidden")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError("Frame timestamps must be strictly increasing")
    interval_s = float(np.median(intervals))
    if not np.allclose(intervals, interval_s, rtol=1e-5, atol=1e-7):
        raise ValueError("Nonuniform timestamps: explicitly resample upstream and preserve provenance")
    return timestamps.reshape(-1, temporal_patch_size).mean(axis=1), 1.0 / interval_s


def video_layout(
    input_ids: np.ndarray,
    video_token_id: int,
    video_grid_thw: np.ndarray,
    merge_size: int,
    expected_temporal_bins: int,
) -> tuple[np.ndarray, int, int]:
    """Validate one video and return its token positions and temporal/spatial sizes."""
    identifiers = np.asarray(input_ids)
    grid = np.asarray(video_grid_thw)
    if identifiers.ndim != 1 or grid.shape != (1, 3):
        raise ValueError("Only a single unpadded video sequence per extraction is supported")
    if not np.issubdtype(grid.dtype, np.integer) or np.any(grid <= 0):
        raise ValueError("video_grid_thw must contain positive integer dimensions")
    temporal_bins, grid_height, grid_width = map(int, grid[0])
    if merge_size < 1 or grid_height % merge_size or grid_width % merge_size:
        raise ValueError("Spatial grid is not divisible by the processor merge_size")
    if temporal_bins != expected_temporal_bins:
        raise ValueError("Processor changed temporal sampling or padded frames")
    spatial_tokens = (grid_height // merge_size) * (grid_width // merge_size)
    positions = np.flatnonzero(identifiers == video_token_id)
    if positions.size != temporal_bins * spatial_tokens:
        raise ValueError("Video token count disagrees with video_grid_thw / merge_size squared")
    if positions.size > 1 and np.any(np.diff(positions) != 1):
        raise ValueError("Expected exactly one contiguous video-token block")
    return positions, temporal_bins, spatial_tokens


def pool_video_tokens(
    hidden_states: np.ndarray,
    input_ids: np.ndarray,
    video_token_id: int,
    video_grid_thw: np.ndarray,
    merge_size: int,
    expected_temporal_bins: int,
) -> np.ndarray:
    """Spatially mean-pool one layer in restored temporal-major video-token order."""
    hidden = np.asarray(hidden_states)
    if hidden.ndim != 2 or len(hidden) != len(input_ids) or not np.isfinite(hidden).all():
        raise ValueError("hidden_states must be finite [sequence_length, hidden_dimension]")
    positions, temporal_bins, spatial_tokens = video_layout(
        input_ids, video_token_id, video_grid_thw, merge_size, expected_temporal_bins
    )
    return hidden[positions].reshape(temporal_bins, spatial_tokens, -1).mean(axis=1)


def load_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    required = {
        "sample_id", "video_id", "subject_id", "session_id", "event_start_s",
        "event_end_s", "frames", "timestamps_s",
    }
    with input_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"Line {line_number}: missing required fields {sorted(required)}")
            sample_id = record["sample_id"]
            if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in sample_ids:
                raise ValueError(f"Line {line_number}: sample_id must be unique and nonempty")
            sample_ids.add(sample_id)
            frames, timestamps = record["frames"], record["timestamps_s"]
            if not isinstance(frames, list) or not isinstance(timestamps, list) or len(frames) != len(timestamps):
                raise ValueError(f"Line {line_number}: frames and timestamps_s must be equally sized lists")
            if not all(isinstance(frame, str) and frame for frame in frames):
                raise ValueError(f"Line {line_number}: frame paths must be nonempty strings")
            validate_timestamps(np.asarray(timestamps), 1)
            start, end = float(record["event_start_s"]), float(record["event_end_s"])
            if not np.isfinite([start, end]).all() or start >= end:
                raise ValueError(f"Line {line_number}: invalid event interval")
            if timestamps[0] < start - 1e-7 or timestamps[-1] > end + 1e-7:
                raise ValueError(f"Line {line_number}: frame timestamps fall outside the event interval")
            for frame in frames:
                if not (input_path.parent / frame).is_file():
                    raise FileNotFoundError(f"Line {line_number}: frame not found: {frame}")
            records.append(record)
    if not records:
        raise ValueError("Input manifest is empty")
    return records


def parse_layers(value: str, layer_count: int) -> list[int]:
    if value == "auto":
        return sorted(set(np.linspace(0, layer_count, 7).round().astype(int).tolist()))
    try:
        layers = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--layers must be 'auto' or comma-separated integer tuple indices") from error
    if not layers or len(set(layers)) != len(layers) or any(layer < 0 or layer > layer_count for layer in layers):
        raise ValueError(f"Layer indices must be unique and within 0..{layer_count}")
    return sorted(layers)


def extract(arguments: argparse.Namespace) -> None:
    input_path = arguments.input.resolve()
    output_dir = arguments.output_dir.resolve()
    records = load_records(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; choose a new directory to preserve previous runs")
    version = importlib.metadata.version("transformers")
    if version != VERIFIED_TRANSFORMERS:
        raise RuntimeError(f"Verified extraction requires transformers=={VERIFIED_TRANSFORMERS}; installed {version}")
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor

    if arguments.max_pixels < 28 * 28 or arguments.max_tokens < 1:
        raise ValueError("--max-pixels must be at least 784 and --max-tokens must be positive")
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install a matching CUDA runtime")
    download_options = {
        "local_files_only": not arguments.allow_download,
        "revision": arguments.revision,
        "trust_remote_code": False,
    }
    config = AutoConfig.from_pretrained(arguments.model, **download_options)
    if not isinstance(config, Qwen2_5_VLConfig):
        raise ValueError("This adapter supports Qwen2.5-VL only; Qwen3-VL needs a separately verified adapter")
    layers = parse_layers(arguments.layers, config.text_config.num_hidden_layers)
    processor = Qwen2VLImageProcessor.from_pretrained(arguments.model, **download_options)
    tokenizer = AutoTokenizer.from_pretrained(arguments.model, **download_options)
    temporal_patch_size = int(processor.temporal_patch_size)
    merge_size = int(processor.merge_size)
    if (
        temporal_patch_size != config.vision_config.temporal_patch_size
        or merge_size != config.vision_config.spatial_merge_size
        or processor.patch_size != config.vision_config.patch_size
    ):
        raise ValueError("Processor and model patch/merge configuration disagree")
    for record in records:
        validate_timestamps(np.asarray(record["timestamps_s"]), temporal_patch_size)
    video_token_id = int(config.video_token_id)
    if tokenizer.convert_tokens_to_ids(VIDEO_PLACEHOLDER) != video_token_id:
        raise ValueError("Tokenizer video token does not match the model configuration")
    dtype_name = arguments.dtype
    if dtype_name == "auto":
        dtype_name = "bfloat16" if arguments.device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
    dtype = getattr(torch, dtype_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        arguments.model, dtype=dtype, attn_implementation="sdpa", **download_options
    ).to(arguments.device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hidden").mkdir()
    provenance = {
        "status": "incomplete", "model": arguments.model, "revision": arguments.revision,
        "resolved_revision": getattr(config, "_commit_hash", None), "transformers": version,
        "torch": torch.__version__, "dtype": dtype_name, "device": arguments.device,
        "layers": layers, "layer_indexing": "hidden_states tuple; final index includes final RMSNorm",
        "prompt_template": PROMPT_TEMPLATE, "pooling": "spatial_mean", "min_pixels": 4 * 28 * 28,
        "max_pixels": arguments.max_pixels, "max_tokens": arguments.max_tokens,
        "temporal_patch_size": temporal_patch_size, "spatial_merge_size": merge_size,
        "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_manifest": str(input_path), "completed_samples": 0,
    }
    run_path = output_dir / "extraction.json"
    run_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    manifests = {layer: [] for layer in layers}
    for record_index, record in enumerate(records):
        frame_arrays = []
        frame_hashes = []
        for frame in record["frames"]:
            frame_path = input_path.parent / frame
            frame_hashes.append(hashlib.sha256(frame_path.read_bytes()).hexdigest())
            with Image.open(frame_path) as source:
                frame_arrays.append(np.asarray(source.convert("RGB")))
        if len({frame.shape for frame in frame_arrays}) != 1:
            raise ValueError(f"{record['sample_id']}: all pre-sampled frames must have the same dimensions")
        timestamps, fps = validate_timestamps(np.asarray(record["timestamps_s"]), temporal_patch_size)
        visual_inputs = processor(
            images=None, videos=[np.stack(frame_arrays)], return_tensors="pt",
            min_pixels=4 * 28 * 28, max_pixels=arguments.max_pixels,
        )
        grid = visual_inputs["video_grid_thw"].cpu().numpy()
        token_count = int(np.prod(grid[0])) // merge_size**2
        prompt = PROMPT_TEMPLATE.format(video_tokens=VIDEO_PLACEHOLDER * token_count)
        text_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if text_inputs["input_ids"].shape[1] > arguments.max_tokens:
            raise ValueError(f"{record['sample_id']}: token limit exceeded; shorten/resample upstream")
        identifiers = text_inputs["input_ids"][0].cpu().numpy()
        positions, temporal_bins, spatial_tokens = video_layout(
            identifiers, video_token_id, grid, merge_size, len(timestamps)
        )
        batch = {**text_inputs, **visual_inputs}
        batch = {name: value.to(arguments.device) for name, value in batch.items()}
        batch["second_per_grid_ts"] = torch.tensor(
            [temporal_patch_size / fps], dtype=torch.float32, device=arguments.device
        )
        with torch.inference_mode():
            outputs = model.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        if outputs.hidden_states is None or len(outputs.hidden_states) != config.text_config.num_hidden_layers + 1:
            raise RuntimeError("Unexpected hidden-state tuple layout")
        device_positions = torch.as_tensor(positions, device=arguments.device)
        sample_hash = hashlib.sha256(record["sample_id"].encode("utf-8")).hexdigest()[:20]
        for layer in layers:
            visual_hidden = outputs.hidden_states[layer][0, device_positions]
            hidden = visual_hidden.float().reshape(temporal_bins, spatial_tokens, -1).mean(dim=1).cpu().numpy()
            if not np.isfinite(hidden).all():
                raise RuntimeError(f"{record['sample_id']}: nonfinite hidden states in layer {layer}")
            relative_path = Path("hidden") / f"{record_index:06d}_{sample_hash}.layer_{layer}.npz"
            with (output_dir / relative_path).open("xb") as stream:
                np.savez_compressed(stream, hidden_states=hidden, timestamps_s=timestamps)
            extracted = dict(record)
            extracted["frames"] = [str((input_path.parent / frame).resolve()) for frame in record["frames"]]
            extracted["hidden_path"] = relative_path.as_posix()
            extracted["hidden_timestamps_s"] = timestamps.tolist()
            extracted["model_id"] = arguments.model
            extracted["model_revision"] = provenance["resolved_revision"] or arguments.revision
            extracted["layer_index"] = layer
            extracted["pooling"] = "spatial_mean"
            extracted["extraction"] = {
                "layer": layer, "model": arguments.model, "resolved_revision": provenance["resolved_revision"],
                "sampling_fps": fps, "hidden_sampling_hz": fps / temporal_patch_size,
                "seconds_per_temporal_grid": temporal_patch_size / fps, "video_grid_thw": grid[0].tolist(),
                "timestamps_convention": "absolute center of each consecutive temporal frame group",
                "frame_sha256": frame_hashes, "prompt_instruction": FIXED_INSTRUCTION,
            }
            manifests[layer].append(extracted)
        del outputs, visual_hidden, batch
        provenance["completed_samples"] = record_index + 1
        run_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Extracted {record_index + 1}/{len(records)}: {record['sample_id']}", flush=True)
    for layer, layer_records in manifests.items():
        with (output_dir / f"manifest.layer_{layer}.jsonl").open("x", encoding="utf-8") as stream:
            for record in layer_records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    provenance["status"] = "complete"
    run_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL of local RGB frame paths and real timestamps")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main", help="Prefer a pinned model revision for reported experiments")
    parser.add_argument("--layers", default="auto", help="Seven evenly spaced tuple indices, or comma-separated indices")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true", help="Permit downloading model/tokenizer files")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        extract(arguments)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Extraction failed: {error}\n")


if __name__ == "__main__":
    main()
