"""One development video, bounded inference, and exact decoder branch capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .resource_guard import ResourceGuard, configure_threads


INSTRUCTION = "Describe the visible events."
BRANCHES = ("decoder_input", "attention_delta", "pre_ffn", "ffn_input", "ffn_delta", "post_ffn")
DEVELOPMENT_IDS = {"video_10993", "video_1580", "video_2564", "video_3501", "video_6860", "video_8241", "video_874", "video_9253"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def capture_branches(backend, batch, layers: tuple[int, ...]) -> tuple[dict, dict]:
    import numpy as np
    import torch

    if batch.video_positions is None or not batch.metadata["uniform_hidden_sampling"]:
        raise ValueError("Probe requires uniformly sampled video token groups")
    if not layers or len(set(layers)) != len(layers) or any(index < 0 or index >= backend.layer_count for index in layers):
        raise ValueError("Decoder layer indices must be unique and in range")
    positions = torch.as_tensor(batch.video_positions, device=backend.device)
    arrays = {"timestamps_s": np.asarray(batch.metadata["hidden_timestamps_s"], dtype=np.float64)}
    pending = {index: {} for index in layers}
    identities = {}
    handles = []
    captured = set()

    def sample(index, branch, states):
        key = f"layer_{index}_{branch}"
        if key in captured:
            raise RuntimeError(f"Repeated hook invocation: {key}")
        selected = states[0, positions].detach().to("cpu").clone()
        values = selected.float()
        if not torch.isfinite(values).all():
            raise RuntimeError(f"Nonfinite branch: {key}")
        arrays[key] = values.mean(dim=1).numpy()
        arrays[f"{key}_spatial_mean_square"] = values.square().mean(dim=(1, 2)).numpy()
        arrays[f"{key}_maximum_token_rms"] = values.square().mean(dim=2).sqrt().max(dim=1).values.numpy()
        captured.add(key)
        return selected

    def identity(index, name, left, right, observed):
        expected = left + right
        error = (expected.float() - observed.float()).abs()
        values = {
            "exact_in_observed_dtype": bool(torch.equal(expected, observed)),
            "maximum_absolute_error": float(error.max()), "dtype": str(observed.dtype),
            "elements_checked": int(observed.numel()),
            "scope": "all visual tokens, all hidden dimensions, before spatial pooling",
        }
        identities[f"layer_{index}_{name}"] = values
        if not values["exact_in_observed_dtype"]:
            raise RuntimeError(f"Residual identity failed: layer {index} {name}")

    def decoder_pre(index):
        def hook(module, arguments):
            pending[index]["input"] = sample(index, "decoder_input", arguments[0])
        return hook

    def attention_post(index):
        def hook(module, arguments, output):
            pending[index]["attention"] = sample(index, "attention_delta", output[0])
        return hook

    def norm_pre(index):
        def hook(module, arguments):
            observed = sample(index, "pre_ffn", arguments[0])
            identity(index, "attention", pending[index].pop("input"), pending[index].pop("attention"), observed)
            pending[index]["pre"] = observed
        return hook

    def mlp_pre(index):
        def hook(module, arguments):
            sample(index, "ffn_input", arguments[0])
        return hook

    def mlp_post(index):
        def hook(module, arguments, output):
            pending[index]["ffn"] = sample(index, "ffn_delta", output)
        return hook

    def decoder_post(index):
        def hook(module, arguments, output):
            observed = sample(index, "post_ffn", output)
            before = pending[index].pop("pre")
            delta = pending[index].pop("ffn")
            identity(index, "ffn", before, delta, observed)
            rounding = observed.float() - before.float() - delta.float()
            arrays[f"layer_{index}_ffn_rounding"] = rounding.mean(dim=1).numpy()
        return hook

    try:
        for index in layers:
            decoder = backend.model.model.language_model.layers[index]
            handles.extend([
                decoder.register_forward_pre_hook(decoder_pre(index)),
                decoder.self_attn.register_forward_hook(attention_post(index)),
                decoder.post_attention_layernorm.register_forward_pre_hook(norm_pre(index)),
                decoder.mlp.register_forward_pre_hook(mlp_pre(index)),
                decoder.mlp.register_forward_hook(mlp_post(index)),
                decoder.register_forward_hook(decoder_post(index)),
            ])
        with torch.inference_mode():
            output = backend.model.model(**batch, use_cache=False, output_hidden_states=False,
                                         output_attentions=False, return_dict=True)
            del output
    finally:
        for handle in handles:
            handle.remove()
    expected = {f"layer_{index}_{branch}" for index in layers for branch in BRANCHES}
    if captured != expected or any(pending.values()):
        raise RuntimeError("Incomplete branch capture")
    return arrays, identities


def run(arguments) -> None:
    configure_threads()
    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = arguments.source_manifest.resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (len(records) != 8 or {record["video_id"] for record in records} != DEVELOPMENT_IDS
            or any(record.get("dataset") != "perception_test_sample" for record in records)):
        raise ValueError("This pilot is restricted to the original eight development videos")
    candidates = sorted(records, key=lambda record: record["video_id"])
    if arguments.video_id is not None:
        candidates = [record for record in candidates if record["video_id"] == arguments.video_id]
    if not candidates:
        raise ValueError("Requested video is absent from the original development manifest")
    record = candidates[0]
    layers = (0, 13, 27)
    metadata = {
        "status": "started", "purpose": "mechanism_engineering_pilot_not_memory_effectiveness",
        "video_id": record["video_id"], "selection": "first lexicographic development ID unless explicitly specified",
        "layers": list(layers), "frame_count": 16, "instruction": INSTRUCTION,
        "source_manifest_sha256": digest(manifest), "probe_source_sha256": digest(Path(__file__)),
        "resource_guard_source_sha256": digest(Path(__file__).with_name("resource_guard.py")),
        "questions_or_answers_used": False, "device": arguments.device,
        "precision_is_separate_from_frozen_QA_run": True,
    }
    write_json(output / "probe.json", metadata)
    try:
        with ResourceGuard(output, arguments.device) as guard:
            guard.phase = "prepare_16_frames"
            import cv2
            import numpy as np
            from PIL import Image
            from .prepare_frames import prepare_video_frames
            from .qwen3_backend import load_backend, DEFAULT_MAX_PIXELS

            cv2.setNumThreads(2)
            video = Path(record["video_path"])
            if not video.is_absolute():
                video = manifest.parent / video
            if digest(video) != record["video_sha256"]:
                raise ValueError("Source video digest differs from the development manifest")
            prepared = prepare_video_frames(video, output / "frames", 16, 512)
            frame_record = {**record, **prepared, "video_path": str(video.resolve()),
                            "sample_id": f"{record['dataset']}:{record['video_id']}:ffn16"}
            (output / "frames.manifest.jsonl").write_text(json.dumps(frame_record) + "\n", encoding="utf-8")
            metadata["frame_sha256"] = [digest(Path(filename)) for filename in prepared["frames"]]
            metadata["frames_manifest_sha256"] = digest(output / "frames.manifest.jsonl")
            guard.phase = "load_model"
            guard.configure_torch()
            backend = load_backend(device=arguments.device)
            frames = []
            for filename in prepared["frames"]:
                with Image.open(filename) as source_image:
                    frames.append(source_image.convert("RGB").copy())
            guard.phase = "prepare_model_input"
            batch = backend.prepare_video(frames, prepared["timestamps_s"], INSTRUCTION, DEFAULT_MAX_PIXELS)
            if batch.metadata["input_tokens"] > 1024 or batch.metadata["visual_tokens"] > 512:
                raise ValueError("Engineering pilot input exceeds its 1024 total / 512 visual token cap")
            guard.phase = "single_forward"
            started = time.perf_counter()
            arrays, identities = capture_branches(backend, batch, layers)
            import torch

            if arguments.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            guard.phase = "save"
            with (output / "branches.npz").open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            metadata.update({
                "status": "complete", "model": backend.metadata, "batch": batch.metadata,
                "forward_seconds": elapsed, "identities": identities,
                "branches_sha256": digest(output / "branches.npz"),
                "source_frame_timestamps_s": prepared["source_frame_timestamps_s"],
                "target_frame_timestamps_s": prepared["timestamps_s"],
                "source_to_target_maximum_error_s": prepared["sampling"]["max_abs_resampling_error_s"],
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if arguments.device == "cuda" else None,
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if arguments.device == "cuda" else None,
                "layer_indexing": "zero-based decoder layer; post_ffn captured before any outer DeepStack addition",
                "capture": "immediate CPU clone of visual tokens; no hidden-state tuple, no LM logits or generation",
                "spectral_axis": "uniform target-grid frame-pair centers; source PTS retained separately",
                "limitations": ["One development video cannot establish a mechanism or task improvement",
                                "Eight temporal bins provide only four non-DC real-FFT bins",
                                "Frame resampling has no temporal anti-alias filter",
                                "Resource limits cannot guarantee operating-system stability"],
            })
            write_json(output / "probe.json", metadata)
            del backend, batch
            if arguments.device == "cuda":
                torch.cuda.empty_cache()
    except BaseException as error:
        metadata.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "probe.json", metadata)
        raise
    print(json.dumps({"status": "complete", "output": str(output), "forward_seconds": elapsed,
                      "identities_passed": len(identities)}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("data/perception_test_sample/videos.jsonl"))
    parser.add_argument("--video-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        parser.exit(2, f"FFN probe failed: {error}\n")


if __name__ == "__main__":
    main()
