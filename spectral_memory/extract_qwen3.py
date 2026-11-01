"""Extract locally cached Qwen3-VL representations from a pre-sampled frame manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .extract_qwen import load_records, parse_layers
from .qwen3_backend import DEFAULT_MAX_PIXELS, DEFAULT_MODEL, DEFAULT_REVISION, load_backend


FIXED_INSTRUCTION = "Describe the visible events."


def extract(arguments: argparse.Namespace) -> None:
    from PIL import Image

    input_path = arguments.input.resolve()
    output_dir = arguments.output_dir.resolve()
    records = load_records(input_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty")
    if any(len(record["frames"]) < 16 or len(record["frames"]) % 2 for record in records):
        raise ValueError("Spectral extraction requires at least 16 frames and complete pairs")
    backend = load_backend(arguments.model, device=arguments.device, revision=arguments.revision)
    layers = tuple(parse_layers(arguments.layers, backend.layer_count))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hidden").mkdir()
    provenance = {
        **backend.metadata, "status": "incomplete", "completed_samples": 0, "layers": list(layers),
        "instruction": FIXED_INSTRUCTION, "max_pixels_per_frame": arguments.max_pixels,
        "input_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    run_path = output_dir / "extraction.json"
    run_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    manifests = {layer: [] for layer in layers}
    for record_index, record in enumerate(records):
        frames = []
        frame_hashes = []
        for filename in record["frames"]:
            frame_path = input_path.parent / filename
            frame_hashes.append(hashlib.sha256(frame_path.read_bytes()).hexdigest())
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
        batch = backend.prepare_video(frames, record["timestamps_s"], FIXED_INSTRUCTION, arguments.max_pixels)
        if not batch.metadata["uniform_hidden_sampling"]:
            raise ValueError("Frequency-domain extraction requires uniform temporal-grid timestamps")
        result = backend.extract(batch, layers)
        sample_hash = hashlib.sha256(record["sample_id"].encode("utf-8")).hexdigest()[:20]
        for layer, hidden in result.hidden_states.items():
            relative_path = Path("hidden") / f"{record_index:06d}_{sample_hash}.layer_{layer}.npz"
            with (output_dir / relative_path).open("xb") as stream:
                np.savez_compressed(stream, hidden_states=hidden, timestamps_s=result.timestamps_s)
            extracted = {
                **record, "hidden_path": relative_path.as_posix(), "hidden_timestamps_s": result.timestamps_s.tolist(),
                "model_id": backend.metadata["model_id"], "model_revision": backend.metadata["model_revision"],
                "layer_index": layer, "pooling": "spatial_mean",
                "frames": [str((input_path.parent / filename).resolve()) for filename in record["frames"]],
                "extraction": {**result.metadata, "layer_index": layer, "frame_sha256": frame_hashes},
            }
            manifests[layer].append(extracted)
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", default="0,14,28")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        extract(arguments)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Qwen3 extraction failed: {error}\n")


if __name__ == "__main__":
    main()
