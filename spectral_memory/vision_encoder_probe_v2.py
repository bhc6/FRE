"""Read-only temporal and spatial spectra of native visual encoder features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil

from .resource_guard import ResourceGuard, configure_threads


PROTOCOL = Path("docs/vision_encoder_probe_protocol_v2.md")
SOURCE_ARCHIVE_LIMIT = 100 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def qwen_premerger_grid(flat, grid_thw, merge_size):
    import numpy as np

    values = np.asarray(flat)
    time_bins, height, width = map(int, grid_thw)
    require(values.ndim == 2 and values.shape[0] == time_bins * height * width and
            height % merge_size == 0 and width % merge_size == 0, "Invalid Qwen pre-merger geometry")
    blocked = values.reshape(time_bins, height // merge_size, width // merge_size, merge_size, merge_size, values.shape[-1])
    return blocked.transpose(0, 1, 3, 2, 4, 5).reshape(time_bins, height, width, values.shape[-1]).copy()


def row_major_grid(values, time_bins, height, width):
    import numpy as np

    values = np.asarray(values)
    require(values.ndim in (2, 3) and values.size == time_bins * height * width * values.shape[-1], "Invalid row-major visual geometry")
    return values.reshape(time_bins, height, width, values.shape[-1]).copy()


def smol_unshuffle_grid(shuffled, height, width, feature_dim, scale_factor):
    import numpy as np

    values = np.asarray(shuffled)
    require(values.ndim == 3 and height == width and height % scale_factor == 0 and
            values.shape[1:] == (height * width // scale_factor**2, feature_dim * scale_factor**2), "Invalid Smol pixel-shuffle geometry")
    grouped = values.reshape(values.shape[0], height // scale_factor, width // scale_factor, scale_factor, scale_factor, feature_dim)
    return grouped.transpose(0, 1, 3, 2, 4, 5).reshape(values.shape[0], height, width, feature_dim).copy()


def spectrum_summary(features, channel_chunk=32):
    import numpy as np

    values = np.asarray(features)
    require(values.ndim == 4 and min(values.shape) > 0 and np.isfinite(values).all(), "Expected finite [T,H,W,D] features")
    time_bins, height, width, channels = values.shape
    multiplicity = np.full(time_bins // 2 + 1, 2., dtype=np.float64)
    multiplicity[0] = 1.
    if time_bins % 2 == 0:
        multiplicity[-1] = 1.
    temporal_power = np.zeros((len(multiplicity), height, width), dtype=np.float64)
    mean_temporal_power = np.zeros(len(multiplicity), dtype=np.float64)
    spatial_power = np.zeros((time_bins, height, width), dtype=np.float64)
    total_energy, mean_total_energy = 0., 0.
    for start in range(0, channels, channel_chunk):
        chunk = values[..., start:start + channel_chunk].astype(np.float64)
        total_energy += float(np.square(chunk).sum())
        temporal = np.fft.rfft(chunk, axis=0, norm="ortho")
        temporal_power += np.square(np.abs(temporal)).sum(axis=-1) * multiplicity[:, None, None]
        del temporal
        spatial = np.fft.fft2(chunk, axes=(1, 2), norm="ortho")
        spatial_power += np.square(np.abs(spatial)).sum(axis=-1)
        del spatial
        averaged = chunk.mean(axis=(1, 2))
        mean_total_energy += float(np.square(averaged).sum())
        mean_temporal = np.fft.rfft(averaged, axis=0, norm="ortho")
        mean_temporal_power += np.square(np.abs(mean_temporal)).sum(axis=-1) * multiplicity
    temporal_by_frequency = temporal_power.sum(axis=(1, 2))
    temporal_dc = float(temporal_by_frequency[0])
    temporal_ac = float(temporal_by_frequency[1:].sum())
    spatial_dc = float(spatial_power[:, 0, 0].sum())
    spatial_ac = float(spatial_power.reshape(time_bins, -1)[:, 1:].sum())
    mean_temporal_dc = float(mean_temporal_power[0])
    mean_temporal_ac = float(mean_temporal_power[1:].sum())
    errors = {"temporal": abs(float(temporal_by_frequency.sum()) - total_energy) / max(total_energy, 1e-20),
              "spatial": abs(float(spatial_power.sum()) - total_energy) / max(total_energy, 1e-20),
              "spatial_mean_temporal": abs(float(mean_temporal_power.sum()) - mean_total_energy) / max(mean_total_energy, 1e-20)}
    require(max(errors.values()) <= 1e-8, "Spectral Parseval identity failed")
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator > 0 else None
    summary = {"shape_time_height_width_channel": list(values.shape), "raw_total_energy": total_energy,
               "energy_per_element": total_energy / values.size,
               "temporal_fullpatch": {"axis": "T independently at each spatial position and channel, then sum power",
                                      "dc_energy": temporal_dc, "ac_energy": temporal_ac,
                                      "ac_fraction": ratio(temporal_ac, total_energy), "frequency_energy": temporal_by_frequency.tolist()},
               "temporal_spatial_mean": {"axis": "average H,W first, then FFT along T independently per channel",
                                         "raw_total_energy": mean_total_energy, "energy_per_element": mean_total_energy / (time_bins * channels),
                                         "dc_energy": mean_temporal_dc, "ac_energy": mean_temporal_ac,
                                         "ac_fraction": ratio(mean_temporal_ac, mean_total_energy),
                                         "frequency_energy": mean_temporal_power.tolist(),
                                         "spatial_cell_count_times_mean_ac_over_fullpatch_ac": ratio(height * width * mean_temporal_ac, temporal_ac)},
               "spatial_fullpatch": {"axes": "H,W independently per time group and channel, then sum power",
                                     "dc_energy": spatial_dc, "ac_energy": spatial_ac,
                                     "ac_fraction": ratio(spatial_ac, total_energy),
                                     "per_time_dc_energy": spatial_power[:, 0, 0].tolist(),
                                     "per_time_ac_energy": spatial_power.reshape(time_bins, -1)[:, 1:].sum(axis=1).tolist()},
               "temporal_frequency_cycles_per_time_group": np.fft.rfftfreq(time_bins).tolist(),
               "temporal_frequency_cycles_per_sampled_clip_extent": (np.fft.rfftfreq(time_bins) * time_bins).tolist(),
               "spatial_frequency_height_cycles_per_cell": np.fft.fftfreq(height).tolist(),
               "spatial_frequency_width_cycles_per_cell": np.fft.fftfreq(width).tolist(),
               "spatial_frequency_height_cycles_per_full_grid_extent": (np.fft.fftfreq(height) * height).tolist(),
               "spatial_frequency_width_cycles_per_full_grid_extent": (np.fft.fftfreq(width) * width).tolist(),
               "normalization": "orthonormal FFT; temporal real FFT interior frequencies doubled; full complex spatial FFT unshifted",
               "channel_policy": "no Fourier transform along feature channels; channel powers summed",
               "parseval_relative_error": errors}
    return summary, {"temporal_power_fhw": temporal_power, "spatial_power_thw": spatial_power,
                     "spatial_mean_temporal_power_f": mean_temporal_power}


def feature_geometry(backend, batch, model_name):
    from .cross_model_probe_v2 import layout

    positions, layout_record = layout(backend, batch, model_name)
    if model_name == "qwen":
        time_bins, height, width = map(int, batch["video_grid_thw"][0].detach().cpu().tolist())
        merge_size = int(backend.model.config.vision_config.spatial_merge_size)
        require(height % merge_size == 0 and width % merge_size == 0, "Qwen spatial grid is not merge-compatible")
        pre_stage, post_stage = "vision_final_premerger", "vision_final_postmerger"
        pre_description = "final vision block output before final merger LayerNorm; block-major order explicitly inverted"
        post_description = "final merger output, row-major coarse spatial cells; excludes DeepStack auxiliary outputs"
    else:
        time_bins = positions.shape[0]
        height = width = int(backend.image_size // backend.model.config.vision_config.patch_size)
        merge_size = int(backend.model.config.scale_factor)
        require(height % merge_size == 0, "Smol spatial grid is not connector-compatible")
        pre_stage, post_stage = "vision_final_postnorm", "vision_connector_output"
        pre_description = "vision_model.last_hidden_state after final post_layernorm; patch Conv2d flatten is row-major"
        post_description = "connector output after pixel-shuffle and modality projection; coarse cells remain row-major"
    require(positions.shape == (time_bins, height * width // merge_size**2), "Visual token positions disagree with reconstructed grid")
    patch_size = int(backend.model.config.vision_config.patch_size)
    geometry = {"pre_stage": pre_stage, "post_stage": post_stage, "time_bins": time_bins, "pre_height": height,
                "pre_width": width, "post_height": height // merge_size, "post_width": width // merge_size,
                "spatial_merge_factor": merge_size, "pre_spatial_stride_processed_pixels": patch_size,
                "post_spatial_stride_processed_pixels": patch_size * merge_size, "pre_description": pre_description,
                "post_description": post_description, "layout": layout_record,
                "time_interpretation": "Qwen pairs of input frames, not the original sixteen-frame axis" if model_name == "qwen" else "one processed image per retained frame"}
    return positions, geometry


def capture(backend, batch, model_name):
    import numpy as np
    import torch
    from .cross_model_probe_v2 import architecture
    from .passive_probe import forward_logits

    positions, geometry = feature_geometry(backend, batch, model_name)
    layers, _ = architecture(backend, model_name)
    features = {}
    handles = []
    selected = torch.as_tensor(positions, device=batch["input_ids"].device, dtype=torch.long)

    def save(name, tensor, pre_stage=False):
        require(name not in features and isinstance(tensor, torch.Tensor), "Unexpected duplicate or non-tensor visual capture")
        values = tensor.detach().float().cpu().numpy().copy()
        if pre_stage and model_name == "qwen":
            grid = qwen_premerger_grid(values, (geometry["time_bins"], geometry["pre_height"], geometry["pre_width"]),
                                      geometry["spatial_merge_factor"])
        else:
            height = geometry["pre_height" if pre_stage else "post_height"]
            width = geometry["pre_width" if pre_stage else "post_width"]
            grid = row_major_grid(values, geometry["time_bins"], height, width)
        require(np.isfinite(grid).all(), "Nonfinite visual features")
        features[name] = grid

    def qwen_pre(module, arguments):
        save(geometry["pre_stage"], arguments[0], True)

    def vision_output(module, arguments, output):
        save(geometry["pre_stage"], output.last_hidden_state, True)

    def projection_output(module, arguments, output):
        save(geometry["post_stage"], output)

    def language_input(module, arguments, keywords):
        hidden = arguments[0] if arguments else keywords["hidden_states"]
        require(hidden.ndim == 3 and hidden.shape[:2] == batch["input_ids"].shape, "LLM layer-zero input geometry changed")
        save("llm_layer0_visual_input", hidden[0, selected])

    try:
        if model_name == "qwen":
            merger = backend.model.model.visual.merger
            handles.extend((merger.register_forward_pre_hook(qwen_pre), merger.register_forward_hook(projection_output)))
        else:
            handles.extend((backend.model.model.vision_model.register_forward_hook(vision_output),
                            backend.model.model.connector.register_forward_hook(projection_output)))
        handles.append(layers[0].register_forward_pre_hook(language_input, with_kwargs=True))
        logits = forward_logits(backend, batch)
    finally:
        for handle in handles:
            handle.remove()
    require(set(features) == {geometry["pre_stage"], geometry["post_stage"], "llm_layer0_visual_input"}, "Visual stage capture is incomplete")
    projection_exact = bool(np.array_equal(features[geometry["post_stage"]], features["llm_layer0_visual_input"]))
    require(projection_exact, "Projector-to-LLM visual position mapping failed exact equality")
    geometry["projector_to_llm0_visual_exact"] = projection_exact
    return logits, features, geometry


def selected_cases(arguments):
    from .cross_model_probe_v2 import cases

    original = cases(arguments)
    result = []
    for index in range(0, len(original), 3):
        group = original[index:index + 3]
        require([case["condition"] for case in group] == ["real", "static_first", "static_last"], "Existing source case order changed")
        result.extend(group)
        real = group[0]
        result.append({**real, "case_id": real["query_id"] + ":gray", "condition": "gray",
                       "question": {key: value for key, value in real["question"].items() if key not in {"answer_index", "answer_text"}},
                       "gold_answer_index": None, "frames": [real["frames"][0]] * 16,
                       "frame_sha256": [real["frame_sha256"][0]] * 16, "decoded_source_timestamps_s": [None] * 16,
                       "gray_policy": "synthetic RGB=(127,127,127), original first-frame dimensions, unchanged displayed timestamps"})
    return result


def pack_features(features, geometry):
    import numpy as np

    arrays, metadata = {}, {}
    post_stage = geometry["post_stage"]
    for stage, values in features.items():
        logical_hash = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        if stage == "llm_layer0_visual_input" and np.array_equal(values, features[post_stage]):
            metadata[stage] = {"encoding": "exact_stage_alias", "alias_of": post_stage,
                               "logical_shape": list(values.shape), "dtype": "float32", "logical_sha256": logical_hash}
            continue
        repeated = np.array_equal(values, np.broadcast_to(values[:1], values.shape))
        stored = values[:1] if repeated else values
        arrays[stage + "_features"] = stored
        metadata[stage] = {"encoding": "exact_time_repeat" if repeated else "dense",
                           "array_key": stage + "_features", "logical_shape": list(values.shape),
                           "stored_shape": list(stored.shape), "dtype": "float32", "logical_sha256": logical_hash,
                           "stored_uncompressed_bytes": int(stored.nbytes)}
    return arrays, metadata


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_runtime import digest, tensor_hash, write_json
    from .cross_model_probe_v2 import architecture
    from .endpoint_control import check_deadline
    from .passive_probe import forward_logits

    output = arguments.output_dir.resolve()
    require(not output.exists(), "Visual probe output must be new")
    cases = selected_cases(arguments)
    protocol = arguments.protocol.resolve()
    source_names = ("vision_encoder_probe_v2.py", "cross_model_probe_v2.py", "controller_runtime.py", "qwen3_backend.py",
                    "smolvlm_backend.py", "passive_probe.py", "evaluate_small.py", "resource_guard.py", "endpoint_control.py")
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_choice": arguments.model,
              "scope": "read-only unweighted vision features; old eight validation sources only",
              "selection": "first sorted question per source, real/static_first/static_last/gray",
              "expected_cases": len(cases), "completed_cases": 0, "conditions": ["real", "static_first", "static_last", "gray"],
              "protocol_sha256": digest(protocol), "questions_sha256": digest(arguments.questions),
              "frames_sha256": digest(arguments.frames_manifest), "maximum_archive_bytes_per_source": SOURCE_ARCHIVE_LIMIT,
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names},
              "frequency_policy": "FFT only T or H,W; no feature-channel FFT or attention weighting",
              "source_archive_bytes": {}}
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(protocol, snapshot / "protocol.md")
    write_json(output / "cases.json", cases)
    config["cases_sha256"] = digest(output / "cases.json")
    write_json(output / "config.json", config)
    try:
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            guard.phase = "load_pinned_model"
            if arguments.model == "qwen":
                from .qwen3_backend import load_backend, DEFAULT_MAX_PIXELS, DEFAULT_REVISION
            else:
                from .smolvlm_backend import load_backend, DEFAULT_MAX_PIXELS, DEFAULT_REVISION
            backend = load_backend(device="cuda")
            layers, implementation = architecture(backend, arguments.model)
            require(backend.metadata["model_revision"] == DEFAULT_REVISION and backend.metadata["processor_revision"] == DEFAULT_REVISION,
                    "Unpinned model or processor revision")
            from transformers.models.qwen3_vl import modeling_qwen3_vl, video_processing_qwen3_vl
            from transformers.models.smolvlm import modeling_smolvlm
            config.update({"model": backend.metadata, "implementation_sha256": {
                module.__name__: {"path": inspect.getfile(module), "sha256": digest(inspect.getfile(module))}
                for module in (implementation, modeling_qwen3_vl, video_processing_qwen3_vl, modeling_smolvlm)}})
            encoded = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in "ABC"]
            require(all(len(value) == 1 for value in encoded) and len({value[0] for value in encoded}) == 3, "Invalid option label tokens")
            option_ids = [value[0] for value in encoded]
            config["option_token_ids"] = option_ids
            write_json(output / "config.json", config)
            with (output / "records.jsonl").open("x", encoding="utf-8") as records, (output / "spectra.jsonl").open("x", encoding="utf-8") as spectra:
                for index, case in enumerate(cases):
                    check_deadline(guard)
                    guard.phase = "prepare:" + case["case_id"]
                    require([digest(path) for path in case["frames"]] == case["frame_sha256"], "Source frame content changed")
                    if case["condition"] == "gray":
                        with Image.open(case["frames"][0]) as first:
                            frames = [Image.new("RGB", first.size, (127, 127, 127)) for unused in range(16)]
                    else:
                        frames = [Image.open(path).convert("RGB") for path in case["frames"]]
                    raw_rgb_hashes = [{"shape": list(np.asarray(frame).shape), "sha256": hashlib.sha256(np.asarray(frame).tobytes()).hexdigest()} for frame in frames]
                    batch = backend.prepare_video(frames, case["display_timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    input_hashes = {name: tensor_hash(value) for name, value in batch.items()}
                    guard.phase = "native:" + case["case_id"]
                    baseline = forward_logits(backend, batch)
                    check_deadline(guard)
                    guard.phase = "passive_vision_capture:" + case["case_id"]
                    passive, features, geometry = capture(backend, batch, arguments.model)
                    require(np.array_equal(baseline, passive), "Passive visual hooks changed complete-vocabulary logits")
                    check_deadline(guard)
                    guard.phase = "restored:" + case["case_id"]
                    restored = forward_logits(backend, batch)
                    require(np.array_equal(baseline, restored) and np.isfinite(baseline).all(), "Native output restoration or finiteness failed")
                    require(input_hashes == {name: tensor_hash(value) for name, value in batch.items()}, "Model inputs changed during passive capture")
                    arrays, feature_metadata = pack_features(features, geometry)
                    arrays.update({"identity_logits": baseline, "passive_logits": passive, "restored_logits": restored,
                                   "input_ids": batch["input_ids"][0].detach().cpu().numpy(), "option_token_ids": np.asarray(option_ids)})
                    stage_summaries = {}
                    for stage, values in features.items():
                        check_deadline(guard)
                        guard.phase = "cpu_spectra:" + case["case_id"] + ":" + stage
                        if feature_metadata[stage]["encoding"] == "exact_stage_alias":
                            stage_summaries[stage] = {**stage_summaries[geometry["post_stage"]], "exact_alias_of": geometry["post_stage"]}
                        else:
                            summary, spectral_arrays = spectrum_summary(values)
                            stage_summaries[stage] = summary
                            arrays.update({stage + "_" + name: value for name, value in spectral_arrays.items()})
                    source_dir = output / case["video_id"]
                    source_dir.mkdir(exist_ok=True)
                    archive = source_dir / (case["condition"] + ".npz")
                    np.savez_compressed(archive, **arrays)
                    config["source_archive_bytes"][case["video_id"]] = config["source_archive_bytes"].get(case["video_id"], 0) + archive.stat().st_size
                    require(config["source_archive_bytes"][case["video_id"]] <= SOURCE_ARCHIVE_LIMIT, "Source archive exceeds the declared 100 MiB limit")
                    common = {"case_id": case["case_id"], "query_id": case["query_id"], "video_id": case["video_id"],
                              "condition": case["condition"], "model": arguments.model,
                              "archive_file": str(archive.relative_to(output)), "archive_sha256": digest(archive)}
                    row = {**common, "question": case["question"], "gold_answer_index": case["gold_answer_index"],
                           "option_logits": baseline[option_ids].tolist(), "option_token_ids": option_ids,
                           "predicted_answer_index": int(baseline[option_ids].argmax()), "scoring_policy": "gold only for offline real-condition scoring",
                           "native_passive_complete_vocabulary_exact": True, "native_restored_complete_vocabulary_exact": True,
                           "input_tensor_hashes": input_hashes, "source_frame_sha256": case["frame_sha256"],
                           "actual_input_rgb_hashes": raw_rgb_hashes, "display_timestamps_s": case["display_timestamps_s"],
                           "decoded_source_timestamps_s": case["decoded_source_timestamps_s"],
                           "gray_policy": case.get("gray_policy"), "geometry": geometry, "features": feature_metadata,
                           "batch": batch.metadata, "archive_bytes": archive.stat().st_size}
                    records.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    for stage, summary in stage_summaries.items():
                        spectra.write(json.dumps({**common, "stage": stage, **summary}, ensure_ascii=False, allow_nan=False) + "\n")
                    records.flush()
                    spectra.flush()
                    config["completed_cases"] += 1
                    write_json(output / "config.json", config)
                    print(f"vision spectrum {arguments.model} {case['case_id']} ({config['completed_cases']}/{len(cases)})", flush=True)
                    del batch, frames, features, arrays, baseline, passive, restored
            config.update({"records_sha256": digest(output / "records.jsonl"), "spectra_sha256": digest(output / "spectra.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            check_deadline(guard)
            del backend, layers
            torch.cuda.empty_cache()
        resources = json.loads((output / "resource_summary.json").read_text(encoding="utf-8"))
        require(resources["status"] == "complete" and resources["exception"] is None and not (output / "resource_stop.json").exists(), "Unclean resource completion")
        require(config["completed_cases"] == config["expected_cases"], "Visual probe is incomplete")
        require(digest(protocol) == config["protocol_sha256"] == digest(snapshot / "protocol.md"), "Protocol changed")
        require(digest(arguments.questions) == config["questions_sha256"] and digest(arguments.frames_manifest) == config["frames_sha256"], "Data manifests changed")
        for name, expected in config["source_sha256"].items():
            require(digest(Path(__file__).with_name(name)) == expected == digest(snapshot / name), "Source changed")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "resource_summary_sha256": digest(output / "resource_summary.json")})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "smol"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--frames-manifest", "--frames", dest="frames_manifest", type=Path, required=True)
    parser.add_argument("--video-ids")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
