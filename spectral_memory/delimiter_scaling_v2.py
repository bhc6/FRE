"""Evaluate an explicitly adapted delimiter-token-scaling baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


PROTOCOL = Path("docs/delimiter_scaling_protocol_v2.md")
SCALES = (.5, 2., 4.)
DELIMITER_IDS = {"qwen": {"<|vision_start|>": 151652, "<|vision_end|>": 151653},
                 "smol": {"<fake_token_around_image>": 49189}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def delimiter_positions(backend, batch, model_name):
    import numpy as np
    from .cross_model_probe_v2 import layout

    visual_positions, visual_layout = layout(backend, batch, model_name)
    tokenizer = backend.processor.tokenizer
    identifiers = batch["input_ids"][0].detach().cpu().numpy()
    configured = DELIMITER_IDS[model_name]
    for name, expected in configured.items():
        require(tokenizer.convert_tokens_to_ids(name) == expected and
                tokenizer.encode(name, add_special_tokens=False) == [expected], "Pinned delimiter tokenizer identity changed")
    if model_name == "qwen":
        require(backend.model.config.vision_start_token_id == configured["<|vision_start|>"] and
                backend.model.config.vision_end_token_id == configured["<|vision_end|>"], "Qwen delimiter configuration mismatch")
        starts, ends = visual_positions[:, 0] - 1, visual_positions[:, -1] + 1
        require(np.all(identifiers[starts] == configured["<|vision_start|>"]) and
                np.all(identifiers[ends] == configured["<|vision_end|>"]), "Qwen visual blocks lack exact delimiters")
    else:
        starts, ends = visual_positions[:, 0] - 2, visual_positions[:, -1] + 1
        global_marker = tokenizer.convert_tokens_to_ids("<global-img>")
        require(global_marker == 49152 and np.all(identifiers[visual_positions[:, 0] - 1] == global_marker),
                "Smol visual blocks lack the pinned global-image marker")
        require(np.all(identifiers[starts] == 49189) and np.all(identifiers[ends] == 49189),
                "Smol visual blocks lack exact surrounding fake tokens")
    expected_positions = np.sort(np.concatenate((starts, ends)))
    found = np.flatnonzero(np.isin(identifiers, list(configured.values())))
    require(np.array_equal(found, expected_positions) and len(set(found)) == len(found),
            "Unexpected delimiter outside a verified visual frame group")
    require(len(found) == (16 if model_name == "qwen" else 32) and found.min() >= 0 and found.max() < len(identifiers) - 1,
            "Delimiter count or final text-token separation changed")
    return found, {"tokens": configured, "positions": found.tolist(), "count": len(found),
                   "start_positions": starts.tolist(), "end_positions": ends.tolist(),
                   "visual_layout": visual_layout, "selection": "all and only verified frame-boundary special tokens"}


def controlled_forward(backend, batch, model_name, positions, scale, mode="delimiter_scale"):
    import numpy as np
    import torch
    from .cross_model_probe_v2 import architecture
    from .passive_probe import forward_logits

    require(mode in ("delimiter_scale", "update_damp"), "Unknown intervention mode")
    require(scale in (.5, 1., 2., 4.) and (mode != "update_damp" or scale == .5), "Unregistered scale or damping control")
    layers, _ = architecture(backend, model_name)
    layer_index = (len(layers) - 1) // 2
    require(len(layers) >= 2 and layer_index + 1 < len(layers), "A following decoder layer is required")
    token_count = batch["input_ids"].shape[1]
    selected = torch.as_tensor(positions, dtype=torch.long, device=batch["input_ids"].device)
    require(batch["input_ids"].shape[0] == 1 and selected.ndim == 1 and selected.numel() > 0 and
            selected.unique().numel() == selected.numel() and int(selected.min()) >= 0 and int(selected.max()) < token_count - 1,
            "Expected unique delimiter positions before the last text query")
    visits = {}
    captures = {}
    metadata = {"mode": mode, "scale": scale, "injection_layer": layer_index,
                "layer_indexing": "zero_based", "layer_selection": "floor((decoder_layer_count-1)/2)",
                "decoder_layer_count": len(layers), "delimiter_count": int(selected.numel()),
                "mechanism_capture_layer": layer_index + 1, "projection_capture": "before Q/K head normalization and rotary embedding"}
    handles = []

    def unpack(output):
        if isinstance(output, torch.Tensor):
            return output, None
        require(isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor), "Unsupported decoder/module output")
        return output[0], output[1:]

    def scaled_output(name, multiplier, observe_residual=False):
        def hook(module, arguments, output):
            require(name not in visits, "Expected exactly one uncached visit per hooked module")
            visits[name] = 1
            hidden, remainder = unpack(output)
            require(hidden.ndim == 3 and hidden.shape[:2] == (1, token_count), "Decoder output sequence geometry changed")
            before = hidden[:, selected]
            if observe_residual:
                metadata["delimiter_residual_l2_before"] = float(before.float().norm())
            if multiplier == 1.:
                if observe_residual:
                    metadata["delimiter_residual_l2_after"] = metadata["delimiter_residual_l2_before"]
                return output
            replaced = hidden.clone()
            replaced[:, selected] = before * multiplier
            if observe_residual:
                metadata["delimiter_residual_l2_after"] = float(replaced[:, selected].float().norm())
            return replaced if remainder is None else (replaced, *remainder)
        return hook

    def capture(name):
        def hook(module, arguments, output):
            require(name not in captures, "Duplicate next-layer mechanism capture")
            require(isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[:2] == (1, token_count),
                    "Mechanism capture geometry changed")
            captures[name] = output[0, selected].detach().float().cpu().numpy().copy()
        return hook

    try:
        handles.append(layers[layer_index].register_forward_hook(
            scaled_output("selected_decoder_output", scale if mode == "delimiter_scale" else 1., True)))
        following = layers[layer_index + 1]
        handles.append(following.input_layernorm.register_forward_hook(capture("next_input_norm")))
        for projection in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(following.self_attn, projection).register_forward_hook(capture("next_" + projection)))
        if mode == "update_damp":
            for index in range(layer_index + 1, len(layers)):
                handles.append(layers[index].self_attn.register_forward_hook(scaled_output(f"attention_{index}", .5)))
                handles.append(layers[index].mlp.register_forward_hook(scaled_output(f"ffn_{index}", .5)))
        logits = forward_logits(backend, batch)
    finally:
        for handle in handles:
            handle.remove()
    expected_visits = 1 + (2 * (len(layers) - layer_index - 1) if mode == "update_damp" else 0)
    require(len(visits) == expected_visits and len(captures) == 4 and np.isfinite(logits).all() and
            all(np.isfinite(value).all() for value in captures.values()), "Incomplete or nonfinite delimiter intervention")
    metadata["hooked_modules_visited"] = len(visits)
    return logits, captures, metadata


def mechanism_comparison(reference, intervention):
    import numpy as np

    result = {}
    for name in reference:
        baseline = np.asarray(reference[name], dtype=np.float64)
        changed = np.asarray(intervention[name], dtype=np.float64)
        require(baseline.shape == changed.shape, "Mechanism tensors must be paired")
        baseline_norm = np.linalg.norm(baseline)
        per_token = np.linalg.norm(baseline, axis=-1)
        result[name] = {"shape": list(baseline.shape), "baseline_l2": float(baseline_norm),
                        "intervention_l2": float(np.linalg.norm(changed)),
                        "norm_ratio": float(np.linalg.norm(changed) / max(baseline_norm, 1e-20)),
                        "relative_l2_difference": float(np.linalg.norm(changed - baseline) / max(baseline_norm, 1e-20)),
                        "per_delimiter_norm_ratio": (np.linalg.norm(changed, axis=-1) / np.maximum(per_token, 1e-20)).tolist(),
                        "per_delimiter_relative_l2_difference": (np.linalg.norm(changed - baseline, axis=-1) / np.maximum(per_token, 1e-20)).tolist(),
                        "zero_reference_delimiters": int((per_token == 0).sum())}
    return result


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_evaluate import cases_for
    from .controller_runtime import digest, tensor_hash, write_json
    from .cross_model_probe_v2 import architecture
    from .endpoint_control import check_deadline
    from .normalized_evaluate import rotate_cases
    from .passive_probe import forward_logits

    output = arguments.output_dir.resolve()
    require(not output.exists(), "Delimiter output must be a new directory")
    cases = rotate_cases(cases_for(arguments), arguments.rotations)
    protocol = arguments.protocol.resolve()
    source_names = ("delimiter_scaling_v2.py", "cross_model_probe_v2.py", "normalized_evaluate.py", "controller_evaluate.py",
                    "controller_runtime.py", "smolvlm_backend.py", "qwen3_backend.py", "passive_probe.py", "resource_guard.py",
                    "evaluate_small.py", "endpoint_control.py")
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_choice": arguments.model, "split": arguments.split, "rotations": arguments.rotations,
              "static_rotation_policy": "rotation_0_only_no_gold", "candidate_scales": list(SCALES),
              "mechanism_only_control": "update_damp0.5_after_selected_layer_compared_to_delimiter_scale2",
              "scope": "published delimiter-token scaling adaptation; no novelty claim; development unless separately frozen",
              "scoring_policy": "save labels as metadata only; compute accuracy offline; no label enters intervention",
              "expected_cases": len(cases), "expected_records": len(cases) * 5, "completed_cases": 0, "completed_records": 0,
              "forwards_per_case": 7, "protocol_sha256": digest(protocol), "questions_sha256": digest(arguments.questions),
              "frames_sha256": digest(arguments.frames_manifest), "source_sha256": {
                  name: digest(Path(__file__).with_name(name)) for name in source_names}}
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
            require(len(layers) == (28 if arguments.model == "qwen" else 32), "Pinned model layer count changed")
            require(backend.metadata["model_revision"] == DEFAULT_REVISION and backend.metadata["processor_revision"] == DEFAULT_REVISION,
                    "Model/processor must resolve to the pinned immutable checkpoint revision")
            config.update({"model": backend.metadata, "injection_layer": (len(layers) - 1) // 2,
                           "delimiter_ids": DELIMITER_IDS[arguments.model], "implementation_sha256": {
                               "path": inspect.getfile(implementation), "sha256": digest(inspect.getfile(implementation))}})
            encoded = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in "ABC"]
            require(all(len(value) == 1 for value in encoded) and len({value[0] for value in encoded}) == 3,
                    "Original option labels require distinct single tokens")
            option_ids = [value[0] for value in encoded]
            config["option_token_ids"] = option_ids
            write_json(output / "config.json", config)
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for case_index, case in enumerate(cases):
                    check_deadline(guard)
                    guard.phase = "prepare:" + case["case_id"]
                    require([digest(path) for path in case["frames"]] == case["frame_sha256"], "Evaluation frame contents changed")
                    frames = [Image.open(path).convert("RGB") for path in case["frames"]]
                    batch = backend.prepare_video(frames, case["timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    positions, delimiter_layout = delimiter_positions(backend, batch, arguments.model)
                    input_hashes = {name: tensor_hash(value) for name, value in batch.items()}
                    guard.phase = "native:" + case["case_id"]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    baseline = forward_logits(backend, batch)
                    native_seconds = time.perf_counter() - started
                    check_deadline(guard)
                    guard.phase = "scale1_identity:" + case["case_id"]
                    identity, reference_mechanism, identity_metadata = controlled_forward(backend, batch, arguments.model, positions, 1.)
                    require(np.array_equal(baseline, identity), "Scale-one hooks changed native complete-vocabulary logits")
                    arrays = {"identity_logits": baseline, "zero_logits": identity,
                              "input_ids": batch["input_ids"][0].detach().cpu().numpy(),
                              "option_token_ids": np.asarray(option_ids), "delimiter_positions": positions}
                    for name, values in reference_mechanism.items():
                        arrays["identity_" + name] = values
                    results = [("identity", 1., "native", baseline, native_seconds, identity_metadata, reference_mechanism)]
                    for scale in SCALES:
                        check_deadline(guard)
                        guard.phase = f"delimiter_scale{scale:g}:" + case["case_id"]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        logits, mechanism, metadata = controlled_forward(backend, batch, arguments.model, positions, scale)
                        results.append((f"delimiter_scale{scale:g}", scale, "candidate" if scale > 1 else "reverse_ablation",
                                        logits, time.perf_counter() - started, metadata, mechanism))
                    check_deadline(guard)
                    guard.phase = "update_damp0.5:" + case["case_id"]
                    started = time.perf_counter()
                    damp_logits, damp_mechanism, damp_metadata = controlled_forward(backend, batch, arguments.model, positions, .5, "update_damp")
                    results.append(("update_damp0.5", .5, "mechanism_only", damp_logits,
                                    time.perf_counter() - started, damp_metadata, damp_mechanism))
                    check_deadline(guard)
                    guard.phase = "restored:" + case["case_id"]
                    restored = forward_logits(backend, batch)
                    require(np.array_equal(baseline, restored), "Native complete-vocabulary logits did not restore")
                    require(input_hashes == {name: tensor_hash(value) for name, value in batch.items()}, "Paired input tensors changed")
                    arrays["restored_logits"] = restored
                    for tag, scale, role, logits, elapsed, metadata, mechanism in results:
                        arrays[tag + "_logits"] = logits
                        for name, values in mechanism.items():
                            arrays[tag + "_" + name] = values
                    dts2 = arrays["delimiter_scale2_logits"]
                    difference = damp_logits.astype(np.float64) - dts2.astype(np.float64)
                    damp_comparison = {"comparison": "update_damp0.5_minus_delimiter_scale2",
                                       "full_vocabulary_maximum_absolute_difference": float(np.abs(difference).max()),
                                       "full_vocabulary_relative_l2_difference": float(np.linalg.norm(difference) / max(np.linalg.norm(dts2.astype(np.float64)), 1e-20)),
                                       "full_vocabulary_exact": bool(np.array_equal(damp_logits, dts2)),
                                       "option_logits_exact": bool(np.array_equal(damp_logits[option_ids], dts2[option_ids])),
                                       "option_prediction_same": bool(damp_logits[option_ids].argmax() == dts2[option_ids].argmax()),
                                       "interpretation": "Exact equivalence only in the ideal positive-scale-invariant pre-norm setting; epsilon and finite precision can differ."}
                    filename = f"{case_index:04d}.npz"
                    np.savez_compressed(output / filename, **arrays)
                    archive_hash = digest(output / filename)
                    common_names = ("case_id", "base_case_id", "query_id", "video_id", "condition", "rotation", "question",
                                    "original_question", "original_options", "original_gold_answer_index", "semantic_original_option_indices",
                                    "gold_answer_index", "supervised", "frame_sha256")
                    common = {name: case[name] for name in common_names}
                    common.update({"model": arguments.model, "split": arguments.split, "batch": batch.metadata,
                                   "input_tensor_hashes": input_hashes, "delimiter_layout": delimiter_layout,
                                   "identity_exact": True, "restoration_exact": True, "option_token_ids": option_ids,
                                   "archive_file": filename, "archive_sha256": archive_hash, "dts2_damping_comparison": damp_comparison})
                    baseline_prediction = int(baseline[option_ids].argmax())
                    for tag, scale, role, logits, elapsed, metadata, mechanism in results:
                        selected_logits = logits[option_ids]
                        prediction = int(selected_logits.argmax())
                        record = {**common, "tag": tag, "scale": scale, "role": role, "option_logits": selected_logits.tolist(),
                                  "predicted_answer_index": prediction,
                                  "predicted_original_option_index": case["semantic_original_option_indices"][prediction],
                                  "baseline_predicted_answer_index": baseline_prediction,
                                  "baseline_predicted_original_option_index": case["semantic_original_option_indices"][baseline_prediction],
                                  "baseline_answer_changed": prediction != baseline_prediction,
                                  "exact_maximum_tie_indices": np.flatnonzero(selected_logits == selected_logits.max()).tolist(),
                                  "forward_seconds": elapsed, "capture": metadata,
                                  "next_layer_mechanism_vs_identity": mechanism_comparison(reference_mechanism, mechanism)}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        config["completed_records"] += 1
                    stream.flush()
                    config["completed_cases"] += 1
                    write_json(output / "config.json", config)
                    print(f"delimiter {arguments.model} {case['case_id']} ({config['completed_cases']}/{len(cases)})", flush=True)
                    del batch, frames, arrays, results, baseline, identity, restored, reference_mechanism, damp_mechanism
            config.update({"records_sha256": digest(output / "records.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            check_deadline(guard)
            del backend, layers
            torch.cuda.empty_cache()
        resources = json.loads((output / "resource_summary.json").read_text(encoding="utf-8"))
        require(resources["status"] == "complete" and resources["exception"] is None and not (output / "resource_stop.json").exists(),
                "Resource guard did not finish cleanly")
        require(config["completed_cases"] == config["expected_cases"] and config["completed_records"] == config["expected_records"],
                "Delimiter matrix is incomplete")
        require(digest(protocol) == config["protocol_sha256"] == digest(snapshot / "protocol.md"), "Protocol changed during execution")
        require(digest(arguments.questions) == config["questions_sha256"] and digest(arguments.frames_manifest) == config["frames_sha256"],
                "Input manifests changed during execution")
        for name, expected in config["source_sha256"].items():
            require(digest(Path(__file__).with_name(name)) == expected == digest(snapshot / name), "Source changed during execution")
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
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--frames-manifest", "--frames", dest="frames_manifest", type=Path, required=True)
    parser.add_argument("--video-ids")
    parser.add_argument("--rotations", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
