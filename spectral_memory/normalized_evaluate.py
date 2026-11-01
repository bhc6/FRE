"""Evaluate normalized internal readout controls with explicit option rotations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


def rotate_cases(cases, rotations):
    from .evaluate_small import build_prompt

    if not rotations or len(rotations) != len(set(rotations)) or any(rotation not in (0, 1, 2) for rotation in rotations):
        raise ValueError("Use distinct option rotations from 0, 1, 2")
    rotated = []
    for case in cases:
        original_options = list(case["options"])
        if len(original_options) != 3:
            raise ValueError("Normalized official evaluation requires three original choices")
        for rotation in rotations if case["supervised"] else (0,):
            semantic_indices = [(index + rotation) % 3 for index in range(3)]
            options = original_options[rotation:] + original_options[:rotation]
            original_gold = case["gold_answer_index"] if case["supervised"] else None
            gold = (original_gold - rotation) % 3 if original_gold is not None else None
            question = {**case["question"], "options": options}
            if case["supervised"]:
                question["answer_index"] = gold
                question["answer_text"] = options[gold]
            else:
                question = {key: value for key, value in question.items() if key not in {"answer_index", "answer_text"}}
            rotated.append({**case, "base_case_id": case["case_id"], "case_id": case["case_id"] + f":rot{rotation}",
                            "rotation": rotation, "original_options": original_options,
                            "original_gold_answer_index": original_gold, "semantic_original_option_indices": semantic_indices,
                            "original_question": case["question"], "question": question, "options": options,
                            "gold_answer_index": gold, "prompt": build_prompt(question)})
    if len({case["case_id"] for case in rotated}) != len(rotated):
        raise ValueError("Rotated case identifiers must be unique")
    return rotated


def checkpoint_tag(method, residual_limit, seed):
    return f"{method}_cap{residual_limit:g}_seed{seed}"


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_evaluate import cases_for
    from .controller_runtime import controlled_forward, digest, tensor_hash, write_json
    from .endpoint_control import check_deadline
    from .normalized_readout import METHODS, NormalizedReadoutController
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Normalized evaluation output must be new")
    output.mkdir(parents=True)
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": arguments.split, "rotations": list(arguments.rotations), "static_rotation_policy": "rotation_0_only_no_gold",
              "evaluation": "complete_uncached_native_model_forward", "completed_cases": 0, "completed_records": 0}
    write_json(output / "config.json", config)
    try:
        protocol_hash = digest(arguments.protocol)
        cases = rotate_cases(cases_for(arguments), arguments.rotations)
        checkpoint_records = []
        payloads = {}
        for path in arguments.checkpoints:
            path = path.resolve()
            checkpoint_hash = digest(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["protocol_sha256"] != protocol_hash or payload["step"] != 200 or payload["method"] not in METHODS:
                raise ValueError("Expected a normalized step-200 checkpoint under the supplied frozen protocol")
            settings = payload["controller_config"]
            if settings.get("head_dim", 128) != 128 or settings.get("time_bins", 8) != 8 or settings.get("hidden_dim", 16) != 16:
                raise ValueError("Checkpoint dimensions differ from the fixed 128/8/16 controller")
            residual_limit = float(settings["residual_limit"])
            if residual_limit not in (0.1, 0.3):
                raise ValueError("Use the declared 0.1 or 0.3 head residual cap")
            if not isinstance(payload["seed"], int) or isinstance(payload["seed"], bool):
                raise ValueError("Checkpoint seed must be an integer")
            controller = NormalizedReadoutController(payload["method"], **settings)
            controller.load_state_dict(payload["state_dict"])
            tag = checkpoint_tag(payload["method"], residual_limit, payload["seed"])
            if tag in payloads:
                raise ValueError("Duplicate method/cap/seed checkpoint tag")
            checkpoint_records.append({"path": str(path), "checkpoint_sha256": checkpoint_hash, "tag": tag,
                                       "method": payload["method"], "seed": payload["seed"], "step": 200,
                                       "controller_config": settings, "residual_limit": residual_limit,
                                       "gain_limit": controller.gain_limit,
                                       "parameter_count": sum(parameter.numel() for parameter in controller.parameters())})
            payloads[tag] = payload
            del controller
        if not checkpoint_records:
            raise ValueError("At least one normalized checkpoint is required")
        checkpoint_records.sort(key=lambda record: record["tag"])
        source_names = ("normalized_evaluate.py", "normalized_readout.py", "controller_evaluate.py", "controller_runtime.py",
                        "learned_readout.py", "resource_guard.py", "qwen3_backend.py", "passive_probe.py", "evaluate_small.py", "endpoint_control.py")
        config.update({"expected_cases": len(cases), "expected_records": len(cases) * (len(checkpoint_records) + 1),
                       "protocol_sha256": protocol_hash, "questions_sha256": digest(arguments.questions),
                       "frames_sha256": digest(arguments.frames_manifest), "checkpoints": checkpoint_records,
                       "option_rotation_semantics": "new_options=old[r:]+old[:r]; presented_gold=(original_gold-r)%3",
                       "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names}})
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for name in source_names:
            shutil.copy2(Path(__file__).with_name(name), snapshot / name)
        shutil.copy2(arguments.protocol, snapshot / "protocol.md")
        write_json(output / "cases.json", cases)
        config["cases_sha256"] = digest(output / "cases.json")
        write_json(output / "config.json", config)
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            guard.phase = "load_model_and_normalized_controllers"
            backend = load_backend(device="cuda")
            config["model"] = backend.metadata
            controllers = []
            for record in checkpoint_records:
                payload = payloads[record["tag"]]
                controller = NormalizedReadoutController(payload["method"], **payload["controller_config"]).to("cuda").eval()
                controller.load_state_dict(payload["state_dict"])
                controller.requires_grad_(False)
                controllers.append((record, controller))
            del payloads
            zero = NormalizedReadoutController("unit_time_mlp").to("cuda").eval().requires_grad_(False)
            option_ids = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in "ABC"]
            if any(len(identifiers) != 1 for identifiers in option_ids):
                raise ValueError("Expected single-token option labels")
            option_ids = [identifiers[0] for identifiers in option_ids]
            if len(set(option_ids)) != 3:
                raise ValueError("Option label tokens must be distinct")
            config["option_token_ids"] = option_ids
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for case_index, case in enumerate(cases):
                    check_deadline(guard)
                    guard.phase = "prepare:" + case["case_id"]
                    if [digest(path) for path in case["frames"]] != case["frame_sha256"]:
                        raise ValueError("Evaluation frames changed")
                    frames = [Image.open(path).convert("RGB") for path in case["frames"]]
                    batch = backend.prepare_video(frames, case["timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    input_hashes = {name: tensor_hash(value) for name, value in batch.items()}
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    baseline = forward_logits(backend, batch)
                    baseline_seconds = time.perf_counter() - started
                    identity, _, identity_metadata = controlled_forward(backend, batch, zero)
                    if not np.array_equal(identity, baseline) or identity_metadata["delta_context_l2"] != 0.0:
                        raise RuntimeError("Normalized zero controller changed native full-vocabulary logits")
                    common_names = ("case_id", "base_case_id", "query_id", "video_id", "condition", "rotation", "question",
                                    "original_question", "original_options", "original_gold_answer_index", "semantic_original_option_indices",
                                    "gold_answer_index", "supervised", "frame_sha256")
                    common = {name: case[name] for name in common_names}
                    common.update({"split": arguments.split, "batch": batch.metadata, "input_tensor_hashes": input_hashes,
                                   "identity_exact": True, "option_token_ids": option_ids})
                    arrays = {"identity_logits": baseline, "zero_logits": identity,
                              "input_ids": batch["input_ids"][0].cpu().numpy(), "option_token_ids": np.asarray(option_ids)}
                    outputs = [("identity", "identity", None, baseline, baseline_seconds, {"delta_context_l2": 0.0}, None, None)]
                    for record, controller in controllers:
                        check_deadline(guard)
                        guard.phase = case["case_id"] + ":" + record["tag"]
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        logits, _, metadata = controlled_forward(backend, batch, controller)
                        elapsed = time.perf_counter() - started
                        if metadata["maximum_head_relative_delta"] > record["residual_limit"] + 1e-6:
                            raise RuntimeError("Normalized controller exceeded its checkpoint head cap")
                        if metadata["maximum_absolute_gain"] > record["gain_limit"] + 1e-6:
                            raise RuntimeError("Normalized controller exceeded its checkpoint gain limit")
                        outputs.append((record["tag"], record["method"], record["seed"], logits, elapsed, metadata,
                                        record["checkpoint_sha256"], record["residual_limit"]))
                        arrays[record["tag"] + "_logits"] = logits
                    restored = forward_logits(backend, batch)
                    if not np.array_equal(restored, baseline):
                        raise RuntimeError("Native full-vocabulary output did not restore after normalized intervention")
                    if input_hashes != {name: tensor_hash(value) for name, value in batch.items()}:
                        raise RuntimeError("Paired evaluation input tensors changed")
                    arrays["restored_logits"] = restored
                    filename = f"{case_index:04d}.npz"
                    np.savez_compressed(output / filename, **arrays)
                    archive_hash = digest(output / filename)
                    teacher = torch.as_tensor(baseline[option_ids], dtype=torch.float64).log_softmax(dim=-1)
                    baseline_prediction = int(baseline[option_ids].argmax())
                    for tag, method, seed, logits, elapsed, metadata, checkpoint_hash, residual_limit in outputs:
                        selected = logits[option_ids]
                        prediction = int(selected.argmax())
                        semantic_prediction = case["semantic_original_option_indices"][prediction]
                        log_probabilities = torch.as_tensor(selected, dtype=torch.float64).log_softmax(dim=-1)
                        record = {**common, "tag": tag, "method": method, "seed": seed, "residual_limit": residual_limit,
                                  "option_logits": selected.tolist(), "predicted_answer_index": prediction,
                                  "predicted_original_option_index": semantic_prediction,
                                  "baseline_predicted_answer_index": baseline_prediction,
                                  "baseline_predicted_original_option_index": case["semantic_original_option_indices"][baseline_prediction],
                                  "correct": int(prediction == case["gold_answer_index"]) if case["supervised"] else None,
                                  "baseline_correct": int(baseline_prediction == case["gold_answer_index"]) if case["supervised"] else None,
                                  "option_KL_from_baseline": float((teacher.exp() * (teacher - log_probabilities)).sum()),
                                  "baseline_answer_changed": prediction != baseline_prediction,
                                  "exact_maximum_tie_indices": np.flatnonzero(selected == selected.max()).tolist(),
                                  "forward_seconds": elapsed, "capture": metadata, "checkpoint_sha256": checkpoint_hash,
                                  "restoration_exact": True, "archive_file": filename, "archive_sha256": archive_hash}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        config["completed_records"] += 1
                    stream.flush()
                    config["completed_cases"] += 1
                    write_json(output / "config.json", config)
                    print(f"evaluated {arguments.split} {case['case_id']} ({config['completed_cases']}/{len(cases)})", flush=True)
                    del batch, frames, arrays, outputs, baseline, identity, restored
            check_deadline(guard)
            config.update({"records_sha256": digest(output / "records.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            del backend, controllers, controller, zero
            torch.cuda.empty_cache()
        resources = json.loads((output / "resource_summary.json").read_text(encoding="utf-8"))
        if (output / "resource_stop.json").exists() or resources["status"] != "complete" or resources["exception"] is not None:
            raise RuntimeError("Normalized evaluation resource guard did not finish cleanly")
        if config["completed_records"] != config["expected_records"] or config["completed_cases"] != config["expected_cases"]:
            raise RuntimeError("Normalized evaluation matrix is incomplete")
        if digest(arguments.protocol) != protocol_hash or digest(snapshot / "protocol.md") != protocol_hash:
            raise RuntimeError("Protocol changed during normalized evaluation")
        if digest(arguments.questions) != config["questions_sha256"] or digest(arguments.frames_manifest) != config["frames_sha256"]:
            raise RuntimeError("Evaluation data manifests changed during the run")
        for name, expected in config["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected or digest(snapshot / name) != expected:
                raise RuntimeError("Evaluation implementation changed during the run")
        for record in checkpoint_records:
            if digest(record["path"]) != record["checkpoint_sha256"]:
                raise RuntimeError("Normalized checkpoint changed during evaluation")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "resource_summary_sha256": digest(output / "resource_summary.json")})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--video-ids")
    parser.add_argument("--rotations", type=int, nargs="+", choices=(0, 1, 2), default=[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
