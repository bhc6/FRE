"""Evaluate frozen controllers with complete native model forwards and paired inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


def cases_for(arguments):
    from .controller_runtime import read_rows
    from .evaluate_small import build_prompt

    if arguments.split == "endpoint":
        from .causal_intervention import load_frames
        from .endpoint_control import INPUT_ROOT, VIDEO_ID, build_cases, input_manifest

        data = load_frames(INPUT_ROOT)[VIDEO_ID]
        provenance = input_manifest(data)
        source = data["record"]
        cases = []
        for case in build_cases():
            if not case["has_video"]:
                continue
            indices = case["source_frame_indices"]
            cases.append({**case, "query_id": case["case_id"], "video_id": VIDEO_ID,
                          "condition": "real", "supervised": True, "question": case,
                          "frames": [source["frames"][index] for index in indices],
                          "frame_sha256": [provenance["source_provenance"]["frame_sha256"][index] for index in indices],
                          "timestamps_s": source["timestamps_s"]})
        return cases
    questions = read_rows(arguments.questions)
    sources = {row["video_id"]: row for row in read_rows(arguments.frames_manifest)}
    if arguments.video_ids:
        selected = set(arguments.video_ids.split(","))
        questions = [row for row in questions if row["video_id"] in selected]
        if {row["video_id"] for row in questions} != selected:
            raise ValueError("Requested evaluation videos are not completely declared")
    questions.sort(key=lambda row: (row["video_id"], row["query_id"]))
    if not questions or any(row["split"] != arguments.split for row in questions):
        raise ValueError("Evaluation questions must match the declared split")
    cases = []
    previous_video = None
    for question in questions:
        source = sources[question["video_id"]]
        if len(source["frames"]) != 16 or len(question["options"]) != 3:
            raise ValueError("Expected sixteen input frames and three official choices")
        conditions = ["real"] if question["video_id"] == previous_video else ["real", "static_first", "static_last"]
        previous_video = question["video_id"]
        for condition in conditions:
            real = condition == "real"
            source_index = 0 if condition == "static_first" else -1
            cases.append({"case_id": question["query_id"] + ":" + condition, "query_id": question["query_id"],
                          "video_id": question["video_id"], "condition": condition, "supervised": real,
                          "question": question if real else {key: value for key, value in question.items() if key not in {"answer_index", "answer_text"}},
                          "options": question["options"], "prompt": build_prompt(question),
                          "gold_answer_index": question["answer_index"] if real else None,
                          "frames": source["frames"] if real else [source["frames"][source_index]] * 16,
                          "frame_sha256": source["frame_sha256"] if real else [source["frame_sha256"][source_index]] * 16,
                          "timestamps_s": source["timestamps_s"]})
    return cases


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_runtime import controlled_forward, digest, tensor_hash, write_json
    from .controller_train import METHODS, SEEDS, STEPS
    from .endpoint_control import check_deadline
    from .learned_readout import ReadoutController
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Evaluation output must be new")
    output.mkdir(parents=True)
    cases = cases_for(arguments)
    protocol_hash = digest(arguments.protocol)
    checkpoint_records = []
    for directory in arguments.training_dirs:
        directory = Path(directory)
        train_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if train_config["status"] != "complete" or train_config["protocol_sha256"] != protocol_hash:
            raise ValueError("Training must be complete under the frozen evaluation protocol")
        for summary in train_config["completed_methods"]:
            path = directory / summary["method"] / "final.pt"
            if digest(path) != summary["checkpoint_sha256"]:
                raise ValueError("Final checkpoint was changed")
            checkpoint_records.append({"path": str(path.resolve()), **summary})
    pairs = {(row["method"], row["seed"]) for row in checkpoint_records}
    if pairs != {(method, seed) for method in METHODS for seed in SEEDS} or len(checkpoint_records) != 12:
        raise ValueError("Evaluation requires all four methods and three declared seeds")
    checkpoint_records.sort(key=lambda row: (row["seed"], METHODS.index(row["method"])))
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": arguments.split, "expected_cases": len(cases), "completed_cases": 0,
              "expected_records": len(cases) * 13, "protocol_sha256": protocol_hash,
              "checkpoints": checkpoint_records, "evaluation": "complete_uncached_native_model_forward",
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                 ("controller_evaluate.py", "controller_runtime.py", "learned_readout.py", "resource_guard.py",
                                  "qwen3_backend.py", "passive_probe.py", "evaluate_small.py", "endpoint_control.py")}}
    if arguments.split != "endpoint":
        config.update({"questions_sha256": digest(arguments.questions), "frames_sha256": digest(arguments.frames_manifest)})
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in config["source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(arguments.protocol, snapshot / "protocol.md")
    write_json(output / "cases.json", cases)
    config["cases_sha256"] = digest(output / "cases.json")
    write_json(output / "config.json", config)
    try:
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            guard.phase = "load_model_and_frozen_controllers"
            backend = load_backend(device="cuda")
            config["model"] = backend.metadata
            controllers = []
            for record in checkpoint_records:
                payload = torch.load(record["path"], map_location="cpu", weights_only=True)
                if payload["step"] != STEPS or payload["protocol_sha256"] != protocol_hash or (payload["method"], payload["seed"]) != (record["method"], record["seed"]):
                    raise ValueError("Checkpoint metadata differs from its frozen declaration")
                controller = ReadoutController(payload["method"], **payload["controller_config"]).to("cuda").eval()
                controller.load_state_dict(payload["state_dict"])
                controller.requires_grad_(False)
                controllers.append((record, controller))
            zero = ReadoutController("spectral_mlp").to("cuda").eval().requires_grad_(False)
            labels = "AB" if arguments.split == "endpoint" else "ABC"
            option_ids = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
            if any(len(identifiers) != 1 for identifiers in option_ids):
                raise ValueError("Expected single-token option labels")
            option_ids = [identifiers[0] for identifiers in option_ids]
            config["option_token_ids"] = option_ids
            total_records = 0
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for case_index, case in enumerate(cases):
                    check_deadline(guard)
                    guard.phase = f"prepare:{case['case_id']}"
                    if [digest(path) for path in case["frames"]] != case["frame_sha256"]:
                        raise ValueError("Evaluation frame content changed")
                    frames = [Image.open(path).convert("RGB") for path in case["frames"]]
                    batch = backend.prepare_video(frames, case["timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    hashes = {name: tensor_hash(value) for name, value in batch.items()}
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    baseline = forward_logits(backend, batch)
                    baseline_seconds = time.perf_counter() - started
                    identity, _, identity_metadata = controlled_forward(backend, batch, zero)
                    if not np.array_equal(identity, baseline) or identity_metadata["delta_context_l2"] != 0.0:
                        raise RuntimeError("Zero-initialized controller changed native whole-vocabulary logits")
                    common = {key: case[key] for key in ("case_id", "query_id", "video_id", "condition", "question", "gold_answer_index", "supervised", "frame_sha256")}
                    common.update({"split": arguments.split, "batch": batch.metadata, "input_tensor_hashes": hashes,
                                   "identity_exact": True, "option_token_ids": option_ids})
                    outputs = [("identity", None, baseline, baseline_seconds, {"delta_context_l2": 0.0}, None)]
                    arrays = {"identity_logits": baseline, "zero_logits": identity,
                              "input_ids": batch["input_ids"][0].cpu().numpy(), "option_token_ids": np.asarray(option_ids)}
                    for record, controller in controllers:
                        check_deadline(guard)
                        guard.phase = f"{case['case_id']}:{record['method']}:{record['seed']}"
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        logits, _, metadata = controlled_forward(backend, batch, controller)
                        elapsed = time.perf_counter() - started
                        if metadata["maximum_head_relative_delta"] > 0.100001 or metadata["maximum_absolute_gain"] > 0.250001:
                            raise RuntimeError("Controller exceeded a predeclared intervention limit")
                        outputs.append((record["method"], record["seed"], logits, elapsed, metadata, record["checkpoint_sha256"]))
                        arrays[f"{record['method']}_{record['seed']}_logits"] = logits
                    restored = forward_logits(backend, batch)
                    if not np.array_equal(restored, baseline):
                        raise RuntimeError("Native output was not restored after intervention")
                    if hashes != {name: tensor_hash(value) for name, value in batch.items()}:
                        raise RuntimeError("Evaluation input tensors changed between methods")
                    arrays["restored_logits"] = restored
                    filename = f"{case_index:04d}.npz"
                    np.savez_compressed(output / filename, **arrays)
                    archive_hash = digest(output / filename)
                    teacher = torch.as_tensor(baseline[option_ids], dtype=torch.float64).log_softmax(dim=-1)
                    for method, seed, logits, elapsed, metadata, checkpoint_hash in outputs:
                        selected = logits[option_ids]
                        predicted = int(selected.argmax())
                        log_probabilities = torch.as_tensor(selected, dtype=torch.float64).log_softmax(dim=-1)
                        row = {**common, "method": method, "seed": seed, "option_logits": selected.tolist(),
                               "predicted_answer_index": predicted, "baseline_predicted_answer_index": int(baseline[option_ids].argmax()),
                               "correct": int(predicted == case["gold_answer_index"]) if case["supervised"] else None,
                               "baseline_correct": int(baseline[option_ids].argmax() == case["gold_answer_index"]) if case["supervised"] else None,
                               "option_KL_from_baseline": float((teacher.exp() * (teacher - log_probabilities)).sum()),
                               "baseline_answer_changed": bool(predicted != int(baseline[option_ids].argmax())),
                               "forward_seconds": elapsed, "capture": metadata, "checkpoint_sha256": checkpoint_hash,
                               "restoration_exact": True, "archive_file": filename, "archive_sha256": archive_hash}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        total_records += 1
                    stream.flush()
                    config["completed_cases"] += 1
                    config["completed_records"] = total_records
                    write_json(output / "config.json", config)
                    print(f"evaluated {arguments.split} {case['case_id']} ({config['completed_cases']}/{len(cases)})", flush=True)
                    del batch, frames, arrays, outputs, baseline, restored, identity
            check_deadline(guard)
            config.update({"records_sha256": digest(output / "records.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            del backend, controllers, controller, zero
            torch.cuda.empty_cache()
        if (output / "resource_stop.json").exists() or total_records != config["expected_records"]:
            raise RuntimeError("Incomplete evaluation or resource guard stop")
        if digest(arguments.protocol) != protocol_hash:
            raise RuntimeError("Protocol changed during evaluation")
        for name, expected in config["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected or digest(snapshot / name) != expected:
                raise RuntimeError("Evaluation implementation changed during the run")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "endpoint"), required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--frames-manifest", type=Path)
    parser.add_argument("--training-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids")
    arguments = parser.parse_args()
    if arguments.split != "endpoint" and (arguments.questions is None or arguments.frames_manifest is None):
        parser.error("Official evaluation requires question and frame manifests")
    run(arguments)


if __name__ == "__main__":
    main()
