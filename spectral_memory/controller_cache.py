"""Acquire only declared source splits for frozen-tail controller training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .resource_guard import ResourceGuard, configure_threads


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_runtime import controlled_forward, digest, read_rows, save_tail, tensor_hash, write_json
    from .evaluate_small import build_prompt
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend
    from .endpoint_control import check_deadline

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Cache output must be new")
    output.mkdir(parents=True)
    questions = read_rows(arguments.questions)
    frames_by_video = {row["video_id"]: row for row in read_rows(arguments.frames_manifest)}
    if arguments.video_ids:
        requested = set(arguments.video_ids.split(","))
        questions = [row for row in questions if row["video_id"] in requested]
    if not questions:
        raise ValueError("No declared source questions")
    questions.sort(key=lambda row: (row["video_id"], row["query_id"]))
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "questions_sha256": digest(arguments.questions), "frames_sha256": digest(arguments.frames_manifest),
              "add_static_distillation": arguments.static_distillation, "completed_cases": 0,
              "protocol_sha256": digest(arguments.protocol),
              "expected_cases": len(questions) + (2 * len({row["video_id"] for row in questions}) if arguments.static_distillation else 0),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                 ("controller_runtime.py", "controller_cache.py", "learned_readout.py", "resource_guard.py", "qwen3_backend.py", "passive_probe.py", "evaluate_small.py")}}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in config["source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(arguments.protocol, snapshot / "protocol.md")
    write_json(output / "config.json", config)
    try:
        rows = []
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            guard.phase = "load_model"
            backend = load_backend(device="cuda")
            config["model"] = backend.metadata
            option_ids = [backend.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
            if any(len(identifiers) != 1 for identifiers in option_ids):
                raise ValueError("Expected three single option tokens")
            option_ids = [identifiers[0] for identifiers in option_ids]
            tail_manifest = save_tail(backend, output / "frozen_tail.pt", option_ids)
            write_json(output / "frozen_tail_manifest.json", tail_manifest)
            config["tail_manifest_sha256"] = digest(output / "frozen_tail_manifest.json")
            config["tail_sha256"] = digest(output / "frozen_tail.pt")
            config["option_token_ids"] = option_ids
            previous_video = None
            frames = None
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for question in questions:
                    check_deadline(guard)
                    source = frames_by_video[question["video_id"]]
                    if len(source["frames"]) != 16:
                        raise ValueError("Use the frozen 16-frame manifest")
                    first_question = previous_video != question["video_id"]
                    if first_question:
                        frames = [Image.open(path).convert("RGB") for path in source["frames"]]
                        frame_hashes = [digest(path) for path in source["frames"]]
                        if frame_hashes != source["frame_sha256"]:
                            raise ValueError("Source frames changed after data freeze")
                        previous_video = question["video_id"]
                    conditions = ["real"]
                    if arguments.static_distillation and first_question:
                        conditions += ["static_first", "static_last"]
                    for condition in conditions:
                        guard.phase = f"{question['query_id']}:{condition}"
                        images = frames if condition == "real" else [frames[0 if condition == "static_first" else -1]] * 16
                        batch = backend.prepare_video(images, source["timestamps_s"], build_prompt(question), DEFAULT_MAX_PIXELS)
                        native = forward_logits(backend, batch)
                        logits, arrays, metadata = controlled_forward(backend, batch, retain_features=True)
                        if not np.array_equal(native, logits):
                            raise RuntimeError("Passive controller feature capture altered native logits")
                        option_count = len(question["options"])
                        if option_count != 3:
                            raise ValueError("Official controller study requires three choices")
                        filename = f"{len(rows):04d}.npz"
                        np.savez_compressed(output / filename, baseline_logits=logits, option_token_ids=np.asarray(option_ids), **arrays)
                        row = {"case_id": question["query_id"] + ":" + condition, "query_id": question["query_id"],
                               "video_id": question["video_id"], "condition": condition, "question": question,
                               "gold_answer_index": question["answer_index"] if condition == "real" else None,
                               "supervised": condition == "real", "option_logits": logits[option_ids].tolist(),
                               "predicted_answer_index": int(logits[option_ids].argmax()),
                               "correct": int(logits[option_ids].argmax() == question["answer_index"]) if condition == "real" else None,
                               "batch": batch.metadata, "input_tensor_hashes": {name: tensor_hash(value) for name, value in batch.items()},
                               "passive_exact": True, "capture": metadata, "archive_file": filename,
                               "archive_sha256": digest(output / filename), "source_frame_sha256": frame_hashes,
                               "frame_sha256": frame_hashes if condition == "real" else [frame_hashes[0 if condition == "static_first" else -1]] * 16}
                        if condition != "real":
                            row["question"] = {key: value for key, value in question.items() if key not in {"answer_index", "answer_text"}}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        config["completed_cases"] = len(rows)
                        write_json(output / "config.json", config)
                        del batch, arrays, logits, native
                    print(f"cached {question['query_id']} ({len(rows)} conditions)", flush=True)
            check_deadline(guard)
            config.update({"records_sha256": digest(output / "records.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            del backend
            torch.cuda.empty_cache()
        if (output / "resource_stop.json").exists():
            raise RuntimeError("Resource guard stopped extraction")
        if len(rows) != config["expected_cases"] or digest(arguments.protocol) != config["protocol_sha256"]:
            raise RuntimeError("Extraction coverage or protocol changed")
        for name, expected in config["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected or digest(snapshot / name) != expected:
                raise RuntimeError("Extraction implementation changed during the run")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--video-ids")
    parser.add_argument("--static-distillation", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
