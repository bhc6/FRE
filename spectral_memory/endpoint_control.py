"""Bounded endpoint-readability controls on explicitly constructed frame sequences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


VIDEO_ID = "video_1580"
INPUT_ROOT = Path("outputs")
PROTOCOL = Path("docs/endpoint_control_protocol_v1.md")
SOURCE_MANIFEST_SHA256 = "d141bfb6876efc155697f4673cb2aa43be46208bddcc5e984d35af6021c77de1"
ENDPOINTS = {
    "A": {"source_index": 0, "word": "men", "sha256": "4a8328d36754adf7fc5ce8b3be2a3034535f2ac93daf9c712fa50a617e2b678d"},
    "B": {"source_index": 15, "word": "enm", "sha256": "52dd2635cbdec968559da7788fb4325404555942255087717ec97f508b278040"},
}
SEQUENCES = (
    ("A16", "A" * 16), ("B16", "B" * 16),
    ("A8B8", "A" * 8 + "B" * 8), ("B8A8", "B" * 8 + "A" * 8),
    ("A2B14", "A" * 2 + "B" * 14), ("B14A2", "B" * 14 + "A" * 2),
    ("B2A14", "B" * 2 + "A" * 14), ("A14B2", "A" * 14 + "B" * 2),
)
OPTION_ORDERS = (("men_enm", ("men", "enm")), ("enm_men", ("enm", "men")))
LABELS = ("A", "B")
SOURCE_FILES = ("endpoint_control.py", "causal_intervention.py", "passive_probe.py", "resource_guard.py", "qwen3_backend.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_prompt(endpoint, options):
    if endpoint not in {"first", "last"} or tuple(options) not in {order for _, order in OPTION_ORDERS}:
        raise ValueError("Unexpected endpoint or option order")
    return (
        "This is a constructed displayed sequence, not the original recording. "
        "Answer only about the frames displayed in this constructed sequence.\n"
        f"Question: In the {endpoint} displayed frame, what word do the three letter cards spell from left to right?\n"
        f"A. {options[0]}\nB. {options[1]}\n"
        "Answer with the option letter only."
    )


def build_cases():
    cases = []
    for sequence, symbols in (*SEQUENCES, ("no_video", "")):
        for endpoint, position in (("first", 0), ("last", -1)):
            for order_name, options in OPTION_ORDERS:
                gold_word = ENDPOINTS[symbols[position]]["word"] if symbols else None
                cases.append({
                    "case_id": f"{VIDEO_ID}:endpoint_v1:{sequence}:{endpoint}:{order_name}",
                    "sequence": sequence, "symbols": symbols, "endpoint": endpoint,
                    "option_order": order_name, "options": list(options), "option_labels": list(LABELS),
                    "prompt": build_prompt(endpoint, options), "has_video": bool(symbols),
                    "source_frame_indices": [ENDPOINTS[symbol]["source_index"] for symbol in symbols],
                    "gold_word": gold_word, "gold_answer_index": options.index(gold_word) if symbols else None,
                    "static_readability_control": sequence in {"A16", "B16"},
                })
    if len(cases) != 36 or len({case["case_id"] for case in cases}) != 36:
        raise RuntimeError("Expected exactly 36 unique endpoint conditions")
    return cases


def input_manifest(data):
    record = data["record"]
    provenance = data["provenance"]
    timestamps = record["timestamps_s"]
    if record["video_id"] != VIDEO_ID or len(data["frames"]) != 16 or len(timestamps) != 16:
        raise ValueError("Expected the fixed video_1580 sixteen-frame source")
    if provenance["frames_manifest_sha256"] != SOURCE_MANIFEST_SHA256:
        raise ValueError("The predeclared source manifest changed")
    if any(not math.isfinite(value) for value in timestamps) or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Display timestamps must remain finite and increasing")
    endpoints = {}
    for symbol, specification in ENDPOINTS.items():
        index = specification["source_index"]
        if provenance["frame_sha256"][index] != specification["sha256"]:
            raise ValueError("A visually verified endpoint image changed")
        frame = data["frames"][index]
        endpoints[symbol] = {
            **specification, "path": record["frames"][index], "rgb_size": list(frame.size),
            "rgb_bytes_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "source_decoded_frame_index": record["source_frame_indices"][index],
            "source_pts_s": record["source_frame_timestamps_s"][index],
        }
    if endpoints["A"]["rgb_size"] != endpoints["B"]["rgb_size"]:
        raise ValueError("Endpoint spatial dimensions differ")
    return {
        "video_id": VIDEO_ID, "source_provenance": provenance,
        "source_frame_record_sha256": canonical_digest(record), "source_frame_record": record,
        "display_timestamps_s": list(timestamps), "endpoints": endpoints,
        "visual_verification": {
            "date": "2026-11-01", "reviewer": "root_agent_visual_inspection_before_run",
            "observation": "Original frame_0000.jpg spells men and frame_0015.jpg spells enm, left to right.",
            "scope": "Full original sampled images; no crop, OCR relabeling, or original shuffling/count labels.",
        },
        "sequences": [{"name": name, "symbols": symbols,
                       "source_frame_indices": [ENDPOINTS[symbol]["source_index"] for symbol in symbols],
                       "frame_sha256": [ENDPOINTS[symbol]["sha256"] for symbol in symbols]}
                      for name, symbols in SEQUENCES],
    }


def summarize(rows):
    cases = build_cases()
    if len(rows) != 36 or {row["case_id"] for row in rows} != {case["case_id"] for case in cases}:
        raise ValueError("A complete summary requires all 36 distinct conditions")
    video_rows = [row for row in rows if row["has_video"]]
    static_rows = [row for row in video_rows if row["static_readability_control"]]
    mixed_rows = [row for row in video_rows if not row["static_readability_control"]]
    prior_rows = [row for row in rows if not row["has_video"]]
    if (len(video_rows), len(static_rows), len(mixed_rows), len(prior_rows)) != (32, 8, 24, 4):
        raise ValueError("Endpoint groups do not match the frozen design")
    if any(row.get(field) is not None for row in prior_rows
           for field in ("correct", "gold_word", "gold_answer_index", "gold_margin")):
        raise ValueError("No-video priors must not have correctness labels")
    static_correct = sum(row["correct"] for row in static_rows)
    gate = static_correct == 8
    groups = {}
    consistency = []
    for sequence, _ in SEQUENCES:
        selected = [row for row in video_rows if row["sequence"] == sequence]
        groups[sequence] = {"correct": sum(row["correct"] for row in selected), "total": len(selected)}
        for endpoint in ("first", "last"):
            pair = [row for row in selected if row["endpoint"] == endpoint]
            consistency.append({"sequence": sequence, "endpoint": endpoint,
                                "same_predicted_word_across_option_orders": len({row["predicted_word"] for row in pair}) == 1})
    return {
        "video_qa_correct": sum(row["correct"] for row in video_rows), "video_qa_total": 32,
        "static_correct": static_correct, "static_total": 8, "static_readability_gate_passed": gate,
        "mixed_correct": sum(row["correct"] for row in mixed_rows), "mixed_total": 24,
        "mixed_interpretation": "interference_or_endpoint_readout_candidate_only" if gate else "static_readability_not_established",
        "long_term_forgetting_inferred": False, "per_sequence": groups,
        "option_order_consistency": consistency, "no_video_prior_count": 4,
        "no_video_accuracy": None, "unit_of_analysis": "one constructed source example, not 32 independent videos",
    }


def write_report(path, summary, rows):
    lines = ["# Constructed endpoint controls", "",
             f"All video QA: {summary['video_qa_correct']}/32; static controls: {summary['static_correct']}/8; mixed: {summary['mixed_correct']}/24.",
             f"Static readability gate passed: {summary['static_readability_gate_passed']}.",
             f"Interpretation: {summary['mixed_interpretation']}. No long-term forgetting claim.", "",
             "All 32 video conditions are reported. These are repeated controls on one source example.", "",
             "| Sequence | Endpoint | Option order | Gold word | Predicted word | Correct | Gold margin |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if row["has_video"]:
            lines.append(f"| {row['sequence']} | {row['endpoint']} | {row['option_order']} | {row['gold_word']} | {row['predicted_word']} | {row['correct']} | {row['gold_margin']:.6f} |")
    lines.extend(["", "No-video prompt priors have no ground truth or accuracy.", "",
                  "| Endpoint | Option order | Predicted word | Option A logit | Option B logit |",
                  "| --- | --- | --- | --- | --- |"])
    for row in rows:
        if not row["has_video"]:
            lines.append(f"| {row['endpoint']} | {row['option_order']} | {row['predicted_word']} | {row['option_logits'][0]:.6f} | {row['option_logits'][1]:.6f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_deadline(guard):
    if guard.stopped.is_set() or (guard.output / "resource_stop.json").exists():
        raise RuntimeError("Resource guard has stopped the run")
    if time.monotonic() - guard.started > guard.timeout_seconds:
        raise TimeoutError("The fixed 300-second process budget was exceeded")


def run(arguments):
    configure_threads()
    import numpy as np
    from .causal_intervention import load_frames
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory must be new; this diagnostic cannot resume")
    output.mkdir(parents=True)
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "video_id": VIDEO_ID, "expected_conditions": 36, "completed_conditions": 0,
              "device": "cuda", "dtype": "torch.bfloat16", "timeout_seconds": 300,
              "maximum_input_tokens": 1024, "maximum_visual_tokens": 512,
              "hooks_registered": False, "forwards_per_condition": 1,
              "purpose": "constructed_endpoint_readability_development_diagnostic"}
    write_json(output / "config.json", config)
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        config["source_sha256"] = {}
        for filename in SOURCE_FILES:
            source = Path(__file__).with_name(filename)
            shutil.copy2(source, snapshot / filename)
            config["source_sha256"][filename] = digest(source)
            if digest(snapshot / filename) != config["source_sha256"][filename]:
                raise RuntimeError("Source changed while snapshotting")
        shutil.copy2(PROTOCOL, snapshot / "protocol.md")
        config["protocol_sha256"] = digest(PROTOCOL)
        if digest(snapshot / "protocol.md") != config["protocol_sha256"]:
            raise RuntimeError("Protocol changed while snapshotting")
        cases = build_cases()
        write_json(output / "cases.json", cases)
        config["cases_sha256"] = digest(output / "cases.json")
        write_json(output / "config.json", config)
        rows = []
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            import torch

            guard.phase = "load_validated_endpoint_frames"
            data = load_frames(INPUT_ROOT)[VIDEO_ID]
            manifest = input_manifest(data)
            write_json(output / "inputs.json", manifest)
            config["inputs_sha256"] = digest(output / "inputs.json")
            check_deadline(guard)
            guard.phase = "load_model"
            backend = load_backend(device="cuda")
            if backend.model.dtype != torch.bfloat16:
                raise ValueError("Endpoint v1 requires BF16 without precision fallback")
            from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation

            shutil.copy2(implementation.__file__, snapshot / "modeling_qwen3_vl.py")
            config["modeling_source_sha256"] = digest(implementation.__file__)
            config["model"] = backend.metadata
            write_json(output / "config.json", config)
            identifiers = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
            if any(len(values) != 1 for values in identifiers) or len({values[0] for values in identifiers}) != 2:
                raise ValueError("Option letters must be distinct single tokens")
            option_ids = [values[0] for values in identifiers]
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    check_deadline(guard)
                    guard.phase = f"prepare:{case['case_id']}"
                    if case["has_video"]:
                        frames = [data["frames"][index] for index in case["source_frame_indices"]]
                        batch = backend.prepare_video(frames, manifest["display_timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    else:
                        batch = backend.prepare_text(case["prompt"])
                    if batch.metadata["input_tokens"] > 1024 or batch.metadata["visual_tokens"] > 512:
                        raise ValueError("Endpoint v1 input exceeds its fixed token budget")
                    tensor_hashes = {}
                    for name, tensor in batch.items():
                        cpu_tensor = tensor.detach().cpu().contiguous()
                        tensor_hashes[name] = {"shape": list(cpu_tensor.shape), "dtype": str(cpu_tensor.dtype),
                                               "sha256": hashlib.sha256(cpu_tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
                        del cpu_tensor
                    input_ids = batch["input_ids"][0].detach().cpu().numpy().copy()
                    guard.phase = f"forward:{case['case_id']}"
                    started = time.perf_counter()
                    logits = forward_logits(backend, batch)
                    elapsed = time.perf_counter() - started
                    check_deadline(guard)
                    if logits.ndim != 1 or not np.isfinite(logits).all():
                        raise RuntimeError("Expected finite full-vocabulary logits")
                    allowed = logits[option_ids]
                    predicted = int(allowed.argmax())
                    probabilities = np.exp(allowed.astype(np.float64) - float(allowed.max()))
                    probabilities /= probabilities.sum()
                    answer = case["gold_answer_index"]
                    filename = f"{len(rows):02d}_{case['sequence']}_{case['endpoint']}_{case['option_order']}.npz"
                    guard.phase = f"save:{case['case_id']}"
                    with (output / filename).open("xb") as handle:
                        np.savez_compressed(handle, baseline_logits=logits, input_ids=input_ids,
                                            option_token_ids=np.asarray(option_ids, dtype=np.int64))
                    row = {**case, "video_id": VIDEO_ID, "batch": batch.metadata,
                           "input_tensor_hashes": tensor_hashes, "input_token_ids": input_ids.tolist(),
                           "option_token_ids": option_ids, "option_logits": allowed.tolist(),
                           "two_option_probabilities": probabilities.tolist(), "option_logit_tie": bool(allowed[0] == allowed[1]),
                           "predicted_answer_index": predicted, "predicted_label": LABELS[predicted],
                           "predicted_word": case["options"][predicted],
                           "correct": int(predicted == answer) if answer is not None else None,
                           "gold_margin": float(allowed[answer] - allowed[1 - answer]) if answer is not None else None,
                           "archive_file": filename, "archive_sha256": digest(output / filename),
                           "full_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                           "full_logits_shape": list(logits.shape), "full_logits_dtype": str(logits.dtype),
                           "forward_seconds": elapsed}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    config["completed_conditions"] = len(rows)
                    write_json(output / "config.json", config)
                    print(f"endpoint {len(rows)}/36 {case['sequence']} {case['endpoint']} {case['option_order']}: {row['predicted_word']}; correct={row['correct']}", flush=True)
                    del batch, logits, input_ids
            guard.phase = "finalize"
            check_deadline(guard)
            summary = summarize(rows)
            write_json(output / "summary.json", summary)
            write_report(output / "report.md", summary, rows)
            config.update({"summary_sha256": digest(output / "summary.json"), "records_sha256": digest(output / "records.jsonl"),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            for filename, expected in config["source_sha256"].items():
                if digest(Path(__file__).with_name(filename)) != expected:
                    raise RuntimeError("Diagnostic source changed during execution")
            if digest(PROTOCOL) != config["protocol_sha256"]:
                raise RuntimeError("Protocol changed during execution")
            del backend
            torch.cuda.empty_cache()
            check_deadline(guard)
        resource_summary = json.loads((output / "resource_summary.json").read_text(encoding="utf-8"))
        if resource_summary["status"] != "complete" or (output / "resource_stop.json").exists():
            raise RuntimeError("Resource summary does not certify a completed run")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "resource_summary_sha256": digest(output / "resource_summary.json")})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
