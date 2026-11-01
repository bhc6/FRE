"""Run descriptive MCQA interventions with a shared, once-loaded small-model backend."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from .data import read_jsonl


CONDITIONS = ("no_memory", "first_half", "last_half", "uniform_full")
PROMPT_TEMPLATE = (
    "Answer the multiple-choice question using the provided visual evidence, if any. "
    "The frames may cover only part of the original video. "
    "Choose exactly one of the listed options. "
    "Return only its capital letter, with no explanation.\n\n"
    "Question: {question}\n{options}\nAnswer:"
)
REWARD_METRIC = "mcq_exact_match_0_1_strict_letter_v1"


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _append_record(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def parse_answer(raw_text, option_count):
    """Accept one capital option letter, optionally bracketed and/or punctuated."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {"status": "empty", "answer_index": None}
    match = re.fullmatch(r"(?:([A-Z])|\(([A-Z])\)|\[([A-Z])\])(?:[.。])?", raw_text.strip())
    if match is None:
        return {"status": "invalid_format", "answer_index": None}
    letter = next(group for group in match.groups() if group is not None)
    index = ord(letter) - ord("A")
    if not 0 <= index < option_count:
        return {"status": "out_of_range", "answer_index": None}
    return {"status": "valid", "answer_index": index}


def build_prompt(question):
    options = "\n".join(f"{chr(ord('A') + index)}. {option}"
                        for index, option in enumerate(question["options"]))
    return PROMPT_TEMPLATE.format(question=question["question"], options=options)


def select_frame_indices(timestamps_s, condition, frame_count):
    """Keep real selected timestamps; rounded uniform indices need not be equispaced."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 2 or frame_count % 2:
        raise ValueError("frame_count must be an even integer >= 2")
    if condition == "no_memory":
        return []
    midpoint = (timestamps_s[0] + timestamps_s[-1]) / 2.0
    indices = list(range(len(timestamps_s)))
    if condition == "first_half":
        indices = [index for index in indices if timestamps_s[index] <= midpoint]
    elif condition == "last_half":
        indices = [index for index in indices if timestamps_s[index] >= midpoint]
    if len(indices) < frame_count:
        raise ValueError(f"{condition} has {len(indices)} frames, fewer than requested {frame_count}; "
                         "use fewer frames per condition or a denser whole-video manifest")
    positions = [round(index * (len(indices) - 1) / (frame_count - 1)) for index in range(frame_count)]
    selected = [indices[position] for position in positions]
    if len(set(selected)) != frame_count:
        raise ValueError("Frame selection duplicated timestamps")
    return selected


def _load_inputs(arguments):
    questions = read_jsonl(arguments.questions)
    frames_source = Path(arguments.frames_manifest).resolve()
    frame_records = read_jsonl(frames_source)
    frames_by_video = {}
    for record in frame_records:
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in frames_by_video:
            raise ValueError("Frame manifest needs unique nonempty video_id values")
        frames, timestamps = record.get("frames"), record.get("timestamps_s")
        if not isinstance(frames, list) or not isinstance(timestamps, list) or len(frames) != len(timestamps):
            raise ValueError(f"{video_id}: frames and timestamps_s must be matching lists")
        if len(frames) < 2 or not all(isinstance(timestamp, (int, float))
                                      and not isinstance(timestamp, bool) and math.isfinite(timestamp)
                                      for timestamp in timestamps):
            raise ValueError(f"{video_id}: at least two finite timestamps are required")
        if any(following <= previous for previous, following in zip(timestamps, timestamps[1:])):
            raise ValueError(f"{video_id}: frame timestamps must strictly increase")
        source_times = record.get("source_frame_timestamps_s")
        if source_times is not None and (not isinstance(source_times, list) or len(source_times) != len(frames)):
            raise ValueError(f"{video_id}: source_frame_timestamps_s length mismatch")
        if source_times is not None and not all(
            isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and math.isfinite(timestamp)
            for timestamp in source_times
        ):
            raise ValueError(f"{video_id}: source frame timestamps must be finite numbers")
        if source_times is not None and any(following < previous for previous, following in zip(source_times, source_times[1:])):
            raise ValueError(f"{video_id}: source frame timestamps must not move backwards")
        absolute_frames = []
        for frame in frames:
            if not isinstance(frame, str) or not frame:
                raise ValueError(f"{video_id}: frame paths must be nonempty strings")
            path = Path(frame)
            if not path.is_absolute():
                path = frames_source.parent / path
            if not path.is_file():
                raise ValueError(f"Missing frame: {path}")
            absolute_frames.append(str(path.resolve()))
        frames_by_video[video_id] = dict(record, frames=absolute_frames)
    query_ids = set()
    for question in questions:
        query_id = question.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise ValueError("Questions need unique nonempty query_id values")
        query_ids.add(query_id)
        options = question.get("options")
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{query_id}: empty question")
        if not isinstance(options, list) or not 2 <= len(options) <= 26 or not all(
            isinstance(option, str) and option.strip() for option in options
        ):
            raise ValueError(f"{query_id}: expected 2–26 nonempty answer options")
        answer = question.get("answer_index")
        if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < len(options):
            raise ValueError(f"{query_id}: invalid zero-based official answer_index")
    requested = set(arguments.query_id or [])
    if requested - query_ids:
        raise ValueError(f"Unknown query IDs: {sorted(requested - query_ids)}")
    selected = [question for question in questions if not requested or question["query_id"] in requested]
    selected.sort(key=lambda question: (question.get("area") != "memory", question["query_id"]))
    if arguments.limit is not None:
        selected = selected[:arguments.limit]
    for question in selected:
        if question.get("video_id") not in frames_by_video:
            raise ValueError(f"No frame manifest for {question['query_id']}")
    return selected, frames_by_video


def _make_payload(record, condition, frame_count, payload_dir, hash_cache):
    indices = select_frame_indices(record["timestamps_s"], condition, frame_count)
    selected_frames = []
    for index in indices:
        path = record["frames"][index]
        stat = Path(path).stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        if path not in hash_cache:
            hash_cache[path] = (signature, _hash_bytes(Path(path).read_bytes()))
        elif hash_cache[path][0] != signature:
            raise ValueError(f"Frame changed during evaluation: {path}")
        item = {"path": path, "sha256": hash_cache[path][1],
                "timestamp_s": record["timestamps_s"][index], "source_index": index}
        if record.get("source_frame_timestamps_s") is not None:
            item["source_frame_timestamp_s"] = record["source_frame_timestamps_s"][index]
        selected_frames.append(item)
    payload = {"schema": "visual_memory_payload_descriptor_v1", "video_id": record["video_id"],
               "video_sha256": record.get("video_sha256"), "selection_condition": condition,
               "frames": selected_frames,
               "scope": "Descriptor binds exact frame bytes and timestamps supplied to the backend"}
    encoded = _canonical_bytes(payload)
    payload_hash = _hash_bytes(encoded)
    path = payload_dir / f"{payload_hash}.json"
    if not path.exists():
        path.write_bytes(encoded)
    return payload, payload_hash, path


def _model_field(response, metadata, name, fallback=None):
    return response.get(name) or metadata.get(name) or fallback


def _report(output, run_config, metadata, records, utilities, status, error=None):
    completed = {(record["query_id"], record["condition"]) for record in records if record["status"] == "completed"}
    pending = [{"query_id": query_id, "condition": condition}
               for query_id in run_config["selected_query_ids"]
               for condition in run_config["executed_conditions"] if (query_id, condition) not in completed]
    summaries = {}
    for condition in run_config["executed_conditions"]:
        rows = [record for record in records if record["condition"] == condition and record["status"] == "completed"]
        summaries[condition] = {
            "questions": len(rows),
            "correct": sum(record["reward"] for record in rows),
            "accuracy": sum(record["reward"] for record in rows) / len(rows) if rows else None,
            "parse_failures": sum(record["parser"]["status"] != "valid" for record in rows),
            "memory_area_questions": sum(record.get("area") == "memory" for record in rows),
        }
    report = {"status": status, "run_config": run_config, "backend": metadata,
              "evaluated_queries": len({record["query_id"] for record in records if record["status"] == "completed"}),
              "evaluated_source_videos": len({record["video_id"] for record in records if record["status"] == "completed"}),
              "forward_calls": len(records), "conditions": summaries, "results": records,
              "paired_utilities": len(utilities), "error": error, "pending_evaluations": pending,
              "recovery": "Completed records are flushed after each call. No automatic resume: rerun pending query IDs "
                          "in a fresh directory, retaining this run and its fixed configuration.",
              "limitations": [
                  "Descriptive small-model mechanism pilot; repeated conditions are not independent samples.",
                  "Equal frame count does not prove equal visual token count or information coverage.",
                  "No-memory has zero visual budget; differences are coverage diagnostics, not fair method comparisons.",
                  "Official QA answers score real generations; they are never injected as memory-utility labels.",
                  "Utility is measured against empty memory for one frozen readout, not lifelong storage quality.",
                  "No probe is trained and no hidden-state path is fabricated by this command.",
                  "Payload descriptors bind frame bytes but do not measure total retained-memory storage cost.",
                  "Uniform-full covers the supplied frame manifest; original full-video coverage requires its sampling provenance.",
              ]}
    _write_json(output / "report.json", report)
    lines = ["# Small-model MCQA pilot", "", f"Status: `{status}`", "",
             "Descriptive diagnostics only; each question's conditions are paired observations.", "",
             "| Condition | Questions | Correct | Accuracy | Parse failures |", "| --- | ---: | ---: | ---: | ---: |"]
    for condition, summary in summaries.items():
        accuracy = "undefined" if summary["accuracy"] is None else f"{summary['accuracy']:.3f}"
        lines.append(f"| {condition} | {summary['questions']} | {summary['correct']} | {accuracy} | {summary['parse_failures']} |")
    lines.extend(["", "## Model and outputs", "",
                  f"Model: `{metadata.get('model_id', run_config['model_path'])}`; "
                  f"revision: `{metadata.get('model_revision', 'unresolved_revision')}`.", "",
                  "Complete prompts, payload hashes and generation metadata are retained in report.json and records.jsonl.", "",
                  "| Query | Condition | Raw answer | Parse | Reward | Input/output tokens | Generation/request seconds | Peak bytes |",
                  "| --- | --- | --- | --- | ---: | --- | ---: | ---: |"])
    for record in records:
        if record["status"] != "completed":
            continue
        generation = record["generation"]
        answer = json.dumps(record["raw_answer"], ensure_ascii=False).replace("|", "\\|").replace("`", "'")
        tokens = f"{generation.get('input_tokens', 'unavailable')}/{generation.get('output_tokens', 'unavailable')}"
        latency = f"{generation.get('elapsed_seconds', 'unavailable')}/{generation.get('request_elapsed_seconds', 'unavailable')}"
        lines.append(f"| {record['query_id']} | {record['condition']} | {answer} | {record['parser']['status']} | "
                     f"{record['reward']} | {tokens} | {latency} | "
                     f"{generation.get('peak_memory_bytes', 'unavailable')} |")
    lines.extend(["", "## Limitations", ""] + [f"- {item}" for item in report["limitations"]])
    if error:
        lines.extend(["", f"Run error: `{error}`", "", f"Pending query/condition pairs: {len(pending)}.",
                      "Completed records remain available; see report.json for the exact pending list and recovery guidance."])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_evaluation(arguments, backend=None):
    conditions = [part.strip() for part in arguments.conditions.split(",") if part.strip()]
    if not conditions or len(set(conditions)) != len(conditions) or any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"Conditions must be unique values from {CONDITIONS}")
    if arguments.frames_per_condition < 2 or arguments.frames_per_condition % 2:
        raise ValueError("frames-per-condition must be positive, even, and at least two")
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("limit must be positive")
    if not 1 <= arguments.max_new_tokens <= 256 or arguments.max_pixels < 1024:
        raise ValueError("max-new-tokens must be within 1..256 and max-pixels >= 1024")
    executed = ["no_memory", *[condition for condition in conditions if condition != "no_memory"]]
    questions, frames_by_video = _load_inputs(arguments)
    if not questions:
        raise ValueError("No questions selected")
    for question in questions:
        for condition in executed:
            select_frame_indices(frames_by_video[question["video_id"]]["timestamps_s"], condition,
                                 arguments.frames_per_condition)
    output = Path(arguments.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    payload_dir = output / "payloads"
    payload_dir.mkdir()
    base_payload = {"schema": "visual_memory_payload_descriptor_v1", "frames": []}
    base_encoded = _canonical_bytes(base_payload)
    base_hash = _hash_bytes(base_encoded)
    (payload_dir / "empty_memory.json").write_bytes(base_encoded)
    decoding = {"do_sample": False, "max_new_tokens": arguments.max_new_tokens,
                "seed": arguments.seed, "max_pixels": arguments.max_pixels,
                "parser": "strict_capital_letter_optional_brackets_punctuation_v1"}
    template_hash = _hash_bytes(PROMPT_TEMPLATE.encode("utf-8"))
    decoding_hash = _hash_bytes(_canonical_bytes(decoding))
    run_config = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "questions_path": str(Path(arguments.questions).resolve()),
                  "questions_sha256": _hash_bytes(Path(arguments.questions).read_bytes()),
                  "frames_manifest_path": str(Path(arguments.frames_manifest).resolve()),
                  "frames_manifest_sha256": _hash_bytes(Path(arguments.frames_manifest).read_bytes()),
                  "requested_conditions": conditions, "executed_conditions": executed,
                  "baseline_added_for_pairing": "no_memory" not in conditions,
                  "selected_query_ids": [question["query_id"] for question in questions],
                  "model_path": arguments.model_path, "requested_revision": arguments.revision,
                  "device": arguments.device, "frames_per_condition": arguments.frames_per_condition,
                  "implementation_sha256": {"evaluate_small.py": _hash_bytes(Path(__file__).read_bytes()),
                                             "qwen3_backend.py": _hash_bytes(Path(__file__).with_name("qwen3_backend.py").read_bytes())},
                  "prompt_template": PROMPT_TEMPLATE, "prompt_hash": template_hash,
                  "decoding": decoding, "decoding_hash": decoding_hash,
                  "reward_metric": REWARD_METRIC, "query_presentation_policy": "after_source_video"}
    _write_json(output / "run_config.json", run_config)
    records, utilities, metadata, hash_cache = [], [], {}, {}
    status, error = "failed", None
    try:
        if backend is None:
            from .qwen3_backend import load_backend
            backend = load_backend(arguments.model_path, device=arguments.device, revision=arguments.revision)
        metadata = dict(backend.metadata)
        _write_json(output / "backend.json", metadata)
        with (output / "records.jsonl").open("w", encoding="utf-8") as records_handle, \
             (output / "evaluator_logs.jsonl").open("w", encoding="utf-8") as evaluations_handle, \
             (output / "candidates.jsonl").open("w", encoding="utf-8") as candidates_handle, \
             (output / "utilities.jsonl").open("w", encoding="utf-8") as utilities_handle:
            for question in questions:
                prompt = build_prompt(question)
                base_result = None
                source = frames_by_video[question["video_id"]]
                for condition in executed:
                    payload, payload_hash, payload_path = _make_payload(
                        source, condition, arguments.frames_per_condition, payload_dir, hash_cache)
                    if condition == "no_memory":
                        payload_hash, payload_path = base_hash, payload_dir / "empty_memory.json"
                    frame_paths = [frame["path"] for frame in payload["frames"]]
                    timestamps = [frame["timestamp_s"] for frame in payload["frames"]]
                    record = {"query_id": question["query_id"], "video_id": question["video_id"],
                              "area": question.get("area"), "reasoning": question.get("reasoning"),
                              "tags": question.get("tags", []), "condition": condition,
                              "prompt": prompt, "input_prompt_sha256": _hash_bytes(prompt.encode("utf-8")),
                              "prompt_hash": template_hash, "decoding_hash": decoding_hash,
                              "candidate_hash": payload_hash, "memory_payload_path": str(payload_path),
                              "frame_count": len(frame_paths), "timestamps_s": timestamps,
                              "manifest_coverage_s": [source["timestamps_s"][0], source["timestamps_s"][-1]],
                              "selected_coverage_s": [timestamps[0], timestamps[-1]] if timestamps else None,
                              "source_sampling_provenance": source.get("sampling"),
                              "forward_id": _hash_bytes(_canonical_bytes([question["query_id"], condition, payload_hash])),
                              "seed": arguments.seed, "status": "started"}
                    request_started = time.perf_counter()
                    try:
                        response = backend.generate_response(
                            question=prompt, frame_paths=frame_paths, timestamps_s=timestamps,
                            max_new_tokens=arguments.max_new_tokens, seed=arguments.seed,
                            max_pixels=arguments.max_pixels)
                        response = dict(response, request_elapsed_seconds=time.perf_counter() - request_started,
                                        request_timing_scope="image I/O, input preprocessing, model generation and decoding; excludes model load and payload hashing")
                        record.update(raw_answer=response.get("raw_text"), generation=response)
                        raw_text = response["raw_text"]
                        if not isinstance(raw_text, str):
                            raise ValueError("Backend raw_text must be a string")
                        if response.get("do_sample", False) is not False or response.get("seed", arguments.seed) != arguments.seed:
                            raise ValueError("Backend decoding configuration differs from the fixed deterministic protocol")
                        parsed = parse_answer(raw_text, len(question["options"]))
                        reward = int(parsed["status"] == "valid" and parsed["answer_index"] == question["answer_index"])
                        model_id = _model_field(response, metadata, "model_id", arguments.model_path)
                        revision = _model_field(response, metadata, "model_revision", "unresolved_revision")
                        if base_result is not None and (model_id, revision) != (base_result["model_id"], base_result["model_revision"]):
                            raise ValueError("Model identity changed between paired no-memory and visual conditions")
                        if metadata.get("model_id") and model_id != metadata["model_id"]:
                            raise ValueError("Generation model_id differs from the loaded backend")
                        if metadata.get("model_revision") and revision != metadata["model_revision"]:
                            raise ValueError("Generation model_revision differs from the loaded backend")
                        record.update(status="completed", raw_answer=raw_text, parser=parsed, reward=reward,
                                      official_answer_index=question["answer_index"],
                                      model_id=model_id, model_revision=revision, generation=response)
                        if isinstance(response.get("rendered_prompt"), str):
                            record["rendered_prompt_sha256"] = _hash_bytes(response["rendered_prompt"].encode("utf-8"))
                    except Exception as generation_error:
                        record.update(status="failed", error=f"{type(generation_error).__name__}: {generation_error}",
                                      request_elapsed_seconds=time.perf_counter() - request_started)
                        records.append(record)
                        _append_record(records_handle, record)
                        raise
                    records.append(record)
                    _append_record(records_handle, record)
                    print(json.dumps({"query_id": question["query_id"], "condition": condition,
                                      "raw_answer": raw_text, "parser_status": parsed["status"],
                                      "reward": reward}, ensure_ascii=False), flush=True)
                    if condition == "no_memory":
                        base_result = record
                        continue
                    sample_id = f"{question['query_id']}::{condition}::{payload_hash[:16]}"
                    memory_id = f"empty::{base_hash[:16]}"
                    candidate = {"schema": "visual_candidates_without_hidden_states_v1", "sample_id": sample_id,
                                 "candidate_id": f"{question['video_id']}::{condition}::{payload_hash[:16]}",
                                 "query_id": question["query_id"], "video_id": question["video_id"],
                                 "subject_id": question.get("subject_id"), "session_id": question.get("session_id"),
                                 "task_mode": "retrieve", "memory_state_id": memory_id,
                                 "evaluation_group": f"{question['query_id']}::{memory_id}",
                                 "candidate_hash": payload_hash, "base_memory_hash": base_hash,
                                 "memory_payload_path": str(payload_path), "condition": condition,
                                 "hidden_status": "not_extracted", "frame_count": len(frame_paths)}
                    _append_record(candidates_handle, candidate)
                    common = {"sample_id": sample_id, "query_id": question["query_id"],
                              "memory_state_id": memory_id, "seed": arguments.seed,
                              "evaluator_id": model_id, "evaluator_revision": revision,
                              "prompt_hash": template_hash, "decoding_hash": decoding_hash,
                              "reward_metric": REWARD_METRIC, "base_memory_hash": base_hash,
                              "candidate_hash": payload_hash, "candidate_in_base": False,
                              "query_after_event": True, "coverage_condition": condition}
                    for label_condition, result in (("without", base_result), ("with", record)):
                        _append_record(evaluations_handle, dict(common, condition=label_condition,
                                                               reward=result["reward"], raw_answer=result["raw_answer"],
                                                               parser=result["parser"],
                                                               source_forward_id=result["forward_id"],
                                                               original_forward_condition=result["condition"]))
                    utility = dict(candidate, utility=record["reward"] - base_result["reward"],
                                   with_reward=record["reward"], without_reward=base_result["reward"],
                                   utility_protocol={key: common[key] for key in
                                                     ("evaluator_id", "evaluator_revision", "prompt_hash", "decoding_hash", "reward_metric")},
                                   label_source="actual_paired_model_generations", descriptive_only=True)
                    utilities.append(utility)
                    _append_record(utilities_handle, utility)
        status = "complete"
    except Exception as run_error:
        error = f"{type(run_error).__name__}: {run_error}"
        raise
    finally:
        report = _report(output, run_config, metadata, records, utilities, status, error)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Actual small-model MCQA pilot with descriptive coverage controls")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--frames-manifest", required=True)
    parser.add_argument("--model-path", "--model", dest="model_path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--frames-per-condition", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main():
    arguments = build_parser().parse_args()
    try:
        report = run_evaluation(arguments)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({"status": report["status"], "queries": report["evaluated_queries"],
                      "report": str(Path(arguments.output_dir).resolve() / "report.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
