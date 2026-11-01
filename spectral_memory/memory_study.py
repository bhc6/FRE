"""Build bounded visual memories and evaluate readers using stored payloads only."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .data import read_jsonl, write_jsonl
from .evaluate_small import build_prompt
from .qwen3_backend import DEFAULT_MODEL, DEFAULT_REVISION, load_backend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def in_subset(row, subset):
    if subset == "all":
        return True
    if subset == "memory":
        return row.get("area") == "memory"
    if subset == "temporal_state":
        tags = set(row.get("tags", []))
        return bool(tags.intersection({"change detection", "sequencing"}) or re.search(
            r"\b(before|after|first|last|beginning|end|order|changed|change|remember)\b",
            row.get("query_text", ""), flags=re.IGNORECASE))
    raise ValueError("Unknown evaluation subset")


def load_memory(row):
    path = Path(row["memory_path"])
    if path.stat().st_size != row["memory_bytes"] or digest(path) != row["memory_sha256"]:
        raise ValueError("Stored memory bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        frames = archive["frames"].copy()
        timestamps = archive["timestamps_s"].copy()
        indices = archive["frame_indices"].copy()
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Stored frames must be uint8 RGB arrays")
    if timestamps.dtype != np.float64 or indices.dtype != np.int32 or len(frames) != len(timestamps) or len(frames) != len(indices):
        raise ValueError("Invalid stored memory temporal arrays")
    if len(frames) % 2 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0) or np.any(np.diff(indices) <= 0):
        raise ValueError("Invalid stored frame pairing or temporal order")
    pairs = np.asarray(row["pair_indices"], dtype=np.int32)
    if len(frames) != 2 * len(pairs) or not np.array_equal(indices[::2], 2 * pairs) or not np.array_equal(indices[1::2], 2 * pairs + 1):
        raise ValueError("Stored frames disagree with the selected adjacent pairs")
    actual = hashlib.sha256(frames.tobytes() + timestamps.tobytes() + indices.tobytes()).hexdigest()
    if actual != row["payload_sha256"]:
        raise ValueError("Stored payload content hash mismatch")
    return frames, timestamps


def diagnostic_indices(count, budget, method):
    if method == "full_reference":
        return list(range(count))
    if method == "last_pair_reference":
        return [count - 1]
    if method == "first_half":
        candidates = np.arange(count // 2)
    elif method == "last_half":
        candidates = np.arange(count // 2, count)
    elif method.startswith("tail"):
        tail, coverage = method.split("_")
        fixed = int(tail[4:])
        if fixed >= budget:
            raise ValueError("Matched-tail diagnostic needs a nonempty preceding context")
        if coverage == "old":
            candidates = np.arange(count // 2)
        elif coverage == "recent":
            candidates = np.arange(count // 2, count - fixed)
        elif coverage == "uniform":
            candidates = np.arange(count - fixed)
        else:
            raise ValueError("Unknown matched-tail context")
        selected = candidates[np.rint(np.linspace(0, len(candidates) - 1, budget - fixed)).astype(int)]
        return sorted([*selected.tolist(), *range(count - fixed, count)])
    else:
        return None
    if len(candidates) < budget:
        raise ValueError("Insufficient candidates for requested diagnostic budget")
    return candidates[np.rint(np.linspace(0, len(candidates) - 1, budget)).astype(int)].tolist()


def build(arguments):
    from PIL import Image, ImageOps
    from .memory_selection import select_memory

    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Memory output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "payloads").mkdir()
    frames_source = arguments.frames_manifest.resolve()
    hidden_source = arguments.hidden_manifest.resolve()
    frame_rows = read_jsonl(frames_source)
    hidden_rows = {row["video_id"]: row for row in read_jsonl(hidden_source)}
    methods = arguments.methods.split(",")
    budgets = [int(value) for value in arguments.budgets.split(",")]
    if len(set(methods)) != len(methods) or len(set(budgets)) != len(budgets):
        raise ValueError("Duplicate methods or budgets")
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "incomplete",
        "frames_manifest_sha256": digest(frames_source), "hidden_manifest_sha256": digest(hidden_source),
        "methods": methods, "budgets_pairs": budgets, "seed": arguments.seed,
        "selection_source_sha256": digest(Path(__file__).with_name("memory_selection.py")),
        "study_source_sha256": digest(__file__),
        "memory_format": "uncompressed NPZ: uint8 RGB frames, float64 timestamps, int32 frame indices",
        "memory_canvas_width_height": [512, 288],
        "spatial_transform": "aspect-preserving bicubic fit into 512x288, centered black padding, no crop",
        "query_access": "only persisted NPZ, question and options; no original video or hidden-state access",
        "write_query_visibility": "none", "label_visibility": "none",
        "timing_limitation": "cached hidden extraction is additional observation cost for feature-based selectors",
    }
    write_json(output / "build.json", config)
    memories = []
    for frames_row in frame_rows:
        video_id = frames_row["video_id"]
        hidden_row = hidden_rows[video_id]
        with np.load(hidden_source.parent / hidden_row["hidden_path"], allow_pickle=False) as archive:
            hidden = archive["hidden_states"].copy()
            hidden_times = archive["timestamps_s"].copy()
        if len(frames_row["frames"]) != 2 * len(hidden):
            raise ValueError("Hidden states do not match source frame pairs")
        times = np.asarray(frames_row["timestamps_s"], dtype=np.float64)
        if not np.allclose(times.reshape(-1, 2).mean(axis=1), hidden_times, atol=1e-7, rtol=1e-5):
            raise ValueError("Frame and hidden timestamps disagree")
        frame_images = []
        started = time.perf_counter()
        hashes = []
        for filename in frames_row["frames"]:
            path = frames_source.parent / filename
            hashes.append(digest(path))
            with Image.open(path) as image:
                image = image.convert("RGB")
                if image.size != (512, 288):
                    resized = ImageOps.contain(image, (512, 288), method=Image.Resampling.BICUBIC)
                    image = Image.new("RGB", (512, 288), color=(0, 0, 0))
                    image.paste(resized, ((512 - resized.width) // 2, (288 - resized.height) // 2))
                frame_images.append(np.asarray(image).copy())
        if hashes != hidden_row["extraction"]["frame_sha256"]:
            raise ValueError("Hidden states were extracted from different frame bytes")
        images = np.stack(frame_images)
        decode_seconds = time.perf_counter() - started
        for budget in budgets:
            if budget < 2 or budget > len(hidden):
                raise ValueError("Memory budget must be between two and all source pairs")
            byte_counts = []
            for method in methods:
                started = time.perf_counter()
                indices = diagnostic_indices(len(hidden), budget, method)
                if indices is None:
                    selected = select_memory(hidden, hidden_times, budget, method, seed=arguments.seed)
                    indices = selected["indices"]
                else:
                    selected = {"indices": indices, "method": method, "uses_hidden_states": False}
                selection_seconds = time.perf_counter() - started
                indices = np.asarray(indices, dtype=np.int32)
                if np.any(np.diff(indices) <= 0) or np.any(indices < 0) or np.any(indices >= len(hidden)):
                    raise ValueError("Invalid selected pair indices")
                frame_indices = np.column_stack([2 * indices, 2 * indices + 1]).reshape(-1).astype(np.int32)
                retained = images[frame_indices]
                retained_times = times[frame_indices]
                payload_hash = hashlib.sha256(retained.tobytes() + retained_times.tobytes() + frame_indices.tobytes()).hexdigest()
                payload_path = output / "payloads" / f"{payload_hash}.npz"
                if not payload_path.exists():
                    np.savez(payload_path, frames=retained, timestamps_s=retained_times, frame_indices=frame_indices)
                archive_bytes = payload_path.stat().st_size
                is_reference = method.endswith("reference")
                if not is_reference:
                    if len(indices) != budget:
                        raise ValueError("Selector violated its pair budget")
                    byte_counts.append(archive_bytes)
                memories.append({
                    "video_id": video_id, "method": method, "budget_pairs": budget,
                    "retained_pairs": len(indices), "frame_indices": frame_indices.tolist(),
                    "pair_indices": indices.tolist(), "memory_path": str(payload_path.resolve()),
                    "memory_sha256": digest(payload_path), "payload_sha256": payload_hash,
                    "memory_bytes": archive_bytes, "frame_count": len(frame_indices),
                    "is_reference": is_reference, "selection_seconds": selection_seconds,
                    "frame_load_seconds": decode_seconds, "selection": selected,
                    "layer_index": hidden_row["layer_index"],
                    "hidden_extraction_seconds": hidden_row["extraction"]["elapsed_seconds"],
                    "observation_input_frames": len(images), "memory_array_shape": list(retained.shape),
                })
            if len(set(byte_counts)) > 1:
                raise ValueError("Same-budget methods do not have identical persistent bytes")
        write_jsonl(output / "memories.jsonl", memories)
        print(json.dumps({"video_id": video_id, "memories": len(memories)}, ensure_ascii=False), flush=True)
    config.update(status="complete", video_count=len(frame_rows), memory_count=len(memories))
    write_json(output / "build.json", config)


def evaluate(arguments):
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(arguments.memory_manifest)
    memory_keys = [(row["video_id"], row["method"], row["budget_pairs"]) for row in rows]
    if len(set(memory_keys)) != len(memory_keys):
        raise ValueError("Memory manifest repeats a video/method/budget combination")
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    questions = read_jsonl(arguments.questions)
    if arguments.video_id:
        questions = [question for question in questions if question["video_id"] in arguments.video_id]
    if arguments.query_id:
        questions = [question for question in questions if question["query_id"] in arguments.query_id]
    if not questions or len({question["query_id"] for question in questions}) != len(questions):
        raise ValueError("Questions must have nonempty unique IDs")
    for question in questions:
        if question["video_id"] not in by_video:
            raise ValueError("Question lacks stored visual memories")
        if not 0 <= question["answer_index"] < len(question["options"]):
            raise ValueError("Invalid official answer")
    backend_module = importlib.import_module(arguments.backend_module)
    model_path = arguments.model_path or backend_module.DEFAULT_MODEL
    revision = arguments.revision or backend_module.DEFAULT_REVISION
    config = {
        "questions_sha256": digest(arguments.questions), "memory_manifest_sha256": digest(arguments.memory_manifest),
        "max_pixels": arguments.max_pixels, "model": model_path, "revision": revision,
        "backend_module": arguments.backend_module,
        "decoding": "argmax_single_token_allowed_option_letters_v1", "seed": arguments.seed,
        "query_ids": sorted(question["query_id"] for question in questions),
        "backend_source_sha256": digest(backend_module.__file__),
        "prompt_source_sha256": digest(Path(__file__).with_name("evaluate_small.py")),
        "study_source_sha256": digest(__file__),
    }
    config_path = output / "config.json"
    records_path = output / "records.jsonl"
    if config_path.exists():
        if not arguments.resume or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output requires --resume and identical configuration/source hashes")
    elif any(output.iterdir()):
        raise ValueError("Output contains unrecognized existing artifacts")
    else:
        write_json(config_path, config)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("memory_study.py", Path(backend_module.__file__).name, "evaluate_small.py", "memory_selection.py"):
        source = Path(__file__).with_name(name)
        target = snapshot / name
        if not target.exists():
            target.write_bytes(source.read_bytes())
    records = read_jsonl(records_path) if records_path.exists() else []
    completed = {(row["query_id"], row["method"], row["budget_pairs"]) for row in records}
    if len(completed) != len(records):
        raise ValueError("Duplicate completed evaluations")
    cache = {(row["query_id"], row["payload_sha256"]): row["response"] for row in records}
    validated_payloads = {}
    run_identity = str(output) + digest(config_path)
    backend = backend_module.load_backend(model_path, device=arguments.device, revision=revision)
    write_json(output / "backend.json", backend.metadata)
    rng = np.random.default_rng(arguments.seed)
    tasks = []
    for question in sorted(questions, key=lambda value: value["query_id"]):
        tasks.append((question, None))
        order = rng.permutation(len(by_video[question["video_id"]]))
        tasks.extend((question, by_video[question["video_id"]][int(index)]) for index in order)
    write_json(output / "status.json", {"status": "running", "expected_records": len(tasks), "completed_records": len(records)})
    try:
        with records_path.open("a", encoding="utf-8") as stream:
            for question, memory in tasks:
                method = memory["method"] if memory else "no_memory"
                budget = memory["budget_pairs"] if memory else 0
                key = (question["query_id"], method, budget)
                if key in completed:
                    continue
                payload_hash = memory["payload_sha256"] if memory else "empty"
                cache_key = (question["query_id"], payload_hash)
                reused = cache_key in cache
                started = time.perf_counter()
                retained = None
                if memory:
                    file_status = Path(memory["memory_path"]).stat()
                    signature = (file_status.st_size, file_status.st_mtime_ns, memory["memory_sha256"], payload_hash)
                    if validated_payloads.get(memory["memory_path"]) != signature:
                        retained = load_memory(memory)
                        validated_payloads[memory["memory_path"]] = signature
                if reused:
                    response = cache[cache_key]
                else:
                    prompt = build_prompt(question)
                    if memory:
                        frames, timestamps = retained if retained is not None else load_memory(memory)
                        batch = backend.prepare_video(list(frames), timestamps.tolist(), prompt, arguments.max_pixels)
                    else:
                        batch = backend.prepare_text(prompt)
                    response = backend.choose_batch(batch, len(question["options"]))
                    response["request_elapsed_seconds"] = time.perf_counter() - started
                    response["forward_id"] = hashlib.sha256((run_identity + question["query_id"] + payload_hash + prompt).encode()).hexdigest()
                    cache[cache_key] = response
                record = {
                    "query_id": question["query_id"], "video_id": question["video_id"],
                    "area": question.get("area"), "reasoning": question.get("reasoning"),
                    "tags": question.get("tags", []), "query_text": question["question"],
                    "method": method, "budget_pairs": budget, "payload_sha256": payload_hash,
                    "memory_bytes": memory["memory_bytes"] if memory else 0,
                    "selected_pairs": memory["pair_indices"] if memory else [],
                    "retained_pairs": len(memory["pair_indices"]) if memory else 0,
                    "is_reference": method.endswith("reference"),
                    "reward": int(response["answer_index"] == question["answer_index"]),
                    "official_answer_index": question["answer_index"], "response": response,
                    "reused_identical_payload_forward": reused,
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                records.append(record)
                completed.add(key)
                if len(records) % 20 == 0 or method == "no_memory":
                    print(json.dumps({"completed": len(records), "total": len(tasks), "query_id": question["query_id"],
                                      "method": method, "budget": budget, "reward": record["reward"]}), flush=True)
        if len(records) != len(tasks):
            raise ValueError("Completed record count does not match the planned experiment")
        write_json(output / "status.json", {"status": "complete", "expected_records": len(tasks), "completed_records": len(records)})
    except BaseException as error:
        write_json(output / "status.json", {"status": "incomplete", "expected_records": len(tasks),
                                           "completed_records": len(records), "error": f"{type(error).__name__}: {error}"})
        raise
    summarize(output, arguments.seed)


def summarize(output, seed=17):
    rows = read_jsonl(Path(output) / "records.jsonl")
    scores = {}
    comparisons = []
    methods = sorted({(row["budget_pairs"], row["method"]) for row in rows})
    for budget, method in methods:
        selected = [row for row in rows if row["budget_pairs"] == budget and row["method"] == method]
        for subset in ("all", "memory", "temporal_state"):
            subset_rows = [row for row in selected if in_subset(row, subset)]
            if not subset_rows:
                continue
            grouped = defaultdict(list)
            for row in subset_rows:
                grouped[row["video_id"]].append(row["reward"])
            values = np.array([np.mean(grouped[video]) for video in sorted(grouped)])
            scores[f"{budget}:{method}:{subset}"] = {
                "budget_pairs": budget, "method": method, "subset": subset,
                "questions": len(subset_rows), "videos": len(grouped),
                "correct": sum(row["reward"] for row in subset_rows),
                "question_accuracy": float(np.mean([row["reward"] for row in subset_rows])),
                "video_macro_accuracy": float(values.mean()),
                "mean_memory_bytes": float(np.mean([row["memory_bytes"] for row in subset_rows])),
                "median_request_seconds": float(np.median([row["response"]["request_elapsed_seconds"] for row in subset_rows])),
            }
    budgets = sorted({row["budget_pairs"] for row in rows if row["method"] == "spectral_balanced"})
    for budget in budgets:
        main = {row["query_id"]: row for row in rows if row["budget_pairs"] == budget and row["method"] == "spectral_balanced"}
        for baseline in sorted({row["method"] for row in rows if row["budget_pairs"] == budget
                                and not row["method"].endswith("reference")} - {"spectral_balanced"}):
            control = {row["query_id"]: row for row in rows if row["budget_pairs"] == budget and row["method"] == baseline}
            if set(main) != set(control):
                raise ValueError("Paired comparison has missing questions")
            for subset in ("all", "memory", "temporal_state"):
                grouped = defaultdict(list)
                for query_id, row in main.items():
                    if not in_subset(row, subset):
                        continue
                    grouped[row["video_id"]].append(row["reward"] - control[query_id]["reward"])
                if not grouped:
                    continue
                values = np.array([np.mean(grouped[video]) for video in sorted(grouped)])
                rng = np.random.default_rng(seed)
                distribution = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
                comparisons.append({"budget_pairs": budget, "baseline": baseline, "subset": subset,
                                    "video_macro_difference": float(values.mean()),
                                    "paired_video_bootstrap_ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
                                    "independent_group_claim": "original videos; unknown participants; shared scenes may remain correlated"})
    report = {"scores": scores, "comparisons": comparisons,
              "limitations": ["No participant IDs; video bootstrap does not establish cross-person generalization.",
                              "Intervals are unadjusted descriptive comparisons; only the frozen primary contrast is confirmatory.",
                              "Full/last-pair references have different budgets and are reader diagnostics.",
                              "Per-video raw-image bytes are matched across same-budget methods; compute is reported separately.",
                              "Short-video event memory is not lifelong streaming memory."]}
    write_json(Path(output) / "report.json", report)
    lines = ["# Stored-memory QA study", "", "| Pairs | Method | Subset | Correct | Questions | Video macro | Mean bytes |",
             "| ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for value in scores.values():
        lines.append(f"| {value['budget_pairs']} | {value['method']} | {value['subset']} | {value['correct']} | {value['questions']} | {value['video_macro_accuracy']:.4f} | {value['mean_memory_bytes']:.0f} |")
    lines.extend(["", "## Paired spectral comparisons", "", "| Pairs | Baseline | Subset | Macro difference | 95% video bootstrap |",
                  "| ---: | --- | --- | ---: | --- |"])
    for value in comparisons:
        lower, upper = value["paired_video_bootstrap_ci95"]
        lines.append(f"| {value['budget_pairs']} | {value['baseline']} | {value['subset']} | {value['video_macro_difference']:.4f} | [{lower:.4f}, {upper:.4f}] |")
    lines.extend(["", "## Limitations", "", *[f"- {item}" for item in report["limitations"]]])
    (Path(output) / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("build")
    building.add_argument("--frames-manifest", type=Path, required=True)
    building.add_argument("--hidden-manifest", type=Path, required=True)
    building.add_argument("--methods", default="uniform,recent,random,temporal_difference,kcenter,temporal_reconstruction,temporal_centered,spectral_balanced,spectral_power")
    building.add_argument("--budgets", default="2,4,8")
    building.add_argument("--output-dir", type=Path, required=True)
    building.add_argument("--seed", type=int, default=17)
    evaluating = commands.add_parser("evaluate")
    evaluating.add_argument("--memory-manifest", type=Path, required=True)
    evaluating.add_argument("--questions", type=Path, required=True)
    evaluating.add_argument("--output-dir", type=Path, required=True)
    evaluating.add_argument("--video-id", action="append")
    evaluating.add_argument("--query-id", action="append")
    evaluating.add_argument("--device", default="cuda")
    evaluating.add_argument("--backend-module", default="spectral_memory.qwen3_backend")
    evaluating.add_argument("--model-path")
    evaluating.add_argument("--revision")
    evaluating.add_argument("--max-pixels", type=int, default=65536)
    evaluating.add_argument("--seed", type=int, default=17)
    evaluating.add_argument("--resume", action="store_true")
    reporting = commands.add_parser("report")
    reporting.add_argument("--output-dir", type=Path, required=True)
    reporting.add_argument("--seed", type=int, default=17)
    arguments = parser.parse_args()
    if arguments.command == "build":
        build(arguments)
    elif arguments.command == "evaluate":
        evaluate(arguments)
    else:
        summarize(arguments.output_dir, arguments.seed)


if __name__ == "__main__":
    main()
