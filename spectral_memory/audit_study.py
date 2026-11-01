"""Independently validate completed stored-memory studies and summarize costs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .memory_study import in_subset, load_memory


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def paired_difference(rows, method, baseline, budget, subset, seed=17):
    main = {row["query_id"]: row for row in rows
            if row["method"] == method and row["budget_pairs"] == budget and in_subset(row, subset)}
    controls = {row["query_id"]: row for row in rows
                if row["method"] == baseline and row["budget_pairs"] == budget and in_subset(row, subset)}
    if not main or set(main) != set(controls):
        raise ValueError("Paired comparison has missing questions")
    videos = defaultdict(list)
    wins = losses = 0
    for query_id, row in main.items():
        difference = row["reward"] - controls[query_id]["reward"]
        videos[row["video_id"]].append(difference)
        wins += int(difference > 0)
        losses += int(difference < 0)
    values = np.asarray([np.mean(videos[video]) for video in sorted(videos)])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "method": method, "baseline": baseline, "budget_pairs": budget, "subset": subset,
        "questions": len(main), "videos": len(videos), "question_wins": wins, "question_losses": losses,
        "video_macro_difference": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def audit(arguments):
    study = arguments.study.resolve()
    config = json.loads((study / "config.json").read_text(encoding="utf-8"))
    status = json.loads((study / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Do not audit an incomplete study")
    if digest(arguments.questions) != config["questions_sha256"]:
        raise ValueError("Question file differs from the evaluated source")
    if digest(arguments.memories) != config["memory_manifest_sha256"]:
        raise ValueError("Memory manifest differs from the evaluated source")
    source_questions = read_jsonl(arguments.questions)
    if len(set(config["query_ids"])) != len(config["query_ids"]):
        raise ValueError("Run configuration repeats question IDs")
    questions = {row["query_id"]: row for row in source_questions
                 if row["query_id"] in config["query_ids"]}
    if set(questions) != set(config["query_ids"]):
        raise ValueError("Configured question IDs are missing from the source")
    memories = read_jsonl(arguments.memories)
    memory_index = {(row["video_id"], row["method"], row["budget_pairs"]): row for row in memories}
    if len(memory_index) != len(memories):
        raise ValueError("Duplicate memory cells")
    expected = set()
    for query_id, question in questions.items():
        expected.add((query_id, "no_memory", 0))
        expected.update((query_id, method, budget) for video, method, budget in memory_index
                        if video == question["video_id"])
    rows = read_jsonl(study / "records.jsonl")
    keys = [(row["query_id"], row["method"], row["budget_pairs"]) for row in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Study has duplicate, missing or extra cells")
    if len(rows) != status["expected_records"] or len(rows) != status["completed_records"]:
        raise ValueError("Completion metadata disagrees with records")
    forward_cache = {}
    for row in rows:
        question = questions[row["query_id"]]
        response = row["response"]
        logits = np.asarray(response["option_logits"], dtype=np.float64)
        probabilities = np.asarray(response["option_probabilities"], dtype=np.float64)
        if logits.shape != (len(question["options"]),) or not np.isfinite(logits).all():
            raise ValueError("Invalid option logits")
        recomputed = np.exp(logits - logits.max())
        recomputed /= recomputed.sum()
        if not np.allclose(probabilities, recomputed, rtol=1e-5, atol=1e-7):
            raise ValueError("Option probabilities disagree with logits")
        if response["answer_index"] != int(logits.argmax()):
            raise ValueError("Prediction differs from allowed-token argmax")
        expected_reward = int(response["answer_index"] == question["answer_index"])
        if row["reward"] != expected_reward or row["official_answer_index"] != question["answer_index"]:
            raise ValueError("Stored reward disagrees with the official answer")
        if row["video_id"] != question["video_id"] or row["query_text"] != question["question"]:
            raise ValueError("Question identity mismatch")
        for field in ("area", "reasoning", "tags"):
            expected_value = question.get(field, [] if field == "tags" else None)
            if row.get(field) != expected_value:
                raise ValueError(f"Question {field} metadata differs from the official source")
        if response["model_id"] != config["model"] or response["model_revision"] != config["revision"]:
            raise ValueError("Model provenance mismatch")
        key = (row["query_id"], row["payload_sha256"])
        if key in forward_cache and forward_cache[key] != response:
            raise ValueError("Identical-payload reuse changed the model response")
        forward_cache[key] = response
        if row["method"] == "no_memory":
            if row["memory_bytes"] != 0 or row["selected_pairs"]:
                raise ValueError("No-memory reference contains visual payload")
        else:
            memory = memory_index[(row["video_id"], row["method"], row["budget_pairs"])]
            for key in ("memory_bytes", "payload_sha256"):
                if row[key] != memory[key]:
                    raise ValueError("Readout differs from persisted memory metadata")
            if row["selected_pairs"] != memory["pair_indices"]:
                raise ValueError("Readout selected-pair mismatch")
    verified = set()
    bytes_by_budget = defaultdict(set)
    for memory in memories:
        if memory["memory_path"] not in verified:
            load_memory(memory)
            verified.add(memory["memory_path"])
        if not memory["method"].endswith("reference"):
            if len(memory["pair_indices"]) != memory["budget_pairs"]:
                raise ValueError("Memory violates its declared budget")
            bytes_by_budget[memory["budget_pairs"]].add(memory["memory_bytes"])
    if any(len(values) != 1 for values in bytes_by_budget.values()):
        raise ValueError("Same-budget payload sizes differ")
    snapshot = study / "source_snapshot"
    for name, field in (("memory_study.py", "study_source_sha256"),
                        (config["backend_module"].split(".")[-1] + ".py", "backend_source_sha256"),
                        ("evaluate_small.py", "prompt_source_sha256")):
        if digest(snapshot / name) != config[field]:
            raise ValueError("Source snapshot differs from frozen run configuration")
    freeze_checks = {}
    frozen_run_binding = False
    if arguments.freeze:
        freeze = json.loads(arguments.freeze.read_text(encoding="utf-8"))
        for filename, expected_digest in freeze["inputs"].items():
            freeze_checks[filename] = digest(filename) == expected_digest
        if not all(freeze_checks.values()):
            raise ValueError("Frozen experiment inputs changed")
        if set(questions) != {row["query_id"] for row in source_questions}:
            raise ValueError("Frozen confirmation audit requires all source questions")
        frozen_run_binding = (
            config["questions_sha256"] == freeze["inputs"].get("data/perception_confirm/questions.jsonl")
            and config["study_source_sha256"] == freeze["inputs"].get("spectral_memory/memory_study.py")
            and config["prompt_source_sha256"] == freeze["inputs"].get("spectral_memory/evaluate_small.py")
            and config["backend_source_sha256"] == freeze["inputs"].get("spectral_memory/qwen3_backend.py")
        )
    performance = {}
    by_method = defaultdict(list)
    for row in rows:
        by_method[(row["budget_pairs"], row["method"])].append(row)
    for (budget, method), group in sorted(by_method.items()):
        unique = {row["response"]["forward_id"]: row["response"] for row in group}
        performance[f"{budget}:{method}"] = {
            "unique_forwards_within_group": len(unique),
            "visual_tokens": sorted({row["response"]["visual_tokens"] for row in group}),
            "input_tokens": distribution([row["response"]["input_tokens"] for row in group]),
            "model_seconds": distribution([response["elapsed_seconds"] for response in unique.values()]),
            "request_seconds": distribution([response["request_elapsed_seconds"] for response in unique.values()]),
            "peak_allocated_bytes": max(response["peak_memory_bytes"] for response in unique.values()),
            "timing_note": "Identical payloads reuse real forwards across groups; these costs are not additive.",
        }
    comparisons = []
    if any(row["method"] == "spectral_balanced" for row in rows):
        comparisons = [paired_difference(rows, "spectral_balanced", baseline, 4, subset)
                       for baseline in ("kcenter", "temporal_centered")
                       for subset in ("all", "memory", "temporal_state")]
    primary = next((value for value in comparisons if value["baseline"] == "kcenter"
                    and value["subset"] == "temporal_state"), None)
    mechanism = next((value for value in comparisons if value["baseline"] == "temporal_centered"
                      and value["subset"] == "temporal_state"), None)
    threshold_pass = bool(primary and mechanism and primary["video_macro_difference"] >= 0.03
                          and primary["ci95"][0] > 0 and mechanism["ci95"][0] > 0)
    primary_reader = (
        config["backend_module"] == "spectral_memory.qwen3_backend"
        and config["model"] == "Qwen/Qwen3-VL-2B-Instruct"
        and config["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    )
    primary_applicable = bool(primary_reader and frozen_run_binding)
    report = {
        "status": "passed_integrity_checks", "records": len(rows), "questions": len(questions),
        "videos": len({row["video_id"] for row in rows}), "unique_payload_files": len(verified),
        "unique_forwards": len({row["response"]["forward_id"] for row in rows}),
        "method_counts": dict(Counter(row["method"] for row in rows)),
        "matched_payload_bytes": {str(budget): sorted(values)[0] for budget, values in bytes_by_budget.items()},
        "primary_comparisons": comparisons,
        "primary_reader_identity": primary_reader,
        "frozen_primary_applicable": primary_applicable,
        "frozen_v1_success": threshold_pass if primary_applicable else None,
        "threshold_pass_descriptive": threshold_pass,
        "freeze_checks": freeze_checks, "performance": performance,
        "hashes": {"records": digest(study / "records.jsonl"), "config": digest(study / "config.json"),
                   "memories": digest(arguments.memories), "questions": digest(arguments.questions),
                   "auditor_source": digest(__file__)},
        "limitations": [
            "Passing integrity checks is not evidence of method effectiveness.",
            "The v1 success flag applies to the primary Qwen reader; transfer results are secondary.",
            "Latency is one sequential run; cached calls, warmup and thermal effects prevent a controlled speed claim.",
            "NPZ byte counts exclude external indexes, observation workspace and model weights.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "records", "unique_forwards", "frozen_v1_success",
                                                  "primary_comparisons")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--memories", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
