"""Independently audit full-forward records and report paired video-level statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np


METHODS = ("time_mlp", "spectral_mlp", "spectral_phaseblind", "constant_spectral")
SEEDS = (20261101, 20261102, 20261103)
PAIRS = (("identity", None),) + tuple((method, seed) for method in METHODS for seed in SEEDS)
TEMPORAL_TAGS = {"sequencing", "action counting", "change detection"}
SUBSETS = ("all", "temporal", "memory", "descriptive_only")
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261101
PROTOCOL_SHA256 = "3954767ef2662e15eb4934f90eea08ca6d48eaaef39fd2d297e3d9a3df590748"
MANIFEST_HASHES = {
    "train.questions.jsonl": "b1cb8b15865aa387bed3cefb7f6a922f41f0c1c7301fe590f2f562920b3c5845",
    "validation.questions.jsonl": "76e60bb8b3cc522dfd3aa6323782d753ef58caa78b7dbcbe864224f34d7d3504",
    "test.questions.jsonl": "10a5fe30f688a512bbb93b3ad049f2f02fae3b6a88da613354d05b42aa72e4ac",
    "frames16/validation.frames.manifest.jsonl": "82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9",
    "frames16/test.frames.manifest.jsonl": "b15fded6c1eecda205b69b04e12d284f5a183d42d54aa8c48bab9d962582eca9",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def log_softmax(values):
    centered = np.asarray(values, dtype=np.float64) - np.max(values)
    return centered - np.log(np.exp(centered).sum())


def subsets_for(question):
    tags = set(question["tags"])
    result = ["all"]
    if tags & TEMPORAL_TAGS:
        result.append("temporal")
    if question["area"] == "memory":
        result.append("memory")
    if question["reasoning"] == "descriptive" and question["area"] != "memory" and not tags & TEMPORAL_TAGS:
        result.append("descriptive_only")
    return result


def expected_cases(manifest_root, split):
    if split == "endpoint":
        sequences = {"A16": "A" * 16, "B16": "B" * 16, "A8B8": "A" * 8 + "B" * 8,
                     "B8A8": "B" * 8 + "A" * 8, "A2B14": "A" * 2 + "B" * 14,
                     "B14A2": "B" * 14 + "A" * 2, "B2A14": "B" * 2 + "A" * 14,
                     "A14B2": "A" * 14 + "B" * 2}
        output = {}
        for sequence, symbols in sequences.items():
            for endpoint, position in (("first", 0), ("last", -1)):
                for order, options in (("men_enm", ["men", "enm"]), ("enm_men", ["enm", "men"])):
                    case_id = f"video_1580:endpoint_v1:{sequence}:{endpoint}:{order}"
                    output[case_id] = {"video_id": "video_1580", "condition": "real", "options": options,
                                       "gold": options.index("men" if symbols[position] == "A" else "enm"),
                                       "subset": "static_readability" if sequence in {"A16", "B16"} else "mixed",
                                       "frame_sha256": [{"A": "4a8328d36754adf7fc5ce8b3be2a3034535f2ac93daf9c712fa50a617e2b678d",
                                                         "B": "52dd2635cbdec968559da7788fb4325404555942255087717ec97f508b278040"}[symbol] for symbol in symbols],
                                       "symbols": symbols, "endpoint": endpoint, "sequence": sequence}
        return output
    question_file = manifest_root / f"{split}.questions.jsonl"
    frame_file = manifest_root / "frames16" / f"{split}.frames.manifest.jsonl"
    require(digest(question_file) == MANIFEST_HASHES[f"{split}.questions.jsonl"], "Frozen questions changed")
    require(digest(frame_file) == MANIFEST_HASHES[f"frames16/{split}.frames.manifest.jsonl"], "Frozen frames changed")
    sources = {row["video_id"]: row for row in read_rows(frame_file)}
    questions = sorted(read_rows(question_file), key=lambda row: (row["video_id"], row["query_id"]))
    output = {}
    visited = set()
    for question in questions:
        video_id = question["video_id"]
        for condition in (["real"] if video_id in visited else ["real", "static_first", "static_last"]):
            real = condition == "real"
            frame_hashes = sources[video_id]["frame_sha256"]
            if not real:
                frame_hashes = [frame_hashes[0 if condition == "static_first" else -1]] * 16
            output[question["query_id"] + ":" + condition] = {
                "video_id": video_id, "condition": condition, "options": question["options"],
                "gold": question["answer_index"] if real else None, "frame_sha256": frame_hashes,
                "question": question if real else {key: value for key, value in question.items() if key not in {"answer_index", "answer_text"}},
            }
        visited.add(video_id)
    return output


def audit_archived_case(directory, rows, expected, checkpoints):
    first = rows[0]
    case_id = first["case_id"]
    require(len(rows) == 13 and {(row["method"], row["seed"]) for row in rows} == set(PAIRS), f"Incomplete method/seed matrix: {case_id}")
    require(first["video_id"] == expected["video_id"] and first["condition"] == expected["condition"], "Case identity changed")
    require(first["gold_answer_index"] == expected["gold"], "Gold label changed")
    require(first["supervised"] == (expected["gold"] is not None), "Static supervision leaked")
    require(first["frame_sha256"] == expected["frame_sha256"], "Frame sequence differs from frozen manifest")
    if first["split"] != "endpoint":
        require(first["question"] == expected["question"], "Question metadata differs from frozen manifest")
    else:
        question = first["question"]
        require(question["options"] == expected["options"] and question["symbols"] == expected["symbols"], "Seen endpoint content changed")
        require(question["endpoint"] == expected["endpoint"] and question["sequence"] == expected["sequence"], "Seen endpoint target changed")
        require(question["gold_answer_index"] == expected["gold"], "Seen endpoint gold changed")
    common_fields = ("case_id", "query_id", "video_id", "split", "condition", "question", "supervised", "gold_answer_index",
                     "frame_sha256", "input_tensor_hashes", "option_token_ids", "archive_file", "archive_sha256", "batch")
    for row in rows:
        require(all(row[field] == first[field] for field in common_fields), "Methods did not share a paired input")
        require(row["identity_exact"] is True and row["restoration_exact"] is True, "Native identity/restoration failed")
    archive_path = directory / first["archive_file"]
    require(archive_path.parent.resolve() == directory.resolve(), "Archive escaped its run directory")
    require(digest(archive_path) == first["archive_sha256"], "Full-vocabulary archive hash changed")
    option_ids = np.asarray(first["option_token_ids"], dtype=np.int64)
    require(len(option_ids) == len(expected["options"]) == len(set(option_ids.tolist())), "Option token mapping is not one-to-one")
    audited = []
    with np.load(archive_path, allow_pickle=False) as archive:
        native = np.asarray(archive["identity_logits"], dtype=np.float64)
        require(native.ndim == 1 and np.isfinite(native).all(), "Native logits invalid")
        require(np.array_equal(archive["zero_logits"], native) and np.array_equal(archive["restored_logits"], native), "Archived zero/restored logits not exactly native")
        require(np.array_equal(option_ids, archive["option_token_ids"]) and np.min(option_ids) >= 0 and np.max(option_ids) < len(native), "Archived option IDs invalid")
        teacher = log_softmax(native[option_ids])
        native_prediction = int(native[option_ids].argmax())
        native_margin = float(np.sort(native[option_ids])[-1] - np.sort(native[option_ids])[-2])
        for row in rows:
            pair = (row["method"], row["seed"])
            key = "identity_logits" if pair[0] == "identity" else f"{pair[0]}_{pair[1]}_logits"
            full_logits = np.asarray(archive[key], dtype=np.float64)
            require(full_logits.shape == native.shape and np.isfinite(full_logits).all(), "Method full-vocabulary logits invalid")
            logits = full_logits[option_ids]
            require(np.array_equal(logits, np.asarray(row["option_logits"])), "Recorded option logits differ from archive")
            prediction = int(logits.argmax())
            correct = int(prediction == expected["gold"]) if first["supervised"] else None
            divergence = float(np.sum(np.exp(teacher) * (teacher - log_softmax(logits))))
            require(row["predicted_answer_index"] == prediction and row["correct"] == correct, "Recorded prediction/accuracy invalid")
            require(row["baseline_predicted_answer_index"] == native_prediction, "Recorded baseline prediction invalid")
            require(row["baseline_correct"] == (int(native_prediction == expected["gold"]) if first["supervised"] else None), "Recorded baseline accuracy invalid")
            require(row["baseline_answer_changed"] == (prediction != native_prediction), "Recorded baseline flip invalid")
            require(abs(row["option_KL_from_baseline"] - divergence) <= 1e-10, "Recorded KL failed independent FP64 recomputation")
            capture = row["capture"]
            if pair[0] == "identity":
                require(row["checkpoint_sha256"] is None and capture["delta_context_l2"] == 0, "Identity changed")
            else:
                require(row["checkpoint_sha256"] == checkpoints[pair], "Row checkpoint differs from frozen checkpoint")
                require(capture["injection_layer"] == 27 and capture["maximum_head_relative_delta"] <= 0.100001, "Intervention placement or head limit changed")
                gains = np.asarray(capture["head_gains"])
                require(gains.shape == (16, 8) and np.isfinite(gains).all(), "Head gain diagnostics invalid")
                require(abs(float(np.max(np.abs(gains))) - capture["maximum_absolute_gain"]) <= 1e-7 and np.max(np.abs(gains)) <= 0.250001, "Head gain limit invalid")
            selected_margin = float(np.sort(logits)[-1] - np.sort(logits)[-2])
            gold_margin = None if expected["gold"] is None else float(logits[expected["gold"]] - np.max(np.delete(logits, expected["gold"])))
            audited.append({**row, "evaluation_directory": str(directory.resolve()), "recomputed_option_KL": max(divergence, 0.0),
                            "exact_maximum_tie_indices": np.flatnonzero(logits == logits.max()).tolist(),
                            "gold_margin": gold_margin, "top_two_margin": selected_margin,
                            "top_two_margin_change": selected_margin - native_margin,
                            "option_logit_maximum_change": float(np.max(np.abs(logits - native[option_ids]))),
                            "full_vocabulary_maximum_change": float(np.max(np.abs(full_logits - native))),
                            "full_vocabulary_rms_change": float(np.sqrt(np.mean((full_logits - native) ** 2))),
                            "analysis_subsets": ([expected["subset"]] if first["split"] == "endpoint" else subsets_for(first["question"])) if first["supervised"] else ["static_no_gold"]})
    return audited


def audit_evaluations(directories, manifest_root, protocol):
    require(digest(protocol) == PROTOCOL_SHA256, "Analysis requires the frozen protocol")
    output = []
    run_records = []
    seen_cases = defaultdict(set)
    expectations = {}
    frozen_checkpoints = None
    frozen_sources = None
    for directory in directories:
        directory = Path(directory).resolve()
        config = read_json(directory / "config.json")
        require(config["status"] == "complete", f"Incomplete run: {directory}")
        require(config["protocol_sha256"] == PROTOCOL_SHA256, "Evaluation protocol differs")
        require(config["evaluation"] == "complete_uncached_native_model_forward", "Primary scores must use full forward")
        require(config["model"]["model_id"] == "Qwen/Qwen3-VL-2B-Instruct" and config["model"]["model_revision"] == "89644892e4d85e24eaac8bacfd4f463576704203", "Base model or revision changed")
        require(digest(directory / "records.jsonl") == config["records_sha256"], "Evaluation records changed")
        require(digest(directory / "cases.json") == config["cases_sha256"], "Evaluation cases changed")
        require(digest(directory / "source_snapshot" / "protocol.md") == PROTOCOL_SHA256, "Protocol snapshot changed")
        for name, expected_hash in config["source_sha256"].items():
            require(digest(directory / "source_snapshot" / name) == expected_hash, "Evaluation source snapshot changed")
        if frozen_sources is None:
            frozen_sources = config["source_sha256"]
        require(config["source_sha256"] == frozen_sources, "Evaluation shards use different implementations")
        checkpoints = {(row["method"], row["seed"]): row["checkpoint_sha256"] for row in config["checkpoints"]}
        require(set(checkpoints) == set(PAIRS[1:]) and len(config["checkpoints"]) == 12, "Frozen checkpoint matrix incomplete")
        for record in config["checkpoints"]:
            require(digest(record["path"]) == record["checkpoint_sha256"], "A final checkpoint changed on disk")
        if frozen_checkpoints is None:
            frozen_checkpoints = checkpoints
        require(checkpoints == frozen_checkpoints, "Evaluation shards use different checkpoints")
        split = config["split"]
        require(split in {"validation", "test", "endpoint"}, "Unknown evaluation split")
        if split not in expectations:
            expectations[split] = expected_cases(manifest_root, split)
        if split != "endpoint":
            require(config["questions_sha256"] == MANIFEST_HASHES[f"{split}.questions.jsonl"], "Evaluation question provenance changed")
            require(config["frames_sha256"] == MANIFEST_HASHES[f"frames16/{split}.frames.manifest.jsonl"], "Evaluation frame provenance changed")
        rows = read_rows(directory / "records.jsonl")
        cases = read_json(directory / "cases.json")
        require(len(rows) == config["expected_records"] == config["completed_records"], "Run record count incomplete")
        require(len(cases) == config["expected_cases"] == config["completed_cases"], "Run case count incomplete")
        grouped = defaultdict(list)
        for row in rows:
            require(row["split"] == split, "Mixed split inside run")
            grouped[row["case_id"]].append(row)
        require(set(grouped) == {case["case_id"] for case in cases}, "Case manifest coverage differs")
        for case_id, case_rows in grouped.items():
            require(case_id in expectations[split] and case_id not in seen_cases[split], "Unknown or duplicate evaluation case")
            output.extend(audit_archived_case(directory, case_rows, expectations[split][case_id], checkpoints))
            seen_cases[split].add(case_id)
        run_records.append({"directory": str(directory), "config_sha256": digest(directory / "config.json"), "config": config})
    require(bool(output), "No evaluation records")
    for split, expected in expectations.items():
        missing = sorted(set(expected) - seen_cases[split])
        require(not missing, f"Partial {split} matrix: missing {len(missing)} cases; first={missing[:3]}")
    return output, run_records


def metric_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        for subset in row["analysis_subsets"]:
            grouped[(row["split"], subset, row["method"], row["seed"])].append(row)
    result = []
    for (split, subset, method, seed), selected in grouped.items():
        videos = defaultdict(list)
        for row in selected:
            videos[row["video_id"]].append(row)
        metric = {"split": split, "subset": subset, "method": method, "seed": seed,
                  "source_videos": len(videos), "cases": len(selected), "identity_is_single_baseline": method == "identity"}
        if subset != "static_no_gold":
            metric.update({"correct": sum(row["correct"] for row in selected),
                           "micro_accuracy": float(np.mean([row["correct"] for row in selected])),
                           "video_macro_accuracy": float(np.mean([np.mean([row["correct"] for row in group]) for group in videos.values()])),
                           "gold_margin_mean": float(np.mean([row["gold_margin"] for row in selected]))})
            if subset == "static_readability":
                metric["static_8_of_8_gate_passed"] = metric["correct"] == metric["cases"] == 8
        else:
            metric.update({"correct": None, "micro_accuracy": None, "video_macro_accuracy": None,
                           "option_KL_mean": float(np.mean([row["recomputed_option_KL"] for row in selected])),
                           "option_KL_max": float(np.max([row["recomputed_option_KL"] for row in selected])),
                           "top1_changed_count": sum(row["baseline_answer_changed"] for row in selected),
                           "top1_changed_fraction": float(np.mean([row["baseline_answer_changed"] for row in selected])),
                           "absolute_top_two_margin_change_mean": float(np.mean([abs(row["top_two_margin_change"]) for row in selected])),
                           "option_logit_maximum_change": float(max(row["option_logit_maximum_change"] for row in selected))})
        metric["exact_option_ties"] = sum(len(row["exact_maximum_tie_indices"]) > 1 for row in selected)
        metric["forward_seconds_sum"] = float(sum(row["forward_seconds"] for row in selected))
        metric["maximum_head_relative_delta"] = max(row["capture"].get("maximum_head_relative_delta", 0.0) for row in selected)
        metric["capped_heads_fraction"] = sum(row["capture"].get("capped_head_count", 0) for row in selected) / (len(selected) * 16)
        result.append(metric)
    return result


def bootstrap_weights(strata, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    generator = np.random.default_rng(seed)
    video_weights = np.zeros((repetitions, len(strata)), dtype=np.int64)
    for stratum in sorted(set(strata)):
        positions = np.flatnonzero(np.asarray(strata) == stratum)
        draws = generator.integers(0, len(positions), size=(repetitions, len(positions)))
        for within, position in enumerate(positions):
            video_weights[:, position] = (draws == within).sum(axis=1)
    seed_draws = generator.integers(0, len(SEEDS), size=(repetitions, len(SEEDS)))
    seed_weights = np.stack([(seed_draws == position).sum(axis=1) for position in range(len(SEEDS))], axis=1) / len(SEEDS)
    return video_weights, seed_weights


def paired_statistics(difference, counts, video_weights, seed_weights, metric):
    if metric == "micro_accuracy":
        contributions = difference
        denominator = video_weights @ counts
        point_per_seed = difference.sum(axis=1) / counts.sum()
    else:
        included = counts > 0
        contributions = np.divide(difference, counts[None, :], out=np.zeros_like(difference), where=included[None, :])
        denominator = video_weights @ included.astype(float)
        point_per_seed = contributions.sum(axis=1) / included.sum()
    valid = denominator > 0
    fixed = (video_weights[valid] @ contributions.mean(axis=0)) / denominator[valid]
    crossed = np.einsum("bs,sv,bv->b", seed_weights[valid], contributions, video_weights[valid]) / denominator[valid]
    return {"difference": float(np.mean(point_per_seed)), "per_seed_differences": point_per_seed.tolist(),
            "seed_difference_sample_sd": float(np.std(point_per_seed, ddof=1)),
            "video_bootstrap_95_percentile": np.percentile(fixed, [2.5, 97.5]).tolist() if len(fixed) else None,
            "video_seed_bootstrap_95_percentile": np.percentile(crossed, [2.5, 97.5]).tolist() if len(crossed) else None,
            "valid_replicates": int(valid.sum()), "undefined_replicates": int((~valid).sum())}


def paired_comparisons(rows, output):
    real = [row for row in rows if row["split"] == "test" and row["condition"] == "real"]
    if not real:
        return [], None
    identities = [row for row in real if row["method"] == "identity"]
    videos = sorted({row["video_id"] for row in identities})
    strata = [next(row["question"]["selection_stratum"] for row in identities if row["video_id"] == video) for video in videos]
    require(len(videos) == 16 and sorted(Counter(strata).values()) == [8, 8], "Test bootstrap strata changed")
    video_weights, seed_weights = bootstrap_weights(strata)
    bootstrap_path = output / "bootstrap_draws.npz"
    np.savez_compressed(bootstrap_path, video_weights=video_weights, seed_weights=seed_weights,
                        video_ids=np.asarray(videos), selection_strata=np.asarray(strata), training_seeds=np.asarray(SEEDS))
    comparisons = [(method, "identity") for method in METHODS] + [("spectral_mlp", reference) for reference in ("time_mlp", "spectral_phaseblind", "constant_spectral")]
    indexed = {(row["case_id"], row["method"], row["seed"]): row for row in real}
    result = []
    for subset in SUBSETS:
        cases = [row for row in identities if subset in row["analysis_subsets"]]
        counts = np.asarray([sum(row["video_id"] == video for row in cases) for video in videos], dtype=float)
        for method, reference in comparisons:
            difference = np.zeros((len(SEEDS), len(videos)))
            repairs = []
            regressions = []
            for seed_index, seed in enumerate(SEEDS):
                repaired = regressed = 0
                for case in cases:
                    actual = indexed[(case["case_id"], method, seed)]["correct"]
                    baseline = indexed[(case["case_id"], reference, None if reference == "identity" else seed)]["correct"]
                    difference[seed_index, videos.index(case["video_id"])] += actual - baseline
                    repaired += int(actual == 1 and baseline == 0)
                    regressed += int(actual == 0 and baseline == 1)
                repairs.append(repaired)
                regressions.append(regressed)
            for metric in ("micro_accuracy", "video_macro_accuracy"):
                result.append({"split": "test", "subset": subset, "method": method, "reference": reference, "metric": metric,
                               "primary": subset == "all" and method == "spectral_mlp" and reference == "time_mlp" and metric == "video_macro_accuracy",
                               "source_videos": int((counts > 0).sum()), "cases": len(cases),
                               "per_seed_repairs": repairs, "per_seed_regressions": regressions,
                               **paired_statistics(difference, counts, video_weights, seed_weights, metric)})
    record = {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "percentile": [2.5, 97.5],
              "video_ids": videos, "selection_strata": strata, "training_seeds": list(SEEDS),
              "draws_sha256": digest(bootstrap_path), "same_draws_for_all_methods_and_subsets": True,
              "seed_resampling": "globally crossed with videos, never independently within video",
              "identity": "one fixed baseline, reused algebraically across seed pairings"}
    return result, record


def audit_training(directories):
    output = []
    seeds_seen = set()
    source_hashes = None
    for directory in directories:
        directory = Path(directory).resolve()
        config = read_json(directory / "config.json")
        seed = config["seed"]
        require(seed in SEEDS and seed not in seeds_seen, "Training seed duplicated or changed")
        seeds_seen.add(seed)
        require(config["status"] == "complete" and config["protocol_sha256"] == PROTOCOL_SHA256, "Training provenance incomplete")
        require(config["steps_per_method"] == 200 and config["checkpoint_policy"] == "final_step_200_no_validation_selection", "Checkpoint policy changed")
        fixed_settings = {"batch_real": 6, "batch_static": 2, "optimizer": "AdamW", "learning_rate": 0.001,
                          "weight_decay": 0.0001, "real_kl_weight": 0.1, "static_kl_weight": 1.0,
                          "delta_penalty_weight": 0.001, "gradient_clip_norm": 1.0, "tail_precision": "bfloat16",
                          "sampling": "with_replacement_uniform_over_real_QA_and_over_static_cases_separately",
                          "backbone_parameter_gradients": False, "training_cases": 206}
        require(all(config.get(key) == value for key, value in fixed_settings.items()), "Training configuration differs from frozen settings")
        optimizer_settings = {"lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0001,
                              "amsgrad": False, "maximize": False}
        require(all(config["optimizer_defaults"].get(key) == value for key, value in optimizer_settings.items()), "AdamW defaults differ from frozen settings")
        require(config["data_provenance"]["training_questions_sha256"] == MANIFEST_HASHES["train.questions.jsonl"], "Training data changed")
        require(digest(directory / "source_snapshot" / "protocol.md") == PROTOCOL_SHA256, "Training protocol snapshot changed")
        for name, expected_hash in config["source_sha256"].items():
            require(digest(directory / "source_snapshot" / name) == expected_hash, "Training source snapshot changed")
        if source_hashes is None:
            source_hashes = config["source_sha256"]
        require(config["source_sha256"] == source_hashes, "Training seeds used different implementations")
        summaries = {item["method"]: item for item in config["completed_methods"]}
        require(set(summaries) == set(METHODS) and len(config["completed_methods"]) == 4, "Incomplete training methods")
        paired_schedule = None
        for method in METHODS:
            method_path = directory / method
            summary = summaries[method]
            require(summary["method"] == method and summary["seed"] == seed and summary["steps"] == 200, "Training method summary identity changed")
            require(summary["parameter_count"] == (8 if method == "constant_spectral" else 18584), "Controller parameter count changed")
            for filename, field in (("final.pt", "checkpoint_sha256"), ("optimizer.pt", "optimizer_sha256"),
                                    ("initial.pt", "initial_state_sha256"), ("training.jsonl", "training_log_sha256")):
                require(digest(method_path / filename) == summary[field], "Training artifact changed")
            history = read_rows(method_path / "training.jsonl")
            require([row["step"] for row in history] == list(range(1, 201)), "Training step coverage invalid")
            schedule = [row["case_ids"] for row in history]
            if paired_schedule is None:
                paired_schedule = schedule
            require(schedule == paired_schedule, "Training methods did not share exact sampled batches")
            for row in history:
                require(len(row["case_ids"]) == 8 and all(case.endswith(":real") for case in row["case_ids"][:6]), "Real training batch changed")
                require(all(case.endswith((":static_first", ":static_last")) for case in row["case_ids"][6:]), "Static training batch changed")
                require(all(np.isfinite(row[field]) for field in ("loss", "real_ce", "real_kl", "static_kl", "relative_delta_squared", "gradient_norm_before_clip")), "Nonfinite training log")
                expected_loss = row["real_ce"] + 0.1 * row["real_kl"] + row["static_kl"] + 0.001 * row["relative_delta_squared"]
                require(abs(expected_loss - row["loss"]) <= 1e-5 * max(1.0, abs(expected_loss)), "Training loss composition changed")
            output.append({"directory": str(directory), "method": method, "seed": seed, **summary,
                           "sampling_schedule_sha256": canonical_digest(schedule), "initial_loss": history[0]["loss"],
                           "final_loss": history[-1]["loss"], "maximum_gradient_norm_before_clip": max(row["gradient_norm_before_clip"] for row in history),
                           "config_sha256": digest(directory / "config.json"), "config": config})
    if directories:
        require(seeds_seen == set(SEEDS), "All three training seeds are required together")
    return output


def resource_ledger(directories):
    output = []
    for directory in sorted({Path(path).resolve() for path in directories}):
        entry = {"directory": str(directory), "artifact_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())}
        for filename, key in (("resource_summary.json", "resources"), ("resource_limits.json", "limits"),
                              ("resource_stop.json", "stop"), ("config.json", "config")):
            if (directory / filename).exists():
                entry[key] = read_json(directory / filename)
                entry[filename + "_sha256"] = digest(directory / filename)
        if (directory / "replay_parity.json").exists():
            parity = read_json(directory / "replay_parity.json")
            entry["replay_parity"] = {"records": len(parity), "prediction_changes": sum(row["prediction_changed"] for row in parity),
                                       "maximum_option_logit_difference": max(row["maximum_option_logit_difference"] for row in parity),
                                       "raw_sha256": digest(directory / "replay_parity.json")}
        output.append(entry)
    return output


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}"


def interval(values):
    return "undefined" if values is None else f"[{100 * values[0]:.2f}, {100 * values[1]:.2f}]"


def write_tables(output, metrics, comparisons, resources, report):
    indexed = {(row["split"], row["subset"], row["method"], row["seed"]): row for row in metrics}
    lines = ["# 学习式控制器：独立CPU审计与完整结果", "",
             "全部准确率来自完整普通模型forward；从原始全词表NPZ独立复算。identity只有一个实际基线。",
             "三个训练seed固定为20261101/20261102/20261103；以下准确率及差值单位为百分比/百分点。",
             "本地test来自官方valid中16个新源视频，14.362–34.910秒；不能称官方隐藏测试或长期记忆结论。", ""]
    main_tex = []
    for split in ("test", "validation"):
        if not any(row["split"] == split for row in metrics):
            continue
        section_start = len(lines)
        counts = {subset: indexed[(split, subset, "identity", None)] for subset in SUBSETS}
        lines.extend([f"## {split}：全部方法和seed", "",
                      "; ".join(f"{subset}: {counts[subset]['source_videos']}源/{counts[subset]['cases']}题" for subset in SUBSETS) + "。标签子集有重叠。", "",
                      "| 方法 | seed | all micro | all macro | temporal micro | temporal macro | memory micro | memory macro | descriptive micro | descriptive macro |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        tex = ["\\begin{tabular}{lr*{8}{r}}", "\\toprule",
               "Method & Seed & All $\\mu$ & All $M$ & Temp. $\\mu$ & Temp. $M$ & Mem. $\\mu$ & Mem. $M$ & Desc. $\\mu$ & Desc. $M$ \\\\", "\\midrule"]
        for method, seed in PAIRS:
            values = [percentage(indexed[(split, subset, method, seed)][metric]) for subset in SUBSETS for metric in ("micro_accuracy", "video_macro_accuracy")]
            lines.append("| " + " | ".join([method, "—" if seed is None else str(seed), *values]) + " |")
            escaped_method = method.replace("_", "\\_")
            tex.append(" & ".join([escaped_method, "--" if seed is None else str(seed), *values]) + " \\\\")
        tex.extend(["\\bottomrule", "\\end{tabular}", ""])
        (output / f"{split}_main.tex").write_text("\n".join(tex), encoding="utf-8")
        if split == "test":
            main_tex = tex
        lines.extend(["", "### seed均值与样本标准差", "", "| 方法 | all micro | all macro | temporal macro | memory macro | descriptive macro |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for method in ("identity", *METHODS):
            cells = []
            for subset, metric in (("all", "micro_accuracy"), *((subset, "video_macro_accuracy") for subset in SUBSETS)):
                values = [indexed[(split, subset, method, seed)][metric] for seed in ([None] if method == "identity" else SEEDS)]
                cells.append(percentage(np.mean(values)) + (" (单基线)" if method == "identity" else f" ± {100 * np.std(values, ddof=1):.2f}"))
            lines.append("| " + " | ".join([method, *cells]) + " |")
        (output / f"{split}_main.md").write_text("\n".join(lines[section_start:]) + "\n", encoding="utf-8")
        if split == "test":
            (output / "main_table.md").write_text("\n".join(lines[section_start:]) + "\n", encoding="utf-8")
        lines.extend(["", f"### {split}：重复帧无金标保持诊断", "", "| 方法 | seed | 源/条件 | KL均值 | top1改变 | 平均绝对margin变化 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for method, seed in PAIRS:
            item = indexed[(split, "static_no_gold", method, seed)]
            lines.append(f"| {method} | {seed if seed is not None else '—'} | {item['source_videos']}/{item['cases']} | {item['option_KL_mean']:.6g} | {item['top1_changed_count']}/{item['cases']} | {item['absolute_top_two_margin_change_mean']:.6g} |")
        lines.extend(["", "这些重复帧条件不计算正确率，也不替代有金标的seen静态8题门槛。", ""])
    if comparisons:
        primary = next(row for row in comparisons if row["primary"])
        lines.extend(["## 预声明主要比较", "", f"全部74题video-macro，spectral−time = {100 * primary['difference']:.2f}百分点；固定seed的视频bootstrap 95%区间 {interval(primary['video_bootstrap_95_percentile'])}；video×seed区间 {interval(primary['video_seed_bootstrap_95_percentile'])}。", "",
                      "每次按两个选源层各抽8个源，所有方法共用抽样。seed全局抽取；identity不伪造独立seed。区间是小样本描述，不能保证总体覆盖率。", "",
                      "| 子集 | 方法−参考 | 指标 | 差值 | 视频95%区间 | 视频×seed95%区间 | 有效重复 |", "| --- | --- | --- | ---: | --- | --- | ---: |"])
        comparison_tex = ["\\begin{tabular}{lllrll}", "\\toprule", "Subset & Comparison & Metric & Difference & Video CI & Video $\\times$ seed CI \\\\", "\\midrule"]
        for item in comparisons:
            values = [item["subset"], item["method"] + " - " + item["reference"], item["metric"], f"{100 * item['difference']:.2f}", interval(item["video_bootstrap_95_percentile"]), interval(item["video_seed_bootstrap_95_percentile"])]
            lines.append("| " + " | ".join([*values, str(item["valid_replicates"])]) + " |")
            comparison_tex.append(" & ".join(value.replace("_", "\\_") for value in values) + " \\\\")
        comparison_tex.extend(["\\bottomrule", "\\end{tabular}", ""])
        (output / "paired_comparisons.tex").write_text("\n".join(comparison_tex), encoding="utf-8")
    endpoint = [row for row in metrics if row["split"] == "endpoint"]
    if endpoint:
        lines.extend(["", "## 已看单源endpoint附录", "", "| 方法 | seed | 静态正确 | mixed正确 | 静态8/8门槛 |", "| --- | ---: | ---: | ---: | --- |"])
        for method, seed in PAIRS:
            static = indexed[("endpoint", "static_readability", method, seed)]
            mixed = indexed[("endpoint", "mixed", method, seed)]
            lines.append(f"| {method} | {seed if seed is not None else '—'} | {static['correct']}/8 | {mixed['correct']}/24 | {static['static_8_of_8_gate_passed']} |")
        lines.extend(["", "此处32条件来自一个已看源，不加入新test样本数或bootstrap。", ""])
    lines.extend(["", "## 资源与复现", "", "| 运行目录 | 状态 | 过程秒数 | CUDA allocated MiB | sampled RSS MiB | sampled GPU °C |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for resource in resources:
        summary = resource.get("resources", {})
        config = resource.get("config", {})
        def rounded(value, divisor=1):
            return "—" if value is None else f"{value / divisor:.2f}"
        lines.append("| " + " | ".join([Path(resource["directory"]).name, config.get("status", summary.get("status", "unknown")), rounded(summary.get("elapsed_seconds")), rounded(config.get("peak_cuda_allocated_bytes"), 1024**2), rounded(summary.get("peak_sampled_rss_bytes"), 1024**2), rounded(summary.get("peak_sampled_gpu_temperature_c"))]) + " |")
    lines.extend(["", "资源遥测为两秒采样，可能遗漏短峰值；CUDA allocator峰值另列。资源清单仅覆盖传入目录，失败proof及旧实验是否齐全由scope字段明确标识。", "",
                  f"审计{report['records']}条记录、{report['cases']}个完整13方法/seed条件；原始文件、源码、checkpoint、抽样表与分析器SHA见report.json及audit_runs.json。", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if main_tex:
        (output / "main_table.tex").write_text("\n".join(main_tex), encoding="utf-8")


def synthetic_archive_check(directory):
    question = {"options": ["one", "two", "three"], "tags": ["sequencing"], "area": "physics", "reasoning": "descriptive"}
    expected = {"question": question, "video_id": "synthetic_only", "condition": "real", "gold": 2,
                "options": question["options"], "frame_sha256": ["synthetic"] * 16}
    native = np.asarray([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    arrays = {"identity_logits": native, "zero_logits": native, "restored_logits": native,
              "option_token_ids": np.asarray([0, 1, 2])}
    for method, seed in PAIRS[1:]:
        arrays[f"{method}_{seed}_logits"] = native
    archive = directory / "synthetic.npz"
    np.savez_compressed(archive, **arrays)
    checkpoints = {pair: "synthetic_checkpoint" for pair in PAIRS[1:]}
    rows = []
    for method, seed in PAIRS:
        capture = {"delta_context_l2": 0.0, "injection_layer": 27, "maximum_head_relative_delta": 0.0,
                   "head_gains": [[0.0] * 8 for _ in range(16)], "maximum_absolute_gain": 0.0}
        rows.append({"case_id": "synthetic:real", "query_id": "synthetic", "video_id": "synthetic_only", "split": "validation",
                     "condition": "real", "question": question, "gold_answer_index": 2, "supervised": True,
                     "frame_sha256": expected["frame_sha256"], "input_tensor_hashes": {}, "option_token_ids": [0, 1, 2], "batch": {},
                     "archive_file": archive.name, "archive_sha256": digest(archive), "identity_exact": True, "restoration_exact": True,
                     "method": method, "seed": seed, "option_logits": [0.0, 1.0, 2.0], "predicted_answer_index": 2,
                     "baseline_predicted_answer_index": 2, "correct": 1, "baseline_correct": 1,
                     "baseline_answer_changed": False, "option_KL_from_baseline": 0.0, "capture": capture,
                     "checkpoint_sha256": None if method == "identity" else "synthetic_checkpoint", "forward_seconds": 0.0})
    audited = audit_archived_case(directory, rows, expected, checkpoints)
    require(len(audited) == 13 and all(row["correct"] == 1 for row in audited), "Valid synthetic archive rejected")
    malformed = [{**row} for row in rows]
    malformed[1]["correct"] = 0
    for invalid in (malformed, rows[:-1], [rows[0], *rows[1:-1], rows[1]]):
        try:
            audit_archived_case(directory, invalid, expected, checkpoints)
        except ValueError:
            pass
        else:
            raise ValueError("Corrupt or incomplete synthetic archive accepted")


def self_check():
    strata = ["memory"] * 8 + ["sequencing"] * 8
    videos, seeds = bootstrap_weights(strata)
    require(np.all(videos[:, :8].sum(axis=1) == 8) and np.all(videos[:, 8:].sum(axis=1) == 8), "Bootstrap lost source strata")
    require(np.allclose(seeds.sum(axis=1), 1), "Global seed weights invalid")
    counts = np.ones(16)
    difference = np.ones((3, 16))
    for metric in ("micro_accuracy", "video_macro_accuracy"):
        result = paired_statistics(difference, counts, videos, seeds, metric)
        require(result["difference"] == 1 and np.allclose(result["video_seed_bootstrap_95_percentile"], [1, 1]), "Constant paired effect bootstrap failed")
        zero = paired_statistics(np.zeros_like(difference), counts, videos, seeds, metric)
        require(zero["video_bootstrap_95_percentile"] == [0.0, 0.0], "Identity comparison not exactly zero")
    sparse_counts = np.zeros(16)
    sparse_counts[0] = 1
    sparse_difference = np.zeros((3, 16))
    sparse_difference[:, 0] = 1
    sparse = paired_statistics(sparse_difference, sparse_counts, videos, seeds, "video_macro_accuracy")
    require(sparse["undefined_replicates"] > 0 and sparse["valid_replicates"] + sparse["undefined_replicates"] == BOOTSTRAP_REPETITIONS, "Undefined subsets were not retained")
    reversed_result = paired_statistics(-difference, counts, videos, seeds, "micro_accuracy")
    require(reversed_result["difference"] == -1, "Paired effect direction reversed")
    unequal_counts = np.asarray([1.0, 9.0])
    unequal_difference = np.asarray([[1.0, 0.0]] * 3)
    unequal_weights = np.ones((4, 2))
    equal_seed_weights = np.full((4, 3), 1 / 3)
    micro = paired_statistics(unequal_difference, unequal_counts, unequal_weights, equal_seed_weights, "micro_accuracy")
    macro = paired_statistics(unequal_difference, unequal_counts, unequal_weights, equal_seed_weights, "video_macro_accuracy")
    require(np.isclose(micro["difference"], 0.1) and np.isclose(macro["difference"], 0.5), "Micro and video-macro weights were conflated")
    with tempfile.TemporaryDirectory(prefix="controller_analysis_selfcheck_") as directory:
        temporary = Path(directory)
        synthetic_archive_check(temporary)
        metrics = []
        for method, seed in PAIRS:
            for subset in SUBSETS:
                metrics.append({"split": "test", "subset": subset, "method": method, "seed": seed,
                                "source_videos": 16, "cases": 74, "micro_accuracy": 0.5, "video_macro_accuracy": 0.5})
            metrics.append({"split": "test", "subset": "static_no_gold", "method": method, "seed": seed,
                            "source_videos": 16, "cases": 32, "option_KL_mean": 0.0, "top1_changed_count": 0,
                            "absolute_top_two_margin_change_mean": 0.0})
        write_tables(temporary, metrics, [], [], {"records": 13, "cases": 1})
        require((temporary / "main_table.md").exists() and (temporary / "main_table.tex").exists(), "Paper table generation failed")
        tex = (temporary / "main_table.tex").read_text(encoding="utf-8")
        require(tex.count(" \\\\") == 14, "LaTeX table lost a seed row or row terminator")
    return {"status": "passed", "checks": ["8+8 stratified source resampling", "global paired seed weights", "constant effect", "identity zero", "undefined subset replicates", "effect direction", "micro versus video-macro weighting", "synthetic raw-logit audit", "reject corrupt accuracy/missing seed/duplicate seed", "Markdown and LaTeX 13-row table smoke"], "gpu_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dirs", type=Path, nargs="+", default=[])
    parser.add_argument("--training-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--resource-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--manifest-root", type=Path, default=Path("data/internal_controller_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/learned_controller_protocol_v1.md"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        print(json.dumps(self_check(), ensure_ascii=False))
        return
    if not arguments.evaluation_dirs or arguments.output_dir is None:
        parser.error("Pass completed evaluation directories and a new output directory")
    require(not arguments.output_dir.exists(), "Analysis output must be new")
    rows, runs = audit_evaluations(arguments.evaluation_dirs, arguments.manifest_root, arguments.protocol)
    training = audit_training(arguments.training_dirs)
    if training:
        trained_checkpoints = {(row["method"], row["seed"]): row["checkpoint_sha256"] for row in training}
        evaluated_checkpoints = {(row["method"], row["seed"]): row["checkpoint_sha256"] for row in runs[0]["config"]["checkpoints"]}
        require(trained_checkpoints == evaluated_checkpoints, "Training audit does not describe the evaluated checkpoints")
    arguments.output_dir.mkdir(parents=True)
    metrics = metric_rows(rows)
    comparisons, bootstrap = paired_comparisons(rows, arguments.output_dir)
    resources = resource_ledger([*arguments.evaluation_dirs, *arguments.training_dirs, *arguments.resource_dirs])
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "records": len(rows),
              "cases": len({(row["split"], row["case_id"]) for row in rows}), "splits": sorted({row["split"] for row in rows}),
              "protocol_sha256": PROTOCOL_SHA256, "analyzer_sha256": digest(__file__), "numpy_version": np.__version__,
              "python_version": sys.version, "primary_comparison": next((row for row in comparisons if row["primary"]), None),
              "bootstrap": bootstrap, "training_audit_complete": bool(training), "resource_scope": "Explicit input directories only; no claim of complete historical research cost",
              "cuda_or_model_execution": False, "limitations": ["16 source clusters and three training seeds", "overlapping diagnostic subsets", "official valid split with local experimental roles", "short videos, not a long-term-memory demonstration"]}
    write_rows(arguments.output_dir / "audited_records.jsonl", rows)
    write_rows(arguments.output_dir / "per_seed_metrics.jsonl", metrics)
    write_rows(arguments.output_dir / "paired_comparisons.jsonl", comparisons)
    write_json(arguments.output_dir / "audit_runs.json", runs)
    write_json(arguments.output_dir / "training_audit.json", training)
    write_json(arguments.output_dir / "resource_ledger.json", resources)
    write_tables(arguments.output_dir, metrics, comparisons, resources, report)
    report["output_sha256"] = {path.name: digest(path) for path in arguments.output_dir.iterdir() if path.is_file()}
    write_json(arguments.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
