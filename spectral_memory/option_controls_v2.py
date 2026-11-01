"""Fit a train-only letter-logit calibration control using CPU NumPy Adam."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .report_learned_controller import MANIFEST_HASHES, PROTOCOL_SHA256, digest, read_json, read_rows, require, write_json, write_rows


STEPS = 200
LEARNING_RATE = 0.03
L2_COEFFICIENT = 0.01
BETAS = (0.9, 0.999)
EPSILON = 1e-8


def normalized_log_probabilities(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def bias_from_parameters(parameters):
    return np.asarray([parameters[0], parameters[1], -parameters.sum()], dtype=np.float64)


def objective(parameters, logits, gold):
    bias = bias_from_parameters(parameters)
    log_probabilities = normalized_log_probabilities(logits + bias)
    cross_entropy = float(-log_probabilities[np.arange(len(gold)), gold].mean())
    penalty = float(L2_COEFFICIENT * np.square(bias).sum())
    derivative = np.exp(log_probabilities).mean(axis=0) - np.bincount(gold, minlength=3) / len(gold)
    derivative += 2 * L2_COEFFICIENT * bias
    gradient = np.asarray([derivative[0] - derivative[2], derivative[1] - derivative[2]])
    return cross_entropy + penalty, gradient, cross_entropy, penalty


def fit(logits, gold):
    parameters = np.zeros(2, dtype=np.float64)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)
    history = []
    for step in range(STEPS + 1):
        loss, gradient, cross_entropy, penalty = objective(parameters, logits, gold)
        require(np.isfinite(loss) and np.isfinite(gradient).all(), "Calibration optimization became nonfinite")
        history.append({"step": step, "loss": loss, "train_nll": cross_entropy, "l2_penalty": penalty,
                        "gradient_l2": float(np.linalg.norm(gradient)), "parameters": parameters.tolist(),
                        "letter_bias": bias_from_parameters(parameters).tolist()})
        if step == STEPS:
            break
        first_moment = BETAS[0] * first_moment + (1 - BETAS[0]) * gradient
        second_moment = BETAS[1] * second_moment + (1 - BETAS[1]) * np.square(gradient)
        corrected_first = first_moment / (1 - BETAS[0] ** (step + 1))
        corrected_second = second_moment / (1 - BETAS[1] ** (step + 1))
        parameters -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + EPSILON)
    return bias_from_parameters(parameters), history


def load_old_inputs(arguments):
    manifests = {}
    manifest_provenance = {}
    for split in ("train", "validation", "test"):
        path = arguments.questions_root / f"{split}.questions.jsonl"
        require(digest(path) == MANIFEST_HASHES[path.name], "Only the frozen, already-seen V1 question sets are allowed")
        manifests[split] = {row["query_id"]: row for row in read_rows(path)}
        manifest_provenance[split] = {"path": str(path.resolve()), "sha256": digest(path)}
    splits = {"train": [], "validation": [], "old_test_seen": []}
    source_records = []
    for directory in arguments.train_cache_dirs:
        config = read_json(directory / "config.json")
        records_path = directory / "records.jsonl"
        require(config["status"] == "complete" and digest(records_path) == config["records_sha256"], "Native training cache provenance failed")
        source_records.append({"path": str(records_path.resolve()), "sha256": digest(records_path)})
        for row in read_rows(records_path):
            if not row["supervised"]:
                continue
            require(row["condition"] == "real" and row["question"] == manifests["train"].get(row["query_id"]), "Unknown training question or changed label")
            require(row["gold_answer_index"] == row["question"]["answer_index"], "Training gold changed")
            splits["train"].append(row)
    analysis = read_json(arguments.analysis_dir / "report.json")
    audited_path = arguments.analysis_dir / "audited_records.jsonl"
    require(analysis["status"] == "complete" and analysis["protocol_sha256"] == PROTOCOL_SHA256, "Old full-forward audit is not complete")
    require(digest(audited_path) == analysis["output_sha256"]["audited_records.jsonl"], "Old full-forward records changed")
    source_records.append({"path": str(audited_path.resolve()), "sha256": digest(audited_path)})
    for row in read_rows(audited_path):
        if row["split"] not in {"validation", "test"} or row["method"] != "identity" or row["condition"] != "real":
            continue
        require(row["question"] == manifests[row["split"]].get(row["query_id"]), "An unseen or changed query entered the calibration diagnostic")
        require(row["gold_answer_index"] == row["question"]["answer_index"], "Development gold changed")
        splits["old_test_seen" if row["split"] == "test" else "validation"].append(row)
    for split, rows in splits.items():
        expected = manifests["test" if split == "old_test_seen" else split]
        require(len(rows) == len(expected) and {row["query_id"] for row in rows} == set(expected), "Old split coverage incomplete or duplicated")
        require(all(len(row["option_logits"]) == 3 and len(row["question"]["options"]) == 3 for row in rows), "Three original options are required")
        rows.sort(key=lambda row: (row["video_id"], row["query_id"]))
    return splits, {"frozen_question_manifests": manifest_provenance, "native_score_sources": source_records}


def score(split, rows, bias):
    original = np.asarray([row["option_logits"] for row in rows], dtype=np.float64)
    gold = np.asarray([row["gold_answer_index"] for row in rows], dtype=np.int64)
    require(np.isfinite(original).all(), "Native scores are nonfinite")
    records = []
    metrics = []
    for method, logits in (("native", original), ("train_only_letter_calibration", original + bias)):
        log_probabilities = normalized_log_probabilities(logits)
        predictions = logits.argmax(axis=-1)
        correct = predictions == gold
        nll = -log_probabilities[np.arange(len(gold)), gold]
        source_positions = defaultdict(list)
        for position, row in enumerate(rows):
            source_positions[row["video_id"]].append(position)
            records.append({"split": split, "method": method, "query_id": row["query_id"], "video_id": row["video_id"],
                            "options": row["question"]["options"], "gold_answer_index": int(gold[position]),
                            "option_logits": logits[position].tolist(), "native_option_logits": original[position].tolist(),
                            "predicted_answer_index": int(predictions[position]), "correct": int(correct[position]),
                            "nll": float(nll[position]), "maximum_tie_indices": np.flatnonzero(logits[position] == logits[position].max()).tolist()})
        metrics.append({"split": split, "method": method, "cases": len(rows), "source_videos": len(source_positions),
                        "correct": int(correct.sum()), "micro_accuracy": float(correct.mean()),
                        "source_macro_accuracy": float(np.mean([correct[positions].mean() for positions in source_positions.values()])),
                        "mean_nll": float(nll.mean()), "source_macro_nll": float(np.mean([nll[positions].mean() for positions in source_positions.values()])),
                        "gold_counts_ABC": np.bincount(gold, minlength=3).tolist(),
                        "prediction_counts_ABC": np.bincount(predictions, minlength=3).tolist(),
                        "exact_maximum_ties": int(sum(np.count_nonzero(row == row.max()) > 1 for row in logits))})
    native_prediction = original.argmax(axis=-1)
    calibrated_prediction = (original + bias).argmax(axis=-1)
    comparison = {"split": split, "repairs": int(((native_prediction != gold) & (calibrated_prediction == gold)).sum()),
                  "regressions": int(((native_prediction == gold) & (calibrated_prediction != gold)).sum()),
                  "prediction_changes": int((native_prediction != calibrated_prediction).sum()),
                  "micro_accuracy_difference": metrics[1]["micro_accuracy"] - metrics[0]["micro_accuracy"],
                  "source_macro_accuracy_difference": metrics[1]["source_macro_accuracy"] - metrics[0]["source_macro_accuracy"],
                  "mean_nll_difference": metrics[1]["mean_nll"] - metrics[0]["mean_nll"]}
    return metrics, records, comparison


def run(arguments):
    require(not arguments.output_dir.exists(), "Calibration output must be new")
    splits, provenance = load_old_inputs(arguments)
    train = splits["train"]
    logits = np.asarray([row["option_logits"] for row in train], dtype=np.float64)
    gold = np.asarray([row["gold_answer_index"] for row in train], dtype=np.int64)
    bias, history = fit(logits, gold)
    require(abs(bias.sum()) <= 1e-12 and len(history) == 201, "Calibration fit violates the fixed two-parameter 200-step protocol")
    all_metrics = []
    all_records = []
    comparisons = []
    for split, rows in splits.items():
        metrics, records, comparison = score(split, rows, bias)
        all_metrics.extend(metrics)
        all_records.extend(records)
        comparisons.append(comparison)
    arguments.output_dir.mkdir(parents=True)
    write_rows(arguments.output_dir / "optimization.jsonl", history)
    write_rows(arguments.output_dir / "case_scores.jsonl", all_records)
    write_rows(arguments.output_dir / "metrics.jsonl", all_metrics)
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "diagnostic scoring calibration, not internal-model or memory improvement",
              "source_sha256": digest(__file__), "provenance": provenance, "fit_data": "142 frozen train QA only",
              "optimization": {"optimizer": "NumPy float64 Adam", "steps": STEPS, "learning_rate": LEARNING_RATE,
                               "betas": BETAS, "epsilon": EPSILON, "l2_coefficient_on_three_biases": L2_COEFFICIENT,
                               "batch": "full training set", "initial_parameters": [0.0, 0.0], "trainable_parameters": 2,
                               "constraint": "bias_C = -bias_A - bias_B", "checkpoint_selection": "fixed final step 200"},
              "letter_bias_ABC": bias.tolist(), "metrics": all_metrics, "comparisons": comparisons,
              "model_forward_executed": False, "gpu_used": False, "new_confirmation_sources_read": False,
              "interpretation": "Validation and old_test_seen are development diagnostics. No source selection, heldout tuning, memory mechanism, or option-permutation robustness is established."}
    lines = ["# V2字母logit校准控制（仅CPU）", "", "此对照只校准已有三个选项分数，不修改模型，不构成记忆方法。只用142条旧训练题拟合两个自由参数；validation和旧test均作为已看开发诊断。没有读取新32源的模型分数。", "",
             "固定全批Adam 200步，lr=0.03，betas=(0.9,0.999)，eps=1e-8；目标为训练平均NLL + 0.01×三个bias的平方和，bias之和为0。初始全0，不挑checkpoint。", "",
             f"最终A/B/C bias：`{bias.tolist()}`。", "",
             "| split | 方法 | 正确/题数 | micro % | source macro % | NLL | source macro NLL |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in all_metrics:
        lines.append(f"| {item['split']} | {item['method']} | {item['correct']}/{item['cases']} | {100*item['micro_accuracy']:.2f} | {100*item['source_macro_accuracy']:.2f} | {item['mean_nll']:.6f} | {item['source_macro_nll']:.6f} |")
    lines.extend(["", "| split | 修复 | 回退 | 预测改变 | source macro差（百分点） |", "| --- | ---: | ---: | ---: | ---: |"])
    for item in comparisons:
        lines.append(f"| {item['split']} | {item['repairs']} | {item['regressions']} | {item['prediction_changes']} | {100*item['source_macro_accuracy_difference']:.2f} |")
    lines.extend(["", "选项位置偏差仍需ABC/BCA/CAB循环换位、按答案内容还原后检验。此处分数校准不能取代真实内部方法，也不能把旧test重新称为未见确认集。", ""])
    (arguments.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    report["output_sha256"] = {path.name: digest(path) for path in arguments.output_dir.iterdir() if path.is_file()}
    write_json(arguments.output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "letter_bias_ABC": report["letter_bias_ABC"], "metrics": all_metrics,
                      "comparisons": comparisons, "source_sha256": report["source_sha256"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache-dirs", type=Path, nargs="+", default=[Path("outputs/controller_train_cache_v1") / f"chunk{index:02d}" for index in range(4)])
    parser.add_argument("--analysis-dir", type=Path, default=Path("outputs/controller_analysis_v1"))
    parser.add_argument("--questions-root", type=Path, default=Path("data/internal_controller_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2_option_controls"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
