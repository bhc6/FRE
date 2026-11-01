"""Audit fixed-input intervention records on CPU, including full-vocabulary controls."""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np

PRIMARY = "middle13_high_up"
STRUCTURAL = ("late27_zero", "late27_temporal_up", "middle13_identity")
MATCHED = ("middle13_temporal_matched_up", "middle13_phase_control_up", "middle13_attention_matched_up")
VIDEOS = {"video_10993", "video_1580", "video_2564", "video_3501", "video_6860", "video_8241", "video_874", "video_9253"}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporal(question):
    return bool(set(question["tags"]) & {"change detection", "sequencing"} or re.search(
        r"\b(before|after|first|last|beginning|end|order|changed|change|remember)\b",
        question["question"], flags=re.IGNORECASE))


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    require(array.size > 0 and np.isfinite(array).all(), "Invalid summary")
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def summarize(rows, reference):
    scores, differences, margin_differences = defaultdict(list), defaultdict(list), defaultdict(list)
    corrections, harms, changes, deltas = [], [], [], []
    for row in rows:
        before = reference[row["query_id"]]
        delta, margin_delta = row["correct"] - before["correct"], row["gold_margin"] - before["gold_margin"]
        scores[row["video_id"]].append(row["correct"])
        differences[row["video_id"]].append(delta)
        margin_differences[row["video_id"]].append(margin_delta)
        deltas.append(margin_delta)
        detail = {"query_id": row["query_id"], "video_id": row["video_id"],
                  "before_index": before["predicted_answer_index"], "after_index": row["predicted_answer_index"],
                  "gold_index": row["gold_answer_index"], "before_correct": before["correct"],
                  "after_correct": row["correct"], "before_margin": before["gold_margin"],
                  "after_margin": row["gold_margin"], "margin_delta": margin_delta}
        if delta > 0:
            corrections.append(detail)
        elif delta < 0:
            harms.append(detail)
        if row["predicted_answer_index"] != before["predicted_answer_index"]:
            changes.append(detail)
    return {
        "questions": len(rows), "videos": len(scores), "correct": sum(row["correct"] for row in rows),
        "accuracy_micro": statistics.mean(row["correct"] for row in rows),
        "accuracy_video_macro": statistics.mean(statistics.mean(values) for values in scores.values()),
        "accuracy_micro_delta": (len(corrections) - len(harms)) / len(rows),
        "accuracy_video_macro_delta": statistics.mean(statistics.mean(values) for values in differences.values()),
        "gold_margin_delta": describe(deltas),
        "gold_margin_video_macro_delta": statistics.mean(statistics.mean(values) for values in margin_differences.values()),
        "corrected_count": len(corrections), "harmed_count": len(harms), "net_corrected": len(corrections) - len(harms),
        "corrected_videos": sorted({row["video_id"] for row in corrections}),
        "harmed_videos": sorted({row["video_id"] for row in harms}),
        "positive_net_videos": sorted(name for name, values in differences.items() if sum(values) > 0),
        "per_video_net_corrected": {name: sum(values) for name, values in sorted(differences.items())},
        "corrections": corrections, "harms": harms, "prediction_changes": changes}


def audit(run, root):
    config, status = read_json(run / "config.json"), read_json(run / "status.json")
    require(config["stage"] == "qa" and config["status"] == status["status"] == "complete", "QA run incomplete")
    require(config["completed_records"] == status["completed"] == 540, "Expected 540 completed records")
    require(config["primary_candidate"] == PRIMARY, "Frozen primary changed")
    source_path, protocol_path = run / "source_snapshot/causal_intervention.py", run / "source_snapshot/protocol.md"
    require(digest(source_path) == config["source_sha256"], "Frozen source hash mismatch")
    require(digest(protocol_path) == config["protocol_sha256"], "Frozen protocol hash mismatch")
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    frozen = next(ast.literal_eval(node.value) for node in syntax.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "CONDITIONS" for target in node.targets))
    require(config["conditions"] == [list(row) for row in frozen], "Source/config conditions differ")
    conditions = [row[0] for row in frozen]
    require(len(conditions) == len(set(conditions)) == 15, "Expected 15 unique conditions")
    question_path = root / "data/perception_test_sample/questions.jsonl"
    require(digest(question_path) == config["questions_sha256"], "Questions hash mismatch")
    questions = read_rows(question_path)
    require(len(questions) == 36 and {row["video_id"] for row in questions} == VIDEOS, "Question/video coverage differs")
    official = {row["query_id"]: row for row in questions}
    require(len(official) == 36 and set(config["inputs"]) == VIDEOS, "Duplicate query/input video mismatch")
    for video in sorted(VIDEOS):
        folder = root / "outputs" / ("ffn_qwen3_dev16_v1" if video == "video_10993" else f"ffn_qwen3_dev16_{video}")
        provenance = config["inputs"][video]
        require(digest(folder / "probe.json") == provenance["probe_sha256"], "Probe hash changed")
        require(digest(folder / "frames.manifest.jsonl") == provenance["frames_manifest_sha256"], "Manifest changed")
        manifest = read_json(folder / "frames.manifest.jsonl")
        require(manifest["video_id"] == video and len(manifest["frames"]) == 16, "Frame identity/count mismatch")
        require([digest(Path(path)) for path in manifest["frames"]] == provenance["frame_sha256"], "Frame content mismatch")
    records = read_rows(run / "records.jsonl")
    require(len(records) == 540, "Records disagree with completion marker")
    indexed = defaultdict(dict)
    for row in records:
        query, condition = row["query_id"], row["condition"]
        require(query in official and condition in conditions and condition not in indexed[query], "Unknown/duplicate condition")
        indexed[query][condition] = row
        question = official[query]
        for recorded, original in (("video_id", "video_id"), ("query_text", "question"), ("options", "options"),
                                   ("area", "area"), ("tags", "tags"), ("gold_answer_index", "answer_index")):
            require(row[recorded] == question[original], f"Official mismatch: {query}/{recorded}")
        require(row["temporal_state"] == temporal(question), "Temporal subset mismatch")
        definition = next(value for value in frozen if value[0] == condition)
        require([row[key] for key in ("layer", "branch", "mode", "gain", "norm_matched")] == list(definition[1:]), "Condition metadata differs")
        logits = np.asarray(row["option_logits"], dtype=np.float64)
        require(logits.shape == (len(question["options"]),) and np.isfinite(logits).all(), "Invalid option logits")
        predicted, gold = int(logits.argmax()), question["answer_index"]
        require(predicted == row["predicted_answer_index"] and row["correct"] == int(predicted == gold), "Argmax/reward mismatch")
        require(float(logits[gold] - np.max(np.delete(logits, gold))) == row["gold_margin"], "Margin mismatch")
        require(len(set(row["option_token_ids"])) == len(logits), "Option token IDs not unique")
        require(row["visual_tokens"] <= 512 and row["input_tokens"] <= 1024, "Input budget exceeded")
        shape = row["intervention"]["selected_shape"]
        require(len(shape) == 3 and shape[0] == 8 and shape[0] * shape[1] == row["visual_tokens"], "Unexpected visual-token shape")
        require(int(np.prod(shape)) == row["intervention"]["selected_elements"], "Selected element count mismatch")
    require(set(indexed) == set(official), "Missing questions")
    structural_checks = []
    for query, group in sorted(indexed.items()):
        require(set(group) == set(conditions), "Missing per-query condition")
        for row in group.values():
            for key in ("prompt_sha256", "query_text", "options", "option_token_ids", "input_tokens", "visual_tokens"):
                require(row[key] == group["baseline"][key], "Within-query input mismatch")
        archive = run / "structural_logits" / (hashlib.sha256(query.encode()).hexdigest()[:20] + ".npz")
        with np.load(archive, allow_pickle=False) as arrays:
            require(set(arrays.files) == {"baseline", *STRUCTURAL}, "Archive conditions differ")
            baseline = arrays["baseline"]
            require(baseline.ndim == 1 and baseline.dtype == np.float32 and np.isfinite(baseline).all(), "Invalid full logits")
            for condition in ("baseline", *STRUCTURAL):
                values, row = arrays[condition], group[condition]
                require(values.dtype == baseline.dtype and np.array_equal(values, baseline), "Structural full logits differ")
                require(hashlib.sha256(values.tobytes()).hexdigest() == row["full_logits_sha256"], "Full logits hash mismatch")
                require(row["full_logits_identical_to_baseline"] and row["full_logits_max_abs_change"] == 0, "Structural metadata differs")
                require(np.array_equal(values[row["option_token_ids"]], row["option_logits"]), "Option/full logits mismatch")
            structural_checks.append({"query_id": query, "archive_sha256": digest(archive),
                                      "vocabulary_size": int(baseline.size), "exact_controls": list(STRUCTURAL)})
    reference = {query: rows["baseline"] for query, rows in indexed.items()}
    by_condition = {condition: [indexed[query][condition] for query in sorted(indexed)] for condition in conditions}
    summaries = {condition: {"all": summarize(rows, reference),
                             "temporal_state": summarize([row for row in rows if row["temporal_state"]], reference)}
                 for condition, rows in by_condition.items()}
    comparisons, norms = {}, {}
    for control in MATCHED:
        comparator = {row["query_id"]: row for row in by_condition[control]}
        comparisons[control] = {"all": summarize(by_condition[PRIMARY], comparator),
                                "temporal_state": summarize([row for row in by_condition[PRIMARY] if row["temporal_state"]], comparator)}
        values = []
        for query, group in sorted(indexed.items()):
            primary, comparison = group[PRIMARY]["intervention"], group[control]["intervention"]
            expected = group["baseline"]["intervention"]["high_component_l2"] * 0.2
            require(np.isclose(primary["requested_perturbation_l2"], expected, rtol=1e-5, atol=1e-6), "Primary norm inconsistent")
            require(np.isclose(comparison["requested_perturbation_l2"], primary["requested_perturbation_l2"], rtol=1e-5, atol=1e-6), "Requested norms differ")
            require(primary["actual_perturbation_l2"] > 0, "Zero primary perturbation")
            values.append({"query_id": query, "primary_requested": primary["requested_perturbation_l2"],
                           "control_requested": comparison["requested_perturbation_l2"],
                           "primary_actual": primary["actual_perturbation_l2"], "control_actual": comparison["actual_perturbation_l2"],
                           "actual_ratio_control_over_primary": comparison["actual_perturbation_l2"] / primary["actual_perturbation_l2"]})
        norms[control] = {"actual_ratio_control_over_primary": describe([row["actual_ratio_control_over_primary"] for row in values]),
                          "actual_norms_exactly_equal_all_queries": all(row["primary_actual"] == row["control_actual"] for row in values),
                          "per_query": values}
    primary_all, primary_time = summaries[PRIMARY]["all"], summaries[PRIMARY]["temporal_state"]
    comparator_time = comparisons["middle13_temporal_matched_up"]["temporal_state"]
    gates = {"all_structural_controls_exact": len(structural_checks) == 36,
             "temporal_net_corrections_at_least_two": primary_time["net_corrected"] >= 2,
             "temporal_positive_net_videos_at_least_two": len(primary_time["positive_net_videos"]) >= 2,
             "all_questions_no_net_degradation": primary_all["net_corrected"] >= 0,
             "primary_beats_norm_matched_temporal_accuracy": comparator_time["net_corrected"] > 0 and comparator_time["accuracy_video_macro_delta"] > 0}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cpu_audit",
        "run_directory": str(run), "gpu_used": False, "scope": "original_8_video_36_question_development_only",
        "primary_candidate": PRIMARY, "conditions": conditions,
        "integrity": {"records": 540, "queries": 36, "conditions_per_query": 15, "validated_input_frames": 128,
                      "structural_checks": structural_checks, "structural_comparisons": len(structural_checks) * 3,
                      "config_sha256": digest(run / "config.json"), "records_sha256": digest(run / "records.jsonl"),
                      "questions_sha256": digest(question_path), "protocol_sha256": digest(protocol_path),
                      "source_sha256": digest(source_path), "analysis_source_sha256": digest(Path(__file__))},
        "summaries": summaries, "primary_against_controls": comparisons, "perturbation_norm_comparisons": norms,
        "development_gates": gates, "passes_all_development_gates": all(gates.values()),
        "gate_interpretation": "Two-video gate requires positive temporal net corrections in two videos; margins alone do not pass.",
        "evidence_gaps": [
            "Only 8 reused development videos; no independent confirmation or cross-person inference.",
            "No condition tests bounded stored-memory selection/compression or question-grounded evidence removal.",
            "Action-coverage overlap does not provide a causal QA evidence label.",
            "Pixel-order edits and legally relabeled counterfactual QA are not part of this run.",
            "BF16 actual perturbation norms differ despite matched requested norms; ratios are reported.",
            "Nonstructural full-vocabulary arrays were not saved; their hashes cannot be independently recomputed.",
            "No repeated baseline estimates runtime margin variability; exact controls verify implementation, not robust task benefit.",
            "No second model; best nonprimary conditions remain exploratory.",
            "Temporal filtering uses the complete observed prefix, not an online causal memory update."]}


def markdown(report):
    baseline, primary = report["summaries"]["baseline"], report["summaries"][PRIMARY]
    lines = ["# V2 固定输入视觉分支干预：CPU审计", "",
             "36题×15条件=540条完整核验；128帧输入哈希、官方题目、选项、金标、margin与冻结源码/协议一致。",
             f"3个结构/identity负对照共{report['integrity']['structural_comparisons']}次全词表逐元素比较全部相同。", "",
             f"预声明主要条件 {PRIMARY}：全部 {primary['all']['correct']}/36（baseline {baseline['all']['correct']}/36）；时序 {primary['temporal_state']['correct']}/8（baseline {baseline['temporal_state']['correct']}/8）。",
             f"开发门槛：**{'通过' if report['passes_all_development_gates'] else '未通过'}**。不以其它最好条件替换主要候选；margin改变不等于解决问答错误。", "",
             "## 全条件结果", "", "| 条件 | 全部正确 | 视频宏平均 | 时序正确 | 时序宏平均 | 时序净纠正 | 平均margin差 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for condition in report["conditions"]:
        overall, timed = report["summaries"][condition]["all"], report["summaries"][condition]["temporal_state"]
        lines.append(f"| {condition} | {overall['correct']}/36 | {overall['accuracy_video_macro']:.4f} | {timed['correct']}/8 | {timed['accuracy_video_macro']:.4f} | {timed['net_corrected']:+d} | {overall['gold_margin_delta']['mean']:+.4f} |")
    lines.extend(["", "## 预声明开发门槛", "", "| 门槛 | 结果 |", "| --- | --- |"])
    lines.extend(f"| {name} | {'通过' if passed else '未通过'} |" for name, passed in report["development_gates"].items())
    lines.extend(["", "两视频门槛按两个视频各自时序题净纠正为正解释，JSON保留全部纠正/损伤视频。", "",
                  "## 主要条件对等扰动范数控制", "",
                  "| 控制 | 全部净纠正 | 时序净纠正 | 时序宏平均差 | 时序margin差 | 实际范数比中位数[范围] |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"])
    for control in MATCHED:
        comparison, norms = report["primary_against_controls"][control], report["perturbation_norm_comparisons"][control]["actual_ratio_control_over_primary"]
        lines.append(f"| {control} | {comparison['all']['net_corrected']:+d} | {comparison['temporal_state']['net_corrected']:+d} | {comparison['temporal_state']['accuracy_video_macro_delta']:+.4f} | {comparison['temporal_state']['gold_margin_delta']['mean']:+.4f} | {norms['median']:.6f} [{norms['min']:.6f}, {norms['max']:.6f}] |")
    lines.extend(["", "范数比=控制实际BF16扰动/主要条件实际扰动。请求范数相等已核验，实际范数不逐bit相等。循环位移控制保留高频子空间，控制时间方向，不是独立随机频带。", "",
                  "## 全部预测改变", "", "| 条件 | query_id | 原→新答案 | 金标 | 正确性 | margin变化 |",
                  "| --- | --- | --- | --- | --- | ---: |"])
    for condition in report["conditions"]:
        for row in report["summaries"][condition]["all"]["prediction_changes"]:
            lines.append(f"| {condition} | {row['query_id']} | {chr(65 + row['before_index'])}→{chr(65 + row['after_index'])} | {chr(65 + row['gold_index'])} | {row['before_correct']}→{row['after_correct']} | {row['margin_delta']:+.3f} |")
    lines.extend(["", "## 独立证据链缺项", ""])
    lines.extend(f"- {item}" for item in report["evidence_gaps"])
    lines.extend(["", "仅为开发探索，不作显著性、泛化、长期记忆或频谱记忆成功推断；v1负结果不变。", "",
                  "## 来源", "", f"- Run: {report['run_directory']}"])
    lines.extend(f"- {name}: {value}" for name, value in report["integrity"].items() if name.endswith("sha256"))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    run, output = (root / arguments.run_dir).resolve(), (root / arguments.output_dir).resolve()
    report = audit(run, root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    changes = [{"condition": condition, **row} for condition in report["conditions"]
               for row in report["summaries"][condition]["all"]["prediction_changes"]]
    (output / "changed_questions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes), encoding="utf-8")
    print(json.dumps({"records": 540, "structural_comparisons": report["integrity"]["structural_comparisons"],
                      "primary_all": report["summaries"][PRIMARY]["all"]["correct"],
                      "primary_temporal": report["summaries"][PRIMARY]["temporal_state"]["correct"],
                      "gates": report["development_gates"], "passes": report["passes_all_development_gates"]}, indent=2))


if __name__ == "__main__":
    main()
