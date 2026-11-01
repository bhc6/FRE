"""CPU-only artifact audit and source-cluster summaries for delimiter scaling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "outputs/normalized_development_analysis_v2/analyze.py"
specification = importlib.util.spec_from_file_location("normalized_analysis_helpers", HELPER)
helpers = importlib.util.module_from_spec(specification)
specification.loader.exec_module(helpers)
np = helpers.np
check = helpers.check
digest = helpers.digest
read_json = helpers.read_json
read_rows = helpers.read_rows
TAGS = ("identity", "delimiter_scale0.5", "delimiter_scale2", "delimiter_scale4", "update_damp0.5")
FIELDS = ("next_input_norm", "next_q_proj", "next_k_proj", "next_v_proj")


def close(actual, expected, tolerance=1e-10):
    return bool(np.allclose(actual, expected, rtol=tolerance, atol=tolerance))


def read_shard(folder):
    folder = folder.resolve()
    configuration = read_json(folder / "config.json")
    check(str(folder) + ":complete", configuration["status"] == "complete")
    helpers.resource_check(folder)
    for filename, field in (("records.jsonl", "records_sha256"), ("cases.json", "cases_sha256"), ("resource_summary.json", "resource_summary_sha256")):
        check(str(folder) + ":" + filename, digest(folder / filename) == configuration[field])
    check(str(folder) + ":protocol", digest(folder / "source_snapshot/protocol.md") == configuration["protocol_sha256"])
    for filename, expected in configuration["source_sha256"].items():
        check(str(folder) + ":source:" + filename, digest(folder / "source_snapshot" / filename) == expected and
              digest(ROOT / "spectral_memory" / filename) == expected)
    implementation = configuration["implementation_sha256"]
    check(str(folder) + ":transformers_implementation", digest(Path(implementation["path"])) == implementation["sha256"])
    check(str(folder) + ":manifests", configuration["questions_sha256"] == "76e60bb8b3cc522dfd3aa6323782d753ef58caa78b7dbcbe864224f34d7d3504" and
          configuration["frames_sha256"] == "82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9")
    rows = read_rows(folder / "records.jsonl")
    check(str(folder) + ":record_count", len(rows) == configuration["expected_records"] == configuration["completed_records"])
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    check(str(folder) + ":case_count", len(groups) == configuration["expected_cases"] == configuration["completed_cases"])
    model = configuration["model_choice"]
    for case_id, group in groups.items():
        check(case_id + ":tags", len(group) == len(TAGS) and {row["tag"] for row in group} == set(TAGS))
        first = group[0]
        archive_path = folder / first["archive_file"]
        check(case_id + ":archive", all(digest(archive_path) == row["archive_sha256"] for row in group))
        with np.load(archive_path, allow_pickle=False) as archive:
            native = archive["identity_logits"]
            identifiers = archive["input_ids"]
            option_ids = archive["option_token_ids"]
            positions = archive["delimiter_positions"]
            baseline = native[option_ids]
            check(case_id + ":whole_vocab_controls", np.array_equal(native, archive["zero_logits"]) and np.array_equal(native, archive["restored_logits"]))
            check(case_id + ":input_hash", hashlib.sha256(identifiers.tobytes()).hexdigest() == first["input_tensor_hashes"]["input_ids"]["sha256"])
            expected_positions = np.flatnonzero(np.isin(identifiers, list(configuration["delimiter_ids"].values())))
            check(case_id + ":delimiter_selection", np.array_equal(positions, expected_positions) and
                  positions.tolist() == first["delimiter_layout"]["positions"] and len(positions) == (16 if model == "qwen" else 32))
            scaled = archive["delimiter_scale2_logits"].astype(np.float64)
            damped = archive["update_damp0.5_logits"].astype(np.float64)
            difference = damped - scaled
            comparison = {"full_vocabulary_maximum_absolute_difference": float(np.max(np.abs(difference))),
                          "full_vocabulary_relative_l2_difference": float(np.linalg.norm(difference) / max(np.linalg.norm(scaled), 1e-30)),
                          "full_vocabulary_exact": bool(np.array_equal(scaled, damped)),
                          "option_logits_exact": bool(np.array_equal(scaled[option_ids], damped[option_ids])),
                          "option_prediction_same": bool(scaled[option_ids].argmax() == damped[option_ids].argmax())}
            for row in group:
                prefix = model + ":" + case_id + ":" + row["tag"]
                values = archive[row["tag"] + "_logits"][option_ids]
                check(prefix + ":scores", np.array_equal(values, row["option_logits"]) and row["option_token_ids"] == option_ids.tolist())
                check(prefix + ":predictions", int(values.argmax()) == row["predicted_answer_index"] and int(baseline.argmax()) == row["baseline_predicted_answer_index"])
                check(prefix + ":tie", np.flatnonzero(values == values.max()).tolist() == row["exact_maximum_tie_indices"])
                check(prefix + ":paired_inputs", row["input_tensor_hashes"] == first["input_tensor_hashes"] and row["frame_sha256"] == first["frame_sha256"])
                check(prefix + ":native_controls", row["identity_exact"] and row["restoration_exact"])
                rotation = row["rotation"]
                mapping = [(index + rotation) % 3 for index in range(3)]
                check(prefix + ":semantic_mapping", row["semantic_original_option_indices"] == mapping and
                      row["predicted_original_option_index"] == mapping[int(values.argmax())] and
                      row["baseline_predicted_original_option_index"] == mapping[int(baseline.argmax())])
                if row["supervised"]:
                    check(prefix + ":rotation_and_gold", rotation in configuration["rotations"] and
                          row["gold_answer_index"] == (row["original_gold_answer_index"] - rotation) % 3 and
                          row["question"]["options"] == row["original_options"][rotation:] + row["original_options"][:rotation])
                    row["correct"] = int(values.argmax() == row["gold_answer_index"])
                    row["baseline_correct"] = int(baseline.argmax() == row["gold_answer_index"])
                else:
                    check(prefix + ":static_no_gold", row["gold_answer_index"] is None and row["original_gold_answer_index"] is None and
                          rotation == 0 and "answer_index" not in row["question"] and "answer_text" not in row["question"])
                    row["correct"] = None
                    row["baseline_correct"] = None
                row["option_KL_from_baseline"] = float(helpers.kl([baseline], [values])[0])
                check(prefix + ":flip", row["baseline_answer_changed"] == bool(values.argmax() != baseline.argmax()))
                check(prefix + ":damping_comparison", all(close(value, row["dts2_damping_comparison"][name]) for name, value in comparison.items()))
                for field in FIELDS:
                    initial = archive["identity_" + field].astype(np.float64)
                    current = archive[row["tag"] + "_" + field].astype(np.float64)
                    measured = row["next_layer_mechanism_vs_identity"][field]
                    initial_norm = np.linalg.norm(initial)
                    current_norm = np.linalg.norm(current)
                    initial_per = np.linalg.norm(initial, axis=-1)
                    current_per = np.linalg.norm(current, axis=-1)
                    check(prefix + ":" + field, list(current.shape) == measured["shape"] and
                          close(initial_norm, measured["baseline_l2"]) and close(current_norm, measured["intervention_l2"]) and
                          close(current_norm / max(initial_norm, 1e-30), measured["norm_ratio"]) and
                          close(np.linalg.norm(current - initial) / max(initial_norm, 1e-30), measured["relative_l2_difference"]) and
                          close(current_per / np.maximum(initial_per, 1e-30), measured["per_delimiter_norm_ratio"]) and
                          close(np.linalg.norm(current - initial, axis=-1) / np.maximum(initial_per, 1e-30), measured["per_delimiter_relative_l2_difference"]))
    return configuration, rows


def method_metrics(rows):
    index = {(row["case_id"], row["tag"]): row for row in rows}
    result = {}
    for tag in TAGS:
        selected = sorted([row for row in rows if row["tag"] == tag and row["supervised"]], key=lambda row: row["case_id"])
        baseline = [index[(row["case_id"], "identity")] for row in selected]
        summary = helpers.paired([row["option_logits"] for row in baseline], [row["option_logits"] for row in selected],
                                 [row["gold_answer_index"] for row in selected], [row["video_id"] for row in selected],
                                 [row["case_id"] for row in selected])
        static = [row for row in rows if row["tag"] == tag and not row["supervised"]]
        summary["static"] = {"count": len(static), "prediction_flips": sum(row["baseline_answer_changed"] for row in static),
                             "option_kl": helpers.distribution([row["option_KL_from_baseline"] for row in static])}
        summary["next_layer_mechanism"] = {field: {
            "norm_ratio": helpers.distribution([row["next_layer_mechanism_vs_identity"][field]["norm_ratio"] for row in selected]),
            "relative_l2_difference": helpers.distribution([row["next_layer_mechanism_vs_identity"][field]["relative_l2_difference"] for row in selected])}
            for field in FIELDS}
        result[tag] = summary
    return result


def aggregate(shards, require_rotations):
    indexed = {}
    provenance = []
    signatures = {}
    for folder in shards:
        configuration, rows = read_shard(folder)
        model = configuration["model_choice"]
        signature = (configuration["source_sha256"], configuration["protocol_sha256"], configuration["injection_layer"], configuration["delimiter_ids"])
        check(model + ":consistent_signature", signatures.setdefault(model, signature) == signature)
        provenance.append({"folder": str(folder), "config_sha256": digest(folder / "config.json"), "status": configuration["status"],
                           "model": model, "rotations": configuration["rotations"], "completed_cases": configuration["completed_cases"]})
        for row in rows:
            key = (model, row["case_id"], row["tag"])
            if key in indexed:
                previous = indexed[key]
                check(str(key) + ":duplicate_static_only", not row["supervised"] and previous["option_logits"] == row["option_logits"] and
                      previous["input_tensor_hashes"] == row["input_tensor_hashes"] and previous["frame_sha256"] == row["frame_sha256"])
            else:
                indexed[key] = row
    result = {}
    for model in ("qwen", "smol"):
        rows = [row for key, row in indexed.items() if key[0] == model]
        check(model + ":present", bool(rows))
        rotations = sorted({row["rotation"] for row in rows if row["supervised"]})
        check(model + ":rotation_set", rotations == [0, 1, 2] if require_rotations else rotations in ([0], [0, 1, 2]))
        for rotation in rotations:
            check(model + ":complete_rotation:" + str(rotation),
                  all(len([row for row in rows if row["tag"] == tag and row["supervised"] and row["rotation"] == rotation]) == 40 for tag in TAGS))
        check(model + ":static_16_per_tag", all(len([row for row in rows if row["tag"] == tag and not row["supervised"]]) == 16 for tag in TAGS))
        per_rotation = {str(rotation): method_metrics([row for row in rows if not row["supervised"] or row["rotation"] == rotation]) for rotation in rotations}
        result[model] = {"rotations": rotations, "rotation_matrix_complete": rotations == [0, 1, 2],
                         "methods": method_metrics(rows), "per_rotation": per_rotation, "query_aggregation": {}}
        for tag in TAGS:
            queries = {}
            for row in rows:
                if row["tag"] == tag and row["supervised"]:
                    queries.setdefault(row["query_id"], []).append(row)
            check(model + ":" + tag + ":per_query_rotations", len(queries) == 40 and all(sorted(row["rotation"] for row in group) == rotations for group in queries.values()))
            result[model]["query_aggregation"][tag] = {
                "queries": 40, "sources": 8, "presentations": 40 * len(rotations),
                "semantic_prediction_consistent_queries": sum(len({row["predicted_original_option_index"] for row in group}) == 1 for group in queries.values()),
                "all_presentations_correct_queries": sum(all(row["correct"] for row in group) for group in queries.values())}
        paired_cases = [row for row in rows if row["tag"] == "identity"]
        controls = [row["dts2_damping_comparison"] for row in paired_cases]
        result[model]["damping_equivalence"] = {
            "unique_cases_including_static": len(controls),
            "full_vocabulary_exact_cases": sum(row["full_vocabulary_exact"] for row in controls),
            "option_logits_exact_cases": sum(row["option_logits_exact"] for row in controls),
            "same_option_prediction_cases": sum(row["option_prediction_same"] for row in controls),
            "maximum_absolute_difference": helpers.distribution([row["full_vocabulary_maximum_absolute_difference"] for row in controls]),
            "relative_l2_difference": helpers.distribution([row["full_vocabulary_relative_l2_difference"] for row in controls])}
    complete_flags = [value["rotation_matrix_complete"] for value in result.values()]
    status = "complete_three_rotations" if all(complete_flags) else ("mixed_complete_matrices_rotations_missing" if any(complete_flags) else "complete_original_order_only")
    return {"models": result, "provenance": provenance, "status": status}


def markdown(report):
    lines = ["# Delimiter scaling V2：开发审计", "", "日期：2026-11-01。公开方法的适配对照，无新方法主张。仅旧validation；没有读取新确认源评分。", "",
             f"状态：{report['status']}。{report['checks_passed']}项校验通过；源码/协议/NPZ/完整词表identity与restoration、输入配对、delimiter布局、选项置换及下一层归一化/QKV统计全部独立复算。", "",
             "NLL为三选项NLL；source-macro先视频内平均。CI为10000次配对source-cluster bootstrap，seed 20261101，整组携带问题与旋转。每模型仅8源且反复开发，区间探索性、未校正多重比较。Static没有答案金标，0 flips不代表准确率。", ""]
    for model, values in report["models"].items():
        lines.extend([f"## {model}", "", f"已完成rotation={values['rotations']}；三rotation完整={values['rotation_matrix_complete']}。缺少旋转时不能作旋转稳健收益主张。", "",
                      "| 方法 | 正确呈现 | Micro % | Source-macro % | NLL micro | ΔNLL [95% CI] | Repair/regression | Static flips |",
                      "|---|---:|---:|---:|---:|---|---:|---:|"])
        for tag, score in values["methods"].items():
            lower, upper = score["nll"]["delta_micro_ci95"]
            lines.append(f"| {tag} | {score['after_correct']}/{score['questions_or_presentations']} | {score['accuracy']['after_micro']*100:.4f} | {score['accuracy']['after_source_macro']*100:.4f} | {score['nll']['after_micro']:.6f} | {score['nll']['delta_micro']:+.6f} [{lower:.6f},{upper:.6f}] | {score['repairs']}/{score['regressions']} | {score['static']['prediction_flips']}/{score['static']['count']} |")
        lines.extend(["", "| 方法减Identity | Δmicro pp [95% CI] | Δmacro pp [95% CI] | ΔNLL macro [95% CI] |", "|---|---|---|---|"])
        for tag, score in values["methods"].items():
            if tag == "identity":
                continue
            cells = []
            for metric, average, factor in (("accuracy", "micro", 100), ("accuracy", "source_macro", 100), ("nll", "source_macro", 1)):
                item = score[metric]
                lower, upper = item[f"delta_{average}_ci95"]
                cells.append(f"{item[f'delta_{average}']*factor:+.6f} [{lower*factor:.6f},{upper*factor:.6f}]")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")
        equivalence = values["damping_equivalence"]
        lines.extend(["", f"scale2与damp0.5：{equivalence['full_vocabulary_exact_cases']}/{equivalence['unique_cases_including_static']}条件完整词表逐位相同，选项预测相同{equivalence['same_option_prediction_cases']}/{equivalence['unique_cases_including_static']}；完整词表最大绝对差的最大值为{equivalence['maximum_absolute_difference']['maximum']:.9g}。这只是限定计算的机制对照，不是额外性能收益。", "",
                      "| 候选 | 下一层norm相对差最大 | Q投影相对差最大 | K投影相对差最大 | V投影相对差最大 |", "|---|---:|---:|---:|---:|"])
        for tag, score in values["methods"].items():
            metrics = [score["next_layer_mechanism"][field]["relative_l2_difference"]["maximum"] for field in FIELDS]
            lines.append(f"| {tag} | " + " | ".join(f"{value:.9g}" for value in metrics) + " |")
        lines.extend(["", "以上下一层摘要仅真实问题；等价条件计数同时包含16个静态条件。", ""])
        if values["rotation_matrix_complete"]:
            lines.extend(["| 方法 | rot0 /40 | rot1 /40 | rot2 /40 | 三次语义一致 /40 | 三次全对 /40 |", "|---|---:|---:|---:|---:|---:|"])
            for tag in TAGS:
                counts = [values["per_rotation"][str(rotation)][tag]["after_correct"] for rotation in (0, 1, 2)]
                query = values["query_aggregation"][tag]
                lines.append(f"| {tag} | {counts[0]} | {counts[1]} | {counts[2]} | {query['semantic_prediction_consistent_queries']} | {query['all_presentations_correct_queries']} |")
        lines.append("")
    lines.extend(["## 解释限制", "",
                  "下一层Q/K/V记录是head norm及RoPE之前的线性投影。残差乘2不等于归一化后的Q/K/V乘2。理想等价依赖精确尺度不变归一化；epsilon、浮点舍入或实现路径可能令实际两路不同，不能用输出接近代替误差归因实验。", "",
                  "模型参数量、输入像素/token预算、时间组和delimiter定义不同，不把模型间绝对分数或变化大小解释为等预算比较。任何正向开发变化必须同时看回退、NLL、source-macro、全部旋转和静态扰动。", "",
                  f"分析源码SHA256：{report['analysis_sha256']}。统计助手SHA256：{report['helper_sha256']}。逐例变化及全部置信区间见同目录report.json。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-rotations", action="store_true")
    parser.add_argument("--document", type=Path)
    arguments = parser.parse_args()
    check("helper_frozen", digest(HELPER) == "48aa81ab13162098afd10706f68edc9d53e52e1311a5c5058984a264987bd356")
    check("new_output", not arguments.output_dir.exists())
    report = aggregate(arguments.input_dir, arguments.require_rotations)
    report.update({"analysis_sha256": digest(Path(__file__)), "helper_sha256": digest(HELPER), "checks_passed": len(helpers.CHECKS)})
    arguments.output_dir.mkdir(parents=True)
    (arguments.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (arguments.output_dir / "checks.json").write_text(json.dumps(helpers.CHECKS, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (arguments.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    if arguments.document is not None:
        arguments.document.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks_passed": len(helpers.CHECKS), "report_sha256": digest(arguments.output_dir / "report.json")}))
    for model, summary in report["models"].items():
        for tag, metrics in summary["methods"].items():
            print(model, tag, metrics["after_correct"], metrics["nll"]["after_micro"], metrics["repairs"], metrics["regressions"])


if __name__ == "__main__":
    main()
