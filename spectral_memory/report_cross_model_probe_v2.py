"""Source-level descriptive aggregation of frozen read-only V2 probes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .cross_model_probe_v2 import CONDITIONS, FRAME_SHA, QUESTION_SHA, digest, read_rows, require, write_json


EXPECTED_VIDEOS = {"video_1024", "video_10267", "video_1414", "video_275", "video_3007", "video_341", "video_6532", "video_8221"}
MODELS = ("qwen", "smol")
SPECTRA = ("value", "post_rope_key", "weighted_visual_context")


def statistics(values):
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    require(not len(finite) or bool(np.isfinite(finite).all()), "Nonfinite summary values")
    if not len(finite):
        return {"n_sources": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {"n_sources": len(finite), "mean": float(finite.mean()), "median": float(np.median(finite)),
            "minimum": float(finite.min()), "maximum": float(finite.max()),
            "quartile25": float(np.quantile(finite, 0.25)), "quartile75": float(np.quantile(finite, 0.75)),
            "positive_sources": int((finite > 0).sum()), "negative_sources": int((finite < 0).sum()),
            "zero_sources": int((finite == 0).sum())}


def softmax(values):
    values = np.asarray(values, dtype=float)
    exponential = np.exp(values - values.max())
    return exponential / exponential.sum()


def output_distance(first, second):
    average = (first + second) / 2
    first_active = first > 0
    second_active = second > 0
    divergence = 0.5 * np.sum(first[first_active] * np.log(first[first_active] / average[first_active]))
    divergence += 0.5 * np.sum(second[second_active] * np.log(second[second_active] / average[second_active]))
    return {"total_variation": float(np.abs(first - second).sum() / 2), "jensen_shannon_nats": float(divergence)}


def spectral_summary(spectrum):
    total = float(sum(spectrum["total_energy"]))
    dc = float(sum(spectrum["dc_energy"]))
    non_dc = float(sum(spectrum["non_dc_energy"]))
    shape = spectrum["shape_head_time_patch_channel"]
    elements = int(np.prod(shape))
    return {"dc_energy": dc, "non_dc_energy": non_dc, "total_energy": total,
            "dc_per_element": dc / elements, "non_dc_per_element": non_dc / elements,
            "non_dc_fraction": non_dc / total if total > 0 else None,
            "parseval_relative_error_maximum": max(spectrum["parseval_relative_error"])}


def layer_summary(row):
    heads = row["head_metrics"]
    norms = row["ffn_metrics"]
    output = {"visual_attention_mass": float(np.mean([head["visual_attention_mass"] for head in heads])),
              "visual_to_native_context_ratio": float(np.mean([head["visual_to_native_context_ratio"] for head in heads])),
              "native_context_l2": float(np.mean([head["native_context_l2"] for head in heads])),
              "visual_context_l2": float(np.mean([head["visual_context_l2"] for head in heads])),
              "attention_entropy_normalized": float(np.mean([head["attention_entropy_normalized"] for head in heads])),
              "ffn_over_attention_output_l2": norms["ffn_over_attention_output_l2"],
              "ffn_over_pre_ffn_residual_l2": norms["ffn_over_pre_ffn_residual_l2"],
              "attention_over_decoder_input_l2": norms["attention_over_decoder_input_l2"],
              "reference_native_relative_l2_error_maximum": max(head["reference_native_relative_l2_error"] for head in heads)}
    for name in SPECTRA:
        output.update({f"{name}_{key}": value for key, value in spectral_summary(row[f"{name}_temporal_spectrum"]).items()})
    return output


def read_shards(paths):
    records = {}
    layers = {}
    provenance = []
    signatures = {}
    for supplied in paths:
        directory = supplied.resolve()
        configuration = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        require(configuration["status"] == "complete", f"Incomplete shard: {directory}")
        require(configuration["expected_cases"] == configuration["completed_cases"], f"Case count mismatch: {directory}")
        require(configuration["questions_sha256"] == QUESTION_SHA and configuration["frames_manifest_sha256"] == FRAME_SHA,
                "Unexpected data manifests")
        require(not (directory / "resource_stop.json").exists(), "Resource-stopped shard cannot be included")
        require(digest(directory / "records.jsonl") == configuration["records_sha256"], "Record hash mismatch")
        require(digest(directory / "layers.jsonl") == configuration["layers_sha256"], "Layer hash mismatch")
        require(digest(directory / "cases.json") == configuration["cases_sha256"], "Case hash mismatch")
        require(digest(directory / "source_snapshot" / "protocol.md") == configuration["protocol_sha256"], "Protocol hash mismatch")
        for filename, expected in configuration["source_sha256"].items():
            require(digest(directory / "source_snapshot" / filename) == expected, "Source snapshot hash mismatch")
        model = configuration["model_choice"]
        require(model in MODELS, "Unexpected model")
        signature = (configuration["source_sha256"], configuration["protocol_sha256"],
                     configuration["model"]["model_revision"], configuration["attention_geometry_query_kv_channel"],
                     configuration["implementation_sha256"])
        if model in signatures:
            require(signature == signatures[model], "Mixed implementations within a model")
        signatures[model] = signature
        case_rows = read_rows(directory / "records.jsonl")
        layer_rows = read_rows(directory / "layers.jsonl")
        count = configuration["layers"]
        require(len(case_rows) == configuration["expected_cases"] and len(layer_rows) == len(case_rows) * count, "Incomplete layer matrix")
        shard_keys = set()
        for row in case_rows:
            key = (model, row["video_id"], row["condition"])
            require(key not in records and key not in shard_keys, "Duplicate source condition")
            require(row["model"] == model and row["video_id"] in EXPECTED_VIDEOS and row["condition"] in CONDITIONS, "Unexpected case identity")
            require(row["native_passive_whole_vocabulary_exact"] and row["native_restored_whole_vocabulary_exact"], "Passive controls failed")
            require(row["captured_layer_count"] == count, "Recorded layer count mismatch")
            require((row["condition"] == "real") == (row["gold_answer_index"] is not None), "Static labels must be absent")
            require(row["condition"] == "real" or row["correct"] is None, "Static accuracy is prohibited")
            archive = directory / row["archive_file"]
            require(digest(archive) == row["archive_sha256"], "Archive hash mismatch")
            with np.load(archive, allow_pickle=False) as values:
                native = values["native_logits"]
                require(np.array_equal(native, values["passive_logits"]) and np.array_equal(native, values["restored_logits"]), "Archived logits differ")
                require(np.array_equal(native[values["option_token_ids"]], row["option_logits"]), "Archived options differ")
            records[key] = row
            shard_keys.add(key)
        shard_layers = defaultdict(set)
        for row in layer_rows:
            case_key = (model, row["video_id"], row["condition"])
            key = (*case_key, row["layer"])
            require(case_key in shard_keys and key not in layers, "Unexpected or duplicate layer")
            require(row["case_id"] == records[case_key]["case_id"] and row["archive_sha256"] == records[case_key]["archive_sha256"], "Layer/case alignment changed")
            require(row["ffn_metrics"]["attention_residual_identity_exact"] and row["ffn_metrics"]["ffn_residual_identity_exact"], "Residual identity failed")
            require(len(row["head_metrics"]) == row["query_heads"], "Query head count mismatch")
            require(len(row["value_temporal_spectrum"]["total_energy"]) == row["kv_heads"], "Real KV head count mismatch")
            layers[key] = layer_summary(row)
            shard_layers[case_key].add(row["layer"])
        require(all(indices == set(range(count)) for indices in shard_layers.values()) and set(shard_layers) == shard_keys,
                "Each case must contain all decoder layers")
        provenance.append({"directory": str(directory), "config_sha256": digest(directory / "config.json"),
                           "records_sha256": configuration["records_sha256"], "layers_sha256": configuration["layers_sha256"],
                           "cases": len(case_rows), "model": model, "sources": configuration["selected_video_ids"]})
    return records, layers, provenance


def aggregate(records, layers, allow_partial):
    coverage = {model: sorted({key[1] for key in records if key[0] == model}) for model in MODELS}
    for model, sources in coverage.items():
        for source in sources:
            require(all((model, source, condition) in records for condition in CONDITIONS), "Missing paired condition")
        if not allow_partial:
            require(set(sources) == EXPECTED_VIDEOS, "Full report requires eight sources for both models")
    source_rows = []
    depth_rows = []
    output_rows = []
    model_summaries = {}
    for model, sources in coverage.items():
        if not sources:
            continue
        count = 28 if model == "qwen" else 32
        bands = {name: indices.tolist() for name, indices in zip(("early", "middle", "late"), np.array_split(np.arange(count), 3))}
        bands["final"] = [count - 1]
        metric_names = tuple(layers[(model, sources[0], "real", 0)])
        for source in sources:
            for layer in range(count):
                real = layers[(model, source, "real", layer)]
                first = layers[(model, source, "static_first", layer)]
                last = layers[(model, source, "static_last", layer)]
                for condition in (*CONDITIONS, "static_mean", "real_minus_static_mean"):
                    if condition in CONDITIONS:
                        metrics = layers[(model, source, condition, layer)]
                    else:
                        metrics = {}
                        for name in metric_names:
                            valid = first[name] is not None and last[name] is not None and real[name] is not None
                            average = (first[name] + last[name]) / 2 if valid else None
                            metrics[name] = (real[name] - average if average is not None else None) if condition == "real_minus_static_mean" else average
                    source_rows.append({"model": model, "video_id": source, "layer": layer,
                                        "relative_depth": (layer + 1) / count, "condition": condition, **metrics})
            real_record = records[(model, source, "real")]
            real_probability = softmax(real_record["option_logits"])
            comparisons = {}
            for condition in ("static_first", "static_last"):
                static_record = records[(model, source, condition)]
                comparisons[condition] = {**output_distance(real_probability, softmax(static_record["option_logits"])),
                                          "prediction_changed": real_record["predicted_answer_index"] != static_record["predicted_answer_index"],
                                          "static_prediction": static_record["predicted_answer_index"]}
            output_rows.append({"model": model, "video_id": source, "query_id": real_record["query_id"],
                                "real_prediction": real_record["predicted_answer_index"], "real_correct": real_record["correct"],
                                "real_gold": real_record["gold_answer_index"], "real_option_probabilities": real_probability.tolist(),
                                "comparisons": comparisons})
        for layer in range(count):
            for condition in (*CONDITIONS, "static_mean", "real_minus_static_mean"):
                matched = [row for row in source_rows if row["model"] == model and row["layer"] == layer and row["condition"] == condition]
                depth_rows.append({"model": model, "layer": layer, "relative_depth": (layer + 1) / count,
                                   "condition": condition, "metrics": {name: statistics([row[name] for row in matched]) for name in metric_names}})
        band_rows = []
        for band, indices in bands.items():
            for condition in (*CONDITIONS, "static_mean", "real_minus_static_mean"):
                per_source = []
                for source in sources:
                    matched = [row for row in source_rows if row["model"] == model and row["video_id"] == source and row["layer"] in indices and row["condition"] == condition]
                    metrics = {name: float(np.mean([row[name] for row in matched])) if all(row[name] is not None for row in matched) else None for name in metric_names}
                    per_source.append({"video_id": source, **metrics})
                band_rows.append({"band": band, "layers": indices, "condition": condition,
                                  "metrics": {name: statistics([row[name] for row in per_source]) for name in metric_names}, "source_values": per_source})
        chosen_outputs = [row for row in output_rows if row["model"] == model]
        model_summaries[model] = {"sources": sources, "n_sources": len(sources), "real_correct": sum(row["real_correct"] for row in chosen_outputs),
                                  "bands": band_rows, "output_sensitivity": {
                                      condition: {"changed_predictions": sum(row["comparisons"][condition]["prediction_changed"] for row in chosen_outputs),
                                                  "total_variation": statistics([row["comparisons"][condition]["total_variation"] for row in chosen_outputs]),
                                                  "jensen_shannon_nats": statistics([row["comparisons"][condition]["jensen_shannon_nats"] for row in chosen_outputs])}
                                      for condition in ("static_first", "static_last")}}
    return {"status": "partial_preview" if allow_partial else "complete", "coverage": coverage,
            "independent_unit": "source video; heads first averaged and spectral energies pooled within layer; layer-band means within source",
            "inference": "descriptive only on eight previously seen validation sources; no p-values, independence assumptions across heads/layers, or causal claims",
            "spectral_pairing": "real minus arithmetic mean of both static endpoint conditions, metric by metric; static spectra can contain causal and positional context",
            "band_definition": "numpy.array_split(all layer indices,3), plus separately reported final layer; bands are architecture-relative",
            "models": model_summaries, "source_layer_metrics": source_rows, "depth_summaries": depth_rows,
            "source_output_sensitivity": output_rows}


def markdown(report):
    lines = ["# 跨模型只读诊断V2：按源汇总", "", f"状态：{report['status']}。只使用旧validation，每源第一题。", "",
             "均值与范围的统计单位均为源视频。先在层内合并head，再在每源内汇总深度区间；不把head或层当成额外样本。以下结果只是描述，不构成故障或因果机制证明。", "",
             "## 深度趋势", "", "| 模型 | 条件 | 层区间 | n源 | 视觉mass | 视觉/native范数比 | FFN/attention范数比 | V非DC占比 | K非DC占比 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, summary in report["models"].items():
        for band in summary["bands"]:
            if band["condition"] not in ("real", "static_mean", "real_minus_static_mean"):
                continue
            names = ("visual_attention_mass", "visual_to_native_context_ratio", "ffn_over_attention_output_l2", "value_non_dc_fraction", "post_rope_key_non_dc_fraction")
            values = [band["metrics"][name]["mean"] for name in names]
            printed = [f"{value:.5g}" if value is not None else "null" for value in values]
            lines.append(f"| {model} | {band['condition']} | {band['band']} ({band['layers'][0]}–{band['layers'][-1]}) | {summary['n_sources']} | " + " | ".join(printed) + " |")
    lines.extend(["", "## 输出敏感性", "", "静态条件没有有效原答案金标；仅比较输出，不计算静态准确率。", "", "| 模型 | 真视频正确 | 与静态首帧预测不同 | 与静态末帧预测不同 | 首帧TV均值 | 末帧TV均值 |", "|---|---:|---:|---:|---:|---:|"])
    for model, summary in report["models"].items():
        first = summary["output_sensitivity"]["static_first"]
        last = summary["output_sensitivity"]["static_last"]
        lines.append(f"| {model} | {summary['real_correct']}/{summary['n_sources']} | {first['changed_predictions']}/{summary['n_sources']} | {last['changed_predictions']}/{summary['n_sources']} | {first['total_variation']['mean']:.5g} | {last['total_variation']['mean']:.5g} |")
    lines.extend(["", "## 解释边界", "", "- 视觉mass小可能来自正常的注意力集中或信息已被中间文本位置整合，不能单独称为视觉失效。", "- FFN范数大不代表其压制视觉；需要保持其他计算不变的因果干预。", "- real/static谱差包含内容与上下文差异；固定图像的后层V与post-RoPE K也可能有非DC。", "- 总非DC能量等价于时间维去均值方差；仅有该量不能证明独特频率结构或顺序推理。", "- 两模型帧分组、视觉token预算和处理像素不同；绝对能量与范数不能直接跨模型排名。", "- 完整词表、源码与产物哈希均已校验；浮点旁路重建误差保留在JSON。", ""])
    return "\n".join(lines)


def plot_depth(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("visual_attention_mass", "Visual attention mass"),
              ("visual_to_native_context_ratio", "Visual / native context L2"),
              ("ffn_over_attention_output_l2", "FFN / attention output L2"),
              ("value_non_dc_fraction", "V temporal non-DC energy fraction"),
              ("post_rope_key_non_dc_fraction", "Post-RoPE K non-DC energy fraction"),
              ("weighted_visual_context_non_dc_fraction", "Weighted visual context non-DC fraction"))
    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.2))
    colors = {"qwen": "#315da8", "smol": "#c5612f"}
    for axis, (metric, label) in zip(axes.flat, panels):
        for model in report["models"]:
            for condition, style in (("real", "-"), ("static_mean", "--")):
                rows = [row for row in report["depth_summaries"] if row["model"] == model and row["condition"] == condition]
                depths = [row["relative_depth"] for row in rows]
                values = [row["metrics"][metric] for row in rows]
                axis.plot(depths, [value["median"] for value in values], color=colors[model], linestyle=style,
                          linewidth=1.5, label=f"{model}: {'real' if condition == 'real' else 'static endpoints mean'}")
                axis.fill_between(depths, [value["quartile25"] for value in values], [value["quartile75"] for value in values],
                                  color=colors[model], alpha=0.07)
        axis.set_title(label, fontsize=10)
        axis.set_xlabel("Decoder relative depth")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.16)
        if metric.endswith("fraction"):
            axis.set_ylim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    counts = ", ".join(f"{model}: {summary['n_sources']} sources" for model, summary in report["models"].items())
    figure.suptitle(f"Read-only development probe ({counts})\nSource medians and interquartile ranges; descriptive, not causal", fontsize=12)
    figure.tight_layout(rect=(0, 0.08, 1, 0.92))
    for extension in ("png", "svg"):
        figure.savefig(directory / f"depth_profiles.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--plot", action="store_true")
    arguments = parser.parse_args()
    require(not arguments.output_dir.exists(), "Analysis output must be new")
    records, layers, provenance = read_shards(arguments.input_dir)
    report = aggregate(records, layers, arguments.allow_partial)
    report["provenance"] = provenance
    report["analysis_source_sha256"] = digest(__file__)
    arguments.output_dir.mkdir(parents=True)
    write_json(arguments.output_dir / "report.json", report)
    (arguments.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    if arguments.plot:
        plot_depth(report, arguments.output_dir)
    write_json(arguments.output_dir / "integrity.json", {"report_sha256": digest(arguments.output_dir / "report.json"),
                                                        "markdown_sha256": digest(arguments.output_dir / "report.md"),
                                                        "analysis_source_sha256": report["analysis_source_sha256"],
                                                        "input_shards": len(provenance), "case_count": len(records), "layer_count": len(layers)})
    print(json.dumps({"status": report["status"], "coverage": report["coverage"], "case_count": len(records), "layer_count": len(layers)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
