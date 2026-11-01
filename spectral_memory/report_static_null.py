"""CPU-only comparison of saved real-video and repeated-frame branch trajectories."""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

BRANCHES = ("decoder_input", "attention_delta", "pre_ffn", "ffn_input", "ffn_delta", "post_ffn")
LAYERS = (0, 13, 27)
CONTROLS = ("static_first", "static_last")
VIDEOS = {"video_10993", "video_1580", "video_2564", "video_3501", "video_6860", "video_8241", "video_874", "video_9253"}
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_hash(path, name, normalize_attention_flag=False):
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    if normalize_attention_flag:
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                node.keywords = [keyword for keyword in node.keywords
                                 if not (keyword.arg == "output_attentions"
                                         and isinstance(keyword.value, ast.Constant)
                                         and keyword.value.value is False)]
    return hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def describe(values):
    finite = [value for value in values if value is not None]
    if not finite:
        return {"n": 0, "median": None, "min": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    require(np.isfinite(array).all(), "Nonfinite aggregate")
    return {"n": len(finite), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def energies(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (8, 2048) and np.isfinite(values).all(), "Expected finite spatial means [8,2048]")
    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    temporal = float(np.mean(centered ** 2))
    dc = float(np.mean(mean ** 2))
    total = float(np.mean(values ** 2))
    spectrum = np.fft.rfft(centered, axis=0, norm="ortho")
    weights = np.array([1.0, 2.0, 2.0, 2.0, 1.0])
    bins = np.sum(weights[:, None] * np.abs(spectrum) ** 2, axis=1) / values.size
    temporal_fft = float(bins[1:].sum())
    require(abs(temporal - temporal_fft) <= 1e-12 * max(1.0, temporal), "Parseval mismatch")
    require(abs(dc + temporal - total) <= 1e-12 * max(1.0, total), "DC decomposition mismatch")
    return {
        "absolute_temporal_energy": temporal, "dc_energy": dc, "total_energy": total,
        "temporal_fraction": ratio(temporal, total),
        "temporal_energy_fft": temporal_fft, "parseval_absolute_error": abs(temporal - temporal_fft),
        "dc_decomposition_absolute_error": abs(dc + temporal - total),
        "one_sided_centered_energy_by_bin": bins.tolist(),
        "temporal_nonzero_exact": temporal > 0,
        "temporal_above_relative_numerical_floor": temporal > max(1e-20, total * 1e-12),
    }


def check_identities(metadata, visual_tokens):
    expected = {f"layer_{layer}_{branch}" for layer in LAYERS for branch in ("attention", "ffn")}
    require(set(metadata["identities"]) == expected, "Residual identity coverage mismatch")
    for identity in metadata["identities"].values():
        require(identity["exact_in_observed_dtype"] and identity["maximum_absolute_error"] == 0, "Residual identity failed")
        require(identity["dtype"] == "torch.bfloat16" and identity["elements_checked"] == visual_tokens * 2048, "Identity dtype/count mismatch")
    return len(expected)


def archive_rows(path, video, condition, batch):
    rows, schema = [], {}
    with np.load(path, allow_pickle=False) as arrays:
        times = np.asarray(arrays["timestamps_s"], dtype=np.float64)
        require(times.shape == (8,) and np.isfinite(times).all(), "Invalid timestamps")
        require(np.allclose(times, batch["hidden_timestamps_s"], rtol=0, atol=1e-10), "Archive/batch timestamps mismatch")
        require(np.all(np.diff(times) > 0) and np.allclose(np.diff(times), np.diff(times).mean()), "Nonuniform temporal sampling")
        required = {"timestamps_s"}
        for layer in LAYERS:
            required.add(f"layer_{layer}_ffn_rounding")
            for branch in BRANCHES:
                key = f"layer_{layer}_{branch}"
                required.update((key, key + "_spatial_mean_square", key + "_maximum_token_rms"))
                values = np.asarray(arrays[key], dtype=np.float64)
                moments = np.asarray(arrays[key + "_spatial_mean_square"], dtype=np.float64)
                maxima = np.asarray(arrays[key + "_maximum_token_rms"], dtype=np.float64)
                require(moments.shape == maxima.shape == (8,) and np.isfinite(moments).all() and np.isfinite(maxima).all(), "Invalid spatial summaries")
                pooled_square = np.mean(values ** 2, axis=1)
                require(np.all(moments >= pooled_square - 2e-6 * np.maximum(1.0, moments)), "Spatial mean-square bound violated")
                require(np.all(maxima >= 0), "Negative maximum token RMS")
                rows.append({
                    "video_id": video, "condition": condition, "layer": layer, "branch": branch,
                    "shape": list(values.shape), "energy_units": "activation_squared_mean_over_time_and_hidden_coordinates",
                    **energies(values),
                    "instantaneous_spatial_second_moment_mean": float(moments.mean()),
                    "instantaneous_spatial_dispersion_mean": float(np.maximum(moments - pooled_square, 0).mean()),
                    "maximum_token_rms_by_time": maxima.tolist(),
                })
            rounding = np.asarray(arrays[f"layer_{layer}_ffn_rounding"], dtype=np.float64)
            require(rounding.shape == (8, 2048) and np.isfinite(rounding).all(), "Invalid saved rounding")
            observed_rounding = arrays[f"layer_{layer}_post_ffn"].astype(np.float64) - arrays[f"layer_{layer}_pre_ffn"].astype(np.float64) - arrays[f"layer_{layer}_ffn_delta"].astype(np.float64)
            rounding_scale = max(1.0, *(float(np.max(np.abs(arrays[f"layer_{layer}_{name}"])))
                                        for name in ("post_ffn", "pre_ffn", "ffn_delta")))
            rounding_error = float(np.max(np.abs(rounding - observed_rounding)))
            rounding_tolerance = 2e-6 * rounding_scale
            require(rounding_error <= rounding_tolerance, "Saved rounding inconsistent after spatial reduction")
            for branch_row in rows[-len(BRANCHES):]:
                branch_row["saved_rounding_max_error"] = rounding_error
                branch_row["saved_rounding_tolerance"] = rounding_tolerance
        require(set(arrays.files) == required, "Unexpected/missing archive arrays")
        schema = {name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)} for name in arrays.files}
        require(all(len(item["shape"]) <= 2 for item in schema.values()), "Unexpected token-level tensor; review spatial interpretation")
    return rows, schema


def audit(root, run):
    config, status = read_json(run / "config.json"), read_json(run / "status.json")
    require(config["stage"] == "null" and config["status"] == status["status"] == "complete", "Static run incomplete")
    require(config["completed_records"] == status["completed"] == 16, "Expected 16 static captures")
    require(set(config["inputs"]) == VIDEOS, "Static video membership differs")
    source = run / "source_snapshot/causal_intervention.py"
    protocol = run / "source_snapshot/protocol.md"
    capture = run / "source_snapshot/probe_ffn.py"
    require(digest(source) == config["source_sha256"], "Static source hash mismatch")
    require(digest(protocol) == config["protocol_sha256"], "Static protocol hash mismatch")
    capture_hash = function_hash(capture, "capture_branches")
    normalized_capture_hash = function_hash(capture, "capture_branches", True)
    records, provenance, schema, identity_count = [], [], {}, 0
    for video in sorted(VIDEOS):
        real_folder = root / "outputs" / ("ffn_qwen3_dev16_v1" if video == "video_10993" else f"ffn_qwen3_dev16_{video}")
        probe_path, real_path = real_folder / "probe.json", real_folder / "branches.npz"
        probe, manifest = read_json(probe_path), read_json(real_folder / "frames.manifest.jsonl")
        frozen_input = config["inputs"][video]
        require(probe["status"] == "complete" and probe["video_id"] == video and probe["layers"] == list(LAYERS), "Real probe identity differs")
        require(digest(probe_path) == frozen_input["probe_sha256"], "Real probe metadata changed")
        require(digest(real_path) == probe["branches_sha256"], "Real archive changed")
        require(digest(real_folder / "frames.manifest.jsonl") == frozen_input["frames_manifest_sha256"], "Real manifest changed")
        require([digest(Path(path)) for path in manifest["frames"]] == frozen_input["frame_sha256"], "Real input frame bytes changed")
        require(frozen_input["frame_sha256"] == probe["frame_sha256"], "Probe/static input frame lists differ")
        real_capture = real_folder / "source_snapshot/probe_ffn.py"
        require(digest(real_capture) == probe["probe_source_sha256"], "Real capture snapshot mismatch")
        require(function_hash(real_capture, "capture_branches", True) == normalized_capture_hash,
                "Real/static capture differs beyond explicit output_attentions=False")
        real_batch = probe["batch"]
        require(real_batch["model_revision"] == REVISION and real_batch["dtype"] == "torch.bfloat16", "Real model/dtype differs")
        require(real_batch["visual_tokens"] == 480 and real_batch["video_grid_thw"] == [8, 12, 20], "Real token geometry differs")
        require(probe["instruction"] == "Describe the visible events." and not probe["questions_or_answers_used"], "Real prompt or label usage differs")
        identity_count += check_identities(probe, 480)
        real_rows, schema = archive_rows(real_path, video, "real", real_batch)
        records.extend(real_rows)
        provenance.append({"video_id": video, "condition": "real", "metadata_sha256": digest(probe_path),
                           "archive_sha256": digest(real_path), "probe_source_sha256": digest(real_capture),
                           "capture_function_ast_sha256": function_hash(real_capture, "capture_branches"),
                           "capture_ast_equal_after_attention_flag_normalization": True})
        for control in CONTROLS:
            metadata_path, archive_path = run / f"{video}.{control}.json", run / f"{video}.{control}.npz"
            metadata = read_json(metadata_path)
            expected_index = 0 if control == "static_first" else 15
            require(metadata["video_id"] == video and metadata["control"] == control and metadata["source_index"] == expected_index, "Static identity/index mismatch")
            require(metadata["source_sha256"] == frozen_input["frame_sha256"][expected_index], "Static source frame mismatch")
            require(metadata["synthetic_video_from_real_static_frame"] and metadata["qa_labels_used"] is False, "Static label/source contract mismatch")
            require(digest(archive_path) == metadata["archive_sha256"], "Static archive hash mismatch")
            batch = metadata["batch"]
            for key in ("model_id", "model_revision", "processor_revision", "dtype", "torch", "transformers",
                        "attention_implementation", "visual_tokens", "input_tokens", "video_grid_thw",
                        "rendered_prompt", "prompt_sha256", "source_timestamps_s", "hidden_timestamps_s"):
                require(batch[key] == real_batch[key], f"Static/real batch differs: {video}/{control}/{key}")
            identity_count += check_identities(metadata, 480)
            static_rows, static_schema = archive_rows(archive_path, video, control, batch)
            require(static_schema == schema, "Static/real archive schema differs")
            records.extend(static_rows)
            provenance.append({"video_id": video, "condition": control, "metadata_sha256": digest(metadata_path),
                               "archive_sha256": digest(archive_path), "source_frame_sha256": metadata["source_sha256"],
                               "source_frame_index_in_16_frame_input": expected_index})
    require(len(records) == 432 and len(provenance) == 24, "Expected 24 captures × 18 branches")
    grouped = defaultdict(list)
    indexed = {}
    for row in records:
        grouped[(row["condition"], row["layer"], row["branch"])].append(row)
        indexed[(row["video_id"], row["condition"], row["layer"], row["branch"])] = row
    summaries = []
    for (condition, layer, branch), rows in sorted(grouped.items()):
        require(len(rows) == 8, "Incomplete video aggregation")
        summaries.append({"condition": condition, "layer": layer, "branch": branch, "videos": 8,
                          "absolute_temporal_energy": describe([row["absolute_temporal_energy"] for row in rows]),
                          "dc_energy": describe([row["dc_energy"] for row in rows]),
                          "temporal_fraction": describe([row["temporal_fraction"] for row in rows]),
                          "temporal_nonzero_exact_count": sum(row["temporal_nonzero_exact"] for row in rows),
                          "above_numerical_floor_count": sum(row["temporal_above_relative_numerical_floor"] for row in rows)})
    paired, paired_groups = [], defaultdict(list)
    for video in sorted(VIDEOS):
        for control in CONTROLS:
            for layer in LAYERS:
                for branch in BRANCHES:
                    real, static = indexed[(video, "real", layer, branch)], indexed[(video, control, layer, branch)]
                    row = {"video_id": video, "control": control, "layer": layer, "branch": branch,
                           "temporal_static_over_real": ratio(static["absolute_temporal_energy"], real["absolute_temporal_energy"]),
                           "dc_static_over_real": ratio(static["dc_energy"], real["dc_energy"]),
                           "temporal_static_minus_real": static["absolute_temporal_energy"] - real["absolute_temporal_energy"],
                           "dc_static_minus_real": static["dc_energy"] - real["dc_energy"]}
                    paired.append(row)
                    paired_groups[(control, layer, branch)].append(row)
    paired_summaries = [
        {"control": control, "layer": layer, "branch": branch,
         **{key: describe([row[key] for row in rows]) for key in
            ("temporal_static_over_real", "dc_static_over_real", "temporal_static_minus_real", "dc_static_minus_real")}}
        for (control, layer, branch), rows in sorted(paired_groups.items())]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cpu_audit", "gpu_used": False,
        "run_directory": str(run), "scope": "within_model_forward_representation_diagnostic_only",
        "integrity": {"static_captures": 16, "real_captures": 8, "branch_records": len(records),
                      "tokenwise_residual_identity_checks": identity_count, "verified_original_frames": 128,
                      "config_sha256": digest(run / "config.json"), "source_sha256": digest(source),
                      "protocol_sha256": digest(protocol), "capture_source_sha256": digest(capture),
                      "capture_function_ast_sha256": capture_hash, "analysis_source_sha256": digest(Path(__file__)),
                      "capture_ast_sha256_without_explicit_output_attentions_false": normalized_capture_hash,
                      "source_difference": "Static capture adds explicit output_attentions=False; archived real capture omitted it. All other capture AST is identical after removing this explicit False keyword.",
                      "maximum_parseval_error": max(row["parseval_absolute_error"] for row in records),
                      "maximum_dc_decomposition_error": max(row["dc_decomposition_absolute_error"] for row in records)},
        "definitions": {"temporal_energy": "mean_{t,d}((spatial_mean(X)[t,d] - mean_t(spatial_mean(X)[:,d]))^2)",
                        "dc_energy": "mean_d(mean_t(spatial_mean(X)[:,d])^2)",
                        "normalization": "Activation squared per hidden coordinate and time point, without total-energy normalization.",
                        "numerical_floor": "max(1e-20,total_energy*1e-12); numerical sensitivity only, not significance.",
                        "rfft": "T=8; orthonormal one-sided FFT, weights [1,2,2,2,1]; four non-DC bins.",
                        "static_construction": "One actual first/last input RGB frame repeated 16 times; original presentation timestamps and prompt retained."},
        "spatial_availability": {
            "saved_token_tensor": False, "saved_spatial_means_shape": [8, 2048],
            "forward_visual_tensor_shape_from_verified_geometry": [8, 60, 2048],
            "saved_summaries": ["spatial mean", "spatial coordinate second moment", "maximum token RMS without argmax"],
            "cannot_recover": ["individual spatial token trajectories", "local temporal spectra", "location of maximum token RMS", "temporal energy removed by pooling"],
            "future_local_extraction": "A future separately authorized capture could save [T,P,D] before mean(dim=1), then use audited processor patch-merging/grid order for P-to-space mapping. These archives cannot provide that tensor, and spatial token index is not an object track."},
        "limitations": [
            "Static RGB content lacks input visual motion but retains timestamps, positions and causal context; nonzero hidden temporal energy cannot be attributed purely to visual motion.",
            "This comparison does not isolate timestamps, positional encoding, contextual accumulation, vision mixing or numerical effects from one another.",
            "Real-minus-static is not a unique motion component: the model is nonlinear and the content/trajectory itself changes.",
            "DC is not identified as background/noise; temporal energy is not identified as task-relevant information.",
            "Spatial pooling can hide local differences; stored moments cannot recover local temporal spectra.",
            "Only 8 reused development videos and 8 hidden time samples; medians/ranges are descriptive, not confidence intervals.",
            "No QA labels or task utility are measured in static controls.",
            "These results do not explain the prior layer-0 selector failure and do not repair prior spectral-memory negative results."],
        "provenance": provenance, "archive_schema": schema, "summaries": summaries,
        "paired_summaries": paired_summaries, "records": records, "paired_records": paired}


def format_stats(value):
    if value["median"] is None:
        return "undefined"
    return f"{value['median']:.6g} [{value['min']:.6g}, {value['max']:.6g}]"


def markdown(report):
    indexed = {(row["condition"], row["layer"], row["branch"]): row for row in report["summaries"]}
    paired = {(row["control"], row["layer"], row["branch"]): row for row in report["paired_summaries"]}
    layer_zero_input_counts = [indexed[(condition, 0, "decoder_input")]["temporal_nonzero_exact_count"]
                               for condition in CONTROLS]
    layer_zero_attention_counts = [indexed[(condition, 0, "attention_delta")]["temporal_nonzero_exact_count"]
                                   for condition in CONTROLS]
    lines = ["# 静态帧对照：模型内部前向时间能量审计", "",
             "16/16静态任务已完成；本报告只用CPU分析已保存的8个真实视频、重复首帧和重复末帧三种输入。",
             "同视频的时间槽、提示、模型revision、BF16、480视觉token和网格均核验一致；共432个层/分支记录。",
             f"来源哈希及{report['integrity']['tokenwise_residual_identity_checks']}个池化前残差恒等式通过；最大Parseval误差 {report['integrity']['maximum_parseval_error']:.3g}。", "",
             "源码差异已保留：真实probe未显式传output_attentions，静态probe增加output_attentions=False。移除这一显式False参数后capture函数AST一致；不称为完全相同源码。", "",
             f"主要观察：重复首/末帧条件的层0 decoder_input空间均值轨迹，非零时间能量视频数分别为{layer_zero_input_counts[0]}/8和{layer_zero_input_counts[1]}/8；同层attention_delta则为{layer_zero_attention_counts[0]}/8和{layer_zero_attention_counts[1]}/8。这定位的是已保存的空间均值分支，不代表已识别具体位置编码或上下文机制。", "",
             f"层13 post_ffn的配对静态/真实时间能量比中位数为{paired[('static_first', 13, 'post_ffn')]['temporal_static_over_real']['median']:.5f}与{paired[('static_last', 13, 'post_ffn')]['temporal_static_over_real']['median']:.5f}；这两个静态参考不能作为可直接相减的运动成分估计。", "",
             "所有数字为8视频的中位数[实际最小值,最大值]；不是置信区间。绝对时间能量是在空间均值轨迹上去时间均值后的均方值，DC单独报告，避免总能量归一化造成误读。", "",
             "## 绝对去均值时间能量", "",
             "| 层 | 分支 | 真实 | 重复首帧 | 重复末帧 | 首/末高于数值底限视频数 |",
             "| ---: | --- | --- | --- | --- | --- |"]
    for layer in LAYERS:
        for branch in BRANCHES:
            real, first, last = [indexed[(condition, layer, branch)] for condition in ("real", *CONTROLS)]
            lines.append(f"| {layer} | {branch} | {format_stats(real['absolute_temporal_energy'])} | {format_stats(first['absolute_temporal_energy'])} | {format_stats(last['absolute_temporal_energy'])} | {first['above_numerical_floor_count']}/8; {last['above_numerical_floor_count']}/8 |")
    lines.extend(["", "## DC能量", "", "| 层 | 分支 | 真实 | 重复首帧 | 重复末帧 |", "| ---: | --- | --- | --- | --- |"])
    for layer in LAYERS:
        for branch in BRANCHES:
            values = [indexed[(condition, layer, branch)] for condition in ("real", *CONTROLS)]
            lines.append(f"| {layer} | {branch} | " + " | ".join(format_stats(row["dc_energy"]) for row in values) + " |")
    lines.extend(["", "## 配对静态/真实能量比", "",
                  "先在每个视频内取比，再汇总；不将比值解释为背景或运动比例。", "",
                  "| 层 | 分支 | 首帧 时间比 | 末帧 时间比 | 首帧 DC比 | 末帧 DC比 |",
                  "| ---: | --- | --- | --- | --- | --- |"])
    for layer in LAYERS:
        for branch in BRANCHES:
            first, last = [paired[(condition, layer, branch)] for condition in CONTROLS]
            values = [first["temporal_static_over_real"], last["temporal_static_over_real"], first["dc_static_over_real"], last["dc_static_over_real"]]
            lines.append(f"| {layer} | {branch} | " + " | ".join(format_stats(value) for value in values) + " |")
    lines.extend(["", "## 可作与不可作的解释", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "## 局部空间信息是否可用", "",
                  "前向hook处理的视觉张量由已验证网格对应为[8,60,2048]，但存盘只有[8,2048]空间均值及[8]二阶矩/最大token RMS。没有逐token轨迹，也没有最大RMS对应的索引。",
                  "因此本轮不能从现有文件恢复局部空间频谱、patch时间能量或运动位置。未来若另行提取，可在mean(dim=1)之前保存[T,P,D]并核验patch合并顺序与网格映射；同一空间索引不等于同一物体轨迹。", "",
                  "重复静态帧仍出现非零隐状态时间能量，只能说明该谱包含模型处理位置/时间/上下文等因素的可能贡献，不能把它全部称为运动，更不能据此解释旧layer0selector失败。", "",
                  "## 复现", "",
                  "python -m spectral_memory.report_static_null --run-dir outputs/causal_static_null_v2 --output-dir outputs/causal_static_null_v2/analysis",
                  "", f"分析源码SHA-256: {report['integrity']['analysis_source_sha256']}", "",
                  "逐视频绝对值、配对比值、完整来源SHA及全部数组shape/dtype见report.json。"])
    return "\n".join(lines) + "\n"


def self_checks():
    constant = np.full((8, 2048), 2.0)
    fixed = energies(constant)
    require(fixed["absolute_temporal_energy"] == 0 and fixed["dc_energy"] == 4, "Constant check failed")
    wave = np.sin(2 * np.pi * np.arange(8) / 8)[:, None] * np.ones((1, 2048))
    original, scaled = energies(wave + 3), energies(2 * (wave + 3))
    require(np.isclose(original["absolute_temporal_energy"], 0.5) and np.isclose(original["dc_energy"], 9), "Sinusoid energy check failed")
    require(np.isclose(scaled["absolute_temporal_energy"], 4 * original["absolute_temporal_energy"]), "Scaling check failed")
    require(ratio(0, 0) is None and ratio(0, 1) == 0, "Zero-denominator check failed")
    return ["constant temporal energy zero", "DC separated from sinusoid", "quadratic amplitude scaling", "zero-denominator remains undefined"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/causal_static_null_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/causal_static_null_v2/analysis"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    checks = self_checks()
    report = audit(root, (root / arguments.run_dir).resolve())
    output = (root / arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    (output / "checks.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": len(report["records"]),
                      "identities": report["integrity"]["tokenwise_residual_identity_checks"],
                      "checks": len(checks), "parseval": report["integrity"]["maximum_parseval_error"]}, indent=2))


if __name__ == "__main__":
    main()
