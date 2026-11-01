"""Validate and summarize the eight-video FFN development probe without inference."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


DEVELOPMENT_IDS = {
    "video_10993", "video_1580", "video_2564", "video_3501",
    "video_6860", "video_8241", "video_874", "video_9253",
}
LAYERS = (0, 13, 27)
BRANCHES = (
    "decoder_input", "attention_delta", "pre_ffn",
    "ffn_input", "ffn_delta", "post_ffn",
)
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
METRICS = (
    "temporal_energy_ratio", "dc_energy_ratio", "temporal_fraction_ratio",
    "total_energy_ratio", "temporal_fraction_difference_pp",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def branch_ratios(before, after):
    before_total = before["temporal"] + before["dc"]
    after_total = after["temporal"] + after["dc"]
    before_fraction = ratio(before["temporal"], before_total)
    after_fraction = ratio(after["temporal"], after_total)
    temporal_ratio = ratio(after["temporal"], before["temporal"])
    total_ratio = ratio(after_total, before_total)
    fraction_ratio = ratio(after_fraction, before_fraction) if before_fraction is not None and after_fraction is not None else None
    identity_error = None
    if fraction_ratio is not None and temporal_ratio is not None and total_ratio is not None and total_ratio > 0:
        identity_error = abs(fraction_ratio - temporal_ratio / total_ratio)
        require(identity_error <= 1e-10 * max(1.0, abs(fraction_ratio)), "Normalized energy ratio identity failed")
    return {
        "temporal_energy_ratio": temporal_ratio,
        "dc_energy_ratio": ratio(after["dc"], before["dc"]),
        "temporal_fraction_ratio": fraction_ratio,
        "total_energy_ratio": total_ratio,
        "temporal_fraction_before": before_fraction,
        "temporal_fraction_after": after_fraction,
        "temporal_fraction_difference_pp": 100 * (after_fraction - before_fraction)
        if before_fraction is not None and after_fraction is not None else None,
        "fraction_ratio_identity_absolute_error": identity_error,
        "fraction_rises_while_temporal_energy_does_not": bool(
            fraction_ratio is not None and temporal_ratio is not None and fraction_ratio > 1 and temporal_ratio <= 1
        ),
        "fraction_rises_with_dc_decrease": bool(
            fraction_ratio is not None and fraction_ratio > 1 and after["dc"] < before["dc"]
        ),
    }


def summary(values):
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    require(np.isfinite(finite).all(), "Nonfinite descriptive value")
    if not len(finite):
        return {"count": 0, "median": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(finite)), "median": float(np.median(finite)),
        "minimum": float(finite.min()), "maximum": float(finite.max()),
    }


def validate_probe(directory, source_manifest):
    directory = Path(directory).resolve()
    probe = read_json(directory / "probe.json")
    analysis = read_json(directory / "analysis" / "report.json")
    require(probe["status"] == "complete" and analysis["status"] == "complete", f"Incomplete probe: {directory}")
    video_id = probe["video_id"]
    require(video_id in DEVELOPMENT_IDS, f"Unexpected development video: {video_id}")
    require(probe["layers"] == list(LAYERS) and set(analysis["layers"]) == {str(layer) for layer in LAYERS},
            "Layer list differs from the declared development protocol")
    require(probe["frame_count"] == 16 and probe["questions_or_answers_used"] is False,
            "Probe must use sixteen frames without question/answer labels")
    require(probe["instruction"] == "Describe the visible events.", "Extraction instruction changed")
    require(probe["source_manifest_sha256"] == digest(source_manifest), "Development source manifest changed")
    for name in ("model", "batch"):
        metadata = probe[name]
        require(metadata["model_id"] == MODEL and metadata["model_revision"] == REVISION,
                "Model identity or revision changed")
        require(metadata["processor_revision"] == REVISION and metadata["dtype"] == "torch.bfloat16",
                "Processor revision or compute dtype changed")
        require(str(metadata["device"]).startswith("cuda"), "Recorded probe did not use the declared CUDA device")
        require(metadata["transformers"] == "4.57.1" and metadata["attention_implementation"] == "sdpa",
                "Verified inference implementation changed")
    batch = probe["batch"]
    require(batch["uniform_hidden_sampling"] is True and batch["processor_resamples_frames"] is False,
            "Processor must preserve the explicit uniform time grid")
    require(batch["max_pixels_per_frame"] == 65536 and batch["temporal_patch_size"] == 2,
            "Pixel budget or temporal patch size changed")
    require(batch["input_tokens"] <= 1024 and batch["visual_tokens"] <= 512,
            "Probe exceeds its recorded engineering input limits")
    grid = np.asarray(batch["video_grid_thw"])
    require(grid.shape == (3,) and grid[0] == 8 and batch["spatial_merge_size"] == 2
            and batch["patch_size"] == 16 and np.all(grid[1:] % 2 == 0),
            "Unexpected merged visual-token grid")
    require(batch["visual_tokens"] == int(grid.prod() // 4), "Visual-token count disagrees with the grid")
    target_times = np.asarray(probe["target_frame_timestamps_s"], dtype=np.float64)
    source_times = np.asarray(probe["source_frame_timestamps_s"], dtype=np.float64)
    require(target_times.shape == (16,) and source_times.shape == (16,), "Frame timestamp count mismatch")
    require(np.isfinite(target_times).all() and np.isfinite(source_times).all(), "Nonfinite frame times")
    require(np.all(np.diff(target_times) > 0) and np.all(np.diff(source_times) > 0), "Non-increasing frame times")
    target_intervals = np.diff(target_times)
    require(np.allclose(target_intervals, target_intervals.mean(), rtol=1e-5, atol=1e-8),
            "Target frame time grid is nonuniform")
    hidden_times = target_times.reshape(8, 2).mean(axis=1)
    require(np.allclose(hidden_times, batch["hidden_timestamps_s"], atol=1e-8, rtol=1e-8),
            "Hidden timestamps do not match pair centers")
    require(batch["frame_groups"] == np.arange(16).reshape(8, 2).tolist(), "Frame-pair mapping changed")
    require(probe["frames_manifest_sha256"] == digest(directory / "frames.manifest.jsonl"),
            "Prepared frame manifest changed")
    frames = [json.loads(line) for line in (directory / "frames.manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    require(len(frames) == 1 and frames[0]["video_id"] == video_id, "Prepared frame identity mismatch")
    source_records = [json.loads(line) for line in Path(source_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    source_matches = [record for record in source_records if record["video_id"] == video_id]
    require(len(source_matches) == 1, "Source manifest must identify exactly one video")
    source_record = source_matches[0]
    source_video = Path(source_record["video_path"])
    if not source_video.is_absolute():
        source_video = Path(source_manifest).resolve().parent / source_video
    require(Path(frames[0]["video_path"]).resolve() == source_video.resolve(), "Prepared source video path changed")
    require(digest(source_video) == frames[0]["video_sha256"] == source_record["video_sha256"],
            "Actual source video bytes differ from the development manifest")
    require(len(frames[0]["frames"]) == 16 and len(probe["frame_sha256"]) == 16, "Prepared frame count mismatch")
    require([digest(directory / filename) for filename in frames[0]["frames"]] == probe["frame_sha256"],
            "Actual frame bytes differ from the extracted inputs")
    for actual, expected, name in (
        (frames[0]["timestamps_s"], target_times, "target"),
        (frames[0]["source_frame_timestamps_s"], source_times, "source"),
        (batch["source_timestamps_s"], target_times, "processor target"),
    ):
        require(np.allclose(actual, expected, rtol=1e-8, atol=1e-8), f"{name} frame timestamp mismatch")
    branch_hash = digest(directory / "branches.npz")
    require(probe["branches_sha256"] == branch_hash == analysis["input_sha256"], "Branch archive hash mismatch")
    snapshot = directory / "source_snapshot"
    source_hashes = {}
    for filename, expected_hash in (
        ("probe_ffn.py", probe["probe_source_sha256"]),
        ("resource_guard.py", probe["resource_guard_source_sha256"]),
        ("analyze_ffn.py", analysis["analysis_source_sha256"]),
    ):
        require(digest(snapshot / filename) == expected_hash, f"Source snapshot mismatch: {filename}")
        source_hashes[filename] = expected_hash
    rows = []
    with np.load(directory / "branches.npz", allow_pickle=False) as archive:
        require(np.allclose(archive["timestamps_s"], hidden_times, rtol=1e-8, atol=1e-8),
                "NPZ hidden timestamps mismatch")
        require(analysis["layout"]["samples"] == 8, "Analysis sample count changed")
        for layer in LAYERS:
            layer_report = analysis["layers"][str(layer)]
            arrays = {}
            energies = {}
            for branch in BRANCHES:
                values = np.asarray(archive[f"layer_{layer}_{branch}"], dtype=np.float64)
                require(values.shape == (8, 2048) and np.isfinite(values).all(),
                        "Invalid branch mean array")
                arrays[branch] = values
                energies[branch] = {
                    "temporal": float(np.mean((values - values.mean(axis=0, keepdims=True))**2)),
                    "dc": float(np.mean(values.mean(axis=0)**2)),
                }
                for metric in ("temporal", "dc"):
                    require(np.isclose(energies[branch][metric], layer_report["branches"][branch]["energy"][metric],
                                       rtol=1e-10, atol=1e-12), f"Saved analysis {metric} differs from direct recomputation")
            shapes = {values.shape for values in arrays.values()}
            require(len(shapes) == 1, "Branch shapes differ within a layer")
            hidden_dimension = arrays["pre_ffn"].shape[1]
            expected_elements = batch["visual_tokens"] * hidden_dimension
            for path in ("attention", "ffn"):
                identity = probe["identities"][f"layer_{layer}_{path}"]
                require(identity["exact_in_observed_dtype"] is True and identity["maximum_absolute_error"] == 0,
                        "Tokenwise original-dtype residual identity did not pass")
                require(identity["dtype"] == "torch.bfloat16" and identity["elements_checked"] == expected_elements,
                        "Residual check does not cover the expected visual token coordinates")
            for path, before, after in (
                ("FFN", "pre_ffn", "post_ffn"), ("Attention", "decoder_input", "pre_ffn"),
            ):
                decomposition = layer_report[f"{path.lower()}_decomposition"]["scopes"]
                for scope in decomposition.values():
                    require(scope["closure_absolute_error"] <= 1e-9 * max(1.0, abs(scope["observed_energy"])),
                            "Spectral residual decomposition did not close")
                rows.append({
                    "video_id": video_id, "layer": layer, "path": path,
                    "before": energies[before], "after": energies[after],
                    **branch_ratios(energies[before], energies[after]),
                    "dc_base_increment_cross_term": decomposition["dc"]["base_increment_cross_term"],
                    "temporal_base_increment_cross_term": decomposition["temporal"]["base_increment_cross_term"],
                    "dc_increment_energy": decomposition["dc"]["increment_energy"],
                    "temporal_increment_energy": decomposition["temporal"]["increment_energy"],
                    "dc_rounding_total_contribution": decomposition["dc"]["rounding_total_contribution"],
                    "temporal_rounding_total_contribution": decomposition["temporal"]["rounding_total_contribution"],
                })
    resource_summary = read_json(directory / "resource_summary.json")
    require(resource_summary["status"] == "complete" and resource_summary["exception"] is None,
            "Resource guard did not finish cleanly")
    limits = read_json(directory / "resource_limits.json")
    require(limits["cpu_threads"] == 2 and limits["cuda_allocator_fraction"] == 0.65
            and limits["timeout_seconds"] == 300 and limits["monitor_interval_seconds"] == 2,
            "Declared resource limits changed")
    provenance = {
        "directory": str(directory), "video_id": video_id, "source_hashes": source_hashes,
        "probe_sha256": digest(directory / "probe.json"), "branches_sha256": branch_hash,
        "analysis_report_sha256": digest(directory / "analysis" / "report.json"),
        "analysis_bin_spacing_hz": analysis["layout"]["bin_spacing_hz"],
        "forward_seconds": probe["forward_seconds"],
        "peak_cuda_allocated_bytes": probe["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": probe["peak_cuda_reserved_bytes"],
        "source_to_target_maximum_error_s": probe["source_to_target_maximum_error_s"],
        "resource_summary": resource_summary, "resource_limits": limits,
    }
    return rows, provenance


def aggregate(rows, provenance):
    video_ids = [value["video_id"] for value in provenance]
    require(len(video_ids) == len(set(video_ids)), "Duplicate development video")
    require(set(video_ids) == DEVELOPMENT_IDS, "Final aggregation requires all eight original development videos")
    require(len(rows) == 48, "Expected eight videos × three layers × two residual paths")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layer"], row["path"])].append(row)
    summaries = []
    for (layer, path), group in sorted(grouped.items()):
        require(len(group) == 8, "Incomplete per-layer path")
        summaries.append({
            "layer": layer, "path": path, "videos": len(group),
            "metrics": {metric: summary([row[metric] for row in group]) for metric in METRICS},
            "fraction_rises_with_dc_decrease_count": sum(row["fraction_rises_with_dc_decrease"] for row in group),
            "fraction_rises_without_temporal_energy_increase_count": sum(
                row["fraction_rises_while_temporal_energy_does_not"] for row in group),
        })
    source_versions = {
        filename: sorted({value["source_hashes"][filename] for value in provenance})
        for filename in ("probe_ffn.py", "resource_guard.py", "analyze_ffn.py")
    }
    return {
        "status": "complete", "scope": "eight_original_development_videos_descriptive_mechanism_diagnostic",
        "video_ids": sorted(video_ids), "layers": list(LAYERS), "model": MODEL, "revision": REVISION,
        "dtype": "torch.bfloat16", "frames_per_video": 16, "temporal_positions_per_video": 8,
        "records": sorted(rows, key=lambda value: (value["layer"], value["video_id"], value["path"])),
        "summaries": summaries, "provenance": provenance, "source_versions": source_versions,
        "source_versions_identical": all(len(values) == 1 for values in source_versions.values()),
        "observed_resources": {
            "maximum_cuda_allocated_bytes": max(value["peak_cuda_allocated_bytes"] for value in provenance),
            "maximum_cuda_reserved_bytes": max(value["peak_cuda_reserved_bytes"] for value in provenance),
            "maximum_sampled_gpu_temperature_c": max(
                value["resource_summary"]["peak_sampled_gpu_temperature_c"] for value in provenance),
            "maximum_sampled_process_rss_bytes": max(
                value["resource_summary"]["peak_sampled_rss_bytes"] for value in provenance),
            "minimum_sampled_available_ram_bytes": min(
                value["resource_summary"]["minimum_sampled_available_ram_bytes"] for value in provenance),
            "forward_seconds": summary([value["forward_seconds"] for value in provenance]),
            "scope": (
                "Largest observed allocator peaks across these runs; maximum/minimum sampled telemetry across runs. "
                "Two-second telemetry may miss short peaks. CUDA allocated/reserved is not driver-wide VRAM usage."
            ),
        },
        "report_source_sha256": digest(__file__),
        "ratio_identity": "Temporal fraction post/pre = temporal energy post/pre divided by total energy post/pre.",
        "limits": [
            "Eight original development videos, not independent confirmation and not a task-effectiveness test.",
            "Only medians and observed min-max ranges are reported; no significance test or confidence interval.",
            "A normalized temporal fraction increase can reflect DC cancellation instead of stronger temporal signal.",
            "The frozen v1 spectral_balanced objective already excludes DC; this diagnosis does not explain its failure or demonstrate a fix.",
            "DC is the trajectory time mean, not a semantic label for background or long-term information.",
            "Eight temporal positions give four non-DC bins, with clip-dependent physical frequencies.",
            "Recorded source snapshots are individually verified; differences are retained instead of claiming identical code.",
            "Probe, resource guard and analysis snapshots are verified; this does not independently hash every imported dependency.",
            "Resource guard success and sampled telemetry do not establish that prior operating-system instability is fixed.",
        ],
    }


def format_value(value):
    return "undefined" if value is None else f"{value:.4f}"


def render_report(report):
    late_ffn = next(value for value in report["summaries"] if value["layer"] == 27 and value["path"] == "FFN")
    late_metrics = late_ffn["metrics"]
    lines = [
        "# 原八视频：FFN 与注意力残差的开发诊断", "",
        "本报告使用原 8 个开发视频，每个 16 帧、8 个隐藏时间位置，固定 Qwen3-VL-2B 的 decoder 0/13/27 层。"
        "只描述表示变化，不进行显著性检验，不评估问答或记忆收益。", "",
        "源视频、帧字节、模型/revision/BF16、时间网格、原 dtype 的 token 级残差恒等式、NPZ 与源码快照均已逐项核查。"
        "各次源码版本见 JSON；首个工程试点与后续强化检查的源码并不完全相同。", "",
        f"最后一层 FFN 的时间占比中位数变为 {late_metrics['temporal_fraction_ratio']['median']:.3f} 倍，"
        f"但绝对时间能量仅变为 {late_metrics['temporal_energy_ratio']['median']:.3f} 倍，"
        f"DC 变为 {late_metrics['dc_energy_ratio']['median']:.3f} 倍。"
        f"全部 {late_ffn['fraction_rises_with_dc_decrease_count']} 个视频都出现占比上升伴随 DC 下降；"
        f"其中 {late_ffn['fraction_rises_without_temporal_energy_increase_count']} 个视频的绝对时间能量没有上升。"
        "这提示归一化占比受 DC 抵消强烈影响，不能把占比倍数当成有用动态信息增益。", "",
        "这针对的是把 DC 纳入分母的诊断指标。冻结 v1 的 spectral_balanced 目标已经去均值、排除 DC；"
        "**当前观察不能解释 v1 的失败，也没有证明某个改动能够修复它。**", "",
        "## 中位数与实际范围", "",
        "每格为中位数 [最小值, 最大值]；这些范围不是置信区间。比值大于 1 表示残差合并后增大。", "",
        "| 层 | 路径 | 时间能量 post/pre | DC post/pre | 时间占比 post/pre | 总能量 post/pre | 时间占比变化 pp |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for value in report["summaries"]:
        formatted = []
        for metric in METRICS:
            stats = value["metrics"][metric]
            formatted.append(f"{format_value(stats['median'])} [{format_value(stats['minimum'])}, {format_value(stats['maximum'])}]")
        lines.append(f"| {value['layer']} | {value['path']} | " + " | ".join(formatted) + " |")
    resources = report["observed_resources"]
    lines.extend([
        "", "## 归一化占比与 DC 抵消", "",
        "令 V 为去时间均值后的能量，D 为 DC 能量，时间占比为 f=V/(V+D)。精确有 "
        "f_post/f_pre=(V_post/V_pre)/((V_post+D_post)/(V_pre+D_pre))。"
        "因此，占比大幅提高并不要求绝对时间能量大幅提高。", "",
        "| 视频 | 层 | 路径 | 时间能量比 | DC 比 | 总能量比 | 时间占比比 | DC 交叉项 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["records"]:
        lines.append(f"| {row['video_id']} | {row['layer']} | {row['path']} | "
                     f"{format_value(row['temporal_energy_ratio'])} | {format_value(row['dc_energy_ratio'])} | "
                     f"{format_value(row['total_energy_ratio'])} | {format_value(row['temporal_fraction_ratio'])} | "
                     f"{row['dc_base_increment_cross_term']:.6g} |")
    lines.extend([
        "", "DC 交叉项是 2Re〈残差 DC, 增量 DC〉，负值可以抵消原均值分量；"
        "它与增量自身能量和舍入项共同决定输出 DC。不能将 DC 自动命名为背景，也不能把抵消写成删除无用信息。", "",
        "## 解释边界", "",
        "- 每个视频都保留，不筛选有利层或有利样本；没有独立确认、QA 成绩、因果干预或长期记忆收益。",
        "- Attention 与 FFN 使用各自正确残差输入；FFN 的归一化输入不是未归一化残差。",
        "- 原 dtype 恒等式在全部视觉 token 上核查；频谱指标基于空间均值，仍可能隐藏局部动态。",
        "- T=8 只有四个非 DC 频率 bin，物理频率随片段时长变化，不能给三个频带赋予固定语义。",
        "- 资源监控成功不能证明先前系统重启原因已经修复；遥测采样可能遗漏短峰值。",
        "", "## 本次实测资源", "",
        f"- 八次运行中最大 CUDA allocated 峰值：{resources['maximum_cuda_allocated_bytes']:,} bytes；"
        f"reserved 峰值：{resources['maximum_cuda_reserved_bytes']:,} bytes。",
        f"- 2 秒采样观察到的最高 GPU 温度：{resources['maximum_sampled_gpu_temperature_c']}°C；"
        f"最大进程 RSS：{resources['maximum_sampled_process_rss_bytes']:,} bytes；"
        f"最低可用 RAM：{resources['minimum_sampled_available_ram_bytes']:,} bytes。",
        "- 这些是本批运行的已观察值，不是未来运行或整机资源的保证上限。"
        "CUDA allocated/reserved 不包括所有驱动显存；2 秒采样可能遗漏短暂温度及系统内存峰值。",
        "", "![分层能量比](ffn_development_ratios.png)", "",
        "图中小点是全部视频，粗标记是中位数，竖线是实际最小到最大范围；不是标准误或置信区间。",
    ])
    return "\n".join(lines) + "\n"


def make_figure(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    metrics = ("temporal_energy_ratio", "dc_energy_ratio", "temporal_fraction_ratio")
    labels = ("Temporal energy\npost / pre", "DC energy\npost / pre", "Temporal fraction\npost / pre")
    colors = {"FFN": "#2463a3", "Attention": "#bc6125"}
    for axis, layer in zip(axes, LAYERS):
        for path, offset in (("FFN", -0.12), ("Attention", 0.12)):
            group = sorted([row for row in report["records"] if row["layer"] == layer and row["path"] == path],
                           key=lambda value: value["video_id"])
            for position, metric in enumerate(metrics):
                values = np.asarray([row[metric] for row in group], dtype=np.float64)
                require(np.isfinite(values).all() and np.all(values > 0), "Log-ratio plot requires positive defined ratios")
                center = position + offset
                jitter = np.linspace(-0.045, 0.045, len(values))
                axis.scatter(center + jitter, values, color=colors[path], alpha=0.46, s=17, zorder=3)
                axis.vlines(center, values.min(), values.max(), color=colors[path], linewidth=1.3, zorder=2)
                axis.scatter([center], [np.median(values)], marker="_", s=180, linewidths=2.2,
                             color=colors[path], zorder=4)
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
        axis.set_yscale("log", base=2)
        axis.set_xticks(range(3), labels, fontsize=9)
        axis.set_xlim(-0.5, 2.5)
        axis.set_title(f"Decoder layer {layer}", fontsize=11)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Post / pre ratio (log scale)", fontsize=10)
    all_values = [row[metric] for row in report["records"] for metric in metrics]
    lower_power = int(np.floor(np.log2(min(all_values))))
    upper_power = int(np.ceil(np.log2(max(all_values))))
    powers = range(lower_power, upper_power + 1)
    tick_values = [2.0**power for power in powers]
    tick_labels = [str(2**power) if power >= 0 else f"1/{2**(-power)}" for power in powers]
    axes[0].set_yticks(tick_values, tick_labels)
    axes[0].set_ylim(min(all_values) * 0.8, max(all_values) * 1.2)
    handles = [Line2D([0], [0], color=colors[path], marker="o", linestyle="", label=path)
               for path in ("FFN", "Attention")]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    figure.suptitle("Residual-branch effects across eight development videos", fontsize=13, y=0.995)
    figure.text(0.5, 0.035,
                "16 frames / video; 8 temporal groups. Dots: videos; markers: medians; lines: observed ranges. "
                "No significance test or task-benefit claim.",
                ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.07, 1, 0.89))
    figure.savefig(output / "ffn_development_ratios.png", dpi=240)
    figure.savefig(output / "ffn_development_ratios.svg")
    plt.close(figure)


def run_self_checks():
    dc_only = branch_ratios({"temporal": 1.0, "dc": 99.0}, {"temporal": 1.0, "dc": 9.0})
    require(dc_only["temporal_energy_ratio"] == 1.0 and dc_only["temporal_fraction_ratio"] == 10.0,
            "DC-only normalized fraction example failed")
    require(dc_only["fraction_rises_while_temporal_energy_does_not"], "DC confound flag failed")
    scaled = branch_ratios({"temporal": 1.0, "dc": 9.0}, {"temporal": 2.0, "dc": 18.0})
    require(scaled["temporal_energy_ratio"] == 2.0 and scaled["temporal_fraction_ratio"] == 1.0,
            "Common amplitude scaling should preserve temporal fraction")
    zero = branch_ratios({"temporal": 0.0, "dc": 0.0}, {"temporal": 0.0, "dc": 0.0})
    require(zero["temporal_fraction_ratio"] is None, "Zero-signal ratios must be undefined")
    stats = summary([1.0, 2.0, 4.0, None])
    require(stats == {"count": 3, "median": 2.0, "minimum": 1.0, "maximum": 4.0},
            "Median/range summary failed")
    return {"status": "passed", "checks": [
        "DC cancellation can raise normalized fraction without raising temporal energy",
        "Common amplitude scaling preserves temporal fraction",
        "Zero-signal ratios remain undefined",
        "Descriptive median/range excludes undefined values",
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", action="append", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/perception_test_sample/videos.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    directories = arguments.probe_dir or sorted(
        directory for directory in arguments.input_root.glob("ffn_qwen3_dev16*") if (directory / "probe.json").exists()
    )
    require(bool(directories), "No probe directories found")
    checks = run_self_checks()
    rows = []
    provenance = []
    for directory in directories:
        local_rows, local_provenance = validate_probe(directory, arguments.source_manifest)
        rows.extend(local_rows)
        provenance.append(local_provenance)
    video_ids = [value["video_id"] for value in provenance]
    require(len(video_ids) == len(set(video_ids)), "Duplicate development videos")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output_dir / "checks.json", checks)
    if arguments.validate_only:
        result = {
            "status": "validated_inputs_only", "video_ids": sorted(video_ids),
            "complete_development_set": set(video_ids) == DEVELOPMENT_IDS,
            "provenance": provenance, "final_report_generated": False,
        }
        write_json(arguments.output_dir / "input_validation.json", result)
        print(json.dumps({"status": result["status"], "videos": len(video_ids), "final_report_generated": False}))
        return
    report = aggregate(rows, provenance)
    make_figure(report, arguments.output_dir)
    write_json(arguments.output_dir / "report.json", report)
    (arguments.output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "videos": len(video_ids), "records": len(rows),
                      "output": str(arguments.output_dir.resolve())}))


if __name__ == "__main__":
    main()
