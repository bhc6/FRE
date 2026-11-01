"""Render fixed passive-probe summaries without selecting layers or outcomes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report_passive_probe import CONDITIONS, LAYERS, VIDEOS, digest, read_json, stats
from .report_probe_tasks import require


COLORS = {"real": "#2464A8", "static_first": "#D07825", "static_last": "#168278"}
LABELS = {"real": "Real video", "static_first": "Repeated first frame", "static_last": "Repeated last frame"}
BRANCH = "post_ffn"


def interval(values):
    summary = stats(values)
    require(summary["n"] == len(VIDEOS), "Every plotted interval must contain eight video values")
    return summary


def error_arrays(summaries, scale=1.0):
    medians = np.array([row["median"] for row in summaries]) * scale
    minima = np.array([row["min"] for row in summaries]) * scale
    maxima = np.array([row["max"] for row in summaries]) * scale
    return medians, np.stack((medians - minima, maxima - medians))


def make_figure(report):
    require(report["status"] == "complete_cpu_audit" and set(report["video_ids"]) == VIDEOS,
            "The figure requires the complete original eight-video CPU audit")
    require(report["integrity"]["passive_full_vocabulary_checks"] == 60, "Passive matrix is incomplete")
    plots = {"branch_fixed_before_full_analysis": BRANCH, "panel_a": [], "panel_b": [], "panel_c": []}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.9))
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.30, top=0.84, wspace=0.31)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    positions = np.arange(len(LAYERS), dtype=float)
    for condition_index, condition in enumerate(CONDITIONS):
        summaries = []
        for layer in LAYERS:
            values = [row["mean_retention_fraction"] for row in report["spectral_records"]
                      if row["condition"] == condition and row["layer"] == layer and row["branch"] == BRANCH]
            summary = interval(values)
            summaries.append(summary)
            plots["panel_a"].append({"condition": condition, "layer": layer,
                                      "mean_retention_fraction": summary})
        medians, errors = error_arrays(summaries, scale=100)
        axes[0].errorbar(positions + (condition_index - 1) * 0.075, medians, yerr=errors,
                        marker="o", markersize=5, capsize=3, linewidth=1.6,
                        color=COLORS[condition], label=LABELS[condition])
    axes[0].set(title="A  Spatial mean retains only part of variation", xlabel="Decoder layer",
                ylabel="Retained temporal energy (%)", xticks=positions, xticklabels=LAYERS)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    groups = [(kind, condition) for kind in ("q", "k") for condition in CONDITIONS[1:]]
    for stage_index, (stage, label, color) in enumerate((
            ("temporal_before", "Before RoPE (after head norm)", "#8B949E"),
            ("temporal_after", "Actual SDPA input after RoPE", "#7955A3"))):
        summaries = []
        for kind, condition in groups:
            selected = [row for row in report["rope_temporal_records"]
                        if row["condition"] == condition and row["layer"] == 0 and row["kind"] == kind]
            summary = interval([row[stage] for row in selected])
            summaries.append(summary)
            plots["panel_b"].append({"condition": condition, "kind": kind, "stage": stage,
                                      "temporal_energy": summary})
        medians, errors = error_arrays(summaries)
        bar_positions = np.arange(len(groups)) + (stage_index - 0.5) * 0.34
        axes[1].bar(bar_positions, medians, width=0.32, yerr=errors, capsize=3,
                    color=color, label=label, alpha=0.9)
        for position, median in zip(bar_positions, medians):
            if median == 0:
                axes[1].annotate("0", (position, 0), xytext=(0, 4), textcoords="offset points",
                                 ha="center", fontsize=9, color="#4A4A4A")
    axes[1].set(title="B  Static content gains a temporal spectrum", xlabel="Layer 0, repeated frame",
                ylabel="Absolute temporal energy (mean squared)",
                xticks=np.arange(len(groups)), xticklabels=["Q / first", "Q / last", "K / first", "K / last"])
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for condition_index, condition in enumerate(CONDITIONS[1:]):
        for metric_index, (metric, label, linestyle, marker) in enumerate((
                ("full_temporal_static_over_real", "full space", "-", "o"),
                ("mean_temporal_static_over_real", "spatial mean", "--", "s"))):
            summaries = []
            for layer in LAYERS:
                values = [row[metric] for row in report["static_paired_records"]
                          if row["condition"] == condition and row["layer"] == layer and row["branch"] == BRANCH]
                summary = interval(values)
                summaries.append(summary)
                plots["panel_c"].append({"condition": condition, "layer": layer, "metric": metric,
                                          "paired_static_over_real_ratio": summary})
            medians, errors = error_arrays(summaries)
            offset = (condition_index * 2 + metric_index - 1.5) * 0.055
            axes[2].errorbar(positions + offset, medians, yerr=errors, marker=marker,
                            markersize=4, capsize=2.5, linewidth=1.5, linestyle=linestyle,
                            markerfacecolor=COLORS[condition] if metric_index == 0 else "white",
                            color=COLORS[condition], label=f"{'First' if condition_index == 0 else 'Last'} / {label}")
    axes[2].axhline(1, color="#616161", linestyle=":", linewidth=1)
    axes[2].set(title="C  Static / real energy depends on pooling", xlabel="Decoder layer",
                ylabel="Paired temporal-energy ratio", xticks=positions, xticklabels=LAYERS)
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False, fontsize=9, ncols=2, loc="best")
    figure.suptitle("Passive internal diagnostics in Qwen3-VL-2B", fontsize=18, y=0.96)
    figure.text(0.055, 0.165,
                "Eight development videos; T = 8 temporal groups. Markers/bars: video medians; whiskers: observed minimum–maximum (not confidence intervals).",
                fontsize=11)
    figure.text(0.055, 0.12,
                "Panels A/C use the fixed post-FFN residual branch; C summarizes paired video ratios. Q/K heads are averaged at their actual head counts.",
                fontsize=11)
    sampling = report["sampling_summary"]
    sampling_hz, nyquist_hz = sampling["hidden_sampling_hz"], sampling["nyquist_hz"]
    figure.text(0.055, 0.075,
                f"Hidden sampling: {sampling_hz['min']:.3f}–{sampling_hz['max']:.3f} Hz; Nyquist: {nyquist_hz['min']:.3f}–{nyquist_hz['max']:.3f} Hz. Sparse frames, no temporal antialias filter; no continuous-motion frequency claim.",
                fontsize=11)
    figure.text(0.055, 0.03,
                "Descriptive activation diagnostics only: spatial energy loss and static positional spectra do not establish lost task evidence or improved memory/QA.",
                fontsize=11, color="#505050")
    return figure, plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("outputs/passive_probe_v4/analysis/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/passive_probe_v4/analysis/figures"))
    arguments = parser.parse_args()
    report = read_json(arguments.report)
    figure, plotted = make_figure(report)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ("png", "svg"):
        destination = arguments.output_dir / f"passive_internal_diagnostics.{extension}"
        figure.savefig(destination, dpi=180)
        files.append({"path": str(destination.resolve()), "sha256": digest(destination)})
    plt.close(figure)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_report": str(arguments.report.resolve()), "report_sha256": digest(arguments.report),
              "plot_source_sha256": digest(Path(__file__)), "source_analysis_sha256": report["integrity"]["analysis_source_sha256"],
              "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
              "video_ids": sorted(VIDEOS), "temporal_groups": 8,
              "sampling_summary": report["sampling_summary"],
              "uncertainty_display": "video median and observed range; no confidence interval",
              "axis_scales": {"panel_a": "log", "panel_b": "linear", "panel_c": "log"},
              "gpu_used": False, "files": files, "plotted_values": plotted}
    output_record = arguments.output_dir / "figure_record.json"
    output_record.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"files": files, "record": str(output_record.resolve())}, indent=2))


if __name__ == "__main__":
    main()
