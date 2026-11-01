"""Plot frozen confirmation contrasts from completed experiment reports."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_completed(path):
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Plots require completed reader experiments")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--smol", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    qwen = load_completed(arguments.qwen)
    smol = load_completed(arguments.smol)
    coverage = json.loads(arguments.coverage.read_text(encoding="utf-8"))
    if coverage["status"] != "complete":
        raise ValueError("Coverage experiment is incomplete")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        ("Qwen3-VL 2B\nTemporal/state QA: 77 questions, 33 videos", qwen, "temporal_state"),
        ("SmolVLM2 500M\nSame memories and questions; reader transfer", smol, "temporal_state"),
        ("Short-action evidence\n144 observable actions, 30 videos", coverage, "short"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.3), sharex=True)
    source_points = []
    for axis, (title, report, subset) in zip(axes, panels):
        for position, baseline in enumerate(("kcenter", "temporal_centered")):
            candidates = [row for row in report["comparisons"]
                          if row["budget_pairs"] == 4 and row["subset"] == subset
                          and row["baseline"] == baseline
                          and ("metric" not in row or row["metric"] == "observable_hit_rate")]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one frozen contrast")
            row = candidates[0]
            difference = 100 * row["video_macro_difference"]
            lower, upper = np.asarray(row["paired_video_bootstrap_ci95"]) * 100
            axis.errorbar(difference, 1 - position,
                          xerr=[[difference - lower], [upper - difference]],
                          fmt="o", color=("#205a9a", "#a35b12")[position], capsize=5,
                          markersize=7, linewidth=2)
            axis.text(difference, 1 - position + 0.16,
                      f"{difference:+.2f} [{lower:+.2f}, {upper:+.2f}]",
                      ha="center", va="bottom", fontsize=9)
            source_points.append({"panel": title, "baseline": baseline, "difference_pp": difference,
                                  "ci95_pp": [float(lower), float(upper)]})
        axis.axvline(0, color="#606770", linewidth=1, linestyle="--")
        axis.set_yticks([1, 0], ["vs k-center", "vs centered time"])
        axis.set_ylim(-0.45, 1.55)
        axis.set_title(title, fontsize=10, pad=14)
        axis.set_xlabel("Spectral − baseline (percentage points)", fontsize=9)
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Fixed 8-frame memory: paired video-bootstrap 95% intervals",
                    fontsize=13, y=0.97)
    figure.text(0.5, 0.015,
                "40 held-out videos. Unadjusted intervals; only the frozen Qwen comparison is primary. "
                "Short clips do not establish lifelong memory.",
                ha="center", fontsize=8, color="#444444")
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    figure.savefig(arguments.output_dir / "confirmation_contrasts.png", dpi=180)
    figure.savefig(arguments.output_dir / "confirmation_contrasts.svg")
    plt.close(figure)
    sources = {name: {"path": str(path.resolve()),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in (("qwen", arguments.qwen), ("smol", arguments.smol),
                                  ("coverage", arguments.coverage))}
    (arguments.output_dir / "plot_sources.json").write_text(
        json.dumps({"sources": sources, "points": source_points}, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
