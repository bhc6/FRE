"""Audit archived training-tail replay numerics without running a model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .report_learned_controller import PROTOCOL_SHA256, SEEDS, digest, read_json, read_rows, require, write_json, write_rows


def summarize(native, replay, identities, recorded_sizes):
    error = replay - native
    margins = error.max(axis=-1) - error.min(axis=-1)
    changed = replay.argmax(axis=-1) != native.argmax(axis=-1)
    return {"cases": len(native), "maximum_absolute_option_logit_error": float(np.max(np.abs(error))),
            "pooled_option_logit_rms_error": float(np.sqrt(np.mean(error**2))),
            "maximum_per_case_option_logit_rms_error": float(np.max(np.sqrt(np.mean(error**2, axis=-1)))),
            "maximum_pairwise_option_margin_error": float(np.max(margins)),
            "prediction_changes": int(changed.sum()), "prediction_change_fraction": float(changed.mean()),
            "changed_case_ids": [case_id for case_id, flip in zip(identities, changed) if flip],
            "exact_option_ties_native": int(sum(np.count_nonzero(row == row.max()) > 1 for row in native)),
            "exact_option_ties_replay": int(sum(np.count_nonzero(row == row.max()) > 1 for row in replay)),
            "actual_replay_batch_sizes_by_case": dict(Counter(str(size) for size in recorded_sizes))}


def run(arguments):
    require(not arguments.output_dir.exists(), "Replay analysis output must be new")
    details = []
    summaries = []
    sources = []
    seeds = set()
    for directory in arguments.training_dirs:
        config = read_json(directory / "config.json")
        seed = config["seed"]
        require(seed in SEEDS and seed not in seeds, "Training seed changed or duplicated")
        seeds.add(seed)
        require(config["status"] == "complete" and config["protocol_sha256"] == PROTOCOL_SHA256, "Training run not frozen and complete")
        path = directory / "replay_parity.json"
        require(digest(path) == config["replay_parity_sha256"], "Replay parity archive hash changed")
        cached = {}
        for cache in config["data_provenance"]["caches"]:
            cache_path = Path(cache["directory"]) / "records.jsonl"
            require(digest(cache_path) == cache["records_sha256"], "Original native teacher records changed")
            for row in read_rows(cache_path):
                require(row["case_id"] not in cached, "Native teacher case duplicated")
                cached[row["case_id"]] = row
        parity = read_json(path)
        require(len(parity) == len(cached) == 206 and {row["case_id"] for row in parity} == set(cached), "Replay case coverage incomplete")
        sources.append({"training_directory": str(directory.resolve()), "seed": seed, "replay_parity_sha256": digest(path),
                        "config_sha256": digest(directory / "config.json")})
        for mode, prefix in (("batch1", "batch1_"), ("batch8", "")):
            native = np.asarray([row["native_option_logits"] for row in parity], dtype=np.float64)
            replay = np.asarray([row[mode + "_option_logits"] for row in parity], dtype=np.float64)
            require(native.shape == replay.shape == (206, 3) and np.isfinite(native).all() and np.isfinite(replay).all(), "Replay logits invalid")
            for position, record in enumerate(parity):
                require(np.array_equal(native[position], cached[record["case_id"]]["option_logits"]), "Teacher does not match original normal forward")
                error = replay[position] - native[position]
                expected = {"maximum_option_logit_difference": float(np.max(np.abs(error))),
                            "rms_option_logit_difference": float(np.sqrt(np.mean(error**2))),
                            "maximum_pairwise_margin_difference": float(np.ptp(error)),
                            "prediction_changed": bool(replay[position].argmax() != native[position].argmax())}
                for field, value in expected.items():
                    require(bool(np.isclose(record[prefix + field], value, rtol=1e-6, atol=1e-10)),
                            f"Stored FP32 replay diagnostic failed FP64 recomputation: {record['case_id']} {mode} {field}")
                details.append({"seed": seed, "replay_mode": mode, "case_id": record["case_id"],
                                "condition": cached[record["case_id"]]["condition"],
                                "actual_replay_batch_size": 1 if mode == "batch1" else record["replay_batch_size"],
                                "native_option_logits": native[position].tolist(), "replayed_option_logits": replay[position].tolist(), **expected})
            for subset in ("all", "real", "static_no_gold"):
                selected = [position for position, record in enumerate(parity) if subset == "all" or (cached[record["case_id"]]["supervised"] if subset == "real" else not cached[record["case_id"]]["supervised"])]
                summaries.append({"seed": seed, "replay_mode": mode, "subset": subset,
                                  **summarize(native[selected], replay[selected], [parity[position]["case_id"] for position in selected],
                                              [1 if mode == "batch1" else parity[position]["replay_batch_size"] for position in selected])})
    require(seeds == set(SEEDS), "Report all three fixed training seeds")
    arguments.output_dir.mkdir(parents=True)
    write_rows(arguments.output_dir / "per_case_replay.jsonl", details)
    write_rows(arguments.output_dir / "summary.jsonl", summaries)
    report = {"status": "complete", "analyzer_sha256": digest(__file__), "protocol_sha256": PROTOCOL_SHA256,
              "independent_recomputation": "FP64 from native, batch1 and batch8 three-option logits; native matched to original cache records",
              "diagnostic_serialization_tolerance": "Recorded FP32 reductions compared to independent FP64 with rtol=1e-6 and atol=1e-10; this is not a replay performance acceptance threshold",
              "sources": sources, "records": len(details), "unique_native_cases": 206, "summaries": summaries,
              "gpu_used": False, "tolerance_policy": "No performance-error acceptance threshold; finite differentiable replay only, per frozen protocol",
              "limitation": "This is zero-controller cached-tail numerical replay; final performance is scored by complete model forward. Repeated seeds reuse the same 206 frozen native inputs."}
    write_json(arguments.output_dir / "report.json", report)
    lines = ["# 冻结尾层重放：独立CPU数值审计", "",
             "从归档的native/batch1/batch8三选项logits以FP64复算，并将native逐项核对到原普通forward缓存。每seed同206输入（真实142、无金标静态64），不是618个独立样本。", "",
             "| seed | 重放模式 | 子集 | 样本 | 最大logit误差 | 合并RMS | 最大margin误差 | argmax变化 |", "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in summaries:
        lines.append(f"| {item['seed']} | {item['replay_mode']} | {item['subset']} | {item['cases']} | {item['maximum_absolute_option_logit_error']:.6g} | {item['pooled_option_logit_rms_error']:.6g} | {item['maximum_pairwise_option_margin_error']:.6g} | {item['prediction_changes']}/{item['cases']} |")
    lines.extend(["", "RMS对所有选项误差平方汇总后开方；margin误差为同一题任意两选项差值误差的最大绝对值。精确平局按原选项次序argmax，平局数量及全部flip的case ID见JSON。", "",
                  "batch8名称指重放批处理策略；尾批实际为6，逐case真实batch size已公开。三seed共享相同缓存，因此重复数值结果不能被作为独立确认。", "",
                  "冻结协议未设回放误差或flip容差，只要求前向与梯度有限可用；这些近似训练分数不进入论文主表。全部最终validation/test/endpoint评分由完整模型普通forward执行。", ""])
    (arguments.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print({"status": "complete", "records": len(details), "output": str(arguments.output_dir.resolve())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
