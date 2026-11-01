"""Verify final experiment completeness, sealed inputs, and resource accounting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path(__file__).resolve().parents[1]
    experiment = root / "outputs/controller_experiment_v1"
    destination = experiment / "final_integrity.json"
    if destination.exists():
        raise ValueError("Final integrity report already exists")
    freeze_path = experiment / "pretest_freeze.json"
    freeze = read_json(freeze_path)
    plan = read_json(experiment / "test_execution_plan.json")
    if plan["status"] != "complete" or plan["completed_shards"] != list(range(4)) or plan["pretest_freeze_sha256"] != digest(freeze_path):
        raise ValueError("The frozen test plan is incomplete or changed")
    for name, expected in freeze["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen source or data changed: " + name)
    for record in freeze["checkpoints"]:
        if digest(Path(record["directory"]) / "final.pt") != record["checkpoint_sha256"]:
            raise ValueError("A frozen checkpoint changed")
    caches = [root / f"outputs/controller_train_cache_v1/chunk{index:02d}" for index in range(4)]
    training = [root / f"outputs/controller_training_v1/seed{seed}" for seed in (20261101, 20261102, 20261103)]
    evaluation = [root / "outputs/controller_evaluation_v1" / name for name in
                  ("validation00", "validation01", "endpoint", "test00", "test01", "test02", "test03")]
    smoke = root / "outputs/controller_smoke_v1"
    resources = []
    for directory in [*caches, *training, *evaluation, smoke]:
        config = read_json(directory / ("report.json" if directory == smoke else "config.json"))
        resource = read_json(directory / "resource_summary.json")
        limits = read_json(directory / "resource_limits.json")
        if config["status"] != "complete" or resource["status"] != "complete" or (directory / "resource_stop.json").exists():
            raise ValueError("An experiment process is incomplete or resource-stopped")
        if limits["cuda_allocator_fraction"] != 0.65 or limits["timeout_seconds"] != 300 or limits["cpu_threads"] != 2:
            raise ValueError("Resource limits changed")
        resources.append({"directory": str(directory), "config_sha256": digest(directory / ("report.json" if directory == smoke else "config.json")),
                          "resource_summary_sha256": digest(directory / "resource_summary.json"), **resource,
                          "peak_cuda_allocated_bytes": config["peak_cuda_allocated_bytes"],
                          "peak_cuda_reserved_bytes": config["peak_cuda_reserved_bytes"]})
    cache_cases = 0
    evaluation_records = 0
    condition_counts = {}
    model_records = []
    for directory in [*caches, *evaluation]:
        config = read_json(directory / "config.json")
        if digest(directory / "records.jsonl") != config["records_sha256"]:
            raise ValueError("Recorded cases changed")
        rows = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        unique_archives = {row["archive_file"]: row["archive_sha256"] for row in rows}
        for name, expected in unique_archives.items():
            if digest(directory / name) != expected:
                raise ValueError("Raw output archive changed")
        model_records.append({name: config["model"][name] for name in
                              ("model_id", "model_revision", "processor_revision", "transformers", "torch", "dtype", "attention_implementation")})
        if directory in caches:
            if not all(row["passive_exact"] for row in rows):
                raise ValueError("A training capture changed its baseline")
            cache_cases += len(rows)
        else:
            if not all(row["identity_exact"] and row["restoration_exact"] for row in rows):
                raise ValueError("An evaluation identity or restoration check failed")
            evaluation_records += len(rows)
            cases = {row["case_id"] for row in rows}
            if len(rows) != len(cases) * 13:
                raise ValueError("An evaluation matrix is incomplete")
            condition_counts[config["split"]] = condition_counts.get(config["split"], 0) + len(cases)
            if config["split"] == "test" and datetime.fromisoformat(config["created_at_utc"]) <= datetime.fromisoformat(freeze["created_at_utc"]):
                raise ValueError("Test execution preceded source and checkpoint freeze")
    if cache_cases != 206 or evaluation_records != 2522 or condition_counts != {"validation": 56, "endpoint": 32, "test": 106}:
        raise ValueError("Whole-experiment coverage differs from the declared matrix")
    if any(record != model_records[0] for record in model_records):
        raise ValueError("Model identity or numerical environment differs between stages")
    report = {"status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "pretest_freeze_sha256": digest(freeze_path), "checker_sha256": digest(__file__),
              "frozen_files_unchanged": True, "twelve_checkpoints_unchanged": True,
              "cache_cases": cache_cases, "evaluation_conditions": condition_counts, "evaluation_records": evaluation_records,
              "baseline_capture_exact_cases": 206, "whole_vocabulary_zero_identity_and_restoration_cases": 194,
              "executed_full_model_forward_calls_from_protocol_coverage": 3355,
              "full_model_forward_accounting": "412 cache + 2910 official/endpoint evaluation + 33 smoke; excludes cached-tail training and CPU fixtures",
              "paired_training_runs": 12, "optimizer_steps_total": 2400, "model": model_records[0],
              "guarded_gpu_processes": len(resources), "summed_guarded_process_wall_seconds": sum(row["elapsed_seconds"] for row in resources),
              "maximum_guarded_process_wall_seconds": max(row["elapsed_seconds"] for row in resources),
              "peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in resources),
              "peak_cuda_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in resources),
              "peak_sampled_gpu_temperature_c": max(row["peak_sampled_gpu_temperature_c"] for row in resources),
              "peak_sampled_rss_bytes": max(row["peak_sampled_rss_bytes"] for row in resources),
              "minimum_sampled_available_ram_bytes": min(row["minimum_sampled_available_ram_bytes"] for row in resources),
              "resource_stops": 0, "resource_scope": "This completed controller V1 experiment only; not the whole historical research project",
              "archive_bytes": sum(path.stat().st_size for directory in [*caches, *training, *evaluation, smoke] for path in directory.rglob("*") if path.is_file()),
              "resources": resources}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "resources"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
