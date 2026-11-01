"""Seal the completed controller experiment before any new-test scoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(arguments):
    from data.internal_controller_v1.evaluation_contract import load_evaluation_cases

    root = Path(__file__).resolve().parents[1]
    output = arguments.output.resolve()
    if output.exists():
        raise ValueError("Freeze manifest must be new")
    protocol = root / "docs/learned_controller_protocol_v1.md"
    expected_protocol = "3954767ef2662e15eb4934f90eea08ca6d48eaaef39fd2d297e3d9a3df590748"
    if digest(protocol) != expected_protocol:
        raise ValueError("The predeclared controller protocol changed")
    data_summary = {}
    for split in ("train", "validation", "test"):
        cases = load_evaluation_cases(split)
        data_summary[split] = {"questions": len(cases), "videos": len({case["question"]["video_id"] for case in cases})}
    methods = ("time_mlp", "spectral_mlp", "spectral_phaseblind", "constant_spectral")
    seeds = (20261101, 20261102, 20261103)
    records = []
    for seed in seeds:
        directory = root / f"outputs/controller_training_v1/seed{seed}"
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["status"] != "complete" or config["protocol_sha256"] != expected_protocol or config["seed"] != seed:
            raise ValueError("Training run is incomplete or inconsistent")
        if {record["method"] for record in config["completed_methods"]} != set(methods) or config["training_cases"] != 206:
            raise ValueError("Training coverage differs from the frozen protocol")
        for name, expected in config["source_sha256"].items():
            if digest(root / "spectral_memory" / name) != expected or digest(directory / "source_snapshot" / name) != expected:
                raise ValueError("Training source differs from its saved snapshot")
        sampling = None
        for summary in config["completed_methods"]:
            method_dir = directory / summary["method"]
            for name, field in (("final.pt", "checkpoint_sha256"), ("optimizer.pt", "optimizer_sha256"),
                                ("initial.pt", "initial_state_sha256"), ("training.jsonl", "training_log_sha256")):
                if digest(method_dir / name) != summary[field]:
                    raise ValueError("A training artifact changed after completion")
            training = [json.loads(line) for line in (method_dir / "training.jsonl").read_text(encoding="utf-8").splitlines()]
            plan = [(row["step"], row["case_ids"]) for row in training]
            if [row["step"] for row in training] != list(range(1, 201)) or (sampling is not None and plan != sampling):
                raise ValueError("Training steps or paired sampling differ")
            sampling = plan
            records.append({"directory": str(method_dir), **summary})
    files = [protocol, root / "data/internal_controller_v1/evaluation_contract.py"]
    files.extend(root / "spectral_memory" / name for name in
                 ("controller_freeze.py", "controller_cache.py", "controller_train.py", "controller_runtime.py", "controller_evaluate.py",
                  "learned_readout.py", "resource_guard.py", "qwen3_backend.py", "passive_probe.py", "evaluate_small.py", "endpoint_control.py",
                  "report_learned_controller.py"))
    for split in ("train", "validation", "test"):
        files.extend((root / f"data/internal_controller_v1/{split}.questions.jsonl",
                      root / f"data/internal_controller_v1/frames16/{split}.frames.manifest.jsonl"))
    report = {"status": "frozen_before_new_test", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": expected_protocol, "data": data_summary, "checkpoints": records,
              "files_sha256": {str(path.relative_to(root)): digest(path) for path in files},
              "paired_sampling_verified": True, "all_final_step_200": True,
              "primary": "all_74_video_macro_accuracy_spectral_mlp_minus_time_mlp_three_seed_mean",
              "test_matrix": "106_conditions_x_13_identity_or_method_seed_rows",
              "test_predictions_read": False, "new_hyperparameter_or_checkpoint_selection_permitted": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checkpoints": len(records), "sha256": digest(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
