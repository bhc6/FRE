"""Run the one frozen new-test matrix in sequential resource-bounded shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(arguments):
    from .resource_guard import gpu_snapshot

    root = Path(__file__).resolve().parents[1]
    freeze_path = root / "outputs/controller_experiment_v1/pretest_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "frozen_before_new_test":
        raise ValueError("The complete pretest freeze is required")
    for name, expected in freeze["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    for record in freeze["checkpoints"]:
        if digest(Path(record["directory"]) / "final.pt") != record["checkpoint_sha256"]:
            raise ValueError("Frozen final checkpoint changed")
    questions_path = root / "data/internal_controller_v1/test.questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines()]
    videos = sorted({row["video_id"] for row in questions})
    if len(videos) != 16 or len(questions) != 74:
        raise ValueError("The new-test matrix changed")
    plan_path = root / "outputs/controller_experiment_v1/test_execution_plan.json"
    if plan_path.exists():
        raise ValueError("This test execution has already been scheduled; inspect its audit before any recovery")
    shards = []
    for index in range(4):
        selected = videos[index * 4:(index + 1) * 4]
        cases = sum(row["video_id"] in selected for row in questions) + 8
        directory = root / f"outputs/controller_evaluation_v1/test{index:02d}"
        if directory.exists():
            raise ValueError("Test shard output already exists")
        shards.append({"index": index, "video_ids": selected, "conditions": cases, "records": cases * 13,
                       "output_dir": str(directory)})
    plan = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pretest_freeze_sha256": digest(freeze_path), "driver_sha256": digest(__file__), "shards": shards,
            "whole_test_conditions": sum(shard["conditions"] for shard in shards), "completed_shards": []}
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for shard in shards:
            cooldown_start = time.monotonic()
            while True:
                snapshot = gpu_snapshot()
                if snapshot["gpu_used_mib"] <= 256 and snapshot["gpu_free_mib"] >= 6500 and snapshot["gpu_temperature_c"] <= 65:
                    break
                if time.monotonic() - cooldown_start >= 60:
                    raise RuntimeError("GPU did not return to the unchanged preflight limits within sixty seconds")
                time.sleep(2)
            command = [sys.executable, "-m", "spectral_memory.controller_evaluate", "--split", "test",
                       "--questions", str(questions_path), "--frames-manifest", str(root / "data/internal_controller_v1/frames16/test.frames.manifest.jsonl"),
                       "--protocol", str(root / "docs/learned_controller_protocol_v1.md"), "--training-dirs"]
            command.extend(str(root / f"outputs/controller_training_v1/seed{seed}") for seed in (20261101, 20261102, 20261103))
            command.extend(("--video-ids", ",".join(shard["video_ids"]), "--output-dir", shard["output_dir"]))
            subprocess.run(command, cwd=root, check=True)
            plan["completed_shards"].append(shard["index"])
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        plan["status"] = "complete"
        plan["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        plan["status"] = "failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
