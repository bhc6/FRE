"""Run a short Qwen RGB-memory evaluation with the shared resource guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .resource_guard import ResourceGuard, configure_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    configure_threads()
    memories = [json.loads(line) for line in arguments.memory_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [json.loads(line) for line in arguments.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not memories or any(not 1 <= len(row["pair_indices"]) <= 8 for row in memories):
        raise ValueError("Guarded reader accepts only bounded memories of at most sixteen frames")
    if not 1 <= len(questions) <= 200:
        raise ValueError("This runner requires 1..200 questions per short run")
    counts = {}
    for row in memories:
        counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
    expected = sum(counts.get(question["video_id"], 0) + 1 for question in questions)
    if expected > 600 or any(question["video_id"] not in counts for question in questions):
        raise ValueError("Run must cover all questions and have at most 600 scoring rows")
    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Guarded output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "expected_records": expected,
        "device": "cuda", "dtype": "torch.bfloat16", "resume": False,
        "memory_manifest_sha256": hashlib.sha256(arguments.memory_manifest.read_bytes()).hexdigest(),
        "questions_sha256": hashlib.sha256(arguments.questions.read_bytes()).hexdigest(),
        "wrapper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "guard_source_sha256": hashlib.sha256(Path(__file__).with_name("resource_guard.py").read_bytes()).hexdigest(),
    }
    (output / "guarded_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (output / "source_snapshot").mkdir()
    for filename in ("guarded_readout.py", "resource_guard.py"):
        shutil.copy2(Path(__file__).with_name(filename), output / "source_snapshot" / filename)
    with ResourceGuard(output, "cuda") as guard:
        guard.configure_torch()
        from .memory_study import evaluate
        guard.phase = "bounded_memory_readout"
        evaluate(argparse.Namespace(
            output_dir=output / "study", memory_manifest=arguments.memory_manifest,
            questions=arguments.questions, video_id=None, query_id=None,
            device="cuda", backend_module="spectral_memory.qwen3_backend", model_path=None,
            revision=None, max_pixels=65536, seed=17, resume=False,
        ))


if __name__ == "__main__":
    main()
