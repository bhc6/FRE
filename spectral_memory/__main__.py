"""Command line entry points; no model weights are downloaded by these commands."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .data import load_samples, read_jsonl, split_samples, write_jsonl
from .experiment import run_experiment
from .features import FeatureConfig, extract_feature_blocks, intervene
from .labels import attach_utilities


def feature_config(arguments):
    edges = tuple(float(value) for value in arguments.bands.split(","))
    return FeatureConfig(band_edges_hz=edges, resample_points=arguments.resample_points,
                         stft_window=arguments.stft_window)


def make_demo(output_dir, seed=17, subjects=24):
    if subjects < 5:
        raise ValueError("Demo needs at least five subjects")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output directory must be empty or new")
    output.mkdir(parents=True, exist_ok=True)
    hidden_dir = output / "hidden"
    hidden_dir.mkdir()
    signal_rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(seed + 100000)
    records, evaluation_rows = [], []
    for subject in range(subjects):
        for event in range(8):
            sample_id = f"person_{subject:03d}_event_{event:02d}"
            start = event * 40.0
            timestamps = start + np.arange(32, dtype=float)
            local_time = timestamps - start
            dimensions = 12
            offset = signal_rng.normal(size=dimensions)
            frequency = signal_rng.choice([0.0625, 0.125, 0.25, 0.375])
            phase = signal_rng.uniform(-np.pi, np.pi, size=dimensions)
            hidden = offset + np.sin(2 * np.pi * frequency * local_time[:, None] + phase)
            hidden += signal_rng.normal(scale=0.12, size=hidden.shape)
            hidden = hidden.astype(np.float32)
            feature_path = hidden_dir / f"{sample_id}.npz"
            np.savez_compressed(feature_path, hidden_states=hidden, timestamps_s=timestamps)
            candidate_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
            utility = float(label_rng.uniform(-0.25, 0.5))
            record = {"sample_id": sample_id, "candidate_id": sample_id,
                      "subject_id": f"person_{subject:03d}", "session_id": f"session_{subject:03d}",
                      "video_id": f"video_{subject:03d}", "query_id": f"query_{subject:03d}",
                      "memory_state_id": f"memory_{subject:03d}", "evaluation_group": f"group_{subject:03d}",
                      "event_start_s": start, "event_end_s": start + 32,
                      "hidden_path": str(feature_path), "cost": int(signal_rng.integers(1, 4)),
                      "cost_unit": "synthetic_units", "utility": utility, "synthetic": True,
                      "task_mode": "retrieve", "context": [1.0, 0.0],
                      "context_source": "query_and_base_memory", "label_source": "independent_random_smoke_labels",
                      "model_id": "synthetic", "layer_index": 0, "pooling": "synthetic"}
            record["candidate_hash"] = candidate_hash
            record["base_memory_hash"] = hashlib.sha256(record["memory_state_id"].encode()).hexdigest()
            record["memory_payload_path"] = str(feature_path)
            record["memory_payload_kind"] = "hidden_states_npz"
            records.append(record)
            for condition, reward in (("without", 0.4), ("with", 0.4 + utility)):
                evaluation_rows.append({"sample_id": sample_id, "query_id": record["query_id"],
                                        "memory_state_id": record["memory_state_id"], "seed": seed,
                                        "condition": condition, "reward": reward,
                                        "evaluator_id": "synthetic_demo", "evaluator_revision": "v1",
                                        "prompt_hash": "synthetic_template", "decoding_hash": "synthetic_deterministic",
                                        "reward_metric": "synthetic_reward", "base_memory_hash": record["base_memory_hash"],
                                        "candidate_hash": candidate_hash, "candidate_in_base": False,
                                        "query_after_event": True})
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, records)
    write_jsonl(output / "unlabeled.jsonl", [{key: value for key, value in record.items() if key != "utility"} for record in records])
    write_jsonl(output / "evaluations.jsonl", evaluation_rows)
    return manifest


def make_interventions(manifest, output_dir, mode, band_hz, seed, task_mode):
    samples = load_samples(manifest, task_mode, require_utility=False)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Intervention directory must be empty or new")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(samples):
        transformed = intervene(sample.hidden_states, sample.timestamps_s, mode, band_hz=band_hz, seed=seed + index)
        path = output / f"intervention_{index:06d}.npz"
        np.savez_compressed(path, hidden_states=transformed, timestamps_s=sample.timestamps_s)
        payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        record = dict(sample.metadata, hidden_path=str(path), intervention=mode,
                      sample_id=f"{sample.metadata['sample_id']}::{mode}::{payload_hash[:16]}",
                      candidate_id=f"{sample.metadata.get('candidate_id', sample.metadata['sample_id'])}::{mode}::{payload_hash[:16]}",
                      candidate_hash=payload_hash, memory_payload_path=str(path), memory_payload_kind="hidden_states_npz",
                      intervention_band_hz=band_hz, intervention_seed=seed + index,
                      original_sample_id=sample.metadata["sample_id"],
                      intervention_requires_new_task_evaluation=True)
        for key in ("utility", "label_source", "label_query_count", "label_pair_count", "label_aggregation", "utility_protocol"):
            record.pop(key, None)
        records.append(record)
    write_jsonl(output / "manifest.jsonl", records)
    return len(records)


def build_parser():
    parser = argparse.ArgumentParser(description="Spectral Memory Probe: offline hypotheses, not proof of memory gains")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True)
        child.add_argument("--task-mode", choices=("retrieve", "write"), default="retrieve")
        child.add_argument("--group-key", choices=("subject_id", "session_id", "video_id"), default="subject_id")
        child.add_argument("--seed", type=int, default=17)
        child.add_argument("--bands", default="0,0.125,0.25,0.5", help="Frequency band boundaries in Hz")
        child.add_argument("--resample-points", type=int, default=16)
        child.add_argument("--stft-window", type=int, default=16)
        if command == "run":
            child.add_argument("--output-dir", required=True)
            child.add_argument("--budget", type=int, default=8)
            child.add_argument("--alphas", default="0.1,1,10,100")
            child.add_argument("--bootstrap-draws", type=int, default=1000)
            child.add_argument("--projection-dim", type=int)
    demo = subparsers.add_parser("demo")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--subjects", type=int, default=24)
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--budget", type=int, default=8)
    labels = subparsers.add_parser("labels")
    labels.add_argument("--manifest", required=True)
    labels.add_argument("--evaluations", required=True)
    labels.add_argument("--output", required=True)
    perturb = subparsers.add_parser("intervene")
    perturb.add_argument("--manifest", required=True)
    perturb.add_argument("--output-dir", required=True)
    perturb.add_argument("--mode", choices=("identity", "reverse", "shuffle", "phase_randomize", "remove_band"), required=True)
    perturb.add_argument("--band", help="lower,upper Hz; required for remove_band")
    perturb.add_argument("--seed", type=int, default=17)
    perturb.add_argument("--task-mode", choices=("retrieve", "write"), default="retrieve")
    return parser


def main():
    arguments = build_parser().parse_args()
    try:
        if arguments.command == "demo":
            manifest = make_demo(arguments.output_dir, arguments.seed, arguments.subjects)
            report = run_experiment(manifest, Path(arguments.output_dir) / "run", seed=arguments.seed, budget=arguments.budget)
            print(json.dumps({"status": report["status"], "report": str(Path(arguments.output_dir).resolve() / "run" / "report.md")}, ensure_ascii=False))
        elif arguments.command == "run":
            report = run_experiment(arguments.manifest, arguments.output_dir, feature_config(arguments),
                                    arguments.task_mode, arguments.group_key, arguments.seed, arguments.budget,
                                    tuple(float(value) for value in arguments.alphas.split(",")),
                                    arguments.bootstrap_draws, arguments.projection_dim)
            print(json.dumps({"status": report["status"], "split_counts": report["split_counts"]}, ensure_ascii=False))
        elif arguments.command == "validate":
            samples = load_samples(arguments.manifest, arguments.task_mode)
            _, masks = split_samples(samples, arguments.group_key, arguments.seed)
            config = feature_config(arguments)
            widths = None
            for sample in samples:
                widths = {key: len(value) for key, value in extract_feature_blocks(sample.hidden_states, sample.timestamps_s, config).items()}
            print(json.dumps({"samples": len(samples), "split_counts": {key: int(mask.sum()) for key, mask in masks.items()},
                              "feature_widths": widths, "config": asdict(config)}, ensure_ascii=False))
        elif arguments.command == "labels":
            count = attach_utilities(arguments.manifest, arguments.evaluations, arguments.output)
            print(json.dumps({"labeled_samples": count, "output": arguments.output}))
        else:
            band = tuple(float(value) for value in arguments.band.split(",")) if arguments.band else None
            count = make_interventions(arguments.manifest, arguments.output_dir, arguments.mode, band,
                                       arguments.seed, arguments.task_mode)
            print(json.dumps({"intervened_samples": count, "requires_new_labels": True}))
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
