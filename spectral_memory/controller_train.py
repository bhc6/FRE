"""Fixed-budget training of four readout controllers on declared cached training data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


METHODS = ("time_mlp", "spectral_mlp", "spectral_phaseblind", "constant_spectral")
SEEDS = (20261101, 20261102, 20261103)
STEPS = 200


def load_dataset(cache_directories, split_questions, protocol_hash):
    import numpy as np
    import torch
    from .controller_runtime import digest, read_rows

    allowed = {row["query_id"]: row for row in read_rows(split_questions)}
    rows = []
    arrays = {}
    seen = set()
    tail_path = None
    tail_hash = None
    tail_manifest_hash = None
    caches = []
    keys = ("query", "context", "decoder_input", "attention_delta", "basis_time", "basis_spectral", "basis_phase_ablation")
    for directory in cache_directories:
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config["status"] != "complete" or digest(directory / "records.jsonl") != config["records_sha256"]:
            raise ValueError("Training cache failed completion or hash validation")
        if config["questions_sha256"] != digest(split_questions) or config["protocol_sha256"] != protocol_hash:
            raise ValueError("Cache question manifest or protocol differs from training")
        if digest(directory / "frozen_tail.pt") != config["tail_sha256"]:
            raise ValueError("Cached tail weight hash mismatch")
        manifest_hash = digest(directory / "frozen_tail_manifest.json")
        if manifest_hash != config["tail_manifest_sha256"] or tail_manifest_hash not in (None, manifest_hash):
            raise ValueError("Frozen tail tensors or configuration differ between training chunks")
        tail_manifest_hash = manifest_hash
        if tail_path is None:
            tail_path = directory / "frozen_tail.pt"
            tail_hash = digest(tail_path)
        caches.append({"directory": str(directory.resolve()), "records_sha256": config["records_sha256"]})
        for row in read_rows(directory / "records.jsonl"):
            if row["query_id"] not in allowed:
                raise ValueError("A non-training query entered the training cache")
            if allowed[row["query_id"]]["split"] != "train":
                raise ValueError("Only the declared train split may enter training")
            declared = allowed[row["query_id"]]
            expected_question = declared if row["supervised"] else {key: value for key, value in declared.items() if key not in {"answer_index", "answer_text"}}
            if row["question"] != expected_question or row["video_id"] != declared["video_id"]:
                raise ValueError("Cached question differs from the frozen official training manifest")
            if row["supervised"] != (row["condition"] == "real") or row["gold_answer_index"] != (declared["answer_index"] if row["supervised"] else None):
                raise ValueError("Cached supervised label or static condition was changed")
            if row["case_id"] in seen:
                raise ValueError("Duplicate cached training condition")
            seen.add(row["case_id"])
            path = directory / row["archive_file"]
            if digest(path) != row["archive_sha256"]:
                raise ValueError("Cache archive hash mismatch")
            with np.load(path, allow_pickle=False) as archive:
                for key in keys:
                    arrays.setdefault(key, []).append(archive[key].copy())
            arrays.setdefault("teacher", []).append(row["option_logits"])
            arrays.setdefault("gold", []).append(row["gold_answer_index"] if row["supervised"] else -100)
            arrays.setdefault("supervised", []).append(row["supervised"])
            rows.append(row)
    if {row["query_id"] for row in rows if row["supervised"]} != set(allowed):
        raise ValueError("Training real-QA coverage is incomplete")
    if sum(not row["supervised"] for row in rows) != 2 * len({row["video_id"] for row in rows}):
        raise ValueError("Exactly two distillation-only static cases per training video are required")
    for video_id in {row["video_id"] for row in rows}:
        static = [row["condition"] for row in rows if row["video_id"] == video_id and not row["supervised"]]
        if sorted(static) != ["static_first", "static_last"]:
            raise ValueError("Static distillation coverage differs by source")
    tensors = {key: torch.as_tensor(np.asarray(values)) for key, values in arrays.items()}
    return tensors, rows, tail_path, {"caches": caches, "tail_sha256": tail_hash, "tail_manifest_sha256": tail_manifest_hash,
                                      "training_questions_sha256": digest(split_questions)}


def run(arguments):
    configure_threads()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from .controller_runtime import FrozenTail, digest, write_json
    from .learned_readout import ReadoutController
    from .endpoint_control import check_deadline

    if arguments.seed not in SEEDS:
        raise ValueError("Use one of the three predeclared paired seeds")
    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Training output must be new")
    output.mkdir(parents=True)
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "seed": arguments.seed, "methods": METHODS, "steps_per_method": STEPS, "batch_real": 6, "batch_static": 2,
              "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
              "real_kl_weight": 0.1, "static_kl_weight": 1.0, "delta_penalty_weight": 0.001,
              "checkpoint_policy": "final_step_200_no_validation_selection", "tail_precision": "bfloat16",
              "loss": "restricted_original_option_cross_entropy_and_baseline_KL; no static QA gold",
              "sampling": "with_replacement_uniform_over_real_QA_and_over_static_cases_separately",
              "gradient_clip_norm": 1.0, "torch_version": torch.__version__, "numpy_version": np.__version__,
              "backbone_parameter_gradients": False, "training_compute": "frozen_cached_final_layer_tail",
              "protocol_sha256": digest(arguments.protocol), "completed_methods": [],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                 ("controller_train.py", "controller_runtime.py", "learned_readout.py", "resource_guard.py")}}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in config["source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(arguments.protocol, snapshot / "protocol.md")
    write_json(output / "config.json", config)
    try:
        dataset, rows, tail_path, provenance = load_dataset(arguments.cache_dirs, arguments.questions, config["protocol_sha256"])
        config["data_provenance"] = provenance
        config["training_cases"] = len(rows)
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            guard.phase = "load_cached_tail"
            payload = torch.load(tail_path, map_location="cpu", weights_only=True)
            tail = FrozenTail(payload).to("cuda").eval()
            dataset = {key: value.to("cuda") for key, value in dataset.items()}
            real_indices = torch.nonzero(dataset["supervised"], as_tuple=False).flatten().cpu().numpy()
            static_indices = torch.nonzero(~dataset["supervised"], as_tuple=False).flatten().cpu().numpy()
            with torch.no_grad():
                parity = []
                for start in range(0, len(rows), 8):
                    selected = slice(start, start + 8)
                    predicted = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected],
                                     torch.zeros_like(dataset["context"][selected]))
                    teacher = dataset["teacher"][selected].float()
                    if not torch.isfinite(predicted).all() or not torch.isfinite(teacher).all():
                        raise RuntimeError("Nonfinite frozen-tail replay or native teacher")
                    for offset in range(len(predicted)):
                        index = start + offset
                        single = tail(dataset["decoder_input"][index:index + 1], dataset["attention_delta"][index:index + 1],
                                      torch.zeros_like(dataset["context"][index:index + 1]))[0]
                        differences = predicted[offset] - teacher[offset]
                        single_difference = single - teacher[offset]
                        parity.append({"case_id": rows[start + offset]["case_id"],
                                       "maximum_option_logit_difference": float(differences.abs().max()),
                                       "rms_option_logit_difference": float(differences.square().mean().sqrt()),
                                       "maximum_pairwise_margin_difference": float(differences.max() - differences.min()),
                                       "prediction_changed": bool(predicted[offset].argmax() != teacher[offset].argmax()),
                                       "batch1_maximum_option_logit_difference": float(single_difference.abs().max()),
                                       "batch1_rms_option_logit_difference": float(single_difference.square().mean().sqrt()),
                                       "batch1_maximum_pairwise_margin_difference": float(single_difference.max() - single_difference.min()),
                                       "batch1_prediction_changed": bool(single.argmax() != teacher[offset].argmax()),
                                       "native_option_logits": teacher[offset].cpu().tolist(),
                                       "batch8_option_logits": predicted[offset].cpu().tolist(),
                                       "batch1_option_logits": single.cpu().tolist(), "replay_batch_size": len(predicted)})
                write_json(output / "replay_parity.json", parity)
                config["replay_parity_sha256"] = digest(output / "replay_parity.json")
            for method in METHODS:
                check_deadline(guard)
                torch.manual_seed(arguments.seed)
                torch.cuda.manual_seed_all(arguments.seed)
                generator = np.random.default_rng(arguments.seed)
                controller = ReadoutController(method).to("cuda")
                optimizer = torch.optim.AdamW(controller.parameters(), lr=0.001, weight_decay=0.0001)
                config["optimizer_defaults"] = optimizer.defaults
                method_dir = output / method
                method_dir.mkdir()
                torch.save(controller.state_dict(), method_dir / "initial.pt")
                torch.cuda.synchronize()
                started = time.perf_counter()
                with (method_dir / "training.jsonl").open("x", encoding="utf-8") as stream:
                    for step in range(STEPS):
                        check_deadline(guard)
                        guard.phase = f"train:{method}:{step + 1}"
                        indices = np.concatenate((generator.choice(real_indices, 6, replace=True),
                                                  generator.choice(static_indices, 2, replace=True)))
                        selected = torch.as_tensor(indices, device="cuda")
                        bases = {name: dataset["basis_" + name][selected] for name in ("time", "spectral", "phase_ablation")}
                        delta, diagnostic = controller(dataset["query"][selected], bases, dataset["context"][selected])
                        logits = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], delta)
                        teacher = dataset["teacher"][selected].float()
                        teacher_log = teacher.log_softmax(dim=-1)
                        divergence = (teacher_log.exp() * (teacher_log - logits.log_softmax(dim=-1))).sum(dim=-1)
                        classification = torch.nn.functional.cross_entropy(logits[:6], dataset["gold"][selected[:6]])
                        ratio = delta.square().sum(dim=-1) / dataset["context"][selected].square().sum(dim=-1).clamp_min(1e-12)
                        loss = classification + 0.1 * divergence[:6].mean() + divergence[6:].mean() + 0.001 * ratio.mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite training loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        record = {"step": step + 1, "loss": float(loss.detach()), "real_ce": float(classification.detach()),
                                  "real_kl": float(divergence[:6].mean().detach()), "static_kl": float(divergence[6:].mean().detach()),
                                  "relative_delta_squared": float(ratio.mean().detach()), "gradient_norm_before_clip": float(norm),
                                  "case_ids": [rows[int(index)]["case_id"] for index in indices]}
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                checkpoint = {"method": method, "seed": arguments.seed, "step": STEPS,
                              "state_dict": {name: tensor.detach().cpu() for name, tensor in controller.state_dict().items()},
                              "controller_config": {"head_dim": 128, "time_bins": 8, "hidden_dim": 16,
                                                    "gain_limit": 0.25, "residual_limit": 0.1},
                              "protocol_sha256": config["protocol_sha256"], "training_data": provenance,
                              "parameter_count": sum(parameter.numel() for parameter in controller.parameters())}
                torch.save(checkpoint, method_dir / "final.pt")
                torch.save(optimizer.state_dict(), method_dir / "optimizer.pt")
                summary = {"method": method, "seed": arguments.seed, "steps": STEPS, "elapsed_seconds": elapsed,
                           "parameter_count": checkpoint["parameter_count"], "checkpoint_sha256": digest(method_dir / "final.pt"),
                           "training_log_sha256": digest(method_dir / "training.jsonl"),
                           "optimizer_sha256": digest(method_dir / "optimizer.pt"),
                           "initial_state_sha256": digest(method_dir / "initial.pt")}
                write_json(method_dir / "summary.json", summary)
                config["completed_methods"].append(summary)
                write_json(output / "config.json", config)
                print(f"trained {method} seed={arguments.seed} steps={STEPS} seconds={elapsed:.2f}", flush=True)
                del optimizer, controller
            config.update({"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            check_deadline(guard)
            del tail, dataset
            torch.cuda.empty_cache()
        if (output / "resource_stop.json").exists() or digest(arguments.protocol) != config["protocol_sha256"]:
            raise RuntimeError("Training resource stop or protocol changed")
        for name, expected in config["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected or digest(snapshot / name) != expected:
                raise RuntimeError("Training implementation changed during the run")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
