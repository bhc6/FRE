"""Train a generic final-token FFN capacity baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


def run(arguments):
    configure_threads()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from .controller_runtime import digest, write_json
    from .ffn_readout_v2 import FFNFrozenTail as FrozenTail
    from .controller_train import load_dataset
    from .endpoint_control import check_deadline
    from .ffn_readout_v2 import FFNReadoutController

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a new training directory")
    if arguments.seed not in (20261101, 20261102, 20261103):
        raise ValueError("Unregistered training seed")
    output.mkdir(parents=True)
    source_names = ("ffn_train_v2.py", "ffn_readout_v2.py", "controller_train.py",
                    "controller_runtime.py", "learned_readout.py", "resource_guard.py")
    config = {
        "status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": arguments.seed, "steps": 200, "methods": arguments.methods,
        "residual_limits": arguments.residual_limits, "batch_real": 6, "batch_static": 2,
        "learning_rate": .001, "weight_decay": .0001, "real_kl": .1,
        "static_kl": 1., "relative_delta_squared_penalty": .001,
        "protocol_sha256": digest(arguments.protocol), "cache_protocol_sha256": digest(arguments.cache_protocol),
        "checkpoint_policy": "final_step_200", "training_only_split": "train",
        "training_precision": "BF16_frozen_tail_replay_approximation",
        "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names},
        "completed_methods": [],
    }
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(arguments.protocol, snapshot / "protocol.md")
    write_json(output / "config.json", config)
    try:
        dataset, rows, tail_path, provenance = load_dataset(arguments.cache_dirs, arguments.questions,
                                                            config["cache_protocol_sha256"])
        config["data_provenance"] = provenance
        config["training_cases"] = len(rows)
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            guard.phase = "load_training_cache"
            tail = FrozenTail(torch.load(tail_path, map_location="cpu", weights_only=True)).to("cuda").eval()
            dataset = {name: value.to("cuda") for name, value in dataset.items()}
            with torch.no_grad():
                residual = dataset["decoder_input"].to(torch.bfloat16) + dataset["attention_delta"].to(torch.bfloat16)
                dataset["query"] = tail.post_norm(residual).float()
                dataset["context"] = tail.mlp(dataset["query"].to(torch.bfloat16)).float()
                del residual
            real_indices = torch.nonzero(dataset["supervised"], as_tuple=False).flatten().cpu().numpy()
            static_indices = torch.nonzero(~dataset["supervised"], as_tuple=False).flatten().cpu().numpy()

            def evaluate_replay(controller):
                entries = []
                with torch.no_grad():
                    for start in range(0, len(rows), 8):
                        check_deadline(guard)
                        selected = slice(start, start + 8)
                        bases = {name: dataset["basis_" + name][selected] for name in ("time", "spectral")}
                        delta, diagnostics = controller(dataset["query"][selected], bases, dataset["context"][selected])
                        logits = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], delta)
                        for offset, values in enumerate(logits.cpu().tolist()):
                            row = rows[start + offset]
                            entries.append({"case_id": row["case_id"], "video_id": row["video_id"],
                                            "supervised": row["supervised"], "gold": row["gold_answer_index"],
                                            "option_logits": values, "native_option_logits": row["option_logits"],
                                            "maximum_relative_ffn_delta": float((diagnostics["delta_norm"][offset] /
                                                diagnostics["context_norm"][offset].clamp_min(1e-12)).max())})
                return entries

            for residual_limit in arguments.residual_limits:
                for method in arguments.methods:
                    check_deadline(guard)
                    torch.manual_seed(arguments.seed)
                    torch.cuda.manual_seed_all(arguments.seed)
                    generator = np.random.default_rng(arguments.seed)
                    controller_config = {"hidden_dim": 2048, "rank": 4, "residual_limit": residual_limit}
                    controller = FFNReadoutController(method, **controller_config).to("cuda")
                    optimizer = torch.optim.AdamW(controller.parameters(), lr=.001, weight_decay=.0001)
                    tag = f"{method}_cap{residual_limit:g}_seed{arguments.seed}"
                    method_dir = output / tag
                    method_dir.mkdir()
                    torch.save(controller.state_dict(), method_dir / "initial.pt")
                    initial_replay = evaluate_replay(controller)
                    write_json(method_dir / "initial_replay.json", initial_replay)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with (method_dir / "training.jsonl").open("x", encoding="utf-8") as stream:
                        for step in range(200):
                            check_deadline(guard)
                            guard.phase = f"train:{tag}:{step + 1}"
                            indices = np.concatenate((generator.choice(real_indices, 6, replace=True),
                                                      generator.choice(static_indices, 2, replace=True)))
                            selected = torch.as_tensor(indices, device="cuda")
                            bases = {name: dataset["basis_" + name][selected] for name in ("time", "spectral")}
                            delta, diagnostics = controller(dataset["query"][selected], bases, dataset["context"][selected])
                            logits = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], delta)
                            teacher_log = dataset["teacher"][selected].float().log_softmax(dim=-1)
                            divergence = (teacher_log.exp() * (teacher_log - logits.log_softmax(dim=-1))).sum(dim=-1)
                            classification = torch.nn.functional.cross_entropy(logits[:6], dataset["gold"][selected[:6]])
                            ratio = delta.square().sum(dim=-1) / dataset["context"][selected].square().sum(dim=-1).clamp_min(1e-12)
                            loss = classification + .1 * divergence[:6].mean() + divergence[6:].mean() + .001 * ratio.mean()
                            if not torch.isfinite(loss):
                                raise RuntimeError("Nonfinite training loss")
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                            record = {"step": step + 1, "loss": float(loss.detach()),
                                      "real_ce": float(classification.detach()),
                                      "real_kl": float(divergence[:6].mean().detach()),
                                      "static_kl": float(divergence[6:].mean().detach()),
                                      "relative_delta_squared": float(ratio.mean().detach()),
                                      "gradient_norm_before_clip": float(gradient_norm),
                                      "capped_head_fraction": float(diagnostics["cap_active"].float().mean()),
                                      "case_ids": [rows[int(index)]["case_id"] for index in indices]}
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    final_replay = evaluate_replay(controller)
                    write_json(method_dir / "final_replay.json", final_replay)
                    checkpoint = {"method": method, "seed": arguments.seed, "step": 200, "tag": tag,
                                  "state_dict": {name: value.detach().cpu() for name, value in controller.state_dict().items()},
                                  "controller_config": controller_config, "protocol_sha256": config["protocol_sha256"],
                                  "training_data": provenance,
                                  "parameter_count": sum(value.numel() for value in controller.parameters())}
                    torch.save(checkpoint, method_dir / "final.pt")
                    torch.save(optimizer.state_dict(), method_dir / "optimizer.pt")
                    summary = {"tag": tag, "method": method, "seed": arguments.seed, "residual_limit": residual_limit,
                               "steps": 200, "elapsed_seconds": elapsed, "parameter_count": checkpoint["parameter_count"],
                               "checkpoint_sha256": digest(method_dir / "final.pt"),
                               "artifacts_sha256": {name: digest(method_dir / name) for name in
                                   ("initial.pt", "optimizer.pt", "initial_replay.json", "final_replay.json", "training.jsonl")}}
                    write_json(method_dir / "summary.json", summary)
                    config["completed_methods"].append(summary)
                    write_json(output / "config.json", config)
                    real_rows = [row for row in final_replay if row["supervised"]]
                    correct = sum(int(np.argmax(row["option_logits"]) == row["gold"]) for row in real_rows)
                    print(f"trained {tag}, replay {correct}/{len(real_rows)} (approximate), {elapsed:.2f}s", flush=True)
                    del controller, optimizer
            config.update({"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            check_deadline(guard)
            del tail, dataset
            torch.cuda.empty_cache()
        if (output / "resource_stop.json").exists() or digest(arguments.protocol) != config["protocol_sha256"]:
            raise RuntimeError("Resource stop or protocol changed")
        for name, expected in config["source_sha256"].items():
            if digest(Path(__file__).with_name(name)) != expected:
                raise RuntimeError("Training source changed")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=["ffn_lowrank"],
                        choices=("ffn_lowrank",))
    parser.add_argument("--residual-limits", nargs="+", type=float, default=[.1, .3], choices=(.1, .3))
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
