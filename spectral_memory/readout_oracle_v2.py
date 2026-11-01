"""Label-assisted training-only reachability diagnostic, never a deployed method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .resource_guard import ResourceGuard, configure_threads


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from .controller_runtime import FrozenTail, digest, write_json
    from .controller_train import load_dataset
    from .endpoint_control import check_deadline

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a new oracle diagnostic directory")
    output.mkdir(parents=True)
    config = {"status": "started", "purpose": "label_assisted_training_only_reachability_not_generalization",
              "methods": ["original_time", "unit_time", "unit_spectral", "free_context"],
              "radius": .3, "steps": 100, "learning_rate": .05, "batch_size": 16,
              "source_sha256": digest(Path(__file__)), "cache_protocol_sha256": digest(arguments.cache_protocol)}
    write_json(output / "config.json", config)
    try:
        dataset, rows, tail_path, provenance = load_dataset(arguments.cache_dirs, arguments.questions,
                                                            config["cache_protocol_sha256"])
        config["training_provenance"] = provenance
        real_indices = [index for index, row in enumerate(rows) if row["supervised"]]
        records = []
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            torch.manual_seed(20261101)
            tail = FrozenTail(torch.load(tail_path, map_location="cpu", weights_only=True)).to("cuda").eval()
            dataset = {name: value.to("cuda") for name, value in dataset.items()}
            for method in config["methods"]:
                for start in range(0, len(real_indices), 16):
                    indices = real_indices[start:start + 16]
                    selected = torch.as_tensor(indices, device="cuda")
                    context = dataset["context"][selected]
                    context_norm = torch.linalg.vector_norm(context, dim=-1)
                    basis_name = "spectral" if method == "unit_spectral" else "time"
                    basis = dataset["basis_" + basis_name][selected]
                    if method.startswith("unit_"):
                        denominator = torch.maximum(torch.linalg.vector_norm(basis, dim=-1), 1e-6 * context_norm[..., None])
                        basis = basis / denominator.clamp_min(1e-12)[..., None] * context_norm[..., None, None] / np.sqrt(8)
                    raw = torch.nn.Parameter(torch.zeros((*context.shape[:2], 128 if method == "free_context" else 8), device="cuda"))
                    optimizer = torch.optim.Adam([raw], lr=.05)

                    def delta_for():
                        if method == "free_context":
                            delta = raw.tanh() * context_norm[..., None] * .3 / np.sqrt(128)
                        else:
                            delta = (.25 * raw.tanh()[..., None] * basis).sum(dim=2)
                        scale = (.3 * context_norm / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-12)).clamp(max=1.)
                        return delta * scale[..., None]

                    with torch.no_grad():
                        initial = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], torch.zeros_like(context))
                    for step in range(100):
                        check_deadline(guard)
                        guard.phase = f"oracle:{method}:{start}:{step}"
                        delta = delta_for()
                        logits = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], delta)
                        loss = torch.nn.functional.cross_entropy(logits, dataset["gold"][selected])
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite oracle loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    with torch.no_grad():
                        delta = delta_for()
                        final = tail(dataset["decoder_input"][selected], dataset["attention_delta"][selected], delta)
                    for offset, index in enumerate(indices):
                        records.append({"method": method, "case_id": rows[index]["case_id"],
                                        "video_id": rows[index]["video_id"], "gold": rows[index]["gold_answer_index"],
                                        "initial_logits": initial[offset].cpu().tolist(),
                                        "final_logits": final[offset].cpu().tolist(),
                                        "native_logits": rows[index]["option_logits"],
                                        "maximum_relative_delta": float((torch.linalg.vector_norm(delta[offset], dim=-1) /
                                                                          context_norm[offset].clamp_min(1e-12)).max())})
                method_rows = [row for row in records if row["method"] == method]
                before = sum(int(np.argmax(row["initial_logits"]) == row["gold"]) for row in method_rows)
                after = sum(int(np.argmax(row["final_logits"]) == row["gold"]) for row in method_rows)
                print(f"training-only oracle {method}: {before} -> {after}/{len(method_rows)}", flush=True)
            config.update({"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            del tail, dataset
            torch.cuda.empty_cache()
        (output / "records.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
        config.update({"status": "complete", "records_sha256": digest(output / "records.jsonl"),
                       "limitation": "finite optimizer diagnostic, not a proven optimum or unbiased task accuracy; BF16 replay approximation"})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
