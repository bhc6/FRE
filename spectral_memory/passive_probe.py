"""Read-only internal activation capture with exact output controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


LAYERS = (0, 7, 13, 20, 27)
BRANCHES = ("decoder_input", "attention_delta", "pre_ffn", "ffn_input", "ffn_delta", "post_ffn")
HEAD_BRANCHES = ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "q_rot", "k_rot")
PROTOCOL = Path("docs/passive_probe_protocol_v4.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def reference_attention(query, key, value, mask, scaling):
    import numpy as np

    head_count = query.shape[0]
    key_heads = key.shape[0]
    if head_count % key_heads or key.shape != value.shape:
        raise ValueError("Unsupported GQA layout")
    mapping = np.arange(head_count) // (head_count // key_heads)
    scores = np.einsum("hd,hnd->hn", query.astype(np.float64), key[mapping].astype(np.float64)) * scaling
    if mask is not None:
        if mask.dtype == np.bool_:
            scores = np.where(mask, scores, -np.inf)
        else:
            scores = scores + mask.astype(np.float64)
    if not np.isfinite(scores.max(axis=-1)).all():
        raise ValueError("The last query has no visible keys")
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    context = np.einsum("hn,hnd->hd", probabilities, value[mapping].astype(np.float64))
    return probabilities, context


def capture(backend, batch, full_visual):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation

    if not batch.metadata["uniform_hidden_sampling"] or batch.video_positions is None:
        raise ValueError("This protocol requires uniform visual time groups")
    positions = torch.as_tensor(batch.video_positions, device=backend.device)
    arrays = {
        "video_positions": np.asarray(batch.video_positions, dtype=np.int64),
        "timestamps_s": np.asarray(batch.metadata["hidden_timestamps_s"], dtype=np.float64),
        "token_ids": batch["input_ids"][0].detach().cpu().numpy().copy(),
    }
    identities = {}
    contexts = {}
    handles = []
    visited = set()
    original_registry = implementation.ALL_ATTENTION_FUNCTIONS
    modules = {id(backend.model.model.language_model.layers[index].self_attn): index for index in LAYERS}

    def to_numpy(values):
        return values.detach().to(device="cpu", dtype=torch.float32).numpy().copy()

    def sample(index, branch, states, heads=False):
        identifier = (index, branch)
        if identifier in visited:
            raise RuntimeError(f"Duplicate capture: {identifier}")
        visited.add(identifier)
        if states.shape[0] != 1 or states.shape[1] != batch["input_ids"].shape[1]:
            raise ValueError(f"Unexpected token layout: {branch}")
        if heads and states.ndim == 3:
            head_dim = backend.model.model.language_model.layers[index].self_attn.head_dim
            states = states.reshape(*states.shape[:2], -1, head_dim)
        last = to_numpy(states[0, -1])
        if not np.isfinite(last).all():
            raise ValueError("Nonfinite readout state")
        arrays[f"layer_{index}_last_{branch}"] = last
        if full_visual:
            visual = to_numpy(states[0, positions])
            if not np.isfinite(visual).all():
                raise ValueError("Nonfinite visual state")
            arrays[f"layer_{index}_{branch}"] = visual

    def input_hook(index, branch):
        def hook(module, arguments):
            sample(index, branch, arguments[0])
        return hook

    def output_hook(index, branch, heads=False, tuple_output=False):
        def hook(module, arguments, output):
            sample(index, branch, output[0] if tuple_output else output, heads)
        return hook

    def norm_hook(index, branch):
        def hook(module, arguments, output):
            sample(index, branch, output, True)
        return hook

    def context_hook(index):
        def hook(module, arguments):
            head_dim = backend.model.model.language_model.layers[index].self_attn.head_dim
            arrays[f"layer_{index}_last_sdpa_context"] = to_numpy(arguments[0][0, -1]).reshape(-1, head_dim)
        return hook

    def observed_sdpa(module, query, key, value, attention_mask, **kwargs):
        if id(module) in modules:
            index = modules[id(module)]
            sample(index, "q_rot", query.transpose(1, 2), True)
            sample(index, "k_rot", key.transpose(1, 2), True)
            mask = None
            if attention_mask is not None:
                if attention_mask.ndim != 4:
                    raise ValueError("Unsupported attention mask layout")
                selected_mask = attention_mask[0, :, -1, :key.shape[-2]].detach().cpu()
                mask = selected_mask.numpy() if selected_mask.dtype == torch.bool else selected_mask.float().numpy()
                arrays[f"layer_{index}_last_attention_mask"] = mask.copy()
            if kwargs.get("dropout", 0.0) != 0.0 or key.shape[-2] != query.shape[-2]:
                raise ValueError("Probe requires uncached, deterministic prefill")
            probabilities, context = reference_attention(
                to_numpy(query[0, :, -1]), to_numpy(key[0]), to_numpy(value[0]),
                mask, kwargs.get("scaling", module.scaling),
            )
            arrays[f"layer_{index}_last_attention_weights"] = probabilities.astype(np.float32)
            arrays[f"layer_{index}_last_reference_context"] = context.astype(np.float32)
            contexts[index] = {"scaling": float(kwargs.get("scaling", module.scaling)),
                               "query_heads": int(query.shape[1]), "kv_heads": int(key.shape[1]),
                               "reference": "CPU FP64 reconstruction, not exported SDPA attention or causal attribution"}
        return original_registry["sdpa"](module, query, key, value, attention_mask, **kwargs)

    class ReadOnlyRegistry:
        def __getitem__(self, name):
            return observed_sdpa if name == "sdpa" else original_registry[name]

        def __getattr__(self, name):
            return getattr(original_registry, name)

    try:
        for index in LAYERS:
            decoder = backend.model.model.language_model.layers[index]
            attention = decoder.self_attn
            if attention.config._attn_implementation != "sdpa":
                raise ValueError("Only the frozen SDPA backend is supported")
            handles.extend([
                decoder.register_forward_pre_hook(input_hook(index, "decoder_input")),
                attention.register_forward_hook(output_hook(index, "attention_delta", tuple_output=True)),
                decoder.post_attention_layernorm.register_forward_pre_hook(input_hook(index, "pre_ffn")),
                decoder.mlp.register_forward_pre_hook(input_hook(index, "ffn_input")),
                decoder.mlp.register_forward_hook(output_hook(index, "ffn_delta")),
                decoder.register_forward_hook(output_hook(index, "post_ffn")),
                attention.q_proj.register_forward_hook(output_hook(index, "q_proj", heads=True)),
                attention.k_proj.register_forward_hook(output_hook(index, "k_proj", heads=True)),
                attention.v_proj.register_forward_hook(output_hook(index, "v_proj", heads=True)),
                attention.q_norm.register_forward_hook(norm_hook(index, "q_norm")),
                attention.k_norm.register_forward_hook(norm_hook(index, "k_norm")),
                attention.o_proj.register_forward_pre_hook(context_hook(index)),
            ])
        implementation.ALL_ATTENTION_FUNCTIONS = ReadOnlyRegistry()
        logits = forward_logits(backend, batch)
    finally:
        implementation.ALL_ATTENTION_FUNCTIONS = original_registry
        for handle in handles:
            handle.remove()
    if visited != {(index, branch) for index in LAYERS for branch in BRANCHES + HEAD_BRANCHES}:
        raise RuntimeError("Incomplete internal capture")
    for index in LAYERS:
        prefixes = [f"layer_{index}_last_"] + ([f"layer_{index}_"] if full_visual else [])
        for prefix in prefixes:
            for name, left, right, observed in (
                ("attention", "decoder_input", "attention_delta", "pre_ffn"),
                ("ffn", "pre_ffn", "ffn_delta", "post_ffn"),
            ):
                expected = (torch.from_numpy(arrays[prefix + left]).to(torch.bfloat16)
                            + torch.from_numpy(arrays[prefix + right]).to(torch.bfloat16)).float().numpy()
                actual = arrays[prefix + observed]
                identical = bool(np.array_equal(expected, actual))
                identities[prefix + name] = {"exact_in_bfloat16": identical,
                                             "maximum_error": float(np.max(np.abs(expected - actual)))}
                if not identical:
                    raise RuntimeError(f"Residual invariant failed: {prefix}{name}")
        reference = arrays[f"layer_{index}_last_reference_context"]
        actual = arrays[f"layer_{index}_last_sdpa_context"]
        contexts[index]["maximum_absolute_error"] = float(np.max(np.abs(reference - actual)))
        contexts[index]["relative_l2_error"] = float(np.linalg.norm(reference - actual) / max(np.linalg.norm(actual), 1e-12))
    return logits, arrays, {"identities": identities, "attention_reference": contexts,
                            "visual_states_persisted": full_visual}


def forward_logits(backend, batch):
    import torch

    with torch.inference_mode():
        result = backend.model(**batch, use_cache=False, output_hidden_states=False,
                               output_attentions=False, return_dict=True, logits_to_keep=1)
        return result.logits[0, -1].float().cpu().numpy().copy()


def run(arguments):
    configure_threads()
    import numpy as np
    from .causal_intervention import DEVELOPMENT_IDS, load_frames
    from .evaluate_small import build_prompt
    from .memory_study import in_subset
    from .probe_ffn import INSTRUCTION
    from .qwen3_backend import load_backend, DEFAULT_MAX_PIXELS

    if arguments.video_id not in DEVELOPMENT_IDS:
        raise ValueError("Only the original development videos are allowed")
    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 10 * 1024**3:
        raise RuntimeError("At least 10 GiB of free output storage is required")
    questions = [json.loads(line) for line in arguments.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 36 or {row["video_id"] for row in questions} != DEVELOPMENT_IDS:
        raise ValueError("Expected all 36 development questions")
    questions = sorted((row for row in questions if row["video_id"] == arguments.video_id), key=lambda row: row["query_id"])
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "video_id": arguments.video_id, "layers": LAYERS, "device": "cuda",
              "expected_conditions": 3 + len(questions), "completed_conditions": 0,
              "protocol_sha256": digest(PROTOCOL), "questions_sha256": digest(arguments.questions),
              "source_sha256": digest(__file__), "purpose": "passive_development_diagnostic_not_memory_improvement"}
    write_json(output / "config.json", config)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for filename in ("passive_probe.py", "resource_guard.py", "qwen3_backend.py", "evaluate_small.py", "causal_intervention.py"):
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    shutil.copy2(PROTOCOL, snapshot / "protocol.md")
    try:
        with ResourceGuard(output, "cuda") as guard:
            guard.configure_torch()
            import torch

            guard.phase = "load_validated_frames"
            data = load_frames(arguments.input_root)[arguments.video_id]
            config["inputs"] = data["provenance"]
            config["source_frame_timestamps_s"] = data["record"]["source_frame_timestamps_s"]
            guard.phase = "load_model"
            backend = load_backend(device="cuda")
            if backend.model.dtype != torch.bfloat16:
                raise ValueError("V4 precision is explicitly BF16")
            config["model"] = backend.metadata
            from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation
            config["modeling_source_sha256"] = digest(implementation.__file__)
            write_json(output / "config.json", config)
            conditions = [("real", data["frames"], INSTRUCTION, None),
                          ("static_first", [data["frames"][0]] * 16, INSTRUCTION, None),
                          ("static_last", [data["frames"][-1]] * 16, INSTRUCTION, None)]
            conditions.extend(("qa_" + question["query_id"].split(":")[-1], data["frames"],
                               build_prompt(question), question) for question in questions)
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for condition, frames, instruction, question in conditions:
                    guard.phase = f"prepare:{condition}"
                    batch = backend.prepare_video(frames, data["record"]["timestamps_s"], instruction, DEFAULT_MAX_PIXELS)
                    if batch.metadata["input_tokens"] > 1024 or batch.metadata["visual_tokens"] > 512:
                        raise ValueError("V4 input exceeds its predeclared token cap")
                    guard.phase = f"baseline:{condition}"
                    baseline = forward_logits(backend, batch)
                    guard.phase = f"passive_probe:{condition}"
                    started = time.perf_counter()
                    with torch.inference_mode():
                        logits, arrays, checks = capture(backend, batch, question is None)
                    elapsed = time.perf_counter() - started
                    identical = bool(np.array_equal(baseline, logits))
                    maximum = float(np.max(np.abs(baseline - logits)))
                    if not identical:
                        raise RuntimeError(f"Probe changed logits: {condition}, max={maximum}")
                    guard.phase = f"save:{condition}"
                    filename = condition + ".npz"
                    with (output / filename).open("xb") as handle:
                        np.savez_compressed(handle, **arrays)
                    with (output / (condition + "_logits.npz")).open("xb") as handle:
                        np.savez_compressed(handle, baseline=baseline, probe=logits)
                    row = {"video_id": arguments.video_id, "condition": condition,
                           "query_id": question["query_id"] if question else None,
                           "prompt_sha256": batch.metadata["prompt_sha256"], "batch": batch.metadata,
                           "arrays_file": filename, "arrays_sha256": digest(output / filename),
                           "arrays_shapes": {name: list(value.shape) for name, value in arrays.items()},
                           "arrays_uncompressed_bytes": sum(value.nbytes for value in arrays.values()),
                           "logits_file": condition + "_logits.npz",
                           "logits_sha256": digest(output / (condition + "_logits.npz")),
                           "full_logits_identical": identical, "full_logits_maximum_error": maximum,
                           "baseline_logits_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
                           "probe_forward_seconds": elapsed, "checks": checks}
                    if question is not None:
                        labels = [chr(ord("A") + index) for index in range(len(question["options"]))]
                        token_ids = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in labels]
                        if any(len(values) != 1 for values in token_ids):
                            raise ValueError("Options must have single-token labels")
                        allowed = logits[[values[0] for values in token_ids]]
                        answer = question["answer_index"]
                        predicted = int(np.argmax(allowed))
                        row.update({"question": question["question"], "options": question["options"],
                                    "gold_answer_index": answer, "predicted_answer_index": predicted,
                                    "correct": int(answer == predicted), "option_logits": allowed.tolist(),
                                    "gold_margin": float(allowed[answer] - np.max(np.delete(allowed, answer))),
                                    "temporal_state": in_subset({**question, "query_text": question["question"]}, "temporal_state")})
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    config["completed_conditions"] += 1
                    write_json(output / "config.json", config)
                    print(f"{arguments.video_id} {condition}: exact logits; {elapsed:.2f}s; {row['arrays_uncompressed_bytes']} raw bytes", flush=True)
                    del arrays, logits
                guard.phase = "verify_restored_forward"
                restored = forward_logits(backend, batch)
                if not np.array_equal(restored, baseline):
                    raise RuntimeError("Forward changed after probe removal")
                config["last_restored_logits_identical"] = True
            config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            write_json(output / "config.json", config)
            del backend, batch
            torch.cuda.empty_cache()
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--questions", type=Path, default=Path("data/perception_test_sample/questions.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
