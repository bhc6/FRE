"""Fixed endpoint attention gates with native and explicit-mask controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .endpoint_control import INPUT_ROOT, VIDEO_ID, build_cases, check_deadline, digest, input_manifest, write_json
from .resource_guard import ResourceGuard, configure_threads


CONDITIONS = ("native", "passthrough", "causal_sham", "first_gate", "last_gate")
PROTOCOL = Path("docs/endpoint_routing_protocol_v1.md")
PRIOR = Path("outputs/endpoint_control_v1")
LAYERS = (0, 7, 13, 20, 27)
SOURCES = ("endpoint_routing.py", "endpoint_control.py", "passive_probe.py", "causal_intervention.py", "qwen3_backend.py", "resource_guard.py")


def tensor_record(tensor):
    import torch

    values = tensor.detach().cpu().contiguous()
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()}


def make_mask(original, length, visual_positions, condition, device):
    import torch

    if condition not in {"causal_sham", "first_gate", "last_gate"}:
        raise ValueError("Unexpected explicit-mask condition")
    if length < 2 or visual_positions.ndim != 2 or visual_positions.numel() == 0:
        raise ValueError("Expected nonempty time-by-space visual positions")
    flat_positions = visual_positions.flatten()
    if visual_positions.dtype != torch.long or int(flat_positions.min()) < 0 or int(flat_positions.max()) >= length - 1:
        raise ValueError("Visual positions must leave a nonempty suffix inside the input")
    if not bool(torch.all(flat_positions[1:] > flat_positions[:-1])):
        raise ValueError("Visual positions must be unique and chronologically ordered")
    causal = torch.ones((length, length), dtype=torch.bool, device=device).tril()[None, None]
    if original is None:
        mask = causal.clone()
    else:
        if original.ndim != 4 or tuple(original.shape) != (1, 1, length, length):
            raise ValueError("Only the fixed full-prefill causal mask is supported")
        mask = original.clone()
        if mask.dtype == torch.bool:
            if bool(torch.any(mask & ~causal)):
                raise ValueError("The original explicit mask permits future keys")
            mask &= causal
        elif mask.dtype.is_floating_point:
            if bool(torch.any(torch.isnan(mask) | torch.isposinf(mask))):
                raise ValueError("The original mask contains NaN or positive infinity")
            blocked = torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min)
            if bool(torch.any(~causal & ~blocked)):
                raise ValueError("The original explicit mask permits future keys")
            mask.masked_fill_(~causal, float("-inf"))
        else:
            raise ValueError("Unexpected attention mask dtype")
    baseline = mask.clone()
    suffix_start = int(visual_positions.max()) + 1
    if condition in {"first_gate", "last_gate"}:
        removed = visual_positions[1:] if condition == "first_gate" else visual_positions[:-1]
        mask[:, :, suffix_start:, removed.flatten()] = False if mask.dtype == torch.bool else float("-inf")
    return baseline, mask, suffix_start


def verify_mask_edges(baseline, mask, visual_positions, condition, suffix_start):
    import torch

    length = mask.shape[-1]
    flat_positions = visual_positions.flatten()
    nonvisual = torch.ones(length, dtype=torch.bool, device=mask.device)
    nonvisual[flat_positions] = False
    allowed = visual_positions[0] if condition == "first_gate" else visual_positions[-1] if condition == "last_gate" else flat_positions
    removed = visual_positions[1:].flatten() if condition == "first_gate" else visual_positions[:-1].flatten() if condition == "last_gate" else flat_positions[:0]
    future = torch.ones((length, length), dtype=torch.bool, device=mask.device).triu(1)[None, None]
    blocked = ~mask if mask.dtype == torch.bool else torch.isneginf(mask) | (mask <= torch.finfo(mask.dtype).min)
    checks = {
        "no_new_edges_or_increased_additive_weights": not bool(torch.any(mask & ~baseline)) if mask.dtype == torch.bool else bool(torch.all(mask <= baseline)),
        "no_future_keys": bool(torch.all(blocked[future])),
        "prefix_mask_unchanged": bool(torch.equal(mask[:, :, :suffix_start], baseline[:, :, :suffix_start])),
        "suffix_nonvisual_mask_unchanged": bool(torch.equal(mask[:, :, suffix_start:, nonvisual], baseline[:, :, suffix_start:, nonvisual])),
        "target_visual_mask_unchanged": bool(torch.equal(mask[:, :, suffix_start:, allowed], baseline[:, :, suffix_start:, allowed])),
        "excluded_visual_keys_blocked": bool(torch.all(blocked[:, :, suffix_start:, removed])),
        "sham_mask_unchanged": condition != "causal_sham" or bool(torch.equal(mask, baseline)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Routing-mask edge invariant failed: {checks}")
    return checks


def routed_forward(backend, batch, condition):
    import numpy as np
    import torch
    from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation
    from .passive_probe import forward_logits

    positions = torch.as_tensor(batch.video_positions, device=backend.device, dtype=torch.long)
    length = int(batch["input_ids"].shape[1])
    if tuple(positions.shape) != (8, 60) or length != 618:
        raise ValueError("The frozen endpoint input layout changed")
    decoders = backend.model.model.language_model.layers
    if len(decoders) != 28:
        raise ValueError("The frozen model layer count changed")
    modules = {id(layer.self_attn): index for index, layer in enumerate(decoders)}
    registry = implementation.ALL_ATTENTION_FUNCTIONS
    handles = []
    hashes = {}
    visits = []
    arrays = {}
    information = {}

    def observed_visual(index):
        def hook(module, arguments, output):
            if index in hashes:
                raise RuntimeError("Duplicate visual-prefix capture")
            hashes[index] = tensor_record(output[0, positions])["sha256"]
        return hook

    def dispatch(module, query, key, value, attention_mask, **kwargs):
        if id(module) not in modules:
            return registry["sdpa"](module, query, key, value, attention_mask, **kwargs)
        index = modules[id(module)]
        visits.append(index)
        if query.shape[0] != 1 or query.shape[-2] != length or key.shape[-2] != length:
            raise ValueError("Only batch-one uncached prefill is supported")
        if kwargs.get("dropout", 0.0) != 0.0 or not module.is_causal:
            raise ValueError("Expected deterministic causal decoder attention")
        if condition == "passthrough":
            return registry["sdpa"](module, query, key, value, attention_mask, **kwargs)
        if attention_mask is None and kwargs.get("is_causal") is False:
            raise ValueError("The native unmasked call explicitly disables causality")
        baseline, mask, suffix_start = make_mask(attention_mask, length, positions, condition, query.device)
        if index == 0:
            arrays["base_mask"] = baseline.cpu().numpy() if baseline.dtype == torch.bool else baseline.float().cpu().numpy()
            arrays["used_mask"] = mask.cpu().numpy() if mask.dtype == torch.bool else mask.float().cpu().numpy()
            information.update({"suffix_start": suffix_start, "mask_dtype": str(mask.dtype),
                                "source_mask_is_none": attention_mask is None,
                                "source_mask": None if attention_mask is None else tensor_record(attention_mask),
                                "source_is_causal_argument": kwargs.get("is_causal"),
                                "base_mask_sha256": tensor_record(baseline)["sha256"],
                                "eligible_suffix_query_positions": list(range(suffix_start, length)),
                                "target_query_positions": list(range(suffix_start, length)) if condition != "causal_sham" else [],
                                "allowed_visual_key_positions": (positions[0] if condition == "first_gate" else positions[-1] if condition == "last_gate" else positions.flatten()).tolist(),
                                "removed_visual_key_positions": (positions[1:].flatten() if condition == "first_gate" else positions[:-1].flatten() if condition == "last_gate" else positions.flatten()[:0]).tolist(),
                                "mask_invariants": verify_mask_edges(baseline, mask, positions, condition, suffix_start),
                                "used_mask_sha256": tensor_record(mask)["sha256"]})
        elif (tensor_record(mask)["sha256"] != information["used_mask_sha256"]
              or tensor_record(baseline)["sha256"] != information["base_mask_sha256"]):
            raise RuntimeError("Decoder layers received different routing masks")
        kwargs["is_causal"] = False
        return registry["sdpa"](module, query, key, value, mask, **kwargs)

    class RoutingRegistry:
        def __getitem__(self, name):
            return dispatch if name == "sdpa" else registry[name]

        def __getattr__(self, name):
            return getattr(registry, name)

    try:
        for index in LAYERS:
            handles.append(decoders[index].register_forward_hook(observed_visual(index)))
        if condition != "native":
            implementation.ALL_ATTENTION_FUNCTIONS = RoutingRegistry()
        logits = forward_logits(backend, batch)
    finally:
        implementation.ALL_ATTENTION_FUNCTIONS = registry
        for handle in handles:
            handle.remove()
    expected = [] if condition == "native" else list(range(28))
    if visits != expected or set(hashes) != set(LAYERS):
        raise RuntimeError("Missing or unexpected attention/visual visits")
    if not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite logits")
    information.update({"visited_layers": visits, "visual_state_sha256": hashes})
    return logits, arrays, information


def summarize(rows):
    groups = {}
    for condition in (*CONDITIONS, "target_gate", "opposite_gate"):
        if condition in {"target_gate", "opposite_gate"}:
            selected = [row for row in rows if row["condition"] ==
                        (f"{row['endpoint']}_gate" if condition == "target_gate" else
                         f"{'last' if row['endpoint'] == 'first' else 'first'}_gate")]
        else:
            selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 32:
            raise RuntimeError("Expected 32 records per reporting condition")
        groups[condition] = {"correct": sum(row["correct"] for row in selected), "total": 32,
                             "static_correct": sum(row["correct"] for row in selected if row["static_readability_control"]),
                             "static_total": 8,
                             "mixed_correct": sum(row["correct"] for row in selected if not row["static_readability_control"]),
                             "mixed_total": 24}
    native = {row["case_id"]: row for row in rows if row["condition"] == "native"}
    sham = [row for row in rows if row["condition"] == "causal_sham"]
    return {"conditions": groups,
            "causal_sham_native_maximum_logit_difference": max(row["native_maximum_logit_difference"] for row in sham),
            "causal_sham_native_prediction_changes": sum(row["predicted_word"] != native[row["case_id"]]["predicted_word"] for row in sham),
            "static_selectivity_gate_passed": all(groups[condition]["static_correct"] == 8 for condition in ("causal_sham", "first_gate", "last_gate")),
            "static_selectivity_rule": "causal_sham, first_gate and last_gate each require all eight static controls correct",
            "unit_of_analysis": "one previously seen source video; constructed conditions are not independent videos",
            "spectral_intervention": False, "long_term_forgetting_inferred": False}


def run(arguments):
    configure_threads()
    import numpy as np
    from .causal_intervention import load_frames
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("A new output directory is required")
    output.mkdir(parents=True)
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "expected_records": 160, "completed_records": 0, "conditions": CONDITIONS,
              "timeout_seconds": 300, "prior_directory": str(PRIOR.resolve())}
    write_json(output / "config.json", config)
    try:
        if json.loads((PRIOR / "config.json").read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("The required baseline study is incomplete")
        priors = {row["case_id"]: row for row in map(json.loads, (PRIOR / "records.jsonl").read_text(encoding="utf-8").splitlines())}
        config["prior_records_sha256"] = digest(PRIOR / "records.jsonl")
        cases = [case for case in build_cases() if case["has_video"]]
        write_json(output / "cases.json", cases)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        config["source_sha256"] = {}
        for filename in SOURCES:
            source = Path(__file__).with_name(filename)
            shutil.copy2(source, snapshot / filename)
            config["source_sha256"][filename] = digest(source)
        shutil.copy2(PROTOCOL, snapshot / "protocol.md")
        config["protocol_sha256"] = digest(PROTOCOL)
        write_json(output / "config.json", config)
        rows = []
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            import torch
            from transformers.integrations import sdpa_attention
            from transformers.models.qwen3_vl import modeling_qwen3_vl

            data = load_frames(INPUT_ROOT)[VIDEO_ID]
            manifest = input_manifest(data)
            write_json(output / "inputs.json", manifest)
            guard.phase = "load_model"
            backend = load_backend(device="cuda")
            if backend.model.dtype != torch.bfloat16:
                raise ValueError("BF16 is required")
            config["model"] = backend.metadata
            for name, module in (("modeling_qwen3_vl.py", modeling_qwen3_vl), ("sdpa_attention.py", sdpa_attention)):
                shutil.copy2(module.__file__, snapshot / name)
                config[name + "_sha256"] = digest(module.__file__)
            option_ids = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
            if any(len(ids) != 1 for ids in option_ids):
                raise ValueError("Option letters must be single tokens")
            option_ids = [ids[0] for ids in option_ids]
            with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                for case_index, case in enumerate(cases):
                    check_deadline(guard)
                    prior = priors[case["case_id"]]
                    frames = [data["frames"][index] for index in case["source_frame_indices"]]
                    batch = backend.prepare_video(frames, manifest["display_timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    input_hashes = {name: tensor_record(tensor) for name, tensor in batch.items()}
                    if input_hashes != prior["input_tensor_hashes"] or option_ids != prior["option_token_ids"]:
                        raise RuntimeError("Inputs changed relative to endpoint_control_v1")
                    baseline = None
                    control_hashes = None
                    for condition in CONDITIONS:
                        check_deadline(guard)
                        guard.phase = f"{case_index}:{condition}"
                        logits, arrays, information = routed_forward(backend, batch, condition)
                        if condition == "native":
                            with np.load(PRIOR / prior["archive_file"], allow_pickle=False) as archive:
                                if not np.array_equal(logits, archive["baseline_logits"]):
                                    raise RuntimeError("Native logits differ from the archived baseline")
                            baseline = logits.copy()
                        if condition == "passthrough" and not np.array_equal(logits, baseline):
                            raise RuntimeError("Passthrough wrapper changed logits")
                        if condition == "causal_sham":
                            control_hashes = information["visual_state_sha256"]
                        if condition in {"first_gate", "last_gate"} and information["visual_state_sha256"] != control_hashes:
                            raise RuntimeError("Routing changed the visual prefix relative to its mask control")
                        allowed = logits[option_ids]
                        predicted = int(allowed.argmax())
                        filename = f"{case_index:02d}_{condition}.npz"
                        np.savez_compressed(output / filename, logits=logits, input_ids=batch["input_ids"][0].cpu().numpy(),
                                            video_positions=np.asarray(batch.video_positions), option_token_ids=np.asarray(option_ids), **arrays)
                        word_order = [case["options"].index(word) for word in ("men", "enm")]
                        row = {**case, "condition": condition, "predicted_word": case["options"][predicted],
                               "predicted_answer_index": predicted, "option_logits": allowed.tolist(),
                               "correct": int(predicted == case["gold_answer_index"]),
                               "gold_margin": float(allowed[case["gold_answer_index"]] - allowed[1 - case["gold_answer_index"]]),
                               "men_minus_enm_margin": float(allowed[word_order[0]] - allowed[word_order[1]]),
                               "option_logit_tie": bool(allowed[0] == allowed[1]), "input_tensor_hashes": input_hashes,
                               "batch": batch.metadata, "capture": information, "archive_file": filename,
                               "archive_sha256": digest(output / filename),
                               "native_maximum_logit_difference": float(np.max(np.abs(logits - baseline)))}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        config["completed_records"] = len(rows)
                        write_json(output / "config.json", config)
                        del arrays, logits
                    print(f"routing {case_index + 1}/32 {case['sequence']} {case['endpoint']} {case['option_order']}", flush=True)
                    if case_index == len(cases) - 1:
                        guard.phase = "unhooked_restoration"
                        if not np.array_equal(forward_logits(backend, batch), baseline):
                            raise RuntimeError("Removing routing hooks did not restore the baseline")
                        config["unhooked_restoration_exact"] = True
                    del batch
            guard.phase = "finalize"
            summary = summarize(rows)
            write_json(output / "summary.json", summary)
            config.update({"peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                           "records_sha256": digest(output / "records.jsonl"), "summary_sha256": digest(output / "summary.json")})
            for filename, expected in config["source_sha256"].items():
                if digest(Path(__file__).with_name(filename)) != expected or digest(snapshot / filename) != expected:
                    raise RuntimeError("Source changed during execution")
            if digest(PROTOCOL) != config["protocol_sha256"] or digest(snapshot / "protocol.md") != config["protocol_sha256"]:
                raise RuntimeError("Protocol changed during execution")
            check_deadline(guard)
            del backend
            torch.cuda.empty_cache()
        resources = json.loads((output / "resource_summary.json").read_text(encoding="utf-8"))
        if resources["status"] != "complete" or (output / "resource_stop.json").exists():
            raise RuntimeError("Resource monitoring did not certify completion")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "resource_summary_sha256": digest(output / "resource_summary.json")})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
