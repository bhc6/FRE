"""Read-only all-layer Qwen3-VL/SmolVLM temporal and last-query diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from .resource_guard import ResourceGuard, configure_threads


QUESTION_SHA = "76e60bb8b3cc522dfd3aa6323782d753ef58caa78b7dbcbe864224f34d7d3504"
FRAME_SHA = "82d93ca7f9acb6fc8062160676a2f7e20885298218a0fd8430bdda11aac6b9b9"
CONDITIONS = ("real", "static_first", "static_last")
PROTOCOL = Path("docs/cross_model_probe_v2_protocol.md")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_hash(values):
    import torch

    copied = values.detach().cpu().contiguous()
    return {"shape": list(copied.shape), "dtype": str(copied.dtype),
            "sha256": hashlib.sha256(copied.view(torch.uint8).numpy().tobytes()).hexdigest()}


def to_list(values):
    return values.detach().double().cpu().tolist()


def temporal_spectrum(values):
    """Transform T in [real KV head,T,spatial token,channel], never channels."""
    import torch

    require(values.ndim == 4 and values.shape[1] in (8, 16), "Expected [head,T,P,D] with T=8 or T=16")
    values = values.float()
    time_bins = values.shape[1]
    spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
    multiplicity = torch.full((spectrum.shape[1],), 2.0, device=values.device)
    multiplicity[0] = 1.0
    multiplicity[-1] = 1.0
    energy = spectrum.abs().square().sum(dim=(-1, -2)) * multiplicity[None, :]
    time_energy = values.square().sum(dim=(1, 2, 3))
    dc = energy[:, 0]
    non_dc = energy[:, 1:].sum(dim=1)
    total = dc + non_dc
    parseval = (total - time_energy).abs() / time_energy.clamp_min(1e-20)
    require(bool(torch.isfinite(energy).all()) and bool(torch.all(parseval <= 2e-5)), "Temporal FFT Parseval or finiteness check failed")
    return {"dc_energy": to_list(dc), "non_dc_energy": to_list(non_dc), "total_energy": to_list(total),
            "non_dc_fraction": [float(ac / full) if full > 0 else None for ac, full in zip(to_list(non_dc), to_list(total))],
            "energy_per_element": to_list(total / values[0].numel()), "frequency_energy": to_list(energy),
            "frequencies_cycles_per_time_group": to_list(torch.fft.rfftfreq(time_bins, device=values.device)),
            "parseval_relative_error": to_list(parseval), "shape_head_time_patch_channel": list(values.shape),
            "fft_axis": "time axis 1; each spatial token/channel transformed separately before energy summation",
            "normalization": "orthonormal rFFT, factor two for interior real frequencies"}


def architecture(backend, model_name):
    if model_name == "qwen":
        from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation

        layers = backend.model.model.language_model.layers
    elif model_name == "smol":
        from transformers.models.llama import modeling_llama as implementation

        layers = backend.model.model.text_model.layers
    else:
        raise ValueError("Unknown model")
    require(all(layer.self_attn.config._attn_implementation == "sdpa" for layer in layers), "Only the pinned native SDPA path is supported")
    return layers, implementation


def layout(backend, batch, model_name):
    import numpy as np

    identifiers = batch["input_ids"][0].detach().cpu().numpy()
    require(batch["input_ids"].shape[0] == 1 and bool((batch["attention_mask"] == 1).all()), "Only unpadded single-example prefill is supported")
    if model_name == "qwen":
        positions = np.asarray(batch.video_positions)
        marker = int(backend.model.config.video_token_id)
        timestamps = np.asarray(batch.metadata["hidden_timestamps_s"], dtype=float)
        require(positions.shape[0] == 8 and batch.metadata["frame_groups"] == np.arange(16).reshape(8, 2).tolist(), "Qwen must retain eight adjacent-frame pairs")
        require(batch.metadata["visual_tokens"] <= 512 and len(identifiers) <= 1024, "Qwen diagnostic token budget exceeded")
    else:
        marker = int(backend.image_token_id)
        found = np.flatnonzero(identifiers == marker)
        require(len(found) == 16 * backend.image_seq_len, "Smol image token count differs from sixteen frames")
        positions = found.reshape(16, backend.image_seq_len)
        timestamps = np.asarray(batch.metadata["source_timestamps_s"], dtype=float)
        require(tuple(batch["pixel_values"].shape) == (1, 16, 3, backend.image_size, backend.image_size), "Smol visual tensor frame layout changed")
        require(batch.metadata["frame_groups"] == [[index] for index in range(16)], "Smol frame group ordering changed")
        require(len(identifiers) <= 2048 and positions.size <= 1024, "Smol diagnostic token budget exceeded")
    require(positions.ndim == 2 and np.array_equal(positions.flatten(), np.flatnonzero(identifiers == marker)), "Visual token groups do not match the actual input IDs")
    require(np.all(np.diff(positions, axis=1) == 1) and np.all(np.diff(positions[:, 0]) > positions.shape[1]), "Visual frame groups must be separated contiguous token blocks")
    require(positions.max() < len(identifiers) - 1 and identifiers[-1] != marker, "The sampled query must be the final text token")
    intervals = np.diff(timestamps)
    require(len(timestamps) == len(positions) and np.all(intervals > 0), "Visual timestamps changed")
    uniform = bool(np.allclose(intervals, np.median(intervals), rtol=1e-5, atol=1e-7))
    require(uniform, "Time FFT requires a uniform displayed grid")
    return positions, {"time_groups": len(positions), "spatial_tokens_per_group": positions.shape[1],
                       "visual_token_id": marker, "visual_token_positions": positions.tolist(),
                       "time_group_timestamps_s": timestamps.tolist(), "uniform_time_grid": uniform,
                       "time_group_spacing_s": float(np.median(intervals)),
                       "sampling_hz": float(1 / np.median(intervals)), "nyquist_hz": float(0.5 / np.median(intervals)),
                       "frequency_interpretation": "sampled displayed grid, without temporal antialiasing; not a direct physical motion spectrum"}


def last_query_reference(module, query, key, value, attention_mask, positions, scaling):
    import torch

    require(query.ndim == key.ndim == value.ndim == 4 and query.shape[0] == key.shape[0] == value.shape[0] == 1, "Expected [1,head,token,channel]")
    require(key.shape == value.shape and query.shape[-2:] == key.shape[-2:], "Only uncached aligned Q/K/V prefill is supported")
    query_heads, kv_heads = query.shape[1], key.shape[1]
    require(query_heads % kv_heads == 0, "Unsupported GQA mapping")
    mapping = torch.arange(query_heads, device=query.device) // (query_heads // kv_heads)
    query_last = query[0, :, -1].float()
    keys = key[0].float()
    values = value[0].float()
    scores = torch.einsum("hd,hnd->hn", query_last, keys[mapping]) * scaling
    allowed = torch.ones_like(scores, dtype=torch.bool)
    mask_description = "last row of uncached causal prefill sees all keys"
    if attention_mask is not None:
        require(attention_mask.ndim == 4 and attention_mask.shape[0] == 1, "Unsupported mask shape")
        selected = attention_mask[0, :, -1, :key.shape[-2]]
        require(selected.shape[0] in (1, query_heads), "Unsupported mask head dimension")
        if selected.dtype == torch.bool:
            allowed = allowed & selected
            scores = scores.masked_fill(~selected, -torch.inf)
            mask_description = "actual boolean mask, True means allowed"
        else:
            selected = selected.float()
            require(bool(((selected == 0) | (selected < -1e20)).all()), "Nonbinary additive attention bias requires a separate audit")
            allowed = allowed & (selected == 0)
            scores = scores + selected
            mask_description = "actual additive mask; allowed entries exactly zero"
    require(bool(allowed.any(dim=-1).all()), "Last query has no visible keys")
    probabilities = scores.softmax(dim=-1)
    reference = torch.einsum("hn,hnd->hd", probabilities, values[mapping])
    visual_attention = probabilities[:, positions]
    original_visual_values = values[:, positions]
    original_visual_keys = keys[:, positions]
    visual_values = original_visual_values[mapping]
    time_context = (visual_attention.unsqueeze(-1) * visual_values).sum(dim=2)
    visual_context = time_context.sum(dim=1)
    visual_mass = visual_attention.sum(dim=(1, 2))
    temporal_mass = visual_attention.sum(dim=2)
    conditional_visual = visual_attention / visual_mass[:, None, None].clamp_min(1e-30)
    conditional_time = temporal_mass / visual_mass[:, None].clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    visual_entropy = -(conditional_visual * conditional_visual.clamp_min(1e-30).log()).sum(dim=(1, 2))
    temporal_entropy = -(conditional_time * conditional_time.clamp_min(1e-30).log()).sum(dim=1)
    metrics = torch.stack((visual_mass, entropy, entropy / allowed.sum(dim=-1).float().log().clamp_min(1e-30),
                           visual_entropy, temporal_entropy,
                           torch.linalg.vector_norm(query_last, dim=-1), torch.linalg.vector_norm(reference, dim=-1),
                           torch.linalg.vector_norm(visual_context, dim=-1), allowed.sum(dim=-1).float()), dim=1)
    require(bool(torch.isfinite(metrics).all()), "Nonfinite attention diagnostics")
    return {"reference_context": reference, "visual_context": visual_context, "metrics": metrics,
            "temporal_attention_mass": temporal_mass, "query_heads": query_heads, "kv_heads": kv_heads,
            "head_dim": query.shape[-1], "gqa_mapping": mapping.detach().cpu().tolist(), "scaling": float(scaling),
            "mask_description": mask_description,
            "value_temporal_spectrum": temporal_spectrum(original_visual_values),
            "post_rope_key_temporal_spectrum": temporal_spectrum(original_visual_keys),
            "weighted_visual_context_temporal_spectrum": temporal_spectrum(time_context.unsqueeze(2))}


def finalize_layer(index, state, model_name):
    import torch

    native = state["native_context"].float()
    reference = state["reference_context"]
    visual = state["visual_context"]
    native_norm = torch.linalg.vector_norm(native, dim=-1)
    error = torch.linalg.vector_norm(reference - native, dim=-1)
    error_max = (reference - native).abs().max(dim=-1).values
    extra = torch.stack((native_norm, torch.linalg.vector_norm(visual, dim=-1) / native_norm.clamp_min(1e-20),
                         error / native_norm.clamp_min(1e-20), error_max,
                         (native * visual).sum(dim=-1) / (native_norm * torch.linalg.vector_norm(visual, dim=-1)).clamp_min(1e-20)), dim=1)
    combined = torch.cat((state["metrics"], extra), dim=1).double().cpu().tolist()
    names = ("visual_attention_mass", "attention_entropy_nats", "attention_entropy_normalized", "conditional_visual_entropy_nats",
             "conditional_temporal_entropy_nats", "query_l2", "reference_context_l2", "visual_context_l2", "allowed_key_count",
             "native_context_l2", "visual_to_native_context_ratio", "reference_native_relative_l2_error", "reference_native_maximum_error",
             "visual_native_context_cosine")
    temporal_mass = to_list(state["temporal_attention_mass"])
    heads = []
    for head, values in enumerate(combined):
        item = {"query_head": head, "source_kv_head": state["gqa_mapping"][head], **dict(zip(names, values)),
                "temporal_attention_mass": temporal_mass[head]}
        if item["visual_attention_mass"] == 0:
            item["conditional_visual_entropy_nats"] = None
            item["conditional_temporal_entropy_nats"] = None
        heads.append(item)
    branches = ("decoder_input", "attention_output", "pre_ffn", "ffn_output", "post_ffn")
    norms = {name: float(torch.linalg.vector_norm(state[name].float())) for name in branches}
    attention_identity = bool(torch.equal(state["decoder_input"] + state["attention_output"], state["pre_ffn"]))
    ffn_identity = bool(torch.equal(state["pre_ffn"] + state["ffn_output"], state["post_ffn"]))
    require(attention_identity and ffn_identity, "Native residual addition identity failed")
    ffn = {"last_query_branch_l2": norms, "attention_residual_identity_exact": attention_identity,
           "ffn_residual_identity_exact": ffn_identity,
           "attention_over_decoder_input_l2": norms["attention_output"] / max(norms["decoder_input"], 1e-20),
           "ffn_over_attention_output_l2": norms["ffn_output"] / max(norms["attention_output"], 1e-20),
           "ffn_over_pre_ffn_residual_l2": norms["ffn_output"] / max(norms["pre_ffn"], 1e-20),
           "post_ffn_over_pre_ffn_l2": norms["post_ffn"] / max(norms["pre_ffn"], 1e-20)}
    aggregate = {name: float(state["metrics"][:, position].mean()) for position, name in enumerate(names[:9])}
    aggregate.update({"visual_to_native_context_ratio_mean": float(extra[:, 1].mean()),
                      "visual_to_native_context_ratio_median": float(extra[:, 1].median()),
                      "reference_native_relative_l2_error_maximum": float(extra[:, 2].max())})
    return {"model": model_name, "layer": index, "query_heads": state["query_heads"], "kv_heads": state["kv_heads"],
            "head_dim": state["head_dim"], "scaling": state["scaling"], "mask": state["mask_description"],
            "reference_attention": "FP32 reconstruction from actual post-RoPE Q, actual K/V, complete-key mask; not exported SDPA weights or causal attribution",
            "head_metrics": heads, "head_aggregate": aggregate, "ffn_metrics": ffn,
            "value_temporal_spectrum": state["value_temporal_spectrum"],
            "post_rope_key_temporal_spectrum": state["post_rope_key_temporal_spectrum"],
            "weighted_visual_context_temporal_spectrum": state["weighted_visual_context_temporal_spectrum"],
            "value_spectrum_unit": "actual KV heads, never GQA-replicated independent samples",
            "key_spectrum_caveat": "actual post-RoPE keys include positional and causal context effects, including on repeated images"}


def capture(backend, batch, model_name, positions):
    import torch
    from .passive_probe import forward_logits

    layers, implementation = architecture(backend, model_name)
    registry = implementation.ALL_ATTENTION_FUNCTIONS
    indices = {id(layer.self_attn): index for index, layer in enumerate(layers)}
    states = {}
    records = {}
    visits = []
    handles = []
    selected_positions = torch.as_tensor(positions, device=batch["input_ids"].device, dtype=torch.long)
    token_count = batch["input_ids"].shape[1]

    def sample(index, name, tensor):
        require(tensor.ndim == 3 and tensor.shape[:2] == (1, token_count), "Unexpected decoder sequence layout")
        bucket = states.setdefault(index, {})
        require(name not in bucket, "Duplicate last-query branch capture")
        bucket[name] = tensor[0, -1].detach().clone()

    def input_hook(index, name):
        def hook(module, arguments):
            sample(index, name, arguments[0])
        return hook

    def decoder_input_hook(index):
        def hook(module, arguments, keywords):
            sample(index, "decoder_input", arguments[0] if arguments else keywords["hidden_states"])
        return hook

    def output_hook(index, name, tuple_output=False):
        def hook(module, arguments, output):
            sample(index, name, output[0] if tuple_output else output)
            if name == "post_ffn":
                records[index] = finalize_layer(index, states.pop(index), model_name)
        return hook

    def projection_hook(index, head_dim):
        def hook(module, arguments):
            values = arguments[0]
            require(values.shape[:2] == (1, token_count), "Native context shape changed")
            require("native_context" not in states[index], "Duplicate native context capture")
            states[index]["native_context"] = values[0, -1].reshape(-1, head_dim).detach().clone()
        return hook

    def observed_sdpa(module, query, key, value, attention_mask, **kwargs):
        if id(module) in indices:
            index = indices[id(module)]
            require(index not in visits and kwargs.get("dropout", 0.0) == 0, "Expected one deterministic prefill per text attention layer")
            require(kwargs.get("head_mask") is None, "Head masks are not supported")
            visits.append(index)
            state = last_query_reference(module, query, key, value, attention_mask, selected_positions,
                                         kwargs.get("scaling", module.scaling))
            states[index].update(state)
        return registry["sdpa"](module, query, key, value, attention_mask, **kwargs)

    class ReadOnlyRegistry:
        def __getitem__(self, name):
            return observed_sdpa if name == "sdpa" else registry[name]

        def __getattr__(self, name):
            return getattr(registry, name)

    try:
        for index, decoder in enumerate(layers):
            handles.extend((decoder.register_forward_pre_hook(decoder_input_hook(index), with_kwargs=True),
                            decoder.self_attn.o_proj.register_forward_pre_hook(projection_hook(index, decoder.self_attn.head_dim)),
                            decoder.self_attn.register_forward_hook(output_hook(index, "attention_output", True)),
                            decoder.post_attention_layernorm.register_forward_pre_hook(input_hook(index, "pre_ffn")),
                            decoder.mlp.register_forward_hook(output_hook(index, "ffn_output")),
                            decoder.register_forward_hook(output_hook(index, "post_ffn"))))
        implementation.ALL_ATTENTION_FUNCTIONS = ReadOnlyRegistry()
        logits = forward_logits(backend, batch)
    finally:
        implementation.ALL_ATTENTION_FUNCTIONS = registry
        for handle in handles:
            handle.remove()
    require(visits == list(range(len(layers))) and set(records) == set(range(len(layers))) and not states, "Not every text layer was captured exactly once")
    return logits, [records[index] for index in range(len(layers))]


def cases(arguments):
    from .evaluate_small import build_prompt

    require(digest(arguments.questions) == QUESTION_SHA and digest(arguments.frames_manifest) == FRAME_SHA, "Only the old eight validation sources are authorized")
    questions = sorted(read_rows(arguments.questions), key=lambda row: (row["video_id"], row["query_id"]))
    frames = {row["video_id"]: row for row in read_rows(arguments.frames_manifest)}
    require(len(questions) == 40 and len(frames) == 8 and {row["video_id"] for row in questions} == set(frames), "Frozen old validation coverage changed")
    selected = set(frames) if not arguments.video_ids else set(arguments.video_ids.split(","))
    require(selected and selected <= set(frames), "Requested video is outside the old validation set")
    first = {}
    for question in questions:
        first.setdefault(question["video_id"], question)
    output = []
    for video_id in sorted(selected):
        source = frames[video_id]
        question = first[video_id]
        require(len(source["frames"]) == len(source["timestamps_s"]) == 16, "Exactly sixteen input frames are required")
        for condition in CONDITIONS:
            real = condition == "real"
            position = 0 if condition == "static_first" else -1
            output.append({"video_id": video_id, "query_id": question["query_id"], "case_id": question["query_id"] + ":" + condition,
                           "condition": condition, "question": question if real else {key: value for key, value in question.items() if key not in {"answer_index", "answer_text"}},
                           "prompt": build_prompt(question), "gold_answer_index": question["answer_index"] if real else None,
                           "options": question["options"], "frames": source["frames"] if real else [source["frames"][position]] * 16,
                           "frame_sha256": source["frame_sha256"] if real else [source["frame_sha256"][position]] * 16,
                           "display_timestamps_s": source["timestamps_s"],
                           "decoded_source_timestamps_s": source["source_frame_timestamps_s"] if real else [source["source_frame_timestamps_s"][position]] * 16,
                           "original_source_video_sha256": source["video_sha256"]})
    return output


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .endpoint_control import check_deadline
    from .passive_probe import forward_logits

    output = arguments.output_dir.resolve()
    require(not output.exists(), "Probe output must be new")
    selected_cases = cases(arguments)
    output.mkdir(parents=True)
    source_names = ("cross_model_probe_v2.py", "smolvlm_backend.py", "qwen3_backend.py", "passive_probe.py", "resource_guard.py", "evaluate_small.py", "endpoint_control.py")
    config = {"status": "started", "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_choice": arguments.model,
              "scope": "paired read-only development diagnostics on old validation; no new confirmation scores",
              "selection": "first query after sorting (video_id,query_id), all three real/static conditions",
              "expected_cases": len(selected_cases), "completed_cases": 0, "full_scope_cases": 24,
              "full_eight_source_scope": len(selected_cases) == 24, "selected_video_ids": sorted({row["video_id"] for row in selected_cases}),
              "questions_sha256": QUESTION_SHA, "frames_manifest_sha256": FRAME_SHA, "protocol_sha256": digest(PROTOCOL),
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in source_names},
              "maximum_input_tokens": 1024 if arguments.model == "qwen" else 2048,
              "maximum_visual_tokens": 512 if arguments.model == "qwen" else 1024,
              "diagnostic_attention_precision": "FP32 reference from actual SDPA inputs; native output untouched",
              "spectrum_storage": "per actual KV head/time-frequency energy; no full Q/K/V archives"}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for name in source_names:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(PROTOCOL, snapshot / "protocol.md")
    write_json(output / "cases.json", selected_cases)
    config["cases_sha256"] = digest(output / "cases.json")
    write_json(output / "config.json", config)
    try:
        with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
            guard.configure_torch()
            guard.phase = "load_model"
            if arguments.model == "qwen":
                from .qwen3_backend import load_backend, DEFAULT_MAX_PIXELS
            else:
                from .smolvlm_backend import load_backend, DEFAULT_MAX_PIXELS
            backend = load_backend(device="cuda")
            layers, implementation = architecture(backend, arguments.model)
            require(len(layers) == (28 if arguments.model == "qwen" else 32), "Pinned model layer count changed")
            expected_geometry = (16, 8, 128) if arguments.model == "qwen" else (15, 5, 64)
            configuration = backend.model.config.text_config
            require((configuration.num_attention_heads, configuration.num_key_value_heads, layers[0].self_attn.head_dim) == expected_geometry, "Pinned model head geometry changed")
            config["model"] = backend.metadata
            config["layers"] = len(layers)
            config["attention_geometry_query_kv_channel"] = list(expected_geometry)
            from transformers.integrations import sdpa_attention
            from transformers.models.smolvlm import modeling_smolvlm
            config["implementation_sha256"] = {module.__name__: {"path": inspect.getfile(module), "sha256": digest(inspect.getfile(module))}
                                                for module in (implementation, sdpa_attention, modeling_smolvlm)}
            tokens = [backend.processor.tokenizer.encode(label, add_special_tokens=False) for label in "ABC"]
            require(all(len(token) == 1 for token in tokens) and len({token[0] for token in tokens}) == 3, "Original options require unique single-token labels")
            option_ids = [token[0] for token in tokens]
            config["option_token_ids"] = option_ids
            write_json(output / "config.json", config)
            with (output / "records.jsonl").open("x", encoding="utf-8") as records_stream, (output / "layers.jsonl").open("x", encoding="utf-8") as layer_stream:
                for case_index, case in enumerate(selected_cases):
                    check_deadline(guard)
                    guard.phase = f"prepare:{case['case_id']}"
                    require([digest(path) for path in case["frames"]] == case["frame_sha256"], "Frame content changed")
                    frames = [Image.open(path).convert("RGB") for path in case["frames"]]
                    batch = backend.prepare_video(frames, case["display_timestamps_s"], case["prompt"], DEFAULT_MAX_PIXELS)
                    positions, layout_record = layout(backend, batch, arguments.model)
                    input_hashes = {name: tensor_hash(value) for name, value in batch.items()}
                    guard.phase = f"native:{case['case_id']}"
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    native = forward_logits(backend, batch)
                    native_seconds = time.perf_counter() - started
                    check_deadline(guard)
                    guard.phase = f"passive_all_layers:{case['case_id']}"
                    started = time.perf_counter()
                    passive, layer_records = capture(backend, batch, arguments.model, positions)
                    passive_seconds = time.perf_counter() - started
                    require(np.array_equal(native, passive), "Read-only capture changed native whole-vocabulary logits")
                    check_deadline(guard)
                    guard.phase = f"restored:{case['case_id']}"
                    restored = forward_logits(backend, batch)
                    require(np.array_equal(native, restored), "Native output did not restore after hooks")
                    require(input_hashes == {name: tensor_hash(value) for name, value in batch.items()}, "Input tensors changed during capture")
                    require(np.isfinite(native).all(), "Native logits are nonfinite")
                    archive = output / f"{case_index:04d}.npz"
                    np.savez_compressed(archive, native_logits=native, passive_logits=passive, restored_logits=restored,
                                        input_ids=batch["input_ids"][0].detach().cpu().numpy(), visual_positions=positions,
                                        time_group_timestamps_s=np.asarray(layout_record["time_group_timestamps_s"]), option_token_ids=np.asarray(option_ids))
                    option_logits = native[option_ids]
                    predicted = int(option_logits.argmax())
                    common = {"case_id": case["case_id"], "query_id": case["query_id"], "video_id": case["video_id"],
                              "condition": case["condition"], "model": arguments.model, "archive_file": archive.name,
                              "archive_sha256": digest(archive)}
                    row = {**common, "question": case["question"], "gold_answer_index": case["gold_answer_index"],
                           "option_logits": option_logits.tolist(), "option_token_ids": option_ids, "predicted_answer_index": predicted,
                           "correct": int(predicted == case["gold_answer_index"]) if case["condition"] == "real" else None,
                           "maximum_tie_indices": np.flatnonzero(option_logits == option_logits.max()).tolist(),
                           "native_passive_whole_vocabulary_exact": True, "native_restored_whole_vocabulary_exact": True,
                           "input_tensor_hashes": input_hashes, "frame_sha256": case["frame_sha256"],
                           "display_timestamps_s": case["display_timestamps_s"], "decoded_source_timestamps_s": case["decoded_source_timestamps_s"],
                           "layout": layout_record, "batch": batch.metadata, "native_forward_seconds": native_seconds,
                           "passive_forward_seconds": passive_seconds, "captured_layer_count": len(layer_records)}
                    records_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    for layer_record in layer_records:
                        layer_stream.write(json.dumps({**common, **layer_record}, ensure_ascii=False, allow_nan=False) + "\n")
                    records_stream.flush()
                    layer_stream.flush()
                    config["completed_cases"] += 1
                    write_json(output / "config.json", config)
                    print(f"probed {arguments.model} {case['case_id']} ({config['completed_cases']}/{config['expected_cases']})", flush=True)
                    del batch, frames, native, passive, restored, layer_records
            config["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            config["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            check_deadline(guard)
            del backend
            torch.cuda.empty_cache()
        require(not (output / "resource_stop.json").exists() and config["completed_cases"] == config["expected_cases"], "Probe incomplete or resource-stopped")
        require(digest(PROTOCOL) == config["protocol_sha256"], "Protocol changed during execution")
        for name, expected in config["source_sha256"].items():
            require(digest(Path(__file__).with_name(name)) == expected and digest(snapshot / name) == expected, "Probe source changed during execution")
        config.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "records_sha256": digest(output / "records.jsonl"), "layers_sha256": digest(output / "layers.jsonl")})
        write_json(output / "config.json", config)
    except BaseException as error:
        config.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def self_check():
    configure_threads()
    import numpy as np
    import torch
    from torch import nn
    from transformers import LlamaConfig, LlamaModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from .passive_probe import forward_logits

    torch.set_num_threads(2)
    checks = []
    for time_bins in (8, 16):
        torch.manual_seed(17)
        repeated = torch.randn(2, 1, 3, 4).expand(-1, time_bins, -1, -1)
        static = temporal_spectrum(repeated)
        require(max(static["non_dc_energy"]) == 0.0, "Static value has spurious temporal energy")
        alternating = torch.tensor([(-1.0) ** index for index in range(time_bins)]).view(1, time_bins, 1, 1)
        cancellation = alternating * torch.tensor([1.0, -1.0]).view(1, 1, 2, 1)
        motion = temporal_spectrum(cancellation)
        require(torch.equal(cancellation.mean(dim=2), torch.zeros_like(cancellation.mean(dim=2))) and motion["non_dc_fraction"] == [1.0], "Spatial averaging leaked into temporal FFT")
        checks.extend([f"static_T{time_bins}", f"spatial_cancellation_T{time_bins}"])
    for model_name in ("smol", "qwen"):
        torch.manual_seed(23)
        if model_name == "smol":
            configuration = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_dropout=0.0)
            configuration._attn_implementation = "sdpa"
            language = LlamaModel(configuration).eval()
        else:
            configuration = Qwen3VLTextConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_dropout=0.0,
                                             rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})
            configuration._attn_implementation = "sdpa"
            language = Qwen3VLTextModel(configuration).eval()

        class TextWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                setattr(self.model, "text_model" if model_name == "smol" else "language_model", language)
                self.lm_head = nn.Linear(64, 128, bias=False)

            def forward(self, input_ids, attention_mask, logits_to_keep=1, **kwargs):
                result = language(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                return SimpleNamespace(logits=self.lm_head(result.last_hidden_state[:, -logits_to_keep:]))

        backend = SimpleNamespace(model=TextWrapper().eval())
        time_bins = 16 if model_name == "smol" else 8
        length = time_bins * 2 + 4
        batch = {"input_ids": torch.randint(0, 128, (1, length)), "attention_mask": torch.ones((1, length), dtype=torch.long)}
        positions = np.arange(time_bins * 2).reshape(time_bins, 2)
        native = forward_logits(backend, batch)
        passive, records = capture(backend, batch, model_name, positions)
        restored = forward_logits(backend, batch)
        require(np.array_equal(native, passive) and np.array_equal(native, restored), "Tiny architecture passive/restoration check failed")
        require(len(records) == 2 and all(len(row["head_metrics"]) == 4 and len(row["value_temporal_spectrum"]["dc_energy"]) == 2 for row in records), "GQA counts or all-layer capture invalid")
        require(max(row["head_aggregate"]["reference_native_relative_l2_error_maximum"] for row in records) < 1e-5, "FP32 attention reconstruction differs from CPU SDPA")
        checks.extend([f"{model_name}_actual_attention_class", f"{model_name}_all_layers_GQA", f"{model_name}_whole_vocab_exact_restored", f"{model_name}_residual_identities"])
    return {"status": "passed", "checks": checks, "device": "CPU", "actual_checkpoint_loaded": False,
            "note": "Tiny random text models test pinned architecture hooks, not GPU/checkpoint behavior"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen", "smol"))
    parser.add_argument("--questions", type=Path, default=Path("data/internal_controller_v1/validation.questions.jsonl"))
    parser.add_argument("--frames-manifest", type=Path, default=Path("data/internal_controller_v1/frames16/validation.frames.manifest.jsonl"))
    parser.add_argument("--video-ids")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        print(json.dumps(self_check(), ensure_ascii=False))
    else:
        if arguments.model is None or arguments.output_dir is None:
            parser.error("A model and new output directory are required")
        run(arguments)


if __name__ == "__main__":
    main()
