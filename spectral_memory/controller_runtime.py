"""Frozen last-layer readout replay and audited online controller insertion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .resource_guard import configure_threads

configure_threads()

import numpy as np
import torch
from torch import nn


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def tensor_hash(tensor):
    values = tensor.detach().cpu().contiguous()
    return {"shape": list(values.shape), "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.view(torch.uint8).numpy().tobytes()).hexdigest()}


def cpu_float(tensor):
    return tensor.detach().float().cpu().numpy().copy()


def extract_features(query, key, value, attention_mask, scaling, positions):
    from .learned_readout import build_bases

    query_last = query[0, :, -1].float()
    mapping = torch.arange(query.shape[1], device=query.device) // (query.shape[1] // key.shape[1])
    mapped_keys = key[0, mapping].float()
    mapped_values = value[0, mapping].float()
    scores = torch.einsum("hd,hnd->hn", query_last, mapped_keys) * scaling
    if attention_mask is not None:
        mask = attention_mask[0, :, -1, :key.shape[-2]]
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.float()
    weights = scores.softmax(dim=-1)
    visual_weights = weights[:, positions]
    visual_values = mapped_values[:, positions]
    bases = build_bases(visual_weights, visual_values)
    reference_context = torch.einsum("hn,hnd->hd", weights, mapped_values)
    return query_last, bases, {"attention_weights": weights, "reference_context": reference_context,
                               "visual_attention_mass": visual_weights.sum(dim=(-1, -2))}


class FrozenTail(nn.Module):
    def __init__(self, payload):
        super().__init__()
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP, Qwen3VLTextRMSNorm

        configuration = Qwen3VLTextConfig(**payload["text_config"])
        self.mlp = Qwen3VLTextMLP(configuration)
        self.post_norm = Qwen3VLTextRMSNorm(configuration.hidden_size, eps=configuration.rms_norm_eps)
        self.final_norm = Qwen3VLTextRMSNorm(configuration.hidden_size, eps=configuration.rms_norm_eps)
        self.register_buffer("output_weight", payload["output_weight"].clone())
        self.register_buffer("answer_weight", payload["answer_weight"].clone())
        self.mlp.load_state_dict(payload["mlp"])
        self.post_norm.load_state_dict(payload["post_norm"])
        self.final_norm.load_state_dict(payload["final_norm"])
        self.to(dtype=torch.bfloat16)
        self.requires_grad_(False)

    def forward(self, decoder_input, attention_delta, delta_context):
        correction = torch.nn.functional.linear(delta_context.flatten(-2).to(torch.bfloat16), self.output_weight)
        modified_attention = attention_delta.to(torch.bfloat16) + correction
        residual = decoder_input.to(torch.bfloat16) + modified_attention
        hidden = residual + self.mlp(self.post_norm(residual))
        hidden = self.final_norm(hidden)
        return torch.nn.functional.linear(hidden, self.answer_weight).float()


def save_tail(backend, path, option_ids):
    decoder = backend.model.model.language_model.layers[-1]
    if decoder.self_attn.o_proj.bias is not None:
        raise ValueError("The frozen residual projection assumes no output bias")
    payload = {"text_config": backend.model.config.text_config.to_dict(), "option_token_ids": option_ids,
               "output_weight": decoder.self_attn.o_proj.weight.detach().cpu(),
               "answer_weight": backend.model.lm_head.weight.detach().cpu()[option_ids],
               "mlp": {name: value.detach().cpu() for name, value in decoder.mlp.state_dict().items()},
               "post_norm": {name: value.detach().cpu() for name, value in decoder.post_attention_layernorm.state_dict().items()},
               "final_norm": {name: value.detach().cpu() for name, value in backend.model.model.language_model.norm.state_dict().items()}}
    torch.save(payload, path)
    return {"option_token_ids": option_ids, "text_config": payload["text_config"],
            "tensors": {name: tensor_hash(value) for name, value in payload.items() if isinstance(value, torch.Tensor)},
            "modules": {name: {key: tensor_hash(value) for key, value in payload[name].items()}
                        for name in ("mlp", "post_norm", "final_norm")}}


def controlled_forward(backend, batch, controller=None, retain_features=False):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as implementation
    from .passive_probe import forward_logits

    if batch.video_positions is None or batch.video_positions.shape[0] != 8:
        raise ValueError("Controller requires eight existing visual time groups")
    if batch.metadata["input_tokens"] > 1024 or batch.metadata["visual_tokens"] > 512:
        raise ValueError("Controller input exceeds the frozen budget")
    decoder = backend.model.model.language_model.layers[-1]
    attention = decoder.self_attn
    if len(backend.model.model.language_model.layers) != 28 or attention.config._attn_implementation != "sdpa":
        raise ValueError("Expected the frozen 28-layer SDPA model")
    positions = torch.as_tensor(batch.video_positions, device=backend.device, dtype=torch.long)
    registry = implementation.ALL_ATTENTION_FUNCTIONS
    state = {}
    handles = []
    visits = []

    def decoder_input(module, arguments):
        state["decoder_input"] = arguments[0][0, -1].detach().clone()

    def projection_input(module, arguments):
        state["context"] = arguments[0][0, -1].reshape(16, 128).detach().clone()

    def dispatch(module, query, key, value, attention_mask, **kwargs):
        if module is attention:
            if visits or query.shape[0] != 1 or query.shape[-2] != key.shape[-2] or kwargs.get("dropout", 0.0):
                raise ValueError("Expected a single deterministic uncached prefill")
            visits.append(27)
            query_last, bases, reference = extract_features(query, key, value, attention_mask,
                                                          kwargs.get("scaling", module.scaling), positions)
            state.update({"query": query_last, "bases": bases, **reference})
        return registry["sdpa"](module, query, key, value, attention_mask, **kwargs)

    def attention_output(module, arguments, output):
        values = output[0]
        state["attention_delta"] = values[0, -1].detach().clone()
        if controller is None:
            state["delta_context"] = torch.zeros_like(state["context"], dtype=torch.float32)
            return output
        delta, diagnostics = controller(state["query"][None],
                                        {name: tensor[None] for name, tensor in state["bases"].items()},
                                        state["context"][None].float())
        state["delta_context"] = delta[0]
        state["controller_diagnostics"] = diagnostics
        projected = torch.nn.functional.linear(delta.flatten(-2).to(values.dtype), attention.o_proj.weight)[0]
        replaced = values.clone()
        replaced[0, -1] = values[0, -1] + projected
        return (replaced, *output[1:])

    class ObservedRegistry:
        def __getitem__(self, name):
            return dispatch if name == "sdpa" else registry[name]

        def __getattr__(self, name):
            return getattr(registry, name)

    try:
        handles.extend((decoder.register_forward_pre_hook(decoder_input),
                        attention.o_proj.register_forward_pre_hook(projection_input),
                        attention.register_forward_hook(attention_output)))
        implementation.ALL_ATTENTION_FUNCTIONS = ObservedRegistry()
        logits = forward_logits(backend, batch)
    finally:
        implementation.ALL_ATTENTION_FUNCTIONS = registry
        for handle in handles:
            handle.remove()
    if visits != [27] or not np.isfinite(logits).all():
        raise RuntimeError("Incomplete or nonfinite readout")
    error = torch.linalg.vector_norm(state["reference_context"] - state["context"].float())
    relative_error = float(error / torch.linalg.vector_norm(state["context"].float()).clamp_min(1e-12))
    delta_norm = float(torch.linalg.vector_norm(state["delta_context"]))
    context_norm = float(torch.linalg.vector_norm(state["context"].float()))
    metadata = {"reference_attention": "FP32 reconstruction from actual Q/K and full-key mask; not exported SDPA weights",
                "reference_context_relative_l2_error": relative_error,
                "delta_context_l2": delta_norm, "context_l2": context_norm,
                "delta_relative_l2": delta_norm / max(context_norm, 1e-12),
                "injection_layer": 27, "injection_token": int(batch["input_ids"].shape[1] - 1)}
    if controller is not None:
        diagnostic = state["controller_diagnostics"]
        metadata["maximum_head_relative_delta"] = float((diagnostic["delta_norm"] / diagnostic["context_norm"].clamp_min(1e-12)).max())
        metadata["maximum_absolute_gain"] = float(diagnostic["gains"].abs().max())
        metadata["capped_head_count"] = int(diagnostic["cap_active"].sum())
        metadata["head_gains"] = cpu_float(diagnostic["gains"][0]).tolist()
    arrays = {}
    if retain_features:
        arrays = {name: cpu_float(state[name]) for name in ("decoder_input", "attention_delta", "context", "query", "reference_context", "attention_weights", "visual_attention_mass")}
        arrays.update({"basis_" + name: cpu_float(tensor) for name, tensor in state["bases"].items()})
        arrays.update({"input_ids": batch["input_ids"][0].cpu().numpy(), "visual_positions": np.asarray(batch.video_positions)})
    return logits, arrays, metadata
