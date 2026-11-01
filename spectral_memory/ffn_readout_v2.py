"""Generic final-token FFN adaptation as a capacity control."""

from __future__ import annotations

import math

from .controller_runtime import FrozenTail
import torch
from torch import nn


METHODS = ("ffn_lowrank",)


class FFNReadoutController(nn.Module):
    def __init__(self, method="ffn_lowrank", hidden_dim=2048, rank=4, residual_limit=.1):
        super().__init__()
        if method not in METHODS or hidden_dim != 2048 or rank != 4 or residual_limit not in (.1, .3):
            raise ValueError("Use the fixed rank-4 FFN capacity control")
        self.method = method
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.residual_limit = residual_limit
        self.gain_limit = 1.
        self.network = nn.Sequential(nn.Linear(hidden_dim, rank), nn.SiLU(), nn.Linear(rank, hidden_dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, query, bases, context):
        if query.ndim != 2 or query.shape != context.shape or query.shape[-1] != self.hidden_dim:
            raise ValueError("Expected last-position FFN input/output with shape [B,2048]")
        gains = self.network(query.float()).tanh()
        context_norm = torch.linalg.vector_norm(context.float(), dim=-1)
        raw_delta = gains * context_norm[:, None] / math.sqrt(self.hidden_dim)
        raw_norm = torch.linalg.vector_norm(raw_delta, dim=-1)
        radius = self.residual_limit * context_norm
        scale = (radius / raw_norm.clamp_min(1e-12)).clamp(max=1.)
        delta = raw_delta * scale[:, None]
        return delta, {"gains": gains, "delta_norm": torch.linalg.vector_norm(delta, dim=-1),
                       "context_norm": context_norm, "cap_active": raw_norm > radius,
                       "raw_delta_norm": raw_norm, "cap_scale": scale}


class FFNFrozenTail(FrozenTail):
    def forward(self, decoder_input, attention_delta, delta_ffn):
        residual = decoder_input.to(torch.bfloat16) + attention_delta.to(torch.bfloat16)
        original_ffn = self.mlp(self.post_norm(residual))
        hidden = residual + (original_ffn + delta_ffn.to(torch.bfloat16))
        return torch.nn.functional.linear(self.final_norm(hidden), self.answer_weight).float()


def controlled_forward(backend, batch, controller=None, retain_features=False):
    import numpy as np
    from .passive_probe import forward_logits

    if retain_features:
        raise ValueError("FFN evaluation does not export attention readout features")
    decoder = backend.model.model.language_model.layers[-1]
    state = {}

    def replace_ffn(module, arguments, output):
        if state:
            raise RuntimeError("Expected exactly one uncached FFN visit")
        original = output[:, -1].float()
        if controller is None:
            delta = torch.zeros_like(original)
            diagnostics = {"gains": delta, "delta_norm": delta.norm(dim=-1),
                           "context_norm": original.norm(dim=-1),
                           "cap_active": torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)}
        else:
            delta, diagnostics = controller(arguments[0][:, -1].float(), {}, original)
        state.update({"delta": delta, "diagnostics": diagnostics})
        replaced = output.clone()
        replaced[:, -1] = output[:, -1] + delta.to(output.dtype)
        return replaced

    handle = decoder.mlp.register_forward_hook(replace_ffn)
    try:
        logits = forward_logits(backend, batch)
    finally:
        handle.remove()
    if not state or not np.isfinite(logits).all():
        raise RuntimeError("Incomplete or nonfinite FFN control")
    diagnostics = state["diagnostics"]
    metadata = {"intervention_space": "final_layer_last_prefill_token_FFN_output",
                "injection_layer": 27, "injection_token": int(batch["input_ids"].shape[-1] - 1),
                "delta_ffn_l2": float(diagnostics["delta_norm"].max()),
                "ffn_l2": float(diagnostics["context_norm"].max()),
                "maximum_relative_ffn_delta": float((diagnostics["delta_norm"] / diagnostics["context_norm"].clamp_min(1e-12)).max()),
                "maximum_absolute_gain": float(diagnostics["gains"].abs().max()),
                "capped_token_count": int(diagnostics["cap_active"].sum())}
    return logits, {}, metadata
