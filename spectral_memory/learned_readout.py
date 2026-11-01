"""Small query-conditioned residuals for a frozen final attention readout."""

from __future__ import annotations

import math

import torch
from torch import nn


METHODS = ("time_mlp", "spectral_mlp", "spectral_phaseblind", "constant_spectral")


def build_bases(attention_visual: torch.Tensor, values_visual: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build [head, time, channel] bases without renormalizing visual attention.

    Attention has shape [head, time, patch] and contains the visual slice of
    full-key softmax probabilities. Values have shape [head, time, patch,
    channel]; callers must already expand grouped-query value heads.
    The phase_ablation output is only a conditioner input. It is never the
    intervention basis used by spectral_phaseblind.
    """
    if attention_visual.ndim != 3 or values_visual.ndim != 4:
        raise ValueError("Expected attention [H,T,P] and values [H,T,P,D]")
    if attention_visual.shape != values_visual.shape[:3] or min(values_visual.shape) < 1:
        raise ValueError("Attention/value axes must match and all dimensions must be nonempty")
    if attention_visual.device != values_visual.device:
        raise ValueError("Attention and values must be on the same device")
    if not attention_visual.is_floating_point() or not values_visual.is_floating_point():
        raise ValueError("Attention and values must be real floating tensors")
    if not bool(torch.isfinite(attention_visual).all()) or not bool(torch.isfinite(values_visual).all()):
        raise ValueError("Attention and values must be finite")
    if bool(torch.any(attention_visual < 0)):
        raise ValueError("Visual attention probabilities cannot be negative")
    dtype = torch.float64 if torch.float64 in (attention_visual.dtype, values_visual.dtype) else torch.float32
    attention = attention_visual.to(dtype=dtype)
    values = values_visual.to(dtype=dtype)
    time_bins = attention.shape[1]
    time_basis = (attention.unsqueeze(-1) * values).sum(dim=2)
    attention_spectrum = torch.fft.rfft(attention, dim=1, norm="ortho")
    value_spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
    cross = (attention_spectrum.conj().unsqueeze(-1) * value_spectrum).sum(dim=2)
    spectral_parts = [cross[:, 0].real]
    phaseblind_parts = [cross[:, 0].abs()]
    for frequency in range(1, (time_bins + 1) // 2):
        spectral_parts.extend((2 * cross[:, frequency].real, -2 * cross[:, frequency].imag))
        phaseblind_parts.extend((2 * cross[:, frequency].abs(), torch.zeros_like(cross[:, frequency].real)))
    if time_bins % 2 == 0:
        spectral_parts.append(cross[:, -1].real)
        phaseblind_parts.append(cross[:, -1].abs())
    return {"time": time_basis, "spectral": torch.stack(spectral_parts, dim=1),
            "phase_ablation": torch.stack(phaseblind_parts, dim=1)}


class ReadoutController(nn.Module):
    """Return only a bounded delta_context; no labels or backbone state edits."""

    def __init__(self, method: str, head_dim: int = 128, time_bins: int = 8,
                 hidden_dim: int = 16, gain_limit: float = 0.25,
                 residual_limit: float = 0.1, epsilon: float = 1e-8):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown controller method: {method}")
        if any(not isinstance(value, int) or value < 1 for value in (head_dim, time_bins, hidden_dim)):
            raise ValueError("Controller dimensions must be positive integers")
        if any(not math.isfinite(value) or value <= 0 for value in (gain_limit, residual_limit, epsilon)):
            raise ValueError("Gain/residual limits and epsilon must be finite and positive")
        self.method = method
        self.head_dim = head_dim
        self.time_bins = time_bins
        self.hidden_dim = hidden_dim
        self.gain_limit = gain_limit
        self.residual_limit = residual_limit
        self.epsilon = epsilon
        if method == "constant_spectral":
            self.constant_gains = nn.Parameter(torch.zeros(time_bins))
            self.network = None
        else:
            self.register_parameter("constant_gains", None)
            self.network = nn.Sequential(
                nn.Linear(head_dim * (time_bins + 1), hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, time_bins),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, query: torch.Tensor, bases: dict[str, torch.Tensor],
                context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Map query/context [B,H,D] and bases [B,H,T,D] to delta [B,H,D].

        Query is the cached per-head normalized query. Only the conditioner
        basis is divided by the native context RMS; the actual intervention
        basis retains its original units. Each head has an independent
        residual-radius limit. A zero native context therefore permits no
        residual in that head.
        """
        if query.ndim != 3 or context.shape != query.shape or query.shape[-1] != self.head_dim:
            raise ValueError("Expected matching query/context [B,H,head_dim]")
        basis_name = "time" if self.method == "time_mlp" else "spectral"
        conditioner_name = "phase_ablation" if self.method == "spectral_phaseblind" else basis_name
        required = {basis_name, conditioner_name}
        expected_shape = (*query.shape[:2], self.time_bins, self.head_dim)
        if any(name not in bases or tuple(bases[name].shape) != expected_shape for name in required):
            raise ValueError("Required bases must have shape [B,H,time_bins,head_dim]")
        parameter = next(self.parameters())
        if query.device != parameter.device or context.device != parameter.device or any(bases[name].device != parameter.device for name in required):
            raise ValueError("Controller and all inputs must be on the same device")
        if not query.is_floating_point() or not context.is_floating_point() or any(not bases[name].is_floating_point() for name in required):
            raise ValueError("Controller inputs must be real floating tensors")
        query = query.to(dtype=parameter.dtype)
        native_context = context.to(dtype=parameter.dtype)
        intervention_basis = bases[basis_name].to(dtype=parameter.dtype)
        context_norm = torch.linalg.vector_norm(native_context, dim=-1)
        feature_scale = (context_norm / math.sqrt(self.head_dim)).clamp_min(self.epsilon)
        if self.network is None:
            raw_gains = self.constant_gains.view(1, 1, self.time_bins).expand(*query.shape[:2], self.time_bins)
        else:
            conditioner = bases[conditioner_name].to(dtype=parameter.dtype) / feature_scale[..., None, None]
            features = torch.cat((query, conditioner.flatten(start_dim=2)), dim=-1)
            raw_gains = self.network(features)
        gains = self.gain_limit * torch.tanh(raw_gains)
        raw_delta = torch.sum(gains.unsqueeze(-1) * intervention_basis, dim=2)
        raw_delta_norm = torch.linalg.vector_norm(raw_delta, dim=-1)
        radius = self.residual_limit * context_norm
        cap_scale = (radius / raw_delta_norm.clamp_min(self.epsilon)).clamp(max=1.0)
        delta = raw_delta * cap_scale.unsqueeze(-1)
        diagnostics = {"gains": gains, "raw_delta_norm": raw_delta_norm,
                       "delta_norm": torch.linalg.vector_norm(delta, dim=-1),
                       "context_norm": context_norm, "residual_radius": radius,
                       "cap_scale": cap_scale, "cap_active": raw_delta_norm > radius,
                       "feature_scale": feature_scale}
        return delta, diagnostics
