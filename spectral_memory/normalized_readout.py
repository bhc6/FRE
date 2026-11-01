"""Direction-normalized bounded readout controls for the V2 scale hypothesis."""

from __future__ import annotations

import math

import torch
from torch import nn


METHODS = ("unit_time_mlp", "unit_spectral_mlp", "unit_time_constant")


class NormalizedReadoutController(nn.Module):
    """Predict bounded per-head residuals after removing basis amplitude."""

    def __init__(self, method: str, head_dim: int = 128, time_bins: int = 8,
                 hidden_dim: int = 16, gain_limit: float = 0.25,
                 residual_limit: float = 0.1, epsilon: float = 1e-6):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown normalized controller method: {method}")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (head_dim, time_bins, hidden_dim)):
            raise ValueError("Controller dimensions must be positive integers")
        if any(not math.isfinite(value) or value <= 0 for value in (gain_limit, residual_limit, epsilon)):
            raise ValueError("Gain/residual limits and relative epsilon must be finite and positive")
        self.method = method
        self.head_dim = head_dim
        self.time_bins = time_bins
        self.hidden_dim = hidden_dim
        self.gain_limit = gain_limit
        self.residual_limit = residual_limit
        self.epsilon = epsilon
        if method == "unit_time_constant":
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
        """Return a residual [B,H,D] and the V1 diagnostic keys plus direction scales.

        Query is the actual post-head-norm, post-RoPE query. Each basis
        direction is divided by max(its L2 norm, epsilon * context L2 norm).
        The conditioner multiplies directions by sqrt(head_dim), while the
        intervention multiplies them by context L2 norm / sqrt(time_bins).
        Feature_scale reports the relative norm floor, not the V1 RMS scale.
        """
        if query.ndim != 3 or context.shape != query.shape or query.shape[-1] != self.head_dim:
            raise ValueError("Expected matching query/context [B,H,head_dim]")
        if min(query.shape) < 1:
            raise ValueError("Query/context dimensions must be nonempty")
        basis_name = "spectral" if self.method == "unit_spectral_mlp" else "time"
        expected_shape = (*query.shape[:2], self.time_bins, self.head_dim)
        if basis_name not in bases or tuple(bases[basis_name].shape) != expected_shape:
            raise ValueError("Required basis must have shape [B,H,time_bins,head_dim]")
        parameter = next(self.parameters())
        if query.device != parameter.device or context.device != parameter.device or bases[basis_name].device != parameter.device:
            raise ValueError("Controller and all inputs must be on the same device")
        if not query.is_floating_point() or not context.is_floating_point() or not bases[basis_name].is_floating_point():
            raise ValueError("Controller inputs must be real floating tensors")
        if not bool(torch.isfinite(query).all()) or not bool(torch.isfinite(context).all()) or not bool(torch.isfinite(bases[basis_name]).all()):
            raise ValueError("Controller inputs must be finite")
        query = query.to(dtype=parameter.dtype)
        native_context = context.to(dtype=parameter.dtype)
        basis = bases[basis_name].to(dtype=parameter.dtype)
        context_norm = torch.linalg.vector_norm(native_context, dim=-1)
        basis_norm = torch.linalg.vector_norm(basis, dim=-1)
        relative_floor = self.epsilon * context_norm
        direction_denominator = torch.maximum(basis_norm, relative_floor.unsqueeze(-1))
        safe_direction_denominator = torch.where(direction_denominator > 0, direction_denominator,
                                                 torch.ones_like(direction_denominator))
        directions = basis / safe_direction_denominator.unsqueeze(-1)
        if self.network is None:
            raw_gains = self.constant_gains.view(1, 1, self.time_bins).expand(*query.shape[:2], self.time_bins)
        else:
            conditioner = directions * math.sqrt(self.head_dim)
            features = torch.cat((query, conditioner.flatten(start_dim=2)), dim=-1)
            raw_gains = self.network(features)
        gains = self.gain_limit * torch.tanh(raw_gains)
        intervention_basis = directions * (context_norm / math.sqrt(self.time_bins))[..., None, None]
        raw_delta = torch.sum(gains.unsqueeze(-1) * intervention_basis, dim=2)
        raw_delta_norm = torch.linalg.vector_norm(raw_delta, dim=-1)
        radius = self.residual_limit * context_norm
        cap_denominator = torch.maximum(raw_delta_norm, relative_floor)
        safe_cap_denominator = torch.where(cap_denominator > 0, cap_denominator, torch.ones_like(cap_denominator))
        cap_scale = (radius / safe_cap_denominator).clamp(max=1.0)
        delta = raw_delta * cap_scale.unsqueeze(-1)
        diagnostics = {"gains": gains, "raw_delta_norm": raw_delta_norm,
                       "delta_norm": torch.linalg.vector_norm(delta, dim=-1),
                       "context_norm": context_norm, "residual_radius": radius,
                       "cap_scale": cap_scale, "cap_active": raw_delta_norm > radius,
                       "feature_scale": relative_floor, "basis_norm": basis_norm,
                       "direction_denominator": direction_denominator,
                       "direction_norm": torch.linalg.vector_norm(directions, dim=-1)}
        return delta, diagnostics
