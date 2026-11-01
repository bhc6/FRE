"""Small, bounded temporal residual adapters for feature-level development.

The spectral and time-domain modules intentionally share the same learned
projection parameterization.  They differ only in whether the bottleneck is
applied before or after an orthonormal temporal Fourier transform.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


METHODS = ("spectral", "time")


def _flatten_spatial(values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.ndim == 4:
        batch, time_bins, patches, feature_dim = values.shape
        return values, (batch, time_bins, patches, feature_dim)
    if values.ndim == 5:
        batch, time_bins, height, width, feature_dim = values.shape
        return values.reshape(batch, time_bins, height * width, feature_dim), tuple(values.shape)
    raise ValueError("Expected [B,T,P,D] or [B,T,H,W,D] features")


def _restore_spatial(values: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if len(shape) == 4:
        return values.reshape(shape)
    batch, time_bins, height, width, feature_dim = shape
    return values.reshape(batch, time_bins, height, width, feature_dim)


class _TemporalResidualBase(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        time_bins: int,
        rank: int = 4,
        max_gate: float = 0.25,
        residual_limit: float = 0.10,
    ) -> None:
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("time_bins", time_bins), ("rank", rank)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("max_gate", max_gate), ("residual_limit", residual_limit)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.feature_dim = feature_dim
        self.time_bins = time_bins
        self.rank = rank
        self.max_gate = float(max_gate)
        self.residual_limit = float(residual_limit)
        self.in_proj = nn.Linear(feature_dim, rank, bias=False)
        self.out_proj = nn.Linear(rank, feature_dim, bias=False)
        # Both variants expose exactly T gate parameters.  The spectral
        # variant uses only the first floor(T/2)+1 bins, preserving capacity
        # parity with the time-domain control.
        self.gate = nn.Parameter(torch.zeros(time_bins))
        # Keep the initial residual exactly zero through out_proj while
        # leaving a gradient path to out_proj on the first optimizer step.
        nn.init.constant_(self.gate, 0.1)
        nn.init.zeros_(self.out_proj.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_dim": self.feature_dim,
            "time_bins": self.time_bins,
            "rank": self.rank,
            "max_gate": self.max_gate,
            "residual_limit": self.residual_limit,
            "parameter_count": self.parameter_count,
            "zero_initialized_output": True,
            "initial_gate_raw": 0.1,
            "input_layouts": ["[B,T,P,D]", "[B,T,H,W,D]"],
            "temporal_axis": 1,
            "channel_fft": False,
        }

    def _bounded_residual(self, original: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        original_float = original.float()
        delta_float = delta.float()
        original_norm = torch.linalg.vector_norm(original_float, dim=-1)
        delta_norm = torch.linalg.vector_norm(delta_float, dim=-1)
        radius = self.residual_limit * original_norm
        scale = torch.minimum(torch.ones_like(delta_norm), radius / delta_norm.clamp_min(1e-12))
        bounded = delta_float * scale.unsqueeze(-1)
        return bounded.to(dtype=original.dtype), {
            "delta_norm": torch.linalg.vector_norm(bounded, dim=-1),
            "input_norm": original_norm,
            "radius": radius,
            "cap_scale": scale,
            "cap_active": delta_norm > radius,
        }

    def _validate(self, values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise ValueError("features must be a real floating-point tensor")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features must be finite")
        flat, shape = _flatten_spatial(values)
        if flat.shape[1] != self.time_bins or flat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected time_bins={self.time_bins}, feature_dim={self.feature_dim}; got {tuple(values.shape)}"
            )
        return flat, shape


class TemporalSpectralResidualAdapter(_TemporalResidualBase):
    """Bounded residual adapter whose bottleneck is modulated in temporal FFT space."""

    method = "spectral"

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flat, shape = self._validate(values)
        original_dtype = flat.dtype
        work = flat.to(dtype=self.in_proj.weight.dtype)
        spectrum = torch.fft.rfft(work, dim=1, norm="ortho")
        in_weight = self.in_proj.weight.to(dtype=spectrum.dtype)
        out_weight = self.out_proj.weight.to(dtype=spectrum.dtype)
        bottleneck = torch.einsum("btpd,rd->btpr", spectrum, in_weight)
        frequency_gate = self.max_gate * torch.tanh(self.gate[: spectrum.shape[1]])
        bottleneck = bottleneck * frequency_gate.view(1, -1, 1, 1)
        transformed = torch.einsum("btpr,dr->btpd", bottleneck, out_weight)
        delta = torch.fft.irfft(transformed, n=self.time_bins, dim=1, norm="ortho")
        bounded, _ = self._bounded_residual(work, delta)
        return _restore_spatial((work + bounded).to(dtype=original_dtype), shape)

    def forward_with_diagnostics(self, values: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat, shape = self._validate(values)
        original_dtype = flat.dtype
        work = flat.to(dtype=self.in_proj.weight.dtype)
        spectrum = torch.fft.rfft(work, dim=1, norm="ortho")
        in_weight = self.in_proj.weight.to(dtype=spectrum.dtype)
        out_weight = self.out_proj.weight.to(dtype=spectrum.dtype)
        bottleneck = torch.einsum("btpd,rd->btpr", spectrum, in_weight)
        gate = self.max_gate * torch.tanh(self.gate[: spectrum.shape[1]])
        bottleneck = bottleneck * gate.view(1, -1, 1, 1)
        transformed = torch.einsum("btpr,dr->btpd", bottleneck, out_weight)
        delta = torch.fft.irfft(transformed, n=self.time_bins, dim=1, norm="ortho")
        bounded, diagnostics = self._bounded_residual(work, delta)
        diagnostics.update({"frequency_gate": gate.detach(),
                            "power_input": spectrum.abs().square().sum(dim=-1).detach()})
        return _restore_spatial((work + bounded).to(dtype=original_dtype), shape), diagnostics


class TemporalResidualAdapter(_TemporalResidualBase):
    """Capacity-matched time-domain residual control."""

    method = "time"

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        flat, shape = self._validate(values)
        original_dtype = flat.dtype
        work = flat.to(dtype=self.in_proj.weight.dtype)
        bottleneck = self.in_proj(work)
        time_gate = self.max_gate * torch.tanh(self.gate).view(1, self.time_bins, 1, 1)
        delta = self.out_proj(bottleneck * time_gate)
        bounded, _ = self._bounded_residual(work, delta)
        return _restore_spatial((work + bounded).to(dtype=original_dtype), shape)

    def forward_with_diagnostics(self, values: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat, shape = self._validate(values)
        original_dtype = flat.dtype
        work = flat.to(dtype=self.in_proj.weight.dtype)
        bottleneck = self.in_proj(work)
        gate = self.max_gate * torch.tanh(self.gate)
        delta = self.out_proj(bottleneck * gate.view(1, self.time_bins, 1, 1))
        bounded, diagnostics = self._bounded_residual(work, delta)
        diagnostics["time_gate"] = gate.detach()
        return _restore_spatial((work + bounded).to(dtype=original_dtype), shape), diagnostics


def build_adapter(method: str, **kwargs: Any) -> _TemporalResidualBase:
    if method == "spectral":
        return TemporalSpectralResidualAdapter(**kwargs)
    if method == "time":
        return TemporalResidualAdapter(**kwargs)
    raise ValueError(f"Unknown adapter method: {method}")

