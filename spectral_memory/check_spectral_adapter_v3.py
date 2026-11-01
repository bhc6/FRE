"""Deterministic CPU checks for the v3 temporal adapters."""

from __future__ import annotations

import torch

from .spectral_adapter_v3 import build_adapter


def main() -> None:
    torch.manual_seed(20261102)
    settings = {"feature_dim": 12, "time_bins": 8, "rank": 3, "max_gate": 0.25, "residual_limit": 0.10}
    spectral = build_adapter("spectral", **settings)
    time = build_adapter("time", **settings)
    assert spectral.parameter_count == time.parameter_count
    values = torch.randn(2, 8, 3, 4, 12, dtype=torch.float32)
    spectral_output = spectral(values)
    time_output = time(values)
    assert spectral_output.shape == values.shape
    assert time_output.shape == values.shape
    assert torch.equal(spectral_output, values)
    assert torch.equal(time_output, values)
    with torch.no_grad():
        spectral.out_proj.weight.normal_(std=0.02)
        time.out_proj.weight.normal_(std=0.02)
    transformed, diagnostics = spectral.forward_with_diagnostics(values)
    assert transformed.shape == values.shape
    assert torch.isfinite(transformed).all()
    delta = (transformed.float() - values.float()).norm(dim=-1)
    radius = settings["residual_limit"] * values.float().norm(dim=-1)
    assert torch.all(delta <= radius + 1e-5)
    assert diagnostics["power_input"].shape == (2, 8 // 2 + 1, 3 * 4)
    flat_values = values.reshape(2, 8, 12, 12)
    flat_output = spectral(flat_values)
    assert flat_output.shape == flat_values.shape
    print("spectral adapter v3 checks passed: 7")


if __name__ == "__main__":
    main()
