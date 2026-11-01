"""CPU checks for a fixed attention head's temporal Fourier readout identity."""

from __future__ import annotations

import os

for thread_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


SEED = 20261101
TOLERANCE = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - np.max(logits))
    return exponentials / exponentials.sum()


def one_sided_weights(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("Time length must be positive")
    weights = np.full(count // 2 + 1, 2.0)
    weights[0] = 1.0
    if count % 2 == 0:
        weights[-1] = 1.0
    return weights


def spectral_contributions(attention: np.ndarray, values: np.ndarray) -> np.ndarray:
    if attention.ndim != 2 or values.ndim != 3 or attention.shape != values.shape[:2]:
        raise ValueError("Expected attention [time, position] and values [time, position, channel]")
    if np.iscomplexobj(attention) or np.iscomplexobj(values):
        raise ValueError("This one-sided identity expects real inputs")
    if not np.isfinite(attention).all() or not np.isfinite(values).all():
        raise ValueError("Inputs must be finite")
    attention_spectrum = np.fft.rfft(attention, axis=0, norm="ortho")
    value_spectrum = np.fft.rfft(values, axis=0, norm="ortho")
    cross_spectrum = np.einsum("kp,kpd->kd", attention_spectrum.conj(), value_spectrum)
    return one_sided_weights(len(attention))[:, None] * cross_spectrum.real


def maximum_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def run_checks(output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise ValueError("Output directory must not already exist")
    random = np.random.default_rng(SEED)
    checks = []
    examples = {}

    def check(name: str, condition: bool, **metrics) -> None:
        checks.append({"name": name, "passed": bool(condition), "metrics": metrics})
        if not condition:
            raise AssertionError(f"{name}: {metrics}")

    for count in (1, 2, 3, 7, 8, 31, 32):
        positions, channels, text_tokens = 3, 5, 4
        all_logits = random.normal(size=count * positions + text_tokens)
        all_attention = softmax(all_logits)
        attention = all_attention[:count * positions].reshape(count, positions)
        text_attention = all_attention[count * positions:]
        values = random.normal(size=(count, positions, channels))
        text_values = random.normal(size=(text_tokens, channels))
        visual_readout = np.einsum("tp,tpd->d", attention, values)
        text_readout = text_attention @ text_values
        all_values = np.concatenate((values.reshape(-1, channels), text_values))
        total_readout = all_attention @ all_values
        attention_spectrum = np.fft.fft(attention, axis=0, norm="ortho")
        value_spectrum = np.fft.fft(values, axis=0, norm="ortho")
        full_readout = np.einsum("kp,kpd->d", attention_spectrum.conj(), value_spectrum)
        single_sided_readout = spectral_contributions(attention, values).sum(axis=0)
        backward_readout = np.einsum(
            "kp,kpd->d", np.fft.fft(attention, axis=0).conj(), np.fft.fft(values, axis=0)
        ) / count
        check(f"time_full_and_weighted_rfft_readouts_T{count}",
              maximum_error(visual_readout, full_readout) < TOLERANCE
              and maximum_error(visual_readout, single_sided_readout) < TOLERANCE
              and maximum_error(visual_readout, backward_readout) < TOLERANCE,
              full_fft_error=maximum_error(visual_readout, full_readout),
              rfft_error=maximum_error(visual_readout, single_sided_readout),
              backward_fft_error=maximum_error(visual_readout, backward_readout))
        visual_mass = float(attention.sum())
        conditional_attention = softmax(all_logits[:count * positions]).reshape(count, positions)
        conditional_readout = np.einsum("tp,tpd->d", conditional_attention, values)
        check(f"visual_mass_and_text_partition_T{count}",
              0 < visual_mass < 1
              and maximum_error(total_readout, single_sided_readout + text_readout) < TOLERANCE
              and maximum_error(visual_readout, visual_mass * conditional_readout) < TOLERANCE,
              visual_mass=visual_mass,
              total_reconstruction_error=maximum_error(total_readout, single_sided_readout + text_readout),
              conditional_readout_scale_error=maximum_error(visual_readout, visual_mass * conditional_readout))

        visual_gate = np.exp(random.normal(scale=0.5, size=count * positions))
        all_gate = np.concatenate((visual_gate, np.ones(text_tokens)))
        gated_attention = all_attention * all_gate
        gated_attention /= gated_attention.sum()
        gated_by_logits = softmax(all_logits + np.log(all_gate))
        gated_visual = gated_attention[:count * positions].reshape(count, positions)
        gated_text = gated_attention[count * positions:]
        gated_total = gated_attention @ all_values
        gated_reconstructed = spectral_contributions(gated_visual, values).sum(axis=0) + gated_text @ text_values
        check(f"fixed_qkv_gate_includes_text_normalizer_T{count}",
              maximum_error(gated_attention, gated_by_logits) < TOLERANCE
              and maximum_error(gated_total, gated_reconstructed) < TOLERANCE
              and abs(gated_attention.sum() - 1) < TOLERANCE,
              attention_error=maximum_error(gated_attention, gated_by_logits),
              readout_error=maximum_error(gated_total, gated_reconstructed),
              before_visual_mass=visual_mass, after_visual_mass=float(gated_visual.sum()))

        zero_gate = all_gate.copy()
        zero_gate[0] = 0.0
        zero_gated_attention = all_attention * zero_gate
        zero_gated_attention /= zero_gated_attention.sum()
        masked_logits = all_logits + np.log(all_gate)
        masked_logits[0] = -np.inf
        check(f"zero_gate_matches_masked_softmax_T{count}",
              zero_gated_attention[0] == 0
              and maximum_error(zero_gated_attention, softmax(masked_logits)) < TOLERANCE)

    count = 8
    values = np.zeros((count, 1, 1))
    values[0, 0, 0] = 1.0
    values[-1, 0, 0] = -1.0
    first_attention = np.zeros((count, 1))
    last_attention = np.zeros((count, 1))
    first_attention[0, 0] = 1.0
    last_attention[-1, 0] = 1.0
    for label, uniform_mass in (("onehot", 0.0), ("finite_softmax", 0.1)):
        first = (1 - uniform_mass) * first_attention + uniform_mass / count
        last = (1 - uniform_mass) * last_attention + uniform_mass / count
        first_power = abs(np.fft.rfft(first, axis=0, norm="ortho")) ** 2
        last_power = abs(np.fft.rfft(last, axis=0, norm="ortho")) ** 2
        first_contributions = spectral_contributions(first, values)
        last_contributions = spectral_contributions(last, values)
        first_readout = float(np.einsum("tp,tpd->d", first, values)[0])
        last_readout = float(np.einsum("tp,tpd->d", last, values)[0])
        check(f"{label}_same_attention_and_value_powers_opposite_readouts",
              maximum_error(first_power, last_power) < TOLERANCE
              and abs(first_readout - (1 - uniform_mass)) < TOLERANCE
              and abs(last_readout + (1 - uniform_mass)) < TOLERANCE
              and abs(first_contributions.sum() - first_readout) < TOLERANCE
              and abs(last_contributions.sum() - last_readout) < TOLERANCE,
              attention_power_error=maximum_error(first_power, last_power),
              value_power_error=0.0, first_readout=first_readout, last_readout=last_readout,
              minimum_first_band_contribution=float(first_contributions.min()),
              minimum_last_band_contribution=float(last_contributions.min()))
        if uniform_mass > 0:
            check("counterexample_uses_finite_softmax_logits",
                  np.isfinite(np.log(first)).all() and np.isfinite(np.log(last)).all()
                  and maximum_error(first, softmax(np.log(first))) < TOLERANCE
                  and maximum_error(last, softmax(np.log(last))) < TOLERANCE)
        examples[f"{label}_first_attention"] = first
        examples[f"{label}_last_attention"] = last
        examples[f"{label}_first_band_contributions"] = first_contributions
        examples[f"{label}_last_band_contributions"] = last_contributions
    examples["shared_values"] = values

    check("one_sided_endpoint_weights_even_and_odd",
          np.array_equal(one_sided_weights(1), [1])
          and np.array_equal(one_sided_weights(2), [1, 1])
          and np.array_equal(one_sided_weights(7), [1, 2, 2, 2])
          and np.array_equal(one_sided_weights(8), [1, 2, 2, 2, 1]))
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Synthetic NumPy identities and counterexamples, not a new theorem or model-performance evidence",
        "seed": SEED, "dtype": "float64", "tolerance": TOLERANCE,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_imported": "torch" in sys.modules, "gpu_used": False,
        "check_count": len(checks), "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "examples.npz", **examples)
    (output / "checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_spectral_checks_v1"))
    arguments = parser.parse_args()
    report = run_checks(arguments.output_dir)
    print(json.dumps({"checks": report["check_count"], "all_passed": report["all_passed"],
                      "torch_imported": report["torch_imported"],
                      "output": str(arguments.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
