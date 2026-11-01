"""Feature-level temporal and spatiotemporal spectral counterfactuals.

This module is deliberately model-free. It operates on logical visual features
with shape ``[T, H, W, D]`` captured by ``vision_encoder_probe_v2``. It does
not claim that reversing a feature tensor is equivalent to reversing pixels.
The functions are intended for controlled development experiments before any
learned intervention is trained.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


POWER_TOLERANCE = 1e-5
REAL_TOLERANCE = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features)
    require(values.ndim == 4, "Expected logical visual features with shape [T,H,W,D]")
    require(all(int(size) > 0 for size in values.shape), "Feature dimensions must be positive")
    require(not np.iscomplexobj(values), "Feature counterfactuals require real-valued features")
    require(np.isfinite(values).all(), "Feature tensor contains non-finite values")
    return np.ascontiguousarray(values)


def feature_hash(features: np.ndarray) -> str:
    values = validate_features(features)
    return hashlib.sha256(values.tobytes()).hexdigest()


def identity(features: np.ndarray) -> np.ndarray:
    """Return an independent copy for the identity round-trip condition."""
    return validate_features(features).copy()


def temporal_reverse(features: np.ndarray) -> np.ndarray:
    """Reverse the sampled feature-time axis without re-encoding pixels."""
    return validate_features(features)[::-1].copy()


def temporal_power(features: np.ndarray) -> np.ndarray:
    """Return full temporal power ``|FFT_T|^2`` summed over channels only.

    The spatial grid is retained, so this is not a spatially pooled spectrum.
    """
    values = validate_features(features).astype(np.float64, copy=False)
    spectrum = np.fft.fft(values, axis=0, norm="ortho")
    return np.square(np.abs(spectrum)).sum(axis=-1)


def joint_3d_power(features: np.ndarray) -> np.ndarray:
    """Return ``|FFT_(T,H,W)|^2`` independently for every feature channel."""
    values = validate_features(features).astype(np.float64, copy=False)
    spectrum = np.fft.fftn(values, axes=(0, 1, 2), norm="ortho")
    return np.square(np.abs(spectrum))


def _random_hermitian_phase(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Create a conjugate-symmetric unit-magnitude phase mask.

    Taking the phase of the FFT of real Gaussian noise gives the required
    conjugate symmetry, so multiplying a real tensor spectrum by this mask and
    applying the inverse FFT yields a real tensor up to floating-point noise.
    """
    noise = rng.normal(size=shape)
    noise_spectrum = np.fft.fftn(noise, axes=(0, 1, 2), norm="ortho")
    magnitude = np.abs(noise_spectrum)
    phase = np.ones(shape, dtype=np.complex128)
    nonzero = magnitude > 1e-15
    phase[nonzero] = noise_spectrum[nonzero] / magnitude[nonzero]
    return phase


def three_d_allpass_phase(features: np.ndarray, seed: int = 20261102) -> np.ndarray:
    """Change joint spatiotemporal phase while preserving joint power.

    The phase mask is sampled from a real Gaussian field and therefore has
    Hermitian symmetry. Zeros in the source spectrum retain phase one. The
    output is rounded to real values only after checking the inverse FFT's
    imaginary residual.
    """
    values = validate_features(features).astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    transformed = np.fft.fftn(values, axes=(0, 1, 2), norm="ortho")
    phase = _random_hermitian_phase(values.shape[:3], rng)[..., None]
    counterfactual = np.fft.ifftn(transformed * phase, axes=(0, 1, 2), norm="ortho")
    imaginary_error = float(np.max(np.abs(counterfactual.imag)))
    require(imaginary_error <= REAL_TOLERANCE, "All-pass phase inverse is not real within tolerance")
    result = np.asarray(counterfactual.real, dtype=np.float64)
    require(np.isfinite(result).all(), "All-pass phase result is non-finite")
    return result.astype(np.asarray(features).dtype, copy=False)


def spatial_mean_control(features: np.ndarray) -> dict[str, np.ndarray | float | list[int]]:
    """Compare full-patch temporal power with spatial-mean temporal power."""
    values = validate_features(features).astype(np.float64, copy=False)
    full_power = temporal_power(values)
    pooled = values.mean(axis=(1, 2))
    pooled_fft = np.fft.fft(pooled, axis=0, norm="ortho")
    pooled_power = np.square(np.abs(pooled_fft)).sum(axis=-1)
    full_non_dc = float(full_power[1:].sum())
    pooled_non_dc = float(pooled_power[1:].sum())
    ratio = float(values.shape[1] * values.shape[2] * pooled_non_dc / full_non_dc) if full_non_dc > 0 else None
    return {
        "full_temporal_power_fhw": full_power,
        "pooled_temporal_power_f": pooled_power,
        "full_non_dc_energy": full_non_dc,
        "pooled_non_dc_energy": pooled_non_dc,
        "spatial_constant_energy_ratio": ratio,
        "shape": list(values.shape),
    }


def counterfactual_bundle(features: np.ndarray, seed: int = 20261102) -> dict[str, object]:
    """Build all fixed conditions and preservation diagnostics for one tensor."""
    values = validate_features(features)
    reversed_values = temporal_reverse(values)
    phase_values = three_d_allpass_phase(values, seed=seed)
    original_temporal = temporal_power(values)
    original_joint = joint_3d_power(values)
    reversed_temporal = temporal_power(reversed_values)
    phase_joint = joint_3d_power(phase_values)
    temporal_error = float(np.max(np.abs(original_temporal - reversed_temporal)) /
                          max(float(np.max(original_temporal)), 1e-20))
    phase_error = float(np.max(np.abs(original_joint - phase_joint)) /
                        max(float(np.max(original_joint)), 1e-20))
    bundle = {
        "conditions": {
            "identity": identity(values),
            "temporal_reverse": reversed_values,
            "3d_allpass_phase": phase_values,
        },
        "metadata": {
            "original_shape": list(values.shape),
            "original_sha256": feature_hash(values),
            "temporal_reverse_sha256": feature_hash(reversed_values),
            "phase_sha256": feature_hash(phase_values),
            "temporal_power_relative_error_reverse": temporal_error,
            "joint_3d_power_relative_error_phase": phase_error,
            "temporal_power_tolerance": POWER_TOLERANCE,
            "phase_seed": seed,
            "feature_level_only": True,
            "pixel_reencoding_performed": False,
            "condition_interpretation": {
                "identity": "native feature round-trip",
                "temporal_reverse": "sampled feature-time reversal, not reversed-pixel video",
                "3d_allpass_phase": "joint spatiotemporal power-preserving phase counterfactual",
            },
        },
        "spatial_mean_control": spatial_mean_control(values),
    }
    require(temporal_error <= POWER_TOLERANCE, "Temporal reverse did not preserve temporal power")
    require(phase_error <= POWER_TOLERANCE, "All-pass phase did not preserve joint 3D power")
    return bundle


def load_feature_archive(archive: Path, feature_key: str, logical_shape: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Load a dense or exact-repeat feature from a probe NPZ archive.

    Probe archives may store an exact temporal repeat as one slice. Callers
    that have the JSON record's logical shape can pass it to reconstruct the
    full tensor without silently treating the stored slice as a one-frame
    sequence.
    """
    with np.load(archive, allow_pickle=False) as data:
        key = feature_key if feature_key in data else feature_key + "_features"
        require(key in data, f"Feature key not found in archive: {feature_key}")
        values = np.asarray(data[key])
    values = validate_features(values)
    if logical_shape is not None:
        require(len(logical_shape) == 4 and all(int(size) > 0 for size in logical_shape),
                "logical_shape must be a positive [T,H,W,D] shape")
        require(tuple(values.shape[1:]) == tuple(logical_shape[1:]),
                "Stored feature shape disagrees with logical spatial/channel shape")
        if values.shape[0] == 1 and int(logical_shape[0]) > 1:
            values = np.broadcast_to(values, tuple(int(size) for size in logical_shape)).copy()
        require(tuple(values.shape) == tuple(int(size) for size in logical_shape),
                "Stored feature shape does not match logical_shape")
    return values


def _run_checks(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Check output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261102)
    values = rng.normal(size=(8, 4, 6, 5)).astype(np.float64)
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, **metrics: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "metrics": metrics})
        if not condition:
            raise AssertionError(f"{name}: {metrics}")

    same = identity(values)
    check("identity_exact", np.array_equal(same, values), hash=feature_hash(same))
    reversed_values = temporal_reverse(values)
    temporal_error = float(np.max(np.abs(temporal_power(values) - temporal_power(reversed_values))))
    check("temporal_reverse_preserves_temporal_power", temporal_error <= 1e-10, max_power_error=temporal_error)
    phase_values = three_d_allpass_phase(values, seed=17)
    joint_error = float(np.max(np.abs(joint_3d_power(values) - joint_3d_power(phase_values))))
    check("allpass_phase_preserves_joint_3d_power", joint_error <= 1e-10,
          max_power_error=joint_error, rms_change=float(np.linalg.norm(values - phase_values)))
    check("allpass_phase_changes_features", not np.array_equal(values, phase_values))
    summary = spatial_mean_control(values)
    check("spatial_mean_control_is_finite", np.isfinite(summary["full_temporal_power_fhw"]).all() and
          np.isfinite(summary["pooled_temporal_power_f"]).all(),
          spatial_constant_energy_ratio=summary["spatial_constant_energy_ratio"])
    bundle = counterfactual_bundle(values, seed=17)
    check("bundle_reports_feature_level_only", bundle["metadata"]["feature_level_only"] and
          not bundle["metadata"]["pixel_reencoding_performed"])
    check("bundle_preservation_errors_pass", bundle["metadata"]["temporal_power_relative_error_reverse"] <= POWER_TOLERANCE and
          bundle["metadata"]["joint_3d_power_relative_error_phase"] <= POWER_TOLERANCE,
          temporal_error=bundle["metadata"]["temporal_power_relative_error_reverse"],
          phase_error=bundle["metadata"]["joint_3d_power_relative_error_phase"])
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU algebra and feature-level counterfactual checks; no model or task claim",
        "python": platform.python_version(), "numpy": np.__version__, "dtype": "float64",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_used": False, "torch_imported": False, "check_count": len(checks),
        "all_passed": all(item["passed"] for item in checks), "checks": checks,
    }
    (output / "checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "examples.npz", original=values, reverse=reversed_values, phase=phase_values)
    (output / "report.md").write_text(
        "# Feature counterfactual CPU checks\n\n"
        f"All {len(checks)} checks passed. These are feature-level algebra checks only; no pixels, models or GPU were used.\n\n"
        f"- Temporal reverse power error: {temporal_error:.3g}.\n"
        f"- Joint 3D all-pass phase power error: {joint_error:.3g}.\n"
        f"- All-pass phase RMS feature change: {np.linalg.norm(values - phase_values):.6g}.\n"
        f"- Spatial-constant temporal-energy ratio: {summary['spatial_constant_energy_ratio']}.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run deterministic CPU checks")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_counterfactual_cpu_checks_v3"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--feature-key", default="vision_final_postmerger_features")
    parser.add_argument("--logical-shape", type=int, nargs=4, metavar=("T", "H", "W", "D"),
                        help="Logical [T,H,W,D] shape when the probe stored an exact temporal repeat")
    args = parser.parse_args()
    if args.check:
        result = _run_checks(args.output_dir)
        print(json.dumps({"checks": result["check_count"], "all_passed": result["all_passed"],
                          "output": str(args.output_dir.resolve())}, ensure_ascii=False))
        return
    if args.archive is None:
        parser.error("--archive is required unless --check is used")
    features = load_feature_archive(args.archive, args.feature_key,
                                    tuple(args.logical_shape) if args.logical_shape is not None else None)
    result = counterfactual_bundle(features)
    print(json.dumps({"shape": result["metadata"]["original_shape"],
                      "temporal_reverse_power_error": result["metadata"]["temporal_power_relative_error_reverse"],
                      "phase_power_error": result["metadata"]["joint_3d_power_relative_error_phase"],
                      "spatial_constant_energy_ratio": result["spatial_mean_control"]["spatial_constant_energy_ratio"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
