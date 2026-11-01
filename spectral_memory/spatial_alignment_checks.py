"""Small CPU counterexamples for temporal spectra and spatial alignment."""

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

import numpy as np


def align(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.stack([np.roll(frame, -int(position), axis=0) for frame, position in zip(values, positions)])


def non_dc_energy(values: np.ndarray) -> float:
    spectrum = np.fft.fft(values, axis=0, norm="ortho")
    return float(np.sum(np.abs(spectrum[1:]) ** 2))


def run_checks(output: Path) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Check output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    count = 16
    patches = 16
    object_feature = np.array([1.0, 0.5, -0.25])
    times = np.arange(count)
    clockwise_positions = times % patches
    counterclockwise_positions = (-times) % patches
    clockwise = np.zeros((count, patches, len(object_feature)))
    counterclockwise = np.zeros_like(clockwise)
    clockwise[times, clockwise_positions] = object_feature
    counterclockwise[times, counterclockwise_positions] = object_feature
    aligned_clockwise = align(clockwise, clockwise_positions)
    aligned_counterclockwise = align(counterclockwise, counterclockwise_positions)
    checks = []

    def check(name: str, condition: bool, **metrics) -> None:
        checks.append({"name": name, "passed": bool(condition), "metrics": metrics})
        if not condition:
            raise AssertionError(f"{name}: {metrics}")

    constant_feature_error = float(np.max(np.abs(clockwise[times, clockwise_positions] - object_feature)))
    check("object_feature_is_constant_while_location_moves", constant_feature_error == 0.0,
          feature_error=constant_feature_error, visited_patches=int(len(np.unique(clockwise_positions))))
    fixed_patch_energy = non_dc_energy(clockwise[:, 0])
    check("fixed_patch_has_temporal_spectrum_from_pure_translation", fixed_patch_energy > 1.0,
          fixed_patch_non_dc_energy=fixed_patch_energy, all_patches_non_dc_energy=non_dc_energy(clockwise))
    aligned_energy = non_dc_energy(aligned_clockwise)
    check("object_alignment_removes_translation_spectrum", aligned_energy < 1e-12,
          aligned_non_dc_energy=aligned_energy)
    energy_error = float(abs(np.sum(clockwise**2) - np.sum(aligned_clockwise**2)))
    check("spatial_permutation_preserves_total_energy", energy_error < 1e-12,
          total_energy_error=energy_error, total_energy=float(np.sum(clockwise**2)))
    collision_error = float(np.max(np.abs(aligned_clockwise - aligned_counterclockwise)))
    unaligned_distance = float(np.linalg.norm(clockwise - counterclockwise))
    check("opposite_motion_labels_collide_after_alignment", collision_error == 0.0 and unaligned_distance > 1.0,
          aligned_difference=collision_error, original_difference=unaligned_distance,
          motion_labels=["clockwise", "counterclockwise"])
    clockwise_step = (np.diff(clockwise_positions) + patches // 2) % patches - patches // 2
    counterclockwise_step = (np.diff(counterclockwise_positions) + patches // 2) % patches - patches // 2
    check("retaining_warp_metadata_preserves_direction", np.all(clockwise_step == 1) and np.all(counterclockwise_step == -1),
          clockwise_step=int(clockwise_step[0]), counterclockwise_step=int(counterclockwise_step[0]))
    mean_error = float(np.max(np.abs(clockwise.mean(axis=1) - aligned_clockwise.mean(axis=1))))
    check("spatial_mean_is_invariant_to_this_translation", mean_error < 1e-12,
          mean_error=mean_error, mean_non_dc_energy=non_dc_energy(clockwise.mean(axis=1)))
    transform_then_warp = align(np.fft.fft(clockwise, axis=0, norm="ortho"), clockwise_positions)
    warp_then_transform = np.fft.fft(aligned_clockwise, axis=0, norm="ortho")
    commutator = float(np.linalg.norm(warp_then_transform - transform_then_warp))
    check("time_varying_warp_does_not_commute_with_temporal_dft", commutator > 1.0,
          commutator_norm=commutator)
    constant_positions = np.full(count, 3)
    constant_first = np.fft.fft(align(clockwise, constant_positions), axis=0, norm="ortho")
    constant_second = align(np.fft.fft(clockwise, axis=0, norm="ortho"), constant_positions)
    constant_error = float(np.max(np.abs(constant_first - constant_second)))
    check("constant_spatial_warp_commutes_with_temporal_dft", constant_error < 1e-12,
          commutator_maximum_error=constant_error)
    warp_matrices = np.stack([np.eye(patches)[(np.arange(patches) + position) % patches]
                              for position in clockwise_positions])
    fourier = np.fft.fft(np.eye(count), axis=0, norm="ortho")
    frequency_operator = np.einsum("kt,tpd,qt->kqpd", fourier, warp_matrices, fourier.conj())
    expected_operator = np.fft.fft(warp_matrices, axis=0, norm="ortho")[(times[:, None] - times[None, :]) % count] / np.sqrt(count)
    formula_error = float(np.max(np.abs(frequency_operator - expected_operator)))
    off_diagonal = frequency_operator.copy()
    off_diagonal[times, times] = 0
    mixing_norm = float(np.linalg.norm(off_diagonal))
    check("time_varying_alignment_mixes_temporal_frequencies", formula_error < 1e-12 and mixing_norm > 1.0,
          frequency_operator_formula_error=formula_error, off_diagonal_frobenius=mixing_norm)

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Synthetic coordinate counterexamples only; no model, tracking or task accuracy claim",
        "python": platform.python_version(), "numpy": np.__version__, "dtype": "float64",
        "processes": 1, "library_thread_limit": 2, "torch_imported": False, "gpu_used": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "time_samples": count, "spatial_patches": patches, "channels": len(object_feature),
        "check_count": len(checks), "all_passed": all(item["passed"] for item in checks), "checks": checks,
    }
    (output / "checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "examples.npz", clockwise=clockwise, counterclockwise=counterclockwise,
                        aligned_clockwise=aligned_clockwise, aligned_counterclockwise=aligned_counterclockwise,
                        clockwise_positions=clockwise_positions, counterclockwise_positions=counterclockwise_positions)
    report = [
        "# Spatial alignment CPU counterexamples", "",
        f"All {len(checks)} checks passed. One NumPy process, two library threads, no torch/model/GPU.", "",
        f"- A constant object crossing patches gives fixed-patch non-DC energy {fixed_patch_energy:.9g}.",
        f"- Object alignment reduces non-DC energy to {aligned_energy:.3g} without reducing total energy.",
        f"- Opposite motion labels become identical aligned features: difference {collision_error:.3g}.",
        "- Retaining the warp/position sequence preserves the two opposite direction labels in this construction.",
        f"- Temporal DFT/time-varying warp commutator norm: {commutator:.9g}; constant-warp control: {constant_error:.3g}.",
        f"- Frequency-domain warp operator off-diagonal norm: {mixing_norm:.9g}.", "",
        "This is a periodic translation construction with oracle alignment, not evidence that tracking is feasible or memory improves.",
        "Exact criteria are in the source check conditions; metrics and source hash are in checks.json.", "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial_alignment_checks"))
    arguments = parser.parse_args()
    result = run_checks(arguments.output_dir)
    print(json.dumps({"checks": result["check_count"], "all_passed": result["all_passed"],
                      "output": str(arguments.output_dir.resolve())}))


if __name__ == "__main__":
    main()
