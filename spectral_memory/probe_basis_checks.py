"""CPU checks for Fourier-basis probe equivalence and spectral identifiability."""

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


SEED = 20261101


def real_fourier(values: np.ndarray, axis: int = 0) -> np.ndarray:
    source = np.asarray(values)
    if np.iscomplexobj(source) or source.ndim == 0 or not np.isfinite(source).all():
        raise ValueError("Expected a finite, real, non-scalar array")
    moved = np.moveaxis(source.astype(np.float64), axis, 0)
    count = len(moved)
    if count < 1:
        raise ValueError("The time axis cannot be empty")
    spectrum = np.fft.rfft(moved, axis=0, norm="ortho")
    packed = [spectrum[0].real]
    for frequency in range(1, (count + 1) // 2):
        packed.extend((np.sqrt(2) * spectrum[frequency].real, np.sqrt(2) * spectrum[frequency].imag))
    if count % 2 == 0:
        packed.append(spectrum[-1].real)
    return np.moveaxis(np.stack(packed), 0, axis)


def inverse_real_fourier(values: np.ndarray, axis: int = 0) -> np.ndarray:
    source = np.asarray(values)
    if np.iscomplexobj(source) or source.ndim == 0 or not np.isfinite(source).all():
        raise ValueError("Expected a finite, real, non-scalar packed array")
    moved = np.moveaxis(source.astype(np.float64), axis, 0)
    count = len(moved)
    if count < 1:
        raise ValueError("The packed axis cannot be empty")
    spectrum = np.zeros((count // 2 + 1, *moved.shape[1:]), dtype=np.complex128)
    spectrum[0] = moved[0]
    for frequency in range(1, (count + 1) // 2):
        spectrum[frequency] = (moved[2 * frequency - 1] + 1j * moved[2 * frequency]) / np.sqrt(2)
    if count % 2 == 0:
        spectrum[-1] = moved[-1]
    return np.moveaxis(np.fft.irfft(spectrum, n=count, axis=0, norm="ortho"), 0, axis)


def one_sided_weights(count: int) -> np.ndarray:
    weights = np.full(count // 2 + 1, 2.0)
    weights[0] = 1.0
    if count % 2 == 0:
        weights[-1] = 1.0
    return weights


def maximum_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second)))


def ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0)
    target_mean = targets.mean(axis=0)
    centered = features - feature_mean
    centered_target = targets - target_mean
    weights = np.linalg.solve(centered.T @ centered + penalty, centered.T @ centered_target)
    return weights, target_mean - feature_mean @ weights


def predict(features: np.ndarray, fitted: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    weights, intercept = fitted
    return features @ weights + intercept


def silu(values: np.ndarray) -> np.ndarray:
    return values / (1 + np.exp(-values))


def silu_derivative(values: np.ndarray) -> np.ndarray:
    sigmoid = 1 / (1 + np.exp(-values))
    return sigmoid + values * sigmoid * (1 - sigmoid)


def run_checks(output: Path) -> dict:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Check output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(SEED)
    checks = []
    examples = {}

    def check(name: str, condition: bool, **metrics) -> None:
        checks.append({"name": name, "passed": bool(condition), "metrics": metrics})
        if not condition:
            raise AssertionError(f"{name}: {metrics}")

    for count in (1, 2, 3, 7, 8, 31, 32):
        source = random.normal(size=(count, 3))
        basis = real_fourier(np.eye(count))
        packed = real_fourier(source)
        inverse = inverse_real_fourier(packed)
        orthogonality_error = maximum_error(basis.T @ basis, np.eye(count))
        roundtrip_error = maximum_error(inverse, source)
        energy_error = float(abs(np.sum(source**2) - np.sum(packed**2)))
        complex_spectrum = np.fft.rfft(source, axis=0, norm="ortho")
        weighted_energy_error = float(abs(np.sum(source**2) - np.sum(one_sided_weights(count)[:, None] * abs(complex_spectrum)**2)))
        check(f"complete_real_fourier_is_orthogonal_T{count}",
              orthogonality_error < 1e-12 and roundtrip_error < 1e-12
              and energy_error < 1e-10 and weighted_energy_error < 1e-10,
              orthogonality_error=orthogonality_error, inverse_error=roundtrip_error,
              energy_error=energy_error, rfft_weighted_energy_error=weighted_energy_error,
              real_dimension=int(packed.shape[0]))
        dc = real_fourier(np.ones(count))
        check(f"dc_is_single_unscaled_endpoint_T{count}",
              abs(dc[0] - np.sqrt(count)) < 1e-12 and np.linalg.norm(dc[1:]) < 1e-12)
        if count % 2 == 0:
            nyquist = real_fourier((-1.0)**np.arange(count))
            check(f"even_nyquist_is_single_unscaled_endpoint_T{count}",
                  abs(nyquist[-1] - np.sqrt(count)) < 1e-12 and np.linalg.norm(nyquist[:-1]) < 1e-12)
        elif count > 1:
            largest_positive = np.sin(2 * np.pi * (count // 2) * np.arange(count) / count)
            packed_largest = real_fourier(largest_positive)
            check(f"odd_last_rfft_bin_retains_imaginary_component_T{count}",
                  abs(abs(packed_largest[-1]) - np.sqrt(count / 2)) < 1e-11)

    ridge_metrics = []
    for count, samples, channels in ((7, 25, 3), (8, 7, 3), (32, 28, 2)):
        trajectories = random.normal(size=(samples + 11, count, channels))
        time_features = trajectories.reshape(samples + 11, -1)
        frequency_features = real_fourier(trajectories, axis=1).reshape(samples + 11, -1)
        basis = np.kron(real_fourier(np.eye(count)), np.eye(channels))
        mapping_error = maximum_error(frequency_features, time_features @ basis.T)
        arbitrary_weights = random.normal(size=(count * channels, 2))
        arbitrary_intercept = random.normal(size=2)
        hypothesis_error = maximum_error(time_features @ arbitrary_weights + arbitrary_intercept,
                                         frequency_features @ (basis @ arbitrary_weights) + arbitrary_intercept)
        norm_error = abs(float(np.sum(arbitrary_weights**2) - np.sum((basis @ arbitrary_weights)**2)))
        check(f"linear_function_class_and_l2_norm_equivalent_T{count}",
              mapping_error < 1e-12 and hypothesis_error < 1e-11 and norm_error < 1e-10,
              feature_mapping_error=mapping_error, prediction_error=hypothesis_error, norm_error=norm_error)
        targets = time_features[:samples] @ arbitrary_weights + arbitrary_intercept + random.normal(scale=0.2, size=(samples, 2))
        for regularization in (0.03, 0.7, 12.0):
            penalty = regularization * np.eye(count * channels)
            time_fit = ridge_fit(time_features[:samples], targets, penalty)
            frequency_fit = ridge_fit(frequency_features[:samples], targets, penalty)
            time_predictions = predict(time_features[samples:], time_fit)
            frequency_predictions = predict(frequency_features[samples:], frequency_fit)
            prediction_error = maximum_error(time_predictions, frequency_predictions)
            coefficient_error = maximum_error(frequency_fit[0], basis @ time_fit[0])
            intercept_error = maximum_error(time_fit[1], frequency_fit[1])
            check(f"ridge_same_predictions_T{count}_lambda{regularization}",
                  prediction_error < 1e-9 and coefficient_error < 1e-9 and intercept_error < 1e-9,
                  test_prediction_error=prediction_error, coefficient_rotation_error=coefficient_error,
                  intercept_error=intercept_error, samples=samples, features=count * channels)
            ridge_metrics.append({"time_samples": count, "training_examples": samples, "features": count * channels,
                                  "lambda": regularization, "test_prediction_error": prediction_error})
        time_centered = time_features[:samples] - time_features[:samples].mean(axis=0)
        frequency_centered = frequency_features[:samples] - frequency_features[:samples].mean(axis=0)
        target_centered = targets - targets.mean(axis=0)
        time_minimum_norm = np.linalg.pinv(time_centered, rcond=1e-12) @ target_centered
        frequency_minimum_norm = np.linalg.pinv(frequency_centered, rcond=1e-12) @ target_centered
        minimum_norm_error = maximum_error(frequency_minimum_norm, basis @ time_minimum_norm)
        check(f"zero_ridge_minimum_euclidean_norm_solution_equivalent_T{count}",
              minimum_norm_error < 1e-10, coefficient_error=minimum_norm_error)
        if count == 8:
            examples["ridge_time_test_predictions"] = time_predictions
            examples["ridge_frequency_test_predictions"] = frequency_predictions

    count = 16
    basis = real_fourier(np.eye(count))
    train = random.normal(size=(40, count))
    test = random.normal(size=(13, count))
    targets = train @ random.normal(size=(count, 2)) + random.normal(scale=0.3, size=(40, 2))
    transformed_train, transformed_test = train @ basis.T, test @ basis.T
    regularization = 8.0
    isotropic = regularization * np.eye(count)
    baseline_fit = ridge_fit(train, targets, isotropic)
    baseline_prediction = predict(test, baseline_fit)
    scaling = np.full(count, 1 / np.sqrt(2))
    scaling[[0, -1]] = 1.0
    unweighted_train, unweighted_test = transformed_train * scaling, transformed_test * scaling
    unweighted_fit = ridge_fit(unweighted_train, targets, isotropic)
    unweighted_difference = maximum_error(baseline_prediction, predict(unweighted_test, unweighted_fit))
    corrected_fit = ridge_fit(unweighted_train, targets, regularization * np.diag(scaling**2))
    corrected_error = maximum_error(baseline_prediction, predict(unweighted_test, corrected_fit))
    check("omitting_sqrt2_changes_isotropic_ridge_but_not_invertibility",
          unweighted_difference > 1e-3 and corrected_error < 1e-11,
          unequal_prediction_difference=unweighted_difference, corrected_penalty_prediction_error=corrected_error)

    frequency_penalty = np.diag(np.geomspace(0.2, 15, count))
    frequency_fit = ridge_fit(transformed_train, targets, frequency_penalty)
    matching_time_fit = ridge_fit(train, targets, basis.T @ frequency_penalty @ basis)
    anisotropic_error = maximum_error(predict(transformed_test, frequency_fit), predict(test, matching_time_fit))
    anisotropic_difference = maximum_error(predict(transformed_test, frequency_fit), baseline_prediction)
    check("anisotropic_frequency_regularizer_equals_rotated_time_regularizer",
          anisotropic_error < 1e-11 and anisotropic_difference > 1e-3,
          matched_prediction_error=anisotropic_error, difference_from_isotropic_time=anisotropic_difference)

    selected_rows = np.arange(5)
    subbasis = basis[selected_rows]
    time_projection = subbasis.T @ subbasis
    truncated_fit = ridge_fit(train @ subbasis.T, targets, regularization * np.eye(len(selected_rows)))
    projected_fit = ridge_fit(train @ time_projection, targets, isotropic)
    truncation_error = maximum_error(predict(test @ subbasis.T, truncated_fit), predict(test @ time_projection, projected_fit))
    collision_first = random.normal(size=count)
    collision_second = collision_first + basis[-1]
    collision_error = maximum_error(subbasis @ collision_first, subbasis @ collision_second)
    check("truncated_spectral_probe_equals_same_time_subspace",
          truncation_error < 1e-11 and collision_error < 1e-12,
          matched_prediction_error=truncation_error, discarded_direction_collision_error=collision_error,
          original_feature_distance=float(np.linalg.norm(collision_first - collision_second)))
    per_trajectory_centered = train - train.mean(axis=1, keepdims=True)
    dropped_dc = transformed_train.copy()
    dropped_dc[:, 0] = 0
    error = maximum_error(real_fourier(per_trajectory_centered, axis=1), dropped_dc)
    check("time_mean_removal_equals_dc_projection_not_intercept_removal", error < 1e-12, maximum_error=error)

    train_standard_deviation = train.std(axis=0)
    spectrum_standard_deviation = transformed_train.std(axis=0)
    time_scaled = train / train_standard_deviation
    spectrum_scaled = transformed_train / spectrum_standard_deviation
    time_scaled_fit = ridge_fit(time_scaled, targets, isotropic)
    spectrum_scaled_fit = ridge_fit(spectrum_scaled, targets, isotropic)
    standardization_difference = maximum_error(predict(test / train_standard_deviation, time_scaled_fit),
                                               predict(transformed_test / spectrum_standard_deviation, spectrum_scaled_fit))
    check("independent_per_coordinate_standardization_changes_ridge_prior",
          standardization_difference > 1e-3, prediction_difference=standardization_difference)

    count = 31
    original = random.normal(size=count)
    reversed_time = original[::-1]
    shifted = np.roll(original, 7)
    original_transform = np.fft.fft(original, norm="ortho")
    frequency = np.arange(count)
    reversal_predicted = np.exp(2j * np.pi * frequency / count) * original_transform.conj()
    shift_predicted = np.exp(-2j * np.pi * frequency * 7 / count) * original_transform
    reversal_formula_error = maximum_error(np.fft.fft(reversed_time, norm="ortho"), reversal_predicted)
    shift_formula_error = maximum_error(np.fft.fft(shifted, norm="ortho"), shift_predicted)
    reversal_power_error = maximum_error(abs(original_transform)**2, abs(np.fft.fft(reversed_time, norm="ortho"))**2)
    shift_power_error = maximum_error(abs(original_transform)**2, abs(np.fft.fft(shifted, norm="ortho"))**2)
    check("unwindowed_power_cannot_distinguish_reversal_or_circular_shift",
          reversal_formula_error < 1e-11 and shift_formula_error < 1e-11
          and reversal_power_error < 1e-11 and shift_power_error < 1e-11,
          reversal_formula_error=reversal_formula_error, shift_formula_error=shift_formula_error,
          reversal_power_error=reversal_power_error, shift_power_error=shift_power_error)
    separator = original - reversed_time
    two_scores = np.array([original @ separator, reversed_time @ separator])
    check("phase_preserving_linear_probe_can_separate_one_reversal_pair",
          two_scores[0] > 0 and two_scores[1] < 0, signed_scores=two_scores.tolist(),
          limitation="This is a pairwise construction, not generalization evidence.")

    count = 16
    basis = real_fourier(np.eye(count))
    trajectories = random.normal(size=(12, count))
    band_coefficients = random.normal(size=count // 2 + 1)
    real_coefficients = [band_coefficients[0]]
    for frequency_index in range(1, count // 2):
        real_coefficients.extend([band_coefficients[frequency_index]] * 2)
    real_coefficients.append(band_coefficients[-1])
    quadratic = basis.T @ np.diag(real_coefficients) @ basis
    powers = one_sided_weights(count) * abs(np.fft.rfft(trajectories, axis=1, norm="ortho"))**2
    power_scores = powers @ band_coefficients
    quadratic_scores = np.einsum("ni,ij,nj->n", trajectories, quadratic, trajectories)
    score_error = maximum_error(power_scores, quadratic_scores)
    circulation_error = maximum_error(quadratic, np.stack([np.roll(quadratic[0], index) for index in range(count)]))
    check("linear_power_probe_is_a_restricted_time_quadratic_probe",
          score_error < 1e-11 and circulation_error < 1e-12,
          score_error=score_error, symmetric_circulant_error=circulation_error)

    count = 64
    time = np.arange(count)
    operating_trajectory = 0.5 + 0.2 * np.cos(2 * np.pi * 3 * time / count)
    perturbation = np.cos(2 * np.pi * 5 * time / count)
    jacobian = silu_derivative(operating_trajectory) * operating_trajectory + silu(operating_trajectory)
    response = jacobian * perturbation
    step = 1e-5
    positive = operating_trajectory + step * perturbation
    negative = operating_trajectory - step * perturbation
    finite_difference = (silu(positive) * positive - silu(negative) * negative) / (2 * step)
    derivative_error = maximum_error(response, finite_difference)
    response_spectrum = np.fft.fft(response, norm="ortho")
    perturbation_spectrum = np.fft.fft(perturbation, norm="ortho")
    dft = np.fft.fft(np.eye(count), axis=0, norm="ortho")
    frequency_operator = dft @ np.diag(jacobian) @ dft.conj().T
    off_diagonal = frequency_operator - np.diag(np.diag(frequency_operator))
    constant_operator = dft @ (np.eye(count) * float(jacobian.mean())) @ dft.conj().T
    constant_off_diagonal = constant_operator - np.diag(np.diag(constant_operator))
    shift_noncommutation = maximum_error(jacobian * np.roll(perturbation, 1), np.roll(response, 1))
    check("trajectory_local_swiglu_jacobian_is_generally_time_varying",
          derivative_error < 1e-8 and np.linalg.norm(off_diagonal) > 0.1
          and abs(perturbation_spectrum[8]) < 1e-12 and abs(response_spectrum[8]) > 0.01
          and shift_noncommutation > 1e-3,
          finite_difference_error=derivative_error, frequency_off_diagonal_frobenius=float(np.linalg.norm(off_diagonal)),
          new_sideband_bin_8_magnitude=float(abs(response_spectrum[8])),
          shift_noncommutation_error=shift_noncommutation)
    check("constant_local_jacobian_is_frequency_diagonal",
          np.linalg.norm(constant_off_diagonal) < 1e-11,
          frequency_off_diagonal_frobenius=float(np.linalg.norm(constant_off_diagonal)))
    jacobian_spectrum = np.fft.fft(jacobian, norm="ortho")
    predicted_operator = np.array([[jacobian_spectrum[(output_frequency - input_frequency) % count] / np.sqrt(count)
                                   for input_frequency in range(count)] for output_frequency in range(count)])
    error = maximum_error(predicted_operator, frequency_operator)
    check("time_varying_jacobian_fourier_mixing_formula", error < 1e-12, maximum_error=error)

    for invalid in (np.array([1 + 1j]), np.array([]), np.array([np.nan]), np.array(1.0)):
        try:
            real_fourier(invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid Fourier input was silently accepted")
    check("invalid_complex_empty_nonfinite_scalar_inputs_rejected", True)

    examples.update({"reversal_original": original, "reversal_reversed": reversed_time,
                     "reversal_shifted": shifted, "local_swiglu_operating_trajectory": operating_trajectory,
                     "local_swiglu_perturbation": perturbation, "local_swiglu_response": response,
                     "local_swiglu_frequency_operator": frequency_operator})
    np.savez(output / "examples.npz", **examples)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "purpose": "Synthetic CPU theorem checks and counterexamples; no model, video or task-performance claim",
        "python": platform.python_version(), "numpy": np.__version__, "dtype": "float64",
        "processes": 1, "thread_limits": {name: os.environ[name] for name in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "torch_imported": False, "gpu_used": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "check_count": len(checks), "all_passed": all(item["passed"] for item in checks),
        "ridge_equivalence_cases": ridge_metrics, "checks": checks,
    }
    (output / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    maximum_ridge_error = max(item["test_prediction_error"] for item in ridge_metrics)
    summary = [
        "# Probe basis CPU checks", "",
        f"All {len(checks)} checks passed. NumPy {np.__version__}, float64, seed {SEED}, one process, two library threads.", "",
        "These are synthetic numerical checks, not Qwen/VLM or QA results.", "",
        f"- Complete orthogonal Fourier versus time-domain ridge: maximum test-prediction discrepancy {maximum_ridge_error:.3e}.",
        f"- Missing sqrt(2) scaling with unchanged isotropic penalty: prediction discrepancy {unweighted_difference:.6g}; corrected penalty error {corrected_error:.3e}.",
        f"- Frequency-anisotropic penalty versus its rotated time penalty: error {anisotropic_error:.3e}.",
        f"- Reversal/circular-shift power discrepancies: {reversal_power_error:.3e} / {shift_power_error:.3e}.",
        f"- Time-varying local SwiGLU Jacobian: off-diagonal frequency-operator norm {np.linalg.norm(off_diagonal):.6g}; constant-J control {np.linalg.norm(constant_off_diagonal):.3e}.", "",
        "Full metrics: checks.json. Exact acceptance criteria: source check conditions. Small synthetic examples: examples.npz.", "",
        "| Check | Status |", "| --- | --- |",
        *[f"| {item['name']} | PASS |" for item in checks], "",
    ]
    (output / "report.md").write_text("\n".join(summary), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/probe_basis_checks"))
    arguments = parser.parse_args()
    report = run_checks(arguments.output_dir)
    print(json.dumps({"checks": report["check_count"], "all_passed": report["all_passed"],
                      "output": str(arguments.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
