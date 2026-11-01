"""Query-free selection of consecutive frame pairs from a hidden trajectory."""

from numbers import Integral
from time import perf_counter

import numpy as np


METHODS = (
    "uniform",
    "recent",
    "random",
    "temporal_difference",
    "kcenter",
    "temporal_reconstruction",
    "temporal_centered",
    "spectral_balanced",
    "spectral_power",
)
BAND_EDGES_CYCLES_PER_SAMPLE = (0.0, 0.125, 0.25, 0.5)
BAND_ENERGY_FLOOR_FRACTION = 0.01


def _validate(
    hidden_states: np.ndarray,
    timestamps_s: np.ndarray,
    budget_pairs: int,
    method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(hidden_states)
    if np.iscomplexobj(source):
        raise ValueError("hidden_states must be real-valued")
    states = np.asarray(source, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 1:
        raise ValueError("hidden_states must have shape [T, D], T >= 2, D >= 1")
    if times.ndim != 1 or times.size != states.shape[0]:
        raise ValueError("timestamps_s must have shape [T] matching hidden_states")
    if not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError("hidden_states and timestamps_s must be finite")
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("timestamps_s must be strictly increasing")
    if not np.allclose(intervals, intervals.mean(), rtol=1e-5, atol=1e-8):
        raise ValueError("timestamps_s must be uniform; resample explicitly first")
    if isinstance(budget_pairs, bool) or not isinstance(budget_pairs, Integral):
        raise ValueError("budget_pairs must be an integer")
    if not 2 <= budget_pairs <= states.shape[0]:
        raise ValueError("budget_pairs must satisfy 2 <= budget_pairs <= T")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return source, states, times


def _gram_coordinates(states: np.ndarray) -> tuple[np.ndarray, dict]:
    gram = states @ states.T
    if not np.isfinite(gram).all():
        raise ValueError("hidden_states are too large for finite float64 energies")
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > 0
    coordinates = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    if coordinates.shape[1] == 0:
        coordinates = np.zeros((states.shape[0], 1), dtype=np.float64)
    denominator = float(np.linalg.norm(gram))
    residual = float(np.linalg.norm(coordinates @ coordinates.T - gram))
    relative_residual = residual / denominator if denominator else 0.0
    if relative_residual > 1e-10:
        raise ValueError("Gram coordinates failed the metric-equivalence check")
    return coordinates, {
        "method": "gram_eigendecomposition",
        "input_channels": int(states.shape[1]),
        "coordinate_channels": int(coordinates.shape[1]),
        "relative_gram_residual": relative_residual,
        "negative_eigenvalues_clipped": int(np.count_nonzero(eigenvalues < 0)),
        "smallest_eigenvalue": float(eigenvalues.min()),
        "equivalence": (
            "For a temporal linear operator A, ||A X||_F^2 = "
            "tr(A X X^T A*) = ||A Z||_F^2 when Z Z^T = X X^T. "
            "This preserves interpolation residuals, channel-summed complex "
            "Fourier residuals and Euclidean distances, up to recorded roundoff. "
            "It does not preserve per-channel magnitude-spectrum differences."
        ),
    }


def _reconstruct(states: np.ndarray, times: np.ndarray, selected: list[int]) -> np.ndarray:
    anchors = np.asarray(sorted(selected), dtype=np.int64)
    positions = np.arange(states.shape[0])
    right = np.searchsorted(anchors, positions, side="left")
    right = np.clip(right, 0, anchors.size - 1)
    left = np.clip(right - 1, 0, anchors.size - 1)
    left = np.where(positions >= anchors[-1], anchors.size - 1, left)
    left_times = times[anchors[left]]
    spans = times[anchors[right]] - left_times
    fractions = np.divide(
        times - left_times,
        spans,
        out=np.zeros_like(times),
        where=spans > 0,
    )
    reconstructed = states[anchors[left]] + fractions[:, None] * (
        states[anchors[right]] - states[anchors[left]]
    )
    reconstructed[anchors] = states[anchors]
    return reconstructed


def _spectrum(states: np.ndarray) -> np.ndarray:
    centered = states - states.mean(axis=0, keepdims=True)
    return np.fft.rfft(centered, axis=0, norm="ortho")


def _spectral_setup(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    transformed = _spectrum(states)
    frequencies = np.fft.rfftfreq(states.shape[0], d=1.0)
    weights = np.full(frequencies.size, 2.0)
    weights[0] = 0.0
    if states.shape[0] % 2 == 0:
        weights[-1] = 1.0
    bands = [
        (frequencies > lower) & (frequencies <= upper)
        for lower, upper in zip(
            BAND_EDGES_CYCLES_PER_SAMPLE[:-1],
            BAND_EDGES_CYCLES_PER_SAMPLE[1:],
        )
    ]
    return transformed, weights, bands


def _band_errors(
    original: np.ndarray,
    reconstructed: np.ndarray,
    weights: np.ndarray,
    bands: list[np.ndarray],
    magnitude_only: bool,
) -> tuple[float, list[dict]]:
    original_energy = np.sum(np.abs(original) ** 2, axis=1) * weights
    total_dynamic_energy = float(original_energy.sum())
    if magnitude_only:
        difference = np.abs(reconstructed) - np.abs(original)
    else:
        difference = reconstructed - original
    error_energy = np.sum(np.abs(difference) ** 2, axis=1) * weights
    diagnostics = []
    for band_index, mask in enumerate(bands):
        band_energy = float(original_energy[mask].sum())
        error = float(error_energy[mask].sum())
        denominator = band_energy + BAND_ENERGY_FLOOR_FRACTION * total_dynamic_energy
        relative_error = error / denominator if denominator else 0.0
        diagnostics.append(
            {
                "lower_exclusive_cycles_per_sample": BAND_EDGES_CYCLES_PER_SAMPLE[band_index],
                "upper_inclusive_cycles_per_sample": BAND_EDGES_CYCLES_PER_SAMPLE[band_index + 1],
                "bin_count": int(mask.sum()),
                "original_energy": band_energy,
                "reconstruction_error_energy": error,
                "normalization_denominator": denominator,
                "relative_error": relative_error,
            }
        )
    return float(np.mean([item["relative_error"] for item in diagnostics])), diagnostics


def _diagnostics(states: np.ndarray, times: np.ndarray, selected: list[int]) -> dict:
    reconstructed = _reconstruct(states, times, selected)
    original_spectrum, weights, bands = _spectral_setup(states)
    reconstruction_spectrum = _spectrum(reconstructed)
    complex_error, complex_bands = _band_errors(
        original_spectrum, reconstruction_spectrum, weights, bands, False
    )
    magnitude_error, magnitude_bands = _band_errors(
        original_spectrum, reconstruction_spectrum, weights, bands, True
    )
    squared_error = float(np.sum((reconstructed - states) ** 2))
    residual = reconstructed - states
    centered_residual = residual - residual.mean(axis=0, keepdims=True)
    centered_squared_error = float(np.sum(centered_residual**2))
    dynamic_energy = float(np.sum((states - states.mean(axis=0)) ** 2))
    return {
        "temporal_squared_error": squared_error,
        "temporal_mean_squared_error": squared_error / states.size,
        "temporal_centered_squared_error": centered_squared_error,
        "temporal_centered_mean_squared_error": centered_squared_error / states.size,
        "relative_temporal_squared_error": squared_error / dynamic_energy if dynamic_energy else 0.0,
        "original_dynamic_energy": dynamic_energy,
        "mean_state_squared_bias": float(np.mean((reconstructed.mean(axis=0) - states.mean(axis=0)) ** 2)),
        "spectral_balanced_error": complex_error,
        "spectral_power_magnitude_error": magnitude_error,
        "complex_band_errors": complex_bands,
        "magnitude_band_errors": magnitude_bands,
    }


def select_memory(
    hidden_states: np.ndarray,
    timestamps_s: np.ndarray,
    budget_pairs: int,
    method: str,
    seed: int = 17,
) -> dict:
    """Return chronological indices for a fixed budget of adjacent frame pairs.

    Hidden row g represents original frames [2*g, 2*g+1]. The caller must
    verify this mapping against extraction provenance. All policies retain
    both endpoints except recent, which keeps only the most recent rows.
    This is offline prefix selection: the entire supplied hidden trajectory
    must already be available. No query, label, reward or future suffix is used.

    Spectral objectives use an unwindowed, mean-centered, orthonormal real FFT
    with one-sided Parseval weights and fixed bands in cycles per sample.
    spectral_power compares magnitudes in the original hidden channel basis.
    It intentionally drops phase and is not invariant to channel rotations.
    The three band-relative errors are averaged, with empty bands contributing
    zero. A constant trajectory has zero spectral error by convention.
    """
    started = perf_counter()
    source, states, times = _validate(hidden_states, timestamps_s, budget_pairs, method, seed)
    budget_pairs = int(budget_pairs)
    count, channels = states.shape
    shifted = states - states[:1]
    coordinates, reduction = _gram_coordinates(shifted)
    objective_states = shifted if method == "spectral_power" else coordinates
    original_spectrum, weights, bands = _spectral_setup(objective_states)
    selection_trace = []
    objective_evaluations = 0

    def objective(candidate: list[int]) -> float:
        reconstruction = _reconstruct(objective_states, times, candidate)
        if method in ("temporal_reconstruction", "temporal_centered"):
            residual = objective_states - reconstruction
            if method == "temporal_centered":
                residual = residual - residual.mean(axis=0, keepdims=True)
            return float(np.sum(residual**2) / (count * channels))
        transformed = _spectrum(reconstruction)
        return _band_errors(
            original_spectrum, transformed, weights, bands, method == "spectral_power"
        )[0]

    selected = [0, count - 1]
    if method == "uniform":
        selected = np.rint(np.linspace(0, count - 1, budget_pairs)).astype(int).tolist()
    elif method == "recent":
        selected = list(range(count - budget_pairs, count))
    elif method == "random":
        generator = np.random.default_rng(int(seed))
        selected += generator.choice(np.arange(1, count - 1), budget_pairs - 2, replace=False).tolist()
    elif method == "temporal_difference":
        differences = np.sum(np.diff(coordinates, axis=0) ** 2, axis=1)
        candidates = sorted(range(1, count - 1), key=lambda index: (-differences[index - 1], index))
        selected += candidates[: budget_pairs - 2]
    elif method == "kcenter":
        distances = np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=2)
        while len(selected) < budget_pairs:
            candidates = [index for index in range(count) if index not in selected]
            nearest = distances[:, selected].min(axis=1)
            chosen = max(candidates, key=lambda index: (nearest[index], -index))
            selection_trace.append({"added_pair_index": chosen, "nearest_selected_squared_distance": float(nearest[chosen])})
            selected.append(chosen)
    else:
        while len(selected) < budget_pairs:
            candidates = [index for index in range(count) if index not in selected]
            scores = [objective(selected + [index]) for index in candidates]
            objective_evaluations += len(candidates)
            best_position = int(np.argmin(scores))
            chosen = candidates[best_position]
            selection_trace.append({"added_pair_index": chosen, "objective_after_addition": scores[best_position]})
            selected.append(chosen)

    selected = sorted(selected)
    diagnostics = _diagnostics(shifted, times, selected)
    reduced_diagnostics = _diagnostics(coordinates, times, selected)
    for key in ("temporal_squared_error", "temporal_centered_squared_error", "spectral_balanced_error"):
        reference = diagnostics[key]
        difference = abs(reference - reduced_diagnostics[key])
        tolerance = 1e-9 * max(1.0, abs(reference))
        if difference > tolerance:
            raise ValueError(f"Gram reduction failed final {key} equivalence")
        reduction[f"final_{key}_absolute_discrepancy"] = difference
    selected_frame_indices = [frame for index in selected for frame in (2 * index, 2 * index + 1)]
    interval = float(np.diff(times).mean())
    return {
        "method": method,
        "seed": int(seed),
        "budget_pairs": budget_pairs,
        "candidate_pair_count": count,
        "selected_indices": selected,
        "indices": selected,
        "selected_pair_indices": selected,
        "selected_timestamps_s": times[selected].tolist(),
        "selected_original_frame_indices": selected_frame_indices,
        "selected_input_frame_positions": selected_frame_indices,
        "selected_frame_count": 2 * budget_pairs,
        "retains_first_pair": selected[0] == 0,
        "retains_last_pair": selected[-1] == count - 1,
        "endpoint_policy": "latest contiguous pairs; first pair not forced" if method == "recent" else "first and last pairs retained",
        "diagnostics": diagnostics,
        "selection_trace": selection_trace,
        "objective_evaluations": objective_evaluations,
        "metadata": {
            "query_free": True,
            "uses_hidden_states_for_selection": method not in ("uniform", "recent", "random"),
            "uses_hidden_states": method not in ("uniform", "recent", "random"),
            "selection_scope": "offline over supplied available prefix",
            "frame_pair_mapping": "hidden row g maps to original frames [2*g, 2*g+1]; caller verifies provenance",
            "frame_index_semantics": "positions in the supplied sampled frame sequence; map through frame-manifest source_frame_indices to obtain decoded MP4 frame numbers",
            "feature_basis": (
                "not_used_by_selection_policy"
                if method in ("uniform", "recent", "random")
                else "original_hidden_channels" if method == "spectral_power" else "Gram_metric_coordinates"
            ),
            "diagnostic_basis": "original_hidden_channels",
            "gram_reduction": reduction,
            "input_dtype": str(source.dtype),
            "compute_dtype": "float64",
            "hidden_pair_sample_interval_s": interval,
            "frequency_units": "cycles_per_hidden_pair_sample",
            "band_edges_cycles_per_sample": list(BAND_EDGES_CYCLES_PER_SAMPLE),
            "band_edges_hz_for_this_clip": [edge / interval for edge in BAND_EDGES_CYCLES_PER_SAMPLE],
            "band_energy_floor_fraction": BAND_ENERGY_FLOOR_FRACTION,
            "spectrum": "mean-centered, boxcar, orthonormal rFFT, one-sided Parseval weights; DC excluded",
            "spectral_power_definition": "squared difference of per-channel magnitudes, not squared difference of powers",
            "interpolation": "piecewise linear in actual timestamps, constant extension outside retained anchors",
            "tie_break": "earliest candidate among exactly equal computed objectives",
            "temporal_difference_definition": "largest squared backward hidden differences, excluding forced endpoints",
            "temporal_centered_definition": "mean squared interpolation residual after subtracting its temporal per-channel mean; equivalent to flat-weight non-DC complex Fourier residual",
            "spectral_mean_bias": "DC is excluded from spectral objectives; mean-state bias is reported separately",
            "constant_signal_convention": "zero spectral error; optimized ties choose earliest candidate",
        },
        "storage_accounting": {
            "input_hidden_array_bytes": int(source.nbytes),
            "selected_hidden_array_bytes_if_retained": int(source[selected].nbytes),
            "selected_pair_indices_bytes_int64": 8 * budget_pairs,
            "selected_timestamps_bytes_float64": 8 * budget_pairs,
            "selected_original_frame_indices_bytes_int64": 16 * budget_pairs,
            "selected_rgb_payload_bytes": None,
            "reader_tokens": None,
            "reader_compute_included": False,
            "hidden_extraction_compute_included": False,
            "scope": "numeric array bytes; excludes JSON/container overhead, RGB storage, model weights, and reader; caller measures selected RGB files and reader tokens",
        },
        "selection_elapsed_seconds": perf_counter() - started,
        "selection_timing_scope": (
            "CPU selection plus shared Gram reduction and full reconstruction diagnostics; "
            "excludes hidden extraction, RGB persistence and reader; uniform/recent/random "
            "do not require hidden extraction or shared audit computations for deployment"
        ),
    }
