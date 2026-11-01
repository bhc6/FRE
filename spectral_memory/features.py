"""Temporal features with explicit physical frequencies and reversible controls."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Bands are in Hz; ordered samples span the whole event's relative time."""

    band_edges_hz: tuple[float, ...] = (0.0, 0.125, 0.25, 0.5)
    resample_points: int = 16
    stft_window: int = 16


def _validate_signal(
    hidden_states: np.ndarray, timestamps_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    states = np.asarray(hidden_states, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 8 or states.shape[1] < 1:
        raise ValueError("hidden_states must have shape [T, D], with T >= 8 and D >= 1")
    if times.ndim != 1 or times.size != states.shape[0]:
        raise ValueError("timestamps_s must have shape [T] matching hidden_states")
    if not np.isfinite(states).all() or not np.isfinite(times).all():
        raise ValueError("hidden_states and timestamps_s must be finite")
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        raise ValueError("timestamps_s must be strictly increasing")
    interval = float(np.mean(intervals))
    if not np.allclose(intervals, interval, rtol=1e-5, atol=1e-8):
        raise ValueError("FFT requires uniformly sampled timestamps; resample explicitly first")
    return states, times, interval


def _validate_config(config: FeatureConfig, interval: float) -> np.ndarray:
    edges = np.asarray(config.band_edges_hz, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("band_edges_hz must contain at least two finite edges")
    if edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("band_edges_hz must be nonnegative and strictly increasing")
    if edges[-1] > 0.5 / interval + 1e-10:
        raise ValueError("band_edges_hz exceed the Nyquist frequency of the timestamps")
    if not isinstance(config.resample_points, int) or config.resample_points < 8:
        raise ValueError("resample_points must be an integer >= 8")
    if config.resample_points & (config.resample_points - 1):
        raise ValueError("resample_points must be a power of two for three-level Haar analysis")
    if not isinstance(config.stft_window, int) or config.stft_window < 8:
        raise ValueError("stft_window must be an integer >= 8")
    return edges


def _periodogram(states: np.ndarray, interval: float, window: str) -> tuple[np.ndarray, np.ndarray]:
    if window == "hann":
        taper = np.hanning(states.shape[0])
    elif window == "boxcar":
        taper = np.ones(states.shape[0])
    else:
        raise ValueError("window must be 'hann' or 'boxcar'")
    centered = states - states[:1]
    centered -= centered.mean(axis=0, keepdims=True)
    transformed = np.fft.rfft(centered * taper[:, None], axis=0)
    density = np.abs(transformed) ** 2 * interval / np.sum(taper**2)
    if states.shape[0] % 2 == 0:
        density[1:-1] *= 2.0
    else:
        density[1:] *= 2.0
    frequencies = np.fft.rfftfreq(states.shape[0], d=interval)
    return frequencies, density


def power_spectrum(
    hidden_states: np.ndarray, timestamps_s: np.ndarray, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hz and one-sided mean-detrended PSD [frequency, channel].

    PSD units are activation squared per Hz. Its frequency-bin sum times df
    equals tapered centered energy divided by the taper's squared norm.
    The DC bin is returned, including any DC introduced by Hann tapering.
    """
    states, _, interval = _validate_signal(hidden_states, timestamps_s)
    return _periodogram(states, interval, window)


def _spectral_summary(
    states: np.ndarray, interval: float, edges: np.ndarray
) -> np.ndarray:
    frequencies, density = _periodogram(states, interval, "hann")
    density[0] = 0.0
    positive = density[1:]
    total = positive.sum(axis=0)
    valid = total > np.finfo(np.float64).tiny
    fractions = np.divide(positive, total, out=np.zeros_like(positive), where=valid)
    band_features = []
    boundary_tolerance = np.finfo(np.float64).eps * frequencies[-1] * 16
    for lower, upper in zip(edges[:-1], edges[1:]):
        band = (frequencies[1:] > lower + boundary_tolerance) & (
            frequencies[1:] <= upper + boundary_tolerance
        )
        if not band.any():
            raise ValueError(
                f"No positive-frequency FFT bin in ({lower}, {upper}] Hz; "
                "use longer windows or wider bands"
            )
        band_features.append(fractions[band].sum(axis=0))
    logarithms = np.log(np.maximum(fractions, np.finfo(np.float64).tiny))
    entropy = -(fractions * logarithms).sum(axis=0) / np.log(positive.shape[0])
    centroid = (fractions * frequencies[1:, None]).sum(axis=0)
    flatness = np.zeros_like(total)
    if valid.any():
        normalized = fractions[:, valid] * positive.shape[0]
        flatness[valid] = np.exp(np.mean(np.log(np.maximum(normalized, 1e-30)), axis=0))
    energy = np.log1p(total * (frequencies[1] - frequencies[0]))
    return np.concatenate([*band_features, entropy, centroid, flatness, energy])


def _lag_correlation(states: np.ndarray, lag: int) -> np.ndarray:
    previous = states[:-lag] - states[:1]
    following = states[lag:] - states[lag : lag + 1]
    previous -= previous.mean(axis=0)
    following -= following.mean(axis=0)
    denominator = np.sqrt((previous**2).sum(axis=0) * (following**2).sum(axis=0))
    return np.divide(
        (previous * following).sum(axis=0),
        denominator,
        out=np.zeros(states.shape[1]),
        where=denominator > np.finfo(np.float64).tiny,
    )


def _haar_features(ordered: np.ndarray) -> np.ndarray:
    approximation = ordered - ordered[:1]
    approximation -= approximation.mean(axis=0, keepdims=True)
    energies = []
    for _level in range(3):
        detail = (approximation[::2] - approximation[1::2]) / np.sqrt(2.0)
        approximation = (approximation[::2] + approximation[1::2]) / np.sqrt(2.0)
        energies.append(np.mean(detail**2, axis=0))
    return np.log1p(np.concatenate(energies))


def extract_feature_blocks(
    hidden_states: np.ndarray,
    timestamps_s: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, np.ndarray]:
    """Extract fixed-width blocks for a fixed channel count and configuration.

    Semantic features contain means and four activation-norm statistics.
    Temporal features contain channel variance, derivative magnitudes in
    seconds, endpoint changes, and lag-1/2/4 correlation.
    Spectral features contain global Hann PSD summaries, mean/std/end-minus-
    start summaries of local windows, and three Haar detail energies. DC is
    excluded from spectral summaries; constant signals receive zero dynamic
    energy, entropy, centroid, flatness and correlation. Bands are (low, high]
    in Hz and each must have a positive-frequency bin in every window.
    Ordered features use linear interpolation on event-relative time.
    Phase retains real/imaginary rFFT coefficients, including DC, of that
    same resampling. Interior coefficients are multiplied by sqrt(2) so its
    squared norm equals the ordered block's norm. This is an invertible change
    of coordinates, not additional information. Haar scales are relative to
    event duration; they are not absolute frequency bands. Duration varies
    across records, but all returned block dimensions stay fixed.
    """
    states, times, interval = _validate_signal(hidden_states, timestamps_s)
    edges = _validate_config(config, interval)
    norms = np.linalg.norm(states, axis=1)
    semantic = np.concatenate(
        [states.mean(axis=0), [norms.mean(), norms.std(), norms.min(), norms.max()]]
    )
    first_derivative = np.diff(states, axis=0) / interval
    second_derivative = np.diff(states, n=2, axis=0) / interval**2
    temporal = np.concatenate(
        [
            (states - states[:1]).var(axis=0),
            np.mean(np.abs(first_derivative), axis=0),
            np.sqrt(np.mean(first_derivative**2, axis=0)),
            np.mean(np.abs(second_derivative), axis=0),
            states[-1] - states[0],
            *[_lag_correlation(states, lag) for lag in (1, 2, 4)],
        ]
    )
    relative_times = (times - times[0]) / (times[-1] - times[0])
    sample_positions = np.linspace(0.0, 1.0, config.resample_points)
    ordered = np.column_stack(
        [np.interp(sample_positions, relative_times, channel) for channel in states.T]
    )
    global_spectral = _spectral_summary(states, interval, edges)
    window_length = min(config.stft_window, states.shape[0])
    hop = max(1, window_length // 2)
    starts = list(range(0, states.shape[0] - window_length + 1, hop))
    if starts[-1] != states.shape[0] - window_length:
        starts.append(states.shape[0] - window_length)
    local_spectral = np.stack(
        [_spectral_summary(states[start : start + window_length], interval, edges) for start in starts]
    )
    spectral = np.concatenate(
        [
            global_spectral,
            local_spectral.mean(axis=0),
            local_spectral.std(axis=0),
            local_spectral[-1] - local_spectral[0],
            _haar_features(ordered),
        ]
    )
    phase_spectrum = np.fft.rfft(ordered, axis=0, norm="ortho")
    phase_spectrum[1:-1] *= np.sqrt(2.0)
    phase = np.concatenate([phase_spectrum.real.ravel(), phase_spectrum.imag.ravel()])
    blocks = {
        "semantic": semantic,
        "temporal": temporal,
        "spectral": spectral,
        "ordered": ordered.ravel(),
        "phase": phase,
    }
    if not all(np.isfinite(block).all() for block in blocks.values()):
        raise ValueError("Feature extraction overflowed; inspect the activation scale")
    return blocks


def intervene(
    hidden_states: np.ndarray,
    timestamps_s: np.ndarray,
    mode: str,
    band_hz: tuple[float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return a transformed copy; times remain unchanged.

    FFT interventions use a rectangular window and preserve the channel DC.
    Phase randomization preserves each channel's complete power spectrum,
    including real DC/Nyquist coefficients. It does not preserve cross-channel
    coherence. These interventions operate on recorded representations; task
    effects require feeding the result through a specified memory/readout.
    """
    states, _, interval = _validate_signal(hidden_states, timestamps_s)
    if mode not in {"identity", "reverse", "shuffle", "phase_randomize", "remove_band"}:
        raise ValueError(f"Unknown intervention: {mode}")
    if mode != "remove_band" and band_hz is not None:
        raise ValueError("band_hz is only valid for remove_band")
    if mode == "identity":
        return states.copy()
    if mode == "reverse":
        return states[::-1].copy()
    random = np.random.default_rng(seed)
    if mode == "shuffle":
        return states[random.permutation(states.shape[0])].copy()
    spectrum = np.fft.rfft(states, axis=0)
    if mode == "phase_randomize":
        stop = -1 if states.shape[0] % 2 == 0 else None
        interior = spectrum[1:stop]
        interior *= np.exp(1j * random.uniform(-np.pi, np.pi, size=interior.shape))
    else:
        if band_hz is None or len(band_hz) != 2:
            raise ValueError("remove_band requires a (low, high) Hz pair")
        lower, upper = band_hz
        if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper:
            raise ValueError("band_hz must be finite with 0 <= low < high")
        if upper > 0.5 / interval + 1e-10:
            raise ValueError("band_hz exceeds the Nyquist frequency")
        frequencies = np.fft.rfftfreq(states.shape[0], d=interval)
        boundary_tolerance = np.finfo(np.float64).eps * frequencies[-1] * 16
        selected = (frequencies > lower + boundary_tolerance) & (
            frequencies <= upper + boundary_tolerance
        )
        if not selected.any():
            raise ValueError("No non-DC FFT bins lie in the requested intervention band")
        spectrum[selected] = 0.0
    return np.fft.irfft(spectrum, n=states.shape[0], axis=0)
