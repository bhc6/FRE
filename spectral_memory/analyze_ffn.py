"""CPU-only spectral audit of saved Qwen FFN and attention branch means."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


BRANCHES = (
    "decoder_input", "attention_delta", "pre_ffn",
    "ffn_input", "ffn_delta", "post_ffn",
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_times(values):
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError("Timestamps must be real")
    times = np.asarray(source, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Need at least two finite timestamps")
    intervals = np.diff(times)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals.mean(), rtol=1e-5, atol=1e-8):
        raise ValueError("Timestamps must be strictly increasing and uniformly spaced")
    return times


def validate_states(values, sample_count, name):
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError(f"{name} must be real")
    states = np.asarray(source, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] != sample_count or states.shape[1] < 1:
        raise ValueError(f"{name} must have shape [T,D]")
    if not np.isfinite(states).all():
        raise ValueError(f"{name} contains nonfinite values")
    return states


def spectral_layout(times):
    sample_count = len(times)
    interval = float(np.diff(times).mean())
    frequencies = np.fft.rfftfreq(sample_count, interval)
    weights = np.full(len(frequencies), 2.0, dtype=np.float64)
    weights[0] = 1.0
    if sample_count % 2 == 0:
        weights[-1] = 1.0
    parts = np.array_split(np.arange(1, len(frequencies)), 3)
    scopes = {"dc": np.asarray([0]), "temporal": np.arange(1, len(frequencies))}
    scopes.update({f"band_{index + 1}": indices for index, indices in enumerate(parts)})
    metadata = {
        "samples": sample_count, "interval_seconds": interval,
        "first_to_last_seconds": float(times[-1] - times[0]),
        "dft_period_seconds": sample_count * interval,
        "bin_spacing_hz": float(1 / (sample_count * interval)),
        "rfft_frequencies_hz": frequencies.tolist(), "one_sided_multiplicity": weights.tolist(),
        "bands": [
            {"name": f"band_{index + 1}", "indices": indices.tolist(),
             "frequencies_hz": frequencies[indices].tolist(), "bin_count": len(indices)}
            for index, indices in enumerate(parts)
        ],
        "band_rule": "np.array_split of ascending non-DC bin indices into three nearly equal-count groups",
        "resolution_warning": (
            "Few temporal samples: band estimates have weak frequency resolution; T=8 gives only four non-DC bins."
            if sample_count <= 8 else
            "Band boundaries depend on T and sample interval; these are descriptive bins, not semantic frequency labels."
        ),
    }
    return weights, scopes, metadata


def transformed(states):
    raw = np.fft.rfft(states, axis=0, norm="ortho")
    centered = np.fft.rfft(states - states.mean(axis=0, keepdims=True), axis=0, norm="ortho")
    centered[0] = raw[0]
    return centered


def inner(first, second, indices, weights, normalization):
    return float(np.sum(weights[indices, None] * np.real(first[indices] * np.conj(second[indices]))) / normalization)


def energy_statistics(states, weights, scopes):
    spectrum = transformed(states)
    normalization = states.size
    energy = {name: inner(spectrum, spectrum, indices, weights, normalization)
              for name, indices in scopes.items()}
    raw_mean_square = float(np.mean(states**2))
    variance = float(np.mean((states - states.mean(axis=0, keepdims=True))**2))
    energy.update({
        "raw_mean_square": raw_mean_square,
        "temporal_variance_direct": variance,
        "parseval_temporal_absolute_error": abs(energy["temporal"] - variance),
        "parseval_raw_absolute_error": abs(energy["dc"] + energy["temporal"] - raw_mean_square),
        "temporal_fraction_of_raw": energy["temporal"] / raw_mean_square if raw_mean_square else None,
        "band_fractions_of_temporal": {
            name: energy[name] / energy["temporal"] if energy["temporal"] else None
            for name in scopes if name.startswith("band_")
        },
    })
    return energy


def spatial_moments(states, values):
    source = np.asarray(values)
    if np.iscomplexobj(source):
        raise ValueError("Spatial second moments must be real")
    moments = np.asarray(source, dtype=np.float64)
    if moments.shape != (len(states),) or not np.isfinite(moments).all() or np.any(moments < 0):
        raise ValueError("Spatial second moments must be finite nonnegative [T]")
    pooled = np.mean(states**2, axis=1)
    variance = moments - pooled
    tolerance = 2e-6 * np.maximum(1.0, np.maximum(moments, pooled))
    if np.any(variance < -tolerance):
        raise ValueError("Spatial second moment is smaller than its mean-square bound")
    total = float(moments.mean())
    return {
        "mean_patch_coordinate_square": total,
        "mean_spatial_mean_coordinate_square": float(pooled.mean()),
        "mean_spatial_variance": float(np.maximum(variance, 0).mean()),
        "unclipped_minimum_spatial_variance": float(variance.min()),
        "spatial_mean_retained_second_moment_fraction": min(1.0, float(pooled.mean()) / total) if total else None,
        "spatial_variance_by_time": variance.tolist(),
        "identifiable": "Instantaneous spatial dispersion and second-moment retention only.",
        "not_identifiable": (
            "Patchwise temporal means are absent; temporal energy removed by spatial pooling, "
            "action information loss, and downstream task utility cannot be recovered from these moments."
        ),
    }


def additive_decomposition(base, increment, observed, weights, scopes):
    rounding = observed - base - increment
    spectra = [transformed(value) for value in (base, increment, rounding, observed)]
    base_spectrum, increment_spectrum, rounding_spectrum, observed_spectrum = spectra
    normalization = base.size
    report = {}
    for name, indices in scopes.items():
        base_energy = inner(base_spectrum, base_spectrum, indices, weights, normalization)
        increment_energy = inner(increment_spectrum, increment_spectrum, indices, weights, normalization)
        rounding_energy = inner(rounding_spectrum, rounding_spectrum, indices, weights, normalization)
        cross_base_increment = 2 * inner(base_spectrum, increment_spectrum, indices, weights, normalization)
        cross_base_rounding = 2 * inner(base_spectrum, rounding_spectrum, indices, weights, normalization)
        cross_increment_rounding = 2 * inner(increment_spectrum, rounding_spectrum, indices, weights, normalization)
        observed_energy = inner(observed_spectrum, observed_spectrum, indices, weights, normalization)
        rounding_contribution = rounding_energy + cross_base_rounding + cross_increment_rounding
        reconstructed = base_energy + increment_energy + cross_base_increment + rounding_contribution
        cosine_denominator = 2 * np.sqrt(base_energy * increment_energy)
        report[name] = {
            "base_energy": base_energy, "increment_energy": increment_energy,
            "base_increment_cross_term": cross_base_increment,
            "rounding_energy": rounding_energy,
            "base_rounding_cross_term": cross_base_rounding,
            "increment_rounding_cross_term": cross_increment_rounding,
            "rounding_total_contribution": rounding_contribution,
            "observed_energy": observed_energy, "reconstructed_energy": reconstructed,
            "closure_absolute_error": abs(observed_energy - reconstructed),
            "observed_minus_base_energy": observed_energy - base_energy,
            "increment_plus_cross_without_rounding": increment_energy + cross_base_increment,
            "base_increment_spectral_cosine": (
                float(np.clip(cross_base_increment / cosine_denominator, -1.0, 1.0))
                if cosine_denominator else None
            ),
        }
    return {
        "scopes": report, "effective_rounding_max_abs": float(np.max(np.abs(rounding))),
        "rounding_definition": (
            "Observed saved mean minus base mean minus increment mean, recomputed in float64. "
            "It includes residual-addition and spatial-reduction rounding; it is not only BF16 quantization."
        ),
    }


def run_self_checks():
    times = np.arange(8, dtype=np.float64) * 0.5
    weights, scopes, metadata = spectral_layout(times)
    checks = []

    def check(name, condition, values=None):
        if not condition:
            raise AssertionError(f"FFN analysis check failed: {name}")
        checks.append({"name": name, "passed": True, "values": values})

    constant = np.full((8, 3), 2.5)
    energy = energy_statistics(constant, weights, scopes)
    check("constant_has_only_dc", energy["temporal"] == 0 and abs(energy["dc"] - 6.25) < 1e-12)
    sine = np.sin(2 * np.pi * np.arange(8) / 8)[:, None]
    energy = energy_statistics(sine, weights, scopes)
    check("single_sine_parseval_and_bin", abs(energy["temporal"] - 0.5) < 1e-12
          and abs(energy["band_1"] - 0.5) < 1e-12 and energy["band_2"] < 1e-25
          and energy["band_3"] < 1e-25)
    anti = -0.75 * sine
    decomposition = additive_decomposition(sine, anti, sine + anti, weights, scopes)["scopes"]["temporal"]
    check("anti_increment_cross_term", abs(decomposition["observed_energy"] - 0.5 / 16) < 1e-12
          and abs(decomposition["base_increment_cross_term"] + 0.75) < 1e-12
          and decomposition["closure_absolute_error"] < 1e-12)
    rng = np.random.default_rng(17)
    discrepancies = []
    for sample_count in (2, 7, 8, 31, 32):
        local_times = np.arange(sample_count, dtype=np.float64) * 0.3
        local_weights, local_scopes, _ = spectral_layout(local_times)
        states = rng.normal(size=(sample_count, 5))
        energy = energy_statistics(states, local_weights, local_scopes)
        discrepancies.extend([energy["parseval_temporal_absolute_error"], energy["parseval_raw_absolute_error"]])
        check(f"parseval_T{sample_count}", max(discrepancies) < 1e-12
              and abs(sum(energy[f"band_{index}"] for index in (1, 2, 3)) - energy["temporal"]) < 1e-12)
    base = rng.normal(size=(8, 4)).astype(np.float16).astype(np.float64)
    increment = (0.07 * rng.normal(size=(8, 4))).astype(np.float16).astype(np.float64)
    observed = (base + increment).astype(np.float16).astype(np.float64)
    decomposition = additive_decomposition(base, increment, observed, weights, scopes)
    check("rounded_residual_decomposition",
          max(value["closure_absolute_error"] for value in decomposition["scopes"].values()) < 1e-12
          and decomposition["effective_rounding_max_abs"] > 0)
    tokens = np.stack([sine, -sine], axis=1)
    moments = np.mean(tokens**2, axis=(1, 2))
    spatial = spatial_moments(tokens.mean(axis=1), moments)
    check("opposite_spatial_signs_cancel_mean", spatial["spatial_mean_retained_second_moment_fraction"] == 0
          and abs(spatial["mean_spatial_variance"] - 0.5) < 1e-12)
    invalid_calls = (
        lambda: validate_times([0, 0]),
        lambda: validate_times([0, 1, 3]),
        lambda: validate_times([0, np.nan]),
        lambda: validate_times([0 + 0j, 1 + 1j]),
        lambda: validate_states(np.zeros((7, 2)), 8, "invalid"),
        lambda: validate_states(np.full((8, 2), np.nan), 8, "invalid"),
        lambda: validate_states(np.zeros((8, 2), dtype=complex), 8, "invalid"),
        lambda: spatial_moments(np.ones((8, 2)), np.zeros(8)),
        lambda: spatial_moments(np.zeros((8, 2)), np.full(8, -1)),
    )
    for index, call in enumerate(invalid_calls):
        rejected = False
        try:
            call()
        except ValueError:
            rejected = True
        check(f"invalid_input_{index + 1}_rejected", rejected)
    return {
        "status": "passed", "checks": checks, "check_count": len(checks),
        "max_parseval_absolute_error": max(discrepancies),
        "eight_sample_band_indices": [value["indices"] for value in metadata["bands"]],
        "scope": "Independent synthetic numerical checks, no torch, model loading, GPU, or task-performance claims.",
    }


def analyze(input_path):
    with np.load(input_path, allow_pickle=False) as archive:
        times = validate_times(archive["timestamps_s"])
        weights, scopes, layout = spectral_layout(times)
        layer_ids = sorted({int(match.group(1)) for name in archive.files
                            if (match := re.fullmatch(r"layer_(\d+)_decoder_input", name))})
        if not layer_ids:
            raise ValueError("No layer_{idx}_decoder_input arrays found")
        layers = {}
        for layer in layer_ids:
            branch_arrays = {}
            branch_report = {}
            for branch in BRANCHES:
                key = f"layer_{layer}_{branch}"
                values = validate_states(archive[key], len(times), key)
                branch_arrays[branch] = values
                branch_report[branch] = {
                    "shape": list(values.shape),
                    "energy": energy_statistics(values, weights, scopes),
                    "spatial_moments": spatial_moments(values, archive[key + "_spatial_mean_square"]),
                }
            if len({value.shape for value in branch_arrays.values()}) != 1:
                raise ValueError("All six branches of a layer must share shape")
            saved_rounding = validate_states(archive[f"layer_{layer}_ffn_rounding"], len(times), "ffn_rounding")
            effective = branch_arrays["post_ffn"] - branch_arrays["pre_ffn"] - branch_arrays["ffn_delta"]
            if saved_rounding.shape != effective.shape:
                raise ValueError("FFN rounding shape mismatch")
            maximum = float(np.max(np.abs(effective - saved_rounding)))
            scale = max(1.0, *(float(np.max(np.abs(branch_arrays[name])))
                               for name in ("pre_ffn", "ffn_delta", "post_ffn")))
            tolerance = 2e-6 * scale
            if maximum > tolerance:
                raise ValueError("Saved FFN rounding does not match the residual identity")
            ffn = additive_decomposition(branch_arrays["pre_ffn"], branch_arrays["ffn_delta"],
                                         branch_arrays["post_ffn"], weights, scopes)
            attention = additive_decomposition(branch_arrays["decoder_input"], branch_arrays["attention_delta"],
                                               branch_arrays["pre_ffn"], weights, scopes)
            layers[str(layer)] = {
                "branches": branch_report, "ffn_decomposition": ffn, "attention_decomposition": attention,
                "saved_rounding_validation": {
                    "max_absolute_discrepancy": maximum, "tolerance": tolerance,
                    "saved_rounding_energy": energy_statistics(saved_rounding, weights, scopes),
                },
            }
    return {
        "status": "complete", "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layout": layout, "layers": layers,
        "energy_units": "Squared state units, averaged over T temporal samples and D coordinates; DC plus temporal equals raw mean square.",
        "transform": "Temporal mean-centered orthonormal rFFT for non-DC bins; raw orthonormal rFFT coefficient for DC; one-sided multiplicity.",
        "decomposition": "E(post)=E(base)+E(delta)+2Re<base,delta>+E(rounding)+2Re<base,rounding>+2Re<delta,rounding>.",
        "limitations": [
            "Descriptive representation audit only: no task benefit, memory utility, or causal FFN contribution is established.",
            "FFN input is normalized pre-FFN residual; comparing its energy directly with the residual confounds normalization.",
            "Spatial second moments do not identify temporal dynamics removed by spatial averaging.",
            "post_ffn must be captured before DeepStack addition; next-layer input is not a substitute.",
            "FFN and attention increments may reinforce or cancel their base through cross terms; increment energy is not explained task variance.",
            "Short trajectories have weak spectral resolution; frequency bands do not automatically denote semantic timescales.",
        ],
    }


def render_report(report):
    layout = report["layout"]
    lines = [
        "# FFN 与注意力分支的频谱诊断", "",
        "这是保存张量的 CPU 描述性分析，不评估问答收益，也不证明 FFN 对记忆有效。", "",
        f"时间位置数：{layout['samples']}；间隔：{layout['interval_seconds']:.6g} 秒；"
        f"DFT 频率分辨率：{layout['bin_spacing_hz']:.6g} Hz。", "",
        "非 DC 频率索引按顺序分成三个近等数量组；T=8 时只有 4 个非 DC bin，频带解释很弱。"
        "这里不是 v1 主确认的固定 cycles/sample 频带，不能合并比较。", "",
        "所有能量均为 T×D 归一化的均方值。DC 表示每通道时间均值的平方；时间能量表示去时间均值后的方差。", "",
        "| 频带 | bin 索引 | Hz |", "| --- | --- | --- |",
    ]
    for band in layout["bands"]:
        lines.append(f"| {band['name']} | {band['indices']} | {band['frequencies_hz']} |")
    for layer, values in report["layers"].items():
        lines.extend(["", f"## Decoder 层 {layer}", "",
                      "| 分支 | DC | 时间能量 | band 1 | band 2 | band 3 | 空间均值二阶矩保留比例 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for branch, value in values["branches"].items():
            energy = value["energy"]
            retained = value["spatial_moments"]["spatial_mean_retained_second_moment_fraction"]
            retention_text = f"{retained:.4f}" if retained is not None else "undefined (zero signal)"
            lines.append(f"| {branch} | {energy['dc']:.6g} | {energy['temporal']:.6g} | "
                         f"{energy['band_1']:.6g} | {energy['band_2']:.6g} | {energy['band_3']:.6g} | {retention_text} |")
        lines.extend(["", "| 残差路径/频段 | 输入能量 | 增量能量 | 输入×增量交叉项 | 舍入总贡献 | 输出能量 | 恒等式误差 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for name, decomposition in (("FFN", values["ffn_decomposition"]),
                                     ("Attention", values["attention_decomposition"])):
            for scope, value in decomposition["scopes"].items():
                lines.append(f"| {name}/{scope} | {value['base_energy']:.6g} | {value['increment_energy']:.6g} | "
                             f"{value['base_increment_cross_term']:.6g} | {value['rounding_total_contribution']:.6g} | "
                             f"{value['observed_energy']:.6g} | {value['closure_absolute_error']:.3g} |")
    lines.extend([
        "", "## 解释边界", "",
        "- 能量上升或下降不等于任务改善；增量和残差的交叉项可以增强或抵消动态。",
        "- 舍入贡献包括观测到的残差加法与空间归约数值误差；它可带符号，不能只计舍入项自身能量。",
        "- 空间二阶矩仅识别瞬时空间离散程度。未保存每个 patch 的时间均值，无法恢复空间平均删除的时间动态。",
        "- FFN 的输入是归一化状态；归一化前后能量不同不能直接归因于 FFN 学习了什么。",
        "- DeepStack 在 decoder 层返回后额外加入视觉特征，post_ffn 必须在这之前捕获。",
        "- 合成数值检查只验证公式和输入约束；结果见 checks.json。",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--self-check-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.self_check_only and arguments.input is None:
        parser.error("--input is required unless --self-check-only is used")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checks = run_self_checks()
    write_json(arguments.output_dir / "checks.json", checks)
    if arguments.self_check_only:
        print(json.dumps({"status": checks["status"], "checks": checks["check_count"]}))
        return
    report = analyze(arguments.input)
    write_json(arguments.output_dir / "report.json", report)
    (arguments.output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "layers": list(report["layers"]),
                      "samples": report["layout"]["samples"], "output": str(arguments.output_dir.resolve())}))


if __name__ == "__main__":
    main()
