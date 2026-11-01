"""Training-only preprocessing, held-out probes, and additive budget diagnostics."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import rankdata

from .data import load_samples, split_samples, write_jsonl
from .features import FeatureConfig, extract_feature_blocks


FEATURE_SETS = {
    "M1_semantic": ("semantic",),
    "M2_semantic_temporal": ("semantic", "temporal"),
    "M3_semantic_spectral": ("semantic", "spectral"),
    "M4_all": ("semantic", "temporal", "spectral"),
    "M5_ordered_control": ("semantic", "temporal", "ordered"),
    "M6_complex_control": ("semantic", "temporal", "phase"),
}


def fit_ridge(features, targets, alpha):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-10] = 1.0
    normalized = (features - mean) / scale
    intercept = float(targets.mean())
    centered_targets = targets - intercept
    row_count, column_count = normalized.shape
    if column_count <= row_count:
        gram = normalized.T @ normalized + alpha * np.eye(column_count)
        weights = np.linalg.solve(gram, normalized.T @ centered_targets)
    else:
        gram = normalized @ normalized.T + alpha * np.eye(row_count)
        weights = normalized.T @ np.linalg.solve(gram, centered_targets)
    return {"mean": mean, "scale": scale, "weights": weights, "intercept": intercept}


def predict_ridge(model, features):
    return ((features - model["mean"]) / model["scale"]) @ model["weights"] + model["intercept"]


def correlation(targets, predictions):
    if len(targets) < 2 or np.ptp(targets) == 0 or np.ptp(predictions) == 0:
        return None
    return float(np.corrcoef(rankdata(targets), rankdata(predictions))[0, 1])


def metrics(targets, predictions):
    return {"mse": float(np.mean((targets - predictions) ** 2)),
            "mae": float(np.mean(np.abs(targets - predictions))),
            "spearman": correlation(targets, predictions)}


def knapsack(scores, costs, budget):
    """Maximize additive predicted scores with integer costs; negative items are skipped."""
    if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 100000:
        raise ValueError("Budget must be an integer in [0,100000]")
    scores = np.asarray(scores, dtype=float)
    costs_array = np.asarray(costs)
    if scores.ndim != 1 or costs_array.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid knapsack vectors")
    if not np.issubdtype(costs_array.dtype, np.integer) or np.any(costs_array <= 0):
        raise ValueError("Knapsack costs must be positive integers")
    best = np.zeros(budget + 1)
    paths = [tuple() for _ in range(budget + 1)]
    for item_index, (score, cost) in enumerate(zip(scores, costs_array)):
        if score <= 0 or cost > budget:
            continue
        for capacity in range(budget, int(cost) - 1, -1):
            candidate = best[capacity - cost] + score
            if candidate > best[capacity]:
                best[capacity] = candidate
                paths[capacity] = paths[capacity - cost] + (item_index,)
    return list(paths[budget])


def group_diagnostics(samples, predictions, budget, random_seed=17):
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(sample.metadata["evaluation_group"], []).append(index)
    rows = []
    generator = np.random.default_rng(random_seed)
    for group, indices in sorted(groups.items()):
        targets = np.array([samples[index].metadata["utility"] for index in indices])
        costs = np.array([samples[index].metadata["cost"] for index in indices])
        scores = predictions[indices]
        selected = knapsack(scores, costs, budget)
        oracle = knapsack(targets, costs, budget)
        order = np.argsort(-scores, kind="stable")
        discount = 1 / np.log2(np.arange(len(indices)) + 2)
        gains = np.maximum(targets, 0)
        ideal = float(np.sort(gains)[::-1] @ discount)
        baseline_selections = {}
        for name, ordering in {
            "random": generator.permutation(len(indices)),
            "recency": np.argsort([-samples[index].metadata["event_end_s"] for index in indices], kind="stable"),
        }.items():
            remaining = budget
            chosen = []
            for candidate in ordering:
                if costs[candidate] <= remaining:
                    chosen.append(int(candidate))
                    remaining -= costs[candidate]
            baseline_selections[name] = chosen
        row = {
            "evaluation_group": group,
            "query_id": samples[indices[0]].metadata["query_id"],
            "memory_state_id": samples[indices[0]].metadata["memory_state_id"],
            "linear_ndcg": float(gains[order] @ discount / ideal) if ideal > 0 else None,
            "spearman": correlation(targets, scores),
            "selected_ids": [samples[indices[index]].metadata["sample_id"] for index in selected],
            "selected_cost": int(costs[selected].sum()),
            "budget": budget,
            "sum_individual_utility_proxy": float(targets[selected].sum()),
            "additive_oracle_proxy": float(targets[oracle].sum()),
        }
        for name, selected_baseline in baseline_selections.items():
            row[f"{name}_sum_utility_proxy"] = float(targets[selected_baseline].sum())
        rows.append(row)
    return rows


def paired_bootstrap(targets, baseline, augmented, group_ids, seed=17, draws=1000):
    """Equal-group paired MSE improvement; independent groups are the resampling units."""
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    unique_groups = sorted(set(group_ids))
    differences = np.array([
        np.mean((targets[group_ids == group] - baseline[group_ids == group]) ** 2
                - (targets[group_ids == group] - augmented[group_ids == group]) ** 2)
        for group in unique_groups
    ])
    result = {"mean_group_mse_improvement": float(differences.mean()),
              "independent_test_groups": len(unique_groups), "draws": draws,
              "ci95": None, "warning": None}
    if len(unique_groups) < 2:
        result["warning"] = "Fewer than two independent groups; no uncertainty interval"
        return result
    generator = np.random.default_rng(seed)
    bootstrap_means = np.array([generator.choice(differences, len(differences), replace=True).mean()
                                for _ in range(draws)])
    result["ci95"] = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
    if len(unique_groups) < 10:
        result["warning"] = "Few independent groups; interval is exploratory"
    return result


def run_experiment(manifest, output_dir, config=None, task_mode="retrieve", group_key="subject_id",
                   seed=17, budget=8, alphas=(0.1, 1.0, 10.0, 100.0), bootstrap_draws=1000,
                   projection_dim=None):
    config = config or FeatureConfig()
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge penalties must be finite and positive")
    if projection_dim is not None and projection_dim < 1:
        raise ValueError("projection_dim must be positive")
    samples = load_samples(manifest, task_mode)
    assignments, masks = split_samples(samples, group_key, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("Output already contains a run; choose a fresh output directory")
    blocks = [extract_feature_blocks(sample.hidden_states, sample.timestamps_s, config) for sample in samples]
    contexts = np.array([sample.metadata.get("context", []) for sample in samples], dtype=float)
    targets = np.array([sample.metadata["utility"] for sample in samples], dtype=float)
    test_samples = [sample for sample, keep in zip(samples, masks["test"]) if keep]
    test_targets = targets[masks["test"]]
    test_groups = np.array([sample.metadata[group_key] for sample in test_samples])
    report = {
        "status": "synthetic_smoke_test" if all(sample.metadata.get("synthetic") for sample in samples) else "offline_utility_probe",
        "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "task_mode": task_mode, "group_key": group_key, "seed": seed, "budget": budget,
        "cost_unit": samples[0].metadata["cost_unit"], "feature_config": asdict(config),
        "split_counts": {key: int(mask.sum()) for key, mask in masks.items()},
        "projection_dim": projection_dim,
        "models": {}, "comparisons": {},
        "limitations": [
            "Synthetic runs test plumbing only; their labels are independent of hidden-state features.",
            "Summed individual utility is a proxy: selected sets require fresh downstream evaluation.",
            "Ridge probes cannot replace matched nonlinear temporal baselines or causal task interventions.",
            "Intervals use equal-weight independent split groups, not individual event rows.",
            "Spectral features include ordered local STFT summaries; they are not power-only features.",
        ],
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    write_jsonl(output / "split_manifest.jsonl", [dict(sample.metadata, split=str(split), content_hash=sample.content_hash)
                                               for sample, split in zip(samples, assignments)])
    predictions_by_model = {}
    prediction_records = []
    for model_number, (name, feature_names) in enumerate(FEATURE_SETS.items()):
        features = np.stack([np.concatenate([block[key] for key in feature_names]) for block in blocks])
        raw_width = features.shape[1]
        projection_state = {}
        if projection_dim is not None:
            projection_mean = features[masks["train"]].mean(axis=0)
            projection_scale = features[masks["train"]].std(axis=0)
            projection_scale[projection_scale < 1e-10] = 1.0
            projection = np.random.default_rng(seed + model_number).normal(
                size=(raw_width, projection_dim)) / np.sqrt(projection_dim)
            features = ((features - projection_mean) / projection_scale) @ projection
            projection_state = {"projection": projection, "projection_mean": projection_mean,
                                "projection_scale": projection_scale}
        features = np.concatenate((features, contexts), axis=1)
        candidates = []
        for alpha in alphas:
            model = fit_ridge(features[masks["train"]], targets[masks["train"]], alpha)
            validation_predictions = predict_ridge(model, features[masks["validation"]])
            validation_mse = float(np.mean((targets[masks["validation"]] - validation_predictions) ** 2))
            candidates.append((validation_mse, float(alpha), model))
        validation_mse, alpha, model = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        predictions = predict_ridge(model, features[masks["test"]])
        if not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite predictions in {name}")
        predictions_by_model[name] = predictions
        diagnostics = group_diagnostics(test_samples, predictions, budget, seed)
        valid_ndcg = [row["linear_ndcg"] for row in diagnostics if row["linear_ndcg"] is not None]
        report["models"][name] = {
            "raw_feature_width": raw_width, "fitted_feature_width": features.shape[1],
            "alpha": alpha, "validation_mse": validation_mse,
            "test": metrics(test_targets, predictions),
            "mean_group_linear_ndcg": float(np.mean(valid_ndcg)) if valid_ndcg else None,
            "mean_sum_individual_utility_proxy": float(np.mean([row["sum_individual_utility_proxy"] for row in diagnostics])),
        }
        np.savez_compressed(output / f"{name}.npz", **model, **projection_state)
        write_jsonl(output / f"{name}.selections.jsonl", diagnostics)
        for sample, prediction in zip(test_samples, predictions):
            prediction_records.append({"sample_id": sample.metadata["sample_id"], "model": name,
                                       "utility": sample.metadata["utility"], "prediction": float(prediction),
                                       "evaluation_group": sample.metadata["evaluation_group"]})
    for baseline, augmented in (("M1_semantic", "M3_semantic_spectral"), ("M2_semantic_temporal", "M4_all"),
                                ("M5_ordered_control", "M4_all"), ("M6_complex_control", "M4_all")):
        report["comparisons"][f"{augmented}_vs_{baseline}"] = paired_bootstrap(
            test_targets, predictions_by_model[baseline], predictions_by_model[augmented],
            test_groups, seed, bootstrap_draws)
    write_jsonl(output / "predictions.jsonl", prediction_records)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Spectral Memory Probe", "", f"Run type: `{report['status']}`", "",
             "This is an offline probe, not an end-to-end memory evaluation.", "",
             "| Model | Features | Test MSE | Spearman | Group NDCG |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in report["models"].items():
        rho = result["test"]["spearman"]
        ndcg = result["mean_group_linear_ndcg"]
        rho_text = "undefined" if rho is None else f"{rho:.4f}"
        ndcg_text = "undefined" if ndcg is None else f"{ndcg:.4f}"
        lines.append(f"| {name} | {result['fitted_feature_width']} | {result['test']['mse']:.6f} | {rho_text} | {ndcg_text} |")
    lines.extend(["", "## Limits", ""] + [f"- {limitation}" for limitation in report["limitations"]])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
