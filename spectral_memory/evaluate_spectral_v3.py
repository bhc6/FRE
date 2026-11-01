"""CPU-only evaluator for the v3 spectral counterfactual/adapter protocol.

The evaluator consumes JSONL records produced by a v3 forward runner.  It never
opens model weights or video files and refuses records marked as fresh
confirmation data unless ``--allow-confirmation`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SEED = 20261101


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("records JSONL is empty")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every JSONL record must be an object")
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(row: dict[str, Any]) -> str:
    value = row.get("source_id", row.get("video_id"))
    require(value is not None, "record requires source_id or video_id")
    return str(value)


def _method(row: dict[str, Any]) -> str:
    value = row.get("method", row.get("tag"))
    require(value is not None, "record requires method or tag")
    return str(value)


def _query(row: dict[str, Any]) -> str:
    value = row.get("query_id", row.get("case_id"))
    require(value is not None, "record requires query_id or case_id")
    return str(value)


def _supervised(row: dict[str, Any]) -> bool:
    value = row.get("supervised")
    if value is None:
        return row.get("gold_answer_index", row.get("gold_original_option_index")) is not None
    return bool(value)


def _original_prediction(row: dict[str, Any]) -> int:
    if row.get("predicted_original_option_index") is not None:
        return int(row["predicted_original_option_index"])
    prediction = int(row["predicted_answer_index"])
    mapping = row.get("semantic_original_option_indices")
    return int(mapping[prediction]) if mapping is not None else prediction


def _original_gold(row: dict[str, Any]) -> int | None:
    if row.get("original_gold_answer_index") is not None:
        return int(row["original_gold_answer_index"])
    if row.get("gold_original_option_index") is not None:
        return int(row["gold_original_option_index"])
    if row.get("gold_answer_index") is None:
        return None
    gold = int(row["gold_answer_index"])
    mapping = row.get("semantic_original_option_indices")
    if mapping is None:
        return gold
    return int(mapping[gold])


def _presented_gold(row: dict[str, Any]) -> int | None:
    value = row.get("gold_answer_index")
    return None if value is None else int(value)


def _option_logits(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == (3,) and np.isfinite(array).all(), f"{key} must be finite length-3 logits")
    return array


def _nll(logits: np.ndarray, gold: int) -> float:
    shifted = logits - np.max(logits)
    return float(-shifted[gold] + np.log(np.exp(shifted).sum()))


def _validate_rows(rows: list[dict[str, Any]], allow_confirmation: bool) -> None:
    methods = set()
    for row in rows:
        role = row.get("data_role", row.get("split", "development"))
        if not allow_confirmation:
            require(role not in {"fresh_confirmation", "confirmation", "internal_routing_v2"},
                    "fresh confirmation records are sealed; pass --allow-confirmation only after freezing")
        methods.add(_method(row))
        require(_query(row) and _source(row), "source_id/video_id and query_id/case_id cannot be empty")
        supervised = _supervised(row)
        if supervised:
            require(_original_gold(row) is not None, "supervised record requires a gold answer")
        else:
            require(_original_gold(row) is None, "unsupervised/static record cannot carry a gold answer")
        rotation = int(row.get("rotation", 0))
        require(rotation in (0, 1, 2), "rotation must be 0, 1 or 2")
        if row.get("option_count") is not None:
            require(int(row["option_count"]) == 3, "v3 requires three options")
        _option_logits(row, "option_logits")
        _option_logits(row, "baseline_option_logits")
    require(methods, "no methods found")


def _query_rows(rows: list[dict[str, Any]], method: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _method(row) == method and _supervised(row):
            grouped[_query(row)].append(row)
    return grouped


def _query_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    rotations = sorted({int(row.get("rotation", 0)) for row in group})
    require(rotations == [0, 1, 2], f"supervised query {_query(group[0])} must contain rotations 0,1,2")
    source_ids = {_source(row) for row in group}
    require(len(source_ids) == 1, "a query must belong to one source")
    predictions = [_original_prediction(row) for row in group]
    golds = {_original_gold(row) for row in group}
    require(len(golds) == 1, "gold answer changed across option rotations")
    gold = next(iter(golds))
    baseline_predictions = []
    method_nll = []
    baseline_nll = []
    for row in group:
        baseline = row.get("baseline_predicted_original_option_index")
        baseline_predictions.append(int(baseline) if baseline is not None else _original_prediction({**row, "predicted_original_option_index": None, "predicted_answer_index": row.get("baseline_predicted_answer_index", 0)}))
        logits = _option_logits(row, "option_logits")
        base_logits = _option_logits(row, "baseline_option_logits")
        presented_gold = _presented_gold(row)
        if logits is not None and presented_gold is not None:
            method_nll.append(_nll(logits, presented_gold))
        if base_logits is not None and presented_gold is not None:
            baseline_nll.append(_nll(base_logits, presented_gold))
    return {
        "query_id": _query(group[0]),
        "source_id": next(iter(source_ids)),
        "gold": int(gold),
        "correct_mean": float(np.mean([prediction == gold for prediction in predictions])),
        "baseline_correct_mean": float(np.mean([prediction == gold for prediction in baseline_predictions])),
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "semantic_consistent": len(set(predictions)) == 1,
        "baseline_semantic_consistent": len(set(baseline_predictions)) == 1,
        "nll_mean": float(np.mean(method_nll)) if method_nll else None,
        "baseline_nll_mean": float(np.mean(baseline_nll)) if baseline_nll else None,
        "presentations": len(group),
    }


def _bootstrap(values: dict[str, float], repetitions: int, seed: int) -> list[float]:
    source_ids = sorted(values)
    if not source_ids:
        return []
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(source_ids), size=(repetitions, len(source_ids)))
    source_values = np.asarray([values[source] for source in source_ids], dtype=np.float64)
    return np.mean(source_values[sample_indices], axis=1).tolist()


def _ci(values: dict[str, float], repetitions: int, seed: int) -> list[float]:
    samples = _bootstrap(values, repetitions, seed)
    if not samples:
        return [0.0, 0.0]
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _method_metrics(rows: list[dict[str, Any]], method: str, repetitions: int, seed: int) -> dict[str, Any]:
    grouped = _query_rows(rows, method)
    queries = [_query_summary(group) for group in grouped.values()]
    require(queries, f"method {method} has no supervised queries")
    source_accuracy: dict[str, list[float]] = defaultdict(list)
    source_baseline: dict[str, list[float]] = defaultdict(list)
    source_nll: dict[str, list[float]] = defaultdict(list)
    source_baseline_nll: dict[str, list[float]] = defaultdict(list)
    for query in queries:
        source_accuracy[query["source_id"]].append(query["correct_mean"])
        source_baseline[query["source_id"]].append(query["baseline_correct_mean"])
        if query["nll_mean"] is not None:
            source_nll[query["source_id"]].append(query["nll_mean"])
        if query["baseline_nll_mean"] is not None:
            source_baseline_nll[query["source_id"]].append(query["baseline_nll_mean"])
    source_after = {source: float(np.mean(values)) for source, values in source_accuracy.items()}
    source_before = {source: float(np.mean(values)) for source, values in source_baseline.items()}
    delta_accuracy = {source: source_after[source] - source_before[source] for source in source_after}
    source_after_nll = {source: float(np.mean(values)) for source, values in source_nll.items() if values}
    source_before_nll = {source: float(np.mean(values)) for source, values in source_baseline_nll.items() if values}
    delta_nll = {source: source_after_nll[source] - source_before_nll[source] for source in source_after_nll if source in source_before_nll}
    repairs = sum(query["correct_mean"] > query["baseline_correct_mean"] for query in queries)
    regressions = sum(query["correct_mean"] < query["baseline_correct_mean"] for query in queries)
    static = [row for row in rows if _method(row) == method and not _supervised(row)]
    static_flips = sum(int(row.get("baseline_predicted_original_option_index", _original_prediction(row)) != _original_prediction(row)) for row in static)
    return {
        "method": method,
        "sources": len(source_after),
        "queries": len(queries),
        "presentations": sum(query["presentations"] for query in queries),
        "accuracy_micro": float(np.mean([query["correct_mean"] for query in queries])),
        "baseline_accuracy_micro": float(np.mean([query["baseline_correct_mean"] for query in queries])),
        "accuracy_source_macro": float(np.mean(list(source_after.values()))),
        "baseline_accuracy_source_macro": float(np.mean(list(source_before.values()))),
        "delta_accuracy_micro": float(np.mean([query["correct_mean"] - query["baseline_correct_mean"] for query in queries])),
        "delta_accuracy_source_macro": float(np.mean(list(delta_accuracy.values()))),
        "delta_accuracy_source_macro_ci95": _ci(delta_accuracy, repetitions, seed),
        "nll_micro": float(np.mean([query["nll_mean"] for query in queries if query["nll_mean"] is not None])) if source_after_nll else None,
        "baseline_nll_micro": float(np.mean([query["baseline_nll_mean"] for query in queries if query["baseline_nll_mean"] is not None])) if source_before_nll else None,
        "delta_nll_source_macro": float(np.mean(list(delta_nll.values()))) if delta_nll else None,
        "delta_nll_source_macro_ci95": _ci(delta_nll, repetitions, seed) if delta_nll else None,
        "repairs": int(repairs),
        "regressions": int(regressions),
        "semantic_consistent_queries": int(sum(query["semantic_consistent"] for query in queries)),
        "baseline_semantic_consistent_queries": int(sum(query["baseline_semantic_consistent"] for query in queries)),
        "static_count": len(static),
        "static_prediction_flips": int(static_flips),
        "query_summaries": queries,
    }


def acceptance(report: dict[str, Any], role: str) -> dict[str, Any]:
    methods = report["methods"]
    baseline_name = report["baseline_method"]
    baseline = methods[baseline_name]
    gates = {}
    for method, score in methods.items():
        if method == baseline_name:
            continue
        minimum_delta = 0.05 if role == "development" else 0.03
        gates[method] = {
            "diagnostic_or_validation_delta_gate": score["delta_accuracy_source_macro"] >= minimum_delta,
            "micro_non_decrease": score["accuracy_micro"] >= baseline["accuracy_micro"],
            "static_no_flip_gate": score["static_count"] > 0 and score["static_prediction_flips"] == 0,
            "at_least_two_positive_seed_placeholder": True,
            "minimum_source_macro_delta": minimum_delta,
        }
        gates[method]["passed"] = all(value for key, value in gates[method].items() if key.endswith("gate") or key in {"micro_non_decrease", "static_no_flip_gate"})
    return gates


def evaluate(rows: list[dict[str, Any]], baseline_method: str, role: str, repetitions: int, seed: int) -> dict[str, Any]:
    _validate_rows(rows, allow_confirmation=role == "confirmation")
    methods = sorted({_method(row) for row in rows})
    require(baseline_method in methods, f"baseline method {baseline_method!r} absent")
    scores = {method: _method_metrics(rows, method, repetitions, seed + index) for index, method in enumerate(methods)}
    baseline_static = scores[baseline_method]["static_count"]
    require(baseline_static > 0, "v3 evaluation requires at least one static control")
    require(all(score["static_count"] == baseline_static for score in scores.values()),
            "all methods must contain the same number of static controls")
    report = {"status": "complete", "role": role, "baseline_method": baseline_method,
              "bootstrap_repetitions": repetitions, "bootstrap_seed": seed, "methods": scores}
    report["acceptance_gates"] = acceptance(report, role)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-method", default="native")
    parser.add_argument("--role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    require(args.bootstrap_repetitions >= 100, "bootstrap repetitions must be at least 100")
    rows = read_rows(args.records)
    report = evaluate(rows, args.baseline_method, args.role, args.bootstrap_repetitions, args.seed)
    report["records_sha256"] = digest(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "methods": sorted(report["methods"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
