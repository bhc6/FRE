"""Synthetic CPU checks for evaluate_spectral_v3."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

try:
    from .evaluate_spectral_v3 import evaluate
except ImportError:  # pragma: no cover - supports direct repository invocation
    from evaluate_spectral_v3 import evaluate


def make_rows():
    rows = []
    for source_index in range(2):
        source = f"source_{source_index}"
        for query_index in range(2):
            query = f"{source}:q{query_index}"
            gold = (source_index + query_index) % 3
            for rotation in range(3):
                presented_gold = (gold - rotation) % 3
                mapping = [(index + rotation) % 3 for index in range(3)]
                native_presented = (gold + rotation) % 3
                method_presented = presented_gold
                base_logits = [0.0, 0.0, 0.0]
                base_logits[native_presented] = 1.0
                method_logits = [0.0, 0.0, 0.0]
                method_logits[method_presented] = 1.0
                rows.append({
                    "method": "native", "source_id": source, "query_id": query,
                    "rotation": rotation, "supervised": True,
                    "gold_answer_index": presented_gold,
                    "original_gold_answer_index": gold,
                    "semantic_original_option_indices": mapping,
                    "predicted_answer_index": native_presented,
                    "predicted_original_option_index": mapping[native_presented],
                    "baseline_predicted_original_option_index": mapping[native_presented],
                    "option_logits": base_logits,
                    "baseline_option_logits": base_logits,
                })
                rows.append({
                    "method": "spectral", "source_id": source, "query_id": query,
                    "rotation": rotation, "supervised": True,
                    "gold_answer_index": presented_gold,
                    "original_gold_answer_index": gold,
                    "semantic_original_option_indices": mapping,
                    "predicted_answer_index": method_presented,
                    "predicted_original_option_index": mapping[method_presented],
                    "baseline_predicted_original_option_index": mapping[native_presented],
                    "option_logits": method_logits,
                    "baseline_option_logits": base_logits,
                })
        rows.append({
            "method": "native", "source_id": source, "query_id": f"{source}:static",
            "rotation": 0, "supervised": False,
            "predicted_answer_index": 0, "predicted_original_option_index": 0,
            "baseline_predicted_original_option_index": 0,
        })
        rows.append({
            "method": "spectral", "source_id": source, "query_id": f"{source}:static",
            "rotation": 0, "supervised": False,
            "predicted_answer_index": 0, "predicted_original_option_index": 0,
            "baseline_predicted_original_option_index": 0,
        })
    return rows


def main():
    rows = make_rows()
    report = evaluate(rows, "native", "development", repetitions=200, seed=20261101)
    assert report["status"] == "complete"
    assert report["methods"]["spectral"]["accuracy_source_macro"] == 1.0
    assert report["methods"]["spectral"]["static_prediction_flips"] == 0
    assert report["acceptance_gates"]["spectral"]["passed"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "records.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        sealed = dict(rows[0])
        sealed["data_role"] = "fresh_confirmation"
        try:
            evaluate([sealed], "native", "development", repetitions=100, seed=1)
        except ValueError as error:
            assert "sealed" in str(error)
        else:
            raise AssertionError("fresh confirmation record was not rejected")
    print("spectral evaluation v3 checks passed: 4")


if __name__ == "__main__":
    main()
