"""Join paired evaluator logs without calling a model or inventing task rewards."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import read_jsonl, write_jsonl


def attach_utilities(manifest, evaluations, output):
    records = read_jsonl(manifest)
    rows = read_jsonl(evaluations)
    identities = {record["sample_id"]: record for record in records}
    if len(identities) != len(records):
        raise ValueError("Duplicate sample IDs in manifest")
    for record in records:
        if any(not isinstance(record.get(field), str) or not record[field]
               for field in ("candidate_hash", "base_memory_hash")):
            raise ValueError("Manifest must bind candidate_hash and base_memory_hash to evaluated payloads")
    pairs = defaultdict(dict)
    protocol_identity = None
    protocol_fields = ("evaluator_id", "evaluator_revision", "prompt_hash", "decoding_hash", "reward_metric")
    required = ("sample_id", "query_id", "memory_state_id", "evaluator_id", "evaluator_revision",
                "prompt_hash", "decoding_hash", "reward_metric", "base_memory_hash", "candidate_hash")
    for row in rows:
        if any(not isinstance(row.get(field), str) or not row[field] for field in required):
            raise ValueError("Evaluator rows need complete provenance strings")
        if row["sample_id"] not in identities:
            raise ValueError("Evaluator log references an unknown sample")
        sample = identities[row["sample_id"]]
        for field in ("candidate_hash", "base_memory_hash"):
            if row[field] != sample[field]:
                raise ValueError(f"Evaluation does not match manifest {field}")
        identity = tuple(row[field] for field in protocol_fields)
        if protocol_identity is not None and identity != protocol_identity:
            raise ValueError("Do not aggregate different evaluator/reward/prompt/decoding protocols")
        protocol_identity = identity
        if row.get("condition") not in {"with", "without"}:
            raise ValueError("Evaluator condition must be with or without")
        if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
            raise ValueError("Every evaluation needs an explicit integer seed")
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not np.isfinite(reward):
            raise ValueError("Rewards must be finite numbers")
        if row.get("candidate_in_base") is not False:
            raise ValueError("Must verify candidate_in_base=false for addition utility")
        if row.get("query_after_event") is not True:
            raise ValueError("Must verify query_after_event=true")
        key = (row["sample_id"], row["query_id"], row["memory_state_id"], row["seed"])
        if row["condition"] in pairs[key]:
            raise ValueError("Duplicate evaluation condition/seed")
        pairs[key][row["condition"]] = row
    per_sample = defaultdict(lambda: defaultdict(list))
    for key, pair in pairs.items():
        if set(pair) != {"with", "without"}:
            raise ValueError(f"Unpaired counterfactual evaluation: {key}")
        included, excluded = pair["with"], pair["without"]
        for field in required:
            if included[field] != excluded[field]:
                raise ValueError(f"Paired provenance differs: {field}")
        sample = identities[key[0]]
        if included["memory_state_id"] != sample["memory_state_id"]:
            raise ValueError("Evaluation uses the wrong base memory state")
        if sample.get("task_mode", "retrieve") == "write":
            if sample.get("query_id") != "__expected__" or sample.get("label_scope") != "expected_future_queries":
                raise ValueError("Write labels must aggregate future queries")
        elif included["query_id"] != sample["query_id"]:
            raise ValueError("Retrieval label uses the wrong query")
        per_sample[key[0]][key[1]].append(float(included["reward"] - excluded["reward"]))
    updated = []
    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in per_sample:
            raise ValueError(f"Missing paired labels for {sample_id}")
        query_gains = per_sample[sample_id]
        expected = record.get("label_query_ids")
        if record.get("task_mode", "retrieve") == "write":
            if not expected or len(set(expected)) != len(expected) or set(expected) != set(query_gains):
                raise ValueError("Write rows must predeclare exactly the evaluated label_query_ids")
        gains = [float(np.mean(values)) for values in query_gains.values()]
        feature_path = Path(record["hidden_path"])
        if not feature_path.is_absolute():
            feature_path = Path(manifest).resolve().parent / feature_path
        if record.get("memory_payload_path") is not None:
            payload_path = Path(record["memory_payload_path"])
            if not payload_path.is_absolute():
                payload_path = Path(manifest).resolve().parent / payload_path
            record = dict(record, memory_payload_path=str(payload_path.resolve()))
        updated.append(dict(record, hidden_path=str(feature_path.resolve()), utility=float(np.mean(gains)),
                            label_query_count=len(gains), label_pair_count=sum(map(len, query_gains.values())),
                            label_source="paired_evaluator_logs", label_aggregation="mean_of_query_seed_means",
                            utility_protocol=dict(zip(protocol_fields, protocol_identity)),
                            intervention_requires_new_task_evaluation=False))
    target = Path(output).resolve()
    if target in {Path(manifest).resolve(), Path(evaluations).resolve()} or target.exists():
        raise ValueError("Label output must be a new file")
    write_jsonl(target, updated)
    return len(updated)
