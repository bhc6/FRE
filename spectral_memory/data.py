"""Read event records and enforce split boundaries before feature fitting."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Sample:
    metadata: dict
    hidden_states: np.ndarray
    timestamps_s: np.ndarray
    content_hash: str


def read_jsonl(path):
    source = Path(path)
    records = []
    with source.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{source}: empty manifest")
    return records


def write_jsonl(path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def load_samples(manifest, task_mode="retrieve", require_utility=True):
    if task_mode not in {"retrieve", "write"}:
        raise ValueError("task_mode must be retrieve or write")
    source = Path(manifest).resolve()
    records = read_jsonl(source)
    samples = []
    seen_ids = set()
    required = ("sample_id", "video_id", "subject_id", "session_id", "query_id",
                "memory_state_id", "evaluation_group", "hidden_path", "cost_unit")
    for record in records:
        for key in required:
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Missing nonempty string: {key}")
        sample_id = record["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        cost = record.get("cost")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
            raise ValueError(f"{sample_id}: cost must be a positive integer")
        if record.get("task_mode", task_mode) != task_mode:
            raise ValueError(f"{sample_id}: mixed task modes")
        context = np.asarray(record.get("context", []), dtype=np.float64)
        if context.ndim != 1 or not np.isfinite(context).all():
            raise ValueError(f"{sample_id}: context must be a finite vector")
        if task_mode == "write":
            if context.size and record.get("context_source") != "write_observable":
                raise ValueError(f"{sample_id}: write context must be marked write_observable")
            if record.get("label_scope") != "expected_future_queries":
                raise ValueError(f"{sample_id}: write labels require expected_future_queries")
            if record.get("query_id") != "__expected__":
                raise ValueError(f"{sample_id}: per-query rows are retrieval, not query-free writing")
            future_queries = record.get("label_query_ids")
            if not isinstance(future_queries, list) or not future_queries or not all(
                isinstance(query, str) and query for query in future_queries
            ) or len(set(future_queries)) != len(future_queries):
                raise ValueError(f"{sample_id}: predeclare a unique nonempty list of label_query_ids")
        if require_utility:
            if record.get("intervention_requires_new_task_evaluation"):
                raise ValueError(f"{sample_id}: intervention still requires new task evaluations")
            utility = record.get("utility")
            if isinstance(utility, bool) or not isinstance(utility, (int, float)) or not np.isfinite(utility):
                raise ValueError(f"{sample_id}: utility must be finite")
        feature_path = Path(record["hidden_path"])
        if not feature_path.is_absolute():
            feature_path = source.parent / feature_path
        if record.get("memory_payload_path") is not None:
            payload_path = Path(record["memory_payload_path"])
            if not payload_path.is_absolute():
                payload_path = source.parent / payload_path
            payload_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if payload_hash != record.get("candidate_hash"):
                raise ValueError(f"{sample_id}: memory payload differs from candidate_hash")
            record["memory_payload_path"] = str(payload_path.resolve())
        with np.load(feature_path, allow_pickle=False) as archive:
            hidden = np.asarray(archive["hidden_states"], dtype=np.float64)
            timestamps = np.asarray(archive["timestamps_s"], dtype=np.float64)
        if hidden.ndim != 2 or hidden.shape[0] < 8 or hidden.shape[1] < 1:
            raise ValueError(f"{sample_id}: hidden_states must be [T>=8,D>=1]")
        if timestamps.shape != (hidden.shape[0],) or not np.isfinite(hidden).all() or not np.isfinite(timestamps).all():
            raise ValueError(f"{sample_id}: malformed or nonfinite hidden states/timestamps")
        steps = np.diff(timestamps)
        if np.any(steps <= 0) or not np.allclose(steps, steps.mean(), rtol=1e-4, atol=1e-8):
            raise ValueError(f"{sample_id}: timestamps must be uniformly increasing")
        start = record.get("event_start_s")
        end = record.get("event_end_s")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"{sample_id}: event_start_s and event_end_s are required numbers")
        if not np.isfinite([start, end]).all() or not start <= timestamps[0] < timestamps[-1] <= end:
            raise ValueError(f"{sample_id}: absolute timestamps must lie within event bounds")
        digest = hashlib.sha256(np.ascontiguousarray(hidden).tobytes()).hexdigest()
        samples.append(Sample(dict(record, hidden_path=str(feature_path.resolve())), hidden, timestamps, digest))
    dimensions = {(sample.hidden_states.shape[1], len(sample.metadata.get("context", []))) for sample in samples}
    if len(dimensions) != 1:
        raise ValueError("Do not mix hidden dimensions or context dimensions in one experiment")
    if len({sample.metadata["cost_unit"] for sample in samples}) != 1:
        raise ValueError("All memory costs must use the same unit")
    for field in ("model_id", "model_revision", "layer_index", "pooling"):
        values = {str(sample.metadata.get(field, "__unspecified__")) for sample in samples}
        if len(values) != 1:
            raise ValueError(f"Do not mix {field} values within one experiment")
    protocols = {json.dumps(sample.metadata.get("utility_protocol"), sort_keys=True) for sample in samples}
    if len(protocols) != 1:
        raise ValueError("Do not mix utility evaluation protocols within one experiment")
    group_identities = {}
    candidates = set()
    for sample in samples:
        record = sample.metadata
        group = record["evaluation_group"]
        identity = (record["query_id"], record["memory_state_id"], record.get("base_memory_hash"))
        if task_mode == "write":
            identity += (tuple(sorted(record["label_query_ids"])),)
        if group in group_identities and group_identities[group] != identity:
            raise ValueError("One evaluation_group must identify one query/distribution and base memory")
        group_identities[group] = identity
        candidate = (group, record.get("candidate_id", sample.content_hash))
        if candidate in candidates:
            raise ValueError("Duplicate candidate within an evaluation_group")
        candidates.add(candidate)
    return samples


def split_samples(samples, group_key="subject_id", seed=17):
    """Use disjoint people by default; reject crossing sources even with explicit splits."""
    if group_key not in {"subject_id", "session_id", "video_id"}:
        raise ValueError("Unsupported split group key")
    explicit = [sample.metadata.get("split") for sample in samples]
    if any(value is not None for value in explicit):
        if not all(value in {"train", "validation", "test"} for value in explicit):
            raise ValueError("Every row must declare train/validation/test, or none may")
        assignments = np.array(explicit)
    else:
        groups = np.array(sorted({sample.metadata[group_key] for sample in samples}))
        if len(groups) < 5:
            raise ValueError("Need >=5 independent groups for automatic train/validation/test splits")
        np.random.default_rng(seed).shuffle(groups)
        validation_count = max(1, round(len(groups) * 0.2))
        test_count = max(1, round(len(groups) * 0.2))
        mapping = {group: "train" for group in groups}
        mapping.update({group: "validation" for group in groups[:validation_count]})
        mapping.update({group: "test" for group in groups[validation_count:validation_count + test_count]})
        assignments = np.array([mapping[sample.metadata[group_key]] for sample in samples])
    for field in {group_key, "video_id", "evaluation_group", "content_hash"}:
        observed = {}
        for sample, split in zip(samples, assignments):
            value = sample.content_hash if field == "content_hash" else sample.metadata[field]
            if value in observed and observed[value] != split:
                raise ValueError(f"Split leakage: {field}={value} crosses partitions")
            observed[value] = split
    masks = {split: assignments == split for split in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Every split must be nonempty")
    return assignments, masks
