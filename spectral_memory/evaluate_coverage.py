"""Score predeclared action-time evidence coverage without reading QA results."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .data import read_jsonl, write_jsonl


DECLARED_AT_UTC = "2026-11-01T09:49:42Z"
SHORT_ACTION_MAX_MICROSECONDS = 2_000_000
PRIMARY_BUDGET_PAIRS = 4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 17
METRICS = (
    "observable_hit_rate",
    "two_frame_retention_rate",
    "pair_witness_retention_rate",
    "unconditional_hit_rate",
    "compression_loss_rate",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _unique_rows(rows, key, description):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {description} identifiers")
    return indexed


def _normalize_actions(annotation_rows, videos, source_provenance):
    missing = sorted(set(videos) - set(annotation_rows))
    if missing:
        raise ValueError(f"Official action annotation missing videos: {missing}")
    normalized = []
    for video_id, video in videos.items():
        official = annotation_rows[video_id]
        if official["metadata"]["split"] != video["benchmark_split"]:
            raise ValueError(f"{video_id}: annotation split mismatch")
        if "action_localisation" not in official:
            raise ValueError(f"{video_id}: absent action field is not an empty annotation list")
        identifiers = set()
        for action in official["action_localisation"]:
            if action["id"] in identifiers:
                raise ValueError(f"{video_id}: duplicate action ID")
            identifiers.add(action["id"])
            boundaries = action["timestamps"]
            if (
                len(boundaries) != 2
                or any(isinstance(value, bool) or not isinstance(value, Integral) for value in boundaries)
                or boundaries[0] < 0
                or boundaries[1] < boundaries[0]
            ):
                raise ValueError(f"{video_id}: invalid official microsecond interval")
            duration_us = int(boundaries[1] - boundaries[0])
            normalized.append({
                "event_id": f"{video_id}:action:{action['id']}",
                "video_id": video_id,
                "benchmark_split": video["benchmark_split"],
                "action_id": action["id"],
                "label_id": action["label_id"],
                "label": action["label"],
                "start_s": boundaries[0] / 1_000_000,
                "end_s": boundaries[1] / 1_000_000,
                "duration_microseconds": duration_us,
                "short_action": duration_us <= SHORT_ACTION_MAX_MICROSECONDS,
                "source_annotation": action,
                "time_mapping": "official timestamps integer microseconds divided by 1000000; closed interval",
                "source_url": source_provenance["url"],
                "source_archive_sha256": source_provenance["archive_sha256"],
                "source_member": source_provenance["member"],
                "source_member_sha256": source_provenance["member_sha256"],
            })
    return normalized


def _validate_frame_row(row):
    times = np.asarray(row["source_frame_timestamps_s"], dtype=np.float64)
    source_indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
    grid_times = np.asarray(row["timestamps_s"], dtype=np.float64)
    count = len(row["frames"])
    if count != 64 or times.shape != (count,) or source_indices.shape != (count,) or grid_times.shape != (count,):
        raise ValueError("Frozen coverage protocol requires 64-frame manifests with source PTS and indices")
    if (
        not np.isfinite(times).all()
        or not np.isfinite(grid_times).all()
        or np.any(np.diff(times) < 0)
        or np.any(np.diff(source_indices) < 0)
        or np.any(np.diff(grid_times) <= 0)
        or np.any(source_indices < 0)
    ):
        raise ValueError("Invalid source frame ordering or timestamps")
    for source_index in np.unique(source_indices):
        if np.ptp(times[source_indices == source_index]) > 1e-9:
            raise ValueError("Repeated source frame has inconsistent source PTS")
    return times, source_indices, grid_times


def _observe_event(event, source_times, source_indices):
    mask = (source_times >= event["start_s"]) & (source_times <= event["end_s"])
    positions = np.flatnonzero(mask).tolist()
    decoded = sorted(set(source_indices[mask].tolist()))
    pair_indices = [
        index
        for index in range(len(source_times) // 2)
        if mask[2 * index]
        and mask[2 * index + 1]
        and source_indices[2 * index] != source_indices[2 * index + 1]
    ]
    return {
        **event,
        "observation_sample_positions": positions,
        "observation_source_frame_indices": decoded,
        "observation_source_timestamps_s": source_times[mask].tolist(),
        "observation_distinct_frame_count": len(decoded),
        "observation_pair_indices": pair_indices,
        "observation_has_one_frame": bool(decoded),
        "observation_has_two_frames": len(decoded) >= 2,
        "observation_has_pair_witness": bool(pair_indices),
    }


def _load_memory_indices(memory, frame_row, memory_manifest):
    pair_indices = np.asarray(memory["pair_indices"], dtype=np.int64)
    positions = np.asarray(memory["frame_indices"], dtype=np.int64)
    expected = np.column_stack([2 * pair_indices, 2 * pair_indices + 1]).reshape(-1)
    if (
        pair_indices.ndim != 1
        or len(pair_indices) == 0
        or np.any(np.diff(pair_indices) <= 0)
        or np.any(pair_indices < 0)
        or np.any(pair_indices >= 32)
        or not np.array_equal(positions, expected)
    ):
        raise ValueError("Memory indices violate consecutive source-pair mapping")
    is_reference = bool(memory.get("is_reference", memory["method"].endswith("reference")))
    if not is_reference and len(pair_indices) != memory["budget_pairs"]:
        raise ValueError("Nonreference memory violates its requested pair budget")
    if memory["budget_pairs"] not in (2, 4, 8):
        raise ValueError("Memory budget is outside frozen coverage protocol")
    payload_path = Path(memory["memory_path"])
    if not payload_path.is_absolute():
        payload_path = memory_manifest.parent / payload_path
    if payload_path.stat().st_size != memory["memory_bytes"]:
        raise ValueError("Persisted memory byte count changed")
    with np.load(payload_path, allow_pickle=False) as archive:
        stored_positions = archive["frame_indices"].copy()
        stored_times = archive["timestamps_s"].copy()
    if not np.array_equal(stored_positions, positions):
        raise ValueError("Persisted frame indices differ from memory manifest")
    expected_times = np.asarray(frame_row["timestamps_s"], dtype=np.float64)[positions]
    if not np.array_equal(stored_times, expected_times):
        raise ValueError("Persisted frame times differ from frame manifest")
    return pair_indices, positions, is_reference


def _event_retention(observation, selected_positions, selected_pairs, source_indices, source_times):
    observed_sources = set(observation["observation_source_frame_indices"])
    retained_sources = sorted(observed_sources.intersection(source_indices[selected_positions].tolist()))
    retained_pairs = sorted(set(observation["observation_pair_indices"]).intersection(selected_pairs.tolist()))
    hit_positions = [
        int(position)
        for position in selected_positions
        if int(source_indices[position]) in observed_sources
    ]
    return {
        "event_id": observation["event_id"],
        "video_id": observation["video_id"],
        "short_action": observation["short_action"],
        "observation_has_one_frame": observation["observation_has_one_frame"],
        "observation_has_two_frames": observation["observation_has_two_frames"],
        "observation_has_pair_witness": observation["observation_has_pair_witness"],
        "retained_source_frame_indices": retained_sources,
        "retained_source_timestamps_s": [float(source_times[position]) for position in hit_positions],
        "retained_pair_witness_indices": retained_pairs,
        "retained_distinct_frame_count": len(retained_sources),
        "hit_one": bool(retained_sources),
        "hit_two": len(retained_sources) >= 2,
        "hit_pair_witness": bool(retained_pairs),
    }


def _rates(events):
    counts = {
        "events": len(events),
        "observable_one": sum(row["observation_has_one_frame"] for row in events),
        "observable_two": sum(row["observation_has_two_frames"] for row in events),
        "observable_pair": sum(row["observation_has_pair_witness"] for row in events),
        "retained_one": sum(row["hit_one"] for row in events),
        "retained_two": sum(row["hit_two"] for row in events),
        "retained_pair": sum(row["hit_pair_witness"] for row in events),
    }
    counts["zero_observation"] = counts["events"] - counts["observable_one"]
    counts["lost_observable"] = counts["observable_one"] - counts["retained_one"]
    fractions = {
        "observable_hit_rate": ("retained_one", "observable_one"),
        "two_frame_retention_rate": ("retained_two", "observable_two"),
        "pair_witness_retention_rate": ("retained_pair", "observable_pair"),
        "unconditional_hit_rate": ("retained_one", "events"),
        "compression_loss_rate": ("lost_observable", "observable_one"),
    }
    return {
        "counts": counts,
        "metrics": {
            name: counts[numerator] / counts[denominator] if counts[denominator] else None
            for name, (numerator, denominator) in fractions.items()
        },
        "metric_fraction_fields": fractions,
    }


def _aggregate(video_scores):
    grouped = defaultdict(list)
    for row in video_scores:
        grouped[(row["budget_pairs"], row["method"], row["subset"], row["is_reference"])].append(row)
    scores = []
    for (budget, method, subset, is_reference), rows in sorted(grouped.items()):
        counts = {name: sum(row["counts"][name] for row in rows) for name in rows[0]["counts"]}
        metrics = {}
        for name in METRICS:
            values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
            numerator, denominator = rows[0]["metric_fraction_fields"][name]
            metrics[name] = {
                "video_macro": float(np.mean(values)) if values else None,
                "eligible_videos": len(values),
                "event_micro": counts[numerator] / counts[denominator] if counts[denominator] else None,
                "event_numerator": counts[numerator],
                "event_denominator": counts[denominator],
            }
        scores.append({
            "budget_pairs": budget,
            "method": method,
            "subset": subset,
            "is_reference": is_reference,
            "total_videos": len(rows),
            "counts": counts,
            "metrics": metrics,
        })
    return scores


def _comparisons(video_scores):
    indexed = {
        (row["budget_pairs"], row["method"], row["subset"], row["video_id"]): row
        for row in video_scores
    }
    policies = sorted({
        (row["budget_pairs"], row["method"], row["subset"])
        for row in video_scores if not row["is_reference"]
    })
    comparisons = []
    for budget, baseline, subset in policies:
        if baseline == "spectral_balanced":
            continue
        videos = sorted({
            row["video_id"] for row in video_scores
            if row["budget_pairs"] == budget and row["method"] == baseline and row["subset"] == subset
        })
        if not all((budget, "spectral_balanced", subset, video) in indexed for video in videos):
            continue
        for metric in METRICS:
            differences = []
            eligible_videos = []
            for video in videos:
                main = indexed[(budget, "spectral_balanced", subset, video)]["metrics"][metric]
                control = indexed[(budget, baseline, subset, video)]["metrics"][metric]
                if (main is None) != (control is None):
                    raise ValueError("Paired methods have inconsistent eligibility denominators")
                if main is not None:
                    differences.append(main - control)
                    eligible_videos.append(video)
            if not differences:
                continue
            values = np.asarray(differences, dtype=np.float64)
            generator = np.random.default_rng(BOOTSTRAP_SEED)
            samples = values[
                generator.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
            ].mean(axis=1)
            comparisons.append({
                "budget_pairs": budget,
                "main_method": "spectral_balanced",
                "baseline": baseline,
                "subset": subset,
                "metric": metric,
                "is_primary_comparison": budget == PRIMARY_BUDGET_PAIRS and subset == "short" and metric == "observable_hit_rate" and baseline in ("kcenter", "temporal_centered"),
                "video_macro_difference": float(values.mean()),
                "paired_video_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "eligible_videos": eligible_videos,
                "video_differences": differences,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "independence_limitation": "original videos only; participant identities unknown; no multiple-comparison adjustment",
                "insufficient_videos_for_inference": len(values) < 2,
            })
    return comparisons


def evaluate(arguments):
    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Coverage output must be new or empty")
    videos = _unique_rows(read_jsonl(arguments.video_manifest), "video_id", "video")
    if len(videos) != 40:
        raise ValueError("Frozen coverage protocol requires its 40-video manifest")
    frames = _unique_rows(read_jsonl(arguments.frames_manifest), "video_id", "frame")
    if set(videos) != set(frames):
        raise ValueError("Video and frame manifests must contain exactly the same videos")
    annotations = json.loads(arguments.action_annotations.read_text(encoding="utf-8"))
    provenance = json.loads(arguments.annotation_provenance.read_text(encoding="utf-8"))
    if provenance["selected_sha256"] != _digest(arguments.action_annotations):
        raise ValueError("Action JSON differs from saved source provenance")
    if provenance["selection_video_manifest_sha256"] != _digest(arguments.video_manifest):
        raise ValueError("Action source selection used a different video manifest")
    protocol = arguments.protocol.read_text(encoding="utf-8")
    if DECLARED_AT_UTC not in protocol:
        raise ValueError("Coverage protocol declaration timestamp differs")
    actions = _normalize_actions(annotations, videos, provenance)
    actions_by_video = defaultdict(list)
    for action in actions:
        actions_by_video[action["video_id"]].append(action)
    source_data = {video: _validate_frame_row(row) for video, row in frames.items()}
    observations = [
        _observe_event(action, source_data[action["video_id"]][0], source_data[action["video_id"]][1])
        for action in actions
    ]
    observations_by_video = defaultdict(list)
    for row in observations:
        observations_by_video[row["video_id"]].append(row)
    memories = read_jsonl(arguments.memory_manifest)
    memory_keys = {(row["video_id"], row["method"], row["budget_pairs"]) for row in memories}
    if len(memory_keys) != len(memories):
        raise ValueError("Duplicate video/method/budget memory entries")
    if {row["video_id"] for row in memories} != set(videos):
        raise ValueError("Memory manifest must cover exactly the frozen video set")
    policies_by_video = {
        video: {(row["method"], row["budget_pairs"]) for row in memories if row["video_id"] == video}
        for video in videos
    }
    if len({tuple(sorted(policies)) for policies in policies_by_video.values()}) != 1:
        raise ValueError("Different videos have different memory policies or budgets")
    output.mkdir(parents=True, exist_ok=True)
    event_scores = []
    video_scores = []
    for memory in memories:
        video = memory["video_id"]
        pair_indices, positions, is_reference = _load_memory_indices(memory, frames[video], arguments.memory_manifest)
        source_times, source_indices, _ = source_data[video]
        retained = [
            _event_retention(event, positions, pair_indices, source_indices, source_times)
            for event in observations_by_video[video]
        ]
        for event in retained:
            event_scores.append({
                **event,
                "method": memory["method"],
                "budget_pairs": memory["budget_pairs"],
                "actual_retained_pairs": len(pair_indices),
                "is_reference": is_reference,
                "memory_bytes": memory["memory_bytes"],
            })
        for subset in ("all", "short"):
            subset_events = retained if subset == "all" else [row for row in retained if row["short_action"]]
            video_scores.append({
                "video_id": video,
                "method": memory["method"],
                "budget_pairs": memory["budget_pairs"],
                "actual_retained_pairs": len(pair_indices),
                "is_reference": is_reference,
                "subset": subset,
                **_rates(subset_events),
            })
    observation_summary = {}
    for subset in ("all", "short"):
        selected = observations if subset == "all" else [row for row in observations if row["short_action"]]
        observation_summary[subset] = {
            "actions": len(selected),
            "videos_with_actions": len({row["video_id"] for row in selected}),
            "observable_one_frame": sum(row["observation_has_one_frame"] for row in selected),
            "observable_two_frames": sum(row["observation_has_two_frames"] for row in selected),
            "observable_pair_witness": sum(row["observation_has_pair_witness"] for row in selected),
            "zero_observation_actions": sum(not row["observation_has_one_frame"] for row in selected),
            "zero_observation_event_ids": [row["event_id"] for row in selected if not row["observation_has_one_frame"]],
        }
    report = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_at_utc": DECLARED_AT_UTC,
        "source_sha256": _digest(__file__),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _digest(path)}
            for name, path in (
                ("video_manifest", arguments.video_manifest),
                ("frames_manifest", arguments.frames_manifest),
                ("memory_manifest", arguments.memory_manifest),
                ("action_annotations", arguments.action_annotations),
                ("annotation_provenance", arguments.annotation_provenance),
                ("protocol", arguments.protocol),
            )
        },
        "official_source_provenance": provenance,
        "short_action_max_seconds": SHORT_ACTION_MAX_MICROSECONDS / 1_000_000,
        "primary_budget_pairs": PRIMARY_BUDGET_PAIRS,
        "total_videos": len(videos),
        "official_empty_action_list_video_ids": sorted(set(videos) - set(actions_by_video)),
        "empty_list_interpretation": "no action instances supplied; not proof that no action occurs",
        "observation": observation_summary,
        "scores": _aggregate(video_scores),
        "comparisons": _comparisons(video_scores),
        "QA_inputs_read": False,
        "payload_validation": "stored frame indices and grid times matched manifests; file sizes checked; RGB bytes not read and whole archive hashes not recomputed",
        "limitations": [
            "Temporal overlap is a coverage proxy, not evidence of visibility, recognition or QA utility.",
            "Zero-observation events are reported separately and cannot be recovered by selecting observed frames.",
            "Video bootstrap does not establish participant independence; intervals are unadjusted.",
            "Official empty action lists are reported without treating their videos as action-free.",
        ],
    }
    write_jsonl(output / "actions.normalized.jsonl", actions)
    write_jsonl(output / "observation.events.jsonl", observations)
    write_jsonl(output / "coverage.events.jsonl", event_scores)
    write_jsonl(output / "coverage.videos.jsonl", video_scores)
    _write_json(output / "report.json", report)
    lines = [
        "# Action-time evidence coverage",
        "",
        f"Declared: {DECLARED_AT_UTC}; primary budget: 4 pairs; short duration: <= 2.0 seconds.",
        "",
        f"Videos: {len(videos)}; official action instances: {len(actions)}; videos with empty official action lists: {len(report['official_empty_action_list_video_ids'])}.",
        "",
        "Primary outcome: video-macro probability of retaining at least one source frame, conditional on a short event being observable in the original 64 frames.",
        "",
        "| Method | Macro hit rate | Eligible videos | Retained / observable events |",
        "| --- | ---: | ---: | ---: |",
    ]
    for score in report["scores"]:
        if score["budget_pairs"] != PRIMARY_BUDGET_PAIRS or score["subset"] != "short" or score["is_reference"]:
            continue
        metric = score["metrics"]["observable_hit_rate"]
        value = "undefined" if metric["video_macro"] is None else f"{metric['video_macro']:.4f}"
        lines.append(f"| {score['method']} | {value} | {metric['eligible_videos']} | {metric['event_numerator']} / {metric['event_denominator']} |")
    lines.extend(["", "## Predeclared main contrasts", ""])
    for comparison in report["comparisons"]:
        if comparison["is_primary_comparison"]:
            lower, upper = comparison["paired_video_bootstrap_ci95"]
            lines.append(
                f"- spectral_balanced minus {comparison['baseline']}: {comparison['video_macro_difference']:.4f}; "
                f"95% paired-video percentile interval [{lower:.4f}, {upper:.4f}]."
            )
    lines.extend(["", "## Observation limitations", ""])
    for subset, summary in observation_summary.items():
        lines.append(f"- {subset}: {summary['actions']} actions, {summary['zero_observation_actions']} with no observed source frame, {summary['observable_one_frame']} observable with at least one frame.")
    lines.extend(["", *[f"- {limitation}" for limitation in report["limitations"]], ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "actions": len(actions),
        "observation": observation_summary,
        "primary_comparisons": [row for row in report["comparisons"] if row["is_primary_comparison"]],
    }, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--memory-manifest", type=Path, required=True)
    parser.add_argument("--action-annotations", type=Path, required=True)
    parser.add_argument("--annotation-provenance", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("docs/coverage_protocol.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
