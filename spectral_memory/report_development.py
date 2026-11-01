"""Independently summarize complete eight-video development QA and coverage."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .data import read_jsonl, write_jsonl
from .evaluate_coverage import (
    _aggregate,
    _event_retention,
    _load_memory_indices,
    _observe_event,
    _rates,
    _validate_frame_row,
)


DEVELOPMENT_VIDEO_IDS = {
    "video_10993", "video_2564", "video_874", "video_3501",
    "video_6860", "video_9253", "video_8241", "video_1580",
}
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 17
QA_SUBSETS = ("all", "temporal_state", "memory")
COVERAGE_METRICS = (
    "observable_hit_rate", "two_frame_retention_rate",
    "pair_witness_retention_rate", "unconditional_hit_rate", "compression_loss_rate",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _subset(row, subset):
    if subset == "all":
        return True
    if subset == "memory":
        return row.get("area") == "memory"
    return bool(
        set(row.get("tags", [])).intersection({"change detection", "sequencing"})
        or re.search(
            r"\b(before|after|first|last|beginning|end|order|changed|change|remember)\b",
            row.get("query_text", ""), re.IGNORECASE,
        )
    )


def _paired_difference(first, second):
    if set(first) != set(second):
        raise ValueError("Paired comparison has different eligible video sets")
    videos = sorted(first)
    if not videos:
        return None
    differences = np.asarray([first[video] - second[video] for video in videos], dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = differences[
        generator.integers(0, len(videos), size=(BOOTSTRAP_SAMPLES, len(videos)))
    ].mean(axis=1)
    return {
        "video_macro_difference": float(differences.mean()),
        "paired_video_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "eligible_video_ids": videos,
        "eligible_videos": len(videos),
        "per_video_differences": differences.tolist(),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fewer_than_two_video_clusters": len(videos) < 2,
        "interpretation": "exploratory development contrast; unadjusted descriptive interval, not independent confirmation",
    }


def _load_and_validate(arguments):
    frame_rows = read_jsonl(arguments.frames_manifest)
    frames = {row["video_id"]: row for row in frame_rows}
    if len(frame_rows) != 8 or set(frames) != DEVELOPMENT_VIDEO_IDS:
        raise ValueError("Only the original eight development videos are permitted")
    memories = read_jsonl(arguments.memory_manifest)
    if {row["video_id"] for row in memories} != DEVELOPMENT_VIDEO_IDS:
        raise ValueError("Memory manifest must contain only and all original development videos")
    indexed_memories = {
        (row["video_id"], row["method"], row["budget_pairs"]): row for row in memories
    }
    if len(indexed_memories) != len(memories):
        raise ValueError("Duplicate video/method/budget memory records")
    policies_by_video = {
        video: {(row["method"], row["budget_pairs"]) for row in memories if row["video_id"] == video}
        for video in DEVELOPMENT_VIDEO_IDS
    }
    if len({tuple(sorted(value)) for value in policies_by_video.values()}) != 1:
        raise ValueError("Each development video must have the same method/budget policies")
    byte_counts = defaultdict(set)
    for memory in memories:
        if not memory.get("is_reference", memory["method"].endswith("reference")):
            byte_counts[(memory["video_id"], memory["budget_pairs"])].add(memory["memory_bytes"])
    if any(len(values) != 1 for values in byte_counts.values()):
        raise ValueError("Same-video same-budget methods have different persistent memory bytes")
    status_path = arguments.readout_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Readout is incomplete; intermediate QA records are not read")
    config_path = arguments.readout_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("memory_manifest_sha256") != _digest(arguments.memory_manifest):
        raise ValueError("Readout configuration is bound to a different memory manifest")
    rows = read_jsonl(arguments.readout_dir / "records.jsonl")
    if status.get("expected_records") != len(rows) or status.get("completed_records") != len(rows):
        raise ValueError("Complete status counts do not match persisted records")
    if {row["video_id"] for row in rows} != DEVELOPMENT_VIDEO_IDS:
        raise ValueError("Readout contains nondevelopment videos or misses a development video")
    query_ids = {row["query_id"] for row in rows}
    if len(query_ids) != 36 or query_ids != set(config.get("query_ids", [])):
        raise ValueError("Expected the complete configured set of 36 development questions")
    record_keys = {(row["query_id"], row["method"], row["budget_pairs"]) for row in rows}
    if len(record_keys) != len(rows):
        raise ValueError("Duplicate question/method/budget readout records")
    by_query = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    questions = {}
    for query_id, records in by_query.items():
        first = records[0]
        video = first["video_id"]
        expected = policies_by_video[video] | {("no_memory", 0)}
        actual = {(row["method"], row["budget_pairs"]) for row in records}
        if actual != expected:
            raise ValueError(f"{query_id}: missing or unexpected method/budget conditions")
        question_fields = ("video_id", "official_answer_index", "query_text", "area", "reasoning", "tags")
        for row in records:
            if any(row.get(field) != first.get(field) for field in question_fields):
                raise ValueError(f"{query_id}: inconsistent question metadata across conditions")
            if row["reward"] not in (0, 1):
                raise ValueError("Rewards must be binary")
            if row["reward"] != int(row["response"]["answer_index"] == row["official_answer_index"]):
                raise ValueError("Stored reward disagrees with official answer and chosen option")
            if row["method"] == "no_memory":
                if row["budget_pairs"] != 0 or row["selected_pairs"] or row["memory_bytes"] != 0:
                    raise ValueError("Invalid no-memory control")
                row["is_reference"] = False
                row["retained_pairs"] = 0
            else:
                memory = indexed_memories[(video, row["method"], row["budget_pairs"])]
                if (
                    row["selected_pairs"] != memory["pair_indices"]
                    or row["payload_sha256"] != memory["payload_sha256"]
                    or row["memory_bytes"] != memory["memory_bytes"]
                ):
                    raise ValueError("QA record does not match its persisted memory manifest")
                row["is_reference"] = bool(memory.get("is_reference", memory["method"].endswith("reference")))
                row["retained_pairs"] = len(memory["pair_indices"])
        questions[query_id] = {field: first.get(field) for field in question_fields}
    if any(baseline not in {row["method"] for row in rows} for baseline in arguments.baseline):
        raise ValueError("A requested baseline does not exist in this readout")
    return frames, memories, rows, questions, status, config


def _qa_summary(rows):
    policies = sorted({(row["budget_pairs"], row["method"]) for row in rows})
    summaries = []
    video_rows = []
    for budget, method in policies:
        selected = [row for row in rows if row["budget_pairs"] == budget and row["method"] == method]
        reference_values = {row["is_reference"] for row in selected}
        if len(reference_values) != 1:
            raise ValueError("A policy mixes reference and normal memories")
        for subset in QA_SUBSETS:
            subset_rows = [row for row in selected if _subset(row, subset)]
            if not subset_rows:
                continue
            by_video = defaultdict(list)
            for row in subset_rows:
                by_video[row["video_id"]].append(row)
            macro_values = []
            for video, items in sorted(by_video.items()):
                value = float(np.mean([row["reward"] for row in items]))
                macro_values.append(value)
                video_rows.append({
                    "video_id": video, "method": method, "budget_pairs": budget, "subset": subset,
                    "questions": len(items), "correct": sum(row["reward"] for row in items),
                    "accuracy": value, "query_ids": sorted(row["query_id"] for row in items),
                    "is_reference": selected[0]["is_reference"],
                })
            summaries.append({
                "method": method, "budget_pairs": budget, "subset": subset,
                "is_reference": selected[0]["is_reference"],
                "actual_retained_pair_counts": sorted({row["retained_pairs"] for row in selected}),
                "questions": len(subset_rows),
                "correct": sum(row["reward"] for row in subset_rows),
                "question_micro_accuracy": float(np.mean([row["reward"] for row in subset_rows])),
                "video_macro_accuracy": float(np.mean(macro_values)),
                "eligible_videos": len(by_video),
                "query_ids": sorted(row["query_id"] for row in subset_rows),
                "mean_memory_bytes": float(np.mean([row["memory_bytes"] for row in subset_rows])),
            })
    return summaries, video_rows


def _coverage(arguments, frames, memories):
    action_rows = read_jsonl(arguments.actions)
    if len(action_rows) != 71 or {row["video_id"] for row in action_rows} != DEVELOPMENT_VIDEO_IDS:
        raise ValueError("Expected all 71 official action instances from the original eight development videos")
    if len({(row["video_id"], row["action_id"]) for row in action_rows}) != len(action_rows):
        raise ValueError("Duplicate official action instance")
    source_data = {video: _validate_frame_row(row) for video, row in frames.items()}
    observations = []
    for action in action_rows:
        start, end = float(action["start_s"]), float(action["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end < start:
            raise ValueError("Invalid official action interval")
        duration_us = round(end * 1_000_000) - round(start * 1_000_000)
        event = {
            **action, "event_id": f"{action['video_id']}:action:{action['action_id']}",
            "short_action": duration_us <= 2_000_000,
            "duration_microseconds": duration_us,
        }
        source_times, source_indices, _ = source_data[action["video_id"]]
        observations.append(_observe_event(event, source_times, source_indices))
    by_video = defaultdict(list)
    for event in observations:
        by_video[event["video_id"]].append(event)
    video_scores = []
    event_scores = []
    for memory in memories:
        video = memory["video_id"]
        pairs, positions, is_reference = _load_memory_indices(memory, frames[video], arguments.memory_manifest)
        source_times, source_indices, _ = source_data[video]
        retained = [_event_retention(event, positions, pairs, source_indices, source_times) for event in by_video[video]]
        for event in retained:
            event_scores.append({
                **event, "method": memory["method"], "budget_pairs": memory["budget_pairs"],
                "is_reference": is_reference, "actual_retained_pairs": len(pairs),
            })
        for subset in ("all", "short"):
            selected = retained if subset == "all" else [row for row in retained if row["short_action"]]
            video_scores.append({
                "video_id": video, "method": memory["method"], "budget_pairs": memory["budget_pairs"],
                "subset": subset, "is_reference": is_reference, "actual_retained_pairs": len(pairs),
                **_rates(selected),
            })
    observation_summary = {}
    for subset in ("all", "short"):
        selected = observations if subset == "all" else [row for row in observations if row["short_action"]]
        observation_summary[subset] = {
            "official_action_instances": len(selected),
            "observable_one_distinct_source_frame": sum(row["observation_has_one_frame"] for row in selected),
            "observable_two_distinct_source_frames": sum(row["observation_has_two_frames"] for row in selected),
            "observable_intact_input_pair": sum(row["observation_has_pair_witness"] for row in selected),
            "zero_observation_events": [row["event_id"] for row in selected if not row["observation_has_one_frame"]],
            "videos_without_action_instances_in_input": sorted(DEVELOPMENT_VIDEO_IDS - {row["video_id"] for row in selected}),
        }
    return _aggregate(video_scores), video_scores, event_scores, observations, observation_summary


def _comparisons(qa_scores, qa_videos, coverage_scores, coverage_videos, baselines):
    qa_index = {(row["budget_pairs"], row["method"], row["subset"]): row for row in qa_scores}
    coverage_index = {(row["budget_pairs"], row["method"], row["subset"]): row for row in coverage_scores}
    output = {"qa": [], "coverage": [], "skipped": []}
    for score in qa_scores:
        if score["is_reference"]:
            continue
        budget, method, subset = score["budget_pairs"], score["method"], score["subset"]
        for baseline in baselines:
            if baseline == method:
                continue
            control = qa_index.get((budget, baseline, subset))
            if control is None or control["is_reference"]:
                output["skipped"].append({"domain":"qa","method":method,"baseline":baseline,"budget_pairs":budget,"subset":subset,"reason":"no nonreference baseline at the same budget and subset"})
                continue
            if score["query_ids"] != control["query_ids"]:
                raise ValueError("QA comparison has different question sets")
            main_values = {row["video_id"]:row["accuracy"] for row in qa_videos if row["method"]==method and row["budget_pairs"]==budget and row["subset"]==subset}
            control_values = {row["video_id"]:row["accuracy"] for row in qa_videos if row["method"]==baseline and row["budget_pairs"]==budget and row["subset"]==subset}
            output["qa"].append({
                "method":method,"baseline":baseline,"budget_pairs":budget,"subset":subset,
                "questions":score["questions"],"question_correct_difference":score["correct"]-control["correct"],
                "same_question_set":True,"higher_is_better":True,
                **_paired_difference(main_values,control_values),
            })
    for score in coverage_scores:
        if score["is_reference"]:
            continue
        budget, method, subset = score["budget_pairs"], score["method"], score["subset"]
        for baseline in baselines:
            if baseline == method:
                continue
            control = coverage_index.get((budget, baseline, subset))
            if control is None or control["is_reference"]:
                output["skipped"].append({"domain":"coverage","method":method,"baseline":baseline,"budget_pairs":budget,"subset":subset,"reason":"no nonreference baseline at the same budget and subset"})
                continue
            for metric in COVERAGE_METRICS:
                main_values = {row["video_id"]:row["metrics"][metric] for row in coverage_videos if row["method"]==method and row["budget_pairs"]==budget and row["subset"]==subset and row["metrics"][metric] is not None}
                control_values = {row["video_id"]:row["metrics"][metric] for row in coverage_videos if row["method"]==baseline and row["budget_pairs"]==budget and row["subset"]==subset and row["metrics"][metric] is not None}
                difference = _paired_difference(main_values,control_values)
                if difference is not None:
                    output["coverage"].append({
                        "method":method,"baseline":baseline,"budget_pairs":budget,"subset":subset,
                        "metric":metric,"higher_is_better":metric!="compression_loss_rate",
                        **difference,
                    })
    return output


def _number(value):
    return "undefined" if value is None else f"{value:.4f}"


def report(arguments):
    if len(arguments.baseline) != len(set(arguments.baseline)):
        raise ValueError("Repeated identical --baseline values are redundant")
    output = (arguments.output_dir or arguments.readout_dir / "development_analysis").resolve()
    if output == arguments.readout_dir.resolve():
        raise ValueError("Use an independent output directory; the readout report must remain unchanged")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Development report output must be new or empty")
    frames, memories, records, questions, status, config = _load_and_validate(arguments)
    source_paths = {
        "readout_records":arguments.readout_dir/"records.jsonl",
        "readout_status":arguments.readout_dir/"status.json",
        "readout_config":arguments.readout_dir/"config.json",
        "memory_manifest":arguments.memory_manifest,
        "frames_manifest":arguments.frames_manifest,
        "official_actions":arguments.actions,
        "report_source":Path(__file__),
        "coverage_helpers_source":Path(__file__).with_name("evaluate_coverage.py"),
    }
    sources = {name:{"path":str(path.resolve()),"sha256":_digest(path)} for name,path in source_paths.items()}
    qa_scores, qa_video_rows = _qa_summary(records)
    coverage_scores, coverage_video_rows, coverage_events, observations, observation_summary = _coverage(arguments,frames,memories)
    comparisons = _comparisons(qa_scores,qa_video_rows,coverage_scores,coverage_video_rows,arguments.baseline)
    result = {
        "status":"complete",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"original eight development videos and all 36 questions; exploratory, not independent confirmation",
        "sources":sources,
        "readout_completion_status":status,
        "reader_model":config.get("model"),
        "reader_revision":config.get("revision"),
        "reader_backend_module":config.get("backend_module","spectral_memory.qwen3_backend"),
        "video_ids":sorted(DEVELOPMENT_VIDEO_IDS),
        "questions":len(questions),
        "records":len(records),
        "method_budget_conditions_per_question":len(records)//len(questions),
        "baseline_methods":arguments.baseline,
        "QA":qa_scores,
        "coverage":coverage_scores,
        "observation":observation_summary,
        "comparisons":comparisons,
        "definitions":{
            "QA_micro":"correct questions divided by questions",
            "QA_video_macro":"unweighted mean of within-video question accuracy; videos with no questions in a subset are excluded and counted",
            "short_actions":"official duration <= 2.0 seconds, compared in microseconds",
            "event_time":"closed official interval, using actual decoded source PTS; source frame IDs deduplicated",
            "coverage_micro":"retained eligible official action instances divided by eligible action instances",
            "coverage_video_macro":"unweighted mean of within-video conditional event retention, excluding videos with no eligible events",
            "observable_hit_rate":"one retained source frame conditional on one source frame originally observed",
            "two_frame_retention_rate":"two DISTINCT retained source frames conditional on two distinct frames originally observed; not a readability score",
            "pair_witness_retention_rate":"one complete retained input pair with two distinct in-interval source frames, conditional on such a pair originally existing; not an action recognition score",
            "zero_observation":"actions absent from all 64 observed source frames; separately reported, not treated as selector-induced losses",
        },
        "limitations":[
            "All analyses are development exploration with no multiple-testing correction or independent-confirmation claim.",
            "Paired bootstrap uses videos, 10000 resamples and seed 17; participant identity is unknown.",
            "Overlapping official action labels count as separate instances, not independent temporal repetitions.",
            "Temporal frame coverage does not establish readability, action recognition or QA benefit.",
            "No-memory and full/last references are summarized, but different actual budgets are excluded from matched-budget comparisons.",
        ],
    }
    for name,path in source_paths.items():
        if _digest(path) != sources[name]["sha256"]:
            raise ValueError("An input changed while the development report was computed")
    output.mkdir(parents=True,exist_ok=True)
    write_jsonl(output/"qa.videos.jsonl",qa_video_rows)
    write_jsonl(output/"coverage.videos.jsonl",coverage_video_rows)
    write_jsonl(output/"coverage.events.jsonl",coverage_events)
    write_jsonl(output/"observation.events.jsonl",observations)
    _write_json(output/"report.json",result)
    lines=[
        "# Development-only QA and evidence-coverage report","",
        "All 8 original development videos and all 36 questions; exploratory comparisons only. No multiple-testing adjustment or independent-confirmation claim.","",
        f"Reader: {config.get('model')}; completed records: {len(records)}; method/budget conditions per question: {len(records)//36}.","",
    ]
    for subset in QA_SUBSETS:
        lines.extend([f"## QA: {subset}","",
            "| Budget pairs | Method | Actual pairs | Correct / questions | Video macro | Videos | Reference |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |"])
        for score in qa_scores:
            if score["subset"]==subset:
                actual=",".join(str(value) for value in score["actual_retained_pair_counts"])
                lines.append(f"| {score['budget_pairs']} | {score['method']} | {actual} | {score['correct']}/{score['questions']} | {score['video_macro_accuracy']:.4f} | {score['eligible_videos']} | {score['is_reference']} |")
        lines.append("")
    for subset in ("all","short"):
        lines.extend([f"## Temporal evidence coverage: {subset}","",
            "| Pairs | Method | One-frame macro | One-frame micro | Events retained/observable | Two-frame macro | Intact-pair macro |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
        for score in coverage_scores:
            if score["subset"]!=subset or score["is_reference"]:
                continue
            one=score["metrics"]["observable_hit_rate"]
            two=score["metrics"]["two_frame_retention_rate"]
            pair=score["metrics"]["pair_witness_retention_rate"]
            lines.append(f"| {score['budget_pairs']} | {score['method']} | {_number(one['video_macro'])} | {_number(one['event_micro'])} | {one['event_numerator']}/{one['event_denominator']} | {_number(two['video_macro'])} | {_number(pair['video_macro'])} |")
        observed=observation_summary[subset]
        lines.extend(["",f"Official instances: {observed['official_action_instances']}; zero-observation instances: {len(observed['zero_observation_events'])}. Two-frame and intact-pair values are temporal coverage, not visual readability or action-recognition accuracy.",""])
    lines.extend(["## Same-budget paired QA contrasts","",
        "| Pairs | Method minus baseline | Subset | Video macro difference | 95% video-bootstrap interval |",
        "| ---: | --- | --- | ---: | --- |"])
    for comparison in comparisons["qa"]:
        lower,upper=comparison["paired_video_bootstrap_ci95"]
        lines.append(f"| {comparison['budget_pairs']} | {comparison['method']} minus {comparison['baseline']} | {comparison['subset']} | {comparison['video_macro_difference']:.4f} | [{lower:.4f}, {upper:.4f}] |")
    lines.extend(["","## Same-budget one-frame coverage contrasts","",
        "| Pairs | Method minus baseline | Action subset | Video macro difference | 95% video-bootstrap interval |",
        "| ---: | --- | --- | ---: | --- |"])
    for comparison in comparisons["coverage"]:
        if comparison["metric"]!="observable_hit_rate":
            continue
        lower,upper=comparison["paired_video_bootstrap_ci95"]
        lines.append(f"| {comparison['budget_pairs']} | {comparison['method']} minus {comparison['baseline']} | {comparison['subset']} | {comparison['video_macro_difference']:.4f} | [{lower:.4f}, {upper:.4f}] |")
    lines.extend(["","## Limits","",*[f"- {item}" for item in result["limitations"]],""])
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":"complete","output":str(output),"questions":len(questions),"records":len(records),"QA_score_groups":len(qa_scores),"coverage_score_groups":len(coverage_scores),"qa_comparisons":len(comparisons["qa"]),"coverage_comparisons":len(comparisons["coverage"])},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir",type=Path,required=True)
    parser.add_argument("--memory-manifest",type=Path,required=True)
    parser.add_argument("--frames-manifest",type=Path,required=True)
    parser.add_argument("--actions",type=Path,required=True)
    parser.add_argument("--baseline",action="append",default=[])
    parser.add_argument("--output-dir",type=Path)
    report(parser.parse_args())


if __name__=="__main__":
    main()
