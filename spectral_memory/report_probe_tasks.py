"""Summarize all original development tasks for passive model-internal probes."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .causal_task_audit import audit_category


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def build(root):
    paths = {
        "questions": root / "data/perception_test_sample/questions.jsonl",
        "fixed16_run": root / "outputs/causal_qa_dev_v2/records.jsonl",
        "fixed16_config": root / "outputs/causal_qa_dev_v2/config.json",
        "old_development_readout": root / "outputs/spectral_memory_dev_readout/records.jsonl",
        "old_development_config": root / "outputs/spectral_memory_dev_readout/config.json",
        "official": root / "data/perception_test_sample/annotations/sample.json",
    }
    sources = {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()}
    questions = read_rows(paths["questions"])
    official = json.loads(paths["official"].read_text(encoding="utf-8"))
    baseline = {row["query_id"]: row for row in read_rows(paths["fixed16_run"]) if row["condition"] == "baseline"}
    older = read_rows(paths["old_development_readout"])
    empty = {row["query_id"]: row for row in older if row["method"] == "no_memory"}
    full = {row["query_id"]: row for row in older if row["method"] == "full_reference" and row["budget_pairs"] == 4}
    require(len(questions) == len(baseline) == len(empty) == len(full) == 36, "Expected original 36 questions")
    identifiers = {row["query_id"] for row in questions}
    require(set(baseline) == set(empty) == set(full) == identifiers, "Task coverage mismatch")
    for key in ("fixed16_config", "old_development_config"):
        config = json.loads(paths[key].read_text(encoding="utf-8"))
        require(config["questions_sha256"] == sources["questions"]["sha256"], "Question source mismatch")
    rows = []
    for question in sorted(questions, key=lambda row: row["query_id"]):
        query = question["query_id"]
        original = next(row for row in official[question["video_id"]]["mc_question"]
                        if row["id"] == int(query.rsplit(":", 1)[-1]))
        for local, upstream in (("question", "question"), ("options", "options"), ("answer_index", "answer_id"),
                                ("tags", "tag"), ("area", "area"), ("reasoning", "reasoning")):
            require(question[local] == original[upstream], "Official annotation mismatch")
        current, without, reference = baseline[query], empty[query], full[query]
        require(current["query_text"] == question["question"] and current["options"] == question["options"], "Baseline input mismatch")
        require(current["gold_answer_index"] == without["official_answer_index"] == reference["official_answer_index"] == question["answer_index"], "Gold mismatch")
        category = audit_category(question)
        rows.append({
            "query_id": query, "video_id": question["video_id"],
            "official_question": question["question"], "official_options": question["options"],
            "official_answer_index": question["answer_index"], "official_answer_text": question["answer_text"],
            "official_area": question["area"], "official_reasoning": question["reasoning"],
            "official_tags": question["tags"], "audit_category": category,
            "temporal_state_under_frozen_rule": current["temporal_state"],
            "fixed16_baseline": {"answer_index": current["predicted_answer_index"],
                                 "answer_text": question["options"][current["predicted_answer_index"]],
                                 "correct": current["correct"], "gold_margin": current["gold_margin"],
                                 "option_logits": current["option_logits"], "prompt_sha256": current["prompt_sha256"],
                                 "input_tokens": current["input_tokens"], "visual_tokens": current["visual_tokens"]},
            "old_no_video": {"answer_index": without["response"]["answer_index"], "correct": without["reward"],
                             "option_logits": without["response"]["option_logits"],
                             "cross_run_reference": True},
            "old_full64": {"answer_index": reference["response"]["answer_index"], "correct": reference["reward"],
                           "option_logits": reference["response"]["option_logits"],
                           "complete_visual_evidence_oracle": False},
            "prior_to_fixed16_transition": f"{without['reward']}->{current['correct']}",
            "fixed16_vs_full64_agreement": current["correct"] == reference["reward"],
            "can_label_forgetting_from_current_records": False,
            "task_relevance_requirement": "Question-specific evidence access must be separately demonstrated; video-level spectra and response correctness alone do not label memory retention.",
        })
    transitions = Counter(row["prior_to_fixed16_transition"] for row in rows)
    temporal_rows = [row for row in rows if row["temporal_state_under_frozen_rule"]]
    errors = [row["query_id"] for row in rows if not row["fixed16_baseline"]["correct"]]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cpu_task_audit",
        "source_scope": "original_8_development_videos_only_no_40_video_confirmation_access",
        "gpu_used": False, "sources": sources,
        "summary": {
            "videos": len({row["video_id"] for row in rows}), "questions": len(rows),
            "fixed16_correct": sum(row["fixed16_baseline"]["correct"] for row in rows),
            "fixed16_errors": len(errors), "error_query_ids": errors,
            "temporal_questions": len(temporal_rows), "temporal_videos": len({row["video_id"] for row in temporal_rows}),
            "temporal_correct": sum(row["fixed16_baseline"]["correct"] for row in temporal_rows),
            "temporal_error_query_ids": [row["query_id"] for row in temporal_rows if not row["fixed16_baseline"]["correct"]],
            "old_no_video_to_fixed16_transitions": dict(sorted(transitions.items())),
            "full64_vs_fixed16_changed_query_ids": [row["query_id"] for row in rows if not row["fixed16_vs_full64_agreement"]],
            "established_forgetting_cases": 0,
        },
        "positive_and_negative_cases": {
            "endpoint_change_video": "video_1580: official beginning men, end enm; one video only",
            "endpoint_invariant_videos": ["video_8241", "video_9253"],
            "endpoint_invariant_state": "amor at both official endpoints",
            "only_official_memory_query": "perception_test:video_1580:mc:1",
            "memory_query_no_video_correct": True,
            "count_error": "perception_test:video_1580:mc:0 is wrong even with the prior full64 and local references; forgetting is not identified",
            "last_state_error": "perception_test:video_1580:mc:2 is wrong in fixed16/full64; one prior last-pair reference is correct with margin 0.25 and a different budget",
            "broad_visual_positive_cases": "14 no-video-wrong to fixed16-correct transitions, spanning general visual tasks; not 14 memory cases",
            "static_controls": "16 repeated-first/last-frame traces have no original-question gold labels for edited-video scoring",
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/passive_probe_task_audit"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    report = build(root)
    output = (root / arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 只读内部探针：全部36题标签审计", "",
             "固定16帧无干预baseline为25/36，原时序规则下6/8；所有标签逐题对照官方sample.json。没有已识别的遗忘案例。", "",
             "| query_id | 官方问题 | 官方答案 | baseline答案 | margin | 原时序子集 | 旧full64正确 |",
             "| --- | --- | --- | --- | ---: | --- | --- |"]
    for row in report["rows"]:
        if row["fixed16_baseline"]["correct"]:
            continue
        lines.append(f"| {row['query_id']} | {row['official_question']} | {row['official_answer_text']} | {row['fixed16_baseline']['answer_text']} | {row['fixed16_baseline']['gold_margin']:+.3f} | {row['temporal_state_under_frozen_rule']} | {row['old_full64']['correct']} |")
    lines.extend(["", "旧no-video→新固定16帧的跨run四格：错→对14、错→错9、对→错2、对→对11。仅为描述，不能把所有差异归因为视觉证据保存。",
                  "固定16与旧full64总分相同，但2564:mc:4和3501:mc:3的正确性相反，不是同一输入条件。", "",
                  "所有36题与正反例边界见report.json；独立确认40视频未访问。"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
