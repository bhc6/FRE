"""Independently audit archived endpoint routing outputs and masks on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .report_passive_probe import digest, read_json, stats
from .report_probe_tasks import read_rows, require


CONDITIONS = ("native", "passthrough", "causal_sham", "first_gate", "last_gate")
REPORT_CONDITIONS = CONDITIONS + ("target_gate", "opposite_gate")
LAYERS = (0, 7, 13, 20, 27)
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
CASE_FIELDS = ("case_id", "sequence", "symbols", "endpoint", "option_order", "options", "option_labels",
               "prompt", "has_video", "source_frame_indices", "gold_word", "gold_answer_index", "static_readability_control")
SYMBOL_WORDS = {"A": "men", "B": "enm"}
MASK_INVARIANTS = ("no_new_edges_or_increased_additive_weights", "no_future_keys", "prefix_mask_unchanged",
                   "suffix_nonvisual_mask_unchanged", "target_visual_mask_unchanged", "excluded_visual_keys_blocked",
                   "sham_mask_unchanged")


def selected_condition(endpoint, condition):
    if condition == "target_gate":
        return endpoint + "_gate"
    if condition == "opposite_gate":
        return ("last" if endpoint == "first" else "first") + "_gate"
    return condition


def canonical_score(logits, option_ids, options, gold):
    allowed = logits[option_ids]
    prediction = int(np.argmax(allowed))
    return {"option_logits": allowed.tolist(), "predicted_answer_index": prediction,
            "predicted_word": options[prediction], "correct": int(prediction == gold),
            "gold_margin": float(allowed[gold] - allowed[1 - gold]),
            "men_minus_enm_margin": float(allowed[options.index("men")] - allowed[options.index("enm")]),
            "option_logit_tie": bool(allowed[0] == allowed[1])}


def mask_digest(mask, dtype):
    if dtype == "torch.bool":
        require(mask.dtype == np.bool_, "Stored boolean mask dtype changed")
        raw = mask.tobytes()
    elif dtype == "torch.bfloat16":
        require(mask.dtype == np.float32, "BF16 mask was not expanded to FP32")
        words = mask.view(np.uint32)
        require(np.all((words & np.uint32(65535)) == 0), "Saved values are not exact BF16 mask values")
        raw = (words >> 16).astype(np.uint16).tobytes()
    elif dtype in ("torch.float16", "torch.float32"):
        dtype_numpy = np.float16 if dtype == "torch.float16" else np.float32
        require(np.array_equal(mask.astype(dtype_numpy).astype(np.float32), mask), "Mask dtype conversion was lossy")
        raw = mask.astype(dtype_numpy).tobytes()
    else:
        raise ValueError("Unsupported saved mask dtype")
    return hashlib.sha256(raw).hexdigest()


def validate_mask(base, used, positions, condition, capture, length):
    shape = (1, 1, length, length)
    require(base.shape == used.shape == shape and base.dtype == used.dtype, "Mask shape/dtype mismatch")
    require(base.dtype == np.bool_ or np.issubdtype(base.dtype, np.floating), "Invalid mask dtype")
    require(not np.isnan(base).any() and not np.isnan(used).any(), "NaN in attention mask")
    require(not np.isposinf(base).any() and not np.isposinf(used).any(), "Positive infinite attention bias")
    causal = np.tri(length, dtype=bool)[None, None]
    if capture["source_mask_is_none"]:
        require(base.dtype == np.bool_ and np.array_equal(base, causal), "Default causal mask differs")
        require(capture["source_mask"] is None and capture["source_is_causal_argument"] is not False, "Unmasked source causality differs")
    else:
        source = capture["source_mask"]
        require(source["shape"] == list(shape) and source["dtype"] == capture["mask_dtype"] and len(source["sha256"]) == 64,
                "Original mask description differs")
    suffix = int(positions.max()) + 1
    require(capture["suffix_start"] == suffix and suffix < length, "Suffix query boundary differs")
    eligible = list(range(suffix, length))
    kept = positions[0] if condition == "first_gate" else positions[-1] if condition == "last_gate" else positions.flatten()
    removed = positions[1:].flatten() if condition == "first_gate" else positions[:-1].flatten() if condition == "last_gate" else positions.flatten()[:0]
    require(capture["eligible_suffix_query_positions"] == eligible, "Eligible suffix query metadata differs")
    require(capture["target_query_positions"] == ([] if condition == "causal_sham" else eligible), "Target query metadata differs")
    require(capture["allowed_visual_key_positions"] == kept.tolist() and capture["removed_visual_key_positions"] == removed.tolist(),
            "Visual key metadata differs")
    require(set(capture["mask_invariants"]) == set(MASK_INVARIANTS) and all(capture["mask_invariants"][name] is True for name in MASK_INVARIANTS),
            "Runtime edge checks failed or changed")
    expected = base.copy()
    if condition in ("first_gate", "last_gate"):
        expected[:, :, suffix:, removed] = False if expected.dtype == np.bool_ else -np.inf
    require(np.array_equal(used, expected), "Gate modifies an unintended attention edge")
    if used.dtype == np.bool_:
        blocked = ~used
    else:
        minimum = (float(np.array([0xFF7F0000], dtype=np.uint32).view(np.float32)[0]) if capture["mask_dtype"] == "torch.bfloat16"
                   else float(np.finfo(np.float16 if capture["mask_dtype"] == "torch.float16" else np.float32).min))
        blocked = np.isneginf(used) | (used <= minimum)
    allowed = ~blocked
    require(not np.any(allowed & ~causal), "Future keys became visible")
    require(np.all(allowed.any(axis=-1)), "An attention row has no allowed key")
    remaining = allowed[0, 0][suffix:][:, positions.flatten()].sum(axis=-1)
    require(np.all(remaining == (480 if condition == "causal_sham" else 60)), "Unexpected visible visual-key count")
    require(mask_digest(used, capture["mask_dtype"]) == capture["used_mask_sha256"], "Saved mask byte hash differs")
    require(mask_digest(base, capture["mask_dtype"]) == capture["base_mask_sha256"], "Saved base-mask byte hash differs")
    return {"condition": condition, "suffix_start": suffix, "suffix_queries": length - suffix,
            "visual_keys_per_suffix_query": int(remaining[0]),
            "mask_elements_changed": int(np.count_nonzero(base != used)),
            "source_mask_is_none": capture["source_mask_is_none"],
            "source_mask": capture["source_mask"], "source_is_causal_argument": capture["source_is_causal_argument"],
            "eligible_suffix_query_positions": eligible, "target_query_positions": capture["target_query_positions"],
            "allowed_visual_key_positions": kept.tolist(), "removed_visual_key_positions": removed.tolist(),
            "base_mask_sha256": capture["base_mask_sha256"],
            "mask_dtype": capture["mask_dtype"], "used_mask_sha256": capture["used_mask_sha256"]}


def summary_group(rows):
    static = [row for row in rows if row["static_readability_control"]]
    mixed = [row for row in rows if not row["static_readability_control"]]
    require((len(rows), len(static), len(mixed)) == (32, 8, 24), "Reporting group is incomplete")
    return {"correct": sum(row["correct"] for row in rows), "total": 32,
            "static_correct": sum(row["correct"] for row in static), "static_total": 8,
            "mixed_correct": sum(row["correct"] for row in mixed), "mixed_total": 24,
            "ties": sum(row["option_logit_tie"] for row in rows),
            "gold_margin": stats([row["gold_margin"] for row in rows]),
            "mixed_gold_margin": stats([row["gold_margin"] for row in mixed])}


def analyze(directory, prior_directory):
    config = read_json(directory / "config.json")
    require(config["status"] == "complete" and config["completed_records"] == config["expected_records"] == 160,
            "The full routing run is not complete")
    require(config["conditions"] == list(CONDITIONS) and config["unhooked_restoration_exact"], "Condition/restoration mismatch")
    require(config["model"]["model_revision"] == REVISION and config["model"]["dtype"] == "torch.bfloat16", "Model revision/dtype mismatch")
    require(not (directory / "resource_stop.json").exists(), "Resource guard recorded a stop")
    resources = read_json(directory / "resource_summary.json")
    require(resources["status"] == "complete", "Resource monitoring is incomplete")
    for filename, expected in config["source_sha256"].items():
        require(digest(directory / "source_snapshot" / filename) == expected, "Captured source hash differs")
    for filename, field in (("protocol.md", "protocol_sha256"), ("modeling_qwen3_vl.py", "modeling_qwen3_vl.py_sha256"),
                            ("sdpa_attention.py", "sdpa_attention.py_sha256")):
        require(digest(directory / "source_snapshot" / filename) == config[field], "Protocol/model implementation hash differs")
    for filename, field in (("records.jsonl", "records_sha256"), ("summary.json", "summary_sha256"),
                            ("resource_summary.json", "resource_summary_sha256")):
        require(digest(directory / filename) == config[field], "Recorded output hash differs")
    prior_config = read_json(prior_directory / "config.json")
    require(prior_config["status"] == "complete" and digest(prior_directory / "records.jsonl") == config["prior_records_sha256"],
            "Frozen endpoint baseline changed")
    prior_all = read_rows(prior_directory / "records.jsonl")
    priors = {row["case_id"]: row for row in prior_all if row["has_video"]}
    require(len(prior_all) == 36 and len(priors) == 32, "Prior scope differs")
    require(read_json(directory / "inputs.json") == read_json(prior_directory / "inputs.json"), "Source input manifest differs")
    manifest = read_json(directory / "inputs.json")
    for endpoint in manifest["endpoints"].values():
        require(digest(Path(endpoint["path"])) == endpoint["sha256"], "Source endpoint image changed")
    cases = read_json(directory / "cases.json")
    require(len(cases) == 32 and {case["case_id"] for case in cases} == set(priors), "Case coverage differs")
    for case in cases:
        require(all(case[field] == priors[case["case_id"]][field] for field in CASE_FIELDS), "Case differs from frozen baseline")
        symbol = case["symbols"][0 if case["endpoint"] == "first" else -1]
        require(case["gold_word"] == SYMBOL_WORDS[symbol] and case["options"][case["gold_answer_index"]] == case["gold_word"],
                "Constructed endpoint label differs")
    rows = read_rows(directory / "records.jsonl")
    require(len(rows) == 160 and len({(row["case_id"], row["condition"]) for row in rows}) == 160, "Routing records are incomplete/duplicated")
    indexed = {(row["case_id"], row["condition"]): row for row in rows}
    require(set(indexed) == {(case_id, condition) for case_id in priors for condition in CONDITIONS}, "Routing matrix differs")
    audited, masks, numeric_comparisons, visual_comparisons = [], [], [], []
    for case in cases:
        case_id = case["case_id"]
        prior = priors[case_id]
        require(digest(prior_directory / prior["archive_file"]) == prior["archive_sha256"], "Prior logit archive changed")
        with np.load(prior_directory / prior["archive_file"], allow_pickle=False) as archive:
            prior_logits = archive["baseline_logits"].copy()
            prior_tokens = archive["input_ids"].copy()
        saved_logits, saved_masks, saved_hashes = {}, {}, {}
        for condition in CONDITIONS:
            row = indexed[(case_id, condition)]
            require(all(row[field] == case[field] for field in CASE_FIELDS), "Record case metadata differs")
            require(row["input_tensor_hashes"] == prior["input_tensor_hashes"], "Processed input hash declarations differ")
            batch = row["batch"]
            require(batch["input_tokens"] == 618 and batch["visual_tokens"] == 480, "Frozen input budget changed")
            require(batch["model_revision"] == REVISION and batch["dtype"] == "torch.bfloat16", "Record model differs")
            require(batch["prompt_sha256"] == prior["batch"]["prompt_sha256"], "Prompt digest differs")
            capture = row["capture"]
            require(capture["visited_layers"] == ([] if condition == "native" else list(range(28))), "Dispatch layer coverage differs")
            hashes = capture["visual_state_sha256"]
            require(set(hashes) == {str(layer) for layer in LAYERS}, "Visual capture layer coverage differs")
            require(all(len(value) == 64 and all(character in "0123456789abcdef" for character in value) for value in hashes.values()),
                    "Invalid visual-state hash")
            path = directory / row["archive_file"]
            require(digest(path) == row["archive_sha256"], "Routing archive hash differs")
            with np.load(path, allow_pickle=False) as archive:
                expected_keys = {"logits", "input_ids", "video_positions", "option_token_ids"}
                if condition in CONDITIONS[2:]:
                    expected_keys |= {"base_mask", "used_mask"}
                require(set(archive.files) == expected_keys, "Archive keys differ")
                logits, tokens, positions, option_ids = [archive[name] for name in ("logits", "input_ids", "video_positions", "option_token_ids")]
                require(logits.dtype == np.float32 and logits.shape == prior_logits.shape and np.isfinite(logits).all(), "Invalid full vocabulary logits")
                require(tokens.shape == (618,) and np.array_equal(tokens, prior_tokens), "Archived input tokens differ")
                require(hashlib.sha256(tokens.tobytes()).hexdigest() == row["input_tensor_hashes"]["input_ids"]["sha256"], "Input-ID byte hash differs")
                require(positions.shape == (8, 60) and np.issubdtype(positions.dtype, np.integer), "Visual positions shape/dtype differs")
                require(np.all(np.diff(positions.flatten()) > 0) and positions.min() >= 0 and positions.max() < len(tokens) - 1, "Invalid visual ordering")
                require(np.array_equal(np.flatnonzero(tokens == tokens[positions[0, 0]]), positions.flatten()), "Visual placeholder positions differ")
                require(option_ids.tolist() == prior["option_token_ids"], "Option token mapping changed")
                scored = canonical_score(logits, option_ids, row["options"], row["gold_answer_index"])
                require(all(scored[field] == row[field] for field in scored), "Archived logits disagree with score metadata")
                saved_logits[condition] = logits.copy()
                saved_hashes[condition] = hashes
                if condition in CONDITIONS[2:]:
                    base_mask, used_mask = archive["base_mask"], archive["used_mask"]
                    mask_check = validate_mask(base_mask, used_mask, positions, condition, capture, len(tokens))
                    masks.append({"case_id": case_id, **mask_check})
                    saved_masks[condition] = base_mask.copy()
            audited.append({**{field: case[field] for field in CASE_FIELDS if field != "prompt"},
                            "condition": condition, **scored, "archive_sha256": row["archive_sha256"],
                            "input_tokens": 618, "visual_tokens": 480, "visual_state_sha256": hashes})
        native = saved_logits["native"]
        require(np.array_equal(native, prior_logits), "Native full vocabulary differs from prior")
        require(np.array_equal(saved_logits["passthrough"], native), "Passthrough changed full vocabulary")
        require(saved_hashes["passthrough"] == saved_hashes["native"], "Passthrough changed visual-prefix hashes")
        for condition in CONDITIONS:
            row = indexed[(case_id, condition)]
            maximum = float(np.max(np.abs(saved_logits[condition] - native)))
            require(maximum == row["native_maximum_logit_difference"], "Reported numerical difference differs")
        for condition in ("first_gate", "last_gate"):
            require(np.array_equal(saved_masks[condition], saved_masks["causal_sham"]), "Gate baseline mask differs from sham")
            require(saved_hashes[condition] == saved_hashes["causal_sham"], "Gate visual-prefix hashes differ from sham")
            visual_comparisons.append({"case_id": case_id, "condition": condition, "layers": list(LAYERS),
                                       "recorded_hashes_equal_sham": True, "raw_states_archived": False})
        sham_difference = saved_logits["causal_sham"].astype(np.float64) - native
        numeric_comparisons.append({"case_id": case_id, "full_logits_identical": bool(np.array_equal(saved_logits["causal_sham"], native)),
                                    "maximum_absolute_logit_difference": float(np.abs(sham_difference).max()),
                                    "rms_logit_difference": float(np.sqrt(np.mean(sham_difference ** 2))),
                                    "prediction_changed": indexed[(case_id, "causal_sham")]["predicted_word"] != indexed[(case_id, "native")]["predicted_word"],
                                    "canonical_margin_change": indexed[(case_id, "causal_sham")]["men_minus_enm_margin"] - indexed[(case_id, "native")]["men_minus_enm_margin"],
                                    "visual_hash_changed_layers": [layer for layer in LAYERS if saved_hashes["causal_sham"][str(layer)] != saved_hashes["native"][str(layer)]]})
    by_key = {(row["case_id"], row["condition"]): row for row in audited}
    groups = {condition: [by_key[(case["case_id"], selected_condition(case["endpoint"], condition))] for case in cases]
              for condition in REPORT_CONDITIONS}
    summaries = {condition: summary_group(selected) for condition, selected in groups.items()}
    producer = read_json(directory / "summary.json")
    for condition, values in summaries.items():
        require(all(values[field] == producer["conditions"][condition][field]
                    for field in ("correct", "total", "static_correct", "static_total", "mixed_correct", "mixed_total")),
                "Producer condition summary differs")
    case_comparisons, gate_selectivity = [], []
    for case in cases:
        case_id = case["case_id"]
        sham = by_key[(case_id, "causal_sham")]
        target = by_key[(case_id, selected_condition(case["endpoint"], "target_gate"))]
        opposite = by_key[(case_id, selected_condition(case["endpoint"], "opposite_gate"))]
        opposite_word = SYMBOL_WORDS[case["symbols"][-1 if case["endpoint"] == "first" else 0]]
        case_comparisons.append({"case_id": case_id, "sequence": case["sequence"], "endpoint": case["endpoint"],
                                 "option_order": case["option_order"], "static": case["static_readability_control"],
                                 "gold_word": case["gold_word"], "sham_correct": sham["correct"],
                                 "target_correct": target["correct"], "opposite_correct": opposite["correct"],
                                 "target_recovered": not sham["correct"] and bool(target["correct"]),
                                 "target_regressed": bool(sham["correct"]) and not target["correct"],
                                 "target_gold_margin_change": target["gold_margin"] - sham["gold_margin"],
                                 "opposite_gold_margin_change": opposite["gold_margin"] - sham["gold_margin"],
                                 "opposite_reads_other_endpoint_word": opposite["predicted_word"] == opposite_word})
        first, last = by_key[(case_id, "first_gate")], by_key[(case_id, "last_gate")]
        first_word, last_word = [SYMBOL_WORDS[case["symbols"][position]] for position in (0, -1)]
        difference = first["men_minus_enm_margin"] - last["men_minus_enm_margin"]
        directional = difference * (1 if first_word == "men" else -1) if first_word != last_word else None
        gate_selectivity.append({"case_id": case_id, "sequence": case["sequence"], "endpoint": case["endpoint"],
                                 "option_order": case["option_order"], "static": case["static_readability_control"],
                                 "first_gate_word": first["predicted_word"], "last_gate_word": last["predicted_word"],
                                 "first_gate_reads_first_word": first["predicted_word"] == first_word,
                                 "last_gate_reads_last_word": last["predicted_word"] == last_word,
                                 "canonical_first_gate_minus_last_gate": difference,
                                 "source_directed_gate_margin_difference": directional})
    endpoint_pairs, option_pairs, sequence_summaries = [], [], []
    sequences = list(dict.fromkeys(case["sequence"] for case in cases))
    for condition, selected in groups.items():
        for sequence in sequences:
            sequence_rows = [row for row in selected if row["sequence"] == sequence]
            sequence_summaries.append({"condition": condition, "sequence": sequence,
                                       "correct": sum(row["correct"] for row in sequence_rows), "total": len(sequence_rows)})
            for order in ("men_enm", "enm_men"):
                pair = {row["endpoint"]: row for row in sequence_rows if row["option_order"] == order}
                first, last = pair["first"], pair["last"]
                difference = first["men_minus_enm_margin"] - last["men_minus_enm_margin"]
                directional = difference * (1 if first["gold_word"] == "men" else -1) if first["gold_word"] != last["gold_word"] else None
                endpoint_pairs.append({"condition": condition, "sequence": sequence, "option_order": order,
                                       "same_predicted_word": first["predicted_word"] == last["predicted_word"],
                                       "both_endpoints_correct": bool(first["correct"] and last["correct"]),
                                       "first_minus_last_word_margin": difference, "source_directed_question_margin_difference": directional})
            for endpoint in ("first", "last"):
                pair = [row for row in sequence_rows if row["endpoint"] == endpoint]
                option_pairs.append({"condition": condition, "sequence": sequence, "endpoint": endpoint,
                                     "same_predicted_word_across_option_orders": pair[0]["predicted_word"] == pair[1]["predicted_word"],
                                     "canonical_margin_men_enm_order": next(row["men_minus_enm_margin"] for row in pair if row["option_order"] == "men_enm"),
                                     "canonical_margin_enm_men_order": next(row["men_minus_enm_margin"] for row in pair if row["option_order"] == "enm_men")})
    protocol_gate = all(summaries[condition]["static_correct"] == 8 for condition in ("causal_sham", "first_gate", "last_gate"))
    require(protocol_gate == producer["static_selectivity_gate_passed"], "Producer static selectivity gate differs")
    mixed_comparisons = [row for row in case_comparisons if not row["static"]]
    mixed_selectivity = [row for row in gate_selectivity if not row["static"]]
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cpu_audit", "gpu_used": False,
            "integrity": {"conditions": list(CONDITIONS), "records": 160, "native_prior_full_logit_equalities": 32,
                          "passthrough_native_full_logit_equalities": 32, "independent_mask_checks": len(masks),
                          "recorded_gate_sham_visual_hash_comparisons": len(visual_comparisons) * len(LAYERS),
                          "raw_visual_state_hashes_recomputed": False, "unhooked_restoration_recorded_exact": True,
                          "unhooked_restoration_independently_recomputed": False,
                          "input_tokens": 618, "visual_tokens": 480, "visual_hash_layers": list(LAYERS),
                          "source_records_sha256": config["records_sha256"], "prior_records_sha256": config["prior_records_sha256"],
                          "analysis_source_sha256": digest(Path(__file__)), "capture_source_sha256": config["source_sha256"],
                          "protocol_sha256": config["protocol_sha256"]},
            "condition_summaries": summaries, "sequence_summaries": sequence_summaries,
            "numerical_control": {"main_reference": "causal_sham", "full_logits_identical_to_native": sum(row["full_logits_identical"] for row in numeric_comparisons),
                                  "prediction_changes_from_native": sum(row["prediction_changed"] for row in numeric_comparisons),
                                  "maximum_absolute_logit_difference": max(row["maximum_absolute_logit_difference"] for row in numeric_comparisons),
                                  "records": numeric_comparisons},
            "main_outcome": {"protocol_static_gate_passed": protocol_gate,
                             "both_fixed_gates_static_all_correct": summaries["first_gate"]["static_correct"] == summaries["last_gate"]["static_correct"] == 8,
                             "full_mixed_construction_solved_under_target_gate": protocol_gate and summaries["target_gate"]["mixed_correct"] == 24,
                             "all_target_recoveries": sum(row["target_recovered"] for row in case_comparisons),
                             "all_target_regressions": sum(row["target_regressed"] for row in case_comparisons),
                             "static_target_regressions": sum(row["target_regressed"] for row in case_comparisons if row["static"]),
                             "mixed_target_recoveries": sum(row["target_recovered"] for row in mixed_comparisons),
                             "mixed_target_regressions": sum(row["target_regressed"] for row in mixed_comparisons),
                             "mixed_opposite_reads_other_endpoint_word": sum(row["opposite_reads_other_endpoint_word"] for row in mixed_comparisons),
                             "mixed_both_fixed_gates_read_respective_source_words": sum(row["first_gate_reads_first_word"] and row["last_gate_reads_last_word"] for row in mixed_selectivity),
                             "mixed_source_directed_gate_effect_positive": sum(row["source_directed_gate_margin_difference"] > 0 for row in mixed_selectivity),
                             "mixed_source_directed_gate_margin_difference": stats([row["source_directed_gate_margin_difference"] for row in mixed_selectivity])},
            "case_comparisons": case_comparisons, "gate_selectivity": gate_selectivity,
            "endpoint_question_pairs": endpoint_pairs, "option_order_pairs": option_pairs,
            "mask_checks": masks, "visual_hash_checks": visual_comparisons, "records": audited,
            "no_video_priors": [{field: row[field] for field in ("endpoint", "option_order", "predicted_word", "option_logits", "gold_word", "correct")}
                                for row in prior_all if not row["has_video"]],
            "resource_summary": resources,
            "limitations": [
                "One previously seen source example, two source images, repeated constructed controls; no population significance or independent confirmation.",
                "The main numerical reference is causal_sham; all explicit-mask differences from native are retained.",
                "Visual hashes are compared as runtime records at five fixed layers; original visual tensors and restoration logits were not archived for independent reconstruction.",
                "Input token bytes and source image files are checked directly; other processed input tensors are verified through matching recorded hashes, not regenerated on CPU.",
                "Text/timestamp keys remain available as indirect visual-information paths; success is total intervention sufficiency and failure does not rule out routing.",
                "All-layer gates change downstream text states and text Q/K/V; they do not identify one layer, FFN damage, original attention insufficiency, or long-term forgetting.",
                "Scoring selects between option-letter logits, not unrestricted word generation; option-order effects and exact ties are retained.",
                "This is a non-spectral routing control with no trained parameters; no frequency or memory-method novelty is claimed."]}


def markdown(report):
    outcome, numeric = report["main_outcome"], report["numerical_control"]
    lines = ["# 端点内部路由：独立CPU审计", "",
             "完整160条件、32条视觉QA；32次native对旧基线、32次passthrough对native的全词表核验通过。96个显式mask从NPZ独立重算通过。",
             "固定5层共320组gate/sham视觉状态哈希一致；未保存这些原始状态张量，所以这是运行时哈希的一致性核验。", "",
             f"causal_sham对native：全词表完全相同{numeric['full_logits_identical_to_native']}/32，答案变化{numeric['prediction_changes_from_native']}/32，最大logit差{numeric['maximum_absolute_logit_difference']:.6g}。所有主效果以causal_sham为参照。", "",
             "| 条件 | 全部 | 静态 | 混合 | 平局数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for condition, row in report["condition_summaries"].items():
        lines.append(f"| {condition} | {row['correct']}/32 | {row['static_correct']}/8 | {row['mixed_correct']}/24 | {row['ties']} |")
    lines.extend(["", f"协议静态门槛：{outcome['protocol_static_gate_passed']}。混合24题中target gate相对sham修复{outcome['mixed_target_recoveries']}题、退化{outcome['mixed_target_regressions']}题；另有静态{outcome['static_target_regressions']}题退化。全部32题共修复{outcome['all_target_recoveries']}题、退化{outcome['all_target_regressions']}题。完整解决本构造混合24题：{outcome['full_mixed_construction_solved_under_target_gate']}。",
                  f"相反gate读出另一端点词{outcome['mixed_opposite_reads_other_endpoint_word']}/24；两个固定gate均读出各自指定源词{outcome['mixed_both_fixed_gates_read_respective_source_words']}/24；按源内容校正的gate词级margin差为正{outcome['mixed_source_directed_gate_effect_positive']}/24。", "",
                  "## 全部题目", "", "单元格为预测词和men−enm连续margin；正号支持men，负号支持enm，0为原始平局。passthrough与native逐元素相等。", "",
                  "| 序列 | 问题 | 选项顺序 | Gold | native | causal_sham | first_gate | last_gate |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    indexed = {(row["case_id"], row["condition"]): row for row in report["records"]}
    for case in report["case_comparisons"]:
        cells = []
        for condition in ("native", "causal_sham", "first_gate", "last_gate"):
            row = indexed[(case["case_id"], condition)]
            cells.append(f"{row['predicted_word']} ({row['men_minus_enm_margin']:+.3f})")
        lines.append(f"| {case['sequence']} | {case['endpoint']} | {case['option_order']} | {case['gold_word']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## 首末问题与选项反转", "",
                  "| 条件 | 混合首末问题同词 | 混合首末均正确 | 两选项顺序同词 |",
                  "| --- | ---: | ---: | ---: |"])
    for condition in REPORT_CONDITIONS:
        endpoints = [row for row in report["endpoint_question_pairs"] if row["condition"] == condition and row["source_directed_question_margin_difference"] is not None]
        options = [row for row in report["option_order_pairs"] if row["condition"] == condition]
        lines.append(f"| {condition} | {sum(row['same_predicted_word'] for row in endpoints)}/12 | {sum(row['both_endpoints_correct'] for row in endpoints)}/12 | {sum(row['same_predicted_word_across_option_orders'] for row in options)}/16 |")
    lines.extend(["", "全部连续margin差、反转序列、静态位置效果及逐case掩码检查均保存在JSON；没有按效果选择层、序列或选项顺序。", "",
                  "## 解释边界", ""])
    lines.extend("- " + item for item in report["limitations"])
    lines.extend(["", f"分析源码SHA256：{report['integrity']['analysis_source_sha256']}",
                  f"原始记录SHA256：{report['integrity']['source_records_sha256']}"])
    return "\n".join(lines) + "\n"


def self_checks():
    first = canonical_score(np.array([3, 1], dtype=np.float32), [0, 1], ["men", "enm"], 0)
    swapped = canonical_score(np.array([1, 3], dtype=np.float32), [0, 1], ["enm", "men"], 1)
    require(first["men_minus_enm_margin"] == swapped["men_minus_enm_margin"] == 2, "Semantic margin depends on option order")
    tied = canonical_score(np.ones(2, dtype=np.float32), [0, 1], ["enm", "men"], 1)
    require(tied["option_logit_tie"] and tied["predicted_word"] == "enm" and not tied["correct"], "Tie semantics changed")
    require(selected_condition("first", "opposite_gate") == "last_gate" and selected_condition("last", "target_gate") == "last_gate", "Gate selection depends on wrong field")
    positions = np.arange(10, 490).reshape(8, 60)
    base = np.tri(618, dtype=bool)[None, None]
    for condition in CONDITIONS[2:]:
        used = base.copy()
        removed = positions.flatten()[:0]
        if condition != "causal_sham":
            removed = positions[1:] if condition == "first_gate" else positions[:-1]
            used[:, :, 490:, removed.flatten()] = False
        capture = {"source_mask_is_none": True, "suffix_start": 490, "mask_dtype": "torch.bool",
                   "source_mask": None, "source_is_causal_argument": None,
                   "eligible_suffix_query_positions": list(range(490, 618)),
                   "target_query_positions": [] if condition == "causal_sham" else list(range(490, 618)),
                   "allowed_visual_key_positions": (positions[0] if condition == "first_gate" else positions[-1] if condition == "last_gate" else positions.flatten()).tolist(),
                   "removed_visual_key_positions": removed.flatten().tolist(),
                   "mask_invariants": {name: True for name in MASK_INVARIANTS},
                   "used_mask_sha256": mask_digest(used, "torch.bool"), "base_mask_sha256": mask_digest(base, "torch.bool")}
        validate_mask(base, used, positions, condition, capture, 618)
    return ["semantic margin invariant under option-label swap", "exact ties preserve first-index argmax",
            "gate selection uses endpoint only", "three masks preserve causal edges and endpoint key counts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/endpoint_routing_v1"))
    parser.add_argument("--prior-dir", type=Path, default=Path("outputs/endpoint_control_v1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-check-only", action="store_true")
    arguments = parser.parse_args()
    checks = self_checks()
    if arguments.self_check_only:
        print(json.dumps({"status": "passed", "checks": checks}, indent=2))
        return
    report = analyze(arguments.input_dir.resolve(), arguments.prior_dir.resolve())
    output = (arguments.output_dir or arguments.input_dir / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    (output / "checks.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"integrity": report["integrity"], "conditions": report["condition_summaries"], "main_outcome": report["main_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
