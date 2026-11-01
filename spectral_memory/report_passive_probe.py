"""Stream passive-probe archives on CPU without fitting task predictors."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .causal_task_audit import audit_category
from .report_probe_tasks import require, read_rows
from .report_causal_intervention import temporal

LAYERS = (0, 7, 13, 20, 27)
BRANCHES = ("decoder_input", "attention_delta", "pre_ffn", "ffn_input", "ffn_delta", "post_ffn")
HEAD_BRANCHES = ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "q_rot", "k_rot")
CONDITIONS = ("real", "static_first", "static_last")
VIDEOS = {"video_10993", "video_1580", "video_2564", "video_3501", "video_6860", "video_8241", "video_874", "video_9253"}
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def stats(values):
    usable = [value for value in values if value is not None]
    if not usable:
        return {"n": 0, "median": None, "min": None, "max": None}
    array = np.asarray(usable, dtype=np.float64)
    require(np.isfinite(array).all(), "Nonfinite aggregate")
    return {"n": len(usable), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def bf16_round(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    rounded = (words + np.uint32(0x7FFF) + ((words >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def spatial_spectrum(source):
    values = np.asarray(source, dtype=np.float64)
    require(values.ndim in (3, 4) and np.isfinite(values).all(), "Invalid full visual tensor")
    if values.ndim == 3:
        values = values[:, :, None, :]
    samples, patches, heads, channels = values.shape
    mean_time = values.mean(axis=0, keepdims=True)
    centered = values - mean_time
    pooled = centered.mean(axis=1)
    frequency = np.fft.rfft(centered, axis=0, norm="ortho")
    pooled_frequency = frequency.mean(axis=1)
    residual_frequency = frequency - pooled_frequency[:, None, :, :]
    weights = np.full(frequency.shape[0], 2.0)
    weights[0] = 1.0
    if samples % 2 == 0:
        weights[-1] = 1.0
    full_bins = np.sum(np.abs(frequency) ** 2, axis=(1, 3)) * weights[:, None] / (samples * patches * channels)
    mean_bins = np.sum(np.abs(pooled_frequency) ** 2, axis=2) * weights[:, None] / (samples * channels)
    residual_bins = np.sum(np.abs(residual_frequency) ** 2, axis=(1, 3)) * weights[:, None] / (samples * patches * channels)
    direct = np.mean(centered ** 2, axis=(0, 1, 3))
    pooled_direct = np.mean(pooled ** 2, axis=(0, 2))
    parseval = float(np.max(np.abs(full_bins.sum(axis=0) - direct)))
    decomposition = float(np.max(np.abs(full_bins - mean_bins - residual_bins)))
    require(parseval <= 1e-11 * max(1.0, float(direct.max())), "Temporal Parseval failure")
    require(decomposition <= 1e-11 * max(1.0, float(full_bins.max())), "Spatial-frequency decomposition failure")
    require(np.allclose(mean_bins.sum(axis=0), pooled_direct, rtol=1e-11, atol=1e-12), "Spatial mean Parseval failure")
    full_dc = np.mean(mean_time ** 2, axis=(0, 1, 3))
    pooled_dc = np.mean(mean_time.mean(axis=1) ** 2, axis=(0, 2))
    spatial_dc = np.mean((mean_time - mean_time.mean(axis=1, keepdims=True)) ** 2, axis=(0, 1, 3))
    require(np.allclose(full_dc, pooled_dc + spatial_dc, rtol=1e-11, atol=1e-12), "DC spatial decomposition failure")
    per_head = []
    for head in range(heads):
        full_energy = float(direct[head])
        mean_energy = float(pooled_direct[head])
        per_head.append({
            "head": head, "temporal_full": full_energy, "temporal_mean": mean_energy,
            "temporal_spatial_residual": float(residual_bins[1:, head].sum()),
            "mean_retention_fraction": ratio(mean_energy, full_energy),
            "dc_full": float(full_dc[head]), "dc_mean": float(pooled_dc[head]),
            "dc_spatial_residual": float(spatial_dc[head]),
            "full_energy_by_frequency": full_bins[:, head].tolist(),
            "mean_energy_by_frequency": mean_bins[:, head].tolist(),
            "spatial_residual_energy_by_frequency": residual_bins[:, head].tolist(),
        })
    total = float(direct.mean())
    mean = float(pooled_direct.mean())
    return {
        "shape": list(source.shape), "head_count": heads,
        "temporal_full": total, "temporal_mean": mean,
        "temporal_spatial_residual": float(residual_bins[1:].sum(axis=0).mean()),
        "mean_retention_fraction": ratio(mean, total),
        "dc_full": float(full_dc.mean()), "dc_mean": float(pooled_dc.mean()),
        "dc_spatial_residual": float(spatial_dc.mean()),
        "parseval_maximum_error": parseval, "spatial_decomposition_maximum_error": decomposition,
        "per_head": per_head,
    }


def rope_comparison(before, after):
    require(before.shape == after.shape and before.ndim in (2, 4), "RoPE tensor shape mismatch")
    first, second = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    before_energy, after_energy = np.sum(first ** 2, axis=-1), np.sum(second ** 2, axis=-1)
    nonzero = before_energy > 0
    require(np.all(after_energy[~nonzero] == 0), "RoPE changed zero vectors")
    relative = np.abs(after_energy[nonzero] - before_energy[nonzero]) / before_energy[nonzero]
    return {
        "vectors": int(before_energy.size),
        "relative_squared_norm_error": stats(relative.tolist()),
        "relative_squared_norm_error_mean": float(relative.mean()) if relative.size else 0.0,
        "sum_squared_norm_after_over_before": ratio(float(after_energy.sum()), float(before_energy.sum())),
        "interpretation": "Exact RoPE is norm preserving; observed BF16 arithmetic may perturb norms. No temporal-spectrum invariance follows from vector-norm preservation.",
    }


def validate_array(archive, row, key, shape=None):
    require(key in archive.files and key in row["arrays_shapes"], f"Missing array {key}")
    values = archive[key]
    require(list(values.shape) == row["arrays_shapes"][key], f"Saved shape mismatch: {key}")
    if shape is not None:
        require(values.shape == shape, f"Unexpected shape: {key}/{values.shape}")
    require(np.isfinite(values).all(), f"Nonfinite array: {key}")
    return values


def validate_residual(archive, row, prefix, shape):
    input_state = validate_array(archive, row, prefix + "decoder_input", shape)
    attention = validate_array(archive, row, prefix + "attention_delta", shape)
    pre_ffn = validate_array(archive, row, prefix + "pre_ffn", shape)
    require(np.array_equal(bf16_round(input_state + attention), pre_ffn), "Attention BF16 identity failed")
    del input_state, attention
    delta = validate_array(archive, row, prefix + "ffn_delta", shape)
    post = validate_array(archive, row, prefix + "post_ffn", shape)
    require(np.array_equal(bf16_round(pre_ffn + delta), post), "FFN BF16 identity failed")
    for name in ("attention", "ffn"):
        item = row["checks"]["identities"][prefix + name]
        require(item["exact_in_bfloat16"] and item["maximum_error"] == 0, "Recorded residual identity differs")


def attention_summary(archive, row, layer, positions, token_count):
    prefix = f"layer_{layer}_last_"
    weights = validate_array(archive, row, prefix + "attention_weights", (16, token_count)).astype(np.float64)
    require(np.all(weights >= 0) and np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=2e-7), "Invalid reference attention probabilities")
    mask_name = prefix + "attention_mask"
    if mask_name in archive.files:
        mask = validate_array(archive, row, mask_name)
        require(mask.shape in ((1, token_count), (16, token_count)), "Unexpected attention mask shape")
        blocked = ~mask if mask.dtype == np.bool_ else np.isneginf(mask) | (mask < -1e4)
        require(np.all(weights[np.broadcast_to(blocked, weights.shape)] == 0), "Reference attention violates mask")
    visual_mass = weights[:, positions].sum(axis=2)
    total_visual = visual_mass.sum(axis=1)
    reference = validate_array(archive, row, prefix + "reference_context", (16, 128)).astype(np.float64)
    actual = validate_array(archive, row, prefix + "sdpa_context", (16, 128)).astype(np.float64)
    absolute_error = float(np.max(np.abs(reference - actual)))
    relative_error = float(np.linalg.norm(reference - actual) / max(np.linalg.norm(actual), 1e-12))
    saved = row["checks"]["attention_reference"][str(layer)]
    require(saved["query_heads"] == 16 and saved["kv_heads"] == 8, "GQA reference metadata mismatch")
    require(np.isclose(saved["maximum_absolute_error"], absolute_error, rtol=1e-6, atol=1e-8), "Reference context maximum error differs")
    require(np.isclose(saved["relative_l2_error"], relative_error, rtol=1e-5, atol=1e-8), "Reference context relative error differs")
    head_details = []
    for head in range(16):
        conditional = visual_mass[head] / total_visual[head] if total_visual[head] > 0 else None
        probabilities = weights[head]
        positive = probabilities[probabilities > 0]
        head_details.append({
            "head": head, "visual_mass": float(total_visual[head]),
            "visual_time_mass": visual_mass[head].tolist(),
            "conditional_visual_time_mass": conditional.tolist() if conditional is not None else None,
            "full_key_entropy_nats": float(-np.sum(positive * np.log(positive))),
        })
    return {
        "layer": layer, "video_id": row["video_id"], "condition": row["condition"], "query_id": row["query_id"],
        "mean_visual_mass": float(total_visual.mean()),
        "mean_first_slot_mass": float(visual_mass[:, 0].mean()),
        "mean_last_slot_mass": float(visual_mass[:, -1].mean()),
        "mean_middle_slots_mass": float(visual_mass[:, 1:-1].sum(axis=1).mean()),
        "mean_visual_time_mass": visual_mass.mean(axis=0).tolist(),
        "context_reference_maximum_error": absolute_error,
        "context_reference_relative_l2_error": relative_error,
        "reference_label": "Capture-time CPU FP64 QK/mask/scaling reconstruction rounded to FP32; not exported SDPA probabilities or causal attribution",
        "heads": head_details,
    }


def validate_config(root, directory, video, questions_hash, count):
    config = read_json(directory / "config.json")
    require(config["status"] == "complete" and config["video_id"] == video, "Passive video incomplete or wrong identity")
    require(config["expected_conditions"] == config["completed_conditions"] == count, "Incomplete passive conditions")
    require(config["layers"] == list(LAYERS) and config["questions_sha256"] == questions_hash, "Layer/question source mismatch")
    require(config["last_restored_logits_identical"] is True, "Probe cleanup equality failed")
    require(config["model"]["model_revision"] == REVISION and config["model"]["dtype"] == "torch.bfloat16", "Model mismatch")
    require(digest(directory / "source_snapshot/passive_probe.py") == config["source_sha256"], "Passive source snapshot mismatch")
    require(digest(directory / "source_snapshot/protocol.md") == config["protocol_sha256"], "Protocol snapshot mismatch")
    original_folder = root / "outputs" / ("ffn_qwen3_dev16_v1" if video == "video_10993" else f"ffn_qwen3_dev16_{video}")
    metadata = config["inputs"]
    require(digest(original_folder / "probe.json") == metadata["probe_sha256"], "Original probe source changed")
    require(digest(original_folder / "frames.manifest.jsonl") == metadata["frames_manifest_sha256"], "Original manifest changed")
    manifest = read_json(original_folder / "frames.manifest.jsonl")
    require([digest(Path(path)) for path in manifest["frames"]] == metadata["frame_sha256"], "Original frames changed")
    require(config["source_frame_timestamps_s"] == manifest["source_frame_timestamps_s"], "Source PTS changed")
    return config, manifest


def analyze(root, input_root, expected_videos):
    questions_path = root / "data/perception_test_sample/questions.jsonl"
    questions = {row["query_id"]: row for row in read_rows(questions_path)}
    require(len(questions) == 36 and {row["video_id"] for row in questions.values()} == VIDEOS, "Original question scope changed")
    v2_rows = [row for row in read_rows(root / "outputs/causal_qa_dev_v2/records.jsonl") if row["condition"] == "baseline"]
    v2 = {row["query_id"]: row for row in v2_rows}
    directories = {}
    for path in input_root.rglob("records.jsonl"):
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        video = read_json(config_path).get("video_id")
        if video in expected_videos:
            require(video not in directories, "Duplicate video directories")
            directories[video] = path.parent
    require(set(directories) == expected_videos, "Missing requested passive videos")
    spectral_rows, rope_rows, attention_rows, task_rows, provenance, sampling_rows = [], [], [], [], [], []
    passive_checks, residual_checks, archive_bytes = 0, 0, 0
    for video, directory in sorted(directories.items()):
        expected_questions = {query: question for query, question in questions.items() if question["video_id"] == video}
        config, manifest = validate_config(root, directory, video, digest(questions_path), len(expected_questions) + 3)
        records = read_rows(directory / "records.jsonl")
        require(len(records) == len(expected_questions) + 3, "Record count mismatch")
        require({row["condition"] for row in records} == set(CONDITIONS) | {"qa_" + query.rsplit(":", 1)[-1] for query in expected_questions}, "Condition coverage mismatch")
        require(len({row["condition"] for row in records}) == len(records), "Duplicate conditions")
        require({row["query_id"] for row in records if row["query_id"] is not None} == set(expected_questions), "QA coverage mismatch")
        neutral_tokens, neutral_positions, neutral_batch = None, None, None
        for row in sorted(records, key=lambda item: (item["condition"] != "real", item["condition"])):
            neutral = row["query_id"] is None
            require(row["video_id"] == video, "Record video mismatch")
            require(row["checks"]["visual_states_persisted"] == neutral, "Visual persistence flag mismatch")
            batch = row["batch"]
            require(batch["model_revision"] == REVISION and batch["dtype"] == "torch.bfloat16", "Record model/dtype mismatch")
            require(batch["visual_tokens"] == 480 and batch["input_tokens"] <= 1024, "Unexpected token count")
            require(batch["source_timestamps_s"] == manifest["timestamps_s"], "Presentation timestamps mismatch")
            require(row["prompt_sha256"] == batch["prompt_sha256"], "Prompt hash metadata mismatch")
            archive_path, logits_path = directory / row["arrays_file"], directory / row["logits_file"]
            require(digest(archive_path) == row["arrays_sha256"] and digest(logits_path) == row["logits_sha256"], "Archive hash mismatch")
            with np.load(logits_path, allow_pickle=False) as logits_archive:
                require(set(logits_archive.files) == {"baseline", "probe"}, "Logits archive keys mismatch")
                baseline, probe = logits_archive["baseline"], logits_archive["probe"]
                require(baseline.dtype == probe.dtype == np.float32 and baseline.ndim == 1, "Logit shape/dtype mismatch")
                require(np.isfinite(baseline).all() and np.array_equal(baseline, probe), "Passive probe changed full vocabulary")
                require(hashlib.sha256(baseline.tobytes()).hexdigest() == row["baseline_logits_sha256"], "Baseline logits hash mismatch")
                require(row["full_logits_identical"] and row["full_logits_maximum_error"] == 0, "Passive equality metadata mismatch")
                if not neutral:
                    question = expected_questions[row["query_id"]]
                    for field, official_field in (("question", "question"), ("options", "options"), ("gold_answer_index", "answer_index")):
                        require(row[field] == question[official_field], "QA official mismatch")
                    option_ids = row.get("option_token_ids", v2[row["query_id"]]["option_token_ids"])
                    require(np.array_equal(baseline[option_ids], row["option_logits"]), "Option/full vocabulary mismatch")
                    chosen = int(np.argmax(row["option_logits"]))
                    gold = question["answer_index"]
                    margin = float(row["option_logits"][gold] - max(value for index, value in enumerate(row["option_logits"]) if index != gold))
                    require(chosen == row["predicted_answer_index"] and int(chosen == gold) == row["correct"] and margin == row["gold_margin"], "QA reward/margin mismatch")
                    require(row["temporal_state"] == temporal(question), "QA subset mismatch")
                    task_rows.append({
                        "query_id": row["query_id"], "video_id": video, "question": question["question"],
                        "options": question["options"], "gold_answer_index": gold, "predicted_answer_index": chosen,
                        "correct": row["correct"], "gold_margin": margin, "audit_category": audit_category(question),
                        "temporal_state": row["temporal_state"],
                        "option_token_ids": option_ids,
                        "option_token_ids_source": "current_record" if "option_token_ids" in row else "same_query_frozen_v2_baseline_same_model_revision",
                        "same_prediction_as_v2": chosen == v2[row["query_id"]]["predicted_answer_index"],
                        "same_option_logits_as_v2": row["option_logits"] == v2[row["query_id"]]["option_logits"],
                        "visual_trajectories_saved_for_this_question": False,
                    })
            passive_checks += 1
            with np.load(archive_path, allow_pickle=False) as arrays:
                require(set(arrays.files) == set(row["arrays_shapes"]), "Array manifest keys mismatch")
                positions = validate_array(arrays, row, "video_positions", (8, 60))
                tokens = validate_array(arrays, row, "token_ids", (batch["input_tokens"],))
                times = validate_array(arrays, row, "timestamps_s", (8,))
                require(np.issubdtype(positions.dtype, np.integer) and len(np.unique(positions)) == 480, "Visual positions invalid")
                require(positions.min() >= 0 and positions.max() < len(tokens) - 1 and np.all(np.diff(positions.flatten()) > 0), "Visual ordering invalid")
                require(len(np.unique(tokens[positions])) == 1, "Visual positions contain multiple token IDs")
                require(np.allclose(times, batch["hidden_timestamps_s"], rtol=0, atol=1e-10), "Hidden timestamps mismatch")
                require(np.all(np.diff(times) > 0) and np.allclose(np.diff(times), np.diff(times).mean()), "Nonuniform time samples")
                hidden_interval = float(np.diff(times).mean())
                hidden_sampling_hz = 1.0 / hidden_interval
                require(np.isclose(batch["hidden_sampling_hz"], hidden_sampling_hz, rtol=1e-12, atol=1e-12), "Hidden sampling metadata mismatch")
                if row["condition"] == "real":
                    neutral_tokens, neutral_positions, neutral_batch = tokens.copy(), positions.copy(), batch
                    sampling_rows.append({"video_id": video, "hidden_timestamps_s": times.tolist(),
                                          "hidden_interval_s": hidden_interval, "hidden_sampling_hz": hidden_sampling_hz,
                                          "nyquist_hz": hidden_sampling_hz / 2,
                                          "presented_frame_span_s": float(batch["source_timestamps_s"][-1] - batch["source_timestamps_s"][0]),
                                          "frequencies_hz": np.fft.rfftfreq(8, hidden_interval).tolist()})
                else:
                    require(np.array_equal(positions, neutral_positions), "Visual positions changed within video")
                    require(np.array_equal(tokens[:positions.max() + 1], neutral_tokens[:positions.max() + 1]), "Visual prefix token IDs changed")
                    if neutral:
                        require(np.array_equal(tokens, neutral_tokens) and batch["prompt_sha256"] == neutral_batch["prompt_sha256"], "Neutral prompt/tokens changed")
                observed_bytes = sum(np.dtype(np.float32).itemsize * np.prod(shape) for key, shape in row["arrays_shapes"].items()
                                     if key not in ("token_ids", "video_positions", "timestamps_s"))
                observed_bytes += tokens.nbytes + positions.nbytes + times.nbytes
                require(observed_bytes == row["arrays_uncompressed_bytes"], "Uncompressed archive byte count mismatch")
                archive_bytes += int(observed_bytes)
                for layer in LAYERS:
                    prefix, last = f"layer_{layer}_", f"layer_{layer}_last_"
                    validate_residual(arrays, row, last, (2048,))
                    residual_checks += 2
                    for branch in BRANCHES:
                        state = validate_array(arrays, row, last + branch, (2048,))
                        require(state.dtype == np.float32, "Last residual dtype mismatch")
                    for branch in HEAD_BRANCHES:
                        heads = 16 if branch.startswith("q_") else 8
                        state = validate_array(arrays, row, last + branch, (heads, 128))
                        require(state.dtype == np.float32, "Last head dtype mismatch")
                    for kind in ("q", "k"):
                        before = validate_array(arrays, row, last + kind + "_norm")
                        after = validate_array(arrays, row, last + kind + "_rot")
                        rope_rows.append({"video_id": video, "condition": row["condition"], "query_id": row["query_id"],
                                          "layer": layer, "kind": kind, "scope": "last_prefill_token",
                                          **rope_comparison(before, after)})
                    attention_rows.append(attention_summary(arrays, row, layer, positions, len(tokens)))
                    if neutral:
                        validate_residual(arrays, row, prefix, (8, 60, 2048))
                        residual_checks += 2
                        for branch in BRANCHES + HEAD_BRANCHES:
                            shape = (8, 60, 2048) if branch in BRANCHES else (8, 60, 16 if branch.startswith("q_") else 8, 128)
                            state = validate_array(arrays, row, prefix + branch, shape)
                            require(state.dtype == np.float32, "Visual dtype mismatch")
                            spectral_rows.append({"video_id": video, "condition": row["condition"], "layer": layer, "branch": branch,
                                                  "frequencies_hz": np.fft.rfftfreq(8, float(np.diff(times).mean())).tolist(),
                                                  **spatial_spectrum(state)})
                            del state
                        for kind in ("q", "k"):
                            before = validate_array(arrays, row, prefix + kind + "_norm")
                            after = validate_array(arrays, row, prefix + kind + "_rot")
                            rope_rows.append({"video_id": video, "condition": row["condition"], "query_id": None,
                                              "layer": layer, "kind": kind, "scope": "all_visual_tokens",
                                              **rope_comparison(before, after)})
                    else:
                        require(not any(prefix + branch in arrays.files for branch in BRANCHES + HEAD_BRANCHES), "QA unexpectedly contains full visual trajectories")
                provenance.append({"video_id": video, "condition": row["condition"], "arrays_sha256": row["arrays_sha256"],
                                   "logits_sha256": row["logits_sha256"], "source_sha256": config["source_sha256"],
                                   "protocol_sha256": config["protocol_sha256"], "modeling_source_sha256": config["modeling_source_sha256"],
                                   "last_prefill_token_position": len(tokens) - 1, "last_prefill_token_id": int(tokens[-1]),
                                   "input_tokens": len(tokens), "arrays_uncompressed_bytes": int(observed_bytes)})
            print(f"CPU audited {video} {row['condition']}", flush=True)
    expected_question_count = sum(question["video_id"] in expected_videos for question in questions.values())
    require(len(task_rows) == expected_question_count and passive_checks == 3 * len(expected_videos) + expected_question_count, "Incomplete overall passive matrix")
    indexed_spectra = {(row["video_id"], row["condition"], row["layer"], row["branch"]): row for row in spectral_rows}
    grouped = defaultdict(list)
    for row in spectral_rows:
        grouped[(row["condition"], row["layer"], row["branch"])].append(row)
    summary_keys = ("temporal_full", "temporal_mean", "temporal_spatial_residual", "mean_retention_fraction", "dc_full")
    spectral_summaries = [{"condition": condition, "layer": layer, "branch": branch,
                          **{key: stats([row[key] for row in rows]) for key in summary_keys}}
                         for (condition, layer, branch), rows in sorted(grouped.items())]
    paired_rows, rope_temporal_rows = [], []
    for video in sorted(expected_videos):
        for layer in LAYERS:
            for branch in BRANCHES + HEAD_BRANCHES:
                real = indexed_spectra[(video, "real", layer, branch)]
                for condition in CONDITIONS[1:]:
                    static = indexed_spectra[(video, condition, layer, branch)]
                    paired_rows.append({"video_id": video, "condition": condition, "layer": layer, "branch": branch,
                                        "full_temporal_static_over_real": ratio(static["temporal_full"], real["temporal_full"]),
                                        "mean_temporal_static_over_real": ratio(static["temporal_mean"], real["temporal_mean"]),
                                        "spatial_residual_static_over_real": ratio(static["temporal_spatial_residual"], real["temporal_spatial_residual"])})
            for condition in CONDITIONS:
                for kind in ("q", "k"):
                    before = indexed_spectra[(video, condition, layer, kind + "_norm")]
                    after = indexed_spectra[(video, condition, layer, kind + "_rot")]
                    rope_temporal_rows.append({"video_id": video, "condition": condition, "layer": layer, "kind": kind,
                                               "temporal_before": before["temporal_full"], "temporal_after": after["temporal_full"],
                                               "temporal_after_over_before": ratio(after["temporal_full"], before["temporal_full"]),
                                               "temporal_after_minus_before": after["temporal_full"] - before["temporal_full"]})
    tasks_by_id = {row["query_id"]: row for row in task_rows}
    for row in attention_rows:
        if row["query_id"] is not None:
            task = tasks_by_id[row["query_id"]]
            row.update({key: task[key] for key in ("correct", "gold_margin", "audit_category", "temporal_state")})
            target = {"endpoint_first_letters": 0, "endpoint_last_letters": 7}.get(task["audit_category"])
            row["descriptive_endpoint_slot"] = target
            row["descriptive_endpoint_slot_mass"] = row["mean_visual_time_mass"][target] if target is not None else None
    task_summary = {"questions": len(task_rows), "correct": sum(row["correct"] for row in task_rows),
                    "temporal_questions": sum(row["temporal_state"] for row in task_rows),
                    "temporal_correct": sum(row["correct"] for row in task_rows if row["temporal_state"]),
                    "errors": [row for row in task_rows if not row["correct"]],
                    "v2_prediction_changes": [row["query_id"] for row in task_rows if not row["same_prediction_as_v2"]]}
    per_video = []
    for video in sorted(expected_videos):
        selected = [row for row in task_rows if row["video_id"] == video]
        per_video.append({"video_id": video, "questions": len(selected),
                          "correct": sum(row["correct"] for row in selected),
                          "accuracy": sum(row["correct"] for row in selected) / len(selected),
                          "temporal_questions": sum(row["temporal_state"] for row in selected),
                          "temporal_correct": sum(row["correct"] for row in selected if row["temporal_state"])})
    task_summary["per_video"] = per_video
    task_summary["macro_video_accuracy"] = float(np.mean([row["accuracy"] for row in per_video]))
    task_summary["by_audit_category"] = [
        {"category": category, "questions": len(selected), "correct": sum(row["correct"] for row in selected),
         "videos": len({row["video_id"] for row in selected})}
        for category in sorted({row["audit_category"] for row in task_rows})
        for selected in [[row for row in task_rows if row["audit_category"] == category]]]
    first_layer = [row for row in spectral_rows if row["condition"] == "real" and row["layer"] == 0 and row["branch"] == "decoder_input"]
    static_rotation = []
    for condition in CONDITIONS[1:]:
        for kind in ("q", "k"):
            selected = [row for row in rope_temporal_rows if row["condition"] == condition and row["layer"] == 0 and row["kind"] == kind]
            static_rotation.append({"condition": condition, "kind": kind, "videos": len(selected),
                                    "zero_before_positive_after": sum(row["temporal_before"] == 0 and row["temporal_after"] > 0 for row in selected),
                                    "temporal_before": stats([row["temporal_before"] for row in selected]),
                                    "temporal_after": stats([row["temporal_after"] for row in selected])})
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete_cpu_audit", "gpu_used": False,
        "video_ids": sorted(expected_videos), "scope": "development_descriptive_internal_probe",
        "integrity": {"passive_full_vocabulary_checks": passive_checks, "independent_bf16_residual_checks": residual_checks,
                      "neutral_full_visual_captures": len(expected_videos) * 3, "qa_last_token_captures": len(task_rows),
                      "total_arrays_uncompressed_bytes": archive_bytes,
                      "maximum_parseval_error": max(row["parseval_maximum_error"] for row in spectral_rows),
                      "maximum_spatial_decomposition_error": max(row["spatial_decomposition_maximum_error"] for row in spectral_rows),
                      "questions_sha256": digest(questions_path), "analysis_source_sha256": digest(Path(__file__))},
        "definitions": {
            "energy": "Mean activation squared over time, spatial tokens, heads and coordinates; no normalization by total/DC energy.",
            "spatial_identity": "After per-patch temporal centering: E_full(f)=E_spatial_mean(f)+E_spatial_residual(f), with one-sided orthonormal FFT weights.",
            "head_aggregation": "Actual Q heads=16 and KV heads=8; head statistics use their actual count without duplicated KV observations.",
            "attention": "Saved probabilities were reconstructed at capture time on CPU from actual full-key Q/K/mask/scaling; this analysis validates probabilities and compares saved reference context with o_proj-input SDPA vectors.",
            "attention_archive_scope": "Full text-position K/V are not archived, and QA archives contain only last-position Q/K/V. This offline audit cannot independently reconstruct the complete attention row or its weighted context from archived Q/K/V.",
            "last_position": "Last input/prefill text position that produces the first output logits, not a generated answer token.",
        },
        "diagnostic_summary": {
            "real_layer0_decoder_input_mean_retention": stats([row["mean_retention_fraction"] for row in first_layer]),
            "layer0_static_rotation": static_rotation,
            "reference_context_relative_l2_error": stats([row["context_reference_relative_l2_error"] for row in attention_rows]),
            "interpretation": "Spatial pooling removes measured temporal variation, not established task evidence. Static-content rotation changes are positional diagnostics, not memory or motion evidence."},
        "sampling_summary": {
            "sampled_frames_per_video": 16, "hidden_time_groups": 8,
            "hidden_sampling_hz": stats([row["hidden_sampling_hz"] for row in sampling_rows]),
            "nyquist_hz": stats([row["nyquist_hz"] for row in sampling_rows]),
            "hidden_interval_s": stats([row["hidden_interval_s"] for row in sampling_rows]),
            "presented_frame_span_s": stats([row["presented_frame_span_s"] for row in sampling_rows]),
            "frequencies_cycles_per_time_group": np.fft.rfftfreq(8).tolist(), "records": sampling_rows,
            "interpretation": "Frequencies describe only the eight observed hidden time groups. Sparse extraction of sixteen source frames has no temporal antialias filter, so these frequencies do not identify continuous original-video motion frequencies."},
        "spectral_summaries": spectral_summaries, "spectral_records": spectral_rows, "static_paired_records": paired_rows,
        "rope_vector_records": rope_rows, "rope_temporal_records": rope_temporal_rows,
        "attention_records": attention_rows, "task_summary": task_summary, "task_records": task_rows, "provenance": provenance,
        "limitations": [
            "Only original development videos; no independent confirmation, no high-dimensional probe training or layer selection.",
            "Static input retains position, timestamp and causal-context effects; real-minus-static is not uniquely visual motion.",
            "Only eight hidden time groups are observed from sixteen sparsely extracted frames, without temporal antialias filtering; cycles per group and timestamp-converted Hz cannot identify fast continuous motion or rule out aliasing.",
            "Spatial tokens are fixed grid locations, not object tracks; high residual energy is not task evidence.",
            "RoPE can change time spectra while approximately preserving individual vector norms; this is not added information.",
            "Reference attention is not exported SDPA probability or causal necessity/sufficiency; full-key context error is reported.",
            "Full text-position K/V were not archived; this CPU audit checks saved row normalization/masks and reference-versus-SDPA context errors, but cannot independently recompute the whole attention row or weighted context from archived Q/K/V.",
            "Endpoint-slot mass only describes first/last visual group, not a complete independently grounded evidence interval.",
            "QA visual trajectories are absent; neutral visual spectra cannot be treated as independent question-specific representations.",
            "Many heads/layers/conditions from one video are not independent samples; no significance or memory-success claim.",
            "Old layer-0 selection failures remain unexplained by these passive measurements."]}


def show(value):
    if value["median"] is None:
        return "undefined"
    return f"{value['median']:.5g} [{value['min']:.5g}, {value['max']:.5g}]"


def markdown(report):
    integrity, tasks = report["integrity"], report["task_summary"]
    lines = ["# V4 模型内部只读探针：CPU独立审计", "",
             f"视频{len(report['video_ids'])}个；全词表被动性核验{integrity['passive_full_vocabulary_checks']}次，CPU重算BF16残差恒等式{integrity['independent_bf16_residual_checks']}次，均通过。",
             f"逐频空间分解最大误差{integrity['maximum_spatial_decomposition_error']:.3g}；Parseval最大误差{integrity['maximum_parseval_error']:.3g}。",
             f"QA baseline {tasks['correct']}/{tasks['questions']}；时序题{tasks['temporal_correct']}/{tasks['temporal_questions']}。没有训练probe或将本结果称为记忆改善。", "",
             f"真实输入第0层decoder_input的空间均值仅保留时间能量比例{show(report['diagnostic_summary']['real_layer0_decoder_input_mean_retention'])}；这表示时间变化被平均消除，尚未证明消除了任务证据。",
             "静态输入第0层Q/K归一化后与旋转后的比较逐项见JSON diagnostic_summary；位置旋转产生的时间变化不能解释为运动或记忆。", "",
             f"每视频16个稀疏采样帧形成8个hidden时间格：hidden_sampling_hz为{show(report['sampling_summary']['hidden_sampling_hz'])} Hz；对应Nyquist为{show(report['sampling_summary']['nyquist_hz'])} Hz；展示帧跨度为{show(report['sampling_summary']['presented_frame_span_s'])}秒。",
             "FFT仅描述这8个已观察时间格，频点为0、0.125、0.25、0.375、0.5 cycles/time-group。原视频抽帧没有时间抗混叠滤波；0.25 cycles/time-group只表示每4个已采样时间格一个周期，不能称为原视频的高频动作。转换为Hz亦不能排除混叠或恢复未采样的快速变化。", "",
             "数字为视频中位数[实际范围]，不是置信区间。完整逐head/逐频数值保留在JSON。", "",
             "## 全空间、均值和空间残差时间能量", "",
             "| 条件 | 层 | 分支 | 全空间时间能量 | 空间均值时间能量 | 空间残差时间能量 | 均值保留份额 |",
             "| --- | ---: | --- | --- | --- | --- | --- |"]
    for row in report["spectral_summaries"]:
        if row["branch"] in BRANCHES:
            lines.append(f"| {row['condition']} | {row['layer']} | {row['branch']} | {show(row['temporal_full'])} | {show(row['temporal_mean'])} | {show(row['temporal_spatial_residual'])} | {show(row['mean_retention_fraction'])} |")
    lines.extend(["", "## Q/K/V与RoPE", "",
                  "| 条件 | 层 | 分支 | 全空间时间能量 | DC能量 | 空间均值保留份额 |",
                  "| --- | ---: | --- | --- | --- | --- |"])
    for row in report["spectral_summaries"]:
        if row["branch"] in HEAD_BRANCHES:
            lines.append(f"| {row['condition']} | {row['layer']} | {row['branch']} | {show(row['temporal_full'])} | {show(row['dc_full'])} | {show(row['mean_retention_fraction'])} |")
    rope_errors = [row["relative_squared_norm_error"]["max"] for row in report["rope_vector_records"]]
    lines.extend(["", f"norm→rot逐向量平方范数相对误差最大值：{max(rope_errors):.6g}。该数值单独报告BF16偏差；幅值近似守恒不要求时间频谱相同。", "",
                  "## 按视频与预定题型汇总", "",
                  f"视频宏平均准确率{tasks['macro_video_accuracy']:.5f}；每个视频只作为一个聚合单位，不将题目、head或层复制成独立样本。", "",
                  "| 视频 | 正确/题目 | 时序正确/题目 |", "| --- | ---: | ---: |"])
    for row in tasks["per_video"]:
        lines.append(f"| {row['video_id']} | {row['correct']}/{row['questions']} | {row['temporal_correct']}/{row['temporal_questions']} |")
    lines.extend(["", "| 预定题型 | 正确/题目 | 视频数 |", "| --- | ---: | ---: |"])
    for row in tasks["by_audit_category"]:
        lines.append(f"| {row['category']} | {row['correct']}/{row['questions']} | {row['videos']} |")
    lines.extend(["",
                  "## QA错误与末位置参考attention", "",
                  "| query_id | 官方答案 | baseline答案 | margin |", "| --- | --- | --- | ---: |"])
    for row in tasks["errors"]:
        lines.append(f"| {row['query_id']} | {row['options'][row['gold_answer_index']]} | {row['options'][row['predicted_answer_index']]} | {row['gold_margin']:+.3f} |")
    errors = {row["query_id"] for row in tasks["errors"]}
    lines.extend(["", "以下仅描述错误题的全head平均位置质量，不解释必要性、充分性或遗忘。", "",
                  "| query_id | 层 | visual总质量 | 首槽质量 | 末槽质量 | CPU参考context相对误差 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in report["attention_records"]:
        if row["query_id"] in errors:
            lines.append(f"| {row['query_id']} | {row['layer']} | {row['mean_visual_mass']:.5f} | {row['mean_first_slot_mass']:.5f} | {row['mean_last_slot_mass']:.5f} | {row['context_reference_relative_l2_error']:.5g} |")
    context_errors = stats([row["context_reference_relative_l2_error"] for row in report["attention_records"]])
    lines.extend(["", f"全部条件CPU参考context与实际SDPA context相对L2误差中位数[范围]：{show(context_errors)}。CPU参考行不是SDPA内部导出的概率。", "",
                  "归档没有所有文本位置的K/V，QA亦只保存末位置Q/K/V。本次离线CPU审计可复核保存行的归一化/mask及参考context相对实际SDPA的误差，但不能用归档张量独立重新计算完整attention行或加权context。", "",
                  "## 解释边界", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(["", "QA未保存完整视觉轨迹；Q/K实际旋转后张量来自原SDPA入口，参考概率另由CPU计算。所有数组source/hash、位置、head及误差明细见JSON。",
                  "", f"分析源码SHA-256：{integrity['analysis_source_sha256']}"])
    return "\n".join(lines) + "\n"


def self_checks():
    values = np.zeros((8, 2, 1, 3), dtype=np.float32)
    wave = np.sin(2 * np.pi * np.arange(8) / 8).astype(np.float32)
    values[:, 0, 0, :] = wave[:, None]
    values[:, 1, 0, :] = -wave[:, None]
    result = spatial_spectrum(values)
    require(result["temporal_mean"] == 0 and result["temporal_full"] > 0, "Opposite-patch cancellation check failed")
    require(np.isclose(result["temporal_full"], result["temporal_spatial_residual"]), "Residual retention check failed")
    fixed = spatial_spectrum(np.ones((8, 2, 3), dtype=np.float32))
    require(fixed["temporal_full"] == 0 and fixed["dc_full"] == 1, "Constant check failed")
    require(np.array_equal(bf16_round(np.array([1.0, -1.0, 0.0], dtype=np.float32)), [1.0, -1.0, 0.0]), "BF16 check failed")
    return ["opposite spatial signs cancel only pooled signal", "full=mean+residual per frequency", "constant temporal energy zero", "BF16 representable values preserved"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, default=Path("outputs/passive_probe_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/passive_probe_v4/analysis"))
    parser.add_argument("--videos", nargs="+")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    requested = set(arguments.videos) if arguments.videos else VIDEOS
    require(requested and requested <= VIDEOS, "Only original development videos are allowed")
    checks = self_checks()
    report = analyze(root, (root / arguments.input_root).resolve(), requested)
    output = (root / arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    (output / "checks.json").write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(requested), "integrity": report["integrity"], "qa": report["task_summary"]["correct"]}, indent=2))


if __name__ == "__main__":
    main()
