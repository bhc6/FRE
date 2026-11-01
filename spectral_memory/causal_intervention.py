"""Bounded visual-branch interventions with full-vocabulary structural controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from .resource_guard import ResourceGuard, configure_threads


CONDITIONS = (
    ("baseline", 13, "ffn", "observe", 1.0, False),
    ("late27_zero", 27, "ffn", "all", 0.0, False),
    ("late27_temporal_up", 27, "ffn", "temporal", 1.2, False),
    ("middle13_identity", 13, "ffn", "all", 1.0, False),
    ("middle13_zero", 13, "ffn", "all", 0.0, False),
    ("middle13_dc_down", 13, "ffn", "dc", 0.8, False),
    ("middle13_temporal_down", 13, "ffn", "temporal", 0.8, False),
    ("middle13_temporal_up", 13, "ffn", "temporal", 1.2, False),
    ("middle13_high_down", 13, "ffn", "high", 0.8, False),
    ("middle13_high_up", 13, "ffn", "high", 1.2, False),
    ("middle13_low_up", 13, "ffn", "low", 1.2, False),
    ("middle13_temporal_matched_up", 13, "ffn", "temporal", 1.2, True),
    ("middle13_phase_control_up", 13, "ffn", "shifted_high", 1.2, False),
    ("middle13_attention_matched_up", 13, "attention", "temporal", 1.2, True),
    ("early0_high_up", 0, "ffn", "high", 1.2, False),
)
STRUCTURAL_CONTROLS = {"late27_zero", "late27_temporal_up", "middle13_identity"}
DEVELOPMENT_IDS = {"video_10993", "video_1580", "video_2564", "video_3501", "video_6860", "video_8241", "video_874", "video_9253"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def component(values, mode):
    import torch

    if mode in {"all", "observe"}:
        return values
    mean = values.mean(dim=0, keepdim=True)
    if mode == "dc":
        return mean.expand_as(values)
    if mode == "temporal":
        return values - mean
    spectrum = torch.fft.rfft(values, dim=0, norm="ortho")
    frequencies = torch.fft.rfftfreq(values.shape[0], device=values.device)
    if mode in {"high", "shifted_high"}:
        mask = frequencies > 0.25
    elif mode == "low":
        mask = (frequencies > 0) & (frequencies <= 0.25)
    else:
        raise ValueError(f"Unknown intervention component: {mode}")
    result = torch.fft.irfft(spectrum * mask[:, None, None], n=values.shape[0], dim=0, norm="ortho")
    return result.roll(1, dims=0) if mode == "shifted_high" else result


def make_hook(positions, mode, gain, stats, target_norm=None, attention=False):
    import torch

    def hook(module, arguments, output):
        if stats:
            raise RuntimeError("Expected one prefill invocation per intervention")
        states = output[0] if attention else output
        selected = states[0, positions].float()
        direction = component(selected, mode)
        requested_norm = float(torch.linalg.vector_norm(direction))
        scale = 1.0
        if target_norm is not None:
            if requested_norm == 0 and target_norm > 0:
                raise RuntimeError("Cannot match a positive perturbation to a zero component")
            scale = target_norm / requested_norm if requested_norm else 0.0
        modified = (selected + (gain - 1.0) * scale * direction).to(states.dtype)
        actual_delta = modified.float() - selected
        stats.update({
            "branch_l2": float(torch.linalg.vector_norm(selected)), "component_l2": requested_norm,
            "target_component_l2": target_norm, "component_scale": scale,
            "requested_perturbation_l2": abs(gain - 1.0) * scale * requested_norm,
            "actual_perturbation_l2": float(torch.linalg.vector_norm(actual_delta)),
            "selected_shape": list(selected.shape), "selected_elements": selected.numel(),
        })
        if mode == "observe":
            stats["high_component_l2"] = float(torch.linalg.vector_norm(component(selected, "high")))
            return output
        replaced = states.clone()
        replaced[0, positions] = modified
        return (replaced, *output[1:]) if attention else replaced
    return hook


def forward(backend, batch, condition, target_norm=None):
    import torch

    name, layer, branch, mode, gain, matched = condition
    positions = torch.as_tensor(batch.video_positions, device=backend.device)
    stats = {}
    decoder = backend.model.model.language_model.layers[layer]
    module = decoder.mlp if branch == "ffn" else decoder.self_attn
    handle = module.register_forward_hook(make_hook(positions, mode, gain, stats,
                                                   target_norm if matched else None, branch == "attention"))
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            result = backend.model(**batch, use_cache=False, output_hidden_states=False,
                                   output_attentions=False, return_dict=True, logits_to_keep=1)
            logits = result.logits[0, -1].float().cpu().numpy().copy()
            del result
    finally:
        handle.remove()
    if not stats:
        raise RuntimeError("Intervention hook was not called")
    return logits, {**stats, "forward_seconds": time.perf_counter() - started}


def load_frames(root):
    from PIL import Image

    records = {}
    for directory in sorted(root.glob("ffn_qwen3_dev16*")):
        metadata = json.loads((directory / "probe.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            raise ValueError("Incomplete source probe")
        video_id = metadata["video_id"]
        if video_id in records:
            raise ValueError("Duplicate source video")
        manifest = directory / "frames.manifest.jsonl"
        if digest(manifest) != metadata["frames_manifest_sha256"]:
            raise ValueError("Source frame manifest changed")
        frame_record = json.loads(manifest.read_text(encoding="utf-8"))
        frames = []
        for filename, expected in zip(frame_record["frames"], metadata["frame_sha256"], strict=True):
            path = Path(filename)
            if digest(path) != expected:
                raise ValueError("Source frame bytes changed")
            with Image.open(path) as image:
                frames.append(image.convert("RGB").copy())
        if len(frames) != 16:
            raise ValueError("This stage requires exactly sixteen source frames")
        records[video_id] = {"frames": frames, "record": frame_record,
                             "provenance": {"probe_sha256": digest(directory / "probe.json"),
                                            "frames_manifest_sha256": digest(manifest),
                                            "frame_sha256": metadata["frame_sha256"]}}
    if set(records) != DEVELOPMENT_IDS:
        raise ValueError("Expected all original eight development videos")
    return records


def run(arguments):
    configure_threads()
    import numpy as np
    from .evaluate_small import build_prompt
    from .memory_study import in_subset
    from .qwen3_backend import load_backend, DEFAULT_MAX_PIXELS

    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(line) for line in arguments.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions.sort(key=lambda row: row["query_id"])
    if len(questions) != 36 or {row["video_id"] for row in questions} != DEVELOPMENT_IDS:
        raise ValueError("Expected all 36 original development questions")
    if len({row["query_id"] for row in questions}) != 36:
        raise ValueError("Duplicate query ID")
    config = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "incomplete",
              "stage": arguments.stage, "device": "cuda", "conditions": CONDITIONS,
              "questions_sha256": digest(arguments.questions), "expected_questions": 36,
              "protocol_sha256": digest("docs/causal_intervention_protocol_v2.md"),
              "source_sha256": digest(__file__), "primary_candidate": "middle13_high_up",
              "scope": "development_mechanism_screen_not_independent_confirmation"}
    write_json(output / "config.json", config)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for filename in ("causal_intervention.py", "resource_guard.py", "qwen3_backend.py", "evaluate_small.py", "probe_ffn.py"):
        shutil.copy2(Path(__file__).with_name(filename), snapshot / filename)
    shutil.copy2("docs/causal_intervention_protocol_v2.md", snapshot / "protocol.md")
    completed = 0
    try:
        with ResourceGuard(output, "cuda") as guard:
            guard.configure_torch()
            import torch
            guard.phase = "load_inputs"
            frames_by_video = load_frames(arguments.input_root)
            config["inputs"] = {name: value["provenance"] for name, value in frames_by_video.items()}
            write_json(output / "config.json", config)
            guard.phase = "load_model"
            backend = load_backend(device="cuda")
            config["model"] = backend.metadata
            write_json(output / "config.json", config)
            if arguments.stage == "null":
                from .probe_ffn import capture_branches, INSTRUCTION
                for video_id, data in sorted(frames_by_video.items()):
                    for control, index in (("static_first", 0), ("static_last", -1)):
                        guard.phase = f"{video_id}:{control}"
                        batch = backend.prepare_video([data["frames"][index]] * 16,
                                                      data["record"]["timestamps_s"], INSTRUCTION, DEFAULT_MAX_PIXELS)
                        arrays, identities = capture_branches(backend, batch, (0, 13, 27))
                        archive = output / f"{video_id}.{control}.npz"
                        with archive.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                        write_json(output / f"{video_id}.{control}.json", {
                            "video_id": video_id, "control": control, "source_index": index % 16,
                            "source_sha256": data["provenance"]["frame_sha256"][index],
                            "archive_sha256": digest(archive), "identities": identities,
                            "batch": batch.metadata, "synthetic_video_from_real_static_frame": True,
                            "qa_labels_used": False,
                        })
                        completed += 1
                        write_json(output / "status.json", {"status": "running", "completed": completed, "expected": 16})
                        print(f"null {completed}/16 {video_id} {control}", flush=True)
                        del batch
            else:
                (output / "structural_logits").mkdir()
                with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
                    for question in questions:
                        data = frames_by_video[question["video_id"]]
                        safe_question = {"question": question["question"], "options": question["options"]}
                        batch = backend.prepare_video(data["frames"], data["record"]["timestamps_s"],
                                                      build_prompt(safe_question), DEFAULT_MAX_PIXELS)
                        if batch.metadata["input_tokens"] > 1024 or batch.metadata["visual_tokens"] > 512:
                            raise ValueError("Input exceeds bounded intervention budget")
                        letters = [chr(ord("A") + index) for index in range(len(question["options"]))]
                        identifiers = [backend.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in letters]
                        if any(len(value) != 1 for value in identifiers) or len({value[0] for value in identifiers}) != len(letters):
                            raise ValueError("Answers must map to unique single tokens")
                        baseline_logits = None
                        target_norm = None
                        controls = {}
                        for condition in CONDITIONS:
                            name = condition[0]
                            guard.phase = f"{question['query_id']}:{name}"
                            logits, stats = forward(backend, batch, condition, target_norm)
                            if not np.isfinite(logits).all():
                                raise RuntimeError("Nonfinite full-vocabulary logits")
                            if name == "baseline":
                                baseline_logits = logits
                                target_norm = stats["high_component_l2"]
                                controls[name] = logits
                            maximum = float(np.max(np.abs(logits - baseline_logits)))
                            identical = bool(np.array_equal(logits, baseline_logits))
                            if name in STRUCTURAL_CONTROLS:
                                if not identical:
                                    raise RuntimeError(f"Structural/no-op control changed logits: {name} by {maximum}")
                                controls[name] = logits
                            allowed = logits[[value[0] for value in identifiers]]
                            predicted = int(allowed.argmax())
                            answer = question["answer_index"]
                            margin = float(allowed[answer] - np.max(np.delete(allowed, answer)))
                            row = {"query_id": question["query_id"], "video_id": question["video_id"],
                                   "condition": name, "layer": condition[1], "branch": condition[2],
                                   "mode": condition[3], "gain": condition[4], "norm_matched": condition[5],
                                   "query_text": question["question"], "options": question["options"],
                                   "area": question["area"], "tags": question["tags"],
                                   "temporal_state": in_subset({**question, "query_text": question["question"]}, "temporal_state"),
                                   "gold_answer_index": answer, "predicted_answer_index": predicted,
                                   "correct": int(predicted == answer), "option_logits": allowed.tolist(),
                                   "option_token_ids": [value[0] for value in identifiers], "gold_margin": margin,
                                   "full_logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                                   "full_logits_identical_to_baseline": identical,
                                   "full_logits_max_abs_change": maximum, "intervention": stats,
                                   "prompt_sha256": batch.metadata["prompt_sha256"],
                                   "input_tokens": batch.metadata["input_tokens"],
                                   "visual_tokens": batch.metadata["visual_tokens"]}
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            stream.flush()
                            completed += 1
                        filename = hashlib.sha256(question["query_id"].encode()).hexdigest()[:20] + ".npz"
                        with (output / "structural_logits" / filename).open("xb") as handle:
                            np.savez_compressed(handle, **controls)
                        write_json(output / "status.json", {"status": "running", "completed": completed,
                                                            "expected": len(questions) * len(CONDITIONS)})
                        print(f"qa {completed}/{len(questions) * len(CONDITIONS)} {question['query_id']}", flush=True)
                        del batch
            config.update({"status": "complete", "completed_records": completed,
                           "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                           "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())})
            write_json(output / "config.json", config)
            write_json(output / "status.json", {"status": "complete", "completed": completed})
            del backend
            torch.cuda.empty_cache()
    except BaseException as error:
        config.update({"status": "failed", "completed_records": completed, "error": str(error)})
        write_json(output / "config.json", config)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("qa", "null"), required=True)
    parser.add_argument("--input-root", type=Path, default=Path("outputs"))
    parser.add_argument("--questions", type=Path, default=Path("data/perception_test_sample/questions.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
