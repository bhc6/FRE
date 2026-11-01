"""Development-only, query-free exact and radius-constrained memory refinement."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import read_jsonl, write_jsonl
from .memory_selection import (
    _band_errors,
    _gram_coordinates,
    _reconstruct,
    _spectral_setup,
    _spectrum,
    _validate,
    select_memory,
)


METHODS = (
    "uniform",
    "kcenter",
    "spectral_balanced",
    "temporal_centered",
    "spectral_exact",
    "temporal_centered_exact",
    "kcenter_swap_spectral",
    "kcenter_swap_centered",
    "kcenter_radius_exact_spectral",
    "kcenter_radius_exact_centered",
)
DEVELOPMENT_VIDEO_IDS = {
    "video_10993", "video_2564", "video_874", "video_3501",
    "video_6860", "video_9253", "video_8241", "video_1580",
}
MAX_SWAP_ROUNDS = 30


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _radius_swap(candidates, initial_indices, objective_key, radius_limit):
    by_indices = {tuple(row["indices"]): row for row in candidates}
    current = by_indices[tuple(initial_indices)]
    trace = []
    checked_candidates = 0
    converged = False
    for round_index in range(MAX_SWAP_ROUNDS):
        current_set = set(current["indices"])
        eligible = []
        for row in candidates:
            if row["radius_squared"] > radius_limit:
                continue
            proposed_set = set(row["indices"])
            removed = current_set - proposed_set
            added = proposed_set - current_set
            if len(removed) != 1 or len(added) != 1:
                continue
            checked_candidates += 1
            if row[objective_key] >= current[objective_key] - 1e-12 * max(1.0, abs(current[objective_key])):
                continue
            eligible.append((row[objective_key], min(removed), min(added), row))
        if not eligible:
            converged = True
            break
        value, removed, added, current = min(eligible, key=lambda item: item[:3])
        trace.append({
            "round": round_index + 1, "removed": removed, "added": added,
            "objective": value, "radius_squared": current["radius_squared"],
        })
    return current, {
        "swap_trace": trace,
        "feasible_neighbor_objective_lookups": checked_candidates,
        "converged_to_one_swap_local_minimum": converged,
        "maximum_rounds": MAX_SWAP_ROUNDS,
    }


def refine_selections(hidden_states, timestamps_s, budget_pairs=4, seed=17):
    """Compute ten fixed-budget choices without accepting labels or questions.

    All exact and swap methods share one exhaustive candidate-objective table.
    Spectral and centered-time targets use the same endpoint-retaining sets.
    The fixed-radius bound is the radius of the frozen k-center initialization.
    """
    started = perf_counter()
    _, states, times = _validate(hidden_states, timestamps_s, budget_pairs, "uniform", seed)
    if budget_pairs != 4:
        raise ValueError("This development refinement is restricted to four pairs")
    count = len(states)
    candidate_count = (count - 2) * (count - 3) // 2
    if candidate_count > 100_000:
        raise ValueError("Exact refinement exceeds the 100000-candidate CPU guard")
    coordinates, reduction = _gram_coordinates(states - states[:1])
    distances = np.sum((coordinates[:, None] - coordinates[None, :]) ** 2, axis=2)
    original, weights, bands = _spectral_setup(coordinates)
    candidates = []
    for interior in combinations(range(1, count - 1), 2):
        indices = [0, *interior, count - 1]
        reconstructed = _reconstruct(coordinates, times, indices)
        residual = coordinates - reconstructed
        residual -= residual.mean(axis=0, keepdims=True)
        candidates.append({
            "indices": indices,
            "spectral_objective": _band_errors(original, _spectrum(reconstructed), weights, bands, False)[0],
            "centered_objective": float(np.sum(residual**2) / states.size),
            "radius_squared": float(distances[:, indices].min(axis=1).max()),
        })
    by_indices = {tuple(row["indices"]): row for row in candidates}
    base = {
        method: select_memory(hidden_states, times, budget_pairs, method, seed=seed)
        for method in ("uniform", "kcenter", "spectral_balanced", "temporal_centered")
    }
    initial = by_indices[tuple(base["kcenter"]["indices"])]
    initial_radius = initial["radius_squared"]
    radius_limit = initial_radius * (1 + 1e-10) + 1e-12
    feasible = [row for row in candidates if row["radius_squared"] <= radius_limit]
    selected = {method: (by_indices[tuple(value["indices"])], {}) for method, value in base.items()}
    for target, exact_method, swap_method, constrained_exact_method in (
        ("spectral_objective", "spectral_exact", "kcenter_swap_spectral", "kcenter_radius_exact_spectral"),
        ("centered_objective", "temporal_centered_exact", "kcenter_swap_centered", "kcenter_radius_exact_centered"),
    ):
        selected[exact_method] = (min(candidates, key=lambda row: (row[target], row["indices"])), {})
        selected[constrained_exact_method] = (min(feasible, key=lambda row: (row[target], row["indices"])), {})
        selected[swap_method] = _radius_swap(candidates, initial["indices"], target, radius_limit)
    result = {}
    for method in METHODS:
        choice, extra = selected[method]
        constrained = method.startswith(("kcenter_swap", "kcenter_radius_exact"))
        if constrained and choice["radius_squared"] > radius_limit:
            raise ValueError("Constrained selection exceeded the initial k-center radius")
        result[method] = {
            "method": method,
            "indices": choice["indices"],
            "selected_pair_indices": choice["indices"],
            "selected_input_frame_positions": [position for index in choice["indices"] for position in (2 * index, 2 * index + 1)],
            "budget_pairs": budget_pairs,
            "seed": seed,
            "uses_hidden_states": method != "uniform",
            "objectives": {name: choice[name] for name in ("spectral_objective", "centered_objective", "radius_squared")},
            "initial_kcenter_indices": initial["indices"],
            "initial_kcenter_radius_squared": initial_radius,
            "radius_squared_limit": radius_limit if constrained else None,
            "candidate_set_count": len(candidates),
            "radius_feasible_candidate_count": len(feasible),
            "query_access": False,
            "label_access": False,
            "candidate_space": "all four-pair sets retaining first and last pair",
            "tie_break": "objective then lexicographic indices for exact; objective then removed/added indices for swaps",
            "gram_reduction": reduction,
            **extra,
        }
    return {
        "selections": result,
        "candidate_objectives": candidates,
        "shared_objective_evaluations": {"spectral": len(candidates), "centered_time": len(candidates)},
        "joint_selection_elapsed_seconds": perf_counter() - started,
        "timing_scope": "joint ten-method construction plus shared exhaustive objectives and frozen-method audit; not isolated deployment timing",
    }


def build(arguments):
    from PIL import Image

    output = arguments.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refinement memory output must be new or empty")
    frame_rows = read_jsonl(arguments.frames_manifest)
    hidden_rows_list = read_jsonl(arguments.hidden_manifest)
    hidden_rows = {row["video_id"]: row for row in hidden_rows_list}
    if (
        len(frame_rows) != 8
        or len(hidden_rows_list) != 8
        or {row["video_id"] for row in frame_rows} != DEVELOPMENT_VIDEO_IDS
        or set(hidden_rows) != DEVELOPMENT_VIDEO_IDS
    ):
        raise ValueError("This builder is restricted to the original eight development videos")
    if any(row["layer_index"] != 0 for row in hidden_rows_list):
        raise ValueError("This development readout comparison is frozen to layer 0")
    output.mkdir(parents=True, exist_ok=True)
    (output / "payloads").mkdir()
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete",
        "scope": "development-only layer0 budget4 refinement; no new confirmation claim",
        "methods": list(METHODS),
        "budget_pairs": 4,
        "seed": arguments.seed,
        "frames_manifest_sha256": _digest(arguments.frames_manifest),
        "hidden_manifest_sha256": _digest(arguments.hidden_manifest),
        "selection_source_sha256": _digest(Path(__file__).with_name("memory_selection.py")),
        "refinement_source_sha256": _digest(__file__),
        "query_access": "reader accesses only persisted RGB NPZ, question and options",
        "selection_question_access": False,
        "selection_action_or_QA_label_access": False,
        "memory_format": "uncompressed NPZ uint8 RGB frames, float64 grid timestamps, int32 input-frame positions",
        "comparison_compute": "spectral and centered targets evaluated on the same complete 435-set candidate table",
    }
    _write_json(output / "build.json", config)
    memories = []
    candidate_rows = []
    summaries = []
    for frame_row in frame_rows:
        video = frame_row["video_id"]
        hidden_row = hidden_rows[video]
        hidden_path = arguments.hidden_manifest.parent / hidden_row["hidden_path"]
        with np.load(hidden_path, allow_pickle=False) as archive:
            hidden = archive["hidden_states"].copy()
            hidden_times = archive["timestamps_s"].copy()
        times = np.asarray(frame_row["timestamps_s"], dtype=np.float64)
        if hidden.shape[0] != 32 or len(frame_row["frames"]) != 64:
            raise ValueError("Expected the frozen 32 hidden pairs / 64 source samples")
        if not np.allclose(times.reshape(32, 2).mean(axis=1), hidden_times, rtol=1e-5, atol=1e-7):
            raise ValueError("Hidden and source-pair timestamps disagree")
        images = []
        hashes = []
        for filename in frame_row["frames"]:
            path = Path(filename)
            if not path.is_absolute():
                path = arguments.frames_manifest.parent / path
            hashes.append(_digest(path))
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
        if hashes != hidden_row["extraction"]["frame_sha256"]:
            raise ValueError("Source frame bytes differ from hidden extraction")
        images = np.stack(images)
        result = refine_selections(hidden, hidden_times, 4, arguments.seed)
        byte_counts = []
        for candidate in result["candidate_objectives"]:
            candidate_rows.append({"video_id": video, "layer_index": 0, **candidate})
        for method in METHODS:
            selected = result["selections"][method]
            indices = np.asarray(selected["selected_input_frame_positions"], dtype=np.int32)
            retained = images[indices]
            retained_times = times[indices]
            payload_hash = hashlib.sha256(retained.tobytes() + retained_times.tobytes() + indices.tobytes()).hexdigest()
            payload_path = output / "payloads" / f"{payload_hash}.npz"
            if not payload_path.exists():
                np.savez(payload_path, frames=retained, timestamps_s=retained_times, frame_indices=indices)
            archive_bytes = payload_path.stat().st_size
            byte_counts.append(archive_bytes)
            memories.append({
                "video_id": video,
                "method": method,
                "budget_pairs": 4,
                "retained_pairs": 4,
                "frame_indices": indices.tolist(),
                "pair_indices": selected["indices"],
                "memory_path": str(payload_path.resolve()),
                "memory_sha256": _digest(payload_path),
                "payload_sha256": payload_hash,
                "memory_bytes": archive_bytes,
                "frame_count": 8,
                "is_reference": False,
                "selection_seconds": result["joint_selection_elapsed_seconds"],
                "selection_timing_scope": result["timing_scope"],
                "selection": selected,
                "layer_index": 0,
                "hidden_extraction_seconds": hidden_row["extraction"]["elapsed_seconds"],
                "hidden_array_sha256": _digest(hidden_path),
                "observation_input_frames": 64,
                "memory_array_shape": list(retained.shape),
            })
            summaries.append({"video_id": video, "method": method, **selected["objectives"]})
        if len(set(byte_counts)) != 1:
            raise ValueError("Same-video same-budget refinement payload bytes differ")
        print(json.dumps({"video_id": video, "memories": len(memories)}), flush=True)
    method_summary = {}
    for method in METHODS:
        rows = [row for row in summaries if row["method"] == method]
        method_summary[method] = {
            "videos": len(rows),
            "mean_spectral_objective": float(np.mean([row["spectral_objective"] for row in rows])),
            "mean_centered_time_objective": float(np.mean([row["centered_objective"] for row in rows])),
            "mean_radius_squared": float(np.mean([row["radius_squared"] for row in rows])),
        }
    write_jsonl(output / "memories.jsonl", memories)
    write_jsonl(output / "candidate_objectives.jsonl", candidate_rows)
    write_jsonl(output / "selected_objectives.jsonl", summaries)
    _write_json(output / "objective_summary.json", method_summary)
    config.update(
        status="complete",
        video_count=len(frame_rows),
        memory_count=len(memories),
        candidate_set_count=len(candidate_rows),
        memory_manifest_sha256=_digest(output / "memories.jsonl"),
    )
    _write_json(output / "build.json", config)
    print(json.dumps({"status":"complete","memory_manifest":str((output / "memories.jsonl").resolve()),"methods":method_summary}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-manifest", type=Path, required=True)
    parser.add_argument("--hidden-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
