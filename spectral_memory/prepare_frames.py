"""Resample videos on an explicit time grid while retaining decoded source PTS."""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from .data import read_jsonl, write_jsonl
from .prepare_resources import file_sha256


def prepare_video_frames(video_path, output_dir, frame_count=64, max_side=512):
    import cv2

    if frame_count < 16 or frame_count % 2:
        raise ValueError("frame_count must be even and >=16")
    if max_side < 64:
        raise ValueError("max_side must be >=64")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if source_count < frame_count:
            raise ValueError("Upsampling with duplicate source frames is excluded from this pilot")
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_count - 1)
        success, final_frame = capture.read()
        if not success:
            raise ValueError("Cannot decode the video endpoint")
        final_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()
        if not success:
            raise ValueError("Cannot decode the first video frame")
        first_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        if not np.isfinite([first_pts, final_pts]).all() or final_pts <= first_pts:
            raise ValueError("Decoder did not expose valid presentation timestamps")
        targets = np.linspace(first_pts, final_pts, frame_count)
        paths, source_times, source_indices = [], [], []
        previous_frame = None
        previous_pts = None
        previous_index = None
        current_index = 0
        target_index = 0
        all_pts = []
        while success:
            current_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not np.isfinite(current_pts) or (previous_pts is not None and current_pts <= previous_pts):
                raise ValueError("Decoded presentation timestamps must be strictly increasing")
            all_pts.append(current_pts)
            while target_index < frame_count and targets[target_index] <= current_pts + 1e-8:
                target = targets[target_index]
                if previous_pts is not None and abs(previous_pts - target) <= abs(current_pts - target):
                    chosen_frame, chosen_pts, chosen_index = previous_frame, previous_pts, previous_index
                else:
                    chosen_frame, chosen_pts, chosen_index = frame, current_pts, current_index
                height, width = chosen_frame.shape[:2]
                factor = min(1.0, max_side / max(height, width))
                resized = cv2.resize(chosen_frame, (max(1, round(width * factor)), max(1, round(height * factor))),
                                     interpolation=cv2.INTER_AREA)
                target_path = output / f"frame_{target_index:04d}.jpg"
                if target_path.exists():
                    raise ValueError("Frame output already exists")
                if not cv2.imwrite(str(target_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Failed to write a resampled frame")
                paths.append(str(target_path.resolve()))
                source_times.append(chosen_pts)
                source_indices.append(int(chosen_index))
                target_index += 1
            previous_frame, previous_pts, previous_index = frame, current_pts, current_index
            success, frame = capture.read()
            current_index += 1
        if len(paths) != frame_count or len(set(source_indices)) != frame_count:
            raise ValueError("Could not produce distinct source frames for all grid positions")
        errors = np.abs(targets - np.asarray(source_times))
        maximum_gap = float(np.max(np.diff(all_pts)))
        if errors.max() > maximum_gap / 2 + 1e-6:
            raise ValueError("Nearest-frame resampling error exceeds the source timestamp tolerance")
        return {
            "frames": paths, "timestamps_s": targets.tolist(),
            "source_frame_timestamps_s": source_times, "source_frame_indices": source_indices,
            "event_start_s": first_pts, "event_end_s": final_pts,
            "sampling": {
                "method": "explicit_uniform_time_grid_with_nearest_decoded_frame",
                "frame_count": frame_count, "resampled_fps": float(1 / (targets[1] - targets[0])),
                "source_metadata_fps": metadata_fps, "decoded_source_frames": len(all_pts),
                "source_pts_interval_min_s": float(np.min(np.diff(all_pts))),
                "source_pts_interval_max_s": maximum_gap,
                "max_abs_resampling_error_s": float(errors.max()),
                "temporal_lowpass_applied": False,
                "covers_decoded_first_and_last_frame": source_indices[0] == 0 and source_indices[-1] == source_count - 1,
                "spatial_max_side": max_side, "jpeg_quality": 95,
            },
        }
    finally:
        capture.release()


def prepare(manifest, output_dir, frame_count=64, max_side=512):
    source_manifest = Path(manifest).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frame output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(source_manifest)
    results = []
    seen_ids = set()
    for record in records:
        video_id = record["video_id"]
        if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id) or video_id in seen_ids:
            raise ValueError("Video IDs must be unique safe directory names")
        seen_ids.add(video_id)
        video = Path(record["video_path"])
        if not video.is_absolute():
            video = source_manifest.parent / video
        digest = file_sha256(video)
        if record.get("video_sha256") and digest != record["video_sha256"]:
            raise ValueError("Source video hash mismatch")
        started = time.perf_counter()
        prepared = prepare_video_frames(video, output / video_id, frame_count, max_side)
        result = {**record, **prepared, "video_path": str(video.resolve()), "video_sha256": digest,
                  "sample_id": f"{record.get('dataset', 'video')}:{video_id}:full",
                  "subject_id": record.get("subject_id"), "session_id": record.get("session_id")}
        results.append(result)
        print(json.dumps({"video_id": video_id, "frames": len(result["frames"]),
                          "max_timing_error_s": result["sampling"]["max_abs_resampling_error_s"],
                          "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    write_jsonl(output / "frames.manifest.jsonl", results)
    (output / "preparation.json").write_text(json.dumps({
        "source_manifest": str(source_manifest), "source_manifest_sha256": file_sha256(source_manifest),
        "video_count": len(results), "frame_count_per_video": frame_count,
        "timing_definition": "explicit resampled grid; original decoded PTS and indices preserved per frame",
        "limitation": "No temporal low-pass filtering; future spectral interpretation must control frame-sampling aliasing.",
    }, indent=2), encoding="utf-8")
    return output / "frames.manifest.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-count", type=int, default=64)
    parser.add_argument("--max-side", type=int, default=512)
    arguments = parser.parse_args()
    result = prepare(arguments.manifest, arguments.output_dir, arguments.frame_count, arguments.max_side)
    print(str(result))


if __name__ == "__main__":
    main()
