"""Acquire the public Perception Test sample and preserve its real annotations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .data import write_jsonl


SOURCES = {
    "annotations": "https://storage.googleapis.com/dm-perception-test/zip_data/sample_annotations.zip",
    "videos": "https://storage.googleapis.com/dm-perception-test/zip_data/sample_videos.zip",
    "readme": "https://raw.githubusercontent.com/google-deepmind/perception_test/main/README.md",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, maximum_bytes):
    target = Path(destination)
    if target.exists():
        raise ValueError(f"Download destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    request = Request(url, headers={"User-Agent": "SpectralMemoryProbe/0.1 resource-audit"})
    with urlopen(request, timeout=60) as response:
        advertised = response.headers.get("Content-Length")
        if advertised and int(advertised) > maximum_bytes:
            raise ValueError(f"Resource exceeds explicit download bound: {advertised}")
        response_metadata = {"etag": response.headers.get("ETag"),
                             "last_modified": response.headers.get("Last-Modified")}
        with temporary.open("xb") as stream:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > maximum_bytes:
                    raise ValueError("Download exceeded explicit size bound")
                digest.update(block)
                stream.write(block)
    if advertised and count != int(advertised):
        raise ValueError("Incomplete download")
    temporary.replace(target)
    return {"source": url, "bytes": count, "sha256": digest.hexdigest(), **response_metadata}


def extract_archive(archive_path, output_dir, maximum_bytes):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = []
    with ZipFile(archive_path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > maximum_bytes:
            raise ValueError("Archive exceeds explicit uncompressed size bound")
        for entry in archive.infolist():
            portable = PurePosixPath(entry.filename.replace("\\", "/"))
            if portable.is_absolute() or ".." in portable.parts or ":" in entry.filename:
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive symlinks are unsupported")
            target = (root / Path(*portable.parts)).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive target escaped output directory")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            files.append(target)
    return files


def prepare(output_dir, annotations_only=False):
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    if shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free < 1024**3:
        raise ValueError("At least 1 GiB free space is required for the bounded sample")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": "Perception Test official sample", "status": "incomplete",
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "license": "CC-BY-4.0 for non-code dataset materials, per official README",
                  "sources": {}, "limitations": [
                      "Eight short sample videos are interface and task-design resources, not a long-term benchmark.",
                      "Official answer indices are task ground truth, not counterfactual memory utility.",
                      "Subject and session identities are not supplied; no cross-person generalization claim is valid.",
                      "Annotation frame_rate is metadata; true decoded PTS must be checked before spectral extraction.",
                  ]}
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_dir = output / "archives"
    provenance["sources"]["readme"] = download(SOURCES["readme"], output / "UPSTREAM_README.md", 1024**2)
    readme = (output / "UPSTREAM_README.md").read_text(encoding="utf-8")
    if "CC-BY" not in readme and "CC BY" not in readme and "creativecommons.org/licenses/by/4.0" not in readme:
        raise ValueError("Expected dataset-license reference missing; inspect the upstream README")
    print("Downloading official annotations...", flush=True)
    provenance["sources"]["annotations"] = download(
        SOURCES["annotations"], archive_dir / "sample_annotations.zip", 8 * 1024**2)
    annotation_files = extract_archive(archive_dir / "sample_annotations.zip", output / "annotations", 64 * 1024**2)
    annotation_file = next(path for path in annotation_files if path.name == "sample.json")
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    video_files = {}
    if not annotations_only:
        print("Downloading the bounded official video sample (under 256 MiB)...", flush=True)
        provenance["sources"]["videos"] = download(
            SOURCES["videos"], archive_dir / "sample_videos.zip", 256 * 1024**2)
        extracted = extract_archive(archive_dir / "sample_videos.zip", output / "media", 512 * 1024**2)
        for path in extracted:
            if path.suffix.lower() == ".mp4":
                if path.stem in video_files:
                    raise ValueError("Ambiguous duplicate video filename")
                video_files[path.stem] = path
    videos, questions, actions = [], [], []
    for video_id, entry in annotations.items():
        metadata = entry["metadata"]
        if metadata["video_id"] != video_id:
            raise ValueError("Annotation video identity mismatch")
        if metadata.get("is_cup_game"):
            raise ValueError("Cup-game video requires the official cut-frame mapping before evaluation")
        video_path = video_files.get(video_id)
        if not annotations_only and video_path is None:
            raise ValueError(f"Annotated video missing: {video_id}")
        common = {"dataset": "perception_test_sample", "video_id": video_id,
                  "video_path": video_path.relative_to(output).as_posix() if video_path else None,
                  "benchmark_split": metadata["split"], "subject_id": None, "session_id": None}
        videos.append({**common, "metadata": metadata,
                       "video_sha256": file_sha256(video_path) if video_path else None,
                       "identity_status": "not_provided_by_sample_annotations"})
        for question in entry.get("mc_question", []):
            answer_index = question["answer_id"]
            if not isinstance(answer_index, int) or not 0 <= answer_index < len(question["options"]):
                raise ValueError("Invalid official answer index")
            questions.append({**common,
                              "query_id": f"perception_test:{video_id}:mc:{question['id']}",
                              "question": question["question"], "options": question["options"],
                              "answer_index": answer_index, "answer_text": question["options"][answer_index],
                              "area": question["area"], "reasoning": question["reasoning"],
                              "tags": question["tag"], "label_source": "official_mc_question_annotation",
                              "utility_status": "not_evaluated"})
        for action in entry.get("action_localisation", []):
            actions.append({**common, "action_id": action["id"], "label_id": action["label_id"],
                            "label": action["label"], "start_s": action["timestamps"][0] / 1e6,
                            "end_s": action["timestamps"][1] / 1e6,
                            "frame_ids": action["frame_ids"],
                            "time_basis": "official_annotation_microseconds"})
    write_jsonl(output / "videos.jsonl", videos)
    write_jsonl(output / "questions.jsonl", questions)
    write_jsonl(output / "actions.jsonl", actions)
    memory_questions = [question for question in questions if question["area"].lower() == "memory"]
    write_jsonl(output / "memory_questions.jsonl", memory_questions)
    provenance.update(status="annotations_only" if annotations_only else "complete",
                      video_count=len(videos), downloaded_video_count=len(video_files),
                      mcqa_count=len(questions), memory_mcqa_count=len(memory_questions),
                      action_count=len(actions), no_cup_game_videos=True,
                      annotation_sha256=hashlib.sha256(annotation_file.read_bytes()).hexdigest())
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "ATTRIBUTION.md").write_text(
        "# Attribution\n\nPerception Test dataset, Google DeepMind and the Perception Test authors.\n\n"
        "Source: https://github.com/google-deepmind/perception_test\n\n"
        "Non-code materials: Creative Commons Attribution 4.0 (per upstream README).\n\n"
        "The original annotations are unchanged in annotations/sample.json; JSONL files normalize field names "
        "and convert official action timestamps from microseconds to seconds. Raw videos are unchanged.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "status": provenance["status"],
                      "videos": len(video_files), "questions": len(questions),
                      "memory_questions": len(memory_questions), "actions": len(actions)}, ensure_ascii=False))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/perception_test_sample")
    parser.add_argument("--annotations-only", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.output_dir, arguments.annotations_only)


if __name__ == "__main__":
    main()
