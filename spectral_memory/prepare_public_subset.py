"""Fetch a frozen, bounded Perception Test validation subset using HTTP ZIP ranges."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

from .data import read_jsonl, write_jsonl
from .prepare_resources import download, file_sha256


ROOT = "https://storage.googleapis.com/dm-perception-test/"
VIDEO_URL = ROOT + "zip_data/valid_videos.zip"
SELECTION_SEED = "spectral-confirmation-20261101:"
SOURCES = {
    "annotations": ROOT + "zip_data/mc_question_valid_annotations.zip",
    "cut_valid": ROOT + "misc/cut_frame_mapping_valid.json",
    "cut_train": ROOT + "misc/cut_frame_mapping_train.json",
    "readme": "https://raw.githubusercontent.com/google-deepmind/perception_test/main/README.md",
}


class HttpRangeReader(io.RawIOBase):
    """Seek within an immutable remote ZIP without ever accepting a full-body response."""

    def __init__(self, url: str, maximum_bytes: int):
        self.url = url
        self.position = 0
        self.maximum_bytes = maximum_bytes
        self.reserved_request_bytes = 0
        self.successful_bytes = 0
        self.ranges = []
        with urlopen(Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
            self.last_modified = response.headers.get("Last-Modified")
        if not self.etag:
            raise ValueError("An archive ETag is required to bind all range requests to one object")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        bases = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise ValueError("Invalid range-reader seek")
        self.position = bases[whence] + offset
        return self.position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Reader is closed")
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b""
        if size > 32 * 1024**2:
            raise ValueError("Single ZIP range request exceeds the 32 MiB guard")
        start, end = self.position, self.position + size - 1
        for attempt in range(3):
            if self.reserved_request_bytes + size > self.maximum_bytes:
                raise ValueError("Remote ZIP requests would exceed the explicit byte budget")
            self.reserved_request_bytes += size
            request = Request(self.url, headers={
                "Range": f"bytes={start}-{end}", "If-Match": self.etag,
                "User-Agent": "SpectralMemoryProbe/0.1 bounded-range-import",
            })
            try:
                with urlopen(request, timeout=90) as response:
                    expected_range = f"bytes {start}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server ignored or changed the requested byte range; full archive rejected")
                    if response.headers.get("ETag") != self.etag:
                        raise ValueError("Archive changed between range requests")
                    block = response.read(size + 1)
                    if len(block) != size:
                        raise ValueError("Range response length mismatch")
                self.position += size
                self.successful_bytes += size
                self.ranges.append({"start": start, "end_inclusive": end, "bytes": size,
                                    "sha256": hashlib.sha256(block).hexdigest()})
                return block
            except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("Unreachable download retry state")


def select_videos(annotations, cuts, archive_entries, excluded_ids):
    eligible, excluded = [], []
    for video_id, entry in annotations.items():
        metadata = entry["metadata"]
        if metadata["video_id"] != video_id or metadata["split"] != "valid":
            raise ValueError("Official validation annotation identity mismatch")
        reasons = []
        if video_id in excluded_ids:
            reasons.append("development_video")
        if video_id not in archive_entries:
            reasons.append("not_in_video_archive")
        if video_id not in cuts:
            reasons.append("missing_official_cut_rule")
        elif not isinstance(cuts[video_id], int) or cuts[video_id] != -1:
            reasons.append("requires_official_cut_or_invalid_rule")
        if metadata.get("is_cup_game"):
            reasons.append("cup_game")
        if not entry.get("mc_question"):
            reasons.append("no_official_mcqa")
        if reasons:
            excluded.append({"video_id": video_id, "reasons": reasons})
        else:
            eligible.append(video_id)
    memory_ids = [video_id for video_id in eligible if any(
        question["area"] == "memory" for question in annotations[video_id]["mc_question"])]
    memory_set = set(memory_ids)
    other_ids = [video_id for video_id in eligible if video_id not in memory_set]
    rank = lambda video_id: hashlib.sha256((SELECTION_SEED + video_id).encode("utf-8")).hexdigest()
    if min(len(memory_ids), len(other_ids)) < 20:
        raise ValueError("The declared two strata do not contain twenty eligible videos each")
    selected = sorted(memory_ids, key=rank)[:20] + sorted(other_ids, key=rank)[:20]
    selection = {
        "selection_seed": SELECTION_SEED,
        "rule": "Within each stratum sort SHA256(UTF8(seed + video_id)) ascending; take first 20; memory stratum first",
        "strata": "20 videos with at least one official area=memory MCQ; 20 with no memory MCQ",
        "selection_inputs": "video identity, official split, cut rule, cup-game flag, question presence/area only",
        "answer_values_used_for_selection": False, "model_outputs_used_for_selection": False,
        "usage_role": "reserved_confirmation_after_method_freeze", "selected_video_ids": selected,
        "development_excluded_video_ids": sorted(excluded_ids), "eligible_count": len(eligible),
        "eligible_memory_count": len(memory_ids), "eligible_nonmemory_count": len(other_ids),
        "selected": [{"video_id": video_id, "selection_rank_sha256": rank(video_id),
                      "stratum": "memory" if video_id in memory_set else "nonmemory",
                      "archive_member": archive_entries[video_id].filename,
                      "compressed_bytes": archive_entries[video_id].compress_size,
                      "uncompressed_bytes": archive_entries[video_id].file_size,
                      "zip_crc32": f"{archive_entries[video_id].CRC:08x}",
                      "compression_type": archive_entries[video_id].compress_type,
                      "official_cut_frame": cuts[video_id]} for video_id in selected],
    }
    return selection, excluded


def prepare(output_dir, development_manifest, maximum_bytes=1_000_000_000):
    output = Path(output_dir).resolve()
    development_path = Path(development_manifest).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    if maximum_bytes <= 0 or maximum_bytes >= 3_000_000_000:
        raise ValueError("Set a positive total request budget below 3 GB")
    if shutil.disk_usage(Path.cwd()).free < 2 * maximum_bytes:
        raise ValueError("Insufficient space for the bounded import and derived frame preparation")
    development = read_jsonl(development_path)
    excluded_ids = {entry["video_id"] for entry in development}
    development_hashes = {entry.get("video_sha256") for entry in development if entry.get("video_sha256")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "sources").mkdir()
    (output / "media").mkdir()
    provenance = {
        "dataset": "Perception Test official validation diagnostic subset", "status": "incomplete",
        "usage_role": "reserved_confirmation_after_method_freeze", "benchmark_split": "valid",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {}, "maximum_requested_archive_bytes": maximum_bytes,
        "development_manifest_sha256": file_sha256(development_path),
        "license": "CC-BY-4.0 for non-code materials, per official README",
        "limitations": ["Memory-stratified diagnostic subset, not a representative full-benchmark score.",
                        "Official subjects/sessions unavailable; no cross-subject independence claim.",
                        "Clip identity and content hashes are held out from the eight-video development set.",
                        "Raw official answer indices are not measured memory utility.",
                        "All cut-required and cup-game videos are excluded; retained videos have official cut=-1."]}
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    source_files = {}
    for name, url in SOURCES.items():
        filename = PurePosixPath(url).name
        source_files[name] = output / "sources" / filename
        provenance["sources"][name] = download(url, source_files[name], 8 * 1024**2)
    with ZipFile(source_files["annotations"]) as annotation_archive:
        raw_annotations = annotation_archive.read("mc_question_valid.json")
    if len(raw_annotations) > 32 * 1024**2:
        raise ValueError("Unexpected annotation expansion")
    (output / "sources" / "mc_question_valid.json").write_bytes(raw_annotations)
    provenance["annotation_json_sha256"] = hashlib.sha256(raw_annotations).hexdigest()
    annotations = json.loads(raw_annotations)
    cuts = json.loads(source_files["cut_valid"].read_text(encoding="utf-8"))
    train_cuts = json.loads(source_files["cut_train"].read_text(encoding="utf-8"))
    development_audit = []
    for entry in development:
        split = entry["benchmark_split"]
        mapping = train_cuts if split == "train" else cuts if split == "valid" else {}
        development_audit.append({"video_id": entry["video_id"], "benchmark_split": split,
                                  "official_cut_frame": mapping.get(entry["video_id"]),
                                  "is_cup_game": entry.get("metadata", {}).get("is_cup_game"),
                                  "video_sha256": entry.get("video_sha256")})
    write_jsonl(output / "development_cut_audit.jsonl", development_audit)
    reader = HttpRangeReader(VIDEO_URL, maximum_bytes)
    videos, questions = [], []
    try:
        with ZipFile(reader) as archive:
            entries = {}
            for entry in archive.infolist():
                member = PurePosixPath(entry.filename)
                if member.suffix.lower() != ".mp4":
                    continue
                video_id = member.stem
                if not re.fullmatch(r"video_\d+", video_id) or video_id in entries:
                    raise ValueError("Ambiguous or unexpected archive video identity")
                entries[video_id] = entry
            selection, excluded = select_videos(annotations, cuts, entries, excluded_ids)
            selection["annotation_json_sha256"] = provenance["annotation_json_sha256"]
            selection["cut_mapping_sha256"] = provenance["sources"]["cut_valid"]["sha256"]
            selection_bytes = json.dumps(selection, ensure_ascii=False, indent=2).encode("utf-8")
            (output / "selection.json").write_bytes(selection_bytes)
            provenance["selection_sha256"] = hashlib.sha256(selection_bytes).hexdigest()
            write_jsonl(output / "selection_exclusions.jsonl", excluded)
            compressed_bytes = sum(entry["compressed_bytes"] for entry in selection["selected"])
            raw_bytes = sum(entry["uncompressed_bytes"] for entry in selection["selected"])
            if compressed_bytes + reader.reserved_request_bytes + 1024**2 > maximum_bytes or raw_bytes > maximum_bytes:
                raise ValueError("Declared subset exceeds the download or extraction byte budget")
            provenance["archive"] = {
                "url": VIDEO_URL, "full_archive_bytes": reader.size, "etag": reader.etag,
                "last_modified": reader.last_modified, "full_archive_downloaded": False,
                "selected_compressed_bytes": compressed_bytes, "selected_video_bytes": raw_bytes,
                "full_archive_sha256": None, "hash_note": "Full archive was not downloaded; range and member hashes are retained",
            }
            provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            print(json.dumps({"stage": "frozen_selection", "videos": 40,
                              "questions": sum(len(annotations[item]["mc_question"]) for item in selection["selected_video_ids"]),
                              "selected_compressed_bytes": compressed_bytes,
                              "selection_sha256": provenance["selection_sha256"]}), flush=True)
            for index, chosen in enumerate(selection["selected"], start=1):
                video_id = chosen["video_id"]
                entry = entries[video_id]
                target = output / "media" / f"{video_id}.mp4"
                temporary = target.with_suffix(".mp4.partial")
                digest, crc, count = hashlib.sha256(), 0, 0
                with archive.open(entry) as source, temporary.open("xb") as destination:
                    while block := source.read(8 * 1024**2):
                        count += len(block)
                        if count > entry.file_size:
                            raise ValueError("Member expanded beyond its declared byte size")
                        destination.write(block)
                        digest.update(block)
                        crc = zlib.crc32(block, crc)
                if count != entry.file_size or crc != entry.CRC:
                    raise ValueError("Video member size/CRC check failed")
                video_hash = digest.hexdigest()
                if video_hash in development_hashes or any(row["video_sha256"] == video_hash for row in videos):
                    raise ValueError("Duplicate video content violates the frozen independent subset")
                temporary.replace(target)
                common = {
                    "dataset": "perception_test_valid_confirmation40", "video_id": video_id,
                    "video_path": target.relative_to(output).as_posix(), "benchmark_split": "valid",
                    "subject_id": None, "session_id": None, "usage_role": "confirmation",
                    "official_cut_frame": -1, "cut_policy": "excluded_all_cut_required_videos",
                }
                videos.append({**common, "metadata": annotations[video_id]["metadata"],
                               "video_sha256": video_hash, "video_bytes": count,
                               "identity_status": "subject_and_session_not_provided_by_official_annotations",
                               "selection_stratum": chosen["stratum"], "source_archive_member": entry.filename,
                               "source_archive_etag": reader.etag, "source_zip_crc32": f"{crc:08x}"})
                for question in annotations[video_id]["mc_question"]:
                    answer = question["answer_id"]
                    if not isinstance(answer, int) or not 0 <= answer < len(question["options"]):
                        raise ValueError("Invalid official answer index")
                    questions.append({**common, "query_id": f"perception_test:{video_id}:mc:{question['id']}",
                                      "question": question["question"], "options": question["options"],
                                      "answer_index": answer, "answer_text": question["options"][answer],
                                      "area": question["area"], "reasoning": question["reasoning"],
                                      "tags": question["tag"], "label_source": "official_mc_question_valid_annotation",
                                      "utility_status": "not_evaluated", "selection_stratum": chosen["stratum"]})
                write_jsonl(output / "videos.jsonl", videos)
                write_jsonl(output / "questions.jsonl", questions)
                print(json.dumps({"stage": "download", "completed": index, "total": 40,
                                  "video_id": video_id, "video_bytes": count,
                                  "successful_archive_range_bytes": reader.successful_bytes}), flush=True)
    finally:
        provenance["archive_range_successful_bytes"] = reader.successful_bytes
        provenance["archive_range_reserved_bytes_including_retries"] = reader.reserved_request_bytes
        write_jsonl(output / "archive_ranges.jsonl", reader.ranges)
        provenance["archive_ranges_sha256"] = file_sha256(output / "archive_ranges.jsonl")
        reader.close()
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    memory_questions = [question for question in questions if question["area"] == "memory"]
    write_jsonl(output / "memory_questions.jsonl", memory_questions)
    provenance.update(status="complete", video_count=len(videos), mcqa_count=len(questions),
                      memory_mcqa_count=len(memory_questions), no_development_video_id_or_hash_overlap=True,
                      video_manifest_sha256=file_sha256(output / "videos.jsonl"),
                      question_manifest_sha256=file_sha256(output / "questions.jsonl"))
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "ATTRIBUTION.md").write_text(
        "# Attribution\n\nPerception Test, Google DeepMind and the Perception Test authors.\n\n"
        "Official source: https://github.com/google-deepmind/perception_test\n\n"
        "Dataset non-code materials: CC-BY-4.0 per the saved upstream README.\n\n"
        "Selected original video bytes and QA answer labels are unchanged. Fields are normalized to JSONL. "
        "The frozen subset excludes all cut-required videos and all eight development videos. "
        "Selection uses official question areas, never model outputs or answer values. "
        "This memory-stratified subset does not represent the complete benchmark distribution.\n",
        encoding="utf-8")
    print(json.dumps({"stage": "complete", "output": str(output), "videos": len(videos),
                      "questions": len(questions), "memory_questions": len(memory_questions)}), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/perception_confirm")
    parser.add_argument("--development-manifest", default="data/perception_test_sample/videos.jsonl")
    parser.add_argument("--maximum-bytes", type=int, default=1_000_000_000)
    arguments = parser.parse_args()
    prepare(arguments.output_dir, arguments.development_manifest, arguments.maximum_bytes)


if __name__ == "__main__":
    main()
