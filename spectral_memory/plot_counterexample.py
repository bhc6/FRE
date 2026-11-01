"""Render an explicitly posthoc development counterexample to proxy guarantees."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .data import read_jsonl


def main():
    root = Path.cwd()
    diagnostic_path = root / "outputs/next_iteration_diagnostics/diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    source_path = root / "data/perception_test_frames64/frames.manifest.jsonl"
    source = next(row for row in read_jsonl(source_path) if row["video_id"] == "video_6860")
    action_path = root / "data/perception_test_sample/actions.jsonl"
    actions = [row for row in read_jsonl(action_path)
               if row["video_id"] == "video_6860" and row["action_id"] in (4, 11)]
    methods = ("kcenter", "kcenter_swap_spectral")
    selected = [next(row for row in diagnostic["selections"]
                     if row["video_id"] == "video_6860" and row["layer"] == 28
                     and row["budget_pairs"] == 4 and row["method"] == method)
                for method in methods]
    width, height = 1728, 680
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    regular = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    title = ImageFont.truetype(str(font_path), 26) if font_path.exists() else regular
    draw.text((24, 16), "Better spectral loss and feature radius can still discard observed evidence", fill="#17273c", font=title)
    draw.text((24, 53), "Posthoc development example: video_6860, layer 28, 8-frame memory; not a confirmatory effect", fill="#555555", font=regular)
    source_times = np.asarray(source["source_frame_timestamps_s"])
    frame_hashes = []
    for row_index, row in enumerate(selected):
        top = 102 + row_index * 220
        label = "k-center: spectral loss 0.9193, squared radius 514.41" if row_index == 0 else "Constrained spectral swaps: loss 0.7379, squared radius 421.59"
        draw.text((24, top), label, fill="#17273c", font=regular)
        positions = [position for pair in row["indices"] for position in (2 * pair, 2 * pair + 1)]
        for column, position in enumerate(positions):
            left = 24 + column * 212
            path = source_path.parent / source["frames"][position]
            frame_hashes.append({"position": position, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            with Image.open(path) as frame:
                preview = ImageOps.contain(frame.convert("RGB"), (200, 120))
                sheet.paste(preview, (left, top + 35))
            hit = any(action["start_s"] <= source_times[position] <= action["end_s"] for action in actions)
            if hit:
                draw.rectangle((left - 2, top + 33, left + 202, top + 157), outline="#ba3d20", width=4)
            draw.text((left, top + 165), f"slot {position}; PTS {source_times[position]:.2f}s", fill="#333333", font=small)
    draw.text((24, 559), "Orange outline: source frame inside the official short-action intervals.", fill="#ba3d20", font=regular)
    draw.text((24, 590), "The two annotation instances overlap in one window: moving object(s) [16.59, 17.89]s;", fill="#333333", font=regular)
    draw.text((24, 618), "holding something [16.13, 18.03]s. Interval coverage does not establish recognizability or a QA causal effect.", fill="#333333", font=regular)
    output = root / "outputs/next_iteration_diagnostics"
    sheet.save(output / "proxy_counterexample.png")
    (output / "proxy_counterexample_sources.json").write_text(
        json.dumps({"scope": "posthoc development diagnostic", "video_id": "video_6860",
                    "layer": 28, "budget_pairs": 4, "methods": selected,
                    "official_actions": actions, "frame_hashes": frame_hashes,
                    "inputs": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (diagnostic_path, source_path, action_path)}},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output / "proxy_counterexample.png")


if __name__ == "__main__":
    main()
