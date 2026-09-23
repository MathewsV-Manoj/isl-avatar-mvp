"""Incrementally convert verified INCLUDE video archives to avatar landmarks.

Each output clip uses the project's 30 x 225 MediaPipe contract. Archive and
video paths are retained so that data from this CC-BY source is auditable.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from build_dictionary import (
    FEATURES_PER_FRAME,
    HAND_LANDMARKS,
    POSE_LANDMARKS,
    _fill_landmarks,
    _normalize,
    resample_to_fixed_length,
    smooth_sequence,
)
from reanchor_hands import fix_sequence


ARCHIVES = Path("datasets/include/videos")
OUTPUT = Path("signal_dictionary_include_candidates.json")
REPORT = Path("include_candidate_manifest.json")


def canonical(path: str) -> str:
    # Archives sometimes include an auxiliary subfolder below the numbered
    # label directory (for example, "5. Beautiful/Extra/video.MOV").
    # Select the numbered label component rather than the deepest directory.
    parents = Path(path).parts[:-1]
    label = next((part for part in reversed(parents) if re.match(r"^\d+\.\s*", part)), Path(path).parent.name)
    label = re.sub(r"^\d+\.\s*", "", label)
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_").upper()


def extract_clip(data: bytes, suffix: str, holistic: object) -> np.ndarray | None:
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(data)
        temporary.flush()
        capture = cv2.VideoCapture(temporary.name)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            capture.release()
            return None
        wanted = np.linspace(0, count - 1, min(count, 36), dtype=int)
        frames: list[np.ndarray] = []
        for index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                continue
            result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            vector = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
            _fill_landmarks(vector, 0, result.pose_landmarks, POSE_LANDMARKS)
            _fill_landmarks(vector, POSE_LANDMARKS * 3, result.left_hand_landmarks, HAND_LANDMARKS)
            _fill_landmarks(vector, (POSE_LANDMARKS + HAND_LANDMARKS) * 3, result.right_hand_landmarks, HAND_LANDMARKS)
            frames.append(vector)
        capture.release()
    return _normalize(np.stack(frames)) if len(frames) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-archives", type=int)
    parser.add_argument("--rebuild", action="store_true", help="Re-extract verified archives after parser changes.")
    args = parser.parse_args()
    clips: dict[str, list[object]] = defaultdict(list)
    provenance: dict[str, list[dict[str, str]]] = defaultdict(list)
    completed: set[str] = set()
    failures: list[str] = []
    if not args.rebuild and OUTPUT.exists() and REPORT.exists():
        clips.update(json.loads(OUTPUT.read_text(encoding="utf-8")))
        old = json.loads(REPORT.read_text(encoding="utf-8"))
        provenance.update(old.get("provenance", {}))
        completed.update(old.get("completed_samples", []))
        failures.extend(old.get("failures", []))

    archives = sorted(ARCHIVES.glob("*.zip"))
    if args.max_archives:
        archives = archives[: args.max_archives]
    processed = 0
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or member.filename.rsplit(".", 1)[-1].lower() not in {"mov", "mp4", "avi"}:
                        continue
                    source_id = f"{archive_path.name}:{member.filename}"
                    if source_id in completed:
                        continue
                    label = canonical(member.filename)
                    try:
                        sequence = extract_clip(archive.read(member), Path(member.filename).suffix, holistic)
                        if sequence is None:
                            raise ValueError("no usable landmarks")
                        sequence = smooth_sequence(resample_to_fixed_length(sequence, 30))
                        fixed, _ = fix_sequence(sequence)
                        clips[label].append(fixed.tolist())
                        provenance[label].append({"archive": archive_path.name, "video": member.filename})
                    except Exception as error:  # Keep a bad clip from blocking the full corpus.
                        failures.append(f"{source_id}: {error}")
                    completed.add(source_id)
                    processed += 1
                    if processed % 10 == 0:
                        OUTPUT.write_text(json.dumps(dict(clips), separators=(",", ":")), encoding="utf-8")
                        REPORT.write_text(json.dumps({"source": "AI4Bharat/INCLUDE", "license": "CC-BY-4.0", "provenance": provenance, "completed_samples": sorted(completed), "failures": failures}, indent=2), encoding="utf-8")
                        print(f"processed={processed} clips={sum(len(v) for v in clips.values())}", flush=True)

    OUTPUT.write_text(json.dumps(dict(clips), separators=(",", ":")), encoding="utf-8")
    REPORT.write_text(json.dumps({"source": "AI4Bharat/INCLUDE", "license": "CC-BY-4.0", "provenance": provenance, "completed_samples": sorted(completed), "failures": failures}, indent=2), encoding="utf-8")
    print(json.dumps({"processed": processed, "labels": len(clips), "clips": sum(len(v) for v in clips.values()), "failures": len(failures)}))


if __name__ == "__main__":
    main()
