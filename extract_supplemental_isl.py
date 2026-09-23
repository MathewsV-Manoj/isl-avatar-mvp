"""Extract landmark sequences from the downloaded supplemental ISL clips.

The output keeps every clip grouped by normalized label instead of collapsing
multiple signers into one example. This makes it suitable for later training
and signer-variation analysis without changing the browser dictionary format.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def extract_clip(path: Path, holistic: object) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        capture.release()
        return None
    wanted = set(np.linspace(0, frame_count - 1, min(frame_count, 36), dtype=int).tolist())
    frames: list[np.ndarray] = []
    frame_number = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame_number in wanted:
            result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            vector = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
            _fill_landmarks(vector, 0, result.pose_landmarks, POSE_LANDMARKS)
            _fill_landmarks(vector, POSE_LANDMARKS * 3, result.left_hand_landmarks, HAND_LANDMARKS)
            _fill_landmarks(vector, (POSE_LANDMARKS + HAND_LANDMARKS) * 3, result.right_hand_landmarks, HAND_LANDMARKS)
            frames.append(vector)
        frame_number += 1
    capture.release()
    return _normalize(np.stack(frames)) if len(frames) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("datasets/isl-isolated-40words"))
    parser.add_argument("--output", type=Path, default=Path("signal_dictionary_supplemental.json"))
    args = parser.parse_args()

    rows = list(csv.DictReader((args.root / "metadata.csv").open(encoding="utf-8", newline="")))
    clips: dict[str, list[list[list[float]]]] = defaultdict(list)
    failures: list[str] = []
    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for number, row in enumerate(rows, start=1):
            label = row["normalized_word"].strip().upper().replace(" ", "_")
            video = args.root / row["video_path"]
            if not video.exists():
                failures.append(f"{label}: missing {row['video_path']}")
                continue
            sequence = extract_clip(video, holistic)
            if sequence is None:
                failures.append(f"{label}: no usable landmarks in {row['video_path']}")
                continue
            sequence = smooth_sequence(resample_to_fixed_length(sequence, 30))
            fixed, _ = fix_sequence(sequence)
            clips[label].append(fixed.tolist())
            if number % 25 == 0 or number == len(rows):
                print(f"processed {number}/{len(rows)} clips", flush=True)

    args.output.write_text(json.dumps(dict(clips), separators=(",", ":")), encoding="utf-8")
    report = {
        "source": "vidit031/isl-isolated-40words",
        "clips": sum(len(items) for items in clips.values()),
        "labels": len(clips),
        "failures": failures,
    }
    args.output.with_name("supplemental_isl_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}))
    if failures:
        print(f"failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
