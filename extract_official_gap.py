"""Convert the selected official-derived videos into the project landmark format."""

from __future__ import annotations

import json
import os
import re
import threading
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from build_dictionary import FEATURES_PER_FRAME, HAND_LANDMARKS, POSE_LANDMARKS, _fill_landmarks
from build_dictionary import resample_to_fixed_length, smooth_sequence
from reanchor_hands import fix_sequence


ROOT = Path("datasets/official_isl_gap")
SELECTION = Path("official_isl_selected_manifest.json")
OUTPUT = Path("signal_dictionary_official_candidates.json")
REPORT = Path("official_isl_candidate_report.json")
STATE = Path("official_isl_extractor_state.json")


def canonical(value: str) -> str:
    return re.sub(r"^_|_$", "", re.sub(r"[^A-Z0-9]+", "_", value.upper()))


def extract_file_clip(path: Path, holistic: object) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        capture.release()
        return None
    wanted = np.linspace(0, frame_count - 1, min(frame_count, 18), dtype=int).tolist()
    frames = []
    wanted_set = set(wanted)
    for frame_number in range(frame_count):
        ok = capture.grab()
        if not ok:
            continue
        if frame_number not in wanted_set:
            continue
        ok, image = capture.retrieve()
        if not ok:
            continue
        height, width = image.shape[:2]
        if width > 960:
            scale = 960 / width
            image = cv2.resize(image, (960, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        vector = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
        _fill_landmarks(vector, 0, result.pose_landmarks, POSE_LANDMARKS)
        _fill_landmarks(vector, POSE_LANDMARKS * 3, result.left_hand_landmarks, HAND_LANDMARKS)
        _fill_landmarks(vector, (POSE_LANDMARKS + HAND_LANDMARKS) * 3, result.right_hand_landmarks, HAND_LANDMARKS)
        frames.append(vector)
    capture.release()
    return np.stack(frames) if len(frames) >= 2 else None


def main() -> None:
    global ROOT, SELECTION, OUTPUT, REPORT, STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--state", type=Path, default=STATE)
    args = parser.parse_args()
    ROOT, SELECTION, OUTPUT, REPORT, STATE = args.root, args.selection, args.output, args.report, args.state
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]
    clips = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"completed": [], "failures": []}
    completed = set(report.get("completed", []))
    failures = list(report.get("failures", []))
    failed_paths = {row["path"] for row in failures}
    completed -= failed_paths
    failures = []
    inflight = json.loads(STATE.read_text(encoding="utf-8")).get("inflight") if STATE.exists() else None
    if inflight and inflight not in completed:
        completed.add(inflight)
        failures.append({"path": inflight, "error": "interrupted during extraction"})

    def checkpoint() -> None:
        OUTPUT.write_text(json.dumps(clips, separators=(",", ":")), encoding="utf-8")
        REPORT.write_text(json.dumps({"source": "ISLRTC official-derived dictionary", "completed": sorted(completed), "failures": failures, "clips": sum(len(v) for v in clips.values()), "labels": len(clips)}, indent=2), encoding="utf-8")
        STATE.write_text(json.dumps({"inflight": None}), encoding="utf-8")

    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for item in selected:
            path = item["path"]
            if path in completed:
                continue
            video = ROOT / path
            if not video.exists():
                failures.append({"path": path, "error": "downloaded file missing"})
                completed.add(path)
                continue
            STATE.write_text(json.dumps({"inflight": path}), encoding="utf-8")
            timer = threading.Timer(45, lambda: os._exit(124))
            timer.daemon = True
            timer.start()
            try:
                sequence = extract_file_clip(video, holistic)
            except Exception as error:
                sequence = None
                failures.append({"path": path, "error": str(error)})
            finally:
                timer.cancel()
            completed.add(path)
            if sequence is None:
                if not any(row.get("path") == path for row in failures):
                    failures.append({"path": path, "error": "no usable landmarks"})
            else:
                label = canonical(item["label"])
                sequence = smooth_sequence(resample_to_fixed_length(sequence, 30))
                fixed, _ = fix_sequence(sequence)
                clips.setdefault(label, []).append(fixed.tolist())
            if len(completed) % 25 == 0:
                checkpoint()
                print(f"processed {len(completed)}/{len(selected)} clips", flush=True)
    checkpoint()
    print(json.dumps({"completed": len(completed), "failures": len(failures), "clips": sum(len(v) for v in clips.values()), "labels": len(clips)}))


if __name__ == "__main__":
    main()
