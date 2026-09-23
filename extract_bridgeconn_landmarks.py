"""Extract compatible MediaPipe landmarks from new BridgeConn sign videos.

BridgeConn's supplied DWPose layout is not equivalent to this project's
33-pose + 21-left-hand + 21-right-hand representation. Re-extracting from the
source MP4 files keeps the retargeting coordinate contract consistent.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
import threading
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


ROOT = Path("datasets/bridgeconn-sign-dictionary-isl")
MANIFEST = Path("signal_dictionary_cislr/manifest.json")
OUTPUT = Path("signal_dictionary_bridgeconn_candidates.json")
SOURCE_REPORT = Path("bridgeconn_candidate_manifest.json")
STATE_FILE = Path("bridgeconn_extractor_state.json")
# This MP4 repeatedly stalls OpenCV decoding despite ordinary metadata. Keep it
# out of the batch and preserve the rest of the corpus.
QUARANTINED_SAMPLES = {"shard_00004-train.tar:2416"}


def canonical(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)", "", value.lower())
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def extract_clip(video_bytes: bytes, holistic: object) -> np.ndarray | None:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temporary:
        temporary.write(video_bytes)
        temporary.flush()
        capture = cv2.VideoCapture(temporary.name)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 2:
            capture.release()
            return None
        wanted = np.linspace(0, frame_count - 1, min(frame_count, 36), dtype=int).tolist()
        frames: list[np.ndarray] = []
        for frame_number in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
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
    cislr = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing = {canonical(label) for label in cislr["signs"]}
    clips: dict[str, list[list[list[float]]]] = defaultdict(list)
    provenance: dict[str, list[dict[str, str]]] = defaultdict(list)
    failures: list[str] = []
    completed: set[str] = set()
    inflight_sample: str | None = None
    if OUTPUT.exists() and SOURCE_REPORT.exists():
        clips.update(json.loads(OUTPUT.read_text(encoding="utf-8")))
        saved = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
        provenance.update(saved.get("provenance", {}))
        failures.extend(saved.get("failures", []))
        completed.update(saved.get("completed_samples", []))
        inflight_sample = saved.get("inflight_sample")
    if STATE_FILE.exists():
        inflight_sample = json.loads(STATE_FILE.read_text(encoding="utf-8")).get("inflight_sample", inflight_sample)
    if inflight_sample and inflight_sample not in completed:
        completed.add(inflight_sample)
        failures.append(f"{inflight_sample}: interrupted during extraction")
        inflight_sample = None

    def write_state(sample: str | None) -> None:
        STATE_FILE.write_text(json.dumps({"inflight_sample": sample}), encoding="utf-8")

    def checkpoint() -> None:
        OUTPUT.write_text(json.dumps(dict(clips), separators=(",", ":")), encoding="utf-8")
        SOURCE_REPORT.write_text(json.dumps({
            "source": "bridgeconn/sign-dictionary-isl",
            "candidate_clips": sum(len(items) for items in clips.values()),
            "candidate_labels": len(clips),
            "provenance": provenance,
            "failures": failures,
            "completed_samples": sorted(completed),
            "inflight_sample": inflight_sample,
        }, indent=2), encoding="utf-8")
        write_state(None)

    processed = 0

    with mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for shard in sorted(ROOT.glob("shard_*-train.tar")):
            with tarfile.open(shard, "r") as archive:
                metadata: dict[str, str] = {}
                for member in archive:
                    if member.isfile() and member.name.endswith(".json"):
                        stream = archive.extractfile(member)
                        if not stream:
                            continue
                        payload = json.load(stream)
                        transcript = payload.get("transcript", {})
                        text = transcript.get("text") if isinstance(transcript, dict) else None
                        if isinstance(text, str):
                            metadata[Path(member.name).stem] = canonical(text)

                for member in archive:
                    if not member.isfile() or not member.name.endswith(".mp4"):
                        continue
                    sample_id = Path(member.name).stem
                    label = metadata.get(sample_id, "")
                    if not label or label in existing:
                        continue
                    source_id = f"{shard.name}:{sample_id}"
                    if source_id in completed:
                        continue
                    if source_id in QUARANTINED_SAMPLES:
                        completed.add(source_id)
                        failures.append(f"{source_id}: quarantined decoder stall")
                        checkpoint()
                        continue
                    stream = archive.extractfile(member)
                    if not stream:
                        failures.append(f"{shard.name}:{sample_id}: missing video stream")
                        completed.add(source_id)
                        continue
                    inflight_sample = source_id
                    write_state(inflight_sample)
                    # OpenCV can block indefinitely on a malformed MP4. The
                    # checkpoint above lets the supervisor restart and mark
                    # only this sample as failed.
                    timer = threading.Timer(45, lambda: os._exit(124))
                    timer.daemon = True
                    timer.start()
                    try:
                        sequence = extract_clip(stream.read(), holistic)
                    finally:
                        timer.cancel()
                    processed += 1
                    if sequence is None:
                        failures.append(f"{shard.name}:{sample_id}: no usable landmarks")
                        completed.add(source_id)
                        inflight_sample = None
                        write_state(None)
                        continue
                    sequence = smooth_sequence(resample_to_fixed_length(sequence, 30))
                    fixed, _ = fix_sequence(sequence)
                    clips[label.upper()].append(fixed.tolist())
                    provenance[label.upper()].append({"shard": shard.name, "sample": sample_id})
                    completed.add(source_id)
                    inflight_sample = None
                    write_state(None)
                    if processed % 25 == 0:
                        checkpoint()
                        print(f"processed {processed} new-label clips", flush=True)

    checkpoint()
    report = {
        "source": "bridgeconn/sign-dictionary-isl",
        "candidate_clips": sum(len(items) for items in clips.values()),
        "candidate_labels": len(clips),
        "provenance": provenance,
        "failures": failures,
        "completed_samples": sorted(completed),
    }
    print(json.dumps({key: value for key, value in report.items() if key not in {"provenance", "failures"}}, indent=2))
    if failures:
        print(f"failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
