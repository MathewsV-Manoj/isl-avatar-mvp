"""Build an ISL avatar dictionary from the gated CISLR word-video corpus.

Prerequisite: authenticate once with ``isl_env\\Scripts\\hf.exe auth login`` and
accept the CISLR access conditions on Hugging Face.

Examples:
  isl_env\\Scripts\\python import_cislr.py --download --limit 25
  isl_env\\Scripts\\python import_cislr.py --limit 25
  isl_env\\Scripts\\python import_cislr.py

The no-limit command processes the 4,765 official prototype clips. It is an
intentionally long batch job; start with a pilot to validate the result first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import httpx
import mediapipe as mp
import numpy as np
# The Xet client can stall on some Windows networks; regular HTTPS is resumable and reliable here.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import get_hf_file_metadata, hf_hub_download
from huggingface_hub.utils import build_hf_headers

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

REPO_ID = "Exploration-Lab/CISLR"
ARCHIVE_NAME = "CISLR_v1.5-a_videos/CISLR_v1.5-a_videos.zip"
PROTOTYPE_NAME = "prototype.csv"
TARGET_FRAMES = 30
MAX_SOURCE_FRAMES = 90


def download_archive_https(destination: Path) -> Path:
    """Download CISLR with ordinary authenticated HTTP, resumable on interruption."""
    source_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ARCHIVE_NAME}"
    metadata = get_hf_file_metadata(source_url, headers=build_hf_headers())
    if not metadata.location or not metadata.size:
        raise RuntimeError("Hugging Face did not return a download location for CISLR.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    if start > metadata.size:
        partial.unlink()
        start = 0
    print(f"Downloading CISLR archive ({metadata.size / 1_000_000:.0f} MB), resuming at {start / 1_000_000:.0f} MB...", flush=True)
    # The current network accepts small range transfers but stalls on one large stream.
    # Each completed range is persisted, so a later run resumes from the exact byte.
    chunk_size = 5 * 1024 * 1024
    with partial.open("ab" if start else "wb") as stream:
        downloaded = start
        reported = start
        while downloaded < metadata.size:
            end = min(metadata.size - 1, downloaded + chunk_size - 1)
            response = httpx.get(
                metadata.location,
                headers={"Range": f"bytes={downloaded}-{end}"},
                timeout=90,
                follow_redirects=True,
            )
            response.raise_for_status()
            expected = end - downloaded + 1
            if len(response.content) != expected:
                raise RuntimeError(f"Range response was {len(response.content)} bytes; expected {expected}.")
            stream.write(response.content)
            downloaded += len(response.content)
            if downloaded - reported >= 25 * 1024 * 1024 or downloaded == metadata.size:
                print(f"  {downloaded / metadata.size * 100:.1f}%", flush=True)
                reported = downloaded
    if partial.stat().st_size != metadata.size:
        raise RuntimeError(f"Incomplete CISLR download: {partial.stat().st_size} of {metadata.size} bytes.")
    partial.replace(destination)
    return destination


def download_source(source_dir: Path, include_archive: bool) -> tuple[Path, Path | None]:
    source_dir.mkdir(parents=True, exist_ok=True)
    prototype = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=PROTOTYPE_NAME,
            local_dir=source_dir,
        )
    )
    archive = None
    if include_archive:
        archive_path = source_dir / ARCHIVE_NAME
        # A completed source archive is reusable across resumable extraction
        # runs. Avoid a second 1.17 GB transfer after a code-only restart.
        archive = archive_path if archive_path.exists() and archive_path.stat().st_size > 0 else download_archive_https(archive_path)
    return prototype, archive


def read_prototypes(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {"uid", "gloss"} - set(rows[0]):
        raise ValueError(f"Unexpected CISLR prototype columns in {path}")
    return rows[:limit] if limit else rows


def archive_index(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    index: dict[str, zipfile.ZipInfo] = {}
    for entry in archive.infolist():
        if entry.filename.lower().endswith(".mp4"):
            index[Path(entry.filename).stem] = entry
    if not index:
        raise RuntimeError("CISLR archive contains no MP4 videos.")
    return index


def extract_landmarks_from_video(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        capture.release()
        return None
    wanted = set(np.linspace(0, frame_count - 1, min(frame_count, MAX_SOURCE_FRAMES), dtype=int).tolist())
    frames: list[np.ndarray] = []
    frame_number = 0
    holistic_api = mp.solutions.holistic
    with holistic_api.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_number in wanted:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = holistic.process(rgb)
                vector = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
                _fill_landmarks(vector, 0, result.pose_landmarks, POSE_LANDMARKS)
                _fill_landmarks(vector, POSE_LANDMARKS * 3, result.left_hand_landmarks, HAND_LANDMARKS)
                _fill_landmarks(vector, (POSE_LANDMARKS + HAND_LANDMARKS) * 3, result.right_hand_landmarks, HAND_LANDMARKS)
                frames.append(vector)
            frame_number += 1
    capture.release()
    if len(frames) < 2:
        return None
    return _normalize(np.stack(frames))


def extract_dictionary(archive_path: Path, prototypes: list[dict[str, str]], output_path: Path) -> tuple[int, list[str]]:
    result: dict[str, list[list[float]]] = {}
    failures: list[str] = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(prefix="cislr_") as temp_dir:
        entries = archive_index(archive)
        temp_root = Path(temp_dir)
        for number, row in enumerate(prototypes, start=1):
            uid, gloss = row["uid"], row["gloss"].strip()
            entry = entries.get(uid)
            if entry is None:
                failures.append(f"{gloss}: missing video {uid}")
                print(f"[{number}/{len(prototypes)}] {gloss}: missing video", flush=True)
                continue
            video_path = temp_root / f"{uid}.mp4"
            with archive.open(entry) as source, video_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            try:
                sequence = extract_landmarks_from_video(video_path)
            finally:
                video_path.unlink(missing_ok=True)
            if sequence is None:
                failures.append(f"{gloss}: no usable landmarks")
                print(f"[{number}/{len(prototypes)}] {gloss}: no usable landmarks", flush=True)
                continue
            key = gloss.upper().replace(" ", "_")
            processed = smooth_sequence(resample_to_fixed_length(sequence, TARGET_FRAMES))
            # Keep CISLR compatible with the avatar's wrist-anchored hand convention.
            fixed, _ = fix_sequence(processed)
            result[key] = fixed.tolist()
            print(f"[{number}/{len(prototypes)}] {key}: ok", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result), encoding="utf-8")
    return len(result), failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the 1.17 GB CISLR archive before extraction.")
    parser.add_argument("--limit", type=int, help="Process only the first N prototype clips for a pilot run.")
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/cislr"))
    parser.add_argument("--output", type=Path, default=Path("signal_dictionary_cislr_smoothed.json"))
    args = parser.parse_args()

    prototype, downloaded_archive = download_source(args.source_dir, args.download)
    archive_path = downloaded_archive or args.source_dir / ARCHIVE_NAME
    if not archive_path.exists():
        raise SystemExit("CISLR archive is absent. Run again with --download.")
    prototypes = read_prototypes(prototype, args.limit)
    print(f"Extracting {len(prototypes)} CISLR prototype clips from {archive_path}")
    saved, failures = extract_dictionary(archive_path, prototypes, args.output)
    print(f"Saved {saved} signs to {args.output}.")
    if failures:
        (args.output.with_suffix(".failures.txt")).write_text("\n".join(failures), encoding="utf-8")
        print(f"{len(failures)} clips failed; see {args.output.with_suffix('.failures.txt')}")


if __name__ == "__main__":
    main()
