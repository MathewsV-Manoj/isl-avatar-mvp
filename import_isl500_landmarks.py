"""Import one signer-balanced sample per ISL-DATA class into our pose format.

ISL-DATA stores MMPose whole-body arrays and MediaPipe hand arrays separately.
This importer maps the 17 COCO body joints into the project's 33-pose layout,
joins the two 21-point hand streams, normalizes around the shoulders, and
resamples every clip to 30 frames.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import requests
from huggingface_hub import hf_hub_download


REPO = "ISL500/ISL-DATA"
API = "https://huggingface.co/api/datasets/ISL500/ISL-DATA/tree/main"


def list_files(path: str) -> list[str]:
    url = f"{API}/{path}"
    params = {"recursive": "true", "expand": "false", "limit": 1000}
    files: list[str] = []
    while url:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        files.extend(item["path"] for item in response.json() if item.get("type") == "file")
        url = response.links.get("next", {}).get("url")
        params = None
    return files


def stem_label(path: str) -> str:
    return re.sub(r"__.*$", "", Path(path).stem).strip().upper()


def resample(sequence: np.ndarray, frames: int = 30) -> np.ndarray:
    if len(sequence) == frames:
        return sequence.astype(np.float32)
    positions = np.linspace(0, len(sequence) - 1, frames)
    out = np.empty((frames,) + sequence.shape[1:], dtype=np.float32)
    for i, position in enumerate(positions):
        lo, hi = int(np.floor(position)), int(np.ceil(position))
        if lo == hi:
            out[i] = sequence[lo]
        else:
            alpha = position - lo
            out[i] = (1.0 - alpha) * sequence[lo] + alpha * sequence[hi]
    return out


def fill_pose(whole_body: np.ndarray) -> np.ndarray:
    """Map COCO-17 body joints into MediaPipe's 33-pose index space."""
    # MMPose whole-body order: COCO-17, feet, face, left hand, right hand.
    pose = np.zeros((len(whole_body), 33, 3), dtype=np.float32)
    mapping = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13,
               8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26,
               15: 27, 16: 28}
    for source, target in mapping.items():
        pose[:, target] = whole_body[:, source]
    # Fill the missing torso/face joints from stable neighboring joints.
    pose[:, 1] = pose[:, 0]
    pose[:, 3] = pose[:, 0]
    pose[:, 4] = pose[:, 0]
    pose[:, 6] = (pose[:, 11] + pose[:, 12]) / 2.0
    pose[:, 8] = pose[:, 6]
    pose[:, 9] = pose[:, 6]
    pose[:, 10] = pose[:, 6]
    pose[:, 22] = (pose[:, 23] + pose[:, 24]) / 2.0
    pose[:, 29] = pose[:, 27]
    pose[:, 30] = pose[:, 28]
    pose[:, 31] = pose[:, 27]
    pose[:, 32] = pose[:, 28]
    return pose


def normalize_clip(pose: np.ndarray, hands: np.ndarray) -> np.ndarray | None:
    # The source hand tensor is [T, 2, 21, 3], in left/right image order.
    if len(hands) != len(pose):
        hands = resample(hands, len(pose))
    hands = np.nan_to_num(hands.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pose = np.nan_to_num(pose.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.any(np.abs(hands).sum(axis=(1, 2)) > 1e-5, axis=0) if False else None
    if not np.any(np.abs(pose).sum(axis=(1, 2)) > 1e-5):
        return None
    center = (pose[:, 11] + pose[:, 12]) / 2.0
    width = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=1)
    scale = float(np.median(width[width > 1e-5])) if np.any(width > 1e-5) else 1.0
    scale = max(scale, 1e-5)
    pose = (pose - center[:, None, :]) / scale
    hand_out = np.zeros_like(hands)
    for side, wrist_idx in enumerate((15, 16)):
        wrist = pose[:, wrist_idx]
        raw_wrist = hands[:, side, 0]
        # Source hand coordinates are normalized to the image. Anchor them to
        # the mapped body wrist while retaining their local geometry.
        local = hands[:, side] - raw_wrist[:, None, :]
        local *= 0.45 / max(float(np.nanmedian(np.linalg.norm(local[:, 9] - local[:, 0], axis=1))), 1e-5)
        hand_out[:, side] = local + wrist[:, None, :]
    clip = np.concatenate([pose, hand_out[:, 0], hand_out[:, 1]], axis=1)
    clip = resample(clip, 30)
    if not np.isfinite(clip).all():
        return None
    return clip.reshape(30, 225).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/isl500_candidates.json")
    parser.add_argument("--max-classes", type=int, default=500)
    parser.add_argument("--users", default="001", help="Comma-separated signer ids, e.g. 001,002")
    args = parser.parse_args()
    pose_paths: list[tuple[str, str]] = []
    hand_paths: dict[tuple[str, str], str] = {}
    hand_by_label: dict[tuple[str, str], str] = {}
    user_ids = [int(value.strip()) for value in args.users.split(",") if value.strip()]
    for user_index in user_ids:
        user = f"User{user_index:03d}"
        pose_paths.extend((user, path) for path in list_files(f"Landmarks/Pose/{user}") if path.endswith(".npy"))
        for path in list_files(f"Landmarks/MediaPipe/ISL_DATA_{user.upper()}"):
            if path.endswith(".h5"):
                hand_paths[(user, Path(path).stem)] = path
                hand_by_label.setdefault((user, stem_label(path)), path)
    selected: dict[str, tuple[str, str]] = {}
    for user, path in pose_paths:
        if not path.endswith(".npy"):
            continue
        label = stem_label(path)
        if label not in selected:
            selected[label] = (user, path)
    selected = dict(sorted(selected.items())[: args.max_classes])
    output: dict[str, list[list[float]]] = {}
    provenance = []
    failed = []
    def download_pair(item: tuple[str, str]):
        label, (user, pose_path) = item
        stem = Path(pose_path).stem
        hand_path = hand_paths.get((user, stem)) or hand_by_label.get((user, label))
        if not hand_path:
            return label, None, {"label": label, "reason": "missing hand landmarks"}
        try:
            local_pose = hf_hub_download(REPO, pose_path, repo_type="dataset")
            local_hand = hf_hub_download(REPO, hand_path, repo_type="dataset")
            pose = np.load(local_pose, allow_pickle=False)
            with h5py.File(local_hand, "r") as handle:
                hands = np.asarray(handle["intermediate"])
            clip = normalize_clip(fill_pose(pose), hands)
            if clip is None:
                return label, None, {"label": label, "reason": "empty or invalid pose"}
            return label, clip, None
        except Exception as error:
            return label, None, {"label": label, "reason": str(error)}

    items = list(selected.items())
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_pair, item) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            label, clip, error = future.result()
            if clip is not None:
                output[label] = clip.tolist()
                user, pose_path = selected[label]
                hand_path = hand_paths.get((user, Path(pose_path).stem)) or hand_by_label[(user, label)]
                provenance.append({"label": label, "signer": user, "pose": pose_path, "hands": hand_path, "source": REPO})
            else:
                failed.append(error)
            if index % 25 == 0:
                print(f"processed {index}/{len(selected)}; kept={len(output)} failed={len(failed)}", flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output), encoding="utf-8")
    destination.with_name(destination.stem + "_provenance.json").write_text(json.dumps({"source": REPO, "license": "research/academic only; commercial use requires author permission", "items": provenance, "failed": failed}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "kept": len(output), "failed": len(failed), "output": str(destination)}))


if __name__ == "__main__":
    main()
