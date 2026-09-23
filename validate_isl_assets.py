"""Validate the browser dictionaries produced by the ISL extraction pipeline.

Run while the CISLR builder is active; only manifest-listed, completed shards
are inspected. A non-zero exit means the browser should not be deployed yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FRAME_COUNT = 30
FEATURE_COUNT = 225
POSE_POINTS = 33
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HAND = 33
RIGHT_HAND = 54


def validate_clip(name: str, frames: object) -> tuple[int, int]:
    array = np.asarray(frames, dtype=np.float32)
    if array.shape != (FRAME_COUNT, FEATURE_COUNT):
        raise ValueError(f"{name}: expected {(FRAME_COUNT, FEATURE_COUNT)}, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite landmark values")
    points = array.reshape(FRAME_COUNT, POSE_POINTS + 42, 3)
    if not np.array_equal(points[:, LEFT_HAND], points[:, LEFT_WRIST]):
        raise ValueError(f"{name}: left hand wrist is not anchored to pose wrist")
    if not np.array_equal(points[:, RIGHT_HAND], points[:, RIGHT_WRIST]):
        raise ValueError(f"{name}: right hand wrist is not anchored to pose wrist")

    collapsed = 0
    observed = 0
    for start in (LEFT_HAND, RIGHT_HAND):
        hand = points[:, start : start + 21]
        spread = np.linalg.norm(hand - hand[:, :1], axis=2).max(axis=1)
        collapsed += int((spread < 1e-5).sum())
        observed += len(spread)
    return observed, collapsed


def validate_dictionary(path: Path) -> tuple[int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: expected a non-empty dictionary")
    hand_frames = collapsed = 0
    for name, frames in data.items():
        observed, bad = validate_clip(f"{path.name}:{name}", frames)
        hand_frames += observed
        collapsed += bad
    return len(data), hand_frames, collapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("signal_dictionary_reanchored_v2.json"))
    parser.add_argument("--cislr-dir", type=Path, default=Path("signal_dictionary_cislr"))
    parser.add_argument("--demo", type=Path, default=Path("signal_dictionary_demo.json"))
    args = parser.parse_args()

    total_signs = hand_frames = collapsed = 0
    for label, path in (("base", args.base), ("demo", args.demo)):
        signs, observed, bad = validate_dictionary(path)
        print(f"{label}: {signs} signs, {observed} hand frames, {bad} collapsed")
        total_signs += signs
        hand_frames += observed
        collapsed += bad

    manifest = json.loads((args.cislr_dir / "manifest.json").read_text(encoding="utf-8"))
    shard_rows = manifest.get("shards", [])
    seen: set[str] = set()
    for row in shard_rows:
        shard_path = args.cislr_dir / row["file"]
        signs, observed, bad = validate_dictionary(shard_path)
        keys = set(json.loads(shard_path.read_text(encoding="utf-8")))
        if len(keys) != row["count"] or keys != set(row["keys"]):
            raise ValueError(f"{shard_path}: manifest keys or count do not match shard")
        if seen & keys:
            raise ValueError(f"{shard_path}: duplicate CISLR glosses found")
        seen.update(keys)
        total_signs += signs
        hand_frames += observed
        collapsed += bad

    if len(seen) != manifest.get("sign_count") or seen != set(manifest.get("signs", {})):
        raise ValueError("CISLR manifest sign index does not match completed shards")
    print(f"cislr: {len(seen)} signs in {len(shard_rows)} shards")
    print(f"total validated clips: {total_signs}; collapsed hands: {collapsed}/{hand_frames}")


if __name__ == "__main__":
    main()
