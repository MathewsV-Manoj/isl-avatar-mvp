"""Validate every local landmark clip before it can be selected for signing."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ("cislr", "signal_dictionary_cislr"),
    ("bridgeconn", "signal_dictionary_bridgeconn"),
    ("official", "signal_dictionary_official"),
    ("include", "signal_dictionary_include"),
    ("islrtc", "signal_dictionary_islrtc_batch"),
    ("islrtc_v3", "signal_dictionary_islrtc_batch_v3"),
    ("isl500", "signal_dictionary_isl500"),
    ("isign", "signal_dictionary_isign"),
    ("kaggle_social", "signal_dictionary_kaggle_social"),
)


def hand_collapsed(frame: list[float], start: int) -> bool:
    wrist = frame[start * 3 : start * 3 + 3]
    return all(
        sum((frame[(start + point) * 3 + axis] - wrist[axis]) ** 2 for axis in range(3)) < 1e-10
        for point in range(1, 21)
    )


def sequences(raw: list) -> list[list[list[float]]]:
    """Return either one direct clip or the valid takes nested under a word."""
    if not raw:
        return []
    if isinstance(raw[0], list) and len(raw[0]) == 225:
        return [raw]
    return [take for take in raw if take and isinstance(take[0], list) and len(take[0]) == 225]


def main() -> None:
    report = {"sources": {}, "rejected": [], "repairable_hand_frames": 0}
    for source, directory in SOURCES:
        root = ROOT / directory
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = Counter(signs=len(manifest.get("signs", {})), clips=0, takes=0, frames=0, bad_frames=0, collapsed_hand_frames=0, both_hands_collapsed_frames=0, unsupported_shapes=0)
        shard_files = sorted({entry["shard"] for entry in manifest.get("signs", {}).values()})
        for shard_name in shard_files:
            for name, raw in json.loads((root / shard_name).read_text(encoding="utf-8")).items():
                metrics["clips"] += 1
                clips = sequences(raw)
                if not clips:
                    metrics["unsupported_shapes"] += 1
                    report["rejected"].append({"source": source, "sign": name, "reason": "unsupported_shape"})
                    continue
                invalid = False
                for clip in clips:
                    metrics["takes"] += 1
                    for frame in clip:
                        metrics["frames"] += 1
                        if len(frame) != 225 or not all(math.isfinite(float(value)) for value in frame):
                            metrics["bad_frames"] += 1
                            invalid = True
                            continue
                        left_missing = hand_collapsed(frame, 33)
                        right_missing = hand_collapsed(frame, 54)
                        collapsed = left_missing + right_missing
                        metrics["collapsed_hand_frames"] += collapsed
                        metrics["both_hands_collapsed_frames"] += left_missing and right_missing
                if invalid:
                    report["rejected"].append({"source": source, "sign": name, "reason": "invalid_frame"})
        report["sources"][source] = dict(metrics)
        report["repairable_hand_frames"] += metrics["collapsed_hand_frames"]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "dictionary_clip_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
