"""Build a server-side denylist for clips with prolonged total hand collapse."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ("cislr", "signal_dictionary_cislr"),
    ("bridgeconn", "signal_dictionary_bridgeconn"),
    ("official", "signal_dictionary_official"),
    ("include", "signal_dictionary_include"),
    ("islrtc", "signal_dictionary_islrtc_batch"),
    ("islrtc_v3", "signal_dictionary_islrtc_batch_v3"),
    ("isign", "signal_dictionary_isign"),
    ("kaggle_social", "signal_dictionary_kaggle_social"),
)


def frame_is_valid(frame: object) -> bool:
    return isinstance(frame, list) and len(frame) == 225 and all(isinstance(value, (int, float)) and math.isfinite(value) for value in frame)


def hand_collapsed(frame: list[float], start: int) -> bool:
    base = start * 3
    wrist = frame[base : base + 3]
    return all(
        sum((frame[(start + point) * 3 + axis] - wrist[axis]) ** 2 for axis in range(3)) <= 1e-10
        for point in range(1, 21)
    )


def usable(sequence: list[list[float]]) -> int:
    return sum(not (hand_collapsed(frame, 33) and hand_collapsed(frame, 54)) for frame in sequence)


def best_sequence(raw: object) -> list[list[float]] | None:
    if not isinstance(raw, list) or not raw:
        return None
    if frame_is_valid(raw[0]):
        return raw
    options = [take for take in raw if isinstance(take, list) and take and frame_is_valid(take[0])]
    return max(options, key=usable, default=None)


def main() -> None:
    reports = {}
    for source, directory_name in SOURCES:
        directory = ROOT / directory_name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        by_shard: dict[str, list[str]] = {}
        for key, item in manifest["signs"].items():
            by_shard.setdefault(item["shard"], []).append(key)
        blocked: list[str] = []
        for shard, keys in by_shard.items():
            data = json.loads((directory / shard).read_text(encoding="utf-8"))
            for key in keys:
                sequence = best_sequence(data.get(key))
                if sequence is None or usable(sequence) < math.ceil(len(sequence) * 0.5):
                    blocked.append(key)
        reports[source] = {
            "total_labels": manifest["sign_count"],
            "blocked_labels": blocked,
            "eligible_labels": manifest["sign_count"] - len(blocked),
        }
    report = {
        "sources": reports,
        "rule": "Reject a label when its best stored take has both hands collapsed in more than half of frames.",
    }
    target = ROOT / "reports" / "clip_eligibility.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({source: {"total_labels": item["total_labels"], "eligible_labels": item["eligible_labels"]} for source, item in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
