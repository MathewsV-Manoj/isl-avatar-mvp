"""Reject INCLUDE clips whose hands are absent or collapsed for most frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_include_candidates.json")
OUTPUT = Path("signal_dictionary_include_filtered.json")
REPORT = Path("include_filter_report.json")


def main() -> None:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    kept: dict[str, list[object]] = {}
    rejected: list[dict[str, object]] = []
    for label, clips in source.items():
        for index, frames in enumerate(clips):
            points = np.asarray(frames, dtype=np.float32)
            if points.shape != (30, 225) or not np.isfinite(points).all():
                rejected.append({"label": label, "clip": index, "reason": "invalid_shape_or_values"})
                continue
            points = points.reshape(30, 75, 3)
            missing = []
            for start in (33, 54):
                hand = points[:, start : start + 21]
                spread = np.linalg.norm(hand - hand[:, :1], axis=2).max(axis=1)
                missing.append(spread < 1e-5)
            missing_rate = float(np.logical_and(missing[0], missing[1]).mean())
            if missing_rate > 0.5:
                rejected.append({"label": label, "clip": index, "reason": "both_hands_missing", "rate": missing_rate})
                continue
            kept.setdefault(label, []).append(frames)
    OUTPUT.write_text(json.dumps(kept, separators=(",", ":")), encoding="utf-8")
    report = {"source": str(INPUT), "kept_clips": sum(len(v) for v in kept.values()), "kept_labels": len(kept), "rejected_clips": len(rejected), "rejected": rejected}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}))


if __name__ == "__main__":
    main()
