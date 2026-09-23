"""Reject BridgeConn clips with prolonged loss of both hands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_bridgeconn_candidates.json")
OUTPUT = Path("signal_dictionary_bridgeconn_filtered.json")
REPORT = Path("bridgeconn_filter_report.json")


def main() -> None:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    filtered: dict[str, list[object]] = {}
    rejected = []
    for label, clips in source.items():
        for index, frames in enumerate(clips):
            points = np.asarray(frames, dtype=np.float32).reshape(30, 75, 3)
            collapsed = []
            for start in (33, 54):
                hand = points[:, start : start + 21]
                spread = np.linalg.norm(hand - hand[:, :1], axis=2).max(axis=1)
                collapsed.append(spread < 1e-5)
            both_missing_rate = float(np.logical_and(collapsed[0], collapsed[1]).mean())
            if both_missing_rate > 0.5:
                rejected.append({"label": label, "clip": index, "both_missing_rate": both_missing_rate})
                continue
            filtered.setdefault(label, []).append(frames)

    OUTPUT.write_text(json.dumps(filtered, separators=(",", ":")), encoding="utf-8")
    REPORT.write_text(json.dumps({
        "source": str(INPUT),
        "kept_clips": sum(len(items) for items in filtered.values()),
        "kept_labels": len(filtered),
        "rejected_clips": len(rejected),
        "rejected": rejected,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "kept_clips": sum(len(items) for items in filtered.values()),
        "kept_labels": len(filtered),
        "rejected_clips": len(rejected),
    }))


if __name__ == "__main__":
    main()
