"""Filter official-derived clips with prolonged loss of both hands."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_official_candidates.json")
OUTPUT = Path("signal_dictionary_official_filtered.json")
REPORT = Path("official_isl_filter_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    filtered: dict[str, list[object]] = {}
    rejected = []
    for label, clips in source.items():
        for index, frames in enumerate(clips):
            points = np.asarray(frames, dtype=np.float32).reshape(30, 75, 3)
            missing = []
            for start in (33, 54):
                hand = points[:, start : start + 21]
                spread = np.linalg.norm(hand - hand[:, :1], axis=2).max(axis=1)
                missing.append(spread < 1e-5)
            rate = float(np.logical_and(missing[0], missing[1]).mean())
            if rate > 0.5:
                rejected.append({"label": label, "clip": index, "both_missing_rate": rate})
            else:
                filtered.setdefault(label, []).append(frames)
    args.output.write_text(json.dumps(filtered, separators=(",", ":")), encoding="utf-8")
    args.report.write_text(json.dumps({"kept_clips": sum(map(len, filtered.values())), "kept_labels": len(filtered), "rejected_clips": len(rejected), "rejected": rejected}, indent=2), encoding="utf-8")
    print(json.dumps({"kept_clips": sum(map(len, filtered.values())), "kept_labels": len(filtered), "rejected_clips": len(rejected)}))


if __name__ == "__main__":
    main()
