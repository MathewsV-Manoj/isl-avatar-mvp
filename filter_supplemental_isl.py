"""Filter supplemental clips with prolonged loss of both hands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("signal_dictionary_supplemental.json"))
    parser.add_argument("--output", type=Path, default=Path("signal_dictionary_supplemental_filtered.json"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
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
            both_missing = np.logical_and(collapsed[0], collapsed[1])
            if float(both_missing.mean()) > 0.5:
                rejected.append({"label": label, "clip": index, "both_missing_rate": float(both_missing.mean())})
                continue
            filtered.setdefault(label, []).append(frames)

    args.output.write_text(json.dumps(filtered, separators=(",", ":")), encoding="utf-8")
    report = {
        "source": str(args.input),
        "kept_clips": sum(len(items) for items in filtered.values()),
        "kept_labels": len(filtered),
        "rejected_clips": len(rejected),
        "rejected": rejected,
    }
    args.output.with_name("supplemental_isl_filter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}))


if __name__ == "__main__":
    main()
