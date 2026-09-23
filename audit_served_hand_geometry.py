"""Audit every clip exposed by the validated ISL production dictionary."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from production_server import INDEX, clips_for, valid_clip
from verify_hand_orientation import (
    HAND,
    LH_OFF,
    L_ELBOW,
    L_WRIST,
    RH_OFF,
    R_ELBOW,
    R_WRIST,
    angle_deg,
    corrected_hand,
    summarize,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    by_source = Counter()
    collapsed = Counter()
    before: list[float] = []
    after: list[float] = []
    invalid: list[str] = []
    fully_collapsed_labels: defaultdict[str, int] = defaultdict(int)

    for key, record in INDEX.items():
        clips, missing = clips_for([key])
        raw = clips.get(record["gloss"])
        if missing or not valid_clip(raw):
            invalid.append(key)
            continue
        by_source[record["source"]] += 1
        frames = np.asarray(raw, dtype=float).reshape(30, 75, 3)
        both_collapsed = 0
        for frame in frames:
            frame_collapsed = 0
            for side, offset, elbow, wrist in (
                ("L", LH_OFF, L_ELBOW, L_WRIST),
                ("R", RH_OFF, R_ELBOW, R_WRIST),
            ):
                hand = frame[offset : offset + HAND]
                if np.linalg.norm(hand - hand[0], axis=1).max() < 1e-6:
                    collapsed[record["source"]] += 1
                    frame_collapsed += 1
                    continue
                original = angle_deg(hand[9] - hand[0], frame[wrist] - frame[elbow])
                fixed = corrected_hand(frame, side)
                corrected = angle_deg(fixed[9] - fixed[0], frame[wrist] - frame[elbow])
                if original is not None and corrected is not None:
                    before.append(original)
                    after.append(corrected)
            if frame_collapsed == 2:
                both_collapsed += 1
        if both_collapsed > 15:
            fully_collapsed_labels[key] = both_collapsed

    report = {
        "served_signs": len(INDEX),
        "valid_clip_count": sum(by_source.values()),
        "served_by_source": dict(sorted(by_source.items())),
        "collapsed_hand_frames_by_source": dict(sorted(collapsed.items())),
        "invalid_keys": invalid,
        "labels_with_both_hands_collapsed_over_half": dict(sorted(fully_collapsed_labels.items())),
        "orientation_before_degrees": summarize(before),
        "orientation_after_degrees": summarize(after),
    }
    (ROOT / "reports" / "served_hand_geometry.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if invalid:
        raise SystemExit(f"served invalid clips: {invalid[:10]}")
    if fully_collapsed_labels:
        raise SystemExit("served labels exceed collapsed-hand threshold")
    if max(after, default=180.0) > 1e-4:
        raise SystemExit("served palm orientation exceeds tolerance")


if __name__ == "__main__":
    main()
