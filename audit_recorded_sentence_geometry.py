"""Geometry gate for the recorded multiword clips exposed by production_server."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from production_server import INDEX, SENTENCE_SOURCE, clips_for, valid_sequence
from verify_hand_orientation import HAND, LH_OFF, L_ELBOW, L_WRIST, RH_OFF, R_ELBOW, R_WRIST, angle_deg, corrected_hand, summarize


ROOT = Path(__file__).resolve().parent


def main() -> None:
    sentence_records = [(key, record) for key, record in INDEX.items() if record["source"] == SENTENCE_SOURCE[0]]
    invalid: list[str] = []
    collapsed: list[str] = []
    after: list[float] = []
    for key, record in sentence_records:
        clips, missing = clips_for([key])
        clip = clips.get(record["gloss"])
        if missing or not valid_sequence(clip):
            invalid.append(key)
            continue
        frames = np.asarray(clip, dtype=float).reshape(30, 75, 3)
        both_collapsed = 0
        for frame in frames:
            left = frame[LH_OFF : LH_OFF + HAND]
            right = frame[RH_OFF : RH_OFF + HAND]
            if np.ptp(left, axis=0).max() < 1e-6 and np.ptp(right, axis=0).max() < 1e-6:
                both_collapsed += 1
            for side, offset, elbow, wrist in (("L", LH_OFF, L_ELBOW, L_WRIST), ("R", RH_OFF, R_ELBOW, R_WRIST)):
                hand = frame[offset : offset + HAND]
                if np.ptp(hand, axis=0).max() < 1e-6:
                    continue
                value = angle_deg(corrected_hand(frame, side)[9] - corrected_hand(frame, side)[0], frame[wrist] - frame[elbow])
                if value is not None:
                    after.append(value)
        if both_collapsed > 15:
            collapsed.append(key)

    report = {
        "recorded_sentence_clips": len(sentence_records),
        "valid_clip_count": len(sentence_records) - len(invalid),
        "invalid_keys": invalid,
        "both_hands_collapsed_over_half": collapsed,
        "orientation_after_degrees": summarize(after),
    }
    (ROOT / "reports" / "recorded_sentence_geometry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if invalid or collapsed or max(after, default=180.0) > 1e-4:
        raise SystemExit("recorded sentence geometry gate failed")


if __name__ == "__main__":
    main()
