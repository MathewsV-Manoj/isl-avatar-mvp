"""Run palm-to-forearm orientation diagnostics on representative CISLR shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cislr-dir", type=Path, default=Path("signal_dictionary_cislr"))
    args = parser.parse_args()

    manifest = json.loads((args.cislr_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["shards"]
    selected = [rows[0], rows[len(rows) // 2], rows[-1]]
    before: list[float] = []
    after: list[float] = []
    collapsed = 0

    for row in selected:
        shard = json.loads((args.cislr_dir / row["file"]).read_text(encoding="utf-8"))
        for frames in shard.values():
            for frame in np.asarray(frames, dtype=float).reshape(-1, 75, 3):
                for side, off, elbow, wrist in (
                    ("L", LH_OFF, L_ELBOW, L_WRIST),
                    ("R", RH_OFF, R_ELBOW, R_WRIST),
                ):
                    hand = frame[off : off + HAND]
                    if np.linalg.norm(hand - hand[0], axis=1).max() < 1e-6:
                        collapsed += 1
                        continue
                    source_angle = angle_deg(hand[9] - hand[0], frame[wrist] - frame[elbow])
                    fixed = corrected_hand(frame, side)
                    fixed_angle = angle_deg(fixed[9] - fixed[0], frame[wrist] - frame[elbow])
                    if source_angle is not None and fixed_angle is not None:
                        before.append(source_angle)
                        after.append(fixed_angle)

    print("sample shards    : " + ", ".join(row["file"] for row in selected))
    print(f"valid hand frames: {len(before)}")
    print(f"collapsed skipped: {collapsed}")
    print("before degrees   : " + ", ".join(f"{key}={value:.3f}" for key, value in summarize(before).items()))
    print("after degrees    : " + ", ".join(f"{key}={value:.6f}" for key, value in summarize(after).items()))
    if max(after, default=180.0) > 1e-4:
        raise SystemExit("orientation correction failed tolerance")


if __name__ == "__main__":
    main()
