"""
Numerically verify the hand-orientation correction used by ybot-fingers.html.

The fixed dictionary anchors each MediaPipe hand wrist to the pose wrist, but
the hand cluster can still retain MediaPipe's arbitrary local-frame rotation.
This script measures the angle between each hand's forward axis (landmark
0 -> 9) and the body forearm direction (elbow -> wrist), then applies the same
minimal swing rotation used by the browser retargeter and measures again.

Run:
    python verify_hand_orientation.py signal_dictionary_fixed.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

POSE, HAND = 33, 21
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
LH_OFF, RH_OFF = POSE, POSE + HAND
EPS = 1e-9


def unit(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return None
    return v / n


def angle_deg(a, b):
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0))))


def rotation_between(a, b):
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    v = np.cross(ua, ub)
    c = float(np.dot(ua, ub))
    if c < -1.0 + 1e-8:
        axis = np.cross(ua, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < EPS:
            axis = np.cross(ua, np.array([0.0, 1.0, 0.0]))
        axis = unit(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def corrected_hand(points, side):
    off = LH_OFF if side == "L" else RH_OFF
    elbow = L_ELBOW if side == "L" else R_ELBOW
    wrist = L_WRIST if side == "L" else R_WRIST
    hand = points[off : off + HAND].copy()
    source = hand[9] - hand[0]
    target = points[wrist] - points[elbow]
    rot = rotation_between(source, target)
    if rot is None:
        return hand
    return (hand - hand[0]) @ rot.T + points[wrist]


def summarize(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main(path):
    data = json.loads(Path(path).read_text())
    before, after, collapsed = [], [], 0
    for seq in data.values():
        arr = np.array(seq, dtype=float).reshape(len(seq), POSE + 2 * HAND, 3)
        for frame in arr:
            for side, off, elbow, wrist in (
                ("L", LH_OFF, L_ELBOW, L_WRIST),
                ("R", RH_OFF, R_ELBOW, R_WRIST),
            ):
                hand = frame[off : off + HAND]
                if np.linalg.norm(hand - hand[0], axis=1).max() < 1e-6:
                    collapsed += 1
                    continue
                base_angle = angle_deg(hand[9] - hand[0], frame[wrist] - frame[elbow])
                fixed = corrected_hand(frame, side)
                fixed_angle = angle_deg(fixed[9] - fixed[0], frame[wrist] - frame[elbow])
                if base_angle is not None and fixed_angle is not None:
                    before.append(base_angle)
                    after.append(fixed_angle)

    print(f"file             : {path}")
    print(f"valid hand frames: {len(before)}")
    print(f"collapsed skipped: {collapsed}")
    print("before degrees   : " + ", ".join(f"{k}={v:.3f}" for k, v in summarize(before).items()))
    print("after degrees    : " + ", ".join(f"{k}={v:.6f}" for k, v in summarize(after).items()))
    if max(after, default=180.0) > 1e-4:
        raise SystemExit("orientation correction failed tolerance")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "signal_dictionary_fixed.json")
