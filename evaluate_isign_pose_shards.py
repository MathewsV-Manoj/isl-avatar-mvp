"""Validate every native iSign sentence-pose shard before model use."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("datasets/isign_sentence_pose_v2")
    files = sorted(root.glob("*.npz"))
    clips = 0
    bad = []
    min_value = float("inf")
    max_value = float("-inf")
    for path in files:
        with np.load(path, allow_pickle=False) as pack:
            poses = pack["poses"]
            count = len(pack["uids"])
            clips += count
            min_value = min(min_value, float(poses.min()))
            max_value = max(max_value, float(poses.max()))
            if poses.shape != (count, 30, 225) or not np.isfinite(poses).all():
                bad.append(path.name)
    report = {"shards": len(files), "clips": clips, "bad_shards": bad, "min": min_value, "max": max_value, "valid": not bad}
    (root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
