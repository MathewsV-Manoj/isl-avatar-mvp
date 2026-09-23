"""Validate the exact-phrase browser shard and its 30x225 landmark clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("signal_dictionary_islrtc_sentences")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_shape") != [30, 225]:
        raise ValueError("sentence manifest feature shape is not [30, 225]")
    total = 0
    for shard in manifest.get("shards", []):
        data = json.loads((root / shard["file"]).read_text(encoding="utf-8"))
        if set(data) != set(shard["keys"]):
            raise ValueError(f"shard key mismatch: {shard['file']}")
        for key, frames in data.items():
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (30, 225) or not np.isfinite(array).all():
                raise ValueError(f"invalid clip: {key}")
            points = array.reshape(30, 75, 3)
            if not np.array_equal(points[:, 33], points[:, 15]) or not np.array_equal(points[:, 54], points[:, 16]):
                raise ValueError(f"wrist anchor mismatch: {key}")
            total += 1
    if total != manifest.get("sign_count"):
        raise ValueError("manifest sign_count mismatch")
    print(f"sentence browser shard: {total} exact phrases; shapes and wrist anchors valid")


if __name__ == "__main__":
    main()
