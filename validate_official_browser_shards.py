"""Validate browser-ready official-derived shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("signal_dictionary_official")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    seen = set()
    clips = 0
    for shard in manifest["shards"]:
        data = json.loads((ROOT / shard["file"]).read_text(encoding="utf-8"))
        if set(data) != set(shard["keys"]):
            raise ValueError(f"{shard['file']}: key mismatch")
        for key, sequences in data.items():
            if key in seen:
                raise ValueError(f"duplicate key: {key}")
            seen.add(key)
            for frames in sequences:
                array = np.asarray(frames, dtype=np.float32)
                if array.shape != (30, 225) or not np.isfinite(array).all():
                    raise ValueError(f"{key}: invalid feature shape")
                clips += 1
    if len(seen) != manifest["sign_count"]:
        raise ValueError("manifest count mismatch")
    print(f"official browser: {clips} clips across {len(seen)} labels in {len(manifest['shards'])} shards")


if __name__ == "__main__":
    main()
