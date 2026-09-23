"""Validate the lazy-loadable BridgeConn browser dictionary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("signal_dictionary_bridgeconn")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    expected = set(manifest["signs"])
    observed: set[str] = set()
    clips = 0
    for shard in manifest["shards"]:
        data = json.loads((ROOT / shard["file"]).read_text(encoding="utf-8"))
        if set(data) != set(shard["keys"]):
            raise ValueError(f"{shard['file']}: manifest key mismatch")
        for key, sequences in data.items():
            if key in observed:
                raise ValueError(f"{key}: duplicate key across browser shards")
            observed.add(key)
            for frames in sequences:
                array = np.asarray(frames, dtype=np.float32)
                if array.shape != (30, 225) or not np.isfinite(array).all():
                    raise ValueError(f"{key}: invalid browser clip")
                clips += 1
    if observed != expected or len(observed) != manifest["sign_count"]:
        raise ValueError("manifest sign count does not match browser shards")
    print(f"bridgeconn browser: {clips} clips across {len(observed)} labels in {len(manifest['shards'])} shards")


if __name__ == "__main__":
    main()
