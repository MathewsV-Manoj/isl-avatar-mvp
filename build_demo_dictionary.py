"""Create a compact, browser-first dictionary for the multilingual demo.

The full CISLR dictionary stays in 100-sign lazy-load shards. This tool copies
only the clips used by the curated meeting phrase pack into one small file so
their first playback does not wait for a full shard download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reanchor_hands import fix_sequence


DEMO_KEYS = ("FINE", "SLOWER", "MEET", "TODAY", "NAME", "UNDERSTAND")


def main() -> None:
    source_dir = Path("signal_dictionary_cislr")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    locations = manifest["signs"]
    cached_shards: dict[str, dict] = {}
    selected: dict[str, list] = {}

    for key in DEMO_KEYS:
        location = locations.get(key)
        if location is None:
            raise RuntimeError(f"Demo sign is not yet available: {key}")
        shard_name = location["shard"]
        if shard_name not in cached_shards:
            cached_shards[shard_name] = json.loads((source_dir / shard_name).read_text(encoding="utf-8"))
        frames = cached_shards[shard_name].get(key)
        array = np.asarray(frames, dtype=np.float32)
        if array.shape != (30, 225) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid frame data for {key}")
        # Reapply the current shared hand policy so a bundle made from an older
        # shard cannot retain an all-zero, detached missing hand.
        fixed, _ = fix_sequence(array)
        selected[key] = fixed.tolist()

    target = Path("signal_dictionary_demo.json")
    target.write_text(json.dumps(selected, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(selected)} demo clips to {target} ({target.stat().st_size / 1024:.1f} KiB).")


if __name__ == "__main__":
    main()
