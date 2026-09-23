"""Apply the current wrist-anchor policy to already completed CISLR shards.

This repairs landmark JSON directly; it never re-runs MediaPipe or changes a
gloss key. It is safe to run after an interrupted extraction before resuming.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reanchor_hands import fix_sequence


def main() -> None:
    output_dir = Path("signal_dictionary_cislr")
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_shards = changed_clips = 0

    for row in manifest.get("shards", []):
        path = output_dir / row["file"]
        source = json.loads(path.read_text(encoding="utf-8"))
        repaired: dict[str, list] = {}
        shard_changed = False
        for key, frames in source.items():
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (30, 225) or not np.isfinite(array).all():
                raise ValueError(f"Invalid clip in {path}: {key}")
            fixed, _ = fix_sequence(array)
            repaired[key] = fixed.tolist()
            if not np.array_equal(array, fixed):
                shard_changed = True
                changed_clips += 1
        if shard_changed:
            path.write_text(json.dumps(repaired), encoding="utf-8")
            changed_shards += 1
            print(f"Repaired {path.name}", flush=True)

    print(f"Repaired {changed_clips} clips in {changed_shards} completed shards.")


if __name__ == "__main__":
    main()
