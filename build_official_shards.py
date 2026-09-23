"""Build lazy browser shards from validated official-derived clips."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_official_filtered.json")
SELECTION = Path("official_isl_selected_manifest.json")
OUTPUT = Path("signal_dictionary_official")


def main() -> None:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    selection = {row["label"]: row["path"] for row in json.loads(SELECTION.read_text(encoding="utf-8"))["selected"]}
    for label, clips in source.items():
        for frames in clips:
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (30, 225) or not np.isfinite(array).all():
                raise ValueError(f"{label}: invalid clip")
    OUTPUT.mkdir(exist_ok=True)
    keys = sorted(source)
    locations = {}
    shards = []
    for number, start in enumerate(range(0, len(keys), 100)):
        batch = keys[start : start + 100]
        filename = f"signs-{number:04d}.json"
        (OUTPUT / filename).write_text(json.dumps({key: source[key] for key in batch}, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch), "keys": batch})
        for key in batch:
            locations[key] = {"shard": filename, "gloss": key.lower().replace("_", " "), "source_video": selection.get(key, "")}
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "format": "isl-avatar-dictionary-v1",
        "source": "ISLRTC official-derived dictionary subset",
        "license": "Verify original ISLRTC conditions before product use; no resale/profiteering without permission",
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "shards": shards,
        "signs": locations,
    }), encoding="utf-8")
    print(f"official browser shards: {len(locations)} labels in {len(shards)} files")


if __name__ == "__main__":
    main()
