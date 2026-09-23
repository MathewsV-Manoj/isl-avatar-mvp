"""Build browser shards from filtered INCLUDE landmark clips."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_include_filtered.json")
PROVENANCE = Path("include_candidate_manifest.json")
OUTPUT = Path("signal_dictionary_include")


def main() -> None:
    clips = json.loads(INPUT.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    valid = {key: value for key, value in clips.items() if value and all(np.asarray(clip, dtype=np.float32).shape == (30, 225) and np.isfinite(np.asarray(clip, dtype=np.float32)).all() for clip in value)}
    if len(valid) != len(clips):
        raise RuntimeError("INCLUDE validation changed after filtering.")
    OUTPUT.mkdir(exist_ok=True)
    locations: dict[str, dict[str, object]] = {}
    shards = []
    keys = sorted(valid)
    for number, start in enumerate(range(0, len(keys), 100)):
        batch = keys[start : start + 100]
        filename = f"signs-{number:04d}.json"
        (OUTPUT / filename).write_text(json.dumps({key: valid[key] for key in batch}, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch), "keys": batch})
        for key in batch:
            locations[key] = {"shard": filename, "gloss": key.lower().replace("_", " "), "source_samples": provenance.get("provenance", {}).get(key, [])}
    (OUTPUT / "manifest.json").write_text(json.dumps({"format": "isl-avatar-dictionary-v1", "source": "AI4Bharat INCLUDE", "license": "CC-BY-4.0", "feature_shape": [30, 225], "created_at": datetime.now(timezone.utc).isoformat(), "sign_count": len(locations), "shards": shards, "signs": locations}), encoding="utf-8")
    print(f"INCLUDE shards: {len(locations)} labels in {len(shards)} files")


if __name__ == "__main__":
    main()
