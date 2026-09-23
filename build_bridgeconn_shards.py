"""Build browser-loadable, source-provenanced shards from validated BridgeConn clips."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


INPUT = Path("signal_dictionary_bridgeconn_filtered.json")
PROVENANCE = Path("bridgeconn_candidate_manifest.json")
OUTPUT = Path("signal_dictionary_bridgeconn")
SHARD_SIZE = 100


def valid_clip(frames: object) -> bool:
    array = np.asarray(frames, dtype=np.float32)
    return array.shape == (30, 225) and bool(np.isfinite(array).all())


def main() -> None:
    clips = json.loads(INPUT.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    validated = {key: value for key, value in clips.items() if value and all(valid_clip(clip) for clip in value)}
    if len(validated) != len(clips):
        raise RuntimeError("BridgeConn validation changed after filtering; refusing to build shards.")

    OUTPUT.mkdir(exist_ok=True)
    keys = sorted(validated)
    shards = []
    locations = {}
    for number, start in enumerate(range(0, len(keys), SHARD_SIZE)):
        batch_keys = keys[start : start + SHARD_SIZE]
        filename = f"signs-{number:04d}.json"
        payload = {key: validated[key] for key in batch_keys}
        (OUTPUT / filename).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch_keys), "keys": batch_keys})
        for key in batch_keys:
            locations[key] = {
                "shard": filename,
                "gloss": key.lower().replace("_", " "),
                "source_samples": provenance.get(key, []),
            }

    manifest = {
        "format": "isl-avatar-dictionary-v1",
        "source": "BridgeConn Sign Dictionary ISL",
        "license": "CC-BY-SA-4.0",
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "shards": shards,
        "signs": locations,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"BridgeConn shards: {len(locations)} labels in {len(shards)} files")


if __name__ == "__main__":
    main()
