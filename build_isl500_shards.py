"""Promote validated ISL500 variation clips as a separately attributed shard."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("datasets/isl500_candidates.json"))
    parser.add_argument("--base", type=Path, default=Path("signal_dictionary_fixed.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("signal_dictionary_isl500"))
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8"))
    selected = {}
    for key, frames in source.items():
        array = np.asarray(frames, dtype=np.float32)
        if array.shape != (30, 225) or not np.isfinite(array).all():
            raise ValueError(f"{key}: invalid clip")
        if key not in base:
            selected[key] = frames
    output = args.output_dir
    output.mkdir(exist_ok=True)
    keys = sorted(selected)
    locations = {}
    shards = []
    for number, start in enumerate(range(0, len(keys), 100)):
        batch = keys[start : start + 100]
        filename = f"signs-{number:04d}.json"
        (output / filename).write_text(json.dumps({key: selected[key] for key in batch}, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch), "keys": batch})
        for key in batch:
            locations[key] = {"shard": filename, "gloss": key.lower().replace("_", " ")}
    (output / "manifest.json").write_text(json.dumps({
        "format": "isl-avatar-dictionary-v1",
        "source": "ISL500/ISL-DATA landmark variation corpus",
        "license": "research/academic only; commercial use requires author permission",
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "base_overlap_excluded": len(source) - len(selected),
        "shards": shards,
        "signs": locations,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": len(source), "promoted": len(selected), "overlap_excluded": len(source) - len(selected)}))


if __name__ == "__main__":
    main()
