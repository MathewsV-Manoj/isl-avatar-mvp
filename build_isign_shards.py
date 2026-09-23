"""Build source-attributed lazy shards from validated iSign word candidates."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(r"E:\ISL_Project_Datasets\isign\isign_word_candidates.json.gz"))
    parser.add_argument("--base", type=Path, default=Path("signal_dictionary_fixed.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("signal_dictionary_isign"))
    args = parser.parse_args()
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    selected = {}
    for key, frames in source.items():
        array = np.asarray(frames, dtype=np.float32)
        if array.shape != (30, 225) or not np.isfinite(array).all():
            raise ValueError(f"{key}: invalid clip")
        if key not in base:
            selected[key] = frames
    args.output_dir.mkdir(exist_ok=True)
    keys = sorted(selected)
    locations = {}
    shards = []
    for number, start in enumerate(range(0, len(keys), 100)):
        batch = keys[start : start + 100]
        filename = f"signs-{number:04d}.json"
        (args.output_dir / filename).write_text(json.dumps({key: selected[key] for key in batch}, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch), "keys": batch})
        for key in batch:
            locations[key] = {"shard": filename, "gloss": key.lower().replace("_", " ")}
    manifest = {
        "format": "isl-avatar-dictionary-v1",
        "source": "Exploration-Lab/iSign pose-format corpus",
        "license": "cc-by-nc-sa-4.0; research use; commercial use requires permission",
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "base_overlap_excluded": len(source) - len(selected),
        "shards": shards,
        "signs": locations,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": len(source), "promoted": len(selected), "overlap_excluded": len(source) - len(selected)}))


if __name__ == "__main__":
    main()
