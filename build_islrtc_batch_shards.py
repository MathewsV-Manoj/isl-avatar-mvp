"""Build browser shards for the validated ISLRTC sample batch."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "datasets/islrtc_batch_filtered.json"
SELECTION = ROOT / "datasets/islrtc_extraction_selection.json"
OUTPUT = ROOT / "signal_dictionary_islrtc_batch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--source", default="ISLRTC official-derived validated sample batch")
    parser.add_argument("--license", dest="license_name", default="MIT per source dataset card; verify upstream data.gov.in terms before product use")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    selected = {row["label"]: row["path"] for row in json.loads(args.selection.read_text(encoding="utf-8"))["selected"]}
    for label, clips in source.items():
        for frames in clips:
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (30, 225) or not np.isfinite(array).all():
                raise ValueError(f"{label}: invalid clip")
    output = args.output_dir
    output.mkdir(exist_ok=True)
    keys = sorted(source)
    locations = {}
    shards = []
    for number, start in enumerate(range(0, len(keys), 100)):
        batch = keys[start : start + 100]
        filename = f"signs-{number:04d}.json"
        (output / filename).write_text(json.dumps({key: source[key] for key in batch}, separators=(",", ":")), encoding="utf-8")
        shards.append({"file": filename, "count": len(batch), "keys": batch})
        for key in batch:
            locations[key] = {"shard": filename, "gloss": key.lower().replace("_", " "), "source_video": selected.get(key, "")}
    (output / "manifest.json").write_text(json.dumps({
        "format": "isl-avatar-dictionary-v1",
        "source": args.source,
        "license": args.license_name,
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "shards": shards,
        "signs": locations,
    }), encoding="utf-8")
    print(f"ISLRTC batch browser shards: {len(locations)} labels in {len(shards)} files")


if __name__ == "__main__":
    main()
