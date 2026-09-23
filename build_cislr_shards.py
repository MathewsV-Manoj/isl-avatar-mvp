"""Run the CISLR avatar build as resumable, browser-loadable sign shards.

The first ten author-selected prototype videos form a quality gate. When they
pass, the script processes all 4,765 CISLR prototypes in 100-sign shards.
Interrupted work is resumed: a valid existing shard is never reprocessed.

Run once in the project virtual environment:
  isl_env\\Scripts\\python build_cislr_shards.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from import_cislr import download_source, extract_dictionary, read_prototypes


def read_valid_shard(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    for sequence in data.values():
        array = np.asarray(sequence, dtype=np.float32)
        if array.shape != (30, 225) or not np.isfinite(array).all():
            return None
    return data


def validate_pilot(path: Path, expected_minimum: int) -> dict:
    data = read_valid_shard(path)
    if data is None or len(data) < expected_minimum:
        raise RuntimeError(f"Pilot validation failed: expected at least {expected_minimum} usable signs in {path}.")
    return data


def write_manifest(output_dir: Path, prototypes: list[dict[str, str]], shards: list[dict]) -> None:
    locations: dict[str, dict[str, str]] = {}
    prototype_by_key = {row["gloss"].upper().replace(" ", "_"): row for row in prototypes}
    for shard in shards:
        for key in shard["keys"]:
            source = prototype_by_key.get(key, {})
            locations[key] = {
                "shard": shard["file"],
                "gloss": source.get("gloss", key),
                "category": source.get("category", ""),
            }
    manifest = {
        "format": "isl-avatar-dictionary-v1",
        "source": "CISLR v1.5 prototype clips",
        "feature_shape": [30, 225],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sign_count": len(locations),
        "shards": shards,
        "signs": locations,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/cislr"))
    parser.add_argument("--output-dir", type=Path, default=Path("signal_dictionary_cislr"))
    parser.add_argument("--pilot-size", type=int, default=10)
    parser.add_argument("--shard-size", type=int, default=100)
    args = parser.parse_args()
    if args.pilot_size < 1 or args.shard_size < 1:
        raise SystemExit("--pilot-size and --shard-size must be positive.")

    prototype_path, archive_path = download_source(args.source_dir, include_archive=True)
    assert archive_path is not None
    prototypes = read_prototypes(prototype_path, limit=None)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilot_path = args.output_dir / "pilot.json"
    if read_valid_shard(pilot_path) is None:
        print(f"Running {args.pilot_size}-sign quality gate...", flush=True)
        extract_dictionary(archive_path, prototypes[: args.pilot_size], pilot_path)
    pilot = validate_pilot(pilot_path, max(1, args.pilot_size // 2))
    print(f"Pilot passed: {len(pilot)} validated signs.", flush=True)

    shards: list[dict] = []
    for start in range(0, len(prototypes), args.shard_size):
        batch = prototypes[start : start + args.shard_size]
        number = start // args.shard_size
        filename = f"signs-{number:04d}.json"
        shard_path = args.output_dir / filename
        data = read_valid_shard(shard_path)
        if data is None:
            print(f"Building shard {number + 1}: signs {start + 1}-{start + len(batch)}...", flush=True)
            extract_dictionary(archive_path, batch, shard_path)
            data = read_valid_shard(shard_path)
        if data is None:
            raise RuntimeError(f"Shard validation failed after extraction: {shard_path}")
        shards.append({"file": filename, "count": len(data), "keys": sorted(data)})
        write_manifest(args.output_dir, prototypes, shards)
        print(f"Shard {number + 1} complete: {len(data)} signs. Manifest updated.", flush=True)

    total = sum(shard["count"] for shard in shards)
    print(f"CISLR shard build complete: {total} signs in {len(shards)} shards.", flush=True)


if __name__ == "__main__":
    main()
