"""Check that the local deployable API bundle has its required runtime data."""

from __future__ import annotations

import json
from pathlib import Path

from production_server import ELIGIBILITY_REPORT, INDEX, SENTENCE_SOURCE, SOURCES


ROOT = Path(__file__).resolve().parent


def main() -> None:
    required = [ELIGIBILITY_REPORT, ROOT / "regional_translation.py", ROOT / "isl-avatar-prototype.html"]
    source_counts: dict[str, int] = {}
    source_bytes: dict[str, int] = {}
    for source, directory in (*SOURCES, SENTENCE_SOURCE):
        manifest_path = ROOT / directory / "manifest.json"
        required.append(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = {entry["shard"] for entry in manifest.get("signs", {}).values()}
        for name in shards:
            shard = ROOT / directory / name
            required.append(shard)
            if shard.exists():
                source_bytes[source] = source_bytes.get(source, 0) + shard.stat().st_size
        source_counts[source] = len(manifest.get("signs", {}))

    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, missing
    assert len(INDEX) >= 10_000
    print(json.dumps({
        "release_runtime_indexed_entries": len(INDEX),
        "runtime_sources": source_counts,
        "runtime_pose_shards_mb": round(sum(source_bytes.values()) / (1024 * 1024), 2),
        "runtime_pose_shards_by_source_mb": {key: round(value / (1024 * 1024), 2) for key, value in source_bytes.items()},
        "required_runtime_files": len(required),
        "missing_runtime_files": missing,
    }, indent=2))


if __name__ == "__main__":
    main()
