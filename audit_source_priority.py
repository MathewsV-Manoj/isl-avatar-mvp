"""Audit which source wins for overlapping runtime dictionary labels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from production_server import BLOCKED_SOURCE_KEYS, ROOT, SOURCES, canonical


def main() -> None:
    owners: dict[str, list[str]] = defaultdict(list)
    for source, directory in SOURCES:
        manifest = json.loads((ROOT / directory / "manifest.json").read_text(encoding="utf-8"))
        for raw_key in manifest.get("signs", {}):
            if raw_key not in BLOCKED_SOURCE_KEYS.get(source, set()):
                owners[canonical(raw_key)].append(source)
    overlaps = {key: values for key, values in owners.items() if len(values) > 1}
    preferred = {key: values[0] for key, values in owners.items()}
    report = {
        "source_order": [source for source, _ in SOURCES],
        "runtime_keys": len(owners),
        "overlapping_runtime_keys": len(overlaps),
        "selected_by_source": {source: sum(value == source for value in preferred.values()) for source, _ in SOURCES},
        "rule": "The first eligible source in source_order wins; lower-priority sources are fallbacks only.",
    }
    output = ROOT / "reports" / "production_source_priority.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
