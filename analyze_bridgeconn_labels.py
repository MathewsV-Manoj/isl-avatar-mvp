"""Compare BridgeConn transcript labels with the existing CISLR sign index."""

from __future__ import annotations

import json
import re
import tarfile
from collections import Counter
from pathlib import Path


ROOT = Path("datasets/bridgeconn-sign-dictionary-isl")
CISLR_MANIFEST = Path("signal_dictionary_cislr/manifest.json")
OUTPUT = Path("bridgeconn_label_overlap_report.json")


def canonical(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)", "", value.lower())
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def main() -> None:
    cislr = json.loads(CISLR_MANIFEST.read_text(encoding="utf-8"))
    existing = {canonical(label) for label in cislr["signs"]}
    labels: Counter[str] = Counter()
    for shard in sorted(ROOT.glob("shard_*-train.tar")):
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                stream = archive.extractfile(member)
                if not stream:
                    continue
                payload = json.load(stream)
                transcript = payload.get("transcript", {})
                text = transcript.get("text") if isinstance(transcript, dict) else None
                if isinstance(text, str) and canonical(text):
                    labels[canonical(text)] += 1

    overlap = sorted(label for label in labels if label in existing)
    new = sorted(label for label in labels if label not in existing)
    report = {
        "shards": len(list(ROOT.glob("shard_*-train.tar"))),
        "samples": sum(labels.values()),
        "unique_bridgeconn_labels": len(labels),
        "overlap_with_cislr": len(overlap),
        "new_vs_cislr": len(new),
        "overlap_labels": overlap,
        "new_labels": new,
        "label_sample_counts": dict(labels.most_common()),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in {"overlap_labels", "new_labels", "label_sample_counts"}}, indent=2))


if __name__ == "__main__":
    main()
