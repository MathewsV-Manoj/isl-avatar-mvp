"""Inspect a BridgeConn WebDataset shard without extracting its video payload."""

from __future__ import annotations

import json
import tarfile
import argparse
from collections import Counter
from pathlib import Path


LABEL_FIELDS = ("gloss", "label", "word", "text", "sentence", "translation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path, nargs="?", default=Path("datasets/bridgeconn-sign-dictionary-isl/shard_00001-train.tar"))
    args = parser.parse_args()
    shard = args.shard
    report_path = Path(f"bridgeconn_{shard.stem}_report.json")
    if not shard.exists():
        raise SystemExit(f"Missing completed shard: {shard}")

    suffixes: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    samples: list[str] = []
    json_members = 0
    with tarfile.open(shard, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffixes[Path(member.name).suffix.lower() or "[none]"] += 1
            if len(samples) < 20:
                samples.append(member.name)
            if Path(member.name).suffix.lower() != ".json":
                continue
            json_members += 1
            stream = archive.extractfile(member)
            if not stream:
                continue
            try:
                payload = json.load(stream)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            for field in LABEL_FIELDS:
                value = payload.get(field)
                if isinstance(value, str) and value.strip():
                    labels[value.strip()] += 1
                    break
            else:
                transcript = payload.get("transcript")
                if isinstance(transcript, dict):
                    value = transcript.get("text")
                    if isinstance(value, str) and value.strip():
                        labels[value.strip()] += 1

    report = {
        "shard": str(shard),
        "suffix_counts": dict(sorted(suffixes.items())),
        "json_members": json_members,
        "unique_labels_found": len(labels),
        "label_examples": labels.most_common(100),
        "member_examples": samples,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
