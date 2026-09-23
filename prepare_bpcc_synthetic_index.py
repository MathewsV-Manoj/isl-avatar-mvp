"""Align BPCC sentence annotations to the local ISL pose dictionary.

The output is an index, not a claim of native continuous signing. It is useful
for training sequence ordering, duration, and transition models from validated
isolated clips while preserving missing-token and provenance information.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def key(word: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", word.upper()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200000)
    parser.add_argument("--min-covered", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sign_keys = set()
    for manifest_path in (
        root / "signal_dictionary_cislr/manifest.json",
        root / "signal_dictionary_bridgeconn/manifest.json",
        root / "signal_dictionary_official/manifest.json",
    ):
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sign_keys.update(manifest.get("signs", {}).keys())
    with (root / "signal_dictionary_normalized.json").open(encoding="utf-8") as f:
        sign_keys.update(json.load(f).keys())
    source = root / "datasets/posestitch_isl/BPCC-ISL.csv"
    out_dir = root / "datasets/bpcc_synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.jsonl"
    counts = {"rows_read": 0, "kept": 0, "rejected_short": 0, "covered_tokens": 0, "missing_tokens": 0}
    with source.open(encoding="utf-8", newline="") as src, out_path.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        for row in reader:
            if counts["rows_read"] >= args.limit:
                break
            counts["rows_read"] += 1
            raw = row.get("Processed_Sentence") or row.get("sentence_without_punctuation") or ""
            tokens = [x for x in re.findall(r"[A-Za-z0-9']+", raw.lower()) if x]
            if len(tokens) < args.min_covered:
                counts["rejected_short"] += 1
                continue
            covered, missing = [], []
            for token in tokens[:32]:
                candidate = key(token)
                if candidate in sign_keys:
                    covered.append(candidate)
                else:
                    missing.append(token)
            if len(covered) < args.min_covered:
                counts["rejected_short"] += 1
                continue
            dst.write(json.dumps({
                "source": "PoseStitch-ISL/BPCC-ISL.csv",
                "sentence_id": row.get("sentence_id"),
                "text": row.get("Original_Sentence", ""),
                "tokens": tokens[:32],
                "sign_keys": covered,
                "missing": missing,
                "synthetic": True,
            }, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            counts["covered_tokens"] += len(covered)
            counts["missing_tokens"] += len(missing)
    counts["dictionary_keys"] = len(sign_keys)
    counts["coverage"] = round(counts["covered_tokens"] / max(1, counts["covered_tokens"] + counts["missing_tokens"]), 4)
    (out_dir / "report.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
