"""Create conservative sentence-context rows from ISL-Fingerspelling metadata."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models/bpcc_gloss_tagger.pt"
CSV_PATH = ROOT / "datasets/isl_fingerspelling/automatic_segments_detection.csv"
OUT = ROOT / "datasets/isl_fingerspelling/sentence_index.jsonl"
REPORT = ROOT / "datasets/isl_fingerspelling/sentence_index_report.json"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def clean(token: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", token.upper()).strip("_")


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    in_vocab = checkpoint["in_vocab"]
    out_vocab = checkpoint["out_vocab"]
    rows = []
    total = 0
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            total += 1
            tokens = words(item.get("transcript_text", ""))
            known = [token for token in tokens if token in in_vocab]
            if len(known) < 2 or len(known) / max(1, len(tokens)) < 0.5:
                continue
            glosses = [clean(token) for token in known if clean(token) in out_vocab]
            if not glosses:
                continue
            rows.append({
                "tokens": known,
                "sign_keys": sorted(set(glosses)),
                "source": "ISL-Fingerspelling",
                "uid": item.get("uid", ""),
                "matched_word": item.get("matched_word", ""),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    report = {
        "source": "kirandevraj/ISL-Fingerspelling",
        "source_license": "CC-BY-NC-4.0",
        "input_rows": total,
        "kept_rows": len(rows),
        "minimum_known_tokens": 2,
        "minimum_known_fraction": 0.5,
        "native_gloss_supervision": False,
        "continuous_pose_supervision": False,
        "output": str(OUT.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
