"""Prepare conservative sentence-context augmentation from ISLTranslate metadata.

ISLTranslate provides English sentence/video IDs, not native gloss or pose targets.
This script therefore creates only weak rows whose every token is already in the
production sentence model vocabulary. It never fabricates continuous pose labels.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "datasets/isltranslate/ISLTranslate.csv"
CHECKPOINT = ROOT / "models/bpcc_gloss_tagger.pt"
OUT = ROOT / "datasets/isltranslate/weak_index.jsonl"
REPORT = ROOT / "datasets/isltranslate/weak_index_report.json"
SPARSE_OUT = ROOT / "datasets/isltranslate/weak_index_sparse.jsonl"
SPARSE_REPORT = ROOT / "datasets/isltranslate/weak_index_sparse_report.json"


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def clean(token: str) -> str:
    return "_".join("".join(ch for ch in token.upper() if ch.isalnum()).split())


def main() -> None:
    checkpoint = json.loads("{}")
    import torch

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    in_vocab = checkpoint["in_vocab"]
    out_vocab = checkpoint["out_vocab"]
    rows = []
    sparse_rows = []
    skipped_unknown = 0
    skipped_short = 0
    seen_tokens: set[str] = set()
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            sentence = item.get("text", "").strip()
            words = tokens(sentence)
            if len(words) < 2:
                skipped_short += 1
                continue
            known_words = [word for word in words if word in in_vocab]
            # Retain partially-known sentences only when enough of the
            # context survives. Unknown words are removed, never relabeled.
            if len(known_words) >= 2 and len(known_words) / len(words) >= 0.5:
                sparse_glosses = [clean(word) for word in known_words if clean(word) in out_vocab]
                if sparse_glosses:
                    sparse_rows.append({
                        "tokens": known_words,
                        "sign_keys": sorted(set(sparse_glosses)),
                        "source": "ISLTranslate_sparse_known_tokens",
                        "uid": item.get("uid", ""),
                    })
            if any(word not in in_vocab for word in words):
                skipped_unknown += 1
                continue
            glosses = [clean(word) for word in words if clean(word) in out_vocab]
            if not glosses:
                continue
            rows.append({
                "tokens": words,
                "sign_keys": sorted(set(glosses)),
                "source": "ISLTranslate",
                "uid": item.get("uid", ""),
            })
            seen_tokens.update(words)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    with SPARSE_OUT.open("w", encoding="utf-8") as handle:
        for row in sparse_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    report = {
        "source": "Exploration-Lab/ISLTranslate",
        "source_license": "CC-BY-NC",
        "input_rows": sum(1 for _ in CSV_PATH.open(encoding="utf-8")) - 1,
        "kept_rows": len(rows),
        "skipped_unknown_token_rows": skipped_unknown,
        "skipped_short_rows": skipped_short,
        "known_input_tokens": len(seen_tokens),
        "native_gloss_supervision": False,
        "continuous_pose_supervision": False,
        "output": str(OUT.relative_to(ROOT)),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sparse_report = {
        "source": "Exploration-Lab/ISLTranslate",
        "source_license": "CC-BY-NC",
        "input_rows": report["input_rows"],
        "kept_rows": len(sparse_rows),
        "minimum_known_tokens": 2,
        "minimum_known_fraction": 0.5,
        "native_gloss_supervision": False,
        "continuous_pose_supervision": False,
        "output": str(SPARSE_OUT.relative_to(ROOT)),
    }
    SPARSE_REPORT.write_text(json.dumps(sparse_report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(json.dumps(sparse_report, indent=2))


if __name__ == "__main__":
    main()
