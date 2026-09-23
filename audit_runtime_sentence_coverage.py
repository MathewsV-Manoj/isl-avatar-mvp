"""Measure exact word-lookup completeness across held local English sentences.

This does not score ISL grammar. A sentence counts only when every non-omitted
token has a deterministic dictionary resolution.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from production_server import resolve_phrase


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "datasets" / "posestitch_isl" / "BPCC-ISL.csv"


def main() -> None:
    total = complete = resolved_terms = missing_terms = 0
    missing = Counter()
    with CORPUS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = (row.get("Processed_Sentence") or row.get("sentence_without_punctuation") or "").strip()
            if not text:
                continue
            terms, unresolved = resolve_phrase(text)
            total += 1
            resolved_terms += len(terms)
            missing_terms += len(unresolved)
            missing.update(unresolved)
            if terms and not unresolved:
                complete += 1
    report = {
        "purpose": "Dictionary-resolution completeness only; this is not ISL grammatical correctness.",
        "sentences": total,
        "sentences_with_complete_lookup": complete,
        "complete_lookup_rate": complete / max(1, total),
        "resolved_terms": resolved_terms,
        "missing_terms": missing_terms,
        "top_missing_terms": [{"term": term, "count": count} for term, count in missing.most_common(100)],
    }
    target = ROOT / "reports" / "runtime_sentence_coverage.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sentences", "sentences_with_complete_lookup", "complete_lookup_rate", "missing_terms")}, indent=2))


if __name__ == "__main__":
    main()
