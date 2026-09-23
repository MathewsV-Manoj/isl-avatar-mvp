"""Find frequent unresolved input from the local sentence corpus.

This is a lookup-coverage audit, not a linguistic ISL-quality evaluation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from production_server import GLOSS_OMISSIONS, resolve_phrase


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "datasets" / "isltranslate" / "ISLTranslate.csv"
REPORT = ROOT / "reports" / "corpus_lookup_coverage.json"


def main() -> None:
    total = resolved_inputs = partially_resolved = 0
    missing = Counter()
    examples: dict[str, str] = {}
    with CORPUS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            total += 1
            _, unresolved = resolve_phrase(text)
            if not unresolved:
                resolved_inputs += 1
                continue
            partially_resolved += 1
            for token in unresolved:
                if token in GLOSS_OMISSIONS:
                    continue
                missing[token] += 1
                examples.setdefault(token, text)

    top = [
        {"token": token, "count": count, "example": examples[token]}
        for token, count in missing.most_common(250)
    ]
    report = {
        "corpus": str(CORPUS.relative_to(ROOT)),
        "inputs": total,
        "fully_resolved_inputs": resolved_inputs,
        "inputs_with_unresolved_content": partially_resolved,
        "full_input_resolution_rate": round(resolved_inputs / total, 6) if total else 0.0,
        "unique_unresolved_content_tokens": len(missing),
        "top_unresolved_content_tokens": top,
        "limit": "A resolved dictionary label does not establish semantic ISL correctness; signer review remains required.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "inputs": total,
        "full_input_resolution_rate": report["full_input_resolution_rate"],
        "unique_unresolved_content_tokens": len(missing),
        "top": top[:20],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
