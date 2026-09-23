"""Measure product-server lexical coverage without claiming translation quality."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from production_server import GLOSS_OMISSIONS, INDEX, candidates


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "datasets" / "posestitch_isl" / "BPCC-ISL.csv"


def main() -> None:
    counts: Counter[str] = Counter()
    with CORPUS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = row.get("Processed_Sentence") or row.get("sentence_without_punctuation") or ""
            counts.update(re.findall(r"[a-z0-9']+", text.lower()))

    total = sum(counts.values())
    omitted = sum(count for token, count in counts.items() if token in GLOSS_OMISSIONS)
    resolved = sum(count for token, count in counts.items() if any(key in INDEX for key in candidates(token)))
    unresolved = [(token, count) for token, count in counts.most_common() if not any(key in INDEX for key in candidates(token)) and token not in GLOSS_OMISSIONS]
    report = {
        "purpose": "Lexical lookup coverage only; it is not ISL translation or avatar accuracy.",
        "runtime_resolvable_sign_keys": len(INDEX),
        "token_instances": total,
        "function_word_omissions": omitted,
        "resolvable_token_instances": resolved,
        "content_token_instances": total - omitted,
        "content_token_resolution": resolved / max(1, total - omitted),
        "top_unresolved": [{"token": token, "count": count} for token, count in unresolved[:100]],
    }
    target = ROOT / "reports" / "runtime_lexical_coverage.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("runtime_resolvable_sign_keys", "content_token_instances", "resolvable_token_instances", "content_token_resolution")}, indent=2))


if __name__ == "__main__":
    main()
