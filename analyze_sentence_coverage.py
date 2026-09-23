"""Measure sentence token coverage and identify high-value missing vocabulary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SENTENCES = Path("datasets/isign_sentence_metadata/normalized.jsonl")
REPORT = Path("sentence_coverage_report.json")


def variants(word: str) -> set[str]:
    values = {word}
    if word.endswith("ies") and len(word) > 4:
        values.add(word[:-3] + "y")
    if word.endswith("es") and len(word) > 4:
        values.add(word[:-2])
    if word.endswith("s") and len(word) > 3:
        values.add(word[:-1])
    if word.endswith("ing") and len(word) > 5:
        values.add(word[:-3])
    if word.endswith("ed") and len(word) > 4:
        values.add(word[:-2])
    return values


def main() -> None:
    keys = set()
    for path in [Path("signal_dictionary_cislr/manifest.json"), Path("signal_dictionary_bridgeconn/manifest.json"), Path("signal_dictionary_official/manifest.json")]:
        keys.update(json.loads(path.read_text(encoding="utf-8"))["signs"])
    keys.update(json.loads(Path("signal_dictionary_supplemental_filtered.json").read_text(encoding="utf-8")))
    vocabulary = {key.lower().replace("_", " ") for key in keys}
    counts = Counter()
    total = covered = lemma_covered = 0
    for line in SENTENCES.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for word in row["tokens"]:
            total += 1
            if word in vocabulary:
                covered += 1
            elif any(variant in vocabulary for variant in variants(word)):
                lemma_covered += 1
            else:
                counts[word] += 1
    REPORT.write_text(json.dumps({
        "tokens": total,
        "exact_covered": covered,
        "exact_coverage": round(covered / total, 4),
        "additional_inflection_covered": lemma_covered,
        "coverage_with_basic_inflections": round((covered + lemma_covered) / total, 4),
        "top_missing_tokens": counts.most_common(200),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"exact_coverage": round(covered / total, 4), "coverage_with_basic_inflections": round((covered + lemma_covered) / total, 4), "unique_missing": len(counts), "top_missing": counts.most_common(10)}))


if __name__ == "__main__":
    main()
