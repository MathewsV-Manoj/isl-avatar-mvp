"""Audit local ISL dictionaries and report usable vocabulary coverage.

The report deliberately measures keys that can be resolved by the application,
not the number of videos downloaded.  It also records cross-source duplicates
so a weak supplemental clip never silently replaces a preferred CISLR clip.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = (
    ("cislr", "signal_dictionary_cislr/manifest.json"),
    ("bridgeconn", "signal_dictionary_bridgeconn/manifest.json"),
    ("official", "signal_dictionary_official/manifest.json"),
    ("include", "signal_dictionary_include/manifest.json"),
    ("islrtc", "signal_dictionary_islrtc_batch/manifest.json"),
    ("islrtc_v3", "signal_dictionary_islrtc_batch_v3/manifest.json"),
    ("isl500", "signal_dictionary_isl500/manifest.json"),
    ("isign", "signal_dictionary_isign/manifest.json"),
    ("kaggle_social", "signal_dictionary_kaggle_social/manifest.json"),
)


def key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def candidates(token: str) -> set[str]:
    token = key(token)
    result = {token}
    if token.endswith("IES") and len(token) > 3:
        result.add(token[:-3] + "Y")
    if token.endswith("ING") and len(token) > 5:
        result.update((token[:-3], token[:-3] + "E"))
    if token.endswith("ED") and len(token) > 4:
        result.update((token[:-2], token[:-1]))
    if token.endswith("ES") and len(token) > 4:
        result.add(token[:-2])
    if token.endswith("S") and len(token) > 3:
        result.add(token[:-1])
    return {item for item in result if item}


def main() -> None:
    source_keys: dict[str, set[str]] = {}
    owners: dict[str, list[str]] = defaultdict(list)
    malformed: dict[str, list[str]] = defaultdict(list)
    for source, relative in SOURCES:
        path = ROOT / relative
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        keys = set(manifest.get("signs", {}))
        source_keys[source] = keys
        for sign in keys:
            owners[sign].append(source)
            if sign != key(sign):
                malformed[source].append(sign)

    preferred_order = [name for name, _ in SOURCES]
    preferred: dict[str, str] = {}
    for sign, sources in owners.items():
        preferred[sign] = next(source for source in preferred_order if source in sources)

    total = set().union(*source_keys.values()) if source_keys else set()
    canonical_owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, keys in source_keys.items():
        for raw_key in keys:
            canonical_owners[key(raw_key)].append((source, raw_key))
    canonical_collisions = {
        canonical_key: values
        for canonical_key, values in canonical_owners.items()
        if len({raw_key for _, raw_key in values}) > 1
    }
    vocab = Counter()
    bpcc = ROOT / "datasets/posestitch_isl/BPCC-ISL.csv"
    if bpcc.exists():
        with bpcc.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("Processed_Sentence") or row.get("sentence_without_punctuation") or ""
                vocab.update(re.findall(r"[A-Za-z0-9']+", text.lower()))

    weighted_total = sum(vocab.values())
    weighted_hit = sum(count for token, count in vocab.items() if candidates(token) & total)
    report = {
        "sources": {name: len(keys) for name, keys in source_keys.items()},
        "raw_unique_sign_keys": len(total),
        "runtime_resolvable_sign_keys": len(canonical_owners),
        "duplicate_sign_keys": sum(1 for sources in owners.values() if len(sources) > 1),
        "canonical_key_collisions": {
            canonical_key: [{"source": source, "raw_key": raw_key} for source, raw_key in values]
            for canonical_key, values in canonical_collisions.items()
        },
        "malformed_keys": {name: values[:50] for name, values in malformed.items() if values},
        "preferred_source_order": preferred_order,
        "bpcc": {
            "unique_tokens": len(vocab),
            "weighted_tokens": weighted_total,
            "weighted_resolvable_tokens": weighted_hit,
            "weighted_resolution": round(weighted_hit / max(1, weighted_total), 6),
            "top_unresolved": [
                {"token": token, "count": count}
                for token, count in vocab.most_common()
                if not (candidates(token) & total)
            ][:250],
        },
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "dictionary_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
