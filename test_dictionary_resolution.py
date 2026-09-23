"""Regression checks for browser-equivalent ISL dictionary resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFESTS = (
    "signal_dictionary_cislr/manifest.json",
    "signal_dictionary_bridgeconn/manifest.json",
    "signal_dictionary_official/manifest.json",
    "signal_dictionary_include/manifest.json",
    "signal_dictionary_islrtc_batch/manifest.json",
    "signal_dictionary_islrtc_batch_v3/manifest.json",
    "signal_dictionary_isl500/manifest.json",
    "signal_dictionary_isign/manifest.json",
    "signal_dictionary_kaggle_social/manifest.json",
)


def key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


def resolve(keys: set[str], text: str) -> str | None:
    canonical = {key(raw): raw for raw in keys}
    return canonical.get(key(text))


def segment(keys: set[str], text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    result: list[str] = []
    index = 0
    while index < len(tokens):
        match = None
        for width in range(min(5, len(tokens) - index), 0, -1):
            phrase = " ".join(tokens[index : index + width])
            if resolve(keys, phrase):
                match = phrase
                break
        result.append(match or tokens[index])
        index += len(match.split()) if match else 1
    return result


def main() -> None:
    keys: set[str] = set()
    for name in MANIFESTS:
        path = ROOT / name
        if path.exists():
            keys.update(json.loads(path.read_text(encoding="utf-8")).get("signs", {}))
    cases = {
        "e-book": "E-BOOK",
        "thank you": "THANK_YOU",
        "what is your name": "WHAT_IS_YOUR_NAME",
        "see you tomorrow": "SEE_YOU_TOMORROW",
    }
    for phrase, expected in cases.items():
        assert resolve(keys, phrase) == expected, (phrase, resolve(keys, phrase), expected)
        assert segment(keys, phrase) == [key(phrase).lower().replace("_", " ")], (phrase, segment(keys, phrase))
    print(f"dictionary resolution regression passed: {len(cases)} phrases across {len(keys)} source keys")


if __name__ == "__main__":
    main()
