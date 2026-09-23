"""Resolve translated ISL glosses into an avatar plan with no unknown-token state.

This module runs after Malayalam/Hindi/Tamil/English speech is converted to an
ISL gloss sequence. It does not translate spoken language itself: keeping that
boundary prevents English word order from leaking into the avatar layer.

Resolution priority:
  1. exact dictionary sign
  2. curated gloss alias (for tense, inflection, and common variants)
  3. multi-sign compound split
  4. fingerspelling (names, acronyms, genuinely new terms)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ALIASES = {
    "HELLO": "HELLO_HI",
    "HI": "HELLO_HI",
    "AM": "I_ME_MINE_MY",
    "ARE": "BECOME",
    "GOING": "GO",
    "WENT": "GO",
    "GONE": "GO",
    "SCHOOLS": "COLLEGE_SCHOOL",
    "SCHOOL": "COLLEGE_SCHOOL",
    "THANKS": "GRATEFUL",
    "THANK_YOU": "GRATEFUL",
    "PLEASED": "PLEASE",
    "FRIENDS": "FRIEND",
    "CHILDREN": "CHILD",
    "PEOPLE": "PERSON",
    "DOCTOR": "MEDICINE",
    "MOBILE": "PHONE",
}


@dataclass(frozen=True)
class SignPlanItem:
    source: str
    kind: str  # sign | alias | compound | fingerspell
    signs: list[str]


def canonical(token: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", token.upper()).strip("_")


def load_vocabulary(paths: Iterable[Path]) -> set[str]:
    vocabulary: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        # CISLR is sharded, so its manifest is the lightweight source of every
        # available gloss. Ordinary dictionaries remain simple key-to-frames maps.
        keys = data["signs"] if isinstance(data, dict) and isinstance(data.get("signs"), dict) else data
        if not isinstance(keys, dict):
            raise ValueError(f"Dictionary must contain an object: {path}")
        vocabulary.update(canonical(key) for key in keys)
    return vocabulary


def resolve_glosses(glosses: Iterable[str], vocabulary: set[str]) -> list[SignPlanItem]:
    plan: list[SignPlanItem] = []
    for raw in glosses:
        token = canonical(raw)
        if not token:
            continue
        if token in vocabulary:
            plan.append(SignPlanItem(raw, "sign", [token]))
            continue
        alias = ALIASES.get(token)
        if alias and alias in vocabulary:
            plan.append(SignPlanItem(raw, "alias", [alias]))
            continue
        parts = token.split("_")
        if len(parts) > 1 and all(part in vocabulary for part in parts):
            plan.append(SignPlanItem(raw, "compound", parts))
            continue
        # A rendering plan always exists. The avatar must implement A-Z / 0-9
        # fingerspelling before this is enabled in the live interface.
        plan.append(SignPlanItem(raw, "fingerspell", list(token.replace("_", ""))))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("glosses", nargs="+", help="ISL gloss tokens to resolve")
    parser.add_argument(
        "--dictionary",
        type=Path,
        action="append",
        default=[Path("signal_dictionary_reanchored_v2.json"), Path("signal_dictionary_demo.json"), Path("signal_dictionary_cislr/manifest.json")],
    )
    args = parser.parse_args()
    vocabulary = load_vocabulary(args.dictionary)
    plan = resolve_glosses(args.glosses, vocabulary)
    print(json.dumps([asdict(item) for item in plan], indent=2))


if __name__ == "__main__":
    main()
