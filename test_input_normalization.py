"""Regression suite for unambiguous speech and typing normalization."""

from __future__ import annotations

from production_server import resolve_phrase


CASES = {
    "hai all": ["HELLO", "EVERYONE"],
    "HI ALL!!!": ["HELLO", "EVERYONE"],
    "helo everyone": ["HELLO", "EVERYONE"],
    "helloo all": ["HELLO", "EVERYONE"],
    "i'm fine": ["I", "FINE"],
    "i’m fine": ["I", "FINE"],
    "im fine": ["I", "FINE"],
    "i m fine": ["I", "FINE"],
    "i can't hear you": ["I", "CAN_NOT", "HEAR", "YOU"],
    "cant hear you": ["CAN_NOT", "HEAR", "YOU"],
    "can t hear you": ["CAN_NOT", "HEAR", "YOU"],
    "please don't repeat that": ["PLEASE", "DO_NOT", "REPEAT", "THAT"],
    "please dont repeat that": ["PLEASE", "DO_NOT", "REPEAT", "THAT"],
    "please don t repeat that": ["PLEASE", "DO_NOT", "REPEAT", "THAT"],
    "you re ready": ["YOU", "READY"],
    "hai guys how ar you i am fine": ["HELLO", "EVERYONE", "HOW_ARE_YOU", "I", "FINE"],
}


def main() -> None:
    for text, expected in CASES.items():
        resolved, missing = resolve_phrase(text)
        actual = [item["key"] for item in resolved]
        assert actual == expected and not missing, (text, actual, missing)
        assert not any(key.endswith("_T") or key in {"IM", "M", "RE", "LL", "S"} for key in actual), (text, actual)
    print(f"input normalization regression passed: {len(CASES)} cases")


if __name__ == "__main__":
    main()
