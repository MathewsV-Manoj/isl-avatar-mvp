"""Produce a transparent acceptance report for the server-backed MVP."""

from __future__ import annotations

import json
from pathlib import Path

from production_server import INDEX, REVIEWED_PHRASES, clips_for, resolve_phrase


ROOT = Path(__file__).resolve().parent
ENGLISH_CASES = (
    "hello", "thank you", "please help me", "what is your name",
    "see you tomorrow", "can you hear me", "repeat slowly", "good morning",
    "conference tomorrow", "meeting today", "i am fine", "i am happy",
    "please repeat", "i do not understand", "start the meeting", "my name",
    "my phone", "help", "where", "when", "why", "no", "yes", "understand", "india",
)


def check_english(text: str) -> dict[str, object]:
    resolved, missing = resolve_phrase(text)
    keys = [item["key"] for item in resolved]
    clips, unavailable = clips_for(keys)
    valid_clips = all(
        isinstance(raw, list) and raw and (
            (isinstance(raw[0], list) and len(raw[0]) == 225) or
            (isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list) and len(raw[0][0]) == 225)
        )
        for raw in clips.values()
    )
    return {
        "input": text, "keys": keys, "missing": missing, "clip_count": len(clips),
        "clip_unavailable": unavailable, "passed": bool(resolved) and not missing and not unavailable and len(clips) == len(keys) and valid_clips,
    }


def main() -> None:
    english = [check_english(text) for text in ENGLISH_CASES]
    regional = []
    for language, phrases in REVIEWED_PHRASES.items():
        for source, translated in phrases.items():
            result = check_english(translated)
            regional.append({"language": language, "input": source, "english": translated, **result})
    report = {
        "mode": "validated_dictionary_mvp",
        "runtime_resolvable_sign_keys": len(INDEX),
        "english": {"total": len(english), "passed": sum(item["passed"] for item in english), "cases": english},
        "reviewed_regional": {"total": len(regional), "passed": sum(item["passed"] for item in regional), "cases": regional},
        "guarantee": "Passed cases resolve only to stored sign clips. This report does not measure ISL linguistic adequacy or avatar fidelity.",
    }
    output = ROOT / "reports" / "mvp_acceptance.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("mode", "runtime_resolvable_sign_keys", "english", "reviewed_regional") if key != "english" and key != "reviewed_regional"}, indent=2))
    print(f"english={report['english']['passed']}/{report['english']['total']} regional={report['reviewed_regional']['passed']}/{report['reviewed_regional']['total']}")


if __name__ == "__main__":
    main()
