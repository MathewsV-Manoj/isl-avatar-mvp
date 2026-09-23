"""Exhaustively exercise the local validated dictionary product contract.

It tests every sign the runtime advertises, then separately tests the limited
reviewed multilingual surface and deliberate unknown-input rejections. It does
not infer whether a sign is linguistically correct; that is a signer review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mvp_acceptance import ENGLISH_CASES, check_english
from production_server import GLOSS_OMISSIONS, INDEX, REVIEWED_PHRASES, clips_for, resolve_phrase, reviewed_translation


ROOT = Path(__file__).resolve().parent


def input_for(raw_key: str) -> str:
    return re.sub(r"[_-]+", " ", raw_key).lower()


def main() -> None:
    delivery_failures: list[str] = []
    resolution_failures: list[str] = []
    intentional_omissions: list[str] = []
    for number, (key, record) in enumerate(INDEX.items(), 1):
        clips, missing = clips_for([key])
        raw = clips.get(record["gloss"])
        if missing or not isinstance(raw, list) or len(raw) != 30 or any(not isinstance(frame, list) or len(frame) != 225 for frame in raw):
            delivery_failures.append(key)
        resolved, missing_words = resolve_phrase(input_for(record["raw_key"]))
        got = [entry["key"] for entry in resolved]
        if key not in got:
            if key.lower() in GLOSS_OMISSIONS and not missing_words:
                intentional_omissions.append(key)
            else:
                resolution_failures.append(key)
        if number % 1000 == 0:
            print(f"checked {number}/{len(INDEX)} signs", flush=True)

    english = [check_english(text) for text in ENGLISH_CASES]
    regional = []
    for language, phrases in REVIEWED_PHRASES.items():
        for text, english_text in phrases.items():
            translated, method = reviewed_translation(text, language)
            checked = check_english(english_text)
            regional.append({
                "language": language, "text": text, "translation": translated,
                "method": method, "passed": translated == english_text and method == "reviewed_phrase" and checked["passed"],
            })
    unknown = []
    for language, text in (("en-IN", "qzxvfoo"), ("hi-IN", "अपरिचित परीक्षण"), ("ml-IN", "പരിശോധിക്കാത്ത വാചകം"), ("ta-IN", "சோதிக்கப்படாத சொற்றொடர்")):
        translated, _ = reviewed_translation(text, language)
        if language == "en-IN":
            resolved, missing = resolve_phrase(text)
            passed = not resolved and missing == [text]
        else:
            passed = translated is None
        unknown.append({"language": language, "text": text, "passed": passed})

    report = {
        "runtime_signs": len(INDEX),
        "dictionary_delivery": {"passed": len(INDEX) - len(delivery_failures), "failed_keys": delivery_failures},
        "natural_label_resolution": {"passed": len(INDEX) - len(resolution_failures) - len(intentional_omissions), "failed_keys": resolution_failures, "intentional_function_word_omissions": intentional_omissions},
        "english_meeting_phrases": {"passed": sum(item["passed"] for item in english), "total": len(english)},
        "reviewed_regional_phrases": {"passed": sum(item["passed"] for item in regional), "total": len(regional), "cases": regional},
        "unknown_input_rejection": {"passed": sum(item["passed"] for item in unknown), "total": len(unknown), "cases": unknown},
        "limits": "This checks server lookup and pose delivery, not the linguistic accuracy or naturalness of each sign.",
    }
    path = ROOT / "reports" / "exhaustive_product_acceptance.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "runtime_signs": report["runtime_signs"],
        "delivery": f"{report['dictionary_delivery']['passed']}/{len(INDEX)}",
        "natural_resolution": f"{report['natural_label_resolution']['passed']}/{len(INDEX)}",
        "english": f"{report['english_meeting_phrases']['passed']}/{report['english_meeting_phrases']['total']}",
        "regional": f"{report['reviewed_regional_phrases']['passed']}/{report['reviewed_regional_phrases']['total']}",
        "unknown_rejection": f"{report['unknown_input_rejection']['passed']}/{report['unknown_input_rejection']['total']}",
        "delivery_failures": len(delivery_failures),
        "resolution_failures": len(resolution_failures),
    }, indent=2))
    if delivery_failures:
        raise SystemExit("served clip delivery failures found")


if __name__ == "__main__":
    main()
