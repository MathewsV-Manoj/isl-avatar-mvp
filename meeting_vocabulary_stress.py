"""Stress test common online-meeting vocabulary against validated ISL clips."""

from __future__ import annotations

import json
from pathlib import Path

from production_server import clips_for, resolve_phrase


ROOT = Path(__file__).resolve().parent
TERMS = (
    "agenda", "participant", "participants", "join", "joined", "leave", "mute", "unmute",
    "camera", "video", "audio", "network", "internet", "connection", "reconnect", "screen",
    "share", "presentation", "document", "file", "link", "message", "chat", "question", "answer",
    "vote", "poll", "recording", "start", "stop", "pause", "resume", "schedule", "time",
    "today", "tomorrow", "yesterday", "morning", "afternoon", "evening", "meeting", "class",
    "webinar", "conference", "speak", "listen", "hear", "understand", "repeat", "slowly",
)
PHRASES = (
    "please share your screen", "can you hear me", "i cannot hear you", "please repeat that",
    "start the meeting", "the internet connection is poor", "please speak slowly",
    "i do not understand", "the camera is not working", "join the meeting tomorrow",
    "please send the document", "ask your question", "stop the recording", "the class starts today",
)


def check(text: str) -> dict[str, object]:
    resolved, missing = resolve_phrase(text)
    clips, unavailable = clips_for([item["key"] for item in resolved])
    return {
        "input": text,
        "resolved": [item["key"] for item in resolved],
        "missing": missing,
        "clip_unavailable": unavailable,
        "passed": bool(resolved) and not missing and not unavailable and len(clips) == len(resolved),
    }


def main() -> None:
    terms = [check(term) for term in TERMS]
    phrases = [check(phrase) for phrase in PHRASES]
    report = {
        "terms": {"passed": sum(row["passed"] for row in terms), "total": len(terms), "cases": terms},
        "phrases": {"passed": sum(row["passed"] for row in phrases), "total": len(phrases), "cases": phrases},
        "unresolved_terms": [row["input"] for row in terms if row["missing"]],
        "unresolved_phrase_tokens": sorted({token for row in phrases for token in row["missing"]}),
    }
    (ROOT / "reports" / "meeting_vocabulary_stress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "terms": f"{report['terms']['passed']}/{report['terms']['total']}",
        "phrases": f"{report['phrases']['passed']}/{report['phrases']['total']}",
        "unresolved_terms": report["unresolved_terms"],
        "unresolved_phrase_tokens": report["unresolved_phrase_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
