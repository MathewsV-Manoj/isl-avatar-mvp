"""Build a compact, quality-checked first-playback cache for the static app."""

from __future__ import annotations

import json
from pathlib import Path

from regional_translation import REVIEWED_PHRASES


ROOT = Path(__file__).resolve().parent
INDEX = json.loads((ROOT / "runtime_index.json").read_text(encoding="utf-8"))["index"]
OUT = ROOT / "runtime_quickstart.json"
BOOTSTRAP_OUT = ROOT / "runtime_bootstrap.json"

# These are common, unambiguous demo and meeting concepts. They remain sourced
# from the same validated clips as the full runtime; this file only avoids a
# multi-megabyte shard download on the first interaction.
# Full set used by the reviewed English/Hindi/Malayalam/Tamil meeting-demo
# prompts. Keeping it local to the browser avoids unpredictable multi-megabyte
# source shards during a live team demonstration.
KEYS = [
    "ABOUT", "ALLOW_LET", "ANY", "BEGIN", "CAMERA", "CAN", "CANNOT", "CAN_NOT",
    "CLASS", "CONNECTION", "COULD_YOU_PLEASE_TALK_SLOWER", "DAY", "DO", "DOCUMENT",
    "DO_NOT", "EVERYONE", "FINE", "FOR", "GOOD", "GOOD_MORNING", "HAPPY", "HAVE",
    "HEAR", "HELLO", "HELP_ME", "HOW_ARE_YOU", "I", "INTERNET", "INTRODUCE", "JOIN",
    "KNOW", "ME", "MEETING", "MICROPHONE", "MUTE", "MY", "NAME", "NO", "NOT", "NOW",
    "OPINION", "PHONE", "PLEASE", "POOR", "QUESTION", "REPEAT", "SCREEN",
    "SEE_YOU_TOMORROW", "SEND", "SHARE", "SLOWLY", "SPEAK", "START", "TELL", "THANK_YOU",
    "THAT", "TODAY", "UNDERSTAND", "WE", "WELCOME", "WHAT", "WHAT_IS_YOUR_NAME", "WILL",
    "WORK", "YES", "YOU", "YOUR", "YOURSELF",
]

# Small enough for a phone's first visit, while covering the conversation
# controls most likely to be used before the larger demo cache is warm.
BOOTSTRAP_KEYS = [
    "HELLO", "EVERYONE", "THANK_YOU", "PLEASE", "HELP_ME", "YES", "NO",
    "GOOD_MORNING", "MEETING", "START", "MUTE", "YOUR", "MICROPHONE",
    "HOW_ARE_YOU", "I", "FINE", "CAN", "YOU", "HEAR", "ME", "REPEAT",
    "SPEAK", "SLOWLY", "UNDERSTAND", "DO_NOT",
]


def valid_clip(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 30
        and all(isinstance(frame, list) and len(frame) == 225 and all(isinstance(n, (int, float)) for n in frame) for frame in value)
    )


def clip_score(clip: list[list[float]]) -> int:
    # Favor samples whose hand landmarks actually vary over time.
    return sum(any(abs(value - frame[99 + (index % 3)]) > 1e-6 for index, value in enumerate(frame[99:225])) for frame in clip)


def select_clip(raw: object) -> list[list[float]] | None:
    if valid_clip(raw):
        return raw
    if not isinstance(raw, list):
        return None
    options = [candidate for candidate in raw if valid_clip(candidate)]
    return max(options, key=clip_score, default=None)


def build_payload(keys: list[str], version: int) -> dict[str, object]:
    clips = []
    lookup = {}
    for key in keys:
        record = INDEX.get(key)
        if not record:
            continue
        shard_path = ROOT / record["directory"] / record["shard"]
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        clip = select_clip(shard.get(record["raw_key"]))
        if clip:
            clips.append({"gloss": record["gloss"], "clip": clip})
            lookup[key] = record
    # The demo cache carries its own compact resolver. This lets the public
    # browser serve the reviewed demo phrases even when the larger index is
    # delayed by a slow connection.
    return {
        "version": version,
        "clip_count": len(clips),
        "clips": clips,
        "index": lookup,
        "reviewed_phrases": REVIEWED_PHRASES,
    }


def main() -> None:
    bootstrap = build_payload(BOOTSTRAP_KEYS, version=1)
    # Include the compact full resolver index once, so a mobile page has no
    # separate index request before it can recognize a common phrase.
    bootstrap["index"] = INDEX
    BOOTSTRAP_OUT.write_text(json.dumps(bootstrap, separators=(",", ":")), encoding="utf-8")
    demo = build_payload(KEYS, version=2)
    OUT.write_text(json.dumps(demo, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {BOOTSTRAP_OUT.name}: {bootstrap['clip_count']} clips, {BOOTSTRAP_OUT.stat().st_size:,} bytes")
    print(f"wrote {OUT.name}: {demo['clip_count']} clips, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
