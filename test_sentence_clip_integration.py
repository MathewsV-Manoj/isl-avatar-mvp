"""Regression checks for recorded exact-sentence delivery."""

from __future__ import annotations

from production_server import INDEX, SENTENCE_SOURCE, clips_for, resolve_phrase, valid_sequence


CASES = (
    "are you free today",
    "can you repeat that please",
    "could you please talk slower",
)


def main() -> None:
    for text in CASES:
        resolved, missing = resolve_phrase(text)
        assert not missing, (text, missing)
        assert len(resolved) == 1, (text, resolved)
        record = resolved[0]
        assert record["source"] == "islrtc_sentences", (text, record)
        clips, unavailable = clips_for([record["key"]])
        assert not unavailable, (text, unavailable)
        assert valid_sequence(clips[record["gloss"]]), text
        assert record["source"] == SENTENCE_SOURCE[0]

    promoted = [record for record in INDEX.values() if record["source"] == "islrtc_sentences"]
    assert promoted, "no multiword sentence clips were promoted"
    print(f"recorded sentence integration passed: {len(promoted)} multiword clips")


if __name__ == "__main__":
    main()
