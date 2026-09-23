"""Regression tests for the server-backed dictionary MVP."""

from production_server import BLOCKED_SOURCE_KEYS, INDEX, clips_for, resolve_phrase, reviewed_translation, valid_clip


def main() -> None:
    assert len(INDEX) >= 10_000, len(INDEX)
    assert "" not in INDEX
    assert sum(map(len, BLOCKED_SOURCE_KEYS.values())) >= 5
    resolved, missing = resolve_phrase("what is your name")
    assert not missing, missing
    assert [item["key"] for item in resolved] == ["WHAT_IS_YOUR_NAME"], resolved
    for phrase in ("hi how are you", "i dont agree", "no need to worry dont worry"):
        resolved, missing = resolve_phrase(phrase)
        assert not missing and len(resolved) == 1, (phrase, resolved, missing)
    clips, unavailable = clips_for([item["key"] for item in resolved])
    assert not unavailable, unavailable
    assert len(clips) == 1, clips.keys()
    raw = next(iter(clips.values()))
    assert isinstance(raw, list) and len(raw) == 30, type(raw)
    assert all(isinstance(frame, list) and len(frame) == 225 for frame in raw)
    assert valid_clip(raw)
    assert not valid_clip([[0.0] * 225] * 29)
    assert not valid_clip([[0.0] * 224] * 30)
    assert valid_clip([raw, [[0.0] * 224] * 30])
    _, overflow = clips_for(["HELLO"] * 41)
    assert overflow == ["HELLO"]
    resolved, missing = resolve_phrase("conference tomorrow")
    assert len(resolved) == 2 and not missing, (resolved, missing)
    resolved, missing = resolve_phrase("please share the screen")
    assert [item["key"] for item in resolved] == ["PLEASE", "SHARE", "SCREEN"] and not missing, (resolved, missing)
    resolved, missing = resolve_phrase("united nations convention on the rights of persons with disabilities uncrpd")
    assert [item["key"] for item in resolved] == ["UNITED_NATIONS_CONVENTION_ON_THE_RIGHTS_OF_PERSONS_WITH_DISABILITIES_UNCRPD"] and not missing, (resolved, missing)
    resolved, missing = resolve_phrase("i am fine")
    assert [item["key"] for item in resolved] == ["I", "FINE"] and not missing, (resolved, missing)
    for input_word, expected_key in (("observe", "OBSERVE_WATCH"), ("began", "BEGAN_BEGUN"), ("let", "ALLOW_LET")):
        resolved, missing = resolve_phrase(input_word)
        assert not missing and [item["key"] for item in resolved] == [expected_key], (input_word, resolved, missing)
    for greeting in ("hai", "hi", "helo", "helloo", "hai all", "hi all", "helo all", "hello all", "hai everyone"):
        resolved, missing = resolve_phrase(greeting)
        assert not missing, (greeting, missing)
        expected = ["HELLO", "EVERYONE"] if " " in greeting else ["HELLO"]
        assert [item["key"] for item in resolved] == expected, (greeting, resolved)
    for contracted, expanded in (
        ("i'm fine", "i am fine"),
        ("i can't hear you", "i can not hear you"),
        ("please don't repeat that", "please do not repeat that"),
        ("im fine", "i am fine"),
        ("cant hear you", "can not hear you"),
        ("can t hear you", "can not hear you"),
        ("dont understand", "do not understand"),
        ("don t understand", "do not understand"),
    ):
        assert resolve_phrase(contracted) == resolve_phrase(expanded), (contracted, expanded)
    assert reviewed_translation("നമസ്കാരം", "ml-IN") == ("hello", "reviewed_phrase")
    assert reviewed_translation("வணக்கம்", "ta-IN") == ("hello", "reviewed_phrase")
    assert reviewed_translation("नमस्ते", "hi-IN") == ("hello", "reviewed_phrase")
    assert reviewed_translation("പരിശോധിക്കാത്ത വാചകം", "ml-IN")[0] is None
    print(f"production dictionary regression passed: {len(INDEX)} indexed signs")


if __name__ == "__main__":
    main()
