"""Verify that unknown words are not represented by unrelated sign motion."""

from production_server import resolve_phrase


def main() -> None:
    resolved, missing = resolve_phrase("qzxvfoo")
    assert not resolved and missing == ["qzxvfoo"], (resolved, missing)
    print("unknown-word guard: unsupported input is reported as unavailable")


if __name__ == "__main__":
    main()
