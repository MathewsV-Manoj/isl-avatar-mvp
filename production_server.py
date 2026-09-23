"""Local production-style server for the ISL avatar MVP.

It keeps the large pose dictionaries on the server and sends the browser only
the verified clips required by each request.  This is intentionally a
dictionary resolver, not a claim of unrestricted sentence-to-ISL translation.
"""

from __future__ import annotations

import json
import re
import gzip
import os
from collections import OrderedDict
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from regional_translation import LANGUAGE_NAMES, REVIEWED_PHRASES


ROOT = Path(__file__).resolve().parent
ALLOWED_ORIGIN = os.environ.get("ISL_ALLOWED_ORIGIN", "").strip()
SOURCES = (
    ("cislr", "signal_dictionary_cislr"),
    ("bridgeconn", "signal_dictionary_bridgeconn"),
    ("official", "signal_dictionary_official"),
    ("include", "signal_dictionary_include"),
    ("islrtc", "signal_dictionary_islrtc_batch"),
    ("islrtc_v3", "signal_dictionary_islrtc_batch_v3"),
    # ISL500 is intentionally excluded from serving. Its validation report
    # found both hands collapsed in 9,436 of 14,220 frames, making it unsafe
    # for a hand-critical signing MVP.
    ("isign", "signal_dictionary_isign"),
    ("kaggle_social", "signal_dictionary_kaggle_social"),
)

# These are recorded multiword sequences, kept separate from the isolated-sign
# sources.  An exact sentence match may use one of them; all other input still
# follows the validated isolated-sign resolver below.
SENTENCE_SOURCE = ("islrtc_sentences", "signal_dictionary_islrtc_sentences")

# Exact source-label synonym pairs.  These aliases are accepted only where a
# served dataset label explicitly names both forms; they are not substitutions
# invented from a generic thesaurus.
VALIDATED_ALIASES = {
    "observe": "OBSERVE_WATCH",
    "began": "BEGAN_BEGUN",
    "let": "ALLOW_LET",
}

ELIGIBILITY_REPORT = ROOT / "reports" / "clip_eligibility.json"
ELIGIBILITY = json.loads(ELIGIBILITY_REPORT.read_text(encoding="utf-8")).get("sources", {}) if ELIGIBILITY_REPORT.exists() else {}
BLOCKED_SOURCE_KEYS = {source: set(report.get("blocked_labels", [])) for source, report in ELIGIBILITY.items()}
MAX_PHRASE_TOKENS = 16
MAX_INPUT_CHARS = 1_000
MAX_REQUEST_BYTES = 32_768
MAX_CLIP_REQUEST_KEYS = 100


def canonical(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def candidates(value: str) -> list[str]:
    value = str(value or "").strip().lower()
    result = [value]
    if value.endswith("ies") and len(value) > 3:
        result.append(value[:-3] + "y")
    if value.endswith("ing") and len(value) > 5:
        result.extend((value[:-3], value[:-3] + "e"))
    if value.endswith("ed") and len(value) > 4:
        result.extend((value[:-2], value[:-1]))
    if value.endswith("es") and len(value) > 4:
        result.append(value[:-2])
    if value.endswith("s") and len(value) > 3:
        result.append(value[:-1])
    keys = [canonical(item) for item in result if canonical(item)]
    alias = VALIDATED_ALIASES.get(value)
    if alias and alias not in keys:
        keys.append(alias)
    return keys


def load_index() -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}

    sentence_source, sentence_directory = SENTENCE_SOURCE
    sentence_manifest = ROOT / sentence_directory / "manifest.json"
    if sentence_manifest.exists():
        sentence_signs = json.loads(sentence_manifest.read_text(encoding="utf-8")).get("signs", {})
        for raw_key, entry in sentence_signs.items():
            # Do not replace an isolated sign with a sentence take for a
            # one-word input.  Exact multiword recordings are the only
            # candidates promoted into the product lookup path.
            if len(re.findall(r"[A-Za-z0-9]+", str(raw_key))) < 2:
                continue
            key = canonical(raw_key)
            if not key:
                continue
            index[key] = {
                "source": sentence_source,
                "directory": sentence_directory,
                "shard": str(entry["shard"]),
                "raw_key": raw_key,
                "gloss": str(entry.get("gloss") or raw_key.replace("_", " ").lower()),
            }

    for source, directory in SOURCES:
        manifest_path = ROOT / directory / "manifest.json"
        if not manifest_path.exists():
            continue
        signs = json.loads(manifest_path.read_text(encoding="utf-8")).get("signs", {})
        for raw_key, entry in signs.items():
            if raw_key in BLOCKED_SOURCE_KEYS.get(source, set()):
                continue
            key = canonical(raw_key)
            if not key:
                continue
            # Source order is quality priority.  Never replace an earlier clip.
            index.setdefault(key, {
                "source": source,
                "directory": directory,
                "shard": str(entry["shard"]),
                "raw_key": raw_key,
                "gloss": str(entry.get("gloss") or raw_key.replace("_", " ").lower()),
            })
    return index


INDEX = load_index()
GLOSS_OMISSIONS = frozenset({"a", "an", "the", "am", "is", "are", "was", "were", "be", "to", "of"})

# Speech recognition and informal typing commonly produce "hai all" or
# "hi all". These are unambiguous greeting variants in the meeting surface,
# so normalize them before phrase resolution instead of dropping "hai" and
# signing only "all".
GREETING_NORMALIZATIONS = {
    "hai": "hello",
    "hi": "hello",
    "hii": "hello",
    "helo": "hello",
    "helloo": "hello",
    "hai all": "hello everyone",
    "hi all": "hello everyone",
    "hii all": "hello everyone",
    "helo all": "hello everyone",
    "helloo all": "hello everyone",
    "hello all": "hello everyone",
    "hai everyone": "hello everyone",
    "hi everyone": "hello everyone",
}

CONTRACTION_NORMALIZATIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "can't": "can not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "let's": "let us",
}

# Some speech engines omit apostrophes altogether, and a few produce a
# separated final letter. Handle only unambiguous conversational forms here;
# arbitrary fuzzy correction remains intentionally disabled.
COMPACT_SPEECH_NORMALIZATIONS = {
    "im": "i am", "youre": "you are", "theyre": "they are",
    "hes": "he is", "shes": "she is", "its": "it is", "thats": "that is",
    "cant": "can not", "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "arent": "are not", "wasnt": "was not", "werent": "were not",
    "wont": "will not", "wouldnt": "would not", "shouldnt": "should not",
    "couldnt": "could not", "havent": "have not", "hasnt": "has not", "hadnt": "had not",
    "ill": "i will", "youll": "you will", "theyll": "they will",
    "lets": "let us", "can t": "can not", "don t": "do not",
    "i m": "i am", "you re": "you are", "we re": "we are", "they re": "they are",
    "he s": "he is", "she s": "she is", "it s": "it is", "that s": "that is",
    # Speech recognizers commonly emit the clipped copula as "ar".
    "ar": "are",
}
GREETING_WORD_ALIASES = frozenset({"hai", "hi", "hii", "helo", "helloo"})
GROUP_ADDRESS_NORMALIZATIONS = {"guys": "everyone"}


def normalize_english_input(text: str) -> str:
    normalized = str(text).lower().replace("’", "'")
    for contraction, expanded in CONTRACTION_NORMALIZATIONS.items():
        normalized = re.sub(rf"(?<![a-z]){re.escape(contraction)}(?![a-z])", expanded, normalized)
    for spoken, expanded in COMPACT_SPEECH_NORMALIZATIONS.items():
        normalized = re.sub(rf"(?<![a-z]){re.escape(spoken)}(?![a-z])", expanded, normalized)
    normalized = " ".join(re.findall(r"[a-z0-9]+", normalized))
    words = normalized.split()
    if words and words[0] in GREETING_WORD_ALIASES:
        words[0] = "hello"
    if len(words) >= 2 and words[0] == "hello" and words[1] in {"all", "everyone"}:
        words[1] = "everyone"
    words = [GROUP_ADDRESS_NORMALIZATIONS.get(word, word) for word in words]
    normalized = " ".join(words)
    return GREETING_NORMALIZATIONS.get(normalized, normalized)


def resolve_phrase(text: str) -> tuple[list[dict[str, str]], list[str]]:
    raw_tokens = re.findall(r"[a-z0-9]+", str(text).lower().replace("’", "'"))
    raw_key = canonical(" ".join(raw_tokens))
    # Prefer an exact recorded/source phrase before generic speech cleanup.
    # This preserves a verified multiword clip such as HI_HOW_ARE_YOU while
    # still allowing unambiguous forms such as "cant" to normalize below.
    if raw_key in INDEX and (len(raw_tokens) > 1 or raw_tokens[0] not in GLOSS_OMISSIONS):
        record = dict(INDEX[raw_key])
        record["key"] = raw_key
        return [record], []

    tokens = normalize_english_input(text).split()
    resolved: list[dict[str, str]] = []
    missing: list[str] = []
    position = 0
    while position < len(tokens):
        match_key = None
        width_used = 0
        for width in range(min(MAX_PHRASE_TOKENS, len(tokens) - position), 0, -1):
            phrase = " ".join(tokens[position : position + width])
            # Keep a reviewed multiword entry such as DO_NOT, but do not
            # promote a standalone English article or copula merely because a
            # source happened to contain a label for it.
            if width == 1 and phrase in GLOSS_OMISSIONS:
                continue
            for key in candidates(phrase):
                if key in INDEX:
                    match_key, width_used = key, width
                    break
            if match_key:
                break
        if match_key:
            record = dict(INDEX[match_key])
            record["key"] = match_key
            resolved.append(record)
            position += width_used
        else:
            # These English function words do not add an isolated ISL gloss in
            # the supported meeting-phrase path. They are omitted explicitly,
            # never replaced with another sign.
            if tokens[position] not in GLOSS_OMISSIONS:
                missing.append(tokens[position])
            position += 1
    return resolved, missing


def reviewed_translation(text: str, language: str) -> tuple[str | None, str]:
    text = " ".join(str(text).strip().split())
    if language == "en-IN":
        return text, "identity"
    translated = REVIEWED_PHRASES.get(language, {}).get(text)
    if translated:
        return translated, "reviewed_phrase"
    return None, "unreviewed"


@lru_cache(maxsize=24)
def shard(directory: str, name: str) -> dict[str, object]:
    path = ROOT / directory / name
    return json.loads(path.read_text(encoding="utf-8"))


def clips_for(keys: list[str]) -> tuple[dict[str, object], list[str]]:
    clips: dict[str, object] = {}
    # A bounded payload keeps speech-driven playback responsive. Anything
    # beyond the bound is reported, never silently dropped.
    missing: list[str] = [str(key) for key in keys[40:]]
    for key in keys[:40]:
        record = INDEX.get(canonical(key))
        if not record:
            missing.append(str(key))
            continue
        selected = selected_clip(record["directory"], record["shard"], record["raw_key"])
        if selected is None:
            missing.append(str(key))
        else:
            clips[record["gloss"]] = selected
    return clips, missing


@lru_cache(maxsize=2_048)
def selected_clip(directory: str, name: str, raw_key: str) -> list[list[float | int]] | None:
    """Cache the chosen best take for repeated meeting vocabulary."""
    return select_clip(shard(directory, name).get(raw_key))


def valid_sequence(value: object) -> bool:
    """Check one browser-ready 30-frame landmark sequence."""
    return (
        isinstance(value, list)
        and len(value) == 30
        and all(
            isinstance(frame, list)
            and len(frame) == 225
            and all(isinstance(feature, (int, float)) for feature in frame)
            for frame in value
        )
    )


def take_score(sequence: list[list[float | int]]) -> int:
    """Prefer takes with visible hand geometry when a source stores variants."""
    score = 0
    for frame in sequence:
        left = frame[99:162]
        right = frame[162:225]
        if any(abs(value - left[index % 3]) > 1e-6 for index, value in enumerate(left)):
            score += 1
        if any(abs(value - right[index % 3]) > 1e-6 for index, value in enumerate(right)):
            score += 1
    return score


def select_clip(value: object) -> list[list[float | int]] | None:
    """Select one eligible take and never send a multi-take container to JS."""
    if valid_sequence(value):
        return value
    if not isinstance(value, list):
        return None
    candidates = [take for take in value if valid_sequence(take)]
    return max(candidates, key=take_score, default=None)


def valid_clip(value: object) -> bool:
    """Accept a direct sequence or a source container with a valid take."""
    return select_clip(value) is not None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        print("[isl-mvp] " + format % args, flush=True)

    def json_response(self, body: dict[str, object], status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        compressed = "gzip" in self.headers.get("Accept-Encoding", "").lower() and len(payload) >= 1_024
        if compressed:
            payload = gzip.compress(payload, compresslevel=5)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if ALLOWED_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        if urlparse(self.path).path not in {"/api/health", "/api/resolve", "/api/clips"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        if ALLOWED_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/health":
            self.json_response({"ok": True, "mode": "validated_dictionary", "sign_count": len(INDEX)})
            return
        if urlparse(self.path).path == "/":
            self.path = "/isl-avatar-prototype.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/resolve", "/api/clips"}:
            self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            content_type = self.headers.get("Content-Type", "").lower()
            if not content_type.startswith("application/json"):
                raise ValueError("Content-Type must be application/json")
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > MAX_REQUEST_BYTES:
                raise ValueError("request body exceeds the supported size")
            payload = json.loads(self.rfile.read(size))
            if path == "/api/resolve":
                language = str(payload.get("language", "en-IN"))
                text = str(payload.get("text", ""))
                if len(text) > MAX_INPUT_CHARS:
                    raise ValueError("text exceeds the supported length")
                english, method = reviewed_translation(text, language)
                if english is None:
                    self.json_response({
                        "resolved": [], "missing": [], "mode": "validated_dictionary",
                        "translation_available": False, "translation_method": method,
                        "source_language": LANGUAGE_NAMES.get(language, language),
                    })
                    return
                resolved, missing = resolve_phrase(english)
                self.json_response({
                    "resolved": resolved, "missing": missing, "mode": "validated_dictionary",
                    "translation_available": True, "translation_method": method,
                    "english": english, "source_language": LANGUAGE_NAMES.get(language, language),
                    "recorded_sentence": len(resolved) == 1 and resolved[0]["source"] == SENTENCE_SOURCE[0],
                })
                return
            requested = payload.get("keys", [])
            if not isinstance(requested, list):
                raise ValueError("keys must be an array")
            if len(requested) > MAX_CLIP_REQUEST_KEYS:
                raise ValueError("too many clip keys requested")
            clips, missing = clips_for([str(key) for key in requested])
            self.json_response({"clips": clips, "missing": missing})
        except (ValueError, json.JSONDecodeError) as exc:
            self.json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


if __name__ == "__main__":
    host = os.environ.get("ISL_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    print(f"ISL MVP server ready on http://{host}:{port} with {len(INDEX)} indexed signs.", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
