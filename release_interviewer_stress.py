"""Run a release-facing interview simulation through the real local API.

The suite distinguishes validated meeting prompts from terms that must remain
unavailable. It measures API/clip delivery only; it is not an ISL grammar or
signer-accuracy evaluation.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from production_server import Handler
from regional_translation import REVIEWED_PHRASES


ROOT = Path(__file__).resolve().parent

INTERVIEWER_PROMPTS = (
    "good morning everyone",
    "welcome to the meeting",
    "please introduce yourself",
    "what is your name",
    "can you hear me",
    "please speak slowly",
    "could you please talk slower",
    "please share your screen",
    "the internet connection is poor",
    "please repeat that",
    "i do not understand",
    "can you repeat the question",
    "thank you for joining",
    "start the meeting",
    "we will begin now",
    "tell me about yourself",
    "what is your opinion",
    "do you have any question",
    "please send the document",
    "the camera is not working",
    "i cannot hear you",
    "let me know",
    "see you tomorrow",
    "thank you everyone",
    "have a good day",
    "please mute your microphone",
    "hai all",
    "cant hear you",
    "please don't repeat that",
)

# These gaps are intentional safety behavior until matching validated clips are
# available. A pass means the API refuses to invent a nearby sign.
MUST_REJECT = (
    "please unmute your microphone",
    "the webinar starts now",
    "can you share the presentation",
    "please reconnect to the meeting",
)


class QuietHandler(Handler):
    """Keep the release report readable while retaining the real API logic."""

    def log_message(self, format: str, *args: object) -> None:
        return


def request(connection: HTTPConnection, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    connection.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    return response.status, json.loads(response.read().decode("utf-8"))


def check_resolved(connection: HTTPConnection, text: str, language: str = "en-IN") -> dict[str, object]:
    status, resolved = request(connection, "/api/resolve", {"text": text, "language": language})
    keys = [str(item["key"]) for item in resolved.get("resolved", [])]
    clip_status, clip_response = request(connection, "/api/clips", {"keys": keys}) if keys else (200, {"clips": {}, "missing": []})
    passed = (
        status == 200
        and resolved.get("translation_available") is True
        and bool(keys)
        and not resolved.get("missing")
        and clip_status == 200
        and not clip_response.get("missing")
        and len(clip_response.get("clips", {})) == len(keys)
    )
    return {"input": text, "language": language, "keys": keys, "missing": resolved.get("missing", []), "passed": passed}


def check_rejection(connection: HTTPConnection, text: str) -> dict[str, object]:
    status, response = request(connection, "/api/resolve", {"text": text, "language": "en-IN"})
    missing = response.get("missing", [])
    return {
        "input": text,
        "resolved": response.get("resolved", []),
        "missing": missing,
        # The resolver may expose the surrounding validated words, but the
        # browser product guard refuses to play a partial utterance whenever
        # this missing list is nonempty.
        "passed": status == 200 and bool(missing),
    }


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=15)
    try:
        interviewer = [check_resolved(connection, prompt) for prompt in INTERVIEWER_PROMPTS]
        regional = [
            check_resolved(connection, input_text, language)
            for language, phrases in REVIEWED_PHRASES.items()
            for input_text in phrases
        ]
        rejected = [check_rejection(connection, prompt) for prompt in MUST_REJECT]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    report = {
        "purpose": "Local API and validated clip-delivery stress test; not ISL semantic correctness.",
        "interviewer_prompts": {"total": len(interviewer), "passed": sum(row["passed"] for row in interviewer), "cases": interviewer},
        "reviewed_regional_prompts": {"total": len(regional), "passed": sum(row["passed"] for row in regional), "cases": regional},
        "intentional_rejections": {"total": len(rejected), "passed": sum(row["passed"] for row in rejected), "cases": rejected},
    }
    target = ROOT / "reports" / "release_interviewer_stress.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "interviewer": f"{report['interviewer_prompts']['passed']}/{report['interviewer_prompts']['total']}",
        "reviewed_regional": f"{report['reviewed_regional_prompts']['passed']}/{report['reviewed_regional_prompts']['total']}",
        "intentional_rejections": f"{report['intentional_rejections']['passed']}/{report['intentional_rejections']['total']}",
    }, indent=2))
    if not all(
        section["passed"] == section["total"]
        for section in (report["interviewer_prompts"], report["reviewed_regional_prompts"], report["intentional_rejections"])
    ):
        raise SystemExit("release interviewer stress test failed")


if __name__ == "__main__":
    main()
