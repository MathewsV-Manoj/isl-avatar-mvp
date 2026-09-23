"""Exercise the ISL MVP's actual HTTP contract on an isolated local port."""

from __future__ import annotations

import json
import gzip
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import production_server
from production_server import Handler, INDEX


def request(connection: HTTPConnection, method: str, path: str, payload: dict | None = None, gzip_ok: bool = False) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    if gzip_ok:
        headers["Accept-Encoding"] = "gzip"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    if response.getheader("Content-Encoding") == "gzip":
        raw = gzip.decompress(raw)
    return response.status, json.loads(raw.decode("utf-8"))


def main() -> None:
    production_server.ALLOWED_ORIGIN = "https://frontend.example"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=15)
    try:
        status, health = request(connection, "GET", "/api/health")
        assert status == 200 and health["ok"] is True
        assert health["sign_count"] == len(INDEX) >= 10_000

        connection.request("GET", "/api/health")
        cors = connection.getresponse()
        assert cors.status == 200
        assert cors.getheader("Access-Control-Allow-Origin") == "https://frontend.example"
        assert cors.getheader("X-Content-Type-Options") == "nosniff"
        assert cors.getheader("Referrer-Policy") == "no-referrer"
        cors.read()

        connection.request("OPTIONS", "/api/resolve")
        options = connection.getresponse()
        assert options.status == 204
        assert options.getheader("Access-Control-Allow-Origin") == "https://frontend.example"
        assert options.getheader("Access-Control-Allow-Methods") == "GET, POST, OPTIONS"
        options.read()

        status, resolved = request(connection, "POST", "/api/resolve", {
            "text": "what is your name", "language": "en-IN",
        })
        assert status == 200 and resolved["missing"] == []
        assert [item["key"] for item in resolved["resolved"]] == ["WHAT_IS_YOUR_NAME"]

        status, sentence = request(connection, "POST", "/api/resolve", {
            "text": "could you please talk slower", "language": "en-IN",
        })
        assert status == 200 and sentence["recorded_sentence"] is True
        assert sentence["missing"] == [] and len(sentence["resolved"]) == 1

        status, clips = request(connection, "POST", "/api/clips", {
            "keys": [item["key"] for item in sentence["resolved"]],
        })
        assert status == 200 and clips["missing"] == []
        clip = next(iter(clips["clips"].values()))
        assert len(clip) == 30 and all(len(frame) == 225 for frame in clip)

        status, compressed = request(connection, "POST", "/api/clips", {
            "keys": [item["key"] for item in sentence["resolved"]],
        }, gzip_ok=True)
        assert status == 200 and compressed == clips

        status, regional = request(connection, "POST", "/api/resolve", {
            "text": "നമസ്കാരം", "language": "ml-IN",
        })
        assert status == 200 and regional["translation_available"] is True
        assert regional["english"] == "hello" and regional["translation_method"] == "reviewed_phrase"

        status, unsupported = request(connection, "POST", "/api/resolve", {
            "text": "പരിശോധിക്കാത്ത വാചകം", "language": "ml-IN",
        })
        assert status == 200 and unsupported["translation_available"] is False

        status, partial = request(connection, "POST", "/api/resolve", {
            "text": "please unmute your microphone", "language": "en-IN",
        })
        assert status == 200 and partial["missing"] == ["unmute"]
        assert partial["resolved"], partial

        status, invalid = request(connection, "POST", "/api/resolve", {
            "text": "x" * 1_001, "language": "en-IN",
        })
        assert status == 400 and "length" in invalid["error"]

        status, invalid = request(connection, "POST", "/api/clips", {
            "keys": ["HELLO"] * 101,
        })
        assert status == 400 and "too many" in invalid["error"]

        connection.request("POST", "/api/resolve", body=b'{}', headers={"Content-Type": "text/plain"})
        response = connection.getresponse()
        invalid_content_type = json.loads(response.read().decode("utf-8"))
        assert response.status == 400 and "Content-Type" in invalid_content_type["error"]
        print(f"production HTTP contract passed: {health['sign_count']} indexed signs")
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
