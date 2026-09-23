"""Resumable downloader for the public CC BY 4.0 INCLUDE ISL corpus.

The corpus provides isolated-sign video variation. It is kept separate from
sentence-to-pose data and is only used after label and landmark validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


RECORD_URL = "https://zenodo.org/api/records/4010759"
ROOT = Path("datasets/include/videos")
MANIFEST = Path("datasets/include/download_manifest.json")


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_file(url: str, destination: Path, expected_bytes: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size > expected_bytes:
        partial.unlink()
    # Zenodo's large-file redirect can stall under urllib on this host, while
    # the Windows curl client follows it and safely resumes partial archives.
    subprocess.run([
        "curl.exe", "--fail", "--location", "--continue-at", "-",
        "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
        "--speed-time", "120", "--speed-limit", "1024",
        "--output", str(partial), url,
    ], check=True)
    actual = partial.stat().st_size
    if actual != expected_bytes:
        raise RuntimeError(f"{destination.name}: expected {expected_bytes}, received {actual}")
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Download only the first N archives for a pilot.")
    parser.add_argument("--category", help="Only download archives whose filename starts with this category.")
    args = parser.parse_args()

    record = fetch_json(RECORD_URL)
    files = [item for item in record["files"] if item["key"].endswith(".zip")]
    if args.category:
        prefix = args.category.lower() + "_"
        files = [item for item in files if item["key"].lower().startswith(prefix)]
    files.sort(key=lambda item: item["key"])
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No INCLUDE archives match the requested selection.")

    ROOT.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for number, item in enumerate(files, start=1):
        destination = ROOT / item["key"]
        expected = int(item["size"])
        if destination.exists() and destination.stat().st_size == expected:
            print(f"[{number}/{len(files)}] verified {destination.name}", flush=True)
        else:
            print(f"[{number}/{len(files)}] downloading {destination.name} ({expected / 1e9:.2f} GB)", flush=True)
            download_file(item["links"]["self"], destination, expected)
            print(f"[{number}/{len(files)}] complete {destination.name}", flush=True)
        completed.append({"name": item["key"], "bytes": expected, "url": item["links"]["self"]})

    MANIFEST.write_text(json.dumps({
        "source": "Zenodo record 4010759",
        "license": "CC-BY-4.0",
        "archives": completed,
    }, indent=2), encoding="utf-8")
    print(f"INCLUDE download complete: {len(completed)} archives", flush=True)


if __name__ == "__main__":
    main()
