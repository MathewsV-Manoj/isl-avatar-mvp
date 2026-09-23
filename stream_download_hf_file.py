"""Resumable authenticated streaming download for a large Hugging Face file."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from huggingface_hub import get_token
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Authorization": f"Bearer {get_token()}"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{args.file}?download=true"
    session = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
        response.raise_for_status()
        total = response.headers.get("Content-Range", "").split("/")[-1] or response.headers.get("Content-Length", "?")
        mode = "ab" if existing and response.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        written = existing
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    handle.write(block)
                    written += len(block)
                    if written % (512 * 1024 * 1024) < len(block):
                        print(f"downloaded={written} total={total}", flush=True)
    partial.replace(output)
    print(f"complete={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
