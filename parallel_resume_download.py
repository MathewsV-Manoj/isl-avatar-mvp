"""Download the remaining bytes of a gated Hugging Face file concurrently."""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import get_token


def fetch(url: str, token: str, start: int, end: int, path: Path, retries: int = 6) -> None:
    headers = {"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                with path.open("wb") as handle:
                    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            expected = end - start + 1
            if path.stat().st_size != expected:
                raise RuntimeError(f"short range: {path.stat().st_size}/{expected}")
            return
        except Exception:
            if attempt == retries - 1:
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-gib", type=float, default=4.0)
    args = parser.parse_args()
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing >= args.total:
        partial.replace(output)
        print(f"complete={output} bytes={output.stat().st_size}", flush=True)
        return
    root = partial.parent / (partial.name + ".ranges")
    root.mkdir(parents=True, exist_ok=True)
    step = max(1, int(args.chunk_gib * 1024**3))
    ranges = []
    start = existing
    index = 0
    while start < args.total:
        end = min(args.total - 1, start + step - 1)
        ranges.append((index, start, end))
        start = end + 1
        index += 1
    token = get_token()
    url = f"https://huggingface.co/datasets/{args.repo}/resolve/main/{args.file}?download=true"
    print(f"existing={existing} remaining={args.total-existing} ranges={len(ranges)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(fetch, url, token, start, end, root / f"{i:04d}.part"): (i, start, end)
            for i, start, end in ranges
        }
        for job in as_completed(jobs):
            i, start, end = jobs[job]
            job.result()
            print(f"range_complete={i} bytes={end-start+1}", flush=True)
    with partial.open("ab") as destination:
        for i, _, _ in ranges:
            chunk = root / f"{i:04d}.part"
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            chunk.unlink()
    root.rmdir()
    if partial.stat().st_size != args.total:
        raise RuntimeError(f"final size mismatch: {partial.stat().st_size}/{args.total}")
    partial.replace(output)
    print(f"complete={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
