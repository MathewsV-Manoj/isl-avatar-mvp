"""Download a bounded, resumable official-derived vocabulary gap."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
INPUT = Path("official_isl_gap_manifest.json")
OUT = Path("datasets/official_isl_gap")
SELECTION = Path("official_isl_selected_manifest.json")


def choose(items: list[dict]) -> list[dict]:
    single = [item for item in items if re.fullmatch(r"[A-Z][A-Z0-9]*", item["label"]) and not item["label"].endswith(("_SIGN", "_ENGLISH"))]
    selected = single + sorted([item for item in items if item not in single], key=lambda item: item["size"] or 0)[:200]
    return selected


def download(item: dict) -> tuple[str, str]:
    target = OUT / item["path"]
    if target.exists() and target.stat().st_size == item["size"]:
        return item["label"], str(target)
    last_error = None
    for attempt in range(5):
        try:
            downloaded = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=item["path"], local_dir=OUT)
            return item["label"], downloaded
        except Exception as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{item['path']}: {last_error}")


def main() -> None:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    selected = choose(source["missing"])
    SELECTION.write_text(json.dumps({
        "source": REPO,
        "source_license": "MIT repository; verify original ISLRTC conditions before product use",
        "selected_count": len(selected),
        "estimated_bytes": sum(item["size"] or 0 for item in selected),
        "selected": selected,
    }, indent=2), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, item): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
                completed += 1
                if completed % 50 == 0:
                    print(f"downloaded={completed}/{len(selected)}", flush=True)
            except Exception as error:
                failed.append({"path": item["path"], "error": str(error)})
    Path("official_isl_download_report.json").write_text(json.dumps({
        "selected": len(selected),
        "completed": completed,
        "failed": failed,
    }, indent=2), encoding="utf-8")
    print(f"official gap download complete: {completed}/{len(selected)}; failed={len(failed)}")


if __name__ == "__main__":
    main()
