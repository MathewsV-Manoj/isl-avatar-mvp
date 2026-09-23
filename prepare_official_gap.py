"""Enumerate missing official-derived ISL videos without downloading the corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi


REPO = "silentone0725/Indian_Sign_Language_Data.gov_Rencoded"
OUT = Path("official_isl_gap_manifest.json")


def key(value: str) -> str:
    return re.sub(r"^_|_$", "", re.sub(r"[^A-Z0-9]+", "_", value.upper()))


def load_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "signs" in data:
        return {key(name) for name in data["signs"]}
    return {key(name) for name in data}


def main() -> None:
    covered = set()
    covered |= load_keys(Path("signal_dictionary_cislr/manifest.json"))
    covered |= load_keys(Path("signal_dictionary_bridgeconn/manifest.json"))
    covered |= load_keys(Path("signal_dictionary_supplemental_filtered.json"))

    api = HfApi()
    missing = []
    total_videos = 0
    for item in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=False):
        path = getattr(item, "path", "")
        if not path.lower().endswith((".mp4", ".webm", ".mov")):
            continue
        total_videos += 1
        label = key(Path(path).stem)
        if label not in covered:
            missing.append({"label": label, "path": path, "size": getattr(item, "size", None)})

    OUT.write_text(json.dumps({
        "source": REPO,
        "source_license": "MIT repository; verify original ISLRTC conditions before product use",
        "covered_labels": len(covered),
        "remote_video_count": total_videos,
        "missing_video_count": len(missing),
        "missing": missing,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "covered_labels": len(covered),
        "remote_video_count": total_videos,
        "missing_video_count": len(missing),
    }))


if __name__ == "__main__":
    main()
