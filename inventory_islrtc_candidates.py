"""Compare public ISLRTC re-encode filenames with the local lookup labels."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def canonical(value: str) -> str:
    value = re.sub(r"\s*\(Sign\s*\d+\)\s*$", "", value, flags=re.I)
    value = re.sub(r"\s+", "_", value.replace("&", "AND"))
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def labels(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("signs"), dict):
        data = data["signs"]
    return {canonical(str(key)) for key in data}


def main() -> None:
    entries = json.loads((ROOT / "datasets/islrtc_files.json").read_text(encoding="utf-8"))
    files = [item["path"] for item in entries if item.get("path", "").lower().endswith(".mp4")]
    by_label: dict[str, list[str]] = {}
    for path in files:
        stem = Path(path).stem
        key = canonical(stem)
        by_label.setdefault(key, []).append(path)
    known: set[str] = set()
    for path in (
        ROOT / "signal_dictionary_cislr/manifest.json",
        ROOT / "signal_dictionary_official/manifest.json",
        ROOT / "signal_dictionary_bridgeconn/manifest.json",
        ROOT / "signal_dictionary_include/manifest.json",
    ):
        known |= labels(path)
    new_labels = sorted(set(by_label) - known)
    report = {
        "source": "Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded",
        "source_license": "MIT per dataset card; verify upstream data.gov.in terms before redistribution",
        "video_files": len(files),
        "unique_filename_labels": len(by_label),
        "known_label_overlap": len(set(by_label) & known),
        "new_filename_labels": len(new_labels),
        "new_label_examples": new_labels[:250],
        "metadata_only": True,
        "note": "No videos downloaded; filenames are not sufficient to prove sign semantics or quality.",
    }
    (ROOT / "datasets/islrtc_candidate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "datasets/islrtc_candidate_labels.json").write_text(json.dumps({key: by_label[key] for key in new_labels}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
