"""Select a bounded, simple-label batch from the ISLRTC metadata inventory."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def canonical(value: str) -> str:
    value = re.sub(r"\s*\(Sign\s*\d+\)\s*$", "", value, flags=re.I)
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def main() -> None:
    entries = json.loads((ROOT / "datasets/islrtc_files.json").read_text(encoding="utf-8"))
    known = set()
    for manifest_name in (
        "signal_dictionary_cislr/manifest.json",
        "signal_dictionary_official/manifest.json",
        "signal_dictionary_bridgeconn/manifest.json",
        "signal_dictionary_include/manifest.json",
        "signal_dictionary_islrtc_batch/manifest.json",
        "signal_dictionary_kaggle_social/manifest.json",
    ):
        manifest = json.loads((ROOT / manifest_name).read_text(encoding="utf-8"))
        known.update(manifest.get("signs", {}).keys())
    candidates = []
    seen = set()
    for item in entries:
        path = item.get("path", "")
        if not path.lower().endswith(".mp4"):
            continue
        key = canonical(Path(path).stem)
        if not key or key in seen:
            continue
        if key in known:
            continue
        seen.add(key)
        words = key.split("_")
        if len(words) > 4 or any(word.startswith("SIGN") for word in words):
            continue
        if any(term in key for term in ("RHYME", "SYMBOL", "FORMULA", "THEORY", "PRINCIPLE", "SILHOUETTE")):
            continue
        candidates.append({"label": key, "path": path, "bytes": item.get("size", 0)})
    candidates.sort(key=lambda item: (len(item["label"].split("_")), item["label"]))
    batch = candidates[:30]
    report = {
        "source": "Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded",
        "source_license": "MIT per dataset card; verify upstream data.gov.in terms before redistribution",
        "selected": len(batch),
        "estimated_bytes": sum(item["bytes"] or 0 for item in batch),
        "selection": "simple labels, one clip per label, no explicit Sign N suffix, metadata only",
        "items": batch,
    }
    (ROOT / "datasets/islrtc_batch_100.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected", "estimated_bytes", "selection")}, indent=2))


if __name__ == "__main__":
    main()
