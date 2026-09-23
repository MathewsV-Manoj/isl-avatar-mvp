"""Select only new labels from the downloaded MIT social-interaction corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def main() -> None:
    known = set()
    for name in (
        "signal_dictionary_cislr/manifest.json",
        "signal_dictionary_official/manifest.json",
        "signal_dictionary_bridgeconn/manifest.json",
        "signal_dictionary_include/manifest.json",
        "signal_dictionary_islrtc_batch/manifest.json",
    ):
        known.update(json.loads((ROOT / name).read_text(encoding="utf-8")).get("signs", {}))
    selected = []
    for path in sorted((ROOT / "datasets/kaggle_social/unpacked/isl_videos").glob("*.mp4")):
        label = key(path.stem)
        if label not in known:
            selected.append({"path": path.name, "label": label})
    output = {"source": "prasadshet/indian-sign-language-video-dataset", "license": "MIT", "selected": selected}
    (ROOT / "datasets/kaggle_social/selection.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "labels": [row["label"] for row in selected]}, indent=2))


if __name__ == "__main__":
    main()
