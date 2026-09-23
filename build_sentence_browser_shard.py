"""Build a compact exact-phrase browser shard from validated ISL sentence clips."""

from __future__ import annotations

import json
import re
from pathlib import Path


def key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def main() -> None:
    source_path = Path("datasets/isl_csltr_sentence_filtered.json")
    output = Path("signal_dictionary_islrtc_sentences")
    output.mkdir(exist_ok=True)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    selected = {key(label): clips[0] for label, clips in sorted(source.items()) if clips}
    shard = "sentences-0000.json"
    (output / shard).write_text(json.dumps(selected, separators=(",", ":")), encoding="utf-8")
    signs = {name: {"shard": shard, "gloss": name.lower().replace("_", " ")} for name in selected}
    manifest = {
        "format": "isl-avatar-sentence-dictionary-v1",
        "source": "ISL-CSLTR sentence videos",
        "license": "CC BY-SA 4.0",
        "feature_shape": [30, 225],
        "sign_count": len(selected),
        "shards": [{"file": shard, "count": len(selected), "keys": sorted(selected)}],
        "signs": signs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"sentences": len(selected), "shard": str(output / shard)}))


if __name__ == "__main__":
    main()
