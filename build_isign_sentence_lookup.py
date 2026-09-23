"""Build an exact-text index for signer-recorded iSign pose sequences."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).lower()))


def main() -> None:
    root = Path("datasets/isign_sentence_pose_v2")
    index: dict[str, dict[str, object]] = {}
    duplicates = 0
    for path in sorted(root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as pack:
            for position, text in enumerate(pack["texts"].astype(str)):
                normalized = normalize(text)
                if not normalized:
                    continue
                if normalized in index:
                    duplicates += 1
                    continue
                # Defer pose quality checks until retrieval. Reading and
                # scoring every compressed pose here takes hours on CPU.
                index[normalized] = {"file": path.name, "index": position}
    output = Path("models/isign_sentence_pose_lookup.json")
    output.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    report = {"source": str(root), "exact_sentences": len(index), "duplicates": duplicates, "output": str(output)}
    output.with_name("isign_sentence_pose_lookup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
