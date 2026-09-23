"""Build a conservative weak sentence index from iSign transcripts.

Only transcript tokens that already have a validated local sign are emitted as
targets; unknown tokens stay in the input sentence but are not fabricated as
glosses.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def main() -> None:
    root = Path(r"E:\ISL_Project_Datasets\isign")
    project = Path(__file__).resolve().parent
    known: set[str] = set(json.loads((project / "signal_dictionary_fixed.json").read_text(encoding="utf-8")))
    for manifest in project.glob("signal_dictionary_*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            known.update(data.get("signs", {}).keys())
        except Exception:
            pass
    rows = kept = 0
    output = project / "datasets" / "isign" / "isign_weak_index.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with (root / "iSign_v1.1.csv").open(encoding="utf-8-sig", newline="") as source, output.open("w", encoding="utf-8") as target:
        for row in csv.DictReader(source):
            text = row["text"].strip().lower()
            tokens = re.findall(r"[a-z0-9]+", text)
            if not tokens:
                continue
            sign_keys = [token.upper() for token in tokens if token.upper() in known]
            rows += 1
            if not sign_keys:
                continue
            target.write(json.dumps({"tokens": tokens, "sign_keys": sign_keys, "source": "Exploration-Lab/iSign", "uid": row["uid"]}) + "\n")
            kept += 1
    print(json.dumps({"rows_seen": rows, "rows_kept": kept, "known_signs": len(known), "output": str(output)}))


if __name__ == "__main__":
    main()
