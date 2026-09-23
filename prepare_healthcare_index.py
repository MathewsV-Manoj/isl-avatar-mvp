"""Convert the public ISL healthcare text/gloss corpus into safe weak rows."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "datasets/isl_healthcare/isl_medical.jsonl"
SIGNS = ROOT / "signal_dictionary_normalized.json"
OUTPUT = ROOT / "datasets/isl_healthcare/bpcc_index.jsonl"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def main() -> None:
    signs = json.loads(SIGNS.read_text(encoding="utf-8"))
    rows = []
    skipped = 0
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        source_tokens = tokens(item.get("text_en", ""))
        glosses = [g.strip().upper() for g in item.get("gloss_sequence", "").split("+") if g.strip()]
        known = [g for g in glosses if g in signs]
        if not source_tokens or not known:
            skipped += 1
            continue
        rows.append({
            "tokens": source_tokens,
            "sign_keys": known,
            "source": "sahilmaniyar888/isl-healthcare-instruction-corpus",
            "sample_id": item.get("sample_id"),
            "category": item.get("category"),
            "native_pose_supervision": False,
        })
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    (OUTPUT.with_name("bpcc_index_report.json")).write_text(json.dumps({
        "source_rows": len(rows) + skipped,
        "kept_rows": len(rows),
        "skipped_rows": skipped,
        "known_glosses": len({g for row in rows for g in row["sign_keys"]}),
        "native_pose_supervision": False,
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    print(json.dumps({"kept_rows": len(rows), "skipped_rows": skipped}))


if __name__ == "__main__":
    main()
