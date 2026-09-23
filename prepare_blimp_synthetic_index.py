from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def key(word: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", word.upper()).strip("_")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=500000); args = ap.parse_args()
    root = Path(__file__).resolve().parent
    sign_keys = set()
    for name in ("signal_dictionary_cislr/manifest.json", "signal_dictionary_bridgeconn/manifest.json", "signal_dictionary_official/manifest.json"):
        p = root / name
        if p.exists(): sign_keys.update(json.loads(p.read_text(encoding="utf-8")).get("signs", {}).keys())
    out_dir = root / "datasets/bpcc_synthetic"; out_dir.mkdir(exist_ok=True)
    out = out_dir / "blimp_index.jsonl"
    counts = {"rows_read": 0, "kept": 0, "covered_tokens": 0, "missing_tokens": 0}
    with (root / "datasets/posestitch_isl/BLIMP-ISL.csv").open(encoding="utf-8", newline="") as src, out.open("w", encoding="utf-8") as dst:
        for row in csv.DictReader(src):
            if counts["rows_read"] >= args.limit: break
            counts["rows_read"] += 1
            try: tokens = ast.literal_eval(row.get("tokenized_sentences", "[]"))
            except (SyntaxError, ValueError): tokens = []
            tokens = [str(x).lower() for x in tokens if str(x).strip()]
            if not tokens: continue
            covered = [key(t) for t in tokens if key(t) in sign_keys]
            missing = [t for t in tokens if key(t) not in sign_keys]
            if len(covered) < 2: continue
            dst.write(json.dumps({"source":"PoseStitch-ISL/BLIMP-ISL.csv","sentence_id":row.get("sentence_id"),"text":row.get("sentence_good", ""),"tokens":tokens[:32],"sign_keys":covered,"missing":missing,"synthetic":True}, ensure_ascii=False)+"\n")
            counts["kept"] += 1; counts["covered_tokens"] += len(covered); counts["missing_tokens"] += len(missing)
    counts["dictionary_keys"] = len(sign_keys); counts["coverage"] = round(counts["covered_tokens"] / max(1, counts["covered_tokens"] + counts["missing_tokens"]), 4)
    (out_dir / "blimp_report.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__": main()
