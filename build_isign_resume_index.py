"""Filter iSign weak rows to the live model vocabulary for safe fine-tuning."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> None:
    root = Path(__file__).resolve().parent
    checkpoint = torch.load(root / "models/bpcc_gloss_tagger.pt", map_location="cpu")
    input_vocab = set(checkpoint["in_vocab"])
    output_vocab = set(checkpoint["out_vocab"])
    source = root / "datasets/isign/isign_weak_index.jsonl"
    destination = root / "datasets/isign/isign_resume_index.jsonl"
    seen = kept = 0
    with source.open(encoding="utf-8") as source_handle, destination.open("w", encoding="utf-8") as target:
        for line in source_handle:
            row = json.loads(line)
            tokens = [token for token in row["tokens"] if token in input_vocab]
            if not tokens:
                continue
            sign_keys = [key for key in row["sign_keys"] if key in output_vocab]
            seen += 1
            target.write(json.dumps({**row, "tokens": tokens, "sign_keys": sign_keys}) + "\n")
            kept += 1
    print(json.dumps({"rows_seen": seen, "rows_kept": kept, "input_vocab": len(input_vocab), "output_vocab": len(output_vocab), "output": str(destination)}))


if __name__ == "__main__":
    main()
