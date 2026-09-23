"""Measure gloss accuracy separately for each sentence source.

This is a diagnostic, not a signer-independent benchmark: the current
checkpoint was trained on both sources. It is still more informative than a
single mixed random-split number because it exposes source-specific drift.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from train_bpcc_gloss_tagger import GlossTagger


ROOT = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--checkpoint", default="models/bpcc_gloss_tagger.pt")
ARGS = PARSER.parse_args()
CHECKPOINT = torch.load(ROOT / ARGS.checkpoint, map_location="cpu")
MODEL = GlossTagger(len(CHECKPOINT["in_vocab"]), len(CHECKPOINT["out_vocab"]))
MODEL.load_state_dict(CHECKPOINT["state_dict"])
MODEL.eval()


def evaluate(path: Path, limit: int = 50000):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    xs, ys = [], []
    for row in rows:
        tokens = row["tokens"][: CHECKPOINT.get("max_len", 32)]
        x, y = [], []
        for token in tokens:
            x.append(CHECKPOINT["in_vocab"].get(token, CHECKPOINT["in_vocab"]["<unk>"]))
            clean = "_".join("".join(ch for ch in token.upper() if ch.isalnum()).split())
            y.append(CHECKPOINT["out_vocab"].get(clean, CHECKPOINT["out_vocab"]["<drop>"]))
        if x:
            xs.append(torch.tensor(x))
            ys.append(torch.tensor(y))
    correct = total = 0
    for start in range(0, len(xs), 256):
        x = pad_sequence(xs[start:start + 256], batch_first=True, padding_value=0)
        y = pad_sequence(ys[start:start + 256], batch_first=True, padding_value=0)
        with torch.no_grad():
            pred = MODEL(x).argmax(-1)
        mask = y != 0
        correct += ((pred == y) & mask).sum().item()
        total += mask.sum().item()
    return {"rows": len(rows), "tokens": total, "correct": correct, "token_accuracy": round(correct / max(1, total), 6)}


def main():
    report = {
        "weak_supervision": True,
        "continuous_pose_supervision": False,
        "note": "Diagnostic only; checkpoint was trained on both sources.",
        "bpcc": evaluate(ROOT / "datasets/bpcc_synthetic/index.jsonl"),
        "blimp": evaluate(ROOT / "datasets/bpcc_synthetic/blimp_index.jsonl"),
    }
    output_name = "source_split_diagnostic.json" if ARGS.checkpoint == "models/bpcc_gloss_tagger.pt" else "source_split_diagnostic_isltranslate.json"
    (ROOT / output_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
