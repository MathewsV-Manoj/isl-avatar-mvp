"""Train a sentence-context ISL gloss tagger from the BPCC synthetic index.

This model predicts one gloss or DROP per input token. It is a weakly
supervised bridge model: the targets come from isolated-sign dictionary keys,
not native continuous gloss annotations.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


PAD = "<pad>"
UNK = "<unk>"
DROP = "<drop>"


class GlossDataset(Dataset):
    def __init__(self, rows, in_vocab, out_vocab, max_len):
        self.items = []
        for row in rows:
            tokens = row["tokens"][:max_len]
            inputs, targets = [], []
            for token in tokens:
                inputs.append(in_vocab.get(token, in_vocab[UNK]))
                clean = "_".join("".join(ch for ch in token.upper() if ch.isalnum()).split())
                targets.append(out_vocab.get(clean, out_vocab[DROP]))
            if inputs:
                self.items.append((torch.tensor(inputs), torch.tensor(targets)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    x = pad_sequence([a for a, _ in batch], batch_first=True, padding_value=0)
    y = pad_sequence([b for _, b in batch], batch_first=True, padding_value=0)
    return x, y


class GlossTagger(nn.Module):
    def __init__(self, input_size, output_size, hidden=192):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, output_size))

    def forward(self, x):
        encoded, _ = self.encoder(self.embedding(x))
        return self.head(encoded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=199963)
    ap.add_argument("--index", nargs="+", default=["datasets/bpcc_synthetic/index.jsonl"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--resume", default="")
    ap.add_argument("--mask-prob", type=float, default=0.0)
    ap.add_argument("--valid-index", nargs="+", default=[])
    ap.add_argument("--output", default="models/bpcc_gloss_tagger.pt")
    args = ap.parse_args()
    random.seed(7)
    torch.manual_seed(7)
    root = Path(__file__).resolve().parent
    rows = []
    for index_name in args.index:
        with (root / index_name).open(encoding="utf-8") as f:
            for line in f:
                if len(rows) >= args.limit:
                    break
                rows.append(json.loads(line))
        if len(rows) >= args.limit:
            break
    valid_rows = []
    for index_name in args.valid_index:
        with (root / index_name).open(encoding="utf-8") as f:
            for line in f:
                valid_rows.append(json.loads(line))
    tokens = sorted({t for row in rows for t in row["tokens"][:args.max_len]})
    glosses = sorted({g for row in rows for g in row["sign_keys"]})
    in_vocab = {PAD: 0, UNK: 1, **{t: i + 2 for i, t in enumerate(tokens)}}
    out_vocab = {PAD: 0, DROP: 1, **{g: i + 2 for i, g in enumerate(glosses)}}
    data = GlossDataset(rows, in_vocab, out_vocab, args.max_len)
    if valid_rows:
        train = data
        valid = GlossDataset(valid_rows, in_vocab, out_vocab, args.max_len)
        train_len = len(train)
        valid_len = len(valid)
    else:
        train_len = max(1, int(len(data) * .9))
        valid_len = len(data) - train_len
        train, valid = random_split(data, [train_len, valid_len], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = GlossTagger(len(in_vocab), len(out_vocab))
    if args.resume:
        checkpoint = torch.load(root / args.resume, map_location="cpu")
        if checkpoint.get("in_vocab") != in_vocab or checkpoint.get("out_vocab") != out_vocab:
            raise RuntimeError("Resume checkpoint vocabulary does not match the current training indices.")
        model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4 if args.resume else 2e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_valid_accuracy = 0.0
    final_valid_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            train_x = x.clone()
            if args.mask_prob > 0:
                mask = (torch.rand_like(train_x, dtype=torch.float32) < args.mask_prob) & (train_x != 0)
                train_x[mask] = in_vocab[UNK]
            loss = loss_fn(model(train_x).transpose(1, 2), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * x.size(0)
        model.eval(); correct = seen = 0
        with torch.no_grad():
            for x, y in valid_loader:
                pred = model(x).argmax(-1)
                mask = y != 0
                correct += ((pred == y) & mask).sum().item(); seen += mask.sum().item()
        final_valid_accuracy = correct / max(1, seen)
        best_valid_accuracy = max(best_valid_accuracy, final_valid_accuracy)
        print(f"epoch={epoch} train_loss={total / len(train):.4f} valid_token_accuracy={final_valid_accuracy:.4f}", flush=True)
    out = root / "models"; out.mkdir(exist_ok=True)
    output_path = root / args.output
    output_path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_vocab": in_vocab, "out_vocab": out_vocab, "max_len": args.max_len}, output_path)
    report = {"rows": len(data), "train_rows": train_len, "valid_rows": valid_len, "input_vocab": len(in_vocab), "output_vocab": len(out_vocab), "epochs": args.epochs, "resumed": bool(args.resume), "mask_prob": args.mask_prob, "best_valid_token_accuracy": best_valid_accuracy, "final_valid_token_accuracy": final_valid_accuracy, "weak_supervision": True, "continuous_pose_supervision": False, "source": args.index, "validation_source": args.valid_index or "random_split", "output": args.output}
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
