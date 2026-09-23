"""Train a small text-to-ISL-sign bootstrap model.

This is a vocabulary/sequence baseline, not a substitute for training on
continuous ISL pose clips. It uses the local sentence metadata to create
weak targets from the validated sign dictionary and records that provenance
in the checkpoint report.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DROP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "with"}
PAD, BOS, EOS, UNK, FALLBACK = "<pad>", "<bos>", "<eos>", "<unk>", "<fallback>"


def norm_key(token: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", token.upper()).strip("_")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def weak_target(tokens, signs):
    out = []
    for token in tokens:
        if token in DROP:
            continue
        key = norm_key(token)
        if key in signs:
            out.append(key)
        else:
            out.append(FALLBACK)
    return out or [FALLBACK]


class SentenceDataset(Dataset):
    def __init__(self, rows, in_vocab, out_vocab, max_len):
        self.items = []
        for row in rows:
            src = [in_vocab.get(t, in_vocab[UNK]) for t in row["tokens"][:max_len]]
            tgt = [out_vocab[BOS]] + [out_vocab[t] for t in weak_target(row["tokens"][:max_len], signs)] + [out_vocab[EOS]]
            self.items.append((src, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    src_len = max(len(x[0]) for x in batch)
    tgt_len = max(len(x[1]) for x in batch)
    src = torch.zeros(len(batch), src_len, dtype=torch.long)
    tgt = torch.zeros(len(batch), tgt_len, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, :len(s)] = torch.tensor(s)
        tgt[i, :len(t)] = torch.tensor(t)
    return src, tgt


class ContextTagger(nn.Module):
    def __init__(self, in_size, out_size, hidden=192):
        super().__init__()
        self.embedding = nn.Embedding(in_size, hidden, padding_idx=0)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(hidden * 2, hidden, batch_first=True)
        self.output = nn.Linear(hidden, out_size)

    def forward(self, src, target_len):
        encoded, _ = self.encoder(self.embedding(src))
        context = encoded.mean(dim=1, keepdim=True).expand(-1, target_len, -1)
        decoded, _ = self.decoder(context)
        return self.output(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent
    rows = load_jsonl(root / "datasets/isign_sentence_metadata/normalized.jsonl")[:args.limit]
    global signs
    with (root / "signal_dictionary_normalized.json").open(encoding="utf-8") as f:
        signs = json.load(f)

    tokens = sorted({t for row in rows for t in row["tokens"]})
    in_vocab = {PAD: 0, UNK: 1, **{t: i + 2 for i, t in enumerate(tokens)}}
    targets = sorted({x for row in rows for x in weak_target(row["tokens"][:args.max_len], signs) if x != FALLBACK})
    out_tokens = [PAD, BOS, EOS, FALLBACK] + targets
    out_vocab = {t: i for i, t in enumerate(out_tokens)}
    dataset = SentenceDataset(rows, in_vocab, out_vocab, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = ContextTagger(len(in_vocab), len(out_vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for src, tgt in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(src, tgt.shape[1])
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * src.shape[0]
        print(f"epoch={epoch} loss={total / len(dataset):.4f}", flush=True)

    out = root / "models"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_vocab": in_vocab, "out_vocab": out_vocab, "max_len": args.max_len}, out / "sentence_bootstrap.pt")
    report = {"rows": len(rows), "input_vocab": len(in_vocab), "output_vocab": len(out_vocab), "epochs": args.epochs, "weak_target": True, "pose_supervision": False, "source": "local iSign sentence metadata + local validated sign dictionary"}
    (out / "sentence_bootstrap_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
