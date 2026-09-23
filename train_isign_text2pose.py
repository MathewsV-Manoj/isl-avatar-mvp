"""Train a real text-to-pose baseline from normalized iSign sentence clips.

This is deliberately separate from the weak text-to-gloss tagger.  It learns
from the native iSign pose arrays and predicts the complete 30 x 225 motion
sequence.  The checkpoint is only promoted after the held-out pose error is
reported by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
PAD, UNK = 0, 1


def tokens(text: str) -> list[str]:
    return [x.lower() for x in TOKEN_RE.findall(str(text))]


def is_valid(uid: str, valid_fraction: float) -> bool:
    value = int(hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < valid_fraction


def build_vocab(files: list[Path], max_vocab: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as pack:
            for text in pack["texts"]:
                for token in tokens(str(text)):
                    counts[token] = counts.get(token, 0) + 1
    words = sorted(counts, key=lambda x: (-counts[x], x))[: max_vocab - 2]
    return {"<pad>": PAD, "<unk>": UNK, **{word: i + 2 for i, word in enumerate(words)}}


def make_batch(texts: list[str], vocab: dict[str, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [[vocab.get(token, UNK) for token in tokens(text)] or [UNK] for text in texts]
    width = max(len(row) for row in rows)
    x = torch.full((len(rows), width), PAD, dtype=torch.long, device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for row, values in enumerate(rows):
        x[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = True
    return x, mask


class TextToPose(nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 192, output_size: int = 30 * 225):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, output_size),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.embedding(x))
        lengths = mask.sum(dim=1).clamp_min(1)
        pooled = encoded.sum(dim=1) / lengths.unsqueeze(1)
        return self.head(pooled).view(-1, 30, 225)


def run_epoch(model, files, vocab, device, optimizer, batch_size, valid_fraction, train):
    model.train(train)
    total_loss = 0.0
    total_count = 0
    rng = random.Random(11)
    ordered = list(files)
    if train:
        rng.shuffle(ordered)
    for path in ordered:
        with np.load(path, allow_pickle=False) as pack:
            poses = pack["poses"].astype(np.float32, copy=False)
            uids = pack["uids"].astype(str)
            texts = pack["texts"].astype(str)
            selected = [i for i, uid in enumerate(uids) if is_valid(uid, valid_fraction) == (not train)]
            for start in range(0, len(selected), batch_size):
                indices = selected[start : start + batch_size]
                target = torch.from_numpy(poses[indices]).to(device)
                x, mask = make_batch([texts[i] for i in indices], vocab, device)
                with torch.set_grad_enabled(train):
                    prediction = model(x, mask)
                    loss = nn.functional.smooth_l1_loss(prediction, target)
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                total_loss += float(loss.item()) * len(indices)
                total_count += len(indices)
    return total_loss / max(total_count, 1), total_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/isign_sentence_pose_v2")
    ap.add_argument("--output", default="models/isign_text2pose_candidate.pt")
    ap.add_argument("--report", default="models/isign_text2pose_candidate_report.json")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--max-vocab", type=int, default=20000)
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    files = sorted(Path(args.data).glob("*.npz"))
    if not files:
        raise SystemExit(f"no pose shards found under {args.data}")
    vocab = build_vocab(files, args.max_vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextToPose(len(vocab), hidden=args.hidden).to(device)
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        if state.get("vocab") != vocab:
            raise SystemExit("resume vocabulary does not match current dataset vocabulary")
        model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    report = {
        "source": "Exploration-Lab/iSign",
        "native_pose_supervision": True,
        "shards": len(files),
        "vocab_size": len(vocab),
        "device": str(device),
        "epochs": [],
    }
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_count = run_epoch(model, files, vocab, device, optimizer, args.batch_size, args.valid_fraction, True)
        with torch.no_grad():
            valid_loss, valid_count = run_epoch(model, files, vocab, device, optimizer, args.batch_size, args.valid_fraction, False)
        item = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "train_count": train_count, "valid_count": valid_count}
        report["epochs"].append(item)
        print(json.dumps(item), flush=True)
        if valid_loss < best:
            best = valid_loss
            torch.save({"model": model.state_dict(), "vocab": vocab, "config": {"hidden": args.hidden, "feature_shape": [30, 225]}}, args.output)
    report["best_valid_smooth_l1"] = best
    report["checkpoint"] = str(Path(args.output))
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
