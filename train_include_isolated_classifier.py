"""Train a baseline temporal classifier on validated INCLUDE landmarks.

This model is a recognition benchmark only. It is not promoted as the
sentence-to-ISL translator and does not alter the browser avatar checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA = Path("signal_dictionary_include_filtered.json")
OUT = Path("models/include_isolated_classifier.pt")
REPORT = Path("models/include_isolated_classifier_report.json")


class TemporalClassifier(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(225, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2)))


class StrongTemporalClassifier(nn.Module):
    """Bidirectional recurrent model that preserves frame order explicitly."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(225),
            nn.Linear(225, 192),
            nn.GELU(),
            nn.Dropout(0.08),
        )
        self.temporal = nn.GRU(192, 128, num_layers=2, batch_first=True,
                               dropout=0.15, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(256, 96), nn.Tanh(), nn.Linear(96, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
            nn.Dropout(0.25), nn.Linear(256, classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.frame_encoder(x)
        sequence, _ = self.temporal(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (sequence * weights).sum(dim=1)
        summary = torch.cat((pooled, sequence[:, 0], sequence[:, -1]), dim=1)
        return self.head(summary)


class AugmentedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, augment: bool) -> None:
        self.x, self.y, self.augment = x, y, augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.x[index].clone()
        if self.augment:
            # Small landmark noise and time masking model MediaPipe jitter and
            # occasional low-confidence frames without changing the sign label.
            sample = sample + torch.randn_like(sample) * 0.006
            if random.random() < 0.35:
                start = random.randrange(0, max(1, sample.shape[0] - 2))
                width = random.choice((1, 2))
                sample[start : start + width] = sample[max(0, start - 1)]
            if random.random() < 0.25:
                sample = sample * random.uniform(0.96, 1.04)
        return sample, self.y[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--archive-holdout", action="store_true", help="Hold out one archive per label when provenance supports it.")
    parser.add_argument("--strong", action="store_true", help="Use the order-aware bidirectional temporal model.")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source = json.loads(args.data.read_text(encoding="utf-8"))
    labels = sorted(source)
    label_to_id = {label: index for index, label in enumerate(labels)}
    train, valid = [], []
    provenance_path = Path("include_candidate_manifest.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")).get("provenance", {}) if provenance_path.exists() else {}
    for label in labels:
        clips = [np.asarray(clip, dtype=np.float32) for clip in source[label]]
        if args.archive_holdout:
            entries = provenance.get(label, [])
            archives = [str(item.get("archive", "")) for item in entries]
            groups = sorted({item for item in archives if item})
            holdout = groups[-1] if len(groups) > 1 else None
            for index, clip in enumerate(clips):
                archive = archives[index] if index < len(archives) else ""
                if holdout and archive == holdout:
                    valid.append((clip, label_to_id[label]))
                else:
                    train.append((clip, label_to_id[label]))
        else:
            random.Random(args.seed + label_to_id[label]).shuffle(clips)
            cut = max(1, round(len(clips) * 0.2)) if len(clips) > 1 else 0
            valid.extend((clip, label_to_id[label]) for clip in clips[:cut])
            train.extend((clip, label_to_id[label]) for clip in clips[cut:])
    if not valid:
        raise RuntimeError("Need at least two clips for a validation split.")

    def tensors(items: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.stack([item[0] for item in items])), torch.tensor([item[1] for item in items], dtype=torch.long)

    x_train, y_train = tensors(train)
    x_valid, y_valid = tensors(valid)
    loader = DataLoader(AugmentedDataset(x_train, y_train, args.augment), batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (StrongTemporalClassifier(len(labels)) if args.strong else TemporalClassifier(len(labels))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = {"accuracy": 0.0, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(x_valid.to(device)).argmax(1).cpu()
        accuracy = float((pred == y_valid).float().mean().item())
        history.append({"epoch": epoch, "train_loss": total_loss / len(x_train), "valid_accuracy": accuracy})
        print(json.dumps(history[-1]), flush=True)
        if accuracy >= best["accuracy"]:
            best = {"accuracy": accuracy, "state": {key: value.detach().cpu() for key, value in model.state_dict().items()}}

    OUT.parent.mkdir(exist_ok=True)
    args.output.parent.mkdir(exist_ok=True)
    source_name = "vidit031/isl-isolated-40words" if args.data.name == "isl_isolated_40words_filtered.json" else "AI4Bharat/INCLUDE"
    license_name = "research; respect upstream licenses" if source_name.startswith("vidit031") else "CC-BY-4.0"
    torch.save({"state_dict": best["state"], "labels": labels, "feature_shape": [30, 225], "source": source_name, "license": license_name, "architecture": "strong_temporal" if args.strong else "conv_temporal"}, args.output)
    args.report.write_text(json.dumps({"source": source_name, "license": license_name, "split": "archive_holdout" if args.archive_holdout else "random_clip", "train_clips": len(train), "valid_clips": len(valid), "labels": len(labels), "best_valid_accuracy": best["accuracy"], "epochs": args.epochs, "augmentation": args.augment, "architecture": "strong_temporal" if args.strong else "conv_temporal", "device": str(device), "history": history}, indent=2), encoding="utf-8")
    print(json.dumps({"best_valid_accuracy": best["accuracy"], "train_clips": len(train), "valid_clips": len(valid), "labels": len(labels)}), flush=True)


if __name__ == "__main__":
    main()
