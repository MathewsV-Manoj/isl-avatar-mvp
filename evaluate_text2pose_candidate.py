"""Compare a text-to-pose checkpoint against a train-set mean-pose baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_isign_text2pose import TextToPose, is_valid, make_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/isign_sentence_pose_v2")
    parser.add_argument("--checkpoint", default="models/isign_text2pose_48h_candidate.pt")
    parser.add_argument("--output", default="models/isign_text2pose_48h_evaluation.json")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    files = sorted(Path(args.data).glob("*.npz"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = TextToPose(len(checkpoint["vocab"]), hidden=int(checkpoint["config"]["hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total, count = None, 0
    for path in files:
        with np.load(path, allow_pickle=False) as pack:
            poses, uids = pack["poses"].astype(np.float32), pack["uids"].astype(str)
            chosen = [i for i, uid in enumerate(uids) if not is_valid(uid, args.valid_fraction)]
            if not chosen:
                continue
            values = poses[chosen]
            total = values.sum(axis=0, dtype=np.float64) if total is None else total + values.sum(axis=0, dtype=np.float64)
            count += len(values)
    baseline = torch.from_numpy((total / count).astype(np.float32))

    candidate_loss = baseline_loss = 0.0
    valid_count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as pack:
            poses, uids, texts = pack["poses"].astype(np.float32), pack["uids"].astype(str), pack["texts"].astype(str)
            chosen = [i for i, uid in enumerate(uids) if is_valid(uid, args.valid_fraction)]
            for start in range(0, len(chosen), args.batch_size):
                indices = chosen[start : start + args.batch_size]
                if not indices:
                    continue
                target = torch.from_numpy(poses[indices])
                source, mask = make_batch([texts[i] for i in indices], checkpoint["vocab"], torch.device("cpu"))
                with torch.no_grad():
                    prediction = model(source, mask)
                candidate_loss += torch.nn.functional.smooth_l1_loss(prediction, target, reduction="sum").item()
                baseline_loss += torch.nn.functional.smooth_l1_loss(baseline.expand_as(target), target, reduction="sum").item()
                valid_count += len(indices)

    elements = valid_count * 30 * 225
    result = {
        "checkpoint": args.checkpoint,
        "held_out_clips": valid_count,
        "candidate_smooth_l1": candidate_loss / elements,
        "mean_pose_baseline_smooth_l1": baseline_loss / elements,
        "relative_improvement_over_mean": 1 - candidate_loss / max(baseline_loss, 1e-12),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
