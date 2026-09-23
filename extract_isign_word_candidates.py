"""Extract one normalized, source-provenanced candidate per iSign single-word label."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from pose_format import Pose

from import_isl500_landmarks import normalize_clip
from inspect_isign_split_zip import SplitFile


WORD = re.compile(r"^[A-Za-z]+$")
PARTS = ("aa", "ab", "ac", "ad")


def load_texts(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["uid"]: row["text"].strip() for row in csv.DictReader(handle)}


def decode_clip(raw: bytes) -> np.ndarray | None:
    pose = Pose.read(io.BytesIO(raw))
    selected = pose.get_components([
        "POSE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "RIGHT_HAND_LANDMARKS",
    ])
    values = np.ma.filled(selected.body.data, 0.0).astype(np.float32)[:, 0, :, :3]
    if values.shape[1] != 75:
        return None
    body = values[:, :33]
    hands = np.stack((values[:, 33:54], values[:, 54:75]), axis=1)
    return normalize_clip(body, hands)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"E:\ISL_Project_Datasets\isign")
    parser.add_argument("--max-words", type=int, default=10000)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.root)
    texts = load_texts(root / "iSign_v1.1.csv")
    paths = [root / f"iSign-poses_v1.1_part_{part}" for part in PARTS]
    output = {}
    provenance = {}
    stats = {"entries_seen": 0, "single_word_entries": 0, "decoded": 0, "failed": 0, "skipped_existing": 0}
    stream = SplitFile(paths)
    try:
        with zipfile.ZipFile(stream) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".pose"):
                    continue
                stats["entries_seen"] += 1
                uid = Path(info.filename).stem
                text = texts.get(uid, "")
                if not WORD.fullmatch(text):
                    continue
                stats["single_word_entries"] += 1
                label = text.upper()
                if label in output:
                    stats["skipped_existing"] += 1
                    continue
                try:
                    clip = decode_clip(archive.read(info))
                    if clip is None or not np.isfinite(clip).all() or float(np.max(np.abs(clip))) > 20.0:
                        stats["failed"] += 1
                        continue
                    output[label] = clip.tolist()
                    provenance[label] = {"uid": uid, "text": text, "source": "Exploration-Lab/iSign", "fps": 25}
                    stats["decoded"] += 1
                    if len(output) >= args.max_words:
                        break
                except Exception:
                    stats["failed"] += 1
                if stats["entries_seen"] % 5000 == 0:
                    print(json.dumps({**stats, "unique_words": len(output)}), flush=True)
    finally:
        stream.close()
    destination = root / "isign_word_candidates.json.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        json.dump(output, handle, separators=(",", ":"))
    (root / "isign_word_candidates_provenance.json").write_text(
        json.dumps({"license": "cc-by-nc-sa-4.0; research use; commercial use requires permission", "items": provenance, "stats": stats}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({**stats, "unique_words": len(output), "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
