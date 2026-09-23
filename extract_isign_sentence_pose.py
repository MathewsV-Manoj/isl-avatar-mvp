"""Stream all iSign pose records into compact normalized sentence-pose shards."""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from pose_format import Pose

from import_isl500_landmarks import normalize_clip
from inspect_isign_split_zip import SplitFile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"E:\ISL_Project_Datasets\isign")
    parser.add_argument("--output", default="datasets/isign_sentence_pose")
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    texts = {}
    with (root / "iSign_v1.1.csv").open(encoding="utf-8-sig", newline="") as handle:
        texts = {row["uid"]: row["text"].strip() for row in csv.DictReader(handle)}
    paths = [root / f"iSign-poses_v1.1_part_{part}" for part in ("aa", "ab", "ac", "ad")]
    poses, uids, labels = [], [], []
    manifest = {"source": "Exploration-Lab/iSign", "feature_shape": [30, 225], "shards": [], "seen": 0, "kept": 0, "failed": 0}

    def flush() -> None:
        if not poses:
            return
        number = len(manifest["shards"])
        path = output / f"shard-{number:05d}.npz"
        np.savez_compressed(path, poses=np.asarray(poses, dtype=np.float32), uids=np.asarray(uids), texts=np.asarray(labels))
        manifest["shards"].append({"file": path.name, "count": len(poses)})
        poses.clear(); uids.clear(); labels.clear()
        print(json.dumps({"seen": manifest["seen"], "kept": manifest["kept"], "failed": manifest["failed"], "shards": len(manifest["shards"]) }), flush=True)

    stream = SplitFile(paths)
    try:
        with zipfile.ZipFile(stream) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".pose"):
                    continue
                manifest["seen"] += 1
                uid = Path(info.filename).stem
                text = texts.get(uid)
                if not text:
                    manifest["failed"] += 1
                    continue
                try:
                    pose = Pose.read(io.BytesIO(archive.read(info)))
                    selected = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
                    values = np.ma.filled(selected.body.data, 0.0).astype(np.float32)[:, 0, :, :3]
                    if values.shape[1] != 75:
                        manifest["failed"] += 1
                        continue
                    clip = normalize_clip(values[:, :33], np.stack((values[:, 33:54], values[:, 54:75]), axis=1))
                    if clip is None or not np.isfinite(clip).all() or float(np.max(np.abs(clip))) > 20.0:
                        manifest["failed"] += 1
                        continue
                    poses.append(clip); uids.append(uid); labels.append(text); manifest["kept"] += 1
                    if len(poses) >= args.shard_size:
                        flush()
                except Exception:
                    manifest["failed"] += 1
    finally:
        stream.close()
    flush()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
