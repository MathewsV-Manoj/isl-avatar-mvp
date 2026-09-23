"""Parallel continuation of iSign sentence-pose extraction."""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pose_format import Pose

from import_isl500_landmarks import normalize_clip
from inspect_isign_split_zip import SplitFile


def load_texts(root: Path) -> dict[str, str]:
    with (root / "iSign_v1.1.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["uid"]: row["text"].strip() for row in csv.DictReader(handle)}


def worker(args):
    worker_id, names, root_text, output_text, texts = args
    root = Path(root_text); output = Path(output_text); output.mkdir(parents=True, exist_ok=True)
    paths = [root / f"iSign-poses_v1.1_part_{part}" for part in ("aa", "ab", "ac", "ad")]
    poses=[]; uids=[]; labels=[]; kept=failed=0; shard=0
    def flush():
        nonlocal shard
        if not poses: return
        np.savez_compressed(output / f"parallel-{worker_id:02d}-{shard:04d}.npz", poses=np.asarray(poses,dtype=np.float32), uids=np.asarray(uids), texts=np.asarray(labels))
        shard += 1; poses.clear(); uids.clear(); labels.clear()
    stream=SplitFile(paths)
    try:
        with zipfile.ZipFile(stream) as archive:
            for name in names:
                uid=Path(name).stem; text=texts.get(uid)
                if not text: failed+=1; continue
                try:
                    pose=Pose.read(io.BytesIO(archive.read(archive.getinfo(name))))
                    selected=pose.get_components(["POSE_LANDMARKS","LEFT_HAND_LANDMARKS","RIGHT_HAND_LANDMARKS"])
                    values=np.ma.filled(selected.body.data,0.0).astype(np.float32)[:,0,:,:3]
                    if values.shape[1]!=75: failed+=1; continue
                    clip=normalize_clip(values[:,:33],np.stack((values[:,33:54],values[:,54:75]),axis=1))
                    if clip is None or not np.isfinite(clip).all() or float(np.max(np.abs(clip)))>20.0: failed+=1; continue
                    poses.append(clip); uids.append(uid); labels.append(text); kept+=1
                    if len(poses)>=1000: flush()
                except Exception: failed+=1
    finally: stream.close()
    flush()
    return {"worker":worker_id,"kept":kept,"failed":failed,"shards":shard}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=r"E:\ISL_Project_Datasets\isign"); ap.add_argument("--output",default="datasets/isign_sentence_pose_v2"); ap.add_argument("--skip",default="datasets/isign_sentence_pose_v2"); ap.add_argument("--workers",type=int,default=4); args=ap.parse_args()
    root=Path(args.root); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    texts=load_texts(root); skip=set()
    for path in Path(args.skip).glob("*.npz"):
        try: skip.update(np.load(path,allow_pickle=False)["uids"].tolist())
        except Exception: pass
    stream=SplitFile([root/f"iSign-poses_v1.1_part_{part}" for part in ("aa","ab","ac","ad")])
    try:
        with zipfile.ZipFile(stream) as archive:
            names=[i.filename for i in archive.infolist() if i.filename.endswith('.pose') and Path(i.filename).stem not in skip]
    finally: stream.close()
    chunks=[names[i::args.workers] for i in range(args.workers)]
    jobs=[(i,chunks[i],str(root),str(output),texts) for i in range(args.workers)]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result(); results.append(result); print(json.dumps(result),flush=True)
    report={"source":"Exploration-Lab/iSign","skipped_existing":len(skip),"remaining_entries":len(names),"workers":args.workers,"results":results,"feature_shape":[30,225]}
    (output/"parallel_extraction_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__ == "__main__": main()
