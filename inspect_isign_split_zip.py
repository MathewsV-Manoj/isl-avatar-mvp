"""Inspect the iSign split ZIP without creating a second 170 GB file."""
from __future__ import annotations

import argparse
import json
import zipfile
from bisect import bisect_right
from pathlib import Path


class SplitFile:
    def __init__(self, paths: list[Path]):
        self.handles = [p.open("rb") for p in paths]
        self.ends = []
        total = 0
        for handle in self.handles:
            total += handle.seek(0, 2)
            self.ends.append(total)
        self.size = total
        self.pos = 0

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self.pos + offset
        elif whence == 2:
            target = self.size + offset
        else:
            raise ValueError(whence)
        self.pos = max(0, min(self.size, target))
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.pos
        remaining = min(count, self.size - self.pos)
        chunks = []
        while remaining:
            index = bisect_right(self.ends, self.pos)
            base = 0 if index == 0 else self.ends[index - 1]
            take = min(remaining, self.ends[index] - self.pos)
            handle = self.handles[index]
            handle.seek(self.pos - base)
            chunks.append(handle.read(take))
            self.pos += take
            remaining -= take
        return b"".join(chunks)

    def close(self) -> None:
        for handle in self.handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"E:\ISL_Project_Datasets\isign")
    args = parser.parse_args()
    root = Path(args.root)
    names = [
        "iSign-poses_v1.1_part_aa",
        "iSign-poses_v1.1_part_ab",
        "iSign-poses_v1.1_part_ac",
        "iSign-poses_v1.1_part_ad",
    ]
    paths = [root / name for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing split parts: " + ", ".join(missing))
    stream = SplitFile(paths)
    try:
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            sample = infos[:20]
            report = {
                "parts": [{"name": p.name, "bytes": p.stat().st_size} for p in paths],
                "split_zip_valid": True,
                "entry_count": len(infos),
                "sample_entries": [info.filename for info in sample],
            }
            (root / "isign_pose_archive_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report), flush=True)
    finally:
        stream.close()


if __name__ == "__main__":
    main()
