"""Fetch the remaining BridgeConn shards and create format/label audit reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path("datasets/bridgeconn-sign-dictionary-isl")
SHARDS = [f"shard_{index:05d}-train.tar" for index in range(2, 8)]


def main() -> None:
    snapshot_download(
        repo_id="bridgeconn/sign-dictionary-isl",
        repo_type="dataset",
        local_dir=ROOT,
        allow_patterns=SHARDS,
    )
    for shard in [ROOT / "shard_00001-train.tar", *(ROOT / name for name in SHARDS)]:
        subprocess.run([sys.executable, "inspect_bridgeconn_shard.py", str(shard)], check=True)


if __name__ == "__main__":
    main()
