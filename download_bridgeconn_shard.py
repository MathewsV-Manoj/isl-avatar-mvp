"""Resumably fetch and inspect the first BridgeConn ISL dictionary shard."""

import subprocess
import sys

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download(
        repo_id="bridgeconn/sign-dictionary-isl",
        repo_type="dataset",
        local_dir="datasets/bridgeconn-sign-dictionary-isl",
        allow_patterns=["README.md", "shard_00001-train.tar"],
    )
    subprocess.run([sys.executable, "inspect_bridgeconn_shard.py"], check=True)
