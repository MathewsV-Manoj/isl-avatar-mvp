"""Resumably download authorized iSign parts without staging the full corpus."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True)
    parser.add_argument("--output", default=r"E:\ISL_Project_Datasets\isign")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        "Exploration-Lab/iSign",
        args.part,
        repo_type="dataset",
        local_dir=str(output),
        force_download=False,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
