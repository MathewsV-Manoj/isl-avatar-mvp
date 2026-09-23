"""Publish the full static ISL MVP after authenticating with a write token.

This intentionally uploads only the quality-filtered browser runtime, never
the unrelated training folders or experimental model checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
ACCOUNT = "MathewSVM"
RUNTIME_REPO = f"{ACCOUNT}/isl-avatar-runtime"
SPACE_REPO = f"{ACCOUNT}/isl-avatar-mvp"
RUNTIME_PATTERNS = [
    "runtime_index.json",
    "runtime_bootstrap.json",
    "runtime_quickstart.json",
    "signal_dictionary_cislr/**",
    "signal_dictionary_bridgeconn/**",
    "signal_dictionary_official/**",
    "signal_dictionary_include/**",
    "signal_dictionary_islrtc_batch/**",
    "signal_dictionary_islrtc_batch_v3/**",
    "signal_dictionary_isign/**",
    "signal_dictionary_kaggle_social/**",
    "signal_dictionary_islrtc_sentences/**",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--quickstart-only", action="store_true")
    args = parser.parse_args()
    if sum((args.runtime_only, args.frontend_only, args.quickstart_only)) > 1:
        parser.error("choose only one deployment target")
    api = HfApi()

    if args.quickstart_only:
        api.upload_file(
            path_or_fileobj=ROOT / "runtime_quickstart.json",
            path_in_repo="runtime_quickstart.json",
            repo_id=RUNTIME_REPO,
            repo_type="dataset",
            commit_message="Cache common meeting signs for fast browser playback",
        )
    elif not args.frontend_only:
        api.create_repo(RUNTIME_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=ROOT / "deployment" / "runtime_dataset_README.md",
            path_in_repo="README.md",
            repo_id=RUNTIME_REPO,
            repo_type="dataset",
            commit_message="Add validated runtime dataset card",
        )
        api.upload_large_folder(
            repo_id=RUNTIME_REPO,
            repo_type="dataset",
            folder_path=ROOT,
            allow_patterns=RUNTIME_PATTERNS,
            num_workers=4,
            print_report=True,
        )

    if not args.runtime_only and not args.quickstart_only:
        api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=SPACE_REPO,
            repo_type="space",
            folder_path=ROOT / "deployment" / "static_frontend",
            commit_message="Deploy static ISL avatar frontend",
        )

    if not args.frontend_only:
        print(f"Runtime: https://huggingface.co/datasets/{RUNTIME_REPO}")
    if not args.runtime_only:
        print(f"App: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
