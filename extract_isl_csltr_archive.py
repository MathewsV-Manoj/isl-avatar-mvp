"""Extract the validated ISL-CSLTR archive without PowerShell archive limitations."""

import json
import zipfile
from pathlib import Path


def main() -> None:
    archive = Path("E:/ISL_Project_Datasets/isl_csltr/isl-csltr-indian-sign-language-dataset.zip")
    output = Path("E:/ISL_Project_Datasets/isl_csltr/unpacked")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        if bad:
            raise RuntimeError(f"CRC failure: {bad}")
        handle.extractall(output)
        entries = len(handle.infolist())
    frames = sum(1 for path in output.rglob("*.jpg"))
    sentences = sum(1 for path in output.rglob("Frames_Sentence_Level/*/*"))
    status = {"state": "complete", "archive_entries": entries, "jpg_frames": frames, "sentence_paths": sentences}
    (output / "extraction_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
