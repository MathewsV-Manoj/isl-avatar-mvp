"""Download only the bounded, metadata-selected ISLRTC sample batch."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
BASE = "https://huggingface.co/datasets/Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded/resolve/main/"


def main() -> None:
    batch = json.loads((ROOT / "datasets/islrtc_batch_100.json").read_text(encoding="utf-8"))
    out_root = ROOT / "datasets/islrtc_batch_videos"
    downloaded = []
    for item in batch["items"]:
        target = out_root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == item["bytes"]:
            downloaded.append(item | {"local_path": str(target.relative_to(ROOT))})
            continue
        url = BASE + item["path"].replace(" ", "%20")
        with urlopen(url, timeout=120) as response, target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if target.stat().st_size != item["bytes"]:
            raise RuntimeError(f"size mismatch for {item['path']}: {target.stat().st_size} != {item['bytes']}")
        downloaded.append(item | {"local_path": str(target.relative_to(ROOT))})
        print(f"downloaded {len(downloaded)}/{len(batch['items'])} {item['label']}", flush=True)
    (ROOT / "datasets/islrtc_batch_download_manifest.json").write_text(json.dumps(downloaded, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": len(downloaded), "bytes": sum(item["bytes"] for item in downloaded)}, indent=2))


if __name__ == "__main__":
    main()
