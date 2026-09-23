"""Launch the resumable official-gap downloader without holding the agent terminal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
log = (root / "official_isl_download.log").open("a", encoding="utf-8")
error = (root / "official_isl_download.error.log").open("a", encoding="utf-8")
flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
subprocess.Popen([sys.executable, str(root / "download_official_gap.py")], cwd=root, stdout=log, stderr=error, creationflags=flags)
print("official gap downloader launched")
