"""Launch the post-extraction quality pipeline in the background."""

from __future__ import annotations

import subprocess
from pathlib import Path


root = Path(__file__).resolve().parent
python = root / "isl_env" / "Scripts" / "python.exe"
log = (root / "official_isl_pipeline.log").open("a", encoding="utf-8")
error = (root / "official_isl_pipeline.error.log").open("a", encoding="utf-8")
flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008) | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
subprocess.Popen([str(python), str(root / "run_official_pipeline.py")], cwd=root, stdout=log, stderr=error, creationflags=flags)
print("official pipeline launched")
