"""Resume official-gap extraction after a decoder timeout."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


root = Path(__file__).resolve().parent
python = root / "isl_env" / "Scripts" / "python.exe"
for attempt in range(1, 101):
    result = subprocess.run([str(python), str(root / "extract_official_gap.py")], cwd=root)
    if result.returncode == 0:
        break
    print(f"extractor restart {attempt} after exit {result.returncode}", flush=True)
    time.sleep(1)
