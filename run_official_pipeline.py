"""Wait for extraction, then run the official vocabulary quality pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "official_isl_candidate_report.json"
EXPECTED = 3353


def run(script: str) -> None:
    result = subprocess.run([str(ROOT / "isl_env" / "Scripts" / "python.exe"), str(ROOT / script)], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


while True:
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        if len(report.get("completed", [])) >= EXPECTED:
            break
    time.sleep(30)

run("filter_official_candidates.py")
run("build_official_shards.py")
print("official vocabulary pipeline complete", flush=True)
