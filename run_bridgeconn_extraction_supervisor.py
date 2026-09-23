"""Restart the BridgeConn extractor after an isolated decoder timeout."""

from __future__ import annotations

import subprocess
import sys
import time


MAX_RESTARTS = 100


def main() -> None:
    for attempt in range(MAX_RESTARTS):
        result = subprocess.run([sys.executable, "extract_bridgeconn_landmarks.py"])
        if result.returncode == 0:
            print("bridgeconn extraction completed", flush=True)
            return
        print(f"extractor exited with {result.returncode}; restarting from checkpoint ({attempt + 1}/{MAX_RESTARTS})", flush=True)
        time.sleep(1)
    raise SystemExit("BridgeConn extraction restart limit reached")


if __name__ == "__main__":
    main()
