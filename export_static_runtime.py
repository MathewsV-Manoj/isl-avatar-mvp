"""Export the validated product index for a static, lazy-loaded deployment.

Pose shards remain separate files. The browser loads this compact index first
and requests only the shards needed for a user utterance.
"""

from __future__ import annotations

import json
from pathlib import Path

from production_server import INDEX
from regional_translation import REVIEWED_PHRASES


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runtime_index.json"


def main() -> None:
    payload = {
        "mode": "validated_dictionary_static_runtime",
        "sign_count": len(INDEX),
        "index": INDEX,
        "reviewed_phrases": REVIEWED_PHRASES,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"path": str(OUT), "sign_count": len(INDEX), "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    main()
