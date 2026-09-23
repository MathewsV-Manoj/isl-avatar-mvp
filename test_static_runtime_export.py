"""Verify the static no-server release assets preserve the runtime contract."""

from __future__ import annotations

import json
from pathlib import Path

from production_server import INDEX


ROOT = Path(__file__).resolve().parent


def main() -> None:
    runtime = json.loads((ROOT / "runtime_index.json").read_text(encoding="utf-8"))
    assert runtime["sign_count"] == len(INDEX)
    assert runtime["index"] == INDEX
    assert set(runtime["reviewed_phrases"]) == {"hi-IN", "ml-IN", "ta-IN"}
    quickstart = json.loads((ROOT / "runtime_quickstart.json").read_text(encoding="utf-8"))
    assert quickstart["clip_count"] >= 68
    assert all(item["gloss"] and len(item["clip"]) == 30 for item in quickstart["clips"])
    assert len(quickstart["index"]) == quickstart["clip_count"]
    assert set(quickstart["reviewed_phrases"]) == {"hi-IN", "ml-IN", "ta-IN"}
    bootstrap = json.loads((ROOT / "runtime_bootstrap.json").read_text(encoding="utf-8"))
    assert bootstrap["clip_count"] >= 20
    assert len(bootstrap["index"]) == len(INDEX)
    assert any(item["gloss"] == "how are you" for item in quickstart["clips"])
    cached_glosses = {item["gloss"] for item in quickstart["clips"]}
    assert {"please", "mute", "your", "microphone"} <= cached_glosses

    frontend = (ROOT / "deployment" / "static_frontend" / "isl-avatar-prototype.html").read_text(encoding="utf-8")
    assert 'isl-runtime-base" content="https://huggingface.co/datasets/MathewSVM/isl-avatar-runtime/resolve/main"' in frontend
    assert '<meta name="isl-api-base" content="">' in frontend
    assert "ODYSSEY | Real-Time ISL Avatar" in frontend
    assert "ODYSSEY | ISL MVP" in frontend
    assert 'isl-runtime-index-url" content="runtime_index.json"' in frontend
    assert 'isl-bootstrap-url" content="runtime_bootstrap.json"' in frontend
    assert (ROOT / "deployment" / "static_frontend" / "runtime_index.json").read_bytes() == (ROOT / "runtime_index.json").read_bytes()
    assert (ROOT / "deployment" / "static_frontend" / "runtime_bootstrap.json").read_bytes() == (ROOT / "runtime_bootstrap.json").read_bytes()
    assert (ROOT / "deployment" / "static_frontend" / "service-worker.js").exists()
    assert "this.palm.position.set(0,7,0);" in frontend
    assert "this.palm.quaternion.identity();" in frontend
    assert "a stable open hand is safer" in frontend
    assert "const runtime=await loadBootstrapRuntime();" in frontend
    assert "serviceWorker" in frontend
    assert "warmStaticRuntime();" in frontend
    assert (ROOT / "deployment" / "static_frontend" / "index.html").read_text(encoding="utf-8") == frontend
    print(f"static runtime export passed: {runtime['sign_count']} signs")


if __name__ == "__main__":
    main()
