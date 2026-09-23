"""Verify the non-secret frontend/backend deployment handoff files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    html = (ROOT / "isl-avatar-prototype.html").read_text(encoding="utf-8")
    assert 'name="isl-api-base" content=""' in html
    assert 'const API_BASE=' in html and 'fetch(apiUrl("/api/resolve")' in html

    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["rewrites"] == [{"source": "/", "destination": "/isl-avatar-prototype.html"}]

    environment = (ROOT / "deployment" / "api.env.example").read_text(encoding="utf-8")
    assert "ISL_HOST=0.0.0.0" in environment
    assert "ISL_ALLOWED_ORIGIN=https://YOUR-VERCEL-PROJECT.vercel.app" in environment

    guide = (ROOT / "deployment" / "README.md").read_text(encoding="utf-8")
    assert "Vercel" in guide and "persistent backend" in guide

    checklist = (ROOT / "deployment" / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    assert "ISL_ALLOWED_ORIGIN" in checklist
    assert "run_product_preflight.ps1" in checklist
    assert "independent ISL signer" in checklist

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "deployment/RELEASE_CHECKLIST.md" in root_readme
    assert "run_release_metrics.ps1" in root_readme
    assert (ROOT / "run_release_metrics.ps1").exists()
    print("deployment configuration regression passed")


if __name__ == "__main__":
    main()
