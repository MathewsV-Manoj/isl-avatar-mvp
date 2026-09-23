"""Generate the small static frontend package for the public release."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deployment" / "static_frontend"
DEFAULT_RUNTIME_URL = "https://huggingface.co/datasets/MathewSVM/isl-avatar-runtime/resolve/main"


def main() -> None:
    runtime_url = os.environ.get("ISL_RUNTIME_BASE", DEFAULT_RUNTIME_URL).rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    html = (ROOT / "isl-avatar-prototype.html").read_text(encoding="utf-8")
    configured = html.replace(
        '<meta name="isl-runtime-base" content="">',
        f'<meta name="isl-runtime-base" content="{runtime_url}">',
        1,
    )
    if configured == html:
        raise RuntimeError("static runtime meta tag was not found")
    configured = configured.replace(
        '<meta name="isl-runtime-index-url" content="">',
        '<meta name="isl-runtime-index-url" content="runtime_index.json">',
        1,
    )
    if 'isl-runtime-index-url" content="runtime_index.json"' not in configured:
        raise RuntimeError("static runtime index meta tag was not found")
    configured = configured.replace(
        '<meta name="isl-bootstrap-url" content="">',
        '<meta name="isl-bootstrap-url" content="runtime_bootstrap.json">',
        1,
    )
    if 'isl-bootstrap-url" content="runtime_bootstrap.json"' not in configured:
        raise RuntimeError("static bootstrap meta tag was not found")
    configured = configured.replace(
        '<meta name="isl-runtime-base" content="' + runtime_url + '">',
        '<meta name="isl-runtime-base" content="' + runtime_url + '">\n'
        '<link rel="preconnect" href="https://huggingface.co" crossorigin>\n'
        '<link rel="preconnect" href="https://cdn-lfs.huggingface.co" crossorigin>',
        1,
    )
    (OUT / "isl-avatar-prototype.html").write_text(configured, encoding="utf-8")
    # Static Spaces serve index.html at their root. Keep both names so the
    # local prototype URL and the deployed root render the same app.
    (OUT / "index.html").write_text(configured, encoding="utf-8")
    # Keep the lookup index near the static page. It is compact enough to
    # cache locally and removes a cross-origin redirect from first playback.
    (OUT / "runtime_index.json").write_bytes((ROOT / "runtime_index.json").read_bytes())
    (OUT / "runtime_bootstrap.json").write_bytes((ROOT / "runtime_bootstrap.json").read_bytes())
    (OUT / "service-worker.js").write_text(
        'const CACHE="odyssey-shell-v2";\n'
        'const ASSETS=["./","./index.html","./runtime_bootstrap.json"];\n'
        'self.addEventListener("install",event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));\n'
        'self.addEventListener("activate",event=>event.waitUntil(self.clients.claim()));\n'
        'self.addEventListener("fetch",event=>{if(new URL(event.request.url).origin===location.origin)event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request)));});\n',
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "---\n"
        "title: ODYSSEY | ISL MVP\n"
        "emoji: 🤟\n"
        "colorFrom: green\n"
        "colorTo: yellow\n"
        "sdk: static\n"
        "pinned: false\n"
        "---\n\n"
        "# ODYSSEY | ISL MVP\n\n"
        "Static frontend for the validated ISL dictionary runtime. The app loads only requested pose shards from the paired public dataset repository.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
