"""Prevent the browser product path from silently using unvalidated fallbacks."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    html = Path(__file__).with_name("isl-avatar-prototype.html").read_text(encoding="utf-8")
    start = html.index("async function playText(")
    end = html.index("function setPlaybackRate", start)
    play_text = html[start:end]
    assert "const productLookup=await resolveProductDictionary(text,language);" in play_text
    assert "This build needs its ISL API" in play_text
    assert "const words=productLookup.words;" in play_text
    assert "No complete validated ISL sequence for this utterance." in play_text
    assert "timeline=[]; playing=false;" in play_text
    assert 'const NATIVE_POSE_MODE=false;' in html
    assert 'get("nativePose")' not in html
    for unsafe_fallback in (
        "sentenceModelWords(",
        "ensureCislrSigns(",
        "ensureBridgeconnSigns(",
        "ensureOfficialSigns(",
        "ensureIncludeSigns(",
        "ensureIslrtcSigns(",
        "ensureIslrtcV3Signs(",
        "ensureIsl500Signs(",
        "ensureIsignSigns(",
        "ensureKaggleSocialSigns(",
    ):
        assert unsafe_fallback not in play_text, unsafe_fallback
    assert "const baseDatasetPromise=loadDatasetSigns();" not in html
    assert "const isl500ManifestPromise=loadIsl500Manifest();" not in html
    print("product-mode guard passed: browser playback requires validated API clips")


if __name__ == "__main__":
    main()
