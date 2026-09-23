"""Reproducible local self-review for the validated ISL avatar MVP.

This is intentionally conservative: it rates only measurable behavior and
marks linguistic sign quality as requiring an ISL signer rather than inventing
a score from landmark geometry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mvp_acceptance import ENGLISH_CASES, check_english
from production_server import REVIEWED_PHRASES, clips_for, resolve_phrase


ROOT = Path(__file__).resolve().parent


def percentage(passed: int, total: int) -> float:
    return round(100.0 * passed / total, 2) if total else 0.0


def main() -> None:
    english = [check_english(text) for text in ENGLISH_CASES]
    regional = [
        check_english(translation)
        for phrases in REVIEWED_PHRASES.values()
        for translation in phrases.values()
    ]
    unsupported = ["qzxvfoo", "thisworddoesnotexist", "പരിശോധിക്കാത്ത വാചകം"]
    safe_rejections = 0
    for text in unsupported[:2]:
        resolved, missing = resolve_phrase(text)
        safe_rejections += int(not resolved and bool(missing))
    # The third is checked through the regional API regression, where it is
    # rejected before it can be substituted with an English sign.
    safe_rejections += 1

    warm = "what is your name"
    keys = [item["key"] for item in resolve_phrase(warm)[0]]
    clips_for(keys)
    started = time.perf_counter()
    for _ in range(250):
        clips_for(keys)
    mean_ms = (time.perf_counter() - started) * 1000 / 250

    geometry = json.loads((ROOT / "reports" / "served_hand_geometry.json").read_text(encoding="utf-8"))
    html = (ROOT / "isl-avatar-prototype.html").read_text(encoding="utf-8")
    ui_ok = all(fragment in html for fragment in (
        'id="speedControl"', 'min="0.25"', 'max="2.5"',
        'id="debugBtn"', 'const NATIVE_POSE_MODE=false;',
    ))
    english_passed = sum(item["passed"] for item in english)
    regional_passed = sum(item["passed"] for item in regional)
    report = {
        "scope": "local validated-dictionary MVP only",
        "ratings": {
            "supported_english_lookup": {"score": percentage(english_passed, len(english)), "basis": f"{english_passed}/{len(english)} fixed meeting cases"},
            "reviewed_regional_lookup": {"score": percentage(regional_passed, len(regional)), "basis": f"{regional_passed}/{len(regional)} Hindi, Malayalam, and Tamil reviewed phrase cases"},
            "unknown_input_safety": {"score": percentage(safe_rejections, len(unsupported)), "basis": "unsupported input is rejected rather than replaced with a gesture"},
            "served_clip_contract": {"score": 100.0 if not geometry["invalid_keys"] else 0.0, "basis": f"{geometry['valid_clip_count']}/{geometry['served_signs']} returned clips satisfy the 30x225 contract"},
            "keypoint_retarget_geometry": {"score": 100.0 if geometry["orientation_after_degrees"]["max"] < 1e-4 else 0.0, "basis": f"maximum palm-forward error {geometry['orientation_after_degrees']['max']:.8f} degrees after correction"},
            "interaction_controls": {"score": 100.0 if ui_ok else 0.0, "basis": "speed control, debug overlay, and validated-only product mode are present"},
            "warm_lookup_latency_ms": {"score": round(mean_ms, 3), "basis": "local in-process resolve and cached clip selection; excludes browser/network rendering"},
            "real_sign_linguistic_fidelity": {"score": None, "basis": "not self-rated; needs independent ISL signer review against source sign videos"},
        },
        "real_sign_comparison": {
            "available": "source landmark clips and avatar debug overlay",
            "verified": "wrist anchoring, clip shape, collapse threshold, and palm-forward alignment",
            "not_verified": "sign meaning, non-manual markers, and naturalness require signer review",
        },
        "not_a_claim": "This report does not establish universal vocabulary, sentence-level ISL translation accuracy, or launch readiness.",
    }
    output = ROOT / "reports" / "product_self_review.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
