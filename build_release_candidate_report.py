"""Create a concise, evidence-only local release-candidate report."""

from __future__ import annotations

import json
from pathlib import Path

from production_server import BLOCKED_SOURCE_KEYS, INDEX


ROOT = Path(__file__).resolve().parent


def read_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> None:
    acceptance = read_json("reports/mvp_acceptance.json")
    lexical = read_json("reports/runtime_lexical_coverage.json")
    sentence = read_json("reports/runtime_sentence_coverage.json")
    pose = read_json("models/isign_text2pose_48h_evaluation.json")
    report = {
        "status": "local_release_candidate_not_publicly_deployed",
        "served_dictionary": {
            "runtime_resolvable_labels": len(INDEX),
            "blocked_labels_by_source": {source: len(keys) for source, keys in BLOCKED_SOURCE_KEYS.items() if keys},
            "excluded_source": "isl500 due to widespread hand collapse",
        },
        "acceptance": {
            "english": {"passed": acceptance["english"]["passed"], "total": acceptance["english"]["total"]},
            "reviewed_regional": {"passed": acceptance["reviewed_regional"]["passed"], "total": acceptance["reviewed_regional"]["total"]},
        },
        "lookup_metrics": {
            "content_token_resolution": lexical["content_token_resolution"],
            "complete_lookup_rate": sentence["complete_lookup_rate"],
            "corpus_rows": sentence["sentences"],
        },
        "text_to_pose_candidate": {
            "relative_improvement_over_mean": pose["relative_improvement_over_mean"],
            "decision": "not promoted; experimental only",
        },
        "remaining_blockers": [
            "No independently evaluated broad English-to-ISL sentence translator.",
            "Malayalam, Tamil, and Hindi are limited to reviewed phrases for safe use.",
            "Avatar needs signer-led visual linguistic validation before public accessibility claims.",
            "India-region public deployment requires cloud account credentials and deployment review.",
        ],
    }
    target = ROOT / "reports" / "release_candidate_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
