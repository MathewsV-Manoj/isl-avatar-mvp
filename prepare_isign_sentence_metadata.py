"""Normalize public iSign sentence metadata for sentence-stage evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq


INPUT = Path("datasets/isign_sentence_metadata/data/train-00000-of-00001.parquet")
OUTPUT = Path("datasets/isign_sentence_metadata/normalized.jsonl")
REPORT = Path("isign_sentence_metadata_report.json")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def main() -> None:
    table = pq.read_table(INPUT, columns=["video_id", "text"])
    vocabulary = set()
    for path in [Path("signal_dictionary_cislr/manifest.json"), Path("signal_dictionary_bridgeconn/manifest.json"), Path("signal_dictionary_official/manifest.json")]:
        vocabulary.update(json.loads(path.read_text(encoding="utf-8"))["signs"])
    vocabulary.update(json.loads(Path("signal_dictionary_supplemental_filtered.json").read_text(encoding="utf-8")))
    vocabulary_words = {key.lower().replace("_", " ") for key in vocabulary}

    rows = []
    covered_tokens = 0
    total_tokens = 0
    unique_videos = set()
    with OUTPUT.open("w", encoding="utf-8") as stream:
        for video_id, raw_text in zip(table.column("video_id").to_pylist(), table.column("text").to_pylist()):
            text = normalize(raw_text or "")
            word_tokens = tokens(text)
            if not text or not word_tokens:
                continue
            unique_videos.add(video_id.split("-", 1)[0])
            total_tokens += len(word_tokens)
            covered_tokens += sum(token in vocabulary_words for token in word_tokens)
            row = {"video_id": video_id, "text": text, "tokens": word_tokens}
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows.append(row)
    REPORT.write_text(json.dumps({
        "source": "Navneeth017/neo_isign_metadata_ref",
        "rows": len(rows),
        "unique_videos": len(unique_videos),
        "tokens": total_tokens,
        "dictionary_token_coverage": round(covered_tokens / total_tokens, 4) if total_tokens else 0,
        "note": "Text normalization only; this is not an ISL grammar translation or pose dataset.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "unique_videos": len(unique_videos), "tokens": total_tokens, "dictionary_token_coverage": round(covered_tokens / total_tokens, 4)}))


if __name__ == "__main__":
    main()
