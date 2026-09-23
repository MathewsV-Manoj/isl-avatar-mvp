"""Inventory local ISL datasets before training or promotion."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"


def file_info(path: Path) -> dict:
    item = {"path": str(path.relative_to(ROOT)), "exists": path.exists()}
    if path.exists():
        item["bytes"] = path.stat().st_size
    return item


def count_csv(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return max(0, sum(1 for _ in csv.DictReader(handle)))


def main() -> None:
    inventory = {
        "local": [],
        "pose_arrays": [],
        "blockers": [],
    }
    expected = [
        DATASETS / "isltranslate" / "ISLTranslate.csv",
        DATASETS / "isltranslate" / "ISL-signer_validation.csv",
        DATASETS / "isign_sentence_metadata" / "normalized.jsonl",
        DATASETS / "posestitch_isl" / "BPCC-ISL.csv",
        DATASETS / "posestitch_isl" / "BLIMP-ISL.csv",
        DATASETS / "cislr" / "prototype.csv",
    ]
    for path in expected:
        item = file_info(path)
        if item["exists"] and path.suffix == ".csv":
            if item["bytes"] <= 100_000_000:
                item["rows"] = count_csv(path)
            else:
                item["rows"] = "deferred_large_file_scan"
        inventory["local"].append(item)
    include_manifest = ROOT / "signal_dictionary_include" / "manifest.json"
    if include_manifest.exists():
        manifest = json.loads(include_manifest.read_text(encoding="utf-8"))
        inventory["include_avatar_dictionary"] = {
            "path": str(include_manifest.relative_to(ROOT)),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "shards": len(manifest.get("shards", [])),
            "source": manifest.get("source"),
            "feature_shape": manifest.get("feature_shape"),
        }
    islrtc_manifest = ROOT / "signal_dictionary_islrtc_batch" / "manifest.json"
    if islrtc_manifest.exists():
        manifest = json.loads(islrtc_manifest.read_text(encoding="utf-8"))
        inventory["islrtc_validated_batch"] = {
            "path": str(islrtc_manifest.relative_to(ROOT)),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "shards": len(manifest.get("shards", [])),
            "source": manifest.get("source"),
            "feature_shape": manifest.get("feature_shape"),
        }
    islrtc_v3_manifest = ROOT / "signal_dictionary_islrtc_batch_v3" / "manifest.json"
    if islrtc_v3_manifest.exists():
        manifest = json.loads(islrtc_v3_manifest.read_text(encoding="utf-8"))
        inventory["islrtc_validated_batch_2"] = {
            "path": str(islrtc_v3_manifest.relative_to(ROOT)),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "shards": len(manifest.get("shards", [])),
            "source": manifest.get("source"),
            "feature_shape": manifest.get("feature_shape"),
        }
    kaggle_social_manifest = ROOT / "signal_dictionary_kaggle_social" / "manifest.json"
    if kaggle_social_manifest.exists():
        manifest = json.loads(kaggle_social_manifest.read_text(encoding="utf-8"))
        inventory["kaggle_social_validated_batch"] = {
            "path": str(kaggle_social_manifest.relative_to(ROOT)),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "shards": len(manifest.get("shards", [])),
            "source": manifest.get("source"),
            "feature_shape": manifest.get("feature_shape"),
        }
    isolated_report = DATASETS / "isl_isolated_40words_filter_report.json"
    isolated_data = DATASETS / "isl_isolated_40words_filtered.json"
    if isolated_report.exists() and isolated_data.exists():
        report = json.loads(isolated_report.read_text(encoding="utf-8"))
        inventory["isolated_variation_corpus"] = {
            "source": "vidit031/isl-isolated-40words",
            "license": "research; respect upstream licenses",
            "path": str(isolated_data.relative_to(ROOT)),
            "clips": report.get("kept_clips", 0),
            "labels": report.get("kept_labels", 0),
            "rejected_clips": report.get("rejected_clips", 0),
            "feature_shape": [30, 225],
        }
    sentence_manifest = ROOT / "signal_dictionary_islrtc_sentences" / "manifest.json"
    if sentence_manifest.exists():
        manifest = json.loads(sentence_manifest.read_text(encoding="utf-8"))
        inventory["sentence_motion_candidate"] = {
            "path": str(sentence_manifest.relative_to(ROOT)),
            "source": manifest.get("source"),
            "license": manifest.get("license"),
            "sentences": manifest.get("sign_count", 0),
            "feature_shape": manifest.get("feature_shape"),
            "promoted": True,
        }
    isl500_manifest = ROOT / "signal_dictionary_isl500" / "manifest.json"
    if isl500_manifest.exists():
        manifest = json.loads(isl500_manifest.read_text(encoding="utf-8"))
        inventory["isl500_variation_candidate"] = {
            "path": str(isl500_manifest.relative_to(ROOT)),
            "source": manifest.get("source"),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "excluded_overlap": manifest.get("base_overlap_excluded", 0),
            "feature_shape": manifest.get("feature_shape"),
            "promoted": True,
        }
    isign_manifest = ROOT / "signal_dictionary_isign" / "manifest.json"
    if isign_manifest.exists():
        manifest = json.loads(isign_manifest.read_text(encoding="utf-8"))
        validation = Path(r"E:\ISL_Project_Datasets\isign\isign_pose_archive_validation.json")
        validation_data = json.loads(validation.read_text(encoding="utf-8")) if validation.exists() else {}
        inventory["isign_word_dictionary"] = {
            "path": str(isign_manifest.relative_to(ROOT)),
            "source": manifest.get("source"),
            "license": manifest.get("license"),
            "labels": manifest.get("sign_count", 0),
            "excluded_overlap": manifest.get("base_overlap_excluded", 0),
            "feature_shape": manifest.get("feature_shape"),
            "pose_archive_valid": validation_data.get("split_zip_valid", False),
            "pose_archive_entries": validation_data.get("entry_count", 0),
            "promoted": True,
        }
    isign_weak_report = ROOT / "datasets" / "isign" / "isign_weak_index.jsonl"
    if isign_weak_report.exists():
        inventory["isign_sentence_context"] = {
            "source": "Exploration-Lab/iSign",
            "index": str(isign_weak_report.relative_to(ROOT)),
            "rows": sum(1 for _ in isign_weak_report.open(encoding="utf-8")),
            "native_gloss_supervision": False,
            "continuous_pose_supervision": True,
        }
    classifier_report = ROOT / "models" / "isl_isolated_40words_conv_report.json"
    if classifier_report.exists():
        report = json.loads(classifier_report.read_text(encoding="utf-8"))
        inventory["isolated_classifier_candidate"] = {
            "path": str(classifier_report.relative_to(ROOT)),
            "architecture": report.get("architecture"),
            "best_valid_accuracy": report.get("best_valid_accuracy"),
            "train_clips": report.get("train_clips"),
            "valid_clips": report.get("valid_clips"),
            "promoted": False,
        }
    sentence_report = ROOT / "models" / "bpcc_gloss_tagger_report.json"
    if sentence_report.exists():
        report = json.loads(sentence_report.read_text(encoding="utf-8"))
        inventory["sentence_model"] = {
            "path": str(sentence_report.relative_to(ROOT)),
            "rows": report.get("rows"),
            "best_valid_token_accuracy": report.get("best_valid_token_accuracy"),
            "input_vocab": report.get("input_vocab"),
            "output_vocab": report.get("output_vocab"),
            "weak_supervision": report.get("weak_supervision"),
            "continuous_pose_supervision": report.get("continuous_pose_supervision"),
            "promoted": True,
        }
    # ISLTranslate supplies sentence text/context, not landmark supervision.
    # Keep it explicit in the inventory so it can enrich the sentence model
    # without being mistaken for a new word-level avatar dictionary.
    isltranslate_report = DATASETS / "isltranslate" / "weak_index_report.json"
    if isltranslate_report.exists():
        report = json.loads(isltranslate_report.read_text(encoding="utf-8"))
        inventory["sentence_context"] = {
            "source": report.get("source"),
            "license": report.get("source_license"),
            "input_rows": report.get("input_rows", 0),
            "kept_rows": report.get("kept_rows", 0),
            "known_input_tokens": report.get("known_input_tokens", 0),
            "native_gloss_supervision": report.get("native_gloss_supervision", False),
            "continuous_pose_supervision": report.get("continuous_pose_supervision", False),
            "index": str((DATASETS / "isltranslate" / "weak_index.jsonl").relative_to(ROOT)),
            "sparse_index": str((DATASETS / "isltranslate" / "weak_index_sparse.jsonl").relative_to(ROOT)),
        }
        sparse_report = DATASETS / "isltranslate" / "weak_index_sparse_report.json"
        if sparse_report.exists():
            inventory["sentence_context"]["sparse_kept_rows"] = json.loads(
                sparse_report.read_text(encoding="utf-8")
            ).get("kept_rows", 0)
    pose_roots = [DATASETS / "isign_pose_parts", DATASETS / "isign_sentence_metadata", DATASETS / "posestitch_isl"]
    for base in pose_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".npy", ".npz", ".pt", ".pkl"} and any(word in path.name.lower() for word in ("pose", "landmark", "keypoint", "skeleton")):
                inventory["pose_arrays"].append(file_info(path))
    archive = DATASETS / "cislr" / "CISLR_v1.5-a_videos" / "CISLR_v1.5-a_videos.zip"
    if archive.exists():
        try:
            with zipfile.ZipFile(archive) as handle:
                inventory["cislr_archive_entries"] = len(handle.infolist())
                inventory["cislr_archive_test"] = "deferred_full_crc_check"
        except zipfile.BadZipFile as exc:
            inventory["blockers"].append(f"CISLR archive invalid: {exc}")
    if not inventory["pose_arrays"] and not inventory.get("isign_word_dictionary", {}).get("pose_archive_valid"):
        inventory["blockers"].append("No continuous native ISL pose arrays found locally.")
    (ROOT / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory))


if __name__ == "__main__":
    main()
