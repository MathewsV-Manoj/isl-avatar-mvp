"""Local inference bridge for the BPCC gloss tagger."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from train_bpcc_gloss_tagger import GlossTagger
from train_isign_text2pose import TextToPose, make_batch
from regional_translation import LANGUAGE_NAMES, RegionalTranslator

ROOT = Path(__file__).resolve().parent
checkpoint = torch.load(ROOT / "models/bpcc_gloss_tagger.pt", map_location="cpu")
model = GlossTagger(len(checkpoint["in_vocab"]), len(checkpoint["out_vocab"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
inverse = {value: key for key, value in checkpoint["out_vocab"].items()}
vocab = checkpoint["in_vocab"]
pose_model = None
pose_vocab = None
pose_checkpoint_path = ROOT / "models/isign_text2pose_candidate.pt"
if pose_checkpoint_path.exists():
    pose_checkpoint = torch.load(pose_checkpoint_path, map_location="cpu")
    pose_vocab = pose_checkpoint["vocab"]
    pose_config = pose_checkpoint.get("config", {})
    pose_model = TextToPose(len(pose_vocab), hidden=int(pose_config.get("hidden", 192)))
    pose_model.load_state_dict(pose_checkpoint["model"])
    pose_model.eval()
sentence_pose_lookup_path = ROOT / "models/isign_sentence_pose_lookup.json"
sentence_pose_lookup = json.loads(sentence_pose_lookup_path.read_text(encoding="utf-8")) if sentence_pose_lookup_path.exists() else {}
metrics = {"requests": 0, "tokens": 0, "validated": 0, "fallback": 0}
translator = RegionalTranslator()
manifest_keys = set()
for manifest_name in (
    "signal_dictionary_cislr/manifest.json",
    "signal_dictionary_bridgeconn/manifest.json",
    "signal_dictionary_official/manifest.json",
    "signal_dictionary_islrtc_batch/manifest.json",
    "signal_dictionary_islrtc_batch_v3/manifest.json",
    "signal_dictionary_kaggle_social/manifest.json",
):
    path = ROOT / manifest_name
    if path.exists():
        manifest_keys.update(json.loads(path.read_text(encoding="utf-8")).get("signs", {}).keys())


def normalized_key(token: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", token.upper()).strip("_")


def tokenize(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())[: checkpoint.get("max_len", 32)]


def normalized_sentence(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).lower()))


@lru_cache(maxsize=24)
def sentence_pose_file(name: str):
    with np.load(ROOT / "datasets/isign_sentence_pose_v2" / name, allow_pickle=False) as pack:
        return pack["poses"].astype(np.float32)


def exact_sentence_pose(text: str):
    item = sentence_pose_lookup.get(normalized_sentence(text))
    if not item:
        return None
    pose = sentence_pose_file(str(item["file"]))[int(item["index"])]
    if pose.shape != (30, 225) or not np.isfinite(pose).all():
        return None
    points = pose.reshape(30, 75, 3)
    left = np.linalg.norm(points[:, 33:54] - points[:, 33:34], axis=2).max(axis=1)
    right = np.linalg.norm(points[:, 54:75] - points[:, 54:55], axis=2).max(axis=1)
    if np.logical_or(left > 1e-5, right > 1e-5).mean() < 0.5:
        return None
    return pose.tolist()


def predict(text: str):
    metrics["requests"] += 1
    tokens = tokenize(text)
    metrics["tokens"] += len(tokens)
    if not tokens:
        return []
    source = torch.tensor([[vocab.get(token, vocab["<unk>"]) for token in tokens]])
    with torch.no_grad():
        logits = model(source)[0]
        probabilities = torch.softmax(logits, dim=-1)
        confidence, output = probabilities.max(-1)
    result = []
    for token, index, score in zip(tokens, output.tolist(), confidence.tolist()):
        gloss = inverse.get(index, "<drop>")
        direct = normalized_key(token)
        # Never let a weak model invent an unrelated sign for a name or
        # unseen word. Known dictionary words stay deterministic; unknown
        # words are returned unchanged so the browser can fingerspell them.
        if direct in manifest_keys:
            result.append(direct.lower().replace("_", " "))
            metrics["validated"] += 1
        else:
            result.append(token)
            metrics["fallback"] += 1
    return result


def predict_pose(text: str):
    exact = exact_sentence_pose(text)
    if exact is not None:
        return exact, "exact_isign_lookup", False
    if pose_model is None or pose_vocab is None:
        raise RuntimeError("native pose candidate is not available")
    with torch.no_grad():
        source, mask = make_batch([text], pose_vocab, torch.device("cpu"))
        pose = pose_model(source, mask)[0].clamp(-20.0, 20.0)
    return pose.tolist(), "text2pose_candidate", True


class Handler(BaseHTTPRequestHandler):
    def _headers(self, status=200):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._headers(); self.wfile.write(json.dumps({"ok": True, "model": "bpcc_gloss_tagger", "regional_languages": LANGUAGE_NAMES}).encode()); return
        if self.path == "/metrics":
            self._headers(); self.wfile.write(json.dumps({**metrics, "mode": "guarded"}).encode()); return
        self._headers(404); self.wfile.write(b'{"error":"not found"}')

    def do_POST(self):
        if self.path not in ("/predict", "/predict_pose", "/translate", "/predict_regional"):
            self._headers(404); self.wfile.write(b'{"error":"not found"}'); return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            text = str(payload.get("text", ""))
            language = str(payload.get("language", "en-IN"))
            if self.path == "/predict_pose":
                pose, source, candidate = predict_pose(text)
                self._headers(); self.wfile.write(json.dumps({"pose": pose, "shape": [30, 225], "source_text": text, "native_pose_supervision": True, "candidate": candidate, "source": source}).encode()); return
            if self.path in ("/translate", "/predict_regional"):
                translated = translator.translate(text, language)
                result = {
                    "source_text": text,
                    "source_language": language,
                    "source_language_name": LANGUAGE_NAMES.get(language, language),
                    **translated,
                }
                if self.path == "/translate":
                    self._headers(); self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8")); return
                result.update({"words": predict(translated["english"]), "guarded": True})
                self._headers(); self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8")); return
            words = predict(text)
            self._headers(); self.wfile.write(json.dumps({"words": words, "source_text": text, "weak_supervision": True, "guarded": True}).encode())
        except Exception as exc:
            self._headers(500); self.wfile.write(json.dumps({"error": str(exc)}).encode())


if __name__ == "__main__":
    print("sentence inference listening on http://127.0.0.1:8090", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8090), Handler).serve_forever()
