"""
Stage 1 + Stage 2 combined: extract ALL words from ISL-CSLTR and smooth them.

- Auto-discovers every word folder (no need to list them by hand).
- Extracts pose+hands keypoints (guaranteed 225-width per frame).
- Normalizes (shoulder-centered) and resamples to 30 frames.
- Applies Savitzky-Golay smoothing to kill MediaPipe jitter.
- Saves BOTH raw and smoothed dictionaries so you can compare.

Run:
    python build_dictionary.py
"""

import os
import re
import json
import numpy as np
import cv2
import mediapipe as mp
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FRAMES_WORD_LEVEL_DIR = r"C:\Users\Mathews\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"

# Empty list = auto-discover ALL word folders. Or set specific words to limit.
WORDS = []

TARGET_FRAMES = 30
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
NUM_POINTS = POSE_LANDMARKS + 2 * HAND_LANDMARKS      # 75
FEATURES_PER_FRAME = NUM_POINTS * 3                   # 225

# Smoothing: window must be odd and <= TARGET_FRAMES; poly < window.
SMOOTH_WINDOW = 7
SMOOTH_POLY = 3

mp_holistic = mp.solutions.holistic
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def natural_sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else 0


def _fill_landmarks(frame_vec, start_idx, landmark_obj, expected_count):
    if landmark_obj is None:
        return
    for i, lm in enumerate(landmark_obj.landmark):
        if i >= expected_count:
            break
        base = start_idx + i * 3
        frame_vec[base] = lm.x
        frame_vec[base + 1] = lm.y
        frame_vec[base + 2] = lm.z


def extract_landmarks_from_image_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTS)]
    image_files.sort(key=natural_sort_key)
    if len(image_files) == 0:
        return None

    frames = []
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
        for fname in image_files:
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)
            frame_vec = np.zeros(FEATURES_PER_FRAME, dtype=float)
            _fill_landmarks(frame_vec, 0, result.pose_landmarks, POSE_LANDMARKS)
            _fill_landmarks(frame_vec, POSE_LANDMARKS * 3, result.left_hand_landmarks, HAND_LANDMARKS)
            _fill_landmarks(frame_vec, (POSE_LANDMARKS + HAND_LANDMARKS) * 3, result.right_hand_landmarks, HAND_LANDMARKS)
            frames.append(frame_vec)

    if len(frames) == 0:
        return None
    arr = np.stack(frames)
    return _normalize(arr)


def _normalize(arr):
    n = arr.shape[0]
    pts = arr.reshape(n, NUM_POINTS, 3)
    ls, rs = pts[:, 11, :], pts[:, 12, :]
    center = (ls + rs) / 2.0
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    pts = (pts - center[:, None, :]) / scale[:, :, None]
    return pts.reshape(n, FEATURES_PER_FRAME)


def resample_to_fixed_length(arr, target_len=TARGET_FRAMES):
    n = arr.shape[0]
    if n == target_len:
        return arr
    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, arr.shape[1]))
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(new_idx, old_idx, arr[:, col])
    return out


def smooth_sequence(arr):
    """Savitzky-Golay filter along the time axis for each of the 225 tracks.
    Reduces MediaPipe jitter while preserving the overall motion shape."""
    win = min(SMOOTH_WINDOW, arr.shape[0] if arr.shape[0] % 2 == 1 else arr.shape[0] - 1)
    if win < 3:
        return arr  # too short to smooth
    poly = min(SMOOTH_POLY, win - 1)
    out = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        out[:, col] = savgol_filter(arr[:, col], win, poly)
    return out


def discover_words():
    if WORDS:
        return WORDS
    return sorted([d for d in os.listdir(FRAMES_WORD_LEVEL_DIR)
                   if os.path.isdir(os.path.join(FRAMES_WORD_LEVEL_DIR, d))])


def build():
    words = discover_words()
    print(f"Found {len(words)} word folders to process.\n")

    raw_dict, smooth_dict = {}, {}
    failed = []

    for idx, word in enumerate(words, 1):
        word_dir = os.path.join(FRAMES_WORD_LEVEL_DIR, word)
        subdirs = [d for d in os.listdir(word_dir) if os.path.isdir(os.path.join(word_dir, d))]
        sample_dirs = [os.path.join(word_dir, d) for d in subdirs] if subdirs else [word_dir]

        got = None
        for sample_dir in sample_dirs:
            landmarks = extract_landmarks_from_image_folder(sample_dir)
            if landmarks is not None:
                got = resample_to_fixed_length(landmarks)
                break

        if got is None:
            failed.append(word)
            print(f"[{idx}/{len(words)}] {word}: FAILED (no usable frames)")
            continue

        raw_dict[word] = got.tolist()
        smooth_dict[word] = smooth_sequence(got).tolist()
        print(f"[{idx}/{len(words)}] {word}: ok")

    with open("signal_dictionary_raw.json", "w") as f:
        json.dump(raw_dict, f)
    with open("signal_dictionary_smoothed.json", "w") as f:
        json.dump(smooth_dict, f)

    print(f"\nDone. {len(smooth_dict)} words saved.")
    print("  signal_dictionary_raw.json      (unsmoothed)")
    print("  signal_dictionary_smoothed.json (use this one for the avatar)")
    if failed:
        print(f"\n{len(failed)} words failed extraction: {', '.join(failed)}")


if __name__ == "__main__":
    build()
