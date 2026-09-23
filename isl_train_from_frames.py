"""
Week 1 pipeline for ISL-CSLTR (reads pre-extracted image frames).
Clean rewrite — every function agrees the feature width is exactly 225.

Run:
    python isl_train_from_frames.py
"""

import os
import re
import json
import numpy as np
import cv2
import mediapipe as mp

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FRAMES_WORD_LEVEL_DIR = r"C:\Users\Mathews\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"

WORDS = ["HELLO,HI", "YOU", "PLEASE", "SORRY", "HELP", "STOP",
         "NAME", "WATER", "FOOD", "THANK", "UNDERSTAND", "WELCOME"]

TARGET_FRAMES = 30
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
NUM_POINTS = POSE_LANDMARKS + 2 * HAND_LANDMARKS      # 75
FEATURES_PER_FRAME = NUM_POINTS * 3                   # 225

mp_holistic = mp.solutions.holistic
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def natural_sort_key(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else 0


def _fill_landmarks(frame_vec, start_idx, landmark_obj, expected_count):
    """Write up to expected_count landmarks into frame_vec at start_idx.
    Guarantees nothing outside the reserved slots is touched."""
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
    """Returns [num_frames, 225] array, or None. Width is ALWAYS 225."""
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

    arr = np.stack(frames)   # [num_frames, 225]
    return _normalize(arr)


def _normalize(arr):
    """Center on shoulder midpoint, scale by shoulder width.
    Input and output are both [num_frames, 225]."""
    n = arr.shape[0]
    pts = arr.reshape(n, NUM_POINTS, 3)     # [n, 75, 3]
    left_shoulder = pts[:, 11, :]           # pose landmark 11
    right_shoulder = pts[:, 12, :]          # pose landmark 12
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    pts = (pts - center[:, None, :]) / scale[:, :, None]
    return pts.reshape(n, FEATURES_PER_FRAME)   # back to [n, 225]


def resample_to_fixed_length(arr, target_len=TARGET_FRAMES):
    """Resample time axis to target_len frames. Width stays 225."""
    num_frames = arr.shape[0]
    if num_frames == target_len:
        return arr
    old_idx = np.linspace(0, 1, num_frames)
    new_idx = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, arr.shape[1]))
    for col in range(arr.shape[1]):
        resampled[:, col] = np.interp(new_idx, old_idx, arr[:, col])
    return resampled


def build_dataset():
    X, y, per_word_samples = [], [], {w: [] for w in WORDS}

    for label_idx, word in enumerate(WORDS):
        word_dir = os.path.join(FRAMES_WORD_LEVEL_DIR, word)
        if not os.path.isdir(word_dir):
            print(f"[MISSING] no folder for '{word}'")
            continue

        subdirs = [d for d in os.listdir(word_dir) if os.path.isdir(os.path.join(word_dir, d))]
        sample_dirs = [os.path.join(word_dir, d) for d in subdirs] if subdirs else [word_dir]

        for sample_dir in sample_dirs:
            landmarks = extract_landmarks_from_image_folder(sample_dir)
            if landmarks is None:
                continue
            fixed = resample_to_fixed_length(landmarks)   # [30, 225]
            X.append(fixed)
            y.append(label_idx)
            per_word_samples[word].append(fixed)

        print(f"[{word}] {len(per_word_samples[word])} sample(s) extracted")

    if len(X) == 0:
        raise RuntimeError("No usable data extracted. Check FRAMES_WORD_LEVEL_DIR and WORDS.")
    return np.stack(X), np.array(y), per_word_samples


def build_signal_dictionary(per_word_samples, out_path="signal_dictionary.json"):
    dictionary = {word: samples[0].tolist() for word, samples in per_word_samples.items() if samples}
    with open(out_path, "w") as f:
        json.dump(dictionary, f)
    print(f"\nSaved signal dictionary for {len(dictionary)} words -> {out_path}")


if __name__ == "__main__":
    print(f"Extracting from: {FRAMES_WORD_LEVEL_DIR}\n")
    X, y, per_word_samples = build_dataset()
    print(f"\nDataset shape: {X.shape}  (should be (12, 30, 225))")
    build_signal_dictionary(per_word_samples)
    print("\nDone. Load signal_dictionary.json in hand-avatar-verifier.html to view.")
