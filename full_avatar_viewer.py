"""
Full-body + Face + Hands Avatar Viewer
========================================
Upgrades the week-1 pipeline:
  - Extracts POSE (33) + FACE (468) + LEFT HAND (21) + RIGHT HAND (21) landmarks
  - Opens a live matplotlib 3D window animating the full skeleton, incl. facial
    expression, when you run this script on a video
  - This is a skeleton/wireframe avatar (dots + connecting lines), NOT a
    textured 3D character — that rigging step belongs to your avatar teammates.
    What this proves is that your MODEL's output (the motion signal) is correct.

Recommended dataset: iSign (Hugging Face) — it already ships pose files in
pose-format and defines a Text2Pose task, which is literally your model's job.
  pip install pose-format datasets --break-system-packages

Install (this script):
  pip install mediapipe opencv-python matplotlib numpy --break-system-packages
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)

mp_holistic = mp.solutions.holistic

POSE_COUNT = 33
FACE_COUNT = 468
HAND_COUNT = 21

# Connections for drawing lines between joints
POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
HAND_CONNECTIONS = list(mp_holistic.HAND_CONNECTIONS)
# Full face tessellation is very dense (>1000 edges) and unreadable at this
# scale — use a lightweight face outline instead (eyes, brows, lips, jaw).
FACE_OUTLINE_IDX = list(range(0, 17)) + list(range(17, 27)) + \
                    list(range(36, 48)) + list(range(48, 68))
FACE_OUTLINE_IDX = [i for i in FACE_OUTLINE_IDX if i < FACE_COUNT]


# ---------------------------------------------------------------------
# STEP 1 — Extract full-body + face + hands landmarks from a video
# ---------------------------------------------------------------------
def extract_full_sequence(video_path):
    """Returns dict of per-frame arrays: pose[N,33,3], face[N,468,3],
    left_hand[N,21,3], right_hand[N,21,3] — all normalized to shoulder frame."""
    cap = cv2.VideoCapture(video_path)
    pose_frames, face_frames, lh_frames, rh_frames = [], [], [], []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                               refine_face_landmarks=True) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            pose_frames.append(_to_array(result.pose_landmarks, POSE_COUNT))
            face_frames.append(_to_array(result.face_landmarks, FACE_COUNT))
            lh_frames.append(_to_array(result.left_hand_landmarks, HAND_COUNT))
            rh_frames.append(_to_array(result.right_hand_landmarks, HAND_COUNT))
    cap.release()

    pose = np.stack(pose_frames)
    face = np.stack(face_frames)
    lh = np.stack(lh_frames)
    rh = np.stack(rh_frames)

    return _normalize_all(pose, face, lh, rh)


def _to_array(landmark_obj, expected_count):
    if landmark_obj is None:
        return np.zeros((expected_count, 3))
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmark_obj.landmark])
    if pts.shape[0] != expected_count:  # safety pad/crop
        fixed = np.zeros((expected_count, 3))
        fixed[:min(expected_count, pts.shape[0])] = pts[:expected_count]
        return fixed
    return pts


def _normalize_all(pose, face, lh, rh):
    """Center everything on shoulder midpoint, scale by shoulder width —
    applied identically to pose/face/hands so proportions stay correct."""
    left_shoulder = pose[:, 11, :]
    right_shoulder = pose[:, 12, :]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def norm(arr):
        return (arr - center[:, None, :]) / scale[:, None, None]

    return {
        "pose": norm(pose),
        "face": norm(face),
        "left_hand": norm(lh),
        "right_hand": norm(rh),
    }


# ---------------------------------------------------------------------
# STEP 2 — Pop-up window avatar viewer (matplotlib 3D animation)
# ---------------------------------------------------------------------
def play_avatar_window(sequence, fps=15):
    """sequence: dict with 'pose','face','left_hand','right_hand', each [N, K, 3].
    Opens a live window and animates through all N frames, looping."""
    num_frames = sequence["pose"].shape[0]

    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    def set_axes():
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_axis_off()
        ax.view_init(elev=100, azim=-90)  # front-on view

    def draw_frame(i):
        ax.cla()
        set_axes()

        pose = sequence["pose"][i]
        face = sequence["face"][i]
        lh = sequence["left_hand"][i]
        rh = sequence["right_hand"][i]

        # Body skeleton
        ax.scatter(pose[:, 0], -pose[:, 1], pose[:, 2], c="#f2a541", s=12)
        for a, b in POSE_CONNECTIONS:
            ax.plot([pose[a, 0], pose[b, 0]],
                    [-pose[a, 1], -pose[b, 1]],
                    [pose[a, 2], pose[b, 2]], c="#4fb286", linewidth=1.2)

        # Hands
        for hand in (lh, rh):
            ax.scatter(hand[:, 0], -hand[:, 1], hand[:, 2], c="#f2a541", s=8)
            for a, b in HAND_CONNECTIONS:
                ax.plot([hand[a, 0], hand[b, 0]],
                        [-hand[a, 1], -hand[b, 1]],
                        [hand[a, 2], hand[b, 2]], c="#4fb286", linewidth=0.8)

        # Face (lightweight outline points only — full mesh is unreadable at this scale)
        outline = face[FACE_OUTLINE_IDX]
        ax.scatter(outline[:, 0], -outline[:, 1], outline[:, 2], c="#eaf2f0", s=4)

        ax.set_title(f"frame {i+1}/{num_frames}", color="white")

    anim = FuncAnimation(fig, draw_frame, frames=num_frames, interval=1000 / fps, repeat=True)
    plt.show()
    return anim  # keep a reference alive


# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python full_avatar_viewer.py path/to/clip.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Extracting landmarks from {video_path} ...")
    seq = extract_full_sequence(video_path)
    print(f"Extracted {seq['pose'].shape[0]} frames. Opening avatar window...")
    play_avatar_window(seq)
