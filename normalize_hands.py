"""
Hand normalization pass — improves handshape consistency without retraining.

Two fixes, based on published findings that wrist-centring plus zoom
normalization substantially improves hand pose accuracy:

  1. ZOOM NORMALIZATION. Hand scale in the current dictionary varies by
     thousands of times across frames (measured). The same handshape at
     different camera distances produces completely different numbers.
     Fix: rescale every hand so the wrist-to-knuckle span is constant.

  2. DROPPED-FRAME REPAIR. MediaPipe returns all-zero hands on a large
     fraction of frames. Instead of leaving holes, interpolate from the
     nearest valid frames so the avatar holds a plausible pose.

Input : signal_dictionary_smoothed.json
Output: signal_dictionary_normalized.json
"""
import json, numpy as np, sys

POSE, HAND = 33, 21
LH_OFF, RH_OFF = POSE*3, (POSE+HAND)*3
TARGET_SCALE = 0.25          # canonical wrist->MCP span
MCP = [5, 9, 13, 17]         # knuckle landmarks

def hand_scale(rel):
    """rel: [T,21,3] wrist-centred. Returns [T] scale per frame."""
    return np.linalg.norm(rel[:, MCP, :], axis=2).mean(axis=1)

def repair_and_normalize(hand):
    """hand: [T,21,3] absolute. Returns normalized [T,21,3] + n_repaired."""
    T = hand.shape[0]
    wrist = hand[:, 0:1, :]
    rel = hand - wrist
    scale = hand_scale(rel)

    valid = scale > 1e-4
    n_bad = int((~valid).sum())

    if valid.sum() == 0:
        return hand, T                      # nothing usable, leave as-is

    # --- fix 2: fill dropped frames by nearest-valid interpolation ---
    if n_bad:
        idx = np.arange(T)
        good = idx[valid]
        for j in range(21):
            for k in range(3):
                rel[:, j, k] = np.interp(idx, good, rel[good, j, k])
        scale = hand_scale(rel)
        valid = scale > 1e-4

    # --- fix 1: zoom normalization ---
    safe = np.where(valid, scale, 1.0)
    rel = rel * (TARGET_SCALE / safe)[:, None, None]

    # wrist position itself is kept from the (shoulder-normalized) pose,
    # so the hand still sits in the right place on the body
    return rel + wrist, n_bad

def main(inp, outp):
    d = json.load(open(inp))
    out = {}
    tot_bad = 0
    tot_frames = 0
    for word, seq in d.items():
        arr = np.array(seq)                   # [T,225]
        T = arr.shape[0]
        tot_frames += T*2
        for off in (LH_OFF, RH_OFF):
            hand = arr[:, off:off+HAND*3].reshape(T, HAND, 3)
            fixed, nbad = repair_and_normalize(hand)
            tot_bad += nbad
            arr[:, off:off+HAND*3] = fixed.reshape(T, HAND*3)
        out[word] = arr.tolist()

    json.dump(out, open(outp, "w"))
    print(f"words processed : {len(out)}")
    print(f"hand-frames total: {tot_frames}")
    print(f"dropped frames repaired: {tot_bad} ({tot_bad/tot_frames*100:.1f}%)")
    print(f"saved -> {outp}")

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv)>1 else "signal_dictionary_smoothed.json"
    b = sys.argv[2] if len(sys.argv)>2 else "signal_dictionary_normalized.json"
    main(a, b)
