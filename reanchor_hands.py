"""
CRITICAL FIX: re-anchor hand landmarks onto the body's wrists.

Diagnosis: MediaPipe returns hand landmarks in their own frame. The original
extraction normalized them by the BODY's shoulder centre, which left the hands
floating 4-6 shoulder-widths away from the wrists. Any avatar driven from this
data computes bone directions from points that are in the wrong place, which is
why arms, elbows and fingers were all wrong at once.

Fix, per frame per hand:
  1. make the hand wrist-relative
  2. rescale it to a sensible size relative to shoulder width
  3. translate it so landmark 0 sits exactly on the pose wrist (15 / 16)
  4. if the hand is collapsed (undetected), interpolate from nearby frames

Run:
    python reanchor_hands.py signal_dictionary_smoothed.json signal_dictionary_fixed.json
"""
import json, sys, numpy as np

POSE, HAND = 33, 21
NUMP = POSE + 2*HAND
L_WRIST, R_WRIST = 15, 16
HAND_LEN = 0.45          # hand span as a fraction of shoulder width

def fix_sequence(arr):
    """arr: [T,225] -> fixed [T,225], plus count of collapsed frames."""
    T = arr.shape[0]
    pts = arr.reshape(T, NUMP, 3).copy()
    pose = pts[:, :POSE, :]
    collapsed = 0

    for hand_i, (off, wi) in enumerate([(POSE, L_WRIST), (POSE+HAND, R_WRIST)]):
        hand = pts[:, off:off+HAND, :]
        sh = np.linalg.norm(pose[:, 11, :] - pose[:, 12, :], axis=1)
        sh = np.where(sh < 1e-6, 1.0, sh)

        rel = hand - hand[:, 0:1, :]
        span = np.linalg.norm(rel, axis=2).max(axis=1)
        valid = span > 1e-6
        collapsed += int((~valid).sum())

        # interpolate collapsed frames from valid neighbours
        if valid.any() and not valid.all():
            idx = np.arange(T); good = idx[valid]
            for j in range(HAND):
                for k in range(3):
                    rel[:, j, k] = np.interp(idx, good, rel[good, j, k])
            span = np.linalg.norm(rel, axis=2).max(axis=1)
            valid = span > 1e-6
        elif not valid.any():
            # No neighbouring hand frame exists to interpolate. Keep this
            # intentionally collapsed hand attached to its anatomical wrist;
            # downstream retargeting then falls back to the neutral hand pose.
            pts[:, off:off+HAND, :] = pose[:, wi:wi+1, :]
            continue

        # rescale to body proportions
        safe = np.where(valid, span, 1.0)
        rel = rel * (HAND_LEN * sh / safe)[:, None, None]

        # anchor onto the pose wrist
        pts[:, off:off+HAND, :] = rel + pose[:, wi:wi+1, :]

    return pts.reshape(T, NUMP*3), collapsed

def main(inp, outp):
    d = json.load(open(inp))
    out, tot_collapsed, tot = {}, 0, 0
    for word, seq in d.items():
        arr = np.array(seq, dtype=float)
        fixed, c = fix_sequence(arr)
        out[word] = fixed.tolist()
        tot_collapsed += c
        tot += arr.shape[0]*2

    json.dump(out, open(outp, "w"))
    print(f"words              : {len(out)}")
    print(f"hand-frames total  : {tot}")
    print(f"collapsed repaired : {tot_collapsed} ({tot_collapsed/tot*100:.1f}%)")
    print(f"saved -> {outp}")

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv)>1 else "signal_dictionary_smoothed.json"
    b = sys.argv[2] if len(sys.argv)>2 else "signal_dictionary_fixed.json"
    main(a, b)
