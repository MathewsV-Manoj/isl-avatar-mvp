"""Validate the multi-clip supplemental ISL landmark corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=Path("signal_dictionary_supplemental.json"))
    args = parser.parse_args()
    data = json.loads(args.dictionary.read_text(encoding="utf-8"))
    clips = 0
    for label, sequences in data.items():
        for index, frames in enumerate(sequences):
            array = np.asarray(frames, dtype=np.float32)
            if array.shape != (30, 225) or not np.isfinite(array).all():
                raise ValueError(f"{label}[{index}]: expected finite shape (30, 225)")
            points = array.reshape(30, 75, 3)
            if not np.array_equal(points[:, 33], points[:, 15]):
                raise ValueError(f"{label}[{index}]: left wrist is not anchored")
            if not np.array_equal(points[:, 54], points[:, 16]):
                raise ValueError(f"{label}[{index}]: right wrist is not anchored")
            clips += 1
    print(f"supplemental: {clips} clips across {len(data)} labels; all shapes and wrist anchors valid")


if __name__ == "__main__":
    main()
