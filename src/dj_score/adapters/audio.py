"""Audio file decoding."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


class LibrosaLoader:
    """`AudioLoader` via `librosa.load`, resampled to `sr`, mono."""

    def load(self, path: Path, sr: int) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y

    def load_window(self, path: Path, sr: int, start: float, duration: float) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=sr, mono=True, offset=float(start), duration=float(duration))
        return y
