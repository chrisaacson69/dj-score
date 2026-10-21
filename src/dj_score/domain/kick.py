"""The kick-drum onset envelope shared by rhythm classification and alignment.

camelot carried two identical copies (`alignment.py`, `rhythm_detect.py`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def kick_envelope(audio, sr=44100, low_hz=30, high_hz=120, smooth_ms=3.0) -> np.ndarray:
    """Bandpass filter + rectify + moving-average smooth; same length as `audio`."""
    sos = butter(4, [low_hz, high_hz], btype="band", fs=sr, output="sos")
    env = np.abs(sosfilt(sos, audio))
    win = max(int(smooth_ms / 1000 * sr), 1)
    return np.convolve(env, np.ones(win) / win, mode="same")
