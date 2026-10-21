"""Whole-signal measurements: RMS/peak loudness, safe playback volume, display waveform.

Converted from camelot `AnalysisStore._compute_loudness` / `store_bpm`. Integrated
LUFS needs a meter (the `LufsMeter` port), so it is passed in, not computed here.
"""

from __future__ import annotations

import numpy as np

from dj_score.domain.model import Loudness, Waveform

DISPLAY_POINTS = 4000


def measure_loudness(y, lufs: float | None = None) -> Loudness:
    """RMS and peak dBFS (-120 for silence), and the playback volume that keeps the
    peak 0.5 dB under full scale (1.0 when the peak is already below -0.1 dBFS)."""
    if y is None or len(y) == 0:
        return Loudness(safe_volume=1.0, lufs=lufs)
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    loudness_dbfs = round(20 * np.log10(rms), 1) if rms > 0 else -120.0
    peak_dbfs = round(20 * np.log10(peak), 1) if peak > 0 else -120.0
    if peak_dbfs > -0.1:
        safe_volume = round(min(1.0, 10 ** (-(peak_dbfs + 0.5) / 20)), 2)
    else:
        safe_volume = 1.0
    return Loudness(loudness_dbfs=loudness_dbfs, peak_dbfs=peak_dbfs, safe_volume=safe_volume, lufs=lufs)


def make_waveform(y, sr, keep_signal=True) -> Waveform:
    """Decimate to about `DISPLAY_POINTS` samples for drawing; keep the full signal unless told not to."""
    factor = max(1, len(y) // DISPLAY_POINTS)
    y_ds = y[::factor]
    return Waveform(y=y if keep_signal else None, y_ds=y_ds, t_ds=np.arange(len(y_ds)) * (factor / sr))
