"""Integrated LUFS via pyloudnorm (the optional `loudness` extra)."""

from __future__ import annotations

import numpy as np


class PyloudnormMeter:
    """`LufsMeter`. None when pyloudnorm is not installed or cannot measure the
    signal, which is how camelot treated both cases."""

    def integrated_lufs(self, y: np.ndarray, sr: int) -> float | None:
        try:
            import pyloudnorm as pyln
            return round(float(pyln.Meter(sr).integrated_loudness(y)), 1)
        except Exception:
            return None


class NoMeter:
    """`LufsMeter` that never measures, for light consumers and tests."""

    def integrated_lufs(self, y: np.ndarray, sr: int) -> float | None:
        return None
