"""Per-bar third-octave spectra and their summary.

Converted from camelot `spectral_analysis.py`; the heatmap lives in `adapters/plots`.
"""

from __future__ import annotations

import librosa
import numpy as np

from dj_score.domain.model import Measure, SpectralSummary, SpectrumAnalysis

# ISO 266 third-octave center frequencies (31 bands, 20 Hz - 20 kHz)
THIRD_OCTAVE_CENTERS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
    20000,
])

_THIRD_OCTAVE_FACTOR = 2 ** (1.0 / 6.0)  # band edge multiplier

NOISE_FLOOR_DB = -80.0


def compute_band_edges(centers):
    """(n_bands, 2) array of [low_hz, high_hz] per third-octave band."""
    f = _THIRD_OCTAVE_FACTOR
    return np.column_stack([centers / f, centers * f])


def compute_bar_spectra(y, sr, measures: list[Measure], centers=THIRD_OCTAVE_CENTERS,
                        hop_length=512) -> SpectrumAnalysis:
    """Per-bar energy (dB) in third-octave bands via one full-track STFT.

    Bars under 0.1 s, or with no frames, sit at the noise floor. The summary is
    left empty; fill it with `compute_spectral_summary`.
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)

    edges = compute_band_edges(centers)
    n_bands = len(centers)
    band_masks = [(freqs >= low) & (freqs < high) for low, high in edges]

    bar_spectra = []
    for m in measures:
        if m.end - m.start < 0.1:
            bar_spectra.append(np.full(n_bands, NOISE_FLOOR_DB))
            continue

        start_frame = int(m.start * sr / hop_length)
        end_frame = min(int(m.end * sr / hop_length), S.shape[1])
        if end_frame <= start_frame:
            bar_spectra.append(np.full(n_bands, NOISE_FLOOR_DB))
            continue

        # Mean power across frames, then mean within each band
        mean_power_per_bin = (S[:, start_frame:end_frame] ** 2).mean(axis=1)

        band_db = np.full(n_bands, NOISE_FLOOR_DB)
        for i, mask in enumerate(band_masks):
            if mask.any():
                band_power = mean_power_per_bin[mask].mean()
                band_db[i] = max(10.0 * np.log10(band_power + 1e-12), NOISE_FLOOR_DB)
        bar_spectra.append(band_db)

    return SpectrumAnalysis(
        bar_spectra=np.array(bar_spectra),
        band_centers=centers.copy(),
        band_edges=edges,
        summary=None,
    )


def compute_spectral_summary(spectrum: SpectrumAnalysis) -> SpectralSummary:
    """Mean/peak spectrum, dominant band, centroid, active bandwidth, and a one-line profile."""
    spectra = spectrum.bar_spectra
    centers = spectrum.band_centers

    mean_spectrum = spectra.mean(axis=0)
    peak_spectrum = spectra.max(axis=0)

    dominant_idx = int(np.argmax(mean_spectrum))
    dominant_hz = float(centers[dominant_idx])

    # Spectral centroid: energy-weighted mean frequency, weighting in linear power
    linear = 10.0 ** (mean_spectrum / 10.0)
    total = linear.sum()
    centroid = float(np.sum(centers * linear) / total) if total > 0 else 0.0

    # Bandwidth: frequency range within 10 dB of the peak
    active_indices = np.where(peak_spectrum >= (peak_spectrum.max() - 10.0))[0]
    if len(active_indices) > 0:
        bw_low = float(centers[active_indices[0]])
        bw_high = float(centers[active_indices[-1]])
    else:
        bw_low, bw_high = 0.0, 0.0

    if dominant_hz < 200:
        char = "Bass-heavy"
    elif dominant_hz < 1000:
        char = "Mid-focused"
    else:
        char = "Bright"

    return SpectralSummary(
        mean_spectrum=[round(float(x), 1) for x in mean_spectrum],
        peak_spectrum=[round(float(x), 1) for x in peak_spectrum],
        dominant_band_idx=dominant_idx,
        dominant_band_hz=dominant_hz,
        spectral_centroid_hz=round(centroid, 1),
        bandwidth_hz=[bw_low, bw_high],
        energy_profile=f"{char} (dominant {dominant_hz:.0f} Hz), active {bw_low:.0f}–{bw_high:.0f} Hz",
    )
