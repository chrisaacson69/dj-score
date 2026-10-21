"""Per-bar spectral heatmap, from camelot `spectral_analysis.plot_spectrogram`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dj_score.domain.model import Measure, SpectrumAnalysis
from dj_score.domain.spectral import NOISE_FLOOR_DB

FREQ_TICKS = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
FREQ_TICK_LABELS = ["50", "100", "200", "500", "1k", "2k", "5k", "10k"]


def set_log_freq_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(20, 20000)
    ax.set_yticks(FREQ_TICKS)
    ax.set_yticklabels(FREQ_TICK_LABELS)


def freq_edges(band_edges):
    return [band_edges[0, 0]] + [band_edges[i, 1] for i in range(len(band_edges))]


def plot_spectrogram(spectrum: SpectrumAnalysis, measures: list[Measure], title="Spectral Energy"):
    """Time (bar boundaries) x log frequency, colored by dB. `measures` are the
    bars the spectra were computed over."""
    t_edges = [measures[0].start] + [m.end for m in measures]
    T, F = np.meshgrid(t_edges, freq_edges(spectrum.band_edges))

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18)

    pcm = ax.pcolormesh(T, F, spectrum.bar_spectra.T, shading="flat", cmap="inferno",
                        vmin=NOISE_FLOOR_DB, vmax=0)
    set_log_freq_axis(ax)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("dB", fontsize=9)

    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    return fig
