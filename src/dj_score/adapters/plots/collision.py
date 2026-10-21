"""Three-panel collision figure, from camelot `collision.plot_collision`."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dj_score.adapters.plots.spectral import freq_edges, set_log_freq_axis
from dj_score.domain.collision import CollisionResult, GapResult


def plot_collision(col: CollisionResult, gaps: GapResult, name_a, name_b):
    """Collision heatmap with suggested crossovers, collision per band, collision per bar."""
    centers = col.band_centers
    n_bars = col.collision.shape[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [4, 2, 2]})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.06, hspace=0.35)

    T, F = np.meshgrid(np.arange(n_bars + 1), freq_edges(col.band_edges))
    pcm = ax1.pcolormesh(T, F, col.collision.T, shading="flat", cmap="YlOrRd", vmin=0,
                         vmax=np.percentile(col.collision, 95))
    set_log_freq_axis(ax1)

    for x in gaps.suggested_crossovers:
        ax1.axhline(y=x.frequency_hz, color="cyan", linewidth=1.5, linestyle="--", alpha=0.8)
        ax1.text(n_bars * 0.02, x.frequency_hz * 1.1, f"{x.frequency_hz:.0f} Hz",
                 color="cyan", fontsize=8, fontweight="bold")

    cb = fig.colorbar(pcm, ax=ax1, pad=0.02)
    cb.set_label("Collision Score", fontsize=8)
    ax1.set_xlabel("Bar (within overlap region)", fontsize=9)
    ax1.set_ylabel("Frequency (Hz)", fontsize=9)
    ax1.set_title(f"Spectral Collision: {name_a} vs {name_b}", fontsize=10)
    ax1.tick_params(labelsize=8)

    band_labels = [f"{c:.0f}" if c < 1000 else f"{c/1000:.1f}k" for c in centers]
    y_pos = np.arange(len(centers))
    ax2.barh(y_pos, col.total_per_band, color="#cc6633", alpha=0.8)
    ax2.set_yticks(y_pos[::3])
    ax2.set_yticklabels(band_labels[::3], fontsize=7)
    ax2.set_xlabel("Total Collision", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)
    ax2.set_title("Collision by Frequency Band", fontsize=10)
    ax2.tick_params(labelsize=8)

    bars = np.arange(n_bars)
    ax3.plot(bars, col.total_per_bar, color="#336699", linewidth=1.2)
    ax3.fill_between(bars, col.total_per_bar, alpha=0.3, color="#336699")
    ax3.set_xlabel("Bar (within overlap region)", fontsize=9)
    ax3.set_ylabel("Total Collision", fontsize=9)
    ax3.set_title("Collision Over Time", fontsize=10)
    ax3.tick_params(labelsize=8)
    return fig
