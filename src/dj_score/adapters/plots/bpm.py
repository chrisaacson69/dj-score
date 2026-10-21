"""Beat-grid chart, from camelot `bpm_detect.plot_bpm_analysis`."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from dj_score.domain.model import Beatgrid
from dj_score.domain.timefmt import format_time


def plot_bpm_analysis(y, sr, grid: Beatgrid, beats_per_bar, title="", zoom_start=None, zoom_end=None):
    """Waveform with beat ticks and measure lines, over per-measure BPM bars with the
    smoothed span curve and segment medians. Bar numbers appear when zoomed."""
    measures, segments, span = grid.measures, grid.segments, grid.span
    duration = len(y) / sr
    times = np.arange(len(y)) / sr

    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3, figure=fig)

    # Panel 1: waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888", linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.6)

    seg_colors = plt.cm.Pastel1(np.linspace(0, 0.8, max(len(segments), 1)))
    for i, seg in enumerate(segments):
        ax1.axvspan(seg.start_time, seg.end_time, alpha=0.2, color=seg_colors[i % len(seg_colors)])
        mid = (seg.start_time + seg.end_time) / 2
        if view_start <= mid <= view_end:
            ax1.text(mid, 0, f"{seg.bpm:.1f}", fontsize=8, ha="center", va="bottom", color="#333333",
                     fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    for bt in grid.beat_times:
        if view_start <= bt <= view_end:
            ax1.axvline(x=bt, color="#cc4444", linewidth=0.8 if zoomed else 0.2, alpha=0.5 if zoomed else 0.3)

    for m in measures:
        if view_start <= m.start <= view_end:
            ax1.axvline(x=m.start, color="#2244aa", linewidth=1.5 if zoomed else 0.6, alpha=0.7 if zoomed else 0.5)
            if zoomed:
                ax1.text(m.start, 0.98, f"{m.measure_num}", fontsize=7, ha="center", va="top", color="#2244aa",
                         fontweight="bold", transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Beat Grid ({beats_per_bar}/4)  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    # Panel 2: BPM
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for i, seg in enumerate(segments):
        ax2.axvspan(seg.start_time, seg.end_time, alpha=0.15, color=seg_colors[i % len(seg_colors)])

    for m in measures:
        ax2.bar(m.start, m.bpm, width=m.end - m.start, align="edge", color="#aabbdd",
                edgecolor="#667799", linewidth=0.3, alpha=0.6)

    if span is not None and len(span.times) > 0:
        ax2.plot(span.times, span.bpms, color="#cc3333", linewidth=1.5, alpha=0.9,
                 label=f"{span.span}-beat span (median: {span.median:.1f})")

    for seg in segments:
        ax2.hlines(seg.bpm, seg.start_time, seg.end_time, colors="#224488", linewidth=2, alpha=0.7)

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("BPM")
    ax2.set_title("BPM per Measure (bars) with Smoothed Span Curve")
    ax2.legend(fontsize=9, loc="upper right")

    all_bpms = [m.bpm for m in measures]
    if all_bpms:
        bpm_min, bpm_max = min(all_bpms), max(all_bpms)
        margin = max(2, (bpm_max - bpm_min) * 0.3)
        ax2.set_ylim(bpm_min - margin, bpm_max + margin)

    return fig
