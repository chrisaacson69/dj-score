"""Three-panel key chart, from camelot `key_detect.plot_key_analysis`."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from dj_score.domain.model import KeyPhrase, KeySegment, Measure
from dj_score.domain.timefmt import format_time


def plot_key_analysis(y, sr, beat_times, measures: list[Measure], phrases: list[KeyPhrase],
                      segments: list[KeySegment], beats_per_bar, title="",
                      zoom_start=None, zoom_end=None, phrase_bars=4):
    """Waveform tinted by key segment, the key timeline, and per-phrase confidence.

    Beat ticks and measure lines appear only when zoomed; the full view marks
    phrase boundaries instead.
    """
    duration = len(y) / sr
    times = np.arange(len(y)) / sr

    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    zoomed = zoom_start is not None or zoom_end is not None

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 0.6, 0.5], hspace=0.3, figure=fig)

    all_codes = sorted({p.camelot for p in phrases})
    cmap = plt.cm.Set3 if len(all_codes) <= 12 else plt.cm.tab20
    code_colors = {code: cmap(i / max(len(all_codes), 1)) for i, code in enumerate(all_codes)}

    phrase_boundaries = {measures[i].start for i in range(0, len(measures), phrase_bars)}
    if measures:
        phrase_boundaries.add(measures[-1].end)

    # Panel 1: waveform with key segment colors
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888", linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.6)

    for seg in segments:
        c = code_colors.get(seg.camelot, "#cccccc")
        ax1.axvspan(seg.start_time, seg.end_time, alpha=0.25, color=c)
        mid = (seg.start_time + seg.end_time) / 2
        if view_start <= mid <= view_end:
            ax1.text(mid, 0, f"{seg.camelot}\n{seg.key_name}", fontsize=8, ha="center", va="bottom",
                     color="#222222", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    if zoomed:
        for bt in beat_times:
            if view_start <= bt <= view_end:
                ax1.axvline(x=bt, color="#cc4444", linewidth=0.6, alpha=0.4)
        for m in measures:
            if view_start <= m.start <= view_end:
                ax1.axvline(x=m.start, color="#2244aa", linewidth=1.2, alpha=0.6)
    else:
        for pb in phrase_boundaries:
            if view_start <= pb <= view_end:
                ax1.axvline(x=pb, color="#2244aa", linewidth=0.5, alpha=0.35)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Key Segments ({beats_per_bar}/4)  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    patches = [mpatches.Patch(color=code_colors[c], label=c) for c in all_codes]
    ax1.legend(handles=patches, fontsize=7, loc="upper right", ncol=min(len(all_codes), 6))

    # Panel 2: key timeline, one block per segment
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for seg in segments:
        c = code_colors.get(seg.camelot, "#cccccc")
        seg_dur = seg.end_time - seg.start_time
        ax2.barh(0, seg_dur, left=seg.start_time, height=0.8, color=c, edgecolor="#333333", linewidth=0.8)

        mid = (seg.start_time + seg.end_time) / 2
        if view_start <= mid <= view_end:
            share = seg_dur / (view_end - view_start)
            if share > 0.06:
                ax2.text(mid, 0, f"{seg.camelot}\n{seg.key_name}", fontsize=8, ha="center", va="center",
                         fontweight="bold", color="#111111")
            elif share > 0.025:
                ax2.text(mid, 0, seg.camelot, fontsize=7, ha="center", va="center",
                         fontweight="bold", color="#111111")

    ax2.set_yticks([])
    ax2.set_ylabel("Key")
    ax2.set_title("Key Timeline")
    ax2.set_ylim(-0.5, 0.5)

    # Panel 3: confidence per phrase
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for p in phrases:
        ax3.bar(p.start, p.confidence, width=p.end - p.start, align="edge",
                color=code_colors.get(p.camelot, "#cccccc"), edgecolor="#667799", linewidth=0.3, alpha=0.7)
    ax3.set_ylabel("Conf")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_title("Key Detection Confidence per Phrase")
    ax3.set_ylim(0, 1.1)

    return fig
