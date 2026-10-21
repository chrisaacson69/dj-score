"""Event charts, from camelot `event_detect.plot_stem_analysis` / `plot_feature_analysis`."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from dj_score.domain.events import STEM_NAMES, normalize_feature
from dj_score.domain.model import Event, EventPhrase, Measure
from dj_score.domain.timefmt import format_time

STEM_COLORS = {"drums": "#cc4444", "bass": "#4488cc", "vocals": "#44aa66", "other": "#aa8844"}


def _view(y, sr, zoom_start, zoom_end):
    duration = len(y) / sr
    view_start = zoom_start if zoom_start is not None else 0
    view_end = zoom_end if zoom_end is not None else duration
    return np.arange(len(y)) / sr, view_start, view_end, zoom_start is not None or zoom_end is not None


def _phrase_boundaries(measures, phrase_bars):
    boundaries = {measures[i].start for i in range(0, len(measures), phrase_bars)}
    if measures:
        boundaries.add(measures[-1].end)
    return boundaries


def _grid_lines(ax, measures, boundaries, view_start, view_end, zoomed):
    """Measure lines when zoomed, phrase boundaries otherwise."""
    if zoomed:
        for m in measures:
            if view_start <= m.start <= view_end:
                ax.axvline(x=m.start, color="#2244aa", linewidth=0.8, alpha=0.4)
    else:
        for pb in boundaries:
            if view_start <= pb <= view_end:
                ax.axvline(x=pb, color="#2244aa", linewidth=0.4, alpha=0.25)


def plot_stem_analysis(y, sr, measures: list[Measure], phrases: list[EventPhrase], events: list[Event],
                       stem_energies: dict[str, list[float]], stem_presence: dict[str, list[bool]],
                       beats_per_bar, title="", phrase_bars=4, zoom_start=None, zoom_end=None):
    """Waveform with events, then one energy panel per stem (shaded where present)."""
    times, view_start, view_end, zoomed = _view(y, sr, zoom_start, zoom_end)
    stem_names = [s for s in STEM_NAMES if s in stem_energies]

    fig = plt.figure(figsize=(16, 4 + len(stem_names) * 1.8), constrained_layout=True)
    gs = gridspec.GridSpec(1 + len(stem_names), 1, height_ratios=[1.5] + [0.7] * len(stem_names),
                           hspace=0.3, figure=fig)
    boundaries = _phrase_boundaries(measures, phrase_bars)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888", linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.5)
    for e in events:
        if view_start <= e.time <= view_end:
            ax1.axvline(x=e.time, color="#cc2222", linewidth=2.0, alpha=0.7)
            ax1.text(e.time, 0.95, f"Bar {e.bar}\n{e.type}", transform=ax1.get_xaxis_transform(),
                     fontsize=6, ha="center", va="top", color="#cc2222", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
    _grid_lines(ax1, measures, boundaries, view_start, view_end, zoomed)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Events  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    for panel_idx, stem_name in enumerate(stem_names):
        ax = fig.add_subplot(gs[1 + panel_idx], sharex=ax1)
        energies, presence = stem_energies[stem_name], stem_presence[stem_name]
        color = STEM_COLORS.get(stem_name, "#888888")
        for pi, p in enumerate(phrases):
            if pi < len(energies):
                is_present = presence[pi] if pi < len(presence) else False
                ax.bar(p.start, energies[pi], width=p.end - p.start, align="edge", color=color,
                       alpha=0.8 if is_present else 0.25, edgecolor="#333333", linewidth=0.3)
        for e in events:
            if view_start <= e.time <= view_end:
                ax.axvline(x=e.time, color="#cc2222", linewidth=1.5, alpha=0.5, linestyle="--")
        ax.set_ylabel(stem_name.capitalize())
        ax.set_title(f"{stem_name.capitalize()} Energy (shaded = present)")

    fig.axes[-1].set_xlabel("Time (seconds)")
    return fig


_CHANGE_WEIGHTS = {"onset_percussive": 0.25, "onset_harmonic": 0.20, "rms": 0.25,
                   "centroid": 0.15, "low_freq_ratio": 0.15}


def plot_feature_analysis(y, sr, measures: list[Measure], phrases: list[EventPhrase], events: list[Event],
                          beats_per_bar, title="", phrase_bars=4, zoom_start=None, zoom_end=None):
    """Waveform with events, then percussive onset, harmonic onset, RMS and the
    combined change score per phrase; phrases that start an event are red."""
    times, view_start, view_end, zoomed = _view(y, sr, zoom_start, zoom_end)

    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.2, 0.5, 0.5, 0.5, 0.5], hspace=0.3, figure=fig)
    boundaries = _phrase_boundaries(measures, phrase_bars)
    event_bars = {e.bar for e in events}

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(times, y, color="#888888", linewidth=0.4 if zoomed else 0.2, alpha=0.7 if zoomed else 0.5)
    for e in events:
        if view_start <= e.time <= view_end:
            color = "#cc2222" if e.type == "decrease" else ("#2266cc" if e.type == "increase" else "#888800")
            ax1.axvline(x=e.time, color=color, linewidth=2.0, alpha=0.7)
            ax1.text(e.time, 0.95, f"Bar {e.bar}\n{e.type or ''}", transform=ax1.get_xaxis_transform(),
                     fontsize=5, ha="center", va="top", color=color, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85))
    _grid_lines(ax1, measures, boundaries, view_start, view_end, zoomed)

    ax1.set_ylabel("Amplitude")
    zoom_label = f"  [{format_time(view_start)}-{format_time(view_end)}]" if zoomed else ""
    ax1.set_title(f"Waveform with Structural Events  {title}{zoom_label}")
    ax1.set_xlim(view_start, view_end)

    for row, (field, color, ylabel, panel_title) in enumerate([
        ("onset_percussive", "#cc8866", "Perc", "Percussive Onset per Phrase (drums/hits)"),
        ("onset_harmonic", "#6688aa", "Harm", "Harmonic Onset per Phrase (melodic/tonal)"),
        ("rms", "#66aa88", "RMS", "RMS Energy per Phrase"),
    ], start=1):
        ax = fig.add_subplot(gs[row], sharex=ax1)
        for p in phrases:
            ax.bar(p.start, getattr(p, field), width=p.end - p.start, align="edge",
                   color="#cc4444" if p.start_bar in event_bars else color,
                   edgecolor="#445566", linewidth=0.3, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)

    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    normed = {f: normalize_feature(np.array([getattr(p, f) for p in phrases])) for f in _CHANGE_WEIGHTS}
    context = 2
    scores = np.zeros(len(phrases))
    for i in range(context, len(phrases)):
        for f, w in _CHANGE_WEIGHTS.items():
            scores[i] += abs(normed[f][i] - np.mean(normed[f][max(0, i - context):i])) * w

    for i, p in enumerate(phrases):
        ax5.bar(p.start, scores[i], width=p.end - p.start, align="edge",
                color="#cc4444" if p.start_bar in event_bars else "#aa8866",
                edgecolor="#445566", linewidth=0.3, alpha=0.7)
    ax5.axhline(y=0.15, color="#cc2222", linewidth=1.0, linestyle="--", alpha=0.5, label="min_score threshold")
    ax5.legend(fontsize=7, loc="upper right")
    ax5.set_ylabel("Score")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_title("Combined Feature Change Score per Phrase")
    return fig
