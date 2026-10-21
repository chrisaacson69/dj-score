"""Timeline views, from camelot `visualize_timeline.py`. They render the timeline report
JSON (what `dj-score timeline` writes, or a camelot `--timeline` file) as-is.

One addition: an event may carry a "label" (dj-score's event type), shown as given;
without one, the label comes from camelot's energy-event "types" as before.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from dj_score.domain.timefmt import format_time

CAMELOT_COLORS = {
    # A codes (minor): cooler colours
    "1A": "#1e3a5f", "2A": "#2e5984", "3A": "#3d78a8", "4A": "#4a97cc", "5A": "#5bb5e0", "6A": "#6bcfef",
    "7A": "#5fa89e", "8A": "#52827d", "9A": "#456b5c", "10A": "#38543b", "11A": "#4a6b3a", "12A": "#5c8239",
    # B codes (major): warmer colours
    "1B": "#8b4513", "2B": "#a0522d", "3B": "#cd853f", "4B": "#daa520", "5B": "#ffd700", "6B": "#ffb347",
    "7B": "#ff8c00", "8B": "#ff6347", "9B": "#dc143c", "10B": "#c71585", "11B": "#9932cc", "12B": "#6a0dad",
}

EVENT_COLORS = {
    "DROP": "#ff0000", "bass drop": "#ff4500", "build": "#ffa500", "drums in": "#32cd32",
    "drums out": "#228b22", "breakdown": "#4169e1", "energy drop": "#1e90ff", "bass in": "#ff6347",
    "bass out": "#8b0000", "new element": "#9400d3", "texture shift": "#da70d6", "change": "#808080",
}


def get_camelot_color(camelot: str) -> str:
    return CAMELOT_COLORS.get(camelot, "#cccccc")


def classify_energy_event(event_types: list[str]) -> str:
    """camelot's energy-event types -> a display label."""
    if "energy_increase" in event_types and "drums_increase" in event_types:
        return "DROP"
    if "energy_increase" in event_types and "bass_increase" in event_types:
        return "bass drop"
    if "energy_increase" in event_types:
        return "build"
    if "drums_increase" in event_types:
        return "drums in"
    if "energy_decrease" in event_types and "drums_decrease" in event_types:
        return "breakdown"
    if "energy_decrease" in event_types:
        return "energy drop"
    if "drums_decrease" in event_types:
        return "drums out"
    if "bass_increase" in event_types:
        return "bass in"
    if "bass_decrease" in event_types:
        return "bass out"
    if "timbre_change" in event_types:
        return "new element"
    if "texture_change" in event_types:
        return "texture shift"
    return "change"


def event_label(event: dict) -> str:
    return event.get("label") or classify_energy_event(event.get("types", []))


def _seg_end(s):
    return s.get("end", s.get("start", 0) + s.get("duration", 0))


def plot_summary_view(data: dict):
    """Key regions, structural events, and the normalized energy/onset curves."""
    estimate = data.get("estimate", {})
    timeline = estimate.get("timeline", [])
    energy_events = estimate.get("energy_events", [])
    duration = estimate.get("duration", 0)
    bpm = estimate.get("bpm", 120)
    summary = estimate.get("summary", {})

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), height_ratios=[2, 1, 2],
                             gridspec_kw={"hspace": 0.3}, constrained_layout=True)

    ax_key = axes[0]
    ax_key.set_xlim(0, duration)
    ax_key.set_ylim(0, 1)
    ax_key.set_ylabel("Key")
    ax_key.set_title(f"Key Timeline - Dominant: {summary.get('dominant', {}).get('camelot', '?')} "
                     f"({summary.get('dominant', {}).get('key', '?')})")
    for entry in timeline:
        start, end = entry.get("start", 0), entry.get("end", 0)
        camelot = entry.get("camelot", "?")
        alpha = 0.9 if entry.get("type", "unknown") == "stable" else 0.5
        ax_key.add_patch(Rectangle((start, 0), end - start, 1, facecolor=get_camelot_color(camelot), alpha=alpha,
                                   edgecolor="white", linewidth=0.5))
        if end - start > 20:  # only label wide regions
            ax_key.text((start + end) / 2, 0.5, camelot, ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
    ax_key.set_yticks([])

    ax_energy = axes[1]
    ax_energy.set_xlim(0, duration)
    ax_energy.set_ylim(0, 1)
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Energy Events (Structural Changes)")
    for event in energy_events:
        time = event.get("time", 0)
        label = event_label(event)
        color = EVENT_COLORS.get(label, "#808080")
        ax_energy.axvline(x=time, color=color, alpha=0.8, linewidth=2)
        ax_energy.text(time, 0.9, label[:6], ha="center", va="top", fontsize=7, rotation=45, color=color)
    ax_energy.set_yticks([])

    ax_metrics = axes[2]
    raw_segments = estimate.get("raw_segments", [])
    if raw_segments:
        times = [s.get("start", 0) + (_seg_end(s) - s.get("start", 0)) / 2 for s in raw_segments]
        rms = [s.get("energy", {}).get("rms_mean", 0) for s in raw_segments]
        onset = [s.get("energy", {}).get("onset_density", 0) for s in raw_segments]
        rms_norm = np.array(rms) / max(rms) if max(rms) > 0 else rms
        onset_norm = np.array(onset) / max(onset) if max(onset) > 0 else onset
        ax_metrics.fill_between(times, 0, rms_norm, alpha=0.5, label="Energy (RMS)", color="blue")
        ax_metrics.plot(times, onset_norm, "g-", alpha=0.7, label="Onset Density", linewidth=1.5)
        ax_metrics.legend(loc="upper right")
    ax_metrics.set_xlim(0, duration)
    ax_metrics.set_ylim(0, 1.1)
    ax_metrics.set_xlabel("Time (seconds)")
    ax_metrics.set_ylabel("Normalized")
    ax_metrics.set_title("Energy & Rhythm Overview")

    bar_times = np.arange(0, duration, (4 * 60.0) / bpm * estimate.get("bars_per_segment", 8))
    for ax in axes:
        for t in bar_times:
            ax.axvline(x=t, color="gray", alpha=0.2, linewidth=0.5)
    for ax in axes[:-1]:
        ax.set_xticks([])
    time_ticks = np.arange(0, duration, 60)
    axes[-1].set_xticks(time_ticks)
    axes[-1].set_xticklabels([format_time(t) for t in time_ticks])
    return fig


def plot_detailed_view(data: dict):
    """Per-window key confidence and energy metrics; None without raw segments."""
    estimate = data.get("estimate", {})
    raw_segments = estimate.get("raw_segments", [])
    duration = estimate.get("duration", 0)
    bpm = estimate.get("bpm", 120)
    if not raw_segments:
        return None

    fig, axes = plt.subplots(6, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"hspace": 0.1}, constrained_layout=True)

    times = [s.get("start", 0) for s in raw_segments]
    widths = [_seg_end(s) - s.get("start", 0) for s in raw_segments]
    energy = [s.get("energy", {}) for s in raw_segments]
    confidence = [s.get("confidence", 0) for s in raw_segments]
    rms = [e.get("rms_mean", 0) for e in energy]
    rms_std = [e.get("rms_std", 0) for e in energy]
    onset = [e.get("onset_density", 0) for e in energy]
    centroid = [e.get("spectral_centroid", 0) for e in energy]
    low_freq = [e.get("low_freq_ratio", 0) for e in energy]
    colors = [get_camelot_color(s.get("camelot", "?")) for s in raw_segments]

    ax = axes[0]
    ax.bar(times, confidence, width=widths, align="edge", color=colors, alpha=0.8)
    ax.set_ylabel("Key Conf.")
    ax.set_ylim(0, max(confidence) * 1.1 if confidence else 1)
    ax.set_title("Key Detection Confidence (color = Camelot code)")
    for i, s in enumerate(raw_segments):
        if confidence[i] > 0.05:
            ax.text(times[i] + widths[i] / 2, confidence[i], s.get("camelot", ""),
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax = axes[1]
    ax.bar(times, rms, width=widths, align="edge", color="steelblue", alpha=0.7)
    ax.errorbar([t + w / 2 for t, w in zip(times, widths)], rms, yerr=rms_std,
                fmt="none", color="darkblue", alpha=0.5, capsize=2)
    ax.set_ylabel("RMS Energy")
    ax.set_title("RMS Energy (with std dev)")

    for ax, values, color, ylabel, title in (
        (axes[2], onset, "forestgreen", "Onsets/sec", "Onset Density (rhythmic activity)"),
        (axes[3], centroid, "darkorange", "Centroid (Hz)", "Spectral Centroid (brightness/timbre)"),
        (axes[4], low_freq, "darkred", "Low Freq %", "Low Frequency Ratio (bass presence)"),
    ):
        ax.bar(times, values, width=widths, align="edge", color=color, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    def norm(arr):
        arr = np.array(arr)
        return arr / arr.max() if arr.max() > 0 else arr

    ax = axes[5]
    x_centers = [t + w / 2 for t, w in zip(times, widths)]
    ax.plot(x_centers, norm(rms), "b-", label="RMS", alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(onset), "g-", label="Onset", alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(centroid), "orange", label="Centroid", alpha=0.7, linewidth=1.5)
    ax.plot(x_centers, norm(low_freq), "r-", label="LowFreq", alpha=0.7, linewidth=1.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("Normalized")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("All Metrics Normalized (for comparison)")

    bar_times = np.arange(0, duration, (4 * 60.0) / bpm * estimate.get("bars_per_segment", 8))
    for ax in axes:
        for t in bar_times:
            ax.axvline(x=t, color="gray", alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, duration)
    time_ticks = np.arange(0, duration, 30)
    axes[-1].set_xticks(time_ticks)
    axes[-1].set_xticklabels([format_time(t) for t in time_ticks], rotation=45)
    return fig
