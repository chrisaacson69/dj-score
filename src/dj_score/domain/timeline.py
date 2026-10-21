"""Key timeline: key + energy per multi-bar window, merged into regions, with mix points.

Converted from camelot `camelot_from_youtube.py` (`analyze_energy_segment`,
`merge_timeline_segments`, `compute_timeline_summary`). The windows now come from
dj-score's beat grid; camelot's own BPM/downbeat and its energy/micro event detectors
are retired, and structural events come from `domain.events` instead.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np

from dj_score.domain.key import WindowCandidate
from dj_score.domain.model import Measure


@dataclass
class EnergyProfile:
    rms_mean: float
    rms_std: float
    onset_density: float
    """Detected onsets per second."""
    spectral_centroid: float
    low_freq_ratio: float
    """Share of power below 250 Hz."""


def energy_profile(y, sr) -> EnergyProfile:
    """Loudness, rhythmic activity, brightness and bass share of one window (full mix)."""
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration_sec = len(y) / sr
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total_energy = np.sum(S ** 2)
    low_freq_ratio = float(np.sum(S[freqs < 250, :] ** 2) / total_energy) if total_energy > 0 else 0
    return EnergyProfile(
        rms_mean=round(float(np.mean(rms)), 6),
        rms_std=round(float(np.std(rms)), 6),
        onset_density=round(len(onsets) / duration_sec if duration_sec > 0 else 0, 2),
        spectral_centroid=round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])), 1),
        low_freq_ratio=round(low_freq_ratio, 3),
    )


@dataclass
class Window:
    start: float
    end: float
    bar_start: int
    bar_end: int
    """Exclusive: the bar after the window's last bar."""


def bar_windows(measures: list[Measure], bars_per_segment=8) -> list[Window]:
    """Windows of `bars_per_segment` bars on the grid, plus a pre-roll window
    (bar_start -1) for audio before the first bar."""
    windows = []
    if measures and measures[0].start > 0:
        windows.append(Window(0.0, measures[0].start, -1, measures[0].measure_num))
    for i in range(0, len(measures), bars_per_segment):
        g = measures[i:i + bars_per_segment]
        windows.append(Window(g[0].start, g[-1].end, g[0].measure_num, g[0].measure_num + len(g)))
    return windows


@dataclass
class TimelineSegment:
    """One analysed window: its key (as `key.WindowKey`) and energy. A window too
    short to analyse carries `error` instead of a key."""
    start: float
    duration: float
    end: float
    bar_start: int | None = None
    bar_end: int | None = None
    key: str | None = None
    camelot: str | None = None
    confidence: float | None = None
    score: float | None = None
    top_candidates: list[WindowCandidate] | None = None
    error: str | None = None
    energy: EnergyProfile | None = None
    absorbed: bool | None = None
    """Set by `merge_timeline_segments` on a low-confidence blip it relabelled."""
    original_camelot: str | None = None


@dataclass
class RegionConfidence:
    average: float
    max: float


@dataclass
class TimelineRegion:
    start: float
    end: float
    type: str
    """"stable", "transition", "uncertain" or "error"."""
    key: str | None = None
    camelot: str | None = None
    confidence: RegionConfidence | None = None
    score: float | None = None
    segments_merged: int | None = None
    bar_start: int | None = None
    bar_end: int | None = None
    absorbed_uncertain: int | None = None
    error: str | None = None


def _conf(s: TimelineSegment) -> float:
    return s.confidence or 0


def _end(s: TimelineSegment) -> float:
    return s.end if s.end is not None else s.start + (s.duration or 0)


def merge_timeline_segments(segments: list[TimelineSegment], min_confidence=0.1,
                            min_run_length=2) -> list[TimelineRegion]:
    """Runs of the same key become regions; a short low-confidence blip between two
    runs of one key is absorbed into them; error windows stay on their own."""
    if not segments:
        return []

    runs, current = [], [segments[0]]
    for seg in segments[1:]:
        if seg.error:
            if current:
                runs.append(current)
            runs.append([seg])
            current = []
        elif not current:
            current = [seg]
        elif seg.camelot == current[-1].camelot:
            current.append(seg)
        else:
            runs.append(current)
            current = [seg]
    if current:
        runs.append(current)

    merged, i = [], 0
    while i < len(runs):
        run = runs[i]
        avg_conf = sum(_conf(s) for s in run) / len(run)
        if len(run) < min_run_length and avg_conf < min_confidence and merged and i + 1 < len(runs):
            prev_key = merged[-1][-1].camelot if merged[-1] else None
            next_key = runs[i + 1][0].camelot if runs[i + 1] else None
            if prev_key == next_key and prev_key is not None:
                for seg in run:
                    seg.absorbed = True
                    seg.original_camelot = seg.camelot
                    seg.camelot = prev_key
                    seg.key = merged[-1][-1].key
                merged[-1].extend(run)
                i += 1
                continue
        merged.append(run)
        i += 1

    timeline = []
    for run in merged:
        if not run:
            continue
        if run[0].error:
            timeline.append(TimelineRegion(start=run[0].start, end=_end(run[-1]), type="error", error=run[0].error))
            continue

        avg_confidence = sum(_conf(s) for s in run) / len(run)
        if len(run) >= min_run_length and avg_confidence >= min_confidence:
            seg_type = "stable"
        elif avg_confidence < min_confidence * 0.5:
            seg_type = "uncertain"
        else:
            seg_type = "transition"

        region = TimelineRegion(
            start=round(run[0].start, 3),
            end=round(_end(run[-1]), 3),
            type=seg_type,
            key=run[0].key,
            camelot=run[0].camelot,
            confidence=RegionConfidence(round(avg_confidence, 3), round(max(_conf(s) for s in run), 3)),
            score=round(sum(s.score or 0 for s in run) / len(run), 4),
            segments_merged=len(run),
        )
        if run[0].bar_start is not None:
            region.bar_start, region.bar_end = run[0].bar_start, run[-1].bar_end
        absorbed = sum(1 for s in run if s.absorbed)
        if absorbed:
            region.absorbed_uncertain = absorbed
        timeline.append(region)
    return timeline


@dataclass
class DominantRegionKey:
    key: str
    camelot: str
    duration: float
    coverage: float


@dataclass
class MixPoint:
    time: float
    key: str
    camelot: str


@dataclass
class TimelineSummary:
    error: str | None = None
    dominant: DominantRegionKey | None = None
    key_changes: list[dict] | None = None
    """[{"time", "bar", "from", "to"}]. Plain dicts because "bar" is written as null when
    the region has no bars, and the codec omits None dataclass fields."""
    total_regions: int | None = None
    stable_regions: int | None = None
    mix_points: dict[str, MixPoint] | None = None
    """"intro" = first stable region, "outro" = last stable region."""


def compute_timeline_summary(timeline: list[TimelineRegion], total_duration) -> TimelineSummary:
    """Dominant key by duration, key changes between regions, and the intro/outro keys."""
    if not timeline:
        return TimelineSummary(error="No timeline data")
    valid = [t for t in timeline if t.camelot and t.type != "error"]
    if not valid:
        return TimelineSummary(error="No valid key regions found")

    key_durations = Counter()
    for t in valid:
        key_durations[t.camelot] += t.end - t.start
    dominant_camelot, dominant_duration = key_durations.most_common(1)[0]
    dominant_entry = next(t for t in valid if t.camelot == dominant_camelot)

    stable = [t for t in valid if t.type == "stable"]
    intro, outro = (stable[0], stable[-1]) if stable else (valid[0], valid[-1])
    return TimelineSummary(
        dominant=DominantRegionKey(dominant_entry.key, dominant_camelot, round(dominant_duration, 2),
                                   round(dominant_duration / total_duration, 3)),
        key_changes=[{"time": cur.start, "bar": cur.bar_start, "from": prev.camelot, "to": cur.camelot}
                     for prev, cur in zip(valid, valid[1:]) if cur.camelot != prev.camelot],
        total_regions=len(valid),
        stable_regions=len(stable),
        mix_points={"intro": MixPoint(intro.start, intro.key, intro.camelot),
                    "outro": MixPoint(outro.start, outro.key, outro.camelot)},
    )
