"""Per-bar kick-pattern classification: 4/4, 8th-note, half-time, no-kick, complex.

Converted from camelot `rhythm_detect.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from dj_score.domain.kick import kick_envelope
from dj_score.domain.model import Measure


@dataclass
class BarRhythm:
    pattern: str
    """"4/4", "8th-note", "half-time", "no-kick" or "complex"."""
    kick_density: float
    """Onsets per beat: 0 silent, 1 quarter notes, 2 eighth notes."""
    onsets: int
    confidence: float
    bar: int | None = None


@dataclass
class PatternChange:
    bar: int
    from_: str = field(metadata={"key": "from"})
    to: str


@dataclass
class RhythmSummary:
    dominant_pattern: str
    pattern_coverage: dict[str, float]
    pattern_changes: list[PatternChange]
    total_bars: int


@dataclass
class RhythmSection:
    start_bar: int
    end_bar: int
    pattern: str
    bar_count: int
    avg_kick_density: float
    start_time: float | None = None
    end_time: float | None = None


@dataclass
class Bar:
    """A bar on a rigid grid: what `classify_rhythm` needs when no beat grid exists."""
    measure_num: int
    start: float
    end: float


def rigid_bars(duration, bpm, first_downbeat=0.0, beats_per_bar=4) -> list[Bar]:
    """Equal bars from `first_downbeat` at `bpm`; the last may be cut short by the
    track end (kept while at least half a bar remains)."""
    bar_duration = 60.0 / bpm * beats_per_bar
    bars, t, n = [], first_downbeat, 0
    while t + bar_duration <= duration + bar_duration * 0.5:
        bars.append(Bar(n, t, min(t + bar_duration, duration)))
        t += bar_duration
        n += 1
    return bars


def classify_bar_rhythm(audio_segment, sr, beat_interval, beats_per_bar=4,
                        onset_threshold=None) -> BarRhythm:
    """Classify one bar's kick pattern from peak-picked onsets on the kick envelope."""
    if len(audio_segment) < sr * 0.3:
        return BarRhythm("no-kick", 0.0, 0, 0.5)

    # Heavier (30 ms) smoothing: kick resonance oscillates at 50-100 ms
    env = kick_envelope(audio_segment, sr, smooth_ms=30.0)

    env_peak = np.max(env)
    env_rms = np.sqrt(np.mean(env ** 2))
    if env_peak < 1e-5 or env_rms < 1e-6:
        return BarRhythm("no-kick", 0.0, 0, 0.9)

    bar_duration = len(audio_segment) / sr

    # 40% of peak, to skip resonance secondary peaks
    if onset_threshold is None:
        onset_threshold = env_peak * 0.4

    # Local maxima above threshold, at least 60% of an 8th note apart
    eighth_note_samples = int(beat_interval * 0.5 * sr)
    min_spacing_samples = max(int(eighth_note_samples * 0.6), int(beat_interval * 0.3 * sr))
    onset_positions = []

    i = 0
    while i < len(env):
        if env[i] >= onset_threshold:
            peak_start = i
            while i < len(env) and env[i] >= onset_threshold * 0.5:
                i += 1
            peak_idx = peak_start + np.argmax(env[peak_start:i])
            onset_positions.append(peak_idx)
            i = peak_idx + min_spacing_samples
        else:
            i += 1

    n_onsets = len(onset_positions)
    kick_density = n_onsets / beats_per_bar if beats_per_bar > 0 else 0

    if n_onsets == 0:
        return BarRhythm("no-kick", 0.0, 0, 0.7)

    # Expected counts: half-time = 2, 4/4 = 4, 8th-note = 8
    eighth_note_interval = beat_interval / 2
    quarter_note_interval = beat_interval

    if n_onsets >= 2:
        iois = np.diff(np.array(onset_positions) / sr)
        median_ioi = float(np.median(iois))
        ioi_std = float(np.std(iois))
        regularity = 1.0 - min(ioi_std / median_ioi, 1.0) if median_ioi > 0 else 0.0
    else:
        median_ioi = bar_duration
        regularity = 0.5

    pattern = "complex"
    confidence = 0.5

    if n_onsets <= 1:
        pattern = "half-time"
        confidence = 0.6

    elif 1.5 <= n_onsets <= 2.5 or abs(n_onsets - 2) <= 0.5:
        if regularity > 0.5:
            pattern = "half-time"
            confidence = min(0.5 + regularity * 0.4, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif 3 <= n_onsets <= 5:
        ioi_ratio = median_ioi / quarter_note_interval if quarter_note_interval > 0 else 0
        if 0.6 < ioi_ratio < 1.5 and regularity > 0.25:
            pattern = "4/4"
            confidence = min(0.5 + regularity * 0.3 + (1.0 - abs(ioi_ratio - 1.0)) * 0.2, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif 6 <= n_onsets <= 10:
        ioi_ratio = median_ioi / eighth_note_interval if eighth_note_interval > 0 else 0
        if 0.6 < ioi_ratio < 1.5 and regularity > 0.2:
            pattern = "8th-note"
            confidence = min(0.5 + regularity * 0.3 + (1.0 - abs(ioi_ratio - 1.0)) * 0.2, 0.95)
        else:
            pattern = "complex"
            confidence = 0.4

    elif n_onsets > 10:
        # Very dense: likely 16th notes or noise
        pattern = "complex"
        confidence = 0.3

    return BarRhythm(pattern, round(kick_density, 2), n_onsets, round(confidence, 3))


def classify_rhythm(y, sr, measures: list[Measure] | list[Bar], beat_interval, beats_per_bar=4) -> list[BarRhythm]:
    """`classify_bar_rhythm` for every measure, tagged with its bar number."""
    results = []
    for m in measures:
        start_sample = max(0, int(m.start * sr))
        end_sample = min(len(y), int(m.end * sr))
        if end_sample <= start_sample:
            r = BarRhythm("no-kick", 0.0, 0, 0.5)
        else:
            r = classify_bar_rhythm(y[start_sample:end_sample], sr, beat_interval, beats_per_bar)
        r.bar = m.measure_num
        results.append(r)
    return results


def summarize_rhythm(bar_rhythms: list[BarRhythm]) -> RhythmSummary:
    """Dominant pattern, coverage per pattern, and where the pattern changes."""
    if not bar_rhythms:
        return RhythmSummary("unknown", {}, [], 0)

    pattern_counts = Counter(r.pattern for r in bar_rhythms)
    total = len(bar_rhythms)

    changes = []
    prev_pattern = bar_rhythms[0].pattern
    for r in bar_rhythms[1:]:
        if r.pattern != prev_pattern:
            changes.append(PatternChange(bar=r.bar, from_=prev_pattern, to=r.pattern))
            prev_pattern = r.pattern

    return RhythmSummary(
        dominant_pattern=pattern_counts.most_common(1)[0][0],
        pattern_coverage={p: round(c / total, 3) for p, c in pattern_counts.most_common()},
        pattern_changes=changes,
        total_bars=total,
    )


def summarize_rhythm_sections(bar_rhythms: list[BarRhythm], measures: list[Measure] | list[Bar] | None = None) -> list[RhythmSection]:
    """Runs of consecutive bars with the same pattern; times filled when `measures` is given."""
    if not bar_rhythms:
        return []

    time_lookup = {m.measure_num: (m.start, m.end) for m in measures or []}

    def close(start_bar, end_bar, pattern, densities):
        section = RhythmSection(
            start_bar=start_bar,
            end_bar=end_bar,
            pattern=pattern,
            bar_count=len(densities),
            avg_kick_density=round(sum(densities) / len(densities), 2),
        )
        if start_bar in time_lookup:
            section.start_time = round(time_lookup[start_bar][0], 2)
        if end_bar in time_lookup:
            section.end_time = round(time_lookup[end_bar][1], 2)
        return section

    sections = []
    current_pattern = bar_rhythms[0].pattern
    section_start = bar_rhythms[0].bar
    densities = [bar_rhythms[0].kick_density]

    for prev, r in zip(bar_rhythms, bar_rhythms[1:]):
        if r.pattern != current_pattern:
            sections.append(close(section_start, prev.bar, current_pattern, densities))
            current_pattern = r.pattern
            section_start = r.bar
            densities = [r.kick_density]
        else:
            densities.append(r.kick_density)

    sections.append(close(section_start, bar_rhythms[-1].bar, current_pattern, densities))
    return sections
