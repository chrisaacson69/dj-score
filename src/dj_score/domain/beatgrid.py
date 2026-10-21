"""Beat grid: librosa DP beat tracking, intro trimming, downbeat phase, measures,
rolling BPM and tempo segments.

Converted from camelot `bpm_detect.py`. This is today's grid (`source="librosa-dp"`);
ledger step 6 replaces it with the in-house rigid grid and demotes this to a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from dj_score.domain.model import LEGACY_GRID_SOURCE, Beatgrid, Measure, TempoSegment, TempoSpan


@dataclass
class DownbeatEstimate:
    phase: int
    """Beat index that is beat 1 (0 = already aligned)."""
    confidence: float
    """Margin between the best and second-best phase, 0-1."""
    scores: list[float]


@dataclass
class TrimResult:
    beat_times: np.ndarray
    beats_trimmed: int
    bars_trimmed: int


@dataclass
class QuantizationReport:
    frame_duration_ms: float
    unique_interval_count: int
    unique_intervals: np.ndarray
    interval_step_ms: float
    matches_frame_hop: bool


def track_beats(y, sr, hop_length=512) -> tuple[float, np.ndarray]:
    """librosa `beat_track` -> (tempo, beat times in seconds)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else float(tempo)
    else:
        tempo = float(tempo)
    return tempo, beat_times


def _phase_sums(values, beats_per_bar):
    """Sum of `values` over every `beats_per_bar`-th beat, for each starting offset."""
    scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        idxs = list(range(offset, len(values), beats_per_bar))
        if idxs:
            scores[offset] = np.sum(values[idxs])
    return scores


def estimate_downbeat_phase(y, sr, beat_times, beats_per_bar=4) -> DownbeatEstimate:
    """Score each phase offset by four cues and pick beat 1: onset strength (0.3),
    bass energy under 150 Hz (0.3), chroma flux (0.2), interval consistency (0.2)."""
    if len(beat_times) < beats_per_bar * 2:
        return DownbeatEstimate(0, 0.0, [])

    beat_times = np.asarray(beat_times, dtype=float)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_scores = _phase_sums(onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)], beats_per_bar)

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_energy = np.sum(S[freqs < 150, :], axis=0)
    bass_scores = _phase_sums(bass_energy[np.clip(beat_frames, 0, len(bass_energy) - 1)], beats_per_bar)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_flux = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
    harmony_scores = _phase_sums(chroma_flux[np.clip(beat_frames, 0, len(chroma_flux) - 1)], beats_per_bar)

    beat_intervals = np.diff(beat_times)
    interval_scores = np.zeros(beats_per_bar)
    for offset in range(beats_per_bar):
        bar_consistencies = []
        for bar_start in range(offset, len(beat_times) - beats_per_bar, beats_per_bar):
            if bar_start + beats_per_bar - 1 < len(beat_intervals):
                bar_ivl = beat_intervals[bar_start:bar_start + beats_per_bar]
                if len(bar_ivl) == beats_per_bar:
                    bar_consistencies.append(1.0 / (1.0 + np.std(bar_ivl)))
        if bar_consistencies:
            interval_scores[offset] = np.mean(bar_consistencies)

    def _norm(scores):
        mn, mx = scores.min(), scores.max()
        if mx - mn > 1e-12:
            return (scores - mn) / (mx - mn)
        return np.ones_like(scores) * 0.25

    combined = (0.30 * _norm(onset_scores)
                + 0.30 * _norm(bass_scores)
                + 0.20 * _norm(harmony_scores)
                + 0.20 * _norm(interval_scores))

    sorted_scores = np.sort(combined)[::-1]
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    else:
        confidence = 1.0

    return DownbeatEstimate(
        phase=int(np.argmax(combined)),
        confidence=round(float(confidence), 4),
        scores=[round(float(s), 4) for s in combined],
    )


def trim_silent_beats(y, sr, beat_times, hop_length=512, beats_per_bar=4, energy_ratio=0.1) -> TrimResult:
    """Drop whole leading bars whose beats carry almost no bass (< `energy_ratio` x the
    median beat's bass energy under 150 Hz): the tracker locking onto an intro pad."""
    untrimmed = TrimResult(beat_times, 0, 0)
    if len(beat_times) < beats_per_bar * 2:
        return untrimmed

    beat_times = np.asarray(beat_times, dtype=float)
    untrimmed = TrimResult(beat_times, 0, 0)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_energy = np.sum(S[freqs < 150, :], axis=0)
    beat_bass = bass_energy[np.clip(beat_frames, 0, len(bass_energy) - 1)]

    median_bass = float(np.median(beat_bass))
    if median_bass < 1e-6:
        return untrimmed

    threshold = energy_ratio * median_bass
    low_bass_count = 0
    for b in beat_bass:
        if b < threshold:
            low_bass_count += 1
        else:
            break

    bars_to_trim = low_bass_count // beats_per_bar
    if bars_to_trim == 0:
        return untrimmed
    beats_to_trim = bars_to_trim * beats_per_bar
    return TrimResult(beat_times[beats_to_trim:], int(beats_to_trim), int(bars_to_trim))


def compute_span_bpm(beat_times: np.ndarray, span: int) -> TempoSpan:
    """BPM over `span`-beat windows, centered at each window's midpoint; long spans
    average out the ~11.6 ms frame quantization of single-beat intervals."""
    if span >= len(beat_times):
        return TempoSpan(times=np.array([]), bpms=np.array([]))

    span_bpms = 60.0 * span / (beat_times[span:] - beat_times[:-span])
    return TempoSpan(
        times=(beat_times[span:] + beat_times[:-span]) / 2,
        bpms=span_bpms,
        span=span,
        median=float(np.median(span_bpms)),
        std=float(np.std(span_bpms)),
    )


def group_beats_into_measures(beat_times: np.ndarray, beats_per_bar: int = 4, phase: int = 0) -> list[Measure]:
    """Measures from the downbeat at beat index `phase`; earlier beats form a pickup
    measure 0. Per-measure BPM is taken over the whole span to the next downbeat."""
    measures = []
    total_beats = len(beat_times)

    if 0 < phase < total_beats:
        pickup_beats = beat_times[0:phase]
        if len(pickup_beats) >= 2:
            next_time = float(beat_times[phase])
            span_time = next_time - float(pickup_beats[0])
            beat_count = len(pickup_beats)
            bpm = 60.0 * beat_count / span_time if span_time > 0 else 0.0
            measures.append(Measure(0, 0, beat_count, float(pickup_beats[0]), next_time,
                                    float(span_time), round(bpm, 2)))

    bar_num = 1
    for i in range(phase if phase > 0 else 0, total_beats, beats_per_bar):
        end_idx = min(i + beats_per_bar, total_beats)
        measure_beats = beat_times[i:end_idx]
        if len(measure_beats) < 2:
            continue

        if end_idx < total_beats:
            span_time = beat_times[end_idx] - measure_beats[0]
            end_time = float(beat_times[end_idx])
        else:
            # Last (possibly partial) measure: extend by its median interval
            span_time = measure_beats[-1] - measure_beats[0]
            end_time = float(measure_beats[-1] + np.median(np.diff(measure_beats)))

        beat_count = end_idx - i
        bpm = 60.0 * beat_count / span_time if span_time > 0 else 0.0
        measures.append(Measure(bar_num, i, beat_count, float(measure_beats[0]), end_time,
                                float(span_time), round(bpm, 2)))
        bar_num += 1

    return measures


def _median_bpm(group):
    return float(np.median([m.bpm for m in group]))


def find_bpm_segments(measures: list[Measure], tolerance: float = 2.0, min_segment_bars: int = 4) -> list[TempoSegment]:
    """Runs of measures at the same tempo. Short runs (< `min_segment_bars`) are almost
    always frame-quantization blips, so they are absorbed into the closer-BPM neighbour."""
    if not measures:
        return []

    # Pass 1: group by tolerance against the running median
    raw_segments = []
    current = [measures[0]]
    for m in measures[1:]:
        if abs(m.bpm - _median_bpm(current)) <= tolerance:
            current.append(m)
        else:
            raw_segments.append(current)
            current = [m]
    raw_segments.append(current)

    if len(raw_segments) <= 1:
        return [_finalize_segment(g) for g in raw_segments]

    # Pass 2: absorb short segments; pass 3: re-merge neighbours the running median split
    final = _merge_similar_adjacent(_absorb_short_segments(raw_segments, min_segment_bars), tolerance)
    return [_finalize_segment(g) for g in final]


def _absorb_short_segments(raw_segments, min_bars):
    groups = list(raw_segments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            if len(groups[i]) < min_bars and len(groups) > 1:
                best_idx = None
                best_diff = float("inf")
                short_median = _median_bpm(groups[i])
                for neighbor_idx in [i - 1, i + 1]:
                    if 0 <= neighbor_idx < len(groups):
                        diff = abs(short_median - _median_bpm(groups[neighbor_idx]))
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = neighbor_idx
                if best_idx is not None:
                    if best_idx < i:
                        groups[best_idx].extend(groups[i])
                    else:
                        groups[best_idx] = groups[i] + groups[best_idx]
                    groups.pop(i)
                    changed = True
                    continue
            i += 1
    return groups


def _merge_similar_adjacent(groups, tolerance):
    if len(groups) <= 1:
        return groups
    merged = [groups[0]]
    for group in groups[1:]:
        if abs(_median_bpm(merged[-1]) - _median_bpm(group)) <= tolerance:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def _finalize_segment(group: list[Measure]) -> TempoSegment:
    bpms = [m.bpm for m in group]
    return TempoSegment(
        start_measure=group[0].measure_num,
        end_measure=group[-1].measure_num,
        start_time=group[0].start,
        end_time=group[-1].end,
        bpm=round(float(np.median(bpms)), 2),
        bpm_std=round(float(np.std(bpms)), 3),
        measure_count=len(group),
    )


def quantization_report(intervals: np.ndarray, frame_duration: float) -> QuantizationReport:
    """How many discrete beat-interval values exist, and whether their step is the frame hop."""
    unique_intervals = np.unique(np.round(intervals, 6))
    diffs = np.diff(unique_intervals)
    return QuantizationReport(
        frame_duration_ms=round(frame_duration * 1000, 2),
        unique_interval_count=len(unique_intervals),
        unique_intervals=unique_intervals,
        interval_step_ms=round(float(np.median(diffs)) * 1000, 2) if len(diffs) > 0 else 0,
        matches_frame_hop=len(diffs) > 0 and abs(float(np.median(diffs)) - frame_duration) < 0.001,
    )


@dataclass
class GridDetection:
    """A grid plus how it was found: the downbeat vote and the intro trim."""
    grid: Beatgrid
    downbeat: DownbeatEstimate
    trim: TrimResult
    frame_duration: float
    """Seconds per hop frame: the beat tracker's time resolution."""


def detect_grid(y, sr, hop_length=512, beats_per_bar=4, span=8,
                tolerance=2.0, min_segment_bars=4) -> GridDetection:
    """The full librosa-dp grid as camelot builds it: track, trim the silent intro,
    find the downbeat phase, then measures, rolling BPM and tempo segments."""
    tempo, beat_times = track_beats(y, sr, hop_length)
    trim = trim_silent_beats(y, sr, beat_times, hop_length=hop_length)
    downbeat = estimate_downbeat_phase(y, sr, trim.beat_times)
    measures = group_beats_into_measures(trim.beat_times, beats_per_bar, phase=downbeat.phase)
    grid = Beatgrid(
        tempo=tempo,
        beat_times=trim.beat_times,
        measures=measures,
        segments=find_bpm_segments(measures, tolerance=tolerance, min_segment_bars=min_segment_bars),
        span=compute_span_bpm(trim.beat_times, span),
        downbeat_phase=downbeat.phase,
        bars_trimmed=trim.bars_trimmed,
        source=LEGACY_GRID_SOURCE,
    )
    return GridDetection(grid, downbeat, trim, hop_length / sr)


def build_beatgrid(y, sr, hop_length=512, beats_per_bar=4, span=8,
                   tolerance=2.0, min_segment_bars=4) -> Beatgrid:
    """`detect_grid` without the detection details."""
    return detect_grid(y, sr, hop_length, beats_per_bar, span, tolerance, min_segment_bars).grid
