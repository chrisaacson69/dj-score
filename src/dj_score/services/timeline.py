"""Key timeline use case (camelot_from_youtube's `--timeline`), composed from dj-score parts.

Windows of `bars_per_segment` bars on the dj-score grid; per window, the key
(`key.window_key`) and an energy profile; windows merged into regions with a summary
and mix points. Structural events are dj-score's (stem or feature mode), and rhythm is
classified per grid bar. camelot's own BPM/downbeat and energy/micro event detectors,
which `--timeline` used, are retired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dj_score.domain import key, rhythm, timeline
from dj_score.domain.model import TrackAnalysis
from dj_score.ports import AudioLoader, LufsMeter, StemSource
from dj_score.services import analyze

SEGMENT_METHOD = "dj-score beat grid"


@dataclass
class TimelineEvent:
    time: float
    bar: int
    label: str
    """The dj-score event type, shown as-is by the timeline views."""
    types: list[str]
    score: float
    description: str


@dataclass
class TimelineRhythm:
    summary: rhythm.RhythmSummary
    sections: list[rhythm.RhythmSection]
    per_bar: list[rhythm.BarRhythm]


@dataclass
class TimelineGrid:
    source: str
    beat_count: int
    first_downbeat: float | None
    downbeat_phase: int
    bars_trimmed: int


@dataclass
class TimelineReport:
    duration: float
    bpm: float
    bars_per_segment: int
    total_segments_analyzed: int
    segment_method: str
    summary: timeline.TimelineSummary
    timeline: list[timeline.TimelineRegion]
    energy_events: list[TimelineEvent]
    event_mode: str
    raw_segments: list[timeline.TimelineSegment]
    rhythm: TimelineRhythm | None
    beat_grid: TimelineGrid


def analyze_timeline(audio_path: Path, loader: AudioLoader, meter: LufsMeter, stems: StemSource,
                     sr=22050, bars_per_segment=8, use_harmonic=True, min_confidence=0.1,
                     min_stable_segments=2, classify_rhythm=True,
                     track: TrackAnalysis | None = None) -> TimelineReport:
    """Grid and events at the UI's 44.1 kHz (or from `track`); window keys and energy at `sr`."""
    a = track or analyze.analyze_track(audio_path, loader, meter)
    if a.events is None:
        a.events = analyze.analyze_events(a, audio_path, stems, loader)

    y = loader.load(audio_path, sr)
    duration = len(y) / sr
    segments = []
    for w in timeline.bar_windows(a.grid.measures, bars_per_segment):
        clip = y[int(w.start * sr):int(w.end * sr)]
        k = key.window_key(clip, sr, w.start, w.end - w.start, use_harmonic)
        seg = timeline.TimelineSegment(
            start=k.start, duration=k.duration, end=w.end, bar_start=w.bar_start, bar_end=w.bar_end,
            key=k.key, camelot=k.camelot, confidence=k.confidence, score=k.score,
            top_candidates=k.top_candidates, error=k.error)
        if len(clip) > sr:
            seg.energy = timeline.energy_profile(clip, sr)
        segments.append(seg)

    regions = timeline.merge_timeline_segments(segments, min_confidence=min_confidence,
                                               min_run_length=min_stable_segments)

    bar_rhythm = None
    if classify_rhythm and len(a.grid.beat_times) >= 8:
        per_bar = rhythm.classify_rhythm(a.waveform.y, a.sr, a.grid.measures, 60.0 / a.grid.tempo)
        bar_rhythm = TimelineRhythm(rhythm.summarize_rhythm(per_bar),
                                    rhythm.summarize_rhythm_sections(per_bar, a.grid.measures), per_bar)

    first_bar = next((m for m in a.grid.measures if m.measure_num >= 1), None)
    return TimelineReport(
        duration=round(duration, 2),
        bpm=a.grid.tempo,
        bars_per_segment=bars_per_segment,
        total_segments_analyzed=len(segments),
        segment_method=SEGMENT_METHOD,
        summary=timeline.compute_timeline_summary(regions, duration),
        timeline=regions,
        energy_events=[TimelineEvent(e.time, e.bar, e.type, [e.type], e.score, e.description)
                       for e in a.events.events],
        event_mode=a.events.event_mode,
        raw_segments=segments,
        rhythm=bar_rhythm,
        beat_grid=TimelineGrid(a.grid.source, len(a.grid.beat_times), first_bar.start if first_bar else None,
                               a.grid.downbeat_phase, a.grid.bars_trimmed),
    )
