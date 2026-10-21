"""Per-track analysis use cases: one function per UI action (camelot's `_run_*`) or CLI command.

`analyze_track` starts a `TrackAnalysis` from audio; the others compute one more
part from an existing analysis, which the caller attaches and saves. Defaults are
the values camelot's UI uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dj_score.domain import events, key, rhythm, spectral
from dj_score.domain.audio import make_waveform, measure_loudness
from dj_score.domain.beatgrid import GridDetection, detect_grid
from dj_score.domain.model import EventAnalysis, KeyAnalysis, SpectrumAnalysis, StemActivity, TrackAnalysis
from dj_score.ports import AudioLoader, LufsMeter, StemSeparator, StemSource


def detect_beats(audio_path: Path, loader: AudioLoader, sr=44100, hop_length=512, beats_per_bar=4,
                 span=8, tolerance=2.0, min_segment_bars=4) -> tuple[np.ndarray, GridDetection]:
    """Load audio and detect its grid, keeping the detection details. -> (signal, detection)"""
    y = loader.load(audio_path, sr)
    return y, detect_grid(y, sr, hop_length=hop_length, beats_per_bar=beats_per_bar, span=span,
                          tolerance=tolerance, min_segment_bars=min_segment_bars)


def analyze_track(audio_path: Path, loader: AudioLoader, meter: LufsMeter,
                  sr=44100, hop_length=512, beats_per_bar=4, span=8,
                  tolerance=2.0, min_segment_bars=4) -> TrackAnalysis:
    """Load audio, build the beat grid, measure loudness: camelot's "BPM" action."""
    y, detection = detect_beats(audio_path, loader, sr, hop_length, beats_per_bar, span,
                                tolerance, min_segment_bars)
    return TrackAnalysis(
        sr=sr,
        duration=len(y) / sr,
        grid=detection.grid,
        waveform=make_waveform(y, sr),
        loudness=measure_loudness(y, meter.integrated_lufs(y, sr)),
        audio_name=Path(audio_path).name,
    )


def analyze_key(a: TrackAnalysis, phrase_bars=4, min_segment_phrases=2, use_harmonic=True) -> KeyAnalysis:
    """Key per measure and per phrase, key segments, and the summary."""
    y, sr, measures = a.waveform.y, a.sr, a.grid.measures
    phrases = key.compute_phrase_chroma(y, sr, measures, phrase_bars=phrase_bars, use_harmonic=use_harmonic)
    segments = key.find_key_segments(phrases, min_segment_phrases=min_segment_phrases)
    return KeyAnalysis(
        measure_keys=key.compute_measure_chroma(y, sr, measures, use_harmonic=use_harmonic),
        phrases=phrases,
        key_segments=segments,
        key_summary=key.compute_summary(segments, phrases, a.duration),
    )


def analyze_spectrum(a: TrackAnalysis) -> SpectrumAnalysis:
    """Per-bar third-octave spectra and their summary."""
    s = spectral.compute_bar_spectra(a.waveform.y, a.sr, a.grid.measures)
    s.summary = spectral.compute_spectral_summary(s)
    return s


def analyze_events(a: TrackAnalysis, audio_path: Path, stems: StemSource, loader: AudioLoader,
                   phrase_bars=4, presence_threshold=0.15, context=2, min_score=0.15,
                   min_features=2, min_gap_phrases=2, separator: StemSeparator | None = None) -> EventAnalysis:
    """Stem mode when separated stems exist for the track (separating first if a
    `separator` is given and none do), else feature mode."""
    stem_paths = stems.find(audio_path)
    if not stem_paths and separator is not None:
        separator.separate(audio_path)
        stem_paths = stems.find(audio_path)
    if stem_paths:
        energies = {name: events.compute_stem_phrase_energy(loader.load(path, a.sr), a.sr, a.grid.measures,
                                                            phrase_bars=phrase_bars)
                    for name, path in stem_paths.items()}
        presence = events.compute_stem_presence(energies, presence_threshold=presence_threshold)
        phrases = events.stem_phrases(a.grid.measures, phrase_bars)
        found = events.detect_stem_events(presence, phrases, min_gap_phrases=min_gap_phrases)
        mode, extra = "stems", StemActivity(stem_energies=energies, stem_presence=presence)
    else:
        bar_features = events.compute_bar_features(a.waveform.y, a.sr, a.grid.measures)
        phrases = events.compute_phrase_features(bar_features, phrase_bars=phrase_bars)
        found = events.detect_feature_events(phrases, context=context, min_score=min_score,
                                             min_features=min_features, min_gap_phrases=min_gap_phrases,
                                             bar_features=bar_features)
        mode, extra = "features", None
    return EventAnalysis(
        events=found,
        event_phrases=phrases,
        event_mode=mode,
        event_summary=events.compute_summary(found, phrases, a.duration, mode=mode),
        event_extra=extra,
    )


@dataclass
class RhythmReport:
    summary: rhythm.RhythmSummary
    sections: list[rhythm.RhythmSection]
    per_bar: list[rhythm.BarRhythm]


def analyze_rhythm(audio_path: Path, loader: AudioLoader, bpm: float, first_downbeat=0.0, sr=44100) -> RhythmReport:
    """Kick pattern per bar on a rigid grid from `first_downbeat` at `bpm` (no beat tracking)."""
    y = loader.load(audio_path, sr)
    bars = rhythm.rigid_bars(len(y) / sr, bpm, first_downbeat)
    per_bar = rhythm.classify_rhythm(y, sr, bars, 60.0 / bpm)
    return RhythmReport(rhythm.summarize_rhythm(per_bar), rhythm.summarize_rhythm_sections(per_bar, bars), per_bar)


def key_window(audio_path: Path, loader: AudioLoader, sr=22050, start=90.0, duration=120.0,
               use_harmonic=True) -> key.WindowKey:
    """Key of one window: camelot_from_youtube's default mode."""
    return key.window_key(loader.load_window(audio_path, sr, start, duration), sr, start, duration, use_harmonic)


def key_consensus(audio_path: Path, loader: AudioLoader, starts: list[float], sr=22050, window=120.0,
                  use_harmonic=True, min_conf=0.05) -> tuple[key.Consensus, list[key.WindowKey]]:
    """Strict consensus over windows at `starts` -> (consensus, windows)."""
    windows = [key_window(audio_path, loader, sr, s, window, use_harmonic) for s in starts]
    return key.strict_consensus(windows, min_conf), windows


def bar_starts(bpm: float, bar_start: int, bar_step: int, num_windows: int, beats_per_bar=4.0) -> list[float]:
    """Window starts every `bar_step` bars from `bar_start`, in seconds."""
    return [round((bar_start + i * bar_step) * beats_per_bar * 60.0 / bpm, 3) for i in range(num_windows)]
