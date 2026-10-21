"""Per-track analysis use cases: one function per UI action (camelot's `_run_*`).

`analyze_track` starts a `TrackAnalysis` from audio; the others compute one more
part from an existing analysis, which the caller attaches and saves.
"""

from __future__ import annotations

from pathlib import Path

from dj_score.domain import key, spectral
from dj_score.domain.audio import make_waveform, measure_loudness
from dj_score.domain.beatgrid import build_beatgrid
from dj_score.domain.model import KeyAnalysis, SpectrumAnalysis, TrackAnalysis
from dj_score.ports import AudioLoader, LufsMeter


def analyze_track(audio_path: Path, loader: AudioLoader, meter: LufsMeter,
                  sr=44100, hop_length=512, beats_per_bar=4) -> TrackAnalysis:
    """Load audio, build the beat grid, measure loudness: camelot's "BPM" action."""
    y = loader.load(audio_path, sr)
    return TrackAnalysis(
        sr=sr,
        duration=len(y) / sr,
        grid=build_beatgrid(y, sr, hop_length=hop_length, beats_per_bar=beats_per_bar),
        waveform=make_waveform(y, sr),
        loudness=measure_loudness(y, meter.integrated_lufs(y, sr)),
        audio_name=Path(audio_path).name,
    )


def analyze_key(a: TrackAnalysis, phrase_bars=4) -> KeyAnalysis:
    """Key per measure and per phrase, key segments, and the summary."""
    y, sr, measures = a.waveform.y, a.sr, a.grid.measures
    phrases = key.compute_phrase_chroma(y, sr, measures, phrase_bars=phrase_bars, use_harmonic=True)
    segments = key.find_key_segments(phrases, min_segment_phrases=2)
    return KeyAnalysis(
        measure_keys=key.compute_measure_chroma(y, sr, measures, use_harmonic=True),
        phrases=phrases,
        key_segments=segments,
        key_summary=key.compute_summary(segments, phrases, a.duration),
    )


def analyze_spectrum(a: TrackAnalysis) -> SpectrumAnalysis:
    """Per-bar third-octave spectra and their summary."""
    s = spectral.compute_bar_spectra(a.waveform.y, a.sr, a.grid.measures)
    s.summary = spectral.compute_spectral_summary(s)
    return s
