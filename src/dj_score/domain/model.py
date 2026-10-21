"""Analysis results as dataclasses. The cache schema IS this module.

Field names follow the keys camelot's `AnalysisStore` has always written to
`analysis_cache.json`, so a cache written here stays readable by camelot and vice
versa. Fields that are absent from some caches (older files, or one event mode but
not the other) default to None. `adapters/cache_store.py` maps the file layout onto
these types; nothing here knows about files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEGACY_GRID_SOURCE = "librosa-dp"
"""Every camelot cache before dj-score came from `bpm_detect.detect_beats`
(librosa `beat_track` dynamic programming)."""


# -- beat grid ----------------------------------------------------------------

@dataclass
class Measure:
    measure_num: int
    beat_start_idx: int
    beat_count: int
    start: float
    end: float
    span_time: float
    bpm: float


@dataclass
class TempoSegment:
    start_measure: int
    end_measure: int
    start_time: float
    end_time: float
    bpm: float
    bpm_std: float
    measure_count: int


@dataclass
class TempoSpan:
    """Rolling BPM over `span` beats (camelot's `span_data`). With too few beats
    for one span, only the empty `times`/`bpms` are set."""
    times: np.ndarray
    bpms: np.ndarray
    span: int | None = None
    median: float | None = None
    std: float | None = None


@dataclass
class Beatgrid:
    tempo: float
    beat_times: np.ndarray
    measures: list[Measure]
    segments: list[TempoSegment]
    span: TempoSpan | None
    downbeat_phase: int = 0
    bars_trimmed: int = 0
    source: str = LEGACY_GRID_SOURCE
    """Provenance. A grid read back from Rekordbox is only a reference when
    Rekordbox analysed the track natively (2026-09-15 contamination finding)."""


# -- audio --------------------------------------------------------------------

@dataclass
class Waveform:
    y: np.ndarray | None
    """Full mono signal. Optional: golden fixtures and light consumers omit it."""
    y_ds: np.ndarray
    t_ds: np.ndarray


@dataclass
class Loudness:
    loudness_dbfs: float | None = None
    peak_dbfs: float | None = None
    safe_volume: float | None = None
    lufs: float | None = None


# -- key ----------------------------------------------------------------------

@dataclass
class KeyCandidate:
    score: float
    key: str
    camelot: str


@dataclass
class KeyEstimate:
    """Best Krumhansl-profile match for one chroma vector."""
    tonic_pc: int
    mode: str
    key_name: str
    camelot: str
    score: float
    confidence: float
    top_candidates: list[KeyCandidate]


@dataclass
class MeasureKey(KeyEstimate):
    measure_num: int
    start: float
    end: float
    bpm: float


@dataclass
class KeyPhrase(KeyEstimate):
    phrase_num: int
    start_measure: int
    end_measure: int
    start: float
    end: float
    measure_count: int


@dataclass
class KeySegment:
    start_measure: int
    end_measure: int
    start_time: float
    end_time: float
    camelot: str
    key_name: str
    confidence_mean: float
    confidence_max: float
    score_mean: float
    phrase_count: int
    camelot_counts: dict[str, int]


@dataclass
class DominantKey:
    key_name: str
    camelot: str
    coverage: float


@dataclass
class KeyChange:
    time: float
    bar: int
    from_camelot: str
    from_key: str
    to_camelot: str
    to_key: str


@dataclass
class KeySummary:
    dominant: DominantKey
    key_changes: list[KeyChange]
    segment_count: int
    phrase_count: int


@dataclass
class KeyAnalysis:
    measure_keys: list[MeasureKey | None]
    """None where a measure was too short to analyse."""
    phrases: list[KeyPhrase]
    key_segments: list[KeySegment]
    key_summary: KeySummary


# -- events -------------------------------------------------------------------

@dataclass
class Event:
    """One structural event. Feature mode fills `n_features`/`pass_`; stem mode
    fills `n_stems`/`stems_in`/`stems_out`."""
    bar: int
    time: float
    time_fmt: str
    type: str
    score: float
    description: str
    n_features: int | None = None
    pass_: int | None = field(default=None, metadata={"key": "pass"})
    n_stems: int | None = None
    stems_in: list[str] | None = None
    stems_out: list[str] | None = None
    source: str | None = None
    """"manual" for events entered in camelot's event dialog."""
    cue: str | None = None
    """Rekordbox cue override: "hot", "memory" or "auto" (the default when absent)."""


@dataclass
class EventPhrase:
    phrase_num: int
    start_bar: int
    end_bar: int
    start: float
    end: float
    bar_count: int | None = None
    onset_strength: float | None = None
    onset_percussive: float | None = None
    onset_harmonic: float | None = None
    rms: float | None = None
    centroid: float | None = None
    low_freq_ratio: float | None = None


@dataclass
class EventSummary:
    """With no events, only `mode` and `event_count` (0) are set."""
    mode: str
    event_count: int
    events_per_minute: float | None = None
    avg_score: float | None = None
    max_score: float | None = None
    phrase_count: int | None = None


@dataclass
class StemActivity:
    """Per-phrase stem energy and presence (stem mode only)."""
    stem_energies: dict[str, list[float]] | None = None
    stem_presence: dict[str, list[bool]] | None = None


@dataclass
class EventAnalysis:
    events: list[Event]
    event_phrases: list[EventPhrase]
    event_mode: str
    event_summary: EventSummary
    event_extra: StemActivity | None = None


# -- spectrum -----------------------------------------------------------------

@dataclass
class SpectralSummary:
    mean_spectrum: list[float]
    peak_spectrum: list[float]
    dominant_band_idx: int
    dominant_band_hz: float
    spectral_centroid_hz: float
    bandwidth_hz: list[float]
    energy_profile: str


@dataclass
class SpectrumAnalysis:
    bar_spectra: np.ndarray
    """(n_bars, n_bands) dB."""
    band_centers: np.ndarray
    band_edges: np.ndarray
    """(n_bands, 2) Hz."""
    summary: SpectralSummary | None


# -- the track ----------------------------------------------------------------

@dataclass
class TrackAnalysis:
    sr: int
    duration: float
    grid: Beatgrid
    waveform: Waveform
    loudness: Loudness
    key: KeyAnalysis | None = None
    events: EventAnalysis | None = None
    spectrum: SpectrumAnalysis | None = None
    audio_name: str | None = None
