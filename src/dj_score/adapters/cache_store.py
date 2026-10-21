"""The ONE reader/writer of a project's `analysis_cache.{json,npz}`.

Replaces camelot's `AnalysisStore.save/load`, `rekordbox_export.load_store_data` and
`bench/beat_bench.dump_camelot`. The file layout is camelot's, unchanged, so either
side reads the other's caches: large arrays in the `.npz`, everything else in the
`.json`. One addition, `beatgrid_source`, which camelot's loader ignores; caches
without it are legacy `librosa-dp` grids.

Decoding is strict: an unknown key raises instead of being dropped, so a schema
variant shows up as a failing golden test rather than as silently lost data.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from dj_score.domain.codec import CodecError, from_plain, to_plain as _encode
from dj_score.domain.model import (
    LEGACY_GRID_SOURCE,
    Beatgrid,
    EventAnalysis,
    KeyAnalysis,
    Loudness,
    Measure,
    SpectralSummary,
    SpectrumAnalysis,
    TempoSegment,
    TempoSpan,
    TrackAnalysis,
    Waveform,
)

JSON_NAME = "analysis_cache.json"
NPZ_NAME = "analysis_cache.npz"

_TOP_LEVEL_KEYS = {
    "sr", "tempo", "duration", "downbeat_phase", "bars_trimmed",
    "loudness_dbfs", "peak_dbfs", "safe_volume", "lufs",
    "measures", "segments", "span_data", "key", "events_block", "spectrum",
    "_audio_name", "beatgrid_source",
}


class CacheSchemaError(CodecError):
    """The cache holds a key or shape that `domain/model.py` does not describe."""


def _decode(tp, value):
    try:
        return from_plain(tp, value)
    except CacheSchemaError:
        raise
    except CodecError as e:
        raise CacheSchemaError(str(e)) from e


class CacheStore:
    """`AnalysisCache` port over a project directory's json+npz pair."""

    def load(self, project_dir: Path) -> TrackAnalysis | None:
        return load(project_dir)

    def save(self, analysis: TrackAnalysis, project_dir: Path) -> None:
        save(analysis, project_dir)


def load(project_dir) -> TrackAnalysis | None:
    """Read a project's cache. None if either file is missing."""
    d = Path(project_dir)
    json_path, npz_path = d / JSON_NAME, d / NPZ_NAME
    if not json_path.exists() or not npz_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return from_cache(data, arrays)


def save(analysis: TrackAnalysis, project_dir) -> None:
    d = Path(project_dir)
    data, arrays = to_cache(analysis)
    np.savez_compressed(d / NPZ_NAME, **arrays)
    with open(d / JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -- file layout <-> model ----------------------------------------------------

def from_cache(data: dict, arrays: dict[str, np.ndarray]) -> TrackAnalysis:
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        raise CacheSchemaError(f"unknown top-level keys: {sorted(unknown)}")

    sd = data.get("span_data")
    grid = Beatgrid(
        tempo=data["tempo"],
        beat_times=arrays["beat_times"],
        measures=_decode(list[Measure], data["measures"]),
        segments=_decode(list[TempoSegment], data["segments"]),
        span=_decode(TempoSpan, sd) if sd else None,
        downbeat_phase=data.get("downbeat_phase", 0),
        bars_trimmed=data.get("bars_trimmed", 0),
        source=data.get("beatgrid_source", LEGACY_GRID_SOURCE),
    )

    spectrum = None
    if "bar_spectra" in arrays:
        block = data.get("spectrum") or {}
        summary = block.get("spectral_summary")
        spectrum = SpectrumAnalysis(
            bar_spectra=arrays["bar_spectra"],
            band_centers=arrays["band_centers"],
            band_edges=arrays["band_edges"],
            summary=_decode(SpectralSummary, summary) if summary else None,
        )

    key, events = data.get("key"), data.get("events_block")
    return TrackAnalysis(
        sr=data["sr"],
        duration=data["duration"],
        grid=grid,
        waveform=Waveform(y=arrays.get("y"), y_ds=arrays["y_ds"], t_ds=arrays["t_ds"]),
        loudness=Loudness(**{f.name: data.get(f.name) for f in dataclasses.fields(Loudness)}),
        key=_decode(KeyAnalysis, key) if key else None,
        events=_decode(EventAnalysis, events) if events else None,
        spectrum=spectrum,
        audio_name=data.get("_audio_name"),
    )


def to_cache(a: TrackAnalysis) -> tuple[dict, dict[str, np.ndarray]]:
    """Inverse of `from_cache`. Key order follows camelot's `AnalysisStore.save`."""
    g = a.grid
    arrays = {"beat_times": np.asarray(g.beat_times), "y_ds": a.waveform.y_ds, "t_ds": a.waveform.t_ds}
    if a.waveform.y is not None:
        arrays["y"] = a.waveform.y
    if a.spectrum is not None:
        arrays["bar_spectra"] = a.spectrum.bar_spectra
        arrays["band_centers"] = a.spectrum.band_centers
        arrays["band_edges"] = a.spectrum.band_edges

    data = {
        "sr": a.sr,
        "tempo": g.tempo,
        "duration": a.duration,
        "downbeat_phase": g.downbeat_phase,
        "bars_trimmed": g.bars_trimmed,
        **dataclasses.asdict(a.loudness),
        "measures": _encode(g.measures),
        "segments": _encode(g.segments),
        "span_data": _encode(g.span),
    }
    if a.key is not None:
        data["key"] = _encode(a.key)
    if a.events is not None:
        data["events_block"] = _encode(a.events)
    if a.spectrum is not None and a.spectrum.summary is not None:
        data["spectrum"] = {"spectral_summary": _encode(a.spectrum.summary)}
    data["_audio_name"] = a.audio_name
    data["beatgrid_source"] = g.source
    return data, arrays
