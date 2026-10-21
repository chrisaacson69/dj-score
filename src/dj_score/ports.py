"""Protocols the services depend on; adapters implement them structurally (PEP 544)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from dj_score.domain.model import TrackAnalysis


class AnalysisCache(Protocol):
    """Per-project persisted analysis. Implemented by `adapters.cache_store.CacheStore`."""

    def load(self, project_dir: Path) -> TrackAnalysis | None: ...

    def save(self, analysis: TrackAnalysis, project_dir: Path) -> None: ...


class AudioLoader(Protocol):
    """Decode an audio file to a mono float signal. `adapters.audio.LibrosaLoader`."""

    def load(self, path: Path, sr: int) -> np.ndarray: ...


class LufsMeter(Protocol):
    """Integrated loudness (LUFS), or None when no meter is available. `adapters.loudness`."""

    def integrated_lufs(self, y: np.ndarray, sr: int) -> float | None: ...


class StemSource(Protocol):
    """Existing separated stems for a track, as {stem name: wav path}, or None.
    `adapters.stems.StemFolder`."""

    def find(self, audio_path: Path) -> dict[str, Path] | None: ...


class RekordboxExporter(Protocol):
    """Writes analyses into a Rekordbox XML collection. `adapters.rekordbox_xml.RekordboxXml`."""

    def export_track(self, audio_path: Path, analysis: TrackAnalysis,
                     include_cues: bool = True, max_hot_cues: int = 8) -> int: ...

    def export_set(self, tracks: list[tuple[Path, TrackAnalysis]], playlist_name: str = "Camelot Set") -> int: ...
