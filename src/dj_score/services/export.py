"""Export use cases: send analyses to Rekordbox."""

from __future__ import annotations

from pathlib import Path

from dj_score.domain.model import TrackAnalysis
from dj_score.ports import RekordboxExporter


def export_to_rekordbox(audio_path: Path, analysis: TrackAnalysis, exporter: RekordboxExporter,
                        include_cues=True, max_hot_cues=8) -> int:
    """BPM, key, tempo grid and cues for one track; returns the collection's track count."""
    return exporter.export_track(audio_path, analysis, include_cues=include_cues, max_hot_cues=max_hot_cues)


def export_set_to_rekordbox(tracks: list[tuple[Path, TrackAnalysis]], exporter: RekordboxExporter,
                            playlist_name="Camelot Set") -> int:
    """Several tracks plus a playlist of them, in order."""
    return exporter.export_set(tracks, playlist_name=playlist_name)
