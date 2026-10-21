"""Composition root: the one place concrete adapters are wired into services."""

from __future__ import annotations

from pathlib import Path

from dj_score.adapters.audio import LibrosaLoader
from dj_score.adapters.cache_store import CacheStore
from dj_score.adapters.loudness import PyloudnormMeter
from dj_score.adapters.stems import StemFolder


def loader():
    return LibrosaLoader()


def meter():
    return PyloudnormMeter()


def cache():
    return CacheStore()


def stem_source(stems_dir: Path | None = None):
    return StemFolder(stems_dir)


def separator(model_name="htdemucs", device=None, segment_length=None, shifts=1,
              mp3_output=False, float_output=False):
    from dj_score.adapters.demucs import DemucsSeparator
    return DemucsSeparator(model_name=model_name, device=device, segment_length=segment_length,
                           shifts=shifts, mp3_output=mp3_output, float_output=float_output)


def separation_reader():
    from dj_score.adapters.demucs import SeparationFolder
    return SeparationFolder()


def rekordbox(path: Path):
    from dj_score.adapters.rekordbox_xml import RekordboxXml
    return RekordboxXml(path)
