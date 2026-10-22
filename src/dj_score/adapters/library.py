"""Where a track's derived files live.

A music library is a folder of original audio with an `analysis/` folder in it: a track's
project (analysis cache, detail files) is `analysis/<name>/` and its stems are
`analysis/<name>_stems/`, so the originals folder holds only originals. Audio anywhere
else keeps the older layout, with both beside the audio. The layout is read from disk,
so no tool needs configuring.
"""

from __future__ import annotations

from pathlib import Path

from dj_score.adapters.cache_store import JSON_NAME

ANALYSIS = "analysis"


def is_library(folder: Path) -> bool:
    return (Path(folder) / ANALYSIS).is_dir()


def data_dir(audio_path: Path) -> Path:
    """The folder holding this track's derived files."""
    folder = Path(audio_path).parent
    return folder / ANALYSIS if is_library(folder) else folder


def project_dir(audio_path: Path) -> Path:
    return data_dir(audio_path) / Path(audio_path).stem


def stems_dir(audio_path: Path) -> Path:
    return data_dir(audio_path) / f"{Path(audio_path).stem}_stems"


def projects(root: Path) -> list[Path]:
    """Every analysed track in a library, as project folders."""
    return sorted(p.parent for p in (Path(root) / ANALYSIS).glob(f"*/{JSON_NAME}"))
