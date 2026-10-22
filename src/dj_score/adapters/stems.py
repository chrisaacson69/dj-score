"""Separated stems on disk: `<audio stem>_stems/{drums,bass,vocals,other}.wav`."""

from __future__ import annotations

from pathlib import Path

from dj_score.adapters import library
from dj_score.domain.events import STEM_NAMES


class StemFolder:
    """`StemSource` over `<audio stem>_stems/`: an explicit `stems_dir` if given and
    populated, else the track's stems folder (`library.stems_dir`)."""

    def __init__(self, stems_dir: Path | None = None):
        self.stems_dir = Path(stems_dir) if stems_dir else None

    def find(self, audio_path: Path) -> dict[str, Path] | None:
        audio_path = Path(audio_path)
        for d in (self.stems_dir, library.stems_dir(audio_path)):
            if d is not None and d.exists():
                stems = {name: d / f"{name}.wav" for name in STEM_NAMES if (d / f"{name}.wav").exists()}
                if stems:
                    return stems
        return None
