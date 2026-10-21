"""Separated stems on disk: `<audio stem>_stems/{drums,bass,vocals,other}.wav`."""

from __future__ import annotations

from pathlib import Path

from dj_score.domain.events import STEM_NAMES


class StemFolder:
    """`StemSource` over the folder layout camelot's `event_detect` uses: an explicit
    `stems_dir` if given and populated, else `<audio>_stems/` beside the audio."""

    def __init__(self, stems_dir: Path | None = None):
        self.stems_dir = Path(stems_dir) if stems_dir else None

    def find(self, audio_path: Path) -> dict[str, Path] | None:
        audio_path = Path(audio_path)
        for d in (self.stems_dir, audio_path.parent / f"{audio_path.stem}_stems"):
            if d is not None and d.exists():
                stems = {name: d / f"{name}.wav" for name in STEM_NAMES if (d / f"{name}.wav").exists()}
                if stems:
                    return stems
        return None
