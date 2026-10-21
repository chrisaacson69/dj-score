"""Stem use cases: separate a track, and inspect a separation (G7 pilot checks)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dj_score.domain import stems
from dj_score.domain.stems import Resummation, VocalProbe
from dj_score.ports import AudioLoader, SeparationReader, StemSeparator

PROBE_SR = 22050


@dataclass
class StemInspection:
    track: str
    model: str
    resummation: Resummation
    probes: dict[str, VocalProbe]
    """Per stem file. Unvalidated (see `domain.stems.vocal_probe`)."""


def separate_track(audio_path: Path, separator: StemSeparator, output_dir: Path | None = None) -> dict[str, Path]:
    return separator.separate(audio_path, output_dir)


def inspect_stems(stems_dir: Path, reader: SeparationReader, loader: AudioLoader, probe=True) -> StemInspection:
    """Check 1 (re-summation residual) and the vocal-vs-synth probe for each stem."""
    sep = reader.read(stems_dir)
    probes = {}
    if probe:
        for p in sorted(Path(stems_dir).glob("*.wav")):
            probes[p.stem] = stems.vocal_probe(loader.load(p, PROBE_SR), PROBE_SR)
    return StemInspection(
        track=sep.input.stem,
        model=sep.model,
        resummation=stems.resummation(sep.mix, sep.stems, sep.sample_rate),
        probes=probes,
    )
