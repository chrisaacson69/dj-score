"""Two-track spectral collision in a transition zone, spectral gaps, EQ advice.

Converted from camelot `collision.py` (itself ported from CyborgDJ's
`spectral_collision.py`, the fork step 7 retires). Loading moved to
`adapters/cache_store`; the text/JSON reports move with the `compare_pair`
service; the figure lives in `adapters/plots`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dj_score.domain.model import Loudness, SpectrumAnalysis
from dj_score.domain.spectral import NOISE_FLOOR_DB


@dataclass
class CollisionResult:
    collision: np.ndarray
    """(n_bars, n_bands): min(dB_A, dB_B) above the noise floor."""
    total_per_bar: np.ndarray
    total_per_band: np.ndarray
    hotspot_bars: list[int]
    """Bars above the 75th percentile of total collision."""
    hotspot_bands: list[int]
    bar_spectra_a: np.ndarray
    bar_spectra_b: np.ndarray
    band_centers: np.ndarray
    band_edges: np.ndarray


@dataclass
class SpectralGap:
    frequency_hz: float
    band_idx: int
    a_dominates: bool
    mean_gap_db: float
    consistency: float
    """Fraction of bars where the dominant track actually leads."""


@dataclass
class Crossover:
    frequency_hz: float
    below_dominant: str
    above_dominant: str
    confidence: float
    gap_below_db: float
    gap_above_db: float


@dataclass
class GapResult:
    gaps: list[SpectralGap]
    suggested_crossovers: list[Crossover]


@dataclass
class EqRecommendation:
    strategy: str
    """"bass_swap" or "simple_fade"."""
    bass_handoff_hz: float | None
    incoming_eq: dict | None
    """CyborgDJ `eq_3band` spec, verbatim: {"low": [from, to], "mid": ..., "high": ..., "low_freq": hz}."""
    rationale: str


def parse_bar_range(bar_spec: str, n_bars: int) -> tuple[int, int]:
    """'200-260' (1-based measure numbers) -> (start, end) as a 0-based, end-exclusive slice."""
    parts = bar_spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bar range must be 'start-end', got '{bar_spec}'")
    start = int(parts[0]) - 1
    end = int(parts[1])
    if start < 0 or end > n_bars or start >= end:
        raise ValueError(f"Bar range {bar_spec} out of bounds (track has {n_bars} bars)")
    return start, end


def _normalize_spectra(spectra, loudness: Loudness | None):
    """Shift spectra by -LUFS (else -RMS dBFS) so both tracks share a reference."""
    ref = None
    if loudness is not None:
        ref = loudness.lufs if loudness.lufs is not None else loudness.loudness_dbfs
    if ref is None:
        return spectra
    return np.clip(spectra - ref, NOISE_FLOOR_DB, None)


def _paired_regions(a: SpectrumAnalysis, b: SpectrumAnalysis, bars_a, bars_b, loudness_a, loudness_b):
    start_a, end_a = parse_bar_range(bars_a, a.bar_spectra.shape[0])
    start_b, end_b = parse_bar_range(bars_b, b.bar_spectra.shape[0])
    region_a = a.bar_spectra[start_a:end_a].copy()
    region_b = b.bar_spectra[start_b:end_b].copy()

    if region_a.shape[0] != region_b.shape[0]:
        raise ValueError(
            f"Bar ranges must be equal length: A has {region_a.shape[0]} bars, "
            f"B has {region_b.shape[0]} bars")
    if not np.allclose(a.band_centers, b.band_centers):
        raise ValueError("Tracks have different band_centers")

    return _normalize_spectra(region_a, loudness_a), _normalize_spectra(region_b, loudness_b)


def compute_collision(a: SpectrumAnalysis, b: SpectrumAnalysis, bars_a: str, bars_b: str,
                      loudness_a: Loudness | None = None, loudness_b: Loudness | None = None) -> CollisionResult:
    """Where both tracks carry energy in the same bars and bands."""
    region_a, region_b = _paired_regions(a, b, bars_a, bars_b, loudness_a, loudness_b)

    collision = np.maximum(np.minimum(region_a, region_b) - NOISE_FLOOR_DB, 0.0)
    total_per_bar = collision.sum(axis=1)
    total_per_band = collision.sum(axis=0)

    bar_threshold = np.percentile(total_per_bar, 75) if len(total_per_bar) > 4 else 0
    band_threshold = np.percentile(total_per_band, 75) if len(total_per_band) > 4 else 0

    return CollisionResult(
        collision=collision,
        total_per_bar=total_per_bar,
        total_per_band=total_per_band,
        hotspot_bars=[int(i) for i in np.where(total_per_bar > bar_threshold)[0]],
        hotspot_bands=[int(i) for i in np.where(total_per_band > band_threshold)[0]],
        bar_spectra_a=region_a,
        bar_spectra_b=region_b,
        band_centers=a.band_centers,
        band_edges=a.band_edges,
    )


def find_spectral_gaps(a: SpectrumAnalysis, b: SpectrumAnalysis, bars_a: str, bars_b: str,
                       loudness_a: Loudness | None = None, loudness_b: Loudness | None = None,
                       min_gap_db=10.0) -> GapResult:
    """Bands where one track leads by >= `min_gap_db`, and crossovers where the leader flips."""
    region_a, region_b = _paired_regions(a, b, bars_a, bars_b, loudness_a, loudness_b)
    centers = a.band_centers

    gaps = []
    for i in range(region_a.shape[1]):
        diff = region_a[:, i] - region_b[:, i]
        mean_diff = float(np.mean(diff))
        abs_gap = abs(mean_diff)
        if abs_gap >= min_gap_db:
            a_dom = mean_diff > 0
            consistency = float(np.mean(diff > 0)) if a_dom else float(np.mean(diff < 0))
            gaps.append(SpectralGap(
                frequency_hz=float(centers[i]),
                band_idx=i,
                a_dominates=a_dom,
                mean_gap_db=round(abs_gap, 1),
                consistency=round(consistency, 2),
            ))

    crossovers = [
        Crossover(
            frequency_hz=round(float(np.sqrt(g1.frequency_hz * g2.frequency_hz)), 0),
            below_dominant="A" if g1.a_dominates else "B",
            above_dominant="A" if g2.a_dominates else "B",
            confidence=round(min(g1.consistency, g2.consistency), 2),
            gap_below_db=g1.mean_gap_db,
            gap_above_db=g2.mean_gap_db,
        )
        for g1, g2 in zip(gaps, gaps[1:])
        if g1.a_dominates != g2.a_dominates
    ]
    return GapResult(gaps=gaps, suggested_crossovers=crossovers)


def generate_eq_recommendations(gaps: GapResult, collision: CollisionResult, name_a: str) -> EqRecommendation:
    """Bass swap when bass dominates the collision, else a simple equal-power fade.

    `name_a` is the outgoing track, quoted in the rationale.
    """
    # First confident bass-range crossover (<= 400 Hz)
    bass_xover = next((x for x in gaps.suggested_crossovers
                       if x.frequency_hz <= 400 and x.confidence >= 0.5), None)

    bass_mask = collision.band_centers < 200
    band_totals = collision.total_per_band
    bass_collision = float(band_totals[bass_mask].sum()) if bass_mask.any() else 0
    total_collision = float(band_totals.sum())
    bass_fraction = bass_collision / total_collision if total_collision > 0 else 0

    def bass_swap_eq(hz):
        return {"low": [0.0, 1.0], "mid": [1.0, 1.0], "high": [1.0, 1.0], "low_freq": int(hz)}

    if bass_xover and bass_fraction > 0.15:
        hz = bass_xover.frequency_hz
        return EqRecommendation(
            strategy="bass_swap",
            bass_handoff_hz=hz,
            incoming_eq=bass_swap_eq(hz),
            rationale=(
                f"Outgoing track ({name_a}) dominates below {hz:.0f} Hz. "
                f"Bass swap recommended: kill incoming bass initially, ramp to full. "
                f"Bass accounts for {bass_fraction:.0%} of total collision."
            ),
        )
    if total_collision > 0 and bass_fraction > 0.3:
        return EqRecommendation(
            strategy="bass_swap",
            bass_handoff_hz=200,
            incoming_eq=bass_swap_eq(200),
            rationale=(
                f"Significant bass collision ({bass_fraction:.0%} of total) "
                f"but no clear crossover point. Defaulting to 200 Hz bass swap."
            ),
        )
    return EqRecommendation(
        strategy="simple_fade",
        bass_handoff_hz=None,
        incoming_eq=None,
        rationale=(
            f"Low bass collision ({bass_fraction:.0%} of total). "
            f"Simple equal-power crossfade should work."
        ),
    )
