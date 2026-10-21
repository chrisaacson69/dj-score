"""Two-track transition analysis: spectral collision + EQ advice, kick phase alignment.

The `--compare` flow of camelot `camelot_from_youtube.py`, with its reports
(`collision_report`, `collision_to_json` from `collision.py`, `alignment_to_json`
from `alignment.py`) converted as-is so the JSON CyborgDJ reads keeps its shape.

Known inconsistency, kept from camelot: a bar range "200-260" means measure numbers
200-260 (1-based) to the collision half, but bar index 200 (0-based) on a rigid grid
from `first_beat` to the alignment half, so the two halves look one bar apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dj_score.domain.codec import to_plain as _encode
from dj_score.domain import alignment, collision
from dj_score.domain.alignment import AlignmentResult
from dj_score.domain.collision import CollisionResult, EqRecommendation, GapResult
from dj_score.domain.model import TrackAnalysis
from dj_score.ports import AudioLoader


@dataclass
class PairCollision:
    collision: CollisionResult
    gaps: GapResult
    eq: EqRecommendation
    name_a: str
    name_b: str
    bars_a: str
    bars_b: str


def compare_spectra(a: TrackAnalysis, b: TrackAnalysis, bars_a: str, bars_b: str,
                    name_a: str, name_b: str, min_gap_db=10.0) -> PairCollision:
    """Loudness-normalized collision, gaps and EQ advice for the overlap regions."""
    if a.spectrum is None or b.spectrum is None:
        raise ValueError("Both tracks need spectral analysis (run analyze_spectrum).")
    col = collision.compute_collision(a.spectrum, b.spectrum, bars_a, bars_b, a.loudness, b.loudness)
    gaps = collision.find_spectral_gaps(a.spectrum, b.spectrum, bars_a, bars_b, a.loudness, b.loudness,
                                        min_gap_db=min_gap_db)
    return PairCollision(col, gaps, collision.generate_eq_recommendations(gaps, col, name_a),
                         name_a, name_b, bars_a, bars_b)


def align_pair(audio_a: Path, audio_b: Path, loader: AudioLoader, sr: int, bpm: float,
               first_beat_a: int, first_beat_b: int, bars_a: str, bars_b: str,
               max_offset_ms=100.0) -> AlignmentResult:
    """Multi-probe kick alignment, one 4-bar probe per 4 bars of overlap."""
    start_a, end_a = (int(x) for x in bars_a.split("-"))
    start_b = int(bars_b.split("-")[0])
    return alignment.multi_probe_alignment(
        loader.load(audio_a, sr), loader.load(audio_b, sr), sr, bpm,
        first_beat_a, first_beat_b, start_a, start_b,
        n_probes=max(1, (end_a - start_a) // 4), probe_length_bars=4, max_offset_ms=max_offset_ms,
    )


# -- reports ----------------------------------------------------------------------

def collision_report(p: PairCollision) -> str:
    """Human-readable collision summary."""
    col, gaps, eq_recs, name_a, name_b = p.collision, p.gaps, p.eq, p.name_a, p.name_b
    centers = col.band_centers
    lines = [f"Spectral Collision Report: {name_a} vs {name_b}", "=" * 60]

    lines.append(f"\nTotal collision score: {float(col.collision.sum()):.0f}")
    lines.append(f"Hotspot bars (top 25%): {len(col.hotspot_bars)} of {col.collision.shape[0]}")

    band_totals = col.total_per_band
    lines.append("\nTop 5 colliding frequency bands:")
    for idx in np.argsort(band_totals)[::-1][:5]:
        lines.append(f"  {centers[idx]:>8.0f} Hz  score={band_totals[idx]:.0f}")

    bar_totals = col.total_per_bar
    lines.append("\nTop 5 colliding bars (0-indexed within region):")
    for idx in np.argsort(bar_totals)[::-1][:5]:
        lines.append(f"  Bar {idx:>3d}  score={bar_totals[idx]:.0f}")

    lines.append(f"\nSpectral gaps (>= threshold): {len(gaps.gaps)} bands")
    for g in gaps.gaps:
        dom = name_a if g.a_dominates else name_b
        lines.append(f"  {g.frequency_hz:>8.0f} Hz  {dom} dominates by "
                     f"{g.mean_gap_db:.1f} dB ({g.consistency:.0%} consistent)")

    if gaps.suggested_crossovers:
        lines.append("\nSuggested crossover frequencies:")
        for x in gaps.suggested_crossovers:
            below = name_a if x.below_dominant == "A" else name_b
            above = name_a if x.above_dominant == "A" else name_b
            lines.append(f"  {x.frequency_hz:.0f} Hz  (below: {below}, above: {above}, "
                         f"confidence: {x.confidence:.0%})")
    else:
        lines.append("\nNo clear crossover suggestions found.")

    if eq_recs:
        lines.append("\nEQ Recommendation:")
        lines.append(f"  Strategy: {eq_recs.strategy}")
        if eq_recs.bass_handoff_hz:
            lines.append(f"  Bass handoff: {eq_recs.bass_handoff_hz} Hz")
        lines.append(f"  {eq_recs.rationale}")

    return "\n".join(lines)


def collision_to_json(p: PairCollision) -> dict:
    """camelot's collision JSON, field for field."""
    col, centers = p.collision, p.collision.band_centers
    band_totals = col.total_per_band
    return {
        "track_a": {"name": p.name_a, "bars": p.bars_a},
        "track_b": {"name": p.name_b, "bars": p.bars_b},
        "collision_score": round(float(col.collision.sum()), 1),
        "n_bars": int(col.collision.shape[0]),
        "hotspot_bars": col.hotspot_bars,
        "hotspot_bands_hz": [float(centers[i]) for i in col.hotspot_bands],
        "top_colliding_bands": [{"frequency_hz": float(centers[i]), "score": round(float(band_totals[i]), 1)}
                                for i in np.argsort(band_totals)[::-1][:5]],
        "per_bar_collision": [round(float(v), 1) for v in col.total_per_bar],
        "spectral_gaps": _encode(p.gaps.gaps),
        "suggested_crossovers": _encode(p.gaps.suggested_crossovers),
        "eq_recommendation": {**_encode(p.eq), "bass_handoff_hz": p.eq.bass_handoff_hz,
                              "incoming_eq": p.eq.incoming_eq},
    }


def alignment_to_json(result: AlignmentResult, name_a: str, name_b: str,
                      first_beat_a=None, first_beat_b=None) -> dict:
    """camelot's phase-alignment JSON, field for field."""
    output = {
        "track_a": {"name": name_a},
        "track_b": {"name": name_b},
        "alignment": {"offset_samples": result.offset_samples, "offset_ms": result.offset_ms,
                      "consistency": result.consistency},
        "recommendation": _encode(result.recommendation),
    }
    if first_beat_a is not None:
        output["track_a"]["first_beat_sample"] = first_beat_a
    if first_beat_b is not None:
        output["track_b"]["first_beat_sample"] = first_beat_b
        output["recommendation"]["new_first_beat_sample"] = first_beat_b + result.offset_samples
    output["probes"] = _encode(result.probes)
    return output
