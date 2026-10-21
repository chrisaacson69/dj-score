"""Kick-envelope cross-correlation: the sample offset that phase-aligns two tracks.

Converted from camelot `alignment.py` (itself ported from CyborgDJ's
`align_to_master.py`). The JSON report moves with the `compare_pair` service.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from dj_score.domain.kick import kick_envelope


@dataclass
class KickOffset:
    offset_samples: int
    """Positive = B should shift later relative to A."""
    offset_samples_refined: float
    offset_ms: float
    correlation_peak: float
    """0-1, higher = more confident."""
    probe_bars_a: str | None = None
    probe_bars_b: str | None = None


@dataclass
class AlignmentRecommendation:
    suggested_rb_offset: int
    rationale: str
    apply_to: str | None = None
    offset_ms: float | None = None


@dataclass
class AlignmentResult:
    offset_samples: int
    """Median over probes."""
    offset_ms: float
    consistency: float
    """Fraction of probes within 10 ms of the median."""
    probes: list[KickOffset]
    recommendation: AlignmentRecommendation


def _envelopes(seg_a, seg_b, sr):
    env_a, env_b = kick_envelope(seg_a, sr), kick_envelope(seg_b, sr)
    n = min(len(env_a), len(env_b))
    return env_a[:n], env_b[:n]


def _xcorr_region(env_a, env_b, max_off):
    xcorr = correlate(env_a, env_b, mode="full")
    center = len(env_a) - 1
    return xcorr[center - max_off:center + max_off + 1]


def fine_alignment(seg_a, seg_b, sr, max_offset_ms=100) -> tuple[int, float]:
    """(integer_lag, refined_lag) in samples, refined by parabolic interpolation."""
    env_a, env_b = _envelopes(seg_a, seg_b, sr)
    max_off = int(max_offset_ms / 1000 * sr)
    region = _xcorr_region(env_a, env_b, max_off)
    lags = np.arange(-max_off, max_off + 1)
    peak_idx = np.argmax(region)
    best_lag = int(lags[peak_idx])

    if 0 < peak_idx < len(region) - 1:
        a, b, c = float(region[peak_idx - 1]), float(region[peak_idx]), float(region[peak_idx + 1])
        d = a - 2 * b + c
        refined = peak_idx + 0.5 * (a - c) / d if abs(d) > 1e-12 else float(peak_idx)
        refined_lag = refined - max_off
    else:
        refined_lag = float(best_lag)

    return best_lag, refined_lag


def find_kick_offset(audio_a, audio_b, sr, max_offset_ms=100) -> KickOffset:
    """Offset that best aligns the two segments' kicks, with a normalized correlation peak."""
    lag_int, lag_refined = fine_alignment(audio_a, audio_b, sr, max_offset_ms)

    env_a, env_b = _envelopes(audio_a, audio_b, sr)
    norm_a = np.sqrt(np.sum(env_a ** 2))
    norm_b = np.sqrt(np.sum(env_b ** 2))
    if norm_a > 0 and norm_b > 0:
        region = _xcorr_region(env_a, env_b, int(max_offset_ms / 1000 * sr))
        peak_val = float(np.max(region)) / (norm_a * norm_b)
    else:
        peak_val = 0.0

    return KickOffset(
        offset_samples=lag_int,
        offset_samples_refined=round(lag_refined, 2),
        offset_ms=round(lag_int / sr * 1000, 2),
        correlation_peak=round(min(peak_val, 1.0), 4),
    )


def extract_bar_audio(audio, sr, bpm, first_beat_sample, bar_start, bar_end):
    """Audio for bars [bar_start, bar_end) on a rigid 4/4 grid from `first_beat_sample`."""
    samples_per_bar = int(60.0 / bpm * 4 * sr)
    start_sample = max(0, first_beat_sample + bar_start * samples_per_bar)
    end_sample = min(len(audio), first_beat_sample + bar_end * samples_per_bar)
    return audio[start_sample:end_sample]


def multi_probe_alignment(audio_a, audio_b, sr, bpm,
                          first_beat_a, first_beat_b,
                          bar_start_a, bar_start_b,
                          n_probes=4, probe_length_bars=4,
                          max_offset_ms=100) -> AlignmentResult:
    """Align at several points across the overlap and report the median and its consistency."""
    probes = []
    for i in range(n_probes):
        bar_off = i * probe_length_bars
        a0, b0 = bar_start_a + bar_off, bar_start_b + bar_off
        seg_a = extract_bar_audio(audio_a, sr, bpm, first_beat_a, a0, a0 + probe_length_bars)
        seg_b = extract_bar_audio(audio_b, sr, bpm, first_beat_b, b0, b0 + probe_length_bars)
        if len(seg_a) < sr * 0.5 or len(seg_b) < sr * 0.5:
            continue

        probe = find_kick_offset(seg_a, seg_b, sr, max_offset_ms)
        probe.probe_bars_a = f"{a0}-{a0 + probe_length_bars}"
        probe.probe_bars_b = f"{b0}-{b0 + probe_length_bars}"
        probes.append(probe)

    if not probes:
        return AlignmentResult(
            offset_samples=0,
            offset_ms=0.0,
            consistency=0.0,
            probes=[],
            recommendation=AlignmentRecommendation(
                suggested_rb_offset=0,
                rationale="No valid probes — insufficient audio in overlap region.",
            ),
        )

    offsets = [p.offset_samples for p in probes]
    median_offset = int(np.median(offsets))
    median_ms = round(median_offset / sr * 1000, 2)

    tolerance_samples = int(0.010 * sr)  # 10ms
    agreeing = sum(1 for o in offsets if abs(o - median_offset) <= tolerance_samples)

    lag = median_offset > 0
    return AlignmentResult(
        offset_samples=median_offset,
        offset_ms=median_ms,
        consistency=round(agreeing / len(probes), 2),
        probes=probes,
        recommendation=AlignmentRecommendation(
            suggested_rb_offset=median_offset,
            apply_to="track_b",
            offset_ms=median_ms,
            rationale=(
                f"Track B kicks {'lag' if lag else 'lead'} Track A "
                f"by {abs(median_ms):.1f}ms ({abs(median_offset)} samples at {sr}Hz). "
                f"{'Add' if lag else 'Subtract'} {abs(median_offset)} "
                f"{'to' if lag else 'from'} Track B's first_beat_sample."
            ),
        ),
    )
