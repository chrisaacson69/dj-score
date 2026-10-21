"""Measure-aligned key detection: CQT chroma against Krumhansl key profiles.

Converted from camelot `key_detect.py`. `estimate_key` is also the scoring core
of `camelot_from_youtube.estimate_key_from_audio` (same profiles, roll, sort and
confidence), so the single-window estimator is `estimate_key(chroma_mean(y, sr))`.
"""

from __future__ import annotations

import warnings
from collections import Counter

import librosa
import numpy as np

from dj_score.domain.camelot import key_name, to_camelot
from dj_score.domain.model import (
    DominantKey,
    KeyCandidate,
    KeyChange,
    KeyEstimate,
    KeyPhrase,
    KeySegment,
    KeySummary,
    Measure,
    MeasureKey,
)

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                            2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                            2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def chroma_mean(segment, sr) -> np.ndarray:
    """Normalized mean CQT chroma of an audio segment."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
    return _normalize(chroma.mean(axis=1))


def estimate_key(chroma_mean) -> KeyEstimate:
    """Estimate key from a mean chroma vector (12-dim, already normalized)."""
    maj_prof = _normalize(KRUMHANSL_MAJOR)
    min_prof = _normalize(KRUMHANSL_MINOR)

    candidates = []
    for tonic in range(12):
        maj_score = float(np.dot(chroma_mean, np.roll(maj_prof, tonic)))
        min_score = float(np.dot(chroma_mean, np.roll(min_prof, tonic)))
        candidates.append((maj_score, tonic, "major"))
        candidates.append((min_score, tonic, "minor"))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best_score, best_tonic, best_mode = candidates[0]
    second_score = candidates[1][0]

    gap = best_score - second_score
    confidence = float(np.clip(gap / 0.05, 0.0, 1.0))

    top = [KeyCandidate(score=round(float(score), 4), key=key_name(tonic, mode), camelot=to_camelot(tonic, mode))
           for score, tonic, mode in candidates[:6]]

    return KeyEstimate(
        tonic_pc=int(best_tonic),
        mode=best_mode,
        key_name=key_name(best_tonic, best_mode),
        camelot=to_camelot(best_tonic, best_mode),
        score=round(float(best_score), 4),
        confidence=round(float(confidence), 3),
        top_candidates=top,
    )


def _estimate_fields(est: KeyEstimate) -> dict:
    return {f: getattr(est, f) for f in KeyEstimate.__dataclass_fields__}


def compute_measure_chroma(y, sr, measures: list[Measure], use_harmonic=True) -> list[MeasureKey | None]:
    """Key estimate per measure, aligned to the beat grid. None for measures under 0.2 s."""
    y_analysis = librosa.effects.harmonic(y) if use_harmonic else y

    results = []
    for m in measures:
        segment = y_analysis[int(m.start * sr):int(m.end * sr)]
        if len(segment) < sr * 0.2:
            results.append(None)
            continue
        est = estimate_key(chroma_mean(segment, sr))
        results.append(MeasureKey(**_estimate_fields(est), measure_num=m.measure_num,
                                  start=m.start, end=m.end, bpm=m.bpm))
    return results


def compute_phrase_chroma(y, sr, measures: list[Measure], phrase_bars=4, use_harmonic=True) -> list[KeyPhrase]:
    """Key estimate per multi-bar phrase, the primary analysis unit (longer windows,
    more stable chroma). Phrases under 0.5 s are skipped."""
    y_analysis = librosa.effects.harmonic(y) if use_harmonic else y

    phrases = []
    for i in range(0, len(measures), phrase_bars):
        group = measures[i:i + phrase_bars]
        if not group:
            continue
        start_time, end_time = group[0].start, group[-1].end
        segment = y_analysis[int(start_time * sr):int(end_time * sr)]
        if len(segment) < sr * 0.5:
            continue
        est = estimate_key(chroma_mean(segment, sr))
        phrases.append(KeyPhrase(
            **_estimate_fields(est),
            phrase_num=len(phrases) + 1,
            start_measure=group[0].measure_num,
            end_measure=group[-1].measure_num,
            start=start_time,
            end=end_time,
            measure_count=len(group),
        ))
    return phrases


def find_key_segments(phrases: list[KeyPhrase], min_segment_phrases=2) -> list[KeySegment]:
    """Group consecutive phrases with the same Camelot code; short blips get absorbed."""
    if not phrases:
        return []

    # Pass 1: group consecutive same-key phrases
    raw = []
    current = [phrases[0]]
    for p in phrases[1:]:
        if p.camelot == current[-1].camelot:
            current.append(p)
        else:
            raw.append(current)
            current = [p]
    raw.append(current)

    if len(raw) <= 1:
        return [_finalize_key_segment(g) for g in raw]

    # Pass 2: absorb short segments into the neighbour with higher total confidence
    groups = list(raw)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            if len(groups[i]) < min_segment_phrases and len(groups) > 1:
                best_idx = None
                best_conf = -1
                for ni in [i - 1, i + 1]:
                    if 0 <= ni < len(groups):
                        nc = sum(p.confidence for p in groups[ni])
                        if nc > best_conf:
                            best_conf = nc
                            best_idx = ni
                if best_idx is not None:
                    if best_idx < i:
                        groups[best_idx].extend(groups[i])
                    else:
                        groups[best_idx] = groups[i] + groups[best_idx]
                    groups.pop(i)
                    changed = True
                    continue
            i += 1

    # Pass 3: merge adjacent groups whose majority key now matches
    merged = [groups[0]]
    for g in groups[1:]:
        if _majority_camelot(merged[-1]) == _majority_camelot(g):
            merged[-1].extend(g)
        else:
            merged.append(g)

    return [_finalize_key_segment(g) for g in merged]


def _majority_camelot(phrases):
    return Counter(p.camelot for p in phrases).most_common(1)[0][0]


def _finalize_key_segment(phrases: list[KeyPhrase]) -> KeySegment:
    camelot_counts = Counter(p.camelot for p in phrases)
    dominant_camelot = camelot_counts.most_common(1)[0][0]
    # Representative: highest-confidence phrase with the dominant key
    rep = max((p for p in phrases if p.camelot == dominant_camelot), key=lambda p: p.confidence)
    confidences = [p.confidence for p in phrases]
    scores = [p.score for p in phrases]

    return KeySegment(
        start_measure=phrases[0].start_measure,
        end_measure=phrases[-1].end_measure,
        start_time=phrases[0].start,
        end_time=phrases[-1].end,
        camelot=dominant_camelot,
        key_name=rep.key_name,
        confidence_mean=round(float(np.mean(confidences)), 3),
        confidence_max=round(float(max(confidences)), 3),
        score_mean=round(float(np.mean(scores)), 4),
        phrase_count=len(phrases),
        camelot_counts=dict(camelot_counts),
    )


def compute_summary(segments: list[KeySegment], phrases: list[KeyPhrase], duration) -> KeySummary | None:
    """Dominant key by time coverage, plus the key changes. None without segments."""
    if not segments:
        return None

    coverage = {}
    for seg in segments:
        coverage[seg.camelot] = coverage.get(seg.camelot, 0) + (seg.end_time - seg.start_time)

    dominant_code = max(coverage, key=coverage.get)
    dominant_seg = next(s for s in segments if s.camelot == dominant_code)

    changes = [
        KeyChange(
            time=round(cur.start_time, 2),
            bar=cur.start_measure,
            from_camelot=prev.camelot,
            from_key=prev.key_name,
            to_camelot=cur.camelot,
            to_key=cur.key_name,
        )
        for prev, cur in zip(segments, segments[1:])
        if cur.camelot != prev.camelot
    ]

    return KeySummary(
        dominant=DominantKey(
            key_name=dominant_seg.key_name,
            camelot=dominant_code,
            coverage=round(coverage[dominant_code] / duration, 3),
        ),
        key_changes=changes,
        segment_count=len(segments),
        phrase_count=len(phrases),
    )
