"""Checks on separated stems, the G7 pilot's oracles (Vault: research/eyes-and-ears-for-llms.md).

`resummation`: do the stems add back up to the mix? (camelot `resum_check.py`,
check 1). Demucs does not enforce re-summation, so the residual is a measurement.

`vocal_probe`: does the "vocals" stem hold a voice or a lead synth? (camelot
`analyze_stems.py`). **Unvalidated**: hand-set thresholds, folded as a probe for
the G7 pilot to grade, not as a classifier to trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class Resummation:
    seconds: float
    residual_db: float
    """10*log10(|mix - sum(stems)|^2 / |mix|^2); -40 dB or below re-sums almost exactly."""
    clip: dict[str, float]
    """Per stem: fraction of samples at full scale (clamp-on-save adds residual)."""
    share: dict[str, float]
    """Per stem: energy / total stem energy; a large "other" share is the catch-all hazard."""


def resummation(mix: np.ndarray, stems: dict[str, np.ndarray], sr: int) -> Resummation:
    """`mix` and each stem are (channels, samples) at the same rate; compared over the shortest."""
    n = min(mix.shape[1], *(s.shape[1] for s in stems.values()))
    mix = mix[:, :n]
    total = sum(s[:, :n] for s in stems.values())

    mix_energy = np.sum(mix ** 2)
    resid_energy = np.sum((mix - total) ** 2)
    stem_energy = {k: np.sum(s[:, :n] ** 2) for k, s in stems.items()}
    stem_sum = sum(stem_energy.values())

    return Resummation(
        seconds=round(n / sr, 1),
        residual_db=round(10 * np.log10(resid_energy / mix_energy), 2),
        clip={k: round(float(np.mean(np.abs(s[:, :n]) >= 0.9999)), 6) for k, s in stems.items()},
        share={k: round(float(e / stem_sum), 3) for k, e in stem_energy.items()},
    )


@dataclass
class ProbeFeatures:
    mfcc_formant_variance: float
    spectral_centroid_mean: float
    spectral_centroid_cv: float
    spectral_bandwidth: float
    spectral_flatness: float
    zcr_mean: float
    zcr_std: float
    pitch_mean_hz: float
    pitch_cv: float
    voiced_ratio: float
    harmonic_ratio: float
    spectral_contrast_bands: list[float]


@dataclass
class ProbeSegment:
    start: float
    end: float
    rms: float
    voiced_ratio: float
    pitch_hz: float | None
    vocal_likelihood: float
    has_content: bool


@dataclass
class VocalProbe:
    duration: float
    features: ProbeFeatures
    vocal_likelihood: float
    classification: str
    """"likely_vocals" (> 0.65), "mixed_or_uncertain" (> 0.45) or "likely_instruments"."""
    scoring_reasons: list[str]
    segments: list[ProbeSegment]


def _pyin(y, sr):
    return librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)


def vocal_probe(y, sr, segment_duration=5.0) -> VocalProbe:
    """Global timbre/pitch features scored into a 0-1 vocal likelihood, plus a
    simpler per-segment score. The thresholds are camelot's, as calibrated by hand
    on one set of stems."""
    mfcc_var = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = np.mean(centroid)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    try:
        f0, voiced_flag, _ = _pyin(y, sr)
        voiced_f0 = f0[voiced_flag]
        if len(voiced_f0) > 10:
            pitch_mean = np.nanmean(voiced_f0)
            pitch = (float(pitch_mean), float(np.nanstd(voiced_f0) / pitch_mean if pitch_mean > 0 else 0),
                     float(np.sum(voiced_flag) / len(voiced_flag)))
        else:
            pitch = (0.0, 0.0, 0.0)
    except Exception:
        pitch = (0.0, 0.0, 0.0)

    harmonic, _ = librosa.effects.hpss(y)
    f = ProbeFeatures(
        mfcc_formant_variance=float(np.mean(mfcc_var[1:6])),  # MFCCs 2-6 are formant-related
        spectral_centroid_mean=float(centroid_mean),
        spectral_centroid_cv=float(np.std(centroid) / centroid_mean if centroid_mean > 0 else 0),
        spectral_bandwidth=float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])),
        spectral_flatness=float(np.mean(librosa.feature.spectral_flatness(y=y)[0])),
        zcr_mean=float(np.mean(zcr)),
        zcr_std=float(np.std(zcr)),
        pitch_mean_hz=pitch[0],
        pitch_cv=pitch[1],
        voiced_ratio=pitch[2],
        harmonic_ratio=float(np.sum(harmonic ** 2) / (np.sum(y ** 2) + 1e-10)),
        spectral_contrast_bands=[float(c) for c in np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)],
    )

    score, reasons = _score(f)
    if score > 0.65:
        classification = "likely_vocals"
    elif score > 0.45:
        classification = "mixed_or_uncertain"
    else:
        classification = "likely_instruments"

    return VocalProbe(
        duration=len(y) / sr,
        features=f,
        vocal_likelihood=float(score),
        classification=classification,
        scoring_reasons=reasons,
        segments=_segments(y, sr, segment_duration),
    )


def _score(f: ProbeFeatures) -> tuple[float, list[str]]:
    pitch_hz, voiced_ratio, pitch_cv = f.pitch_mean_hz, f.voiced_ratio, f.pitch_cv
    centroid_cv, centroid_mean = f.spectral_centroid_cv, f.spectral_centroid_mean
    score, reasons = 0.0, []

    # Pitch range, the strongest discriminator: vocals 150-600 Hz, bass below
    if 150 < pitch_hz < 600:
        score += 0.25
        reasons.append(f"pitch_in_vocal_range ({pitch_hz:.0f}Hz)")
    elif pitch_hz < 150:
        score -= 0.3
        reasons.append(f"pitch_in_bass_range ({pitch_hz:.0f}Hz)")
    elif pitch_hz > 600:
        score -= 0.1
        reasons.append(f"pitch_high ({pitch_hz:.0f}Hz)")

    if 1000 < centroid_mean < 4000:
        score += 0.15
        reasons.append(f"centroid_in_vocal_range ({centroid_mean:.0f}Hz)")
    elif centroid_mean < 500:
        score -= 0.2
        reasons.append(f"centroid_too_low ({centroid_mean:.0f}Hz)")

    # Natural vibrato vs a stable synth lead (only above the bass range)
    if 0.1 < pitch_cv < 0.5 and pitch_hz > 150:
        score += 0.15
        reasons.append(f"natural_pitch_variation ({pitch_cv:.2f})")
    elif pitch_cv < 0.05 and pitch_hz > 150:
        score -= 0.15
        reasons.append(f"pitch_too_stable_for_vocals ({pitch_cv:.2f})")

    if voiced_ratio > 0.4:
        score += 0.1
        reasons.append(f"high_voiced_ratio ({voiced_ratio:.2f})")
    elif voiced_ratio < 0.2:
        score -= 0.15
        reasons.append(f"low_voiced_ratio ({voiced_ratio:.2f})")

    if centroid_cv > 0.4:
        score += 0.1
        reasons.append(f"varied_timbre ({centroid_cv:.2f})")
    elif centroid_cv < 0.2:
        score -= 0.05
        reasons.append(f"stable_timbre ({centroid_cv:.2f})")

    if f.mfcc_formant_variance > 500 and pitch_hz > 150:
        score += 0.1
        reasons.append(f"high_formant_variance ({f.mfcc_formant_variance:.1f})")

    if f.harmonic_ratio < 0.5:
        score -= 0.2
        reasons.append(f"percussive_content ({f.harmonic_ratio:.2f})")

    if 0.02 < f.spectral_flatness < 0.15:
        score += 0.05
        reasons.append(f"slight_breathiness ({f.spectral_flatness:.3f})")
    elif f.spectral_flatness < 0.01:
        score -= 0.05
        reasons.append(f"too_pure_tonal ({f.spectral_flatness:.3f})")

    return max(0, min(1, score + 0.5)), reasons


def _segments(y, sr, segment_duration) -> list[ProbeSegment]:
    segment_samples = int(segment_duration * sr)
    out = []
    for i in range(int(np.ceil(len(y) / segment_samples))):
        start_sample, end_sample = i * segment_samples, min((i + 1) * segment_samples, len(y))
        segment = y[start_sample:end_sample]
        if len(segment) < sr:
            continue

        seg_rms = np.sqrt(np.mean(segment ** 2))
        try:
            seg_f0, seg_voiced, _ = _pyin(segment, sr)
            seg_voiced_ratio = np.sum(seg_voiced) / len(seg_voiced) if len(seg_voiced) > 0 else 0
            seg_pitch = np.nanmean(seg_f0[seg_voiced]) if np.sum(seg_voiced) > 0 else 0
        except Exception:
            seg_voiced_ratio, seg_pitch = 0, 0

        seg_score = 0.5
        if 80 < seg_pitch < 800:
            seg_score += 0.2
        if seg_voiced_ratio > 0.3:
            seg_score += 0.2
        if seg_rms > 0.01:
            seg_score += 0.1

        out.append(ProbeSegment(
            start=float(start_sample / sr),
            end=float(end_sample / sr),
            rms=float(seg_rms),
            voiced_ratio=float(seg_voiced_ratio),
            pitch_hz=float(seg_pitch) if seg_pitch else None,
            vocal_likelihood=float(max(0, min(1, seg_score))),
            has_content=bool(seg_rms > 0.005),
        ))
    return out
