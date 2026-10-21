"""Structural events per phrase, in two modes.

Stem mode: which stems (drums, bass, vocals, other) are present per phrase, and
where they turn on or off ("bass in", "drop"). Feature mode, without stems: a
phrase-level multi-feature change pass plus a bar-level onset-drop pass.

Converted from camelot `event_detect.py`. Finding and separating stems are
adapters (`StemSource`, and later `StemSeparator`); figures live in `adapters/plots`.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from dj_score.domain.model import Event, EventPhrase, EventSummary, Measure
from dj_score.domain.timefmt import format_time

STEM_NAMES = ["drums", "bass", "vocals", "other"]


@dataclass
class BarFeatures:
    bar_num: int
    start: float
    end: float
    onset_strength: float
    onset_percussive: float
    onset_harmonic: float
    rms: float
    centroid: float
    low_freq_ratio: float
    """Share of power below 250 Hz."""


# -- stem mode ----------------------------------------------------------------

def stem_phrases(measures: list[Measure], phrase_bars=4) -> list[EventPhrase]:
    """Phrase boundaries alone, for stem mode (feature mode's phrases carry features)."""
    phrases = []
    for i in range(0, len(measures), phrase_bars):
        g = measures[i:i + phrase_bars]
        if g:
            phrases.append(EventPhrase(phrase_num=len(phrases) + 1, start_bar=g[0].measure_num,
                                       end_bar=g[-1].measure_num, start=g[0].start, end=g[-1].end))
    return phrases


def compute_stem_phrase_energy(stem_audio, sr, measures: list[Measure], phrase_bars=4) -> list[float]:
    """RMS per phrase of one stem."""
    energies = []
    for i in range(0, len(measures), phrase_bars):
        group = measures[i:i + phrase_bars]
        if not group:
            continue
        segment = stem_audio[int(group[0].start * sr):int(group[-1].end * sr)]
        energies.append(float(np.sqrt(np.mean(segment ** 2))) if len(segment) > 0 else 0.0)
    return energies


def compute_stem_presence(stem_energies_by_name: dict[str, list[float]], presence_threshold=0.15) -> dict[str, list[bool]]:
    """A stem is present in a phrase when above `presence_threshold` x its own loudest phrase."""
    presence = {}
    for stem_name, energies in stem_energies_by_name.items():
        arr = np.array(energies)
        max_e = arr.max()
        if max_e < 1e-8:
            presence[stem_name] = [False] * len(energies)
        else:
            threshold = max_e * presence_threshold
            presence[stem_name] = [bool(e > threshold) for e in arr]
    return presence


def detect_stem_events(stem_presence: dict[str, list[bool]], phrases: list[EventPhrase], min_gap_phrases=2) -> list[Event]:
    """One event per phrase where any stem's presence flips; changes closer than
    `min_gap_phrases` merge (the one with more changes wins, else changes pool)."""
    stem_names = list(stem_presence.keys())

    raw_events = []
    for i in range(1, len(phrases)):
        changes = []
        for stem in stem_names:
            prev, curr = stem_presence[stem][i - 1], stem_presence[stem][i]
            if prev != curr:
                label = "melodic" if stem in ("vocals", "other") else stem
                changes.append({"stem": stem, "label": label, "direction": "in" if curr else "out"})
        if changes:
            raw_events.append({"phrase_idx": i, "bar": phrases[i].start_bar, "time": phrases[i].start,
                               "changes": changes})

    merged = []
    for evt in raw_events:
        if merged and (evt["phrase_idx"] - merged[-1]["phrase_idx"]) < min_gap_phrases:
            if len(evt["changes"]) > len(merged[-1]["changes"]):
                merged[-1] = evt
            else:
                merged[-1]["changes"].extend(evt["changes"])
        else:
            merged.append(evt)

    events = []
    for evt in merged:
        changes = evt["changes"]
        ins = [c for c in changes if c["direction"] == "in"]
        outs = [c for c in changes if c["direction"] == "out"]

        if len(outs) >= 2 and any(c["stem"] == "drums" for c in outs):
            event_type = "breakdown"
        elif len(ins) >= 2 and any(c["stem"] == "drums" for c in ins):
            event_type = "drop"
        elif len(ins) >= 2:
            event_type = "build"
        elif len(outs) >= 2:
            event_type = "breakdown"
        else:
            event_type = f"{changes[0]['label']}_{changes[0]['direction']}"

        events.append(Event(
            bar=evt["bar"],
            time=round(evt["time"], 2),
            time_fmt=format_time(evt["time"]),
            type=event_type,
            score=round(min(1.0, len(changes) * 0.3 + 0.1), 3),
            description=", ".join(f"{c['label']} {c['direction']}" for c in changes),
            n_stems=len(changes),
            stems_in=[c["label"] for c in ins],
            stems_out=[c["label"] for c in outs],
        ))
    return events


# -- feature mode -------------------------------------------------------------

def compute_bar_features(y, sr, measures: list[Measure]) -> list[BarFeatures | None]:
    """Per bar: full / percussive / harmonic (HPSS) onset strength, RMS, spectral
    centroid, low-frequency ratio. None for bars under 0.1 s."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_perc = librosa.onset.onset_strength(y=y_percussive, sr=sr)
    onset_harm = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    onset_times = librosa.times_like(onset_env, sr=sr)

    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    hop = 512

    def bar_mean(env, mask):
        vals = env[mask]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    results = []
    for m in measures:
        segment = y[int(m.start * sr):int(m.end * sr)]
        if len(segment) < sr * 0.1:
            results.append(None)
            continue

        bar_mask = (onset_times >= m.start) & (onset_times < m.end)

        start_frame = int(m.start * sr / hop)
        end_frame = min(int(m.end * sr / hop), S.shape[1])
        centroid, low_ratio = 0.0, 0.0
        if end_frame > start_frame:
            bar_power = S[:, start_frame:end_frame] ** 2
            total_power = bar_power.sum()
            if total_power > 1e-12:
                centroid = float(np.sum(freqs[:, None] * bar_power) / total_power)
                low_ratio = float(bar_power[freqs < 250].sum() / total_power)

        results.append(BarFeatures(
            bar_num=m.measure_num,
            start=m.start,
            end=m.end,
            onset_strength=bar_mean(onset_env, bar_mask),
            onset_percussive=bar_mean(onset_perc, bar_mask),
            onset_harmonic=bar_mean(onset_harm, bar_mask),
            rms=float(np.sqrt(np.mean(segment ** 2))),
            centroid=centroid,
            low_freq_ratio=low_ratio,
        ))
    return results


_FEATURES = ["onset_strength", "onset_percussive", "onset_harmonic", "rms", "centroid", "low_freq_ratio"]


def compute_phrase_features(bar_features: list[BarFeatures | None], phrase_bars=4) -> list[EventPhrase]:
    """Bar features averaged over each run of `phrase_bars` analysable bars."""
    valid_bars = [b for b in bar_features if b is not None]
    phrases = []
    for i in range(0, len(valid_bars), phrase_bars):
        group = valid_bars[i:i + phrase_bars]
        if not group:
            continue
        phrases.append(EventPhrase(
            phrase_num=len(phrases) + 1,
            start_bar=group[0].bar_num,
            end_bar=group[-1].bar_num,
            start=group[0].start,
            end=group[-1].end,
            bar_count=len(group),
            **{f: float(np.mean([getattr(b, f) for b in group])) for f in _FEATURES},
        ))
    return phrases


def normalize_feature(values) -> np.ndarray:
    """Min-max to 0-1; a constant feature maps to zeros."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


_PASS1_WEIGHTS = {
    "onset_percussive": 0.25,  # drums/hits
    "onset_harmonic": 0.20,    # melodic/tonal
    "rms": 0.25,               # overall energy
    "centroid": 0.15,          # brightness/timbre
    "low_freq_ratio": 0.15,    # bass presence
}
_PASS1_LABELS = {
    "onset_percussive": "perc",
    "onset_harmonic": "melodic",
    "rms": "energy",
    "centroid": "brightness",
    "low_freq_ratio": "bass",
}


def detect_feature_events(phrases: list[EventPhrase], context=2, min_score=0.15,
                          min_features=2, min_gap_phrases=2,
                          bar_features: list[BarFeatures | None] | None = None,
                          min_onset_drop=0.40, bar_context=2) -> list[Event]:
    """Pass 1 (phrases): several features changing at once against the previous
    `context` phrases. Pass 2 (bars, when `bar_features` is given): percussive or
    harmonic onset drops, kept only where pass 1 found nothing nearby."""
    events = []

    # Pass 1: phrase-level multi-feature change, on HPSS-split onsets
    if len(phrases) >= context + 1:
        feature_names = list(_PASS1_WEIGHTS)
        normed = {f: normalize_feature(np.array([getattr(p, f) for p in phrases])) for f in feature_names}

        changes = []
        for i in range(context, len(phrases)):
            feature_deltas = {}
            weighted_change = 0.0
            for f in feature_names:
                delta = normed[f][i] - np.mean(normed[f][max(0, i - context):i])
                feature_deltas[f] = float(delta)
                weighted_change += abs(delta) * _PASS1_WEIGHTS[f]
            changes.append({
                "phrase_idx": i,
                "bar": phrases[i].start_bar,
                "time": phrases[i].start,
                "score": float(weighted_change),
                "n_features": sum(1 for f in feature_names if abs(feature_deltas[f]) > 0.15),
                "deltas": feature_deltas,
            })

        candidates = [c for c in changes if c["score"] >= min_score and c["n_features"] >= min_features]
        if not candidates:
            candidates = [c for c in changes if c["score"] >= min_score and c["n_features"] >= 1]

        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            kept, used = [], set()
            for c in candidates:
                if not any(abs(c["phrase_idx"] - u) < min_gap_phrases for u in used):
                    kept.append(c)
                    used.add(c["phrase_idx"])
            kept.sort(key=lambda c: c["time"])

            for c in kept:
                d = c["deltas"]
                avg_onset = (d["onset_percussive"] + d["onset_harmonic"]) / 2
                rms_dir = d["rms"]
                if avg_onset < -0.1 and rms_dir < -0.1:
                    direction = "decrease"
                elif avg_onset > 0.1 and rms_dir > 0.1:
                    direction = "increase"
                elif abs(avg_onset) > abs(rms_dir):
                    direction = "decrease" if avg_onset < 0 else "increase"
                elif abs(rms_dir) > 0.1:
                    direction = "decrease" if rms_dir < 0 else "increase"
                else:
                    direction = "change"

                desc = [f"{_PASS1_LABELS[f]}{'+' if delta > 0 else '-'}" for f, delta in d.items() if abs(delta) > 0.15]
                events.append(Event(
                    bar=c["bar"],
                    time=round(c["time"], 2),
                    time_fmt=format_time(c["time"]),
                    type=direction,
                    score=round(c["score"], 3),
                    description=", ".join(desc) if desc else "subtle change",
                    n_features=c["n_features"],
                    pass_=1,
                ))

    # Pass 2: bar-level onset drops
    if bar_features is not None:
        valid_bars = [b for b in bar_features if b is not None]
        if len(valid_bars) > bar_context * 3:
            p1_bars = {e.bar for e in events}
            bar_events = []  # (event, bar index)

            for signal_name, attr in (("perc", "onset_percussive"), ("harmonic", "onset_harmonic")):
                onset_norm = normalize_feature(np.array([getattr(b, attr) for b in valid_bars]))
                for i in range(bar_context, len(valid_bars) - bar_context):
                    before = np.mean(onset_norm[i - bar_context:i])
                    current = onset_norm[i]
                    after = np.mean(onset_norm[i + 1:i + bar_context + 1])
                    if before < 0.05:
                        continue  # already low

                    drop_pct = (before - current) / before
                    if drop_pct < min_onset_drop:
                        continue
                    abs_drop = before - current
                    recovery = (after - current) / abs_drop if abs_drop > 0.01 else 0
                    recovery = float(np.clip(recovery, 0, 2))

                    stays_low = drop_pct >= 0.75 and recovery < 0.50
                    bounces = drop_pct >= 0.50 and recovery > 0.50
                    if signal_name == "perc":
                        if stays_low:
                            etype, edesc = "breakdown", "perc drop (stays low)"
                        elif bounces:
                            etype, edesc = "fill", f"perc drop {drop_pct:.0%}, recovery {recovery:.0%}"
                        else:
                            etype, edesc = "perc_event", f"perc drop {drop_pct:.0%}"
                    else:
                        if stays_low:
                            etype, edesc = "melodic_out", "melodic drop (stays low)"
                        elif bounces:
                            etype, edesc = "melodic_change", f"melodic drop {drop_pct:.0%}, recovery {recovery:.0%}"
                        else:
                            etype, edesc = "melodic_event", f"melodic drop {drop_pct:.0%}"

                    bar = valid_bars[i]
                    bar_events.append((Event(
                        bar=bar.bar_num,
                        time=round(bar.start, 2),
                        time_fmt=format_time(bar.start),
                        type=etype,
                        score=round(float(drop_pct), 3),
                        description=edesc,
                        n_features=1,
                        pass_=2,
                    ), i))

            # Strongest per 3-bar window
            bar_events.sort(key=lambda e: e[0].score, reverse=True)
            kept_bar, used_bar_idx = [], set()
            for be, idx in bar_events:
                if any(abs(idx - u) < 3 for u in used_bar_idx):
                    continue
                used_bar_idx.add(idx)
                kept_bar.append(be)

            # Keep those more than half a phrase from any pass-1 event
            phrase_bars = phrases[0].end_bar - phrases[0].start_bar + 1 if phrases else 4
            gap_bars = max(phrase_bars // 2, 2)
            events.extend(be for be in kept_bar if not any(abs(be.bar - p1) <= gap_bars for p1 in p1_bars))

    events.sort(key=lambda e: e.time)
    return events


def compute_summary(events: list[Event], phrases: list, duration, mode="feature") -> EventSummary:
    if not events:
        return EventSummary(mode=mode, event_count=0)
    minutes = duration / 60.0
    scores = [e.score for e in events]
    return EventSummary(
        mode=mode,
        event_count=len(events),
        events_per_minute=round(len(events) / minutes, 2) if minutes > 0 else 0,
        avg_score=round(float(np.mean(scores)), 3),
        max_score=round(float(max(scores)), 3),
        phrase_count=len(phrases),
    )
