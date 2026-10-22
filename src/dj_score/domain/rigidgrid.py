"""Fixed-tempo grid: one tempo for the whole track, phase taken from the audio.

Rekordbox's method. librosa's DP tracker drifts +-200 ms through a constant-tempo
track, and its tempogram estimate is a bin centre up to ~0.2 BPM off, so the tempo is
searched instead: a 2-D comb over tempo x phase against the onset envelope. The beat
phase is then the offset with the most kick-band onset on the grid, and the downbeat
phase reuses `beatgrid.estimate_downbeat_phase`. Because the grid is extrapolated from
one tempo and phase, it reaches back to the start of the track, where the DP tracker
can begin late (Breeder: first beat at 3.0 s).

Converted from camelot `bench/rigid_grid.py --comb` (`comb_refine`, `phase_from_onsets`,
`kick_onsets` unchanged). Scored 2026-09-15 on the three tracks with a native Rekordbox
analysis: beat F 1.000, downbeat F 0.67 (the downbeat vote, not the grid, is the gap).
"""

from __future__ import annotations

import librosa
import numpy as np

from dj_score.domain.beatgrid import (
    GridDetection,
    TrimResult,
    compute_span_bpm,
    estimate_downbeat_phase,
    find_bpm_segments,
    group_beats_into_measures,
    track_beats,
)
from dj_score.domain.model import Beatgrid

COMB_GRID_SOURCE = "comb"


def phase_from_onsets(onset_env, sr, hop, period, steps=64):
    """Best grid offset in [0, period): maximise summed onset strength on the grid.
    Coarse search at `steps` per period, then refine at 1 ms."""
    frame_t = np.arange(len(onset_env)) * hop / sr
    dur = frame_t[-1]

    def score(off):
        grid = off + period * np.arange(int((dur - off) / period) + 1)
        idx = np.clip(np.round(grid * sr / hop).astype(int), 0, len(onset_env) - 1)
        return onset_env[idx].sum()

    coarse = np.linspace(0, period, steps, endpoint=False)
    best = coarse[np.argmax([score(o) for o in coarse])]
    fine = best + np.arange(-period / steps, period / steps, 0.001)
    fine = fine[fine >= 0]
    return float(fine[np.argmax([score(o) for o in fine])])


def comb_refine(onset_env, sr, hop, bpm0, span=1.5, step=0.01):
    """2-D comb search: for each tempo in bpm0 +- span, the best-phase onset sum on a
    rigid grid; return the tempo with the highest peak. Immune to the half-beat phase
    ambiguity (it takes the max over phase) and to librosa's tempogram bin quantisation."""
    fps = sr / hop
    n = len(onset_env)
    best_bpm, best_score = bpm0, -1
    for bpm in np.arange(bpm0 - span, bpm0 + span + 1e-9, step):
        period_f = 60.0 / bpm * fps
        # phase resolution of one frame is enough to rank tempos
        score = 0
        for off in np.arange(0, period_f, 1.0):
            idx = np.arange(off, n, period_f).astype(int)
            sc = onset_env[idx].sum() / len(idx)
            if sc > score:
                score = sc
        if score > best_score:
            best_bpm, best_score = bpm, score
    return float(best_bpm)


def kick_onsets(y, sr, hop, fmax=150.0):
    """Onset strength restricted to the kick band, so hats/snares can't win the phase."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=16, fmax=fmax)
    return librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=hop)


def rigid_beats(duration, period, offset) -> np.ndarray:
    """Every beat of a constant-tempo grid from `offset` to the end of the track."""
    return offset + period * np.arange(int((duration - offset) / period) + 1)


def detect_comb_grid(y, sr, hop_length=512, beats_per_bar=4, span=8, tolerance=2.0,
                     min_segment_bars=4, bpm0: float | None = None) -> GridDetection:
    """The fixed-tempo grid. `bpm0` seeds the comb search (+-1.5 BPM); by default it is
    the DP tracker's tempo, as in the 2026-09-15 bench run."""
    if bpm0 is None:
        bpm0, _ = track_beats(y, sr, hop_length)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    bpm = comb_refine(onset_env, sr, hop_length, bpm0)
    period = 60.0 / bpm
    offset = phase_from_onsets(kick_onsets(y, sr, hop_length), sr, hop_length, period)
    beats = rigid_beats(len(y) / sr, period, offset)
    downbeat = estimate_downbeat_phase(y, sr, beats)
    measures = group_beats_into_measures(beats, beats_per_bar, phase=downbeat.phase)
    grid = Beatgrid(
        tempo=bpm,
        beat_times=beats,
        measures=measures,
        segments=find_bpm_segments(measures, tolerance=tolerance, min_segment_bars=min_segment_bars),
        span=compute_span_bpm(beats, span),
        downbeat_phase=downbeat.phase,
        bars_trimmed=0,
        source=COMB_GRID_SOURCE,
    )
    return GridDetection(grid, downbeat, TrimResult(beats, 0, 0), hop_length / sr)
