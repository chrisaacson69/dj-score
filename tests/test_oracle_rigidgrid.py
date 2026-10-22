"""domain.rigidgrid against camelot `bench/rigid_grid.py --comb`: the same onset
envelopes, comb tempo, beat phase and grid, given the same audio and seed tempo."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import analysis, music, needs_camelot
from dj_score.adapters.audio import LibrosaLoader
from dj_score.domain import beatgrid, rigidgrid

pytestmark = needs_camelot

SR, HOP = 44100, 512
TRACKS = ["NL123D.wav", "Waves.mp3"]


@pytest.fixture(scope="module")
def bench(camelot):
    """The bench module at the pinned revision (its functions are unchanged since)."""
    tree = Path(camelot("bpm_detect").__file__).parent
    sys.path.insert(0, str(tree / "bench"))
    return camelot("rigid_grid")


@pytest.fixture(scope="module", params=TRACKS)
def track(request):
    y = LibrosaLoader().load(music(request.param), SR)
    cache = analysis(Path(request.param).stem) / "analysis_cache.json"
    bpm0 = json.loads(cache.read_text(encoding="utf-8"))["tempo"]  # the bench's seed
    return request.param, y, bpm0


def test_onset_envelopes(bench, track):
    _, y, _ = track
    np.testing.assert_array_equal(rigidgrid.kick_onsets(y, SR, HOP), bench.kick_onsets(y, SR, HOP))


def test_comb_tempo_and_phase(bench, track):
    import librosa
    _, y, bpm0 = track
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    bpm = rigidgrid.comb_refine(env, SR, HOP, bpm0)
    assert bpm == bench.comb_refine(env, SR, HOP, bpm0)
    assert bpm != bpm0, "the search must move the seed, or it was not exercised"
    kick = rigidgrid.kick_onsets(y, SR, HOP)
    assert rigidgrid.phase_from_onsets(kick, SR, HOP, 60 / bpm) == bench.phase_from_onsets(kick, SR, HOP, 60 / bpm)


def test_grid(bench, camelot, track):
    """detect_comb_grid against the bench's main loop, step for step."""
    import librosa
    name, y, bpm0 = track
    det = rigidgrid.detect_comb_grid(y, SR, HOP, bpm0=bpm0)

    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    bpm = bench.comb_refine(env, SR, HOP, bpm0)
    period = 60.0 / bpm
    off = bench.phase_from_onsets(bench.kick_onsets(y, SR, HOP), SR, HOP, period)
    grid = off + period * np.arange(int((len(y) / SR - off) / period) + 1)
    db = camelot("bpm_detect").estimate_downbeat_phase(y, SR, grid)

    g = det.grid
    assert g.source == "comb" and g.tempo == bpm
    np.testing.assert_array_equal(g.beat_times, grid)
    assert det.downbeat.phase == db["phase"] and det.downbeat.confidence == db["confidence"]
    downbeats = grid[db["phase"]::4]
    starts = [m.start for m in g.measures if m.beat_count == 4]
    np.testing.assert_allclose(starts, downbeats[:len(starts)], rtol=0, atol=0)
    assert g.beat_times[0] < period, f"{name}: a fixed grid starts within one beat of 0 s"
    assert np.ptp(np.diff(g.beat_times)) < 1e-9, "one tempo throughout"
