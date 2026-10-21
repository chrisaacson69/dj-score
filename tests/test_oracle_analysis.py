"""domain.{spectral,kick,rhythm,alignment,collision,cues} against camelot's originals."""

import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import CAMELOT_CACHES, encode, needs_camelot
from dj_score.adapters import cache_store
from dj_score.domain import alignment, collision, cues, kick, rhythm, spectral
from dj_score.domain.model import Loudness

pytestmark = needs_camelot

SPECTRAL_CACHES = [p for p in CAMELOT_CACHES if (p / cache_store.JSON_NAME).read_text(encoding="utf-8").find('"spectrum"') >= 0]


def _strip_nulls(v):
    if isinstance(v, dict):
        return {k: _strip_nulls(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_strip_nulls(x) for x in v]
    return v


def _assert_same(new, old):
    """A dataclass result equals camelot's dict; arrays compared exactly."""
    if isinstance(old, dict) and not isinstance(new, dict):
        for k, v in old.items():
            if v is None:
                continue
            _assert_same(getattr(new, {"pass": "pass_", "from": "from_"}.get(k, k)), v)
    elif isinstance(old, np.ndarray):
        np.testing.assert_array_equal(new, old)
    elif isinstance(old, (list, tuple)) and old and not isinstance(old[0], (int, float, str, bool)):
        assert len(new) == len(old)
        for n, o in zip(new, old):
            _assert_same(n, o)
    else:
        assert _strip_nulls(encode(new)) == _strip_nulls(old)


@pytest.fixture(scope="module")
def excerpt():
    """First 24 bars of the first local track, with the audio up to them."""
    a = cache_store.load(CAMELOT_CACHES[0])
    measures = a.grid.measures[:24]
    return a, measures, a.waveform.y[: int(measures[-1].end * a.sr) + a.sr]


# -- spectral -------------------------------------------------------------------

def test_bar_spectra(camelot, excerpt):
    sa = camelot("spectral_analysis")
    a, measures, y = excerpt
    old = sa.compute_bar_spectra(y, a.sr, [encode(m) for m in measures])
    new = spectral.compute_bar_spectra(y, a.sr, measures)
    for k in ("bar_spectra", "band_centers", "band_edges"):
        np.testing.assert_array_equal(getattr(new, k), old[k])


@pytest.mark.parametrize("src", SPECTRAL_CACHES, ids=lambda p: p.name)
def test_spectral_summary(camelot, src):
    sa = camelot("spectral_analysis")
    a = cache_store.load(src)
    old = sa.compute_spectral_summary({"bar_spectra": a.spectrum.bar_spectra, "band_centers": a.spectrum.band_centers})
    new = spectral.compute_spectral_summary(a.spectrum)
    assert encode(new) == old == encode(a.spectrum.summary)


# -- kick / rhythm --------------------------------------------------------------

def test_one_kick_envelope(camelot):
    noise = np.random.default_rng(1).standard_normal(44100 * 3)
    for smooth in (3.0, 30.0):
        new = kick.kick_envelope(noise, 44100, smooth_ms=smooth)
        np.testing.assert_array_equal(new, camelot("alignment").kick_envelope(noise, 44100, smooth_ms=smooth))
        np.testing.assert_array_equal(new, camelot("rhythm_detect").kick_envelope(noise, 44100, smooth_ms=smooth))


def test_rhythm(camelot, excerpt):
    rd = camelot("rhythm_detect")
    a, measures, y = excerpt
    old_measures = [encode(m) for m in measures]
    beat_interval = 60.0 / a.grid.tempo
    old = rd.classify_rhythm(y, a.sr, old_measures, beat_interval)
    new = rhythm.classify_rhythm(y, a.sr, measures, beat_interval)
    assert encode(new) == old
    assert encode(rhythm.summarize_rhythm(new)) == rd.summarize_rhythm(old)
    assert encode(rhythm.summarize_rhythm_sections(new, measures)) == rd.summarize_rhythm_sections(old, old_measures)
    assert encode(rhythm.summarize_rhythm_sections(new)) == rd.summarize_rhythm_sections(old)
    assert encode(rhythm.summarize_rhythm([])) == rd.summarize_rhythm([])


# -- alignment ------------------------------------------------------------------

def test_multi_probe_alignment(camelot):
    al = camelot("alignment")
    ta, tb = (cache_store.load(p) for p in CAMELOT_CACHES[:2])
    sr, bpm = ta.sr, ta.grid.tempo
    fa, fb = int(ta.grid.measures[0].start * sr), int(tb.grid.measures[0].start * sr)
    ya, yb = ta.waveform.y[: sr * 120], tb.waveform.y[: sr * 120]
    for bars in ((8, 8), (16, 4), (10_000, 0)):  # the last one yields no probes
        old = al.multi_probe_alignment(ya, yb, sr, bpm, fa, fb, *bars)
        new = alignment.multi_probe_alignment(ya, yb, sr, bpm, fa, fb, *bars)
        assert encode(new) == old


# -- collision ------------------------------------------------------------------

BAR_PAIRS = [("1-32", "1-32"), ("33-64", "9-40"), ("65-96", "33-64")]


def test_collision_all_pairs(camelot):
    """Every pair of spectral caches x bar ranges x gap thresholds x loudness normalization.
    Hand-picked pairs at the default threshold produced no gaps at all, so coverage is asserted."""
    from itertools import combinations
    co = camelot("collision")
    old = {p: co.load_track_spectra(p) for p in SPECTRAL_CACHES}
    new = {p: cache_store.load(p) for p in SPECTRAL_CACHES}
    seen = {"cases": 0, "crossovers": 0, "strategies": set()}
    for pa, pb in combinations(SPECTRAL_CACHES, 2):
        for bars in BAR_PAIRS:
            for min_gap in (10.0, 6.0):
                for normalize in (False, True):
                    old_kw = dict(track_a_data=old[pa], track_b_data=old[pb]) if normalize else {}
                    new_kw = dict(loudness_a=new[pa].loudness, loudness_b=new[pb].loudness) if normalize else {}
                    old_c = co.compute_collision(old[pa], old[pb], *bars, **old_kw)
                    new_c = collision.compute_collision(new[pa].spectrum, new[pb].spectrum, *bars, **new_kw)
                    _assert_same(new_c, old_c)
                    old_g = co.find_spectral_gaps(old[pa], old[pb], *bars, min_gap_db=min_gap, **old_kw)
                    new_g = collision.find_spectral_gaps(new[pa].spectrum, new[pb].spectrum, *bars,
                                                         min_gap_db=min_gap, **new_kw)
                    assert encode(new_g) == old_g
                    old_eq = co.generate_eq_recommendations(old_g, old_c, "A", "B")
                    new_eq = collision.generate_eq_recommendations(new_g, new_c, "A")
                    assert _strip_nulls(encode(new_eq)) == _strip_nulls(old_eq)
                    seen["cases"] += 1
                    seen["crossovers"] += len(new_g.suggested_crossovers)
                    seen["strategies"].add((new_eq.strategy, new_eq.bass_handoff_hz == 200))
    assert seen["cases"] > 500 and seen["crossovers"] > 100, seen
    assert seen["strategies"] == {("bass_swap", False), ("bass_swap", True), ("simple_fade", False)}, seen


def test_collision_rejects_unequal_ranges():
    a = cache_store.load(SPECTRAL_CACHES[0])
    with pytest.raises(ValueError, match="equal length"):
        collision.find_spectral_gaps(a.spectrum, a.spectrum, "1-8", "1-9")


def test_loudness_reference_prefers_lufs(camelot):
    co = camelot("collision")
    s = np.full((2, 3), -20.0)
    for lufs, rms in ((-9.0, -12.0), (None, -12.0), (None, None)):
        np.testing.assert_array_equal(collision._normalize_spectra(s, Loudness(loudness_dbfs=rms, lufs=lufs)),
                                      co._normalize_spectra(s, rms, lufs))


# -- cues -----------------------------------------------------------------------

class _FakeTrack:
    def __init__(self):
        self.marks = []
        self.TrackID = 1

    def set(self, *a):
        pass

    def add_tempo(self, **kw):
        pass

    def add_mark(self, **kw):
        self.marks.append(kw)


class _FakeXml:
    def __init__(self):
        self.track = _FakeTrack()

    def get_tracks(self):
        return []

    def get_track_ids(self):
        return []

    def add_track(self, path):
        return self.track


def _camelot_marks(rx, store_data, max_hot):
    xml = _FakeXml()
    rx._build_track(xml, "x.wav", store_data, include_cues=True, max_hot_cues=max_hot)
    return [(m["Name"], m["Start"], m["Num"]) for m in xml.track.marks]


def _marks(plan):
    return [(c.name, c.start, c.num) for c in plan.hot + plan.memory]


@pytest.mark.parametrize("src", CAMELOT_CACHES, ids=lambda p: p.name)
@pytest.mark.parametrize("max_hot", [8, 2])
def test_cue_plan(camelot, src, max_hot):
    rx = camelot("rekordbox_export")
    raw = json.loads((src / cache_store.JSON_NAME).read_text(encoding="utf-8"))
    a = cache_store.load(src)
    plan = cues.plan_cues(a.events.events, a.key.key_summary.key_changes, max_hot)
    assert _marks(plan) == _camelot_marks(rx, raw, max_hot)


def test_cue_overrides(camelot):
    """The camelot event dialog's `cue` field: forced hot, forced memory, and an odd value."""
    rx = camelot("rekordbox_export")
    src = CAMELOT_CACHES[0]
    raw = json.loads((src / cache_store.JSON_NAME).read_text(encoding="utf-8"))
    a = cache_store.load(src)
    choices = ["hot", "memory", "auto", None, ""]
    for i, ev in enumerate(raw["events_block"]["events"]):
        if choices[i % 5] is not None:
            ev["cue"] = choices[i % 5]
            ev["source"] = "manual"
    events = [replace(e, cue=choices[i % 5]) for i, e in enumerate(a.events.events)]
    assert _marks(cues.plan_cues(events, a.key.key_summary.key_changes, 3)) == _camelot_marks(rx, raw, 3)
