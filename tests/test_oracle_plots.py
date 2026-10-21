"""adapters.plots against camelot's figures: rendered pixels must be identical."""

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import CAMELOT_CACHES, encode, needs_camelot
from dj_score.adapters import cache_store
from dj_score.adapters.plots import collision as collision_plot
from dj_score.adapters.plots import key as key_plot
from dj_score.adapters.plots import spectral as spectral_plot
from dj_score.domain import collision

pytestmark = needs_camelot


def _pixels(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=60)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _assert_same_figure(new_fig, old_fig):
    np.testing.assert_array_equal(_pixels(new_fig), _pixels(old_fig))


@pytest.fixture(scope="module")
def track():
    src = next(p for p in CAMELOT_CACHES if '"spectrum"' in (p / cache_store.JSON_NAME).read_text(encoding="utf-8"))
    return cache_store.load(src), json.loads((src / cache_store.JSON_NAME).read_text(encoding="utf-8"))


@pytest.mark.parametrize("zoom", [(None, None), (30.0, 45.0)])
def test_key_plot(camelot, track, zoom):
    kd = camelot("key_detect")
    a, raw = track
    k = raw["key"]
    old = kd.plot_key_analysis(a.waveform.y, a.sr, a.grid.beat_times, raw["measures"], k["measure_keys"],
                               k["phrases"], k["key_segments"], 4, title="t", zoom_start=zoom[0], zoom_end=zoom[1])
    new = key_plot.plot_key_analysis(a.waveform.y, a.sr, a.grid.beat_times, a.grid.measures, a.key.phrases,
                                     a.key.key_segments, 4, title="t", zoom_start=zoom[0], zoom_end=zoom[1])
    _assert_same_figure(new, old)


def test_spectrogram(camelot, track):
    sa = camelot("spectral_analysis")
    a, raw = track
    n = a.spectrum.bar_spectra.shape[0]
    measures = a.grid.measures[:n]
    old = sa.plot_spectrogram({"bar_spectra": a.spectrum.bar_spectra, "band_edges": a.spectrum.band_edges,
                               "bar_times": [(m.start, m.end) for m in measures]}, title="t")
    _assert_same_figure(spectral_plot.plot_spectrogram(a.spectrum, measures, title="t"), old)


def _crossover_case():
    """The first (pair, bars) among the local caches whose gaps include a crossover."""
    from itertools import combinations
    spectral_caches = [p for p in CAMELOT_CACHES if '"spectrum"' in (p / cache_store.JSON_NAME).read_text(encoding="utf-8")]
    loaded = {p: cache_store.load(p) for p in spectral_caches}
    for pa, pb in combinations(spectral_caches, 2):
        for bars in (("1-32", "1-32"), ("33-64", "9-40"), ("65-96", "33-64")):
            if collision.find_spectral_gaps(loaded[pa].spectrum, loaded[pb].spectrum, *bars).suggested_crossovers:
                return (pa, pb), bars
    pytest.fail("no local pair has a spectral crossover")


def test_collision_plot(camelot):
    co = camelot("collision")
    pair, bars = _crossover_case()
    old_a, old_b = (co.load_track_spectra(p) for p in pair)
    a, b = (cache_store.load(p) for p in pair)
    old_c = co.compute_collision(old_a, old_b, *bars)
    old_g = co.find_spectral_gaps(old_a, old_b, *bars)
    new_c = collision.compute_collision(a.spectrum, b.spectrum, *bars)
    new_g = collision.find_spectral_gaps(a.spectrum, b.spectrum, *bars)
    assert new_g.suggested_crossovers, "pick bars with a crossover so the overlay is exercised"
    _assert_same_figure(collision_plot.plot_collision(new_c, new_g, "A", "B"),
                        co.plot_collision(old_c, old_g, "A", "B"))
