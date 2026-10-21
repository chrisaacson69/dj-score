"""Golden tests: dj-score reads camelot's caches and writes them back unchanged.

Fixtures are data, so they are not in the repo: generate them locally with
`python tests/fixtures/make_golden.py ../camelot_from_youtube`. Set DJ_SCORE_CAMELOT_DIR
to that checkout to also run every real cache there (full `y` included).
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from dj_score.adapters import cache_store
from dj_score.adapters.cache_store import CacheSchemaError, CacheStore
from dj_score.domain.model import LEGACY_GRID_SOURCE
from dj_score.ports import AnalysisCache

GOLDEN = Path(__file__).parent / "fixtures" / "golden"
FIXTURES = sorted(p for p in GOLDEN.iterdir() if p.is_dir()) if GOLDEN.exists() else []
needs_golden = pytest.mark.skipif(not FIXTURES, reason="golden fixtures not generated (see module docstring)")

CAMELOT = os.environ.get("DJ_SCORE_CAMELOT_DIR")
LOCAL = sorted(p.parent for p in Path(CAMELOT).glob("*/analysis_cache.json")) if CAMELOT else []


def _strip_nulls(v):
    """A key camelot never wrote and a key written as null mean the same thing."""
    if isinstance(v, dict):
        return {k: _strip_nulls(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_strip_nulls(x) for x in v]
    return v


# Keys older caches lack, filled the way camelot's `AnalysisStore.load` fills them.
CAMELOT_LOAD_DEFAULTS = {"downbeat_phase": 0, "bars_trimmed": 0}


def _read(d):
    with open(d / cache_store.JSON_NAME, encoding="utf-8") as f:
        data = json.load(f)
    with np.load(d / cache_store.NPZ_NAME, allow_pickle=False) as z:
        return data, {k: z[k] for k in z.files}


def _assert_round_trip(src, tmp_path):
    analysis = cache_store.load(src)
    cache_store.save(analysis, tmp_path)
    (data0, arr0), (data1, arr1) = _read(src), _read(tmp_path)
    assert data1.pop("beatgrid_source") == LEGACY_GRID_SOURCE
    assert _strip_nulls(data1) == _strip_nulls({**CAMELOT_LOAD_DEFAULTS, **data0})
    assert arr1.keys() == arr0.keys()
    for k in arr0:
        assert arr1[k].dtype == arr0[k].dtype, k
        np.testing.assert_array_equal(arr1[k], arr0[k], err_msg=k)


@pytest.mark.parametrize("src", FIXTURES, ids=lambda p: p.name)
def test_golden_round_trip(src, tmp_path):
    _assert_round_trip(src, tmp_path)


@pytest.mark.skipif(not CAMELOT, reason="DJ_SCORE_CAMELOT_DIR not set")
@pytest.mark.parametrize("src", LOCAL, ids=lambda p: p.name)
def test_local_camelot_round_trip(src, tmp_path):
    _assert_round_trip(src, tmp_path)


@needs_golden
def test_fixtures_cover_both_event_modes_and_spectrum():
    loaded = {p.name: cache_store.load(p) for p in FIXTURES}
    assert {a.events.event_mode for a in loaded.values()} == {"features", "stems"}
    assert any(a.spectrum is not None for a in loaded.values())
    assert any(a.loudness.lufs is None for a in loaded.values())


@needs_golden
def test_records_are_typed():
    a = cache_store.load(GOLDEN / "quivver")
    assert a.grid.source == LEGACY_GRID_SOURCE
    assert a.grid.measures[0].bpm > 0
    assert a.key.key_summary.dominant.camelot == "1A"
    assert a.events.events[0].pass_ == 1
    assert a.waveform.y is None


@needs_golden
def test_unknown_key_is_an_error(tmp_path):
    shutil.copytree(GOLDEN / "waves", tmp_path, dirs_exist_ok=True)
    path = tmp_path / cache_store.JSON_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["events_block"]["events"][0]["surprise"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CacheSchemaError, match="surprise"):
        cache_store.load(tmp_path)


def test_missing_cache_is_none(tmp_path):
    assert cache_store.load(tmp_path) is None


@needs_golden
def test_store_satisfies_port():
    store: AnalysisCache = CacheStore()
    assert store.load(GOLDEN / "waves").events.event_mode == "stems"
