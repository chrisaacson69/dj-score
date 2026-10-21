"""domain.key / domain.camelot against camelot's key_detect and camelot_from_youtube."""

import json
from dataclasses import fields

import numpy as np
import pytest

from conftest import CAMELOT_CACHES, encode, needs_camelot
from dj_score.adapters import cache_store
from dj_score.domain import camelot as wheel
from dj_score.domain import key
from dj_score.domain.model import KeyEstimate

pytestmark = needs_camelot


def test_wheel_tables_are_both_originals(camelot):
    kd, cfy = camelot("key_detect"), camelot("camelot_from_youtube")
    for name in ("CAMELOT_MAJOR", "CAMELOT_MINOR"):
        assert getattr(wheel, name) == getattr(kd, name) == getattr(cfy, name)
    assert wheel.PC_TO_NOTE == kd.PC_TO_NOTE == cfy.PC_TO_NOTE_FLAT
    for code in wheel.CAMELOT_CODES + ["", None, "13", "x"]:
        assert wheel.camelot_number(code) == cfy.camelot_number(code)


def test_estimate_key(camelot):
    kd = camelot("key_detect")
    rng = np.random.default_rng(0)
    for _ in range(500):
        chroma = key._normalize(rng.random(12) ** 3)
        assert encode(key.estimate_key(chroma)) == kd.estimate_key(chroma)


def test_single_window_estimator_is_the_same_core(camelot):
    """camelot_from_youtube.estimate_key_from_audio folds into estimate_key."""
    cfy = camelot("camelot_from_youtube")
    a = cache_store.load(CAMELOT_CACHES[0])
    y, sr = a.waveform.y, a.sr
    for start in (30, 60, 90):
        seg = y[start * sr:(start + 20) * sr]
        old, new = cfy.estimate_key_from_audio(seg, sr), key.estimate_key(key.chroma_mean(seg, sr))
        assert (new.tonic_pc, new.mode, new.score, new.confidence) == (old.tonic_pc, old.mode, old.score, old.confidence)
        assert [(c.score, c.key) for c in new.top_candidates] == \
            [(c["score"], f"{c['tonic']} {c['mode']}") for c in old.top_candidates]


@pytest.mark.parametrize("src", CAMELOT_CACHES[:2], ids=lambda p: p.name)
def test_measure_and_phrase_chroma(camelot, src):
    kd = camelot("key_detect")
    a = cache_store.load(src)
    measures = a.grid.measures[:24]
    y = a.waveform.y[: int(measures[-1].end * a.sr) + a.sr]
    old_measures = [encode(m) for m in measures]
    assert encode(key.compute_measure_chroma(y, a.sr, measures)) == kd.compute_measure_chroma(y, a.sr, old_measures)
    assert encode(key.compute_phrase_chroma(y, a.sr, measures)) == kd.compute_phrase_chroma(y, a.sr, old_measures)


@pytest.mark.parametrize("src", CAMELOT_CACHES, ids=lambda p: p.name)
def test_segments_and_summary(camelot, src):
    """Over every cache's stored phrases: same segments and summary as the original,
    and both match what the cache recorded."""
    kd = camelot("key_detect")
    raw = json.loads((src / cache_store.JSON_NAME).read_text(encoding="utf-8"))
    a = cache_store.load(src)
    phrases = a.key.phrases

    segs = key.find_key_segments(phrases)
    old_segs = kd.find_key_segments(raw["key"]["phrases"])
    assert encode(segs) == old_segs
    summary = key.compute_summary(segs, phrases, a.duration)
    assert encode(summary) == kd.compute_summary(old_segs, raw["key"]["phrases"], a.duration)
    assert encode(segs) == raw["key"]["key_segments"]
    assert encode(summary) == raw["key"]["key_summary"]


def test_key_estimate_fields_lead_measure_key():
    assert [f.name for f in fields(KeyEstimate)] == ["tonic_pc", "mode", "key_name", "camelot",
                                                    "score", "confidence", "top_candidates"]
