"""services.analyze against camelot's UI pipeline (`AudioAnalysisApp._run_*` bodies),
compared on the cache each side persists."""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import CAMELOT_DIR, encode, needs_camelot
from dj_score.adapters import cache_store
from dj_score.adapters.audio import LibrosaLoader
from dj_score.adapters.loudness import PyloudnormMeter
from dj_score.domain import beatgrid
from dj_score.services import analyze

pytestmark = needs_camelot

# Short real tracks: one wav, one mp3
AUDIO = ["NL123D.wav", "Waves.mp3"]


def _camelot_run_bpm(bd, path, beats_per_bar=4, hop_length=512):
    """Verbatim body of camelot's `_run_bpm.do`."""
    d = bd.detect_beats(str(path), sr=44100, hop_length=hop_length)
    phase = d.get("downbeat_phase", 0)
    m = bd.group_beats_into_measures(d["beat_times"], beats_per_bar, phase=phase)
    sp = bd.compute_span_bpm(d["beat_times"], 8)
    seg = bd.find_bpm_segments(m, tolerance=2.0, min_segment_bars=4)
    return {"y": d["y"], "sr": d["sr"], "beat_times": d["beat_times"], "tempo": d["tempo"],
            "downbeat_phase": phase, "bars_trimmed": d.get("bars_trimmed", 0),
            "measures": m, "span_data": sp, "segments": seg}


@pytest.fixture(scope="module", params=AUDIO)
def both(request, camelot):
    """(camelot AnalysisStore after store_bpm, dj-score TrackAnalysis) for one track."""
    path = Path(CAMELOT_DIR) / request.param
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    store = camelot("audio_ui").AnalysisStore()
    store.store_bpm(_camelot_run_bpm(camelot("bpm_detect"), path))
    store._audio_name = path.name  # what camelot's file-open sets after store.reset()
    return store, analyze.analyze_track(path, LibrosaLoader(), PyloudnormMeter())


def _assert_same_cache(old_dir, new_dir):
    (d0, a0), (d1, a1) = _read(old_dir), _read(new_dir)
    assert d1.pop("beatgrid_source") == "librosa-dp"
    assert d1 == d0
    assert a1.keys() == a0.keys()
    for k in a0:
        assert a1[k].dtype == a0[k].dtype, k
        np.testing.assert_array_equal(a1[k], a0[k], err_msg=k)


def _read(d):
    data = json.loads((d / cache_store.JSON_NAME).read_text(encoding="utf-8"))
    with np.load(d / cache_store.NPZ_NAME) as z:
        return data, {k: z[k] for k in z.files}


def test_analyze_track_persists_what_camelot_persists(both, tmp_path):
    store, a = both
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    store.save(tmp_path / "old")
    cache_store.save(a, tmp_path / "new")
    _assert_same_cache(tmp_path / "old", tmp_path / "new")
    assert a.loudness.lufs is not None, "pyloudnorm installed, so LUFS must be measured on both sides"


def test_analyze_key_and_spectrum(both, tmp_path):
    store, a = both
    kd, sa = (__import__("key_detect"), __import__("spectral_analysis"))
    s = store
    # Verbatim bodies of camelot's `_run_key.do` and `_run_spectrum.do`
    ph = kd.compute_phrase_chroma(s.y, s.sr, s.measures, phrase_bars=4, use_harmonic=True)
    seg = kd.find_key_segments(ph, min_segment_phrases=2)
    store.store_key({"measure_keys": kd.compute_measure_chroma(s.y, s.sr, s.measures, use_harmonic=True),
                     "phrases": ph, "segments": seg, "summary": kd.compute_summary(seg, ph, s.duration)})
    r = sa.compute_bar_spectra(s.y, s.sr, s.measures)
    store.store_spectrum({"bar_spectra": r["bar_spectra"], "band_centers": r["band_centers"],
                          "band_edges": r["band_edges"], "summary": sa.compute_spectral_summary(r)})

    a.key = analyze.analyze_key(a)
    a.spectrum = analyze.analyze_spectrum(a)
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    store.save(tmp_path / "old")
    cache_store.save(a, tmp_path / "new")
    _assert_same_cache(tmp_path / "old", tmp_path / "new")


def test_trim_and_span_edge_cases(camelot):
    bd = camelot("bpm_detect")
    few = np.array([0.5, 1.0, 1.5])
    span = bd.compute_span_bpm(few, 8)
    assert {k: list(v) for k, v in span.items()} == {"times": [], "bpms": []}
    assert encode(beatgrid.compute_span_bpm(few, 8)) == {"times": [], "bpms": []}
    assert encode(beatgrid.group_beats_into_measures(few, 4, phase=2)) == bd.group_beats_into_measures(few, 4, phase=2)
    assert encode(beatgrid.find_bpm_segments([])) == bd.find_bpm_segments([])
    y = np.zeros(44100 * 4)
    assert encode(beatgrid.estimate_downbeat_phase(y, 44100, few)) == bd.estimate_downbeat_phase(y, 44100, few)
    for beats in (few, np.arange(0.5, 3.5, 0.5)):  # too few beats; silent signal
        t, o = beatgrid.trim_silent_beats(y, 44100, beats), bd.trim_silent_beats(y, 44100, beats)
        assert (t.beats_trimmed, t.bars_trimmed) == (o["beats_trimmed"], o["bars_trimmed"])
        np.testing.assert_array_equal(t.beat_times, o["trimmed_beat_times"])


def test_quantization_report(camelot):
    bd = camelot("bpm_detect")
    iv = np.diff(np.cumsum(np.full(64, 512 / 44100)) * np.array([1, 1.0001] * 32))
    o, n = bd.quantization_report(iv, 512 / 44100), beatgrid.quantization_report(iv, 512 / 44100)
    np.testing.assert_array_equal(n.unique_intervals, o["unique_intervals"])
    assert (n.frame_duration_ms, n.unique_interval_count, n.interval_step_ms, n.matches_frame_hop) == \
        (o["frame_duration_ms"], o["unique_interval_count"], o["interval_step_ms"], o["matches_frame_hop"])
