"""Rekordbox export and the two-track comparison against camelot's originals."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from conftest import CAMELOT_CACHES, CAMELOT_DIR, needs_camelot
from dj_score.adapters import cache_store
from dj_score.adapters.audio import LibrosaLoader
from dj_score.adapters.rekordbox_xml import RekordboxXml, build_track
from dj_score.services import export, pair

pytestmark = needs_camelot


def _raw(src):
    return json.loads((src / cache_store.JSON_NAME).read_text(encoding="utf-8"))


# -- Rekordbox ----------------------------------------------------------------------

class _Recorder:
    """Stands in for a pyrekordbox document and its track; records every call in order."""

    def __init__(self):
        self.calls = []
        self.TrackID = 1

    def get_tracks(self):
        return []

    def get_track_ids(self):
        return []

    def add_track(self, path, **kw):
        self.calls.append(("add_track", Path(path).name))
        return self

    def set(self, k, v):
        if k != "TrackID":  # dj-score passes the ID to add_track instead (see rekordbox_xml)
            self.calls.append(("set", k, v))

    def add_tempo(self, **kw):
        self.calls.append(("tempo", tuple(sorted(kw.items()))))

    def add_mark(self, **kw):
        self.calls.append(("mark", tuple(sorted(kw.items()))))


@pytest.mark.parametrize("src", CAMELOT_CACHES, ids=lambda p: p.name)
@pytest.mark.parametrize("max_hot", [8, 3])
def test_build_track_calls(camelot, src, max_hot):
    rx = camelot("rekordbox_export")
    old, new = _Recorder(), _Recorder()
    rx._build_track(old, "x.wav", _raw(src), include_cues=True, max_hot_cues=max_hot)
    build_track(new, "x.wav", cache_store.load(src), include_cues=True, max_hot_cues=max_hot)
    assert new.calls == old.calls
    assert any(c[0] == "mark" for c in old.calls) and any(c[0] == "tempo" for c in old.calls)


def test_build_track_without_cues(camelot):
    rx = camelot("rekordbox_export")
    old, new = _Recorder(), _Recorder()
    rx._build_track(old, "x.wav", _raw(CAMELOT_CACHES[0]), include_cues=False)
    build_track(new, "x.wav", cache_store.load(CAMELOT_CACHES[0]), include_cues=False)
    assert new.calls == old.calls


def test_multi_segment_tempo_grid_exercised():
    assert any(len(cache_store.load(p).grid.segments) > 1 for p in CAMELOT_CACHES), \
        "no local cache has >1 tempo segment; the per-segment tempo branch would be untested"


def _collection(path):
    """An XML collection as plain data, with TrackIDs replaced by the track's location
    (they only link playlists to tracks inside the file)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    tracks = root.find("COLLECTION").findall("TRACK")
    loc = {t.get("TrackID"): t.get("Location") for t in tracks}
    data = {t.get("Location"): ({k: v for k, v in t.attrib.items() if k != "TrackID"},
                                [(c.tag, dict(c.attrib)) for c in t]) for t in tracks}
    playlists = [(n.get("Name"), [loc[k.get("Key")] for k in n.findall("TRACK")])
                 for n in root.iter("NODE") if n.get("Type") == "1"]
    return data, playlists, [t.get("Location") for t in tracks]


def test_real_xml_collections(camelot, tmp_path):
    """Through pyrekordbox for real. Single exports agree with camelot except TrackID
    numbering; re-exporting known tracks as a set crashes camelot and works here."""
    from pyrekordbox.rbxml import XmlDuplicateError
    rx = camelot("rekordbox_export")
    srcs = CAMELOT_CACHES[:3]
    old_xml, new_xml = tmp_path / "old.xml", tmp_path / "new.xml"
    exporter = RekordboxXml(new_xml)
    for src in srcs:
        audio = tmp_path / f"{src.name}.mp3"
        rx.export_track(audio, _raw(src), output_path=old_xml)
        export.export_to_rekordbox(audio, cache_store.load(src), exporter)
    assert _collection(new_xml) == _collection(old_xml)

    # Re-export a known (middle) track alone: both sides fine and equal
    audio = tmp_path / f"{srcs[1].name}.mp3"
    rx.export_track(audio, _raw(srcs[1]), output_path=old_xml)
    export.export_to_rekordbox(audio, cache_store.load(srcs[1]), exporter)
    assert _collection(new_xml) == _collection(old_xml)

    # Re-export all three as a playlist set: camelot's bug, then ours
    tracks = [(tmp_path / f"{s.name}.mp3", s) for s in srcs[::-1]]
    with pytest.raises(XmlDuplicateError):
        rx.export_set([{"audio_path": a, "store_data": _raw(s)} for a, s in tracks], output_path=old_xml)
    # And camelot's playlist keys, on a fresh file where its export_set does not crash
    fresh = tmp_path / "fresh.xml"
    rx.export_set([{"audio_path": a, "store_data": _raw(s)} for a, s in tracks], output_path=fresh)
    import xml.etree.ElementTree as ET
    keys = [k.get("Key") for n in ET.parse(fresh).getroot().iter("NODE") if n.get("Type") == "1" for k in n]
    assert keys and all(k.startswith("<Track(") for k in keys), keys
    assert export.export_set_to_rekordbox([(a, cache_store.load(s)) for a, s in tracks], exporter) == 3
    data, playlists, _ = _collection(new_xml)
    assert len(data) == 3
    (name, members), = playlists
    assert name == "Camelot Set"
    from urllib.parse import unquote
    assert [Path(unquote(m)).name for m in members] == [a.name for a, _ in tracks]  # every key resolves, in order
    from pyrekordbox.rbxml import RekordboxXml as _Xml
    ids = _Xml(path=str(new_xml)).get_track_ids()
    assert len(ids) == len(set(ids)) == 3


# -- pair ---------------------------------------------------------------------------

SPECTRAL = [p for p in CAMELOT_CACHES if '"spectrum"' in (p / cache_store.JSON_NAME).read_text(encoding="utf-8")]


def test_collision_reports(camelot):
    """Every spectral pair x bar ranges, normalized as camelot's --compare does;
    asserts crossover cases were among them."""
    co = camelot("collision")
    crossovers = 0
    for pa, pb in combinations(SPECTRAL[:6], 2):
        da, db = co.load_track_spectra(pa), co.load_track_spectra(pb)
        ta, tb = cache_store.load(pa), cache_store.load(pb)
        for bars in (("1-32", "1-32"), ("33-64", "9-40")):
            col = co.compute_collision(da, db, *bars, track_a_data=da, track_b_data=db)
            gaps = co.find_spectral_gaps(da, db, *bars, track_a_data=da, track_b_data=db, min_gap_db=6.0)
            eq = co.generate_eq_recommendations(gaps, col, da["name"], db["name"])
            p = pair.compare_spectra(ta, tb, *bars, da["name"], db["name"], min_gap_db=6.0)
            assert pair.collision_report(p) == co.collision_report(col, gaps, da["name"], db["name"], eq_recs=eq)
            assert pair.collision_to_json(p) == co.collision_to_json(col, gaps, eq, da["name"], db["name"], *bars)
            crossovers += len(gaps["suggested_crossovers"])
    assert crossovers > 0


def test_align_pair(camelot):
    """camelot's --compare alignment block vs align_pair + alignment_to_json."""
    import librosa
    al = camelot("alignment")
    pa, pb = (Path(CAMELOT_DIR) / n for n in ("NL123D.wav", "NothingLeft1+2_bar400-bar445.wav"))
    sr, bpm, fa, fb, bars_a, bars_b = 44100, 132.5, 1000, 2500, "8-24", "4-20"
    y_a, _ = librosa.load(str(pa), sr=sr, mono=True)
    y_b, _ = librosa.load(str(pb), sr=sr, mono=True)
    a0, a1 = (int(x) for x in bars_a.split("-"))
    old = al.multi_probe_alignment(y_a, y_b, sr, bpm, fa, fb, a0, int(bars_b.split("-")[0]),
                                   n_probes=max(1, (a1 - a0) // 4), probe_length_bars=4, max_offset_ms=100.0)
    new = pair.align_pair(pa, pb, LibrosaLoader(), sr, bpm, fa, fb, bars_a, bars_b)
    assert len(new.probes) == 4
    import copy
    for fbs in ((fa, fb), (None, None)):
        # camelot's alignment_to_json writes new_first_beat_sample into the result it is
        # given, so each call gets a fresh copy; dj-score's leaves its input alone.
        assert pair.alignment_to_json(new, "A", "B", *fbs) == al.alignment_to_json(copy.deepcopy(old), "A", "B", *fbs)
    assert "new_first_beat_sample" not in pair.alignment_to_json(new, "A", "B")["recommendation"]
