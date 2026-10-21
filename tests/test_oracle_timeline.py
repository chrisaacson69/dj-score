"""domain.timeline, the timeline report and views against camelot_from_youtube --timeline
and visualize_timeline. The composition itself (dj-score grid + events) has no camelot
counterpart; its conversions are checked here, and its output for structure."""

import copy
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import CAMELOT_DIR, encode, needs_camelot
from dj_score.adapters.audio import LibrosaLoader
from dj_score.adapters.plots import timeline as views
from dj_score.cli import reports
from dj_score.domain import timeline
from dj_score.domain.codec import from_plain

pytestmark = needs_camelot


@pytest.fixture(scope="module")
def camelot_timeline(camelot, tmp_path_factory):
    """A real `camelot_from_youtube --timeline` result for a short track."""
    import contextlib
    import sys
    out = tmp_path_factory.mktemp("tl") / "timeline.json"
    cfy = camelot("camelot_from_youtube")
    old_argv = sys.argv
    sys.argv = ["camelot_from_youtube.py", "--audio", str(Path(CAMELOT_DIR) / "NL123D.wav"), "--timeline",
                "--json-out", str(out)]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cfy.main()
    finally:
        sys.argv = old_argv
    return json.loads(out.read_text(encoding="utf-8"))


def _segments(raw):
    return [from_plain(timeline.TimelineSegment, s) for s in raw]


@pytest.mark.parametrize("min_conf,min_run", [(0.05, 2), (0.1, 2), (0.5, 3), (0.95, 1)])
def test_merge_and_summary_on_camelot_segments(camelot, camelot_timeline, min_conf, min_run):
    cfy = camelot("camelot_from_youtube")
    raw = camelot_timeline["estimate"]["raw_segments"]
    old = cfy.merge_timeline_segments(copy.deepcopy(raw), min_conf, min_run)
    new = timeline.merge_timeline_segments(_segments(raw), min_conf, min_run)
    assert encode(new) == old
    duration = camelot_timeline["estimate"]["duration"]
    assert encode(timeline.compute_timeline_summary(new, duration)) == cfy.compute_timeline_summary(old, duration)


def test_merge_absorbs_blips_and_types_regions(camelot):
    cfy = camelot("camelot_from_youtube")

    def seg(i, cam, conf):
        return {"start": i * 10.0, "duration": 10.0, "end": i * 10.0 + 10, "bar_start": i * 8, "bar_end": i * 8 + 8,
                "key": f"key {cam}", "camelot": cam, "confidence": conf, "score": 0.8}
    raw = [seg(0, "8A", 0.5), seg(1, "8A", 0.4), seg(2, "9B", 0.02), seg(3, "8A", 0.6),
           {"start": 40.0, "duration": 3.0, "end": 43.0, "error": "Window too short or failed to load."},
           seg(5, "1A", 0.03), seg(6, "2A", 0.07), seg(7, "2A", 0.2)]
    old = cfy.merge_timeline_segments(copy.deepcopy(raw), 0.1, 2)
    new = timeline.merge_timeline_segments(_segments(raw), 0.1, 2)
    assert encode(new) == old
    assert {r["type"] for r in old} >= {"stable", "error", "uncertain"} and any("absorbed_uncertain" in r for r in old)
    for empty in ([], [raw[4]]):
        assert encode(timeline.compute_timeline_summary(
            timeline.merge_timeline_segments(_segments(empty)), 60.0)) == \
            cfy.compute_timeline_summary(cfy.merge_timeline_segments(copy.deepcopy(empty)), 60.0)


def test_energy_profile(camelot):
    cfy = camelot("camelot_from_youtube")
    y = LibrosaLoader().load(Path(CAMELOT_DIR) / "NL123D.wav", 22050)
    for a, b in ((0, 10), (20, 35), (60, 61.5)):
        clip = y[int(a * 22050):int(b * 22050)]
        assert encode(timeline.energy_profile(clip, 22050)) == cfy.analyze_energy_segment(clip, 22050)


def test_classify_energy_event_copies_agree(camelot):
    """camelot has two classify_energy_event copies; the report uses visualize_timeline's."""
    cfy, vt = camelot("camelot_from_youtube"), camelot("visualize_timeline")
    names = ["energy_increase", "energy_decrease", "drums_increase", "drums_decrease", "bass_increase",
             "bass_decrease", "timbre_change", "texture_change", "other"]
    from itertools import combinations
    for n in range(0, 3):
        for combo in combinations(names, n):
            assert views.classify_energy_event(list(combo)) == vt.classify_energy_event(list(combo))
            assert cfy.classify_energy_event(list(combo)) == vt.classify_energy_event(list(combo)), combo


def test_report_text(camelot, camelot_timeline):
    cfy = camelot("camelot_from_youtube")
    assert reports.timeline_report(camelot_timeline["estimate"]) == cfy.format_timeline_human(camelot_timeline["estimate"])


@pytest.mark.parametrize("view", ["plot_summary_view", "plot_detailed_view"])
def test_views(camelot, camelot_timeline, tmp_path, view):
    vt = camelot("visualize_timeline")
    old_png, new_png = tmp_path / "old.png", tmp_path / "new.png"
    getattr(vt, view)(camelot_timeline, str(old_png))
    fig = getattr(views, view)(camelot_timeline)
    fig.savefig(new_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.testing.assert_array_equal(plt.imread(new_png), plt.imread(old_png))


def test_dj_score_timeline_end_to_end(tmp_path):
    """Structure of `dj-score timeline`: regions tile the analysed windows, each window's
    key is window_key on its slice, events and views render."""
    from test_oracle_cli import _ours
    from dj_score.domain.key import window_key
    audio = Path(CAMELOT_DIR) / "NL123D.wav"
    out = tmp_path / "t.json"
    text = _ours(["timeline", str(audio), "--json-out", str(out), "--plot", "both", "--plot-save", str(tmp_path / "v")])
    data = json.loads(out.read_text(encoding="utf-8"))
    e = data["estimate"]
    assert text.startswith("Dominant:") and "Full JSON saved to" in text
    assert e["segment_method"] == "dj-score beat grid" and e["beat_grid"]["source"] == "librosa-dp"
    assert abs(e["bpm"] - 132.5) < 1.5, e["bpm"]  # camelot's retired tracker said 89.1 for this track
    assert e["timeline"][0]["start"] == 0.0
    assert all(r["end"] <= e["duration"] + 0.05 for r in e["timeline"])
    y = LibrosaLoader().load(audio, 22050)
    for s in e["raw_segments"][:4]:
        k = encode(window_key(y[int(s["start"] * 22050):int(s["end"] * 22050)], 22050, s["start"], s["duration"]))
        assert {f: s.get(f) for f in k} == k
    assert e["energy_events"] and all(ev["label"] == ev["types"][0] for ev in e["energy_events"])
    assert (tmp_path / "v_summary.png").exists() and (tmp_path / "v_detailed.png").exists()


def test_timeline_plot_command(camelot, camelot_timeline, tmp_path):
    """`dj-score timeline-plot` on a saved camelot timeline file = visualize_timeline's output."""
    from test_oracle_cli import _camelot, _ours
    src = tmp_path / "t.json"
    src.write_text(json.dumps(camelot_timeline), encoding="utf-8")
    _camelot(camelot, "visualize_timeline", [str(src), "--save", str(tmp_path / "old")])
    _ours(["timeline-plot", str(src), "--save", str(tmp_path / "new")])
    for mode in ("summary", "detailed"):
        np.testing.assert_array_equal(plt.imread(tmp_path / f"new_{mode}.png"), plt.imread(tmp_path / f"old_{mode}.png"))
