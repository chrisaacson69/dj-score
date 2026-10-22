"""`dj-score` subcommands against the camelot CLIs they replace: same arguments, same
stdout report, same JSON, same saved chart (pixels).

`key` and `events` intentionally use the downbeat-aligned grid where camelot's CLIs
grouped bars from beat 0; the two agree exactly when the detected phase is 0, so the
excerpts used here are cut at an offset where it is.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import analysis, music, needs_camelot
from dj_score.adapters.audio import LibrosaLoader
from dj_score.cli import main as dj_main
from dj_score.domain.beatgrid import detect_grid

pytestmark = needs_camelot


def _run(fn, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        fn(argv)
    plt.close("all")
    return out.getvalue()


def _camelot(camelot, module, argv):
    mod = camelot(module)

    def call(args):
        old = sys.argv
        sys.argv = [f"{module}.py", *args]
        try:
            if module == "rhythm_detect":  # its CLI lives under `if __name__ == "__main__"`
                import runpy
                runpy.run_path(mod.__file__, run_name="__main__")
            else:
                mod.main()
        finally:
            sys.argv = old
    return _run(call, argv)


def _ours(argv):
    return _run(dj_main, argv)


def _same_png(a, b):
    np.testing.assert_array_equal(plt.imread(a), plt.imread(b))


def _cut(src, dst, start, seconds):
    import soundfile as sf
    with sf.SoundFile(src) as f:
        f.seek(int(start * f.samplerate))
        sf.write(dst, f.read(int(seconds * f.samplerate)), f.samplerate)
    return dst


def _phase0_excerpt(tmp, name, seconds=60, stems_dir=None):
    """An excerpt of `name` (and its stems) starting where the detected downbeat phase is 0."""
    src = music(name)
    for start in np.arange(30.0, 34.0, 0.25):
        clip = _cut(src, tmp / f"clip{Path(name).suffix if src.suffix == '.wav' else '.wav'}", start, seconds)
        y = LibrosaLoader().load(clip, 44100)
        if detect_grid(y, 44100).downbeat.phase == 0:
            if stems_dir:
                out = tmp / "clip_stems"
                out.mkdir(exist_ok=True)
                for w in analysis(stems_dir).glob("*.wav"):
                    _cut(w, out / w.name, start, seconds)
            return clip
    pytest.skip(f"no phase-0 offset found in {name}")


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def clip(tmp):
    return _phase0_excerpt(tmp, "NL123D.wav")


def _norm(text, *paths):
    for p in paths:
        text = text.replace(str(p), "<PATH>")
    return text


def test_bpm(camelot, clip, tmp):
    a, b = tmp / "bpm_old.png", tmp / "bpm_new.png"
    old = _camelot(camelot, "bpm_detect", [str(clip), "--save-plot", str(a)])
    new = _ours(["bpm", str(clip), "--save-plot", str(b)])
    assert _norm(new, b) == _norm(old, a)
    _same_png(b, a)


@pytest.mark.parametrize("extra", [[], ["--full-mix", "--phrase-bars", "8"]])
def test_key(camelot, clip, tmp, extra):
    ja, jb, pa, pb = tmp / "k_old.json", tmp / "k_new.json", tmp / "k_old.png", tmp / "k_new.png"
    old = _camelot(camelot, "key_detect", [str(clip), *extra, "--json-out", str(ja), "--save-plot", str(pa)])
    new = _ours(["key", str(clip), *extra, "--json-out", str(jb), "--save-plot", str(pb)])
    assert _norm(new, jb, pb) == _norm(old, ja, pa)
    assert json.loads(jb.read_text(encoding="utf-8")) == json.loads(ja.read_text())
    _same_png(pb, pa)


def test_events_features(camelot, clip, tmp):
    ja, jb, pa, pb = tmp / "e_old.json", tmp / "e_new.json", tmp / "e_old.png", tmp / "e_new.png"
    old = _camelot(camelot, "event_detect", [str(clip), "--json-out", str(ja), "--save-plot", str(pa)])
    new = _ours(["events", str(clip), "--json-out", str(jb), "--save-plot", str(pb)])
    assert "Mode: features" in new
    assert _norm(new, jb, pb) == _norm(old, ja, pa)
    assert json.loads(jb.read_text(encoding="utf-8")) == json.loads(ja.read_text())
    _same_png(pb, pa)


def test_events_stems(camelot, tmp_path):
    clip = _phase0_excerpt(tmp_path, "Waves.mp3", stems_dir="Waves_stems")
    sd = tmp_path / "clip_stems"
    ja, jb, pa, pb = (tmp_path / n for n in ("old.json", "new.json", "old.png", "new.png"))
    old = _camelot(camelot, "event_detect", [str(clip), "--stems-dir", str(sd), "--json-out", str(ja),
                                             "--save-plot", str(pa)])
    new = _ours(["events", str(clip), "--stems-dir", str(sd), "--json-out", str(jb), "--save-plot", str(pb)])
    assert "Mode: stems" in new
    assert _norm(new, jb, pb) == _norm(old, ja, pa)
    assert json.loads(jb.read_text(encoding="utf-8")) == json.loads(ja.read_text())
    _same_png(pb, pa)


def test_spectrum(camelot, clip, tmp):
    ja, jb, pa, pb = tmp / "s_old.json", tmp / "s_new.json", tmp / "s_old.png", tmp / "s_new.png"
    old = _camelot(camelot, "spectral_analysis", [str(clip), "--json-out", str(ja), "--save-plot", str(pa)])
    new = _ours(["spectrum", str(clip), "--json-out", str(jb), "--save-plot", str(pb)])
    assert _norm(new, jb, pb) == _norm(old, ja, pa)
    assert json.loads(jb.read_text(encoding="utf-8")) == json.loads(ja.read_text(encoding="utf-8"))
    _same_png(pb, pa)


def test_rhythm(camelot, clip):
    args = [str(clip), "--bpm", "132.5", "--first-downbeat", "0.4"]
    assert json.loads(_ours(["rhythm", *args])) == json.loads(_camelot(camelot, "rhythm_detect", args))


def test_export(camelot, tmp_path):
    from test_oracle_export_pair import _collection
    from conftest import CAMELOT_CACHES
    src = CAMELOT_CACHES[0]
    audio = tmp_path / f"{src.name}.mp3"
    audio.write_bytes(b"")  # dj-score requires the file to exist; camelot did not check
    xa, xb = tmp_path / "old.xml", tmp_path / "new.xml"
    old = _camelot(camelot, "rekordbox_export", ["--project", str(src), "--audio", str(audio), "--output", str(xa)])
    new = _ours(["export", "--project", str(src), "--audio", str(audio), "--output", str(xb)])
    assert _norm(new, xb) == _norm(old, xa)
    assert _collection(xb) == _collection(xa)


def test_stems_probe(camelot, tmp_path):
    clip = _cut(analysis("Waves_stems") / "vocals.wav", tmp_path / "vocals.wav", 60, 20)
    ja, jb = tmp_path / "old.json", tmp_path / "new.json"
    old = _camelot(camelot, "analyze_stems", ["--audio", str(clip), "--output", str(ja)])
    new = _ours(["stems", "probe", "--audio", str(clip), "--output", str(jb)])
    tail = lambda s: s[s.index("\nResults:"):].replace(str(ja), "<P>").replace(str(jb), "<P>")
    assert tail(new) == tail(old)
    strip = lambda d: {**d, "segments": [{k: v for k, v in s.items() if v is not None} for s in d["segments"]]}
    assert json.loads(jb.read_text(encoding="utf-8")) == strip(json.loads(ja.read_text()))


def test_export_refuses_missing_audio(tmp_path):
    from conftest import CAMELOT_CACHES
    with pytest.raises(SystemExit):
        _ours(["export", "--project", str(CAMELOT_CACHES[0]), "--audio", str(tmp_path / "gone.mp3"),
               "--output", str(tmp_path / "x.xml")])
    assert not (tmp_path / "x.xml").exists()


def test_help_lists_every_subcommand():
    from dj_score.cli import build_parser
    choices = build_parser()._subparsers._group_actions[0].choices
    assert {"bpm", "key", "events", "spectrum", "rhythm", "export", "stems"} <= set(choices)


@pytest.mark.skipif(__import__("os").environ.get("DJ_SCORE_FFMPEG") != "1",
                    reason="needs an ffmpeg decode; set DJ_SCORE_FFMPEG=1 where ffmpeg may run")
def test_stems_check(camelot, tmp_path):
    from test_oracle_stems import SEPARATIONS, _relocated
    dirs = [str(_relocated(tmp_path, *s)) for s in SEPARATIONS]
    ca, cb = tmp_path / "old.csv", tmp_path / "new.csv"
    old = _camelot(camelot, "resum_check", [*dirs, "--csv", str(ca)])
    new = _ours(["stems", "check", *dirs, "--csv", str(cb)])
    assert new == old
    assert cb.read_text() == ca.read_text()


KEY_WINDOW_CASES = [
    ["--start", "10", "--duration", "40"],
    ["--start", "70", "--duration", "40"],                           # runs past the end: error window
    ["--start", "10", "--duration", "40", "--full-mix"],
    ["--consensus", "--starts", "5,25,45", "--window", "30"],
    ["--consensus", "--start", "5", "--window", "30", "--num-windows", "3"],     # fallback rolling
    ["--consensus", "--bpm", "132.5", "--bar-start", "2", "--bar-step", "8", "--num-windows", "3",
     "--window", "20", "--min-confidence", "0.9"],                  # bars, and a strict threshold
]


@pytest.mark.parametrize("extra", KEY_WINDOW_CASES, ids=lambda a: " ".join(a)[:40])
def test_key_window(camelot, extra):
    audio = str(music("NL123D.wav"))
    old = json.loads(_camelot(camelot, "camelot_from_youtube", ["--audio", audio, *extra]))
    new = json.loads(_ours(["key-window", audio, *extra]))
    assert new == old


def test_consensus_edge_cases(camelot):
    from dj_score.domain import key
    from conftest import encode
    cfy = camelot("camelot_from_youtube")
    w = lambda c, conf, s=0.8: key.WindowKey(0, 30, key="A minor", camelot=c, confidence=conf, score=s,
                                              top_candidates=[])
    cases = [[], [key.WindowKey(0, 30, error="x")], [w("8A", 0.01), w("8B", 0.02, 0.9)],
             [w("8A", 0.5), w("8B", 0.6), w("9A", 0.7)], [w("8A", 0.5), w("8A", 0.6), w("1B", 0.9)]]
    for windows in cases:
        for min_conf in (0.05, 0.55):
            old = cfy.strict_consensus([encode(x) for x in windows], min_conf)
            assert encode(key.strict_consensus(windows, min_conf)) == old


def test_youtube_ids(camelot):
    from dj_score.adapters.youtube import extract_youtube_id
    cfy = camelot("camelot_from_youtube")
    for url in ["https://youtu.be/abc123", "https://www.youtube.com/watch?v=xyz&t=5", "https://youtube.com/shorts/s1",
                "https://www.youtube.com/embed/e9?x", "https://example.com/watch?v=no", "not a url", ""]:
        assert extract_youtube_id(url) == cfy.extract_youtube_id(url)


def test_key_window_bpm_auto_uses_the_grid():
    """Deliberate change: camelot's --bpm auto ran its own (retired) 60 s tracker."""
    from dj_score.services.analyze import bar_starts, detect_beats
    audio = music("NL123D.wav")
    r = json.loads(_ours(["key-window", str(audio), "--consensus", "--bpm", "auto", "--bar-start", "2",
                          "--bar-step", "8", "--num-windows", "2", "--window", "20"]))
    tempo = detect_beats(audio, LibrosaLoader(), sr=22050)[1].grid.tempo
    assert r["bpm_detection"]["bpm"] == tempo and "dj-score beat grid" in r["bpm_detection"]["detected_from"]
    assert r["analysis"]["starts"] == bar_starts(tempo, 2, 8, 2) and r["analysis"]["generated_from"] == "bpm/bars (auto-detected)"


def _spectral_pair():
    from conftest import CAMELOT_CACHES
    from dj_score.adapters import cache_store
    return [p for p in CAMELOT_CACHES if cache_store.load(p).spectrum is not None][:2]


@pytest.mark.parametrize("bars,gap", [(("1-32", "1-32"), "10"), (("33-64", "9-40"), "6")])
def test_pair_collision_matches_compare(camelot, bars, gap):
    pa, pb = _spectral_pair()
    common = ["--analysis-a", str(pa), "--analysis-b", str(pb), "--bars-a", bars[0], "--bars-b", bars[1],
              "--min-gap-db", gap]
    old = json.loads(_camelot(camelot, "camelot_from_youtube", ["--compare", "--audio-a", "a.wav", "--audio-b", "b.wav",
                                                                *common]))
    new = json.loads(_ours(["pair", *common]))
    assert new["compare"]["collision"] == old["compare"]["collision"]
    assert new["analysis"]["bars"] == "1-based, inclusive"


def test_pair_alignment_is_compare_one_bar_later(camelot):
    """1-based bars: dj-score's "9-25"/"5-21" are camelot --compare's 0-based "8-24"/"4-20"."""
    a, b = (str(music(n)) for n in ("NL123D.wav", "NothingLeft1+2_bar400-bar445.wav"))
    beats = ["--first-beat-a", "1000", "--first-beat-b", "2500"]
    old = json.loads(_camelot(camelot, "camelot_from_youtube", ["--compare", "--audio-a", a, "--audio-b", b,
                                                                "--bars-a", "8-24", "--bars-b", "4-20",
                                                                "--bpm-a", "132.5", "--bpm-b", "132.5", *beats]))
    new = json.loads(_ours(["pair", "--audio-a", a, "--audio-b", b, "--analysis-a", "none", "--analysis-b", "none",
                            "--bars-a", "9-25", "--bars-b", "5-21", "--bpm", "132.5", *beats,
                            "--sr", "22050"]))  # --compare aligned at its shared --sr default
    assert new["compare"]["phase_alignment"] == old["compare"]["phase_alignment"]
    assert "collision_error" in new["compare"]  # no analyses at "none"


def test_pair_needs_something_to_compare():
    with pytest.raises(SystemExit):
        _ours(["pair", "--bars-a", "1-4", "--bars-b", "1-4"])


def test_pair_report_and_plot(tmp_path):
    pa, pb = _spectral_pair()
    j, png = tmp_path / "p.json", tmp_path / "p.png"
    text = _ours(["pair", "--analysis-a", str(pa), "--analysis-b", str(pb), "--bars-a", "1-32", "--bars-b", "1-32",
                  "--json-out", str(j), "--save-plot", str(png)])
    assert text.startswith("Spectral Collision Report:") and "JSON saved to" in text
    assert png.exists() and json.loads(j.read_text(encoding="utf-8"))["compare"]["collision"]["n_bars"] == 32
