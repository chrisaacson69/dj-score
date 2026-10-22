"""domain.stems / adapters.demucs / services.stems against camelot's resum_check,
analyze_stems and separate_stems. Needs the `stems` extra (torch, demucs)."""

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import analysis, encode, music, needs_camelot

pytest.importorskip("demucs")
from dj_score.adapters.audio import LibrosaLoader  # noqa: E402
from dj_score.adapters.demucs import DemucsSeparator, SeparationFolder  # noqa: E402
from dj_score.domain import stems  # noqa: E402
from dj_score.services import stems as stem_service  # noqa: E402

pytestmark = needs_camelot

# Demucs decodes through ffmpeg/ffprobe subprocesses. On the work machine the PATH
# ffprobe is blocked by Windows Application Control (WinError 4551, 2026-10-21), so
# these run only on request; nothing here probes for ffmpeg on its own.
needs_ffmpeg = pytest.mark.skipif(os.environ.get("DJ_SCORE_FFMPEG") != "1",
                                  reason="needs an ffmpeg decode; set DJ_SCORE_FFMPEG=1 where ffmpeg may run")

SEPARATIONS = [("Waves_stems", "Waves.mp3"), ("NothingLeftBreeder_stems", "NothingLeftBreeder.wav")]


def _relocated(tmp_path, folder, audio):
    """A copy of a camelot stem folder whose separation_info.json points at where things
    are now (camelot's records the repo's old location). camelot's files are not touched."""
    dst = tmp_path / folder
    shutil.copytree(analysis(folder), dst)
    info = json.loads((dst / "separation_info.json").read_text())
    info["input"] = str(music(audio).resolve())
    info["stems"] = {k: str(dst / Path(v).name) for k, v in info["stems"].items()}
    (dst / "separation_info.json").write_text(json.dumps(info, indent=2))
    return dst


@needs_ffmpeg
@pytest.mark.parametrize("folder,audio", SEPARATIONS)
def test_resummation(camelot, tmp_path, folder, audio):
    d = _relocated(tmp_path, folder, audio)
    old = camelot("resum_check").check_track(d)
    new = stem_service.inspect_stems(d, SeparationFolder(), LibrosaLoader(), probe=False)
    r = new.resummation
    flat = {"track": new.track, "model": new.model, "seconds": r.seconds, "residual_db": r.residual_db,
            **{f"clip_{k}": v for k, v in r.clip.items()}, **{f"share_{k}": v for k, v in r.share.items()}}
    assert flat == old


@needs_ffmpeg
def test_stale_stem_paths(camelot, tmp_path):
    """Moved folder: stems are found beside the info file."""
    d = _relocated(tmp_path, *SEPARATIONS[0])
    info = json.loads((d / "separation_info.json").read_text())
    info["stems"] = {k: str(Path("C:/gone") / Path(v).name) for k, v in info["stems"].items()}
    (d / "separation_info.json").write_text(json.dumps(info))
    assert SeparationFolder().read(d).stems.keys() == set(info["stems"])


def test_stale_input_is_a_clear_error(camelot, tmp_path):
    d = _relocated(tmp_path, *SEPARATIONS[0])
    info = json.loads((d / "separation_info.json").read_text())
    info["input"] = "C:/gone/Waves.mp3"
    (d / "separation_info.json").write_text(json.dumps(info))
    with pytest.raises(FileNotFoundError, match="moved"):
        SeparationFolder().read(d)


def _excerpt(src, dst, start_s, seconds):
    import soundfile as sf
    with sf.SoundFile(src) as f:
        f.seek(int(start_s * f.samplerate))
        data = f.read(int(seconds * f.samplerate))
        sf.write(dst, data, f.samplerate)
    return dst


@pytest.mark.parametrize("stem", ["vocals", "bass", "drums", "other"])
def test_vocal_probe(camelot, tmp_path, stem):
    an = camelot("analyze_stems")
    clip = _excerpt(analysis("Waves_stems") / f"{stem}.wav", tmp_path / f"{stem}.wav", 60, 25)
    old = an.analyze_vocal_likelihood(str(clip))
    old.pop("file")
    new = stems.vocal_probe(LibrosaLoader().load(clip, 22050), 22050)
    assert encode(new) == {**old, "segments": [{k: v for k, v in s.items() if v is not None} for s in old["segments"]]}


def test_probe_reaches_several_classes(camelot, tmp_path):
    classes = set()
    for stem in ("vocals", "bass", "drums", "other"):
        clip = _excerpt(analysis("Waves_stems") / f"{stem}.wav", tmp_path / f"{stem}.wav", 60, 25)
        classes.add(stems.vocal_probe(LibrosaLoader().load(clip, 22050), 22050).classification)
    assert len(classes) >= 2, classes


@needs_ffmpeg
def test_separation(camelot, tmp_path):
    """Both separators on the same 8 s clip, CPU, same seed: identical stems."""
    import soundfile as sf
    import torch
    clip = _excerpt(music("NL123D.wav"), tmp_path / "clip.wav", 20, 8)
    outs = {}
    for side, run in (("old", lambda o: camelot("separate_stems").separate_audio(str(clip), str(o), device="cpu")),
                      ("new", lambda o: DemucsSeparator(device="cpu").separate(clip, o))):
        random.seed(0)
        torch.manual_seed(0)
        out = tmp_path / side
        run(out)
        outs[side] = out
    names = sorted(p.name for p in outs["old"].glob("*.wav"))
    assert names == sorted(p.name for p in outs["new"].glob("*.wav")) and len(names) == 4
    for n in names:
        np.testing.assert_array_equal(sf.read(outs["new"] / n)[0], sf.read(outs["old"] / n)[0], err_msg=n)
    i0, i1 = (json.loads((outs[s] / "separation_info.json").read_text()) for s in ("old", "new"))
    for k in ("processing_time_seconds", "stems"):
        i0.pop(k), i1.pop(k)
    assert i1 == i0


def test_resummation_math_without_ffmpeg(camelot, tmp_path, monkeypatch):
    """The residual/clip/share arithmetic against resum_check.check_track, on a synthetic
    separation read with soundfile (camelot's ffmpeg mix decode patched out)."""
    import soundfile as sf
    rc = camelot("resum_check")
    rng = np.random.default_rng(3)
    sr, n = 44100, 44100 * 3
    parts = {k: rng.uniform(-0.3, 0.3, (n, 2)) for k in ("drums", "bass", "vocals", "other")}
    parts["drums"][:500] = 1.0  # some full-scale samples, so clip_* is non-zero
    paths = {}
    for k, v in parts.items():
        paths[k] = tmp_path / f"{k}.wav"
        sf.write(paths[k], v, sr, subtype="FLOAT")
    mix_path = tmp_path / "mix.wav"
    sf.write(mix_path, sum(parts.values()) + rng.normal(0, 1e-3, (n + 100, 2))[:n], sr, subtype="FLOAT")
    (tmp_path / "separation_info.json").write_text(json.dumps(
        {"input": str(mix_path), "model": "synthetic", "sample_rate": sr, "stems": {k: str(p) for k, p in paths.items()}}))

    def sf_mix(path, samplerate, channels):
        return sf.read(path, dtype="float64", always_2d=True)[0].T

    monkeypatch.setattr(rc, "load_mix", sf_mix)
    old = rc.check_track(tmp_path)
    mix = sf_mix(mix_path, sr, 2)
    loaded = {k: sf.read(p, dtype="float64", always_2d=True)[0].T for k, p in paths.items()}
    r = stems.resummation(mix, loaded, sr)
    assert {"seconds": r.seconds, "residual_db": r.residual_db,
            **{f"clip_{k}": v for k, v in r.clip.items()}, **{f"share_{k}": v for k, v in r.share.items()}} == \
        {k: v for k, v in old.items() if k not in ("track", "model")}
    assert r.clip["drums"] > 0 and -80 < r.residual_db < -20
