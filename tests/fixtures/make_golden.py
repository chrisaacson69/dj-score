"""Regenerate the golden fixtures from a camelot_from_youtube checkout.

    python tests/fixtures/make_golden.py ../camelot_from_youtube

Copies each chosen track's `analysis_cache.json` verbatim and its `.npz` minus the
full signal `y` (no audio enters this repo), and converts the hand-labelled
`KNOWN_EVENTS` table out of `analyze_onset_drops.py` (read with `ast`, not run).
"""

import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# slug -> camelot project dir; chosen to cover the schema variants
TRACKS = {
    "quivver": "Quivver - Brothers & Sisters (Original Mix)",  # feature-mode events, spectrum
    "waves": "Waves",                                          # stem-mode events, event_extra
    "kasablanca": "Kasablanca - Am I Dreaming？ (@Kasablanca)",  # legacy: no loudness keys
}


def main(camelot_dir):
    root = Path(camelot_dir)
    for slug, name in TRACKS.items():
        src, dst = root / name, HERE / "golden" / slug
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "analysis_cache.json", dst / "analysis_cache.json")
        with np.load(src / "analysis_cache.npz", allow_pickle=False) as z:
            np.savez_compressed(dst / "analysis_cache.npz", **{k: z[k] for k in z.files if k != "y"})
        print(f"  {slug:12} <- {name}")

    tree = ast.parse((root / "analyze_onset_drops.py").read_text(encoding="utf-8"))
    (known,) = [ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "KNOWN_EVENTS"]
    events = [{"bar": int(k.split("_")[1]), **v} for k, v in known.items()]
    for e, t in zip(events, label_times(root, [e["bar"] for e in events])):
        e["time"] = round(t, 2)
    with open(HERE / "breeder_known_events.json", "w", encoding="utf-8") as f:
        json.dump({"track": "NothingLeftBreeder.wav", "source": "camelot analyze_onset_drops.KNOWN_EVENTS",
                   "time_source": "visualize_audio grid (bars 0-based from its first downbeat)",
                   "labelled_bars": [[0, 32], [64, 128]], "events": events}, f, indent=2)
    print(f"  breeder_known_events.json <- {len(events)} events")


def label_times(root, bars):
    """The labels' bars are on `visualize_audio`'s grid (the plots they were read from).
    Rebuild that grid and check it against the timestamps written into the labels.
    It is not dj-score's grid: librosa's first beat on this track is at 3.0 s, and
    visualize_audio extrapolates back over the intro where dj-score does not."""
    import re

    import librosa
    sys.path.insert(0, str(root))
    import visualize_audio

    y, sr = librosa.load(root / "NothingLeftBreeder.wav", sr=22050, mono=True)
    g = visualize_audio.detect_beats_and_downbeat(y, sr)
    times = [float(g["beat_times"][b * 4 + g["first_downbeat_idx"]]) for b in bars]
    tree = ast.parse((root / "analyze_onset_drops.py").read_text(encoding="utf-8"))
    (known,) = [ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "KNOWN_EVENTS"]
    stamps = {int(k.split("_")[1]): re.search(r"\((\d+):(\d\d)\)", v["description"]) for k, v in known.items()}
    checked = 0
    for b, t in zip(bars, times):
        if stamps[b]:
            stamp = int(stamps[b][1]) * 60 + int(stamps[b][2])
            assert abs(t - stamp) < 1.0, f"bar {b}: grid says {t:.2f}s, label says {stamp}s; wrong env?"
            checked += 1
    assert checked >= 4, checked
    return times


if __name__ == "__main__":
    main(sys.argv[1])
