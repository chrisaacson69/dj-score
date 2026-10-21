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
    with open(HERE / "breeder_known_events.json", "w", encoding="utf-8") as f:
        json.dump({"track": "NothingLeftBreeder.wav", "source": "camelot analyze_onset_drops.KNOWN_EVENTS",
                   "events": events}, f, indent=2)
    print(f"  breeder_known_events.json <- {len(events)} events")


if __name__ == "__main__":
    main(sys.argv[1])
