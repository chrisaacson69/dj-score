"""Shared test plumbing.

Oracle tests run camelot's original function and the dj-score conversion on the
same input and require identical output. They need a camelot_from_youtube checkout
at DJ_SCORE_CAMELOT_DIR (and its module-level deps, e.g. `pip install -e .[plots]`);
without one they skip.

Camelot's code is imported from its git history at CAMELOT_PRE_REPOINT, not from the
working tree, so the originals can move to camelot's `archive/` without breaking the
oracles. The data (audio, analysis caches, stems) comes from the music library at
DJ_SCORE_MUSIC (see `adapters/library`).
"""

import importlib
import importlib.util
import io
import os
import sys
from pathlib import Path

import matplotlib
import pytest

from dj_score.adapters import library
from dj_score.domain.codec import to_plain

matplotlib.use("Agg")

CAMELOT_DIR = os.environ.get("DJ_SCORE_CAMELOT_DIR")
MUSIC_DIR = os.environ.get("DJ_SCORE_MUSIC")
CAMELOT_CACHES = library.projects(MUSIC_DIR) if MUSIC_DIR else []
if MUSIC_DIR and not CAMELOT_CACHES:  # parametrized oracles would pass with no cases
    raise RuntimeError(f"no analysed tracks under {MUSIC_DIR}/{library.ANALYSIS}")
needs_camelot = pytest.mark.skipif(not (CAMELOT_DIR and MUSIC_DIR),
                                   reason="DJ_SCORE_CAMELOT_DIR and DJ_SCORE_MUSIC not both set")


def music(name):
    """An original in the music library."""
    return Path(MUSIC_DIR) / name


def analysis(name):
    """A folder under the library's analysis/ (a project, or `<track>_stems`)."""
    return Path(MUSIC_DIR) / library.ANALYSIS / name


# camelot's last commit before its UI moved onto dj-score: the code every oracle runs.
CAMELOT_PRE_REPOINT = "150422a"


@pytest.fixture(scope="session")
def camelot(tmp_path_factory):
    """Import a camelot module by name, as it was at CAMELOT_PRE_REPOINT: `camelot("key_detect")`."""
    import subprocess
    import tarfile
    if not CAMELOT_DIR:
        pytest.skip("DJ_SCORE_CAMELOT_DIR not set")
    tree = tmp_path_factory.mktemp(f"camelot_{CAMELOT_PRE_REPOINT}")
    tar = subprocess.run(["git", "-C", str(Path(CAMELOT_DIR).resolve()), "archive", "--format=tar",
                          CAMELOT_PRE_REPOINT], capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(tree, filter="data")
    sys.path.insert(0, str(tree))
    return importlib.import_module


def encode(value):
    """A dj-score result in the plain JSON shape camelot's dicts have."""
    return to_plain(value)


@pytest.fixture(scope="session")
def camelot_at(camelot, tmp_path_factory):
    """Import a camelot module as it was at a revision: `camelot_at("audio_ui")`."""
    import subprocess
    cache = {}
    root = Path(CAMELOT_DIR).resolve()
    out = tmp_path_factory.mktemp("camelot_rev")

    def load(module, rev=CAMELOT_PRE_REPOINT):
        if (module, rev) not in cache:
            src = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{module}.py"],
                                 capture_output=True, check=True).stdout
            path = out / f"{module}_{rev}.py"
            path.write_bytes(src)
            spec = importlib.util.spec_from_file_location(f"{module}_{rev}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # the old audio_ui selects TkAgg on import; keep the session headless
            import matplotlib.pyplot as plt
            plt.switch_backend("Agg")
            cache[(module, rev)] = mod
        return cache[(module, rev)]
    return load
