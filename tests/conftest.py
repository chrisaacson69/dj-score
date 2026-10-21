"""Shared test plumbing.

Oracle tests run camelot's original function and the dj-score conversion on the
same input and require identical output. They need a camelot_from_youtube checkout
at DJ_SCORE_CAMELOT_DIR (and its module-level deps, e.g. `pip install -e .[plots]`);
without one they skip.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib
import pytest

from dj_score.domain.codec import to_plain

matplotlib.use("Agg")

CAMELOT_DIR = os.environ.get("DJ_SCORE_CAMELOT_DIR")
CAMELOT_CACHES = sorted(p.parent for p in Path(CAMELOT_DIR).glob("*/analysis_cache.json")) if CAMELOT_DIR else []
needs_camelot = pytest.mark.skipif(not CAMELOT_DIR, reason="DJ_SCORE_CAMELOT_DIR not set")


@pytest.fixture(scope="session")
def camelot():
    """Import a camelot module by name: `camelot("key_detect")`."""
    if not CAMELOT_DIR:
        pytest.skip("DJ_SCORE_CAMELOT_DIR not set")
    root = str(Path(CAMELOT_DIR).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module


def encode(value):
    """A dj-score result in the plain JSON shape camelot's dicts have."""
    return to_plain(value)


# camelot's last commit before its UI moved onto dj-score. Modules that have since been
# repointed (audio_ui) are compared at this revision, not the working copy.
CAMELOT_PRE_REPOINT = "150422a"


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
            cache[(module, rev)] = mod
        return cache[(module, rev)]
    return load
