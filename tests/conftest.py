"""Shared test plumbing.

Oracle tests run camelot's original function and the dj-score conversion on the
same input and require identical output. They need a camelot_from_youtube checkout
at DJ_SCORE_CAMELOT_DIR (and its module-level deps, e.g. `pip install -e .[plots]`);
without one they skip.
"""

import importlib
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
