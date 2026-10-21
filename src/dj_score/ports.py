"""Protocols the services depend on; adapters implement them structurally (PEP 544)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dj_score.domain.model import TrackAnalysis


class AnalysisCache(Protocol):
    """Per-project persisted analysis. Implemented by `adapters.cache_store.CacheStore`."""

    def load(self, project_dir: Path) -> TrackAnalysis | None: ...

    def save(self, analysis: TrackAnalysis, project_dir: Path) -> None: ...
