# CLAUDE.md — dj-score

Rules above this file: the kernel (`~/.claude/CLAUDE.md`) and the projects SDK (`Vault/projects/CLAUDE.md`).
Vault pointer: `Vault/projects/dj-score/README.md`. Design source: `Vault/research/well-structured-python.md`.

## What this is

The **one shared library** for DJ track analysis. Two apps depend on it and hold no analysis code of their own:
- `camelot-from-youtube` — the Tk UI (and per-track data). Source specimen this library is extracted from.
- `cyborgdj` — the renderer; its forked `scripts/{spectral_collision,export_rekordbox,analyze_track}.py` go away.

Install into a consumer's env: `pip install -e ../dj-score[plots,loudness,rekordbox]`.

## The constitution (machine-checked)

`lint-imports` must pass; `pytest` must pass. Contracts live in `pyproject.toml`:
- Layers: `cli → bootstrap → (adapters | services) → ports → domain`. Dependencies point inward only.
- `domain`, `ports`, `services` never import matplotlib, tkinter, pygame, torch, demucs, pyrekordbox,
  soundfile, requests. librosa/numpy/scipy *feature math* is allowed in the domain; loading files is not.
- Results are dataclasses (`domain/model.py`), not dicts. The cache schema IS `model.py`.
- `cli.py` stays thin: parse, call one service, print. No analysis in entry points.

## How code arrives here

**Copy → repoint → archive**, one module at a time, following `MIGRATION.md`. Golden tests from existing
`analysis_cache.json` files come first, so every move is checked against the code it came from. Never
re-derive a function that exists in camelot; convert it. Never fold a near-duplicate without diffing it.
**This repo is code only**: fixtures and all generated data are gitignored and regenerated locally
(`tests/fixtures/make_golden.py`). **Nothing is purged**: retired camelot tools and their outputs move to
camelot's `archive/`. Set `DJ_SCORE_CAMELOT_DIR=../camelot_from_youtube` to also run the golden tests over
every real cache there. Camelot's UI env is its `env1` (the only one with matplotlib + pygame).

## Decisions on record

- **Beat grid:** rigid grid — comb-search tempo + onset phase, computed in-house (no Rekordbox run needed
  per track). Downbeat phase sits behind a `DownbeatVoter` port: in-house estimator first, `beat_this`
  an optional adapter. Every `Beatgrid` carries `source` provenance — a grid read back from Rekordbox is
  only a reference when Rekordbox analysed the track natively (the 2026-09-15 contamination finding).
- **Stems:** `StemSeparator` port; Demucs is today's adapter. `inspect_stems()` = re-summation residual +
  the (unvalidated) vocal-vs-synth probe from `analyze_stems.py`; the G7 pilot grades it.
- Audio, per-track outputs, and model weights never enter this repo.
