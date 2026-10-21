# dj-score

Shared library for DJ track analysis: rigid beat grid, Camelot key, structural events, per-bar
third-octave spectral score, two-track collision / EQ crossover, loudness, Rekordbox export.

Extracted from [camelot_from_youtube](https://github.com/chrisaacson69/camelot_from_youtube), which
remains the GUI. Also consumed by [CyborgDJ](https://github.com/chrisaacson69/CyborgDJ).

```
src/dj_score/
  domain/      pure analysis — dataclasses + numeric functions
  ports.py     Protocols the services depend on
  services/    one function per use case
  adapters/    audio files, analysis cache, Rekordbox, Demucs, plots
  bootstrap.py wires adapters into services
  cli.py       `dj-score` entry point
```

```
pip install -e .[dev]
pytest
lint-imports
```

Status: skeleton. Migration plan and per-tool verdicts: [MIGRATION.md](MIGRATION.md).
