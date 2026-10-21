# Migration ledger — camelot_from_youtube → dj-score

Audit 2026-10-21. Rule: copy → repoint → delete. Tick a row only when the camelot original is deleted.

## Order

1. [ ] Golden tests from 2–3 existing `analysis_cache.{json,npz}` + Breeder hand-labelled events (`analyze_onset_drops.KNOWN_EVENTS`).
2. [ ] `domain/model.py` (TrackAnalysis, Beatgrid+`source`, KeySegment, Event, Cue) + `adapters/cache_store.py` — the ONE cache reader/writer (replaces `AnalysisStore._save/_load`, `rekordbox_export.load_store_data`, `bench/beat_bench.dump_camelot`).
3. [ ] Domain leaves: `camelot`, `key`, `spectral`, `collision`, `alignment`, `rhythm`, `cues`; plots split out to `adapters/plots`.
4. [ ] Services (`analyze_*`, `compare_pair`, `inspect_stems`, `export_to_rekordbox`); camelot UI's `_run_*` become service calls.
5. [ ] `dj-score` subcommands replace the camelot CLIs; delete the retire list.
6. [ ] Grid swap: `domain/beatgrid` = comb tempo + onset phase (from `bench/rigid_grid.py`, `bench/rigidify.py`); `DownbeatVoter` port.
7. [ ] CyborgDJ drops its forks; import-linter in CI.

## Verdicts

| camelot module | Verdict | Destination |
|---|---|---|
| `audio_ui.py` | keep in camelot (the app) | `AnalysisStore` → `model` + `cache_store` + services; UI keeps charts/dialogs/playback |
| `camelot_from_youtube.py` | split, mostly retire | key single/consensus → `domain/key`; YouTube → `adapters/youtube`; `--compare` → `dj-score pair`. Retire own BPM/downbeat (Phase 3: downbeat at chance), synthetic segments, energy/micro events (diff vs `events` feature mode first) |
| `key_detect.py` | fold | canonical `domain/key`; plot → plots |
| `bpm_detect.py` | fold | measures/segments/quantization → `domain/beatgrid`; `beat_track` DP demoted to fallback (`source="librosa-dp"`) |
| `event_detect.py` | fold | `domain/events` (feature + stem mode); `_run_demucs` → `adapters/demucs` |
| `spectral_analysis.py` | fold | `domain/spectral` |
| `collision.py` | fold | `domain/collision` |
| `alignment.py`, `rhythm_detect.py` | fold | `domain/alignment`, `domain/rhythm`; one shared `kick_envelope` |
| `rekordbox_export.py` | split | tier logic → `domain/cues`; XML → `adapters/rekordbox_xml` |
| `separate_stems.py` | fold | `adapters/demucs` behind `StemSeparator` |
| `resum_check.py` | fold | `services/stems.inspect_stems` (G7 check 1) |
| `analyze_stems.py` | fold as unvalidated probe | features → `domain/stems`; reported by `inspect_stems`; G7 pilot decides keep/retire |
| `visualize_timeline.py` | fold | `adapters/plots` |
| `detect_stem_events.py` | retire after harvest | port `full_drop` (multi-stem coincidence) into `domain/events` |
| `detect_events.py` | retire | v1 onset-drop detector, superseded by `event_detect` feature mode |
| `analyze_onset_drops.py` | retire → convert | `KNOWN_EVENTS` becomes a golden test fixture |
| `visualize_audio.py`, `visualize_stems.py`, `check_beats.py` | retire | one-off exploration / diagnostics |
| `dynamic/dynamic_beat_timeline.py` | retire | experiment, superseded by the rigid grid |
| `bench/*` | stays in camelot as a harness | promotes its grid method; `dump_rekordbox` → `adapters/rekordbox_anlz` (oracle reader, native-analysed tracks only) |

## Open

- In-house downbeat: comb+onset alone scored 0.67 downbeat F on the 3 clean tracks (`beat_this` vote: 0.999). Closing that gap without `beat_this` is a G1 work item, not a blocker — the voter is a port.
- Half-beat phase flip (full-band vs kick-band onset scoring disagree per track).
