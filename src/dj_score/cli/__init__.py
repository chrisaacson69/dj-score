"""`dj-score` console entry point. Thin: parse arguments, call a service, print.

Subcommands replace camelot's per-module CLIs (`bpm_detect.py`, `key_detect.py`,
`event_detect.py`, `spectral_analysis.py`, `rhythm_detect.py`, `rekordbox_export.py`,
`separate_stems.py`, `resum_check.py`, `analyze_stems.py`), with their arguments.
One deliberate difference: `key` and `events` use the same downbeat-aligned grid as
the UI and the other commands (camelot's two CLIs grouped bars from beat 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dj_score import __version__, bootstrap
from dj_score.cli import reports
from dj_score.domain.codec import to_plain


def _write_json(path, data, label="JSON written to"):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"{label} {path}")


def _show_or_save(fig, save_plot, label="Chart saved to"):
    import matplotlib.pyplot as plt
    if save_plot:
        fig.savefig(save_plot, dpi=150, bbox_inches="tight")
        print(f"{label} {save_plot}")
    else:
        plt.show()


def _audio(path_str) -> Path:
    path = Path(path_str)
    if not path.exists():
        print(f"Error: File not found: {path_str}", file=sys.stderr)
        sys.exit(1)
    return path


def _track(args):
    """Load + grid + loudness for the grid-based commands, with camelot's progress lines."""
    from dj_score.services import analyze
    path = _audio(args.audio)
    print(f"Loading {path.name} (sr={args.sr}, hop={args.hop_length})...", file=sys.stderr)
    a = analyze.analyze_track(path, bootstrap.loader(), bootstrap.meter(), sr=args.sr,
                              hop_length=args.hop_length, beats_per_bar=args.beats_per_bar)
    print(f"Detected {len(a.grid.beat_times)} beats (librosa tempo: {a.grid.tempo:.1f} BPM)", file=sys.stderr)
    return path, a


# -- commands -----------------------------------------------------------------------

def cmd_bpm(args):
    from dj_score.services import analyze
    path = _audio(args.audio)
    print(f"Loading {path.name} (sr={args.sr}, hop={args.hop_length})...", file=sys.stderr)
    y, det = analyze.detect_beats(path, bootstrap.loader(), sr=args.sr, hop_length=args.hop_length,
                                  beats_per_bar=args.beats_per_bar, span=args.span,
                                  tolerance=args.tolerance, min_segment_bars=args.min_segment_bars)
    grid = det.grid
    print(f"Detected {len(grid.beat_times)} beats (librosa tempo: {grid.tempo:.1f} BPM)", file=sys.stderr)
    if det.trim.bars_trimmed > 0:
        print(f"Trimmed {det.trim.bars_trimmed} silent intro bar(s) ({det.trim.beats_trimmed} beats) from beat grid",
              file=sys.stderr)
    if len(grid.beat_times) < 2:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)
    print(f"Downbeat phase: {det.downbeat.phase} (confidence: {det.downbeat.confidence:.3f})", file=sys.stderr)
    if det.downbeat.scores:
        print(f"  Phase scores: {det.downbeat.scores}", file=sys.stderr)

    print(reports.bpm_report(path.name, grid, det.frame_duration, args.tolerance, args.beats_per_bar))
    if not args.no_plot:
        from dj_score.adapters.plots.bpm import plot_bpm_analysis
        _show_or_save(plot_bpm_analysis(y, args.sr, grid, args.beats_per_bar, title=f"- {path.name}",
                                        zoom_start=args.zoom_start, zoom_end=args.zoom_end), args.save_plot)


def cmd_key(args):
    from dj_score.services import analyze
    path, a = _track(args)
    if len(a.grid.beat_times) < 2:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)
    print(f"Grouped into {len(a.grid.measures)} measures", file=sys.stderr)
    print(f"Analyzing key (harmonic={'off' if args.full_mix else 'on'}, {args.phrase_bars}-bar phrases)...",
          file=sys.stderr)
    k = analyze.analyze_key(a, phrase_bars=args.phrase_bars, min_segment_phrases=args.min_segment_phrases,
                            use_harmonic=not args.full_mix)

    print(reports.key_report(path.name, a.grid.tempo, k, args.phrase_bars))
    if args.json_out:
        _write_json(args.json_out, {
            "file": path.name, "duration": round(a.duration, 2), "bpm": round(a.grid.tempo, 2), "sr": a.sr,
            "beats_per_bar": args.beats_per_bar, "phrase_bars": args.phrase_bars,
            "summary": to_plain(k.key_summary) or {}, "segments": to_plain(k.key_segments),
            "phrases": to_plain(k.phrases)})
    if not args.no_plot:
        from dj_score.adapters.plots.key import plot_key_analysis
        _show_or_save(plot_key_analysis(a.waveform.y, a.sr, a.grid.beat_times, a.grid.measures, k.phrases,
                                        k.key_segments, args.beats_per_bar, title=f"- {path.name}",
                                        zoom_start=args.zoom_start, zoom_end=args.zoom_end,
                                        phrase_bars=args.phrase_bars), args.save_plot)


def cmd_events(args):
    from dj_score.services import analyze
    path, a = _track(args)
    if len(a.grid.beat_times) < 8:
        print("Error: Too few beats detected.", file=sys.stderr)
        sys.exit(1)
    print(f"Grouped into {len(a.grid.measures)} measures", file=sys.stderr)
    ev = analyze.analyze_events(
        a, path, bootstrap.stem_source(args.stems_dir), bootstrap.loader(), phrase_bars=args.phrase_bars,
        presence_threshold=args.presence_threshold, context=args.context, min_score=args.min_score,
        min_features=args.min_features, min_gap_phrases=args.min_gap,
        separator=bootstrap.separator() if args.auto_separate else None)
    print(f"{ev.event_mode.capitalize()} mode: {len(ev.event_phrases)} phrases", file=sys.stderr)

    print(reports.events_report(path.name, a.grid.tempo, a.duration, ev, args.phrase_bars))
    stems = ev.event_mode == "stems"
    if args.json_out:
        result = {"file": path.name, "duration": round(a.duration, 2), "bpm": round(a.grid.tempo, 2),
                  "sr": a.sr, "beats_per_bar": args.beats_per_bar, "phrase_bars": args.phrase_bars,
                  "summary": to_plain(ev.event_summary), "events": to_plain(ev.events)}
        if stems:
            result["stem_presence_summary"] = {s: sum(p) for s, p in ev.event_extra.stem_presence.items()}
        _write_json(args.json_out, result)
    if not args.no_plot:
        from dj_score.adapters.plots import events as plots
        kw = dict(title=f"- {path.name}", phrase_bars=args.phrase_bars,
                  zoom_start=args.zoom_start, zoom_end=args.zoom_end)
        if stems:
            fig = plots.plot_stem_analysis(a.waveform.y, a.sr, a.grid.measures, ev.event_phrases, ev.events,
                                           ev.event_extra.stem_energies, ev.event_extra.stem_presence,
                                           args.beats_per_bar, **kw)
        else:
            fig = plots.plot_feature_analysis(a.waveform.y, a.sr, a.grid.measures, ev.event_phrases, ev.events,
                                              args.beats_per_bar, **kw)
        _show_or_save(fig, args.save_plot)


def cmd_spectrum(args):
    from dj_score.services import analyze
    path = _audio(args.audio)
    print(f"Loading: {path.name}")
    a = analyze.analyze_track(path, bootstrap.loader(), bootstrap.meter())
    print(f"  {len(a.grid.measures)} bars, ~{a.grid.tempo:.1f} BPM")
    print("Computing bar spectra...")
    s = analyze.analyze_spectrum(a)
    print("\nSpectral Summary:")
    print(f"  Profile: {s.summary.energy_profile}")
    print(f"  Centroid: {s.summary.spectral_centroid_hz} Hz")
    print(f"  Dominant band: {s.summary.dominant_band_hz} Hz")
    if args.json_out:
        _write_json(args.json_out, to_plain(s.summary), label="Saved JSON:")
    from dj_score.adapters.plots.spectral import plot_spectrogram
    _show_or_save(plot_spectrogram(s, a.grid.measures, title=f"Spectral Energy — {path.stem}"),
                  args.save_plot, label="Saved plot:")


def cmd_rhythm(args):
    from dj_score.services import analyze
    r = analyze.analyze_rhythm(_audio(args.audio), bootstrap.loader(), args.bpm,
                               first_downbeat=args.first_downbeat, sr=args.sr)
    out = json.dumps({"summary": to_plain(r.summary), "sections": to_plain(r.sections),
                      "per_bar": to_plain(r.per_bar)}, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(out, encoding="utf-8")
        print(f"Saved to {args.json_out}")
    else:
        print(out)


def cmd_export(args):
    from dj_score.services import export
    a = bootstrap.cache().load(Path(args.project))
    if a is None:
        print(f"Error: no analysis cache in {args.project}", file=sys.stderr)
        sys.exit(1)
    n = export.export_to_rekordbox(_audio(args.audio), a, bootstrap.rekordbox(Path(args.output)),
                                   include_cues=not args.no_cues)
    print(f"Exported to: {args.output}")
    print(f"  Tracks in XML: {n}")
    print(f"  BPM: {a.grid.tempo}")
    if a.key and a.key.key_summary:
        dom = a.key.key_summary.dominant
        print(f"  Key: {dom.key_name} ({dom.camelot})")
    if a.events:
        print(f"  Events: {len(a.events.events)}")
    print(f"\nPoint Rekordbox at: {args.output}")
    print("  Preferences > Advanced > rekordbox xml > Browse")


UNSUPPORTED_HINT_EXTS = {".weba", ".webm", ".opus"}


def cmd_key_window(args):
    """camelot_from_youtube's single-window and --consensus key modes."""
    import os

    from dj_score.services import analyze
    path = _audio(args.audio)
    if path.suffix.lower() in UNSUPPORTED_HINT_EXTS:
        raise SystemExit(f"Input format {path.suffix.lower()} often requires ffmpeg. Convert to WAV/MP3/FLAC and re-run.")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = args.yt_api_key or os.environ.get("YT_API_KEY")

    yt_meta = {}
    if args.url and api_key:
        from dj_score.adapters.youtube import video_metadata
        yt_meta = video_metadata(args.url, api_key)

    use_harmonic = not args.full_mix
    loader = bootstrap.loader()
    result = {"youtube": yt_meta,
              "audio": {"path": args.audio, "sr": args.sr, "mono": True, "harmonic_only": use_harmonic}}

    bpm_value = None
    if args.bpm is not None:
        if args.bpm.lower() == "auto":
            # camelot used its own 60 s beat tracker here; dj-score uses the full-track grid
            _, det = analyze.detect_beats(path, loader, sr=args.sr)
            bpm_value = det.grid.tempo
            result["bpm_detection"] = {"bpm": bpm_value, "detected_from": "dj-score beat grid (librosa-dp, full track)",
                                       "beat_count": int(len(det.grid.beat_times))}
        else:
            try:
                bpm_value = float(args.bpm)
            except ValueError:
                raise SystemExit(f"Invalid --bpm value: {args.bpm}. Use a number or 'auto'.")

    if args.consensus:
        if args.starts:
            starts = [float(s.strip()) for s in args.starts.split(",") if s.strip()]
        elif bpm_value is not None and args.bar_start is not None:
            starts = analyze.bar_starts(bpm_value, args.bar_start, args.bar_step, args.num_windows, args.beats_per_bar)
        else:
            step = max(30.0, float(args.window) / 2.0)
            starts = [round(float(args.start) + i * step, 3) for i in range(int(args.num_windows))]
        result["analysis"] = {
            "type": "consensus_strict", "window": float(args.window), "starts": starts,
            "min_confidence": float(args.min_confidence), "bpm": bpm_value,
            "generated_from": (
                "explicit --starts" if args.starts else
                "bpm/bars (auto-detected)" if (args.bpm and args.bpm.lower() == "auto" and bpm_value
                                               and args.bar_start is not None) else
                "bpm/bars" if (bpm_value is not None and args.bar_start is not None) else
                "fallback rolling"),
        }
        consensus, windows = analyze.key_consensus(path, loader, starts, sr=args.sr, window=args.window,
                                                   use_harmonic=use_harmonic, min_conf=args.min_confidence)
        result["estimate"] = {"consensus": to_plain(consensus), "windows": to_plain(windows)}
    else:
        result["analysis"] = {"type": "single", "start": float(args.start), "duration": float(args.duration)}
        result["estimate"] = to_plain(analyze.key_window(path, loader, sr=args.sr, start=args.start,
                                                         duration=args.duration, use_harmonic=use_harmonic))

    out = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(out, encoding="utf-8")
    print(out)


def cmd_timeline(args):
    """Key timeline report (camelot_from_youtube --timeline), on the dj-score grid and events."""
    import os

    from dj_score.services.timeline import analyze_timeline
    path = _audio(args.audio)
    api_key = args.yt_api_key or os.environ.get("YT_API_KEY")
    yt_meta = {}
    if args.url and api_key:
        from dj_score.adapters.youtube import video_metadata
        yt_meta = video_metadata(args.url, api_key)

    use_harmonic = not args.full_mix
    report = analyze_timeline(path, bootstrap.loader(), bootstrap.meter(), bootstrap.stem_source(args.stems_dir),
                              sr=args.sr, bars_per_segment=args.bars_per_segment, use_harmonic=use_harmonic,
                              min_confidence=args.min_confidence, min_stable_segments=args.min_stable_segments,
                              classify_rhythm=not args.no_rhythm)
    result = {
        "youtube": yt_meta,
        "audio": {"path": args.audio, "sr": args.sr, "mono": True, "harmonic_only": use_harmonic},
        "analysis": {"type": "timeline", "bpm": report.bpm, "bars_per_segment": args.bars_per_segment,
                     "min_confidence": float(args.min_confidence), "min_stable_segments": args.min_stable_segments,
                     "segment_method": report.segment_method, "downbeat_method": report.beat_grid.source},
        "estimate": to_plain(report),
    }
    out = json.dumps(result, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(out, encoding="utf-8")
        print(reports.timeline_report(result["estimate"]))
        print(f"\nFull JSON saved to: {args.json_out}")
    else:
        print(out)

    if args.plot:
        import matplotlib.pyplot as plt

        from dj_score.adapters.plots import timeline as views
        for mode, view in (("summary", views.plot_summary_view), ("detailed", views.plot_detailed_view)):
            if args.plot in (mode, "both"):
                fig = view(result)
                if fig is None:
                    print("No raw segment data available for detailed view")
                elif args.plot_save:
                    fig.savefig(f"{args.plot_save}_{mode}.png", dpi=150, bbox_inches="tight")
                    print(f"{mode.capitalize()} view saved to: {args.plot_save}_{mode}.png")
                    plt.close(fig)
                else:
                    plt.show()


def cmd_pair(args):
    """Two-track transition: spectral collision + EQ advice, and kick phase alignment
    (camelot's --compare, collision.py and alignment.py). Bars are 1-based in both halves."""
    from dj_score.services import pair

    def project(audio, analysis):
        if analysis:
            return Path(analysis)
        return Path(audio).parent / Path(audio).stem if audio else None

    compare, report_lines = {}, []
    proj_a, proj_b = project(args.audio_a, args.analysis_a), project(args.audio_b, args.analysis_b)
    if proj_a and proj_b:
        try:
            ta, tb = bootstrap.cache().load(proj_a), bootstrap.cache().load(proj_b)
            if ta is None or tb is None:
                raise FileNotFoundError(f"No analysis cache in {proj_a if ta is None else proj_b}")
            pc = pair.compare_spectra(ta, tb, args.bars_a, args.bars_b, ta.audio_name or proj_a.name,
                                      tb.audio_name or proj_b.name, min_gap_db=args.min_gap_db)
            compare["collision"] = pair.collision_to_json(pc)
            report_lines.append(pair.collision_report(pc))
            if args.save_plot:
                from dj_score.adapters.plots.collision import plot_collision
                plot_collision(pc.collision, pc.gaps, pc.name_a, pc.name_b).savefig(args.save_plot, dpi=150,
                                                                                     bbox_inches="tight")
                report_lines.append(f"Saved plot: {args.save_plot}")
        except Exception as e:
            print(f"Warning: Collision analysis failed: {e}", file=sys.stderr)
            compare["collision_error"] = str(e)

    if args.bpm and args.first_beat_a is not None and args.first_beat_b is not None and args.audio_a and args.audio_b:
        try:
            res = pair.align_pair(Path(args.audio_a), Path(args.audio_b), bootstrap.loader(), args.sr, args.bpm,
                                  args.first_beat_a, args.first_beat_b, args.bars_a, args.bars_b,
                                  max_offset_ms=args.max_offset_ms)
            compare["phase_alignment"] = pair.alignment_to_json(res, Path(args.audio_a).stem, Path(args.audio_b).stem,
                                                                args.first_beat_a, args.first_beat_b)
            report_lines += [f"\nPhase alignment: {res.offset_ms}ms ({res.offset_samples} samples)",
                             f"Consistency: {res.consistency}",
                             f"Recommendation: {res.recommendation.rationale}"]
        except Exception as e:
            print(f"Warning: Phase alignment failed: {e}", file=sys.stderr)
            compare["alignment_error"] = str(e)

    if not compare:
        args.parser.error("nothing to compare: give two analysed tracks (--audio-a/-b or --analysis-a/-b) "
                          "and/or --bpm with --first-beat-a/-b for alignment")
    out = json.dumps({"analysis": {"type": "compare", "bars": "1-based, inclusive"}, "compare": compare}, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(out, encoding="utf-8")
        print("\n".join(report_lines))
        print(f"\nJSON saved to: {args.json_out}")
    else:
        print(out)


def cmd_timeline_plot(args):
    """Render a saved timeline report (visualize_timeline.py)."""
    import matplotlib.pyplot as plt

    from dj_score.adapters.plots import timeline as views
    path = Path(args.json_file)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("analysis", {}).get("type") != "timeline":
        print("Warning: This doesn't appear to be timeline analysis output.")
        print("Make one with: dj-score timeline AUDIO --json-out FILE")
    for mode, view in (("summary", views.plot_summary_view), ("detailed", views.plot_detailed_view)):
        if args.mode not in (mode, "both"):
            continue
        fig = view(data)
        if fig is None:
            print("No raw segment data available for detailed view")
        elif args.save:
            fig.savefig(f"{args.save}_{mode}.png", dpi=150, bbox_inches="tight")
            print(f"{mode.capitalize()} view saved to: {args.save}_{mode}.png")
            plt.close(fig)
    if not args.save:
        plt.show()


def cmd_stems_separate(args):
    from dj_score.services import stems
    paths = stems.separate_track(Path(args.audio), bootstrap.separator(
        model_name=args.model, device="cpu" if args.cpu else None, segment_length=args.segment,
        shifts=args.shifts, mp3_output=args.mp3, float_output=args.float),
        Path(args.output_dir) if args.output_dir else None)
    print("\n" + "=" * 50 + "\nStem separation complete!\n" + "=" * 50)
    for name, p in paths.items():
        print(f"  {name}: {p}")


def cmd_stems_check(args):
    import numpy as np

    from dj_score.services import stems
    dirs = [Path(d) for d in args.dirs]
    if args.root:
        dirs += sorted(p.parent for p in Path(args.root).rglob("separation_info.json"))
    if not dirs:
        args.parser.error("no stem folders given")

    rows = []
    for d in dirs:
        try:
            r = stems.inspect_stems(d, bootstrap.separation_reader(), bootstrap.loader(), probe=False)
        except Exception as e:
            print(f"FAIL {d}: {e}", file=sys.stderr)
            continue
        rs = r.resummation
        rows.append({"track": r.track, "model": r.model, "seconds": rs.seconds, "residual_db": rs.residual_db,
                     **{f"clip_{k}": v for k, v in rs.clip.items()}, **{f"share_{k}": v for k, v in rs.share.items()}})
    if not rows:
        sys.exit(1)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))
    res = np.array([r["residual_db"] for r in rows])
    print(f"\n{len(rows)} tracks  residual_db  median {np.median(res):.2f}  worst {res.max():.2f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def cmd_stems_probe(args):
    """Unvalidated vocal-vs-synth probe (see domain.stems.vocal_probe)."""
    from dj_score.domain.stems import vocal_probe
    from dj_score.services.stems import PROBE_SR
    if not args.stems_dir and not args.audio:
        args.parser.error("Either --stems-dir or --audio is required")

    def probe(p):
        result = {"file": str(p), **to_plain(vocal_probe(bootstrap.loader().load(p, PROBE_SR), PROBE_SR,
                                                         segment_duration=args.segment_duration))}
        return result

    if args.audio:
        r = probe(Path(args.audio))
        print("\nResults:")
        print(f"  Vocal likelihood: {r['vocal_likelihood']:.2f}")
        print(f"  Classification: {r['classification']}")
        print(f"  Reasons: {', '.join(r.get('scoring_reasons', []))}")
        if args.output:
            _write_json(args.output, r, label="\nSaved to:")
        return

    d = Path(args.stems_dir)
    files = sorted(d.glob("*.wav"))
    if not files:
        print(f"No .wav files found in {d}")
        sys.exit(1)
    results = {"stems_dir": str(d), "stems": {f.stem: probe(f) for f in files}}
    for name, r in results["stems"].items():
        bar = "#" * int(r["vocal_likelihood"] * 20) + "-" * (20 - int(r["vocal_likelihood"] * 20))
        print(f"{name:10s}: [{bar}] {r['vocal_likelihood']:.2f} - {r['classification']}")
    _write_json(Path(args.output) if args.output else d / "stem_analysis.json", results,
                label="\nResults saved to:")


# -- parser -------------------------------------------------------------------------

def _grid_args(p, sr=44100):
    p.add_argument("audio", help="Path to audio file (WAV/MP3/FLAC)")
    p.add_argument("--sr", type=int, default=sr, help="Sample rate (default: 44100 for 11.6ms frame resolution)")
    p.add_argument("--hop-length", type=int, default=512, help="Hop length for beat tracker (default: 512)")
    p.add_argument("--beats-per-bar", type=int, default=4, help="Beats per bar (default: 4)")


def _plot_args(p, json_out=True):
    p.add_argument("--zoom-start", type=float, default=None, help="Zoom chart start time in seconds")
    p.add_argument("--zoom-end", type=float, default=None, help="Zoom chart end time in seconds")
    p.add_argument("--no-plot", action="store_true", help="Skip chart")
    p.add_argument("--save-plot", help="Save chart to file instead of showing")
    if json_out:
        p.add_argument("--json-out", help="Write JSON results to file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dj-score", description="DJ track analysis.")
    parser.add_argument("--version", action="version", version=f"dj-score {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("bpm", help="Beat grid and BPM with frame-quantization awareness")
    _grid_args(p)
    p.add_argument("--tolerance", type=float, default=2.0, help="BPM tolerance for segment grouping (default: 2.0)")
    p.add_argument("--min-segment-bars", type=int, default=4,
                   help="Minimum bars for a segment to stand alone (default: 4)")
    p.add_argument("--span", type=int, default=8, help="Beat span for smoothed BPM curve (default: 8)")
    _plot_args(p, json_out=False)
    p.set_defaults(func=cmd_bpm)

    p = sub.add_parser("key", help="Measure-aligned key and Camelot segments")
    _grid_args(p)
    p.add_argument("--phrase-bars", type=int, default=4, help="Bars per phrase (default: 4)")
    p.add_argument("--min-segment-phrases", type=int, default=2,
                   help="Minimum phrases for a key segment to stand alone (default: 2)")
    p.add_argument("--full-mix", action="store_true", help="Use the full mix (disable harmonic-only)")
    _plot_args(p)
    p.set_defaults(func=cmd_key)

    p = sub.add_parser("key-window", help="Key of one window, or strict consensus over several (+ YouTube metadata)")
    p.add_argument("audio", help="Path to local audio file (WAV/MP3/FLAC recommended)")
    p.add_argument("--sr", type=int, default=22050, help="Sample rate for analysis (default: 22050)")
    p.add_argument("--start", type=float, default=90.0, help="Start time for single-window analysis (seconds)")
    p.add_argument("--duration", type=float, default=120.0, help="Duration for single-window analysis (seconds)")
    p.add_argument("--consensus", action="store_true", help="Run strict consensus (multi-window)")
    p.add_argument("--window", type=float, default=120.0, help="Window length for consensus analysis (seconds)")
    p.add_argument("--starts", help="Comma-separated explicit start times for consensus, e.g. '90,148,206'.")
    p.add_argument("--bpm", help="BPM used to generate starts from bars. 'auto' uses the detected beat grid.")
    p.add_argument("--bar-start", type=int, help="Starting bar number (e.g., 48)")
    p.add_argument("--bar-step", type=int, default=32, help="Step in bars between windows (default: 32)")
    p.add_argument("--num-windows", type=int, default=4, help="Number of windows to generate (default: 4)")
    p.add_argument("--beats-per-bar", type=float, default=4.0, help="Beats per bar (default: 4.0)")
    p.add_argument("--min-confidence", type=float, default=0.05,
                   help="Minimum window confidence to vote in the consensus (default: 0.05)")
    p.add_argument("--full-mix", action="store_true", help="Use full mix (disable harmonic-only)")
    p.add_argument("--url", help="YouTube URL (optional; metadata only)")
    p.add_argument("--yt-api-key", default=None, help="YouTube Data API key (default: $YT_API_KEY)")
    p.add_argument("--json-out", help="Also write the JSON to a file.")
    p.set_defaults(func=cmd_key_window)

    p = sub.add_parser("timeline", help="Key timeline: per-window key + energy, regions, mix points, events")
    p.add_argument("audio", help="Path to local audio file (WAV/MP3/FLAC recommended)")
    p.add_argument("--sr", type=int, default=22050, help="Sample rate for window analysis (default: 22050)")
    p.add_argument("--bars-per-segment", type=int, default=8, help="Bars per analysis window (default: 8)")
    p.add_argument("--min-stable-segments", type=int, default=2,
                   help="Windows a key must hold to count as stable (default: 2)")
    p.add_argument("--min-confidence", type=float, default=0.05, help="Confidence below which a window is uncertain")
    p.add_argument("--full-mix", action="store_true", help="Use full mix (disable harmonic-only)")
    p.add_argument("--stems-dir", default=None, help="Stems folder for stem-mode events")
    p.add_argument("--no-rhythm", action="store_true", help="Skip per-bar rhythm classification")
    p.add_argument("--url", help="YouTube URL (optional; metadata only)")
    p.add_argument("--yt-api-key", default=None, help="YouTube Data API key (default: $YT_API_KEY)")
    p.add_argument("--json-out", help="Write the JSON to a file and print the human-readable report")
    p.add_argument("--plot", choices=["summary", "detailed", "both"], help="Timeline views")
    p.add_argument("--plot-save", help="Save views to files (prefix, e.g. 'track1' -> track1_summary.png)")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("timeline-plot", help="Render a saved timeline report (summary and/or detailed view)")
    p.add_argument("json_file", help="Timeline JSON written by `dj-score timeline --json-out`")
    p.add_argument("--mode", choices=["summary", "detailed", "both"], default="both", help="Which view (default: both)")
    p.add_argument("--save", help="Save to files (prefix, e.g. 'track1' -> track1_summary.png) instead of showing")
    p.set_defaults(func=cmd_timeline_plot)

    p = sub.add_parser("pair", help="Two-track transition: spectral collision, EQ advice, kick alignment")
    p.add_argument("--audio-a", help="Track A audio (alignment; also locates its analysis)")
    p.add_argument("--audio-b", help="Track B audio")
    p.add_argument("--analysis-a", help="Track A project folder with analysis_cache.* (default: <audio>/<stem>)")
    p.add_argument("--analysis-b", help="Track B project folder")
    p.add_argument("--bars-a", required=True, help="Track A overlap bars, 1-based inclusive, e.g. 200-260")
    p.add_argument("--bars-b", required=True, help="Track B overlap bars, same length, e.g. 1-61")
    p.add_argument("--bpm", type=float, help="Shared BPM after time-stretching (alignment)")
    p.add_argument("--first-beat-a", type=int, help="Sample index of Track A's first beat (alignment)")
    p.add_argument("--first-beat-b", type=int, help="Sample index of Track B's first beat (alignment)")
    p.add_argument("--sr", type=int, default=44100,
                   help="Sample rate for alignment; --first-beat-* are sample indices at this rate (default: 44100, "
                        "as alignment.py; camelot --compare used 22050)")
    p.add_argument("--min-gap-db", type=float, default=10.0, help="Spectral gap threshold in dB (default: 10)")
    p.add_argument("--max-offset-ms", type=float, default=100.0, help="Alignment search window (default: 100)")
    p.add_argument("--save-plot", help="Save the collision figure")
    p.add_argument("--json-out", help="Write the JSON to a file and print the reports")
    p.set_defaults(func=cmd_pair, parser=p)

    p = sub.add_parser("events", help="Structural events (stem mode when stems exist, else feature mode)")
    _grid_args(p)
    p.add_argument("--stems-dir", default=None, help="Folder with drums/bass/vocals/other.wav")
    p.add_argument("--auto-separate", action="store_true", help="Run Demucs when no stems exist (stems extra)")
    p.add_argument("--presence-threshold", type=float, default=0.15, help="Stem presence threshold (default: 0.15)")
    p.add_argument("--phrase-bars", type=int, default=4, help="Bars per phrase (default: 4)")
    p.add_argument("--context", type=int, default=2, help="Phrases of context for change detection (default: 2)")
    p.add_argument("--min-score", type=float, default=0.15, help="Minimum change score (default: 0.15)")
    p.add_argument("--min-features", type=int, default=2, help="Minimum changing features (default: 2)")
    p.add_argument("--min-gap", type=int, default=2, help="Minimum phrases between events (default: 2)")
    _plot_args(p)
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("spectrum", help="Per-bar third-octave spectral analysis")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--save-plot", default=None, help="Save chart to file instead of showing")
    p.add_argument("--json-out", default=None, help="Write the spectral summary to file")
    p.set_defaults(func=cmd_spectrum)

    p = sub.add_parser("rhythm", help="Kick pattern per bar on a rigid grid")
    p.add_argument("audio", help="Path to audio file.")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM.")
    p.add_argument("--first-downbeat", type=float, default=0.0, help="First downbeat in seconds (default: 0).")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate.")
    p.add_argument("--json-out", help="Write JSON output to file.")
    p.set_defaults(func=cmd_rhythm)

    p = sub.add_parser("export", help="Export an analysed track to a Rekordbox XML collection")
    p.add_argument("--project", "-p", required=True, help="Project directory with analysis_cache.json")
    p.add_argument("--audio", "-a", required=True, help="Path to the audio file")
    p.add_argument("--output", "-o", required=True, help="Collection XML to add the track to (created if missing)")
    p.add_argument("--no-cues", action="store_true", help="Skip exporting event markers as cue points")
    p.set_defaults(func=cmd_export)

    stems = sub.add_parser("stems", help="Stem separation and checks").add_subparsers(dest="stems_command",
                                                                                      metavar="ACTION")
    p = stems.add_parser("separate", help="Separate audio into stems with Demucs (stems extra)")
    p.add_argument("--audio", "-a", required=True, help="Path to input audio file")
    p.add_argument("--output-dir", "-o", help="Output directory for stems (default: <audio>_stems/)")
    p.add_argument("--model", "-m", default="htdemucs",
                   choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"],
                   help="Demucs model to use (default: htdemucs)")
    p.add_argument("--cpu", action="store_true", help="Force CPU processing")
    p.add_argument("--segment", type=int, default=None, help="Segment length in seconds (for memory management)")
    p.add_argument("--shifts", type=int, default=1, help="Random shifts to average (1-10, higher=better, slower)")
    p.add_argument("--mp3", action="store_true", help="Output MP3 instead of WAV")
    p.add_argument("--float", action="store_true", help="32-bit float WAV without clamping (for re-summation checks)")
    p.set_defaults(func=cmd_stems_separate)

    p = stems.add_parser("check", help="Do separated stems re-sum to the mix? (G7 check 1)")
    p.add_argument("dirs", nargs="*", help="Stem folders containing separation_info.json")
    p.add_argument("--root", help="Scan this folder for */separation_info.json")
    p.add_argument("--csv", help="Also write results to this CSV")
    p.set_defaults(func=cmd_stems_check, parser=p)

    p = stems.add_parser("probe", help="Vocal-vs-synth probe per stem (unvalidated)")
    p.add_argument("--stems-dir", "-d", help="Directory containing stem .wav files")
    p.add_argument("--audio", "-a", help="Single audio file to analyze")
    p.add_argument("--output", "-o", help="Output JSON file path")
    p.add_argument("--segment-duration", type=float, default=5.0,
                   help="Duration of analysis segments in seconds (default: 5.0)")
    p.set_defaults(func=cmd_stems_probe, parser=p)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
