"""The human-readable reports the CLI prints, as camelot's CLIs print them."""

from __future__ import annotations

import numpy as np

from dj_score.domain.beatgrid import compute_span_bpm, quantization_report
from dj_score.domain.timefmt import format_time


def bpm_report(name, grid, frame_duration, tolerance, beats_per_bar) -> str:
    beat_times, measures, segments = grid.beat_times, grid.measures, grid.segments
    quant = quantization_report(np.diff(beat_times), frame_duration)
    out = ["", f"=== BPM Analysis: {name} ===", "",
           "Frame Quantization:",
           f"  Hop frame duration: {quant.frame_duration_ms:.2f}ms",
           f"  Unique beat intervals: {quant.unique_interval_count} (step: {quant.interval_step_ms:.2f}ms)",
           f"  Matches hop size: {quant.matches_frame_hop}", "",
           "BPM at different resolutions:"]
    for s in [1, 2, 4, 8, 16]:
        sd = compute_span_bpm(beat_times, s)
        if len(sd.bpms) > 0:
            out.append(f"  {s:2d}-beat span: {sd.median:7.2f} BPM  (std: {sd.std:.3f})")
    out.append("")

    out.append(f"BPM Segments ({len(segments)}, tolerance={tolerance} BPM):")
    if len(segments) > 10:
        out += ["  Many segments detected - likely frame quantization noise.",
                f"  Try increasing --tolerance (current: {tolerance})",
                "  Showing first 10:"]
    out += [f"  {'#':<4} {'Bars':<12} {'Time':<15} {'BPM':>7}  {'Measures':>8}", f"  {'-'*55}"]
    for i, seg in enumerate(segments[:10], 1):
        t = f"{format_time(seg.start_time)}-{format_time(seg.end_time)}"
        b = f"{seg.start_measure}-{seg.end_measure}"
        out.append(f"  {i:<4} {b:<12} {t:<15} {seg.bpm:>7.1f}  {seg.measure_count:>8}")
    if len(segments) > 10:
        out.append(f"  ... ({len(segments) - 10} more)")
    out.append("")

    out.append(f"Per-Measure BPM ({len(measures)} measures, {beats_per_bar}/4 time):")
    for row_start in range(0, min(len(measures), 64), 8):
        row = measures[row_start:row_start + 8]
        out.append("  Bar: " + "  ".join(f"{m.measure_num:>3}" for m in row))
        out.append("  BPM: " + "  ".join(f"{m.bpm:>6.1f}" for m in row))
        out.append("")
    if len(measures) > 64:
        out += [f"  ... ({len(measures) - 64} more measures)", ""]
    return "\n".join(out)


def key_report(name, tempo, k, phrase_bars) -> str:
    summary, segments, phrases = k.key_summary, k.key_segments, k.phrases
    out = ["", f"=== Key Analysis: {name} ===", ""]
    if summary:
        d = summary.dominant
        out.append(f"Dominant Key: {d.key_name}  ({d.camelot})  coverage: {d.coverage*100:.1f}%")
    out += [f"Tempo: {tempo:.1f} BPM (librosa)",
            f"Phrases: {len(phrases)} ({phrase_bars}-bar),  Segments: {len(segments)}", ""]

    if summary and summary.key_changes:
        out.append("Key Changes:")
        for kc in summary.key_changes:
            out.append(f"  Bar {kc.bar:>4}  ({format_time(kc.time)})  "
                       f"{kc.from_camelot} ({kc.from_key}) -> {kc.to_camelot} ({kc.to_key})")
    else:
        out.append("No key changes detected.")
    out.append("")

    out += ["Key Segments:",
            f"  {'#':<3} {'Bars':<12} {'Time':<14} {'Key':<12} {'Camelot':<8} {'Conf':>5}  {'Phrases':>7}",
            f"  {'-'*68}"]
    for i, seg in enumerate(segments, 1):
        t = f"{format_time(seg.start_time)}-{format_time(seg.end_time)}"
        b = f"{seg.start_measure}-{seg.end_measure}"
        out.append(f"  {i:<3} {b:<12} {t:<14} {seg.key_name:<12} {seg.camelot:<8} "
                   f"{seg.confidence_mean:>5.2f}  {seg.phrase_count:>7}")
    out.append("")

    out.append(f"Per-Phrase Key ({len(phrases)} phrases, {phrase_bars}-bar):")
    for row_start in range(0, min(len(phrases), 48), 8):
        row = phrases[row_start:row_start + 8]
        out.append("  Bar:  " + "  ".join(f"{p.start_measure:>4}" for p in row))
        out.append("  Key:  " + "  ".join(f"{p.camelot:>4}" for p in row))
        out.append("  Conf: " + "  ".join(f"{p.confidence:>4.2f}" for p in row))
        out.append("")
    if len(phrases) > 48:
        out += [f"  ... ({len(phrases) - 48} more phrases)", ""]
    return "\n".join(out)


def events_report(name, tempo, duration, ev, phrase_bars) -> str:
    events = ev.events
    out = ["", f"=== Event Detection: {name} ===", f"Mode: {ev.event_mode}", "",
           f"Tempo: {tempo:.1f} BPM", f"Duration: {format_time(duration)}",
           f"Phrases: {len(ev.event_phrases)} ({phrase_bars}-bar)",
           f"Events detected: {ev.event_summary.event_count}", ""]
    if not events:
        return "\n".join(out + ["No significant structural events detected.", ""])
    if ev.event_mode == "stems":
        out += [f"{'#':>3} {'Bar':>5} {'Time':>8} {'Score':>6} {'Type':<14} {'Description'}", f"  {'-'*65}"]
        for i, e in enumerate(events, 1):
            out.append(f"{i:>3} {e.bar:>5} {e.time_fmt:>8} {e.score:>6.3f} {e.type:<14} {e.description}")
    else:
        out += [f"{'#':>3} {'Bar':>5} {'Time':>8} {'Score':>6} {'Pass':>4} {'Type':<12} {'Description'}",
                f"  {'-'*75}"]
        for i, e in enumerate(events, 1):
            p = e.pass_ if e.pass_ is not None else "?"
            out.append(f"{i:>3} {e.bar:>5} {e.time_fmt:>8} {e.score:>6.3f} {p:>4} {e.type or '':<12} {e.description}")
    out.append("")
    return "\n".join(out)
