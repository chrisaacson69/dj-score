"""Clock formatting for labels."""


def format_time(seconds: float) -> str:
    """M:SS, minutes unbounded (3600 s -> "60:00"), as camelot's `bpm_detect` and
    `visualize_timeline` format it. `camelot_from_youtube` switches to H:MM:SS
    past an hour; that variant is reconciled when its CLI folds (ledger step 5)."""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"
