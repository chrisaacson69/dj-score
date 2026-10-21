"""Clock formatting for labels."""


def format_time(seconds: float) -> str:
    """M:SS, minutes unbounded (3600 s -> "60:00"), as camelot's `bpm_detect` and
    `visualize_timeline` format it. `camelot_from_youtube` switches to H:MM:SS
    past an hour: that is `format_clock`, which its timeline report uses."""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def format_clock(seconds: float) -> str:
    """M:SS under an hour, H:MM:SS from an hour (camelot_from_youtube's timeline report)."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
