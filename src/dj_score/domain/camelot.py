"""The Camelot wheel: pitch class + mode -> key name and Camelot code.

Folded from two value-identical copies in camelot (`key_detect.py` and
`camelot_from_youtube.py`, flat spellings). Rekordbox's own spellings ("F#m")
belong to the Rekordbox XML adapter, not here.
"""

from __future__ import annotations

import re

PC_TO_NOTE = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

CAMELOT_MAJOR = {
    "B": "1B", "Gb": "2B", "Db": "3B", "Ab": "4B", "Eb": "5B", "Bb": "6B",
    "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B",
}

CAMELOT_MINOR = {
    "Ab": "1A", "Eb": "2A", "Bb": "3A", "F": "4A", "C": "5A", "G": "6A",
    "D": "7A", "A": "8A", "E": "9A", "B": "10A", "Gb": "11A", "Db": "12A",
}

CAMELOT_CODES = [f"{n}A" for n in range(1, 13)] + [f"{n}B" for n in range(1, 13)]


def key_name(tonic_pc: int, mode: str) -> str:
    return f"{PC_TO_NOTE[int(tonic_pc)]} {mode}"


def to_camelot(tonic_pc: int, mode: str) -> str:
    """Camelot code for a key, "?" if the wheel has no entry."""
    table = CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR
    return table.get(PC_TO_NOTE[int(tonic_pc)], "?")


def camelot_number(code: str | None) -> int | None:
    """The wheel position (1-12) of a code like "8A", ignoring the letter."""
    if not code:
        return None
    m = re.fullmatch(r"(\d+)[AB]", code.strip())
    return int(m.group(1)) if m else None
