"""Rekordbox XML export via pyrekordbox (the optional `rekordbox` extra).

Converted from camelot `rekordbox_export.py`: track metadata, the tempo grid, and
the cue plan from `domain.cues`. All exports accumulate in one XML file, so
Rekordbox is pointed at it once (Preferences > Advanced > rekordbox xml).
"""

from __future__ import annotations

from pathlib import Path

from dj_score.domain.cues import MAX_HOT_CUES, plan_cues
from dj_score.domain.model import TrackAnalysis

# Rekordbox's own key spellings (sharps where it uses them)
CAMELOT_TO_TONALITY = {
    "1A": "Abm", "2A": "Ebm", "3A": "Bbm", "4A": "Fm",
    "5A": "Cm",  "6A": "Gm",  "7A": "Dm",  "8A": "Am",
    "9A": "Em",  "10A": "Bm", "11A": "F#m", "12A": "Dbm",
    "1B": "B",   "2B": "F#",  "3B": "Db",  "4B": "Ab",
    "5B": "Eb",  "6B": "Bb",  "7B": "F",   "8B": "C",
    "9B": "G",   "10B": "D",  "11B": "A",  "12B": "E",
}


class RekordboxXml:
    """`RekordboxExporter` writing to one accumulating XML file at `path`."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _open(self):
        from pyrekordbox.rbxml import RekordboxXml as _Xml
        if self.path.exists():
            return _Xml(path=str(self.path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _Xml(name="camelot_from_youtube", version="1.0.0")

    def export_track(self, audio_path: Path, analysis: TrackAnalysis,
                     include_cues=True, max_hot_cues=MAX_HOT_CUES) -> int:
        """Add or replace one track; returns the number of tracks now in the file."""
        xml = self._open()
        build_track(xml, audio_path, analysis, include_cues, max_hot_cues)
        xml.save(str(self.path))
        return len(xml.get_tracks())

    def export_set(self, tracks: list[tuple[Path, TrackAnalysis]], playlist_name="Camelot Set") -> int:
        """Add or replace several tracks and (re)create a playlist of them, in order."""
        xml = self._open()
        ids = [build_track(xml, audio_path, analysis).TrackID for audio_path, analysis in tracks]
        playlist = xml.add_playlist(playlist_name)
        for tid in ids:
            # The key is the TrackID. camelot passed the Track object, so its playlists
            # held "<Track(Location=...)>" keys that resolve to nothing.
            playlist.add_track(tid)
        xml.save(str(self.path))
        return len(xml.get_tracks())


def _next_track_id(xml):
    existing = xml.get_track_ids()
    return max(existing) + 1 if existing else 1


def build_track(xml, audio_path, a: TrackAnalysis, include_cues=True, max_hot_cues=MAX_HOT_CUES):
    """Write one track into an open pyrekordbox document, replacing any entry for
    the same file. Takes the document as a parameter so it can be inspected."""
    audio_path = Path(audio_path).resolve()
    for t in xml.get_tracks():
        if Path(t.Location).resolve() == audio_path:
            xml.remove_track(t)
            break

    # The ID goes in with the track. camelot set it afterwards, which left pyrekordbox's
    # ID index one behind the file, so a set that re-exported known tracks crashed
    # (XmlDuplicateError). Fresh collections now number from 1, not 2.
    track = xml.add_track(str(audio_path), TrackID=_next_track_id(xml))
    track.set("Name", audio_path.stem)
    track.set("TotalTime", int(a.duration))
    track.set("AverageBpm", round(float(a.grid.tempo), 2))

    summary = a.key.key_summary if a.key else None
    if summary and summary.dominant.camelot:
        camelot = summary.dominant.camelot
        tonality = CAMELOT_TO_TONALITY.get(camelot, "")
        if tonality:
            track.set("Tonality", tonality)
        track.set("Comments", f"{camelot} {summary.dominant.key_name}")

    tempo = a.grid.tempo
    if len(a.grid.segments) > 1:
        for seg in a.grid.segments:
            track.add_tempo(Inizio=round(float(seg.start_time), 3), Bpm=round(float(seg.bpm), 2),
                            Metro="4/4", Battito=1)
    elif tempo:
        first_beat = float(a.grid.measures[0].start) if a.grid.measures else 0.0
        track.add_tempo(Inizio=round(first_beat, 3), Bpm=round(float(tempo), 2), Metro="4/4", Battito=1)

    if include_cues:
        plan = plan_cues(a.events.events if a.events else [], summary.key_changes if summary else [],
                         max_hot_cues)
        for cue in plan.hot + plan.memory:
            track.add_mark(Name=cue.name, Type="cue", Start=cue.start, Num=cue.num)

    return track
