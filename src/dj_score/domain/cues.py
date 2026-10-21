"""Which events become Rekordbox hot cues and which become memory cues.

The tier logic out of camelot `rekordbox_export._build_track`, without the XML;
`adapters/rekordbox_xml` writes the plan. Events co-timed with a key change get
one merged "KEY 1A >> 8A | <type>" label; other key changes become cues of their own.
"""

from __future__ import annotations

from dataclasses import dataclass

from dj_score.domain.model import Event, KeyChange

MAX_HOT_CUES = 8  # Rekordbox A-H
MAX_MEMORY_CUES = 10
LABEL_LIMIT = 40

# Tier 1: structural moments a DJ cues to. Tier 2: fills, which signal transitions.
HOT_CUE_TIER1 = {
    "drop", "breakdown", "build", "bass_in", "bass_out",
    "melodic_in", "melodic_out", "vocal_in", "vocal_out",
    "transition",
}
HOT_CUE_TIER2 = {"fill"}


@dataclass
class Cue:
    name: str
    start: float
    """Seconds, rounded to ms."""
    num: int
    """Hot-cue slot 0-7, or -1 for a memory cue."""


@dataclass
class CuePlan:
    hot: list[Cue]
    memory: list[Cue]


@dataclass
class _Candidate:
    time: float
    type: str | None
    label: str
    has_key_change: bool
    cue: str = "auto"


def event_label(ev: Event) -> str:
    if ev.description:
        return f"{ev.type}: {ev.description}"
    return f"{ev.type} @{ev.bar}" if ev.bar else ev.type


def _candidates(events: list[Event], key_changes: list[KeyChange]) -> list[_Candidate]:
    key_change_map = {round(float(kc.time), 3): kc for kc in key_changes}

    merged, used_kc_times = [], set()
    for ev in sorted(events, key=lambda e: e.time):
        t = round(float(ev.time), 3)
        kc = key_change_map.get(t)
        if kc:
            used_kc_times.add(t)
            suffix = f" | {ev.type}" if ev.type else ""
            label = f"KEY {kc.from_camelot} >> {kc.to_camelot}{suffix}"
        else:
            label = event_label(ev)
        merged.append(_Candidate(ev.time, ev.type, label, bool(kc), "auto" if ev.cue is None else ev.cue))

    for t, kc in key_change_map.items():
        if t not in used_kc_times:
            label = f"KEY {kc.from_camelot} >> {kc.to_camelot}"
            if kc.bar:
                label += f" @{kc.bar}"
            merged.append(_Candidate(t, "key_change", label, True))

    merged.sort(key=lambda c: float(c.time))
    return merged


def plan_cues(events: list[Event], key_changes: list[KeyChange], max_hot_cues=MAX_HOT_CUES) -> CuePlan:
    """Manual overrides first (`Event.cue`), then tier 1, then tier 2 fill the hot
    slots; everything else competes for memory slots, key changes first."""
    merged = _candidates(events, key_changes)

    forced_hot = [c for c in merged if c.cue == "hot"]
    forced_mem = [c for c in merged if c.cue == "memory"]
    auto = [c for c in merged if c.cue == "auto"]

    tier1 = [c for c in auto if c.type in HOT_CUE_TIER1]
    tier2 = [c for c in auto if c.type in HOT_CUE_TIER2]
    rest = [c for c in auto if c.type not in HOT_CUE_TIER1 and c.type not in HOT_CUE_TIER2]

    hot = list(forced_hot)
    auto_memory = []
    for c in tier1 + tier2:
        (hot if len(hot) < max_hot_cues else auto_memory).append(c)
    auto_memory.extend(rest)
    hot.sort(key=lambda c: float(c.time))

    pool = sorted(forced_mem + auto_memory, key=lambda c: float(c.time))
    memory = [c for c in pool if c.has_key_change][:MAX_MEMORY_CUES]
    memory.extend([c for c in pool if not c.has_key_change][:MAX_MEMORY_CUES - len(memory)])
    memory.sort(key=lambda c: float(c.time))

    def cue(c, num):
        return Cue(name=c.label[:LABEL_LIMIT], start=round(float(c.time), 3), num=num)

    return CuePlan(hot=[cue(c, i) for i, c in enumerate(hot)], memory=[cue(c, -1) for c in memory])
