"""Dataclass <-> plain JSON-shaped values, driven by the type hints in `model.py`.

`from_plain` is strict (an unknown key raises), so a schema variant shows up as an
error instead of silently lost data. `to_plain` omits None-valued fields, matching
records where that key never existed. Field metadata {"key": ...} maps a Python
name to its JSON key (`pass_` -> "pass").
"""

from __future__ import annotations

import dataclasses
import types
import typing

import numpy as np


class CodecError(ValueError):
    """A value does not fit the dataclass it is decoded into."""


def _json_key(f: dataclasses.Field) -> str:
    return f.metadata.get("key", f.name)


def from_plain(tp, value):
    """Build `tp` from parsed JSON, recursing through dataclasses, lists, dicts
    and optionals. Scalars pass through untouched so values round-trip exactly."""
    if value is None:
        return None
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        (inner,) = [t for t in args if t is not type(None)]
        return from_plain(inner, value)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise CodecError(f"{tp.__name__}: expected an object, got {type(value).__name__}")
        hints = typing.get_type_hints(tp)
        fields = {_json_key(f): f for f in dataclasses.fields(tp)}
        unknown = set(value) - set(fields)
        if unknown:
            raise CodecError(f"{tp.__name__}: unknown keys {sorted(unknown)}")
        return tp(**{f.name: from_plain(hints[f.name], value[k]) for k, f in fields.items() if k in value})
    if origin is list:
        return [from_plain(args[0], v) for v in value]
    if origin is dict:
        return {k: from_plain(args[1], v) for k, v in value.items()}
    if tp is np.ndarray:
        return np.asarray(value)
    return value


def to_plain(value):
    """Inverse of `_decode`. None-valued dataclass fields are omitted, matching
    caches where that key never existed."""
    if dataclasses.is_dataclass(value):
        return {_json_key(f): to_plain(v) for f in dataclasses.fields(value)
                if (v := getattr(value, f.name)) is not None}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
