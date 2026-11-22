"""Frozen MTA subway ``entrance_type`` vocabulary.

The dataset's ``entrance_type`` column has 13 distinct values as of
2026-04-28 — Stair, Elevator, Escalator, Station House, Easement -
Street, Easement - Passage, Ramp, Underpass, Walkway, Overpass, Stair/
Escalator, Stair/Ramp, Stair/Ramp/Walkway. The vocab changes very rarely;
freezing it lets the CLI fail fast on typos with suggestions, the same
way :mod:`..facdb.categorization` does for FacDB hierarchy values.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

__all__ = [
    "ENTRANCE_TYPES_PATH",
    "UnknownEntranceTypeError",
    "load_entrance_types",
    "validate_entrance_types",
]


ENTRANCE_TYPES_PATH = os.path.join(os.path.dirname(__file__), "entrance_types.json")


class UnknownEntranceTypeError(ValueError):
    """Raised when an ``entrance_type`` filter value isn't in the frozen vocab."""


@lru_cache(maxsize=1)
def load_entrance_types() -> dict[str, Any]:
    with open(ENTRANCE_TYPES_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _legal_lower() -> dict[str, str]:
    return {t.lower(): t for t in load_entrance_types()["entrance_types"]}


def validate_entrance_types(values: list[str]) -> list[str]:
    legal_map = _legal_lower()
    canon: list[str] = []
    unknown: list[str] = []
    for v in values:
        c = v.strip().lower()
        if c in legal_map:
            canon.append(legal_map[c])
        else:
            unknown.append(v)
    if unknown:
        import difflib
        legal_canonical = sorted(legal_map.values())
        hints = []
        for u in unknown:
            matches = difflib.get_close_matches(u, legal_canonical, n=3, cutoff=0.5)
            hints.append(f"  {u!r} — did you mean: {matches}" if matches else f"  {u!r}")
        raise UnknownEntranceTypeError(
            "unknown entrance_type values (not in frozen MTA vocab):\n"
            + "\n".join(hints)
        )
    return canon
