"""Frozen DOHMH cuisine_description vocabulary.

The DOHMH dataset's only useful categorical filter is ``cuisine_description``
(91 distinct values as of 2026-04-28 — e.g. "American", "Pizza", "Mexican",
"Coffee/Tea"). The vocabulary changes very rarely; freezing it lets the
CLI fail fast on typos with suggestions, the same way
:mod:`..facdb.categorization` does for FacDB hierarchy values.

The runtime source of truth is :data:`CUISINES_PATH` (``cuisines.json``).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

__all__ = [
    "CUISINES_PATH",
    "UnknownCuisineError",
    "load_cuisines",
    "validate_cuisines",
]


CUISINES_PATH = os.path.join(os.path.dirname(__file__), "cuisines.json")


class UnknownCuisineError(ValueError):
    """Raised when a ``cuisine_description`` filter value is not in the
    frozen vocabulary."""


@lru_cache(maxsize=1)
def load_cuisines() -> dict[str, Any]:
    """Return the frozen cuisine vocab dict.

    Keys: ``source``, ``fetched_at``, ``cuisines`` (sorted list of strings,
    matching the case used by Socrata).
    """
    with open(CUISINES_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _legal_lower() -> dict[str, str]:
    """Lowercase → canonical-case map for case-insensitive validation."""
    return {c.lower(): c for c in load_cuisines()["cuisines"]}


def validate_cuisines(values: list[str]) -> list[str]:
    """Canonicalize (case-insensitive) and validate cuisine values.

    Raises :class:`UnknownCuisineError` on typos so CLI users hit the
    fail-fast path rather than silently pulling zero rows.
    """
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
        raise UnknownCuisineError(
            "unknown cuisine_description values (not in frozen DOHMH vocab):\n"
            + "\n".join(hints)
        )
    return canon
