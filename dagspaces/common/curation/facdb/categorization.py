"""FacDB facility categorization hierarchy (4-level).

Loaded from ``categorization.json``, which was frozen from the NYC DCP
``facilities_data_dictionary.xlsx`` (Categorization sheet). The xlsx lives
at ``curation/facilities_data_dictionary.xlsx`` for reference; the JSON
is the runtime source of truth so we don't depend on an external path or
``openpyxl`` at import time.

Hierarchy (highest to lowest granularity):

    facdomain   (7)      — e.g. "EDUCATION, CHILD WELFARE, AND YOUTH"
    facgroup    (25)     — e.g. "SCHOOLS (K-12)"
    facsubgrp   (72)     — e.g. "HIGH SCHOOLS"
    factype     (609)    — e.g. "PUBLIC HIGH SCHOOL"
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

__all__ = [
    "load_categorization",
    "validate_filter_values",
    "HIERARCHY_LEVELS",
    "CATEGORIZATION_PATH",
    "UnknownCategoryError",
]

HIERARCHY_LEVELS: tuple[str, ...] = ("facdomain", "facgroup", "facsubgrp", "factype")
CATEGORIZATION_PATH = os.path.join(os.path.dirname(__file__), "categorization.json")


class UnknownCategoryError(ValueError):
    """Raised when a filter value doesn't appear in the frozen hierarchy."""


@lru_cache(maxsize=1)
def load_categorization() -> dict[str, Any]:
    """Return the frozen hierarchy dict.

    Keys: ``version``, ``domains``, ``groups``, ``subgroups``, ``types``,
    ``hierarchy`` (nested dict), ``rows`` (flat list of every combo).
    """
    with open(CATEGORIZATION_PATH, "r") as f:
        return json.load(f)


def _level_values(level: str) -> set[str]:
    if level not in HIERARCHY_LEVELS:
        raise ValueError(
            f"unknown hierarchy level {level!r}; pick one of {HIERARCHY_LEVELS}"
        )
    key = {
        "facdomain": "domains",
        "facgroup": "groups",
        "facsubgrp": "subgroups",
        "factype": "types",
    }[level]
    return set(load_categorization()[key])


def validate_filter_values(level: str, values: list[str]) -> list[str]:
    """Canonicalize (strip + uppercase) and validate values against the
    dictionary. Raises :class:`UnknownCategoryError` on typos so CLI users
    hit the fail-fast path rather than silently pulling zero rows."""
    legal = _level_values(level)
    canon: list[str] = []
    unknown: list[str] = []
    for v in values:
        c = v.strip().upper()
        if c in legal:
            canon.append(c)
        else:
            unknown.append(v)
    if unknown:
        # Try to offer up to 3 near-misses per unknown value
        import difflib
        hints = []
        for u in unknown:
            matches = difflib.get_close_matches(u.upper(), sorted(legal), n=3, cutoff=0.5)
            hints.append(f"  {u!r} — did you mean: {matches}" if matches else f"  {u!r}")
        raise UnknownCategoryError(
            f"unknown {level} values (not in frozen FacDB dictionary):\n"
            + "\n".join(hints)
        )
    return canon
