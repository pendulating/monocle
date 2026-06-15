"""Frozen NYC "Dining Out NYC" outdoor-dining ``license_type`` vocabulary.

The Open Restaurants / Dining Out NYC licensing dataset (``fpeh-f7ci``) has
exactly two ``license_type`` values: ``Sidewalk`` and ``Roadway`` — the two
kinds of outdoor-dining setups a restaurant can be licensed for. The vocab is
tiny and changes very rarely; freezing it lets the CLI fail fast on typos with
suggestions, the same way :mod:`..subway.entrance_types` and
:mod:`..facdb.categorization` do.
"""

from __future__ import annotations

import difflib

__all__ = [
    "LICENSE_TYPES",
    "UnknownLicenseTypeError",
    "validate_license_types",
]


# Canonical (title-case) license_type values as published by DCWP.
LICENSE_TYPES: tuple[str, ...] = ("Sidewalk", "Roadway")


class UnknownLicenseTypeError(ValueError):
    """Raised when a ``license_type`` filter value isn't in the frozen vocab."""


def validate_license_types(values: list[str]) -> list[str]:
    """Canonicalize a list of ``license_type`` filter values (case-insensitive).

    Raises :class:`UnknownLicenseTypeError` with close-match hints on typos.
    """
    legal_map = {t.lower(): t for t in LICENSE_TYPES}
    canon: list[str] = []
    unknown: list[str] = []
    for v in values:
        c = v.strip().lower()
        if c in legal_map:
            canon.append(legal_map[c])
        else:
            unknown.append(v)
    if unknown:
        legal_canonical = sorted(legal_map.values())
        hints = []
        for u in unknown:
            matches = difflib.get_close_matches(u, legal_canonical, n=3, cutoff=0.4)
            hints.append(f"  {u!r} — did you mean: {matches}" if matches else f"  {u!r}")
        raise UnknownLicenseTypeError(
            "unknown license_type values (expected one of "
            f"{list(LICENSE_TYPES)}):\n" + "\n".join(hints)
        )
    return canon
