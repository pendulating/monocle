"""Fetch restaurant inspection rows from DOHMH (``43nn-pn8j``).

One Socrata endpoint, one (optionally cuisine- and borough-filtered) pull.
The raw dataset has one row per (inspection, violation) — ~296k rows for
~31k unique CAMIS as of 2026-04-28. Dedup to one-row-per-restaurant
happens in :mod:`.normalize`.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..socrata import FetchResult, fetch_socrata

__all__ = ["fetch_dohmh", "DOHMH_URL", "DOHMH_COLUMNS"]

log = logging.getLogger(__name__)

DOHMH_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"

# Every column we care about for restaurant-level metadata. Violation-level
# columns (violation_code, violation_description, critical_flag) are kept so
# the most-recent inspection's primary violation can ride along on the
# deduped row, but they are not authoritative.
DOHMH_COLUMNS: tuple[str, ...] = (
    "camis",
    "dba",
    "boro",
    "building",
    "street",
    "zipcode",
    "phone",
    "cuisine_description",
    "inspection_date",
    "action",
    "violation_code",
    "violation_description",
    "critical_flag",
    "score",
    "grade",
    "grade_date",
    "record_date",
    "inspection_type",
    "latitude",
    "longitude",
    "community_board",
    "council_district",
    "census_tract",
    "bin",
    "bbl",
    "nta",
)


def _quote_in(values: list[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def fetch_dohmh(
    *,
    cuisines: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = DOHMH_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
) -> FetchResult:
    """DOHMH inspection rows matching the optional filters.

    ``cuisines`` is matched case-insensitively against ``cuisine_description``;
    ``boroughs`` against ``boro`` (the dataset uses title-case borough names
    plus ``"0"`` for unknown). Both are ANDed; within each, values are ORed.

    Pass no filters to pull the full dataset (~296k rows / ~31k restaurants
    as of 2026-04-28).
    """
    where_parts: list[str] = []
    if cuisines:
        where_parts.append(f"upper(cuisine_description) IN ({_quote_in([c.upper() for c in cuisines])})")
    if boroughs:
        where_parts.append(f"upper(boro) IN ({_quote_in([b.upper() for b in boroughs])})")
    # We do not server-side filter on latitude/bin nullness — keeping every
    # row lets the dedup step pick a non-placeholder inspection where one
    # exists, even if the most-recent one has a missing coordinate.
    where = " AND ".join(where_parts) if where_parts else "camis IS NOT NULL"
    select = ",".join(columns)

    log.info("fetch: DOHMH — where=%s", where)
    return fetch_socrata(
        DOHMH_URL,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
    )
