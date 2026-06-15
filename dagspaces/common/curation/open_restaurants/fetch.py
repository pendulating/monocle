"""Fetch NYC outdoor-dining licenses from Socrata (``fpeh-f7ci``).

This is the "Dining Out NYC" / permanent Open Restaurants licensing dataset
published by DCWP — the successor to the COVID-era Open Restaurants program.
Each row is a restaurant licensed to operate a ``Sidewalk`` or ``Roadway``
outdoor-dining setup. One Socrata endpoint, one date-free pull (the dataset is
small — ~1.3k issued licenses). Filters translate to a SoQL ``WHERE`` clause
against ``license_type`` and/or ``borough``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..socrata import FetchResult, fetch_socrata

__all__ = ["fetch_open_restaurants", "OPEN_RESTAURANTS_URL", "OPEN_RESTAURANTS_COLUMNS"]

log = logging.getLogger(__name__)

OPEN_RESTAURANTS_URL = "https://data.cityofnewyork.us/resource/fpeh-f7ci.json"

OPEN_RESTAURANTS_COLUMNS: tuple[str, ...] = (
    "business_legal_name",
    "assumed_name_s",
    "street",
    "city",
    "borough",
    "postcode",
    "license_type",
    "license_status",
    "license_issue_date",
    "license_expiration_date",
    "latitude",
    "longitude",
    "council_district",
    "community_board",
    "bin",
    "bbl",
    "ct2020",
    "nta2020",
)


def _quote_in(values: list[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def fetch_open_restaurants(
    *,
    license_types: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = OPEN_RESTAURANTS_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
) -> FetchResult:
    """Fetch outdoor-dining license rows matching optional filters.

    ``license_types``: e.g. ``['Sidewalk', 'Roadway']`` — exact match
        (case-insensitive). Already canonicalized by
        :func:`.license_types.validate_license_types` before getting here.
    ``boroughs``: canonical uppercase names (``MANHATTAN`` … ``STATEN ISLAND``);
        the source stores title-case, so we compare with ``upper()``.

    All filters AND together; values within a filter OR. Every row must have a
    non-null ``latitude`` (the dataset has 0 null lat/lon as of 2026-06).
    """
    where_parts: list[str] = []
    if license_types:
        where_parts.append(
            f"upper(license_type) IN ({_quote_in([t.upper() for t in license_types])})"
        )
    if boroughs:
        where_parts.append(
            f"upper(borough) IN ({_quote_in([b.upper() for b in boroughs])})"
        )
    where_parts.append("latitude IS NOT NULL")

    where = " AND ".join(where_parts)
    select = ",".join(columns)
    log.info("fetch: open-restaurants — where=%s", where)
    return fetch_socrata(
        OPEN_RESTAURANTS_URL,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
    )
