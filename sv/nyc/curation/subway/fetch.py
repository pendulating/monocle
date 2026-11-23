"""Fetch subway entrances/exits from NY State Open Data (``i9wp-a4ja``).

This is a **state** Socrata host (``data.ny.gov``), not the city one
(``data.cityofnewyork.us``). The SoQL is the same, and the framework reader
``dagspaces.sources.opendata.socrata.fetch_socrata`` reads both hosts.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from dagspaces.sources.opendata.socrata import FetchResult, fetch_socrata

from .. import NY_STATE_OPEN_DATA

__all__ = ["fetch_subway_entrances", "SUBWAY_DATASET", "SUBWAY_COLUMNS"]

log = logging.getLogger(__name__)

#: The Socrata dataset id (on NY_STATE_OPEN_DATA).
SUBWAY_DATASET = "i9wp-a4ja"

SUBWAY_COLUMNS: tuple[str, ...] = (
    "division",
    "line",
    "borough",
    "stop_name",
    "complex_id",
    "constituent_station_name",
    "station_id",
    "gtfs_stop_id",
    "daytime_routes",
    "entrance_type",
    "entry_allowed",
    "exit_allowed",
    "entrance_latitude",
    "entrance_longitude",
)


def _quote_in(values: list[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in values)


def fetch_subway_entrances(
    *,
    entrance_types: Optional[Iterable[str]] = None,
    divisions: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    routes: Optional[Iterable[str]] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = SUBWAY_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
) -> FetchResult:
    """Fetch subway entrance rows matching optional filters.

    ``entrance_types``: e.g. ``['Stair', 'Elevator']`` — exact match (case-insensitive).
    ``divisions``: e.g. ``['IRT', 'BMT']``.
    ``boroughs``: dataset stores single-letter codes (``M``, ``B``, ``Bx``, ``Q``, ``SI``).
    ``routes``: list of route IDs (``['L', '4', 'Q']``); matched as **whole tokens**
        in the space-separated ``daytime_routes`` column. ``'4'`` does not match
        ``'42'`` because the column has no two-digit subway routes.

    All filters AND together; values within a filter OR.
    """
    where_parts: list[str] = []
    if entrance_types:
        where_parts.append(
            f"upper(entrance_type) IN ({_quote_in([t.upper() for t in entrance_types])})"
        )
    if divisions:
        where_parts.append(
            f"upper(division) IN ({_quote_in([d.upper() for d in divisions])})"
        )
    if boroughs:
        where_parts.append(
            f"upper(borough) IN ({_quote_in([b.upper() for b in boroughs])})"
        )
    if routes:
        # Wrap the routes column in spaces so " 4 " matches "4" but not "40".
        # Socrata SoQL has like; chain ORs across the requested routes.
        ors = []
        for r in routes:
            esc = r.replace("'", "''")
            ors.append(f"upper(' ' || daytime_routes || ' ') LIKE '% {esc.upper()} %'")
        where_parts.append("(" + " OR ".join(ors) + ")")

    where = (
        " AND ".join(where_parts)
        if where_parts
        else "entrance_latitude IS NOT NULL"
    )
    select = ",".join(columns)
    log.info("fetch: subway entrances — where=%s", where)
    return fetch_socrata(
        NY_STATE_OPEN_DATA,
        SUBWAY_DATASET,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
        frame="polars",
    )
