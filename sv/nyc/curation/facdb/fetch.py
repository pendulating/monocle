"""Fetch facilities from NYC DCP Facilities Database (FacDB, ``ji82-xba5``).

One Socrata endpoint, one date-free pull. Filters translate to a SoQL
``WHERE`` clause against any/all of the 4 hierarchy columns
(``facdomain``, ``facgroup``, ``facsubgrp``, ``factype``). Values are
already canonicalized (uppercase, trimmed) by
:func:`.categorization.validate_filter_values` before getting here.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..socrata import FetchResult, fetch_socrata

__all__ = ["fetch_facdb", "FACDB_URL", "FACDB_COLUMNS"]

log = logging.getLogger(__name__)

FACDB_URL = "https://data.cityofnewyork.us/resource/ji82-xba5.json"

FACDB_COLUMNS: tuple[str, ...] = (
    "uid",
    "facname",
    "address",
    "city",
    "zipcode",
    "boro",
    "borocode",
    "bin",
    "bbl",
    "latitude",
    "longitude",
    "xcoord",
    "ycoord",
    "facdomain",
    "facgroup",
    "facsubgrp",
    "factype",
    "capacity",
    "captype",
    "opname",
    "opabbrev",
    "optype",
    "overagency",
    "overabbrev",
    "overlevel",
    "servarea",
    "cd",
    "council",
    "nta2020",
    "ct2020",
    "schooldist",
    "policeprct",
    "datasource",
)


def _in_clause(col: str, values: list[str]) -> str:
    # Each value is a trusted, dict-validated uppercase string; still, escape
    # single quotes defensively in case a future dictionary entry contains one.
    quoted = ",".join("'" + v.replace("'", "''") + "'" for v in values)
    return f"upper({col}) IN ({quoted})"


def fetch_facdb(
    *,
    facdomain: Optional[Iterable[str]] = None,
    facgroup: Optional[Iterable[str]] = None,
    facsubgrp: Optional[Iterable[str]] = None,
    factype: Optional[Iterable[str]] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = FACDB_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
) -> FetchResult:
    """FacDB facilities matching the filter (AND across levels).

    Each level is optional; within a level, values are ORed. The four
    levels are ANDed together. Pass no filters to pull the full database
    (~34.7k rows as of 25v2).
    """
    where_parts: list[str] = []
    if facdomain:
        where_parts.append(_in_clause("facdomain", list(facdomain)))
    if facgroup:
        where_parts.append(_in_clause("facgroup", list(facgroup)))
    if facsubgrp:
        where_parts.append(_in_clause("facsubgrp", list(facsubgrp)))
    if factype:
        where_parts.append(_in_clause("factype", list(factype)))

    where = " AND ".join(where_parts) if where_parts else "latitude IS NOT NULL"
    select = ",".join(columns)
    log.info("fetch: FacDB — where=%s", where)
    return fetch_socrata(
        FACDB_URL,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
    )
