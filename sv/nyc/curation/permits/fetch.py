"""Fetch scaffold/shed permits from NYC DOB Socrata endpoints.

Two sources:

- **DOB NOW** (``w9ak-ipjd``) — modern DOB filing dataset (2020+). Scaffold and
  shed are boolean-like columns (``'1'``/``'0'``). Date field used here:
  ``first_permit_date`` (when the permit was actually issued).
- **BIS** (``ipu4-2q9a``) — legacy DOB Permit Issuance (pre-2020). Scaffold is
  implicit via ``permit_subtype`` ∈ {``SH`` (sidewalk shed), ``SD``, ``SF``
  (supported scaffold)}. Date field: ``issuance_date``.

Both filters push ``<= cutoff`` to the server (Socrata accepts ISO timestamp
comparisons even on fields that display ``MM/DD/YYYY``). Unauthenticated.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from ..socrata import FetchResult, fetch_socrata

__all__ = [
    "fetch_dob_now",
    "fetch_bis",
    "DOB_NOW_URL",
    "BIS_URL",
    "DOB_NOW_COLUMNS",
    "BIS_COLUMNS",
    "BIS_SCAFFOLD_SUBTYPES",
]

log = logging.getLogger(__name__)

DOB_NOW_URL = "https://data.cityofnewyork.us/resource/w9ak-ipjd.json"
BIS_URL = "https://data.cityofnewyork.us/resource/ipu4-2q9a.json"

BIS_SCAFFOLD_SUBTYPES: tuple[str, ...] = ("SH", "SD", "SF")

DOB_NOW_COLUMNS: tuple[str, ...] = (
    "job_filing_number",
    "filing_status",
    "filing_date",
    "first_permit_date",
    "current_status_date",
    "signoff_date",
    "latitude",
    "longitude",
    "scaffold",
    "shed",
    "borough",
    "house_no",
    "street_name",
    "block",
    "lot",
    "bin",
    "initial_cost",
    "job_type",
)

BIS_COLUMNS: tuple[str, ...] = (
    "borough",
    "bin__",
    "house__",
    "street_name",
    "job__",
    "job_type",
    "block",
    "lot",
    "work_type",
    "permit_status",
    "filing_status",
    "permit_type",
    "permit_subtype",
    "permit_sequence__",
    "filing_date",
    "issuance_date",
    "expiration_date",
    "job_start_date",
    "permit_si_no",
    "gis_latitude",
    "gis_longitude",
    "owner_s_business_name",
)


def _iso_cutoff(cutoff: str) -> str:
    """Accept 'YYYY-MM-DD' or full ISO, return full ISO end-of-day timestamp."""
    s = cutoff.strip()
    if "T" in s:
        return s
    return f"{s}T23:59:59"


def _iso_since(since: str) -> str:
    """Accept 'YYYY-MM-DD' or full ISO, return full ISO start-of-day timestamp."""
    s = since.strip()
    if "T" in s:
        return s
    return f"{s}T00:00:00"


def fetch_dob_now(
    cutoff: str,
    *,
    since: Optional[str] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = DOB_NOW_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
) -> FetchResult:
    """DOB NOW scaffold/shed filings with ``first_permit_date <= cutoff``.

    Args:
        cutoff: Upper bound on ``first_permit_date``.
        since: Optional lower bound on ``first_permit_date`` (ISO / YYYY-MM-DD).
            ``None`` → no lower bound (earliest DOB NOW row wins, ~2017).

    Filters out filings with null ``first_permit_date`` at the server so the
    "issued" contract is satisfied by construction (see design decision #1 in
    the plan). ``first_permit_date`` is a real floating timestamp on DOB NOW,
    so ISO comparisons at the server are correct.
    """
    iso_cutoff = _iso_cutoff(cutoff)
    where_parts = [
        "(scaffold='1' OR shed='1')",
        "latitude IS NOT NULL",
        "first_permit_date IS NOT NULL",
        f"first_permit_date <= '{iso_cutoff}'",
    ]
    if since is not None:
        where_parts.append(f"first_permit_date >= '{_iso_since(since)}'")
    where = " AND ".join(where_parts)
    select = ",".join(columns)
    log.info("fetch: DOB NOW — where=%s", where)
    return fetch_socrata(
        DOB_NOW_URL,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
    )


def fetch_bis(
    cutoff: str,
    *,
    since: Optional[str] = None,
    cache_path: Optional[str] = None,
    columns: Iterable[str] = BIS_COLUMNS,
    refresh: bool = False,
    limit: int = 50_000,
    subtypes: Iterable[str] = BIS_SCAFFOLD_SUBTYPES,
) -> FetchResult:
    """BIS permits with ``permit_subtype`` ∈ subtypes.

    Args:
        cutoff: Upper bound on ``issuance_date``.
        since: Optional lower bound on ``issuance_date`` (ISO / YYYY-MM-DD).
            Year-level prune applied server-side; exact clip happens client-side.

    NOTE on date filtering: ``issuance_date`` is a **plain-text** ``MM/DD/YYYY``
    column in the BIS dataset, not a Socrata floating timestamp, so the natural
    ``issuance_date <= '2025-12-31T23:59:59'`` server filter does lexicographic
    string comparison and silently passes half the rows (``'02/20/2019' <
    '2025-12-31T23:59:59'`` is True by ASCII order). We therefore:

    1. Send ``IS NOT NULL`` + ``permit_subtype`` to the server, plus
       **year-level** ``substring(issuance_date, 7, 4) BETWEEN 'SSSS' AND 'YYYY'``
       prune to cut obviously-out-of-range rows cheaply.
    2. Clip exactly to ``[since, cutoff]`` in :mod:`.scaffolding_permits`
       after the ``MM/DD/YYYY`` → datetime parse in :mod:`.normalize`.

    Correctness comes from the client-side filter; server filter is a volume
    optimization.
    """
    year_hi = _iso_cutoff(cutoff)[:4]
    subtype_list = ",".join(f"'{s}'" for s in subtypes)
    # substring() is 1-indexed in SoQL; characters 7..10 of "MM/DD/YYYY" are "YYYY".
    where_parts = [
        f"permit_subtype IN ({subtype_list})",
        "issuance_date IS NOT NULL",
        f"substring(issuance_date, 7, 4) <= '{year_hi}'",
    ]
    if since is not None:
        year_lo = _iso_since(since)[:4]
        where_parts.append(f"substring(issuance_date, 7, 4) >= '{year_lo}'")
    where = " AND ".join(where_parts)
    select = ",".join(columns)
    log.info("fetch: BIS — where=%s", where)
    return fetch_socrata(
        BIS_URL,
        where=where,
        select=select,
        cache_path=cache_path,
        limit=limit,
        refresh=refresh,
    )
