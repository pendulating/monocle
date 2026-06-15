"""Paginated Socrata (SODA) fetch with on-disk parquet cache.

Unauthenticated — no app token. The existing scaffolding-compliance notebook
has been running this pattern against NYC Open Data without issues at a 50k
page size.

Usage::

    from dagspaces.common.curation.socrata import fetch_socrata, FetchResult
    r = fetch_socrata(
        url="https://data.cityofnewyork.us/resource/w9ak-ipjd.json",
        where="(scaffold='1' OR shed='1') AND latitude IS NOT NULL",
        select="job_filing_number,bin,first_permit_date,...",
        cache_path="curation/.../dob_now_raw.parquet",
    )
    print(r.df.shape, r.pages, r.truncated_likely)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import polars as pl
import requests

__all__ = ["fetch_socrata", "FetchResult", "SocrataError"]

log = logging.getLogger(__name__)


class SocrataError(RuntimeError):
    """Raised on a non-recoverable Socrata failure (HTTP error after retries)."""


@dataclass
class FetchResult:
    df: pl.DataFrame
    pages: int
    total_rows: int
    page_rows: list[int] = field(default_factory=list)
    truncated_likely: bool = False  # last page == limit → probably more rows
    elapsed_s: float = 0.0
    cached: bool = False
    where: Optional[str] = None
    select: Optional[str] = None


def _request_with_retry(
    url: str,
    params: dict,
    timeout: int,
    max_retries: int,
    backoff_s: float,
) -> list[dict]:
    """GET with exponential backoff on transient failures.

    4xx responses (except 429) are surfaced immediately without retry — a 400
    bad request won't become valid by waiting."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise SocrataError(
                    f"socrata: {r.status_code} from {url} (body: {r.text[:300]}); "
                    "not retrying a client error"
                )
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} transient", response=r)
            r.raise_for_status()
            return r.json()
        except SocrataError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = backoff_s * (2 ** attempt)
            log.warning(
                "socrata: %s on attempt %d/%d (offset=%s), sleeping %.1fs",
                type(exc).__name__, attempt + 1, max_retries + 1,
                params.get("$offset"), sleep_s,
            )
            time.sleep(sleep_s)
    raise SocrataError(f"socrata: giving up after {max_retries + 1} attempts: {last_exc}")


def fetch_socrata(
    url: str,
    *,
    where: str,
    select: str,
    cache_path: Optional[str] = None,
    limit: int = 50_000,
    order: str = ":id",
    timeout: int = 120,
    max_retries: int = 4,
    backoff_s: float = 2.0,
    refresh: bool = False,
) -> FetchResult:
    """Fetch a Socrata endpoint with pagination and parquet caching.

    Args:
        url: Full resource URL including ``.json`` suffix.
        where: SoQL ``$where`` clause.
        select: Comma-joined ``$select`` column list.
        cache_path: If given, read from / write to this parquet file.
        limit: Page size. 50k is the Socrata unauthenticated upper bound.
        order: SoQL ``$order`` — ``:id`` gives a stable deterministic order.
        refresh: If True, ignore an existing cache file and re-fetch.

    Returns a :class:`FetchResult`. ``truncated_likely`` is True when the final
    page returned exactly ``limit`` rows, which is a strong signal Socrata cut
    us off before the result set ended.
    """
    if cache_path and not refresh and os.path.isfile(cache_path):
        df = pl.read_parquet(cache_path)
        log.info("socrata: loaded %d cached rows from %s", df.height, cache_path)
        return FetchResult(
            df=df,
            pages=0,
            total_rows=df.height,
            cached=True,
            where=where,
            select=select,
        )

    t0 = time.monotonic()
    all_rows: list[dict] = []
    page_rows: list[int] = []
    offset = 0
    while True:
        params = {
            "$where": where,
            "$select": select,
            "$limit": limit,
            "$offset": offset,
            "$order": order,
        }
        batch = _request_with_retry(url, params, timeout, max_retries, backoff_s)
        n = len(batch)
        page_rows.append(n)
        log.info("socrata: fetched %d rows at offset=%d (cum=%d)",
                 n, offset, len(all_rows) + n)
        if n == 0:
            break
        all_rows.extend(batch)
        offset += limit
        if n < limit:
            break

    df = pl.DataFrame(all_rows) if all_rows else pl.DataFrame()
    elapsed = time.monotonic() - t0

    truncated_likely = len(page_rows) > 0 and page_rows[-1] == limit
    result = FetchResult(
        df=df,
        pages=len(page_rows),
        total_rows=df.height,
        page_rows=page_rows,
        truncated_likely=truncated_likely,
        elapsed_s=elapsed,
        where=where,
        select=select,
    )

    if cache_path and df.height > 0:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        df.write_parquet(cache_path)
        log.info("socrata: cached %d rows to %s", df.height, cache_path)

    log.info(
        "socrata: done — %d rows in %d pages, %.1fs elapsed%s",
        df.height, len(page_rows), elapsed,
        " (TRUNCATED LIKELY)" if truncated_likely else "",
    )
    return result
