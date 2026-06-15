"""Normalize DOB NOW + BIS responses to a common flat schema.

Canonical columns (output of :func:`normalize_to_common`):

    source            str  — "dob_now" / "bis"
    permit_id         str  — dob_now: job_filing_number; bis: permit_si_no
    job_id            str  — dob_now: job_filing_number; bis: job__
    bin               str  — API-provided Building Identification Number
    borough           str  — uppercase: MANHATTAN / BROOKLYN / QUEENS / BRONX / STATEN ISLAND
    issue_date        datetime (UTC)
    expiration_date   datetime (UTC) or null  (BIS only)
    signoff_date      datetime (UTC) or null  (DOB NOW only)
    scaffold_type     str — scaffold / shed / both
    permit_status     str (raw)
    permit_subtype    str (BIS only)
    address           str — "{house_no} {street_name}"
    block, lot        str
    job_type          str
    initial_cost      float  (DOB NOW only; BIS rows null)
    raw_latitude      float
    raw_longitude     float
"""

from __future__ import annotations

import logging
from typing import Optional

import polars as pl

__all__ = [
    "normalize_dob_now",
    "normalize_bis",
    "normalize_to_common",
    "COMMON_COLUMNS",
    "BOROUGH_MAP",
]

log = logging.getLogger(__name__)


COMMON_COLUMNS: tuple[str, ...] = (
    "source",
    "permit_id",
    "job_id",
    "bin",
    "borough",
    "issue_date",
    "expiration_date",
    "signoff_date",
    "scaffold_type",
    "permit_status",
    "permit_subtype",
    "address",
    "block",
    "lot",
    "job_type",
    "initial_cost",
    "raw_latitude",
    "raw_longitude",
)


BOROUGH_MAP: dict[str, str] = {
    # DOB NOW already emits uppercase canonical names — passthrough.
    "MANHATTAN": "MANHATTAN",
    "BROOKLYN": "BROOKLYN",
    "QUEENS": "QUEENS",
    "BRONX": "BRONX",
    "STATEN ISLAND": "STATEN ISLAND",
    # BIS also emits uppercase but drop any trailing/leading whitespace in normalize.
}


def _borough_norm(s: pl.Expr) -> pl.Expr:
    return s.cast(pl.Utf8).str.strip_chars().str.to_uppercase()


def _bis_permit_subtype_to_type(subtype: pl.Expr) -> pl.Expr:
    """SH → shed; SD → scaffold; SF → scaffold; everything else → null."""
    return (
        pl.when(subtype == "SH").then(pl.lit("shed"))
          .when(subtype.is_in(["SD", "SF"])).then(pl.lit("scaffold"))
          .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _dob_now_scaffold_type(scaffold: pl.Expr, shed: pl.Expr) -> pl.Expr:
    """Both → both; scaffold-only → scaffold; shed-only → shed; neither → null."""
    s = (scaffold == "1")
    d = (shed == "1")
    return (
        pl.when(s & d).then(pl.lit("both"))
          .when(s).then(pl.lit("scaffold"))
          .when(d).then(pl.lit("shed"))
          .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _parse_iso_datetime(col: pl.Expr) -> pl.Expr:
    """DOB NOW emits '2017-10-17T16:13:39.000'. Keep as UTC-naive (upstream data
    has no tz on these timestamps; they are presumed NY local but we don't
    localize because downstream only does equality/comparison, not tz math).

    Cast to Utf8 first so an all-null source column (polars null dtype) doesn't
    break strptime.
    """
    s = col.cast(pl.Utf8)
    return s.str.strptime(pl.Datetime(time_unit="us"), format="%Y-%m-%dT%H:%M:%S%.f", strict=False)


def _parse_mmddyyyy(col: pl.Expr) -> pl.Expr:
    """BIS emits 'MM/DD/YYYY'. Some rows have 'MM/DD/YYYY HH:MM:SS'. Try both."""
    s = col.cast(pl.Utf8)
    a = s.str.strptime(pl.Datetime(time_unit="us"), format="%m/%d/%Y", strict=False)
    b = s.str.strptime(pl.Datetime(time_unit="us"), format="%m/%d/%Y %H:%M:%S", strict=False)
    return pl.coalesce([a, b])


def _safe_float(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def normalize_dob_now(raw: pl.DataFrame) -> pl.DataFrame:
    """DOB NOW → common schema."""
    if raw.is_empty():
        return _empty_common_frame()
    df = raw.with_columns(
        pl.lit("dob_now").alias("source"),
        pl.col("job_filing_number").cast(pl.Utf8).alias("permit_id"),
        pl.col("job_filing_number").cast(pl.Utf8).alias("job_id"),
        pl.col("bin").cast(pl.Utf8).alias("bin"),
        _borough_norm(pl.col("borough")).alias("borough"),
        _parse_iso_datetime(pl.col("first_permit_date")).alias("issue_date"),
        pl.lit(None, dtype=pl.Datetime(time_unit="us")).alias("expiration_date"),
        _parse_iso_datetime(pl.col("signoff_date")).alias("signoff_date")
            if "signoff_date" in raw.columns
            else pl.lit(None, dtype=pl.Datetime(time_unit="us")).alias("signoff_date"),
        _dob_now_scaffold_type(pl.col("scaffold"), pl.col("shed")).alias("scaffold_type"),
        pl.col("filing_status").cast(pl.Utf8).alias("permit_status")
            if "filing_status" in raw.columns
            else pl.lit(None, dtype=pl.Utf8).alias("permit_status"),
        pl.lit(None, dtype=pl.Utf8).alias("permit_subtype"),
        (
            pl.col("house_no").cast(pl.Utf8).fill_null("")
            + pl.lit(" ")
            + pl.col("street_name").cast(pl.Utf8).fill_null("")
        ).str.strip_chars().alias("address"),
        pl.col("block").cast(pl.Utf8).alias("block"),
        pl.col("lot").cast(pl.Utf8).alias("lot"),
        pl.col("job_type").cast(pl.Utf8).alias("job_type"),
        _safe_float(pl.col("initial_cost")).alias("initial_cost")
            if "initial_cost" in raw.columns
            else pl.lit(None, dtype=pl.Float64).alias("initial_cost"),
        _safe_float(pl.col("latitude")).alias("raw_latitude"),
        _safe_float(pl.col("longitude")).alias("raw_longitude"),
    )
    return df.select(list(COMMON_COLUMNS))


def normalize_bis(raw: pl.DataFrame) -> pl.DataFrame:
    """BIS → common schema."""
    if raw.is_empty():
        return _empty_common_frame()
    df = raw.with_columns(
        pl.lit("bis").alias("source"),
        pl.col("permit_si_no").cast(pl.Utf8).alias("permit_id"),
        pl.col("job__").cast(pl.Utf8).alias("job_id"),
        pl.col("bin__").cast(pl.Utf8).alias("bin"),
        _borough_norm(pl.col("borough")).alias("borough"),
        _parse_mmddyyyy(pl.col("issuance_date")).alias("issue_date"),
        _parse_mmddyyyy(pl.col("expiration_date")).alias("expiration_date")
            if "expiration_date" in raw.columns
            else pl.lit(None, dtype=pl.Datetime(time_unit="us")).alias("expiration_date"),
        pl.lit(None, dtype=pl.Datetime(time_unit="us")).alias("signoff_date"),
        _bis_permit_subtype_to_type(pl.col("permit_subtype")).alias("scaffold_type"),
        pl.col("permit_status").cast(pl.Utf8).alias("permit_status"),
        pl.col("permit_subtype").cast(pl.Utf8).alias("permit_subtype"),
        (
            pl.col("house__").cast(pl.Utf8).fill_null("")
            + pl.lit(" ")
            + pl.col("street_name").cast(pl.Utf8).fill_null("")
        ).str.strip_chars().alias("address"),
        pl.col("block").cast(pl.Utf8).alias("block"),
        pl.col("lot").cast(pl.Utf8).alias("lot"),
        pl.col("job_type").cast(pl.Utf8).alias("job_type"),
        pl.lit(None, dtype=pl.Float64).alias("initial_cost"),
        _safe_float(pl.col("gis_latitude")).alias("raw_latitude")
            if "gis_latitude" in raw.columns
            else pl.lit(None, dtype=pl.Float64).alias("raw_latitude"),
        _safe_float(pl.col("gis_longitude")).alias("raw_longitude")
            if "gis_longitude" in raw.columns
            else pl.lit(None, dtype=pl.Float64).alias("raw_longitude"),
    )
    return df.select(list(COMMON_COLUMNS))


def _empty_common_frame() -> pl.DataFrame:
    """Build a typed empty frame so downstream schema is stable on empty fetches."""
    return pl.DataFrame(schema={
        "source": pl.Utf8,
        "permit_id": pl.Utf8,
        "job_id": pl.Utf8,
        "bin": pl.Utf8,
        "borough": pl.Utf8,
        "issue_date": pl.Datetime(time_unit="us"),
        "expiration_date": pl.Datetime(time_unit="us"),
        "signoff_date": pl.Datetime(time_unit="us"),
        "scaffold_type": pl.Utf8,
        "permit_status": pl.Utf8,
        "permit_subtype": pl.Utf8,
        "address": pl.Utf8,
        "block": pl.Utf8,
        "lot": pl.Utf8,
        "job_type": pl.Utf8,
        "initial_cost": pl.Float64,
        "raw_latitude": pl.Float64,
        "raw_longitude": pl.Float64,
    })


def normalize_to_common(
    dob_now_raw: pl.DataFrame,
    bis_raw: pl.DataFrame,
) -> pl.DataFrame:
    """Concat the two sources in canonical column order."""
    d = normalize_dob_now(dob_now_raw)
    b = normalize_bis(bis_raw)
    return pl.concat([d, b], how="vertical_relaxed")
