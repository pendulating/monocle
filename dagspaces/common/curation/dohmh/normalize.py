"""DOHMH Socrata response → canonical inspection-level schema.

The raw dataset is at the ``(camis × inspection × violation)`` granularity.
The default normalization here **only collapses the violation level**: one
row per ``(camis, inspection_date)``. Per inspection we keep the most-
critical violation's columns (``Critical`` flag preferred, else lowest
``violation_code``) so the kept row carries a substantive citation.

Multiple **inspections** per restaurant are preserved by design — collapsing
across years of inspections is the job of :mod:`.aggregate`, an opt-in
follow-on utility. This split mirrors the user's request:

  * **build** (this module): fetch + geocode every inspection
  * **aggregate-restaurants**: collapse to one row per CAMIS

The downstream ``materialize-cyclomedia`` tool consumes the *aggregated*
parquet (so each unit_uid is unique), but inspection-level history is kept
intact in the build output for time-series / per-inspection analyses.

Output schema:

    permit_id        str   — = camis (shared geom API key)
    uid              str   — = camis (NOT unique across inspections — see aggregate)
    sample_id        str   — "{camis}_{inspection_date_iso}" (unique per row)
    camis            str
    facname          str   — = dba (most-recent-version-of-name from this inspection)
    dba              str
    address          str   — "<building> <street>"
    building         str
    street           str
    city             str   — = borough title-case (DOHMH has no separate city)
    zipcode          str
    borough          str   — canonical uppercase ("MANHATTAN" etc.)
    cuisine_description str
    facdomain, facgroup, facsubgrp, factype  str — FacDB-shaped aliases
    inspection_date  timestamp — null if this row's inspection_date is the
                                  1900-01-01 placeholder
    is_placeholder_inspection bool
    inspection_type  str
    action           str
    grade            str   — A/B/C/N/Z/P or "NOT YET INSPECTED" (placeholder)
    score            f64   — null on placeholder rows
    grade_date       timestamp
    critical_flag    str
    violation_code   str
    violation_description str
    record_date      timestamp
    bin              str
    bbl              str
    latitude         f64
    longitude        f64
    raw_latitude     f64
    raw_longitude    f64
    nta              str
    community_board  str
    council_district str
    census_tract     str
    datasource       str
"""

from __future__ import annotations

import datetime as dt
import logging

import polars as pl

__all__ = ["normalize_dohmh", "COMMON_COLUMNS", "BOROUGH_MAP", "PLACEHOLDER_DATE"]

log = logging.getLogger(__name__)


BOROUGH_MAP: dict[str, str] = {
    "1": "MANHATTAN", "MANHATTAN": "MANHATTAN", "MN": "MANHATTAN",
    "2": "BRONX", "BRONX": "BRONX", "BX": "BRONX",
    "3": "BROOKLYN", "BROOKLYN": "BROOKLYN", "BK": "BROOKLYN",
    "4": "QUEENS", "QUEENS": "QUEENS", "QN": "QUEENS",
    "5": "STATEN ISLAND", "STATEN ISLAND": "STATEN ISLAND", "SI": "STATEN ISLAND",
    # DOHMH uses "0" for unknown / refused-to-state borough.
    "0": "UNKNOWN",
}

# DOHMH writes this timestamp for CAMIS that are registered but never
# actually inspected — see DOHMH data dictionary.
PLACEHOLDER_DATE = dt.datetime(1900, 1, 1)


COMMON_COLUMNS: tuple[str, ...] = (
    # identity (FacDB-shaped + DOHMH-native)
    "permit_id",
    "uid",
    "sample_id",
    "camis",
    "facname",
    "dba",
    "address",
    "building",
    "street",
    "city",
    "zipcode",
    "borough",
    # cuisine + FacDB-shaped hierarchy aliases
    "cuisine_description",
    "facdomain",
    "facgroup",
    "facsubgrp",
    "factype",
    # inspection
    "inspection_date",
    "is_placeholder_inspection",
    "inspection_type",
    "action",
    "grade",
    "score",
    "grade_date",
    "critical_flag",
    "violation_code",
    "violation_description",
    "record_date",
    # spatial keys
    "bin",
    "bbl",
    "latitude",
    "longitude",
    "raw_latitude",
    "raw_longitude",
    # admin geographies
    "nta",
    "community_board",
    "council_district",
    "census_tract",
    # provenance
    "datasource",
)


def _safe_float(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def _borough_expr(col: pl.Expr) -> pl.Expr:
    s = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for k, v in BOROUGH_MAP.items():
        expr = pl.when(s == pl.lit(k)).then(pl.lit(v)).otherwise(expr)
    return expr


def _empty_frame() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {c: pl.Utf8 for c in COMMON_COLUMNS}
    for c in ("latitude", "longitude", "raw_latitude", "raw_longitude", "score"):
        schema[c] = pl.Float64
    for c in ("inspection_date", "record_date", "grade_date"):
        schema[c] = pl.Datetime("us")
    schema["is_placeholder_inspection"] = pl.Boolean
    return pl.DataFrame(schema=schema)


def normalize_dohmh(raw: pl.DataFrame) -> pl.DataFrame:
    """Raw inspection rows → one row per ``(camis, inspection_date)``.

    Violation-level multiplication is collapsed (most-critical violation
    wins). Multiple inspections per CAMIS are **preserved** — call
    :mod:`.aggregate` if you need one-row-per-restaurant output.

    Empty input returns a typed empty frame.
    """
    if raw.is_empty():
        return _empty_frame()

    def _str(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name in raw.columns else pl.lit(None, dtype=pl.Utf8)

    def _f(name: str) -> pl.Expr:
        return _safe_float(pl.col(name)) if name in raw.columns else pl.lit(None, dtype=pl.Float64)

    inspection_dt = (
        pl.col("inspection_date").cast(pl.Utf8).str.strip_chars()
        .str.to_datetime(strict=False)
        if "inspection_date" in raw.columns
        else pl.lit(None, dtype=pl.Datetime("us"))
    )
    record_dt = (
        pl.col("record_date").cast(pl.Utf8).str.strip_chars()
        .str.to_datetime(strict=False)
        if "record_date" in raw.columns
        else pl.lit(None, dtype=pl.Datetime("us"))
    )
    grade_dt = (
        pl.col("grade_date").cast(pl.Utf8).str.strip_chars()
        .str.to_datetime(strict=False)
        if "grade_date" in raw.columns
        else pl.lit(None, dtype=pl.Datetime("us"))
    )

    df = raw.with_columns(
        _str("camis").str.strip_chars().alias("camis"),
        _str("dba").str.strip_chars().alias("dba"),
        _str("boro").alias("_boro_raw"),
        _str("building").str.strip_chars().alias("building"),
        _str("street").str.strip_chars().alias("street"),
        _str("zipcode").str.strip_chars().alias("zipcode"),
        _str("phone").alias("phone"),
        _str("cuisine_description").str.strip_chars().alias("cuisine_description"),
        inspection_dt.alias("inspection_dt"),
        _str("action").alias("action"),
        _str("violation_code").alias("violation_code"),
        _str("violation_description").alias("violation_description"),
        _str("critical_flag").alias("critical_flag"),
        _f("score").alias("score"),
        _str("grade").str.strip_chars().alias("grade"),
        _str("inspection_type").alias("inspection_type"),
        _f("latitude").alias("latitude"),
        _f("longitude").alias("longitude"),
        _str("community_board").alias("community_board"),
        _str("council_district").alias("council_district"),
        _str("census_tract").alias("census_tract"),
        _str("bin").str.strip_chars().alias("bin"),
        _str("bbl").str.strip_chars().alias("bbl"),
        _str("nta").alias("nta"),
        record_dt.alias("record_dt"),
        grade_dt.alias("grade_dt"),
    )
    df = df.filter(pl.col("camis").is_not_null() & (pl.col("camis") != ""))
    if df.is_empty():
        return _empty_frame()

    placeholder = pl.col("inspection_dt") <= pl.lit(PLACEHOLDER_DATE)
    df = df.with_columns(
        placeholder.alias("is_placeholder_inspection"),
        pl.when(pl.col("critical_flag").str.to_uppercase() == "CRITICAL")
          .then(1).otherwise(0).alias("_critical_rank"),
    )

    # ---- Collapse VIOLATION rows only — keep one row per (camis, inspection_dt).
    # Within an inspection, the most-Critical violation row wins. This preserves
    # multiple inspections per restaurant (the aggregate utility does that).
    # We need a stable tie-breaker; violation_code (alpha-sorted) suffices.
    df = df.sort(
        ["camis", "inspection_dt", "_critical_rank", "violation_code"],
        descending=[False, False, True, False],
        nulls_last=True,
    )
    one_per = (
        df.group_by(["camis", "inspection_dt"], maintain_order=True)
          .agg(pl.all().first())
    )

    # ---- Project to canonical schema ------------------------------------
    addr = (
        pl.when(pl.col("building").is_null() | (pl.col("building") == ""))
          .then(pl.col("street"))
          .otherwise(
              pl.when(pl.col("street").is_null() | (pl.col("street") == ""))
                .then(pl.col("building"))
                .otherwise(pl.col("building") + pl.lit(" ") + pl.col("street"))
          )
    )

    cuisine_upper = pl.col("cuisine_description").str.to_uppercase()
    sample_id = (
        pl.col("camis") + pl.lit("_")
        + pl.when(pl.col("is_placeholder_inspection"))
            .then(pl.lit("placeholder"))
            # Full timestamp (not just date) — DOHMH occasionally records two
            # inspections at different times on the same calendar day for one
            # CAMIS (e.g. Cycle + same-day Re-inspection).
            .otherwise(pl.col("inspection_dt").dt.strftime("%Y%m%dT%H%M%S"))
    )

    out = one_per.with_columns(
        pl.col("camis").alias("permit_id"),
        pl.col("camis").alias("uid"),
        sample_id.alias("sample_id"),
        pl.col("dba").alias("facname"),
        addr.alias("address"),
        _borough_expr(pl.col("_boro_raw")).alias("borough"),
        pl.lit("FOOD SERVICE").alias("facdomain"),
        pl.lit("RESTAURANTS").alias("facgroup"),
        cuisine_upper.alias("facsubgrp"),
        cuisine_upper.alias("factype"),
        pl.when(pl.col("is_placeholder_inspection"))
          .then(pl.lit(None, dtype=pl.Datetime("us")))
          .otherwise(pl.col("inspection_dt"))
          .alias("inspection_date"),
        pl.when(pl.col("is_placeholder_inspection"))
          .then(pl.lit("NOT YET INSPECTED"))
          .otherwise(pl.col("grade"))
          .alias("grade"),
        pl.when(pl.col("is_placeholder_inspection"))
          .then(pl.lit(None, dtype=pl.Float64))
          .otherwise(pl.col("score"))
          .alias("score"),
        pl.col("grade_dt").alias("grade_date"),
        pl.col("record_dt").alias("record_date"),
        pl.col("latitude").alias("raw_latitude"),
        pl.col("longitude").alias("raw_longitude"),
        _borough_expr(pl.col("_boro_raw")).str.to_titlecase().alias("city"),
        pl.lit("dohmh:43nn-pn8j").alias("datasource"),
    )

    log.info(
        "normalize: %d raw rows → %d (camis, inspection_date) rows "
        "across %d unique camis "
        "(%d placeholder-only inspections)",
        raw.height,
        out.height,
        out["camis"].n_unique(),
        int(out.filter(pl.col("is_placeholder_inspection")).height),
    )

    return out.select(list(COMMON_COLUMNS))
