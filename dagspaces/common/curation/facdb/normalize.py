"""FacDB Socrata response → canonical flat schema for curation.

Output schema (same column names as the Socrata raw for readability, with
a few derived columns added):

    uid              str    — FacDB primary key (stable across versions)
    facname          str
    address          str
    city             str
    zipcode          str
    borough          str    — canonical uppercase ("MANHATTAN" etc.)
    borocode         str
    bin              str
    bbl              str
    latitude         f64
    longitude        f64
    xcoord           f64    — NY State Plane (EPSG:2263), US ft
    ycoord           f64
    facdomain        str    — hierarchy level 1 (highest)
    facgroup         str    — hierarchy level 2
    facsubgrp        str    — hierarchy level 3
    factype          str    — hierarchy level 4 (lowest)
    capacity         f64
    captype          str
    opname           str
    overagency       str
    overlevel        str
    servarea         str
    cd               str
    council          str
    nta2020          str
    ct2020           str
    datasource       str

Plus the generic shared columns used by :mod:`..geom.attach_geometry`
under its default column names: ``bin``, ``raw_latitude``, ``raw_longitude``,
``permit_id`` (repurposed as the per-row primary key).
"""

from __future__ import annotations

from typing import Optional

import polars as pl

__all__ = ["normalize_facdb", "COMMON_COLUMNS", "BOROUGH_MAP"]


BOROUGH_MAP: dict[str, str] = {
    "1": "MANHATTAN", "MANHATTAN": "MANHATTAN", "MN": "MANHATTAN",
    "2": "BRONX", "BRONX": "BRONX", "BX": "BRONX",
    "3": "BROOKLYN", "BROOKLYN": "BROOKLYN", "BK": "BROOKLYN",
    "4": "QUEENS", "QUEENS": "QUEENS", "QN": "QUEENS",
    "5": "STATEN ISLAND", "STATEN ISLAND": "STATEN ISLAND", "SI": "STATEN ISLAND",
}


COMMON_COLUMNS: tuple[str, ...] = (
    # identity
    "permit_id",    # = uid, repurposed for shared geom.attach_geometry API
    "uid",
    "facname",
    "address",
    "city",
    "zipcode",
    "borough",
    "borocode",
    # hierarchy
    "facdomain",
    "facgroup",
    "facsubgrp",
    "factype",
    # spatial keys
    "bin",
    "bbl",
    "latitude",
    "longitude",
    "raw_latitude",   # alias to latitude (shared geom API)
    "raw_longitude",  # alias to longitude
    "xcoord",
    "ycoord",
    # ops/overview
    "capacity",
    "captype",
    "opname",
    "overagency",
    "overlevel",
    "servarea",
    # admin geographies
    "cd",
    "council",
    "nta2020",
    "ct2020",
    # provenance
    "datasource",
)


def _borough(col: pl.Expr) -> pl.Expr:
    """Normalize boro / borocode to canonical uppercase."""
    s = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    # Build a when/then chain from BOROUGH_MAP keys — polars replace doesn't do
    # case-insensitive mapping cleanly, so do it explicitly.
    expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for k, v in BOROUGH_MAP.items():
        expr = pl.when(s == pl.lit(k)).then(pl.lit(v)).otherwise(expr)
    return expr


def _safe_float(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def normalize_facdb(raw: pl.DataFrame) -> pl.DataFrame:
    """Socrata response → canonical FacDB schema. Empty input gracefully
    returns a typed empty frame."""
    if raw.is_empty():
        return _empty_frame()

    def _or_null_str(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name in raw.columns else pl.lit(None, dtype=pl.Utf8)

    def _or_null_float(name: str) -> pl.Expr:
        return _safe_float(pl.col(name)) if name in raw.columns else pl.lit(None, dtype=pl.Float64)

    # Prefer the already-canonical `boro` text where present; fall back to
    # borocode if needed.
    if "boro" in raw.columns:
        borough_expr = _borough(pl.col("boro")).alias("borough")
    elif "borocode" in raw.columns:
        borough_expr = _borough(pl.col("borocode")).alias("borough")
    else:
        borough_expr = pl.lit(None, dtype=pl.Utf8).alias("borough")

    lat = _or_null_float("latitude")
    lon = _or_null_float("longitude")

    df = raw.with_columns(
        _or_null_str("uid").alias("uid"),
        _or_null_str("uid").alias("permit_id"),
        _or_null_str("facname").alias("facname"),
        _or_null_str("address").alias("address"),
        _or_null_str("city").alias("city"),
        _or_null_str("zipcode").alias("zipcode"),
        borough_expr,
        _or_null_str("borocode").alias("borocode"),
        _or_null_str("facdomain").str.to_uppercase().alias("facdomain"),
        _or_null_str("facgroup").str.to_uppercase().alias("facgroup"),
        _or_null_str("facsubgrp").str.to_uppercase().alias("facsubgrp"),
        _or_null_str("factype").str.to_uppercase().alias("factype"),
        _or_null_str("bin").alias("bin"),
        _or_null_str("bbl").alias("bbl"),
        lat.alias("latitude"),
        lon.alias("longitude"),
        lat.alias("raw_latitude"),
        lon.alias("raw_longitude"),
        _or_null_float("xcoord").alias("xcoord"),
        _or_null_float("ycoord").alias("ycoord"),
        _or_null_float("capacity").alias("capacity"),
        _or_null_str("captype").alias("captype"),
        _or_null_str("opname").alias("opname"),
        _or_null_str("overagency").alias("overagency"),
        _or_null_str("overlevel").alias("overlevel"),
        _or_null_str("servarea").alias("servarea"),
        _or_null_str("cd").alias("cd"),
        _or_null_str("council").alias("council"),
        _or_null_str("nta2020").alias("nta2020"),
        _or_null_str("ct2020").alias("ct2020"),
        _or_null_str("datasource").alias("datasource"),
    )
    return df.select(list(COMMON_COLUMNS))


def _empty_frame() -> pl.DataFrame:
    schema = {c: pl.Utf8 for c in COMMON_COLUMNS}
    for c in ("latitude", "longitude", "raw_latitude", "raw_longitude",
              "xcoord", "ycoord", "capacity"):
        schema[c] = pl.Float64
    return pl.DataFrame(schema=schema)
