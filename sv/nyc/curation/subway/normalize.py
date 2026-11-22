"""MTA subway entrances Socrata response → canonical schema.

One row per entrance/exit. The dataset has no native primary key, so we
synthesize one from the row's identity fields (`station_id`,
`entrance_type`, rounded `(lat, lon)`). The resulting `uid` is stable
across rebuilds as long as MTA hasn't moved the entrance.

Output columns mirror :mod:`..facdb.normalize` so downstream tooling that
already speaks the FacDB schema (`materialize-cyclomedia`,
`filter-facing`, `sample-images`) just works:

    permit_id, uid          str — = synthesized entrance UID (e.g. "27_Stair_40683905_-73978879")
    facname                 str — "<stop_name> (<entrance_type>)"
    station_id              str
    complex_id              str
    gtfs_stop_id            str
    stop_name               str
    constituent_station_name str
    line                    str  — e.g. "4th Av"
    division                str  — IRT / IND / BMT / SIR / IRT/BMT / IND/BMT
    daytime_routes          str  — space-separated route IDs
    entrance_type           str  — frozen vocab; see entrance_types.py
    entry_allowed           bool
    exit_allowed            bool
    address                 str  — "<stop_name> (<entrance_type>)" (no street address in source)
    city                    str  — borough title-case
    borough                 str  — canonical uppercase
    facdomain, facgroup, facsubgrp, factype  str  — FacDB-shaped aliases
                                                    facdomain="TRANSPORTATION", facgroup="SUBWAY ENTRANCES",
                                                    facsubgrp=upper(entrance_type), factype=upper(entrance_type)
    latitude, longitude     f64
    raw_latitude, raw_longitude  f64  — aliases for shared geom API
    datasource              str  — fixed: "mta:i9wp-a4ja"
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = ["normalize_subway_entrances", "COMMON_COLUMNS", "BOROUGH_MAP"]

log = logging.getLogger(__name__)


# Subway dataset uses single/two-letter borough codes.
BOROUGH_MAP: dict[str, str] = {
    "M": "MANHATTAN",
    "B": "BROOKLYN", "BK": "BROOKLYN",
    "BX": "BRONX",
    "Q": "QUEENS",
    "SI": "STATEN ISLAND",
}


COMMON_COLUMNS: tuple[str, ...] = (
    "permit_id",
    "uid",
    "facname",
    "station_id",
    "complex_id",
    "gtfs_stop_id",
    "stop_name",
    "constituent_station_name",
    "line",
    "division",
    "daytime_routes",
    "entrance_type",
    "entry_allowed",
    "exit_allowed",
    "address",
    "city",
    "borough",
    "facdomain",
    "facgroup",
    "facsubgrp",
    "factype",
    "latitude",
    "longitude",
    "raw_latitude",
    "raw_longitude",
    "datasource",
)


def _safe_float(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def _yes_to_bool(col: pl.Expr) -> pl.Expr:
    s = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    return (
        pl.when(s == pl.lit("YES")).then(True)
          .when(s == pl.lit("NO")).then(False)
          .otherwise(None)
    )


def _borough_expr(col: pl.Expr) -> pl.Expr:
    s = col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for k, v in BOROUGH_MAP.items():
        expr = pl.when(s == pl.lit(k)).then(pl.lit(v)).otherwise(expr)
    return expr


def _empty_frame() -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {c: pl.Utf8 for c in COMMON_COLUMNS}
    for c in ("latitude", "longitude", "raw_latitude", "raw_longitude"):
        schema[c] = pl.Float64
    for c in ("entry_allowed", "exit_allowed"):
        schema[c] = pl.Boolean
    return pl.DataFrame(schema=schema)


def normalize_subway_entrances(raw: pl.DataFrame) -> pl.DataFrame:
    """Raw entrance rows → canonical schema with synthetic stable UIDs.

    Empty input returns a typed empty frame.
    """
    if raw.is_empty():
        return _empty_frame()

    def _str(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name in raw.columns else pl.lit(None, dtype=pl.Utf8)

    def _f(name: str) -> pl.Expr:
        return _safe_float(pl.col(name)) if name in raw.columns else pl.lit(None, dtype=pl.Float64)

    df = raw.with_columns(
        _str("station_id").str.strip_chars().alias("station_id"),
        _str("complex_id").str.strip_chars().alias("complex_id"),
        _str("gtfs_stop_id").str.strip_chars().alias("gtfs_stop_id"),
        _str("stop_name").str.strip_chars().alias("stop_name"),
        _str("constituent_station_name").str.strip_chars().alias("constituent_station_name"),
        _str("line").str.strip_chars().alias("line"),
        _str("division").str.strip_chars().alias("division"),
        _str("daytime_routes").str.strip_chars().alias("daytime_routes"),
        _str("entrance_type").str.strip_chars().alias("entrance_type"),
        _yes_to_bool(pl.col("entry_allowed") if "entry_allowed" in raw.columns
                     else pl.lit(None, dtype=pl.Utf8)).alias("entry_allowed"),
        _yes_to_bool(pl.col("exit_allowed") if "exit_allowed" in raw.columns
                     else pl.lit(None, dtype=pl.Utf8)).alias("exit_allowed"),
        _f("entrance_latitude").alias("latitude"),
        _f("entrance_longitude").alias("longitude"),
        _str("borough").alias("_boro_raw"),
    )
    df = df.filter(
        pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()
    )
    if df.is_empty():
        return _empty_frame()

    # Synthetic UID — stable across rebuilds when station_id + entrance_type
    # + rounded coords don't change. Round to 7 dp (~1.1 cm) so jitter from
    # MTA's geocoding doesn't churn the IDs across vintages.
    lat_str = pl.col("latitude").map_elements(
        lambda x: f"{x:.7f}".replace(".", "").replace("-", "n"),
        return_dtype=pl.Utf8,
    )
    lon_str = pl.col("longitude").map_elements(
        lambda x: f"{x:.7f}".replace(".", "").replace("-", "n"),
        return_dtype=pl.Utf8,
    )
    type_slug = pl.col("entrance_type").str.replace_all(r"[^A-Za-z]", "")
    uid = (
        pl.col("station_id").fill_null("NA") + pl.lit("_")
        + type_slug.fill_null("NA") + pl.lit("_")
        + lat_str + pl.lit("_") + lon_str
    )

    facname = (
        pl.col("stop_name").fill_null("UNKNOWN")
        + pl.lit(" (")
        + pl.col("entrance_type").fill_null("Entrance")
        + pl.lit(")")
    )
    et_upper = pl.col("entrance_type").str.to_uppercase()

    out = df.with_columns(
        uid.alias("uid"),
        uid.alias("permit_id"),
        facname.alias("facname"),
        # No street address in the source — repeat facname so it shows up
        # in places like materialize's unit_name fallback.
        facname.alias("address"),
        _borough_expr(pl.col("_boro_raw")).alias("borough"),
        _borough_expr(pl.col("_boro_raw")).str.to_titlecase().alias("city"),
        pl.lit("TRANSPORTATION").alias("facdomain"),
        pl.lit("SUBWAY ENTRANCES").alias("facgroup"),
        et_upper.alias("facsubgrp"),
        et_upper.alias("factype"),
        pl.col("latitude").alias("raw_latitude"),
        pl.col("longitude").alias("raw_longitude"),
        pl.lit("mta:i9wp-a4ja").alias("datasource"),
    )

    log.info(
        "normalize: %d raw rows → %d entrances "
        "(%d unique stations, %d entrance types)",
        raw.height, out.height,
        out["station_id"].n_unique(),
        out["entrance_type"].n_unique(),
    )

    return out.select(list(COMMON_COLUMNS))
