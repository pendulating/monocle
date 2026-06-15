"""Open Restaurants (Dining Out NYC) Socrata response → canonical flat schema.

One row per outdoor-dining **license** (a restaurant may hold more than one —
e.g. separate Sidewalk and Roadway licenses, or licenses at several
locations). The dataset has **no native primary key**, so we synthesize a
stable ``uid`` from the row's identity fields (``bbl`` + ``license_type`` +
rounded ``(lat, lon)``), appending an occurrence index only when that base key
collides. The result is unique per row (required by ``materialize-cyclomedia``,
which refuses duplicate unit IDs) and stable across rebuilds as long as DCWP
hasn't moved the location or relicensed it.

Output columns mirror :mod:`..facdb.normalize` / :mod:`..subway.normalize` so
downstream tooling that already speaks the shared schema (``materialize-
cyclomedia``, ``filter-facing``, ``sample-images``) just works:

    permit_id, uid          str — synthesized license UID (geom API key)
    facname                 str — public-facing name (assumed/DBA name, else legal)
    business_legal_name     str
    assumed_name_s          str — "doing business as" name
    address                 str — street address
    city                    str
    borough                 str — canonical uppercase ("MANHATTAN" etc.)
    postcode                str
    license_type            str — Sidewalk | Roadway
    license_status          str — Issued (all rows, as of 2026-06)
    license_issue_date      str — ISO timestamp
    license_expiration_date str — ISO timestamp
    bin                     str
    bbl                     str
    latitude, longitude     f64
    raw_latitude, raw_longitude  f64 — aliases for the shared geom API
    council_district        str
    community_board         str
    nta2020                 str
    ct2020                  str
    datasource              str — fixed: "dcwp:fpeh-f7ci"
"""

from __future__ import annotations

import logging

import polars as pl

__all__ = ["normalize_open_restaurants", "COMMON_COLUMNS", "BOROUGH_MAP"]

log = logging.getLogger(__name__)


# Source stores title-case borough text; BBL's first digit is the borough code.
BOROUGH_MAP: dict[str, str] = {
    "1": "MANHATTAN", "MANHATTAN": "MANHATTAN", "MN": "MANHATTAN",
    "2": "BRONX", "BRONX": "BRONX", "BX": "BRONX",
    "3": "BROOKLYN", "BROOKLYN": "BROOKLYN", "BK": "BROOKLYN",
    "4": "QUEENS", "QUEENS": "QUEENS", "QN": "QUEENS",
    "5": "STATEN ISLAND", "STATEN ISLAND": "STATEN ISLAND", "SI": "STATEN ISLAND",
}


COMMON_COLUMNS: tuple[str, ...] = (
    # identity
    "permit_id",    # = uid, repurposed for the shared geom.attach_geometry API
    "uid",
    "facname",
    "business_legal_name",
    "assumed_name_s",
    "address",
    "city",
    "borough",
    "postcode",
    # license attributes
    "license_type",
    "license_status",
    "license_issue_date",
    "license_expiration_date",
    # spatial keys
    "bin",
    "bbl",
    "latitude",
    "longitude",
    "raw_latitude",   # alias to latitude (shared geom API)
    "raw_longitude",  # alias to longitude
    # admin geographies
    "council_district",
    "community_board",
    "nta2020",
    "ct2020",
    # provenance
    "datasource",
)


def _safe_float(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


def _borough_expr(text_col: pl.Expr, bbl_col: pl.Expr) -> pl.Expr:
    """Canonical uppercase borough from the title-case text, falling back to
    the BBL's leading borough digit when the text is missing/blank."""
    s = text_col.cast(pl.Utf8).str.strip_chars().str.to_uppercase()
    boro_digit = bbl_col.cast(pl.Utf8).str.strip_chars().str.slice(0, 1)
    expr: pl.Expr = pl.lit(None, dtype=pl.Utf8)
    for k, v in BOROUGH_MAP.items():
        expr = pl.when(s == pl.lit(k)).then(pl.lit(v)).otherwise(expr)
        if k in ("1", "2", "3", "4", "5"):
            expr = pl.when(boro_digit == pl.lit(k)).then(pl.lit(v)).otherwise(expr)
    return expr


def _coord_slug(col: pl.Expr) -> pl.Expr:
    """7-dp coordinate → digits-only token (sign → 'n', dot dropped)."""
    return (
        col.round(7).cast(pl.Utf8)
        .str.replace_all(r"\.", "")
        .str.replace_all(r"-", "n")
    )


def _empty_frame() -> pl.DataFrame:
    schema = {c: pl.Utf8 for c in COMMON_COLUMNS}
    for c in ("latitude", "longitude", "raw_latitude", "raw_longitude"):
        schema[c] = pl.Float64
    return pl.DataFrame(schema=schema)


def normalize_open_restaurants(raw: pl.DataFrame) -> pl.DataFrame:
    """Socrata response → canonical schema with synthetic unique UIDs.

    Empty input returns a typed empty frame.
    """
    if raw.is_empty():
        return _empty_frame()

    def _str(name: str) -> pl.Expr:
        return pl.col(name).cast(pl.Utf8) if name in raw.columns else pl.lit(None, dtype=pl.Utf8)

    def _f(name: str) -> pl.Expr:
        return _safe_float(pl.col(name)) if name in raw.columns else pl.lit(None, dtype=pl.Float64)

    df = raw.with_columns(
        _str("business_legal_name").str.strip_chars().alias("business_legal_name"),
        _str("assumed_name_s").str.strip_chars().alias("assumed_name_s"),
        _str("street").str.strip_chars().alias("address"),
        _str("city").str.strip_chars().alias("city"),
        _str("postcode").str.strip_chars().alias("postcode"),
        _str("license_type").str.strip_chars().alias("license_type"),
        _str("license_status").str.strip_chars().alias("license_status"),
        _str("license_issue_date").str.strip_chars().alias("license_issue_date"),
        _str("license_expiration_date").str.strip_chars().alias("license_expiration_date"),
        _str("bin").str.strip_chars().alias("bin"),
        _str("bbl").str.strip_chars().alias("bbl"),
        _f("latitude").alias("latitude"),
        _f("longitude").alias("longitude"),
        _str("council_district").str.strip_chars().alias("council_district"),
        _str("community_board").str.strip_chars().alias("community_board"),
        _str("nta2020").str.strip_chars().alias("nta2020"),
        _str("ct2020").str.strip_chars().alias("ct2020"),
        _borough_expr(_str("borough"), _str("bbl")).alias("borough"),
    )

    # Drop rows without a usable coordinate before synthesizing UIDs.
    df = df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
    if df.is_empty():
        return _empty_frame()

    # Public-facing name: prefer the assumed/DBA name (what's on the storefront),
    # fall back to the legal entity name.
    facname = (
        pl.when(pl.col("assumed_name_s").is_not_null() & (pl.col("assumed_name_s") != ""))
        .then(pl.col("assumed_name_s"))
        .otherwise(pl.col("business_legal_name"))
    )

    # Synthetic stable base key, then a within-base occurrence index so two
    # rows sharing the same (bbl, license_type, rounded coords) stay distinct.
    type_slug = pl.col("license_type").fill_null("NA").str.replace_all(r"[^A-Za-z]", "").str.to_uppercase()
    base = (
        pl.col("bbl").fill_null("NA").replace("", "NA") + pl.lit("_")
        + type_slug + pl.lit("_")
        + _coord_slug(pl.col("latitude")) + pl.lit("_")
        + _coord_slug(pl.col("longitude"))
    )
    df = df.with_columns(base.alias("_base"))
    df = df.with_columns(
        pl.len().over("_base").alias("_grp_n"),
        pl.int_range(0, pl.len()).over("_base").alias("_grp_i"),
    )
    uid = (
        pl.when(pl.col("_grp_n") > 1)
        .then(pl.col("_base") + pl.lit("_") + pl.col("_grp_i").cast(pl.Utf8))
        .otherwise(pl.col("_base"))
    )

    out = df.with_columns(
        uid.alias("uid"),
        uid.alias("permit_id"),
        facname.alias("facname"),
        pl.col("latitude").alias("raw_latitude"),
        pl.col("longitude").alias("raw_longitude"),
        pl.lit("dcwp:fpeh-f7ci").alias("datasource"),
    )

    log.info(
        "normalize: %d raw rows → %d licenses "
        "(%d unique uid, %d unique bbl, types=%s)",
        raw.height, out.height,
        out["uid"].n_unique(), out["bbl"].n_unique(),
        out["license_type"].drop_nulls().unique().to_list(),
    )

    return out.select(list(COMMON_COLUMNS))
