"""Load and normalize WFS recording catalog CSVs.

The Cyclomedia pull pipeline emits CSVs under `/share/ju/cyclomedia/pull/`:

    recordings_{borough}_2025_chunks/{borough}_2025_partNofM.csv
    recordings_{borough}_2025_part*.csv                  (flat layout)
    out_catalog/recordings_{borough}_2025.csv            (dedup aggregates)

Each CSV has WFS metadata per `imageId` (the recording_id). This module
concats, normalizes, and dedupes them into one Polars DataFrame.
"""

from __future__ import annotations

import glob
import logging
from typing import Iterable

import polars as pl

from .schema import WFS_JOIN_COLUMNS

__all__ = ["load_wfs_catalog", "DEFAULT_CATALOG_GLOB"]

log = logging.getLogger(__name__)

# Glob covering every layout seen in /share/ju/cyclomedia/pull/:
#   - chunked per-borough dirs (manhattan, manhattan_latter, queens)
#   - flat per-part files (bronx parts 1..4)
#   - out_catalog/recordings_{borough}_2025.csv (brooklyn, staten island — only
#     available here; also aggregate copies of other boroughs, harmless because
#     load_wfs_catalog dedupes on recording_id)
DEFAULT_CATALOG_GLOB: tuple[str, ...] = (
    "/share/ju/cyclomedia/pull/recordings_*_chunks/*.csv",
    "/share/ju/cyclomedia/pull/recordings_*_part*.csv",
    "/share/ju/cyclomedia/pull/out_catalog/recordings_*.csv",
)


def _expand_globs(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))


def load_wfs_catalog(patterns: Iterable[str] = DEFAULT_CATALOG_GLOB) -> pl.DataFrame:
    """Load every matching WFS catalog CSV; dedupe on imageId.

    Returns a Polars DataFrame keyed by `recording_id` (renamed from imageId).
    `recordedAt` is parsed to a timezone-aware Datetime in US/Eastern.
    """
    paths = _expand_globs(patterns)
    if not paths:
        raise ValueError(f"No WFS catalog CSVs matched: {list(patterns)}")

    log.info("wfs: loading %d catalog CSVs", len(paths))
    frames = []
    for p in paths:
        # infer_schema_length=0 → all columns read as Utf8; we cast what we need.
        # Avoids pyarrow tripping on mixed-dtype precision columns.
        df = pl.read_csv(p, infer_schema_length=0)
        frames.append(df)

    cat = pl.concat(frames, how="diagonal_relaxed")

    if "imageId" in cat.columns:
        cat = cat.rename({"imageId": "recording_id"})
    if "recording_id" not in cat.columns:
        raise ValueError("WFS catalog is missing 'imageId'/'recording_id' column")

    cat = cat.unique(subset=["recording_id"], keep="first")

    # Typed casts. Anything not present stays absent.
    cast_exprs: list[pl.Expr] = []
    float_cols = {
        "lat", "lon", "recorderDirection", "yawDegrees", "yawPrecisionDegrees",
        "orientation", "orientationPrecision", "statePlaneX", "statePlaneY",
        "height", "groundLevelOffset", "latitudePrecision", "longitudePrecision",
        "heightPrecision",
    }
    int_cols = {"year", "heightSystem"}
    bool_cols = {"hasDepthMap", "isAuthorized"}

    for c in cat.columns:
        if c in float_cols:
            cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        elif c in int_cols:
            cast_exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
        elif c in bool_cols:
            # WFS CSVs emit "True"/"False" strings
            cast_exprs.append(
                pl.col(c).cast(pl.Utf8, strict=False).str.to_lowercase().is_in(["true", "1"]).alias(c)
            )
    if cast_exprs:
        cat = cat.with_columns(cast_exprs)

    if "recordedAt" in cat.columns:
        cat = cat.with_columns(
            pl.col("recordedAt")
            .str.to_datetime(strict=False, time_zone="UTC")
            .dt.convert_time_zone("America/New_York")
            .alias("recordedAt")
        )

    log.info("wfs: %d unique recording_ids loaded", cat.height)
    return cat
