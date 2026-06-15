"""Shared geometry utilities for curation sub-datasets.

``attach_geometry`` is a generic 3-stage spatial matcher:

1. **Exact BIN join** against ``nyc_buildings.parquet`` — ``geom_source='bin_polygon'``.
2. **Nearest-building fallback** via ``sjoin_nearest`` within ``nearest_max_ft``
   feet of the row's ``(lat, lon)`` — ``geom_source='nearest_polygon'``.
3. **Point fallback** — a ``Point(longitude, latitude)`` —
   ``geom_source='point'``.

Rows that have neither a BIN nor a valid lat/lon get ``geom_source='none'``
(for the validator to drop).

All matched geometries are buffered by ``buffer_ft`` in EPSG:2263 (US feet)
and reprojected to EPSG:4326 before return.

Originally lived in :mod:`dagspaces.common.curation.permits.buffer` —
extracted here so curation sub-datasets beyond permits (FacDB, future
layers) can share the same polygon-buffering machinery.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely.geometry import Point

__all__ = [
    "attach_geometry",
    "DEFAULT_BUILDINGS_PATH",
    "DEFAULT_NEAREST_MAX_FT",
    "NYC_SP_CRS",
    "WGS84_CRS",
    "load_buildings",
]

log = logging.getLogger(__name__)

DEFAULT_BUILDINGS_PATH = "/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet"
NYC_SP_CRS = "EPSG:2263"      # NY State Plane Long Island (US feet)
WGS84_CRS = "EPSG:4326"
DEFAULT_NEAREST_MAX_FT = 200.0


def load_buildings(path: str) -> gpd.GeoDataFrame:
    """Load ``nyc_buildings.parquet``, keeping only ``bin`` + ``geometry``.

    The source file has its own ``geom_source`` column (DoITT lineage); we
    drop it here to avoid clashing with our output column of the same name.
    Duplicate-BIN rows are dissolved into a single polygon per BIN.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nyc_buildings.parquet not found at {path}")
    gdf = gpd.read_parquet(path, columns=["bin", "geometry"])
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; cannot proceed")
    if str(gdf.crs).lower() not in ("epsg:4326", "4326"):
        gdf = gdf.to_crs(WGS84_CRS)
    gdf["bin"] = gdf["bin"].astype(str)
    if gdf["bin"].duplicated().any():
        before = len(gdf)
        gdf = gdf.dissolve(by="bin", as_index=False)
        log.info("geom: dissolved %d building rows to %d unique BINs", before, len(gdf))
    log.info("geom: loaded %d buildings from %s", len(gdf), path)
    return gdf


def attach_geometry(
    df: pl.DataFrame,
    buffer_ft: float = 80.0,
    *,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
    id_col: str = "permit_id",
    bin_col: str = "bin",
    lat_col: str = "raw_latitude",
    lon_col: str = "raw_longitude",
) -> gpd.GeoDataFrame:
    """Join ``df`` to building footprints with fallbacks, buffer, emit WGS84.

    Args:
        df: Polars frame with ``bin_col``, ``lat_col``, ``lon_col``, and a
            stable row-level primary key in ``id_col`` (used only for
            debugging / sjoin row-tracking).
        buffer_ft: Buffer distance in feet (applied in EPSG:2263).
        buildings_path: Path to ``nyc_buildings.parquet`` (ignored if
            ``buildings_gdf`` is passed).
        buildings_gdf: Pre-loaded buildings frame; useful for tests.
        nearest_max_ft: Max distance for the nearest-building fallback.
        id_col, bin_col, lat_col, lon_col: Column names in ``df``. Defaults
            match ``permits`` sub-dataset; pass e.g. ``lat_col='latitude'``
            for modules that keep the catalog's naming.

    Returns: GeoDataFrame in WGS84 with df's columns plus ``geometry``
    (buffered), ``geom_source`` ∈ {``bin_polygon``, ``nearest_polygon``,
    ``point``, ``none``}, and ``match_dist_ft`` (distance from row point
    to matched building, or NaN).
    """
    if df.is_empty():
        return gpd.GeoDataFrame(
            df.to_pandas().assign(geom_source="none", match_dist_ft=float("nan")),
            geometry=gpd.GeoSeries([], crs=WGS84_CRS),
            crs=WGS84_CRS,
        )

    for c in (bin_col, lat_col, lon_col, id_col):
        if c not in df.columns:
            raise ValueError(f"attach_geometry: input frame missing column {c!r}")

    buildings = buildings_gdf if buildings_gdf is not None else load_buildings(buildings_path)

    # --- Stage 1: exact BIN left-join ------------------------------------
    df_pd = df.to_pandas()
    df_pd[bin_col] = df_pd[bin_col].astype("string")
    merged = df_pd.merge(
        buildings[["bin", "geometry"]].rename(columns={
            "geometry": "_bin_geom",
            "bin": bin_col,
        }),
        on=bin_col,
        how="left",
    )
    merged["geometry"] = merged["_bin_geom"]
    merged["geom_source"] = merged["_bin_geom"].apply(
        lambda g: "bin_polygon" if g is not None and not pd.isna(g) and not getattr(g, "is_empty", False) else None
    )
    merged["match_dist_ft"] = 0.0
    merged.loc[merged["geom_source"].isna(), "match_dist_ft"] = float("nan")
    merged = merged.drop(columns=["_bin_geom"])

    # --- Stage 2: nearest-building for BIN-miss rows with valid lat/lon --
    need_nearest = (
        merged["geom_source"].isna()
        & merged[lat_col].notna()
        & merged[lon_col].notna()
    )
    n_bin_miss = int(need_nearest.sum())
    if n_bin_miss > 0:
        log.info(
            "geom: running nearest-building fallback on %d BIN-miss rows (max %.0f ft)",
            n_bin_miss, nearest_max_ft,
        )
        miss = merged.loc[need_nearest, [id_col, lat_col, lon_col]].copy()
        miss_gdf = gpd.GeoDataFrame(
            miss,
            geometry=gpd.points_from_xy(miss[lon_col], miss[lat_col]),
            crs=WGS84_CRS,
        ).to_crs(NYC_SP_CRS)
        bldg_sp = buildings[["bin", "geometry"]].to_crs(NYC_SP_CRS).rename(
            columns={"geometry": "_b_geom", "bin": "_nearest_bin"},
        )
        bldg_sp = gpd.GeoDataFrame(bldg_sp, geometry="_b_geom", crs=NYC_SP_CRS)
        nearest = gpd.sjoin_nearest(
            miss_gdf[[id_col, "geometry"]],
            bldg_sp[["_nearest_bin", "_b_geom"]].set_geometry("_b_geom"),
            how="left",
            max_distance=nearest_max_ft,
            distance_col="_nearest_dist",
        ).drop_duplicates(subset=[id_col], keep="first")
        bldg_wgs = buildings[["bin", "geometry"]].rename(columns={"bin": "_nearest_bin"})
        nearest = nearest.merge(
            bldg_wgs.rename(columns={"geometry": "_nearest_geom"}),
            on="_nearest_bin", how="left",
        )
        nearest_lookup = nearest.set_index(id_col)[["_nearest_geom", "_nearest_dist"]]
        idx = merged.loc[need_nearest].set_index(id_col).index
        merged.loc[need_nearest, "geometry"] = nearest_lookup.loc[idx, "_nearest_geom"].values
        merged.loc[need_nearest, "match_dist_ft"] = nearest_lookup.loc[idx, "_nearest_dist"].values
        matched_mask = need_nearest & merged["geometry"].notna()
        merged.loc[matched_mask, "geom_source"] = "nearest_polygon"

    # --- Stage 3: point fallback for everything still unmatched ----------
    need_point = (
        merged["geom_source"].isna()
        & merged[lat_col].notna()
        & merged[lon_col].notna()
    )
    if need_point.any():
        merged.loc[need_point, "geometry"] = merged.loc[need_point].apply(
            lambda r: Point(float(r[lon_col]), float(r[lat_col])), axis=1,
        )
        merged.loc[need_point, "geom_source"] = "point"

    # --- Stage 4: no geometry at all -------------------------------------
    no_geom = merged["geom_source"].isna()
    if no_geom.any():
        merged.loc[no_geom, "geom_source"] = "none"
        merged.loc[no_geom, "geometry"] = None

    counts = merged["geom_source"].value_counts().to_dict()
    log.info(
        "geom: source — bin_polygon=%d nearest_polygon=%d point=%d none=%d (total=%d)",
        counts.get("bin_polygon", 0),
        counts.get("nearest_polygon", 0),
        counts.get("point", 0),
        counts.get("none", 0),
        len(merged),
    )

    # --- Buffer in EPSG:2263 (US ft) then reproject back to WGS84 --------
    gdf_raw = gpd.GeoDataFrame(merged, geometry="geometry", crs=WGS84_CRS)
    have_geom = gdf_raw.geometry.notna() & ~gdf_raw.geometry.is_empty
    buffered = gdf_raw.copy()
    if have_geom.any():
        sub = gdf_raw.loc[have_geom].to_crs(NYC_SP_CRS)
        sub["geometry"] = sub.geometry.buffer(float(buffer_ft))
        sub = sub.to_crs(WGS84_CRS)
        buffered.loc[have_geom, "geometry"] = sub.geometry.values
    buffered = gpd.GeoDataFrame(buffered, geometry="geometry", crs=WGS84_CRS)
    return buffered
