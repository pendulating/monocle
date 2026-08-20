"""Aggregation of unit scores to the 3 NYC geographies.

Every validation-via-statistical-proxy notebook aggregates to the same 3 layers,
so the cases stay comparable:

| Layer | File | Key | CRS on disk |
|-------|------|-----|-------------|
| Neighborhood tabulation area | `nynta2020_26b/nynta2020.shp` | `NTA2020` | EPSG:2263 |
| Community district | `Community_Districts_20260812.geojson` | `boro_cd` | EPSG:4326 |
| Census tract | `2020_Census_Tracts_20260304.geojson` | `boroct2020` | EPSG:4326 |

The 3 files do not share a CRS, so this module puts every layer and every point
into EPSG:2263 before the spatial join. EPSG:2263 is the NYC state-plane system
in US feet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
GEO_DIR = REPO_ROOT / "data" / "geo"

# NYC state plane, US feet. Every join happens in this CRS.
WORKING_CRS = "EPSG:2263"

LAYERS: Dict[str, Dict[str, str]] = {
    "nta": {
        "path": str(GEO_DIR / "nynta2020_26b" / "nynta2020.shp"),
        "key": "NTA2020",
        "name": "NTAName",
        "label": "Neighborhood tabulation area",
    },
    "community_district": {
        "path": str(GEO_DIR / "Community_Districts_20260812.geojson"),
        "key": "boro_cd",
        "name": "boro_cd",
        "label": "Community district",
    },
    "census_tract": {
        "path": str(GEO_DIR / "2020_Census_Tracts_20260304.geojson"),
        "key": "boroct2020",
        "name": "ntaname",
        "label": "Census tract",
    },
}


def load_layer(layer: str):
    """Read one geography layer and put it into the working CRS."""
    import geopandas as gpd

    if layer not in LAYERS:
        raise KeyError(f"unknown layer {layer!r}; use one of {list(LAYERS)}")
    spec = LAYERS[layer]
    g = gpd.read_file(spec["path"])
    if g.crs is None:
        raise ValueError(f"{spec['path']} has no CRS")
    return g.to_crs(WORKING_CRS)


def resolve_curation_root(curation_root: str | Path) -> Path:
    """Resolve a curation path against the repository, not the caller.

    A notebook must not do path math. marimo does not promise a working
    directory, and `__file__` changes when marimo exports a notebook to a
    script. This module holds `REPO_ROOT`, so it resolves the path one way in
    every context.
    """
    root = Path(curation_root)
    return root if root.is_absolute() else REPO_ROOT / root


def load_facilities(curation_root: str | Path,
                    filename: str = "facilities.parquet") -> pd.DataFrame:
    """Read the plain unit table of a curation sub-dataset.

    The file name changes with the case. The FacDB cases (schools, libraries,
    parks) write `facilities.parquet`. The DOHMH restaurants case writes
    `restaurants_aggregated.parquet`, which also carries the inspection proxy.

    Warning: read this file with `pandas`. It holds no geo metadata, so
    `geopandas.read_parquet` raises an error. Use `load_units` when you need
    the point layer.
    """
    path = resolve_curation_root(curation_root) / filename
    if not path.exists():
        raise FileNotFoundError(f"no unit table at {path}")
    return pd.read_parquet(path)


def load_units(curation_root: str, uid_col: str = "uid",
               filename: str = "facilities.parquet"):
    """Read the FacDB facility table and make a point layer.

    Warning: read this file with `pandas`, not `geopandas`. It holds no geo
    metadata, so `geopandas.read_parquet` raises an error.

    The unit position comes from this file, not from the pairs manifest. The
    pairs manifest holds the camera position, which sits up to 80 ft away and can
    fall in the neighbor polygon.
    """
    import geopandas as gpd

    df = load_facilities(curation_root, filename=filename)
    missing = [c for c in (uid_col, "latitude", "longitude") if c not in df.columns]
    if missing:
        raise KeyError(f"{path} has no column(s): {missing}")
    df = df.dropna(subset=["latitude", "longitude"])
    pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(WORKING_CRS)
    return pts.rename(columns={uid_col: "unit_uid"})


def aggregate(unit_scores: pd.DataFrame, units_gdf, layer: str,
              min_units: int = 3) -> pd.DataFrame:
    """Aggregate the unit scores into one geography layer.

    Args:
        unit_scores: The output of `_provenance.score_units`.
        units_gdf: The output of `load_units`.
        min_units: Drop a polygon that holds fewer units than this. A polygon
            with 1 unit gives a mean that is only that unit.

    Returns one row for each polygon that holds enough units.
    """
    import geopandas as gpd

    poly = load_layer(layer)
    spec = LAYERS[layer]
    key, name = spec["key"], spec["name"]

    pts = units_gdf.merge(unit_scores, on="unit_uid", how="inner")
    if pts.empty:
        return pd.DataFrame()

    cols = [key] + ([name] if name != key else [])
    joined = gpd.sjoin(
        pts[["unit_uid", "mean_score", "n_comparisons", "abstention_rate", "geometry"]],
        poly[cols + ["geometry"]],
        how="inner",
        predicate="within",
    )

    out = joined.groupby(key).agg(
        n_units=("unit_uid", "nunique"),
        mean_score=("mean_score", "mean"),
        sd_score=("mean_score", "std"),
        total_comparisons=("n_comparisons", "sum"),
        mean_abstention=("abstention_rate", "mean"),
    ).reset_index()

    if name != key:
        labels = joined.groupby(key)[name].first().reset_index()
        out = out.merge(labels, on=key, how="left")

    out = out[out.n_units >= min_units].sort_values("mean_score", ascending=False)
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)
