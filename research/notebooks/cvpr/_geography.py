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

The areas with no residents
---------------------------
2 of the 3 layers hold areas that no person lives in. NYC calls a community
district of this kind a joint interest area (JIA): a large park, an airport, or
Rikers Island. The NTA file marks the same kind of area in its `NTAType`
column. Such an area has no household income, no school, and almost no crime
record, thus a proxy value there describes nothing.

`load_layer` drops these areas by default. Pass `include_jia=True` to keep them
and reproduce an older number. See `JIA_BORO_CD` and `NTA_RESIDENTIAL_TYPE`.

The census tract file carries no type column of this kind, thus this module
leaves that layer whole. See `excluded_keys`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import pandas as pd

__version__ = "1.1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
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


#: The 12 community districts that hold no residents.
#:
#: NYC names them joint interest areas. The codes are the large parks (164
#: Central Park, 226-228 the Bronx parks, 355-356 Prospect Park and the
#: Brooklyn parks, 480 and 484 the Queens parks, 595 the Staten Island parks),
#: the 2 airports (481 LaGuardia, 482 JFK), and Rikers Island (483).
#:
#: Why the default drops them: a JIA has no household income, no school, and
#: almost no resident crime record. A census proxy there is a boundary
#: artifact, not a measurement of a place. The layer file holds 71 features,
#: thus 12 of 71 areas carry no readable proxy.
JIA_BORO_CD: FrozenSet[str] = frozenset({
    "164", "226", "227", "228", "355", "356",
    "480", "481", "482", "483", "484", "595",
})

#: The `NTAType` value of a residential neighborhood.
#:
#: The NTA file marks each area with a type. `0` is a residential
#: neighborhood, and the other 5 codes mark an area with no residents: `5`
#: Rikers Island, `6` an institution or a rail yard, `7` a cemetery, `8` an
#: airport, and `9` a park. The file holds 262 features and 197 of them are
#: residential, thus the default drops 65.
NTA_RESIDENTIAL_TYPE = "0"

#: The column that carries the NTA type. The default drops a row where this
#: column is not `NTA_RESIDENTIAL_TYPE`.
NTA_TYPE_COLUMN = "NTAType"


def load_layer(layer: str, include_jia: bool = False):
    """Read one geography layer and put it into the working CRS.

    Args:
        include_jia: `True` keeps the areas that hold no residents. The
            default drops them, because a proxy value there describes no
            place. See `JIA_BORO_CD` and `NTA_RESIDENTIAL_TYPE`.

    Warning: the census tract layer takes no such filter. Its file carries no
    type column, thus this module cannot name the non-residential tracts. A
    tract inside a park holds few units, and `min_units` already drops it.
    """
    import geopandas as gpd

    if layer not in LAYERS:
        raise KeyError(f"unknown layer {layer!r}; use one of {list(LAYERS)}")
    spec = LAYERS[layer]
    g = gpd.read_file(spec["path"])
    if g.crs is None:
        raise ValueError(f"{spec['path']} has no CRS")
    if not include_jia:
        if layer == "community_district":
            keep = ~g[spec["key"]].astype(str).isin(JIA_BORO_CD)
            g = g[keep]
        elif layer == "nta" and NTA_TYPE_COLUMN in g.columns:
            keep = g[NTA_TYPE_COLUMN].astype(str).eq(NTA_RESIDENTIAL_TYPE)
            g = g[keep]
    return g.to_crs(WORKING_CRS)


@lru_cache(maxsize=8)
def excluded_keys(layer: str) -> FrozenSet[str]:
    """The keys of one layer that `load_layer` drops by default.

    A frame that a notebook already exported holds those keys, because the
    export came before this filter. Such a frame cannot go through
    `load_layer` again, thus a reader of an export drops the keys directly.
    `_results_table.load_case` does this.

    The census tract layer gives an empty set: its file carries no type
    column, thus this module names no tract.
    """
    if layer == "community_district":
        return JIA_BORO_CD
    if layer == "nta":
        import geopandas as gpd

        spec = LAYERS[layer]
        g = gpd.read_file(spec["path"])
        if NTA_TYPE_COLUMN not in g.columns:
            return frozenset()
        drop = g[~g[NTA_TYPE_COLUMN].astype(str).eq(NTA_RESIDENTIAL_TYPE)]
        return frozenset(drop[spec["key"]].astype(str))
    return frozenset()


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


def weighted_by_area(joined: pd.DataFrame, key: str, value: str,
                     weight: str) -> pd.DataFrame:
    """The weighted mean of `value` in each area, as a 2-column frame.

    A place that took 8,000 comparisons carries a much steadier score than a
    place that took 300. The plain mean gives the 2 places the same voice, and
    the kept comparison counts of one area differ by a factor of 25 after the
    `NotSure` drop. This mean gives each place the voice its evidence buys.

    The second column is always `mean_score_weighted`. An area where every
    weight is 0 gets a NaN, and the caller then falls back to the plain mean.
    """
    w = joined[weight].astype(float).clip(lower=0.0)
    tmp = pd.DataFrame({key: joined[key], "_w": w,
                        "_wv": joined[value].astype(float) * w})
    s = tmp.groupby(key)[["_w", "_wv"]].sum()
    out = (s["_wv"] / s["_w"].where(s["_w"] > 0))
    return out.rename("mean_score_weighted").reset_index()


def aggregate(unit_scores: pd.DataFrame, units_gdf, layer: str,
              min_units: int = 3, units_total=None,
              include_jia: bool = False) -> pd.DataFrame:
    """Aggregate the unit scores into one geography layer.

    Args:
        unit_scores: The output of `_provenance.score_units`.
        units_gdf: The output of `load_units`.
        min_units: Drop a polygon that holds fewer units than this. A polygon
            with 1 unit gives a mean that is only that unit.
        units_total: The point layer of EVERY unit of the case, before the
            score join. Pass `units_gdf` here to get the 2 coverage columns.
            The default is `None`, and then the 2 columns stay absent.
        include_jia: `True` keeps the areas that hold no residents. See
            `load_layer`.

    Returns one row for each polygon that holds enough units.

    | Column | Meaning |
    |--------|---------|
    | `n_units` | The units the model scored in this area |
    | `mean_score` | The plain mean over those units |
    | `mean_score_weighted` | The same mean, weighted by comparison count |
    | `sd_score` | The standard deviation over those units |
    | `total_comparisons` | The comparisons behind the area |
    | `mean_abstention` | The mean `NotSure` rate of those units |
    | `n_units_total` | Every unit of the case in this area (optional) |
    | `coverage` | `n_units` over `n_units_total` (optional) |

    Warning: read `mean_score` against `mean_score_weighted`. The plain mean
    weights each place the same although the kept comparison counts differ by
    a factor of 25, thus the abstention rate enters the plain mean.
    """
    import geopandas as gpd

    poly = load_layer(layer, include_jia=include_jia)
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

    # The comparison-count-weighted mean, beside the plain one. It answers the
    # abstention question: a place that abstained often keeps fewer
    # comparisons, thus the plain mean over-weights it.
    out = out.merge(weighted_by_area(joined, key, "mean_score",
                                     "n_comparisons"),
                    on=key, how="left")
    out["mean_score_weighted"] = out["mean_score_weighted"].fillna(
        out["mean_score"])

    if units_total is not None:
        totals = gpd.sjoin(units_total[["unit_uid", "geometry"]],
                           poly[[key, "geometry"]], how="inner",
                           predicate="within")
        counts = (totals.groupby(key)["unit_uid"].nunique()
                  .rename("n_units_total").reset_index())
        out = out.merge(counts, on=key, how="left")
        out["coverage"] = out["n_units"] / out["n_units_total"]

    if name != key:
        labels = joined.groupby(key)[name].first().reset_index()
        out = out.merge(labels, on=key, how="left")

    out = out[out.n_units >= min_units].sort_values("mean_score", ascending=False)
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)
