"""The corrected road-quality proxies: a roadway denominator, and demand.

Why this module exists
----------------------
`_proxies.py` builds the road-quality proxies that the paper shipped. An audit
of 2026-08-31 found 3 problems in them, and this module holds the corrections.
`_proxies.py` stays as it is, so every old number is still reproducible.

| # | Problem in `_proxies.py` | Correction here |
|---|--------------------------|-----------------|
| 1 | `pothole_by_layer` divides repairs by km^2 | Divide by the centerline mile |
| 2 | `pavement_by_layer` takes a plain segment mean | Weight each segment by its length |
| 3 | The case reads repairs only | Add the 311 pothole complaints |

The denominator argument
------------------------
A pothole happens on a roadway, not on land. A dense street grid holds more
roadway inside the same square kilometre, thus repairs for each km^2 rises with
street density and not only with bad pavement. The rate needs the roadway
itself in the denominator, so this module measures the centerline miles of each
polygon from the NYC street centerline (CSCL).

The length argument
-------------------
The DOT pavement rating scores a street segment. A 60 ft block and a 1 mi
avenue each give 1 rating, thus a plain mean over segments lets the short block
speak as loudly as the avenue. A driver meets the avenue for a mile. This
module weights each rating by the length of its segment.

Supply and demand
-----------------
The 2 pothole sources answer different questions:

| Source | Reads | Warning |
|--------|-------|---------|
| Closed work orders (x9wy-ing4) | The city REPAIR, the supply side | A fast borough looks worse |
| 311 complaints (erm2-nwe9) | The resident REPORT, the demand side | A quiet borough looks better |

Neither one counts potholes. Read the 2 together, and read both beside the DOT
pavement rating, which is the only source that scores the pavement itself.

The image level does not change
-------------------------------
The image comparison of the road case attaches the nearest rated segment to
each image (`_proxies.attach_nearest_segment`). The length weight enters an
AREA mean only, thus this module never touches the image join and the image
numbers of the case stay exactly as they were.

Orientation
-----------
Every value leaves this module oriented "higher is better", the same as
`_proxies`. The 2 pothole rates carry a sign of -1 before they leave.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DIR = REPO_ROOT / "data" / "external" / "nyc_street_centerline"

#: The roadway lines of the city, in the working CRS, as one cache file.
LINES_PARQUET = EXTERNAL_DIR / "cscl_roadway_lines_2263.parquet"

#: The reusable result: the centerline miles of every polygon of every layer.
#: Another case can read this file and needs no centerline pull of its own.
MILES_PARQUET = EXTERNAL_DIR / "centerline_miles_by_layer.parquet"

#: What the pull held, and when it happened.
META_JSON = EXTERNAL_DIR / "centerline_pull.json"

#: NYC Street Centerline (CSCL), on NYC Open Data.
CENTERLINE_RESOURCE = "inkn-q76z"

#: 311 Service Requests from 2010 to Present. This source holds more than
#: 40,000,000 rows. Never fetch it without a server-side filter.
SERVICE_311_RESOURCE = "erm2-nwe9"

#: Street Pothole Work Orders - Closed. The same source that `_proxies` reads.
POTHOLE_RESOURCE = "x9wy-ing4"

#: DOT In-house Street Resurfacing Projects. It records each paving job with
#: the block it covered and the date the work ended.
RESURFACING_RESOURCE = "ffaf-8mrv"

#: The CSCL roadway types that carry traffic: 1 street, 2 highway.
#:
#: The other codes are a bridge (3), a tunnel (4), a boardwalk (5), a path (6),
#: a step street (7), a driveway (8), a ramp (9), an alley (10), an unknown
#: type (11), a non-physical segment (12), a U-turn (13), and a ferry route
#: (14). A pothole work order and a 311 pothole complaint both name a street or
#: a highway, thus these 2 codes make the denominator.
ROADWAY_TYPES = (1, 2)

#: The CSCL status of a constructed segment. 103,429 of the 103,466 roadway
#: segments carry it, thus the filter drops 37 rows that are not in service.
CENTERLINE_STATUS = "2"

FT_PER_MILE = 5280.0

#: The complaint types of the 311 source that name a pothole on a roadway.
#: A bridge deck and a tunnel are not in `ROADWAY_TYPES`, thus their pothole
#: complaints stay out and the numerator matches the denominator.
POTHOLE_311_TYPES = ("Street Condition", "Highway Condition")

#: A polygon under this many centerline miles gives an unstable rate. 0.25 mi
#: is about 3 city blocks.
MIN_MILES = 0.25


# ---------------------------------------------------------------- the layers

def _layer_key(layer: str) -> str:
    """The join key column of one geography layer."""
    from research.notebooks.cvpr import _geography as geo

    return geo.LAYERS[layer]["key"]


# ------------------------------------------------------- the street centerline

def centerline_lines(use_cache: bool = True):
    """Read the CSCL roadway lines and put them into the working CRS.

    Warning: filter on the server. The source holds 122,263 rows, and the
    roadway filter keeps 103,429 of them.

    The cache file is `LINES_PARQUET`. Delete it to force a new pull.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    if use_cache and LINES_PARQUET.exists():
        return gpd.read_parquet(LINES_PARQUET)

    types = ",".join(str(int(t)) for t in ROADWAY_TYPES)
    entry = {"resource_id": CENTERLINE_RESOURCE, "index": "centerline"}
    df = px.fetch(
        entry,
        select="the_geom,physicalid,rw_type,status,full_street_name",
        where=f"rw_type in ({types}) AND status='{CENTERLINE_STATUS}'",
    )
    if df.empty:
        raise RuntimeError("the centerline pull returned no row")

    df = df.dropna(subset=["the_geom"])
    geom = df["the_geom"].apply(px._to_shape)
    g = gpd.GeoDataFrame(
        df.drop(columns=["the_geom"]), geometry=list(geom), crs="EPSG:4326"
    )
    g = g[g.geometry.notna()].to_crs(geo.WORKING_CRS)
    g["length_ft"] = g.geometry.length

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    g.to_parquet(LINES_PARQUET, index=False)
    META_JSON.write_text(json.dumps({
        "resource_id": CENTERLINE_RESOURCE,
        "name": "NYC Street Centerline (CSCL)",
        "documentation": (
            "https://data.cityofnewyork.us/City-Government/Centerline/inkn-q76z"
        ),
        "where": f"rw_type in ({types}) AND status='{CENTERLINE_STATUS}'",
        "n_segments": int(len(g)),
        "total_miles": round(float(g["length_ft"].sum() / FT_PER_MILE), 1),
        "crs": geo.WORKING_CRS,
        "pulled": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
    }, indent=2))
    return g


def centerline_miles(use_cache: bool = True, lines=None) -> pd.DataFrame:
    """The centerline miles of every polygon of the 3 layers, as 1 long frame.

    | Column | Holds |
    |--------|-------|
    | `layer` | `nta`, `community_district`, or `census_tract` |
    | `area_key` | The key of the polygon, as text |
    | `centerline_miles` | The roadway inside the polygon |
    | `n_segments` | The centerline segments that touch the polygon |

    The frame is long, because the 3 layers name their key differently. Use
    `miles_for_layer` to get 1 layer with its own key column.

    Each line is CUT at the polygon border, thus a segment that crosses 2
    polygons gives each one its true part. A sum over 1 layer therefore returns
    the roadway of the city, minus the part in the areas with no residents.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo

    if use_cache and MILES_PARQUET.exists():
        return pd.read_parquet(MILES_PARQUET)

    lines = centerline_lines() if lines is None else lines
    frames: List[pd.DataFrame] = []
    for layer in geo.LAYERS:
        key = _layer_key(layer)
        poly = geo.load_layer(layer)[[key, "geometry"]].copy()
        poly[key] = poly[key].astype(str)
        cut = gpd.overlay(
            lines[["physicalid", "geometry"]], poly, how="intersection",
            keep_geom_type=True,
        )
        cut["length_ft"] = cut.geometry.length
        out = cut.groupby(key).agg(
            length_ft=("length_ft", "sum"),
            n_segments=("physicalid", "nunique"),
        ).reset_index()
        out["centerline_miles"] = out["length_ft"] / FT_PER_MILE
        out = out.rename(columns={key: "area_key"})
        out.insert(0, "layer", layer)
        frames.append(out[["layer", "area_key", "centerline_miles", "n_segments"]])

    miles = pd.concat(frames, ignore_index=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    miles.to_parquet(MILES_PARQUET, index=False)
    return miles


def miles_for_layer(layer: str, use_cache: bool = True) -> pd.DataFrame:
    """The centerline miles of 1 layer, with the key column of that layer."""
    key = _layer_key(layer)
    m = centerline_miles(use_cache=use_cache)
    m = m[m["layer"] == layer].copy()
    return m.rename(columns={"area_key": key})[
        [key, "centerline_miles", "n_segments"]]


def _per_mile(counts: pd.DataFrame, layer: str, key: str,
              count_col: str, min_miles: float = MIN_MILES) -> pd.DataFrame:
    """Turn a count for each polygon into a NEGATED rate for each mile.

    A pothole marks a road that failed, thus the sign is -1 and the value
    leaves this module oriented "higher is better".
    """
    miles = miles_for_layer(layer)
    agg = counts.merge(miles, on=key, how="inner")
    agg = agg[agg["centerline_miles"] >= min_miles]
    agg["proxy_mean"] = -(agg[count_col] / agg["centerline_miles"])
    agg["n_units_proxy"] = agg[count_col]
    out = agg[[key, "proxy_mean", "n_units_proxy", "centerline_miles"]].copy()
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# --------------------------------------------------- the pavement rating, fixed

def pavement_by_layer_weighted(layer: str, segments=None,
                               min_segments: int = 3) -> pd.DataFrame:
    """The DOT pavement rating of 1 layer, weighted by segment length.

    This is `_proxies.pavement_by_layer` with 1 change: each rating enters the
    area mean with the length of its own segment as the weight. A 60 ft block
    and a 1 mi avenue no longer speak with the same voice.

    The segment filters do not change. `_proxies.pavement_segments` still drops
    `systemrating = 0.0`, which means NOT RATED, and it still reads the window
    that the caller sets.

    Each segment goes in by its own midpoint, the same as the plain builder, so
    a long avenue lands in 1 polygon and matches how the image points land.

    Warning: this does not touch the image join. The length weight enters an
    area mean only. `_proxies.attach_nearest_segment` gives an image the rating
    of its nearest segment, and that value carries no weight at all.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    seg = px.pavement_segments() if segments is None else segments
    if seg is None or len(seg) == 0:
        return pd.DataFrame()

    pts = seg[["proxy_value", "geometry"]].copy()
    pts["weight_ft"] = seg.geometry.length
    pts["geometry"] = seg.geometry.interpolate(0.5, normalized=True)

    poly = geo.load_layer(layer)
    key = _layer_key(layer)
    hit = gpd.sjoin(pts, poly[[key, "geometry"]], how="inner", predicate="within")
    if hit.empty:
        return pd.DataFrame()

    hit = hit.assign(_wv=hit["proxy_value"] * hit["weight_ft"])
    out = hit.groupby(key).agg(
        _w=("weight_ft", "sum"),
        _wv=("_wv", "sum"),
        n_units_proxy=("proxy_value", "size"),
    ).reset_index()
    out["proxy_mean"] = out["_wv"] / out["_w"].where(out["_w"] > 0)
    out["rated_miles"] = out["_w"] / FT_PER_MILE
    out = out[out.n_units_proxy >= min_segments].dropna(subset=["proxy_mean"])
    out = out[[key, "proxy_mean", "n_units_proxy", "rated_miles"]]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# --------------------------------------------------- the repairs, per mile

def pothole_repairs_per_mile(layer: str, since_year: int = 2024,
                             min_miles: float = MIN_MILES) -> pd.DataFrame:
    """Closed pothole work orders for each centerline mile, negated.

    This is `_proxies.pothole_by_layer` with a roadway denominator in place of
    the land denominator. The numerator and the filters do not change.

    Warning: this counts REPAIRS, not defects. A borough that fixes potholes
    quickly looks worse here than one that ignores them. Read it beside the
    311 complaint rate and the pavement rating, never alone.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    entry = {"resource_id": POTHOLE_RESOURCE, "index": "pothole"}
    df = px.fetch(
        entry,
        select="the_geom,rptdate,boro",
        where=f"rptdate > '{int(since_year)}-01-01'",
    )
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["the_geom"])
    geom = df["the_geom"].apply(px._to_shape)
    g = gpd.GeoDataFrame(
        df.drop(columns=["the_geom"]), geometry=list(geom), crs="EPSG:4326"
    )
    g = g[g.geometry.notna()].to_crs(geo.WORKING_CRS)
    g["geometry"] = g.representative_point()

    key = _layer_key(layer)
    poly = geo.load_layer(layer)[[key, "geometry"]]
    hit = gpd.sjoin(g[["geometry"]], poly, how="inner", predicate="within")
    counts = hit.groupby(key).size().rename("n_repairs").reset_index()
    counts[key] = counts[key].astype(str)
    return _per_mile(counts, layer, key, "n_repairs", min_miles=min_miles)


# ------------------------------------------------- the complaints, per mile

def pothole_complaints(since_year: int = 2024) -> pd.DataFrame:
    """The 311 pothole complaints, as located points.

    Warning: 311 (erm2-nwe9) holds more than 40,000,000 rows. This call sends
    the complaint type, the descriptor, and the date to the server, thus it
    moves about 148,000 rows and never the whole table.

    The demand side of the case. A resident reports a pothole; the city then
    opens a work order. The 2 counts are not the same number, and the gap
    between them is itself a finding.
    """
    from research.notebooks.cvpr import _proxies as px

    types = ",".join(f"'{t}'" for t in POTHOLE_311_TYPES)
    where = (
        f"created_date >= '{int(since_year)}-01-01T00:00:00' "
        f"AND complaint_type in ({types}) "
        f"AND upper(descriptor) like '%POTHOLE%' "
        f"AND latitude IS NOT NULL"
    )
    entry = {"resource_id": SERVICE_311_RESOURCE, "index": "311_pothole"}
    df = px.fetch(
        entry,
        select=("unique_key,created_date,complaint_type,descriptor,"
                "borough,latitude,longitude"),
        where=where,
    )
    if df.empty:
        return df
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    return df.dropna(subset=["latitude", "longitude", "created_date"])


def pothole_complaints_per_mile(layer: str, since_year: int = 2024,
                                complaints: Optional[pd.DataFrame] = None,
                                min_miles: float = MIN_MILES) -> pd.DataFrame:
    """311 pothole complaints for each centerline mile, negated.

    A complaint marks a road that a resident calls bad, thus the sign is -1.

    Warning: this counts a REPORT, not a pothole. A neighborhood that calls 311
    often looks worse than a quiet one with the same pavement. Read it beside
    the repair rate: the 2 sides differ, and the difference is the point.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo

    df = pothole_complaints(since_year=since_year) if complaints is None else complaints
    if df is None or df.empty:
        return pd.DataFrame()

    pts = gpd.GeoDataFrame(
        df[["unique_key"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(geo.WORKING_CRS)

    key = _layer_key(layer)
    poly = geo.load_layer(layer)[[key, "geometry"]]
    hit = gpd.sjoin(pts, poly, how="inner", predicate="within")
    counts = hit.groupby(key)["unique_key"].nunique().rename(
        "n_complaints").reset_index()
    counts[key] = counts[key].astype(str)
    return _per_mile(counts, layer, key, "n_complaints", min_miles=min_miles)


# ------------------------------------------------ the resurfacing age

def resurfacing_blocks(use_cache: bool = True):
    """The LAST in-house paving of each street block, as lines.

    The source records 252,054 block paving jobs from 2002 onward, and they
    fall on 92,314 different blocks. This keeps the newest job of each block.

    Warning: `location_wkt` already carries EPSG:2263 coordinates, thus the
    geometry needs no transform.

    Warning: the record holds the IN-HOUSE work of DOT only. A block that a
    capital contract rebuilt is not here, and a block that DOT never paved
    since 2002 is not here either. Thus the age is right-censored, and the
    proxy reads "how long since DOT last paved this block", never "how old the
    pavement is".
    """
    import geopandas as gpd
    from shapely import wkt as shapely_wkt

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    entry = {"resource_id": RESURFACING_RESOURCE, "index": "resurfacing"}
    df = px.fetch(
        entry,
        select=("location_segment_id,location_wkt,"
                "location_actual_paving_end_date,location_actual_lane_miles"),
        where=("project_type='Block' "
               "AND location_actual_paving_end_date IS NOT NULL "
               "AND location_wkt IS NOT NULL"),
        use_cache=use_cache,
    )
    if df.empty:
        return df

    df["paved"] = pd.to_datetime(
        df["location_actual_paving_end_date"], errors="coerce")
    df = df.dropna(subset=["paved", "location_segment_id"])
    df = df.sort_values("paved").drop_duplicates(
        "location_segment_id", keep="last")

    geom = df["location_wkt"].apply(
        lambda s: shapely_wkt.loads(s) if isinstance(s, str) else None)
    g = gpd.GeoDataFrame(
        df[["location_segment_id", "paved"]].copy(),
        geometry=list(geom), crs=geo.WORKING_CRS,
    )
    return g[g.geometry.notna()].reset_index(drop=True)


def resurfacing_age_by_layer(layer: str, blocks=None, min_segments: int = 3,
                             as_of: Optional[pd.Timestamp] = None
                             ) -> pd.DataFrame:
    """The years since the last in-house paving, length-weighted and negated.

    An old surface is a worse surface, thus the sign is -1 and the value leaves
    this module oriented "higher is better".

    Each block enters the area mean with its own length, the same rule that
    `pavement_by_layer_weighted` applies, so a short block cannot outvote an
    avenue.

    Warning: read the censoring of `resurfacing_blocks` before you read this
    number. It measures the DOT paving cycle, not the age of the pavement.
    """
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo

    g = resurfacing_blocks() if blocks is None else blocks
    if g is None or len(g) == 0:
        return pd.DataFrame()

    ref = pd.Timestamp(as_of) if as_of is not None else g["paved"].max()
    g = g[g["paved"] <= ref].copy()
    g["age_years"] = (ref - g["paved"]).dt.days / 365.25

    pts = g[["age_years"]].copy()
    pts["weight_ft"] = g.geometry.length
    pts = gpd.GeoDataFrame(
        pts, geometry=g.geometry.interpolate(0.5, normalized=True),
        crs=geo.WORKING_CRS,
    )

    key = _layer_key(layer)
    poly = geo.load_layer(layer)[[key, "geometry"]]
    hit = gpd.sjoin(pts, poly, how="inner", predicate="within")
    if hit.empty:
        return pd.DataFrame()

    hit = hit.assign(_wv=hit["age_years"] * hit["weight_ft"])
    out = hit.groupby(key).agg(
        _w=("weight_ft", "sum"),
        _wv=("_wv", "sum"),
        n_units_proxy=("age_years", "size"),
    ).reset_index()
    out["mean_age_years"] = out["_wv"] / out["_w"].where(out["_w"] > 0)
    out["proxy_mean"] = -out["mean_age_years"]
    out["paved_miles"] = out["_w"] / FT_PER_MILE
    out = out[out.n_units_proxy >= min_segments].dropna(subset=["proxy_mean"])
    out = out[[key, "proxy_mean", "n_units_proxy", "mean_age_years",
               "paved_miles"]]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# ----------------------------------------------------- old against new

def _tau(x, y) -> float:
    """Kendall tau-b, the measure that the master table prints."""
    from scipy import stats

    if len(x) < 3:
        return float("nan")
    return float(stats.kendalltau(x, y, variant="b").statistic)


def compare(model_agg: pd.DataFrame, proxy_frames: Dict[str, pd.DataFrame],
            layer: str = "community_district",
            include_jia: bool = False) -> pd.DataFrame:
    """Correlate 1 model score frame against several proxy frames.

    Args:
        model_agg: The `aggregated` export of the case, with `model`, `layer`,
            the layer key, and `mean_score`.
        proxy_frames: A name for each frame that `_proxies` or this module
            builds. Each frame holds `layer`, the layer key, and `proxy_mean`.
        include_jia: `True` keeps the areas that hold no residents. The default
            drops them, the same as `_results_table.load_case`.

    Returns 1 row for each model and proxy, with n, Pearson r, Spearman rho,
    and Kendall tau-b.
    """
    from research.notebooks.cvpr import _geography as geo

    key = _layer_key(layer)
    drop = frozenset() if include_jia else geo.excluded_keys(layer)
    rows: List[Dict[str, Any]] = []
    m_all = model_agg[model_agg["layer"] == layer].copy()
    m_all[key] = m_all[key].astype(str)
    if drop:
        m_all = m_all[~m_all[key].isin(drop)]

    for name, frame in proxy_frames.items():
        if frame is None or frame.empty:
            continue
        p = frame[frame["layer"] == layer].copy()
        p[key] = p[key].astype(str)
        if drop:
            p = p[~p[key].isin(drop)]
        for model in sorted(m_all["model"].unique()):
            j = m_all[m_all["model"] == model][[key, "mean_score"]].merge(
                p[[key, "proxy_mean"]], on=key, how="inner").dropna()
            rows.append({
                "proxy": name,
                "model": model,
                "layer": layer,
                "n": len(j),
                "pearson_r": round(j["mean_score"].corr(j["proxy_mean"]), 4)
                if len(j) > 2 else float("nan"),
                "spearman_rho": round(
                    j["mean_score"].corr(j["proxy_mean"], method="spearman"), 4)
                if len(j) > 2 else float("nan"),
                "kendall_tau": round(_tau(j["mean_score"], j["proxy_mean"]), 4),
            })
    return pd.DataFrame(rows)


def street_density(layer: str = "community_district") -> pd.DataFrame:
    """The centerline miles for each km^2 of every polygon of 1 layer.

    The audit says that a repair count for each km^2 partly measures street
    density. This frame lets a reader test that: correlate the old proxy with
    this one, and read how much of it the denominator carries.
    """
    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    key = _layer_key(layer)
    poly = geo.load_layer(layer)[[key, "geometry"]].copy()
    poly[key] = poly[key].astype(str)
    poly["area_km2"] = poly.area / px.SQFT_PER_KM2
    out = poly.drop(columns=["geometry"]).merge(
        miles_for_layer(layer), on=key, how="inner")
    out["miles_per_km2"] = out["centerline_miles"] / out["area_km2"]
    out["proxy_mean"] = out["miles_per_km2"]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ the export

#: The 4 new proxies, in the order the notebook shows them.
PROXY_ORDER = (
    "dot_pavement_rating_weighted",
    "pothole_repairs_per_mile",
    "pothole_complaints_per_mile",
    "resurfacing_age",
)

#: What `proxy_metric` holds for each new proxy.
PROXY_METRIC = {
    "dot_pavement_rating_weighted": "systemrating, weighted by segment length",
    "pothole_repairs_per_mile": "closed work orders for each centerline mile",
    "pothole_complaints_per_mile": "311 complaints for each centerline mile",
    "resurfacing_age": "years since the last in-house paving, length-weighted",
}

#: What `proxy_window` holds for each new proxy.
#:
#: Warning: an old window string must never change. `road_quality_proxy_v1.1.0`
#: carries the old strings, and version 1.2.0 repeats every old row exactly.
PROXY_WINDOW = {
    "dot_pavement_rating_weighted": "2024-01 to 2026-07 inspections",
    "pothole_repairs_per_mile": "2024-01 to 2026-07 work orders",
    "pothole_complaints_per_mile": "2024-01 to 2026-08 complaints",
    "resurfacing_age": "2002-03 to 2026-08 paving, age as of 2026-08-14",
}

#: The recipe sources of each new proxy. A rate reads 2 sources: the count,
#: and the centerline that makes the denominator. `proxy_source` then names
#: both, so a reader of the parquet can find the whole recipe.
PROXY_SOURCE_INDEX = {
    "dot_pavement_rating_weighted": ("0",),
    "pothole_repairs_per_mile": ("1", "4"),
    "pothole_complaints_per_mile": ("5", "4"),
    "resurfacing_age": ("6",),
}

#: The label that `_results_table.PROXY_LABEL` needs for each new row.
#: `_results_table.py` is not this module's file, thus the notebook prints
#: these lines and a maintainer copies them.
PROXY_LABEL = {
    "dot_pavement_rating_weighted": "DOT pavement rating (length-weighted)",
    "pothole_repairs_per_mile": "Pothole repairs per mile (negated)",
    "pothole_complaints_per_mile": "Pothole complaints per mile (negated)",
    "resurfacing_age": "Years since resurfacing (negated)",
}


def source_id(entry: Dict[str, Any]) -> str:
    """The dataset id of a recipe source, or the stem of its file."""
    return entry.get("resource_id") or Path(entry["file"]).stem


def source_text(recipe, proxy: str) -> str:
    """The `proxy_source` text of 1 corrected proxy: every id it reads."""
    want = PROXY_SOURCE_INDEX[proxy]
    by_index = {str(e.get("index")): e for e in recipe}
    missing = [i for i in want if i not in by_index]
    if missing:
        raise KeyError(f"recipe holds no source(s) {missing} for {proxy!r}")
    return " + ".join(source_id(by_index[i]) for i in want)


def stamp(frame: pd.DataFrame, proxy: str, recipe) -> pd.DataFrame:
    """Add the proxy name and the 3 provenance columns to 1 frame.

    Warning: never write a fixed proxy name beside a value that a caller
    chooses. The name comes from the same string that selected the builder,
    thus the 2 can never drift apart.
    """
    out = frame.copy()
    out["proxy"] = proxy
    out["proxy_metric"] = PROXY_METRIC[proxy]
    out["proxy_source"] = source_text(recipe, proxy)
    out["proxy_window"] = PROXY_WINDOW[proxy]
    return out


def area_export(recipe, min_segments: int = 3,
                segments=None, complaints: Optional[pd.DataFrame] = None,
                blocks=None, proxies=PROXY_ORDER) -> pd.DataFrame:
    """The corrected proxies of every layer, as 1 long frame.

    Every row carries `proxy`, `proxy_metric`, `proxy_source`, and
    `proxy_window`. The frame holds the export schema only, thus the helper
    columns (`centerline_miles`, `rated_miles`) stay out of it and the parquet
    keeps the shape that version 1.1.0 has.
    """
    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies as px

    seg = px.pavement_segments() if segments is None else segments
    cmp_pts = pothole_complaints() if complaints is None else complaints
    blk = resurfacing_blocks() if blocks is None else blocks

    builders = {
        "dot_pavement_rating_weighted":
            lambda L: pavement_by_layer_weighted(
                L, segments=seg, min_segments=min_segments),
        "pothole_repairs_per_mile": pothole_repairs_per_mile,
        "pothole_complaints_per_mile":
            lambda L: pothole_complaints_per_mile(L, complaints=cmp_pts),
        "resurfacing_age":
            lambda L: resurfacing_age_by_layer(
                L, blocks=blk, min_segments=min_segments),
    }

    keep = ["layer", "proxy_mean", "n_units_proxy"]
    frames: List[pd.DataFrame] = []
    for proxy in proxies:
        for layer in geo.LAYERS:
            out = builders[proxy](layer)
            if out is None or out.empty:
                continue
            key = _layer_key(layer)
            frames.append(stamp(out[keep + [key]], proxy, recipe))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())


def merge_export(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Put the corrected rows under the old rows, with 1 schema.

    The old rows do not change. Version 1.1.0 goes in whole, so a reader of
    version 1.2.0 can still reconcile every published number, and the new rows
    carry the corrected values.

    Warning: this raises an error when a proxy name appears on both sides. A
    duplicate name would give a reader 2 values for 1 polygon.
    """
    clash = set(old["proxy"].unique()) & set(new["proxy"].unique())
    if clash:
        raise ValueError(f"the old and the new export share proxy name(s): "
                         f"{sorted(clash)}")
    cols = list(old.columns)
    for c in cols:
        if c not in new.columns:
            new[c] = None
    return pd.concat([old, new[cols]], ignore_index=True)
