"""Statistical proxies for the CVPR validation notebooks.

Each case directory holds a `recipe.json`. The recipe names the NYC Open Data
sources that supply the proxy. This module reads a recipe, gets the data, and
puts it into a shape that the notebook can compare against the model scores.

Recipe format
-------------
The file maps a string index to one source::

    {
      "0": {
        "name": "School Quality Reports Data",
        "endpoint": "https://data.cityofnewyork.us/api/v3/views/dnpx-dfnc/query.json",
        "documentation": "https://dev.socrata.com/foundry/..."
      }
    }

Add a source with the next index. These keys are optional:

| Key | Purpose |
|-----|---------|
| `resource_id` | The Socrata id. This module reads it from `endpoint` if absent. |
| `select` | A SoQL `$select` clause. |
| `where` | A SoQL `$where` clause. |

Credentials
-----------
**Warning: never write a credential into this repository.** The SODA key lives
in `.env` at the repository root, and `.env` must stay out of git. This module
reads 2 environment variables by name:

- `NYC_API_ID` — the key id
- `NYC_API_SKEY` — the key secret

The 2 values become HTTP basic authentication. Do not print them. Do not put
them in a notebook cell, a log line, or an error message.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(__file__).resolve().parent / ".proxy_cache"

# The environment variable NAMES. The values never appear in this repository.
ENV_KEY_ID = "NYC_API_ID"
ENV_KEY_SECRET = "NYC_API_SKEY"

SODA_PAGE = 50_000


def _load_dotenv_once() -> None:
    """Load `.env` into the environment if the keys are absent."""
    if os.environ.get(ENV_KEY_ID) and os.environ.get(ENV_KEY_SECRET):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:
        pass


def _auth_header() -> Dict[str, str]:
    """Build the basic-auth header from the environment.

    Returns an empty dict when no credential exists. Socrata then applies a
    lower rate limit, but the request still works.
    """
    _load_dotenv_once()
    kid = os.environ.get(ENV_KEY_ID)
    ksec = os.environ.get(ENV_KEY_SECRET)
    if not (kid and ksec):
        return {}
    token = base64.b64encode(f"{kid}:{ksec}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def has_credentials() -> bool:
    """True when both environment variables hold a value.

    This reports presence only. It never returns or logs a value.
    """
    _load_dotenv_once()
    return bool(os.environ.get(ENV_KEY_ID) and os.environ.get(ENV_KEY_SECRET))


def load_recipe(case_dir: str | Path) -> List[Dict[str, Any]]:
    """Read `recipe.json` from a case directory and return its sources."""
    root = Path(case_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    path = root / "recipe.json"
    if not path.exists():
        raise FileNotFoundError(f"no recipe at {path}")
    raw = json.loads(path.read_text())
    out = []
    for k in sorted(raw, key=lambda x: int(x) if str(x).isdigit() else 0):
        entry = dict(raw[k])
        entry["index"] = k
        entry.setdefault("resource_id", _resource_id(entry.get("endpoint", "")))
        out.append(entry)
    return out


def _resource_id(endpoint: str) -> Optional[str]:
    """Take the Socrata 4x4 id out of an endpoint URL."""
    m = re.search(r"/(?:views|resource)/([a-z0-9]{4}-[a-z0-9]{4})", endpoint or "")
    return m.group(1) if m else None


def fetch(entry: Dict[str, Any], use_cache: bool = True,
          max_rows: int = 2_000_000, where: Optional[str] = None,
          select: Optional[str] = None, group: Optional[str] = None) -> pd.DataFrame:
    """Get the rows of one recipe source, with paging and a disk cache.

    Warning: filter on the server. Some sources hold millions of rows, so a
    whole-table fetch is slow and can reach `max_rows`. Pass `where` and
    `select` to cut the table down before it crosses the network.

    Warning: this function raises an error if the source reaches `max_rows`. A
    silent cut would give a wrong answer that looks correct — the School Quality
    Reports table cut at 1,000,000 rows returns 3 schools for a metric that
    truly has 1,841.

    The first call reads the network. The cache then serves later calls. Delete
    `.proxy_cache/` to force a refresh.
    """
    rid = entry.get("resource_id")
    if not rid:
        raise ValueError(f"recipe entry {entry.get('index')} has no resource id")

    where = where if where is not None else entry.get("where")
    select = select if select is not None else entry.get("select")
    group = group if group is not None else entry.get("group")

    CACHE_DIR.mkdir(exist_ok=True)
    tag = re.sub(r"[^a-z0-9_-]", "_",
                 f"{rid}_{select or ''}_{where or ''}_{group or ''}".lower())
    cache = CACHE_DIR / f"{tag[:120]}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    headers = _auth_header()
    frames: List[pd.DataFrame] = []
    offset = 0
    truncated = True
    while offset < max_rows:
        params = {"$limit": SODA_PAGE, "$offset": offset}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        if group:
            params["$group"] = group
        url = (f"https://data.cityofnewyork.us/resource/{rid}.json?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as exc:
            # The credential lives in the Authorization header, not in the URL,
            # so the query text is safe to show. Never add the header here.
            detail = ""
            body = getattr(exc, "read", None)
            if callable(body):
                try:
                    detail = body().decode(errors="replace")[:300]
                except Exception:
                    detail = ""
            raise RuntimeError(
                f"SODA request failed for resource {rid}: {type(exc).__name__}\n"
                f"  query: {urllib.parse.urlencode(params)}\n"
                f"  server said: {detail}"
            ) from None
        if not rows:
            truncated = False
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < SODA_PAGE:
            truncated = False
            break
        offset += SODA_PAGE
        time.sleep(0.2)

    if truncated:
        raise RuntimeError(
            f"resource {rid} reached the {max_rows:,}-row cap. The result is "
            f"incomplete, so do not use it. Add a `where` filter to make the "
            f"query smaller, or raise max_rows."
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df.to_parquet(cache, index=False)
    return df


# --------------------------------------------------------------- schools case

BOROUGH_OF_DBN = {
    "M": "MANHATTAN",
    "X": "BRONX",
    "K": "BROOKLYN",
    "Q": "QUEENS",
    "R": "STATEN ISLAND",
}


def list_metrics(entry: Dict[str, Any], school_year: Optional[str] = None,
                 top: int = 40) -> pd.DataFrame:
    """List the metrics of the School Quality Reports, with a school count.

    This groups on the server, so it moves a small table, not the whole source.
    """
    where = f"school_year='{school_year}'" if school_year else None
    df = fetch(
        entry,
        select="metric_display_name,school_year,count(dbn) as n_schools",
        where=where,
        group="metric_display_name,school_year",
    )
    if df.empty:
        return df
    df["n_schools"] = pd.to_numeric(df["n_schools"], errors="coerce")
    return df.sort_values("n_schools", ascending=False).head(top).reset_index(drop=True)


def fetch_metric(entry: Dict[str, Any], metric: str,
                 school_year: str) -> pd.DataFrame:
    """Get 1 metric for 1 year, filtered on the server.

    Warning: do not fetch the whole School Quality Reports table. It holds more
    than 1,000,000 rows. This function sends the metric and the year to the
    server, so it moves only the rows that the notebook uses.
    """
    safe_metric = metric.replace("'", "''")
    where = f"metric_display_name='{safe_metric}' AND school_year='{school_year}'"
    return fetch(
        entry,
        select="dbn,school_name,metric_display_name,metric_value,school_year",
        where=where,
    )


def normalize_name(value: Any) -> str:
    """Put a school name into a form that both sources share."""
    s = re.sub(r"\s+", " ", str(value).upper())
    return re.sub(r"[^A-Z0-9 ]", "", s).strip()


def school_quality_wide(df: pd.DataFrame, metric: str,
                        school_year: Optional[str] = None) -> pd.DataFrame:
    """Take 1 metric out of the long School Quality Reports table.

    The source holds 1 row for each school, metric, and year. This selects 1
    metric and 1 year, and returns 1 row for each school.

    Args:
        metric: A value of `metric_display_name`.
        school_year: A value of `school_year`. The newest year is the default.
    """
    need = {"dbn", "school_name", "metric_display_name", "metric_value", "school_year"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"proxy table has no column(s): {sorted(missing)}")

    year = school_year or sorted(df["school_year"].dropna().unique())[-1]
    sub = df[(df.metric_display_name == metric) & (df.school_year == year)].copy()
    sub["proxy_value"] = pd.to_numeric(sub["metric_value"], errors="coerce")
    sub = sub.dropna(subset=["proxy_value"])

    out = sub.groupby("dbn").agg(
        school_name=("school_name", "first"),
        proxy_value=("proxy_value", "mean"),
    ).reset_index()
    out["borough"] = out.dbn.str[2].map(BOROUGH_OF_DBN)
    out["join_key"] = out.borough + "|" + out.school_name.map(normalize_name)
    out["metric"] = metric
    out["school_year"] = year
    return out


def load_geocode(entry: Dict[str, Any], case_dir: str | Path):
    """Read a `role: geocode` recipe entry and return a point layer.

    A geocode entry names a local file that gives each proxy key a position.
    The schools case uses the DOE School Point Locations shapefile, where the
    `ATS` column holds the DBN.

    Prefer this over a name join. A key join cannot make a false match, and a
    name join can: 2 different schools can share a name inside a borough.
    """
    import geopandas as gpd

    import _geography as geo

    root = Path(case_dir)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    path = root / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"no geocode file at {path}")

    src = f"zip://{path}" if path.suffix == ".zip" else str(path)
    g = gpd.read_file(src)
    key = entry.get("key", "ATS")
    if key not in g.columns:
        raise KeyError(f"{path.name} has no column {key!r}")
    return g.rename(columns={key: "proxy_key"}).to_crs(geo.WORKING_CRS)


def join_proxy_points(proxy: pd.DataFrame, points, key: str = "dbn") -> Dict[str, Any]:
    """Give each proxy row a position, with an exact key join.

    This replaces the name join for the schools case. It puts the proxy on the
    map without any reference to the model's unit set, so a name that does not
    match cannot move a school to the wrong place.

    Warning: full coverage is not possible, and that is expected. The proxy
    covers public schools. The model unit set comes from FacDB, which also holds
    private, charter, and postgraduate schools. Those have no DBN. The 2 sets
    describe different school populations, so compare them at the geography
    level, not school by school.
    """
    merged = points.merge(
        proxy.rename(columns={key: "proxy_key"}),
        on="proxy_key", how="inner", validate="one_to_one",
    )
    return {
        "joined": merged,
        "n_proxy": len(proxy),
        "n_points": len(points),
        "n_matched": len(merged),
        "match_rate": len(merged) / len(proxy) if len(proxy) else 0.0,
    }


def join_to_units(proxy: pd.DataFrame, facilities: pd.DataFrame) -> Dict[str, Any]:
    """Join the proxy to the FacDB units by borough and school name.

    The proxy is keyed by DBN. FacDB carries no DBN, so the join uses the
    borough and the normalized school name.

    Warning: this join is not complete. Report `match_rate` in the notebook. A
    school that does not match is absent from the comparison, and that absence
    is not random — non-public schools have no DBN at all.

    Returns a dict with the joined frame and the match statistics.
    """
    fac = facilities.copy()
    fac["join_key"] = (
        fac["borough"].map(normalize_name) + "|" + fac["facname"].map(normalize_name)
    )
    # A name can repeat inside a borough. Keep the first unit for each key, so
    # the join stays one-to-one and cannot multiply rows.
    fac = fac.drop_duplicates(subset=["join_key"])

    merged = proxy.merge(
        fac[["join_key", "uid", "facname", "factype"]],
        on="join_key", how="inner", validate="many_to_one",
    ).rename(columns={"uid": "unit_uid"})

    return {
        "joined": merged,
        "n_proxy": len(proxy),
        "n_units": len(fac),
        "n_matched": len(merged),
        "match_rate": len(merged) / len(proxy) if len(proxy) else 0.0,
    }


# ----------------------------------------------------------- restaurants case

# Warning: read the orientation before you read a correlation.
#
# The DOHMH inspection score counts violation points, so a LOW score is a CLEAN
# restaurant. The model answers "which restaurant would you rather eat at", so a
# HIGH model score is a better restaurant. The 2 scales point in opposite
# directions.
#
# Each metric below states `higher_is_better` and supplies a `sign`. The loader
# multiplies the raw value by `sign`, so every proxy value leaves this module
# oriented as "higher is better". A positive correlation then always means the
# model and the proxy agree. Do not undo this in a notebook.
RESTAURANT_METRICS: Dict[str, Dict[str, Any]] = {
    "inspection_score": {
        "column": "last_score",
        "sign": -1,
        "higher_is_better": False,
        "label": "DOHMH inspection score (violation points, negated)",
        "note": "A grade needs 0-13 points, B needs 14-27, C needs 28 or more.",
    },
    "grade_a_rate": {
        "column": None,
        "sign": 1,
        "higher_is_better": True,
        "label": "Share of inspections that earned grade A",
        "note": "n_grade_a divided by n_inspections.",
    },
}


def restaurant_proxy(units: pd.DataFrame, metric: str = "inspection_score",
                     min_inspections: int = 1) -> pd.DataFrame:
    """Build the restaurants proxy from the DOHMH curation table.

    The DOHMH tooling writes `restaurants_aggregated.parquet`. It already holds
    the inspection history and the position, so this case needs no network call
    and no name join. `uid` matches `unit_uid` in the pairs manifest exactly.

    Every returned value is oriented so that higher is better. See the note on
    `RESTAURANT_METRICS`.

    Args:
        min_inspections: Drop a restaurant with fewer inspections than this. A
            single inspection gives a noisy grade.
    """
    if metric not in RESTAURANT_METRICS:
        raise KeyError(f"unknown metric {metric!r}; use one of {list(RESTAURANT_METRICS)}")
    spec = RESTAURANT_METRICS[metric]
    df = units.copy()

    if metric == "grade_a_rate":
        n = pd.to_numeric(df.get("n_inspections"), errors="coerce")
        a = pd.to_numeric(df.get("n_grade_a"), errors="coerce")
        raw = (a / n).where(n > 0)
    else:
        raw = pd.to_numeric(df[spec["column"]], errors="coerce")

    df["proxy_value"] = raw * spec["sign"]
    df["proxy_key"] = df["uid"]

    if "n_inspections" in df.columns:
        df = df[pd.to_numeric(df.n_inspections, errors="coerce").fillna(0) >= min_inspections]

    out = df.dropna(subset=["proxy_value", "proxy_key"])
    keep = [c for c in ("proxy_key", "proxy_value", "camis", "facname",
                        "latitude", "longitude", "cuisine_description") if c in out.columns]
    out = out[keep].copy()
    out.attrs["metric"] = metric
    out.attrs["label"] = spec["label"]
    out.attrs["higher_is_better"] = True  # after the sign flip
    return out


def points_from_latlon(df: pd.DataFrame):
    """Make a point layer from latitude and longitude columns."""
    import geopandas as gpd

    import _geography as geo

    d = df.dropna(subset=["latitude", "longitude"])
    return gpd.GeoDataFrame(
        d,
        geometry=gpd.points_from_xy(d["longitude"], d["latitude"]),
        crs="EPSG:4326",
    ).to_crs(geo.WORKING_CRS)


# ------------------------------------------------------- census-tract income

# ACS 5-Year 2024, table S1901, on disk. `scripts/pairwise_socioeconomic_
# regression.py` reads the same file, so the 2 analyses share one income source.
INCOME_CSV = REPO_ROOT / "data/demo/ct/ACSST5Y2024.S1901-Data.csv"
POP_PARQUET = REPO_ROOT / "data/external/acs_tract_population/acs_b01003_nyc_tract.parquet"
INCOME_COL = "Estimate!!Households!!Median income (dollars)"


def load_tract_income() -> pd.DataFrame:
    """Read the ACS median household income for each census tract.

    Returns `geoid`, `median_income`, and `population`. The geoid is the
    11-digit tract code, which matches the `geoid` column of the tract layer.
    """
    inc = pd.read_csv(INCOME_CSV, header=1, dtype=str, low_memory=False)
    geoid = inc["Geography"].str.extract(r"US(\d{11})$")[0]
    med = pd.to_numeric(inc[INCOME_COL], errors="coerce")
    out = pd.DataFrame({"geoid": geoid, "median_income": med.values})
    out = out.dropna(subset=["geoid"])

    pop = pd.read_parquet(POP_PARQUET)
    pop["geoid"] = pop["geoid"].astype(str)
    return out.merge(pop[["geoid", "population"]], on="geoid", how="left")


def income_by_layer(layer: str, min_tracts: int = 1) -> pd.DataFrame:
    """Put the tract income onto 1 geography layer.

    | Layer | Method |
    |-------|--------|
    | `census_tract` | A direct join. The value is the true ACS median. |
    | `nta`, `community_district` | A population-weighted mean of the tract medians. |

    **Warning: outside the tract layer the value is NOT a median.** You cannot
    average medians, and no median of medians is a median of the whole area.
    This returns a population-weighted mean of the tract medians, which is the
    usual approximation. Call it that in the paper. Use the tract layer when a
    true median matters.

    The method is sound because the geographies nest. Every 2020 census tract
    puts its interior point in exactly 1 NTA and exactly 1 community district —
    checked on 2026-08-12, 2,325 of 2,325 tracts, with no tract in 2 polygons.
    Thus the NYC atomic polygons (`wgbs-damt`) are not needed for this step.
    A strict `within` test matches only 681 tracts, because a shared border
    defeats it. Use an interior point, not `within`.
    """
    import geopandas as gpd

    import _geography as geo

    tracts = gpd.read_file(geo.LAYERS["census_tract"]["path"])
    tracts["geoid"] = tracts["geoid"].astype(str)
    tracts = tracts.to_crs(geo.WORKING_CRS)
    inc = load_tract_income()
    tracts = tracts.merge(inc, on="geoid", how="left")

    if layer == "census_tract":
        key = geo.LAYERS[layer]["key"]
        out = tracts.dropna(subset=["median_income"])[[key, "median_income"]].copy()
        out = out.rename(columns={"median_income": "proxy_mean"})
        out["n_units_proxy"] = 1
        out.insert(0, "layer", layer)
        return out.reset_index(drop=True)

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]

    # An interior point, not `within`. A shared border makes `within` fail.
    pts = tracts.copy()
    pts["geometry"] = tracts.representative_point()
    hit = gpd.sjoin(
        pts[["geoid", "median_income", "population", "geometry"]],
        poly[[key, "geometry"]],
        how="inner", predicate="within",
    ).dropna(subset=["median_income"])

    hit["w"] = pd.to_numeric(hit["population"], errors="coerce").fillna(0.0)
    grouped = hit.groupby(key)

    rows = []
    for gk, g in grouped:
        w = g["w"].to_numpy()
        v = g["median_income"].to_numpy()
        # Fall back to a plain mean when every tract reports no population.
        mean = (v * w).sum() / w.sum() if w.sum() > 0 else v.mean()
        rows.append({key: gk, "proxy_mean": mean, "n_units_proxy": len(g)})

    out = pd.DataFrame(rows)
    out = out[out.n_units_proxy >= min_tracts]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# ----------------------------------------------------------------- PIP proxy

# The Parks Inspection Program rates a park site. Unlike income and crime, it
# measures the unit itself, so it matches the prompt "which one is better
# maintained". A rating is `A` for acceptable or `U` for unacceptable.
#
# Both metrics are already oriented as "higher is better", so neither needs a
# sign flip. A positive correlation means the model and the inspector agree.
PIP_METRICS: Dict[str, Dict[str, Any]] = {
    "cleanliness_acceptable_rate": {
        "column": "cleanliness",
        "label": "Share of inspections with acceptable cleanliness",
        "note": "Litter, glass, and graffiti drive this rating.",
    },
    "condition_acceptable_rate": {
        "column": "overall_condition",
        "label": "Share of inspections with acceptable overall condition",
        "note": "Structural and landscape condition of the site.",
    },
}


def load_parks_properties():
    """Read the DPR Parks Properties layer and return a point layer.

    The layer gives each `gispropnum` a shape. PIP names the same site with
    `prop_id`, so this layer puts a PIP rating on the map.

    Warning: FacDB carries no `prop_id`, so the proxy cannot join to the model
    unit set by key. This layer lets the proxy reach a geography on its own,
    which is the same pattern the schools case uses with the DOE point file.
    """
    import geopandas as gpd

    import _geography as geo

    g = _parks_polygons().to_crs(geo.WORKING_CRS)
    # A park is a polygon. Use an interior point so a site lands in exactly 1
    # polygon of the geography layer.
    g = g.copy()
    g["geometry"] = g.representative_point()
    return g.rename(columns={"gispropnum": "proxy_key"})


# The sources inside the FacDB "PARKS AND PLAZAS" group. They are 2 different
# populations, and only the DPR one has a maintenance proxy. Split them.
PARK_SOURCES = ("dpr_parksproperties", "nysparks_parks", "nysdec_lands")
PLAZA_SOURCES = ("dcp_pops", "dot_pedplazas", "dcp_sfpsd")


def split_parks_plazas(facilities: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Split the FacDB parks-and-plazas units into 2 populations.

    Warning: do not analyze them together. The Parks Inspection Program rates
    only DPR property, which is 2,035 of the 2,582 units. A plaza — a privately
    owned public space or a DOT pedestrian plaza — gets no PIP rating, so a
    mixed table lets the proxy look complete when it covers 79%.

    Args:
        kind: `park` or `plaza`.
    """
    if kind == "park":
        want = PARK_SOURCES
    elif kind == "plaza":
        want = PLAZA_SOURCES
    else:
        raise ValueError(f"kind must be 'park' or 'plaza', not {kind!r}")
    return facilities[facilities["datasource"].isin(want)].copy()


def attach_park_property_id(units_gdf, max_distance_ft: float = 500.0):
    """Give each park unit its DPR `gispropnum`, so PIP can join unit by unit.

    FacDB carries no property id, so the earlier version could only compare PIP
    against the model at the geography level. This spatial join lifts the
    comparison to the unit: 1 park, 1 model score, 1 inspector rating.

    Warning: the FacDB geometry for a park is a NEARBY BUILDING, not the park.
    Its median area is 0.0038 km^2, while Central Park covers 3.41 km^2. Thus
    this function uses the unit POINT and the true DPR polygon, and never the
    FacDB polygon.

    A unit inside a park polygon takes that park. A unit outside every polygon
    takes the nearest park within `max_distance_ft`. Measured 2026-08-12 on the
    2,035 DPR units: 1,684 fall inside (82.7%), and 335 of the remaining 352 sit
    within 500 ft, so 99.2% get an id.
    """
    import geopandas as gpd

    import _geography as geo

    poly = _parks_polygons().to_crs(geo.WORKING_CRS)
    pts = units_gdf.to_crs(geo.WORKING_CRS)[["unit_uid", "geometry"]].copy()

    inside = gpd.sjoin(
        pts, poly[["gispropnum", "geometry"]], how="left", predicate="within"
    )[["unit_uid", "gispropnum"]].drop_duplicates("unit_uid")

    hit = inside.dropna(subset=["gispropnum"])
    miss = pts[~pts.unit_uid.isin(hit.unit_uid)]
    if len(miss):
        near = gpd.sjoin_nearest(
            miss, poly[["gispropnum", "geometry"]],
            max_distance=max_distance_ft, how="left",
        )[["unit_uid", "gispropnum"]].drop_duplicates("unit_uid")
        hit = pd.concat([hit, near.dropna(subset=["gispropnum"])], ignore_index=True)

    return hit.rename(columns={"gispropnum": "proxy_key"}).drop_duplicates("unit_uid")


def _parks_polygons():
    """Read the DPR Parks Properties polygons, with a disk cache."""
    import geopandas as gpd

    cache = CACHE_DIR / "parks_properties.parquet"
    CACHE_DIR.mkdir(exist_ok=True)
    if cache.exists():
        return gpd.read_parquet(cache)
    url = ("https://data.cityofnewyork.us/api/geospatial/enfh-gkve"
           "?method=export&format=GeoJSON")
    g = gpd.read_file(url)
    g = g[[c for c in ("gispropnum", "name311", "acres", "geometry") if c in g.columns]]
    g.to_parquet(cache)
    return g


def parks_pip_proxy(metric: str = "cleanliness_acceptable_rate",
                    since_year: int = 2023,
                    min_inspections: int = 2) -> pd.DataFrame:
    """Build the Parks Inspection Program proxy, 1 row for each site.

    Warning: filter the year on the server. The source holds 151,484
    inspections. A whole-table fetch reaches the row cap and returns an
    incomplete answer.

    Args:
        since_year: Keep inspections from this year onward. 2023 gives about
            20,800 rows across 4 years.
        min_inspections: Drop a site with fewer inspections than this. 1
            inspection gives a rate of exactly 0.0 or 1.0.
    """
    if metric not in PIP_METRICS:
        raise KeyError(f"unknown metric {metric!r}; use one of {list(PIP_METRICS)}")
    col = PIP_METRICS[metric]["column"]

    entry = {"resource_id": "yg3y-7juh", "index": "pip"}
    raw = fetch(
        entry,
        select=f"prop_id,{col},inspection_year",
        where=f"inspection_year >= {int(since_year)}",
    )
    if raw.empty:
        return raw

    raw = raw.dropna(subset=["prop_id", col])
    raw["acceptable"] = raw[col].astype(str).str.upper().eq("A")

    # PIP inspects a ZONE inside a park, and it names a zone `B007-01`. The
    # Parks Properties layer holds only the parent site, `B007`. Thus the raw
    # prop_id matches a shape for only 40.4% of sites. Cut the suffix and roll
    # the zones up to the parent, which raises the match to about 96%.
    raw["proxy_key"] = raw["prop_id"].astype(str).str.split("-").str[0].str.strip()

    out = raw.groupby("proxy_key").agg(
        proxy_value=("acceptable", "mean"),
        n_inspections=("acceptable", "size"),
        n_zones=("prop_id", "nunique"),
    ).reset_index()
    out = out[out.n_inspections >= min_inspections]
    out.attrs["metric"] = metric
    return out.reset_index(drop=True)


# ------------------------------------------------------- road quality proxies

# DOT rates a street segment from 0 to 10, and a HIGH rating is a GOOD road.
# Thus this proxy needs no sign flip.
#
# Warning: a rating of exactly 0.0 means NOT RATED. It does not mean the worst
# pavement. 72,768 of the 514,521 rows carry it. Scored as a zero it would drag
# whole neighborhoods down and look like a real signal. Two facts identify it:
# the zeros fall from 11,043 in 2021 to 1,358 in 2026 while the rated rows hold
# steady, which is a backlog that clears rather than pavement that improves;
# and a sample zero row is an ordinary segment with a normal length and a real
# inspection date. `pavement_segments` drops them.
PAVEMENT_RESOURCE = "6yyb-pb25"
POTHOLE_RESOURCE = "x9wy-ing4"
UNRATED_PAVEMENT = "0.0"


def _to_shape(value):
    """Turn a SODA `the_geom` value into a shapely geometry.

    Warning: the disk cache changes the type. A fresh fetch gives nested Python
    lists, but a parquet round-trip gives numpy arrays, and `shapely.shape`
    raises "truth value of an array is ambiguous" on those. Thus the coordinates
    need a walk back to plain lists first.
    """
    from shapely.geometry import shape

    if not isinstance(value, dict) or "coordinates" not in value:
        return None

    def _plain(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_plain(i) for i in x]
        return x

    try:
        return shape({"type": value["type"], "coordinates": _plain(value["coordinates"])})
    except Exception:
        return None


def pavement_segments(since_year: int = 2024, drop_unrated: bool = True):
    """Read the DOT Street Pavement Ratings as a line layer.

    Warning: filter on the server. The source holds 514,521 rows. Inspections
    since 2024 give about 125,924 rated segments.

    Args:
        drop_unrated: Drop `systemrating = 0.0`, which means not rated. Set it
            to False only to inspect the unrated rows.
    """
    import geopandas as gpd

    import _geography as geo

    where = f"inspection > '{int(since_year)}-01-01'"
    if drop_unrated:
        where += f" AND systemrating != '{UNRATED_PAVEMENT}'"
    entry = {"resource_id": PAVEMENT_RESOURCE, "index": "pavement"}
    df = fetch(
        entry,
        select="the_geom,systemrating,inspection,boroughname,onstreetna",
        where=where,
    )
    if df.empty:
        return df

    df["proxy_value"] = pd.to_numeric(df["systemrating"], errors="coerce")
    df = df.dropna(subset=["proxy_value", "the_geom"])
    geom = df["the_geom"].apply(_to_shape)
    g = gpd.GeoDataFrame(
        df.drop(columns=["the_geom"]), geometry=list(geom), crs="EPSG:4326"
    )
    return g[g.geometry.notna()].to_crs(geo.WORKING_CRS)


def attach_nearest_segment(points, segments, max_distance_ft: float = 100.0):
    """Give each image the rating of the nearest rated street segment.

    This lifts road quality to a UNIT comparison: 1 image, 1 model score, 1 DOT
    rating. Compare it with the parks case, where the same move overturned the
    area result.

    Args:
        max_distance_ft: An image farther than this from any rated segment gets
            no rating and drops. A camera sits in the roadway, so a small cap is
            right. A large cap would attach a rating from another street.
    """
    import geopandas as gpd

    import _geography as geo

    pts = points.to_crs(geo.WORKING_CRS)[["sample_id", "geometry"]].copy()
    seg = segments[["proxy_value", "geometry"]].copy()
    hit = gpd.sjoin_nearest(
        pts, seg, how="inner", max_distance=max_distance_ft, distance_col="dist_ft"
    )
    # A tie can attach 2 segments to 1 image. Keep the closest.
    hit = hit.sort_values("dist_ft").drop_duplicates("sample_id")
    return hit[["sample_id", "proxy_value", "dist_ft"]].reset_index(drop=True)


def pavement_by_layer(layer: str, segments=None, min_segments: int = 3) -> pd.DataFrame:
    """Aggregate the DOT pavement rating into 1 geography layer.

    A rating is a score for each segment, so a mean is valid here. Compare it
    with the income proxy, where a median is not.

    Each segment goes in by its own midpoint, so a long avenue lands in 1
    polygon rather than several. That matches how the image points land.
    """
    import geopandas as gpd

    import _geography as geo

    seg = pavement_segments() if segments is None else segments
    if seg.empty:
        return pd.DataFrame()

    pts = seg[["proxy_value", "geometry"]].copy()
    pts["geometry"] = seg.geometry.interpolate(0.5, normalized=True)

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    hit = gpd.sjoin(pts, poly[[key, "geometry"]], how="inner", predicate="within")
    out = hit.groupby(key).agg(
        proxy_mean=("proxy_value", "mean"),
        n_units_proxy=("proxy_value", "size"),
    ).reset_index()
    out = out[out.n_units_proxy >= min_segments]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


def pothole_by_layer(layer: str, since_year: int = 2024) -> pd.DataFrame:
    """Aggregate closed pothole work orders into 1 geography layer.

    A pothole repair marks a road that failed, so this carries a sign of -1 and
    leaves the module oriented as "higher is better".

    Warning: this counts REPAIRS, not defects. A borough that fixes potholes
    quickly looks worse here than one that ignores them. Read it beside the
    pavement rating, never alone.
    """
    import geopandas as gpd

    import _geography as geo

    entry = {"resource_id": POTHOLE_RESOURCE, "index": "pothole"}
    df = fetch(
        entry,
        select="the_geom,rptdate,boro",
        where=f"rptdate > '{int(since_year)}-01-01'",
    )
    if df.empty:
        return df

    df = df.dropna(subset=["the_geom"])
    geom = df["the_geom"].apply(_to_shape)
    g = gpd.GeoDataFrame(
        df.drop(columns=["the_geom"]), geometry=list(geom), crs="EPSG:4326"
    )
    g = g[g.geometry.notna()].to_crs(geo.WORKING_CRS)
    g["geometry"] = g.representative_point()

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    poly = poly[[key, "geometry"]].copy()
    poly["area_km2"] = poly.area / SQFT_PER_KM2

    hit = gpd.sjoin(g[["geometry"]], poly[[key, "geometry"]], how="inner", predicate="within")
    agg = hit.groupby(key).size().rename("n_potholes").reset_index()
    agg = agg.merge(poly[[key, "area_km2"]], on=key, how="left")
    agg["proxy_mean"] = -(agg["n_potholes"] / agg["area_km2"])
    agg["n_units_proxy"] = agg["n_potholes"]
    out = agg[[key, "proxy_mean", "n_units_proxy"]].copy()
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


# --------------------------------------------------------------- crime proxy

CRIME_CSV = REPO_ROOT / "data/external/nypd_complaints_ytd/nypd_complaint_ytd_2026_pulled20260629.csv"

# The same weights that `scripts/pairwise_socioeconomic_regression.py` uses, so
# the 2 analyses build the same covariate.
SEVERITY_WEIGHTS = {"FELONY": 3.0, "MISDEMEANOR": 2.0, "VIOLATION": 1.0}
SQFT_PER_KM2 = 10_763_910.4

# Warning: more crime means less safety, and the model scores safety upward.
# Each metric carries a sign, and `crime_by_layer` applies it, so every value
# leaves this module oriented as "higher is better". A positive correlation then
# always means the model and the record agree.
CRIME_METRICS: Dict[str, Dict[str, Any]] = {
    "crime_density": {
        "sign": -1,
        "label": "Severity-weighted complaints per km^2 (negated)",
        "note": "Felony 3, misdemeanor 2, violation 1. Divided by land area.",
    },
    "felony_density": {
        "sign": -1,
        "label": "Felony complaints per km^2 (negated)",
        "note": "Felony counts only, divided by land area.",
    },
    "crime_per_capita": {
        "sign": -1,
        "label": "Severity-weighted complaints for each 1,000 residents (negated)",
        "note": "Undefined where a polygon holds under 50 residents, so those rows drop.",
    },
}


def crime_by_layer(layer: str, metric: str = "crime_density",
                   min_population: int = 50) -> pd.DataFrame:
    """Aggregate the NYPD complaint record onto 1 geography layer.

    Crime is point data, so this needs no approximation. A count and an area
    both aggregate exactly, unlike a median. Thus every layer gets a true value,
    and the income caveat does not apply here.

    Args:
        min_population: A per-capita rate needs residents. A polygon under this
            count returns NaN and drops. Parks and cemeteries need this guard.
    """
    import geopandas as gpd

    import _geography as geo

    if metric not in CRIME_METRICS:
        raise KeyError(f"unknown metric {metric!r}; use one of {list(CRIME_METRICS)}")
    spec = CRIME_METRICS[metric]

    cr = pd.read_csv(
        CRIME_CSV, usecols=["law_cat_cd", "latitude", "longitude"], low_memory=False
    ).dropna(subset=["latitude", "longitude"])
    cr["w"] = cr["law_cat_cd"].map(SEVERITY_WEIGHTS).fillna(0.0)
    pts = gpd.GeoDataFrame(
        cr, geometry=gpd.points_from_xy(cr["longitude"], cr["latitude"]), crs="EPSG:4326"
    ).to_crs(geo.WORKING_CRS)

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    poly = poly[[key, "geometry"]].copy()
    poly["area_km2"] = poly.area / SQFT_PER_KM2

    hit = gpd.sjoin(pts, poly[[key, "geometry"]], how="inner", predicate="within")
    agg = hit.groupby(key).agg(
        crime_count=("w", "size"),
        crime_weighted=("w", "sum"),
        felony_count=("law_cat_cd", lambda s: (s == "FELONY").sum()),
    ).reset_index()
    agg = agg.merge(poly[[key, "area_km2"]], on=key, how="left")

    if metric == "crime_per_capita":
        pop = _population_by_layer(layer)
        agg = agg.merge(pop, on=key, how="left")
        ok = agg["population"].where(agg["population"] >= min_population)
        raw = agg["crime_weighted"] / ok * 1000.0
    elif metric == "felony_density":
        raw = agg["felony_count"] / agg["area_km2"]
    else:
        raw = agg["crime_weighted"] / agg["area_km2"]

    agg["proxy_mean"] = raw * spec["sign"]
    agg["n_units_proxy"] = agg["crime_count"]
    out = agg.dropna(subset=["proxy_mean"])[[key, "proxy_mean", "n_units_proxy"]].copy()
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


def _population_by_layer(layer: str) -> pd.DataFrame:
    """Sum the tract population into 1 geography layer.

    A sum aggregates exactly, so this needs no approximation. Compare it with
    `income_by_layer`, where a median does not.
    """
    import geopandas as gpd

    import _geography as geo

    key = geo.LAYERS[layer]["key"]
    tracts = gpd.read_file(geo.LAYERS["census_tract"]["path"]).to_crs(geo.WORKING_CRS)
    tracts["geoid"] = tracts["geoid"].astype(str)
    tracts = tracts.merge(load_tract_income()[["geoid", "population"]], on="geoid", how="left")

    if layer == "census_tract":
        return tracts[[key, "population"]].copy()

    pts = tracts.copy()
    pts["geometry"] = tracts.representative_point()
    poly = geo.load_layer(layer)
    hit = gpd.sjoin(
        pts[["population", "geometry"]], poly[[key, "geometry"]],
        how="inner", predicate="within",
    )
    return hit.groupby(key)["population"].sum().reset_index()


def aggregate_proxy(proxy_points, layer: str, min_units: int = 3) -> pd.DataFrame:
    """Aggregate the located proxy into 1 geography layer.

    Args:
        proxy_points: A point layer with `proxy_key` and `proxy_value`, from
            `join_proxy_points`.

    This uses the same layers and the same minimum as the model scores, so the
    2 tables line up for a correlation.
    """
    import geopandas as gpd

    import _geography as geo

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]

    if proxy_points.empty:
        return pd.DataFrame()

    hit = gpd.sjoin(
        proxy_points[["proxy_key", "proxy_value", "geometry"]],
        poly[[key, "geometry"]],
        how="inner", predicate="within",
    )
    out = hit.groupby(key).agg(
        n_units_proxy=("proxy_key", "nunique"),
        proxy_mean=("proxy_value", "mean"),
    ).reset_index()
    out = out[out.n_units_proxy >= min_units]
    out.insert(0, "layer", layer)
    return out.reset_index(drop=True)


def correlate_units(unit_scores: pd.DataFrame, unit_key: pd.DataFrame,
                    proxy: pd.DataFrame) -> Dict[str, Any]:
    """Compare the model against a proxy UNIT BY UNIT, not by area.

    Prefer this over `correlate` whenever a proxy rates the same object the
    model rates. An area correlation can appear where no unit correlation
    exists, because aggregation into polygons creates its own agreement. The
    parks case shows exactly that: the area value reaches +0.23 while the unit
    value sits at -0.01.

    Args:
        unit_scores: From `_provenance.score_units`, with `unit_uid`.
        unit_key: Maps `unit_uid` to `proxy_key`, from `attach_park_property_id`.
        proxy: Holds `proxy_key` and `proxy_value`.
    """
    j = unit_scores.merge(unit_key, on="unit_uid", how="inner").merge(
        proxy, on="proxy_key", how="inner"
    )
    if len(j) < 3:
        return {"scope": "unit", "n": len(j), "pearson_r": None, "spearman_rho": None}
    return {
        "scope": "unit",
        "n": len(j),
        "pearson_r": round(j["mean_score"].corr(j["proxy_value"]), 4),
        "spearman_rho": round(
            j["mean_score"].corr(j["proxy_value"], method="spearman"), 4
        ),
    }


def correlate(model_agg: pd.DataFrame, proxy_agg: pd.DataFrame,
              layer: str) -> Dict[str, Any]:
    """Compare the model score against the proxy in 1 layer.

    Returns Pearson r and Spearman rho with the count of shared polygons.
    Spearman is the safer read: the model score is an ordinal mean, so a
    monotone agreement matters more than a linear one.
    """
    import _geography as geo

    key = geo.LAYERS[layer]["key"]
    m = model_agg[model_agg.layer == layer]
    p = proxy_agg[proxy_agg.layer == layer]
    if m.empty or p.empty:
        return {"layer": layer, "n": 0, "pearson_r": None, "spearman_rho": None}

    j = m.merge(p, on=key, how="inner", suffixes=("_model", "_proxy"))
    if len(j) < 3:
        return {"layer": layer, "n": len(j), "pearson_r": None, "spearman_rho": None}

    return {
        "layer": layer,
        "n": len(j),
        "pearson_r": round(j["mean_score"].corr(j["proxy_mean"]), 4),
        "spearman_rho": round(j["mean_score"].corr(j["proxy_mean"], method="spearman"), 4),
    }


# ------------------------------------------------------------ vintage proxies
#
# The vintage of a unit is the year it came into being. It is NOT a measure of
# quality, and this module applies no sign to it. A positive correlation says
# that the model calls a NEWER unit better, which is a finding to state and not
# an agreement to score.
#
# Each case needs the vintage field that describes ITS unit. A building year
# fits a library and a school, because the unit IS a building. It does not fit
# a park: the BIN on a park lot belongs to a comfort station or a recreation
# building, and 33% of the park units carry one at all.
#
# | Case | Field | Source | Key |
# |------|-------|--------|-----|
# | Libraries, schools | `construction_year` | Building footprints, on disk | `bin` |
# | Parks | `acquisitiondate` | DPR Parks Properties | `gispropnum` |
# | Plazas (POPS) | `year_completed` | POPS | `bbl` |
#
# Warning: the 92 DOT pedestrian plazas have no vintage field in any city
# source. They drop out, and `pops_vintage` reports how many.

BUILDINGS_PATH = REPO_ROOT / "data" / "geo" / "nyc_buildings.parquet"

# The bounds of a year that this module trusts. The footprints table writes 0
# for an unknown year, and a 0 that reaches a correlation destroys it.
YEAR_MIN = 1600
YEAR_MAX = 2026

# A borough placeholder BIN. It names a borough and not a building, thus it
# joins to nothing and it must never reach the footprints table.
PLACEHOLDER_BINS = {f"{b}000000" for b in range(1, 6)}


def clean_bin(values) -> pd.Series:
    """Return the usable BINs of a column, and NaN for the rest.

    A usable BIN is 7 digits, it starts with the borough code 1 to 5, and it is
    not a borough placeholder such as 1000000.
    """
    s = pd.Series(values).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    ok = s.str.match(r"^[1-5]\d{6}$") & ~s.isin(PLACEHOLDER_BINS)
    return s.where(ok)


def building_year_by_bin(use_cache: bool = True) -> pd.DataFrame:
    """Read the construction year of each building, keyed by BIN.

    The source is `data/geo/nyc_buildings.parquet`, the local copy of the city
    BUILDING dataset (`5zhs-2jue`). It holds 1,082,872 footprints and 99.1% of
    them carry a usable year.

    A BIN can carry more than 1 footprint row, which happens 13 times. This
    keeps the LARGEST footprint of a BIN, because the largest one is the
    building that the address names.

    Returns a frame of `bin` and `build_year`. The result caches to
    `.proxy_cache/`, thus only the first call reads the 211 MB parquet.
    """
    cache = CACHE_DIR / "building_year_by_bin.parquet"
    CACHE_DIR.mkdir(exist_ok=True)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    if not BUILDINGS_PATH.exists():
        raise FileNotFoundError(
            f"no building footprints at {BUILDINGS_PATH}.\n"
            f"Get them again: python data/geo/download_nyc_buildings.py")
    b = pd.read_parquet(BUILDINGS_PATH,
                        columns=["bin", "construction_year", "shape_area"])
    b["bin"] = b["bin"].astype(str).str.strip()
    b["build_year"] = pd.to_numeric(b["construction_year"], errors="coerce")
    b["area"] = pd.to_numeric(b["shape_area"], errors="coerce")
    b = b[b["build_year"].between(YEAR_MIN, YEAR_MAX)]
    b = b.sort_values("area", ascending=False).drop_duplicates("bin")
    out = b[["bin", "build_year"]].reset_index(drop=True)
    out.to_parquet(cache, index=False)
    return out


def building_vintage(facilities: pd.DataFrame) -> pd.DataFrame:
    """Give each facility the construction year of its own building.

    FacDB carries the BIN, thus this is a key join and not a spatial one. The
    unit and the building are the same thing here, so the year needs no
    approximation.

    Returns `unit_uid`, `vintage_year`, `latitude`, and `longitude`, with 1 row
    for each unit that reached a year.
    """
    f = facilities.copy()
    f["bin_clean"] = clean_bin(f["bin"])
    years = building_year_by_bin()
    j = f.merge(years, left_on="bin_clean", right_on="bin", how="left",
                suffixes=("", "_bld"))
    j = j.dropna(subset=["build_year"])
    out = j[["uid", "build_year", "latitude", "longitude"]].rename(
        columns={"uid": "unit_uid", "build_year": "vintage_year"})
    return out.reset_index(drop=True)


def park_acquisition_year(use_cache: bool = True) -> pd.DataFrame:
    """Read the year the city acquired each DPR property.

    A park has no construction year, because a park is land and not a building.
    The acquisition date is the vintage that describes it: the year the land
    became a park.

    Warning: acquisition is not construction. A park that the city acquired in
    1936 can hold a playground of 2015. Read this row as the age of the SITE.

    Returns `proxy_key` (the DPR `gispropnum`) and `vintage_year`.
    """
    entry = {"resource_id": "enfh-gkve", "index": "parks-acquisition"}
    df = fetch(entry, use_cache=use_cache,
               select="gispropnum,acquisitiondate")
    if df.empty:
        return pd.DataFrame(columns=["proxy_key", "vintage_year"])
    year = pd.to_datetime(df["acquisitiondate"], errors="coerce",
                          format="mixed").dt.year
    out = pd.DataFrame({"proxy_key": df["gispropnum"].astype(str).str.strip(),
                        "vintage_year": year})
    out = out[out["vintage_year"].between(YEAR_MIN, YEAR_MAX)]
    # A property can appear more than once. Keep the earliest acquisition,
    # which is the year the site became parkland.
    return (out.sort_values("vintage_year")
               .drop_duplicates("proxy_key").reset_index(drop=True))


def pops_year_completed(use_cache: bool = True) -> pd.DataFrame:
    """Read the completion year of each privately owned public space.

    POPS holds the year of the SPACE, not of the building around it, thus it
    describes the plaza directly. The dataset holds 392 rows, which matches the
    392 `dcp_pops` units of FacDB exactly.

    Returns `bbl` and `vintage_year`.
    """
    entry = {"resource_id": "rvih-nhyn", "index": "pops-year"}
    df = fetch(entry, use_cache=use_cache, select="bbl,year_completed")
    if df.empty:
        return pd.DataFrame(columns=["bbl", "vintage_year"])
    out = pd.DataFrame({
        "bbl": df["bbl"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip(),
        "vintage_year": pd.to_numeric(df["year_completed"], errors="coerce"),
    })
    out = out[out["vintage_year"].between(YEAR_MIN, YEAR_MAX)]
    return out.drop_duplicates("bbl").reset_index(drop=True)


def park_vintage(facilities: pd.DataFrame, units_gdf) -> pd.DataFrame:
    """Give each park unit the acquisition year of its DPR property.

    FacDB carries no property id, thus this uses the same spatial join that the
    inspection proxy uses: a unit inside a park polygon takes that park, and a
    unit outside every polygon takes the nearest park within 500 ft.
    """
    ids = attach_park_property_id(units_gdf)
    years = park_acquisition_year()
    j = ids.merge(years, on="proxy_key", how="inner")
    pos = facilities[["uid", "latitude", "longitude"]].rename(
        columns={"uid": "unit_uid"})
    out = j.merge(pos, on="unit_uid", how="left")
    return out[["unit_uid", "vintage_year", "latitude", "longitude"]].reset_index(
        drop=True)


def pops_vintage(facilities: pd.DataFrame) -> pd.DataFrame:
    """Give each POPS plaza unit the year the space was completed.

    Warning: the DOT pedestrian plazas carry no vintage in any city source, so
    they drop out here. `dcp_pops` is the only plaza source with a year.
    """
    f = facilities.copy()
    f["bbl_clean"] = (f["bbl"].astype(str)
                      .str.replace(r"\.0$", "", regex=True).str.strip())
    years = pops_year_completed()
    j = f.merge(years, left_on="bbl_clean", right_on="bbl", how="inner",
                suffixes=("", "_pops"))
    out = j[["uid", "vintage_year", "latitude", "longitude"]].rename(
        columns={"uid": "unit_uid"})
    return out.drop_duplicates("unit_uid").reset_index(drop=True)


def vintage_by_layer(vintage: pd.DataFrame, layer: str,
                     min_units: int = 3) -> pd.DataFrame:
    """Aggregate the unit vintages into 1 geography layer.

    The value of a polygon is the MEAN year of its units, and `n_units_proxy`
    counts them. This uses the same layers and the same minimum as the model
    scores, thus the 2 tables line up for a correlation.
    """
    if vintage.empty:
        return pd.DataFrame()
    pts = points_from_latlon(vintage.rename(
        columns={"unit_uid": "proxy_key", "vintage_year": "proxy_value"}))
    return aggregate_proxy(pts, layer, min_units=min_units)


def vintage_coverage(facilities: pd.DataFrame,
                     vintage: pd.DataFrame) -> pd.DataFrame:
    """Report how much of a unit set reached a year, by facility type.

    Read this before a correlation. A case that covers a third of its units
    describes that third, and the table must say so.
    """
    f = facilities[["uid", "factype"]].rename(columns={"uid": "unit_uid"})
    f = f.assign(has_year=f["unit_uid"].isin(set(vintage["unit_uid"])))
    g = f.groupby("factype")["has_year"].agg(units="size", with_year="sum")
    g["rate"] = (g["with_year"] / g["units"]).round(3)
    return g.sort_values("units", ascending=False).reset_index()
