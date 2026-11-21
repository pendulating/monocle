"""Station-level proxies for the subway safety case.

The model answers "which station entrance is safer". Until now the case had 2
proxies, and both describe the AREA around an entrance: the household income of
the tract, and the NYPD complaint count for each km^2. Neither one measures the
station. A correlation against them supports 1 claim only — the look of an
entrance tracks the wealth or the crime of the blocks around it.

This module builds a proxy that grades the STATION itself:

    transit crime for each 100,000 rider entries

The numerator counts the NYPD complaints that happened INSIDE the subway
system, which the complaint record marks with `prem_typ_desc = 'TRANSIT - NYC
SUBWAY'`. The denominator counts the rider entries of the same station complex
over the same months. The rate answers the question the prompt asks: a rider who
enters this station meets a recorded crime at this rate.

Why the rate and not the count
------------------------------
Times Sq-42 St takes more transit crime than any other complex, and it also
takes more riders than any other complex. A raw count grades a station by how
busy it is. The rate removes the exposure, thus it separates a busy station from
a dangerous one. `subway_ridership` stays in the export as a CONTEXT row, so a
reader can see the exposure beside the rate.

The 3 keys of this case
-----------------------
| Level | Column | Count |
|-------|--------|-------|
| Entrance | `uid` in `entrances.parquet` | 2,120 |
| Station | `station_id` | 485 |
| Complex | `complex_id` | 445 |

A complex is a set of stations that share 1 fare control area, such as Times
Sq-42 St. The MTA ridership record counts a complex, and the NYPD geocode falls
on a station, thus the complex is the level that both sources reach.

**The roll-up rule: the complex where the entrance carries one, and the station
where it does not.** Every row of `entrances.parquet` carries a `complex_id`
today, thus the rule gives 445 keys. The fall-back stays in the code because the
MTA can add a station with no complex.

The windows
-----------
| Source | Resource | Window |
|--------|----------|--------|
| NYPD complaints, historic | `qgea-i56i` | 2024-01 to 2025-12 |
| NYPD complaints, year to date | `5uac-w243` | 2026-01 to 2026-06 |
| MTA hourly ridership, 2020-2024 | `wujg-7c2s` | 2024-01 to 2024-12 |
| MTA hourly ridership, from 2025 | `5wq4-mkjj` | 2025-01 to 2026-06 |
| MTA subway stations | `39hk-dx4f` | current |

Warning: do not build this proxy from the 2026Q1 file on disk. That window holds
8,903 transit complaints, thus the median complex takes 8 of them and a rate
over 8 events is noise. The 30-month window holds 76,729, and the median complex
takes 66.

Credentials
-----------
**Warning: never write a credential into this repository.** The 2 Socrata hosts
take 2 different keys, and `.env` holds both. This module reads them by name:

| Host | Key id | Key secret |
|------|--------|------------|
| `data.cityofnewyork.us` | `NYC_API_ID` | `NYC_API_SKEY` |
| `data.ny.gov` | `NY_STATE_API_ID` | `NY_STATE_API_SKEY` |

Do not print them. Do not put them in a notebook cell or an error message.

See `vlm-narratives-docs/subway-station-proxy.md`.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

import _proxies as px

__version__ = "1.0.0"

REPO_ROOT = px.REPO_ROOT
EXTERNAL_DIR = REPO_ROOT / "data" / "external"
CACHE_DIR = px.CACHE_DIR

# --------------------------------------------------------------------------
# The 2 Socrata hosts
# --------------------------------------------------------------------------
CITY_DOMAIN = "data.cityofnewyork.us"
STATE_DOMAIN = "data.ny.gov"

#: The environment variable NAMES of each host. The values never appear here.
DOMAIN_ENV: Dict[str, tuple] = {
    CITY_DOMAIN: ("NYC_API_ID", "NYC_API_SKEY"),
    STATE_DOMAIN: ("NY_STATE_API_ID", "NY_STATE_API_SKEY"),
}

SODA_PAGE = 50_000

# --------------------------------------------------------------------------
# The sources
# --------------------------------------------------------------------------
#: The MTA station list. It maps `station_id` to `complex_id` and gives each
#: station its GTFS point, thus it is both the roll-up table and the geocode.
STATIONS_RESOURCE = "39hk-dx4f"
STATIONS_DOMAIN = STATE_DOMAIN
STATIONS_PULL = "mta_subway_stations"

#: The premise value that marks a complaint inside the subway system.
TRANSIT_PREMISE = "TRANSIT - NYC SUBWAY"

#: The window of every station-level number. The end is EXCLUSIVE.
WINDOW_START = "2024-01-01"
WINDOW_END = "2026-07-01"
WINDOW_LABEL = "2024-01 to 2026-06"

#: The complaint sources. The historic table stops on 2025-12-31 and the year to
#: date table starts on 2026-01-01, thus the 2 together cover the window. Both
#: take the same filter, and `transit_complaints` drops a repeated complaint
#: number, so an overlap cannot double count.
COMPLAINT_SOURCES: Sequence[Dict[str, str]] = (
    {"resource": "qgea-i56i", "domain": CITY_DOMAIN,
     "label": "NYPD Complaint Data Historic"},
    {"resource": "5uac-w243", "domain": CITY_DOMAIN,
     "label": "NYPD Complaint Data Current (Year To Date)"},
)
COMPLAINTS_PULL = "nypd_transit_complaints"

#: The ridership sources. The MTA splits the hourly record at the end of 2024.
RIDERSHIP_SOURCES: Sequence[Dict[str, str]] = (
    {"resource": "wujg-7c2s", "domain": STATE_DOMAIN,
     "start": "2024-01-01T00:00:00", "end": "2025-01-01T00:00:00",
     "label": "MTA Subway Hourly Ridership: 2020-2024"},
    {"resource": "5wq4-mkjj", "domain": STATE_DOMAIN,
     "start": "2025-01-01T00:00:00", "end": "2026-07-01T00:00:00",
     "label": "MTA Subway Hourly Ridership: Beginning 2025"},
)
RIDERSHIP_PULL = "mta_subway_ridership"

#: How the provenance columns name each side. A reader of the parquet needs the
#: resource ids, and a rate carries 2 sources.
CRIME_SOURCE_ID = "+".join(s["resource"] for s in COMPLAINT_SOURCES)
RIDER_SOURCE_ID = "+".join(s["resource"] for s in RIDERSHIP_SOURCES)
RATE_SOURCE_ID = f"{CRIME_SOURCE_ID} / {RIDER_SOURCE_ID}"

# --------------------------------------------------------------------------
# The constants of the build
# --------------------------------------------------------------------------
#: A complaint farther than this from every station drops.
#:
#: The NYPD geocodes a transit complaint to the station, not to the platform,
#: thus the distance is small: the median is 145 ft and the 99th percentile is
#: 988 ft. This cap keeps 99.0% of the 76,729 complaints. A larger cap buys
#: 0.5% more and reaches across a river.
MAX_COMPLAINT_FT = 1000.0

#: The rate is "for each 100,000 rider entries".
RIDER_UNIT = 100_000.0

#: A complex under this many rider entries carries no readable rate. Over 30
#: months the smallest served complex takes about 1,000,000 entries, thus this
#: floor removes only a complex whose denominator broke.
MIN_RIDERS = 50_000.0

#: The floor of the polygon aggregation. The model side of this case uses 1
#: (`MIN_UNITS_PER_POLYGON` in the notebook), thus the proxy side must use 1 as
#: well. A different floor would drop areas from one side only, and the 2 tables
#: would no longer describe the same set of polygons.
MIN_STATIONS_PER_POLYGON = 1

#: The division code of the Staten Island Railway.
#:
#: Warning: the SIR is not the subway, and neither source covers it. The NYPD
#: premise `TRANSIT - NYC SUBWAY` records no SIR event, and the MTA hourly
#: subway record holds only 2 of the 21 SIR complexes. A SIR station would
#: therefore take a crime rate of exactly 0 and lead the safety ranking on an
#: absence of data. `station_proxy` drops the 21 SIR complexes, which carry 66
#: of the 2,120 entrances.
SIR_DIVISION = "SIR"

#: How much 2 station names must share before this module calls them the same
#: place. The value is a Jaccard index over the words of the 2 names, after the
#: generic words drop. See `_named_but_unreached`.
NAME_MATCH_MIN = 0.5

# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------
# Warning: more crime means less safety, and the model scores safety upward.
# Each oriented metric carries a sign, and `station_proxy` applies it, thus
# every oriented value leaves this module as "higher is better". A positive
# correlation then always means the model and the record agree.
#
# `subway_ridership` carries NO orientation. A busy station is not a better
# station. Read that row as context, never as agreement.
SUBWAY_METRICS: Dict[str, Dict[str, Any]] = {
    "transit_crime_per_rider": {
        "sign": -1,
        "oriented": True,
        "column": "n_complaints",
        "metric": "complaints_per_100k_rider_entries",
        "source": RATE_SOURCE_ID,
        "label": "Transit crime for each 100,000 rider entries (negated)",
        "note": "Every complaint with premise TRANSIT - NYC SUBWAY, over the "
                "rider entries of the same complex.",
    },
    "transit_felony_per_rider": {
        "sign": -1,
        "oriented": True,
        "column": "n_felonies",
        "metric": "felonies_per_100k_rider_entries",
        "source": RATE_SOURCE_ID,
        "label": "Transit felonies for each 100,000 rider entries (negated)",
        "note": "Felony complaints only. It is 1 of 6 transit complaints.",
    },
    "subway_ridership": {
        "sign": 1,
        "oriented": False,
        "column": None,
        "metric": "rider_entries",
        "source": RIDER_SOURCE_ID,
        "label": "Rider entries (context, no orientation)",
        "note": "The exposure of the rate. A busy station is not a safe "
                "station, thus this row carries no orientation.",
    },
}

#: The metrics that the export writes.
DEFAULT_METRICS: Sequence[str] = tuple(SUBWAY_METRICS)

#: The headline metric of the case.
HEADLINE_METRIC = "transit_crime_per_rider"

CURATION_ROOT = "curation/subway_entrances_all"
UNIT_TABLE = "entrances.parquet"


# --------------------------------------------------------------------------
# Socrata, on either host
# --------------------------------------------------------------------------
def _auth_header(domain: str) -> Dict[str, str]:
    """Build the basic-auth header of 1 host from the environment.

    Returns an empty dict when the host has no credential. Socrata then applies
    a lower rate limit, and the request still works.
    """
    px._load_dotenv_once()
    names = DOMAIN_ENV.get(domain)
    if not names:
        return {}
    kid, ksec = os.environ.get(names[0]), os.environ.get(names[1])
    if not (kid and ksec):
        return {}
    token = base64.b64encode(f"{kid}:{ksec}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def has_credentials(domain: str) -> bool:
    """True when the host holds both environment values.

    This reports presence only. It never returns or logs a value.
    """
    px._load_dotenv_once()
    names = DOMAIN_ENV.get(domain)
    return bool(names and os.environ.get(names[0]) and os.environ.get(names[1]))


def soda_fetch(resource: str, domain: str = CITY_DOMAIN,
               select: Optional[str] = None, where: Optional[str] = None,
               group: Optional[str] = None, max_rows: int = 2_000_000,
               use_cache: bool = True) -> pd.DataFrame:
    """Get the rows of 1 Socrata resource on 1 host, with paging and a cache.

    This repeats `_proxies.fetch` and adds the host. The MTA tables live on
    `data.ny.gov`, and `_proxies.fetch` reaches only the city host.

    Warning: filter on the server. The hourly ridership table holds 120,855,567
    rows, thus a whole-table fetch cannot finish. Pass `select`, `where`, and
    `group` so the server returns 1 row for each station complex.

    Warning: this raises an error when the source reaches `max_rows`. A silent
    cut gives a wrong answer that looks correct.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    tag = re.sub(r"[^a-z0-9_-]", "_",
                 f"{domain}_{resource}_{select or ''}_{where or ''}_{group or ''}"
                 .lower())
    cache = CACHE_DIR / f"{tag[:150]}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    headers = _auth_header(domain)
    frames: List[pd.DataFrame] = []
    offset = 0
    truncated = True
    while offset < max_rows:
        params: Dict[str, Any] = {"$limit": SODA_PAGE, "$offset": offset}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        if group:
            params["$group"] = group
        url = (f"https://{domain}/resource/{resource}.json?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=900))
        except Exception as exc:
            # The credential lives in the Authorization header, not in the URL,
            # thus the query text is safe to show. Never add the header here.
            detail = ""
            body = getattr(exc, "read", None)
            if callable(body):
                try:
                    detail = body().decode(errors="replace")[:300]
                except Exception:
                    detail = ""
            raise RuntimeError(
                f"SODA request failed for {domain}/{resource}: "
                f"{type(exc).__name__}\n"
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
            f"{domain}/{resource} reached the {max_rows:,}-row cap. The result "
            f"is incomplete, so do not use it. Make the `where` filter smaller, "
            f"or raise max_rows."
        )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df.to_parquet(cache, index=False)
    return df


# --------------------------------------------------------------------------
# The pull directory
# --------------------------------------------------------------------------
def pull_dir(name: str, when: Optional[str] = None) -> Path:
    """The directory of 1 pull, `data/external/<name>_pulled<YYYYMMDD>`."""
    stamp = when or date.today().strftime("%Y%m%d")
    return EXTERNAL_DIR / f"{name}_pulled{stamp}"


def newest_pull(name: str) -> Optional[Path]:
    """The newest pull directory of 1 source, or None.

    The date sits at the end of the name in `YYYYMMDD`, thus a lexical sort is
    a date sort.
    """
    hits = sorted(glob.glob(str(EXTERNAL_DIR / f"{name}_pulled*")))
    return Path(hits[-1]) if hits else None


def save_pull(df: pd.DataFrame, name: str, filename: str,
              meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write 1 pull to disk and give back the file.

    The directory also takes a `pull.json` that states the query. A reader of
    the parquet must never have to guess the window.
    """
    out = pull_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    df.to_parquet(path, index=False)
    if meta is not None:
        (out / "pull.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def _load_or_pull(name: str, filename: str, builder, refresh: bool = False
                  ) -> pd.DataFrame:
    """Read the newest pull of a source, or make one when none exists."""
    if not refresh:
        hit = newest_pull(name)
        if hit is not None and (hit / filename).exists():
            return pd.read_parquet(hit / filename)
    df, meta = builder()
    save_pull(df, name, filename, meta)
    return df


# --------------------------------------------------------------------------
# 1. The station list and the roll-up
# --------------------------------------------------------------------------
STATION_COLUMNS = ("station_id,complex_id,gtfs_stop_id,stop_name,borough,"
                   "daytime_routes,division,line,structure,ada,"
                   "gtfs_latitude,gtfs_longitude")


@lru_cache(maxsize=1)
def stations(refresh: bool = False) -> pd.DataFrame:
    """The MTA station list: 1 row for each station, with its complex.

    The table gives every station its GTFS point and its ADA status, and it maps
    `station_id` to `complex_id`. The subway curation reads the same 2 ids, thus
    this table joins to `entrances.parquet` by key and never by name.
    """
    def build():
        df = soda_fetch(STATIONS_RESOURCE, STATIONS_DOMAIN,
                        select=STATION_COLUMNS, use_cache=False)
        meta = {"resource": STATIONS_RESOURCE, "domain": STATIONS_DOMAIN,
                "select": STATION_COLUMNS, "rows": int(len(df)),
                "pulled": date.today().isoformat()}
        return df, meta

    df = _load_or_pull(STATIONS_PULL, "stations.parquet", build, refresh)
    out = df.copy()
    for c in ("station_id", "complex_id", "gtfs_stop_id"):
        out[c] = out[c].astype(str).str.strip()
    out["latitude"] = pd.to_numeric(out["gtfs_latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["gtfs_longitude"], errors="coerce")
    out["ada"] = pd.to_numeric(out.get("ada"), errors="coerce")
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def load_entrances() -> pd.DataFrame:
    """Read the subway curation table, 1 row for each entrance."""
    import _geography as geo

    cols = ["uid", "facname", "station_id", "complex_id", "gtfs_stop_id",
            "stop_name", "borough", "division", "entrance_type",
            "latitude", "longitude"]
    df = geo.load_facilities(CURATION_ROOT, filename=UNIT_TABLE)
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    for c in ("station_id", "complex_id", "gtfs_stop_id"):
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    return out


def station_key_of(complex_id, station_id) -> pd.Series:
    """The roll-up key: the complex where one exists, else the station.

    The 2 id spaces overlap — a `complex_id` of `1` and a `station_id` of `1`
    are different objects — thus the key carries its level as a prefix.
    """
    cx = pd.Series(complex_id).astype(str).str.strip()
    st = pd.Series(station_id).astype(str).str.strip()
    bad = {"", "nan", "None", "<NA>"}
    use_cx = ~cx.isin(bad)
    return pd.Series(
        ["C" + c if ok else "S" + s for c, s, ok in zip(cx, st, use_cx)],
        index=cx.index, dtype=object)


@lru_cache(maxsize=1)
def entrance_station_map() -> pd.DataFrame:
    """Map each entrance to its station key.

    Returns `unit_uid`, `station_key`, `complex_id`, `station_id`, `stop_name`.
    `unit_uid` is the `uid` of `entrances.parquet`, which is the same id that
    the pairs manifest carries, thus the model scores join by key.
    """
    ent = load_entrances()
    out = pd.DataFrame({
        "unit_uid": ent["uid"].astype(str),
        "station_key": station_key_of(ent["complex_id"], ent["station_id"]),
        "complex_id": ent["complex_id"],
        "station_id": ent["station_id"],
        "stop_name": ent.get("stop_name"),
        "division": ent.get("division"),
        "latitude": pd.to_numeric(ent["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(ent["longitude"], errors="coerce"),
    })
    return out


@lru_cache(maxsize=1)
def station_table() -> pd.DataFrame:
    """One row for each station key that the case holds, with its position.

    The position is the MEAN of the entrance positions, not the GTFS platform
    point. The model looked at the entrances, thus the proxy must land in the
    polygon that holds them.
    """
    m = entrance_station_map().dropna(subset=["latitude", "longitude"])
    out = m.groupby("station_key").agg(
        n_entrances=("unit_uid", "nunique"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        complex_id=("complex_id", "first"),
        stop_name=("stop_name", "first"),
        division=("division", "first"),
    ).reset_index()
    names = (m.groupby("station_key")["stop_name"]
             .agg(lambda s: " / ".join(sorted(set(s.dropna().astype(str)))))
             .rename("station_name").reset_index())
    out = out.merge(names, on="station_key", how="left")
    out["is_sir"] = out["division"].astype(str).str.upper().eq(SIR_DIVISION)
    return out


def station_points():
    """The station keys as a point layer in the working CRS."""
    return px.points_from_latlon(station_table())


#: How the entrance scores of 1 station become the score of that station.
#:
#: The choice is the COMPARISON-WEIGHTED mean, not the plain mean. A station
#: holds entrances whose comparison counts differ by a factor of more than 10,
#: and the weighted mean over the entrance means equals the plain mean over the
#: comparisons of the station. Thus a station takes the score it would take if
#: the model had rated the station directly.
ROLLUP_RULE = "comparison_weighted"


def station_scores(entrance_scores: pd.DataFrame) -> pd.DataFrame:
    """Roll the entrance scores of 1 run up to the stations.

    Args:
        entrance_scores: The frame that `_provenance.score_units` returns, with
            `unit_uid`, `mean_score`, `n_comparisons`, and `abstention_rate`.

    Warning: the comparison floor applies to the ENTRANCE, before this step. An
    entrance under the floor never reaches its station, thus a station keeps
    only the entrances that carry evidence.

    The returned frame carries `unit_uid`, and that value is now a STATION key.
    It also carries `mean_score_plain`, so a reader can measure what the weight
    costs.
    """
    key = entrance_station_map()[["unit_uid", "station_key"]]
    j = entrance_scores.merge(key, on="unit_uid", how="inner")
    if j.empty:
        return j.iloc[0:0]

    w = j["n_comparisons"].astype(float).clip(lower=0.0)
    j = j.assign(_w=w, _wv=j["mean_score"].astype(float) * w)
    out = j.groupby("station_key").agg(
        mean_score_plain=("mean_score", "mean"),
        sd_score=("mean_score", "std"),
        n_comparisons=("n_comparisons", "sum"),
        n_entrances=("unit_uid", "nunique"),
        abstention_rate=("abstention_rate", "mean"),
        _w=("_w", "sum"),
        _wv=("_wv", "sum"),
    ).reset_index()
    # A station where every entrance holds 0 comparisons cannot take a weighted
    # mean, thus it falls back to the plain one.
    out["mean_score"] = (out["_wv"] / out["_w"].where(out["_w"] > 0)) \
        .fillna(out["mean_score_plain"])
    out = out.rename(columns={"station_key": "unit_uid"})
    return out[["unit_uid", "mean_score", "mean_score_plain", "sd_score",
                "n_comparisons", "n_entrances", "abstention_rate"]]


# --------------------------------------------------------------------------
# 2. The transit crime of each station
# --------------------------------------------------------------------------
COMPLAINT_COLUMNS = ("cmplnt_num,cmplnt_fr_dt,law_cat_cd,ofns_desc,"
                     "prem_typ_desc,station_name,transit_district,"
                     "latitude,longitude,boro_nm")


@lru_cache(maxsize=1)
def transit_complaints(refresh: bool = False) -> pd.DataFrame:
    """Every NYPD complaint inside the subway system, over the window.

    Warning: filter on the server. The complaint tables hold millions of rows,
    and the premise filter cuts them to 76,729 before they cross the network.
    """
    where = (f"prem_typ_desc='{TRANSIT_PREMISE}' "
             f"AND cmplnt_fr_dt >= '{WINDOW_START}' "
             f"AND cmplnt_fr_dt < '{WINDOW_END}'")

    def build():
        parts, meta_parts = [], []
        for src in COMPLAINT_SOURCES:
            df = soda_fetch(src["resource"], src["domain"],
                            select=COMPLAINT_COLUMNS, where=where,
                            use_cache=False)
            meta_parts.append({"resource": src["resource"],
                               "label": src["label"], "rows": int(len(df))})
            parts.append(df)
        both = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        n_raw = len(both)
        both = both.drop_duplicates("cmplnt_num")
        meta = {"sources": meta_parts, "where": where,
                "window": WINDOW_LABEL, "rows_raw": n_raw,
                "rows": int(len(both)), "pulled": date.today().isoformat()}
        return both, meta

    df = _load_or_pull(COMPLAINTS_PULL, "transit_complaints.parquet",
                       build, refresh)
    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["date"] = pd.to_datetime(out["cmplnt_fr_dt"], errors="coerce")
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def assign_complaints(max_distance_ft: float = MAX_COMPLAINT_FT
                      ) -> pd.DataFrame:
    """Give each transit complaint the station key of its nearest station.

    The rule is a nearest join in EPSG:2263 against the GTFS point of every
    MTA station, and the station then carries the complaint to its complex.

    Why the station point and not the complex centre: a complex can stretch
    1,000 ft, thus its centre sits far from its own platforms. The nearest
    station is always closer, and it names the same complex.

    Warning: the NYPD geocode is not exact. It puts the complaint at the
    station, and 213 of the 1,338 distinct points carry more than 1 station
    name, thus a busy corridor can move a complaint to the neighbor complex.
    `assignment_report` measures the size of that effect.

    Returns `cmplnt_num`, `station_key`, `dist_ft`, `law_cat_cd`, and `weight`.
    """
    import geopandas as gpd

    import _geography as geo

    st = stations()
    st = st.assign(station_key=station_key_of(st["complex_id"],
                                              st["station_id"]))
    gst = gpd.GeoDataFrame(
        st[["station_key", "station_id", "stop_name"]],
        geometry=gpd.points_from_xy(st["longitude"], st["latitude"]),
        crs="EPSG:4326").to_crs(geo.WORKING_CRS)

    cr = transit_complaints()
    gcr = gpd.GeoDataFrame(
        cr[["cmplnt_num", "law_cat_cd", "station_name"]],
        geometry=gpd.points_from_xy(cr["longitude"], cr["latitude"]),
        crs="EPSG:4326").to_crs(geo.WORKING_CRS)

    hit = gpd.sjoin_nearest(gcr, gst, how="inner",
                            max_distance=max_distance_ft,
                            distance_col="dist_ft")
    # A tie can attach 2 stations to 1 complaint. Keep the closest.
    hit = hit.sort_values("dist_ft").drop_duplicates("cmplnt_num")
    out = pd.DataFrame({
        "cmplnt_num": hit["cmplnt_num"].values,
        "station_key": hit["station_key"].values,
        "dist_ft": hit["dist_ft"].values,
        "law_cat_cd": hit["law_cat_cd"].values,
        "nypd_station_name": hit["station_name"].values,
        "mta_stop_name": hit["stop_name"].values,
    })
    out["weight"] = out["law_cat_cd"].map(px.SEVERITY_WEIGHTS).fillna(0.0)
    return out.reset_index(drop=True)


def station_crime(max_distance_ft: float = MAX_COMPLAINT_FT) -> pd.DataFrame:
    """Count the transit complaints of each station key.

    Returns `station_key`, `n_complaints`, `n_felonies`, and `crime_weighted`.
    The weighted column uses the project weights — felony 3, misdemeanor 2,
    violation 1 — which `_proxies.SEVERITY_WEIGHTS` holds.
    """
    a = assign_complaints(max_distance_ft)
    if a.empty:
        return pd.DataFrame(columns=["station_key", "n_complaints",
                                     "n_felonies", "crime_weighted"])
    out = a.groupby("station_key").agg(
        n_complaints=("cmplnt_num", "size"),
        n_felonies=("law_cat_cd", lambda s: int((s == "FELONY").sum())),
        crime_weighted=("weight", "sum"),
    ).reset_index()
    return out


# --------------------------------------------------------------------------
# 3. The exposure
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def complex_ridership(refresh: bool = False) -> pd.DataFrame:
    """Sum the rider entries of each station complex over the window.

    Warning: the hourly table holds 120,855,567 rows for 2020-2024 alone. This
    groups ON THE SERVER, thus the network carries 428 rows and not 120,000,000.

    Returns `complex_id` and `riders`.
    """
    def build():
        parts, meta_parts = [], []
        for src in RIDERSHIP_SOURCES:
            where = (f"transit_timestamp >= '{src['start']}' "
                     f"AND transit_timestamp < '{src['end']}'")
            df = soda_fetch(
                src["resource"], src["domain"],
                select="station_complex_id, sum(ridership) as riders, "
                       "count(*) as n_rows",
                where=where, group="station_complex_id", use_cache=False)
            df["resource"] = src["resource"]
            df["window_start"] = src["start"]
            df["window_end"] = src["end"]
            meta_parts.append({"resource": src["resource"],
                               "label": src["label"], "where": where,
                               "complexes": int(len(df))})
            parts.append(df)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        meta = {"sources": meta_parts, "window": WINDOW_LABEL,
                "rows": int(len(raw)), "pulled": date.today().isoformat()}
        return raw, meta

    raw = _load_or_pull(RIDERSHIP_PULL, "ridership_by_complex.parquet",
                        build, refresh)
    raw = raw.copy()
    raw["riders"] = pd.to_numeric(raw["riders"], errors="coerce").fillna(0.0)
    raw["complex_id"] = raw["station_complex_id"].astype(str).str.strip()
    out = raw.groupby("complex_id", as_index=False)["riders"].sum()
    return out


def station_ridership() -> pd.DataFrame:
    """Put the complex ridership onto the station keys of this case.

    Warning: the 19 Staten Island Railway complexes take no row. The MTA
    hourly subway record covers the subway only, thus SIR keeps its entrances
    and loses its exposure, and every rate metric drops it.

    Returns `station_key` and `riders`.
    """
    r = complex_ridership()
    st = station_table()[["station_key", "complex_id"]].copy()
    j = st.merge(r, on="complex_id", how="left")
    return j[["station_key", "riders"]]


# --------------------------------------------------------------------------
# 4. The station proxy
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _named_but_unreached(max_distance_ft: float = MAX_COMPLAINT_FT
                         ) -> frozenset:
    """The station keys that the record names but the spatial join misses.

    A station with 0 complaints raises 1 question: did the record measure a
    quiet station, or did the geocode of that station point somewhere else?
    The NYPD `station_name` answers it. Roosevelt Island takes 16 complaints in
    the record, but the record puts them 2,900 ft away in Astoria, thus the
    spatial join gives the station a 0 and the safety ranking then puts a
    geocode error at the top.

    A station in this set carries no measurement. `station_proxy` drops it.
    """
    a = assign_complaints(max_distance_ft)
    crime = station_crime(max_distance_ft)
    tab = station_table()
    quiet = tab[~tab["station_key"].isin(set(crime["station_key"]))]
    if quiet.empty:
        return frozenset()

    named = {n for n in transit_complaints()["station_name"].dropna().unique()}
    named_tokens = [(n, _tokens(n)) for n in named]
    hit = set()
    for _, row in quiet.iterrows():
        want = _tokens(row["station_name"])
        if not want:
            continue
        for _, other in named_tokens:
            if not other:
                continue
            score = len(want & other) / len(want | other)
            if score >= NAME_MATCH_MIN:
                hit.add(row["station_key"])
                break
    return frozenset(hit)


def station_proxy(metric: str = HEADLINE_METRIC,
                  max_distance_ft: float = MAX_COMPLAINT_FT,
                  min_riders: float = MIN_RIDERS,
                  drop_sir: bool = True,
                  drop_unreached: bool = True) -> pd.DataFrame:
    """Build 1 station-level proxy, 1 row for each station key.

    Every oriented value leaves this function as "higher is better", thus a
    positive correlation always means agreement. Do not flip a sign again.

    Args:
        drop_sir: Drop the Staten Island Railway. Neither source covers it.
        drop_unreached: Drop a station that the record names but the spatial
            join misses. Its 0 is a geocode error, not a quiet station.

    Returns `proxy_key` (the station key), `proxy_value`, and the parts that
    built it: `n_complaints`, `n_felonies`, `crime_weighted`, `riders`,
    `n_entrances`, `latitude`, and `longitude`.
    """
    if metric not in SUBWAY_METRICS:
        raise KeyError(f"unknown metric {metric!r}; "
                       f"use one of {list(SUBWAY_METRICS)}")
    spec = SUBWAY_METRICS[metric]

    base = station_table()
    if drop_sir:
        base = base[~base["is_sir"]]
    if drop_unreached:
        base = base[~base["station_key"].isin(
            _named_but_unreached(max_distance_ft))]
    crime = station_crime(max_distance_ft)
    riders = station_ridership()
    df = base.merge(crime, on="station_key", how="left") \
             .merge(riders, on="station_key", how="left")
    for c in ("n_complaints", "n_felonies", "crime_weighted"):
        # A station that the guards above keep and that took no complaint took
        # no complaint. That is a zero, not a missing value.
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if metric == "subway_ridership":
        raw = df["riders"]
    else:
        ok = df["riders"].where(df["riders"] >= min_riders)
        raw = df[spec["column"]] / ok * RIDER_UNIT

    df["proxy_value"] = raw * spec["sign"]
    df["proxy_key"] = df["station_key"]
    keep = ["proxy_key", "proxy_value", "station_key", "station_name",
            "n_entrances", "n_complaints", "n_felonies", "crime_weighted",
            "riders", "latitude", "longitude"]
    out = df.dropna(subset=["proxy_value"])[keep].reset_index(drop=True)
    out.attrs["metric"] = metric
    out.attrs["label"] = spec["label"]
    out.attrs["oriented"] = spec["oriented"]
    return out


def station_proxy_by_layer(layer: str, metric: str = HEADLINE_METRIC,
                           min_stations: int = MIN_STATIONS_PER_POLYGON,
                           **kwargs) -> pd.DataFrame:
    """Aggregate 1 station proxy into 1 geography layer.

    The value of a polygon is the UNWEIGHTED mean over the stations inside it,
    and `n_units_proxy` counts them. This is `_proxies.aggregate_proxy`, the
    same function every other unit measure of the project uses, thus the subway
    rows aggregate the way the parks and the schools rows do.
    """
    pts = px.points_from_latlon(station_proxy(metric, **kwargs))
    if pts.empty:
        return pd.DataFrame()
    return px.aggregate_proxy(pts, layer, min_units=min_stations)


# --------------------------------------------------------------------------
# 5. The provenance stamp and the exports
# --------------------------------------------------------------------------
def source_id(entry: Dict[str, Any]) -> str:
    """The dataset id of a recipe source, or the stem of its file."""
    return entry.get("resource_id") or Path(entry["file"]).stem


def stamp(frame: pd.DataFrame, proxy: str, metric: str,
          source: str, window: str) -> pd.DataFrame:
    """Add the proxy name and the 3 provenance columns to 1 frame.

    Warning: never write a fixed proxy name beside a dropdown. Derive the name
    from the selection. A fixed name lets a run export 1 metric under the name
    of another.
    """
    out = frame.copy()
    out["proxy"] = proxy
    out["proxy_metric"] = metric
    out["proxy_source"] = source
    out["proxy_window"] = window
    return out


def stamp_recipe(frame: pd.DataFrame, proxy: str, metric: str,
                 entry: Dict[str, Any],
                 window: Optional[str] = None) -> pd.DataFrame:
    """Stamp a frame whose source is 1 recipe entry."""
    return stamp(frame, proxy, metric, source_id(entry),
                 window or entry["window"])


#: The column order of the area export. It repeats the v1.1.0 order exactly, so
#: a reader of the 2 files sees the same schema.
AREA_COLUMNS: Sequence[str] = (
    "layer", "NTA2020", "proxy_mean", "n_units_proxy", "proxy",
    "boro_cd", "boroct2020", "proxy_metric", "proxy_source", "proxy_window",
)


def build_area_proxy(case: str = "subway",
                     crime_metric: str = "crime_density",
                     station_metrics: Sequence[str] = DEFAULT_METRICS,
                     layers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Build the whole area-layer proxy export of the subway case.

    The frame holds the 2 area proxies that the case already published, and the
    station proxies that this module adds. Every row carries `proxy`,
    `proxy_metric`, `proxy_source`, and `proxy_window`.

    Warning: the area rows come from `_proxies`, unchanged. The station rows use
    the same aggregation function, thus the 2 halves of the export share 1 rule.
    """
    import _geography as geo

    recipe = px.load_recipe(case)
    income_entry = next(e for e in recipe if "S1901" in e["name"])
    # Match the crime entry on its FILE, not on its name. The recipe now names
    # more than 1 NYPD source, and a name test would take the wrong one.
    crime_entry = next(e for e in recipe if e.get("file")
                       and Path(e["file"]).name == px.CRIME_CSV.name)
    use_layers = list(layers) if layers is not None else list(geo.LAYERS)

    frames: List[pd.DataFrame] = []
    for layer in use_layers:
        inc = px.income_by_layer(layer)
        if not inc.empty:
            frames.append(stamp_recipe(inc, "median_household_income",
                                       income_entry["column"], income_entry))
        cri = px.crime_by_layer(layer, metric=crime_metric)
        if not cri.empty:
            # The crime name IS the selection, thus a felony run cannot leave
            # this module under the name of a density run.
            frames.append(stamp_recipe(cri, crime_metric, crime_metric,
                                       crime_entry))
        for metric in station_metrics:
            spec = SUBWAY_METRICS[metric]
            sub = station_proxy_by_layer(layer, metric)
            if not sub.empty:
                frames.append(stamp(sub, metric, spec["metric"],
                                    spec["source"], WINDOW_LABEL))

    if not frames:
        return pd.DataFrame(columns=list(AREA_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    for col in AREA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out["n_units_proxy"] = out["n_units_proxy"].astype("int64")
    return out[list(AREA_COLUMNS)]


def build_station_table(station_metrics: Sequence[str] = DEFAULT_METRICS
                        ) -> pd.DataFrame:
    """Build the station-level proxy table, 1 row for each station and metric.

    This is the UNIT export of the case. The area export aggregates it, and
    `_unit_validation` reads the same numbers through `station_proxy`.
    """
    frames: List[pd.DataFrame] = []
    for metric in station_metrics:
        spec = SUBWAY_METRICS[metric]
        one = station_proxy(metric)
        if one.empty:
            continue
        one = stamp(one, metric, spec["metric"], spec["source"], WINDOW_LABEL)
        one["oriented"] = bool(spec["oriented"])
        frames.append(one)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["proxy", "station_key"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. The reports that a reader needs before a number
# --------------------------------------------------------------------------
def assignment_report(max_distance_ft: float = MAX_COMPLAINT_FT
                      ) -> Dict[str, Any]:
    """Measure the quality of the complaint-to-station join.

    | Key | Meaning |
    |-----|---------|
    | `n_complaints` | The complaints of the window |
    | `n_assigned` | The complaints inside the cap |
    | `assigned_pct` | The share the cap keeps |
    | `median_ft`, `p99_ft` | The distance to the chosen station |
    | `n_stations_hit` | The station keys that took a complaint |
    | `median_per_station` | The median count for each such key |
    | `name_agrees_pct` | The share where the NYPD name and the MTA name share a word |

    The last row is an independent check. The join uses position alone, thus a
    name that agrees is evidence that the position was right.
    """
    cr = transit_complaints()
    a = assign_complaints(max_distance_ft)
    per = a.groupby("station_key").size()
    agree = _name_agreement(a)
    return {
        "window": WINDOW_LABEL,
        "max_distance_ft": max_distance_ft,
        "n_complaints": int(len(cr)),
        "n_assigned": int(len(a)),
        "assigned_pct": round(100.0 * len(a) / len(cr), 2) if len(cr) else 0.0,
        "median_ft": round(float(a["dist_ft"].median()), 1),
        "p99_ft": round(float(a["dist_ft"].quantile(0.99)), 1),
        "n_stations_hit": int(len(per)),
        "median_per_station": float(per.median()),
        "q1_per_station": float(per.quantile(0.25)),
        "name_agrees_pct": agree,
    }


_NAME_STOP = {"AV", "AVE", "AVENUE", "ST", "STREET", "RD", "ROAD", "BLVD",
              "BOULEVARD", "SQ", "SQUARE", "PL", "PLACE", "PKWY", "PARKWAY",
              "THE", "AND", "OF"}


def _tokens(value: Any) -> set:
    text = re.sub(r"[^A-Z0-9]+", " ", str(value).upper())
    return {t for t in text.split() if t and t not in _NAME_STOP}


def _name_agreement(assigned: pd.DataFrame) -> float:
    """The share of assignments where the 2 names share a word.

    The NYPD writes `PACIFIC STREET` where the MTA writes
    `Atlantic Av-Barclays Ctr`, thus an exact name join is impossible and this
    is a check and never a key.
    """
    if assigned.empty:
        return float("nan")
    a = assigned["nypd_station_name"].map(_tokens)
    b = assigned["mta_stop_name"].map(_tokens)
    hit = [bool(x & y) for x, y in zip(a, b)]
    return round(100.0 * sum(hit) / len(hit), 1)


def split_half_reliability(metric: str = HEADLINE_METRIC,
                           split: str = "2025-04-01") -> Dict[str, Any]:
    """Test whether the station rate is a stable property of the station.

    The window holds 30 months. This cuts it in half, counts the complaints of
    each station in each half, and divides both counts by the SAME ridership.
    The 2 halves then give 2 independent estimates of the same rate.

    A high rank agreement says that the rate describes the station and not the
    months. A low one would say that the record is too thin to grade a station.

    Warning: this is a reliability test and not an accuracy test. It cannot see
    an error that both halves share, such as a station that the NYPD geocodes
    in the wrong place for the whole window.
    """
    from scipy import stats

    a = assign_complaints()
    cr = transit_complaints()[["cmplnt_num", "date"]]
    a = a.merge(cr, on="cmplnt_num", how="left")
    cut = pd.Timestamp(split)
    early = a[a["date"] < cut].groupby("station_key").size().rename("n_early")
    late = a[a["date"] >= cut].groupby("station_key").size().rename("n_late")

    keep = station_proxy(metric)[["proxy_key", "riders"]].rename(
        columns={"proxy_key": "station_key"})
    j = keep.join(early, on="station_key").join(late, on="station_key")
    j[["n_early", "n_late"]] = j[["n_early", "n_late"]].fillna(0.0)
    j["rate_early"] = j["n_early"] / j["riders"] * RIDER_UNIT
    j["rate_late"] = j["n_late"] / j["riders"] * RIDER_UNIT

    rho = float(stats.spearmanr(j["rate_early"], j["rate_late"]).statistic)
    r = float(j["rate_early"].corr(j["rate_late"]))
    return {"metric": metric, "split": split, "n_stations": int(len(j)),
            "n_early": int(j["n_early"].sum()), "n_late": int(j["n_late"].sum()),
            "spearman_rho": round(rho, 4), "pearson_r": round(r, 4)}


def cap_sensitivity(caps: Sequence[float] = (500.0, 1000.0, 2000.0),
                    metric: str = HEADLINE_METRIC) -> pd.DataFrame:
    """Rebuild the proxy under other distance caps and compare the orders.

    The cap is a choice. This table shows what the choice buys and what it
    costs, thus a reader sees that the result does not rest on 1,000 ft.
    """
    from scipy import stats

    base = station_proxy(metric, max_distance_ft=MAX_COMPLAINT_FT)
    rows = []
    for cap in caps:
        other = station_proxy(metric, max_distance_ft=float(cap))
        j = base.merge(other, on="proxy_key", suffixes=("", "_cap"))
        rho = float(stats.spearmanr(j["proxy_value"],
                                    j["proxy_value_cap"]).statistic)
        rows.append({"cap_ft": float(cap), "n_stations": int(len(other)),
                     "n_shared": int(len(j)),
                     "rho_against_default": round(rho, 4)})
    return pd.DataFrame(rows)


def coverage_report() -> Dict[str, Any]:
    """State how much of the case each part of the proxy reaches."""
    ent = load_entrances()
    keys = entrance_station_map()
    tab = station_table()
    riders = station_ridership()
    crime = station_crime()
    have_r = riders[riders["riders"].fillna(0) >= MIN_RIDERS]
    prx = station_proxy(HEADLINE_METRIC)
    kept = set(prx["proxy_key"])
    return {
        "n_entrances": int(len(ent)),
        "n_station_keys": int(tab["station_key"].nunique()),
        "n_complexes": int(keys["complex_id"].nunique()),
        "n_stations_id": int(keys["station_id"].nunique()),
        "n_sir_dropped": int(tab["is_sir"].sum()),
        "n_unreached_dropped": int(len(_named_but_unreached())),
        "n_with_ridership": int(len(have_r)),
        "n_with_crime": int(len(crime)),
        "n_headline_rows": int(len(prx)),
        "entrances_kept": int(keys[keys["station_key"].isin(kept)]
                              ["unit_uid"].nunique()),
        "entrances_per_station_median": float(tab["n_entrances"].median()),
    }
