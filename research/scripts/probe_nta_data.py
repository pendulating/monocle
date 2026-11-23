"""Build the neighborhood (NTA 2020) proxy table of the lens-refusal probe.

Round 2 of the probe draws images from the top-10 and the bottom-10 NTAs of
each proxy. This script makes that table.

Writes under `outputs/_monocle/probe_nta/data/`:

| File | Content |
|------|---------|
| `nta_proxies.parquet` | 1 row for each residential NTA (`NTAType == "0"`) |
| `tract_select_crime.parquet` | the 3 selected offence counts for each tract |
| `provenance.json` | the sources, the pull times, and the row counts |
| `build.log` | the console output of the run |

Input
-----

| Source | Use |
|--------|-----|
| `outputs/_monocle/probe/data/tract_proxies.parquet` | every tract proxy |
| `outputs/_monocle/probe/data/catalog_tracts.parquet` | the image count |
| `data/geo/2020_Census_Tracts_20260304.geojson` | the tract to NTA map |
| `data/geo/nynta2020_26b/nynta2020.shp` | the NTA name, borough, type, area |
| NYPD complaints `qgea-i56i` and `5uac-w243` | the 5 selected offences |

How each column aggregates
--------------------------

| Kind | Columns | Method |
|------|---------|--------|
| Sum | `population`, `n_tracts`, `n_img` | a plain sum over the tracts |
| Area | `area_km2` | the NTA polygon area, EPSG:2263 |
| Population weight | `median_income`, `black_share`, `white_share`, `foreign_born_share`, `bachelors_share`, `median_home_value` | a mean over the tracts, weight = tract population |
| Area rate | `homeless_density`, `crime_density`, `prostitution_density` | the NTA total over the NTA area |
| Population rate | `crime_per_capita`, `violent_per_capita` | the NTA total over the population, for each 1,000 residents |

**Warning: outside the tract layer a median is not a median.** You cannot
average medians. `median_income` and `median_home_value` are
population-weighted means of the tract medians. Call them that.

**The severity-weighted crime total.** The tract table holds `crime_density`,
which `_proxies.crime_by_layer` builds as the severity-weighted complaint count
over the tract area (felony 3, misdemeanor 2, violation 1). The weighted total
of a tract is thus `crime_density * area_km2`, and this script recovers it that
way. The tract column `crime_count` is a plain row count, not the weighted
total, so this script does not use it for the rate.

    .venv-vllm025cu129/bin/python research/scripts/probe_nta_data.py
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "outputs/_monocle/probe/data"
OUT = REPO / "outputs/_monocle/probe_nta/data"

TRACT_GEOJSON = REPO / "data/geo/2020_Census_Tracts_20260304.geojson"
NTA_SHP = REPO / "data/geo/nynta2020_26b/nynta2020.shp"

NYPD_CACHE_DIR = REPO / "data/external/nypd_complaints_select_2022plus"
NYPD_CACHE = NYPD_CACHE_DIR / "complaints.parquet"
NYPD_PROV = NYPD_CACHE_DIR / "provenance.json"
# The historic file and the year-to-date file. A complaint can appear in both,
# so the merge drops a repeated `cmplnt_num`.
NYPD_DATASETS = ("qgea-i56i", "5uac-w243")
NYPD_SINCE = "2022-01-01T00:00:00"
NYPD_OFFENCES = (
    "PROSTITUTION & RELATED OFFENSES",
    "MURDER & NON-NEGL. MANSLAUGHTER",
    "FELONY ASSAULT",
    "ROBBERY",
    "RAPE",
)
NYPD_COLS = ("cmplnt_num", "cmplnt_fr_dt", "law_cat_cd", "ofns_desc",
             "latitude", "longitude")
VIOLENT_OFFENCES = ("MURDER & NON-NEGL. MANSLAUGHTER", "FELONY ASSAULT",
                    "ROBBERY", "RAPE")
SODA_PAGE = 50_000
SODA_MAX_TRIES = 3

WORKING_CRS = "EPSG:2263"
SQFT_PER_KM2 = 10_763_910.4
NTA_RESIDENTIAL_TYPE = "0"

# A mean over the tracts, weight = tract population.
POP_WEIGHTED = ("median_income", "black_share", "white_share",
                "foreign_born_share", "bachelors_share", "median_home_value")

# The columns that get a top-10 and a bottom-10 list.
PROXIES = ("pop_density", "median_income", "black_share", "white_share",
           "foreign_born_share", "bachelors_share", "median_home_value",
           "homeless_density", "crime_density", "crime_per_capita",
           "prostitution_density", "violent_per_capita")

# A neighborhood under this population gives an unstable rate, so the extreme
# lists drop it. The table keeps every residential NTA.
MIN_POP = 2000
# The probe needs images. A top-10 entry under this count gets a flag.
MIN_IMG = 300

_LOG_LINES: list[str] = []


def log(m: str) -> None:
    line = f"[probe-nta] {m}"
    _LOG_LINES.append(line)
    print(line, flush=True)


def _auth_header() -> dict:
    """Basic authentication for NYC Open Data, from `.env`.

    Warning: never print the value. `_proxies` reads the 2 environment
    variables and builds the header, so this script holds no credential.
    """
    from research.notebooks.cvpr import _proxies as P

    return P._auth_header()


def soda_pull(rid: str, where: str, select: str) -> pd.DataFrame:
    """Page one Socrata resource and return every row.

    The loop stops on a short page. A failure retries at most
    `SODA_MAX_TRIES` times, then raises an error.
    """
    headers = _auth_header()
    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {"$select": select, "$where": where, "$limit": SODA_PAGE,
                  "$offset": offset, "$order": ":id"}
        url = (f"https://data.cityofnewyork.us/resource/{rid}.json?"
               + urllib.parse.urlencode(params))
        rows = None
        for attempt in range(SODA_MAX_TRIES):
            req = urllib.request.Request(url)
            for k, v in headers.items():
                req.add_header(k, v)
            try:
                rows = json.load(urllib.request.urlopen(req, timeout=300))
                break
            except Exception as exc:
                detail = ""
                body = getattr(exc, "read", None)
                if callable(body):
                    try:
                        detail = body().decode(errors="replace")[:300]
                    except Exception:
                        detail = ""
                log(f"SODA {rid} offset={offset} attempt {attempt + 1}/"
                    f"{SODA_MAX_TRIES} failed: {type(exc).__name__} {detail}")
                if attempt + 1 >= SODA_MAX_TRIES:
                    # The credential sits in the header, not in the URL, so the
                    # query text is safe to show.
                    raise RuntimeError(
                        f"SODA request failed for {rid}: {type(exc).__name__}\n"
                        f"  query: {urllib.parse.urlencode(params)}\n"
                        f"  server said: {detail}") from None
                time.sleep(3.0 * (attempt + 1))
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        log(f"SODA {rid}: {len(rows):,} rows at offset {offset:,}")
        if len(rows) < SODA_PAGE:
            break
        offset += SODA_PAGE
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pull_nypd(refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Get the 5 selected offences since 2022, from the 2 NYPD datasets.

    The result carries only the rows that hold a coordinate.
    """
    if NYPD_CACHE.exists() and not refresh:
        df = pd.read_parquet(NYPD_CACHE)
        prov = json.loads(NYPD_PROV.read_text()) if NYPD_PROV.exists() else {}
        prov["cached"] = True
        log(f"NYPD: {len(df):,} rows from the cache {NYPD_CACHE}")
        return df, prov

    inlist = ",".join("'" + o + "'" for o in NYPD_OFFENCES)
    where = f"cmplnt_fr_dt >= '{NYPD_SINCE}' AND ofns_desc in({inlist})"
    select = ",".join(NYPD_COLS)
    raw_counts = {}
    frames = []
    for rid in NYPD_DATASETS:
        d = soda_pull(rid, where, select)
        d["dataset"] = rid
        raw_counts[rid] = int(len(d))
        log(f"NYPD {rid}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    n_raw = len(df)
    df = df.drop_duplicates(subset=["cmplnt_num"], keep="first")
    n_dedup = len(df)
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    n_geo = len(df)
    df["cmplnt_fr_dt"] = pd.to_datetime(df["cmplnt_fr_dt"], errors="coerce")
    log(f"NYPD: {n_raw:,} rows, {n_dedup:,} after the duplicate drop, "
        f"{n_geo:,} with a coordinate")

    by_off = df["ofns_desc"].value_counts().to_dict()
    prov = {
        "datasets": list(NYPD_DATASETS),
        "endpoint": "https://data.cityofnewyork.us/resource/<id>.json",
        "where": where,
        "select": select,
        "pulled": datetime.now(timezone.utc).isoformat(),
        "rows_by_dataset": raw_counts,
        "rows_raw": int(n_raw),
        "rows_after_dedup": int(n_dedup),
        "rows_with_coordinates": int(n_geo),
        "rows_by_ofns_desc": {k: int(v) for k, v in by_off.items()},
        "date_min": str(df["cmplnt_fr_dt"].min()),
        "date_max": str(df["cmplnt_fr_dt"].max()),
        "cached": False,
    }
    NYPD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NYPD_CACHE, index=False)
    NYPD_PROV.write_text(json.dumps(prov, indent=2))
    log(f"NYPD: wrote {NYPD_CACHE}")
    return df, prov


def select_crime_by_tract(crime: pd.DataFrame) -> pd.DataFrame:
    """Put each selected complaint into its census tract.

    The join runs in EPSG:2263, point in polygon.
    """
    import geopandas as gpd

    tr = gpd.read_file(TRACT_GEOJSON).to_crs(WORKING_CRS)
    tr["geoid"] = tr["geoid"].astype(str)
    tr["boroct2020"] = tr["boroct2020"].astype(str)
    tr = tr[["geoid", "boroct2020", "geometry"]]

    pts = gpd.GeoDataFrame(
        crime[["ofns_desc"]].copy(),
        geometry=gpd.points_from_xy(crime["longitude"], crime["latitude"]),
        crs="EPSG:4326").to_crs(WORKING_CRS)
    hit = gpd.sjoin(pts, tr, how="inner", predicate="within")
    log(f"select crime: {len(hit):,} of {len(pts):,} points inside a tract")

    hit["is_prostitution"] = hit["ofns_desc"].eq("PROSTITUTION & RELATED OFFENSES")
    hit["is_murder"] = hit["ofns_desc"].eq("MURDER & NON-NEGL. MANSLAUGHTER")
    hit["is_violent"] = hit["ofns_desc"].isin(VIOLENT_OFFENCES)
    agg = hit.groupby(["boroct2020", "geoid"], as_index=False).agg(
        prostitution_count=("is_prostitution", "sum"),
        murder_count=("is_murder", "sum"),
        violent_count=("is_violent", "sum"))
    for c in ("prostitution_count", "murder_count", "violent_count"):
        agg[c] = agg[c].astype(int)
    return agg


def nta_frame() -> pd.DataFrame:
    """Read the residential NTAs with their name, borough, and area."""
    import geopandas as gpd

    g = gpd.read_file(NTA_SHP).to_crs(WORKING_CRS)
    g = g[g["NTAType"].astype(str).eq(NTA_RESIDENTIAL_TYPE)].copy()
    g["area_km2"] = g.area / SQFT_PER_KM2
    out = pd.DataFrame({
        "nta2020": g["NTA2020"].astype(str),
        "ntaname": g["NTAName"].astype(str),
        "boroname": g["BoroName"].astype(str),
        "area_km2": g["area_km2"].values,
    })
    return out.sort_values("nta2020").reset_index(drop=True)


def tract_to_nta() -> pd.DataFrame:
    """The tract to NTA map, from the tract file itself.

    The 2 geographies nest, so the file carries the answer. No spatial join is
    necessary here.
    """
    import geopandas as gpd

    tr = gpd.read_file(TRACT_GEOJSON)
    out = pd.DataFrame({
        "boroct2020": tr["boroct2020"].astype(str),
        "geoid": tr["geoid"].astype(str),
        "nta2020": tr["nta2020"].astype(str),
    })
    return out


def wmean(values: pd.Series, weights: pd.Series) -> float:
    """A weighted mean that drops a null value and a null or zero weight."""
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return np.nan
    v = values[ok].astype(float)
    w = weights[ok].astype(float)
    return float((v * w).sum() / w.sum())


def extremes(df: pd.DataFrame, col: str, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The n highest and the n lowest NTAs of one proxy."""
    d = df[df["population"] > MIN_POP].dropna(subset=[col])
    d = d.sort_values(col, ascending=False)
    return d.head(n), d.tail(n).iloc[::-1]


def show_extremes(df: pd.DataFrame, col: str) -> list[str]:
    """Print the top-10 and the bottom-10 of one proxy. Return the flags."""
    top, bot = extremes(df, col)
    flags = []
    log("")
    log(f"=== {col} (NTAs with population > {MIN_POP:,}; "
        f"{len(df[df['population'] > MIN_POP])} of {len(df)}) ===")
    for label, part in (("TOP-10", top), ("BOTTOM-10", bot)):
        log(f"  -- {label} --")
        for _, r in part.iterrows():
            log(f"     {r[col]:>14,.4f}  n_img={int(r['n_img']):>6,}  "
                f"{r['nta2020']}  {r['ntaname']} ({r['boroname']})")
    thin = top[top["n_img"] < MIN_IMG]
    if len(thin):
        for _, r in thin.iterrows():
            flags.append(f"{col}: top-10 NTA {r['nta2020']} {r['ntaname']} "
                         f"has only {int(r['n_img'])} images")
    # A tie at the low end makes the bottom-10 an arbitrary pick.
    d = df[df["population"] > MIN_POP].dropna(subset=[col])
    lo = d[col].min()
    n_tie = int((d[col] == lo).sum())
    if n_tie > 1:
        msg = (f"{col}: {n_tie} NTAs tie at the lowest value {lo:,.4f}, so the "
               f"bottom-10 above is an arbitrary pick among them")
        log(f"  !! {msg}")
        flags.append(msg)
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-nypd", action="store_true",
                    help="pull the NYPD complaints again, ignore the cache")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prov: dict = {"built": datetime.now(timezone.utc).isoformat()}

    tracts = pd.read_parquet(PROBE / "tract_proxies.parquet")
    tracts["boroct2020"] = tracts["boroct2020"].astype(str)
    tracts["geoid"] = tracts["geoid"].astype(str)
    log(f"tract proxies: {len(tracts):,} rows, "
        f"{len(tracts.columns)} columns")
    prov["tract_table"] = {"path": str(PROBE / "tract_proxies.parquet"),
                           "rows": int(len(tracts)),
                           "columns": list(tracts.columns)}

    # The severity-weighted complaint total of a tract. See the module docstring.
    tracts["crime_weighted"] = tracts["crime_density"] * tracts["area_km2"]

    m = tract_to_nta()
    tracts = tracts.merge(m[["boroct2020", "nta2020"]], on="boroct2020",
                          how="left")
    log(f"tract to NTA: {tracts['nta2020'].notna().sum():,} tracts mapped")

    nta = nta_frame()
    log(f"residential NTAs: {len(nta)}")

    cat = pd.read_parquet(PROBE / "catalog_tracts.parquet",
                          columns=["boroct2020"])
    cat["boroct2020"] = cat["boroct2020"].astype(str)
    cat = cat.merge(m[["boroct2020", "nta2020"]], on="boroct2020", how="left")
    img = cat.groupby("nta2020").size().rename("n_img").reset_index()
    log(f"catalog: {len(cat):,} images, {img['n_img'].sum():,} in an NTA")
    prov["catalog"] = {"path": str(PROBE / "catalog_tracts.parquet"),
                       "images": int(len(cat))}

    crime, cprov = pull_nypd(refresh=args.refresh_nypd)
    prov["nypd_select"] = cprov
    sel = select_crime_by_tract(crime)
    sel.to_parquet(OUT / "tract_select_crime.parquet", index=False)
    log(f"wrote {OUT / 'tract_select_crime.parquet'} ({len(sel):,} tracts)")
    prov["tract_select_crime"] = {
        "rows": int(len(sel)),
        "prostitution_count": int(sel["prostitution_count"].sum()),
        "murder_count": int(sel["murder_count"].sum()),
        "violent_count": int(sel["violent_count"].sum()),
    }

    tracts = tracts.merge(sel, on=["boroct2020", "geoid"], how="left")
    for c in ("prostitution_count", "murder_count", "violent_count"):
        tracts[c] = tracts[c].fillna(0).astype(int)

    # -------------------------------------------------------------- aggregate
    rows = []
    for code, g in tracts.dropna(subset=["nta2020"]).groupby("nta2020"):
        rec = {"nta2020": code,
               "n_tracts": int(len(g)),
               "population": float(g["population"].sum(skipna=True)),
               "homeless_count": float(g["homeless_count"].sum(skipna=True)),
               "crime_weighted": float(g["crime_weighted"].sum(skipna=True)),
               "prostitution_count": int(g["prostitution_count"].sum()),
               "murder_count": int(g["murder_count"].sum()),
               "violent_count": int(g["violent_count"].sum())}
        for c in POP_WEIGHTED:
            rec[c] = wmean(g[c], g["population"])
        rows.append(rec)
    agg = pd.DataFrame(rows)

    t = nta.merge(agg, on="nta2020", how="left").merge(img, on="nta2020",
                                                       how="left")
    t["n_img"] = t["n_img"].fillna(0).astype(int)
    t["n_tracts"] = t["n_tracts"].fillna(0).astype(int)
    t["population"] = t["population"].fillna(0.0)
    t["pop_density"] = t["population"] / t["area_km2"]
    t["homeless_density"] = t["homeless_count"] / t["area_km2"]
    t["crime_density"] = t["crime_weighted"] / t["area_km2"]
    pop = t["population"].where(t["population"] > 0)
    t["crime_per_capita"] = t["crime_weighted"] / pop * 1000.0
    t["prostitution_density"] = t["prostitution_count"] / t["area_km2"]
    t["violent_per_capita"] = t["violent_count"] / pop * 1000.0

    cols = ["nta2020", "ntaname", "boroname", "n_tracts", "population",
            "area_km2", "pop_density", "n_img",
            "median_income", "black_share", "white_share",
            "foreign_born_share", "bachelors_share", "median_home_value",
            "homeless_density", "crime_density", "crime_per_capita",
            "prostitution_count", "prostitution_density", "murder_count",
            "violent_count", "violent_per_capita"]
    out = t[cols].sort_values("nta2020").reset_index(drop=True)
    out.to_parquet(OUT / "nta_proxies.parquet", index=False)
    log(f"wrote {OUT / 'nta_proxies.parquet'} ({len(out)} NTAs, "
        f"{time.time() - t0:.0f}s)")

    # ------------------------------------------------------------- verify
    log("")
    log("=== null count for each column ===")
    for c in cols:
        log(f"  {c:<22} nulls={int(out[c].isna().sum()):>4}  "
            f"nonnull={int(out[c].notna().sum()):>4}")
    log("")
    log(f"NTAs with population > {MIN_POP:,}: "
        f"{int((out['population'] > MIN_POP).sum())} of {len(out)}")

    flags: list[str] = []
    for c in PROXIES:
        flags += show_extremes(out, c)

    log("")
    if flags:
        log(f"=== FLAG: {len(flags)} problems ===")
        for f in flags:
            log(f"  {f}")
    else:
        log(f"=== no flag: every top-10 NTA holds {MIN_IMG} images or more, "
            f"and no proxy ties at its lowest value ===")

    prov["nta_table"] = {
        "rows": int(len(out)),
        "rows_min_pop": int((out["population"] > MIN_POP).sum()),
        "min_pop_for_extremes": MIN_POP,
        "min_img_flag": MIN_IMG,
        "nulls": {c: int(out[c].isna().sum()) for c in cols},
        "columns": cols,
    }
    prov["method"] = {
        "population_weighted": list(POP_WEIGHTED),
        "area_km2": "the NTA polygon area, EPSG:2263, from nynta2020.shp",
        "crime_weighted_total": ("tract crime_density * tract area_km2, summed "
                                 "over the tracts of the NTA; the tract "
                                 "crime_density comes from "
                                 "_proxies.crime_by_layer('census_tract'), "
                                 "which weights felony 3, misdemeanor 2, "
                                 "violation 1"),
        "crime_per_capita": "the weighted total over the population * 1000",
        "homeless_density": "the tract homeless_count sum over the NTA area",
        "select_crime": ("a point-in-polygon join to the tracts in EPSG:2263, "
                         "then a sum over the tracts of the NTA"),
        "warning": ("median_income and median_home_value are "
                    "population-weighted means of the tract medians, not "
                    "medians"),
    }
    prov["flags"] = flags
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    (OUT / "build.log").write_text("\n".join(_LOG_LINES) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
