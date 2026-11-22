"""Build the tract table and the proxies of the lens-refusal probe.

Writes under `outputs/_monocle/probe/data/`:

| File | Content |
|------|---------|
| `catalog_tracts.parquet` | every image of the 500k Cyclomedia catalog with its 2020 census tract |
| `tract_proxies.parquet` | 1 row per tract: area, population, income, crime, homelessness, race, migration |
| `provenance.json` | the pulls, their dates, and what was skipped |

The proxies:

| Column | Source | Pull |
|--------|--------|------|
| `median_income` | ACS S1901 on disk (`_proxies.load_tract_income`) | none |
| `crime_density` | NYPD complaints on disk (`_proxies.crime_by_layer`), severity-weighted per km2, NOT negated | none |
| `homeless_density` | 311 `Encampment` + `Homeless Person Assistance` complaints since 2024-01-01, per km2 | Socrata, cached under `data/external/nyc311_homeless/` |
| `black_share`, `white_share` | ACS B03002, non-Hispanic Black and White over total | Census API, needs `CENSUS_API_KEY` |
| `foreign_born_share` | ACS B05002, foreign born over total | Census API, needs `CENSUS_API_KEY` |
| `bachelors_share` | ACS B15003, a bachelor degree or more over the population of 25 years or more | Census API, needs `CENSUS_API_KEY` |
| `median_home_value` | ACS B25077, median value of an owner-occupied home | Census API, needs `CENSUS_API_KEY` |

The Census API refuses a request with no key ("Missing Key"). With no key
in the environment or in `.env`, the script skips the 2 ACS pulls, leaves
the columns empty, and says so in the provenance file. Run it again with
the key and only the ACS step runs.

    .venv-vllm025cu129/bin/python research/scripts/probe_data.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
CVPR = REPO / "research/notebooks/cvpr"
OUT = REPO / "outputs/_monocle/probe/data"
CATALOG = REPO / "data/cyclomedia/cyclomedia_all_2025_citywide_500k.parquet"
NYC311_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
NYC311_CACHE = REPO / "data/external/nyc311_homeless/nyc311_homeless_2024on.parquet"
NYC311_TYPES = ("Encampment", "Homeless Person Assistance")
NYC311_SINCE = "2024-01-01T00:00:00"
ACS_CACHE = (REPO / "data/external/acs_b03002_b05002_b15003_b25077"
             / "acs5_2024_b03002_b05002_b15003_b25077_nyc_tract.parquet")
ACS_URL = "https://api.census.gov/data/2024/acs/acs5"
ACS_VARS = ["B03002_001E", "B03002_003E", "B03002_004E",
            "B05002_001E", "B05002_013E",
            "B15003_001E", "B15003_022E", "B15003_023E", "B15003_024E",
            "B15003_025E",
            "B25077_001E"]
# The columns that the tract table takes from the ACS pull.
ACS_SHARES = ["white_share", "black_share", "foreign_born_share",
              "bachelors_share", "median_home_value"]
# The Census API marks a value that it cannot estimate with a large negative
# sentinel, such as -666666666. A real dollar value is never negative.
ACS_SENTINEL_MIN = 0.0
NYC_COUNTIES = ("005", "047", "061", "081", "085")


def log(m: str) -> None:
    print(f"[probe-data] {m}", flush=True)


def load_dotenv_keys() -> None:
    """Put the KEY=VALUE lines of `.env` into the environment, when unset."""
    p = REPO / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def tract_frame():
    import geopandas as gpd

    import _geography as geo
    import _proxies as P

    tr = gpd.read_file(geo.LAYERS["census_tract"]["path"]).to_crs(geo.WORKING_CRS)
    tr["geoid"] = tr["geoid"].astype(str)
    tr["boroct2020"] = tr["boroct2020"].astype(str)
    tr = tr[["geoid", "boroct2020", "geometry"]].copy()
    tr["area_km2"] = tr.area / P.SQFT_PER_KM2
    return tr


def join_catalog(tr) -> pd.DataFrame:
    import geopandas as gpd

    cat = pd.read_parquet(CATALOG, columns=[
        "sample_id", "image_path", "recording_id", "face", "dataset",
        "latitude", "longitude"])
    pts = gpd.GeoDataFrame(
        cat, geometry=gpd.points_from_xy(cat["longitude"], cat["latitude"]),
        crs="EPSG:4326").to_crs(tr.crs)
    hit = gpd.sjoin(pts, tr[["geoid", "boroct2020", "geometry"]],
                    how="inner", predicate="within")
    hit = hit.drop(columns=["geometry", "index_right"])
    log(f"catalog: {len(cat):,} images, {len(hit):,} inside a tract, "
        f"{hit['geoid'].nunique():,} tracts")
    return pd.DataFrame(hit)


def pull_311(tr) -> tuple[pd.DataFrame, dict]:
    import geopandas as gpd

    sys.path.insert(0, str(REPO))
    from dagspaces.common.curation.socrata import fetch_socrata

    types = ",".join(f"'{t}'" for t in NYC311_TYPES)
    where = f"created_date>='{NYC311_SINCE}' AND complaint_type in({types})"
    NYC311_CACHE.parent.mkdir(parents=True, exist_ok=True)
    res = fetch_socrata(
        NYC311_URL, where=where,
        select="unique_key,created_date,complaint_type,latitude,longitude",
        cache_path=str(NYC311_CACHE))
    df = res.df.to_pandas()
    if not res.cached:
        res.df.write_parquet(str(NYC311_CACHE))
    prov = {"rows": int(len(df)), "cached": bool(res.cached),
            "truncated_likely": bool(res.truncated_likely),
            "where": where, "cache": str(NYC311_CACHE)}
    d = df.dropna(subset=["latitude", "longitude"]).copy()
    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    d = d.dropna(subset=["latitude", "longitude"])
    pts = gpd.GeoDataFrame(
        d, geometry=gpd.points_from_xy(d["longitude"], d["latitude"]),
        crs="EPSG:4326").to_crs(tr.crs)
    hit = gpd.sjoin(pts, tr[["geoid", "geometry"]], how="inner",
                    predicate="within")
    cnt = hit.groupby("geoid").size().rename("homeless_count").reset_index()
    log(f"311: {len(df):,} rows, {len(hit):,} inside a tract "
        f"({', '.join(f'{k}={v}' for k, v in df['complaint_type'].value_counts().items())})")
    return cnt, prov


def pull_acs(tr) -> tuple[pd.DataFrame | None, dict]:
    key = os.environ.get("CENSUS_API_KEY", "")
    if ACS_CACHE.exists():
        df = pd.read_parquet(ACS_CACHE)
        return df, {"cached": True, "rows": int(len(df)), "cache": str(ACS_CACHE)}
    if not key:
        log("ACS: no CENSUS_API_KEY in the environment or .env; SKIPPED ("
            + ", ".join(ACS_SHARES) + " stay empty)")
        return None, {"skipped": "no CENSUS_API_KEY"}
    frames = []
    for county in NYC_COUNTIES:
        q = urllib.parse.urlencode({"get": ",".join(["NAME"] + ACS_VARS),
                                    "for": "tract:*",
                                    "in": f"state:36 county:{county}",
                                    "key": key})
        with urllib.request.urlopen(f"{ACS_URL}?{q}", timeout=120) as r:
            rows = json.loads(r.read().decode())
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
        log(f"ACS: county {county}: {len(rows) - 1} tracts")
    df = pd.concat(frames, ignore_index=True)
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    for v in ACS_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    tot = df["B03002_001E"].where(df["B03002_001E"] > 0)
    df["white_share"] = df["B03002_003E"] / tot
    df["black_share"] = df["B03002_004E"] / tot
    tot2 = df["B05002_001E"].where(df["B05002_001E"] > 0)
    df["foreign_born_share"] = df["B05002_013E"] / tot2
    edu = df["B15003_001E"].where(df["B15003_001E"] > 0)
    deg_cols = ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
    deg = df[deg_cols].where(df[deg_cols] >= ACS_SENTINEL_MIN).sum(
        axis=1, min_count=len(deg_cols))
    df["bachelors_share"] = deg / edu
    # Drop the sentinel, so a tract with no estimate stays null.
    df["median_home_value"] = df["B25077_001E"].where(
        df["B25077_001E"] >= ACS_SENTINEL_MIN)
    df = df[["geoid"] + ACS_SHARES + ACS_VARS]
    ACS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ACS_CACHE, index=False)
    return df, {"cached": False, "rows": int(len(df)), "cache": str(ACS_CACHE),
                "pulled": datetime.now(timezone.utc).isoformat()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-catalog", action="store_true",
                    help="keep the catalog join on disk")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CVPR))
    import _proxies as P

    load_dotenv_keys()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prov: dict = {"built": datetime.now(timezone.utc).isoformat()}

    tr = tract_frame()
    log(f"tracts: {len(tr):,}")

    cat_out = OUT / "catalog_tracts.parquet"
    if args.skip_catalog and cat_out.exists():
        # The join stays on disk, but the provenance keeps the same counts.
        cat = pd.read_parquet(cat_out, columns=["geoid"])
        log(f"catalog: on disk, {len(cat):,} images, "
            f"{cat['geoid'].nunique():,} tracts")
    else:
        cat = join_catalog(tr)
        cat.to_parquet(cat_out, index=False)
    prov["catalog"] = {"images": int(len(cat)),
                       "tracts": int(cat["geoid"].nunique())}

    base = pd.DataFrame(tr.drop(columns="geometry"))

    inc = P.load_tract_income()
    inc["population"] = pd.to_numeric(inc["population"], errors="coerce")
    base = base.merge(inc[["geoid", "median_income", "population"]],
                      on="geoid", how="left")
    prov["income"] = {"tracts_with_value": int(base["median_income"].notna().sum()),
                      "source": str(P.INCOME_CSV)}

    cr = P.crime_by_layer("census_tract")
    cr["boroct2020"] = cr["boroct2020"].astype(str)
    cr = cr.rename(columns={"proxy_mean": "crime_density_neg"})
    cr["crime_density"] = -cr["crime_density_neg"]
    cr = cr.rename(columns={"n_units_proxy": "crime_count"})
    base = base.merge(cr[["boroct2020", "crime_density", "crime_count"]],
                      on="boroct2020", how="left")
    base["crime_density"] = base["crime_density"].fillna(0.0)
    base["crime_count"] = base["crime_count"].fillna(0).astype(int)
    # Complaints per km2 rise with density, so a per-capita rate goes beside
    # it as a check for the gang probe.
    cpc = P.crime_by_layer("census_tract", metric="crime_per_capita")
    cpc["boroct2020"] = cpc["boroct2020"].astype(str)
    cpc["crime_per_capita"] = -cpc["proxy_mean"]
    base = base.merge(cpc[["boroct2020", "crime_per_capita"]], on="boroct2020", how="left")
    prov["crime"] = {"tracts_with_complaints": int((base["crime_count"] > 0).sum()),
                     "tracts_with_per_capita": int(base["crime_per_capita"].notna().sum()),
                     "source": str(P.CRIME_CSV)}

    cnt, p311 = pull_311(tr)
    base = base.merge(cnt, on="geoid", how="left")
    base["homeless_count"] = base["homeless_count"].fillna(0).astype(int)
    base["homeless_density"] = base["homeless_count"] / base["area_km2"]
    prov["homeless_311"] = p311

    acs, pacs = pull_acs(tr)
    if acs is not None:
        base = base.merge(acs[["geoid"] + ACS_SHARES], on="geoid", how="left")
    else:
        for c in ACS_SHARES:
            base[c] = np.nan
    prov["acs_race_migration"] = pacs

    base.to_parquet(OUT / "tract_proxies.parquet", index=False)
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2))
    log(f"wrote {OUT / 'tract_proxies.parquet'} ({len(base):,} tracts, "
        f"{time.time() - t0:.0f}s)")
    log(base.describe().T[["count", "mean", "50%"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
