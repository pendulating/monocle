#!/usr/bin/env python
"""F4: give each 2020 NTA its socioeconomic features, from ACS 2024 5-year tract data.

The NTA rank test against the features (D132 batch, `_f4`) needs 1 row for
each NTA. NTAs are unions of 2020 census tracts; the tract file
(`data/geo/2020_Census_Tracts_20260304.geojson`) names the NTA of each tract.
Thus a count sums over the tracts of an NTA with no spatial join.

| Feature | Source (tract) | NTA value |
|---------|----------------|-----------|
| `pct_white_nh` | B03002_003 / B03002_001 | sum of counts over sum of the population |
| `pct_black_nh` | B03002_004 / B03002_001 | same |
| `pct_asian_nh` | B03002_006 / B03002_001 | same |
| `pct_bachelors` | (B15003_022 + 023 + 024 + 025) / B15003_001: a bachelor's degree or higher, adults 25 and over | same |
| `median_household_income` | S1901: the households in 10 income brackets | the median of the summed brackets, by linear interpolation inside the bracket (the Census method for a median of grouped data) |

The top bracket ($200,000 or more) has no upper edge. An NTA whose median
falls in it gets 200,000 and `income_topcoded` = true. The script checks the
interpolation on the tracts: it compares the interpolated median of each
tract with the published tract median (S1901_C01_012E).

Inputs: `data/external/acs_b03002_b05002_b15003_b25077/...nyc_tract.parquet`
(from `research/scripts/probe_data.py`), `data/demo/ct/ACSST5Y2024.S1901-Data.csv`,
and B03002_006 (Asian alone, not Hispanic), pulled from the Census API 1
time and cached in `data/external/acs_b03002_006/`. The API needs
`CENSUS_API_KEY`; `ensure_dotenv()` loads it, and the script never prints it.

Output: `data/f4_traces/nta_acs.parquet` and `nta_acs_manifest.json`.
Version 1.2.0 also writes the same features, by the same method, for 2 more
layers (`LAYERS`): `tract_acs.parquet` (key `boroct2020`, 1 tract each) and
`cd_acs.parquet` (key `boro_cd`). A community district is the union of the
tracts of its CDTA ("CD equivalent"; MN01 is `boro_cd` 101). A CDTA follows
the tract edges, thus it is near the district and not equal to it. The
joint interest areas (CDTA number 26 or more: parks, airports) get no
`boro_cd`. The NTA file does not change.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/nta_acs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
sys.path.insert(0, ROOT)
from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402

TRACTS = os.path.join(ROOT, "data/geo/2020_Census_Tracts_20260304.geojson")
ACS = os.path.join(ROOT, "data/external/acs_b03002_b05002_b15003_b25077/"
                         "acs5_2024_b03002_b05002_b15003_b25077_nyc_tract.parquet")
ASIAN = os.path.join(ROOT, "data/external/acs_b03002_006/acs5_2024_b03002_006_nyc_tract.parquet")
S1901 = os.path.join(ROOT, "data/demo/ct/ACSST5Y2024.S1901-Data.csv")
OUT = os.path.join(ROOT, "data/f4_traces/nta_acs.parquet")
API = "https://api.census.gov/data/2024/acs/acs5"
COUNTIES = ("005", "047", "061", "081", "085")
# The lower edge of each S1901 bracket (C01_002 to C01_011); the last bracket is open.
EDGES = [0, 10_000, 15_000, 25_000, 35_000, 50_000, 75_000, 100_000, 150_000, 200_000]
BRACKETS = [f"S1901_C01_{i:03d}E" for i in range(2, 12)]

__version__ = "1.2.0"

# layer -> (the key column of the output, the file name)
LAYERS = {"nta": ("nta2020", "nta_acs.parquet"), "tract": ("boroct2020", "tract_acs.parquet"),
          "cd": ("boro_cd", "cd_acs.parquet")}
BORO_DIGIT = {"MN": "1", "BX": "2", "BK": "3", "QN": "4", "SI": "5"}


def boro_cd(cdta: str) -> str | None:
    """Give the `boro_cd` of a CDTA ("MN01" -> "101"), or None for a joint interest area."""
    if not isinstance(cdta, str) or len(cdta) != 4 or int(cdta[2:]) >= 26:
        return None
    return BORO_DIGIT[cdta[:2]] + cdta[2:]


def pull_asian() -> pd.DataFrame:
    """Read B03002_006 of each NYC tract from the cache, or pull it 1 time from the Census API."""
    if os.path.exists(ASIAN):
        return pd.read_parquet(ASIAN)
    ensure_dotenv()
    key = os.environ.get("CENSUS_API_KEY", "")
    if not key:
        raise SystemExit("CENSUS_API_KEY is not set; the Census API refuses a request with no key")
    frames = []
    for county in COUNTIES:
        q = urllib.parse.urlencode({"get": "B03002_001E,B03002_006E", "for": "tract:*",
                                    "in": f"state:36 county:{county}", "key": key})
        with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
            rows = json.loads(r.read().decode())
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
        print(f"[acs] county {county}: {len(rows) - 1} tracts")
    d = pd.concat(frames, ignore_index=True)
    d["geoid"] = d.state + d.county + d.tract
    d = d[["geoid", "B03002_001E", "B03002_006E"]].rename(columns={"B03002_001E": "total_check"})
    for c in ("total_check", "B03002_006E"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    os.makedirs(os.path.dirname(ASIAN), exist_ok=True)
    d.to_parquet(ASIAN, index=False)
    return d


def interpolated_median(counts: np.ndarray) -> tuple:
    """Give the median of grouped incomes and a flag for the open top bracket."""
    total = counts.sum()
    if not total > 0:
        return float("nan"), False
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, total / 2.0))
    if i >= len(EDGES) - 1:
        return float(EDGES[-1]), True
    before = cum[i - 1] if i else 0.0
    lo, hi = EDGES[i], EDGES[i + 1]
    return float(lo + (total / 2.0 - before) / counts[i] * (hi - lo)), False


def income_brackets() -> pd.DataFrame:
    """Give the households of each tract in each bracket (S1901 holds percents)."""
    s = pd.read_csv(S1901, skiprows=[1], dtype=str)
    s["geoid"] = s.GEO_ID.str[-11:]
    out = pd.DataFrame({"geoid": s.geoid,
                        "households": pd.to_numeric(s.S1901_C01_001E, errors="coerce"),
                        "published_median": pd.to_numeric(s.S1901_C01_012E.str.replace(r"[+,]", "", regex=True),
                                                          errors="coerce")})
    for c in BRACKETS:
        out[c] = pd.to_numeric(s[c], errors="coerce").fillna(0.0) / 100.0 * out.households.fillna(0.0)
    return out


def aggregate(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Sum the tract counts over `key` and give the 5 features of each area (1 row for each key)."""
    bach = ["B15003_022E", "B15003_023E", "B15003_024E", "B15003_025E"]
    d = d.dropna(subset=[key])
    # Group by the key only: 1 NTA can hold tracts of 2 boroughs (BX0802 holds Marble Hill, Manhattan).
    # The name and the borough of the area come from the borough with the largest population in it.
    names = (d.groupby([key, "ntaname", "boroname"]).B03002_001E.sum().reset_index()
             .sort_values("B03002_001E", ascending=False).drop_duplicates(key)
             .set_index(key)[["ntaname", "boroname"]])
    g = d.groupby(key, sort=True)
    sums = g[["B03002_001E", "B03002_003E", "B03002_004E", "B03002_006E", "B15003_001E", "households"]
             + bach + BRACKETS].sum(min_count=1)
    pop = sums.B03002_001E.where(sums.B03002_001E > 0)
    adults = sums.B15003_001E.where(sums.B15003_001E > 0)
    out = pd.DataFrame({
        "population": sums.B03002_001E, "households": sums.households, "tracts": g.size(),
        "pct_white_nh": 100 * sums.B03002_003E / pop,
        "pct_black_nh": 100 * sums.B03002_004E / pop,
        "pct_asian_nh": 100 * sums.B03002_006E / pop,
        "pct_bachelors": 100 * sums[bach].sum(axis=1) / adults,
    })
    med = [interpolated_median(r) for r in sums[BRACKETS].to_numpy()]
    out["median_household_income"] = [m for m, _ in med]
    out["income_topcoded"] = [f for _, f in med]
    out = out.join(names).reset_index()
    if key != "nta2020":
        out = out.drop(columns=["ntaname"])
    front = [c for c in (key, "ntaname", "boroname") if c in out.columns]
    out = out[front + [c for c in out.columns if c not in front]]
    if out[key].duplicated().any():
        raise SystemExit(f"a {key} has more than 1 row")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    props = pd.DataFrame([f["properties"] for f in json.load(open(TRACTS))["features"]])
    tracts = props[["geoid", "boroct2020", "nta2020", "ntaname", "cdta2020", "boroname"]]
    acs = pd.read_parquet(ACS)
    asian = pull_asian()
    inc = income_brackets()
    d = (tracts.merge(acs, on="geoid", how="left").merge(asian, on="geoid", how="left")
         .merge(inc, on="geoid", how="left"))
    unmatched = {"acs": int(d.B03002_001E.isna().sum()), "asian": int(d.B03002_006E.isna().sum()),
                 "s1901": int(d.households.isna().sum())}
    if (d.total_check.dropna() != d.B03002_001E[d.total_check.notna()]).any():
        raise SystemExit("B03002_001 of the 2 pulls differ; the pulls are not the same release")

    # The check of the interpolation on single tracts.
    tm = np.array([interpolated_median(r)[0] for r in d[BRACKETS].to_numpy()])
    ok = d.published_median.notna() & np.isfinite(tm) & (d.published_median < 250_000)
    rel = np.abs(tm[ok] - d.published_median[ok]) / d.published_median[ok]
    check = {"tracts": int(ok.sum()), "spearman": round(float(pd.Series(tm[ok]).corr(
        d.published_median[ok].reset_index(drop=True), method="spearman")), 4),
             "median_abs_rel_error": round(float(np.median(rel)), 4)}

    d["boro_cd"] = d.cdta2020.map(boro_cd)
    written = {}
    for layer, (key, name) in LAYERS.items():
        out = aggregate(d, key)
        path = os.path.join(os.path.dirname(args.out), name) if layer != "nta" else args.out
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_parquet(path, index=False)
        written[layer] = {"file": os.path.relpath(path, ROOT), "rows": int(len(out)),
                          "rows_with_population": int(out.population.gt(0).sum()),
                          "income_topcoded": int(out.income_topcoded.sum())}
        if layer == "nta":
            nta = out

    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": [os.path.relpath(p, ROOT) for p in (TRACTS, ACS, ASIAN, S1901)],
        "ntas": int(len(nta)), "ntas_with_population": int(nta.population.gt(0).sum()),
        "tracts_unmatched": unmatched, "income_topcoded": int(nta.income_topcoded.sum()), "layers": written,
        "interpolation_check_on_tracts": check,
    }
    with open(args.out.replace(".parquet", "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
