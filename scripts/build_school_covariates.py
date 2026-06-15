#!/usr/bin/env python3
"""Build a uid-keyed school covariate parquet for regression analyses.

Joins, per FacDB school unit (``uid``):

  * FacDB facilities (uid → BIN, BBL, borough, cd, sector, capacity)
  * DOE school point locations 2019-20 (BIN → DBN; placeholder "million
    BINs" dropped)
  * DOE demographic snapshot 2017-22 (DBN → latest-year enrollment,
    % Poverty, Economic Need Index, % SWD, % ELL); multiple DBNs in one
    building are aggregated enrollment-weighted
  * PLUTO 26v1 (BBL → YearBuilt, BldgClass, NumFloors, AssessTot, BldgArea,
    LandUse)

Output: ``curation/external/school_covariates.parquet`` + README. DOE-derived
fields are null for non-DOE schools (most non-public) — expected, not a bug.

Usage:
    python scripts/build_school_covariates.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/share/pierson/matt/mllmsci")
FACILITIES = ROOT / "curation/facdb_schools_k_12/facilities.parquet"
LOCATIONS = ROOT / "curation/external/doe_schools/school_point_locations_2019_2020_a3nt-yts4.csv"
DEMO = ROOT / "curation/external/doe_schools/demographic_snapshot_school_2017_2022_c7ru-d68s.csv"
PLUTO = ROOT / "curation/external/pluto/pluto_26v1.csv"
OUT = ROOT / "curation/external/school_covariates.parquet"

PCT_COLS = {
    "% Poverty": "pct_poverty",
    "Economic Need Index": "economic_need_index",
    "% Students with Disabilities": "pct_swd",
    "% English Language Learners": "pct_ell",
}


def _parse_pct(series: pd.Series) -> pd.Series:
    """Normalize DOE percent columns to 0-100 floats.

    Handles three formats seen in the snapshot: '84.7%' strings,
    'Above 95%'/'Below 5%' caps, and bare fractions (0.042). A column whose
    numeric values all sit in [0, 1] is treated as a fraction and scaled."""
    s = series.astype(str).str.strip()
    s = s.str.replace("Above ", "", regex=False).str.replace("Below ", "", regex=False)
    had_pct_sign = s.str.endswith("%")
    vals = pd.to_numeric(s.str.rstrip("%"), errors="coerce")
    # Scale bare fractions (no % sign anywhere in the column and max <= 1).
    no_sign = vals[~had_pct_sign].dropna()
    if len(no_sign) and no_sign.max() <= 1.0:
        vals = vals.where(had_pct_sign, vals * 100.0)
    return vals


def _norm_bbl(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def main() -> None:
    fac = pd.read_parquet(
        FACILITIES,
        columns=["uid", "facname", "bin", "bbl", "borough", "cd",
                 "facsubgrp", "factype", "capacity"],
    )
    fac["uid"] = fac["uid"].astype(str)
    fac["bin"] = pd.to_numeric(fac["bin"], errors="coerce").astype("Int64")
    fac["bbl"] = _norm_bbl(fac["bbl"])
    print(f"facilities: {len(fac):,} units")

    # ---- DOE locations: BIN -> DBN(s) ----
    loc = pd.read_csv(LOCATIONS, usecols=["ATS_Code", "BIN"])
    loc = loc.rename(columns={"ATS_Code": "dbn", "BIN": "bin"})
    loc["dbn"] = loc["dbn"].astype(str).str.strip()
    loc["bin"] = pd.to_numeric(loc["bin"], errors="coerce").astype("Int64")
    # Placeholder "million BINs" (1000000, 2000000, ...) carry no building identity.
    loc = loc[loc["bin"].notna() & (loc["bin"] % 1_000_000 != 0)]
    print(f"locations: {len(loc):,} DBN rows with a real BIN ({loc['bin'].nunique():,} BINs)")

    # ---- Demographic snapshot: latest year per DBN ----
    demo = pd.read_csv(
        DEMO, usecols=["DBN", "Year", "Total Enrollment"] + list(PCT_COLS),
    )
    demo = demo.rename(columns={"DBN": "dbn", "Year": "year",
                                "Total Enrollment": "enrollment"})
    demo["dbn"] = demo["dbn"].astype(str).str.strip()
    demo = demo.sort_values("year").drop_duplicates("dbn", keep="last")
    demo["enrollment"] = pd.to_numeric(demo["enrollment"], errors="coerce")
    for src, dst in PCT_COLS.items():
        demo[dst] = _parse_pct(demo[src])
    demo = demo[["dbn", "year", "enrollment"] + list(PCT_COLS.values())]
    print(f"demographics: {len(demo):,} DBNs (latest year per DBN, max={demo['year'].max()})")

    # ---- BIN-level enrollment-weighted aggregation across co-located DBNs ----
    dl = loc.merge(demo, on="dbn", how="inner")
    def _agg(group: pd.DataFrame) -> pd.Series:
        w = group["enrollment"].fillna(0.0).to_numpy(dtype=float)
        out = {"dbn_count": len(group), "total_enrollment": float(np.nansum(w))}
        for c in PCT_COLS.values():
            v = group[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.any():
                ww = np.where(w[ok] > 0, w[ok], 1.0)  # unweighted fallback
                out[c] = float(np.average(v[ok], weights=ww))
            else:
                out[c] = float("nan")
        return pd.Series(out)
    per_bin = dl.groupby("bin").apply(_agg, include_groups=False).reset_index()
    per_bin["bin"] = per_bin["bin"].astype("Int64")
    print(f"per-BIN DOE aggregates: {len(per_bin):,} buildings")

    # ---- PLUTO via BBL ----
    header = pd.read_csv(PLUTO, nrows=0)
    cmap = {c.lower(): c for c in header.columns}
    want = ["bbl", "yearbuilt", "bldgclass", "numfloors", "assesstot",
            "bldgarea", "lotarea", "landuse"]
    usecols = [cmap[w] for w in want if w in cmap]
    pluto = pd.read_csv(PLUTO, usecols=usecols, low_memory=False)
    pluto.columns = [c.lower() for c in pluto.columns]
    pluto["bbl"] = _norm_bbl(pluto["bbl"])
    pluto = pluto.drop_duplicates("bbl")
    pluto["yearbuilt"] = pd.to_numeric(pluto["yearbuilt"], errors="coerce").replace(0, np.nan)
    for c in ("numfloors", "assesstot", "bldgarea", "lotarea"):
        pluto[c] = pd.to_numeric(pluto[c], errors="coerce")
    print(f"pluto: {len(pluto):,} lots")

    # ---- Assemble ----
    cov = fac.merge(per_bin, on="bin", how="left")
    cov = cov.merge(pluto, on="bbl", how="left")
    cov["building_age"] = date.today().year - cov["yearbuilt"]
    cov["assess_per_bldg_sqft"] = np.where(
        cov["bldgarea"] > 0, cov["assesstot"] / cov["bldgarea"], np.nan
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cov["log_assesstot"] = np.where(cov["assesstot"] > 0, np.log10(cov["assesstot"]), np.nan)
    cov["capacity"] = pd.to_numeric(cov["capacity"], errors="coerce").replace(0, np.nan)
    cov["cd"] = cov["cd"].astype(str)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(OUT, index=False)

    # ---- Coverage report ----
    n = len(cov)
    print(f"\nWrote {OUT}  ({n:,} rows)")
    doe_mask = cov["facsubgrp"].isin(["PUBLIC K-12 SCHOOLS", "CHARTER K-12 SCHOOLS"])
    lines = [f"coverage (non-null / {n:,} units; DOE-sector units: {int(doe_mask.sum()):,}):"]
    for c in ["pct_poverty", "economic_need_index", "pct_swd", "pct_ell",
              "total_enrollment", "yearbuilt", "building_age", "numfloors",
              "assesstot", "assess_per_bldg_sqft", "capacity"]:
        nn = int(cov[c].notna().sum())
        nn_doe = int(cov.loc[doe_mask, c].notna().sum())
        lines.append(f"  {c:24} {nn:5,} overall   {nn_doe:5,} within DOE sector")
    report = "\n".join(lines)
    print(report)

    readme = OUT.parent / "school_covariates_README.md"
    readme.write_text(f"""# school_covariates.parquet

Built {date.today().isoformat()} by `scripts/build_school_covariates.py`. One row per
FacDB K-12 school unit (`uid`), for use as `--unit-metadata-parquet` in
`scripts/pairwise_vqa_regression_report.py` / `pairwise_vqa_difference_report.py`.

Sources:
- `curation/facdb_schools_k_12/facilities.parquet` (uid, bin, bbl, borough, cd, sector, capacity)
- DOE school point locations 2019-20 (`a3nt-yts4`): BIN→DBN; placeholder million-BINs dropped
- DOE demographic snapshot 2017-22 (`c7ru-d68s`): latest year per DBN; co-located DBNs
  aggregated enrollment-weighted per BIN (`dbn_count`, `total_enrollment`)
- PLUTO 26v1 (`64uk-42ks`) via BBL: yearbuilt (0→null), bldgclass, numfloors,
  assesstot, bldgarea, lotarea, landuse

Derived: `building_age` ({date.today().year} − yearbuilt), `assess_per_bldg_sqft`,
`log_assesstot` (log10), `capacity` (0→null).

Percent columns normalized to 0–100 ("Above 95%"→95, bare fractions ×100).

DOE-derived fields (pct_poverty, economic_need_index, pct_swd, pct_ell,
total_enrollment) are null for schools without a DOE DBN in the same BIN —
i.e. most non-public schools. Regressions on these describe the DOE sector only.

{report}
""")
    print(f"Wrote {readme}")


if __name__ == "__main__":
    main()
