"""Building-condition proxies for the libraries and the schools cases.

Why this module exists
----------------------
The libraries prompt asks which library BUILDING looks better kept, and the
schools prompt asks the same of a school. Until now the libraries case
validated against household income, which grades the households near the
building and not the building. The schools case validated against student
attendance, which grades the students and not the building. Neither measures
the object the model looks at.

FacDB gives a BIN to 96.8% of the libraries and to 99.0% of the schools, thus
the city building record reaches each unit by key and needs no spatial join.
The record IS a measurement of the building:

| Proxy | Source | What it measures |
|-------|--------|------------------|
| `building_violations_open` | DOB + ECB violations | Open faults of this building |
| `building_violations_total` | DOB + ECB violations | Every fault since the window |
| `alteration_permits` | DOB permit issuance | Money spent on this building |

Orientation
-----------
Every value leaves this module oriented "higher is better", the same contract
that `_proxies` keeps. A violation count carries a sign of -1, thus a positive
correlation always means that the model and the record agree. Do not undo the
sign in a notebook.

Normalization
-------------
A violation count is a RAW COUNT for one building, and this module divides it
by nothing. The unit is the building, so the count already describes 1 object.
A count for each square foot would ask a different question — "does this
building carry more faults than its size predicts" — and no city record states
that the model should read size. `building_footprint_area` is available if a
later reader wants that arm, and this module does not apply it.

Warning: a big campus collects more violations than a small branch library.
Read a correlation inside 1 case, never across the 2 cases.

Credentials
-----------
**Warning: never write a credential into this repository.** This module reads
the SODA key through `_proxies._auth_header`, which reads `.env`. Do not print
a key, and do not put one in a notebook cell or an error message.

The cache
---------
`_proxies.fetch` names its cache file from the first 120 characters of the
query. A BIN `IN` list is thousands of characters long, thus 2 different
batches would truncate to the SAME name and the second batch would read the
first batch's rows. This module therefore keeps its own cache and names each
file from a SHA-256 of the whole query. Delete `.proxy_cache/bld_*.parquet` to
force a refresh.

See `research/docs/building-condition-proxies.md`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

import _proxies as px

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path(__file__).resolve().parent / ".proxy_cache"
EXTERNAL_DIR = REPO_ROOT / "data" / "external"

SODA_PAGE = 50_000

#: How many BINs go into 1 `IN` list. A SODA query string has a length limit,
#: and 150 seven-digit BINs plus their quotes stay near 1,500 characters.
BIN_BATCH = 150

# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
#: The first day of the record window. Every violation and every permit below
#: comes from this day onward, thus the 3 proxies share 1 window and a reader
#: compares them directly.
#:
#: Why 2016: the permit proxy asks for the last 10 years, and the paper battery
#: ran in 2026. A wider window would let a fault of 1990 count against a
#: building that the city repaired in 2005.
WINDOW_START_YEAR = 2016

#: The day this module last read the network. The export writes it into
#: `proxy_window`, thus a reader of the parquet knows how fresh the count is.
PULL_DATE = "2026-08-31"

WINDOW_LABEL = f"{WINDOW_START_YEAR}-01-01 to {PULL_DATE} (pull date)"

# --------------------------------------------------------------------------
# The sources
# --------------------------------------------------------------------------
#: DOB Violations. `bin` and `issue_date` are TEXT columns, and the date reads
#: `YYYYMMDD`, thus a string comparison filters the window on the server.
DOB_VIOLATIONS = "3h2n-5cm9"

#: DOB ECB Violations. The same key and the same date shape.
ECB_VIOLATIONS = "6bgk-3dad"

#: DOB Permit Issuance. The BIN column is `bin__`.
#:
#: Warning: `issuance_date` holds 2 different shapes — `MM/DD/YYYY` for most
#: rows and an ISO timestamp for a few. Thus the server CANNOT filter the
#: window, and this module filters the year in pandas after the pull.
PERMIT_ISSUANCE = "ipu4-2q9a"

#: DOB NOW: Safety — Facades Compliance Filings (Local Law 11 / FISP).
FACADE_FILINGS = "xubg-57si"

#: The DOB job types that mean work on an existing building.
#:
#: | Code | Work |
#: |------|------|
#: | A1 | An alteration that changes use, egress, or occupancy |
#: | A2 | An alteration with no change of use, egress, or occupancy |
#: | A3 | A minor alteration |
#:
#: `NB` (new building) and `DM` (demolition) stay out. A new building is not
#: an investment in the building that the model looked at, and a demolition is
#: the end of one.
ALTERATION_JOB_TYPES = ("A1", "A2", "A3")


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------
def _cache_path(rid: str, params: Dict[str, str]) -> Path:
    """Name a cache file from a hash of the WHOLE query.

    Warning: do not shorten a query to make a name. A BIN `IN` list differs
    from the next batch only after thousands of characters, thus a truncated
    name lets batch 2 read batch 1's rows and the counts come out wrong.
    """
    blob = json.dumps({"rid": rid, **params}, sort_keys=True)
    return CACHE_DIR / f"bld_{rid}_{hashlib.sha256(blob.encode()).hexdigest()[:32]}.parquet"


def fetch(rid: str, select: Optional[str] = None, where: Optional[str] = None,
          group: Optional[str] = None, order: Optional[str] = None,
          use_cache: bool = True, max_rows: int = 2_000_000) -> pd.DataFrame:
    """Get the rows of 1 SODA query, with paging and a disk cache.

    This follows the `_proxies.fetch` pattern. It reads only, it never writes
    to a city dataset, and it puts the credential in the Authorization header
    and never in the URL.

    Warning: this raises an error if the query reaches `max_rows`. A silent cut
    would give a wrong count that looks correct.
    """
    params: Dict[str, str] = {}
    if select:
        params["$select"] = select
    if where:
        params["$where"] = where
    if group:
        params["$group"] = group
    if order:
        params["$order"] = order

    CACHE_DIR.mkdir(exist_ok=True)
    cache = _cache_path(rid, params)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    headers = px._auth_header()
    frames: List[pd.DataFrame] = []
    offset = 0
    truncated = True
    while offset < max_rows:
        page = dict(params)
        page["$limit"] = str(SODA_PAGE)
        page["$offset"] = str(offset)
        url = (f"https://data.cityofnewyork.us/resource/{rid}.json?"
               + urllib.parse.urlencode(page))
        req = urllib.request.Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as exc:
            # The credential lives in the header, not in the URL, thus the
            # query text is safe to show. Never add the header here.
            detail = ""
            body = getattr(exc, "read", None)
            if callable(body):
                try:
                    detail = body().decode(errors="replace")[:300]
                except Exception:
                    detail = ""
            raise RuntimeError(
                f"SODA request failed for resource {rid}: {type(exc).__name__}\n"
                f"  select: {select}\n  group: {group}\n"
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
            f"incomplete, so do not use it. Make the query smaller.")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df.to_parquet(cache, index=False)
    return df


def _in_list(values: Sequence[str]) -> str:
    """Build a SoQL `IN` list. Every value here is a 7-digit BIN."""
    safe = [str(v).replace("'", "") for v in values]
    return ",".join(f"'{v}'" for v in safe)


def fetch_by_bin(rid: str, bins: Sequence[str], select: str,
                 bin_field: str = "bin", where_extra: Optional[str] = None,
                 group: Optional[str] = None,
                 batch: int = BIN_BATCH) -> pd.DataFrame:
    """Read 1 source for a named BIN set only, in batches.

    Warning: the server must do the BIN filter. DOB Violations holds more than
    2,000,000 rows and DOB Permit Issuance holds more than 4,000,000. A whole
    table fetch would reach the row cap and give an incomplete answer.

    Args:
        bins: The BINs of one case. `facility_bins` builds this list.
        bin_field: `bin` for the violation sources, `bin__` for the permits.
        where_extra: More SoQL that this function joins with `AND`.
        group: A `$group` clause. Prefer one — a grouped query moves a small
            table instead of every row.
    """
    keep = sorted({str(b) for b in bins if str(b) and str(b) != "nan"})
    frames: List[pd.DataFrame] = []
    for i in range(0, len(keep), batch):
        where = f"{bin_field} in ({_in_list(keep[i:i + batch])})"
        if where_extra:
            where = f"{where} AND {where_extra}"
        part = fetch(rid, select=select, where=where, group=group)
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# The unit set
# --------------------------------------------------------------------------
def facility_bins(facilities: pd.DataFrame) -> List[str]:
    """The usable BINs of a FacDB unit table, with no repeat.

    `_proxies.clean_bin` states the rule: 7 digits, a borough code of 1 to 5,
    and not a borough placeholder such as 1000000.
    """
    return sorted(px.clean_bin(facilities["bin"]).dropna().unique().tolist())


def bin_coverage(facilities: pd.DataFrame) -> Dict[str, Any]:
    """How much of a unit set carries a usable BIN.

    Read this before a correlation. A case that reaches 96% of its units
    describes that 96%, and the notebook must say so.
    """
    clean = px.clean_bin(facilities["bin"])
    return {
        "units": int(len(facilities)),
        "units_with_bin": int(clean.notna().sum()),
        "bin_rate": float(clean.notna().mean()),
        "distinct_bins": int(clean.dropna().nunique()),
    }


# --------------------------------------------------------------------------
# DOB violations
# --------------------------------------------------------------------------
#: A DOB violation is OPEN when its category ends in `ACTIVE`. The source
#: writes `V-DOB VIOLATION - ACTIVE` for an open one and `V*-DOB VIOLATION -
#: Resolved` or `V*-DOB VIOLATION - DISMISSED` for a closed one. The star marks
#: the closed rows, but the word is the safer test: a category can name a
#: different violation family, such as `VW-VIOLATION WORK WITHOUT PERMIT`, and
#: those families use the same 2 words.
_DOB_OPEN = re.compile(r"ACTIVE\s*$", re.I)


def dob_violations_by_bin(bins: Sequence[str],
                          since_year: int = WINDOW_START_YEAR) -> pd.DataFrame:
    """Count the DOB violations of each BIN, open and total.

    The server groups the rows, thus this moves 1 row for each BIN and
    category and never the violations themselves.

    Returns `bin`, `dob_open`, and `dob_total`. A BIN with no violation is
    absent; `attach_counts` fills it with 0.
    """
    raw = fetch_by_bin(
        DOB_VIOLATIONS, bins,
        select="bin,violation_category,count(*) as n",
        bin_field="bin",
        where_extra=f"issue_date >= '{int(since_year)}0101'",
        group="bin,violation_category",
    )
    if raw.empty:
        return pd.DataFrame(columns=["bin", "dob_open", "dob_total"])
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce").fillna(0)
    raw["is_open"] = raw["violation_category"].astype(str).map(
        lambda s: bool(_DOB_OPEN.search(s)))
    out = raw.groupby("bin").apply(
        lambda g: pd.Series({"dob_open": g.loc[g.is_open, "n"].sum(),
                             "dob_total": g["n"].sum()}),
        include_groups=False).reset_index()
    return out


# --------------------------------------------------------------------------
# ECB violations
# --------------------------------------------------------------------------
def ecb_violations_by_bin(bins: Sequence[str],
                          since_year: int = WINDOW_START_YEAR) -> pd.DataFrame:
    """Count the ECB violations of each BIN, open and total.

    An ECB violation carries a fine, thus it marks a fault that the city took
    to a hearing. `ecb_violation_status` is `ACTIVE` or `RESOLVE`.

    Returns `bin`, `ecb_open`, and `ecb_total`.
    """
    raw = fetch_by_bin(
        ECB_VIOLATIONS, bins,
        select="bin,ecb_violation_status,count(*) as n",
        bin_field="bin",
        where_extra=f"issue_date >= '{int(since_year)}0101'",
        group="bin,ecb_violation_status",
    )
    if raw.empty:
        return pd.DataFrame(columns=["bin", "ecb_open", "ecb_total"])
    raw["n"] = pd.to_numeric(raw["n"], errors="coerce").fillna(0)
    raw["is_open"] = raw["ecb_violation_status"].astype(str).str.upper().eq("ACTIVE")
    out = raw.groupby("bin").apply(
        lambda g: pd.Series({"ecb_open": g.loc[g.is_open, "n"].sum(),
                             "ecb_total": g["n"].sum()}),
        include_groups=False).reset_index()
    return out


# --------------------------------------------------------------------------
# Alteration permits
# --------------------------------------------------------------------------
def alteration_permits_by_bin(bins: Sequence[str],
                              since_year: int = WINDOW_START_YEAR
                              ) -> pd.DataFrame:
    """Count the distinct alteration JOBS of each BIN inside the window.

    A permit is a positive measure: it says that somebody spent money on this
    building. The violations say the opposite, thus the 2 read together tell
    whether a building is neglected or maintained.

    The count is over the DOB job number, not over the permit rows. One job
    takes several permits — an initial permit, a renewal, and 1 permit for each
    trade — and a count of rows would call a long job a busy building.

    Warning: `issuance_date` holds 2 shapes, thus the server cannot filter the
    year. The BIN list and the job type filter on the server, and this
    function filters the year in pandas.

    Returns `bin` and `permit_jobs`.
    """
    types = ",".join(f"'{t}'" for t in ALTERATION_JOB_TYPES)
    raw = fetch_by_bin(
        PERMIT_ISSUANCE, bins,
        select="bin__,job__,issuance_date",
        bin_field="bin__",
        where_extra=f"job_type in ({types}) AND permit_status='ISSUED'",
    )
    if raw.empty:
        return pd.DataFrame(columns=["bin", "permit_jobs"])
    year = pd.to_datetime(raw["issuance_date"], errors="coerce",
                          format="mixed").dt.year
    raw = raw.assign(year=year)
    raw = raw[raw["year"] >= int(since_year)]
    if raw.empty:
        return pd.DataFrame(columns=["bin", "permit_jobs"])
    out = (raw.groupby("bin__")["job__"].nunique()
              .rename("permit_jobs").reset_index()
              .rename(columns={"bin__": "bin"}))
    return out


# --------------------------------------------------------------------------
# Facade filings (optional)
# --------------------------------------------------------------------------
#: The FISP status of a facade, as a number that points "higher is better".
#:
#: | Status | Meaning | Value |
#: |--------|---------|-------|
#: | SAFE | No repair is due | 1.0 |
#: | SWARMP | Safe with a repair programme | 0.5 |
#: | UNSAFE | A hazard the owner must fix now | 0.0 |
FACADE_STATUS_VALUE = {"SAFE": 1.0, "SWARMP": 0.5, "UNSAFE": 0.0}


def facade_status_by_bin(bins: Sequence[str],
                         since_year: int = WINDOW_START_YEAR) -> pd.DataFrame:
    """Read the newest FISP facade status of each BIN.

    **Warning: this covers few units, and the law says why.** Local Law 11
    applies to a building of more than 6 storeys. A branch library and a
    school are mostly low buildings, thus about 12% of the library BINs and
    about 8% of the school BINs carry any filing at all. Measured 2026-08-31.

    This module ships the builder and exports NO proxy row from it. A
    correlation over 8% of a case would describe the tall schools alone, and
    the export must not let a reader mistake that for the case.

    Returns `bin`, `facade_value`, and `n_filings`.
    """
    raw = fetch_by_bin(
        FACADE_FILINGS, bins,
        select="bin,current_status,filing_date",
        bin_field="bin",
        where_extra=f"filing_date >= '{int(since_year)}-01-01T00:00:00'",
    )
    if raw.empty:
        return pd.DataFrame(columns=["bin", "facade_value", "n_filings"])
    raw["value"] = (raw["current_status"].astype(str).str.upper()
                    .map(FACADE_STATUS_VALUE))
    raw = raw.dropna(subset=["value"])
    raw["filing_date"] = pd.to_datetime(raw["filing_date"], errors="coerce",
                                        format="mixed")
    counts = raw.groupby("bin").size().rename("n_filings")
    newest = (raw.sort_values("filing_date").groupby("bin")["value"].last()
                 .rename("facade_value"))
    return pd.concat([newest, counts], axis=1).reset_index()


# --------------------------------------------------------------------------
# The metric table
# --------------------------------------------------------------------------
#: Every metric this module exports, with its sign and its label.
#:
#: Warning: read the sign before you read a correlation. A violation count
#: rises when a building gets worse, thus it carries a sign of -1 and leaves
#: this module oriented "higher is better". A positive correlation then always
#: means that the model and the record agree.
BUILDING_METRICS: Dict[str, Dict[str, Any]] = {
    "building_violations_open": {
        "column": "violations_open",
        "sign": -1,
        "source": "3h2n-5cm9 + 6bgk-3dad",
        "metric": "open DOB and ECB violations of the BIN",
        "label": "Open building violations (negated)",
        "note": "A fault the city recorded and nobody closed.",
    },
    "building_violations_total": {
        "column": "violations_total",
        "sign": -1,
        "source": "3h2n-5cm9 + 6bgk-3dad",
        "metric": "all DOB and ECB violations of the BIN",
        "label": "Building violations (negated)",
        "note": "Every fault since the window, open or closed.",
    },
    "alteration_permits": {
        "column": "permit_jobs",
        "sign": 1,
        "source": "ipu4-2q9a",
        "metric": "distinct A1/A2/A3 jobs with a permit issued to the BIN",
        "label": "Alteration permits",
        "note": "Money spent on the building. Higher is more investment.",
    },
}


def building_record(bins: Sequence[str],
                    since_year: int = WINDOW_START_YEAR) -> pd.DataFrame:
    """The whole building record of a BIN set, 1 row for each BIN.

    A BIN with no violation and no permit takes 0, not NaN. An absent row in
    the city record means "no fault was recorded", which is a measurement and
    not a gap. Compare that with a construction year, where an absent row
    means the city holds no year and the unit must drop.

    Returns `bin` and the columns of `BUILDING_METRICS`, plus the parts.
    """
    keep = sorted({str(b) for b in bins})
    out = pd.DataFrame({"bin": keep})
    dob = dob_violations_by_bin(keep, since_year)
    ecb = ecb_violations_by_bin(keep, since_year)
    per = alteration_permits_by_bin(keep, since_year)
    for part in (dob, ecb, per):
        if not part.empty:
            out = out.merge(part, on="bin", how="left")
    for col in ("dob_open", "dob_total", "ecb_open", "ecb_total",
                "permit_jobs"):
        if col not in out.columns:
            out[col] = 0.0
    out[["dob_open", "dob_total", "ecb_open", "ecb_total", "permit_jobs"]] = (
        out[["dob_open", "dob_total", "ecb_open", "ecb_total", "permit_jobs"]]
        .fillna(0.0).astype(float))
    out["violations_open"] = out["dob_open"] + out["ecb_open"]
    out["violations_total"] = out["dob_total"] + out["ecb_total"]
    return out


def building_condition(facilities: pd.DataFrame, metric: str,
                       since_year: int = WINDOW_START_YEAR,
                       record: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Give each facility 1 building-condition value, oriented higher-better.

    FacDB carries the BIN, thus this is a key join and not a spatial one. The
    unit and the building are the same object here.

    Warning: 2 units can share 1 BIN — a school campus holds several DOE
    organizations in 1 building. Both units then take the SAME value, which is
    correct: the record grades the building, and the 2 units are that
    building. `unit_proxy_by_bin` states the count.

    Returns `unit_uid`, `proxy_value`, `bin`, `latitude`, and `longitude`.

    Args:
        record: A frame from `building_record`. Pass one to build several
            metrics from a single pull.
    """
    if metric not in BUILDING_METRICS:
        raise KeyError(f"unknown metric {metric!r}; "
                       f"use one of {list(BUILDING_METRICS)}")
    spec = BUILDING_METRICS[metric]

    f = facilities.copy()
    f["bin_clean"] = px.clean_bin(f["bin"])
    f = f.dropna(subset=["bin_clean"])
    rec = (building_record(facility_bins(facilities), since_year)
           if record is None else record)

    j = f.merge(rec, left_on="bin_clean", right_on="bin", how="inner",
                suffixes=("", "_rec"))
    j["proxy_value"] = j[spec["column"]].astype(float) * spec["sign"]
    out = j[["uid", "proxy_value", "bin_clean", "latitude", "longitude"]]
    out = out.rename(columns={"uid": "unit_uid", "bin_clean": "bin"})
    out = out.dropna(subset=["proxy_value"])
    out.attrs["metric"] = metric
    out.attrs["label"] = spec["label"]
    out.attrs["higher_is_better"] = True
    return out.reset_index(drop=True)


def unit_proxy_by_bin(facilities: pd.DataFrame, metric: str,
                      since_year: int = WINDOW_START_YEAR,
                      record: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The `proxy_key` / `proxy_value` frame that a unit correlation takes.

    `_unit_validation.correlate_units` and `_proxies.correlate_units` both read
    a frame of this shape. The key is `unit_uid`, because FacDB already put the
    building record on the unit.
    """
    unit = building_condition(facilities, metric, since_year, record)
    return (unit.rename(columns={"unit_uid": "proxy_key"})
                [["proxy_key", "proxy_value"]]
                .drop_duplicates("proxy_key").reset_index(drop=True))


def condition_by_layer(unit_values: pd.DataFrame, layer: str,
                       min_units: int = 3) -> pd.DataFrame:
    """Aggregate the unit values into 1 geography layer.

    The value of a polygon is the MEAN over its units, and `n_units_proxy`
    counts them. This uses the same layers and the same floor as the model
    scores, thus the 2 tables line up for a correlation.
    """
    if unit_values.empty:
        return pd.DataFrame()
    pts = px.points_from_latlon(
        unit_values.rename(columns={"unit_uid": "proxy_key"}))
    return px.aggregate_proxy(pts, layer, min_units=min_units)


def record_coverage(facilities: pd.DataFrame,
                    record: pd.DataFrame) -> pd.DataFrame:
    """Report how much of a unit set the building record reaches.

    | Column | Holds |
    |--------|-------|
    | `units` | Every unit of the case |
    | `with_bin` | Units that carry a usable BIN |
    | `with_record` | Units whose BIN reached the record table |
    | `with_violation` | Units that carry at least 1 violation |
    | `with_permit` | Units that carry at least 1 alteration job |
    """
    f = facilities.copy()
    f["bin_clean"] = px.clean_bin(f["bin"])
    rec = record.set_index("bin")
    has = f["bin_clean"].isin(rec.index)
    viol = f["bin_clean"].map(rec["violations_total"]).fillna(0) > 0
    perm = f["bin_clean"].map(rec["permit_jobs"]).fillna(0) > 0
    return pd.DataFrame([{
        "units": int(len(f)),
        "with_bin": int(f["bin_clean"].notna().sum()),
        "with_record": int(has.sum()),
        "with_violation": int(viol.sum()),
        "with_permit": int(perm.sum()),
        "bin_rate": round(float(f["bin_clean"].notna().mean()), 4),
        "violation_rate": round(float(viol.mean()), 4),
        "permit_rate": round(float(perm.mean()), 4),
    }])


# ==========================================================================
# The schools construct metrics
# ==========================================================================
# The 2 frames below are keyed by DBN and not by BIN, thus they reach the
# schools case only. They live here because the first of them measures the
# BUILDING, which is what this module is for, and because the second is the
# control that belongs beside it.
#
# | Frame | Measures | Key |
# |-------|----------|-----|
# | `school_cleanliness` | Is the school kept clean | DBN |
# | `school_economic_need` | The economic need of the students | DBN |
# ==========================================================================

#: 2017-18 NYC School Survey, the teacher file. The asset is an `.xlsx` blob
#: and not a table, thus a SODA query cannot reach it and this module reads
#: the workbook.
SURVEY_TEACHER_2018 = "sjxh-har6"

#: The survey item that measures the building. The teacher answers on a
#: 5-point agreement scale, and the workbook reports the share that agrees.
SURVEY_CLEAN_ITEM = "6f. This school is kept clean."

#: The workbook sheet that holds the percent-positive columns.
SURVEY_SHEET = "Teacher Pos & Neg %"

SURVEY_DIR = EXTERNAL_DIR / "nyc_school_survey_2018"

#: 2017-18 to 2021-22 Demographic Snapshot. It carries the Economic Need
#: Index of each school and each year.
DEMOGRAPHIC_SNAPSHOT = "c7ru-d68s"

#: The newest year the snapshot holds.
ENI_YEAR = "2021-22"


def _download_survey_workbook(use_cache: bool = True) -> Path:
    """Get the 2017-18 teacher survey workbook and keep it on disk.

    The file is about 15 MB. It lands in `data/external/nyc_school_survey_2018/`
    and stays, thus the notebook reads the network 1 time.
    """
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    path = SURVEY_DIR / "nyc_school_survey_2018_teacher.xlsx"
    if use_cache and path.exists() and path.stat().st_size > 1_000_000:
        return path
    meta_url = f"https://data.cityofnewyork.us/api/views/{SURVEY_TEACHER_2018}.json"
    req = urllib.request.Request(meta_url)
    for k, v in px._auth_header().items():
        req.add_header(k, v)
    meta = json.load(urllib.request.urlopen(req, timeout=180))
    blob_id = meta.get("blobId")
    if not blob_id:
        raise RuntimeError(
            f"{SURVEY_TEACHER_2018} names no blob. The city may have changed "
            f"the asset. Read the dataset page before you change this code.")
    url = (f"https://data.cityofnewyork.us/api/views/{SURVEY_TEACHER_2018}"
           f"/files/{blob_id}?download=true&filename=survey.xlsx")
    req = urllib.request.Request(url)
    for k, v in px._auth_header().items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=600) as src:
        path.write_bytes(src.read())
    return path


_XLSX_CELL = re.compile(r'<c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(.*?)</c>)', re.S)


def _xlsx_rows(book: zipfile.ZipFile, sheet_path: str,
               shared: List[str]) -> Iterable[Dict[str, str]]:
    """Walk the rows of 1 worksheet as dicts of column letter to value.

    Warning: this reads the XML directly. `pandas.read_excel` needs
    `openpyxl`, which the project environments do not carry, and the workbook
    is a simple grid with no formula and no style that matters.
    """
    import html as _html

    xml = book.read(sheet_path).decode("utf-8", errors="replace")
    for rm in re.finditer(r"<row[^>]*>(.*?)</row>", xml, re.S):
        cells: Dict[str, str] = {}
        for cm in _XLSX_CELL.finditer(rm.group(1)):
            ref, attrs, body = cm.group(1), cm.group(3), cm.group(4) or ""
            v = re.search(r"<v>(.*?)</v>", body, re.S)
            if not v:
                continue
            val = v.group(1)
            if 't="s"' in attrs:
                val = shared[int(val)]
            cells[ref] = _html.unescape(val)
        yield cells


def _next_column(ref: str) -> str:
    """The column letter that follows `ref`: A to B, Z to AA."""
    s = list(ref)
    i = len(s) - 1
    while i >= 0:
        if s[i] != "Z":
            s[i] = chr(ord(s[i]) + 1)
            return "".join(s)
        s[i] = "A"
        i -= 1
    return "A" + "".join(s)


def school_cleanliness(use_cache: bool = True) -> pd.DataFrame:
    """Read the share of teachers who agree that the school is kept clean.

    This is the ONE building-condition measurement that a city school source
    holds. The teacher answers item `6f. This school is kept clean.`, and the
    workbook gives the share that agrees or strongly agrees.

    **Warning: the newest such file is the 2017-18 survey.** The city stopped
    publishing the item-level survey after that year, and the 2019 file holds
    only the 6 framework scores. Thus this proxy sits about 7 years before the
    model run, and it describes a building that may have changed. State the
    year in the paper. Read it as a building measure with a stale window, not
    as a measure of 2026.

    Returns `dbn`, `proxy_value` (0 to 100, higher is cleaner), and
    `school_name`.
    """
    import html as _html

    path = _download_survey_workbook(use_cache=use_cache)
    with zipfile.ZipFile(path) as book:
        wb = book.read("xl/workbook.xml").decode("utf-8", errors="replace")
        rels = book.read("xl/_rels/workbook.xml.rels").decode(
            "utf-8", errors="replace")
        sheets = re.findall(r'<sheet name="(.*?)"[^>]*r:id="(rId\d+)"', wb)
        want = None
        for name, rid in sheets:
            if _html.unescape(name) == SURVEY_SHEET:
                want = rid
        if want is None:
            raise RuntimeError(
                f"the workbook holds no sheet {SURVEY_SHEET!r}; "
                f"it holds {[_html.unescape(n) for n, _ in sheets]}")
        target = re.search(rf'Id="{want}"[^>]*Target="(.*?)"', rels)
        sheet_path = "xl/" + target.group(1).lstrip("/")

        ss = book.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
        shared = [_html.unescape(re.sub(r"<.*?>", "", s))
                  for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]

        rows = list(_xlsx_rows(book, sheet_path, shared))

    if len(rows) < 4:
        raise RuntimeError("the survey sheet holds no data rows")
    header, labels = rows[0], rows[1]
    hit = [ref for ref, text in header.items()
           if SURVEY_CLEAN_ITEM.lower() in str(text).lower()]
    if not hit:
        raise RuntimeError(
            f"the survey sheet names no item {SURVEY_CLEAN_ITEM!r}. "
            f"The city may have changed the workbook.")
    first = hit[0]
    # An item owns 2 columns: the share that disagrees and the share that
    # agrees. Take the agree side by its own label, never by position.
    pair = [first, _next_column(first)]
    positive = [c for c in pair
                if "agree" in str(labels.get(c, "")).lower()
                and "disagree" not in str(labels.get(c, "")).lower()]
    if not positive:
        raise RuntimeError(
            f"neither column of {SURVEY_CLEAN_ITEM!r} carries an agree label; "
            f"labels are {[labels.get(c) for c in pair]}")
    col = positive[0]

    out = []
    for r in rows[2:]:
        dbn = str(r.get("A", "")).strip()
        if not re.match(r"^\d{2}[A-Z]\d{3}$", dbn):
            continue
        val = pd.to_numeric(r.get(col), errors="coerce")
        if pd.isna(val):
            continue
        out.append({"dbn": dbn, "school_name": r.get("B", ""),
                    "proxy_value": float(val)})
    df = pd.DataFrame(out).drop_duplicates("dbn").reset_index(drop=True)
    df.attrs["item"] = SURVEY_CLEAN_ITEM
    df.attrs["source"] = SURVEY_TEACHER_2018
    return df


def school_economic_need(year: str = ENI_YEAR,
                         use_cache: bool = True) -> pd.DataFrame:
    """Read the Economic Need Index of each school, oriented higher-better.

    The source writes the index as a percent string, such as `87.9%`, and the
    index rises with need, thus a HIGH raw value marks a POOR student body.
    This function strips the percent sign and multiplies by -1, so the value
    leaves this module oriented the same way as household income: higher means
    a wealthier school community.

    Why it beats the tract income for this case: a school draws its students
    from a zone, not from its own block, and a charter or a specialized school
    draws from the whole city. The index measures the students who actually
    attend, thus it controls the thing the tract income only approximates.

    Warning: this is a CONTROL, not a building measure. A correlation here says
    that the model calls a building in a wealthier school community better. It
    says nothing about upkeep.

    Returns `dbn` and `proxy_value`.
    """
    df = fetch(DEMOGRAPHIC_SNAPSHOT,
               select="dbn,year,economic_need_index",
               where=f"year='{year}'", use_cache=use_cache)
    if df.empty:
        raise RuntimeError(
            f"the demographic snapshot holds no year {year!r}. "
            f"Read the dataset before you change ENI_YEAR.")
    raw = df["economic_need_index"].astype(str).str.replace("%", "", regex=False)
    val = pd.to_numeric(raw, errors="coerce")
    out = pd.DataFrame({"dbn": df["dbn"].astype(str).str.strip(),
                        "proxy_value": -val})
    out = out.dropna(subset=["proxy_value"]).drop_duplicates("dbn")
    if out.empty:
        raise RuntimeError(
            "the demographic snapshot gave rows but no readable index. The "
            "column shape may have changed; read a sample before you edit.")
    return out.reset_index(drop=True)


def school_unit_key(points, facilities: pd.DataFrame,
                    max_distance_ft: float = 200.0) -> pd.DataFrame:
    """Map each FacDB school unit to its nearest DBN point.

    The DBN proxies (`school_kept_clean`, `school_economic_need`) name a DBN,
    and FacDB carries no DBN. This is the ONE join rule for them: each unit
    takes the value of the nearest DBN point inside `max_distance_ft`, in the
    working CRS. A private or a postgraduate school has no DBN point near it
    and drops.

    Warning: a campus can hold several DBNs in 1 building. The nearest point
    wins. State the rule when you quote a number, because a different rule
    moves n by about 5%.

    Returns `unit_uid`, `proxy_key` (the DBN), and `distance_ft`.
    """
    import geopandas as gpd

    import _geography as geo

    fpts = gpd.GeoDataFrame(
        facilities[["uid"]],
        geometry=gpd.points_from_xy(facilities["longitude"],
                                    facilities["latitude"]),
        crs="EPSG:4326",
    ).to_crs(geo.WORKING_CRS)
    near = gpd.sjoin_nearest(fpts, points[["proxy_key", "geometry"]],
                             max_distance=float(max_distance_ft),
                             distance_col="distance_ft")
    near = (near.sort_values("distance_ft").drop_duplicates("uid")
                .rename(columns={"uid": "unit_uid"}))
    return near[["unit_uid", "proxy_key", "distance_ft"]].reset_index(drop=True)


def unit_correlation(unit_scores: pd.DataFrame, unit_proxy: pd.DataFrame,
                     n_boot: int = 400, seed: int = 20260831
                     ) -> Dict[str, Any]:
    """Correlate the model score of each unit with 1 building proxy.

    The interval is a bootstrap over UNITS: a resample draws a unit index and
    reads the score and the proxy at it, thus the pairing holds.

    Args:
        unit_scores: `unit_uid` and `mean_score`, from `score_units`.
        unit_proxy: `unit_uid` and `proxy_value`, oriented higher-better.

    Returns n, `spearman_rho` with `rho_lo`/`rho_hi`, `kendall_tau`, and the
    Spearman `p`.
    """
    import numpy as np
    from scipy import stats

    j = unit_scores.merge(unit_proxy[["unit_uid", "proxy_value"]],
                          on="unit_uid", how="inner")
    out: Dict[str, Any] = {"n": int(len(j))}
    if len(j) < 30:
        out.update(spearman_rho=float("nan"), rho_lo=float("nan"),
                   rho_hi=float("nan"), kendall_tau=float("nan"),
                   p=float("nan"))
        return out
    x = j["mean_score"].to_numpy(dtype=float)
    y = j["proxy_value"].to_numpy(dtype=float)
    rho, p = stats.spearmanr(x, y)
    out["spearman_rho"] = float(rho)
    out["p"] = float(p)
    out["kendall_tau"] = float(stats.kendalltau(x, y, variant="b").statistic)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(j), size=(n_boot, len(j)))
    b = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx])
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    out["rho_lo"], out["rho_hi"] = float(lo), float(hi)
    return out


#: The schools metrics that this module builds, with their provenance.
SCHOOL_METRICS: Dict[str, Dict[str, Any]] = {
    "school_kept_clean": {
        "source": SURVEY_TEACHER_2018,
        "metric": SURVEY_CLEAN_ITEM,
        "window": "2017-18 school year",
        "label": "School kept clean (teacher survey)",
        "sign": 1,
        "note": "The newest item-level NYC School Survey. The city stopped "
                "publishing the item file after 2017-18.",
    },
    "school_economic_need": {
        "source": DEMOGRAPHIC_SNAPSHOT,
        "metric": "economic_need_index",
        "window": f"{ENI_YEAR} school year",
        "label": "School economic need index (negated)",
        "sign": -1,
        "note": "A control, not a building measure. The source writes a "
                "percent, and this module negates it, thus higher means a "
                "wealthier school community.",
    },
}
