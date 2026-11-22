"""The restaurants proxies, second generation.

Why this module exists
----------------------
`_proxies.restaurant_proxy` reads `last_score`: the violation points of the
ONE most recent inspection. That is a single draw of a noisy measurement. The
audit of 2026-08-31 measured the unit-level agreement of that proxy at
rho = -0.05, and a one-shot proxy cannot separate a weak signal from a noisy
ruler.

This module adds the history metrics and the outdoor-dining channel. It does
not change `_proxies`. It delegates the 2 old metrics straight to
`_proxies.restaurant_proxy`, so an old export row comes back bit for bit.

The 4 metrics
-------------
| Name | What it measures | Sign |
|------|------------------|------|
| `inspection_score_history` | The mean score over the cycle initial inspections | -1 |
| `inspection_score` | The score of the last inspection | -1 |
| `grade_a_rate` | The share of inspections that earned grade A | +1 |
| `outdoor_dining` | The unit holds an outdoor-dining license | +1 |

Warning: read the orientation before you read a correlation. A DOHMH score
counts violation points, thus a LOW raw score is a CLEAN restaurant. Each
metric states a `sign`, and this module multiplies the raw value by it. Every
value leaves here oriented "higher is better", so a positive correlation always
means that the model and the record agree. Do not flip a sign again.

Why the history mean restricts the inspection type
--------------------------------------------------
The city gives a re-inspection ONLY after an initial inspection fails. A
re-inspection score is thus conditional on an earlier failure, and a mean over
mixed types compares restaurants on different conditions. The history mean
reads `Cycle Inspection / Initial Inspection` alone: the routine visit that
every operating restaurant receives on a cycle.

`sv.nyc.curation.dohmh.history` writes the sidecar that holds the
numbers. It is keyed by `camis` and it adds no column to
`restaurants_aggregated.parquet`.

Why outdoor dining is a CONFOUNDER and not a validation
--------------------------------------------------------
The prompt asks which restaurant a person would rather eat at. That is a
preference, not a safety measure. Outdoor seating is visible from the street,
thus the model can see it, and it plausibly drives the preference directly.
A correlation with this channel therefore does NOT validate the model against
an outside record. It measures how much of the judgment one visible amenity
explains. State it that way. See `vlm-narratives-docs/restaurant-proxies.md`.

The unit layer
--------------
Restaurants is the case with no join work: `uid` in the unit table IS
`unit_uid` in the pairs manifest. `unit_correlations` reads the place scores
through `_unit_validation.place_scores`, which reads the canonical registry,
and it computes its own Spearman, Kendall, and bootstrap interval.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.0.0"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

CASE = "restaurants"

#: The DOHMH curation sub-dataset.
CURATION_ROOT = "curation/dohmh_restaurants_inspected_all"
UNIT_TABLE = "restaurants_aggregated.parquet"

#: The history sidecar that `dohmh.history.build_inspection_history` writes.
HISTORY_TABLE = "restaurants_inspection_history.parquet"

#: The Open Restaurants (Dining Out NYC) curation sub-dataset.
OPEN_RESTAURANTS_ROOT = "curation/open_restaurants_all"
OPEN_RESTAURANTS_TABLE = "open_restaurants.parquet"

#: The inspection floor of the OLD metrics. It counts every inspection type.
#: `restaurants_validation.py` version 1.1.0 used this value.
MIN_INSPECTIONS = 2

#: The floor on the count of CYCLE INITIAL inspections behind a history mean.
#: A mean over 1 inspection is the old one-shot measurement under a new name.
MIN_HISTORY_INSPECTIONS = 2

#: The bootstrap settings. They match `_unit_validation`.
BOOT_N = 400
BOOT_SEED = 20260831

#: The models of the paper, in the order the tables use.
MODELS: Sequence[str] = ("gemma-4-12b", "qwen3.5-9b")


# --------------------------------------------------------------------------
# The tables on disk
# --------------------------------------------------------------------------
def _path(root: str, filename: str) -> Path:
    p = Path(root)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p / filename


@lru_cache(maxsize=2)
def load_units() -> pd.DataFrame:
    """The one-row-per-CAMIS unit table of the DOHMH curation."""
    import _geography as geo

    return geo.load_facilities(CURATION_ROOT, filename=UNIT_TABLE)


@lru_cache(maxsize=2)
def load_history() -> pd.DataFrame:
    """The inspection-history sidecar, keyed by `camis`.

    Build it with::

        from sv.nyc.curation.dohmh import build_inspection_history
        build_inspection_history(
            "curation/dohmh_restaurants_inspected_all/restaurants.parquet")

    Warning: this needs polars. Use `.venv-nightly` or `.venv-vllm025cu129`.
    """
    path = _path(CURATION_ROOT, HISTORY_TABLE)
    if not path.exists():
        raise FileNotFoundError(
            f"no inspection history at {path} — build it with "
            "sv.nyc.curation.dohmh.build_inspection_history"
        )
    df = pd.read_parquet(path)
    df["camis"] = df["camis"].astype(str)
    return df


@lru_cache(maxsize=2)
def load_open_restaurants() -> pd.DataFrame:
    """The outdoor-dining licenses of the Open Restaurants curation."""
    path = _path(OPEN_RESTAURANTS_ROOT, OPEN_RESTAURANTS_TABLE)
    if not path.exists():
        raise FileNotFoundError(
            f"no open-restaurants table at {path} — build it with "
            "python -m sv.nyc.curation open-restaurants"
        )
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# The outdoor-dining join
# --------------------------------------------------------------------------
#: Words that a licence name carries and an inspection name does not, or the
#: reverse. The join removes them before it compares 2 names.
_FILLER = re.compile(
    r"\b(LLC|INC|INCORPORATED|CORP|CORPORATION|CO|LTD|LP|NYC|NY|THE|"
    r"RESTAURANT|CAFE|BAR|GRILL|PIZZERIA|DELI)\b"
)


def _norm(series: pd.Series) -> pd.Series:
    """Upper case, drop punctuation, and squeeze the spaces."""
    s = series.fillna("").astype(str).str.upper()
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _strip_filler(series: pd.Series) -> pd.Series:
    s = series.str.replace(_FILLER, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def open_restaurants_match(units: Optional[pd.DataFrame] = None,
                           licenses: Optional[pd.DataFrame] = None
                           ) -> pd.DataFrame:
    """Map each outdoor-dining license to a CAMIS.

    The license table carries no CAMIS. It carries a BIN, and the BIN of a
    restaurant is in the unit table, thus the join runs inside a building.

    | Rule | What it matches |
    |------|-----------------|
    | `bin_name` | The same BIN and the same trade name, after normalization |
    | `bin_only` | The same BIN, when that BIN holds exactly 1 restaurant |

    A BIN alone is not enough: one building holds up to 18 restaurants in this
    data. The name rule runs first, and `bin_only` catches the rest.

    Returns one row for each (license uid, camis) pair, with `match_rule`.
    """
    units = load_units() if units is None else units
    lic = load_open_restaurants() if licenses is None else licenses

    u = units.dropna(subset=["bin"])[["camis", "bin", "facname"]].copy()
    u["camis"] = u["camis"].astype(str)
    u["_name"] = _strip_filler(_norm(u["facname"]))

    lg = lic.dropna(subset=["bin"]).copy()
    lg["_trade"] = _strip_filler(_norm(lg["facname"]))
    lg["_assumed"] = _strip_filler(_norm(lg.get("assumed_name_s", pd.Series(dtype=str))))
    lg["_legal"] = _strip_filler(_norm(lg.get("business_legal_name", pd.Series(dtype=str))))

    hits: List[pd.DataFrame] = []
    for col in ("_trade", "_assumed", "_legal"):
        m = lg.merge(u, left_on=["bin", col], right_on=["bin", "_name"],
                     how="inner")
        if not m.empty:
            m = m[["uid", "camis", "license_type", "bin"]].copy()
            m["match_rule"] = "bin_name"
            hits.append(m)
    matched = (pd.concat(hits, ignore_index=True)
               if hits else pd.DataFrame(
                   columns=["uid", "camis", "license_type", "bin", "match_rule"]))
    matched = matched.drop_duplicates(["uid", "camis"])

    # The fallback. A BIN that holds exactly 1 restaurant leaves no doubt.
    per_bin = u.groupby("bin")["camis"].nunique()
    lone = set(per_bin[per_bin == 1].index)
    rest = lg[~lg["uid"].isin(set(matched["uid"]))]
    rest = rest[rest["bin"].isin(lone)]
    if not rest.empty:
        fb = rest.merge(u[u["bin"].isin(lone)][["bin", "camis"]],
                        on="bin", how="inner")
        fb = fb[["uid", "camis", "license_type", "bin"]].copy()
        fb["match_rule"] = "bin_only"
        matched = pd.concat([matched, fb], ignore_index=True)

    return matched.drop_duplicates(["uid", "camis"]).reset_index(drop=True)


def outdoor_dining_units(units: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The outdoor-dining flag of EVERY restaurant in the unit table.

    The frame holds one row for each CAMIS in the unit table, thus the mean of
    `has_outdoor_dining` over an area IS the share of the restaurants of that
    area that hold a license. A frame of the matched restaurants alone would
    give a mean of 1.0 in every area.

    | Column | Holds |
    |--------|-------|
    | `camis` | The restaurant id |
    | `has_outdoor_dining` | 1 when the restaurant holds a license, else 0 |
    | `n_licenses` | The count of licenses that reached this restaurant |
    | `has_sidewalk` | 1 when a Sidewalk license reached it |
    | `has_roadway` | 1 when a Roadway license reached it |
    """
    units = load_units() if units is None else units
    matched = open_restaurants_match(units)

    per_camis = matched.assign(
        _side=matched["license_type"].astype(str).str.upper().eq("SIDEWALK"),
        _road=matched["license_type"].astype(str).str.upper().eq("ROADWAY"),
    ).groupby("camis").agg(
        n_licenses=("uid", "nunique"),
        has_sidewalk=("_side", "max"),
        has_roadway=("_road", "max"),
    ).reset_index()

    out = pd.DataFrame({"camis": units["camis"].astype(str)})
    out = out.merge(per_camis, on="camis", how="left")
    out["n_licenses"] = out["n_licenses"].fillna(0).astype(int)
    for c in ("has_sidewalk", "has_roadway"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out["has_outdoor_dining"] = (out["n_licenses"] > 0).astype(int)
    return out


# --------------------------------------------------------------------------
# The metrics
# --------------------------------------------------------------------------
#: The window string that each metric writes into `proxy_window`.
#:
#: Warning: the 2 old strings must not change. `restaurants_proxy_v1.1.0`
#: carries them, and version 1.2.0 must repeat an old row exactly.
METRIC_WINDOW = {
    "inspection_score": "last inspection of each unit, to 2026-04-25",
    "grade_a_rate": "all inspections, 2007 to 2026-04-25",
    "inspection_score_history":
        "cycle initial inspections, 2019-01-05 to 2026-04-25",
    "outdoor_dining": "issued licenses, 2024-10-09 to 2026-05-29, pulled 2026-06-01",
}

#: The recipe index of the source that each metric reads.
METRIC_SOURCE_INDEX = {
    "inspection_score": "1",
    "grade_a_rate": "1",
    "inspection_score_history": "2",
    "outdoor_dining": "3",
}

METRICS: Dict[str, Dict[str, Any]] = {
    "inspection_score_history": {
        "sign": -1,
        "higher_is_better": False,
        "role": "headline",
        "label": "DOHMH mean inspection score, cycle initial (negated)",
        "note": (
            "The mean of the violation score over the cycle initial "
            "inspections of one restaurant. A re-inspection follows a failure, "
            "thus the mean excludes it. The floor is "
            f"{MIN_HISTORY_INSPECTIONS} inspections."
        ),
    },
    "inspection_score": {
        "sign": -1,
        "higher_is_better": False,
        "role": "context",
        "label": "DOHMH inspection score (violation points, negated)",
        "note": (
            "The score of the LAST inspection of one restaurant, whatever its "
            "type. This was the headline proxy up to export version 1.1.0. It "
            "stays as context, and the paper reads the history mean."
        ),
    },
    "grade_a_rate": {
        "sign": 1,
        "higher_is_better": True,
        "role": "context",
        "label": "Share of inspections that earned grade A",
        "note": "n_grade_a divided by n_inspections, over every inspection type.",
    },
    "outdoor_dining": {
        "sign": 1,
        "higher_is_better": True,
        "role": "confounder",
        "label": "Holds a sidewalk or roadway dining license",
        "note": (
            "1 when the restaurant holds an issued Dining Out NYC license. "
            "The area mean is the share of the restaurants of that area. This "
            "is a CONFOUNDER of the preference judgment, not a validation: "
            "outdoor seating is visible from the street."
        ),
    },
}

#: The order that a table lists the metrics in.
METRIC_ORDER: Sequence[str] = (
    "inspection_score_history",
    "inspection_score",
    "grade_a_rate",
    "outdoor_dining",
)


def restaurant_proxy(metric: str = "inspection_score_history",
                     units: Optional[pd.DataFrame] = None,
                     min_inspections: int = MIN_INSPECTIONS,
                     min_history: int = MIN_HISTORY_INSPECTIONS
                     ) -> pd.DataFrame:
    """Build one restaurants proxy, oriented "higher is better".

    The frame holds `proxy_key` and `proxy_value`, plus the position columns
    that `points_from_latlon` needs. `proxy_key` is the `uid` of the unit
    table, which IS `unit_uid` in the pairs manifest.

    Args:
        metric: One name of `METRICS`.
        min_inspections: The floor that the 2 OLD metrics apply. It counts
            every inspection type, and `_proxies.restaurant_proxy` owns it.
        min_history: The floor on the count of CYCLE INITIAL inspections that
            the history mean reads.

    Warning: `outdoor_dining` keeps EVERY restaurant of the unit table and
    applies no inspection floor. The 0 rows carry the information, thus a
    filter on the inspection record would change what the mean means.
    """
    import _proxies as px

    if metric not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; use one of {list(METRICS)}")

    # The 2 old metrics go to the old function unchanged. This is what keeps an
    # export row of version 1.1.0 identical in version 1.2.0.
    if metric in ("inspection_score", "grade_a_rate"):
        return px.restaurant_proxy(
            load_units() if units is None else units,
            metric=metric, min_inspections=min_inspections)

    spec = METRICS[metric]
    df = (load_units() if units is None else units).copy()
    df["camis"] = df["camis"].astype(str)

    if metric == "inspection_score_history":
        hist = load_history()
        df = df.merge(
            hist[["camis", "history_mean_score", "history_n_inspections"]],
            on="camis", how="left")
        n = pd.to_numeric(df["history_n_inspections"], errors="coerce").fillna(0)
        raw = pd.to_numeric(df["history_mean_score"], errors="coerce")
        raw = raw.where(n >= min_history)
    else:  # outdoor_dining
        flags = outdoor_dining_units(df)
        df = df.merge(flags, on="camis", how="left")
        raw = pd.to_numeric(df["has_outdoor_dining"], errors="coerce").fillna(0)

    df["proxy_value"] = raw * spec["sign"]
    df["proxy_key"] = df["uid"]

    out = df.dropna(subset=["proxy_value", "proxy_key"])
    keep = [c for c in ("proxy_key", "proxy_value", "camis", "facname",
                        "latitude", "longitude", "cuisine_description",
                        "history_n_inspections", "n_licenses")
            if c in out.columns]
    out = out[keep].copy()
    out.attrs["metric"] = metric
    out.attrs["label"] = spec["label"]
    out.attrs["higher_is_better"] = True  # after the sign flip
    return out


def coverage_table(min_inspections: int = MIN_INSPECTIONS,
                   min_history: int = MIN_HISTORY_INSPECTIONS) -> pd.DataFrame:
    """How many units each metric keeps, and the spread of its values."""
    units = load_units()
    rows = []
    for m in METRIC_ORDER:
        p = restaurant_proxy(m, units=units, min_inspections=min_inspections,
                             min_history=min_history)
        v = p["proxy_value"].to_numpy(dtype=float)
        rows.append({
            "metric": m,
            "role": METRICS[m]["role"],
            "n_units": len(p),
            "pct_of_table": round(100 * len(p) / len(units), 1),
            "mean": float(np.mean(v)) if len(v) else float("nan"),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
            "window": METRIC_WINDOW[m],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The unit layer
# --------------------------------------------------------------------------
def _rho_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Spearman rho, Kendall tau-b, and the p value of tau."""
    from scipy import stats

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(stats.spearmanr(x, y).statistic)
    tau = stats.kendalltau(x, y, variant="b")
    return rho, float(tau.statistic), float(tau.pvalue)


def _bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
               ) -> Dict[str, float]:
    """A 95% interval on rho and on tau, from a resample over UNITS.

    The unit is the sampling unit of this table, thus the resample draws units
    and never comparisons.
    """
    if len(x) < 3:
        return {k: float("nan")
                for k in ("rho_lo", "rho_hi", "tau_lo", "tau_hi")}
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot, dtype=float)
    taus = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        rhos[i], taus[i], _ = _rho_tau(x[ix], y[ix])
    lo_r, hi_r = np.nanquantile(rhos, [0.025, 0.975])
    lo_t, hi_t = np.nanquantile(taus, [0.025, 0.975])
    return {"rho_lo": float(lo_r), "rho_hi": float(hi_r),
            "tau_lo": float(lo_t), "tau_hi": float(hi_t)}


def unit_rows(models: Sequence[str] = MODELS,
              metrics: Sequence[str] = METRIC_ORDER,
              arm: str = "canonical",
              n_boot: int = BOOT_N,
              seed: int = BOOT_SEED,
              restrict_to: Optional[str] = None) -> pd.DataFrame:
    """The unit-level agreement of every (metric, model) cell.

    The place scores come from `_unit_validation.place_scores`, which reads the
    canonical registry. This function reads that module and writes nothing to
    it. The statistics below are its own.

    Args:
        restrict_to: The name of a metric. When set, every row runs on the
            units that THAT metric keeps, thus the n of each row is the same
            and a reader compares 2 rulers on one set of restaurants.
    """
    import _unit_validation as U

    units = load_units()
    proxies = {m: restaurant_proxy(m, units=units) for m in metrics}
    common: Optional[set] = None
    if restrict_to is not None:
        common = set(proxies[restrict_to]["proxy_key"].astype(str))

    rows: List[Dict[str, Any]] = []
    for model in models:
        scores = U.place_scores(CASE, model, arm)
        if scores is None or scores.empty:
            continue
        for m in metrics:
            p = proxies[m][["proxy_key", "proxy_value"]].copy()
            p["proxy_key"] = p["proxy_key"].astype(str)
            if common is not None:
                p = p[p["proxy_key"].isin(common)]
            j = scores.assign(proxy_key=scores["unit_uid"].astype(str)) \
                      .merge(p, on="proxy_key", how="inner") \
                      .dropna(subset=["mean_score", "proxy_value"])
            x = j["mean_score"].to_numpy(dtype=float)
            y = j["proxy_value"].to_numpy(dtype=float)
            rho, tau, tau_p = _rho_tau(x, y)
            row: Dict[str, Any] = {
                "case": CASE,
                "model": model,
                "arm": arm,
                "proxy": m,
                "role": METRICS[m]["role"],
                "n_units": int(len(j)),
                "n_scored": int(len(scores)),
                "matched_pct": (round(100 * len(j) / len(scores), 1)
                                if len(scores) else float("nan")),
                "median_comparisons": (float(j["n_comparisons"].median())
                                       if len(j) else float("nan")),
                "rho": rho,
                "tau": tau,
                "tau_p": tau_p,
                "proxy_sd": (float(np.std(y, ddof=1)) if len(y) > 1
                             else float("nan")),
                "restricted_to": restrict_to or "",
                "n_boot": int(n_boot),
            }
            row.update(_bootstrap(x, y, n_boot, seed))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    order = {m: i for i, m in enumerate(METRIC_ORDER)}
    return (pd.DataFrame(rows)
            .sort_values(["model", "proxy"],
                         key=lambda c: (c.map(order) if c.name == "proxy" else c))
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------
def source_entry(recipe: Sequence[Dict[str, Any]], metric: str
                 ) -> Dict[str, Any]:
    """The recipe source that one metric reads."""
    want = METRIC_SOURCE_INDEX[metric]
    for e in recipe:
        if str(e.get("index")) == want:
            return e
    raise KeyError(f"recipe holds no source {want!r} for metric {metric!r}")


def source_id(entry: Dict[str, Any]) -> str:
    """The dataset id of a recipe source, or the stem of its file."""
    return entry.get("resource_id") or Path(entry["file"]).stem


def stamp(frame: pd.DataFrame, metric: str, entry: Dict[str, Any]
          ) -> pd.DataFrame:
    """Add the proxy name and the 3 provenance columns to 1 frame.

    The restaurants naming contract: `proxy` IS the metric name. Warning:
    never write a fixed proxy name beside a metric that a reader chooses. A
    fixed name lets a run export one metric under the name of another.
    """
    out = frame.copy()
    out["proxy"] = metric
    out["proxy_metric"] = metric
    out["proxy_source"] = source_id(entry)
    out["proxy_window"] = METRIC_WINDOW[metric]
    return out


def area_export(recipe: Sequence[Dict[str, Any]],
                metrics: Sequence[str] = METRIC_ORDER,
                min_units: int = 3,
                units: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One long frame: each metric, aggregated into each geography layer.

    Every row carries `proxy`, `proxy_metric`, `proxy_source`, and
    `proxy_window`. The layers come from `_geography.LAYERS`, thus the frame
    lines up with the model-score export of the same case.
    """
    import _geography as geo
    import _proxies as px

    units = load_units() if units is None else units
    frames: List[pd.DataFrame] = []
    for m in metrics:
        pts = px.points_from_latlon(restaurant_proxy(m, units=units))
        entry = source_entry(recipe, m)
        for layer in geo.LAYERS:
            out = px.aggregate_proxy(pts, layer, min_units=min_units)
            if not out.empty:
                frames.append(stamp(out, m, entry))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame())
