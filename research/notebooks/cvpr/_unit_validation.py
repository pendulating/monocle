"""Validation by proxy at the UNIT layer, beside the area layer.

The master table validates the model against an outside measurement at the
community district layer. That layer is a choice, and the choice decides the
result. A community district holds hundreds of facilities, thus the mean over
them is nearly the same in every district: the case-specific proxies vary by a
coefficient of variation of 0.02 to 0.13 across districts, while the
socio-economic proxies vary by 0.4 to 0.95. An area correlation on such a
proxy reads the few districts that move, and the aggregation itself makes some
of that agreement.

This module reads the same question 1 level down. A unit is the object the
model looked at and the object the record grades: 1 park, 1 school, 1
restaurant, 1 image of a street. It is the layer the case README asks for:

    Report the unit value. Use the area value only to show the gap.

The 7 cases that carry a unit key
---------------------------------
| Case | The unit | How the proxy reaches the unit |
|------|----------|--------------------------------|
| subway | 1 station | the entrance rolls up to its complex; a key join |
| libraries | 1 library | FacDB carries the BIN; an exact key join |
| restaurants | 1 restaurant | `uid` is `unit_uid`; an exact key join |
| schools | 1 school | borough and school name; a name join |
| parks | 1 park | the unit point inside the DPR polygon |
| plazas | 1 plaza | the capacity proxies key on the unit; a key join |
| road | 1 image | the nearest rated street segment, 100 ft |

A case can carry more than 1 proxy. `CASE_SPEC` names the headline proxy in
`proxy` and the rest in `alt_proxies`, and `build` gives each proxy its own
row.

**The subway seam is closed.** Subway safety pairs station ENTRANCES, and the
outside measurement grades a STATION. `_proxies_subway` now supplies both halves
of the missing step: the roll-up from an entrance to its station complex, and a
station-level record — transit crime for each 100,000 rider entries. Thus the
unit of the subway case is 1 station, and `SUBWAY_SEAM` records what the step
was and how it closed.

The road from a run to a number
-------------------------------
1. Read the canonical run through the registry (`_canonical`).
2. `_provenance.score_units` or `score_images` gives each place a score.
3. Join the unit-level proxy value with the rule of the case notebook.
4. Report n, Spearman rho, Kendall tau-b, and a bootstrap interval.

Step 3 repeats the rule of the case notebook exactly. Six cases publish a unit
correlation, and `reconcile()` proves that this module gives their published
numbers again. Schools publishes none, thus this module builds the first one
for it.

The pairing with the area value
-------------------------------
`with_area()` puts the unit value of a cell beside the area value of the SAME
cell. It reads the area value through `_results_table.load_case` and
`_results_table.metrics`, which is the arithmetic that the master table runs.
Thus a reader sees the gap in 1 table and needs no second document.

The arm
-------
`arm` selects the prompt arm and defaults to `canonical`, the arm that every
paper number reads. Pass `noabstain`, `nocity`, or `textfirst` to read an
ablation at the unit layer.

See `vlm-narratives-docs/unit-validation.md`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _ablations as A
import _canonical as C
import _geography as geo
import _provenance as prov
import _results_table as R

__version__ = "1.1.0"

HERE = Path(__file__).resolve().parent

#: What each case needs to score its units again and to reach its proxy.
#:
#: The values repeat the constants of the case notebook. Change one here only
#: when the notebook changes, or the 2 numbers part company.
CASE_SPEC: Dict[str, Dict[str, Any]] = {
    "subway": dict(
        canon="subway_safety",
        mode="facility",
        root="curation/subway_entrances_all",
        unit_table="entrances.parquet",
        kind=None,
        join="station",
        proxy="transit_crime_per_rider",
        # `_proxies_subway`: MAX_COMPLAINT_FT and the 2024-01 to 2026-06 window.
        max_distance_ft=1000.0,
        roll_up="comparison_weighted",
    ),
    "libraries": dict(
        canon="libraries",
        mode="facility",
        root="curation/facdb_libraries",
        unit_table="facilities.parquet",
        kind=None,
        join="key",
        # The building record grades the BIN, and the library IS its building.
        proxy="building_violations_total",
        alt_proxies=("building_violations_open", "alteration_permits"),
    ),
    "restaurants": dict(
        canon="restaurants",
        mode="facility",
        root="curation/dohmh_restaurants_inspected_all",
        unit_table="restaurants_aggregated.parquet",
        kind=None,
        join="key",
        # The v1.2.0 headline: the mean score over the CYCLE INITIAL
        # inspections of one restaurant. The last-inspection score stays in the
        # case export as context.
        proxy="inspection_score_history",
        # `_proxies_restaurants`: MIN_INSPECTIONS and MIN_HISTORY_INSPECTIONS.
        min_inspections=2,
        min_history=2,
    ),
    "schools": dict(
        canon="schools",
        mode="facility",
        root="curation/facdb_schools_k_12",
        unit_table="facilities.parquet",
        kind=None,
        join="name",
        # The export renamed this proxy at v1.1.0. The old name said the
        # SOURCE; this name says the measurement.
        proxy="school_attendance",
        # `schools_validation.py`: PROXY_YEAR and DEFAULT_METRIC.
        metric="Average Student Attendance",
        school_year="2024",
    ),
    "parks": dict(
        canon="parks_plazas",
        mode="facility",
        root="curation/facdb_parks_plazas",
        unit_table="facilities.parquet",
        kind="park",
        join="spatial",
        proxy="pip_acceptable_rate",
        # The condition rate reads the structural half of the same program,
        # and the acreage is a SIZE measure, not a quality measure. The master
        # table marks acreage as a context row.
        alt_proxies=("pip_condition_acceptable_rate", "park_acreage"),
        # `parks_validation.py`: PIP_SINCE_YEAR = 2023, the metric dropdown
        # opens on cleanliness, and `attach_park_property_id` keeps its 500 ft.
        pip_metric="cleanliness_acceptable_rate",
        pip_since_year=2023,
        max_distance_ft=500.0,
    ),
    "plazas": dict(
        canon="parks_plazas",
        mode="facility",
        root="curation/facdb_parks_plazas",
        unit_table="facilities.parquet",
        kind="plaza",
        join="key",
        # The 3 capacity proxies key on the UNIT, thus the join is the
        # identity. Warning: a violation names the tax LOT, not the plaza
        # surface. See `_proxies_parks.VIOLATION_LABEL`.
        proxy="dob_ecb_violations",
        alt_proxies=("bid_membership", "plaza_bid_partner"),
    ),
    "road": dict(
        canon="road_quality",
        mode="image",
        root=None,
        unit_table=None,
        kind=None,
        join="nearest",
        proxy="dot_pavement_rating",
        # `road_quality_validation.py`: PAVEMENT_SINCE_YEAR and
        # MAX_SEGMENT_DISTANCE_FT.
        pavement_since_year=2024,
        max_distance_ft=100.0,
    ),
}

#: The order the tables use. It follows the master table, which opens on the
#: subway case.
FOLDERS: Sequence[str] = ("subway", "libraries", "schools", "road", "parks",
                          "plazas", "restaurants")

MODELS: Sequence[str] = A.MODELS

CONTROL_ARM: str = A.CONTROL_ARM

#: The short model name to the Hydra config string, which the area readers use.
MODEL_CONFIG = {"gemma-4-12b": "gemma-4-12b/instruct",
                "qwen3.5-9b": "qwen3.5-9b/instruct"}

#: The column that names a place, by the mode of the case.
PLACE_KEY = {"facility": "unit_uid", "image": "sample_id"}

#: How the subway case reaches a unit row. Read it before you read a subway
#: number.
SUBWAY_SEAM = (
    "Subway safety pairs station ENTRANCES, and the outside measurement grades "
    "a STATION. A station holds 4 entrances at the median, thus a unit row "
    "needs a roll-up from the entrance to its station, and a station-level "
    "record to grade it. `_proxies_subway` supplies both. The roll-up key is "
    "the complex where an entrance carries one, and the station where it does "
    "not. The record is the NYPD transit complaint count for each 100,000 MTA "
    "rider entries, over 2024-01 to 2026-06."
)

#: How the entrance scores of 1 station become the score of that station.
#:
#: The choice is the COMPARISON-WEIGHTED mean, not the plain mean. A station
#: holds entrances whose comparison counts differ by a factor of more than 10,
#: and the weighted mean over the entrance means equals the plain mean over the
#: comparisons of the station. Thus the station takes the score it would take if
#: the model had rated the station directly. `place_scores` also returns
#: `mean_score_plain`, so a reader can measure what the choice costs.
#:
#: `_proxies_subway.ROLLUP_RULE` holds the same string, and
#: `_proxies_subway.station_scores` does the work.
SUBWAY_ROLLUP = "comparison_weighted"

# --------------------------------------------------------------------------
# The floors
# --------------------------------------------------------------------------
#: The comparison floor of a FACILITY case. This is the `_provenance.score_units`
#: default, and the case notebooks call that function bare. A facility sits in
#: hundreds of comparisons, thus the floor costs almost nothing and it drops the
#: unit that 1 or 2 comparisons alone describe.
MIN_COMPARISONS_FACILITY = 5

#: The comparison floor of an IMAGE case. This is the `_provenance.score_images`
#: default.
#:
#: Warning: do NOT raise it to 5. An image case spreads its pairs over a
#: 500,000-image manifest, thus the median image takes about 2 comparisons and a
#: floor of 5 empties the case. The precision of an image row comes from the
#: number of images, not from the number of comparisons of one image.
MIN_COMPARISONS_IMAGE = 1

#: A cell under this many units carries no readable correlation. A Spearman rho
#: over fewer units moves by more than 0.2 when 1 unit moves. The cell stays in
#: the frame and takes a `thin_units` mark, because the reader must see that the
#: cell is thin and not absent.
MIN_UNITS = 30

BOOT_N = 400
BOOT_SEED = 20260831

#: The published unit correlations that `reconcile()` reads.
#:
#: The subway file appears when the subway notebook runs against a set of
#: discovered runs. `reconcile()` skips a name that is not on disk.
RECONCILE_SOURCE = {
    "subway": "subway/outputs/subway_unit_correlations_v1.2.0.parquet",
    "libraries": "libraries/outputs/libraries_unit_correlations_v1.2.0.parquet",
    "plazas": "plazas/outputs/plazas_unit_correlations_v1.2.0.parquet",
    "restaurants":
        "restaurants/outputs/restaurants_unit_correlations_v1.2.0.parquet",
    "parks": "parks/outputs/parks_unit_correlations_v1.0.0.parquet",
    "road": "road/outputs/road_quality_unit_correlations_v1.0.0.parquet",
}


def _folder_label(folder: str) -> str:
    for c in R.CASES:
        if c["folder"] == folder:
            return c["label"]
    return folder


def _spec(folder: str) -> Dict[str, Any]:
    if folder not in CASE_SPEC:
        raise KeyError(
            f"{folder!r} carries no unit key; use one of {list(CASE_SPEC)}. "
            f"For subway, see SUBWAY_SEAM.")
    return CASE_SPEC[folder]


def min_comparisons(folder: str) -> int:
    """The comparison floor of one case, by its mode."""
    return (MIN_COMPARISONS_FACILITY if _spec(folder)["mode"] == "facility"
            else MIN_COMPARISONS_IMAGE)


def case_proxies(folder: str) -> Tuple[str, ...]:
    """Every proxy of one case: the headline first, then the alternates."""
    spec = _spec(folder)
    return (spec["proxy"], *spec.get("alt_proxies", ()))


# --------------------------------------------------------------------------
# Step 1 and 2: the place scores of one run
# --------------------------------------------------------------------------
def _run(folder: str, model: str, arm: str):
    canon = _spec(folder)["canon"]
    hits = [r for r in C.runs(kind="proxy", case=canon, model=model, arm=arm)
            if r.case == canon]
    return hits[0] if hits else None


@lru_cache(maxsize=4)
def _merged(folder: str, model: str, arm: str) -> Optional[pd.DataFrame]:
    """The label of each comparison of one run, with the place of each side.

    This is the frame that `_provenance.score_units` and `score_images` take.
    The cache holds few entries on purpose: a run carries 1,100,000 rows.
    """
    spec = _spec(folder)
    run = _run(folder, model, arm)
    if run is None:
        return None
    res = pd.read_parquet(run.results_link,
                          columns=["pair_id", "relative_label"])
    if spec["mode"] == "facility":
        keep = ["pair_id", "unit_uid_a", "unit_uid_b",
                "unit_name_a", "unit_name_b"]
    else:
        keep = ["pair_id", "sample_id_a", "sample_id_b",
                "latitude_a", "latitude_b", "longitude_a", "longitude_b"]
    pairs = pd.read_parquet(run.pairs_link, columns=keep)
    return res.merge(pairs, on="pair_id", how="left", validate="one_to_one")


def roll_up_to_station(scores: pd.DataFrame) -> pd.DataFrame:
    """Roll the entrance scores of the subway case up to their stations.

    The rule is `SUBWAY_ROLLUP`, and `_proxies_subway.station_scores` holds it.
    The subway notebook calls the same function, thus the 2 paths cannot part
    company and `reconcile()` tests the rest of the road.
    """
    import _proxies_subway as sub

    return sub.station_scores(scores)


@lru_cache(maxsize=16)
def place_scores(folder: str, model: str, arm: str = CONTROL_ARM
                 ) -> Optional[pd.DataFrame]:
    """Give each unit of one (case, model, arm) its mean ordinal score.

    A facility case gets `_provenance.score_units` and an image case gets
    `score_images`, each with the floor of `min_comparisons`. An abstention maps
    to no score and drops out. It is not a zero.

    The subway case takes 1 more step. `score_units` gives each ENTRANCE a
    score, and `roll_up_to_station` turns those into 1 score for each STATION.
    Thus the unit of the subway case is a station, and `unit_uid` holds a
    station key.

    Returns None when the registry holds no such cell.
    """
    df = _merged(folder, model, arm)
    if df is None:
        return None
    spec = _spec(folder)
    floor = min_comparisons(folder)
    if spec["mode"] != "facility":
        return prov.score_images(df, min_comparisons=floor)
    scores = prov.score_units(df, min_comparisons=floor)
    if spec["join"] == "station":
        return roll_up_to_station(scores)
    return scores


# --------------------------------------------------------------------------
# Step 3: the unit-level proxy
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _units_gdf(folder: str):
    """The point layer of the units of one case, filtered to its kind."""
    import _proxies as px

    spec = _spec(folder)
    if spec["mode"] == "image":
        return None
    gdf = geo.load_units(spec["root"], filename=spec["unit_table"])
    if spec["kind"]:
        # Split first. The plaza half takes no maintenance proxy, thus a mixed
        # table would let the proxy look complete when it covers 79%.
        facilities = geo.load_facilities(spec["root"],
                                         filename=spec["unit_table"])
        kept = px.split_parks_plazas(facilities, spec["kind"])
        gdf = gdf[gdf.unit_uid.isin(kept.uid)]
    return gdf


#: The PIP metric behind each parks proxy name. The proxy name is the row key
#: of the master table, and the metric name is what `parks_pip_proxy` takes.
PARKS_PIP_METRIC = {
    "pip_acceptable_rate": "cleanliness_acceptable_rate",
    "pip_condition_acceptable_rate": "condition_acceptable_rate",
}


@lru_cache(maxsize=32)
def unit_proxy(folder: str, proxy: Optional[str] = None
               ) -> Optional[pd.DataFrame]:
    """The outside measurement of each unit of one case.

    The frame holds `proxy_key` and `proxy_value`, 1 row for each key. Every
    value leaves the proxy modules oriented "higher is better", thus a positive
    correlation always means agreement and this module applies no sign.

    `proxy` selects one name of `case_proxies(folder)`, and None takes the
    headline. The road case gets None back: its proxy is a LINE layer, and the
    value reaches an image through the nearest-segment join in `unit_key`.
    """
    import _proxies as px

    spec = _spec(folder)
    proxy = spec["proxy"] if proxy is None else proxy

    if folder == "subway":
        import _proxies_subway as sub

        out = sub.station_proxy(
            metric=proxy,
            max_distance_ft=spec["max_distance_ft"])[["proxy_key",
                                                      "proxy_value"]]

    elif folder == "restaurants":
        import _proxies_restaurants as pr

        # `_proxies_restaurants` loads its own unit table, the same way the
        # case notebook does, and it routes an old metric to `_proxies`
        # unchanged.
        out = pr.restaurant_proxy(metric=proxy,
                                  min_inspections=spec["min_inspections"],
                                  min_history=spec["min_history"])
        out = out[["proxy_key", "proxy_value"]]

    elif folder == "libraries":
        import _proxies_buildings as pb

        facilities = geo.load_facilities(spec["root"],
                                         filename=spec["unit_table"])
        out = pb.unit_proxy_by_bin(facilities, metric=proxy)

    elif folder == "plazas":
        import _proxies_parks as pk

        facilities = geo.load_facilities(spec["root"],
                                         filename=spec["unit_table"])
        kept = px.split_parks_plazas(facilities, spec["kind"])
        if proxy == "dob_ecb_violations":
            vals = pk.plaza_violation_values(kept)
        elif proxy == "bid_membership":
            vals = pk.plaza_bid_values(_units_gdf(folder))
        elif proxy == "plaza_bid_partner":
            vals = pk.plaza_partner_values(_units_gdf(folder), kept)
        else:
            raise KeyError(f"unknown plazas proxy {proxy!r}")
        out = (vals[["unit_uid", "proxy_value"]]
               .rename(columns={"unit_uid": "proxy_key"}))

    elif folder == "schools":
        entry = px.load_recipe(folder)[0]
        wide = px.school_quality_wide(
            px.fetch_metric(entry, spec["metric"], spec["school_year"]),
            spec["metric"], school_year=spec["school_year"])
        facilities = geo.load_facilities(spec["root"],
                                         filename=spec["unit_table"])
        joined = px.join_to_units(wide, facilities)["joined"]
        # A borough and a name can hold 2 DBNs, thus a unit can take 2 proxy
        # rows. Average them: a unit must carry 1 value, or the merge below
        # multiplies its score row and the correlation counts it twice.
        out = (joined.groupby("unit_uid", as_index=False)["proxy_value"]
                     .mean().rename(columns={"unit_uid": "proxy_key"}))

    elif folder == "parks":
        if proxy == "park_acreage":
            import _proxies_parks as pk

            # `proxy_key` is the DPR `gispropnum`, the same key that
            # `attach_park_property_id` writes. This is a SIZE measure.
            out = pk.park_acreage()
        else:
            out = px.parks_pip_proxy(metric=PARKS_PIP_METRIC[proxy],
                                     since_year=spec["pip_since_year"])
        out = out[["proxy_key", "proxy_value"]]

    else:
        return None

    return out.dropna(subset=["proxy_key", "proxy_value"]) \
              .drop_duplicates("proxy_key").reset_index(drop=True)


def unit_key(folder: str, scores: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Map the place of each score row to the key of the proxy.

    | Case | The rule |
    |------|----------|
    | subway | `place_scores` already rolled the entrances up to the station. |
    | libraries | FacDB put the building record on the unit; the identity. |
    | restaurants | `uid` is `unit_uid`, thus the map is the identity. |
    | schools | The proxy already carries `unit_uid` after the name join. |
    | parks | The unit POINT inside the DPR polygon, or within 500 ft. |
    | plazas | The capacity proxies key on the unit; the identity. |
    | road | The nearest rated street segment within 100 ft. |

    The road frame carries `proxy_value` as well, because its proxy is a line
    layer and the join reads the value straight off the segment.

    Warning: never use the FacDB polygon of a park. It is a nearby building with
    a median area of 0.0038 km^2, and Central Park covers 3.41 km^2.
    """
    import geopandas as gpd

    import _proxies as px

    spec = _spec(folder)

    if folder in ("subway", "libraries", "restaurants", "schools", "plazas"):
        return pd.DataFrame({"unit_uid": scores["unit_uid"],
                             "proxy_key": scores["unit_uid"]})

    if folder == "parks":
        return px.attach_park_property_id(
            _units_gdf(folder), max_distance_ft=spec["max_distance_ft"])

    # The image case. Build the point layer from the score frame, the same way
    # the road notebook does, and read the rating of the nearest segment.
    segments = px.pavement_segments(since_year=spec["pavement_since_year"])
    if segments is None or len(segments) == 0:
        return None
    pts = gpd.GeoDataFrame(
        scores,
        geometry=gpd.points_from_xy(scores["longitude"], scores["latitude"]),
        crs="EPSG:4326",
    )
    return px.attach_nearest_segment(pts, segments,
                                     max_distance_ft=spec["max_distance_ft"])


def joined(folder: str, model: str, arm: str = CONTROL_ARM,
           proxy: Optional[str] = None) -> Optional[pd.DataFrame]:
    """One row for each unit that holds a model score AND a proxy value.

    The frame carries `mean_score`, `n_comparisons`, and `proxy_value`. Both
    merges are inner joins, thus a unit that either side misses drops out.
    """
    scores = place_scores(folder, model, arm)
    if scores is None or scores.empty:
        return None
    key = unit_key(folder, scores)
    if key is None or key.empty:
        return None

    if _spec(folder)["mode"] == "image":
        # The key frame already holds the rating of the nearest segment.
        out = scores.merge(key, on="sample_id", how="inner")
    else:
        prx = unit_proxy(folder, proxy)
        if prx is None or prx.empty:
            return None
        out = scores.merge(key, on="unit_uid", how="inner") \
                    .merge(prx, on="proxy_key", how="inner")
    out = out.dropna(subset=["mean_score", "proxy_value"])
    return None if out.empty else out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Step 4: the numbers
# --------------------------------------------------------------------------
def _rho_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Spearman rho, Kendall tau-b, and the p value of tau."""
    from scipy import stats

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), float("nan")
    rho = float(stats.spearmanr(x, y).statistic)
    tau = stats.kendalltau(x, y, variant="b")
    return rho, float(tau.statistic), float(tau.pvalue)


def _ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _bootstrap(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
               ) -> Dict[str, float]:
    """A 95% interval on rho and on tau, from a resample over UNITS.

    The unit is the sampling unit of this table, thus the resample draws units
    and never comparisons. A resample over comparisons would ask a different
    question and give a much narrower interval.
    """
    if len(x) < 3:
        return {k: float("nan") for k in
                ("rho_lo", "rho_hi", "tau_lo", "tau_hi")}
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot, dtype=float)
    taus = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        rhos[i], taus[i], _ = _rho_tau(x[ix], y[ix])
    lo_r, hi_r = _ci(rhos)
    lo_t, hi_t = _ci(taus)
    return {"rho_lo": lo_r, "rho_hi": hi_r, "tau_lo": lo_t, "tau_hi": hi_t}


def _coverage(folder: str, model: str) -> float:
    """The coverage of this cell, on the rule the master table applies.

    A rater under `R.MIN_COVERAGE` keeps too few places to carry its case, and
    the master table DROPS such a row. This table keeps it and marks it: the
    unit layer is where a reader asks what a heavy abstainer left behind.
    """
    cover = R.coverage_by_cell()
    return float(cover.get((_folder_label(folder),
                            R.MODEL_LABEL.get(model, model)), float("nan")))


def unit_row(folder: str, model: str, arm: str = CONTROL_ARM,
             n_boot: int = BOOT_N, seed: int = BOOT_SEED,
             proxy: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Every unit-level number of one (case, model, arm, proxy) cell.

    `proxy` selects one name of `case_proxies(folder)`, and None takes the
    headline. Returns None when the registry holds no such run, or when the
    join keeps nothing. A thin cell comes back with its numbers and a
    `thin_units` mark.
    """
    spec = _spec(folder)
    proxy = spec["proxy"] if proxy is None else proxy
    j = joined(folder, model, arm, proxy)
    if j is None:
        return None
    scores = place_scores(folder, model, arm)
    x = j["mean_score"].to_numpy(dtype=float)
    y = j["proxy_value"].to_numpy(dtype=float)
    rho, tau, tau_p = _rho_tau(x, y)

    row: Dict[str, Any] = {
        "folder": folder,
        "case": _folder_label(folder),
        "canon_case": spec["canon"],
        "model": R.MODEL_LABEL.get(model, model),
        "arm": arm,
        "proxy": proxy,
        "unit_kind": spec["mode"],
        "n_units": int(len(j)),
        "n_scored": int(len(scores)) if scores is not None else 0,
        "matched_pct": (round(100 * len(j) / len(scores), 1)
                        if scores is not None and len(scores) else float("nan")),
        "median_comparisons": float(j["n_comparisons"].median()),
        "min_comparisons": min_comparisons(folder),
        "rho": rho,
        "tau": tau,
        "tau_p": tau_p,
        "score_sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "proxy_sd": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        # The spread of the proxy over the UNITS of this join, as a coefficient
        # of variation. `area_rows` reports the same measure over the AREAS.
        # The 2 numbers carry the audit finding: a proxy that varies over units
        # can be nearly flat over areas, and then an area correlation reads a
        # few areas alone.
        "proxy_cv_units": (float(np.std(y, ddof=1) / abs(np.mean(y)))
                           if len(y) > 1 and np.mean(y) != 0 else float("nan")),
        "coverage": _coverage(folder, model),
        "n_boot": int(n_boot),
    }
    row.update(_bootstrap(x, y, n_boot, seed))
    row["thin_units"] = bool(row["n_units"] < MIN_UNITS)
    row["thin_coverage"] = bool(row["coverage"] < R.MIN_COVERAGE)
    row["thin"] = bool(row["thin_units"] or row["thin_coverage"])
    return row


def build(arm: str = CONTROL_ARM, folders: Sequence[str] = FOLDERS,
          models: Sequence[str] = MODELS, n_boot: int = BOOT_N,
          seed: int = BOOT_SEED) -> pd.DataFrame:
    """The long frame: 1 row for each (case, model, proxy) of one arm.

    Warning: this reads the 1,100,000-row run parquets, and the road case runs
    a spatial join over about 490,000 image points. Allow minutes.
    """
    rows: List[Dict[str, Any]] = []
    for folder in folders:
        for model in models:
            for proxy in case_proxies(folder):
                r = unit_row(folder, model, arm, n_boot, seed, proxy=proxy)
                if r is not None:
                    rows.append(r)
    if not rows:
        return pd.DataFrame()
    # The headline proxy of a case prints before its alternates, thus the sort
    # reads the position of each proxy in `case_proxies`, not its name.
    forder = {f: i for i, f in enumerate(FOLDERS)}
    porder = {(f, p): i for f in folders
              for i, p in enumerate(case_proxies(f))}
    df = pd.DataFrame(rows)
    df["_f"] = df["folder"].map(forder)
    df["_p"] = df.apply(lambda r: porder.get((r["folder"], r["proxy"]), 99),
                        axis=1)
    return (df.sort_values(["_f", "_p", "model"])
            .drop(columns=["_f", "_p"]).reset_index(drop=True))


# --------------------------------------------------------------------------
# The pairing with the area value
# --------------------------------------------------------------------------
def area_rows(layer: str = R.DEFAULT_LAYER,
              folders: Sequence[str] = FOLDERS,
              models: Sequence[str] = MODELS) -> pd.DataFrame:
    """The AREA value of each cell, read through the master-table readers.

    This runs `_results_table.load_case` and `_results_table.metrics`, which is
    the arithmetic behind the master table. It does NOT run
    `_results_table.drop_thin_models`: a dropped rater still needs its area
    value here, or the gap of that cell has only one side. The `area_dropped`
    column names each row the master table refuses.

    This module writes nothing to `_results_table` and changes nothing there.
    """
    cover = R.coverage_by_cell()
    rows: List[Dict[str, Any]] = []
    for folder in folders:
        case = next((c for c in R.CASES if c["folder"] == folder), None)
        if case is None:
            continue
        merged = R.load_case(case, layer=layer)
        if merged is None:
            continue
        for want in case_proxies(folder):
            gp = (merged[merged["proxy"] == want] if "proxy" in merged
                  else merged)
            for model in models:
                cfg = MODEL_CONFIG.get(model, model)
                gm = gp[gp["model"] == cfg]
                if gm.empty:
                    continue
                m = R.metrics(gm["mean_score"], gm["proxy_mean"])
                label = R.MODEL_LABEL.get(cfg, model)
                # The spread of the proxy over the AREAS of this join. Read it
                # beside `proxy_cv_units`: a proxy that is nearly flat here
                # carries little for an area correlation to read.
                pv = gm["proxy_mean"].to_numpy(dtype=float)
                cv = (float(np.std(pv, ddof=1) / abs(np.mean(pv)))
                      if len(pv) > 1 and np.mean(pv) != 0 else float("nan"))
                rows.append({
                    "folder": folder,
                    "model": label,
                    "proxy": want,
                    "layer": layer,
                    "n_areas": m["n"],
                    "area_agreement": m["agreement"],
                    "area_r": m["r"],
                    "area_tau": m["tau"],
                    "area_tau_p": m["tau_p"],
                    "area_proxy_cv": cv,
                    "area_dropped": bool(
                        cover.get((_folder_label(folder), label), 1.0)
                        < R.MIN_COVERAGE),
                })
    return pd.DataFrame(rows)


def with_area(long: pd.DataFrame, layer: str = R.DEFAULT_LAYER) -> pd.DataFrame:
    """Put the unit value of each cell beside the area value of the same cell.

    This is the table the audit asks for. 1 row holds the case, the rater, the
    proxy, the unit count with rho and tau-b, and the area count with tau-b.
    `gap_tau` is the area tau minus the unit tau, thus a large positive value
    says that the aggregation made the agreement.
    """
    if long.empty:
        return long
    area = area_rows(layer)
    out = long.merge(area, on=["folder", "model", "proxy"], how="left")
    out["layer"] = layer
    out["gap_tau"] = out["area_tau"] - out["tau"]
    return out


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------
def reconcile(long: pd.DataFrame,
              sources: Dict[str, str] = RECONCILE_SOURCE,
              tol: float = 0.005) -> pd.DataFrame:
    """Test this module against the unit correlations that a case published.

    Each source is a unit correlation that a case notebook exported. This
    module must give those numbers again, or its join rule differs from the
    notebook's rule. A row of the result names the published value, the value
    here, and whether the 2 agree inside `tol`.

    A published frame can hold several proxies, and its column names differ by
    case, thus this reads `spearman_rho` or `rho`, and `n` or `n_units`. A
    published row whose proxy this frame does not carry drops out.
    """
    rows: List[Dict[str, Any]] = []
    for folder, rel in sources.items():
        path = HERE / rel
        if not path.exists():
            continue
        pub = pd.read_parquet(path)
        pub["model"] = pub["model"].map(lambda m: R.MODEL_LABEL.get(m, m))
        if "arm" in pub.columns:
            pub = pub[pub["arm"] == CONTROL_ARM]
        rho_col = "spearman_rho" if "spearman_rho" in pub.columns else "rho"
        n_col = "n" if "n" in pub.columns else "n_units"
        sub = long[long["folder"] == folder]
        for _, p in pub.iterrows():
            mine = sub[sub["model"] == p["model"]]
            if "proxy" in pub.columns and "proxy" in sub.columns:
                mine = mine[mine["proxy"] == p["proxy"]]
            if mine.empty:
                continue
            m = mine.iloc[0]
            d_rho = float(m["rho"]) - float(p[rho_col])
            rows.append({
                "folder": folder,
                "model": p["model"],
                "proxy": p.get("proxy", m["proxy"]),
                "published_n": int(p[n_col]),
                "n_units": int(m["n_units"]),
                "published_rho": float(p[rho_col]),
                "rho": float(m["rho"]),
                "d_rho": d_rho,
                "d_n": int(m["n_units"]) - int(p[n_col]),
                "agrees": bool(abs(d_rho) <= tol
                               and int(m["n_units"]) == int(p[n_col])),
                "source": rel,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def _fmt(v: float, nd: int = 3) -> str:
    return ("--" if v is None or (isinstance(v, float) and not np.isfinite(v))
            else f"{v:+.{nd}f}")


def _plain(v: float, nd: int = 2) -> str:
    return ("--" if v is None or (isinstance(v, float) and not np.isfinite(v))
            else f"{v:.{nd}f}")


def _pm(v: float, lo: float, hi: float, nd: int = 3) -> str:
    """One signed value with its interval, for a screen table."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """A plain frame of the unit rows, for reading on screen.

    A frame that `with_area` built gains the area columns and the gap. A plain
    `build` frame prints the unit half alone.
    """
    if df.empty:
        return df
    out = {
        "case": df["case"],
        "model": df["model"],
        "unit": df["unit_kind"],
        "proxy": df["proxy"].map(lambda p: R.PROXY_LABEL.get(p, p)),
        "units": df["n_units"].astype(int),
        "matched %": df["matched_pct"].map(_plain),
        "rho": df.apply(lambda r: _pm(r["rho"], r["rho_lo"], r["rho_hi"]),
                        axis=1),
        "tau": df.apply(lambda r: _pm(r["tau"], r["tau_lo"], r["tau_hi"]),
                        axis=1),
    }
    if "area_tau" in df.columns:
        out["areas"] = df["n_areas"].map(
            lambda v: "--" if not np.isfinite(v) else str(int(v)))
        out["area tau"] = df["area_tau"].map(lambda v: _fmt(v, 2))
        out["gap tau"] = df["gap_tau"].map(lambda v: _fmt(v, 2))
        # The spread of the proxy on each layer. A small area value says that
        # the proxy is nearly flat over the areas.
        out["cv units"] = df["proxy_cv_units"].map(_plain)
        out["cv areas"] = df["area_proxy_cv"].map(_plain)
        out["master drops"] = df["area_dropped"].map({True: "yes", False: ""})
    out["thin"] = df.apply(
        lambda r: ", ".join([w for w, on in (("coverage", r["thin_coverage"]),
                                             ("units", r["thin_units"]))
                             if on]), axis=1)
    return pd.DataFrame(out)


# The caption. It states what a row holds and what each column measures, and it
# says nothing about what a number means. A reader draws that.
CAPTION = (
    "Validation by proxy at the unit layer. "
    "A unit is one facility, or one image where the case pairs street-level "
    "shots. "
    "Each row reads the model score of a unit against an outside measurement "
    "of the SAME unit, and the columns give the number of units $n$, the "
    "Spearman $\\rho$, and the Kendall $\\tau_b$. "
    "Brackets hold a 95\\% bootstrap interval over units. "
    "The last two columns hold the area value of the same cell at the {layer} "
    "layer, which stands beside the unit value to show the aggregation gap. "
    "Every proxy points in the same direction, thus a positive value shows "
    "agreement."
)

DEFAULT_LABEL = "tab:unit-validation"


def to_latex(df: pd.DataFrame, caption: Optional[str] = None,
             label: str = DEFAULT_LABEL, layer: str = R.DEFAULT_LAYER,
             star_p: float = 0.05, full_width: bool = True) -> str:
    """The paper table: 1 row for each cell, the unit value and the area value.

    Warning: the table needs `\\usepackage{booktabs}` for the rules.
    """
    if df.empty:
        return "% no unit-validation row\n"
    has_area = "area_tau" in df.columns
    env = "table*" if full_width else "table"
    text = caption if caption is not None else CAPTION.format(
        layer=R.LAYER_LABEL.get(layer, layer))
    head = ["Case", "Rater", "Unit", "Proxy", "$n$", r"$\rho$", r"$\tau_b$"]
    cols = "llll r r r"
    if has_area:
        head += ["Areas", r"area $\tau_b$", r"$\Delta\tau_b$"]
        cols += " r r r"

    lines = [
        "% Generated by _unit_validation.py — do not edit by hand.",
        "% Needs \\usepackage{booktabs}.",
        f"\\begin{{{env}}}[t]", "\\centering",
        f"\\caption{{{text}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}", "\\small",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
        " & ".join(head) + " \\\\", "\\midrule",
    ]
    for _, r in df.iterrows():
        star = ("*" if np.isfinite(r["tau_p"]) and r["tau_p"] < star_p else "")
        cells = [r["case"], r["model"], r["unit_kind"],
                 R.PROXY_LABEL.get(r["proxy"], r["proxy"]),
                 f"{int(r['n_units']):,}".replace(",", "{,}"),
                 f"{r['rho']:+.3f}", f"{r['tau']:+.3f}" + star]
        if has_area:
            n_a = r["n_areas"]
            cells += ["--" if not np.isfinite(n_a) else str(int(n_a)),
                      _fmt(r["area_tau"], 2), _fmt(r["gap_tau"], 2)]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        # The star note lives here, not in the caption: ASD-STE100 holds a
        # paragraph to 6 sentences, and the caption already uses all 6.
        "\\par\\smallskip",
        f"{{\\footnotesize A star marks $p < {star_p}$ on the unit "
        f"$\\tau_b$.}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines) + "\n"


def export(long: pd.DataFrame, paired: pd.DataFrame, out_dir,
           tag: str = "unit_validation", version: str = "1.0.0",
           layer: str = R.DEFAULT_LAYER,
           reconciliation: Optional[pd.DataFrame] = None) -> List[str]:
    """Write the frames and the LaTeX. Give back the names.

    The unit rows carry no layer, thus their stem drops it. The paired rows do
    carry one, thus their stem holds it.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_v{version}"
    paired_stem = f"{tag}_{layer}_v{version}"
    written: List[str] = []

    long.to_csv(out / f"{stem}.csv", index=False)
    written.append(f"{stem}.csv")
    long.to_parquet(out / f"{stem}.parquet", index=False)
    written.append(f"{stem}.parquet")

    if paired is not None and not paired.empty:
        paired.to_csv(out / f"{paired_stem}_paired.csv", index=False)
        written.append(f"{paired_stem}_paired.csv")
        paired.to_parquet(out / f"{paired_stem}_paired.parquet", index=False)
        written.append(f"{paired_stem}_paired.parquet")
        (out / f"{paired_stem}.tex").write_text(to_latex(paired, layer=layer))
        written.append(f"{paired_stem}.tex")
    else:
        (out / f"{stem}.tex").write_text(to_latex(long, layer=layer))
        written.append(f"{stem}.tex")

    if reconciliation is not None and not reconciliation.empty:
        reconciliation.to_csv(out / f"{stem}_reconcile.csv", index=False)
        written.append(f"{stem}_reconcile.csv")
    return written
