"""Validation by proxy, run again on a second prompt arm.

The master table reads what the case notebooks exported, and those exports come
from the CANONICAL arm alone. This module runs the same road from the run
parquet to the 3 metrics for ANY arm, so the no-abstention battery can be read
against the same outside measurements.

The question
------------
The canonical prompt lets the model write `NotSure`. If the model keeps its
harder calls behind that word, then the judgments it DOES write are a selection,
and the proxy agreement of the canonical arm is flattered by it. Forcing the
model to answer adds exactly those held-back calls back in.

| What the numbers do | What it says |
|---------------------|--------------|
| The arm agrees LESS | The abstention held back noise. It was honest. |
| The arm agrees MORE | The abstention held back signal the model had. |
| No change | The abstention carried no information either way. |

**Warning: a fall does not prove the abstention was honest by itself.** The
no-abstention arm also compresses the score of each place toward the middle
(see `_quantile_shift.py`), and a narrower spread lowers a correlation on its
own. `spread_ratio` reports that compression beside the metrics, so a reader
can separate the two.

The held split (`held_table`) asks the sharper form of the question. It scores
the arm's pairs in 2 halves — the pairs the canonical arm abstained on, and a
size-matched draw of the pairs it answered — and reads each half against the
proxy alone. The 2 halves share the rater, the prompt, and the pair count,
thus a difference between them is the content of the held judgments.

The road
--------
The arm changes only the FIRST step. Everything after it is the code the case
notebooks run:

1. Read the run parquet of the arm and join it to `pairs.parquet`.
2. `_provenance.score_units` (or `score_images`) gives each place a score.
3. `_geography.aggregate` puts the places into the polygons of one layer.
4. Join to the proxy and read n, agreement, Pearson r and Kendall tau-b.

Step 4 reads the proxy parquet that the case notebook exported. A proxy is an
outside measurement, thus it does not move with the arm and it must not be
built again.

**Both arms go through THIS module, never through the case export.** A number
here and a number in the master table can differ a little, because a case
notebook can set its own floor. Comparing a recomputed arm with an exported
control would read that difference as a prompt effect.

See `research/docs/proxy-arms.md`.
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
import _style as S

__version__ = "1.1.0"

HERE = Path(__file__).resolve().parent

#: What each proxy row of the master table needs to score its places again.
#:
#: The values repeat the constants of the case notebooks. `min_units` is that
#: notebook's `MIN_UNITS_PER_POLYGON`, and `kind` splits the one parks_plazas
#: run into its 2 table rows.
#:
#: **Warning: the libraries floor moved to 3 on 2026-08-31.** A `min_units` of
#: 1 lets an area score rest on a single place. The plazas floor STAYS at 1 on
#: purpose: a floor of 3 keeps 16 community districts, under `MIN_AREAS`, and
#: the NTA layer does not rescue the case. The unit layer answers the
#: single-plaza problem instead. See
#: `research/docs/parks-plazas-proxies.md` and
#: `research/docs/proxy-audit-fixes.md`.
CASE_SPEC: Dict[str, Dict[str, Any]] = {
    "subway": dict(canon="subway_safety", root="curation/subway_entrances_all",
                   unit_table="entrances.parquet", kind=None, min_units=1,
                   mode="facility"),
    "libraries": dict(canon="libraries", root="curation/facdb_libraries",
                      unit_table="facilities.parquet", kind=None, min_units=3,
                      mode="facility"),
    "schools": dict(canon="schools", root="curation/facdb_schools_k_12",
                    unit_table="facilities.parquet", kind=None, min_units=3,
                    mode="facility"),
    "road": dict(canon="road_quality", root=None, unit_table=None, kind=None,
                 min_units=1, mode="image"),
    "parks": dict(canon="parks_plazas", root="curation/facdb_parks_plazas",
                  unit_table="facilities.parquet", kind="park", min_units=1,
                  mode="facility"),
    "plazas": dict(canon="parks_plazas", root="curation/facdb_parks_plazas",
                   unit_table="facilities.parquet", kind="plaza", min_units=1,
                   mode="facility"),
    "restaurants": dict(canon="restaurants",
                        root="curation/dohmh_restaurants_inspected_all",
                        unit_table="restaurants_aggregated.parquet",
                        kind=None, min_units=3, mode="facility"),
}

#: The order the master table uses.
FOLDERS: Sequence[str] = tuple(c["folder"] for c in R.CASES)

MODELS: Sequence[str] = A.MODELS
CONTROL_ARM = A.CONTROL_ARM
DEFAULT_ARM = "noabstain"

ARM_COLOUR = {
    "canonical": S.PAL["blue"],
    "noabstain": S.PAL["coral"],
    "nocity": S.PAL["amber"],
    "textfirst": S.PAL["teal"],
}

#: The area floor. `_results_table.py` owns it, and this module reads it from
#: there, thus the master table and this table never print 2 different rules.
MIN_AREAS = R.MIN_AREAS


def _folder_label(folder: str) -> str:
    for c in R.CASES:
        if c["folder"] == folder:
            return c["label"]
    return folder


def _prefix(folder: str) -> str:
    for c in R.CASES:
        if c["folder"] == folder:
            return c["prefix"]
    return folder


# --------------------------------------------------------------------------
# Step 1 and 2: the place scores of one arm
# --------------------------------------------------------------------------
def _run(canon_case: str, model: str, arm: str):
    hits = [r for r in C.runs(kind="proxy", case=canon_case, model=model,
                              arm=arm) if r.case == canon_case]
    return hits[0] if hits else None


@lru_cache(maxsize=6)
def _merged(folder: str, model: str, arm: str) -> Optional[pd.DataFrame]:
    """The label of each comparison of one run, with the place of each side.

    This is the frame that `_provenance.score_units` and `score_images` take.
    It is cached because a matched control reads the canonical run again for
    each arm.
    """
    spec = CASE_SPEC[folder]
    run = _run(spec["canon"], model, arm)
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


@lru_cache(maxsize=64)
def place_frame(folder: str, model: str, arm: str,
                match_arm: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Give each place of one (case, model, arm) its score.

    `match_arm` keeps ONLY the comparisons that the named arm also judged.

    **Warning: pass `match_arm` whenever the 2 arms drew different numbers of
    pairs.** `nocity` and `textfirst` drew 110,000 pairs for each case and the
    canonical arm drew 1,100,000, thus a place takes about 770 comparisons in
    those arms against 8,246 in the control. A place score built on 10 times
    fewer comparisons carries more sampling noise, and noise attenuates a
    correlation on its own. Matching the pair set gives both sides the same
    UNIT pairs, so the difference that remains is mostly the prompt.

    What a match does and does not give
    -----------------------------------
    | Level | Does it match? |
    |-------|----------------|
    | The unit pair | Yes. `pair_id` names the 2 places. |
    | The A/B order | Yes. `pair_id` carries the presentation order. |
    | The 2 photographs | `noabstain` yes; `nocity` and `textfirst` almost never. |

    **Warning: in `nocity` and `textfirst` the 2 arms saw DIFFERENT
    photographs of the same 2 units.** A facility case draws a fresh image for
    each side when it builds the pairs, thus a matched `pair_id` names the
    same 2 facilities and not the same 2 pictures. The share of shared pairs
    where both photographs match, measured 2026-08-31:

    | Case | Share |
    |------|-------|
    | Libraries | 0.2% |
    | Schools | 0.2% |
    | Parks and plazas | 0.2% |
    | Subway safety | 0.6% |
    | Restaurants | 1.7% |
    | Road quality | 100% |

    This adds noise to `d_tau` for those 2 arms, and it adds no direction: the
    image draw does not know which arm it serves. Read a small `d_tau` there
    as "no effect that survives the image noise", never as "no effect".

    2 arms escape it. `noabstain` reads the SAME 1,100,000-pair draw, thus its
    photographs match on 100% of the shared pairs. Road quality is an image
    case: there the place IS the photograph, thus it matches by construction.

    The join key is `pair_id`, never `canonical_pair_id`: `pair_id` carries the
    presentation order, and about 8% of the shared pairs came up in the
    opposite A/B order.
    """
    df = _merged(folder, model, arm)
    if df is None:
        return None
    if match_arm and match_arm != arm:
        other = _merged(folder, model, match_arm)
        if other is None:
            return None
        df = df[df["pair_id"].isin(pd.Index(other["pair_id"]))]
        if df.empty:
            return None
    mode = CASE_SPEC[folder]["mode"]
    return (prov.score_units(df) if mode == "facility"
            else prov.score_images(df))


def match_rate(folder: str, model: str, arm: str,
               control: str = CONTROL_ARM) -> float:
    """The share of the arm's comparisons that the control also judged."""
    a = _merged(folder, model, arm)
    b = _merged(folder, model, control)
    if a is None or b is None or len(a) == 0:
        return float("nan")
    return float(a["pair_id"].isin(pd.Index(b["pair_id"])).mean())


# --------------------------------------------------------------------------
# Step 3: the geography
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _units_gdf(folder: str):
    """The point layer of the places of one case, filtered to its kind."""
    import _proxies as px

    spec = CASE_SPEC[folder]
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


@lru_cache(maxsize=256)
def aggregate_arm(folder: str, model: str, arm: str, layer: str,
                  match_arm: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Put the places of one arm into the polygons of one layer.

    The result is cached, because the paired interval of `compare` reads the
    same cell again after `build` wrote its row. Do not change the frame that
    this returns: every caller shares one object.
    """
    scores = place_frame(folder, model, arm, match_arm)
    if scores is None or scores.empty:
        return None
    return _aggregate_scores(folder, scores, layer)


def _aggregate_scores(folder: str, scores: pd.DataFrame, layer: str
                      ) -> Optional[pd.DataFrame]:
    """Put one frame of place scores into the polygons of one layer.

    The 2 modes give the SAME columns, thus a reader of the frame needs no
    test of the mode. See `_geography.aggregate` for what each column holds.
    """
    import geopandas as gpd

    spec = CASE_SPEC[folder]
    if spec["mode"] == "facility":
        out = geo.aggregate(scores, _units_gdf(folder), layer,
                            min_units=spec["min_units"])
        return None if out.empty else out

    # An image case carries its own position, so it needs no facility table.
    pts = gpd.GeoDataFrame(
        scores,
        geometry=gpd.points_from_xy(scores["longitude"], scores["latitude"]),
        crs="EPSG:4326",
    ).to_crs(geo.WORKING_CRS)
    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    hit = gpd.sjoin(pts[["sample_id", "mean_score", "n_comparisons",
                         "abstention_rate", "geometry"]],
                    poly[[key, "geometry"]], how="inner", predicate="within")
    out = hit.groupby(key).agg(
        n_units=("sample_id", "nunique"),
        mean_score=("mean_score", "mean"),
        sd_score=("mean_score", "std"),
        total_comparisons=("n_comparisons", "sum"),
        mean_abstention=("abstention_rate", "mean"),
    ).reset_index()
    out = out.merge(geo.weighted_by_area(hit, key, "mean_score",
                                         "n_comparisons"),
                    on=key, how="left")
    out["mean_score_weighted"] = out["mean_score_weighted"].fillna(
        out["mean_score"])
    out = out[out.n_units >= spec["min_units"]].sort_values(
        "mean_score", ascending=False)
    out.insert(0, "layer", layer)
    out = out.reset_index(drop=True)
    return None if out.empty else out


# --------------------------------------------------------------------------
# Step 4: the proxy
# --------------------------------------------------------------------------
def _newest(pattern: str) -> Optional[str]:
    import glob

    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


@lru_cache(maxsize=16)
def proxy_frame(folder: str, layer: str) -> Optional[pd.DataFrame]:
    """The proxy values of one case on one layer.

    This reads the export of the case notebook. A proxy is an outside
    measurement: it does not move with the prompt arm, thus the 2 arms share
    it and it must not be built again here.
    """
    path = _newest(str(HERE / folder / "outputs"
                       / f"{_prefix(folder)}_proxy_v*.parquet"))
    if not path:
        return None
    prx = pd.read_parquet(path)
    prx = prx[prx["layer"] == layer]
    if prx.empty:
        return None
    if "proxy" not in prx.columns:
        single = next((c.get("single_proxy") for c in R.CASES
                       if c["folder"] == folder), None)
        prx = prx.assign(proxy=single or "proxy")
    key = R.LAYER_KEY[layer]
    return prx[["proxy", key, "proxy_mean"]].dropna(subset=[key])


def cell_rows(folder: str, model: str, arm: str,
              layer: str = R.DEFAULT_LAYER,
              match_arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row for each proxy of one (case, model, arm) cell.

    `match_arm` names the arm whose pair set both sides share. It travels onto
    every row, because a control is only a control for the arm it matches.
    """
    agg = aggregate_arm(folder, model, arm, layer, match_arm)
    prx = proxy_frame(folder, layer)
    if agg is None or prx is None:
        return []
    key = R.LAYER_KEY[layer]
    rows: List[Dict[str, Any]] = []
    for proxy, g in prx.groupby("proxy", sort=False):
        j = agg.merge(g, on=key, how="inner")
        if j.empty:
            continue
        m = R.metrics(j["mean_score"], j["proxy_mean"])
        rows.append({
            "folder": folder,
            "case": _folder_label(folder),
            "canon_case": CASE_SPEC[folder]["canon"],
            "model": model,
            "arm": arm,
            "match_arm": match_arm or arm,
            "layer": layer,
            "proxy": proxy,
            "n_areas": m["n"],
            "agreement": m["agreement"],
            "r": m["r"],
            "tau": m["tau"],
            "tau_p": m["tau_p"],
            # The spread of the model score over the areas of THIS join. A
            # correlation falls when the spread falls, whatever the ranking
            # does, so the reader needs it beside the metrics.
            "score_sd": float(np.std(j["mean_score"], ddof=1)),
            "n_places": int(agg["n_units"].sum()),
            "coverage": _coverage(folder, model),
            "thin_areas": bool(m["n"] < MIN_AREAS),
        })
    return rows


def _coverage(folder: str, model: str) -> float:
    """The coverage of this cell, on the rule the master table applies.

    A rater under `R.MIN_COVERAGE` keeps too few places to carry the case. The
    master table DROPS such a row. This section keeps it and marks it, because
    the arm comparison is about the abstention itself, and the rater that
    abstains most is the one a reader will ask about. Its canonical side is
    still a selection, thus `summary` leaves it out of the count.
    """
    cover = R.coverage_by_cell()
    return float(cover.get((_folder_label(folder), model), float("nan")))


# --------------------------------------------------------------------------
# Robustness: does the abstention enter the area score?
# --------------------------------------------------------------------------
#
# The area score is a plain mean over the places of the area. That mean gives
# each place the same voice although the kept comparison counts differ by a
# factor of 25 after the `NotSure` drop, thus a place that abstained often
# pulls the area as hard as a place that answered every time. The area
# abstention rate also correlates with the proxies, up to r = 0.59, so the
# channel is real and it needs a test.
#
# `mean_score_weighted` weights each place by its comparison count. If the
# 2 taus agree, then the abstention does not carry the correlation.
def weighted_tau_rows(folder: str, model: str, arm: str = CONTROL_ARM,
                      layer: str = R.DEFAULT_LAYER,
                      match_arm: Optional[str] = None) -> List[Dict[str, Any]]:
    """One row for each proxy: the plain tau beside the weighted tau."""
    from scipy import stats

    agg = aggregate_arm(folder, model, arm, layer, match_arm)
    prx = proxy_frame(folder, layer)
    if agg is None or prx is None or "mean_score_weighted" not in agg:
        return []
    key = R.LAYER_KEY[layer]
    rows: List[Dict[str, Any]] = []
    for proxy, g in prx.groupby("proxy", sort=False):
        j = agg.merge(g, on=key, how="inner")
        if len(j) < 3:
            continue
        m = R.metrics(j["mean_score"], j["proxy_mean"])
        w = R.metrics(j["mean_score_weighted"], j["proxy_mean"])
        # How far the abstention itself tracks the proxy. A high value says
        # the plain mean carries a second signal beside the judgments.
        ar = float("nan")
        if "mean_abstention" in j and j["mean_abstention"].notna().any():
            ok = j["mean_abstention"].notna() & j["proxy_mean"].notna()
            if int(ok.sum()) >= 3 and j.loc[ok, "mean_abstention"].std() > 0:
                ar = float(stats.pearsonr(j.loc[ok, "mean_abstention"],
                                          j.loc[ok, "proxy_mean"])[0])
        rows.append({
            "folder": folder,
            "case": _folder_label(folder),
            "model": model,
            "arm": arm,
            "layer": layer,
            "proxy": proxy,
            "n_areas": m["n"],
            "tau": m["tau"],
            "tau_weighted": w["tau"],
            "d_tau_weight": w["tau"] - m["tau"],
            "r": m["r"],
            "r_weighted": w["r"],
            "abstention_proxy_r": ar,
            "thin_areas": bool(m["n"] < MIN_AREAS),
        })
    return rows


def weighted_tau_table(folders: Sequence[str] = FOLDERS,
                       models: Sequence[str] = MODELS,
                       arm: str = CONTROL_ARM,
                       layer: str = R.DEFAULT_LAYER) -> pd.DataFrame:
    """The robustness frame: the plain tau and the weighted tau of each cell.

    A small `d_tau_weight` says the abstention does not carry the proxy
    agreement. A large one says the plain area mean and the weighted area
    mean read the outside measurement differently, and the paper must then
    state which mean it reports.
    """
    rows = [r for folder in folders for model in models
            for r in weighted_tau_rows(folder, model, arm, layer)]
    return pd.DataFrame(rows)


def weighted_to_display(df: pd.DataFrame) -> pd.DataFrame:
    """A plain frame of the robustness rows, for reading on screen."""
    if df.empty:
        return df
    return pd.DataFrame({
        "case": df["case"],
        "model": df["model"],
        "proxy": df["proxy"].map(lambda p: R.PROXY_LABEL.get(p, p)),
        "areas": df["n_areas"].astype(int),
        "tau plain": df["tau"].map(_fmt),
        "tau weighted": df["tau_weighted"].map(_fmt),
        "d tau": df["d_tau_weight"].map(lambda v: f"{v:+.2f}"),
        "abstain vs proxy r": df["abstention_proxy_r"].map(_fmt),
        "thin": np.where(df["thin_areas"], "areas", ""),
    })


# --------------------------------------------------------------------------
# The held pairs: the forced choices alone, against the proxy
# --------------------------------------------------------------------------
#: The seed of the size-matched draw of `_held_split`.
HELD_SEED = 20260831


@lru_cache(maxsize=32)
def _held_split(folder: str, model: str, control: str = CONTROL_ARM,
                seed: int = HELD_SEED
                ) -> Optional[Tuple[pd.Index, pd.Index]]:
    """Split the control's pairs: the held pairs, and a matched answered draw.

    A held pair is one the canonical arm answered `NotSure`. An answered pair
    got a label. A subset with fewer pairs gives each place a noisier score,
    and noise attenuates a correlation on its own, thus BOTH subsets take the
    size of the smaller one, with a seed-fixed draw.
    """
    m = _merged(folder, model, control)
    if m is None:
        return None
    lab = m["relative_label"].astype(str)
    held = m.loc[lab.eq(prov.NOT_SURE), "pair_id"]
    answered = m.loc[~lab.eq(prov.NOT_SURE), "pair_id"]
    n = int(min(len(held), len(answered)))
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    if len(held) > n:
        held = held.sample(n=n, random_state=int(rng.integers(2**31)))
    if len(answered) > n:
        answered = answered.sample(n=n, random_state=int(rng.integers(2**31)))
    return pd.Index(held), pd.Index(answered)


def held_share(folder: str, model: str, control: str = CONTROL_ARM) -> float:
    """The share of the control's pairs that it answered `NotSure`."""
    m = _merged(folder, model, control)
    if m is None or len(m) == 0:
        return float("nan")
    return float(m["relative_label"].astype(str).eq(prov.NOT_SURE).mean())


def _subset_scores(folder: str, model: str, arm: str, ids: pd.Index
                   ) -> Tuple[Optional[pd.DataFrame], int]:
    """Score the places of one arm from the named pairs alone.

    Gives back the frame and the number of the arm's pairs that the subset
    kept. The join key is `pair_id`, thus the presentation order matches.
    """
    df = _merged(folder, model, arm)
    if df is None:
        return None, 0
    df = df[df["pair_id"].isin(ids)]
    if df.empty:
        return None, 0
    mode = CASE_SPEC[folder]["mode"]
    scores = (prov.score_units(df) if mode == "facility"
              else prov.score_images(df))
    return scores, int(len(df))


def subset_rows(folder: str, model: str, arm: str = DEFAULT_ARM,
                layer: str = R.DEFAULT_LAYER,
                control: str = CONTROL_ARM) -> List[Dict[str, Any]]:
    """One row for each proxy and each side of the split.

    The `side` column says which half of the arm's pairs made the row: `held`
    (the canonical arm abstained there) or `answered` (it gave a label). Both
    sides read the SAME rater, prompt, and pair count, thus a difference
    between them is the content of the held judgments.
    """
    split = _held_split(folder, model, control)
    prx = proxy_frame(folder, layer)
    if split is None or prx is None:
        return []
    key = R.LAYER_KEY[layer]
    share = held_share(folder, model, control)
    rows: List[Dict[str, Any]] = []
    for side, ids in zip(("held", "answered"), split):
        scores, n_pairs = _subset_scores(folder, model, arm, ids)
        if scores is None or scores.empty:
            continue
        agg = _aggregate_scores(folder, scores, layer)
        if agg is None:
            continue
        for proxy, g in prx.groupby("proxy", sort=False):
            j = agg.merge(g, on=key, how="inner")
            if j.empty:
                continue
            m = R.metrics(j["mean_score"], j["proxy_mean"])
            rows.append({
                "folder": folder,
                "case": _folder_label(folder),
                "canon_case": CASE_SPEC[folder]["canon"],
                "model": model,
                "arm": arm,
                "side": side,
                "layer": layer,
                "proxy": proxy,
                "held_share": share,
                "n_pairs": n_pairs,
                "median_comparisons": float(scores["n_comparisons"].median()),
                "n_places": int(len(scores)),
                "n_areas": m["n"],
                "agreement": m["agreement"],
                "r": m["r"],
                "tau": m["tau"],
                "tau_p": m["tau_p"],
                "score_sd": float(np.std(j["mean_score"], ddof=1)),
                "thin_areas": bool(m["n"] < MIN_AREAS),
            })
    return rows


def held_table(arm: str = DEFAULT_ARM, folders: Sequence[str] = FOLDERS,
               models: Sequence[str] = MODELS,
               layer: str = R.DEFAULT_LAYER) -> pd.DataFrame:
    """The long frame of the split: one row for each (cell, proxy, side)."""
    rows = [r for folder in folders for model in models
            for r in subset_rows(folder, model, arm, layer)]
    return pd.DataFrame(rows)


def held_to_display(df: pd.DataFrame) -> pd.DataFrame:
    """The 2 sides of each cell beside each other, for reading on screen."""
    if df.empty:
        return df
    keys = ["case", "model", "proxy"]
    h = df[df["side"] == "held"].set_index(keys)
    a = df[df["side"] == "answered"].set_index(keys)
    shared = h.index.intersection(a.index)
    if len(shared) == 0:
        return pd.DataFrame()
    h, a = h.loc[shared], a.loc[shared]
    out = pd.DataFrame({
        "held share": h["held_share"].map(lambda v: f"{v:.0%}"),
        "pairs/side": h["n_pairs"].astype(int),
        "areas held": h["n_areas"].astype(int),
        "areas ans": a["n_areas"].astype(int),
        "tau held": h["tau"].map(_fmt),
        "tau answered": a["tau"].map(_fmt),
        "d tau": (h["tau"] - a["tau"]).map(lambda v: f"{v:+.2f}"),
        "agr held": h["agreement"].map(_fmt),
        "agr answered": a["agreement"].map(_fmt),
        "thin": np.where(h["thin_areas"] | a["thin_areas"], "areas", ""),
    }, index=shared)
    out.index = out.index.set_names(keys)
    return out.reset_index()


#: Every arm the registry holds beside the canonical prompt.
ARMS: Sequence[str] = tuple(A.ABLATION_ARMS)


def build(arms: Sequence[str] = ARMS,
          folders: Sequence[str] = FOLDERS,
          models: Sequence[str] = MODELS,
          layer: str = R.DEFAULT_LAYER,
          control: str = CONTROL_ARM,
          match: bool = True) -> pd.DataFrame:
    """The long frame: one row for each (case, model, arm, proxy).

    Each arm gets its OWN control row, built on that arm's pair set. Thus the
    frame holds one canonical row for each arm, and `match_arm` says which arm
    a control belongs to. See `place_frame` for why the matching is necessary.

    Set `match=False` to read every arm against the full canonical run. That
    reproduces the master table exactly, and it confounds the small arms with
    their sampling noise.

    The loop runs folder and model on the outside so the cached canonical frame
    serves all 3 arms before it leaves the cache.
    """
    rows: List[Dict[str, Any]] = []
    for folder in folders:
        for model in models:
            for arm in arms:
                key = arm if match else None
                rows += cell_rows(folder, model, control, layer, key)
                rows += cell_rows(folder, model, arm, layer, key)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # A control row repeats once for each arm. Keep them: each one carries the
    # pair set of its arm, thus they are NOT copies of one another.
    return out.drop_duplicates(
        subset=["folder", "model", "arm", "match_arm", "layer", "proxy"])


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------
METRICS = ("agreement", "r", "tau")

#: The paired bootstrap of `compare`. 400 resamples give a stable 95%
#: percentile interval and cost about 0.2 s for one row.
BOOT_N = 400
BOOT_SEED = 20260831


def paired_areas(folder: str, model: str, arm: str, layer: str, proxy: str,
                 control: str = CONTROL_ARM) -> Optional[pd.DataFrame]:
    """The area scores of both arms and the proxy, on the shared areas.

    One row is one area, and it carries the canonical score, the arm score,
    and the proxy value. The 2 arms scored the same places, thus a reader of
    `d_tau` must hold the area fixed and read both arms at it.
    """
    a = aggregate_arm(folder, model, control, layer, arm)
    b = aggregate_arm(folder, model, arm, layer, arm)
    prx = proxy_frame(folder, layer)
    if a is None or b is None or prx is None:
        return None
    g = prx[prx["proxy"] == proxy]
    if g.empty:
        return None
    key = R.LAYER_KEY[layer]
    j = (a[[key, "mean_score"]].rename(columns={"mean_score": "canon"})
         .merge(b[[key, "mean_score"]].rename(columns={"mean_score": "arm"}),
                on=key, how="inner")
         .merge(g[[key, "proxy_mean"]], on=key, how="inner"))
    return j if len(j) >= 3 else None


def d_tau_interval(folder: str, model: str, arm: str, layer: str, proxy: str,
                   control: str = CONTROL_ARM, n_boot: int = BOOT_N,
                   seed: int = BOOT_SEED) -> Dict[str, Any]:
    """A paired 95% bootstrap interval on the change of Kendall tau-b.

    One resample draws an AREA index and reads both arms at it, the same
    pattern as `_quantile_shift.shift_row`. The 2 arms score the same areas,
    thus an unpaired interval throws that pairing away and reads much too
    wide.

    The point estimate here uses the shared areas alone, thus it can differ a
    little from the `d_tau` of the long frame, which lets each arm keep the
    areas it holds.
    """
    from scipy import stats

    nan = {"d_tau_paired": float("nan"), "d_tau_lo": float("nan"),
           "d_tau_hi": float("nan"), "n_areas_paired": 0, "n_boot": 0}
    j = paired_areas(folder, model, arm, layer, proxy, control)
    if j is None:
        return nan
    c = j["canon"].to_numpy(dtype=float)
    b = j["arm"].to_numpy(dtype=float)
    p = j["proxy_mean"].to_numpy(dtype=float)

    def _d(ix) -> float:
        tc = stats.kendalltau(c[ix], p[ix], variant="b").statistic
        tb = stats.kendalltau(b[ix], p[ix], variant="b").statistic
        return float(tb - tc)

    full = np.arange(len(j))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(j), size=(n_boot, len(j)))
    vals = np.array([_d(ix) for ix in draws], dtype=float)
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return {"d_tau_paired": _d(full), "d_tau_lo": float(lo),
            "d_tau_hi": float(hi), "n_areas_paired": int(len(j)),
            "n_boot": int(n_boot)}


def compare(long: pd.DataFrame, control: str = CONTROL_ARM,
            arm: str = DEFAULT_ARM, boot: bool = True,
            n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Put one arm beside its control and give the change of each metric.

    A row exists only where BOTH sides produced the cell, thus every delta
    reads the same case, rater, proxy and layer. The control is the row whose
    `match_arm` names this arm, so it carries the arm's pair set.

    `boot` adds `d_tau_lo` and `d_tau_hi`, a paired 95% bootstrap interval
    over the shared areas. Set it to `False` for a fast frame with no
    interval. See `d_tau_interval`.
    """
    if long.empty:
        return long
    keys = ["folder", "case", "canon_case", "model", "layer", "proxy"]
    sub = long[long["match_arm"] == arm] if "match_arm" in long else long
    a = sub[sub["arm"] == control].set_index(keys)
    b = sub[sub["arm"] == arm].set_index(keys)
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return pd.DataFrame()
    a, b = a.loc[shared], b.loc[shared]

    out = pd.DataFrame(index=shared)
    out["arm"] = arm
    out["n_areas_canon"] = a["n_areas"]
    out["n_areas_arm"] = b["n_areas"]
    for m in METRICS:
        out[f"{m}_canon"] = a[m]
        out[f"{m}_arm"] = b[m]
        out[f"d_{m}"] = b[m] - a[m]
    out["tau_p_canon"] = a["tau_p"]
    out["tau_p_arm"] = b["tau_p"]
    out["sd_canon"] = a["score_sd"]
    out["sd_arm"] = b["score_sd"]
    if boot:
        ci = pd.DataFrame(
            [d_tau_interval(folder, model, arm, layer, proxy, control,
                            n_boot, seed)
             for folder, _case, _canon, model, layer, proxy in shared],
            index=shared)
        for col in ("d_tau_paired", "d_tau_lo", "d_tau_hi", "n_areas_paired"):
            out[col] = ci[col]
    # Under 1 the arm squeezed the area scores together. A correlation falls
    # with the spread even when the ranking does not change.
    out["spread_ratio"] = b["score_sd"] / a["score_sd"]
    out["coverage"] = a["coverage"]
    out["thin_areas"] = a["thin_areas"] | b["thin_areas"]
    out["thin_coverage"] = a["coverage"] < R.MIN_COVERAGE
    out["thin"] = out["thin_areas"] | out["thin_coverage"]
    out["layout_confounded"] = [
        (arm, m) in A.LAYOUT_CONFOUNDED
        for m in out.index.get_level_values("model")
    ]
    return out.reset_index()


def drift(long: pd.DataFrame, arms: Sequence[str] = ARMS,
          control: str = CONTROL_ARM, boot: bool = True) -> pd.DataFrame:
    """Every arm beside its own control, in 1 frame.

    The `arm` column names which arm a row belongs to, thus the frame stacks
    what `compare` gives for each arm and every renderer takes it whole.

    `boot` adds the paired interval on `d_tau`. It costs about 0.2 s for one
    row. Set it to `False` for a fast look.
    """
    frames = [c for arm in arms
              if not (c := compare(long, control, arm, boot=boot)).empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    order = {a: i for i, a in enumerate(arms)}
    return out.sort_values(
        ["folder", "model", "proxy", "arm"],
        key=lambda col: col.map(order) if col.name == "arm" else col,
    ).reset_index(drop=True)


def drift_wide(df: pd.DataFrame, metric: str = "tau") -> pd.DataFrame:
    """1 row for each cell and 1 column for each arm, of `d_<metric>`.

    This is the table a reader scans to ask whether the arms move together.
    """
    if df.empty:
        return df
    keys = ["case", "model", "proxy"]
    wide = df.pivot_table(index=keys, columns="arm", values=f"d_{metric}")
    wide.columns = [f"d {metric}: {A.ARM_LABEL.get(c, c)}" for c in wide.columns]
    base = (df[df["arm"] == df["arm"].iloc[0]]
            .set_index(keys)[f"{metric}_canon"].rename(f"{metric} canonical"))
    return wide.join(base).reset_index()


def verdict(df: pd.DataFrame, tol: float = 0.02) -> pd.Series:
    """Name what happened to `tau` in each row.

    `tol` is the PRACTICAL band that counts as no change: a move under 0.02
    changes the order of no area pair a reader would notice.

    **Warning: `tol` is not a standard error.** The marginal error of a
    Kendall tau-b over about 50 areas is near 0.09, but `d_tau` is a PAIRED
    difference: both arms read the same areas, thus most of that error
    cancels and the paired interval is much narrower than 0.09. `d_tau_lo`
    and `d_tau_hi` carry that interval. Read them for significance, and read
    `tol` for size.
    """
    d = df["d_tau"]
    return pd.Series(
        np.where(d.isna(), "--",
                 np.where(d > tol, "agrees more",
                          np.where(d < -tol, "agrees less", "no change"))),
        index=df.index)


def summary(df: pd.DataFrame, tol: float = 0.02,
            drop_thin: bool = True) -> pd.DataFrame:
    """Count the direction of the change, over every cell and by rater.

    `drop_thin` leaves out a row that the master table would not print: a
    rater under `R.MIN_COVERAGE` for its case, or a join under `MIN_AREAS`
    areas. Such a row moves far on noise alone and it would drive the count.
    """
    if df.empty:
        return df
    use = df[~df["thin"]] if (drop_thin and "thin" in df.columns) else df
    if use.empty:
        return pd.DataFrame()

    def _block(name: str, g: pd.DataFrame) -> Dict[str, Any]:
        return {"group": name, **verdict(g, tol).value_counts().to_dict(),
                "cells": int(len(g)),
                "median d_tau": float(g["d_tau"].median()),
                "median spread ratio": float(g["spread_ratio"].median())}

    rows: List[Dict[str, Any]] = []
    if "arm" in use.columns and use["arm"].nunique() > 1:
        # 1 block for each arm first: the question is whether the arms drift
        # the same way, and a pooled count would hide an arm that does not.
        for arm in [a for a in ARMS if a in set(use["arm"])]:
            g = use[use["arm"] == arm]
            rows.append(_block(A.ARM_LABEL.get(arm, arm), g))
            for model, gm in g.groupby("model"):
                rows.append(_block(f"  {A.ARM_LABEL.get(arm, arm)} — {model}",
                                   gm))
    else:
        rows.append(_block("all", use))
        rows += [_block(model, g) for model, g in use.groupby("model")]
    out = pd.DataFrame(rows).fillna(0)
    if drop_thin and "thin" in df.columns:
        out.attrs["n_thin_dropped"] = int(df["thin"].sum())
    return out


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def _fmt(v: float, nd: int = 2) -> str:
    return ("--" if v is None or (isinstance(v, float) and np.isnan(v))
            else f"{v:.{nd}f}")


def _interval(lo: float, hi: float, nd: int = 2) -> str:
    """One 95% interval, for a screen table."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def to_display(df: pd.DataFrame, tol: float = 0.02) -> pd.DataFrame:
    """A plain frame of the comparison, for reading on screen."""
    if df.empty:
        return df
    return pd.DataFrame({
        "arm": df["arm"].map(A.ARM_LABEL).fillna(df["arm"])
               if "arm" in df.columns else DEFAULT_ARM,
        "case": df["case"],
        "model": df["model"],
        "proxy": df["proxy"].map(lambda p: R.PROXY_LABEL.get(p, p)),
        "areas": df["n_areas_canon"].astype(int),
        "agr canon": df["agreement_canon"].map(_fmt),
        "agr arm": df["agreement_arm"].map(_fmt),
        "r canon": df["r_canon"].map(_fmt),
        "r arm": df["r_arm"].map(_fmt),
        "tau canon": df["tau_canon"].map(_fmt),
        "tau arm": df["tau_arm"].map(_fmt),
        "d tau": df["d_tau"].map(lambda v: f"{v:+.2f}"),
        # The paired 95% interval, over the areas that both arms keep. An
        # interval that holds 0 says the arm moved nothing that the areas can
        # carry. See `d_tau_interval`.
        "d tau 95%": (df.apply(
            lambda r: _interval(r["d_tau_lo"], r["d_tau_hi"]), axis=1)
            if "d_tau_lo" in df.columns else ""),
        "spread": df["spread_ratio"].map(_fmt),
        "reads": verdict(df, tol),
        # The master table drops a rater under the coverage floor, and a join
        # over 12 areas carries no correlation. Mark both: a marked row is
        # still worth seeing here, but it must not drive the count.
        "thin": df.apply(
            lambda r: ", ".join(
                [w for w, on in (("coverage", r["thin_coverage"]),
                                 ("areas", r["thin_areas"])) if on]),
            axis=1) if "thin_coverage" in df.columns else "",
        "layout conf.": df["layout_confounded"].map({True: "yes", False: ""}),
    })


CAPTION = (
    "Validation by proxy, run again on each prompt ablation. Each row holds "
    "the same case, rater, proxy and geography layer under the canonical "
    "prompt and under 1 arm, on the pairs that BOTH arms judged. "
    "$\\Delta\\tau$ is the arm minus the "
    "canonical Kendall $\\tau_b$; \\emph{spread} is the ratio of the standard "
    "deviation of the area scores, and a value under 1 says the arm squeezed "
    "the areas together, which lowers a correlation on its own."
)

DEFAULT_LABEL = "tab:proxy-arms"


def to_latex(df: pd.DataFrame, caption: Optional[str] = None,
             label: str = DEFAULT_LABEL, full_width: bool = True) -> str:
    """The paper table: the canonical arm and the no-abstention arm."""
    if df.empty:
        return "% no proxy-arm row\n"
    env = "table*" if full_width else "table"
    head = ["Case", "Rater", "Proxy", "Areas", "agr", "agr$'$", "$r$",
            "$r'$", r"$\tau_b$", r"$\tau_b'$", r"$\Delta\tau_b$", "spread"]
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             "\\begin{tabular}{lll r rr rr rr rr}", "\\toprule",
             " & ".join(head) + " \\\\",
             "\\multicolumn{12}{l}{\\footnotesize A prime marks the "
             "no-abstention arm.} \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join([
            r["case"], r["model"], R.PROXY_LABEL.get(r["proxy"], r["proxy"]),
            f"{int(r['n_areas_canon'])}",
            _fmt(r["agreement_canon"]), _fmt(r["agreement_arm"]),
            _fmt(r["r_canon"]), _fmt(r["r_arm"]),
            _fmt(r["tau_canon"]), _fmt(r["tau_arm"]),
            f"{r['d_tau']:+.2f}", _fmt(r["spread_ratio"]),
        ]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{caption or CAPTION}}}", f"\\label{{{label}}}",
              f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_slopes(df: pd.DataFrame, metric: str = "tau", figsize=(6.6, 4.2)):
    """The canonical value and the arm value of each row, joined by a line.

    A line that falls says the arm agrees less with the outside measurement.
    The colour says which way it went, so the eye reads the direction before
    it reads any number.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("no row to plot")
    rows = df.reset_index(drop=True)
    y = np.arange(len(rows))
    a = rows[f"{metric}_canon"].to_numpy(dtype=float)
    b = rows[f"{metric}_arm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.PAL["warmgrey"], lw=0.7, ls=":", zorder=1)
    for i, (x0, x1) in enumerate(zip(a, b)):
        if not (np.isfinite(x0) and np.isfinite(x1)):
            continue
        colour = S.PAL["coral"] if x1 < x0 else S.PAL["teal"]
        ax.annotate("", xy=(x1, y[i]), xytext=(x0, y[i]),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=2)
    ax.scatter(a, y, s=26, color=ARM_COLOUR["canonical"], edgecolors=S.EDGE,
               linewidths=S.EDGE_LW, zorder=3, label="canonical")
    ax.scatter(b, y, s=26, color=ARM_COLOUR["noabstain"], edgecolors=S.EDGE,
               linewidths=S.EDGE_LW, zorder=3, label="no abstention")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['case']} — {r['model']} — "
         f"{R.PROXY_LABEL.get(r['proxy'], r['proxy'])}"
         for _, r in rows.iterrows()], fontsize=6.5)
    ax.invert_yaxis()
    label = {"tau": r"Kendall $\tau_b$", "r": "Pearson $r$",
             "agreement": "directional agreement"}[metric]
    ax.set_xlabel(f"{label} against the proxy")
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return fig


def plot_spread_vs_delta(df: pd.DataFrame, figsize=(4.8, 3.8)):
    """The change of tau against the compression of the area scores.

    If the arm lost agreement only because it squeezed the scores together,
    the points fall on a rising line through (1, 0). A point that sits well
    below that line lost agreement for another reason.

    A frame of 1 arm colours by rater. A frame of several arms colours by arm,
    because the question is then which arm moves and the rater is secondary.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("no row to plot")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color=S.EDGE, lw=0.7, zorder=1)
    ax.axvline(1.0, color=S.EDGE, lw=0.7, zorder=1)
    thin = (df["thin"] if "thin" in df.columns
            else pd.Series(False, index=df.index))
    multi = "arm" in df.columns and df["arm"].nunique() > 1
    if multi:
        groups = [(A.ARM_LABEL.get(a, a), ARM_COLOUR.get(a, S.PAL["warmgrey"]),
                   df[df["arm"] == a])
                  for a in ARMS if a in set(df["arm"])]
    else:
        groups = [(m, S.PAL["blue"] if "qwen" in m else S.PAL["amber"], g)
                  for m, g in df.groupby("model")]
    for name, colour, g in groups:
        solid = g[~thin.loc[g.index]]
        ax.scatter(solid["spread_ratio"], solid["d_tau"], s=30, color=colour,
                   edgecolors=S.EDGE, linewidths=S.EDGE_LW, zorder=3,
                   label=name)
        # A thin cell stays on the figure and loses its fill. It is the row a
        # reader asks about, and it must not read as a counted point.
        hollow = g[thin.loc[g.index]]
        if not hollow.empty:
            ax.scatter(hollow["spread_ratio"], hollow["d_tau"], s=30,
                       facecolors="none", edgecolors=colour, linewidths=1.1,
                       zorder=3)
    if thin.any():
        ax.scatter([], [], s=30, facecolors="none", edgecolors=S.EDGE,
                   linewidths=1.1, label="under a floor")
    ax.set_xlabel("spread ratio: arm sd / canonical sd")
    ax.set_ylabel(r"$\Delta$ Kendall $\tau_b$")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def _cell_labels(rows: pd.DataFrame) -> List[str]:
    return [f"{r['case']} — {r['model']} — "
            f"{R.PROXY_LABEL.get(r['proxy'], r['proxy'])}"
            for _, r in rows.iterrows()]


def plot_drift(df: pd.DataFrame, metric: str = "tau",
               arms: Optional[Sequence[str]] = None, figsize=(7.0, 7.2)):
    """The drift of every arm against its own control, on one axis.

    Each row is a cell and each colour an arm. A column of markers on the zero
    line says the prompt change moved no agreement anywhere.

    **Warning: each arm carries its OWN control**, built on that arm's pair
    set, thus the figure plots a difference and never an absolute value. The
    canonical number is not the same for `noabstain` and for `nocity`, because
    those arms drew different pairs.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("no row to plot")
    arms = list(arms or [a for a in ARMS if a in set(df["arm"])])
    keys = ["case", "model", "proxy"]
    cells = df[keys].drop_duplicates().reset_index(drop=True)
    pos = {tuple(r): i for i, r in enumerate(cells.itertuples(index=False))}
    off = np.linspace(-0.28, 0.28, len(arms))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, lw=0.9, zorder=2)
    for k, arm in enumerate(arms):
        g = df[df["arm"] == arm]
        if g.empty:
            continue
        y = np.array([pos[(r.case, r.model, r.proxy)] for r in
                      g.itertuples(index=False)], dtype=float) + off[k]
        x = g[f"d_{metric}"].to_numpy(dtype=float)
        thin = (g["thin"].to_numpy() if "thin" in g
                else np.zeros(len(g), dtype=bool))
        colour = ARM_COLOUR.get(arm, S.PAL["warmgrey"])
        ax.scatter(x[~thin], y[~thin], s=24, color=colour, edgecolors=S.EDGE,
                   linewidths=S.EDGE_LW, zorder=4,
                   label=A.ARM_LABEL.get(arm, arm))
        if thin.any():
            ax.scatter(x[thin], y[thin], s=24, facecolors="none",
                       edgecolors=colour, linewidths=1.1, zorder=4)
    for i in range(len(cells)):
        ax.axhline(i, color=S.PAL["warmgrey"], lw=0.4, alpha=0.35, zorder=1)
    if "thin" in df and df["thin"].any():
        ax.scatter([], [], s=24, facecolors="none", edgecolors=S.EDGE,
                   linewidths=1.1, label="under a floor")
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(_cell_labels(cells), fontsize=6.5)
    ax.set_ylim(len(cells) - 0.5, -0.5)
    label = {"tau": r"Kendall $\tau_b$", "r": "Pearson $r$",
             "agreement": "directional agreement"}[metric]
    ax.set_xlabel(f"drift of {label}: arm minus its matched canonical control")
    ax.legend(frameon=False, fontsize=7, ncol=min(len(arms) + 1, 4),
              loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    return fig


def plot_arm_slopes(df: pd.DataFrame, metric: str = "tau",
                    arms: Optional[Sequence[str]] = None, figsize=(9.5, 6.4)):
    """The slope of each cell, 1 panel for each arm, on a shared row order.

    An arrow runs from the matched canonical value to the arm value. A falling
    arrow says the arm agrees less with the outside measurement. The colour
    says the direction, so the eye reads it before it reads a number.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("no row to plot")
    arms = list(arms or [a for a in ARMS if a in set(df["arm"])])
    keys = ["case", "model", "proxy"]
    cells = df[keys].drop_duplicates().reset_index(drop=True)
    pos = {tuple(r): i for i, r in enumerate(cells.itertuples(index=False))}

    fig, axes = plt.subplots(1, len(arms), figsize=figsize, sharey=True,
                             squeeze=False)
    for ax, arm in zip(axes[0], arms):
        g = df[df["arm"] == arm]
        ax.axvline(0.0, color=S.PAL["warmgrey"], lw=0.7, ls=":", zorder=1)
        for r in g.itertuples(index=False):
            y = pos[(r.case, r.model, r.proxy)]
            x0 = getattr(r, f"{metric}_canon")
            x1 = getattr(r, f"{metric}_arm")
            if not (np.isfinite(x0) and np.isfinite(x1)):
                continue
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="->", lw=1.3, shrinkA=0,
                                        shrinkB=0,
                                        color=(S.PAL["coral"] if x1 < x0
                                               else S.PAL["teal"])), zorder=2)
        ax.scatter(g[f"{metric}_canon"],
                   [pos[(r.case, r.model, r.proxy)]
                    for r in g.itertuples(index=False)],
                   s=20, color=ARM_COLOUR["canonical"], edgecolors=S.EDGE,
                   linewidths=S.EDGE_LW, zorder=3)
        # The arm dot takes the DIRECTION colour, not the arm colour. Inside a
        # panel the arm is fixed (the x label names it), thus colour is free to
        # carry the one thing the eye should read first: which way it moved.
        fell = (g[f"{metric}_arm"] < g[f"{metric}_canon"]).to_numpy()
        ax.scatter(g[f"{metric}_arm"],
                   [pos[(r.case, r.model, r.proxy)]
                    for r in g.itertuples(index=False)],
                   s=20, c=[S.PAL["coral"] if f else S.PAL["teal"]
                            for f in fell],
                   edgecolors=S.EDGE, linewidths=S.EDGE_LW, zorder=3)
        ax.set_xlabel(A.ARM_LABEL.get(arm, arm))
    axes[0][0].set_yticks(range(len(cells)))
    axes[0][0].set_yticklabels(_cell_labels(cells), fontsize=6.0)
    axes[0][0].set_ylim(len(cells) - 0.5, -0.5)
    label = {"tau": r"Kendall $\tau_b$", "r": "Pearson $r$",
             "agreement": "directional agreement"}[metric]
    fig.supxlabel(f"{label} against the proxy — the blue dot is the matched "
                  f"canonical control, and the arrow runs to the arm",
                  fontsize=8)
    fig.tight_layout()
    return fig


def export(long: pd.DataFrame, cmp_df: pd.DataFrame, out_dir,
           tag: str = "proxy_arms", version: str = "1.0.0",
           layer: str = R.DEFAULT_LAYER) -> List[str]:
    """Write the frames, the LaTeX, and the 2 figures. Give back the names."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_{layer}_v{version}"
    written: List[str] = []

    long.to_csv(out / f"{stem}_long.csv", index=False)
    written.append(f"{stem}_long.csv")
    cmp_df.to_csv(out / f"{stem}.csv", index=False)
    written.append(f"{stem}.csv")
    (out / f"{stem}.tex").write_text(to_latex(cmp_df))
    written.append(f"{stem}.tex")

    if not cmp_df.empty:
        multi = "arm" in cmp_df.columns and cmp_df["arm"].nunique() > 1
        if multi:
            drift_wide(cmp_df).to_csv(out / f"{stem}_wide.csv", index=False)
            written.append(f"{stem}_wide.csv")
            for name, fig in (("drift", plot_drift(cmp_df)),
                              ("slopes", plot_arm_slopes(cmp_df))):
                fig.savefig(out / f"{stem}_{name}.png", dpi=300,
                            bbox_inches="tight")
                plt.close(fig)
                written.append(f"{stem}_{name}.png")
        else:
            fig = plot_slopes(cmp_df)
            fig.savefig(out / f"{stem}_slopes.png", dpi=300,
                        bbox_inches="tight")
            plt.close(fig)
            written.append(f"{stem}_slopes.png")
        fig = plot_spread_vs_delta(cmp_df)
        fig.savefig(out / f"{stem}_spread.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{stem}_spread.png")
    return written
