"""Tier B of the robustness battery: does the downstream estimand survive?

Tier A (`_robustness.py`) asks whether a wording arm changes the per-pair
judgments. This module asks the question the paper cares about: does the
validation-by-proxy result of the subway-safety case survive the 2 wording
arms that cost the most agreement, `paraphrase` and `persona`, at 25,000
pairs. Plan: `research/docs/robustness-battery-rerun-plan.md`, phase 6.

Every read goes through the canonical registry. The 2 arms register as
`paraphrase` and `persona` (`research/scripts/register_canonical_runs.py`), and each
arm reads against a MATCHED control: the canonical run restricted to the
arm's own `pair_id` rows, which are rows of the 1M table.

The 3 reads
-----------
| Read | Level | Code path |
|------|-------|-----------|
| unit | 1 station, the v1.2.0 proxies (`transit_crime_per_rider`, ...) | `_unit_validation` helpers on the arm and on the matched control |
| area | census tract, every proxy of the case export | `_proxy_arms.build(match=True)` |
| tract-score correlation | census tract | Pearson between the arm's tract scores and the matched control's |

The counterbalanced halves (`halves`) come from the canonical 1M run alone:
the swapped and unswapped presentations each give a tract score, and the 2
must agree if the presentation order does not carry the signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _canonical as C
from research.notebooks.cvpr import _provenance as prov
from research.notebooks.cvpr import _proxy_arms as PA
from research.notebooks.cvpr import _results_table as R
from research.notebooks.cvpr import _unit_validation as UV

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "master" / "outputs"

FOLDER = "subway"
CANON_CASE = "subway_safety"
ARMS: Sequence[str] = ("paraphrase", "persona")
MODELS: Sequence[str] = ("gemma-4-12b", "qwen3.5-9b")
LAYER = "census_tract"
INCOME = "median_household_income"


# --------------------------------------------------------------------------
# Unit level
# --------------------------------------------------------------------------

def _station_scores(df: pd.DataFrame) -> pd.DataFrame:
    scores = prov.score_units(df, min_comparisons=UV.min_comparisons(FOLDER))
    return UV.roll_up_to_station(scores)


def _unit_metrics(scores: pd.DataFrame, proxy: str) -> Dict[str, float]:
    key = UV.unit_key(FOLDER, scores)
    prx = UV.unit_proxy(FOLDER, proxy)
    j = (scores.merge(key, on="unit_uid", how="inner")
               .merge(prx, on="proxy_key", how="inner")
               .dropna(subset=["mean_score", "proxy_value"]))
    x = j["mean_score"].to_numpy(dtype=float)
    y = j["proxy_value"].to_numpy(dtype=float)
    rho, tau, tau_p = UV._rho_tau(x, y)
    return {"n_units": int(len(j)), "rho": rho, "tau": tau, "tau_p": tau_p,
            "score_sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")}


def unit_rows(model: str, arm: str) -> List[Dict[str, object]]:
    """The unit-level proxies of the arm and of its matched control."""
    arm_df = UV._merged(FOLDER, model, arm)
    ctl_df = UV._merged(FOLDER, model, "canonical")
    if arm_df is None or ctl_df is None:
        return []
    ctl_df = ctl_df[ctl_df["pair_id"].isin(pd.Index(arm_df["pair_id"]))]
    sides = {"arm": _station_scores(arm_df), "control": _station_scores(ctl_df)}
    both = sides["arm"].merge(sides["control"], on="unit_uid", suffixes=("_a", "_c"))
    unit_corr = float(both["mean_score_a"].corr(both["mean_score_c"])) if len(both) > 2 else float("nan")
    rows = []
    for proxy in UV.case_proxies(FOLDER):
        for side, scores in sides.items():
            rows.append({"level": "unit", "model": model, "arm": arm, "side": side,
                         "proxy": proxy, "n_pairs": int(len(arm_df)),
                         "match_rate": float(len(ctl_df) / len(arm_df)),
                         "unit_score_corr": unit_corr,
                         **_unit_metrics(scores, proxy)})
    return rows


# --------------------------------------------------------------------------
# Area level
# --------------------------------------------------------------------------

def area_rows(arms: Sequence[str] = ARMS, models: Sequence[str] = MODELS,
              layer: str = LAYER) -> pd.DataFrame:
    df = PA.build(arms=arms, folders=(FOLDER,), models=models, layer=layer, match=True)
    if df.empty:
        return df
    df = df.rename(columns={"n_areas": "n_units"})
    df["level"] = "area"
    df["side"] = np.where(df["arm"] == "canonical", "control", "arm")
    df["arm"] = df["match_arm"]
    return df[["level", "model", "arm", "side", "proxy", "layer", "n_units",
               "agreement", "r", "tau", "tau_p", "score_sd", "n_places"]]


def tract_corr(model: str, arm: str, layer: str = LAYER) -> Dict[str, object]:
    """Pearson between the arm's tract scores and the matched control's."""
    a = PA.aggregate_arm(FOLDER, model, arm, layer)
    c = PA.aggregate_arm(FOLDER, model, "canonical", layer, match_arm=arm)
    key = R.LAYER_KEY[layer]
    if a is None or c is None:
        return {"model": model, "arm": arm, "n_areas": 0, "tract_corr": float("nan")}
    j = a[[key, "mean_score"]].merge(c[[key, "mean_score"]], on=key, suffixes=("_a", "_c"))
    return {"model": model, "arm": arm, "n_areas": int(len(j)),
            "tract_corr": float(j["mean_score_a"].corr(j["mean_score_c"])),
            "tract_spearman": float(j["mean_score_a"].corr(j["mean_score_c"], method="spearman"))}


# --------------------------------------------------------------------------
# The counterbalanced halves of the canonical run
# --------------------------------------------------------------------------

def halves(model: str, layer: str = LAYER, proxy: str = INCOME) -> Dict[str, object]:
    run = UV._run(FOLDER, model, "canonical")
    if run is None:
        return {}
    res = pd.read_parquet(run.results_link, columns=["pair_id", "relative_label", "is_swapped"])
    pairs = pd.read_parquet(run.pairs_link, columns=["pair_id", "unit_uid_a", "unit_uid_b",
                                                     "unit_name_a", "unit_name_b"])
    df = res.merge(pairs, on="pair_id", how="left", validate="one_to_one")
    prx = PA.proxy_frame(FOLDER, layer)
    prx = prx[prx["proxy"] == proxy]
    key = R.LAYER_KEY[layer]
    out: Dict[str, object] = {"model": model, "proxy": proxy, "layer": layer}
    aggs = {}
    for name, half in (("unswapped", df[~df["is_swapped"].astype(bool)]),
                       ("swapped", df[df["is_swapped"].astype(bool)])):
        scores = prov.score_units(half, min_comparisons=UV.min_comparisons(FOLDER))
        agg = PA._aggregate_scores(FOLDER, scores, layer)
        aggs[name] = agg
        j = agg.merge(prx, on=key, how="inner")
        m = R.metrics(j["mean_score"], j["proxy_mean"])
        out[f"n_{name}"] = int(len(half))
        out[f"areas_{name}"] = m["n"]
        out[f"r_{name}"] = m["r"]
        out[f"tau_{name}"] = m["tau"]
    j = aggs["unswapped"][[key, "mean_score"]].merge(aggs["swapped"][[key, "mean_score"]],
                                                    on=key, suffixes=("_u", "_s"))
    out["half_corr"] = float(j["mean_score_u"].corr(j["mean_score_s"]))
    out["areas_shared"] = int(len(j))
    return out


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------

def export(arms: Sequence[str] = ARMS, models: Sequence[str] = MODELS,
           out_dir: Path = OUT_DIR, version: str = "1.0.0",
           with_halves: bool = True) -> List[Path]:
    C.verify_or_raise(quick=True)
    rows: List[Dict[str, object]] = []
    for model in models:
        for arm in arms:
            rows += unit_rows(model, arm)
    long = pd.concat([pd.DataFrame(rows), area_rows(arms, models)], ignore_index=True)
    corr = pd.DataFrame([tract_corr(m, a) for m in models for a in arms])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in [(f"robustness_estimand_v{version}.csv", long),
                     (f"robustness_estimand_tract_corr_v{version}.csv", corr)]:
        (out_dir / name).write_text(df.to_csv(index=False))
        written.append(out_dir / name)
    if with_halves:
        h = pd.DataFrame([halves(m) for m in models])
        p = out_dir / f"robustness_estimand_halves_v{version}.csv"
        p.write_text(h.to_csv(index=False))
        written.append(p)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--no-halves", action="store_true")
    args = ap.parse_args(argv)
    files = export(args.arms, args.models, args.out_dir, args.version,
                   with_halves=not args.no_halves)
    for f in files:
        print(f"[estimand] wrote {f}")
    long = pd.read_csv(files[0])
    cols = ["level", "model", "arm", "side", "proxy", "n_units", "rho", "tau", "tau_p", "r", "score_sd"]
    print(long[[c for c in cols if c in long.columns]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.read_csv(files[1]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(files) > 2:
        print(pd.read_csv(files[2]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
