"""The prompt ablations, read pair for pair against the canonical arm.

Two batteries changed ONE thing each and judged the same image pairs again:

| Arm | What changed | Pairs for each case |
|-----|--------------|---------------------|
| `noabstain` | the decoder cannot emit `NotSure` | 1,000,000 |
| `nocity` | the prompt does not name the city | 100,000 |
| `textfirst` | the prompt text comes BEFORE the images | 100,000 |

Both read against the `canonical` arm, which is the prompt the paper reports.
`research/scripts/register_canonical_runs.py` holds all 3 arms, thus this module chooses
no run: it asks the registry for an arm and joins.

The join key
------------
**Join on `pair_id`, never on `canonical_pair_id`.** `pair_id` carries the
presentation order, and `canonical_pair_id` does not. About 8% of the pairs
that the `nocity` battery shares with the canonical battery came up in the
OPPOSITE A/B order. A join on the canonical id compares a pair with its own
mirror image, thus it reads the order effect as a prompt effect.

Measured on 2026-08-26, over all 7 cases:

| Arm | Shares `canonical_pair_id` | Shares `pair_id` |
|-----|---------------------------|------------------|
| `noabstain` | 100% | 100% of 1,100,000 rows |
| `nocity` | 100% | 91.7% of 110,000 rows |

`match_rate` reports this on every row, so a thin join cannot hide.

What the columns mean
---------------------
| Column | Meaning | No effect |
|--------|---------|-----------|
| `n_pairs` | Pairs both arms judged, same order | -- |
| `match_rate` | Share of the arm's rows that the join kept | 1.0 |
| `abstain_canon`, `abstain_arm` | Share of `NotSure` | -- |
| `label_agreement` | Same label of the 6 | the marginals |
| `label_kappa` | The same, corrected for chance | -- |
| `direction_agreement` | Same side, where both arms answer | the marginals |
| `direction_kappa` | The same, corrected for chance | -- |
| `mean_score_canon`, `mean_score_arm` | Mean ordinal score, both answered | -- |
| `score_delta` | `arm` minus `canon`, paired | 0 |
| `score_delta_se` | Standard error of the paired mean | -- |

Warning: `label_agreement` is not a quality measure. It answers "how much did
the prompt change move?", and a low value is a large effect, not a defect.

Every cell reads cleanly since 2026-09-04. The gemma halves of `nocity`
and `textfirst` re-ran on 2026-09-02 and the `noabstain` gemma half on
2026-09-04, all on the layout of the canonical gemma control. Thus a delta
holds the 1 change the arm names. `LAYOUT_CONFOUNDED` is empty; it stays so
that `to_display` can mark a future arm that runs another layout.

The `textfirst` arm gives each rater ONE change against ITS OWN production
layout:

| Rater | Production | The arm |
|-------|------------|---------|
| qwen3.5-9b | `images_then_text` | `text_first` |
| gemma-4-12b | `interleaved_labels` | `text_first_interleaved` |

gemma-4-12b does not take plain `text_first`: that layout puts the 2 images
next to each other, and this arch then reads them as 1 image. The anchors of
`text_first_interleaved` keep them apart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _canonical as C
from research.notebooks.cvpr import _results_table as R

__version__ = "1.0.0"

#: The ablation arms this module reads. `canonical` is the control, never a row.
ABLATION_ARMS: Sequence[str] = ("noabstain", "nocity", "textfirst")

CONTROL_ARM = "canonical"

ARM_LABEL = {
    "canonical": "canonical",
    "noabstain": "no abstention",
    "nocity": "no city name",
    "textfirst": "text first",
    "matched": "matched control",
    "paraphrase": "paraphrase (subway, 25k)",
    "persona": "persona added (subway, 25k)",
}

#: The cells whose control ran another image layout, thus their delta holds
#: MORE than the 1 change the arm names. Empty since 2026-09-04.
#:
#: The canonical gemma runs use `interleaved_labels` (decision of
#: 2026-09-02). The `nocity` and `textfirst` gemma halves re-ran on
#: 2026-09-02 on `interleaved_labels` and `text_first_interleaved`, over rows
#: cut from the 1,000,000-pair table, thus they join the control on `pair_id`
#: at 100%. The `noabstain` gemma half re-ran on `interleaved_labels` over the
#: full table and registered on 2026-09-04 (`multirun/noabstain_merged_il`).
#: The qwen cells of every arm read cleanly: that rater ran 1 layout
#: throughout. Add a cell here when an arm runs another layout than its
#: control.
LAYOUT_CONFOUNDED: set = set()

#: The 7 cases, in the order the paper uses.
#:
#: Warning: do NOT build this from `R.CASES`. That list drives the
#: validation-by-proxy table, thus it holds only a case that HAS a proxy, and
#: street photography has none. An ablation needs no proxy: it compares a run
#: with a run. `R.RAW_CASE_LABEL` holds all 7 cases in the paper order.
CASES: Sequence[str] = tuple(R.RAW_CASE_LABEL)

MODELS: Sequence[str] = ("gemma-4-12b", "qwen3.5-9b")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def load_arm(case: str, model: str, arm: str) -> Optional[pd.Series]:
    """The label of each comparison of one registered run, keyed by `pair_id`.

    Returns None when the registry holds no such cell.
    """
    return R.load_raw_labels(case, model, kind="proxy", arm=arm)


def paired(case: str, model: str, arm: str) -> Optional[pd.DataFrame]:
    """Join the canonical arm and one ablation arm on `pair_id`.

    The frame holds 2 columns, `canon` and `arm`, and 1 row for each pair that
    BOTH arms judged in the SAME presentation order.
    """
    a = load_arm(case, model, CONTROL_ARM)
    b = load_arm(case, model, arm)
    if a is None or b is None:
        return None
    df = pd.concat([a.rename("canon"), b.rename("arm")], axis=1, join="inner")
    df.attrs["n_arm_rows"] = int(len(b))
    return df


# --------------------------------------------------------------------------
# One row
# --------------------------------------------------------------------------
def arm_row(case: str, model: str, arm: str) -> Optional[Dict[str, Any]]:
    """Every number of one (case, model, arm) cell, or None when it is absent."""
    df = paired(case, model, arm)
    if df is None or df.empty:
        return None
    stats = R.raw_agreement(df["canon"], df["arm"])
    row: Dict[str, Any] = {
        "arm": arm,
        "case": R.RAW_CASE_LABEL.get(case, case),
        "canon_case": case,
        "model": model,
        "n_pairs": stats["n_pairs"],
        "match_rate": (stats["n_pairs"] / df.attrs["n_arm_rows"]
                       if df.attrs.get("n_arm_rows") else float("nan")),
        "abstain_canon": stats["abstain_a"],
        "abstain_arm": stats["abstain_b"],
        "label_agreement": stats["label_agreement"],
        "label_kappa": stats["label_kappa"],
        "n_both": stats["n_both"],
        "thin": stats["n_both"] < R.MIN_BOTH,
        "direction_agreement": stats["direction_agreement"],
        "direction_kappa": stats["direction_kappa"],
        "layout_confounded": (arm, model) in LAYOUT_CONFOUNDED,
    }
    row.update(_score_shift(df))
    return row


def _score_shift(df: pd.DataFrame) -> Dict[str, float]:
    """The paired change of the ordinal score, over the pairs both arms answer.

    A pair where either arm abstains carries no score, thus it drops. The
    standard error is the paired one: it uses the standard deviation of the
    DIFFERENCE, which is much smaller than the standard deviation of either
    arm, because the 2 arms judged the same pair.
    """
    sa = df["canon"].map(R.ORDINAL_SCORE)
    sb = df["arm"].map(R.ORDINAL_SCORE)
    both = sa.notna() & sb.notna()
    n = int(both.sum())
    out = {"mean_score_canon": float("nan"), "mean_score_arm": float("nan"),
           "score_delta": float("nan"), "score_delta_se": float("nan")}
    if n < 2:
        return out
    x = sa[both].to_numpy(dtype=float)
    y = sb[both].to_numpy(dtype=float)
    d = y - x
    out["mean_score_canon"] = float(x.mean())
    out["mean_score_arm"] = float(y.mean())
    out["score_delta"] = float(d.mean())
    out["score_delta_se"] = float(d.std(ddof=1) / np.sqrt(n))
    return out


def arm_table(arm: str, cases: Sequence[str] = CASES,
              models: Sequence[str] = MODELS) -> pd.DataFrame:
    """One row for each (case, model) of one arm."""
    rows = [r for case in cases for model in models
            if (r := arm_row(case, model, arm)) is not None]
    return pd.DataFrame(rows)


def all_arms(arms: Sequence[str] = ABLATION_ARMS,
             cases: Sequence[str] = CASES,
             models: Sequence[str] = MODELS) -> pd.DataFrame:
    """The rows of every ablation arm, in one frame."""
    frames = [arm_table(a, cases, models) for a in arms]
    frames = [f for f in frames if not f.empty]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())


# --------------------------------------------------------------------------
# Where the abstentions went
# --------------------------------------------------------------------------
def abstention_flow(case: str, model: str,
                    arm: str = "noabstain") -> Optional[pd.DataFrame]:
    """What the model wrote instead, on the pairs it abstained on before.

    This is the question the forced-choice battery exists to answer. The rows
    are the pairs the CANONICAL arm answered `NotSure`, and the columns are the
    share of them that the ablation arm gave to each label.

    A flat spread over the 5 labels says the abstention hid no judgment. A
    spread that leans one way says the model held an opinion and declined to
    give it.
    """
    df = paired(case, model, arm)
    if df is None or df.empty:
        return None
    held = df[df["canon"].eq(R.NOT_SURE)]
    if held.empty:
        return None
    share = held["arm"].value_counts(normalize=True)
    order = [k for k in R.ORDINAL_SCORE] + [R.NOT_SURE]
    return pd.DataFrame({
        "label": [k for k in order if k in share.index],
        "share": [float(share[k]) for k in order if k in share.index],
        "n": [int((held["arm"] == k).sum()) for k in order if k in share.index],
    })


def abstention_flow_table(cases: Sequence[str] = CASES,
                          models: Sequence[str] = MODELS,
                          arm: str = "noabstain") -> pd.DataFrame:
    """One row for each (case, model): where the held-back judgments landed.

    `lean` is the mean ordinal score of the forced labels. It is 0 when the
    forced choices split evenly around `Same`, positive when the model calls
    image B better, and negative the other way. A `lean` far from 0 says the
    abstention was not indecision.
    """
    rows = []
    for case in cases:
        for model in models:
            df = paired(case, model, arm)
            if df is None or df.empty:
                continue
            held = df[df["canon"].eq(R.NOT_SURE)]
            if held.empty:
                continue
            s = held["arm"].map(R.ORDINAL_SCORE)
            share = held["arm"].value_counts(normalize=True)
            rows.append({
                "case": R.RAW_CASE_LABEL.get(case, case),
                "canon_case": case,
                "model": model,
                "n_held": int(len(held)),
                "held_share": float(len(held) / len(df)),
                **{f"to_{k}": float(share.get(k, 0.0))
                   for k in list(R.ORDINAL_SCORE) + [R.NOT_SURE]},
                "lean": float(s.mean()) if s.notna().any() else float("nan"),
                "still_notsure": float(share.get(R.NOT_SURE, 0.0)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def _fmt(x: float, nd: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def to_display(df: pd.DataFrame, min_both: int = R.MIN_BOTH) -> pd.DataFrame:
    """A plain frame of the ablation rows, for reading on screen.

    Warning: a row under `min_both` ANSWERED pairs shows no direction and no
    score. Both arms must answer a pair for it to carry a score, thus a case
    where a model nearly always abstains leaves a handful of pairs, and a mean
    over them describes the selection and not the case. The label columns take
    no floor, because they run over every pair.
    """
    if df.empty:
        return df
    thin = df["n_both"] < min_both
    return pd.DataFrame({
        "arm": df["arm"].map(ARM_LABEL).fillna(df["arm"]),
        "case": df["case"],
        "model": df["model"],
        "n": df["n_pairs"].astype(int),
        "match": df["match_rate"].map(lambda v: _fmt(v, 3)),
        "abstain canon": df["abstain_canon"].map(_fmt),
        "abstain arm": df["abstain_arm"].map(_fmt),
        "label agr": df["label_agreement"].map(_fmt),
        "label kappa": df["label_kappa"].map(_fmt),
        "both answer": df["n_both"].astype(int),
        "dir kappa": df["direction_kappa"].mask(thin).map(_fmt),
        "score canon": df["mean_score_canon"].mask(thin).map(lambda v: _fmt(v, 3)),
        "score arm": df["mean_score_arm"].mask(thin).map(lambda v: _fmt(v, 3)),
        "delta": df["score_delta"].mask(thin).map(lambda v: _fmt(v, 3)),
        "se": df["score_delta_se"].mask(thin).map(lambda v: _fmt(v, 4)),
        "layout conf.": df["layout_confounded"].map({True: "yes", False: ""}),
    })
