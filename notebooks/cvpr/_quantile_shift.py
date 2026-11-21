"""The quantile shift of one prompt arm against the canonical arm.

The ablation table in `_ablations.py` gives one number for the change of the
ordinal score: `score_delta`, the paired mean. That number is blind to the
change this battery exists to measure. A pairwise score is antisymmetric — each
comparison adds `+s` to one place and `-s` to the other — thus the mean over
places is near 0 whatever the prompt does. A prompt change that compresses the
scores toward the middle, or that stretches the tails, moves NO mean.

This module reads the change rank by rank instead. It sorts the place scores of
each arm, and `delta(tau)` is the difference of the 2 sorted lists at the same
rank. It is signed, it is per rank, and it is in the units of the score. A
positive `delta` at a low rank beside a negative `delta` at a high rank says
that the arm pulled the 2 ends toward the middle. That is the reading a mean
cannot give.

`spread_ratio` says the same thing in one number: the standard deviation of the
arm over the standard deviation of the canonical arm. A value of 0.73 says the
arm compressed the place scores by 27%. It is the statistic that the `spread`
column of `_proxy_arms.py` reports, thus a reader can hold the 2 tables side by
side.

The sorted-rank read is blind to the identity of a place: a shuffle of the
scores inside one arm moves no `delta`. `rank_tau` reads that other axis. It is
the Kendall tau-b between the paired place scores of the 2 arms, thus 1.0 says
the arm kept the order of the places while it moved the scale.

Version 2.0.0 removes an earlier decomposition of the 1-Wasserstein distance.
The antisymmetry holds the shift term near 0, thus `reshape` only repeated `w1`
and its share was near 100% in every cell. The sorted-rank deltas and the
spread ratio give the same reading directly.

The variable
------------
The unit is a PLACE, not a comparison. A comparison carries 1 of 5 ordinal
labels, and a quantile of 5 values is a step function, so the method needs the
continuous variable that sits one level up: the mean score of a place over
every comparison it took part in. This is the same quantity that
`_provenance.score_units` builds and that the validation-by-proxy table reads
after the geography join.

A place is a facility where the pairs table names one (`unit_uid`), and an
image where it does not (`sample_id`).

**Warning: 2 of the 7 cases have no continuous score.** Road quality and street
photography pair random street-level images, thus a place is 1 image and it
takes about 4 comparisons. A mean of 4 ordinal values takes 17 distinct values,
and its quantiles step. `MIN_COMPARISONS` drops such a place, and those 2 cases
then hold too few places for a quantile. They are absent from the table on
purpose, and `skipped()` says why.

The pairing
-----------
The table reads the places that BOTH arms score. The canonical arm abstains,
thus it scores fewer comparisons for each place and it can lose a place to the
`MIN_COMPARISONS` floor that the arm keeps. A distribution over a different
place set would read that difference as a prompt effect. `n_arm_only` and
`n_canon_only` report what the restriction costs.

The interval
------------
The interval on `delta(tau)`, on `spread_ratio`, and on `rank_tau` comes from a
bootstrap over PLACES, and it resamples the SAME place index for both arms. The 2 arms scored
the same places, so an unpaired interval would throw that away and read much
too wide.

See `vlm-narratives-docs/quantile-shift.md`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _ablations as A
import _canonical as C
import _results_table as R
import _style as S
from _words_across import CASE_COLOUR

__version__ = "2.1.0"

#: The 2 quantiles the figure reports. The tails carry the effect.
TAUS_HEADLINE: Tuple[float, ...] = (0.05, 0.95)

#: The quantiles the table keeps. The middle 3 show that the tails move while
#: the body does not.
TAUS_REPORT: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: The tau grid of the plotted curve. Only the figure reads it.
GRID_N = 999

#: A place under this many comparisons carries a stepped mean, not a
#: continuous score. See the module docstring.
MIN_COMPARISONS = 30

#: A case under this many shared places has no readable quantile. The report
#: stops at 0.05 and 0.95, so 30 places keep each reported rank away from the
#: end of the sorted list.
MIN_PLACES = 30

DEFAULT_ARM = "noabstain"

BOOT_N = 400
BOOT_SEED = 20260828

CASES: Sequence[str] = A.CASES
MODELS: Sequence[str] = A.MODELS


# --------------------------------------------------------------------------
# The place scores
# --------------------------------------------------------------------------
def _place_key(pairs_link: str) -> Tuple[str, str, str]:
    """Name the 2 columns that hold the place of each side, and the kind.

    A facility case names `unit_uid`. An image case names none, and then the
    place is the image itself, `sample_id`.
    """
    import pyarrow.parquet as pq

    cols = set(pq.ParquetFile(pairs_link).schema.names)
    if {"unit_uid_a", "unit_uid_b"} <= cols:
        return "unit_uid_a", "unit_uid_b", "facility"
    return "sample_id_a", "sample_id_b", "image"


@lru_cache(maxsize=64)
def place_scores(case: str, model: str, arm: str,
                 min_comparisons: int = MIN_COMPARISONS
                 ) -> Optional[pd.DataFrame]:
    """The mean ordinal score of each place of 1 run.

    The frame is indexed by the place id and holds `mean_score` and
    `n_comparisons`. `relative_label` already states side A against side B, so
    the presentation order is undone and the sign flips for side B.

    An abstention maps to no score and drops. It is not a zero.

    Returns None when the registry holds no such cell.
    """
    hits = [r for r in C.runs(kind="proxy", case=case, model=model, arm=arm)
            if r.case == case]
    if not hits:
        return None
    run = hits[0]
    key_a, key_b, _kind = _place_key(run.pairs_link)

    res = pd.read_parquet(run.results_link,
                          columns=["pair_id", "relative_label"])
    pairs = pd.read_parquet(run.pairs_link, columns=["pair_id", key_a, key_b])
    df = res.merge(pairs, on="pair_id", how="left", validate="one_to_one")

    score = df["relative_label"].astype(str).map(R.ORDINAL_SCORE)
    ok = score.notna()
    long = pd.concat([
        pd.DataFrame({"place": df.loc[ok, key_a], "score": score[ok]}),
        pd.DataFrame({"place": df.loc[ok, key_b], "score": -score[ok]}),
    ], ignore_index=True).dropna(subset=["place"])

    g = long.groupby("place")["score"].agg(
        mean_score="mean", n_comparisons="size")
    return g[g["n_comparisons"] >= min_comparisons]


def place_kind(case: str, model: str = "gemma-4-12b",
               arm: str = DEFAULT_ARM) -> str:
    """Say whether a place of this case is a facility or an image."""
    hits = [r for r in C.runs(kind="proxy", case=case, model=model, arm=arm)
            if r.case == case]
    return _place_key(hits[0].pairs_link)[2] if hits else "unknown"


def paired_places(case: str, model: str, arm: str = DEFAULT_ARM,
                  min_comparisons: int = MIN_COMPARISONS
                  ) -> Optional[pd.DataFrame]:
    """Join the 2 arms on the places that BOTH of them score.

    The frame holds `canon` and `arm`, 1 row for each shared place. Its `attrs`
    carry `n_canon_only` and `n_arm_only`, which say what the join dropped.
    """
    q = place_scores(case, model, A.CONTROL_ARM, min_comparisons)
    p = place_scores(case, model, arm, min_comparisons)
    if q is None or p is None or q.empty or p.empty:
        return None
    shared = q.index.intersection(p.index)
    out = pd.DataFrame({
        "canon": q.loc[shared, "mean_score"],
        "arm": p.loc[shared, "mean_score"],
        "n_canon": q.loc[shared, "n_comparisons"],
        "n_arm": p.loc[shared, "n_comparisons"],
    })
    out.attrs["n_canon_only"] = int(len(q.index.difference(p.index)))
    out.attrs["n_arm_only"] = int(len(p.index.difference(q.index)))
    return out


# --------------------------------------------------------------------------
# The quantile difference
# --------------------------------------------------------------------------
def quantile_delta(arm: np.ndarray, canon: np.ndarray,
                   taus: Sequence[float]) -> np.ndarray:
    """`delta(tau)`, the arm quantile minus the canonical quantile.

    The 2 samples hold the same places, thus they have the same size. The
    tau-quantile is then a position in the sorted list of a sample, and
    `delta(tau)` reads the 2 sorted score lists at the SAME position.

    A positive value says the arm puts its tau-quantile HIGHER.
    """
    t = np.asarray(taus, dtype=float)
    return np.quantile(arm, t) - np.quantile(canon, t)


def _boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _spread_ratio(arm: np.ndarray, canon: np.ndarray) -> float:
    """The standard deviation of the arm over that of the canonical arm.

    Under 1 the arm squeezed the place scores together. This is the statistic
    that the `spread` column of `_proxy_arms.py` reports.
    """
    sd_c = float(np.std(canon, ddof=1))
    return float(np.std(arm, ddof=1) / sd_c) if sd_c > 0 else float("nan")


def _rank_tau(arm: np.ndarray, canon: np.ndarray) -> float:
    """The Kendall tau-b between the paired place scores of the 2 arms.

    `delta(tau)` sorts each arm on its own, thus it cannot see a place that
    moves in the order. This can. A value of 1.0 says the arm kept the order
    of the places. `_results_table.py` reads the same variant of tau against
    the proxies.
    """
    from scipy import stats

    if len(arm) < 3 or np.std(arm) == 0 or np.std(canon) == 0:
        return float("nan")
    return float(stats.kendalltau(arm, canon, variant="b").statistic)


def shift_row(case: str, model: str, arm: str = DEFAULT_ARM,
              taus: Sequence[float] = TAUS_REPORT,
              min_comparisons: int = MIN_COMPARISONS,
              n_boot: int = BOOT_N, seed: int = BOOT_SEED
              ) -> Optional[Dict[str, Any]]:
    """Every number of one (case, model, arm) cell, or None when it is thin.

    The interval is a paired bootstrap over places: one resample draws a place
    index and reads BOTH arms at it, so the interval keeps the pairing.
    """
    df = paired_places(case, model, arm, min_comparisons)
    if df is None or len(df) < MIN_PLACES:
        return None
    a = df["arm"].to_numpy(dtype=float)
    q = df["canon"].to_numpy(dtype=float)
    t = np.asarray(taus, dtype=float)

    row: Dict[str, Any] = {
        "arm": arm,
        "case": R.RAW_CASE_LABEL.get(case, case),
        "canon_case": case,
        "model": model,
        "place_kind": place_kind(case, model, arm),
        "n_places": int(len(df)),
        "n_canon_only": int(df.attrs.get("n_canon_only", 0)),
        "n_arm_only": int(df.attrs.get("n_arm_only", 0)),
        "median_comparisons": float(df["n_canon"].median()),
        "sd_canon": float(q.std(ddof=1)),
        "sd_arm": float(a.std(ddof=1)),
        "layout_confounded": (arm, model) in A.LAYOUT_CONFOUNDED,
    }
    # The spread of the place scores, arm over canonical. 0.73 says the arm
    # compressed them by 27%. `_proxy_arms.py` reports the same statistic.
    row["spread_ratio"] = _spread_ratio(a, q)
    # The order of the places, which the sorted-rank deltas cannot see.
    row["rank_tau"] = _rank_tau(a, q)

    delta = quantile_delta(a, q, t)
    for tau, d in zip(t, delta):
        row[f"d{int(round(tau * 100)):02d}"] = float(d)

    idx = _boot_indices(len(df), n_boot, seed)
    b_delta = np.empty((n_boot, len(t)), dtype=float)
    b_ratio = np.empty(n_boot, dtype=float)
    b_tau = np.empty(n_boot, dtype=float)
    for i, ix in enumerate(idx):
        aa, qq = a[ix], q[ix]
        b_delta[i] = quantile_delta(aa, qq, t)
        b_ratio[i] = _spread_ratio(aa, qq)
        b_tau[i] = _rank_tau(aa, qq)
    for j, tau in enumerate(t):
        lo, hi = _ci(b_delta[:, j])
        stem = f"d{int(round(tau * 100)):02d}"
        row[f"{stem}_lo"], row[f"{stem}_hi"] = lo, hi
    row["spread_ratio_lo"], row["spread_ratio_hi"] = _ci(b_ratio)
    row["rank_tau_lo"], row["rank_tau_hi"] = _ci(b_tau)
    row["n_boot"] = int(n_boot)
    return row


def shift_table(arm: str = DEFAULT_ARM, cases: Sequence[str] = CASES,
                models: Sequence[str] = MODELS,
                taus: Sequence[float] = TAUS_REPORT,
                min_comparisons: int = MIN_COMPARISONS,
                n_boot: int = BOOT_N) -> pd.DataFrame:
    """One row for each (case, model) that holds enough places."""
    rows = [r for case in cases for model in models
            if (r := shift_row(case, model, arm, taus, min_comparisons,
                               n_boot)) is not None]
    return pd.DataFrame(rows)


def skipped(arm: str = DEFAULT_ARM, cases: Sequence[str] = CASES,
            models: Sequence[str] = MODELS,
            min_comparisons: int = MIN_COMPARISONS) -> pd.DataFrame:
    """Name each cell the table drops, and say why.

    A reader must see that the absent cases are absent for a stated reason and
    not because a run is missing.
    """
    rows = []
    for case in cases:
        for model in models:
            df = paired_places(case, model, arm, min_comparisons)
            if df is not None and len(df) >= MIN_PLACES:
                continue
            kind = place_kind(case, model, arm)
            n = 0 if df is None else int(len(df))
            raw = place_scores(case, model, arm, 1)
            rows.append({
                "case": R.RAW_CASE_LABEL.get(case, case),
                "canon_case": case,
                "model": model,
                "place_kind": kind,
                "places_all": 0 if raw is None else int(len(raw)),
                "median_comparisons": (float("nan") if raw is None
                                       else float(raw["n_comparisons"].median())),
                "places_kept": n,
                "reason": (
                    "no run in the registry" if raw is None else
                    f"a place takes too few comparisons for a continuous "
                    f"score; under {min_comparisons} it steps"),
            })
    return pd.DataFrame(rows)


def delta_curve(case: str, model: str, arm: str = DEFAULT_ARM,
                grid_n: int = GRID_N,
                min_comparisons: int = MIN_COMPARISONS,
                n_boot: int = BOOT_N, seed: int = BOOT_SEED
                ) -> Optional[pd.DataFrame]:
    """`delta(tau)` on a grid, with a paired bootstrap band."""
    df = paired_places(case, model, arm, min_comparisons)
    if df is None or len(df) < MIN_PLACES:
        return None
    a = df["arm"].to_numpy(dtype=float)
    q = df["canon"].to_numpy(dtype=float)
    t = np.linspace(1.0 / (grid_n + 1), grid_n / (grid_n + 1), grid_n)

    out = pd.DataFrame({"tau": t, "delta": quantile_delta(a, q, t)})
    idx = _boot_indices(len(df), n_boot, seed)
    band = np.empty((n_boot, grid_n), dtype=float)
    for i, ix in enumerate(idx):
        band[i] = quantile_delta(a[ix], q[ix], t)
    out["lo"] = np.nanquantile(band, 0.025, axis=0)
    out["hi"] = np.nanquantile(band, 0.975, axis=0)
    out["case"] = case
    out["model"] = model
    out["arm"] = arm
    return out


def delta_curves(arm: str = DEFAULT_ARM, cases: Sequence[str] = CASES,
                 models: Sequence[str] = MODELS, grid_n: int = GRID_N,
                 min_comparisons: int = MIN_COMPARISONS,
                 n_boot: int = BOOT_N) -> pd.DataFrame:
    """The curve of every cell the table holds, in 1 frame."""
    frames = [c for case in cases for model in models
              if (c := delta_curve(case, model, arm, grid_n, min_comparisons,
                                   n_boot)) is not None]
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["tau", "delta", "lo", "hi",
                                       "case", "model", "arm"]))


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------
def _pm(v: float, lo: float, hi: float, nd: int = 3) -> str:
    """One signed value with its interval, for a screen table."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _pm_plain(v: float, lo: float, hi: float, nd: int = 2) -> str:
    """One unsigned value with its interval. A ratio carries no sign."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def to_display(df: pd.DataFrame,
               taus: Sequence[float] = TAUS_REPORT) -> pd.DataFrame:
    """A plain frame of the shift rows, for reading on screen."""
    if df.empty:
        return df
    out = {
        "case": df["case"],
        "model": df["model"],
        "places": df["n_places"].astype(int),
        "spread": df.apply(
            lambda r: _pm_plain(r["spread_ratio"], r["spread_ratio_lo"],
                                r["spread_ratio_hi"]), axis=1),
        "rank tau": df.apply(
            lambda r: _pm_plain(r["rank_tau"], r["rank_tau_lo"],
                                r["rank_tau_hi"]), axis=1),
    }
    for tau in taus:
        stem = f"d{int(round(tau * 100)):02d}"
        if stem in df.columns:
            out[f"d({tau:g})"] = df.apply(
                lambda r, s=stem: _pm(r[s], r[f"{s}_lo"], r[f"{s}_hi"]), axis=1)
    out["layout conf."] = df["layout_confounded"].map({True: "yes", False: ""})
    return pd.DataFrame(out)


CAPTION = (
    "The no-abstention arm against the canonical prompt, read as a quantile "
    "shift of the place score. Sort the place scores of each arm; "
    "$\\delta(\\tau)$ is the arm list minus the canonical list at the same "
    "rank, in ordinal-score units. spread is the ratio of the standard "
    "deviations, arm over canonical, the same statistic as the validation "
    "table: under 1 the arm squeezed the place scores together. rank "
    "$\\tau_b$ is the Kendall correlation between the paired place scores of "
    "the 2 arms: 1.0 says the arm kept the order of the places. Brackets "
    "hold a 95\\% paired bootstrap interval over places."
)

DEFAULT_LABEL = "tab:quantile-shift"


def to_latex(df: pd.DataFrame, caption: Optional[str] = None,
             label: str = DEFAULT_LABEL,
             taus: Sequence[float] = TAUS_REPORT,
             full_width: bool = True) -> str:
    """The paper table: one row for each cell, the 5 report quantiles."""
    if df.empty:
        return "% no quantile-shift row\n"
    env = "table*" if full_width else "table"
    cols = "ll r r r " + "r " * len(taus)
    head = ["Case", "Rater", "Places", "spread", "rank $\\tau_b$"]
    head += [f"$\\delta({t:g})$" for t in taus]

    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\small",
             f"\\begin{{tabular}}{{{cols.strip()}}}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = [r["case"], r["model"], f"{int(r['n_places'])}",
                 f"{r['spread_ratio']:.2f}", f"{r['rank_tau']:.2f}"]
        cells += [f"{r[f'd{int(round(t * 100)):02d}']:+.3f}" for t in taus]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{{caption or CAPTION}}}", f"\\label{{{label}}}",
              f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_delta_curves(curves: pd.DataFrame, models: Sequence[str] = MODELS,
                      taus: Sequence[float] = TAUS_HEADLINE,
                      band: bool = True, figsize=(7.0, 3.4)):
    """`delta(tau)` against tau, 1 panel for each rater and 1 line for each case.

    A flat line at 0 means the arm changed nothing. A line that crosses 0 means
    the arm moved the 2 ends of the distribution in OPPOSITE directions, which
    no mean can show.
    """
    import matplotlib.pyplot as plt

    have = [m for m in models if m in set(curves["model"])]
    fig, axes = plt.subplots(1, max(len(have), 1), figsize=figsize,
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, have):
        sub = curves[curves["model"] == model]
        ax.axhline(0.0, color=S.EDGE, lw=0.7, zorder=1)
        for tau in taus:
            ax.axvline(tau, color=S.PAL["warmgrey"], lw=0.5, ls=":", zorder=1)
        for case, g in sub.groupby("case", sort=False):
            colour = CASE_COLOUR.get(case, S.PAL["warmgrey"])
            if band:
                ax.fill_between(g["tau"], g["lo"], g["hi"], color=colour,
                                alpha=0.16, lw=0, zorder=2)
            ax.plot(g["tau"], g["delta"], color=colour, lw=1.5, zorder=3,
                    label=R.RAW_CASE_LABEL.get(case, case))
        ax.set_xlabel(f"quantile $\\tau$ — {model}")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel(r"$\delta(\tau)$  (score units)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return fig


def plot_headline(df: pd.DataFrame, taus: Sequence[float] = TAUS_HEADLINE,
                  figsize=(6.6, 3.8)):
    """The 2 headline quantiles of each cell, with the bootstrap interval.

    The rows are the cells and the colours are the quantiles, so a reader sees
    at once whether the low quantile and the high quantile moved the same way
    or against each other.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("no row to plot")
    rows = df.reset_index(drop=True)
    y = np.arange(len(rows))
    ramp = [S.CMAP_DIV(v) for v in np.linspace(0.08, 0.92, len(taus))]
    off = np.linspace(-0.28, 0.28, len(taus))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, lw=0.8, zorder=1)
    for k, tau in enumerate(taus):
        stem = f"d{int(round(tau * 100)):02d}"
        v = rows[stem].to_numpy(dtype=float)
        lo = rows[f"{stem}_lo"].to_numpy(dtype=float)
        hi = rows[f"{stem}_hi"].to_numpy(dtype=float)
        yy = y + off[k]
        ax.hlines(yy, lo, hi, color=ramp[k], lw=1.4, zorder=2)
        ax.scatter(v, yy, s=22, color=ramp[k], edgecolors=S.EDGE,
                   linewidths=S.EDGE_LW, zorder=3,
                   label=f"$\\tau={tau:g}$")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['case']} — {r['model']}" for _, r in rows.iterrows()],
                       fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\delta(\tau)$: no abstention minus canonical  (score units)")
    ax.legend(frameon=False, fontsize=7, ncol=len(taus), loc="lower center",
              bbox_to_anchor=(0.5, -0.26))
    fig.tight_layout()
    return fig


def export(table: pd.DataFrame, curves: pd.DataFrame, out_dir,
           tag: str = "quantile_shift", version: str = "2.1.0",
           taus: Sequence[float] = TAUS_HEADLINE,
           table_taus: Sequence[float] = TAUS_REPORT,
           skipped_rows: Optional[pd.DataFrame] = None) -> List[str]:
    """Write the frames, the LaTeX, and the 2 figures. Give back the names.

    `taus` names the quantiles of the headline figure. `table_taus` names the
    quantiles of the LaTeX table, which holds more of them.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{tag}_v{version}"
    written: List[str] = []

    table.to_csv(out / f"{stem}.csv", index=False)
    written.append(f"{stem}.csv")
    curves.to_csv(out / f"{stem}_curves.csv", index=False)
    written.append(f"{stem}_curves.csv")
    (out / f"{stem}.tex").write_text(to_latex(table, taus=table_taus))
    written.append(f"{stem}.tex")
    if skipped_rows is not None and not skipped_rows.empty:
        skipped_rows.to_csv(out / f"{stem}_skipped.csv", index=False)
        written.append(f"{stem}_skipped.csv")

    if not curves.empty:
        fig = plot_delta_curves(curves, taus=taus)
        fig.savefig(out / f"{stem}_curves.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{stem}_curves.png")
    if not table.empty:
        fig = plot_headline(table, taus=taus)
        fig.savefig(out / f"{stem}_headline.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(f"{stem}_headline.png")
    return written
