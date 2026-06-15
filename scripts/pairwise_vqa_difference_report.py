#!/usr/bin/env python3
"""On-the-fly group-difference testing for a urbanpairvqa run.

Answers questions like "would the VLM rather eat at Chinese restaurants than
Italian ones?" from a pairwise VQA stage output. Units (restaurants,
libraries, ...) are assigned to groups via a metadata column — either one
already surfaced on ``pairs.parquet`` (as ``<col>_a`` / ``<col>_b``) or
joined from an external unit-metadata parquet keyed on ``unit_uid`` — and
two complementary tests are run per group pair:

  * **Head-to-head** — pairs where one side is group A and the other group B,
    ``relative_score`` oriented so positive = "A preferred", repeats collapsed
    by ``canonical_pair_id``. One-sample t-test vs 0, Wilcoxon signed-rank,
    binomial sign test, win rate with Wilson CI.
  * **Rating-level** — per-unit TrueSkill μ from the *full* run, Welch's
    t-test + Mann-Whitney U between the groups' μ distributions. Works even
    when direct A-vs-B pairs are sparse; approximate (μ's are coupled through
    the comparison graph).

Modes:

  * Single comparison: ``--group-a Chinese --group-b Italian``
  * Matrix: ``--all-pairs`` (every group pair, Benjamini-Hochberg corrected,
    plus a Kruskal-Wallis omnibus over the per-group μ distributions)
  * ``--list``: print past experiments from the local registry

Every completed experiment is recorded in an append-only JSONL registry
(default ``machine-beholder/difference_tests/registry.jsonl``) keyed by a
deterministic experiment id; reruns are detected and skipped unless
``--force``. Each experiment is also mirrored to W&B (separate analysis
project, default ``URBANPAIRVQA-ANALYSIS``) with summary metrics, the tidy
results table, and the report artifacts. W&B failures are non-fatal.

Examples:

    # Single cuisine comparison on the restaurants sweep (cuisine joined
    # externally — it is not on pairs.parquet)
    python scripts/pairwise_vqa_difference_report.py \\
        .../outputs/pairwise/restaurants_mvp_20260501_115622.parquet \\
        --group-column cuisine_description \\
        --unit-metadata-parquet curation/dohmh_restaurants_inspected_all/restaurants_aggregated.parquet \\
        --unit-metadata-id-column camis \\
        --group-a Chinese --group-b Italian \\
        --attribute "rather eat at" --unit-label restaurant --pdf

    # All-pairs matrix over the 12 biggest cuisines
    python scripts/pairwise_vqa_difference_report.py \\
        .../restaurants_mvp_20260501_115622.parquet \\
        --group-column cuisine_description \\
        --unit-metadata-parquet .../restaurants_aggregated.parquet \\
        --unit-metadata-id-column camis \\
        --all-pairs --top-k-groups 12 --pdf

    # What have we already tested?
    python scripts/pairwise_vqa_difference_report.py --list
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

# Sibling helpers. Importing `pairwise_vqa_report` also applies the shared
# matplotlib rcParams polish as a side effect.
sys.path.insert(0, str(Path(__file__).parent))
from pairwise_vqa_report import (  # noqa: E402
    ACCENT,
    ACCENT_WARM,
    _compute_trueskill,
    _export_pdf,
    _load_run_config,
    _md_table,
)
from pairwise_analysis_common import (  # noqa: E402,F401  (re-exported for tests)
    DEFAULT_REGISTRY,
    DEFAULT_WANDB_PROJECT,
    _centered_image,
    _clean_group_series,
    _fmt_p,
    _slug,
    _stars,
    adjust_pvalues,
    append_registry,
    attach_groups,
    compute_experiment_id,
    find_in_registry,
    list_registry,
    load_run,
    mirror_to_wandb,
    read_registry,
    registry_path,
)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "output_parquet",
        type=Path,
        nargs="?",
        default=None,
        help="Stage output parquet (with pair_id, relative_score). Not needed with --list.",
    )
    p.add_argument(
        "--pairs-parquet",
        type=Path,
        default=None,
        help="Companion pairs.parquet. Default: sibling 'pairs.parquet'.",
    )
    p.add_argument(
        "--aggregation-dir",
        type=Path,
        default=None,
        help=(
            "Multi-model mode: an aggregation directory of per-model runs "
            "(same layouts as pairwise_vqa_aggregation_report.py). Tests run "
            "per model with a cross-model replication summary. Mutually "
            "exclusive with the positional output parquet."
        ),
    )
    p.add_argument(
        "--group-column",
        type=str,
        default=None,
        help=(
            "Metadata column defining the groups (e.g. cuisine_description). "
            "Resolved from pairs.parquet '<col>_a/_b' if present, else from "
            "--unit-metadata-parquet."
        ),
    )
    p.add_argument(
        "--unit-metadata-parquet",
        type=Path,
        default=None,
        help="External unit-metadata parquet to join the group column from.",
    )
    p.add_argument(
        "--unit-metadata-id-column",
        type=str,
        default="unit_uid",
        help=(
            "Id column in --unit-metadata-parquet matched (as string) against "
            "pair unit_uid (e.g. 'camis' for DOHMH restaurants). Default: unit_uid."
        ),
    )
    # ---- comparison selection ----
    p.add_argument("--group-a", type=str, default=None, help="First group (single-comparison mode).")
    p.add_argument("--group-b", type=str, default=None, help="Second group (single-comparison mode).")
    p.add_argument(
        "--all-pairs",
        action="store_true",
        help="Matrix mode: test every pair of eligible groups with BH correction.",
    )
    p.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated explicit group list for --all-pairs (default: all eligible).",
    )
    p.add_argument(
        "--top-k-groups",
        type=int,
        default=None,
        help="With --all-pairs: keep only the K groups with the most rated units.",
    )
    p.add_argument(
        "--min-group-units",
        type=int,
        default=20,
        help="Minimum rated units for a group to be eligible. Default: 20.",
    )
    # ---- stats knobs ----
    p.add_argument(
        "--draw-prob",
        type=float,
        default=0.05,
        help="TrueSkill draw probability. Default: 0.05 (matches pairwise_vqa_report).",
    )
    p.add_argument(
        "--min-comparisons",
        type=int,
        default=1,
        help="Minimum comparisons for a unit's μ to enter the rating-level test. Default: 1.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for stars / 'significant' counts. Default: 0.05.",
    )
    # ---- report cosmetics ----
    p.add_argument("--title", type=str, default=None, help="Report title.")
    p.add_argument(
        "--attribute",
        type=str,
        default="more of the attribute",
        help='Attribute phrase (e.g. "rather eat at"). Used in prose only.',
    )
    p.add_argument("--unit-label", type=str, default="unit", help="Singular noun for the rated entity.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Default: <output_parquet_dir>/<stem>.difference.<slug>.md",
    )
    p.add_argument("--pdf", action="store_true", help="Also render to PDF via pandoc.")
    p.add_argument("--pdf-engine", type=str, default="xelatex")
    p.add_argument(
        "--pdf-orientation", choices=("landscape", "portrait"), default="landscape"
    )
    # ---- registry ----
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help=(
            "Registry JSONL path. Default: $MLLMSCI_DIFFTEST_REGISTRY or "
            f"{DEFAULT_REGISTRY}"
        ),
    )
    p.add_argument("--no-registry", action="store_true", help="Skip registry read/write.")
    p.add_argument("--force", action="store_true", help="Rerun even if the experiment is registered.")
    p.add_argument("--list", action="store_true", help="List registered experiments and exit.")
    # ---- wandb ----
    p.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_ANALYSIS_PROJECT", DEFAULT_WANDB_PROJECT),
        help=f"W&B project for the analysis mirror. Default: {DEFAULT_WANDB_PROJECT}.",
    )
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity. Default: $WANDB_ENTITY, else 'urbanekg'.",
    )
    p.add_argument("--no-wandb", action="store_true", help="Skip the W&B mirror.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Group resolution (difference-testing specific; generic loaders live in
# pairwise_analysis_common)
# ---------------------------------------------------------------------------


def build_unit_group_map(df: pd.DataFrame) -> pd.Series:
    """unit_uid → group from both pair sides. Conflicts keep the first value."""
    a = df[["unit_uid_a", "__group_a__"]].rename(
        columns={"unit_uid_a": "unit_uid", "__group_a__": "group"}
    )
    b = df[["unit_uid_b", "__group_b__"]].rename(
        columns={"unit_uid_b": "unit_uid", "__group_b__": "group"}
    )
    long = pd.concat([a, b], ignore_index=True).dropna(subset=["group"])
    long["unit_uid"] = long["unit_uid"].astype(str)
    n_groups_per_unit = long.groupby("unit_uid")["group"].nunique()
    n_conflicts = int((n_groups_per_unit > 1).sum())
    if n_conflicts:
        print(f"[WARN] {n_conflicts:,} units map to >1 group value; keeping the first seen.")
    return long.drop_duplicates("unit_uid").set_index("unit_uid")["group"]


def resolve_group_name(requested: str, observed: pd.Series) -> str:
    """Case-insensitively match a user-supplied group name to an observed value."""
    counts = observed.value_counts()
    lut = {}
    for val in counts.index:
        lut.setdefault(str(val).casefold(), str(val))
    hit = lut.get(requested.strip().casefold())
    if hit is None:
        top = ", ".join(f"{v!r} ({n})" for v, n in counts.head(20).items())
        raise SystemExit(
            f"Group {requested!r} not found in the group column. "
            f"Top observed values: {top}"
        )
    return hit


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _cohens_d_one_sample(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return float(np.mean(x)) / sd if sd > 0 else float("nan")


def _cohens_d_two_sample(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = math.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def head_to_head_test(df: pd.DataFrame, group_a: str, group_b: str) -> Dict[str, Any]:
    """Direct A-vs-B pairs: orient relative_score so positive = group_a
    preferred, collapse repeats by canonical_pair_id, test mean vs 0.
    """
    mask_ab = (df["__group_a__"] == group_a) & (df["__group_b__"] == group_b)
    mask_ba = (df["__group_a__"] == group_b) & (df["__group_b__"] == group_a)
    oriented = pd.concat(
        [
            df.loc[mask_ab, ["canonical_pair_id", "relative_score"]].assign(
                oriented=lambda d: d["relative_score"].astype(float)
            ),
            df.loc[mask_ba, ["canonical_pair_id", "relative_score"]].assign(
                oriented=lambda d: -d["relative_score"].astype(float)
            ),
        ],
        ignore_index=True,
    )
    n_raw = len(oriented)
    collapsed = oriented.groupby("canonical_pair_id")["oriented"].mean()
    x = collapsed.to_numpy(dtype=float)
    n = x.size

    res: Dict[str, Any] = {
        "n_direct_obs": n_raw,
        "n_direct_pairs": n,
        "mean_oriented": float(np.mean(x)) if n else float("nan"),
        "mean_oriented_se": (
            float(np.std(x, ddof=1) / math.sqrt(n)) if n >= 2 else float("nan")
        ),
        "h2h_t": float("nan"),
        "h2h_p": float("nan"),
        "h2h_wilcoxon_p": float("nan"),
        "h2h_sign_p": float("nan"),
        "win_rate": float("nan"),
        "win_ci_lo": float("nan"),
        "win_ci_hi": float("nan"),
        "h2h_cohens_d": float("nan"),
    }
    if n >= 2 and float(np.std(x, ddof=1)) > 0:
        t = sps.ttest_1samp(x, 0.0)
        res["h2h_t"] = float(t.statistic)
        res["h2h_p"] = float(t.pvalue)
        res["h2h_cohens_d"] = _cohens_d_one_sample(x)
        nonzero = x[x != 0]
        if nonzero.size >= 1:
            try:
                w = sps.wilcoxon(nonzero)
                res["h2h_wilcoxon_p"] = float(w.pvalue)
            except ValueError:
                pass
    if n >= 1:
        wins = int((x > 0).sum())
        losses = int((x < 0).sum())
        decided = wins + losses
        if decided:
            res["win_rate"] = wins / decided
            res["win_ci_lo"], res["win_ci_hi"] = _wilson_ci(wins, decided)
            res["h2h_sign_p"] = float(sps.binomtest(wins, decided, 0.5).pvalue)
    return res


def rating_level_test(
    ratings: pd.DataFrame,
    unit_groups: pd.Series,
    group_a: str,
    group_b: str,
    min_comparisons: int = 1,
) -> Dict[str, Any]:
    """Welch's t + Mann-Whitney U between the two groups' TrueSkill μ values."""
    r = ratings[ratings["n_comparisons"] >= min_comparisons].copy()
    r["group"] = r["unit_uid"].astype(str).map(unit_groups)
    mu_a = r.loc[r["group"] == group_a, "mu"].to_numpy(dtype=float)
    mu_b = r.loc[r["group"] == group_b, "mu"].to_numpy(dtype=float)

    res: Dict[str, Any] = {
        "n_units_a": int(mu_a.size),
        "n_units_b": int(mu_b.size),
        "mu_mean_a": float(np.mean(mu_a)) if mu_a.size else float("nan"),
        "mu_mean_b": float(np.mean(mu_b)) if mu_b.size else float("nan"),
        "delta_mu": float("nan"),
        "delta_mu_se": float("nan"),
        "rating_t": float("nan"),
        "rating_p": float("nan"),
        "rating_mwu_p": float("nan"),
        "rating_cohens_d": float("nan"),
        "cliffs_delta": float("nan"),
    }
    if mu_a.size and mu_b.size:
        res["delta_mu"] = res["mu_mean_a"] - res["mu_mean_b"]
    if mu_a.size >= 2 and mu_b.size >= 2:
        res["delta_mu_se"] = float(
            math.sqrt(np.var(mu_a, ddof=1) / mu_a.size + np.var(mu_b, ddof=1) / mu_b.size)
        )
        t = sps.ttest_ind(mu_a, mu_b, equal_var=False)
        res["rating_t"] = float(t.statistic)
        res["rating_p"] = float(t.pvalue)
        res["rating_cohens_d"] = _cohens_d_two_sample(mu_a, mu_b)
        u = sps.mannwhitneyu(mu_a, mu_b, alternative="two-sided")
        res["rating_mwu_p"] = float(u.pvalue)
        res["cliffs_delta"] = float(2.0 * u.statistic / (mu_a.size * mu_b.size) - 1.0)
    return res


def run_comparison(
    df: pd.DataFrame,
    ratings: pd.DataFrame,
    unit_groups: pd.Series,
    group_a: str,
    group_b: str,
    min_comparisons: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"group_a": group_a, "group_b": group_b}
    row.update(head_to_head_test(df, group_a, group_b))
    row.update(rating_level_test(ratings, unit_groups, group_a, group_b, min_comparisons))
    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_oriented_hist(
    df: pd.DataFrame, group_a: str, group_b: str, out_path: Path
) -> bool:
    mask_ab = (df["__group_a__"] == group_a) & (df["__group_b__"] == group_b)
    mask_ba = (df["__group_a__"] == group_b) & (df["__group_b__"] == group_a)
    oriented = pd.concat(
        [
            df.loc[mask_ab, "relative_score"].astype(float),
            -df.loc[mask_ba, "relative_score"].astype(float),
        ]
    ).to_numpy()
    if oriented.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = np.arange(-2.5, 3.0, 1.0)
    ax.hist(oriented, bins=bins, color=ACCENT, edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.axvline(
        float(oriented.mean()),
        color="#c0392b",
        linestyle="-",
        linewidth=1.4,
        label=f"mean = {oriented.mean():+.3f}",
    )
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([
        f"MuchLess\n({group_b} ≫)", "Less", "Same", "More", f"MuchMore\n({group_a} ≫)",
    ], fontsize=9)
    ax.set_ylabel("count (raw observations)")
    ax.set_title(f"Head-to-head oriented scores  ·  {group_a} vs {group_b}  ·  + = {group_a} preferred")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_group_mu(
    ratings: pd.DataFrame,
    unit_groups: pd.Series,
    groups: List[str],
    out_path: Path,
    min_comparisons: int,
) -> bool:
    r = ratings[ratings["n_comparisons"] >= min_comparisons].copy()
    r["group"] = r["unit_uid"].astype(str).map(unit_groups)
    data = [r.loc[r["group"] == g, "mu"].to_numpy(dtype=float) for g in groups]
    if not any(d.size for d in data):
        return False
    # Order by mean μ descending for readability.
    order = np.argsort([-(np.mean(d) if d.size else -np.inf) for d in data])
    groups_o = [groups[i] for i in order]
    data_o = [data[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8.0, 1.0 * len(groups_o) + 4.0), 5.2))
    bp = ax.boxplot(
        data_o,
        tick_labels=[f"{g}\n(n={d.size})" for g, d in zip(groups_o, data_o)],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        medianprops={"color": "#c0392b", "linewidth": 1.2},
        meanprops={"color": ACCENT_WARM, "linewidth": 1.2, "linestyle": ":"},
        flierprops={"markersize": 3, "alpha": 0.4},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#dce8f5")
        patch.set_edgecolor("#444444")
        patch.set_linewidth(0.7)
    ax.set_ylabel(r"TrueSkill $\mu$")
    ax.set_title(r"Per-unit TrueSkill $\mu$ by group  ·  sorted by group mean")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_matrix_heatmap(
    results: pd.DataFrame,
    groups: List[str],
    out_path: Path,
    *,
    value_col: str,
    p_adj_col: str,
    title: str,
    alpha: float,
) -> None:
    """Antisymmetric group×group heatmap of `value_col` (row preferred = +),
    annotated with significance stars from `p_adj_col`."""
    n = len(groups)
    mat = np.full((n, n), np.nan)
    padj = np.full((n, n), np.nan)
    gi = {g: i for i, g in enumerate(groups)}
    for row in results.itertuples(index=False):
        a, b = gi[row.group_a], gi[row.group_b]
        v = getattr(row, value_col)
        p = getattr(row, p_adj_col)
        mat[a, b] = v
        mat[b, a] = -v if np.isfinite(v) else np.nan
        padj[a, b] = p
        padj[b, a] = p

    finite = mat[np.isfinite(mat)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(7.0, 0.95 * n + 3.0), max(6.0, 0.85 * n + 2.5)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(groups, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(mat[i, j]):
                continue
            norm_v = abs(mat[i, j]) / vmax
            color = "white" if norm_v > 0.6 else "#111111"
            ax.text(
                j, i,
                f"{mat[i, j]:+.2f}{_stars(padj[i, j], alpha)}",
                ha="center", va="center", fontsize=8, color=color,
            )
    ax.set_title(f"{title}\n(+ = row group preferred; stars = BH-adjusted p < {alpha:g}/{alpha/5:g}/{alpha/50:g})")
    ax.tick_params(axis="both", length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-model: replication + plots
# ---------------------------------------------------------------------------


def build_replication(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """One row per group comparison summarizing cross-model agreement."""
    rows: List[Dict[str, Any]] = []
    for (a, b), g in results.groupby(["group_a", "group_b"], sort=False):
        mo = g["mean_oriented"].to_numpy(dtype=float)
        dm = g["delta_mu"].to_numpy(dtype=float)
        dm_finite = dm[np.isfinite(dm)]
        rows.append({
            "group_a": a,
            "group_b": b,
            "n_models": int(len(g)),
            "mean_oriented_mean": float(np.nanmean(mo)) if np.isfinite(mo).any() else float("nan"),
            "n_toward_a_h2h": int((mo > 0).sum()),
            "n_sig_h2h": int((g["h2h_p_adj"] < alpha).sum()),
            "delta_mu_mean": float(np.nanmean(dm)) if np.isfinite(dm).any() else float("nan"),
            "n_toward_a_rating": int((dm > 0).sum()),
            "n_sig_rating": int((g["rating_p_adj"] < alpha).sum()),
            "direction_consistent_rating": bool(
                dm_finite.size and ((dm_finite > 0).all() or (dm_finite < 0).all())
            ),
        })
    return pd.DataFrame(rows)


def _plot_forest(
    results: pd.DataFrame,
    out_path: Path,
    *,
    value_col: str,
    se_col: str,
    p_col: str,
    title: str,
    xlabel: str,
    alpha: float,
) -> bool:
    """Per-model point estimate ± 1.96·SE, one row per model, stars from p."""
    sub = results[np.isfinite(results[value_col].astype(float))].copy()
    if sub.empty:
        return False
    vals = sub[value_col].to_numpy(dtype=float)
    se = sub[se_col].to_numpy(dtype=float)
    ps = sub[p_col].to_numpy(dtype=float)
    labels = sub["model_label"].tolist()
    y = np.arange(len(sub))[::-1]

    fig, ax = plt.subplots(figsize=(9.5, max(2.8, 0.55 * len(sub) + 1.6)))
    xerr = np.where(np.isfinite(se), 1.96 * se, 0.0)
    ax.errorbar(
        vals, y, xerr=xerr, fmt="o", color=ACCENT, ecolor="#999999",
        elinewidth=1.2, capsize=3, markersize=6,
    )
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    span = float(np.max(np.abs(vals) + xerr)) or 1.0
    for yi, v, e, p in zip(y, vals, xerr, ps):
        s = _stars(p, alpha)
        if s:
            ax.text(v + e + span * 0.03, yi, s, va="center", fontsize=10, color="#c0392b")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(-span * 1.15, span * 1.15)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_matrix_heatmap_multi(
    results: pd.DataFrame,
    groups: List[str],
    out_path: Path,
    *,
    value_col: str,
    p_adj_col: str,
    title: str,
    alpha: float,
) -> None:
    """Cross-model mean of `value_col` per cell, annotated with how many
    models reach BH-adjusted significance (k/N)."""
    n = len(groups)
    gi = {g: i for i, g in enumerate(groups)}
    mean_mat = np.full((n, n), np.nan)
    sig_mat = np.zeros((n, n), dtype=int)
    tot_mat = np.zeros((n, n), dtype=int)
    for (a, b), g in results.groupby(["group_a", "group_b"], sort=False):
        i, j = gi[a], gi[b]
        v = float(np.nanmean(g[value_col].astype(float)))
        k = int((g[p_adj_col] < alpha).sum())
        t = int(len(g))
        mean_mat[i, j] = v
        mean_mat[j, i] = -v if np.isfinite(v) else np.nan
        sig_mat[i, j] = sig_mat[j, i] = k
        tot_mat[i, j] = tot_mat[j, i] = t

    finite = mean_mat[np.isfinite(mean_mat)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(7.5, 1.05 * n + 3.0), max(6.5, 0.95 * n + 2.5)))
    im = ax.imshow(mean_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(groups, fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(mean_mat[i, j]):
                continue
            norm_v = abs(mean_mat[i, j]) / vmax
            color = "white" if norm_v > 0.6 else "#111111"
            ax.text(
                j, i,
                f"{mean_mat[i, j]:+.2f}\n{sig_mat[i, j]}/{tot_mat[i, j]}",
                ha="center", va="center", fontsize=7.5, color=color,
            )
    ax.set_title(f"{title}\n(cross-model mean; k/N = models significant at BH-adjusted p < {alpha:g})")
    ax.tick_params(axis="both", length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_report(
    *,
    out_md: Path,
    images_dir: Path,
    title: str,
    args: argparse.Namespace,
    record: Dict[str, Any],
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    omnibus: Optional[Dict[str, float]],
    image_flags: Dict[str, bool],
) -> None:
    now_utc = datetime.now(timezone.utc)
    rel = lambda p: Path(images_dir.name) / p.name  # noqa: E731
    lines: List[str] = []
    mode = record["mode"]

    subtitle = (
        f"Pairwise VQA difference testing  ·  {record['group_column']}  ·  "
        + (f"{record['group_a']} vs {record['group_b']}" if mode == "pair"
           else f"{len(record['groups'])} groups, all pairs")
    )
    lines += ["---", f'title: "{title}"', f'subtitle: "{subtitle}"',
              f'date: "{now_utc.strftime("%B %d, %Y")}"', "---", ""]

    # ---- Overview ----
    lines += ["\\newpage", "", "# Experiment overview", ""]
    lines.append(
        f"**Source:** \\path{{{record['source_parquet']}}}  \n"
        f"**Generated:** {now_utc.isoformat(timespec='seconds')} &nbsp;·&nbsp; "
        f"**Experiment id:** `{record['experiment_id']}`"
    )
    lines.append("")
    lines.append("| field | value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| group column | `{record['group_column']}` |")
    lines.append(f"| group source | {record['group_source']} |")
    if record.get("model_label"):
        lines.append(f"| model | `{record['model_label']}` |")
    lines.append(f"| attribute | *{args.attribute}* |")
    lines.append(f"| rows (responses) | {record['n_rows']:,} |")
    lines.append(f"| rated units | {record['n_units_rated']:,} |")
    lines.append(f"| TrueSkill draw prob | {args.draw_prob} |")
    lines.append(f"| min comparisons / unit | {args.min_comparisons} |")
    lines.append("")

    # ---- Coverage ----
    lines += ["## Group coverage", ""]
    cov = coverage.copy()
    cov["units"] = cov["units"].map("{:,}".format)
    lines.append(_md_table(cov.rename(columns={
        "group": "Group", "units": "Rated units", "mu_mean": "Mean μ",
    })))
    lines.append("")

    if mode == "pair":
        r = results.iloc[0]
        ga, gb = record["group_a"], record["group_b"]

        lines += ["\\newpage", "", f"# Head-to-head  ·  {ga} vs {gb}", ""]
        lines.append(
            f"Direct {ga}-vs-{gb} pairs only. `relative_score` oriented so "
            f"**positive = {ga} preferred** ({args.attribute}); repeated presentations "
            "collapsed to one observation per canonical pair before testing."
        )
        lines.append("")
        h2h = pd.DataFrame([{
            "Direct pairs (collapsed)": f"{int(r['n_direct_pairs']):,}",
            "Raw observations": f"{int(r['n_direct_obs']):,}",
            "Mean oriented score": f"{r['mean_oriented']:+.4f}",
            "t": f"{r['h2h_t']:.3f}" if np.isfinite(r["h2h_t"]) else "",
            "p (t-test)": _fmt_p(r["h2h_p"]),
            "p (Wilcoxon)": _fmt_p(r["h2h_wilcoxon_p"]),
            "p (sign test)": _fmt_p(r["h2h_sign_p"]),
            f"Win rate ({ga})": (
                f"{r['win_rate']:.1%} [{r['win_ci_lo']:.1%}, {r['win_ci_hi']:.1%}]"
                if np.isfinite(r["win_rate"]) else ""
            ),
            "Cohen's d": f"{r['h2h_cohens_d']:+.3f}" if np.isfinite(r["h2h_cohens_d"]) else "",
        }]).T.reset_index()
        h2h.columns = ["metric", "value"]
        lines.append(_md_table(h2h))
        lines.append("")
        if image_flags.get("oriented_hist"):
            lines.extend(_centered_image(rel(images_dir / "oriented_hist.png"),
                                         "Oriented score distribution", "width=70%"))

        lines += ["\\newpage", "", f"# Rating-level  ·  TrueSkill μ by group", ""]
        lines.append(
            "Per-unit TrueSkill μ fit over the **full** run (all comparisons, all "
            "groups), then compared between the two groups. Uses far more data than "
            "the head-to-head subset, but μ estimates are coupled through the "
            "comparison graph, so p-values are approximate."
        )
        lines.append("")
        rt = pd.DataFrame([{
            f"Units ({ga})": f"{int(r['n_units_a']):,}",
            f"Units ({gb})": f"{int(r['n_units_b']):,}",
            f"Mean μ ({ga})": f"{r['mu_mean_a']:.3f}",
            f"Mean μ ({gb})": f"{r['mu_mean_b']:.3f}",
            "Δμ (a − b)": f"{r['delta_mu']:+.3f}",
            "Welch t": f"{r['rating_t']:.3f}" if np.isfinite(r["rating_t"]) else "",
            "p (Welch)": _fmt_p(r["rating_p"]),
            "p (Mann-Whitney)": _fmt_p(r["rating_mwu_p"]),
            "Cohen's d": f"{r['rating_cohens_d']:+.3f}" if np.isfinite(r["rating_cohens_d"]) else "",
            "Cliff's δ": f"{r['cliffs_delta']:+.3f}" if np.isfinite(r["cliffs_delta"]) else "",
        }]).T.reset_index()
        rt.columns = ["metric", "value"]
        lines.append(_md_table(rt))
        lines.append("")
        if image_flags.get("group_mu"):
            lines.extend(_centered_image(rel(images_dir / "group_mu.png"),
                                         "TrueSkill μ by group", "width=70%"))
    else:
        lines += ["\\newpage", "", "# All-pairs matrix", ""]
        if omnibus is not None:
            lines.append(
                f"**Kruskal-Wallis omnibus over per-group μ:** H = {omnibus['H']:.2f}, "
                f"p = {_fmt_p(omnibus['p'])} across {len(record['groups'])} groups."
            )
            lines.append("")
        n_sig_h2h = int((results["h2h_p_adj"] < args.alpha).sum())
        n_sig_rating = int((results["rating_p_adj"] < args.alpha).sum())
        lines.append(
            f"{len(results):,} group pairs tested. Significant after BH correction "
            f"(α = {args.alpha:g}): **{n_sig_h2h}** head-to-head, **{n_sig_rating}** rating-level."
        )
        lines.append("")
        if image_flags.get("heatmap_h2h"):
            lines.extend(_centered_image(rel(images_dir / "heatmap_h2h.png"),
                                         "Head-to-head mean oriented score", "width=80%"))
        if image_flags.get("heatmap_rating"):
            lines += ["\\newpage", ""]
            lines.extend(_centered_image(rel(images_dir / "heatmap_rating.png"),
                                         "Rating-level Δμ", "width=80%"))
        if image_flags.get("group_mu"):
            lines += ["\\newpage", ""]
            lines.extend(_centered_image(rel(images_dir / "group_mu.png"),
                                         "TrueSkill μ by group", "width=86%"))

        lines += ["\\newpage", "", "# Strongest differences", ""]
        lines.append(
            "Group pairs sorted by rating-level BH-adjusted p (head-to-head columns "
            "alongside). Full table in the companion `*.difference_tests.parquet`."
        )
        lines.append("")
        show = results.sort_values("rating_p_adj").head(25).copy()
        disp = pd.DataFrame({
            "A": show["group_a"],
            "B": show["group_b"],
            "Δμ": show["delta_mu"].map(lambda v: f"{v:+.3f}"),
            "p_adj (rating)": show["rating_p_adj"].map(_fmt_p),
            "d": show["rating_cohens_d"].map(
                lambda v: f"{v:+.2f}" if np.isfinite(v) else ""),
            "direct pairs": show["n_direct_pairs"].astype(int),
            "mean oriented": show["mean_oriented"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
            "p_adj (h2h)": show["h2h_p_adj"].map(_fmt_p),
        })
        lines.append(_md_table(disp))
        lines.append("")

    # ---- Caveats ----
    lines += ["\\newpage", "", "# Caveats", ""]
    lines.append(
        "* **Head-to-head**: with `allow_replacement=true` the same unit appears in "
        "many pairs, so observations are not fully independent (shared-unit "
        "clustering is not modeled). Repeats *are* collapsed by canonical pair.\n"
        "* **Rating-level**: TrueSkill μ values are jointly estimated from the full "
        "comparison graph and are not i.i.d. samples; Welch/Mann-Whitney p-values "
        "are approximate. Effect sizes (d, Cliff's δ) are the more robust summary.\n"
        "* Group assignment reflects the metadata join at analysis time, not at "
        "inference time — the VLM never saw the group labels.\n"
        "* When the two tests disagree, trust the head-to-head direction if direct "
        "pairs are plentiful; the rating-level test borrows strength from "
        "comparisons against all other groups."
    )
    lines.append("")
    out_md.write_text("\n".join(lines))


def write_report_multi(
    *,
    out_md: Path,
    images_dir: Path,
    title: str,
    args: argparse.Namespace,
    record: Dict[str, Any],
    results: pd.DataFrame,
    replication: pd.DataFrame,
    coverage: pd.DataFrame,
    omnibus_df: Optional[pd.DataFrame],
    image_flags: Dict[str, bool],
) -> None:
    """Multi-model variant of the report: per-model tables + replication."""
    now_utc = datetime.now(timezone.utc)
    rel = lambda p: Path(images_dir.name) / p.name  # noqa: E731
    lines: List[str] = []
    mode = record["mode"]
    models: List[str] = record["models"]
    n_models = len(models)

    subtitle = (
        f"Multi-model difference testing  ·  {n_models} models  ·  {record['group_column']}  ·  "
        + (f"{record['group_a']} vs {record['group_b']}" if mode == "pair"
           else f"{len(record['groups'])} groups, all pairs")
    )
    lines += ["---", f'title: "{title}"', f'subtitle: "{subtitle}"',
              f'date: "{now_utc.strftime("%B %d, %Y")}"', "---", ""]

    # ---- Overview ----
    lines += ["\\newpage", "", "# Experiment overview", ""]
    lines.append(
        f"**Aggregation dir:** \\path{{{record['source_parquet']}}}  \n"
        f"**Generated:** {now_utc.isoformat(timespec='seconds')} &nbsp;·&nbsp; "
        f"**Experiment id:** `{record['experiment_id']}`"
    )
    lines.append("")
    lines.append("| field | value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| models | {', '.join(f'`{m}`' for m in models)} |")
    lines.append(f"| group column | `{record['group_column']}` |")
    lines.append(f"| group source | {record['group_source']} |")
    lines.append(f"| attribute | *{args.attribute}* |")
    lines.append(f"| rows per model | {record['n_rows']:,} |")
    lines.append(f"| rated units (reference model) | {record['n_units_rated']:,} |")
    lines.append(f"| TrueSkill draw prob | {args.draw_prob} |")
    lines.append("")

    lines += ["## Group coverage  ·  reference model", ""]
    cov = coverage.copy()
    cov["units"] = cov["units"].map("{:,}".format)
    lines.append(_md_table(cov.rename(columns={
        "group": "Group", "units": "Rated units", "mu_mean": "Mean μ",
    })))
    lines.append("")

    if mode == "pair":
        ga, gb = record["group_a"], record["group_b"]
        rep = replication.iloc[0]

        lines += ["\\newpage", "", f"# Replication summary  ·  {ga} vs {gb}", ""]
        lines.append(
            f"* Head-to-head: **{int(rep['n_toward_a_h2h'])}/{n_models}** models point toward "
            f"*{ga}*; **{int(rep['n_sig_h2h'])}/{n_models}** significant (α = {args.alpha:g}). "
            f"Cross-model mean oriented score: **{rep['mean_oriented_mean']:+.3f}**.\n"
            f"* Rating-level: **{int(rep['n_toward_a_rating'])}/{n_models}** models point toward "
            f"*{ga}*; **{int(rep['n_sig_rating'])}/{n_models}** significant. "
            f"Cross-model mean Δμ: **{rep['delta_mu_mean']:+.3f}**."
        )
        lines.append("")

        lines += ["## Head-to-head by model", ""]
        h2h = pd.DataFrame({
            "Model": results["model_label"],
            "Direct pairs": results["n_direct_pairs"].astype(int),
            "Mean oriented": results["mean_oriented"].map(
                lambda v: f"{v:+.4f}" if np.isfinite(v) else ""),
            "p (t-test)": results["h2h_p"].map(_fmt_p),
            f"Win rate ({ga})": [
                f"{w:.1%} [{lo:.1%}, {hi:.1%}]" if np.isfinite(w) else ""
                for w, lo, hi in zip(results["win_rate"], results["win_ci_lo"], results["win_ci_hi"])
            ],
            "d": results["h2h_cohens_d"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
        })
        lines.append(_md_table(h2h))
        lines.append("")
        if image_flags.get("forest_h2h"):
            lines.extend(_centered_image(rel(images_dir / "forest_h2h.png"),
                                         "Head-to-head forest plot", "width=72%"))

        lines += ["\\newpage", "", "## Rating-level by model", ""]
        rt = pd.DataFrame({
            "Model": results["model_label"],
            f"Units ({ga}/{gb})": [
                f"{int(a):,} / {int(b):,}"
                for a, b in zip(results["n_units_a"], results["n_units_b"])
            ],
            "Δμ (a − b)": results["delta_mu"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
            "p (Welch)": results["rating_p"].map(_fmt_p),
            "p (MWU)": results["rating_mwu_p"].map(_fmt_p),
            "d": results["rating_cohens_d"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
            "Cliff's δ": results["cliffs_delta"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
        })
        lines.append(_md_table(rt))
        lines.append("")
        if image_flags.get("forest_rating"):
            lines.extend(_centered_image(rel(images_dir / "forest_rating.png"),
                                         "Rating-level forest plot", "width=72%"))
    else:
        lines += ["\\newpage", "", "# Per-model summary", ""]
        per_model_rows = []
        for m in models:
            sub = results[results["model_label"] == m]
            row = {
                "Model": m,
                "Pairs tested": len(sub),
                f"Sig. h2h (α={args.alpha:g})": int((sub["h2h_p_adj"] < args.alpha).sum()),
                "Sig. rating": int((sub["rating_p_adj"] < args.alpha).sum()),
            }
            if omnibus_df is not None and m in set(omnibus_df["model"]):
                row["Omnibus p (KW)"] = _fmt_p(
                    float(omnibus_df.loc[omnibus_df["model"] == m, "p"].iloc[0])
                )
            per_model_rows.append(row)
        lines.append(_md_table(pd.DataFrame(per_model_rows)))
        lines.append("")
        n_replicated = int(
            ((replication["n_sig_rating"] == replication["n_models"])
             & replication["direction_consistent_rating"]).sum()
        )
        lines.append(
            f"**{n_replicated}** of {len(replication)} comparisons are significant in "
            f"*every* model with a consistent direction (rating-level)."
        )
        lines.append("")
        if image_flags.get("heatmap_h2h"):
            lines.extend(_centered_image(rel(images_dir / "heatmap_h2h.png"),
                                         "Head-to-head, cross-model", "width=80%"))
        if image_flags.get("heatmap_rating"):
            lines += ["\\newpage", ""]
            lines.extend(_centered_image(rel(images_dir / "heatmap_rating.png"),
                                         "Rating-level Δμ, cross-model", "width=80%"))

        lines += ["\\newpage", "", "# Most replicated differences", ""]
        show = replication.sort_values(
            ["n_sig_rating", "delta_mu_mean"],
            ascending=[False, False],
            key=lambda s: s.abs() if s.name == "delta_mu_mean" else s,
        ).head(25)
        disp = pd.DataFrame({
            "A": show["group_a"],
            "B": show["group_b"],
            "mean Δμ": show["delta_mu_mean"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
            "sig (rating)": [
                f"{int(k)}/{int(n)}" for k, n in zip(show["n_sig_rating"], show["n_models"])
            ],
            "toward A (rating)": [
                f"{int(k)}/{int(n)}" for k, n in zip(show["n_toward_a_rating"], show["n_models"])
            ],
            "sig (h2h)": [
                f"{int(k)}/{int(n)}" for k, n in zip(show["n_sig_h2h"], show["n_models"])
            ],
            "consistent": show["direction_consistent_rating"].map({True: "yes", False: "no"}),
        })
        lines.append(_md_table(disp))
        lines.append("")

    # ---- Caveats ----
    lines += ["\\newpage", "", "# Caveats", ""]
    lines.append(
        "* All single-run caveats apply per model (shared-unit clustering in "
        "head-to-head; coupled TrueSkill μ in rating-level).\n"
        "* Models were evaluated on the **same pair set**, so per-model results are "
        "correlated through the shared images — replication counts overstate "
        "independence across models.\n"
        "* BH correction is applied within each model, not across the pooled table.\n"
        "* Group assignment reflects the metadata join at analysis time; the VLMs "
        "never saw the group labels."
    )
    lines.append("")
    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if args.list:
        list_registry(registry_path(args.registry))
        return

    multi = args.aggregation_dir is not None
    if multi:
        if args.output_parquet is not None:
            raise SystemExit("Pass either an output parquet or --aggregation-dir, not both.")
        if not args.aggregation_dir.exists():
            raise SystemExit(f"--aggregation-dir not found: {args.aggregation_dir}")
    else:
        if args.output_parquet is None:
            raise SystemExit("output_parquet or --aggregation-dir is required (or use --list).")
        if not args.output_parquet.exists():
            raise SystemExit(f"Output parquet not found: {args.output_parquet}")
    if not args.group_column:
        raise SystemExit("--group-column is required.")
    single_mode = args.group_a is not None or args.group_b is not None
    if single_mode and (args.group_a is None or args.group_b is None):
        raise SystemExit("Provide both --group-a and --group-b.")
    if not single_mode and not args.all_pairs:
        raise SystemExit("Choose a mode: --group-a/--group-b or --all-pairs.")
    if single_mode and args.all_pairs:
        raise SystemExit("--group-a/--group-b and --all-pairs are mutually exclusive.")

    # ---- Sources (one per model) ----
    if multi:
        from pairwise_vqa_aggregation_report import _discover_runs

        runs, skipped = _discover_runs(args.aggregation_dir)
        for name, reason in skipped:
            print(f"[WARN] skipping {name}: {reason}")
        if not runs:
            raise SystemExit(f"No usable runs under {args.aggregation_dir}.")
        sources = [(r.model_label, r.out_parquet, r.pairs_parquet) for r in runs]
        source_id_path = str(args.aggregation_dir.resolve())
        print(f"Models discovered: {', '.join(label for label, _, _ in sources)}")
    else:
        sources = [(None, args.output_parquet, args.pairs_parquet)]
        source_id_path = str(args.output_parquet.resolve())

    # ---- Load + per-model analysis inputs ----
    per_model: List[Dict[str, Any]] = []
    group_source = ""
    for label, out_pq, pairs_pq in sources:
        df, group_from_pairs = load_run(out_pq, pairs_pq, args.group_column)
        df = attach_groups(
            df,
            group_column=args.group_column,
            group_from_pairs=group_from_pairs,
            unit_metadata_parquet=args.unit_metadata_parquet,
            unit_metadata_id_column=args.unit_metadata_id_column,
        )
        unit_groups = build_unit_group_map(df)
        if unit_groups.empty:
            raise SystemExit("No unit could be assigned a group value — check the join/column.")
        ratings = _compute_trueskill(df, draw_prob=args.draw_prob)
        rated = ratings[ratings["n_comparisons"] >= args.min_comparisons].copy()
        rated["group"] = rated["unit_uid"].astype(str).map(unit_groups)
        if label is None:
            # Source-run provenance from the run's resolved Hydra config.
            run_config = _load_run_config(out_pq, None)
            model_src = str(((run_config or {}).get("model") or {}).get("model_source") or "")
            label = Path(model_src).name if model_src else ""
        per_model.append({
            "label": label, "df": df, "unit_groups": unit_groups,
            "ratings": ratings, "rated": rated,
        })
        group_source = (
            "pairs.parquet metadata"
            if group_from_pairs
            else f"external join: {args.unit_metadata_parquet} on {args.unit_metadata_id_column}"
        )

    n_models = len(per_model)
    ref = per_model[0]
    if multi:
        sizes = {m["label"]: len(m["df"]) for m in per_model}
        if len(set(sizes.values())) > 1:
            print(f"[WARN] row counts differ across models: {sizes}")
    print(f"Group source:      {group_source}")
    print(
        f"Units with groups: {len(ref['unit_groups']):,} across "
        f"{ref['unit_groups'].nunique():,} values"
    )

    group_unit_counts = (
        ref["rated"].dropna(subset=["group"]).groupby("group").size().sort_values(ascending=False)
    )

    # ---- Resolve comparison set (against the reference model) ----
    if single_mode:
        group_a = resolve_group_name(args.group_a, ref["rated"]["group"].dropna())
        group_b = resolve_group_name(args.group_b, ref["rated"]["group"].dropna())
        if group_a == group_b:
            raise SystemExit("--group-a and --group-b resolve to the same group.")
        groups = [group_a, group_b]
        mode = "pair"
    else:
        if args.groups:
            groups = [
                resolve_group_name(g, ref["rated"]["group"].dropna())
                for g in args.groups.split(",") if g.strip()
            ]
        else:
            groups = [
                g for g, n in group_unit_counts.items() if n >= args.min_group_units
            ]
        if args.top_k_groups:
            groups = [g for g in group_unit_counts.index if g in set(groups)][: args.top_k_groups]
        if len(groups) < 2:
            raise SystemExit(
                f"Only {len(groups)} eligible group(s) (min {args.min_group_units} "
                "rated units each); lower --min-group-units or pass --groups."
            )
        group_a = group_b = None
        mode = "matrix"

    # ---- Registry check ----
    id_inputs = {
        "source_parquet": source_id_path,
        "models": sorted(m["label"] for m in per_model) if multi else None,
        "group_column": args.group_column,
        "mode": mode,
        "groups": sorted([group_a, group_b]) if mode == "pair" else sorted(groups),
        "unit_metadata_parquet": (
            str(args.unit_metadata_parquet.resolve()) if args.unit_metadata_parquet else None
        ),
        "unit_metadata_id_column": args.unit_metadata_id_column,
        "draw_prob": args.draw_prob,
        "min_comparisons": args.min_comparisons,
        "min_group_units": args.min_group_units if mode == "matrix" else None,
    }
    experiment_id = compute_experiment_id(id_inputs)
    reg_path = registry_path(args.registry)
    if not args.no_registry:
        prior = find_in_registry(read_registry(reg_path), experiment_id)
        if prior is not None and not args.force:
            print(f"Experiment {experiment_id} already registered ({prior.get('created_at')}).")
            print(f"  report:  {prior.get('report_md')}")
            print(f"  results: {prior.get('results_parquet')}")
            if prior.get("wandb_url"):
                print(f"  wandb:   {prior['wandb_url']}")
            print("Use --force to rerun.")
            return

    # ---- Run tests (per model; BH within model) ----
    pairs_to_test = (
        [(group_a, group_b)] if mode == "pair" else list(combinations(groups, 2))
    )
    frames: List[pd.DataFrame] = []
    for m in per_model:
        rows = [
            run_comparison(m["df"], m["ratings"], m["unit_groups"], a, b, args.min_comparisons)
            for a, b in pairs_to_test
        ]
        sub = adjust_pvalues(pd.DataFrame(rows), ["h2h_p", "rating_p"])
        sub.insert(0, "model_label", m["label"])
        frames.append(sub)
    results = pd.concat(frames, ignore_index=True)
    results.insert(0, "group_column", args.group_column)
    results.insert(0, "experiment_id", experiment_id)

    replication: Optional[pd.DataFrame] = (
        build_replication(results, args.alpha) if multi else None
    )

    omnibus: Optional[Dict[str, float]] = None
    omnibus_df: Optional[pd.DataFrame] = None
    if mode == "matrix":
        omni_rows = []
        for m in per_model:
            samples = [
                m["rated"].loc[m["rated"]["group"] == g, "mu"].to_numpy(dtype=float)
                for g in groups
            ]
            samples = [s for s in samples if s.size >= 2]
            if len(samples) >= 2:
                kw = sps.kruskal(*samples)
                omni_rows.append(
                    {"model": m["label"], "H": float(kw.statistic), "p": float(kw.pvalue)}
                )
        if omni_rows:
            omnibus_df = pd.DataFrame(omni_rows)
            omnibus = {"H": float(omnibus_df["H"].iloc[0]), "p": float(omnibus_df["p"].iloc[0])}

    # ---- Outputs ----
    if mode == "pair":
        slug = f"{_slug(args.group_column)}.{_slug(group_a)}-vs-{_slug(group_b)}"
    else:
        slug = f"{_slug(args.group_column)}.matrix-{len(groups)}"
    if multi:
        slug += f".multi{n_models}"
        out_md = args.out or (args.aggregation_dir / f"difference.{slug}.md")
    else:
        out_md = args.out or args.output_parquet.with_suffix(f".difference.{slug}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    images_dir = out_md.parent / f"{out_md.stem}_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_pq = out_md.with_suffix(".tests.parquet")
    results.to_parquet(results_pq, index=False)

    image_flags: Dict[str, bool] = {}
    if not multi:
        image_flags["group_mu"] = _plot_group_mu(
            ref["ratings"], ref["unit_groups"], groups,
            images_dir / "group_mu.png", args.min_comparisons,
        )
        if mode == "pair":
            image_flags["oriented_hist"] = _plot_oriented_hist(
                ref["df"], group_a, group_b, images_dir / "oriented_hist.png"
            )
        else:
            _plot_matrix_heatmap(
                results, groups, images_dir / "heatmap_h2h.png",
                value_col="mean_oriented", p_adj_col="h2h_p_adj",
                title="Head-to-head mean oriented score", alpha=args.alpha,
            )
            image_flags["heatmap_h2h"] = True
            _plot_matrix_heatmap(
                results, groups, images_dir / "heatmap_rating.png",
                value_col="delta_mu", p_adj_col="rating_p_adj",
                title="Rating-level Δμ (TrueSkill)", alpha=args.alpha,
            )
            image_flags["heatmap_rating"] = True
    elif mode == "pair":
        image_flags["forest_h2h"] = _plot_forest(
            results, images_dir / "forest_h2h.png",
            value_col="mean_oriented", se_col="mean_oriented_se", p_col="h2h_p",
            title=f"Head-to-head  ·  {group_a} vs {group_b}",
            xlabel=f"mean oriented score  (+ = {group_a} preferred; ±1.96 SE)",
            alpha=args.alpha,
        )
        image_flags["forest_rating"] = _plot_forest(
            results, images_dir / "forest_rating.png",
            value_col="delta_mu", se_col="delta_mu_se", p_col="rating_p",
            title=f"Rating-level Δμ (TrueSkill)  ·  {group_a} vs {group_b}",
            xlabel=f"Δμ  (+ = {group_a} higher; ±1.96 SE)",
            alpha=args.alpha,
        )
    else:
        _plot_matrix_heatmap_multi(
            results, groups, images_dir / "heatmap_h2h.png",
            value_col="mean_oriented", p_adj_col="h2h_p_adj",
            title="Head-to-head mean oriented score", alpha=args.alpha,
        )
        image_flags["heatmap_h2h"] = True
        _plot_matrix_heatmap_multi(
            results, groups, images_dir / "heatmap_rating.png",
            value_col="delta_mu", p_adj_col="rating_p_adj",
            title="Rating-level Δμ (TrueSkill)", alpha=args.alpha,
        )
        image_flags["heatmap_rating"] = True

    # ---- Registry record ----
    ref_rated = ref["rated"]
    coverage = pd.DataFrame({
        "group": groups,
        "units": [int(group_unit_counts.get(g, 0)) for g in groups],
        "mu_mean": [
            float(ref_rated.loc[ref_rated["group"] == g, "mu"].mean())
            if group_unit_counts.get(g, 0) else float("nan")
            for g in groups
        ],
    }).sort_values("mu_mean", ascending=False).reset_index(drop=True)

    fin = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
    if mode == "pair" and not multi:
        r0 = results.iloc[0]
        result_summary = {
            k: fin(r0[k])
            for k in (
                "mean_oriented", "h2h_p", "h2h_cohens_d", "win_rate",
                "delta_mu", "rating_p", "rating_cohens_d", "cliffs_delta",
            )
        }
        result_summary["n_direct_pairs"] = int(r0["n_direct_pairs"])
        result_summary["n_units_a"] = int(r0["n_units_a"])
        result_summary["n_units_b"] = int(r0["n_units_b"])
    elif mode == "pair":
        rep = replication.iloc[0]
        result_summary = {
            "n_models": n_models,
            "mean_oriented_mean": fin(rep["mean_oriented_mean"]),
            "delta_mu_mean": fin(rep["delta_mu_mean"]),
            "n_toward_a_h2h": int(rep["n_toward_a_h2h"]),
            "n_toward_a_rating": int(rep["n_toward_a_rating"]),
            "n_sig_h2h": int(rep["n_sig_h2h"]),
            "n_sig_rating": int(rep["n_sig_rating"]),
        }
    elif multi:
        result_summary = {
            "n_models": n_models,
            "n_pairs_tested": int(len(pairs_to_test)),
            "n_replicated_significant": int(
                ((replication["n_sig_rating"] == replication["n_models"])
                 & replication["direction_consistent_rating"]).sum()
            ),
        }
    else:
        result_summary = {
            "n_pairs_tested": int(len(results)),
            "n_significant": int((results["rating_p_adj"] < args.alpha).sum()),
            "n_significant_h2h": int((results["h2h_p_adj"] < args.alpha).sum()),
            "omnibus_p": (omnibus or {}).get("p"),
        }

    record: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "source_parquet": source_id_path,
        "model_label": ref["label"] if not multi else None,
        "models": [m["label"] for m in per_model] if multi else None,
        "n_models": n_models,
        "group_column": args.group_column,
        "group_source": group_source,
        "group_a": group_a,
        "group_b": group_b,
        "groups": groups,
        "unit_metadata_parquet": id_inputs["unit_metadata_parquet"],
        "unit_metadata_id_column": args.unit_metadata_id_column,
        "draw_prob": args.draw_prob,
        "min_comparisons": args.min_comparisons,
        "min_group_units": args.min_group_units,
        "n_rows": int(len(ref["df"])),
        "n_units_rated": int(len(ref_rated)),
        "results": result_summary,
        "report_md": str(out_md.resolve()),
        "results_parquet": str(results_pq.resolve()),
        "wandb_url": None,
    }

    # ---- Report ----
    title = args.title or (
        f"Difference test: {group_a} vs {group_b}" if mode == "pair"
        else f"Difference matrix: {args.group_column} ({len(groups)} groups)"
    )
    if multi:
        title += f"  ·  {n_models} models"
        write_report_multi(
            out_md=out_md,
            images_dir=images_dir,
            title=title,
            args=args,
            record=record,
            results=results,
            replication=replication,
            coverage=coverage,
            omnibus_df=omnibus_df,
            image_flags=image_flags,
        )
    else:
        write_report(
            out_md=out_md,
            images_dir=images_dir,
            title=title,
            args=args,
            record=record,
            results=results,
            coverage=coverage,
            omnibus=omnibus,
            image_flags=image_flags,
        )
    print(f"Wrote report:      {out_md}")
    print(f"Wrote results:     {results_pq}")

    pdf_path = None
    if args.pdf:
        pdf_path = _export_pdf(
            out_md, args.pdf_engine, landscape=(args.pdf_orientation == "landscape")
        )
        if pdf_path is not None:
            print(f"Wrote PDF:         {pdf_path}")

    # ---- W&B mirror ----
    if not args.no_wandb:
        run_label = (
            f"difftest_{_slug(args.group_column)}_{_slug(group_a)}-vs-{_slug(group_b)}"
            if mode == "pair"
            else f"difftest_{_slug(args.group_column)}_matrix{len(groups)}"
        )
        if multi:
            run_label += f"_multi{n_models}"
        url = mirror_to_wandb(
            record=record,
            results=results,
            artifact_paths=[out_md, results_pq] + ([pdf_path] if pdf_path else []),
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_label=run_label,
        )
        record["wandb_url"] = url
        if url:
            print(f"W&B run:           {url}")

    # ---- Registry append ----
    if not args.no_registry:
        append_registry(reg_path, record)
        print(f"Registered:        {experiment_id} → {reg_path}")

    # ---- Console summary ----
    if mode == "pair" and not multi:
        r0 = results.iloc[0]
        direction = group_a if (r0["mean_oriented"] or 0) > 0 else group_b
        print(
            f"\nHead-to-head: n={int(r0['n_direct_pairs'])}, "
            f"mean oriented={r0['mean_oriented']:+.3f} (toward {direction}), "
            f"p={_fmt_p(r0['h2h_p'])}"
        )
        print(
            f"Rating-level: Δμ={r0['delta_mu']:+.3f}, p={_fmt_p(r0['rating_p'])}, "
            f"d={r0['rating_cohens_d']:+.3f}"
        )
    elif mode == "pair":
        rep = replication.iloc[0]
        toward = group_a if rep["delta_mu_mean"] > 0 else group_b
        print(
            f"\n{n_models} models: rating-level {int(rep['n_toward_a_rating'])}/{n_models} "
            f"toward {group_a}, {int(rep['n_sig_rating'])}/{n_models} significant; "
            f"mean Δμ={rep['delta_mu_mean']:+.3f} (toward {toward}). "
            f"Head-to-head {int(rep['n_toward_a_h2h'])}/{n_models} toward {group_a}, "
            f"{int(rep['n_sig_h2h'])}/{n_models} significant."
        )
    elif multi:
        print(
            f"\n{len(pairs_to_test)} comparisons × {n_models} models; "
            f"{result_summary['n_replicated_significant']} significant in every model "
            f"with consistent direction (rating-level, BH α={args.alpha:g})."
        )
    else:
        print(
            f"\n{len(results)} pairs tested; "
            f"{int((results['rating_p_adj'] < args.alpha).sum())} significant "
            f"(rating-level, BH α={args.alpha:g})."
        )


if __name__ == "__main__":
    main()
