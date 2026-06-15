#!/usr/bin/env python3
"""Targeted regressions over a urbanpairvqa run.

Answers questions like "how much of the VLM's per-school rating is explained
by poverty rate?" Per-unit TrueSkill μ is regressed on one or more unit-level
covariates resolved from surfaced pair metadata or an external unit-metadata
parquet (e.g. ``curation/external/school_covariates.parquet``).

Two complementary designs per experiment (mirroring the difference tool's
two tests):

  * **Unit-level (primary)** — OLS of μ on x (+ optional controls) with HC3
    robust SEs; optional ``--wls`` weighting by 1/σ² (TrueSkill per-unit
    uncertainty). Reports R², adjusted R², the focal covariate's coefficient
    with CI, standardized β, partial R² (when controls present), and
    Spearman ρ as the rank-robust companion.
  * **Pair-level (validation)** — for each direct pair where both sides have
    x, regress the ordinal ``relative_score`` on Δx = x_a − x_b (repeats
    collapsed by canonical pair). Immune to TrueSkill coupling: asks whether
    individual judgments track the covariate difference.

Modes:

  * Single: ``--x pct_poverty`` (optionally ``--controls borough,yearbuilt``)
  * Screen: ``--x-list a,b,c`` — one regression per covariate, BH-corrected
  * Multi-model: ``--aggregation-dir`` — per-model fits + replication forest
  * ``--list``: print past experiments from the shared registry

Each experiment is recorded in the shared JSONL registry (mode
``regression`` / ``screen``) and mirrored to W&B (job_type ``regression``).

Examples:

    # The motivating question: VLM school rating vs poverty rate
    python scripts/pairwise_vqa_regression_report.py \\
        <run>/outputs/pairwise/schools_mvp_*.parquet \\
        --x pct_poverty \\
        --unit-metadata-parquet curation/external/school_covariates.parquet \\
        --attribute "rather send your child to" --unit-label school --pdf

    # Same, controlling for borough
    ... --x pct_poverty --controls borough

    # Covariate screen
    ... --x-list pct_poverty,economic_need_index,building_age,assess_per_bldg_sqft
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
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
from pairwise_analysis_common import (  # noqa: E402
    DEFAULT_WANDB_PROJECT,
    _centered_image,
    _fmt_p,
    _slug,
    _stars,
    adjust_pvalues,
    append_registry,
    build_unit_value_map,
    compute_experiment_id,
    find_in_registry,
    list_registry,
    load_run,
    load_unit_metadata,
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
        "output_parquet", type=Path, nargs="?", default=None,
        help="Stage output parquet. Not needed with --list / --aggregation-dir.",
    )
    p.add_argument("--pairs-parquet", type=Path, default=None,
                   help="Companion pairs.parquet. Default: sibling 'pairs.parquet'.")
    p.add_argument(
        "--aggregation-dir", type=Path, default=None,
        help="Multi-model mode: per-model runs (same layouts as the difference tool).",
    )
    # ---- model spec ----
    p.add_argument("--x", type=str, default=None, help="Focal covariate (single mode).")
    p.add_argument("--x-list", type=str, default=None,
                   help="Comma-separated covariates for screen mode (one regression each).")
    p.add_argument(
        "--controls", type=str, default=None,
        help=("Comma-separated control columns. Non-numeric controls are "
              "dummy-coded (C(col)). Applied in single mode only."),
    )
    p.add_argument("--y", choices=("mu", "ts_conservative"), default="mu",
                   help="Response: TrueSkill μ (default) or μ−3σ.")
    p.add_argument("--log-x", action="store_true",
                   help="log10-transform the focal covariate (positive values only).")
    p.add_argument("--wls", action="store_true",
                   help="Weight the unit-level fit by 1/σ² (TrueSkill uncertainty).")
    p.add_argument(
        "--unit-metadata-parquet", type=Path, default=None,
        help="External unit-metadata parquet with the covariate columns.",
    )
    p.add_argument("--unit-metadata-id-column", type=str, default="uid",
                   help="Id column in --unit-metadata-parquet. Default: uid.")
    # ---- stats knobs ----
    p.add_argument("--draw-prob", type=float, default=0.05)
    p.add_argument("--min-comparisons", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.05)
    # ---- report cosmetics ----
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--attribute", type=str, default="more of the attribute")
    p.add_argument("--unit-label", type=str, default="unit")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--pdf-engine", type=str, default="xelatex")
    p.add_argument("--pdf-orientation", choices=("landscape", "portrait"), default="landscape")
    # ---- registry / wandb ----
    p.add_argument("--registry", type=Path, default=None)
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--wandb-project", type=str,
                   default=os.environ.get("WANDB_ANALYSIS_PROJECT", DEFAULT_WANDB_PROJECT))
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _is_categorical(s: pd.Series) -> bool:
    if s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return pd.to_numeric(s, errors="coerce").isna().all()


def fit_unit_level(
    frame: pd.DataFrame,
    *,
    x: str,
    controls: List[str],
    wls: bool,
) -> Dict[str, Any]:
    """OLS/WLS of y on x (+ controls) over the unit frame. ``frame`` must
    contain columns y, x (numeric), sigma, and the controls. Returns a flat
    result dict; NaNs where the fit is not possible."""
    import statsmodels.formula.api as smf

    cols = ["y", x, "sigma"] + controls
    work = frame[cols].dropna().copy()
    n = len(work)
    res: Dict[str, Any] = {
        "x": x, "n_units": n,
        "r2": float("nan"), "r2_adj": float("nan"),
        "beta": float("nan"), "beta_se": float("nan"),
        "beta_ci_lo": float("nan"), "beta_ci_hi": float("nan"),
        "beta_std": float("nan"), "p": float("nan"),
        "partial_r2": float("nan"), "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
    }
    if n < 8 or work[x].nunique() < 3 or float(work[x].std()) == 0.0:
        return res

    terms = [f"Q('{x}')"]
    for c in controls:
        terms.append(f"C(Q('{c}'))" if _is_categorical(work[c]) else f"Q('{c}')")
    formula = "y ~ " + " + ".join(terms)

    weights = (1.0 / np.clip(work["sigma"].to_numpy(dtype=float), 1e-6, None) ** 2
               if wls else None)
    try:
        if wls:
            fit = smf.wls(formula, data=work, weights=weights).fit(cov_type="HC3")
        else:
            fit = smf.ols(formula, data=work).fit(cov_type="HC3")
    except Exception as exc:
        print(f"[WARN] fit failed for x={x}: {exc}")
        return res

    key = f"Q('{x}')"
    ci = fit.conf_int()
    res.update({
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "beta": float(fit.params[key]),
        "beta_se": float(fit.bse[key]),
        "beta_ci_lo": float(ci.loc[key, 0]),
        "beta_ci_hi": float(ci.loc[key, 1]),
        "beta_std": float(fit.params[key] * work[x].std() / work["y"].std())
        if work["y"].std() > 0 else float("nan"),
        "p": float(fit.pvalues[key]),
    })

    if controls:
        # Partial R² of the focal covariate given the controls.
        reduced_formula = "y ~ " + " + ".join(terms[1:]) if len(terms) > 1 else "y ~ 1"
        try:
            if wls:
                red = smf.wls(reduced_formula, data=work, weights=weights).fit()
            else:
                red = smf.ols(reduced_formula, data=work).fit()
            ssr_full = float(fit.ssr)
            ssr_red = float(red.ssr)
            if ssr_red > 0:
                res["partial_r2"] = max(0.0, (ssr_red - ssr_full) / ssr_red)
        except Exception:
            pass
    else:
        res["partial_r2"] = res["r2"]

    rho = sps.spearmanr(work[x], work["y"])
    res["spearman_rho"] = float(rho.statistic)
    res["spearman_p"] = float(rho.pvalue)
    return res


def fit_pair_level(
    df: pd.DataFrame,
    unit_x: pd.Series,
) -> Dict[str, Any]:
    """Regress the canonical ordinal score on Δx = x_a − x_b over direct
    pairs where both sides have a covariate value (repeats collapsed)."""
    work = df[["canonical_pair_id", "unit_uid_a", "unit_uid_b", "relative_score"]].copy()
    work["x_a"] = work["unit_uid_a"].astype(str).map(unit_x)
    work["x_b"] = work["unit_uid_b"].astype(str).map(unit_x)
    work = work.dropna(subset=["x_a", "x_b"])
    work["dx"] = work["x_a"].astype(float) - work["x_b"].astype(float)
    collapsed = work.groupby("canonical_pair_id").agg(
        score=("relative_score", "mean"), dx=("dx", "first")
    )
    n = len(collapsed)
    res: Dict[str, Any] = {
        "n_pairs": n, "pair_slope": float("nan"), "pair_slope_se": float("nan"),
        "pair_p": float("nan"), "pair_r2": float("nan"),
        "pair_spearman_rho": float("nan"), "pair_spearman_p": float("nan"),
    }
    if n < 8 or float(collapsed["dx"].std()) == 0.0:
        return res
    import statsmodels.api as sm

    X = sm.add_constant(collapsed["dx"].to_numpy(dtype=float))
    fit = sm.OLS(collapsed["score"].to_numpy(dtype=float), X).fit(cov_type="HC3")
    res.update({
        "pair_slope": float(fit.params[1]),
        "pair_slope_se": float(fit.bse[1]),
        "pair_p": float(fit.pvalues[1]),
        "pair_r2": float(fit.rsquared),
    })
    rho = sps.spearmanr(collapsed["dx"], collapsed["score"])
    res["pair_spearman_rho"] = float(rho.statistic)
    res["pair_spearman_p"] = float(rho.pvalue)
    return res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_scatter_fit(
    frame: pd.DataFrame, x: str, out_path: Path, *, xlabel: str, ylabel: str,
    r2: float, n_bins: int = 10,
) -> bool:
    work = frame[[x, "y"]].dropna()
    if len(work) < 8:
        return False
    xs = work[x].to_numpy(dtype=float)
    ys = work["y"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.scatter(xs, ys, s=9, c=ACCENT, alpha=0.35, linewidths=0)
    # OLS line + CI band.
    import statsmodels.api as sm

    X = sm.add_constant(xs)
    fit = sm.OLS(ys, X).fit()
    grid = np.linspace(xs.min(), xs.max(), 100)
    pred = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
    ax.plot(grid, pred["mean"], color="#c0392b", linewidth=1.6,
            label=f"OLS fit  ·  R² = {r2:.3f}")
    ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"],
                    color="#c0392b", alpha=0.15, linewidth=0)
    # Binned means.
    try:
        bins = pd.qcut(xs, q=min(n_bins, max(2, len(work) // 30)), duplicates="drop")
        bm = work.groupby(bins, observed=True).agg(bx=(x, "mean"), by=("y", "mean"))
        ax.plot(bm["bx"], bm["by"], "o-", color=ACCENT_WARM, markersize=6,
                linewidth=1.2, label="decile means")
    except ValueError:
        pass
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}  vs  {xlabel}  ·  n = {len(work):,}")
    ax.legend(loc="best")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_residuals(frame: pd.DataFrame, x: str, out_path: Path) -> bool:
    work = frame[[x, "y"]].dropna()
    if len(work) < 8:
        return False
    import statsmodels.api as sm

    X = sm.add_constant(work[x].to_numpy(dtype=float))
    fit = sm.OLS(work["y"].to_numpy(dtype=float), X).fit()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.scatter(fit.fittedvalues, fit.resid, s=9, c=ACCENT, alpha=0.35, linewidths=0)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    ax.set_xlabel("fitted")
    ax.set_ylabel("residual")
    ax.set_title("Residuals vs fitted  ·  simple fit (no controls)")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_pair_level(df: pd.DataFrame, unit_x: pd.Series, out_path: Path,
                     *, xlabel: str) -> bool:
    work = df[["canonical_pair_id", "unit_uid_a", "unit_uid_b", "relative_score"]].copy()
    work["dx"] = (work["unit_uid_a"].astype(str).map(unit_x).astype(float)
                  - work["unit_uid_b"].astype(str).map(unit_x).astype(float))
    work = work.dropna(subset=["dx"])
    collapsed = work.groupby("canonical_pair_id").agg(
        score=("relative_score", "mean"), dx=("dx", "first"))
    if len(collapsed) < 20:
        return False
    fig, ax = plt.subplots(figsize=(10, 5.0))
    try:
        bins = pd.qcut(collapsed["dx"], q=12, duplicates="drop")
        bm = collapsed.groupby(bins, observed=True).agg(
            bx=("dx", "mean"), by=("score", "mean"), n=("score", "size"),
            se=("score", lambda s: s.std() / math.sqrt(max(len(s), 1))))
        ax.errorbar(bm["bx"], bm["by"], yerr=1.96 * bm["se"], fmt="o-",
                    color=ACCENT, markersize=6, linewidth=1.2, capsize=3)
    except ValueError:
        return False
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    ax.axvline(0.0, color="#888888", linestyle=":", linewidth=0.9)
    ax.set_xlabel(f"Δ {xlabel}  (side A − side B)")
    ax.set_ylabel("mean ordinal score  (+ = A preferred)")
    ax.set_title(f"Head-to-head judgment vs covariate difference  ·  {len(collapsed):,} direct pairs")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_screen(results: pd.DataFrame, out_path: Path, alpha: float) -> bool:
    sub = results[np.isfinite(results["r2"].astype(float))].copy()
    if sub.empty:
        return False
    sub = sub.sort_values("r2")
    fig, ax = plt.subplots(figsize=(10, max(2.8, 0.5 * len(sub) + 1.6)))
    colors = [ACCENT if np.sign(b) >= 0 else "#762a83" for b in sub["beta_std"]]
    bars = ax.barh(sub["x"], sub["r2"], color=colors, edgecolor="#444", linewidth=0.4)
    for bar, p, b in zip(bars, sub["p_adj"], sub["beta_std"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  β*={b:+.2f}{_stars(p, alpha)}", va="center", fontsize=8.5)
    ax.set_xlabel("R²  (unit-level, no controls)")
    ax.set_title("Covariate screen  ·  blue = positive slope, purple = negative  ·  stars = BH-adjusted p")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _plot_forest_beta(results: pd.DataFrame, out_path: Path, *, alpha: float,
                      xlabel: str) -> bool:
    sub = results[np.isfinite(results["beta_std"].astype(float))].copy()
    if sub.empty:
        return False
    vals = sub["beta_std"].to_numpy(dtype=float)
    # Scale standardized CI from the raw CI half-width.
    half = (sub["beta_ci_hi"] - sub["beta_ci_lo"]).to_numpy(dtype=float) / 2.0
    scale = np.where(sub["beta"].to_numpy(dtype=float) != 0,
                     vals / sub["beta"].to_numpy(dtype=float), 0.0)
    xerr = np.abs(half * scale)
    labels = sub["model_label"].tolist()
    y = np.arange(len(sub))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(2.8, 0.55 * len(sub) + 1.6)))
    ax.errorbar(vals, y, xerr=xerr, fmt="o", color=ACCENT, ecolor="#999999",
                elinewidth=1.2, capsize=3, markersize=6)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=0.9)
    span = float(np.max(np.abs(vals) + xerr)) or 1.0
    for yi, v, e, p in zip(y, vals, xerr, sub["p"]):
        s = _stars(float(p), alpha)
        if s:
            ax.text(v + e + span * 0.03, yi, s, va="center", fontsize=10, color="#c0392b")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title("Standardized β by model  ·  ±95% CI")
    ax.set_xlim(-span * 1.15, span * 1.15)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def top_influence(frame: pd.DataFrame, x: str, k: int = 8) -> pd.DataFrame:
    """Top-k units by Cook's distance on the simple (no-controls) fit."""
    work = frame[["unit_uid", "unit_name", x, "y"]].dropna()
    if len(work) < 12:
        return pd.DataFrame()
    import statsmodels.api as sm

    X = sm.add_constant(work[x].to_numpy(dtype=float))
    fit = sm.OLS(work["y"].to_numpy(dtype=float), X).fit()
    cooks = fit.get_influence().cooks_distance[0]
    work = work.assign(cooks_d=cooks).nlargest(k, "cooks_d")
    return work[["unit_name", x, "y", "cooks_d"]].round(3)


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
    influence: Optional[pd.DataFrame],
    image_flags: Dict[str, bool],
    attenuation_note: str,
) -> None:
    now_utc = datetime.now(timezone.utc)
    rel = lambda p: Path(images_dir.name) / p.name  # noqa: E731
    lines: List[str] = []
    mode = record["mode"]
    multi = bool(record.get("models"))

    subtitle = "Pairwise VQA regression  ·  " + (
        f"{record['y']} ~ {record['x']}" if mode == "regression"
        else f"{record['y']} ~ screen over {len(record['x_list'])} covariates"
    )
    if record.get("controls"):
        subtitle += f"  ·  controls: {', '.join(record['controls'])}"
    if multi:
        subtitle += f"  ·  {record['n_models']} models"
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
    if multi:
        lines.append(f"| models | {', '.join(f'`{m}`' for m in record['models'])} |")
    elif record.get("model_label"):
        lines.append(f"| model | `{record['model_label']}` |")
    lines.append(f"| response | TrueSkill `{record['y']}` per {args.unit_label} |")
    if mode == "regression":
        lines.append(f"| focal covariate | `{record['x']}`"
                     + (" (log10)" if record.get("log_x") else "") + " |")
    else:
        lines.append(f"| screened covariates | {', '.join(f'`{c}`' for c in record['x_list'])} |")
    if record.get("controls"):
        lines.append(f"| controls | {', '.join(f'`{c}`' for c in record['controls'])} |")
    lines.append(f"| weighting | {'WLS 1/σ²' if record.get('wls') else 'OLS'} (HC3 robust SEs) |")
    lines.append(f"| covariate source | {record['x_source']} |")
    lines.append(f"| rows (responses) | {record['n_rows']:,} |")
    lines.append(f"| rated units | {record['n_units_rated']:,} |")
    lines.append("")
    lines.append(attenuation_note)
    lines.append("")

    def _unit_table(sub: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            **({"Model": sub["model_label"]} if multi else {}),
            **({"x": sub["x"]} if mode == "screen" else {}),
            "n": sub["n_units"].astype(int),
            "R²": sub["r2"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else ""),
            "R² adj": sub["r2_adj"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else ""),
            "β": sub["beta"].map(lambda v: f"{v:+.4g}" if np.isfinite(v) else ""),
            "β (std)": sub["beta_std"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
            "95% CI": [
                f"[{lo:+.4g}, {hi:+.4g}]" if np.isfinite(lo) else ""
                for lo, hi in zip(sub["beta_ci_lo"], sub["beta_ci_hi"])
            ],
            "p": sub["p"].map(_fmt_p),
            **({"p_adj": sub["p_adj"].map(_fmt_p)} if "p_adj" in sub.columns else {}),
            **({"partial R²": sub["partial_r2"].map(
                lambda v: f"{v:.4f}" if np.isfinite(v) else "")}
               if record.get("controls") else {}),
            "Spearman ρ": sub["spearman_rho"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
        })

    def _pair_table(sub: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            **({"Model": sub["model_label"]} if multi else {}),
            **({"x": sub["x"]} if mode == "screen" else {}),
            "direct pairs": sub["n_pairs"].astype(int),
            "slope (score/Δx)": sub["pair_slope"].map(
                lambda v: f"{v:+.4g}" if np.isfinite(v) else ""),
            "p": sub["pair_p"].map(_fmt_p),
            "R²": sub["pair_r2"].map(lambda v: f"{v:.4f}" if np.isfinite(v) else ""),
            "Spearman ρ": sub["pair_spearman_rho"].map(
                lambda v: f"{v:+.3f}" if np.isfinite(v) else ""),
        })

    lines += ["\\newpage", "", "# Unit-level regression", ""]
    lines.append(
        f"Per-{args.unit_label} TrueSkill ratings regressed on the covariate"
        + (" with controls" if record.get("controls") else "")
        + ". HC3 robust standard errors"
        + ("; weighted by 1/σ²." if record.get("wls") else ".")
    )
    lines.append("")
    lines.append(_md_table(_unit_table(results)))
    lines.append("")
    if image_flags.get("scatter"):
        lines.extend(_centered_image(rel(images_dir / "scatter_fit.png"),
                                     "Scatter with fit", "width=74%"))
    if image_flags.get("screen"):
        lines.extend(_centered_image(rel(images_dir / "screen.png"),
                                     "Covariate screen", "width=74%"))
    if image_flags.get("forest"):
        lines.extend(_centered_image(rel(images_dir / "forest_beta.png"),
                                     "Standardized β by model", "width=72%"))

    lines += ["\\newpage", "", "# Pair-level validation", ""]
    lines.append(
        "Ordinal head-to-head score regressed on Δx over direct pairs where "
        "both sides have a covariate value (repeats collapsed by canonical "
        "pair). Immune to TrueSkill coupling — this is the clean causal-of-"
        "judgment check."
    )
    lines.append("")
    lines.append(_md_table(_pair_table(results)))
    lines.append("")
    if image_flags.get("pair_level"):
        lines.extend(_centered_image(rel(images_dir / "pair_level.png"),
                                     "Judgment vs covariate difference", "width=74%"))
    if image_flags.get("residuals"):
        lines += ["\\newpage", "", "# Diagnostics", ""]
        lines.extend(_centered_image(rel(images_dir / "residuals.png"),
                                     "Residuals vs fitted", "width=70%"))
    if influence is not None and not influence.empty:
        lines.append("## Most influential units (Cook's distance, simple fit)")
        lines.append("")
        lines.append(_md_table(influence.rename(columns={
            "unit_name": args.unit_label.capitalize(), "y": record["y"],
            "cooks_d": "Cook's d"})))
        lines.append("")

    # ---- Caveats ----
    lines += ["\\newpage", "", "# Caveats", ""]
    lines.append(
        "* TrueSkill μ values are jointly estimated from the comparison graph "
        "and are not i.i.d.; unit-level CIs/p-values are approximate. The "
        "pair-level slope is the coupling-free check.\n"
        "* Measurement noise in μ attenuates R² toward zero — see the "
        "attenuation note in the overview.\n"
        "* Covariate coverage is rarely complete; units without a value are "
        "dropped (counts in the tables). Joins describe the covered subset "
        "only (e.g. DOE schools for poverty).\n"
        "* Regression describes association, not causation; covariates "
        "correlate with the streetscape the VLM actually sees."
        + ("\n* Screen mode is exploratory: BH within the screen, but "
           "cross-experiment fishing is not corrected." if mode == "screen" else "")
        + ("\n* Models share the same pair set; per-model fits are correlated "
           "and replication counts overstate independence." if multi else "")
    )
    lines.append("")
    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _build_unit_frame(
    ratings: pd.DataFrame,
    *,
    y: str,
    min_comparisons: int,
    value_maps: Dict[str, pd.Series],
) -> pd.DataFrame:
    rated = ratings[ratings["n_comparisons"] >= min_comparisons].copy()
    rated["unit_uid"] = rated["unit_uid"].astype(str)
    frame = rated[["unit_uid", "unit_name", "mu", "sigma", "ts_conservative"]].copy()
    frame["y"] = frame[y]
    for col, m in value_maps.items():
        frame[col] = frame["unit_uid"].map(m)
    return frame


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
    if bool(args.x) == bool(args.x_list):
        raise SystemExit("Provide exactly one of --x or --x-list.")
    mode = "regression" if args.x else "screen"
    x_list = ([args.x] if args.x else
              [c.strip() for c in args.x_list.split(",") if c.strip()])
    controls = [c.strip() for c in (args.controls or "").split(",") if c.strip()]
    if mode == "screen" and controls:
        raise SystemExit("--controls is supported in single (--x) mode only.")
    if mode == "screen" and len(x_list) < 2:
        raise SystemExit("--x-list needs at least two covariates.")

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

    # ---- Load each source + value maps ----
    all_cols = x_list + controls
    per_model: List[Dict[str, Any]] = []
    x_source = ""
    for label, out_pq, pairs_pq in sources:
        # Probe whether the focal covariate is surfaced on pairs.parquet.
        df, x_from_pairs = load_run(out_pq, pairs_pq, x_list[0])
        value_maps: Dict[str, pd.Series] = {}
        for col in all_cols:
            from_pairs = f"{col}_a" in df.columns and f"{col}_b" in df.columns
            numeric = col in x_list  # controls may be categorical
            try:
                value_maps[col] = build_unit_value_map(
                    df, col,
                    column_from_pairs=from_pairs,
                    unit_metadata_parquet=args.unit_metadata_parquet,
                    unit_metadata_id_column=args.unit_metadata_id_column,
                    numeric=numeric,
                )
            except SystemExit:
                raise
        if args.log_x:
            for col in x_list:
                v = value_maps[col]
                value_maps[col] = np.log10(v[v > 0])
        ratings = _compute_trueskill(df, draw_prob=args.draw_prob)
        frame = _build_unit_frame(
            ratings, y=args.y, min_comparisons=args.min_comparisons,
            value_maps=value_maps,
        )
        if label is None:
            run_config = _load_run_config(out_pq, None)
            model_src = str(((run_config or {}).get("model") or {}).get("model_source") or "")
            label = Path(model_src).name if model_src else ""
        per_model.append({
            "label": label, "df": df, "frame": frame, "value_maps": value_maps,
        })
        x_source = (
            "pairs.parquet metadata" if x_from_pairs
            else f"external join: {args.unit_metadata_parquet} on {args.unit_metadata_id_column}"
        )
    n_models = len(per_model)
    ref = per_model[0]
    print(f"Covariate source:  {x_source}")
    cov_n = int(ref["frame"][x_list[0]].notna().sum())
    print(f"Units with {x_list[0]}: {cov_n:,} / {len(ref['frame']):,} rated")

    # ---- Registry check ----
    id_inputs = {
        "tool": "regression",
        "source_parquet": source_id_path,
        "models": sorted(m["label"] for m in per_model) if multi else None,
        "y": args.y,
        "x": args.x,
        "x_list": sorted(x_list) if mode == "screen" else None,
        "controls": sorted(controls) or None,
        "log_x": bool(args.log_x),
        "wls": bool(args.wls),
        "unit_metadata_parquet": (
            str(args.unit_metadata_parquet.resolve()) if args.unit_metadata_parquet else None
        ),
        "unit_metadata_id_column": args.unit_metadata_id_column,
        "draw_prob": args.draw_prob,
        "min_comparisons": args.min_comparisons,
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

    # ---- Fit (per model × per covariate) ----
    rows: List[Dict[str, Any]] = []
    for m in per_model:
        for x in x_list:
            row = {"model_label": m["label"]}
            row.update(fit_unit_level(m["frame"], x=x, controls=controls, wls=args.wls))
            row.update(fit_pair_level(m["df"], m["value_maps"][x]))
            rows.append(row)
    results = pd.DataFrame(rows)
    if mode == "screen":
        # BH within model across the screened covariates.
        results = pd.concat(
            [adjust_pvalues(g, ["p", "pair_p"])
             for _, g in results.groupby("model_label", sort=False)],
            ignore_index=True,
        )
    results.insert(0, "y", args.y)
    results.insert(0, "experiment_id", experiment_id)

    # ---- Outputs ----
    slug = f"{args.y}.{_slug(x_list[0]) if mode == 'regression' else f'screen-{len(x_list)}'}"
    if controls:
        slug += "." + "-".join(_slug(c) for c in controls)[:30]
    if multi:
        slug += f".multi{n_models}"
        out_md = args.out or (args.aggregation_dir / f"regression.{slug}.md")
    else:
        out_md = args.out or args.output_parquet.with_suffix(f".regression.{slug}.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    images_dir = out_md.parent / f"{out_md.stem}_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_pq = out_md.with_suffix(".tests.parquet")
    results.to_parquet(results_pq, index=False)

    image_flags: Dict[str, bool] = {}
    xlabel = x_list[0] + (" (log10)" if args.log_x else "")
    if mode == "regression" and not multi:
        r0 = results.iloc[0]
        image_flags["scatter"] = _plot_scatter_fit(
            ref["frame"], x_list[0], images_dir / "scatter_fit.png",
            xlabel=xlabel, ylabel=f"TrueSkill {args.y}", r2=float(r0["r2"]),
        )
        image_flags["residuals"] = _plot_residuals(
            ref["frame"], x_list[0], images_dir / "residuals.png")
        image_flags["pair_level"] = _plot_pair_level(
            ref["df"], ref["value_maps"][x_list[0]],
            images_dir / "pair_level.png", xlabel=xlabel)
    elif mode == "regression":
        image_flags["forest"] = _plot_forest_beta(
            results, images_dir / "forest_beta.png", alpha=args.alpha,
            xlabel=f"standardized β  ({args.y} ~ {xlabel})")
        image_flags["pair_level"] = _plot_pair_level(
            ref["df"], ref["value_maps"][x_list[0]],
            images_dir / "pair_level.png", xlabel=xlabel)
    else:
        screen_df = results if not multi else results[
            results["model_label"] == ref["label"]]
        image_flags["screen"] = _plot_screen(
            screen_df, images_dir / "screen.png", args.alpha)

    influence = (
        top_influence(ref["frame"], x_list[0])
        if mode == "regression" else None
    )

    # Attenuation context: how big is TrueSkill noise vs the μ spread?
    sigma_ratio = float(ref["frame"]["sigma"].mean() / ref["frame"]["mu"].std()) \
        if ref["frame"]["mu"].std() > 0 else float("nan")
    attenuation_note = (
        f"_Attenuation context: mean TrueSkill σ is {sigma_ratio:.2f}× the "
        f"cross-{args.unit_label} sd of μ; measurement noise biases R² toward "
        "zero by roughly that share of variance._"
    )

    # ---- Registry record ----
    fin = lambda v: float(v) if np.isfinite(v) else None  # noqa: E731
    if mode == "regression":
        if multi:
            r2s = results["r2"].astype(float)
            betas = results["beta_std"].astype(float)
            result_summary = {
                "n_models": n_models,
                "r2_mean": fin(np.nanmean(r2s)),
                "beta_std_mean": fin(np.nanmean(betas)),
                "n_sig": int((results["p"] < args.alpha).sum()),
                "n_positive": int((betas > 0).sum()),
            }
        else:
            r0 = results.iloc[0]
            result_summary = {
                k: fin(r0[k]) for k in (
                    "r2", "r2_adj", "beta", "beta_std", "p", "partial_r2",
                    "spearman_rho", "pair_slope", "pair_p", "pair_r2",
                )
            }
            result_summary["n_units"] = int(r0["n_units"])
            result_summary["n_pairs"] = int(r0["n_pairs"])
    else:
        sig = results[results["p_adj"] < args.alpha]
        best = results.loc[results["r2"].idxmax()] if results["r2"].notna().any() else None
        result_summary = {
            "n_covariates": len(x_list),
            "n_models": n_models,
            "n_significant": int(len(sig)),
            "best_x": str(best["x"]) if best is not None else None,
            "best_r2": fin(best["r2"]) if best is not None else None,
        }

    record: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "source_parquet": source_id_path,
        "model_label": ref["label"] if not multi else None,
        "models": [m["label"] for m in per_model] if multi else None,
        "n_models": n_models,
        "y": args.y,
        "x": args.x,
        "x_list": x_list if mode == "screen" else None,
        "controls": controls or None,
        "log_x": bool(args.log_x),
        "wls": bool(args.wls),
        "x_source": x_source,
        "unit_metadata_parquet": id_inputs["unit_metadata_parquet"],
        "unit_metadata_id_column": args.unit_metadata_id_column,
        "draw_prob": args.draw_prob,
        "min_comparisons": args.min_comparisons,
        "n_rows": int(len(ref["df"])),
        "n_units_rated": int(len(ref["frame"])),
        "results": result_summary,
        "report_md": str(out_md.resolve()),
        "results_parquet": str(results_pq.resolve()),
        "wandb_url": None,
    }

    # ---- Report ----
    title = args.title or (
        f"Regression: {args.y} ~ {xlabel}" if mode == "regression"
        else f"Covariate screen: {args.y} ~ {len(x_list)} covariates"
    )
    if multi:
        title += f"  ·  {n_models} models"
    write_report(
        out_md=out_md, images_dir=images_dir, title=title, args=args,
        record=record, results=results, influence=influence,
        image_flags=image_flags, attenuation_note=attenuation_note,
    )
    print(f"Wrote report:      {out_md}")
    print(f"Wrote results:     {results_pq}")

    pdf_path = None
    if args.pdf:
        pdf_path = _export_pdf(
            out_md, args.pdf_engine, landscape=(args.pdf_orientation == "landscape"))
        if pdf_path is not None:
            print(f"Wrote PDF:         {pdf_path}")

    # ---- W&B mirror ----
    if not args.no_wandb:
        run_label = (
            f"regress_{args.y}_{_slug(x_list[0])}" if mode == "regression"
            else f"regress_{args.y}_screen{len(x_list)}"
        )
        if controls:
            run_label += "_ctrl"
        if multi:
            run_label += f"_multi{n_models}"
        url = mirror_to_wandb(
            record=record, results=results,
            artifact_paths=[out_md, results_pq] + ([pdf_path] if pdf_path else []),
            project=args.wandb_project, entity=args.wandb_entity,
            run_label=run_label, stage="regression",
            extra_tags=[f"x:{x}" for x in x_list],
        )
        record["wandb_url"] = url
        if url:
            print(f"W&B run:           {url}")

    # ---- Registry append ----
    if not args.no_registry:
        append_registry(reg_path, record)
        print(f"Registered:        {experiment_id} → {reg_path}")

    # ---- Console summary ----
    if mode == "regression" and not multi:
        r0 = results.iloc[0]
        print(
            f"\nUnit-level: n={int(r0['n_units'])}, R²={r0['r2']:.4f}, "
            f"β*={r0['beta_std']:+.3f}, p={_fmt_p(r0['p'])}"
            + (f", partial R²={r0['partial_r2']:.4f}" if controls else "")
        )
        print(
            f"Pair-level: n={int(r0['n_pairs'])}, slope={r0['pair_slope']:+.4g}, "
            f"p={_fmt_p(r0['pair_p'])}"
        )
    elif mode == "regression":
        print(
            f"\n{n_models} models: mean R²={result_summary['r2_mean']:.4f}, "
            f"mean β*={result_summary['beta_std_mean']:+.3f}, "
            f"{result_summary['n_sig']}/{n_models} significant, "
            f"{result_summary['n_positive']}/{n_models} positive."
        )
    else:
        print(
            f"\nScreen: {result_summary['n_significant']} of "
            f"{len(x_list) * n_models} fits significant after BH; "
            f"best: {result_summary['best_x']} (R²={result_summary['best_r2']:.4f})."
        )


if __name__ == "__main__":
    main()
