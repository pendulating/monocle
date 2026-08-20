#!/usr/bin/env python
"""Reviewer-2 robustness baselines for the subway-safety pairwise sweep.

Two tiers:

  --tier freebies    No-GPU checks computed from the EXISTING production 100k
                     parquets:
                       P0a  swap-half order robustness — tract-level TrueSkill
                            mu computed separately from the is_swapped==False
                            and ==True halves (balanced counterbalancing gives
                            two disjoint ~50k halves); reports the mu
                            correlation between halves and each half's
                            r(income). High correlation => A/B presentation
                            order does not drive the tract signal.
                       P0b  repeat self-consistency — agreement between
                            repeat_idx==0 and repeat_idx>0 rows of the same
                            canonical pair (the ~10% repeat draw). This is the
                            within-run test-retest ceiling at temperature 0.6
                            and calibrates the Tier-A agreement numbers.

  --tier agreement   Paired per-pair comparison of 1k-prefix perturbation
                     probes against the production baseline, joined on
                     pair_id (the probe draw is a deterministic seed-777
                     prefix of the production draw; the join asserts it).
                     Probe runs are passed as repeated
                       --run <arm>=<model>=<run_dir>
                     where run_dir is the Hydra job dir containing
                     outputs/pairwise/. Arms listed in --flip-arms asked the
                     semantically flipped question ("which looks LESS safe");
                     their labels/scores are negated before comparison — the
                     stage's _canonicalize_label/_invert_label do NOT perform
                     semantic inversion.

Reuses: _canonicalize_label from the pairwise stage, _compute_trueskill from
pairwise_vqa_report (draw_prob=0.05 canon), tract/income machinery from
pairwise_socioeconomic_regression, mirror_to_wandb from
pairwise_analysis_common.

Examples:
    .venv/bin/python scripts/pairwise_reviewer2_baselines.py --tier freebies
    .venv/bin/python scripts/pairwise_reviewer2_baselines.py --tier agreement \\
        --run retest=qwen3.5-9b=multirun/r2a_prompt_subway/22-00-00/1 \\
        --run paraphrase=qwen3.5-9b=multirun/r2a_prompt_subway/22-00-00/3 \\
        --flip-arms flipped --no-wandb
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy import stats as sps  # noqa: E402

from dagspaces.urbanpairvqa.stages.pairwise_vqa import (  # noqa: E402
    _canonicalize_label,
    _INVERT_LABEL,
    _ORDINAL_SCORE,
)
from pairwise_vqa_report import _compute_trueskill  # noqa: E402
from pairwise_socioeconomic_regression import (  # noqa: E402
    DRAW_PROB,
    LAYERS,
    MODEL_NAMES,
    discover_models,
    load_crime,
    load_income,
    load_population,
    load_tracts,
    regress,
    sampleid_to_tract,
)

RESULT_GLOB = "outputs/pairwise/subway_safety_mvp_*.parquet"
ORDINAL = ["MuchLess", "Less", "Same", "More", "MuchMore"]
LABEL_SPACE = ORDINAL + ["NotSure"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_result(run_dir: Path) -> Path:
    hits = sorted(glob.glob(str(run_dir / RESULT_GLOB)))
    if not hits:
        raise SystemExit(f"No {RESULT_GLOB} under {run_dir}")
    return Path(hits[-1])


def _coerce_repeat_idx(df: pd.DataFrame) -> pd.DataFrame:
    if "repeat_idx" in df.columns:
        df["repeat_idx"] = (
            pd.to_numeric(df["repeat_idx"], errors="coerce").fillna(0).astype(int))
    else:
        df["repeat_idx"] = 0
    return df


def _load(pq: Path) -> pd.DataFrame:
    df = pd.read_parquet(pq)
    df["relative_score"] = pd.to_numeric(df["relative_score"], errors="coerce")
    return _coerce_repeat_idx(df)


def production_map(base: Path, gemma12b_run: Path) -> Dict[str, Path]:
    layer = {"base": base, "result_glob": RESULT_GLOB,
             "extra_runs": {"gemma-4-12b": gemma12b_run}}
    return discover_models(layer)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _semantic_flip(labels: pd.Series) -> pd.Series:
    """Negate the safety direction of a flipped-question arm (MuchLess<->
    MuchMore etc.; Same/NotSure fixed). Reuses the stage's inversion table —
    the mapping is the same even though the reason differs (semantic flip
    here, A/B swap there)."""
    return labels.map(lambda v: _INVERT_LABEL.get(str(v), "Same"))


def _weighted_kappa(a: pd.Series, b: pd.Series) -> float:
    """Linear-weighted Cohen's kappa over the 5-point ordinal scale."""
    k = len(ORDINAL)
    idx = {lab: i for i, lab in enumerate(ORDINAL)}
    ai = a.map(idx).to_numpy()
    bi = b.map(idx).to_numpy()
    n = len(ai)
    if n == 0:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (ai, bi), 1.0)
    obs /= n
    pa, pb = obs.sum(axis=1), obs.sum(axis=0)
    exp = np.outer(pa, pb)
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denom = float((w * exp).sum())
    if denom == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / denom)


def _is_parse_fallback(raw: str) -> bool:
    """True when the raw model output only reached "Same" via the stage's
    unparseable-output fallback (no ordinal/equal/abstention token present)."""
    canon = _canonicalize_label(raw)
    if canon != "Same":
        return False
    low = str(raw or "").lower()
    return not any(tok in low for tok in ("same", "equal"))


def _label_dist(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    return np.array([counts.get(lab, 0) for lab in LABEL_SPACE], dtype=float)


def paired_metrics(base: pd.DataFrame, probe: pd.DataFrame, *,
                   flip: bool) -> Dict[str, float]:
    """All Tier-A agreement metrics for one (arm x model), given the
    production baseline and probe frames (both repeat_idx==0)."""
    p = probe.copy()
    if flip:
        p["relative_label"] = _semantic_flip(p["relative_label"])
        p["relative_score"] = -p["relative_score"]

    m = base.merge(p, on="pair_id", suffixes=("_base", "_probe"), how="inner")
    out: Dict[str, float] = {
        "n_probe": float(len(p)),
        "n_joined": float(len(m)),
        "coverage": float(len(m) / max(len(p), 1)),
    }
    if "canonical_pair_id_base" in m.columns and "canonical_pair_id_probe" in m.columns:
        out["canonical_id_match"] = float(
            (m["canonical_pair_id_base"] == m["canonical_pair_id_probe"]).mean())
    if "is_swapped_base" in m.columns and "is_swapped_probe" in m.columns:
        out["swap_match"] = float(
            (m["is_swapped_base"].astype(bool) == m["is_swapped_probe"].astype(bool)).mean())
    img_cols = [c for c in ("image_path_a", "image_path_b")
                if f"{c}_base" in m.columns and f"{c}_probe" in m.columns]
    if img_cols:
        same_img = np.ones(len(m), dtype=bool)
        for c in img_cols:
            same_img &= (m[f"{c}_base"] == m[f"{c}_probe"]).to_numpy()
        out["image_match"] = float(same_img.mean())

    lb, lp = m["relative_label_base"], m["relative_label_probe"]
    out["notsure_base"] = float((lb == "NotSure").mean())
    out["notsure_probe"] = float((lp == "NotSure").mean())
    out["same_base"] = float((lb == "Same").mean())
    out["same_probe"] = float((lp == "Same").mean())
    out["js_divergence"] = float(jensenshannon(_label_dist(lb), _label_dist(lp)) ** 2)

    ok = lb.isin(ORDINAL) & lp.isin(ORDINAL)
    out["n_ordinal"] = float(ok.sum())
    if ok.sum() >= 2:
        out["agree"] = float((lb[ok] == lp[ok]).mean())
        out["kappa_lw"] = _weighted_kappa(lb[ok], lp[ok])
        sb = m.loc[ok, "relative_score_base"].astype(float)
        sp_ = m.loc[ok, "relative_score_probe"].astype(float)
        if sb.nunique() > 1 and sp_.nunique() > 1:
            out["spearman"] = float(sps.spearmanr(sb, sp_)[0])
        out["flip_rate"] = float(((sb > 0) & (sp_ < 0) | (sb < 0) & (sp_ > 0)).mean())
        for flag, tag in ((False, "unswapped"), (True, "swapped")):
            if "is_swapped_base" in m.columns:
                sel = ok & (m["is_swapped_base"].astype(bool) == flag)
                if sel.sum() >= 2:
                    out[f"agree_{tag}"] = float((lb[sel] == lp[sel]).mean())
    return out


def repeat_agreement(df: pd.DataFrame) -> Dict[str, float]:
    """Within-run self-consistency: repeat_idx>0 rows vs their repeat_idx==0
    twin on canonical_pair_id (relative_label is orientation-corrected, so
    counterbalanced repeats compare cleanly)."""
    if "canonical_pair_id" not in df.columns or (df["repeat_idx"] > 0).sum() == 0:
        return {}
    base = df[df["repeat_idx"] == 0][["canonical_pair_id", "relative_label", "relative_score"]]
    rep = df[df["repeat_idx"] > 0][["canonical_pair_id", "relative_label", "relative_score"]]
    m = base.merge(rep, on="canonical_pair_id", suffixes=("_0", "_r"))
    ok = m["relative_label_0"].isin(ORDINAL) & m["relative_label_r"].isin(ORDINAL)
    out = {"repeat_n": float(len(m))}
    if ok.sum() >= 2:
        out["repeat_agree"] = float((m.loc[ok, "relative_label_0"] == m.loc[ok, "relative_label_r"]).mean())
        out["repeat_kappa_lw"] = _weighted_kappa(m.loc[ok, "relative_label_0"], m.loc[ok, "relative_label_r"])
        s0 = m.loc[ok, "relative_score_0"].astype(float)
        sr = m.loc[ok, "relative_score_r"].astype(float)
        out["repeat_flip_rate"] = float(((s0 > 0) & (sr < 0) | (s0 < 0) & (sr > 0)).mean())
    return out


def prefix_check(probe_dir: Path, production_pairs: Path) -> Dict[str, float]:
    """Assert the probe draw is an ordered prefix of the production draw."""
    ppath = _find_result(probe_dir).parent / "pairs.parquet"
    if not ppath.exists() or not production_pairs.exists():
        return {}
    probe = _coerce_repeat_idx(pd.read_parquet(ppath))
    prod = _coerce_repeat_idx(pd.read_parquet(production_pairs))
    pb = probe[probe["repeat_idx"] == 0]["pair_id"].astype(str).tolist()
    pr = prod[prod["repeat_idx"] == 0]["pair_id"].astype(str).tolist()[: len(pb)]
    return {
        "prefix_n": float(len(pb)),
        "prefix_ordered": float(pb == pr),
        "prefix_set_overlap": float(len(set(pb) & set(pr)) / max(len(pb), 1)),
    }


# ---------------------------------------------------------------------------
# Tier: freebies (P0a swap-half + P0b repeat consistency)
# ---------------------------------------------------------------------------

def tier_freebies(models: Dict[str, Path], manifest: Path, min_comp: int,
                  outdir: Path) -> pd.DataFrame:
    print("loading tracts + income for the swap-half analysis ...")
    tracts = load_tracts()
    income = load_income()

    needed: set = set()
    frames: Dict[str, pd.DataFrame] = {}
    for name, pq in models.items():
        df = _load(pq)
        frames[name] = df
        d0 = df[df["repeat_idx"] == 0]
        needed |= set(d0["sample_id_a"].astype(str)) | set(d0["sample_id_b"].astype(str))
    s2t = sampleid_to_tract(manifest, tracts, needed)
    print(f"  sample_ids assigned to a tract: {len(s2t)}")

    rows = []
    for name, df in frames.items():
        row: Dict[str, object] = {"model": name}
        d0 = df[df["repeat_idx"] == 0].dropna(subset=["relative_score"]).copy()
        d0["unit_uid_a"] = d0["sample_id_a"].astype(str).map(s2t)
        d0["unit_uid_b"] = d0["sample_id_b"].astype(str).map(s2t)
        d0 = d0.dropna(subset=["unit_uid_a", "unit_uid_b"])
        d0 = d0[d0["unit_uid_a"] != d0["unit_uid_b"]]

        halves = {}
        for flag, tag in ((False, "unswapped"), (True, "swapped")):
            half = d0[d0["is_swapped"].astype(bool) == flag]
            row[f"n_{tag}"] = int(len(half))
            r = _compute_trueskill(half, DRAW_PROB).rename(columns={"unit_uid": "geoid"})
            r = r[r["n_comparisons"] >= min_comp]
            halves[tag] = r
            inc = r.merge(income, on="geoid", how="inner").dropna(subset=["median_income"])
            if len(inc) >= 10:
                row[f"r_income_{tag}"] = float(sps.pearsonr(inc["mu"], inc["median_income"])[0])

        both = halves["unswapped"].merge(halves["swapped"], on="geoid",
                                         suffixes=("_u", "_s"))
        row["n_tracts_both"] = int(len(both))
        if len(both) >= 10:
            row["mu_pearson_halves"] = float(sps.pearsonr(both["mu_u"], both["mu_s"])[0])
            row["mu_spearman_halves"] = float(sps.spearmanr(both["mu_u"], both["mu_s"])[0])

        row.update(repeat_agreement(df))
        rows.append(row)
        print(f"  [{name}] " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items() if k != "model"))

    res = pd.DataFrame(rows)
    res.to_parquet(outdir / "freebies_metrics.parquet")
    return res


# ---------------------------------------------------------------------------
# Tier: agreement (probe vs production prefix)
# ---------------------------------------------------------------------------

def tier_agreement(runs: List[Tuple[str, str, Path]], models: Dict[str, Path],
                   flip_arms: set, production_pairs: Path,
                   outdir: Path,
                   baseline_runs: Optional[Dict[str, Path]] = None) -> pd.DataFrame:
    """``baseline_runs`` (model -> probe run dir, e.g. the A0 retest run)
    replaces the production parquet as the comparison target for that model.
    Probe-vs-probe comparisons are image-identical (all 1k probes share the
    same within-unit image draw), whereas probe-vs-production comparisons
    include image-draw noise (~7% same photo) on top of decoding noise —
    judge each against the matching A0 ceiling."""
    baseline_runs = baseline_runs or {}
    prod_cache: Dict[str, pd.DataFrame] = {}
    rows = []
    for arm, model, run_dir in runs:
        if model not in models and model not in baseline_runs:
            raise SystemExit(f"--run {arm}={model}=...: unknown model "
                             f"{model!r} (have {sorted(models)})")
        if model not in prod_cache:
            src = (_find_result(baseline_runs[model]) if model in baseline_runs
                   else models[model])
            prod_cache[model] = _load(src)
        base = prod_cache[model]
        base0 = base[base["repeat_idx"] == 0]

        probe_pq = _find_result(run_dir)
        probe = _load(probe_pq)
        probe0 = probe[probe["repeat_idx"] == 0]

        row: Dict[str, object] = {"arm": arm, "model": model,
                                  "run_dir": str(run_dir), "flipped": arm in flip_arms}
        row.update(prefix_check(run_dir, production_pairs))
        row.update(paired_metrics(base0, probe0, flip=arm in flip_arms))
        row.update(repeat_agreement(probe))
        if "answer" in probe0.columns:
            row["parse_fallback_rate"] = float(
                probe0["answer"].astype(str).map(_is_parse_fallback).mean())
        rows.append(row)
        print(f"  [{arm} x {model}] " + "  ".join(
            f"{k}={v:.3f}" for k, v in row.items()
            if isinstance(v, float) and k in
            ("coverage", "agree", "kappa_lw", "spearman", "flip_rate",
             "notsure_probe", "parse_fallback_rate")))

    res = pd.DataFrame(rows)
    res.to_parquet(outdir / "agreement_metrics.parquet")
    return res


# ---------------------------------------------------------------------------
# Tier: outcome (Tier B — tract regression under perturbation, 25k prefixes)
# ---------------------------------------------------------------------------

def _tract_mu(df: pd.DataFrame, s2t: Dict[str, str], min_comp: int) -> pd.DataFrame:
    d = df[df["repeat_idx"] == 0].dropna(subset=["relative_score"]).copy()
    d["unit_uid_a"] = d["sample_id_a"].astype(str).map(s2t)
    d["unit_uid_b"] = d["sample_id_b"].astype(str).map(s2t)
    d = d.dropna(subset=["unit_uid_a", "unit_uid_b"])
    d = d[d["unit_uid_a"] != d["unit_uid_b"]]
    r = _compute_trueskill(d, DRAW_PROB).rename(columns={"unit_uid": "geoid"})
    return r[r["n_comparisons"] >= min_comp]


def tier_outcome(runs: List[Tuple[str, str, Path]], models: Dict[str, Path],
                 manifest: Path, min_comp: int, outdir: Path) -> pd.DataFrame:
    """For each perturbed 25k run: tract TrueSkill -> income+crime regression,
    compared like-for-like against the production judgments restricted to the
    SAME pair_id prefix (identical pairs, production prompt + image draw)."""
    print("loading tracts / income / population / crime ...")
    tracts = load_tracts()
    income = load_income()
    population = load_population()
    crime = load_crime(tracts, population)
    cov = income.merge(crime, on="geoid", how="outer")

    needed: set = set()
    frames: List[Tuple[str, str, Path, pd.DataFrame]] = []
    for arm, model, run_dir in runs:
        probe = _load(_find_result(run_dir))
        frames.append((arm, model, run_dir, probe))
        p0 = probe[probe["repeat_idx"] == 0]
        needed |= set(p0["sample_id_a"].astype(str)) | set(p0["sample_id_b"].astype(str))
    for model in {m for _, m, _, _ in frames}:
        prod = _load(models[model])
        p0 = prod[prod["repeat_idx"] == 0]
        needed |= set(p0["sample_id_a"].astype(str)) | set(p0["sample_id_b"].astype(str))
    s2t = sampleid_to_tract(manifest, tracts, needed)
    print(f"  sample_ids assigned to a tract: {len(s2t)}")

    prod_cache: Dict[str, pd.DataFrame] = {}
    rows = []
    for arm, model, run_dir, probe in frames:
        if model not in prod_cache:
            prod_cache[model] = _load(models[model])
        pair_ids = set(probe.loc[probe["repeat_idx"] == 0, "pair_id"].astype(str))
        prod_prefix = prod_cache[model][
            prod_cache[model]["pair_id"].astype(str).isin(pair_ids)]

        row: Dict[str, object] = {"arm": arm, "model": model,
                                  "run_dir": str(run_dir),
                                  "n_pairs": int(len(pair_ids))}
        mus = {}
        for tag, df in (("probe", probe), ("prod", prod_prefix)):
            mu = _tract_mu(df, s2t, min_comp)
            mus[tag] = mu
            res = regress(mu.merge(cov, on="geoid", how="left"), "mu")
            row[f"n_tracts_{tag}"] = int(res.get("n", 0))
            for k in ("pearson_income", "spearman_income", "beta_income",
                      "beta_jinc_percapita", "beta_jcrime_percapita",
                      "p_jcrime_percapita", "beta_jcrime_density",
                      "p_jcrime_density"):
                if k in res:
                    row[f"{k}_{tag}"] = float(res[k])
        both = mus["probe"].merge(mus["prod"], on="geoid", suffixes=("_x", "_y"))
        if len(both) >= 10:
            row["mu_corr_probe_vs_prod"] = float(
                sps.pearsonr(both["mu_x"], both["mu_y"])[0])
        rows.append(row)
        print(f"  [{arm} x {model}] " + "  ".join(
            f"{k}={v:.3f}" for k, v in row.items() if isinstance(v, float)))

    res = pd.DataFrame(rows)
    res.to_parquet(outdir / "outcome_metrics.parquet")
    return res


# ---------------------------------------------------------------------------
# Report + W&B
# ---------------------------------------------------------------------------

_AGREE_COLS = ["coverage", "image_match", "agree", "kappa_lw", "spearman",
               "flip_rate", "same_base", "same_probe", "notsure_base",
               "notsure_probe", "js_divergence", "agree_unswapped",
               "agree_swapped", "repeat_agree", "parse_fallback_rate",
               "prefix_ordered"]
_OUTCOME_COLS = ["n_pairs", "n_tracts_probe", "n_tracts_prod",
                 "pearson_income_probe", "pearson_income_prod",
                 "spearman_income_probe", "spearman_income_prod",
                 "beta_income_probe", "beta_income_prod",
                 "beta_jinc_percapita_probe", "beta_jinc_percapita_prod",
                 "beta_jcrime_percapita_probe", "beta_jcrime_percapita_prod",
                 "p_jcrime_percapita_probe", "p_jcrime_percapita_prod",
                 "mu_corr_probe_vs_prod"]
_FREEBIE_COLS = ["n_unswapped", "n_swapped", "n_tracts_both",
                 "mu_pearson_halves", "mu_spearman_halves",
                 "r_income_unswapped", "r_income_swapped",
                 "repeat_n", "repeat_agree", "repeat_kappa_lw",
                 "repeat_flip_rate"]


def _md_table(df: pd.DataFrame, index_cols: List[str], cols: List[str]) -> List[str]:
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(index_cols + cols) + " |",
             "|" + "---|" * (len(index_cols) + len(cols))]
    for _, r in df.iterrows():
        cells = [str(r[c]) for c in index_cols]
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v)
                         else ("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_report(tier: str, res: pd.DataFrame, outdir: Path) -> Path:
    lines = [f"# Reviewer-2 baselines — subway safety ({tier})",
             f"_generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    if tier == "freebies":
        lines += [
            "P0a: tract-level TrueSkill mu computed separately from the two "
            "counterbalanced halves of the production 100k run; "
            "`mu_pearson_halves` is the between-half correlation and "
            "`r_income_*` each half's income coupling. P0b: `repeat_*` is the "
            "within-run self-consistency on the ~10% repeated canonical pairs "
            "(test-retest ceiling at temperature 0.6).", ""]
        lines += _md_table(res, ["model"], _FREEBIE_COLS)
    elif tier == "outcome":
        lines += [
            "Tier B: tract-level TrueSkill -> income+crime regression on each "
            "perturbed 25k-prefix run (`_probe` columns) vs the production "
            "judgments restricted to the SAME pairs (`_prod` columns — "
            "like-for-like baseline: identical pairs, production prompt and "
            "image draw). `mu_corr_probe_vs_prod` is the tract-score "
            "correlation between the two.", ""]
        lines += _md_table(res.sort_values(["model", "arm"]),
                           ["arm", "model"], _OUTCOME_COLS)
    else:
        lines += [
            "Each probe (1k seed-777 prefix) joined to the production "
            "baseline on pair_id. `kappa_lw` = linear-weighted Cohen's kappa "
            "on the 5-point ordinal labels (NotSure rows excluded, rates "
            "reported); `flip_rate` = strict sign flips of relative_score. "
            "Flip-question arms are negated before comparison. Judge every "
            "arm against the `retest` anchor row — that is the temp-0.6 "
            "test-retest ceiling, not 1.0.", ""]
        for model in res["model"].unique():
            lines.append(f"## {model}")
            lines.append("")
            lines += _md_table(res[res["model"] == model].sort_values("arm"),
                               ["arm"], _AGREE_COLS)
    path = outdir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def maybe_wandb(tier: str, res: pd.DataFrame, report: Path, outdir: Path,
                project: str, entity: Optional[str]) -> Optional[str]:
    from pairwise_analysis_common import mirror_to_wandb
    flat: Dict[str, float] = {}
    for _, r in res.iterrows():
        key = f"{r['arm']}/{r['model']}" if "arm" in res.columns else str(r["model"])
        for k, v in r.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                flat[f"{key}/{k}"] = float(v)
    record = {
        "experiment_id": f"reviewer2-{tier}-{datetime.now():%Y%m%d_%H%M%S}",
        "mode": f"reviewer2_{tier}",
        "layer": "subway",
        "models": sorted(res["model"].unique().tolist()),
        "results": flat,
    }
    return mirror_to_wandb(
        record=record, results=res,
        artifact_paths=[report, outdir / f"{tier}_metrics.parquet"],
        project=project, entity=entity,
        run_label=f"reviewer2_{tier}_subway", stage="reviewer2_baseline",
        extra_tags=["layer:subway", f"tier:{tier}"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True,
                    choices=["freebies", "agreement", "outcome"])
    ap.add_argument("--production-base", type=Path,
                    default=LAYERS["subway"]["base"])
    ap.add_argument("--gemma12b-run", type=Path,
                    default=LAYERS["subway"]["extra_runs"]["gemma-4-12b"])
    ap.add_argument("--manifest", type=Path, default=LAYERS["subway"]["manifest"])
    ap.add_argument("--run", action="append", default=[],
                    metavar="ARM=MODEL=RUN_DIR",
                    help="agreement tier: probe run to compare (repeatable)")
    ap.add_argument("--baseline", action="append", default=[],
                    metavar="MODEL=RUN_DIR",
                    help="compare against this probe run (e.g. the A0 retest) "
                         "instead of the production parquet for MODEL — "
                         "image-identical contrast (repeatable)")
    ap.add_argument("--flip-arms", default="flipped",
                    help="comma list of arms whose scores must be negated")
    ap.add_argument("--min-comparisons", type=int, default=20,
                    help="freebies tier: per-half tract inclusion threshold")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--project", default="URBANPAIRVQA")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    outdir = args.outdir or (
        REPO / f"outputs/reviewer2_subway/{args.tier}_{datetime.now():%Y%m%d_%H%M%S}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir: {outdir}")

    models = production_map(args.production_base, args.gemma12b_run)
    print(f"production models: {sorted(models)}")

    if args.tier == "freebies":
        res = tier_freebies(models, args.manifest, args.min_comparisons, outdir)
    else:
        if not args.run:
            raise SystemExit(f"--tier {args.tier} requires at least one --run ARM=MODEL=RUN_DIR")
        runs: List[Tuple[str, str, Path]] = []
        for spec in args.run:
            parts = spec.split("=", 2)
            if len(parts) != 3:
                raise SystemExit(f"Bad --run spec {spec!r}; expected ARM=MODEL=RUN_DIR")
            runs.append((parts[0], parts[1], Path(parts[2])))
        flip_arms = {a.strip() for a in args.flip_arms.split(",") if a.strip()}
        if args.tier == "outcome":
            res = tier_outcome(runs, models, args.manifest,
                               args.min_comparisons, outdir)
            report = write_report(args.tier, res, outdir)
            print(f"report: {report}")
            (outdir / "invocation.json").write_text(
                json.dumps(vars(args), indent=2, default=str))
            if not args.no_wandb:
                url = maybe_wandb(args.tier, res, report, outdir,
                                  args.project, args.entity)
                print(f"[wandb] {url}")
            print("DONE")
            return
        baseline_runs: Dict[str, Path] = {}
        for spec in args.baseline:
            parts = spec.split("=", 1)
            if len(parts) != 2:
                raise SystemExit(f"Bad --baseline spec {spec!r}; expected MODEL=RUN_DIR")
            baseline_runs[parts[0]] = Path(parts[1])
        # Any production run's pairs.parquet works as the prefix reference —
        # the draw is model-independent (same manifest/seed/config).
        production_pairs = sorted(args.production_base.glob(
            "*/outputs/pairwise/pairs.parquet"))
        ref_pairs = production_pairs[0] if production_pairs else Path("/nonexistent")
        res = tier_agreement(runs, models, flip_arms, ref_pairs, outdir,
                             baseline_runs)

    report = write_report(args.tier, res, outdir)
    print(f"report: {report}")
    (outdir / "invocation.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if not args.no_wandb:
        url = maybe_wandb(args.tier, res, report, outdir, args.project, args.entity)
        print(f"[wandb] {url}")
    print("DONE")


if __name__ == "__main__":
    main()
