#!/usr/bin/env python3
"""Generate a multi-model aggregation report for a urbanpairvqa task.

Reads an *aggregation directory* — a directory of symlinked run directories,
one per model — verifies that prompts and pair sets are identical across
runs, and produces a markdown + PDF report comparing the models. Meant to
be the multi-model complement to ``scripts/pairwise_vqa_report.py``.

Phase 1 output: discovery + sanity checks only. Later phases will add the
label-distribution comparison, inter-model agreement, pooled + per-model
TrueSkill, and disagreement drill-downs.

Example:

    python scripts/pairwise_vqa_aggregation_report.py \\
        /share/pierson/matt/mllmsci/machine-beholder/aggregations/libraries_well-maintained \\
        --attribute maintained --unit-label library --unit-label-plural libraries \\
        --title "Libraries · well-maintained · multi-model aggregation"
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

# Sibling helpers. Importing `pairwise_vqa_report` also applies the shared
# matplotlib rcParams polish as a side effect.
sys.path.insert(0, str(Path(__file__).parent))
from pairwise_vqa_report import (  # noqa: E402
    ACCENT,
    ACCENT_WARM,
    ORDINAL_COLORS,
    ORDINAL_ORDER,
    _compute_trueskill,
    _export_pdf,
    _label_distribution,
    _label_entropy,
    _load_run_config,
    _md_table,
    _plot_trueskill_distribution,
    _plot_trueskill_ranking,
    _plot_wordcloud,
    _reasoning_lengths,
    _stitch_pair_jpeg,
)


# Stable per-model color palette. Indexed by model discovery order; callers
# keep ordering consistent across all plots so a model keeps one color.
MODEL_COLORS = [
    "#2b6cb0",  # blue
    "#dd8452",  # orange
    "#1b7837",  # green
    "#762a83",  # purple
    "#d1a93b",  # gold
    "#c0392b",  # red
    "#17becf",  # cyan
    "#8c564b",  # brown
]

# Ordinal label ↔ integer score mapping (must match the stage's own convention).
LABEL_TO_INT = {"MuchLess": -2, "Less": -1, "Same": 0, "More": 1, "MuchMore": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def _model_color(i: int) -> str:
    return MODEL_COLORS[i % len(MODEL_COLORS)]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """One completed model's run inside an aggregation directory."""

    model_label: str      # directory / symlink basename (human-readable)
    model_source: str     # absolute path from Hydra config's model.model_source
    model_name: str       # trailing component of model_source
    out_parquet: Path     # libraries_*.parquet (stage output)
    pairs_parquet: Path   # companion pairs.parquet
    hydra_config: dict    # parsed .hydra/config.yaml
    pipeline_dir: Path    # dir holding pipeline_manifest.json + .slurm_jobs/


def _model_label_from_overrides(subrun: Path, fallback: str) -> str:
    """Extract a clean model label from a subrun's .hydra/overrides.yaml.

    Looks for `- model=<group>/<variant>` and returns `<group>_<variant>` so
    labels match the historical per-model symlink naming (e.g.
    `gemma-4-e2b_instruct`). Falls back to the provided value if no model
    override is present.
    """
    overrides_path = subrun / ".hydra" / "overrides.yaml"
    if overrides_path.exists():
        try:
            overrides = yaml.safe_load(overrides_path.read_text()) or []
        except Exception:
            overrides = []
        for ov in overrides:
            if isinstance(ov, str) and ov.startswith("model="):
                return ov[len("model="):].replace("/", "_")
    return fallback


def _build_run(parent_name: str, base: Path) -> tuple[Optional[Run], Optional[str]]:
    """Assemble a Run from a single Hydra job directory (the one that contains
    `.hydra/`, `outputs/`, etc.).

    Returns (run, skip_reason). Exactly one is non-None.
    """
    cfg_path = base / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None, "missing .hydra/config.yaml"
    cfg = _load_run_config(cfg_path, cfg_path) or {}
    pairwise_dir = base / "outputs" / "pairwise"
    pairs_pq = pairwise_dir / "pairs.parquet"
    if not pairs_pq.exists():
        return None, "missing pairs.parquet"
    outs = sorted(p for p in pairwise_dir.glob("*.parquet") if p.name != "pairs.parquet")
    if not outs:
        return None, "no stage output parquet (run likely incomplete)"
    if len(outs) > 1:
        return None, f"multiple stage output parquets: {[p.name for p in outs]}"
    model_src = str((cfg.get("model") or {}).get("model_source") or "")
    label = _model_label_from_overrides(base, fallback=parent_name)
    return (
        Run(
            model_label=label,
            model_source=model_src,
            model_name=Path(model_src).name if model_src else label,
            out_parquet=outs[0],
            pairs_parquet=pairs_pq,
            hydra_config=cfg,
            pipeline_dir=base,
        ),
        None,
    )


def _discover_runs(agg_dir: Path) -> tuple[list[Run], list[tuple[str, str]]]:
    """Walk the aggregation directory, resolving each subdir/symlink into Runs.

    Supports three layouts:

    1. **Per-model symlinks (legacy)** — each child of ``agg_dir`` is (a
       symlink to) a single-run Hydra multirun dir containing ``0/`` with
       ``.hydra/config.yaml``. The symlink's basename becomes the model label.

    2. **Nested sweep symlink** — one child is (a symlink to) a Hydra
       multirun dir containing *many* numeric job subdirs (``0/``, ``1/``,
       ...). Each numeric subdir is one model; labels are derived from that
       subdir's ``.hydra/overrides.yaml`` ``model=<group>/<variant>`` entry.

    3. **Direct sweep dir** — ``agg_dir`` itself is (a symlink to) a Hydra
       multirun dir: its direct children are ``0/``, ``1/``, ..., plus
       ``multirun.yaml``. Same per-sub labelling as layout 2.

    Layouts 1 and 2 can coexist. Hidden children (dot-prefixed, e.g.
    ``.before_facing_fixes``) are skipped.

    Returns (runs, skipped) where skipped is a list of (name, reason) tuples.
    """
    runs: list[Run] = []
    skipped: list[tuple[str, str]] = []

    # Layout 3: agg_dir itself is a Hydra multirun dir.
    top_numeric = sorted(
        (d for d in agg_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    if (agg_dir / "multirun.yaml").exists() and top_numeric:
        for sub in top_numeric:
            run, reason = _build_run(parent_name=agg_dir.name, base=sub)
            if run is None:
                skipped.append((sub.name, reason or "unknown error"))
                continue
            runs.append(run)
        return runs, skipped

    for child in sorted(agg_dir.iterdir()):
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            is_dir = False
        if not is_dir:
            continue

        # Collect numeric Hydra job subdirs (0/, 1/, ...) — present in both
        # layouts, but multiple of them means "sweep".
        try:
            numeric_subdirs = sorted(
                (d for d in child.iterdir() if d.is_dir() and d.name.isdigit()),
                key=lambda d: int(d.name),
            )
        except OSError:
            numeric_subdirs = []

        if not numeric_subdirs:
            skipped.append((child.name, "no numeric job subdirs (expected 0/, 1/, …)"))
            continue

        is_sweep = (child / "multirun.yaml").exists() and len(numeric_subdirs) > 1

        for sub in numeric_subdirs:
            rel_name = f"{child.name}/{sub.name}" if is_sweep else child.name
            run, reason = _build_run(parent_name=child.name, base=sub)
            if run is None:
                skipped.append((rel_name, reason or "unknown error"))
                continue
            runs.append(run)

    return runs, skipped


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _extract_prompts(runs: list[Run]) -> dict[str, tuple[str, str, str]]:
    """Per-run (system, user_template, schema_repr) triple. Empty strings if missing."""
    out: dict[str, tuple[str, str, str]] = {}
    for r in runs:
        p = r.hydra_config.get("prompt") or {}
        schema = (p.get("structured_output") or {}).get("json_schema") or {}
        out[r.model_label] = (
            (p.get("system") or "").strip(),
            (p.get("user_template") or "").strip(),
            repr(schema),
        )
    return out


def _check_prompt_equality(runs: list[Run]) -> tuple[bool, bool, bool]:
    """Return (system_equal, user_equal, schema_equal) across runs."""
    prompts = _extract_prompts(runs)
    sys_vals = {v[0] for v in prompts.values()}
    usr_vals = {v[1] for v in prompts.values()}
    sch_vals = {v[2] for v in prompts.values()}
    return len(sys_vals) == 1, len(usr_vals) == 1, len(sch_vals) == 1


def _check_pair_identity(runs: list[Run]) -> tuple[set[str], dict[str, int]]:
    """Verify pair_id sets across runs. Returns (intersection, per_run_counts)."""
    sets: list[set[str]] = []
    counts: dict[str, int] = {}
    for r in runs:
        df = pd.read_parquet(r.pairs_parquet, columns=["pair_id"])
        s = set(df["pair_id"].astype(str).tolist())
        sets.append(s)
        counts[r.model_label] = len(s)
    intersection: set[str] = set.intersection(*sets) if sets else set()
    return intersection, counts


# ---------------------------------------------------------------------------
# Long-form dataframe (one row per (model, pair_id))
# ---------------------------------------------------------------------------


UNIT_COLS = ("unit_uid_a", "unit_uid_b", "unit_name_a", "unit_name_b")


def _build_long_df(runs: list[Run], pair_id_filter: Optional[set[str]] = None) -> pd.DataFrame:
    """Concatenate per-model stage outputs into a long-form dataframe."""
    frames: list[pd.DataFrame] = []
    for r in runs:
        out = pd.read_parquet(r.out_parquet)
        pairs = pd.read_parquet(r.pairs_parquet)
        have_units = {"unit_uid_a", "unit_uid_b"}.issubset(pairs.columns)
        merge_cols = ["pair_id"] + [c for c in UNIT_COLS if c in pairs.columns]
        pairs_slim = pairs[merge_cols].copy()
        merged = out.merge(pairs_slim, on="pair_id", how="left", validate="one_to_one")
        if pair_id_filter is not None:
            merged = merged[merged["pair_id"].astype(str).isin(pair_id_filter)]
        merged.insert(0, "model_label", r.model_label)
        merged.insert(1, "model_name", r.model_name)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Wide-format pivot + agreement metrics
# ---------------------------------------------------------------------------


def _build_wide_df(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long_df → one row per pair_id with label_<m> and score_<m> columns."""
    labels = long_df.pivot(index="pair_id", columns="model_label", values="relative_label")
    scores = long_df.pivot(index="pair_id", columns="model_label", values="relative_score")
    labels.columns = [f"label_{c}" for c in labels.columns]
    scores.columns = [f"score_{c}" for c in scores.columns]
    return pd.concat([labels, scores], axis=1).reset_index()


def _compute_agreement(wide_df: pd.DataFrame, models: list[str]) -> dict:
    """Compute per-pair agreement metrics. `models` sets the row/col ordering."""
    import krippendorff  # deferred import — only needed here

    # Ordinal-integer matrix: shape (n_models, n_pairs).
    ord_mat = np.stack([
        wide_df[f"label_{m}"].map(LABEL_TO_INT).to_numpy() for m in models
    ])

    n = len(models)
    kappa = np.full((n, n), np.nan)
    exact_agree = np.zeros((n, n))
    sign_agree = np.zeros((n, n))
    confmats: dict[tuple[str, str], np.ndarray] = {}

    for i in range(n):
        for j in range(n):
            yi, yj = ord_mat[i], ord_mat[j]
            if i == j:
                kappa[i, j] = 1.0
                exact_agree[i, j] = 1.0
                sign_agree[i, j] = 1.0
                continue
            kappa[i, j] = cohen_kappa_score(yi, yj, weights="quadratic")
            exact_agree[i, j] = float((yi == yj).mean())
            sign_agree[i, j] = float((np.sign(yi) == np.sign(yj)).mean())
            if i < j:
                cm = np.zeros((5, 5), dtype=int)
                for a, b in zip(yi, yj):
                    cm[a + 2, b + 2] += 1
                confmats[(models[i], models[j])] = cm

    alpha_ordinal = float(
        krippendorff.alpha(reliability_data=ord_mat, level_of_measurement="ordinal")
    )

    # Joint all-models metrics.
    first = ord_mat[0]
    all_exact_mask = (ord_mat == first).all(axis=0)
    all_sign_mask = (np.sign(ord_mat) == np.sign(first)).all(axis=0)
    all_exact_rate = float(all_exact_mask.mean())
    all_sign_rate = float(all_sign_mask.mean())

    # Mean of off-diagonal pairwise metrics.
    mask = ~np.eye(n, dtype=bool)
    mean_kappa = float(np.nanmean(kappa[mask]))
    mean_exact = float(exact_agree[mask].mean())
    mean_sign = float(sign_agree[mask].mean())

    return {
        "models": models,
        "ord_mat": ord_mat,
        "kappa": kappa,
        "exact_agree": exact_agree,
        "sign_agree": sign_agree,
        "confmats": confmats,
        "krippendorff_alpha": alpha_ordinal,
        "all_exact_rate": all_exact_rate,
        "all_sign_rate": all_sign_rate,
        "mean_kappa": mean_kappa,
        "mean_exact": mean_exact,
        "mean_sign": mean_sign,
    }


# ---------------------------------------------------------------------------
# Same-model self-consistency (entropy over deliberate per-model repeats)
# ---------------------------------------------------------------------------


def _self_consistency_stats(
    long_df: pd.DataFrame, models: list[str]
) -> pd.DataFrame:
    """Per-model self-consistency over canonical pairs sent more than once.

    A portion of each model's budget can be spent on repeats (controlled by
    ``pair_sampler.repeat_count`` / ``pair_sampler.repeat_fraction``). Those
    repeats share a ``canonical_pair_id`` but have distinct ``pair_id`` /
    ``repeat_idx``. This function aggregates same-model variability across
    those repeat groups so the report can quote within-model "would the
    model say the same thing twice?" alongside the between-model κ numbers.

    Per-model columns:
        * ``repeat_groups`` — # canonical pairs with ≥ 2 same-model responses
        * ``repeat_responses`` — total # responses inside those groups
        * ``mean_group_size``
        * ``exact_label_agreement_rate`` — fraction of groups where every
          response shares the same ``relative_label``
        * ``sign_agreement_rate`` — fraction where every response shares
          ``sign(relative_score)``
        * ``mean_within_group_entropy_nats`` — within-group entropy of the
          5-class label distribution, averaged across groups
        * ``mean_within_group_score_var`` — within-group variance of
          ``relative_score`` (5-point ordinal), averaged across groups
        * ``weighted_score_agreement`` — replicates the orchestrator's
          ``repeat_weighted_agreement_mean``: per group,
          ``max(0, 1 − span(score) / 4)``, averaged
        * ``self_kappa_quadratic`` — quadratic-weighted Cohen's κ between
          repeat 0 and repeat 1 over groups with exactly two responses
          (NaN if no such groups)
    """
    if "canonical_pair_id" not in long_df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for m in models:
        sub = long_df[long_df["model_label"] == m]
        if sub.empty:
            continue
        grouped = sub.groupby("canonical_pair_id", dropna=False)
        rep_groups = grouped.filter(lambda g: len(g) > 1)
        if rep_groups.empty:
            rows.append({
                "model_label": m,
                "repeat_groups": 0,
                "repeat_responses": 0,
                "mean_group_size": float("nan"),
                "exact_label_agreement_rate": float("nan"),
                "sign_agreement_rate": float("nan"),
                "mean_within_group_entropy_nats": float("nan"),
                "mean_within_group_score_var": float("nan"),
                "weighted_score_agreement": float("nan"),
                "self_kappa_quadratic": float("nan"),
            })
            continue

        rep_grouped = rep_groups.groupby("canonical_pair_id", dropna=False)
        n_groups = int(rep_grouped.ngroups)
        sizes = rep_grouped.size().to_numpy()

        exact = 0
        signs = 0
        ent_acc: list[float] = []
        var_acc: list[float] = []
        weighted_acc: list[float] = []
        # For self-κ we need exactly-two-response groups, paired by repeat_idx ascending.
        kappa_a: list[int] = []
        kappa_b: list[int] = []

        for _, grp in rep_grouped:
            labels = grp["relative_label"].astype(str).tolist()
            scores = grp["relative_score"].astype(float).to_numpy()
            if labels and len(set(labels)) == 1:
                exact += 1
            sign_set = set(np.sign(scores).astype(int).tolist())
            if len(sign_set) == 1:
                signs += 1
            # Within-group entropy (nats) over the 5-class label distribution.
            # Use observed support only — equivalent to summing over all labels
            # because absent labels contribute zero.
            counts = pd.Series(labels).value_counts(normalize=True).to_numpy()
            ent_acc.append(float(-(counts * np.log(counts)).sum()))
            var_acc.append(float(scores.var(ddof=0)))
            span = float(scores.max() - scores.min()) if scores.size else 0.0
            weighted_acc.append(max(0.0, 1.0 - span / 4.0))

            if len(grp) == 2 and "repeat_idx" in grp.columns:
                ordered = grp.sort_values("repeat_idx")
                ints = ordered["relative_label"].map(LABEL_TO_INT).to_numpy()
                if not np.isnan(ints).any():
                    kappa_a.append(int(ints[0]))
                    kappa_b.append(int(ints[1]))

        self_kappa = float("nan")
        if len(kappa_a) >= 2 and len(set(kappa_a + kappa_b)) > 1:
            self_kappa = float(
                cohen_kappa_score(kappa_a, kappa_b, weights="quadratic")
            )

        rows.append({
            "model_label": m,
            "repeat_groups": n_groups,
            "repeat_responses": int(sizes.sum()),
            "mean_group_size": float(sizes.mean()),
            "exact_label_agreement_rate": exact / n_groups,
            "sign_agreement_rate": signs / n_groups,
            "mean_within_group_entropy_nats": float(np.mean(ent_acc)) if ent_acc else float("nan"),
            "mean_within_group_score_var": float(np.mean(var_acc)) if var_acc else float("nan"),
            "weighted_score_agreement": float(np.mean(weighted_acc)) if weighted_acc else float("nan"),
            "self_kappa_quadratic": self_kappa,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference runtime / throughput
# ---------------------------------------------------------------------------


def _parse_dp_world_size(slurm_log_dir: Path) -> Optional[int]:
    """Recover the data-parallel world size from a stage's GPU log.

    Each chunk progress line is ``[urbanpairvqa_pairwise] DP rank N/M: …``.
    The trailing ``M`` is the DP world size. Falls back to ``None`` when no
    log is found or no DP line is parseable. Reads from the end so we don't
    have to slurp a multi-MB log into memory.
    """
    if not slurm_log_dir.exists():
        return None
    log_files = sorted(slurm_log_dir.rglob("*_log.out"))
    import re
    pattern = re.compile(r"\bDP rank \d+/(\d+)\b")
    for log_path in log_files:
        try:
            # Tail the log — DP progress lines appear throughout, so a few
            # KB from the end is plenty.
            with log_path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 64 * 1024))
                tail = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in reversed(tail.splitlines()):
            m = pattern.search(line)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    continue
    return None


def _collect_runtime_stats(runs: list[Run]) -> pd.DataFrame:
    """One row per model: rows / wall duration / throughput / GPU count.

    Sources:
        * ``pipeline_manifest.json`` — ``nodes.<stage>.duration_s`` and
          ``nodes.<stage>.metadata.rows`` (or pair-sampler metadata).
        * ``.slurm_jobs/<stage>/*_log.out`` — DP world size (best-effort).
        * Hydra config — ``model.engine_kwargs.tensor_parallel_size``.

    Throughput is rows/s overall (= rows / wall) and per-GPU
    (= rows / (wall × DP × TP)). ``NaN`` cells indicate the underlying
    metric was not present in the manifest or log.
    """
    import json as _json

    rows: list[dict] = []
    for r in runs:
        row: dict = {
            "model_label": r.model_label,
            "model_name": r.model_name,
        }
        manifest_path = r.pipeline_dir / "pipeline_manifest.json"
        duration_s: Optional[float] = None
        n_rows: Optional[int] = None
        stage_name: Optional[str] = None
        if manifest_path.is_file():
            try:
                manifest = _json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
            nodes = manifest.get("nodes") or {}
            # First node whose stage is pairwise_vqa (typically the only one).
            for nname, ndata in nodes.items():
                if (ndata.get("stage") or "") == "pairwise_vqa":
                    stage_name = nname
                    duration_s = ndata.get("duration_s")
                    md = ndata.get("metadata") or {}
                    n_rows = md.get("rows") or md.get("pairs_sampled")
                    break

        # Tensor parallel size from Hydra config; defaults to 1.
        tp_size = 1
        try:
            engine_kwargs = (
                ((r.hydra_config.get("model") or {}).get("engine_kwargs") or {})
            )
            tp_size = int(engine_kwargs.get("tensor_parallel_size") or 1)
        except (TypeError, ValueError):
            tp_size = 1

        # DP world size from the slurm log (best-effort).
        slurm_log_dir = r.pipeline_dir / ".slurm_jobs" / (stage_name or "pairwise")
        dp_size = _parse_dp_world_size(slurm_log_dir)
        gpu_count = (dp_size or 1) * tp_size

        # Repeat fraction / count from sampler config — surfaces the budget
        # spent on entropy estimation alongside throughput.
        sampler = r.hydra_config.get("pair_sampler") or {}
        repeat_frac = float(sampler.get("repeat_fraction") or 0.0)
        repeat_cnt = int(sampler.get("repeat_count") or 0)

        throughput = (n_rows / duration_s) if (n_rows and duration_s) else float("nan")
        per_gpu = (
            n_rows / (duration_s * gpu_count)
            if (n_rows and duration_s and gpu_count)
            else float("nan")
        )

        row.update({
            "rows": int(n_rows) if n_rows else 0,
            "duration_s": float(duration_s) if duration_s else float("nan"),
            "dp_world_size": int(dp_size) if dp_size else float("nan"),
            "tp_size": int(tp_size),
            "gpu_count": int(gpu_count),
            "throughput_rows_per_s": float(throughput),
            "throughput_per_gpu_rows_per_s": float(per_gpu),
            "seconds_per_row": (duration_s / n_rows) if (n_rows and duration_s) else float("nan"),
            "repeat_count": repeat_cnt,
            "repeat_fraction": repeat_frac,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TrueSkill: pooled, per-model, normalized
# ---------------------------------------------------------------------------


def _compute_pooled_trueskill(long_df: pd.DataFrame, draw_prob: float) -> pd.DataFrame:
    """One TrueSkill fit over the concatenation of all (model × pair) comparisons.

    Each row counts equally — this is the 'non-reweighted' variant.
    """
    return _compute_trueskill(long_df, draw_prob)


def _compute_per_model_trueskill(
    long_df: pd.DataFrame,
    models: list[str],
    draw_prob: float,
) -> dict[str, pd.DataFrame]:
    """Per-model TrueSkill fits. Returns {model_label: ratings_df}."""
    out: dict[str, pd.DataFrame] = {}
    for m in models:
        sub = long_df[long_df["model_label"] == m]
        out[m] = _compute_trueskill(sub, draw_prob)
    return out


def _compute_normalized_trueskill(
    per_model_ratings: dict[str, pd.DataFrame],
    min_comparisons: int = 1,
) -> pd.DataFrame:
    """Z-score each model's μ across libraries, then average across models.

    Returns a DataFrame with:
      unit_uid, unit_name, z_mean, z_std, n_comparisons_min/max,
      mu_<model_label>, z_<model_label>
    sorted by z_mean descending.
    """
    frames: list[pd.DataFrame] = []
    for model, df in per_model_ratings.items():
        sub = df[["unit_uid", "unit_name", "mu", "n_comparisons"]].copy()
        mu_vals = sub["mu"].to_numpy(dtype=float)
        m = float(mu_vals.mean())
        s = float(mu_vals.std(ddof=0))
        sub["z"] = (mu_vals - m) / s if s > 0 else 0.0
        sub["model"] = model
        frames.append(sub)
    long_z = pd.concat(frames, ignore_index=True)

    z_wide = long_z.pivot(index="unit_uid", columns="model", values="z")
    mu_wide = long_z.pivot(index="unit_uid", columns="model", values="mu")
    n_wide = long_z.pivot(index="unit_uid", columns="model", values="n_comparisons")

    # unit_name lookup (should be identical across per-model fits).
    name_map = (
        pd.concat([df[["unit_uid", "unit_name"]] for df in per_model_ratings.values()])
        .drop_duplicates("unit_uid")
        .set_index("unit_uid")["unit_name"]
    )

    out = pd.DataFrame({
        "unit_uid": z_wide.index,
        "unit_name": [name_map.get(u, u) for u in z_wide.index],
        "z_mean": z_wide.mean(axis=1).values,
        "z_std": z_wide.std(axis=1, ddof=0).values,
        "n_comparisons_min": n_wide.min(axis=1).values.astype(int),
        "n_comparisons_max": n_wide.max(axis=1).values.astype(int),
    })
    for model in per_model_ratings.keys():
        out[f"mu_{model}"] = out["unit_uid"].map(mu_wide[model]).astype(float)
        out[f"z_{model}"] = out["unit_uid"].map(z_wide[model]).astype(float)

    out = out[out["n_comparisons_min"] >= min_comparisons]
    return out.sort_values("z_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_throughput(stats: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of per-model throughput (overall + per-GPU).

    Two side-by-side panels: aggregate rows/s on the left, per-GPU rows/s
    on the right. Bars use the same per-model color palette as the rest of
    the report. Models with NaN throughput (manifest missing) are dropped.
    """
    df = stats.dropna(subset=["throughput_rows_per_s"]).copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.text(
            0.5, 0.5, "No throughput data — pipeline_manifest.json missing",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    df = df.sort_values("throughput_rows_per_s", ascending=True).reset_index(drop=True)
    labels = df["model_label"].tolist()
    overall = df["throughput_rows_per_s"].to_numpy()
    per_gpu = df["throughput_per_gpu_rows_per_s"].to_numpy()
    colors = [_model_color(i) for i in range(len(labels))]

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, max(3.0, 0.55 * len(labels) + 1.6)), sharey=True
    )
    for ax, vals, title in (
        (axes[0], overall, "Aggregate throughput  ·  rows / second"),
        (axes[1], per_gpu, "Per-GPU throughput  ·  rows / second / GPU"),
    ):
        bars = ax.barh(labels, vals, color=colors, edgecolor="#444", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {v:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#222",
            )
        ax.set_title(title)
        ax.xaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
        ax.set_axisbelow(True)
        # Headroom for the trailing text label.
        valid = vals[~np.isnan(vals)]
        if valid.size:
            ax.set_xlim(0, valid.max() * 1.18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_per_model_label_distribution(
    long_df: pd.DataFrame,
    runs: list[Run],
    out_path: Path,
) -> pd.DataFrame:
    """Grouped bar chart: ordinal label proportion per model. Returns the
    proportions dataframe for reuse in the overview tables."""
    n_models = len(runs)
    labels = ORDINAL_ORDER
    x = np.arange(len(labels))
    width = 0.8 / max(n_models, 1)

    # Compute proportions in a stable (models × labels) matrix.
    rows = []
    for r in runs:
        sub = long_df[long_df["model_label"] == r.model_label]
        props = sub["relative_label"].value_counts(normalize=True)
        rows.append({"Model": r.model_label, **{lbl: float(props.get(lbl, 0.0)) for lbl in labels}})
    prop_df = pd.DataFrame(rows).set_index("Model")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, r in enumerate(runs):
        vals = prop_df.loc[r.model_label, labels].to_numpy() * 100.0
        offs = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(
            offs,
            vals,
            width=width * 0.92,
            color=_model_color(i),
            edgecolor="#444",
            linewidth=0.4,
            label=r.model_label,
        )
        for bar, v in zip(bars, vals):
            if v >= 0.3:  # skip near-zero bars (no bar to annotate cleanly)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#333",
                )

    # Light vertical dividers between label groups for readability.
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="#eeeeee", linewidth=0.8, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("proportion (%)")
    ax.set_title("Ordinal label distribution  ·  per model")
    ax.set_ylim(0, max(prop_df.to_numpy().max() * 100.0 * 1.28, 1.0))
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(n_models, 4),
        frameon=False,
    )
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return prop_df


def _plot_agreement_heatmap(
    matrix: np.ndarray,
    models: list[str],
    out_path: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = "{:.2f}",
    pct: bool = False,
) -> None:
    """Symmetric N×N heatmap with cell annotations. Works for κ or agreement rates."""
    n = len(models)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * n + 3.5), max(5.5, 1.2 * n + 3)))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_yticklabels(models)
    # Threshold for picking white vs dark text based on normalized cell value.
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            norm_v = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            text_color = "white" if norm_v < 0.28 or norm_v > 0.80 else "#111111"
            s = (f"{v * 100:.1f}%" if pct else fmt.format(v))
            ax.text(j, i, s, ha="center", va="center", color=text_color, fontsize=11, fontweight="semibold")
    ax.set_title(title)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(
    confmats: dict[tuple[str, str], np.ndarray],
    out_path: Path,
) -> None:
    """Small-multiples grid of row-normalized 5×5 confusion matrices."""
    n = len(confmats)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 4.3 * nrows),
        squeeze=False,
    )
    labels = ORDINAL_ORDER
    for idx, ((ma, mb), cm) in enumerate(confmats.items()):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{ma}\nvs  {mb}", fontsize=9, pad=8)
        for i in range(5):
            for j in range(5):
                v = cm_norm[i, j]
                if v >= 0.005:
                    ax.text(
                        j, i,
                        f"{v * 100:.0f}",
                        ha="center", va="center",
                        color="white" if v > 0.55 else "#222222",
                        fontsize=7.5,
                    )
        ax.set_xlabel(f"→ {mb}", fontsize=8)
        ax.set_ylabel(f"{ma} →", fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
    # Hide any unused axes.
    for idx in range(n, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")
    fig.suptitle("Pairwise label confusion  ·  row-normalized, % of row", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_normalized_ranking(
    norm_df: pd.DataFrame,
    out_path: Path,
    attribute: str,
    unit_label_plural: str,
    highlight_n: int = 5,
) -> None:
    """Sorted z-score ranking with cross-model std error bars."""
    if norm_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No normalized ratings", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    sorted_r = norm_df.sort_values("z_mean", ascending=False).reset_index(drop=True)
    n = len(sorted_r)
    x = np.arange(n)
    z = sorted_r["z_mean"].to_numpy()
    s = sorted_r["z_std"].to_numpy()

    h = min(highlight_n, n // 3) if n else 0
    colors = np.full(n, "#9bb4d1", dtype=object)
    if h > 0:
        colors[:h] = "#1b7837"
        colors[-h:] = "#762a83"

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.errorbar(
        x, z, yerr=s,
        fmt="none", ecolor="#cccccc", elinewidth=0.5, capsize=0, alpha=0.7,
    )
    ax.scatter(x, z, s=10, c=list(colors), alpha=0.9, linewidths=0)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, label="z = 0 (cross-library mean)")
    if h > 0:
        ax.annotate(f"top {h}", xy=(x[h - 1], z[:h].min()), xytext=(6, -2),
                    textcoords="offset points", fontsize=9, color="#1b7837", va="top")
        ax.annotate(f"bottom {h}", xy=(x[-h], z[-h:].max()), xytext=(-6, 2),
                    textcoords="offset points", fontsize=9, color="#762a83",
                    va="bottom", ha="right")
    ax.set_xlabel(f"rank  (most → least {attribute})")
    ax.set_ylabel(r"mean z-score across models    (error bar = cross-model σ)")
    ax.set_title(f"Normalized TrueSkill  ·  {n} {unit_label_plural}  ·  per-model μ z-scored, then averaged")
    ax.set_xlim(-1, n)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5, color="#bbbbbb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_model_mu_scatter(
    per_model_ratings: dict[str, pd.DataFrame],
    models: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """Pairwise scatter of per-model μ. Returns a correlation table for the report."""
    import scipy.stats as stats

    pairs = [(i, j) for i in range(len(models)) for j in range(i + 1, len(models))]
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 4.3 * nrows),
        squeeze=False,
    )

    corr_rows = []
    for idx, (i, j) in enumerate(pairs):
        r, c = idx // ncols, idx % ncols
        ax = axes[r][c]
        mi, mj = models[i], models[j]
        df_i = per_model_ratings[mi][["unit_uid", "mu"]].rename(columns={"mu": "mu_i"})
        df_j = per_model_ratings[mj][["unit_uid", "mu"]].rename(columns={"mu": "mu_j"})
        merged = df_i.merge(df_j, on="unit_uid")
        xs = merged["mu_i"].to_numpy()
        ys = merged["mu_j"].to_numpy()

        pearson_r = float(stats.pearsonr(xs, ys).statistic)
        spearman_rho = float(stats.spearmanr(xs, ys).statistic)
        kendall_tau = float(stats.kendalltau(xs, ys).statistic)
        corr_rows.append({
            "Model A": mi,
            "Model B": mj,
            "Pearson r": pearson_r,
            "Spearman ρ": spearman_rho,
            "Kendall τ": kendall_tau,
        })

        ax.scatter(xs, ys, s=10, c=ACCENT, alpha=0.55, linewidths=0)
        lo = float(min(xs.min(), ys.min()))
        hi = float(max(xs.max(), ys.max()))
        pad = (hi - lo) * 0.04 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlabel(mi, fontsize=9)
        ax.set_ylabel(mj, fontsize=9)
        ax.set_title(f"{mi}  vs  {mj}", fontsize=9, pad=6)
        ax.text(
            0.03, 0.97,
            f"r = {pearson_r:+.2f}\nρ = {spearman_rho:+.2f}\nτ = {kendall_tau:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", linewidth=0.4),
        )
        ax.tick_params(labelsize=8)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.4, color="#cccccc")
        ax.set_axisbelow(True)

    for idx in range(n_pairs, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r][c].axis("off")

    fig.suptitle("Per-model TrueSkill μ  ·  pairwise scatter across libraries", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(corr_rows)


# ---------------------------------------------------------------------------
# Example pair selection + rendering (Phase 5)
# ---------------------------------------------------------------------------


def _pair_side(row: pd.Series, unit_uid: str) -> str:
    if str(row.get("unit_uid_a", "")) == unit_uid:
        return "A"
    if str(row.get("unit_uid_b", "")) == unit_uid:
        return "B"
    return "?"


def _pair_metadata_source(long_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per pair_id with all pair-level metadata (image paths, unit identity).

    These fields are model-independent so any one model's rows suffice. Indexed
    by pair_id for fast lookup.
    """
    cols = [
        "pair_id", "image_path_a", "image_path_b", "unit_uid_a", "unit_uid_b",
        "unit_name_a", "unit_name_b", "presented_order", "is_swapped",
    ]
    cols = [c for c in cols if c in long_df.columns]
    first = long_df[long_df["model_label"] == models[0]][cols].copy()
    return first.set_index("pair_id")


def _select_top_bottom_example_pairs(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    normalized: pd.DataFrame,
    models: list[str],
    top_bottom_n: int,
    seed: int,
) -> list[dict]:
    """One random example pair per top-N / bottom-N library (by normalized ẑ).

    Each entry includes the pair's per-model labels so the caption can quote them.
    """
    if normalized.empty or top_bottom_n <= 0:
        return []
    rng = np.random.default_rng(seed)
    sorted_n = normalized.sort_values("z_mean", ascending=False).reset_index(drop=True)
    top = sorted_n.head(top_bottom_n)
    bot = sorted_n.tail(top_bottom_n).iloc[::-1]

    # Any one model's rows suffice for pair metadata (image paths, unit identity).
    meta = _pair_metadata_source(long_df, models)
    label_cols = [f"label_{m}" for m in models]
    wide_indexed = wide_df.set_index("pair_id")[label_cols]

    selections: list[dict] = []
    for section, subset in (("top", top), ("bottom", bot)):
        for rank, rec in enumerate(subset.itertuples(index=False), start=1):
            uid = str(rec.unit_uid)
            candidates = meta[(meta["unit_uid_a"].astype(str) == uid) |
                              (meta["unit_uid_b"].astype(str) == uid)]
            if candidates.empty:
                continue
            pair_id = str(candidates.index[int(rng.integers(0, len(candidates)))])
            pair_row = meta.loc[pair_id]
            per_model_labels = {m: wide_indexed.loc[pair_id, f"label_{m}"] for m in models}
            selections.append({
                "section": section,
                "rank": rank,
                "unit_uid": uid,
                "unit_name": rec.unit_name,
                "z_mean": float(rec.z_mean),
                "z_std": float(rec.z_std),
                "pair_id": pair_id,
                "pair_row": pair_row,
                "per_model_labels": per_model_labels,
            })
    return selections


def _select_high_variance_pairs(
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    models: list[str],
    k: int,
) -> list[dict]:
    """Top-K pairs ranked by cross-model ordinal-label variance."""
    if k <= 0 or wide_df.empty:
        return []
    ord_mat = np.stack([
        wide_df[f"label_{m}"].map(LABEL_TO_INT).to_numpy(dtype=float)
        for m in models
    ])  # (n_models, n_pairs)
    var_vec = ord_mat.var(axis=0, ddof=0)
    scratch = wide_df[["pair_id"]].copy()
    scratch["label_variance"] = var_vec
    scratch["label_range"] = ord_mat.max(axis=0) - ord_mat.min(axis=0)
    top_pairs = scratch.nlargest(k, "label_variance").reset_index(drop=True)

    meta = _pair_metadata_source(long_df, models)
    label_cols = [f"label_{m}" for m in models]
    wide_indexed = wide_df.set_index("pair_id")[label_cols]

    selections: list[dict] = []
    for i, r in enumerate(top_pairs.itertuples(index=False), start=1):
        pair_id = str(r.pair_id)
        if pair_id not in meta.index:
            continue
        pair_row = meta.loc[pair_id]
        per_model_labels = {m: wide_indexed.loc[pair_id, f"label_{m}"] for m in models}
        selections.append({
            "rank": i,
            "pair_id": pair_id,
            "label_variance": float(r.label_variance),
            "label_range": int(r.label_range),
            "pair_row": pair_row,
            "per_model_labels": per_model_labels,
        })
    return selections


def _render_pair_images(
    top_bottom: list[dict],
    high_var: list[dict],
    images_dir: Path,
    *,
    scale: float,
    quality: int,
) -> tuple[list[dict], list[dict]]:
    """Stitch + JPEG-compress each selection's image pair."""
    pairs_dir = images_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    tb_rendered: list[dict] = []
    for sel in top_bottom:
        uid_short = sel["unit_uid"][:10]
        fname = f"{sel['section']}_{sel['rank']:02d}_{uid_short}.jpg"
        out = pairs_dir / fname
        row = sel["pair_row"]
        size = _stitch_pair_jpeg(
            Path(row["image_path_a"]),
            Path(row["image_path_b"]),
            out,
            scale=scale,
            quality=quality,
        )
        if size is None:
            continue
        tb_rendered.append({**sel, "image_path": out, "image_size": size})

    hv_rendered: list[dict] = []
    for sel in high_var:
        fname = f"disputed_{sel['rank']:02d}.jpg"
        out = pairs_dir / fname
        row = sel["pair_row"]
        size = _stitch_pair_jpeg(
            Path(row["image_path_a"]),
            Path(row["image_path_b"]),
            out,
            scale=scale,
            quality=quality,
        )
        if size is None:
            continue
        hv_rendered.append({**sel, "image_path": out, "image_size": size})

    return tb_rendered, hv_rendered


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


def _centered_image(rel_path: Path, alt: str, width_attr: str = "width=86%") -> list[str]:
    return [
        "::: {.center data-latex=\"\"}",
        f"![{alt}]({rel_path}){{ {width_attr} }}",
        ":::",
        "",
    ]


def _write_phase2_markdown(
    *,
    long_df: pd.DataFrame,
    runs: list[Run],
    prop_df: pd.DataFrame,
    agg_dir: Path,
    title: str,
    attribute: str,
    unit_label: str,
    unit_label_plural: str,
    images_dir: Path,
    out_md: Path,
    reasoning_rows: pd.DataFrame,
    wordcloud_tokens: Optional[int],
    agreement: Optional[dict] = None,
    trueskill_bundle: Optional[dict] = None,
    top_n: int = 20,
    min_comparisons: int = 1,
    most_disputed_k: int = 10,
    top_bottom_image_pairs: Optional[list[dict]] = None,
    disputed_image_pairs: Optional[list[dict]] = None,
    self_consistency: Optional[pd.DataFrame] = None,
    runtime_stats: Optional[pd.DataFrame] = None,
) -> None:
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%B %d, %Y")
    rel = lambda p: Path(images_dir.name) / p.name  # noqa: E731
    n_models = len(runs)

    lines: list[str] = []

    def newpage() -> None:
        lines.append("\\newpage")
        lines.append("")

    # ---- YAML title block ----
    subtitle = f"Multi-model aggregation  ·  {n_models} models  ·  {unit_label_plural}  ·  attribute: {attribute}"
    lines.append("---")
    lines.append(f'title: "{title}"')
    lines.append(f'subtitle: "{subtitle}"')
    lines.append(f'date: "{date_str}"')
    lines.append("---")
    lines.append("")

    # ---- Run overview ----
    newpage()
    lines.append("# Run overview")
    lines.append("")

    n_pairs = int(long_df.groupby("model_label").size().iloc[0])
    n_units = (
        int(pd.concat([long_df["unit_uid_a"], long_df["unit_uid_b"]]).astype(str).nunique())
        if "unit_uid_a" in long_df.columns
        else 0
    )
    lines.append(
        f"**Aggregation directory:** \\path{{{agg_dir}}}  \n"
        f"**Generated:** {now_utc.isoformat(timespec='seconds')} &nbsp;·&nbsp; "
        f"**Attribute:** *{attribute}*"
    )
    lines.append("")

    sampler = runs[0].hydra_config.get("pair_sampler") or {}
    lines.append(
        f"**Sampling:** mode=`{sampler.get('mode','?')}`  ·  "
        f"counterbalance=`{sampler.get('counterbalance_mode','?')}`  ·  "
        f"seed=`{sampler.get('pair_seed','?')}`  ·  "
        f"max_pairs=`{sampler.get('max_pairs','?')}`  ·  "
        f"pairs per model=`{n_pairs:,}`  ·  "
        f"distinct {unit_label_plural}=`{n_units:,}`"
    )
    lines.append("")

    # Per-model roster
    lines.append("## Model roster")
    lines.append("")
    roster_rows = []
    for r in runs:
        sub = long_df[long_df["model_label"] == r.model_label]
        dist = _label_distribution(sub)
        ent = _label_entropy(dist)
        lens = _reasoning_lengths(sub)
        cap = int((~lens["is_empty"]).sum())
        cap_rate = cap / len(sub) if len(sub) else 0.0
        roster_rows.append({
            "Model label": r.model_label,
            "Model": r.model_name,
            "Rows": f"{len(sub):,}",
            "Entropy (nats)": f"{ent:.3f}",
            "Reasoning capture": f"{cap_rate:.1%}",
            "Mean relative score": f"{sub['relative_score'].mean():+.3f}",
        })
    lines.append(_md_table(pd.DataFrame(roster_rows)))
    lines.append("")

    # Per-label proportion table (% for readability).
    lines.append("## Per-label proportions (%)")
    lines.append("")
    pct_df = (prop_df * 100.0).round(1)
    pct_df = pct_df.reset_index().rename(columns={"Model": "Model label"})
    for lbl in ORDINAL_ORDER:
        pct_df[lbl] = pct_df[lbl].map(lambda v: f"{v:.1f}")
    lines.append(_md_table(pct_df))
    lines.append("")

    # ---- Inference runtime / throughput ----
    if runtime_stats is not None and not runtime_stats.empty:
        newpage()
        lines.append("# Inference runtime  ·  per model")
        lines.append("")
        lines.append(
            "Wall-clock duration of the `pairwise_vqa` stage on each model "
            "(from `pipeline_manifest.json`), with derived rows-per-second "
            "throughput. Per-GPU throughput divides by the data-parallel × "
            "tensor-parallel world size (DP from the stage's slurm log; TP "
            "from the model's Hydra config). Use this to compare effective "
            "GPU utilization, not just raw wall time."
        )
        lines.append("")

        def _fmt_dur(seconds: float) -> str:
            if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
                return "–"
            seconds = float(seconds)
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            if h:
                return f"{h}h {m:02d}m {s:02d}s"
            if m:
                return f"{m}m {s:02d}s"
            return f"{s}s"

        def _fmt_int(v: float) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "–"
            return f"{int(v):,}"

        def _fmt_float(v: float, fmt: str = "{:.2f}") -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "–"
            return fmt.format(float(v))

        rt_disp = pd.DataFrame({
            "Model label": runtime_stats["model_label"],
            "Rows": runtime_stats["rows"].map(_fmt_int),
            "Wall time": runtime_stats["duration_s"].map(_fmt_dur),
            "GPUs (DP × TP)": [
                f"{_fmt_int(g)}  ({_fmt_int(d)} × {_fmt_int(t)})"
                for g, d, t in zip(
                    runtime_stats["gpu_count"],
                    runtime_stats["dp_world_size"],
                    runtime_stats["tp_size"],
                )
            ],
            "rows/s (overall)": runtime_stats["throughput_rows_per_s"].map(
                lambda v: _fmt_float(v, "{:.2f}")
            ),
            "rows/s/GPU": runtime_stats["throughput_per_gpu_rows_per_s"].map(
                lambda v: _fmt_float(v, "{:.2f}")
            ),
            "ms/row": runtime_stats["seconds_per_row"].map(
                lambda v: _fmt_float(v * 1000.0 if v == v else v, "{:.0f}")
            ),
        })
        lines.append(_md_table(rt_disp))
        lines.append("")

        if any((runtime_stats["repeat_fraction"] > 0)) or any(
            (runtime_stats["repeat_count"] > 0)
        ):
            lines.append(
                "_All rows include the per-model repeat budget set aside for "
                "self-consistency estimation (see Self-consistency section). "
                "Throughput numbers are unaffected — repeats are just extra "
                "samples — but the row count is `pairs × (1 + repeat_fraction)`._"
            )
            lines.append("")
        lines.extend(
            _centered_image(
                rel(images_dir / "throughput.png"),
                "Per-model inference throughput",
                "width=92%",
            )
        )

    # ---- Prompt (shared across runs) ----
    first_cfg = runs[0].hydra_config
    prompt_cfg = first_cfg.get("prompt") or {}
    sys_prompt = (prompt_cfg.get("system") or "").strip()
    user_template = (prompt_cfg.get("user_template") or "").strip()
    schema = (prompt_cfg.get("structured_output") or {}).get("json_schema") or {}

    if sys_prompt or user_template or schema:
        newpage()
        lines.append("# Prompt")
        lines.append("")
        lines.append(f"_Identical across all {n_models} models._")
        lines.append("")
        if sys_prompt:
            lines.append("## System prompt")
            lines.append("")
            lines.append("```")
            lines.append(sys_prompt)
            lines.append("```")
            lines.append("")
        if user_template:
            lines.append("## User template")
            lines.append("")
            lines.append("```")
            lines.append(user_template)
            lines.append("```")
            lines.append("")
        if schema:
            lines.append("## Structured-output schema")
            lines.append("")
            lines.append("```yaml")
            lines.append(yaml.safe_dump(schema, sort_keys=False).rstrip())
            lines.append("```")
            lines.append("")

    # ---- Label distribution: grouped bar ----
    newpage()
    lines.append("# Label distribution  ·  per model")
    lines.append("")
    lines.append(
        f"Canonical A-vs-B ordinal label proportions across the {n_models} models. "
        "Systematic differences here are label-bias / calibration issues that pooled "
        "TrueSkill would average out silently."
    )
    lines.append("")
    lines.extend(
        _centered_image(
            rel(images_dir / "per_model_label_distribution.png"),
            "Per-model label distribution",
            "width=86%",
        )
    )

    # ---- Inter-model agreement ----
    if agreement is not None:
        models_ord = agreement["models"]
        n = len(models_ord)

        newpage()
        lines.append("# Inter-model agreement")
        lines.append("")
        n_pairs_rated = int(long_df.groupby("model_label").size().iloc[0])
        lines.append(
            f"{n} models rating {n_pairs_rated:,} pairs on the 5-point ordinal scale. "
            "Cohen's κ below is quadratic-weighted (ordinal penalty), so confusing "
            "MuchLess with MuchMore is penalized 16× more than MuchLess with Less."
        )
        lines.append("")

        summary = pd.DataFrame([
            ["Pairs rated by all models", f"{n_pairs_rated:,}"],
            ["Krippendorff's α (ordinal)", f"{agreement['krippendorff_alpha']:.4f}"],
            ["Mean pairwise Cohen's κ (quadratic)", f"{agreement['mean_kappa']:.4f}"],
            ["Mean pairwise exact-label agreement", f"{agreement['mean_exact']:.1%}"],
            ["Mean pairwise sign-only agreement", f"{agreement['mean_sign']:.1%}"],
            [f"All {n} models agree on exact label", f"{agreement['all_exact_rate']:.1%}"],
            [f"All {n} models agree on sign", f"{agreement['all_sign_rate']:.1%}"],
        ], columns=["Metric", "Value"])
        lines.append(_md_table(summary))
        lines.append("")

        # Pairwise agreement table (off-diagonal only), sorted by κ desc.
        kappa = agreement["kappa"]
        exact = agreement["exact_agree"]
        sign = agreement["sign_agree"]
        pair_rows = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_rows.append({
                    "Model A": models_ord[i],
                    "Model B": models_ord[j],
                    "κ (quad.)": kappa[i, j],
                    "Exact agreement": exact[i, j],
                    "Sign agreement": sign[i, j],
                })
        pair_df = (
            pd.DataFrame(pair_rows)
            .sort_values("κ (quad.)", ascending=False)
            .reset_index(drop=True)
        )
        pair_df["κ (quad.)"] = pair_df["κ (quad.)"].map(lambda v: f"{v:.3f}")
        pair_df["Exact agreement"] = pair_df["Exact agreement"].map(lambda v: f"{v:.1%}")
        pair_df["Sign agreement"] = pair_df["Sign agreement"].map(lambda v: f"{v:.1%}")
        lines.append("## Pairwise (sorted by κ, highest first)")
        lines.append("")
        lines.append(_md_table(pair_df))
        lines.append("")

        # Kappa heatmap
        newpage()
        lines.append("## Cohen's κ heatmap  ·  quadratic-weighted")
        lines.append("")
        lines.extend(
            _centered_image(
                rel(images_dir / "kappa_heatmap.png"),
                "Pairwise Cohen's κ (quadratic weights)",
                "width=62%",
            )
        )

        # Sign agreement heatmap
        lines.append("## Sign-only agreement heatmap")
        lines.append("")
        lines.append(
            "Fraction of pairs where two models agree on the sign of `relative_score` "
            "— i.e. both say Less/MuchLess, both Same, or both More/MuchMore. "
            "Less sensitive to calibration than κ: captures *who wins*, ignoring intensity."
        )
        lines.append("")
        lines.extend(
            _centered_image(
                rel(images_dir / "sign_agreement_heatmap.png"),
                "Pairwise sign-only agreement",
                "width=62%",
            )
        )

        # Confusion matrices
        newpage()
        lines.append("## Pairwise confusion")
        lines.append("")
        lines.append(
            "Rows = label emitted by the top-listed model; columns = label emitted by the "
            "other model. Each row is normalized so cells show *% of that row*. "
            "Diagonal = agreement; off-diagonal mass shows the direction of disagreement."
        )
        lines.append("")
        lines.extend(
            _centered_image(
                rel(images_dir / "confusion_matrices.png"),
                "Pairwise confusion matrices",
                "width=92%",
            )
        )

    # ---- Self-consistency (same-model repeats / response entropy) ----
    if self_consistency is not None and not self_consistency.empty:
        # Quote the sampler's repeat budget so the reader can see what
        # fraction of completions was set aside for entropy estimation.
        sampler = runs[0].hydra_config.get("pair_sampler") or {}
        rf = float(sampler.get("repeat_fraction") or 0.0)
        rc = int(sampler.get("repeat_count") or 0)
        if rc > 0:
            budget_note = f"`repeat_count = {rc}` extra observations per model"
        elif rf > 0:
            budget_note = f"`repeat_fraction = {rf:g}` of canonical pairs (~{int(rf * 100)}% of budget)"
        else:
            budget_note = "no explicit repeat budget; only organic collisions"

        newpage()
        lines.append("# Self-consistency  ·  same-model response entropy")
        lines.append("")
        lines.append(
            "A portion of every model's budget is intentionally spent re-asking "
            "the same canonical pair (sampler config: " + budget_note + "). "
            "Repeats share a `canonical_pair_id` but get distinct `pair_id` / "
            "`repeat_idx`, so each model's within-pair variability — i.e. "
            '"would the model agree with itself if asked again?" — is recoverable '
            "after the fact. The columns below are computed only over canonical "
            "pairs that have ≥ 2 same-model responses; models with no repeats "
            "show 0."
        )
        lines.append("")
        lines.append(
            "**Metric guide** &nbsp;·&nbsp; "
            "*Exact agreement* = all responses share the same 5-class label. "
            "*Sign agreement* = all responses agree on Less/Same/More direction "
            "(less sensitive to the MuchLess/Less calibration boundary). "
            "*H̄ (nats)* = mean within-group entropy of the label distribution; "
            "0 = perfectly self-consistent, ln 5 ≈ 1.609 = maximally noisy. "
            "*Var(score)* = mean within-group variance of the −2…+2 ordinal "
            "score. *Score-span score* = `mean(max(0, 1 − span / 4))`, "
            "matching the orchestrator's `repeat_weighted_agreement_mean` "
            "diagnostic. *Self-κ* = quadratic-weighted Cohen's κ between "
            "repeats 0 and 1 over canonical pairs that have exactly two "
            "responses (NaN if too few)."
        )
        lines.append("")

        def _fmt(v: float, fmt: str = "{:.3f}") -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "–"
            return fmt.format(float(v))

        def _fmt_pct(v: float) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "–"
            return f"{float(v):.1%}"

        sc_disp = pd.DataFrame({
            "Model label": self_consistency["model_label"],
            "Repeat groups": self_consistency["repeat_groups"].map(
                lambda v: f"{int(v):,}"
            ),
            "Responses (in groups)": self_consistency["repeat_responses"].map(
                lambda v: f"{int(v):,}"
            ),
            "Mean group size": self_consistency["mean_group_size"].map(
                lambda v: _fmt(v, "{:.2f}")
            ),
            "Exact agreement": self_consistency["exact_label_agreement_rate"].map(_fmt_pct),
            "Sign agreement": self_consistency["sign_agreement_rate"].map(_fmt_pct),
            "H̄ (nats)": self_consistency["mean_within_group_entropy_nats"].map(
                lambda v: _fmt(v, "{:.3f}")
            ),
            "Var(score)": self_consistency["mean_within_group_score_var"].map(
                lambda v: _fmt(v, "{:.3f}")
            ),
            "Score-span score": self_consistency["weighted_score_agreement"].map(
                lambda v: _fmt(v, "{:.3f}")
            ),
            "Self-κ (quad.)": self_consistency["self_kappa_quadratic"].map(
                lambda v: _fmt(v, "{:.3f}")
            ),
        })
        lines.append(_md_table(sc_disp))
        lines.append("")
        lines.append(
            "_Sanity check: a model whose self-κ is near zero is essentially "
            "guessing on repeats — its rankings further down should be read "
            "with that noise floor in mind. Conversely, a model with high "
            "self-κ but low pairwise κ in the previous section is **internally "
            "consistent but disagrees with the others** on what the labels "
            "mean — a calibration issue, not a noise issue._"
        )
        lines.append("")

    # ---- TrueSkill (pooled + per-model + normalized) ----
    if trueskill_bundle is not None:
        pooled = trueskill_bundle["pooled"]
        per_model = trueskill_bundle["per_model"]
        normalized = trueskill_bundle["normalized"]
        corr_df = trueskill_bundle["correlations"]
        models_ord = trueskill_bundle["models"]

        # --- Pooled ---
        newpage()
        lines.append("# TrueSkill — pooled (non-reweighted)")
        lines.append("")
        lines.append(
            f"One TrueSkill fit over the concatenation of all ({len(models_ord):,} × 5,000) "
            "model × pair comparisons. Every row counts equally. Because phi-4 and gemma-4-e2b "
            "emit `Same` most of the time, a large fraction of updates are draws — which "
            "compresses the μ dynamic range but lowers σ via sheer sample size."
        )
        lines.append("")
        pooled_summary = (
            pooled[["mu", "sigma", "ts_conservative", "n_comparisons"]]
            .describe().round(3).reset_index().rename(columns={
                "index": "Statistic",
                "mu": "μ",
                "sigma": "σ",
                "ts_conservative": "μ − 3σ",
                "n_comparisons": "# comparisons",
            })
        )
        lines.append(_md_table(pooled_summary))
        lines.append("")
        lines.extend(_centered_image(
            rel(images_dir / "pooled_ts_ranking.png"),
            "Pooled TrueSkill ranking",
            "width=86%",
        ))
        lines.extend(_centered_image(
            rel(images_dir / "pooled_ts_distributions.png"),
            "Pooled TrueSkill μ / σ distributions",
            "width=86%",
        ))

        unit_header = unit_label[:1].upper() + unit_label[1:]
        pooled_rename = {
            "rank": "Rank",
            "unit_name": unit_header,
            "ts_point_estimate": "μ",
            "ts_conservative": "μ − 3σ",
            "sigma": "σ",
            "n_comparisons": "# comparisons",
        }

        def _fmt_pooled(sub: pd.DataFrame) -> pd.DataFrame:
            keep = ["unit_name", "ts_point_estimate", "ts_conservative", "sigma", "n_comparisons"]
            keep = [c for c in keep if c in sub.columns]
            out = sub[keep].reset_index(drop=True).round(3)
            out.insert(0, "rank", np.arange(1, len(out) + 1))
            return out.rename(columns=pooled_rename)

        # Sort pooled by ts_conservative desc (same as single-run convention).
        pooled_sorted = pooled.sort_values("ts_conservative", ascending=False).reset_index(drop=True)
        pooled_filt = pooled_sorted[pooled_sorted["n_comparisons"] >= min_comparisons]
        top_pool = pooled_filt.head(top_n)
        bot_pool = pooled_filt.tail(top_n).iloc[::-1]

        newpage()
        lines.append(f"## Top {len(top_pool)} most {attribute}  ·  pooled")
        lines.append("")
        lines.append(_md_table(_fmt_pooled(top_pool)))
        lines.append("")
        newpage()
        lines.append(f"## Bottom {len(bot_pool)} least {attribute}  ·  pooled")
        lines.append("")
        lines.append(_md_table(_fmt_pooled(bot_pool)))
        lines.append("")

        # --- Per-model μ cross-correlation ---
        newpage()
        lines.append("# TrueSkill — per-model μ correlation")
        lines.append("")
        lines.append(
            f"Per-model TrueSkill was fit separately for each of the {len(models_ord)} models, "
            f"yielding an independent μ per {unit_label}. Scatter plots below put each library "
            "at one point per (model-A, model-B) pair; the identity line is a visual anchor for "
            "'would rate the same'. Near-zero correlations mean models disagree about which "
            f"{unit_label_plural} rank higher, not just how confident they are."
        )
        lines.append("")
        corr_disp = corr_df.copy()
        for col in ["Pearson r", "Spearman ρ", "Kendall τ"]:
            corr_disp[col] = corr_disp[col].map(lambda v: f"{v:+.3f}")
        lines.append(_md_table(corr_disp))
        lines.append("")
        lines.extend(_centered_image(
            rel(images_dir / "per_model_mu_scatter.png"),
            "Per-model TrueSkill μ — pairwise scatter",
            "width=92%",
        ))

        # --- Normalized ---
        newpage()
        lines.append("# TrueSkill — normalized (per-model z-score, averaged)")
        lines.append("")
        lines.append(
            "For each model, μ values are z-scored across libraries "
            r"(subtract the model's mean μ, divide by the model's std of μ). "
            "The z-scores are then averaged across models per library. "
            "**Error bars = std of z-scores across models** — they are the cross-model "
            f"*disagreement* signal, not a within-model posterior. "
            f"This variant puts every model on equal footing; "
            "high-spread libraries are the ones to re-inspect visually."
        )
        lines.append("")
        norm_summary = (
            normalized[["z_mean", "z_std"]]
            .describe().round(3).reset_index().rename(columns={
                "index": "Statistic",
                "z_mean": "ẑ (mean)",
                "z_std": "σ(z) across models",
            })
        )
        lines.append(_md_table(norm_summary))
        lines.append("")
        lines.extend(_centered_image(
            rel(images_dir / "normalized_ts_ranking.png"),
            "Normalized TrueSkill ranking",
            "width=92%",
        ))

        # Normalized top/bottom tables.
        def _fmt_norm(sub: pd.DataFrame) -> pd.DataFrame:
            keep = ["unit_name", "z_mean", "z_std", "n_comparisons_min"]
            out = sub[keep].reset_index(drop=True).round(3)
            out.insert(0, "rank", np.arange(1, len(out) + 1))
            return out.rename(columns={
                "rank": "Rank",
                "unit_name": unit_header,
                "z_mean": "ẑ",
                "z_std": "σ(z)",
                "n_comparisons_min": "# comps (min)",
            })

        norm_filt = normalized[normalized["n_comparisons_min"] >= min_comparisons]
        top_norm = norm_filt.head(top_n)
        bot_norm = norm_filt.tail(top_n).iloc[::-1]

        newpage()
        lines.append(f"## Top {len(top_norm)} most {attribute}  ·  normalized")
        lines.append("")
        lines.append(_md_table(_fmt_norm(top_norm)))
        lines.append("")
        newpage()
        lines.append(f"## Bottom {len(bot_norm)} least {attribute}  ·  normalized")
        lines.append("")
        lines.append(_md_table(_fmt_norm(bot_norm)))
        lines.append("")

        # Most-disputed: largest σ(z). Include per-model μ for diagnostic.
        disputed = normalized.nlargest(most_disputed_k, "z_std").reset_index(drop=True)
        model_mu_cols = [f"mu_{m}" for m in models_ord]
        disp_df = disputed[["unit_name", "z_mean", "z_std"] + model_mu_cols].copy()
        disp_df.insert(0, "rank", np.arange(1, len(disp_df) + 1))
        rename_disp = {
            "rank": "Rank",
            "unit_name": unit_header,
            "z_mean": "ẑ",
            "z_std": "σ(z)",
        }
        for m in models_ord:
            rename_disp[f"mu_{m}"] = f"μ · {m}"
        for col in ["z_mean", "z_std"] + model_mu_cols:
            disp_df[col] = disp_df[col].map(lambda v: f"{v:.2f}")
        disp_df = disp_df.rename(columns=rename_disp)

        newpage()
        lines.append(f"## Top {len(disp_df)} most-disputed  ·  sorted by σ(z) descending")
        lines.append("")
        lines.append(
            f"{unit_label_plural.capitalize()} with the largest cross-model disagreement. "
            "Per-model μ columns help pinpoint which model is the outlier."
        )
        lines.append("")
        lines.append(_md_table(disp_df))
        lines.append("")

    # ---- Example pair images: top/bottom by ẑ ----
    if top_bottom_image_pairs:
        top_imgs = [e for e in top_bottom_image_pairs if e["section"] == "top"]
        bot_imgs = [e for e in top_bottom_image_pairs if e["section"] == "bottom"]

        def _render_per_model_labels(per_model: dict[str, str]) -> str:
            return "  ·  ".join(
                f"**{m}:** `{per_model.get(m, '?')}`" for m in (trueskill_bundle or {}).get("models", per_model.keys())
            )

        def _emit_topbottom(bucket: list[dict], header: str, highlight: str) -> None:
            if not bucket:
                return
            for idx, e in enumerate(bucket):
                newpage()
                row = e["pair_row"]
                side = _pair_side(row, e["unit_uid"])
                a_name = row.get("unit_name_a", "") or ""
                b_name = row.get("unit_name_b", "") or ""
                rel_path = Path(images_dir.name) / "pairs" / e["image_path"].name
                if idx == 0:
                    lines.append(f"# {header}")
                    lines.append("")
                    lines.append(
                        f"One randomly-drawn pair per {unit_label} from the **{highlight}** of the "
                        "normalized ẑ ranking. Caption shows each model's canonical A-vs-B label."
                    )
                    lines.append("")
                lines.append(f"## #{e['rank']}  ·  {e['unit_name']}")
                lines.append("")
                lines.append(
                    f"ẑ = **{e['z_mean']:+.2f}**  ·  σ(z) = **{e['z_std']:.2f}**  ·  "
                    f"highlighted side = **{side}**"
                )
                lines.append("")
                lines.append(
                    f"**A:** {a_name} &nbsp;·&nbsp; **B:** {b_name}  \n"
                    f"**pair_id:** `{e['pair_id']}`"
                )
                lines.append("")
                lines.append(_render_per_model_labels(e["per_model_labels"]))
                lines.append("")
                lines.extend(
                    _centered_image(rel_path, f"{e['unit_name']} example pair", "width=72%")
                )

        _emit_topbottom(
            top_imgs,
            f"Example pairs — top {len(top_imgs)} most {attribute}  ·  by normalized ẑ",
            "top",
        )
        _emit_topbottom(
            bot_imgs,
            f"Example pairs — bottom {len(bot_imgs)} least {attribute}  ·  by normalized ẑ",
            "bottom",
        )

    # ---- Most-disputed pairs (top-K by per-pair label variance) ----
    if disputed_image_pairs:
        models_ord = (trueskill_bundle or {}).get("models", list(disputed_image_pairs[0]["per_model_labels"].keys()))
        for idx, e in enumerate(disputed_image_pairs):
            newpage()
            row = e["pair_row"]
            a_name = row.get("unit_name_a", "") or ""
            b_name = row.get("unit_name_b", "") or ""
            rel_path = Path(images_dir.name) / "pairs" / e["image_path"].name
            if idx == 0:
                lines.append(
                    f"# Most-disputed pairs  ·  top {len(disputed_image_pairs)} by per-pair label variance"
                )
                lines.append("")
                lines.append(
                    "Pairs where the models' 5-point ordinal labels span the widest range. "
                    "These are the qualitative drivers of the Krippendorff α and κ numbers: "
                    "go look at the image and decide which model's label you'd actually endorse."
                )
                lines.append("")
            lines.append(
                f"## #{e['rank']}  ·  pair `{e['pair_id']}`"
            )
            lines.append("")
            lines.append(
                f"label variance = **{e['label_variance']:.2f}**  ·  "
                f"label range across models = **{e['label_range']}**"
            )
            lines.append("")
            lines.append(
                f"**A:** {a_name} &nbsp;·&nbsp; **B:** {b_name}"
            )
            lines.append("")
            lines.append("  ·  ".join(
                f"**{m}:** `{e['per_model_labels'].get(m, '?')}`" for m in models_ord
            ))
            lines.append("")
            lines.extend(
                _centered_image(rel_path, f"Disputed pair {e['pair_id']}", "width=72%")
            )

    # ---- Reasoning word cloud (traced rows only) ----
    newpage()
    lines.append("# Reasoning word cloud")
    lines.append("")
    if wordcloud_tokens is None or wordcloud_tokens == 0 or reasoning_rows.empty:
        contributing = reasoning_rows.groupby("model_label").size().to_dict() if not reasoning_rows.empty else {}
        if contributing:
            detail = ", ".join(f"{m} (n={n:,})" for m, n in contributing.items())
            lines.append(
                f"Reasoning captured from: {detail}, but no tokens remain after stopword filtering."
            )
        else:
            lines.append(
                "_No reasoning traces captured by any model (likely no thinking mode was enabled)._"
            )
        lines.append("")
    else:
        contributing = reasoning_rows.groupby("model_label").size().to_dict()
        contrib_str = ", ".join(f"**{m}** (n={n:,})" for m, n in contributing.items())
        lines.append(
            f"Aggregated from **{len(reasoning_rows):,}** rows with captured `model_reasoning`, "
            f"contributed by {len(contributing)} of {n_models} models: {contrib_str}. "
            "Models run without a thinking mode contribute nothing and are silently omitted here."
        )
        lines.append("")
        lines.extend(_centered_image(rel(images_dir / "wordcloud.png"), "Reasoning word cloud"))

    out_md.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "aggregation_dir",
        type=Path,
        help="Directory of symlinked run dirs (one per model).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Report title. Default derived from the aggregation directory name.",
    )
    p.add_argument(
        "--attribute",
        type=str,
        default="more of the attribute",
        help='Attribute phrase for rankings (e.g., "maintained").',
    )
    p.add_argument(
        "--unit-label",
        type=str,
        default="unit",
        help='Singular noun for the ranked entity (e.g., "library").',
    )
    p.add_argument(
        "--unit-label-plural",
        type=str,
        default=None,
        help='Plural noun (default: unit-label + "s").',
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated list of model labels to include.",
    )
    p.add_argument(
        "--allow-prompt-drift",
        action="store_true",
        help="Proceed even if system/user prompts differ across runs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output markdown path. Default: <agg_dir>/<agg_dir.name>.report.md.",
    )
    p.add_argument(
        "--extra-stopwords",
        type=str,
        default="",
        help="Comma-separated extra stopwords for the word cloud.",
    )
    p.add_argument(
        "--pdf",
        action="store_true",
        help="Also render the markdown to PDF via pandoc.",
    )
    p.add_argument(
        "--pdf-engine",
        type=str,
        default="xelatex",
        help="pandoc --pdf-engine value. Default: xelatex.",
    )
    p.add_argument(
        "--pdf-orientation",
        choices=("landscape", "portrait"),
        default="landscape",
        help="PDF page orientation. Default: landscape.",
    )
    p.add_argument(
        "--discover-only",
        action="store_true",
        help="Stop after Phase 1 (discovery + sanity checks); do not write a report.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Rows in the top/bottom tables for both TrueSkill variants. Default: 20.",
    )
    p.add_argument(
        "--draw-prob",
        type=float,
        default=0.05,
        help="TrueSkill draw probability. Default: 0.05 (matches single-run default).",
    )
    p.add_argument(
        "--min-comparisons",
        type=int,
        default=1,
        help="Minimum comparisons (per model, for normalized) for a unit to appear in tables.",
    )
    p.add_argument(
        "--most-disputed-k",
        type=int,
        default=10,
        help="Rows in the 'most-disputed' table (top-K by cross-model σ). Default: 10.",
    )
    p.add_argument(
        "--top-bottom-image-n",
        type=int,
        default=5,
        help="Number of top and bottom units (by normalized ẑ) to embed pair images for. Default: 5.",
    )
    p.add_argument(
        "--disagreement-pairs-n",
        type=int,
        default=5,
        help="Number of top-label-variance pairs to embed as disagreement exemplars. Default: 5.",
    )
    p.add_argument(
        "--image-scale",
        type=float,
        default=0.5,
        help="Uniform scale for stitched pair images (source 1024x1024). Default: 0.5.",
    )
    p.add_argument(
        "--image-quality",
        type=int,
        default=85,
        help="JPEG quality for stitched pair images (1-95). Default: 85.",
    )
    p.add_argument(
        "--image-seed",
        type=int,
        default=1234,
        help="RNG seed for random pair selection within top/bottom units. Default: 1234.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    agg_dir = args.aggregation_dir.resolve()
    if not agg_dir.is_dir():
        raise SystemExit(f"Aggregation directory not found: {agg_dir}")

    title = args.title or f"Pairwise VQA aggregation: {agg_dir.name}"
    unit_label_plural = args.unit_label_plural or f"{args.unit_label}s"

    runs, skipped = _discover_runs(agg_dir)

    if args.models:
        allowed = {m.strip() for m in args.models.split(",") if m.strip()}
        excluded = [r.model_label for r in runs if r.model_label not in allowed]
        runs = [r for r in runs if r.model_label in allowed]
        for label in excluded:
            skipped.append((label, "excluded via --models"))

    # ---- header ----
    print(f"Aggregation directory: {agg_dir}")
    print(f"Title:                 {title}")
    print(f"Attribute:             {args.attribute}  ({args.unit_label} / {unit_label_plural})")

    if skipped:
        print("\n[info] skipped runs:")
        for name, why in skipped:
            print(f"  - {name}: {why}")

    if not runs:
        raise SystemExit("No runs discovered.")
    if len(runs) < 2:
        raise SystemExit(
            f"Need >= 2 runs to aggregate; discovered {len(runs)} "
            f"({runs[0].model_label}). Use the single-run script instead."
        )

    print(f"\nDiscovered {len(runs)} model runs:")
    for r in runs:
        print(f"  - {r.model_label:28s}  →  {r.model_name}")
        print(f"      stage output: {r.out_parquet}")
        print(f"      pairs:        {r.pairs_parquet}")

    # ---- prompt equality ----
    print("\n=== Prompt equality ===")
    sys_eq, usr_eq, sch_eq = _check_prompt_equality(runs)
    print(f"  system prompts identical:            {'YES' if sys_eq else 'NO'}")
    print(f"  user templates identical:            {'YES' if usr_eq else 'NO'}")
    print(f"  structured-output schemas identical: {'YES' if sch_eq else 'NO'}")
    if not (sys_eq and usr_eq):
        if not args.allow_prompt_drift:
            raise SystemExit(
                "Prompts differ across runs. Re-run with --allow-prompt-drift to proceed "
                "(aggregation will still operate on the intersection of pair_ids)."
            )
        print("  [warn] --allow-prompt-drift set; proceeding with mismatched prompts.")

    # ---- pair identity ----
    print("\n=== Pair identity ===")
    intersection, counts = _check_pair_identity(runs)
    uniform = all(c == len(intersection) for c in counts.values())
    print(f"  all pair_id sets identical: {'YES' if uniform else 'NO'}")
    print(f"  intersection size:          {len(intersection):,}")
    for label, c in counts.items():
        delta = c - len(intersection)
        suffix = "" if delta == 0 else f"  (drops {delta:,})"
        print(f"    {label:28s}  n={c:,}{suffix}")

    # ---- long-form dataframe ----
    print("\n=== Long-form dataframe ===")
    long_df = _build_long_df(runs, pair_id_filter=None if uniform else intersection)
    print(f"  shape: {long_df.shape[0]:,} rows × {long_df.shape[1]} cols")
    print(f"  per-model row counts:")
    for label, c in long_df["model_label"].value_counts().items():
        print(f"    {label:28s}  {c:,}")
    has_units = "unit_uid_a" in long_df.columns and "unit_uid_b" in long_df.columns
    if has_units:
        unique_units = pd.concat([long_df["unit_uid_a"], long_df["unit_uid_b"]]).nunique()
        print(f"  distinct {unit_label_plural}: {unique_units:,}")
    else:
        print("  [warn] pairs.parquet missing unit_uid columns — library-level aggregation unavailable.")

    # ---- per-model quick stats ----
    print(f"\n=== Per-model quick stats (canonical A-vs-B labels) ===")
    for r in runs:
        sub = long_df[long_df["model_label"] == r.model_label]
        dist = _label_distribution(sub)
        entropy = _label_entropy(dist)
        lens = _reasoning_lengths(sub)
        captured = int((~lens["is_empty"]).sum())
        cap_rate = captured / len(sub) if len(sub) else 0.0
        label_pcts = (sub["relative_label"].value_counts(normalize=True) * 100.0).to_dict()
        label_str = "  ".join(
            f"{lbl}={label_pcts.get(lbl, 0.0):5.1f}%" for lbl in ORDINAL_ORDER
        )
        print(f"  {r.model_label}")
        print(f"    entropy={entropy:.3f}  capture={cap_rate:.1%}  "
              f"mean_relative_score={sub['relative_score'].mean():+.3f}")
        print(f"    {label_str}")

    if args.discover_only:
        print("\nPhase 1 complete (--discover-only). Stopping.")
        return

    # -----------------------------------------------------------------------
    # Phase 2: overview + prompt + per-model label distribution + wordcloud
    # -----------------------------------------------------------------------

    out_md = args.out or (agg_dir / f"{agg_dir.name}.report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    images_dir = out_md.parent / f"{out_md.stem}_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Per-model grouped label distribution (also returns the proportion df
    # so the overview table can reuse the same numbers).
    prop_df = _plot_per_model_label_distribution(
        long_df, runs, images_dir / "per_model_label_distribution.png"
    )

    # ---- Phase 3: inter-model agreement ----
    model_order = [r.model_label for r in runs]
    wide_df = _build_wide_df(long_df)
    agreement = _compute_agreement(wide_df, model_order)

    _plot_agreement_heatmap(
        agreement["kappa"],
        model_order,
        images_dir / "kappa_heatmap.png",
        title="Cohen's κ  ·  quadratic-weighted",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=1.0,
        fmt="{:.2f}",
    )
    _plot_agreement_heatmap(
        agreement["sign_agree"],
        model_order,
        images_dir / "sign_agreement_heatmap.png",
        title="Sign-only agreement  ·  fraction of pairs",
        cmap="Blues",
        vmin=0.4,
        vmax=1.0,
        pct=True,
    )
    _plot_confusion_matrices(
        agreement["confmats"],
        images_dir / "confusion_matrices.png",
    )

    # Stdout summary so the agreement numbers appear in the log.
    print("\n=== Inter-model agreement ===")
    print(f"  Krippendorff's α (ordinal):              {agreement['krippendorff_alpha']:+.4f}")
    print(f"  Mean pairwise κ (quadratic):             {agreement['mean_kappa']:+.4f}")
    print(f"  Mean pairwise exact-label agreement:     {agreement['mean_exact']:.1%}")
    print(f"  Mean pairwise sign agreement:            {agreement['mean_sign']:.1%}")
    print(f"  All-models exact-label agreement:        {agreement['all_exact_rate']:.1%}")
    print(f"  All-models sign agreement:               {agreement['all_sign_rate']:.1%}")

    # ---- Same-model self-consistency over deliberate repeats ----
    self_consistency = _self_consistency_stats(long_df, model_order)
    if not self_consistency.empty:
        print("\n=== Self-consistency (same-model repeats) ===")
        for row in self_consistency.itertuples(index=False):
            if row.repeat_groups == 0:
                print(f"  {row.model_label:28s}  no repeat groups")
                continue
            print(
                f"  {row.model_label:28s}  "
                f"groups={row.repeat_groups:>5,}  "
                f"exact={row.exact_label_agreement_rate:.1%}  "
                f"sign={row.sign_agreement_rate:.1%}  "
                f"H̄={row.mean_within_group_entropy_nats:.3f}  "
                f"self-κ={row.self_kappa_quadratic:+.3f}"
            )

    # ---- Per-model runtime / inference throughput ----
    runtime_stats = _collect_runtime_stats(runs)
    if not runtime_stats.empty:
        print("\n=== Inference runtime / throughput ===")
        for row in runtime_stats.itertuples(index=False):
            wall = (
                f"{row.duration_s:8.1f}s"
                if row.duration_s == row.duration_s  # not NaN
                else "      –   "
            )
            tput = (
                f"{row.throughput_rows_per_s:5.2f} rows/s"
                if row.throughput_rows_per_s == row.throughput_rows_per_s
                else "    – rows/s"
            )
            tput_gpu = (
                f"{row.throughput_per_gpu_rows_per_s:5.2f} rows/s/gpu"
                if row.throughput_per_gpu_rows_per_s == row.throughput_per_gpu_rows_per_s
                else "    – rows/s/gpu"
            )
            print(
                f"  {row.model_label:28s}  rows={row.rows:>7,}  wall={wall}  "
                f"DP={row.dp_world_size}  TP={row.tp_size}  {tput}  {tput_gpu}"
            )
        _plot_throughput(runtime_stats, images_dir / "throughput.png")

    # ---- Phase 4: TrueSkill (pooled + per-model + normalized) ----
    print("\n=== TrueSkill ===")
    pooled = _compute_pooled_trueskill(long_df, draw_prob=args.draw_prob)
    per_model_ts = _compute_per_model_trueskill(long_df, model_order, draw_prob=args.draw_prob)
    normalized = _compute_normalized_trueskill(per_model_ts, min_comparisons=args.min_comparisons)
    print(f"  pooled ratings:            {len(pooled):,} {unit_label_plural}")
    print(f"  per-model fits:            {len(per_model_ts)} fits, "
          f"median #comparisons/unit = {int(np.median([df['n_comparisons'].median() for df in per_model_ts.values()]))}")
    print(f"  normalized (z-mean) range: {normalized['z_mean'].min():+.3f} .. {normalized['z_mean'].max():+.3f}")
    print(f"  σ(z) across models range:  {normalized['z_std'].min():.3f} .. {normalized['z_std'].max():.3f}")

    # Pooled ranking plot + distributions (reused from single-run).
    _plot_trueskill_ranking(
        pooled,
        images_dir / "pooled_ts_ranking.png",
        args.attribute,
        unit_label_plural,
    )
    _plot_trueskill_distribution(pooled, images_dir / "pooled_ts_distributions.png")

    # Per-model μ pairwise scatter (returns the correlation table).
    corr_df = _plot_model_mu_scatter(
        per_model_ts, model_order, images_dir / "per_model_mu_scatter.png"
    )

    # Normalized ranking chart.
    _plot_normalized_ranking(
        normalized,
        images_dir / "normalized_ts_ranking.png",
        args.attribute,
        unit_label_plural,
    )

    trueskill_bundle = {
        "models": model_order,
        "pooled": pooled,
        "per_model": per_model_ts,
        "normalized": normalized,
        "correlations": corr_df,
    }

    # ---- Phase 5: example pair images ----
    top_bottom_selections = _select_top_bottom_example_pairs(
        long_df, wide_df, normalized, model_order,
        top_bottom_n=args.top_bottom_image_n,
        seed=args.image_seed,
    )
    disputed_selections = _select_high_variance_pairs(
        long_df, wide_df, model_order, k=args.disagreement_pairs_n,
    )
    top_bottom_rendered, disputed_rendered = _render_pair_images(
        top_bottom_selections, disputed_selections,
        images_dir,
        scale=args.image_scale,
        quality=args.image_quality,
    )
    print(f"\n=== Example pairs ===")
    print(f"  top/bottom images rendered: {len(top_bottom_rendered)}")
    print(f"  disputed images rendered:   {len(disputed_rendered)}")

    # Word cloud — restricted to rows that actually have a captured reasoning trace.
    extra_sw = {s.strip().lower() for s in args.extra_stopwords.split(",") if s.strip()}
    reasoning_rows = long_df[long_df["model_reasoning"].fillna("").astype(str).str.len() > 0].copy()
    if reasoning_rows.empty:
        wordcloud_tokens = 0
    else:
        wordcloud_tokens = _plot_wordcloud(
            reasoning_rows["model_reasoning"],
            images_dir / "wordcloud.png",
            extra_sw,
        )

    _write_phase2_markdown(
        long_df=long_df,
        runs=runs,
        prop_df=prop_df,
        agg_dir=agg_dir,
        title=title,
        attribute=args.attribute,
        unit_label=args.unit_label,
        unit_label_plural=unit_label_plural,
        images_dir=images_dir,
        out_md=out_md,
        reasoning_rows=reasoning_rows,
        wordcloud_tokens=wordcloud_tokens,
        agreement=agreement,
        trueskill_bundle=trueskill_bundle,
        top_n=args.top_n,
        min_comparisons=args.min_comparisons,
        most_disputed_k=args.most_disputed_k,
        top_bottom_image_pairs=top_bottom_rendered,
        disputed_image_pairs=disputed_rendered,
        self_consistency=self_consistency,
        runtime_stats=runtime_stats,
    )

    print(f"\nWrote report:      {out_md}")
    print(f"Images directory:  {images_dir}")

    if args.pdf:
        pdf_path = _export_pdf(
            out_md,
            args.pdf_engine,
            landscape=(args.pdf_orientation == "landscape"),
        )
        if pdf_path is not None:
            print(f"Wrote PDF:         {pdf_path}")


if __name__ == "__main__":
    main()
