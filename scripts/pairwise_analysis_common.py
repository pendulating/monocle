#!/usr/bin/env python3
"""Shared infrastructure for the urbanpairvqa analysis tools.

Used by ``pairwise_vqa_difference_report.py`` (group difference testing) and
``pairwise_vqa_regression_report.py`` (covariate regressions). Provides:

  * Run loading: stage output merged with ``pairs.parquet`` unit identity
  * Unit-metadata attachment (surfaced pair columns or external join)
  * The append-only JSONL experiment registry (deterministic ids, dedupe,
    ``--list``)
  * The W&B mirror (separate analysis project, via the sanctioned
    ``WandbLogger``; failures non-fatal)
  * Small formatting helpers shared by the markdown writers

Registry and W&B conventions are documented in the wiki page
``guide-pairwise-difference-testing``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = _REPO_ROOT / "machine-beholder" / "difference_tests" / "registry.jsonl"
DEFAULT_WANDB_PROJECT = "URBANPAIRVQA-ANALYSIS"

# Metadata values treated as missing after string coercion.
_MISSING_GROUP_VALUES = {"", "nan", "none", "null", "<na>"}


# ---------------------------------------------------------------------------
# Run loading + unit-metadata attachment
# ---------------------------------------------------------------------------


def load_run(
    output_pq: Path,
    pairs_pq: Optional[Path],
    column: Optional[str],
) -> tuple[pd.DataFrame, bool]:
    """Load the stage output merged with unit identity (and, when present,
    the surfaced ``<column>_a/_b`` metadata) from ``pairs.parquet``.

    Returns ``(df, column_from_pairs)``.
    """
    out = pd.read_parquet(output_pq)
    required = {"pair_id", "relative_score"}
    missing = required - set(out.columns)
    if missing:
        raise SystemExit(f"Output parquet {output_pq} missing columns: {sorted(missing)}")

    pairs_path = pairs_pq or (output_pq.parent / "pairs.parquet")
    if not pairs_path.exists():
        raise SystemExit(
            f"pairs.parquet not found at {pairs_path} — unit identity is required."
        )
    pairs = pd.read_parquet(pairs_path)
    if not {"unit_uid_a", "unit_uid_b"}.issubset(pairs.columns):
        raise SystemExit(
            f"{pairs_path} has no unit_uid_a/unit_uid_b — was this a unit-mode run?"
        )

    merge_cols = ["pair_id", "unit_uid_a", "unit_uid_b"]
    for c in ("unit_name_a", "unit_name_b", "canonical_pair_id"):
        if c in pairs.columns and c not in out.columns:
            merge_cols.append(c)
    column_from_pairs = False
    if column:
        ca, cb = f"{column}_a", f"{column}_b"
        if ca in pairs.columns and cb in pairs.columns:
            merge_cols += [ca, cb]
            column_from_pairs = True
    merged = out.merge(pairs[merge_cols], on="pair_id", how="left", validate="one_to_one")
    merged["relative_score"] = pd.to_numeric(merged["relative_score"], errors="coerce")
    merged = merged.dropna(subset=["relative_score", "unit_uid_a", "unit_uid_b"])
    if "canonical_pair_id" not in merged.columns:
        merged["canonical_pair_id"] = merged["pair_id"]
    return merged, column_from_pairs


def _clean_group_series(s: pd.Series) -> pd.Series:
    """String-coerce and null out missing-ish values."""
    out = s.astype(str).str.strip()
    return out.mask(out.str.casefold().isin(_MISSING_GROUP_VALUES))


def load_unit_metadata(
    metadata_parquet: Path,
    id_column: str,
    columns: List[str],
) -> pd.DataFrame:
    """Read an external unit-metadata parquet, deduped and indexed by the
    string-coerced unit id. Returns a frame with the requested columns."""
    if not metadata_parquet.exists():
        raise SystemExit(f"--unit-metadata-parquet not found: {metadata_parquet}")
    meta = pd.read_parquet(metadata_parquet, columns=[id_column] + columns)
    meta = meta.dropna(subset=[id_column]).copy()
    meta["__uid__"] = meta[id_column].astype(str).str.strip()
    meta = meta.drop_duplicates("__uid__").set_index("__uid__")
    return meta[columns]


def attach_groups(
    df: pd.DataFrame,
    *,
    group_column: str,
    group_from_pairs: bool,
    unit_metadata_parquet: Optional[Path],
    unit_metadata_id_column: str,
) -> pd.DataFrame:
    """Add ``__group_a__`` / ``__group_b__`` columns to the merged frame.

    Prefers metadata surfaced on pairs.parquet; falls back to an external
    join on unit_uid.
    """
    df = df.copy()
    if group_from_pairs:
        df["__group_a__"] = _clean_group_series(df[f"{group_column}_a"])
        df["__group_b__"] = _clean_group_series(df[f"{group_column}_b"])
        return df

    if unit_metadata_parquet is None:
        raise SystemExit(
            f"Column {group_column!r} is not surfaced on pairs.parquet "
            f"(no '{group_column}_a'). Supply --unit-metadata-parquet to join it "
            "from an external unit-metadata file."
        )
    meta = load_unit_metadata(unit_metadata_parquet, unit_metadata_id_column, [group_column])
    lut = _clean_group_series(meta[group_column])

    df["__group_a__"] = df["unit_uid_a"].astype(str).str.strip().map(lut)
    df["__group_b__"] = df["unit_uid_b"].astype(str).str.strip().map(lut)

    n_unmatched = int(
        pd.concat([df.loc[df["__group_a__"].isna(), "unit_uid_a"],
                   df.loc[df["__group_b__"].isna(), "unit_uid_b"]]).nunique()
    )
    if n_unmatched:
        print(f"[WARN] {n_unmatched:,} units did not match the unit-metadata join.")
    return df


def build_unit_value_map(
    df: pd.DataFrame,
    column: str,
    *,
    column_from_pairs: bool,
    unit_metadata_parquet: Optional[Path],
    unit_metadata_id_column: str,
    numeric: bool,
) -> pd.Series:
    """unit_uid → value for one metadata column (numeric coercion optional).

    Resolution order matches :func:`attach_groups`: surfaced ``<col>_a/_b``
    pair columns first, else external join.
    """
    if column_from_pairs:
        a = df[["unit_uid_a", f"{column}_a"]].rename(
            columns={"unit_uid_a": "unit_uid", f"{column}_a": "value"})
        b = df[["unit_uid_b", f"{column}_b"]].rename(
            columns={"unit_uid_b": "unit_uid", f"{column}_b": "value"})
        long = pd.concat([a, b], ignore_index=True)
        long["unit_uid"] = long["unit_uid"].astype(str)
        long = long.drop_duplicates("unit_uid").set_index("unit_uid")["value"]
    else:
        if unit_metadata_parquet is None:
            raise SystemExit(
                f"Column {column!r} is not surfaced on pairs.parquet; supply "
                "--unit-metadata-parquet."
            )
        meta = load_unit_metadata(unit_metadata_parquet, unit_metadata_id_column, [column])
        long = meta[column]
    if numeric:
        return pd.to_numeric(long, errors="coerce").dropna()
    return _clean_group_series(long).dropna()


# ---------------------------------------------------------------------------
# Multiple-comparison adjustment
# ---------------------------------------------------------------------------


def adjust_pvalues(results: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """BH-adjust each p-value column independently (NaNs pass through)."""
    from statsmodels.stats.multitest import multipletests

    results = results.copy()
    for col in cols:
        adj = np.full(len(results), np.nan)
        p = results[col].to_numpy(dtype=float)
        valid = np.isfinite(p)
        if valid.sum():
            adj[valid] = multipletests(p[valid], method="fdr_bh")[1]
        results[f"{col}_adj"] = adj
    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def registry_path(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value
    env = os.environ.get("MLLMSCI_DIFFTEST_REGISTRY")
    return Path(env) if env else DEFAULT_REGISTRY


def compute_experiment_id(inputs: Dict[str, Any]) -> str:
    """Deterministic id over the experiment-defining inputs."""
    canon = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


def read_registry(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    n_bad = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        print(f"[WARN] registry {path}: skipped {n_bad} unparseable line(s).")
    return records


def find_in_registry(records: List[Dict[str, Any]], experiment_id: str) -> Optional[Dict[str, Any]]:
    for rec in reversed(records):
        if rec.get("experiment_id") == experiment_id:
            return rec
    return None


def append_registry(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _registry_comparison_label(r: Dict[str, Any]) -> str:
    mode = r.get("mode", "")
    if mode == "pair":
        return f"{r.get('group_a')} vs {r.get('group_b')}"
    if mode == "matrix":
        return f"{len(r.get('groups') or [])} groups"
    if mode in ("regression", "screen"):
        y = r.get("y", "mu")
        if r.get("x"):
            label = f"{y} ~ {r['x']}"
        else:
            label = f"{y} ~ screen[{len(r.get('x_list') or [])}]"
        if r.get("controls"):
            label += f" | {','.join(r['controls'])}"
        return label
    return ""


def list_registry(path: Path) -> None:
    records = read_registry(path)
    if not records:
        print(f"Registry empty or missing: {path}")
        return
    rows = []
    for r in records:
        res = r.get("results") or {}
        rows.append({
            "experiment_id": r.get("experiment_id", ""),
            "created_at": str(r.get("created_at", ""))[:19],
            "mode": r.get("mode", ""),
            "models": r.get("n_models") or 1,
            "group_column": r.get("group_column") or r.get("x") or "",
            "comparison": _registry_comparison_label(r),
            "h2h_p": res.get("h2h_p"),
            "rating_p": res.get("rating_p"),
            "r2": res.get("r2"),
            "n_significant": res.get("n_significant"),
            "source": Path(str(r.get("source_parquet", ""))).name,
            "wandb": r.get("wandb_url") or "",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, max_colwidth=48))


# ---------------------------------------------------------------------------
# W&B mirror
# ---------------------------------------------------------------------------


def mirror_to_wandb(
    *,
    record: Dict[str, Any],
    results: pd.DataFrame,
    artifact_paths: List[Path],
    project: str,
    entity: Optional[str],
    run_label: str,
    stage: str = "difference_test",
    extra_tags: Optional[List[str]] = None,
) -> Optional[str]:
    """Mirror an experiment to W&B via the sanctioned WandbLogger. Returns the
    run URL, or None when W&B is unavailable/failed (never raises)."""
    try:
        from dagspaces.common.stage_utils import ensure_dotenv

        ensure_dotenv()
        from dagspaces.common.wandb_logger import WandbConfig, WandbLogger
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[wandb] dagspaces.common unavailable ({exc}); skipping W&B mirror.")
        return None

    resolved_entity = entity or os.environ.get("WANDB_ENTITY") or "urbanekg"
    tags = [stage, f"mode:{record.get('mode')}"]
    if record.get("group_column"):
        tags.append(f"group_column:{record['group_column']}")
    if record.get("model_label"):
        tags.append(f"model:{record['model_label']}")
    for m in record.get("models") or []:
        tags.append(f"model:{m}")
    tags.extend(extra_tags or [])

    wb_config = WandbConfig(
        enabled=True,
        project=project,
        entity=resolved_entity,
        tags=tags,
        table_sample_rows=5000,
        default_experiment_name=run_label,
        dagspace_name="urbanpairvqa",
    )
    run_config = {k: v for k, v in record.items() if k not in ("results", "wandb_url")}
    logger = WandbLogger.with_config(
        None,
        stage=stage,
        wb_config=wb_config,
        run_id=record.get("experiment_id"),
        run_config=run_config,
    )
    url: Optional[str] = None
    try:
        logger.start()
        if logger._run is None:
            return None
        url = getattr(logger._run, "url", None)

        flat = {}
        for k, v in (record.get("results") or {}).items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                flat[f"{stage}/{k}"] = v
        logger.log_metrics(flat)
        logger.log_table(results, f"{stage}s")
        for p in artifact_paths:
            if p is not None and Path(p).exists():
                # W&B caps artifact names at 128 chars; key on the experiment id.
                name = f"{stage}-{record.get('experiment_id')}-{Path(p).name}"[:128]
                logger.log_artifact(str(p), name=name, type="report")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[wandb] mirror failed (non-fatal): {exc}")
    finally:
        try:
            logger.finish()
        except Exception:
            pass
    return url


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _stars(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return ""
    if p < alpha / 50:  # e.g. 0.001 at alpha=0.05
        return "***"
    if p < alpha / 5:  # e.g. 0.01
        return "**"
    if p < alpha:
        return "*"
    return ""


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.strip())[:40].strip("-").lower()


def _centered_image(rel_path: Path, alt: str, width_attr: str = "width=78%") -> list[str]:
    return [
        "::: {.center data-latex=\"\"}",
        f"![{alt}]({rel_path}){{ {width_attr} }}",
        ":::",
        "",
    ]
