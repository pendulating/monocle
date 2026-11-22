"""Sample K images from a curated parquet to an inspection folder.

Works with any parquet that exposes a row-per-image schema with an
``image_path`` column (and ideally ``sample_id`` + ``dataset`` for
disambiguation). The canonical consumer today is
``curation/.../cyclomedia_near_permits.parquet``, but nothing here is
Cyclomedia-specific.

Output layout::

    <output_dir>/
      images/                            # one file per sampled row
        <dataset>__<sample_id>.jpg       # copy or symlink (see --symlink)
        ...
      manifest.parquet                   # full-row provenance for every export
      manifest.json                      # summary: k, seed, mode, counts, elapsed

The ``<dataset>__<sample_id>`` naming disambiguates cross-dataset duplicates
(recordings on a borough bbox edge that appear in two datasets). If the
source parquet doesn't carry a ``dataset`` column, the prefix is omitted.

By default **copy** is used — safe to tar up and move elsewhere. Pass
``--symlink`` for a fast, local-only materialization that doesn't duplicate
bytes. Absolute source paths are used for symlinks so the inspection folder
stays valid when its own path changes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

import polars as pl

__all__ = ["sample_images", "SampleResult", "DEFAULT_WORKERS"]

log = logging.getLogger(__name__)

DEFAULT_WORKERS = 8


@dataclass
class SampleResult:
    output_dir: str
    images_dir: str
    source_parquet: str
    mode: str
    k_requested: int
    n_sampled: int
    n_exported: int
    n_missing: int
    n_failed: int
    seed: int
    stratify_by: Optional[str]
    manifest_parquet: str
    manifest_json: str
    elapsed_s: float


def _resolve_k(df_height: int, k: int) -> int:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > df_height:
        log.warning(
            "sample: requested k=%d but frame has %d rows; sampling all rows",
            k, df_height,
        )
        return df_height
    return k


def _stratified_sample(
    df: pl.DataFrame, k: int, stratify_by: str, seed: int
) -> pl.DataFrame:
    """Sample K rows split evenly across distinct values of ``stratify_by``.

    Each group is allotted ``k // n_groups`` rows (plus 1 extra to the first
    ``k % n_groups`` groups to hit k exactly). Groups with fewer rows than
    their quota contribute all they have; the shortfall is redistributed to
    larger groups deterministically.
    """
    groups = df[stratify_by].unique().sort().to_list()
    n_groups = len(groups)
    if n_groups == 0:
        return df.head(0)

    base = k // n_groups
    extra = k % n_groups
    quotas = {g: base + (1 if i < extra else 0) for i, g in enumerate(groups)}

    # First pass: honor the quota, tracking shortfalls for redistribution.
    chunks: list[pl.DataFrame] = []
    shortfall = 0
    sizes: dict = {g: 0 for g in groups}
    per_group_available: dict = {}
    per_group: dict[object, pl.DataFrame] = {}
    for g in groups:
        sub = df.filter(pl.col(stratify_by) == g)
        per_group[g] = sub
        per_group_available[g] = sub.height

    # Redistribute shortfall to groups with room.
    final_quotas = dict(quotas)
    for g, q in quotas.items():
        take = min(q, per_group_available[g])
        final_quotas[g] = take
        shortfall += q - take
    if shortfall > 0:
        ordered = sorted(
            groups,
            key=lambda g: per_group_available[g] - final_quotas[g],
            reverse=True,
        )
        i = 0
        while shortfall > 0 and i < len(ordered) * 4:   # bounded loop
            g = ordered[i % len(ordered)]
            if per_group_available[g] > final_quotas[g]:
                final_quotas[g] += 1
                shortfall -= 1
            i += 1

    for g, take in final_quotas.items():
        if take > 0:
            chunks.append(per_group[g].sample(n=take, seed=seed, shuffle=True))
            sizes[g] = take

    log.info(
        "sample: stratified by %s — %s",
        stratify_by,
        ", ".join(f"{g}={n}" for g, n in sizes.items()),
    )
    return pl.concat(chunks, how="vertical_relaxed") if chunks else df.head(0)


def _plan_destination(
    row: dict, images_dir: str, has_dataset_col: bool,
) -> tuple[str, str]:
    """Return (src, dst) absolute paths for one export row."""
    src = os.path.abspath(row["image_path"])
    sid = row.get("sample_id", None)
    ext = os.path.splitext(src)[1] or ".jpg"
    if sid is None:
        # Fall back to image basename if sample_id missing
        name = os.path.basename(src)
    else:
        if has_dataset_col and row.get("dataset"):
            name = f"{row['dataset']}__{sid}{ext}"
        else:
            name = f"{sid}{ext}"
    dst = os.path.join(images_dir, name)
    return src, dst


def _do_one(src: str, dst: str, mode: str) -> str:
    """Copy or symlink ``src`` to ``dst``. Returns "ok" | "missing" | f"fail:{msg}"."""
    if not os.path.isfile(src):
        return "missing"
    try:
        if os.path.lexists(dst):
            os.remove(dst)
        if mode == "symlink":
            os.symlink(src, dst)
        elif mode == "copy":
            # copy2 preserves mtime + permissions, which is helpful for audit.
            shutil.copy2(src, dst)
        else:
            return f"fail:unknown mode {mode}"
    except OSError as exc:
        return f"fail:{exc}"
    return "ok"


def sample_images(
    curated_parquet: str,
    output_dir: str,
    k: int,
    *,
    mode: str = "copy",
    seed: int = 0,
    stratify_by: Optional[str] = None,
    image_path_col: str = "image_path",
    sample_id_col: str = "sample_id",
    dataset_col: Optional[str] = "dataset",
    columns_to_keep: Optional[Iterable[str]] = None,
    force: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> SampleResult:
    """Sample K rows from ``curated_parquet`` and materialize the images.

    Args:
        curated_parquet: Path to a parquet with at minimum an ``image_path``
            column. ``sample_id`` and ``dataset`` are used for filename
            disambiguation when present.
        output_dir: Inspection dir. Created if missing. Must be empty unless
            ``force=True`` — the tool refuses to trample an existing folder.
        k: Number of rows to sample. Capped at ``df.height`` with a warning.
        mode: ``"copy"`` (default, safe to relocate) or ``"symlink"`` (fast,
            local-only; uses absolute source paths).
        seed: RNG seed for ``pl.DataFrame.sample``.
        stratify_by: Optional column name. If set, K is split evenly across
            distinct values. Typical: ``"dataset"``, ``"face"``, ``"source"``.
        image_path_col / sample_id_col / dataset_col: Column-name overrides
            in case the parquet uses different names.
        columns_to_keep: If set, only these columns are kept in the written
            ``manifest.parquet``. Default: all columns.
        force: Allow exporting into a non-empty ``output_dir``.
        workers: Thread count for the copy/symlink phase.
    """
    if mode not in ("copy", "symlink"):
        raise ValueError(f"mode must be 'copy' or 'symlink', got {mode!r}")

    t0 = time.monotonic()
    output_dir = os.path.abspath(output_dir)
    images_dir = os.path.join(output_dir, "images")
    manifest_parquet = os.path.join(output_dir, "manifest.parquet")
    manifest_json = os.path.join(output_dir, "manifest.json")

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not force:
        raise FileExistsError(
            f"{output_dir} is not empty; pass force=True to overwrite"
        )
    os.makedirs(images_dir, exist_ok=True)

    log.info("sample: reading %s", curated_parquet)
    lf = pl.scan_parquet(curated_parquet)
    schema = lf.collect_schema()
    if image_path_col not in schema:
        raise ValueError(
            f"{curated_parquet} missing column {image_path_col!r} "
            f"(have: {list(schema.names())[:10]}...)"
        )

    # Load only the columns we need for sampling + naming. Keep everything in
    # the manifest though.
    df = lf.collect()
    log.info("sample: loaded %d rows × %d cols", df.height, len(df.columns))

    k_eff = _resolve_k(df.height, k)
    if stratify_by is not None:
        if stratify_by not in df.columns:
            raise ValueError(f"stratify_by={stratify_by!r} not a column")
        sampled = _stratified_sample(df, k_eff, stratify_by, seed)
    else:
        sampled = df.sample(n=k_eff, seed=seed, shuffle=True)

    has_dataset_col = dataset_col is not None and dataset_col in sampled.columns
    # Build src/dst pairs
    name_col = dataset_col if has_dataset_col else None
    name_cols = [image_path_col]
    if sample_id_col in sampled.columns:
        name_cols.append(sample_id_col)
    if name_col:
        name_cols.append(name_col)
    tasks = []
    for row in sampled.select(name_cols).to_dicts():
        # Normalize into a consistent dict for _plan_destination
        task_row = {
            "image_path": row[image_path_col],
            "sample_id": row.get(sample_id_col),
            "dataset": row.get(dataset_col) if name_col else None,
        }
        src, dst = _plan_destination(task_row, images_dir, has_dataset_col)
        tasks.append((src, dst))

    log.info("sample: %s %d files with %d workers", mode, len(tasks), workers)
    statuses: list[str] = [""] * len(tasks)
    t_export = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_do_one, src, dst, mode): i
            for i, (src, dst) in enumerate(tasks)
        }
        for f in as_completed(futures):
            i = futures[f]
            statuses[i] = f.result()
    export_elapsed = time.monotonic() - t_export
    n_ok = sum(1 for s in statuses if s == "ok")
    n_missing = sum(1 for s in statuses if s == "missing")
    n_failed = sum(1 for s in statuses if s.startswith("fail"))
    if n_missing + n_failed > 0:
        log.warning(
            "sample: %d missing, %d failed (of %d)",
            n_missing, n_failed, len(tasks),
        )
        # Print a few examples
        for i, s in enumerate(statuses):
            if s != "ok":
                log.warning("  [%d] %s → %s", i, tasks[i][0], s)
                if i > 5:
                    break

    # Write manifest with status column, preserving full-row provenance
    manifest_df = sampled
    if columns_to_keep is not None:
        manifest_df = manifest_df.select(list(columns_to_keep))
    dst_names = [os.path.basename(dst) for _, dst in tasks]
    manifest_df = manifest_df.with_columns(
        pl.Series("export_filename", dst_names),
        pl.Series("export_status", statuses),
    )
    manifest_df.write_parquet(manifest_parquet)

    elapsed = time.monotonic() - t0
    summary = {
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_parquet": os.path.abspath(curated_parquet),
        "output_dir": output_dir,
        "images_dir": images_dir,
        "mode": mode,
        "seed": seed,
        "stratify_by": stratify_by,
        "k_requested": int(k),
        "k_effective": int(k_eff),
        "source_rows": int(df.height),
        "n_sampled": int(len(tasks)),
        "n_exported_ok": int(n_ok),
        "n_missing": int(n_missing),
        "n_failed": int(n_failed),
        "workers": workers,
        "export_elapsed_s": round(export_elapsed, 3),
        "elapsed_s": round(elapsed, 3),
    }
    with open(manifest_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "sample: done — %d %s / %d sampled in %.1fs (%d missing, %d failed)",
        n_ok, mode, len(tasks), elapsed, n_missing, n_failed,
    )

    return SampleResult(
        output_dir=output_dir,
        images_dir=images_dir,
        source_parquet=os.path.abspath(curated_parquet),
        mode=mode,
        k_requested=int(k),
        n_sampled=int(len(tasks)),
        n_exported=int(n_ok),
        n_missing=int(n_missing),
        n_failed=int(n_failed),
        seed=seed,
        stratify_by=stratify_by,
        manifest_parquet=manifest_parquet,
        manifest_json=manifest_json,
        elapsed_s=elapsed,
    )
