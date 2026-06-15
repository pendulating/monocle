"""Cyclomedia catalog indexer.

Pipeline:
  1. Walk dataset trees (fd, see `walker`)
  2. Parse manifest.json per recording (see `manifest`)
  3. Load WFS catalog CSVs (see `wfs`)
  4. Join + derive bearing, geom_wkb, borough, per-face manifest columns
  5. Write hive-partitioned parquet to {output}/by_dataset/dataset=.../year=.../part-0.parquet
  6. Run validation checks; emit validation_report.parquet + summary.md

Call this as a library (`build_catalog`) or via `cli.py`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

import polars as pl
import pyarrow as pa
import pyarrow.dataset as pads
from shapely.geometry import Point
from shapely.wkb import dumps as wkb_dumps

from .manifest import parse_manifests
from .schema import (
    ALL_FACES,
    CATALOG_COLUMNS,
    FACE_BEARING_DEG,
    SCHEMA_VERSION,
    dataset_to_borough,
)
from .walker import WalkResult, walk_dataset
from .wfs import DEFAULT_CATALOG_GLOB, load_wfs_catalog

__all__ = ["build_catalog", "rejoin_wfs", "BuildResult"]

log = logging.getLogger(__name__)


@dataclass
class BuildResult:
    output_root: str
    datasets: list[str]
    row_counts: dict[str, int] = field(default_factory=dict)
    total_rows: int = 0
    elapsed_s: float = 0.0
    validation_summary_path: Optional[str] = None


def _default_datasets(raw_root: str) -> list[str]:
    """Datasets = every top-level subdir of raw_root (skip hidden, skip non-dirs)."""
    out: list[str] = []
    for name in sorted(os.listdir(raw_root)):
        full = os.path.join(raw_root, name)
        if name.startswith(".") or not os.path.isdir(full):
            continue
        out.append(name)
    return out


def _derive_face_columns(recording_df: pl.DataFrame, face: str) -> pl.DataFrame:
    """Given a per-recording DataFrame with wide `face_*_<F>` and `depthmap_*_<F>`
    columns, select the columns for the given face and rename them to their
    non-suffixed names. Used when exploding recording × face."""
    renames = {
        f"face_elapsed_s_{face}": "face_elapsed_s",
        f"face_used_render_{face}": "face_used_render",
        f"depthmap_present_{face}": "depthmap_present",
        f"depthmap_used_render_{face}": "depthmap_used_render",
        f"depthmap_render_size_{face}": "depthmap_render_size",
        f"depthmap_rgb_render_size_{face}": "depthmap_rgb_render_size",
        f"depthmap_downsample_factor_{face}": "depthmap_downsample_factor",
    }
    return recording_df.select(
        "dataset",
        "group",
        "recording_dir",
        "manifest_ok",
        "manifest_image_id",
        "manifest_latitude",
        "manifest_longitude",
        "manifest_zoom",
        "manifest_tile_px",
        "manifest_tile_schema",
        "manifest_name_version",
        "manifest_mode",
        "manifest_checkpoint",
        "manifest_no_tiles",
        "depthmap_stitched",
        *[pl.col(k).alias(v) for k, v in renames.items()],
    ).with_columns(pl.lit(face).cast(pl.Categorical).alias("face"))


def _wkb_point(lat: Optional[float], lon: Optional[float]) -> Optional[bytes]:
    if lat is None or lon is None:
        return None
    try:
        return wkb_dumps(Point(float(lon), float(lat)))  # Point(x=lon, y=lat) in EPSG:4326
    except (TypeError, ValueError):
        return None


def _attach_geom_wkb(df: pl.DataFrame) -> pl.DataFrame:
    """Build geom_wkb from (latitude, longitude). Done in Python since shapely
    is not a first-class Polars expression; one pass over ~M rows is still
    fast (~seconds for 5M)."""
    lats = df["latitude"].to_list()
    lons = df["longitude"].to_list()
    wkbs = [_wkb_point(la, lo) for la, lo in zip(lats, lons)]
    return df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))


def _compute_bearing(face: pl.Expr, recorder_dir: pl.Expr) -> pl.Expr:
    """Absolute compass bearing of each cube face, NULL for U/D.

    Empirical finding (2026-04-22, verified on facdb_libraries): Cyclomedia's
    NYC cube faces are rendered in a **globally-oriented absolute frame** —
    F=North (0°), R=East (90°), B=South (180°), L=West (270°) — regardless of
    the vehicle's direction of travel. The `orientation` column (camera yaw)
    is ~0 (±0.2°) across 100% of NYC rows, confirming the panorama's F-face
    reference axis is fixed to North.

    The earlier implementation added `recorder_direction` (the *vehicle*
    heading, which is a different column than camera yaw — see
    `dagspaces/urbanroamvqa/graph/STREETSMART_API_REFERENCE.md`) to the face
    offset. That treated the cube as vehicle-relative and produced bearings
    that were wrong by the vehicle heading — most visible when the van
    drove opposite directions on the same street and the F face supposedly
    pointed opposite ways while actually showing identical scenes.

    `recorder_dir` is intentionally accepted but unused; kept in the signature
    to avoid breaking callers that still pass it. Remove it in a later cleanup.
    """
    del recorder_dir  # intentionally unused — see docstring
    offset_expr: pl.Expr = pl.lit(None, dtype=pl.Float64)
    for f in ("F", "R", "B", "L"):
        offset_expr = pl.when(face == pl.lit(f)).then(pl.lit(FACE_BEARING_DEG[f])).otherwise(offset_expr)
    return (
        pl.when(face.is_in(["U", "D"]))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(offset_expr % 360.0)
        .cast(pl.Float32)
    )


def _wfs_subset(wfs: pl.DataFrame) -> pl.DataFrame:
    """Pick only the columns we need (safe if any are missing)."""
    keep: list[str] = []
    for c in (
        "recording_id", "recordedAt", "year",
        "lat", "lon",
        "recorderDirection", "yawDegrees", "yawPrecisionDegrees",
        "orientation", "orientationPrecision",
        "statePlaneX", "statePlaneY", "locationSRS",
        "latitudePrecision", "longitudePrecision",
        "height", "heightSystem", "heightPrecision", "groundLevelOffset",
        "productType", "panoramaTileSchema", "tileSchema",
        "hasDepthMap", "isAuthorized",
    ):
        if c in wfs.columns:
            keep.append(c)
    return wfs.select(keep)


def _explode_walk_with_manifests(
    walk: WalkResult,
    manifests: pl.DataFrame,
) -> pl.DataFrame:
    """Walk (per-face) × manifests (per-recording) → one row per face with the
    per-face manifest columns collapsed to non-suffixed names. Produces the
    pre-WFS base frame consumed by `_join_wfs_and_derive`.
    """
    dataset = walk.dataset
    faces_df = walk.frames

    if faces_df.is_empty():
        log.warning("indexer: no face rows for dataset %s", dataset)
        return pl.DataFrame()

    if manifests.is_empty():
        # Empty manifest frame with expected keys+columns so the join still runs.
        empty_cols = {
            "dataset": [], "group": [], "recording_dir": [],
            "manifest_ok": [], "manifest_image_id": [],
            "manifest_latitude": [], "manifest_longitude": [],
            "manifest_zoom": [], "manifest_tile_px": [],
            "manifest_tile_schema": [], "manifest_name_version": [],
            "manifest_mode": [], "manifest_checkpoint": [],
            "manifest_no_tiles": [], "depthmap_stitched": [],
        }
        for f in ALL_FACES:
            for prefix in (
                "face_elapsed_s_", "face_used_render_", "depthmap_present_",
                "depthmap_used_render_", "depthmap_render_size_",
                "depthmap_rgb_render_size_", "depthmap_downsample_factor_",
            ):
                empty_cols[f"{prefix}{f}"] = []
        manifests = pl.DataFrame(empty_cols)

    # Align dtypes on the join keys; walker emits `dataset`/`face` as Categorical
    # but manifests are built from dicts (Utf8). Cast both sides to Utf8.
    faces_df = faces_df.with_columns(
        pl.col("dataset").cast(pl.Utf8),
        pl.col("face").cast(pl.Utf8),
    )
    joined = faces_df.join(
        manifests,
        on=["dataset", "group", "recording_dir"],
        how="left",
    )

    def _coalesce_face_col(col_base: str, dtype: Optional[pl.DataType] = None) -> pl.Expr:
        expr: pl.Expr = pl.lit(None)
        if dtype is not None:
            expr = pl.lit(None, dtype=dtype)
        for f in ALL_FACES:
            col = f"{col_base}_{f}"
            if col in joined.columns:
                expr = pl.when(pl.col("face") == pl.lit(f)).then(pl.col(col)).otherwise(expr)
        return expr

    joined = joined.with_columns(
        _coalesce_face_col("face_elapsed_s", pl.Float64).cast(pl.Float32).alias("face_elapsed_s"),
        _coalesce_face_col("face_used_render", pl.Boolean).alias("face_used_render"),
        _coalesce_face_col("depthmap_present", pl.Boolean).alias("depthmap_present"),
        _coalesce_face_col("depthmap_used_render", pl.Boolean).alias("depthmap_used_render"),
        _coalesce_face_col("depthmap_render_size", pl.Int64).cast(pl.Int32).alias("depthmap_render_size"),
        _coalesce_face_col("depthmap_rgb_render_size", pl.Int64).cast(pl.Int32).alias("depthmap_rgb_render_size"),
        _coalesce_face_col("depthmap_downsample_factor", pl.Float64).cast(pl.Float32).alias("depthmap_downsample_factor"),
    )

    # Resolve recording_id: manifest `imageId` preferred, else dirname.
    joined = joined.with_columns(
        pl.coalesce([pl.col("manifest_image_id"), pl.col("recording_dir")])
          .alias("recording_id")
    )
    return joined


def _join_wfs_and_derive(
    base: pl.DataFrame,
    wfs: pl.DataFrame,
    raw_root: str,
    build_stamp: datetime,
    dataset: str,
) -> pl.DataFrame:
    """Join `base` with WFS and derive every WFS-sourced / computed column.

    `base` must expose: dataset, group, face, image_path, file_size,
    file_mtime_unix, recording_id, manifest_ok, manifest_latitude,
    manifest_longitude, plus all manifest_* / depthmap_* / face_* columns that
    land in CATALOG_COLUMNS. Used by both the full-build path (via
    `_explode_walk_with_manifests`) and the WFS-rejoin path (via
    `rejoin_wfs`)."""
    del raw_root  # reserved for future per-row path validation
    wfs_sub = _wfs_subset(wfs).with_columns(pl.lit(True).alias("_wfs_hit"))
    joined = (
        base.join(wfs_sub, on="recording_id", how="left")
            .with_columns(pl.col("_wfs_hit").fill_null(False).alias("catalog_hit"))
            .drop("_wfs_hit")
    )

    # Lat/lon: prefer manifest, fall back to WFS.
    lat_expr = pl.coalesce([
        pl.col("manifest_latitude"),
        pl.col("lat") if "lat" in joined.columns else pl.lit(None, dtype=pl.Float64),
    ])
    lon_expr = pl.coalesce([
        pl.col("manifest_longitude"),
        pl.col("lon") if "lon" in joined.columns else pl.lit(None, dtype=pl.Float64),
    ])
    joined = joined.with_columns(
        lat_expr.cast(pl.Float64).alias("latitude"),
        lon_expr.cast(pl.Float64).alias("longitude"),
    )

    recorder_dir_expr = (
        pl.col("recorderDirection").cast(pl.Float64)
        if "recorderDirection" in joined.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    joined = joined.with_columns(
        _compute_bearing(pl.col("face").cast(pl.Utf8), recorder_dir_expr).alias("bearing"),
    )

    if "year" in joined.columns:
        year_expr = pl.col("year").cast(pl.Int32, strict=False)
    else:
        year_expr = pl.lit(None, dtype=pl.Int32)
    if "recordedAt" in joined.columns:
        year_expr = pl.coalesce([year_expr, pl.col("recordedAt").dt.year().cast(pl.Int32)])
    year_expr = pl.coalesce([year_expr, pl.lit(build_stamp.year, dtype=pl.Int32)]).cast(pl.Int16).alias("year")
    joined = joined.with_columns(year_expr)

    joined = joined.with_columns(
        (pl.col("recording_id") + pl.lit("_") + pl.col("face").cast(pl.Utf8)).alias("sample_id"),
        pl.lit(dataset_to_borough(dataset)).cast(pl.Categorical).alias("borough"),
        pl.from_epoch(pl.col("file_mtime_unix"), time_unit="s").alias("file_mtime"),
        pl.lit(build_stamp).alias("indexed_at"),
    )

    have = set(joined.columns)
    add_nulls: list[pl.Expr] = []
    for c in CATALOG_COLUMNS:
        if c == "geom_wkb":
            continue
        if c not in have:
            add_nulls.append(pl.lit(None).alias(c))
    if add_nulls:
        joined = joined.with_columns(add_nulls)

    select_cols = [c for c in CATALOG_COLUMNS if c != "geom_wkb"]
    out = joined.select(select_cols)
    out = _attach_geom_wkb(out)
    out = out.select(list(CATALOG_COLUMNS))
    return out


def _build_dataset_rows(
    walk: WalkResult,
    manifests: pl.DataFrame,
    wfs: pl.DataFrame,
    raw_root: str,
    build_stamp: datetime,
) -> pl.DataFrame:
    """Full build path: walk + manifests + WFS → canonical rows for one dataset."""
    base = _explode_walk_with_manifests(walk, manifests)
    if base.is_empty():
        return pl.DataFrame()
    return _join_wfs_and_derive(base, wfs, raw_root, build_stamp, walk.dataset)


def _write_partitioned(df: pl.DataFrame, output_root: str) -> None:
    """Write hive-partitioned parquet: output_root/by_dataset/dataset=X/year=Y/part-0.parquet."""
    if df.is_empty():
        return
    out_dir = os.path.join(output_root, "by_dataset")
    os.makedirs(out_dir, exist_ok=True)

    # Convert to pyarrow table. geom_wkb must stay as binary; cast Categoricals to strings.
    table = df.with_columns(
        pl.col("dataset").cast(pl.Utf8),
        pl.col("face").cast(pl.Utf8),
        pl.col("borough").cast(pl.Utf8),
        # Hive partitioning likes plain strings/ints.
        pl.col("year").cast(pl.Int32),
    ).to_arrow()

    pads.write_dataset(
        table,
        base_dir=out_dir,
        format="parquet",
        partitioning=["dataset", "year"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
        basename_template="part-{i}.parquet",
    )


def build_catalog(
    raw_root: str,
    output_root: str,
    datasets: Optional[Iterable[str]] = None,
    catalog_globs: Iterable[str] = DEFAULT_CATALOG_GLOB,
    fd_path: Optional[str] = None,
    write_validation: bool = True,
) -> BuildResult:
    """Build or rebuild the Cyclomedia catalog rooted at `output_root`.

    `datasets` is a list of directory names under `raw_root`. If None, every
    subdir of `raw_root` is indexed.
    """
    t0 = time.monotonic()
    raw_root = os.path.abspath(raw_root)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    if datasets is None:
        datasets = _default_datasets(raw_root)
    datasets = list(datasets)
    if not datasets:
        raise ValueError(f"No datasets to index under {raw_root}")

    log.info("indexer: building catalog for datasets=%s", datasets)
    wfs = load_wfs_catalog(catalog_globs)

    build_stamp = datetime.now(tz=timezone.utc)
    row_counts: dict[str, int] = {}
    all_rows: list[pl.DataFrame] = []

    for ds in datasets:
        log.info("indexer: [%s] walk", ds)
        walk = walk_dataset(raw_root, ds, fd_path=fd_path)
        if walk.frames.is_empty():
            log.warning("indexer: [%s] walk empty, skipping", ds)
            continue

        log.info("indexer: [%s] parse manifests (%d recordings)", ds, walk.frames.select(["group", "recording_dir"]).unique().height)
        rec_keys = [
            (row["dataset"], row["group"], row["recording_dir"])
            for row in walk.frames.select(["dataset", "group", "recording_dir"]).unique().to_dicts()
        ]
        manifests = parse_manifests(raw_root, rec_keys)

        log.info("indexer: [%s] join + derive", ds)
        rows = _build_dataset_rows(walk, manifests, wfs, raw_root, build_stamp)
        row_counts[ds] = rows.height
        all_rows.append(rows)

        log.info("indexer: [%s] clear + write partitions (%d rows)", ds, rows.height)
        # Clear the old dataset dir so a shrinking walk (or year shift) can't
        # leave stale partition files behind from a prior build.
        _dataset_rmtree(output_root, ds)
        _write_partitioned(rows, output_root)

    total_rows = sum(row_counts.values())

    # Merge into manifest.json instead of overwriting, so a single-dataset
    # rebuild doesn't clobber the metadata for the other datasets on disk.
    manifest_path = os.path.join(output_root, "manifest.json")
    existing_manifest: dict = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)
    merged_row_counts = {**existing_manifest.get("row_counts", {}), **row_counts}
    merged_datasets = sorted(set(existing_manifest.get("datasets", [])) | set(datasets))
    existing_manifest.update({
        "schema_version": SCHEMA_VERSION,
        "built_at": build_stamp.isoformat(),
        "raw_root": raw_root,
        "datasets": merged_datasets,
        "last_built_datasets": datasets,
        "row_counts": merged_row_counts,
        "total_rows": sum(merged_row_counts.values()),
    })
    with open(manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)

    elapsed = time.monotonic() - t0
    log.info("indexer: wrote %d rows across %d datasets in %.1fs",
             total_rows, len(datasets), elapsed)

    result = BuildResult(
        output_root=output_root,
        datasets=datasets,
        row_counts=row_counts,
        total_rows=total_rows,
        elapsed_s=elapsed,
    )

    if write_validation and all_rows:
        from .validation import run_validation
        union = pl.concat(all_rows, how="vertical_relaxed")
        summary_path = run_validation(union, output_root, raw_root=raw_root)
        result.validation_summary_path = summary_path

    return result


# ---- rejoin: refresh WFS-derived columns without re-walking -----------------

# Columns produced by the WFS join or derived from WFS data. Stripped before
# rejoin so the new WFS pass writes fresh values.
_WFS_DERIVED_COLUMNS: tuple[str, ...] = (
    "recordedAt", "recorderDirection", "yawDegrees", "yawPrecisionDegrees",
    "orientation", "orientationPrecision", "statePlaneX", "statePlaneY",
    "locationSRS", "latitudePrecision", "longitudePrecision", "height",
    "heightSystem", "heightPrecision", "groundLevelOffset", "productType",
    "panoramaTileSchema", "tileSchema", "hasDepthMap", "isAuthorized",
    "bearing", "catalog_hit", "year",
    # Recomputed after new WFS join:
    "latitude", "longitude", "geom_wkb",
    # Recomputed in _join_wfs_and_derive:
    "sample_id", "borough", "indexed_at", "file_mtime",
)


def _dataset_rmtree(output_root: str, dataset: str) -> None:
    import shutil
    p = os.path.join(output_root, "by_dataset", f"dataset={dataset}")
    if os.path.isdir(p):
        shutil.rmtree(p)


def rejoin_wfs(
    output_root: str,
    datasets: Iterable[str],
    catalog_globs: Iterable[str] = DEFAULT_CATALOG_GLOB,
    raw_root: str = "/share/ju/cyclomedia/raw",
    write_validation: bool = True,
) -> BuildResult:
    """Re-run the WFS join against existing partitions.

    Reuses the walk + manifest output already written to `by_dataset/`; only
    the WFS-derived columns (recordedAt, recorderDirection, lat/lon fallback,
    bearing, catalog_hit, year, ...) are recomputed against a freshly loaded
    WFS catalog. Orders of magnitude faster than `build_catalog` when the
    fix is to the WFS glob or CSV set, not to the raw tree.
    """
    t0 = time.monotonic()
    output_root = os.path.abspath(output_root)
    datasets = list(datasets)
    if not datasets:
        raise ValueError("rejoin_wfs: datasets is empty")

    log.info("rejoin: rebuilding WFS join for datasets=%s", datasets)
    wfs = load_wfs_catalog(catalog_globs)

    build_stamp = datetime.now(tz=timezone.utc)
    row_counts: dict[str, int] = {}
    all_rows: list[pl.DataFrame] = []

    for ds in datasets:
        ds_dir = os.path.join(output_root, "by_dataset", f"dataset={ds}")
        if not os.path.isdir(ds_dir):
            log.warning("rejoin: [%s] no existing partitions under %s, skipping", ds, ds_dir)
            continue

        log.info("rejoin: [%s] read existing partitions", ds)
        existing = pl.scan_parquet(
            os.path.join(ds_dir, "**", "*.parquet"),
            hive_partitioning=True,
        ).collect()
        if existing.is_empty():
            log.warning("rejoin: [%s] partitions read empty, skipping", ds)
            continue

        # Reconstruct the base-frame contract expected by _join_wfs_and_derive.
        # Existing parquet has latitude/longitude as coalesce(manifest, old_wfs);
        # feed them back as manifest_latitude/manifest_longitude so the new join
        # preserves them where WFS is silent, and overlays WFS where present.
        base = existing.with_columns(
            pl.col("latitude").alias("manifest_latitude"),
            pl.col("longitude").alias("manifest_longitude"),
            # Derive recording_dir (walker basename) from image_path so the
            # manifest/walk downstream code keeps working. Today
            # recording_id == recording_dir for every manifest_ok row (fatal
            # check #1), and the fallback path stores recording_dir in
            # recording_id too.
            pl.col("image_path").str.split("/").list.get(-3).alias("recording_dir"),
            pl.col("file_mtime").dt.epoch("s").alias("file_mtime_unix"),
        )
        drop_cols = [c for c in _WFS_DERIVED_COLUMNS if c in base.columns]
        base = base.drop(drop_cols)

        log.info("rejoin: [%s] join + derive", ds)
        rows = _join_wfs_and_derive(base, wfs, raw_root, build_stamp, ds)
        row_counts[ds] = rows.height
        all_rows.append(rows)

        log.info("rejoin: [%s] clear old partitions and rewrite (%d rows)", ds, rows.height)
        _dataset_rmtree(output_root, ds)
        _write_partitioned(rows, output_root)

    total_rows = sum(row_counts.values())

    # Update manifest.json to reflect the refresh.
    manifest_path = os.path.join(output_root, "manifest.json")
    existing_manifest: dict = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)
    merged_datasets = sorted(
        set(existing_manifest.get("datasets", [])) | set(row_counts.keys())
    )
    existing_manifest.update({
        "schema_version": SCHEMA_VERSION,
        "rejoined_at": build_stamp.isoformat(),
        "rejoined_datasets": datasets,
        "datasets": merged_datasets,
        "row_counts": {**existing_manifest.get("row_counts", {}), **row_counts},
    })
    existing_manifest["total_rows"] = sum(existing_manifest["row_counts"].values())
    with open(manifest_path, "w") as f:
        json.dump(existing_manifest, f, indent=2)

    elapsed = time.monotonic() - t0
    log.info("rejoin: rewrote %d rows across %d datasets in %.1fs",
             total_rows, len(row_counts), elapsed)

    result = BuildResult(
        output_root=output_root,
        datasets=datasets,
        row_counts=row_counts,
        total_rows=total_rows,
        elapsed_s=elapsed,
    )

    if write_validation:
        from .validation import run_validation
        # Validate against the full catalog on disk, not just the rejoined subset,
        # so cross-dataset checks (overlap, uniqueness) see everything.
        full = pl.scan_parquet(
            os.path.join(output_root, "by_dataset", "**", "*.parquet"),
            hive_partitioning=True,
        ).collect()
        summary_path = run_validation(full, output_root, raw_root=raw_root)
        result.validation_summary_path = summary_path

    return result
