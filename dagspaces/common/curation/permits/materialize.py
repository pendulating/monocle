"""Materialize the Cyclomedia sub-dataset for a scaffolding-permits curation.

Given a ``curation/.../coverage.geojson`` produced by
:mod:`.scaffolding_permits`, run a point-in-polygon spatial join against the
Cyclomedia catalog and write the resulting rows to a Parquet alongside the
coverage file. Intended to be invoked from a SLURM job — running over the
full 31M-row catalog is too heavy for an interactive session.

**Why we don't call** ``CyclomediaCatalog.query(within=...)`` **directly:**
that method uses ``polars-st.st.within(literal_multipoly)`` which, at the
time of writing (polars-st 0.7.x / geopolars still prototype — see issue
#27), has no spatial index. It degenerates to a brute-force GEOS containment
test per row against the full MultiPolygon. For a 5,000+ polygon coverage
mask over 4-11M catalog points per borough, that's billions of comparisons
and the job hangs with no progress. Here we use Polars for what it's fast at
(hive-partition pruning, projection, column filter, concat, parquet I/O) and
delegate the one algorithmically-indexed step — the point-in-polygon join —
to ``geopandas.sjoin``, which is STRtree-backed via GEOS. Empirically this
is ~100× faster on our workload.

Chunked by dataset (one borough at a time) to produce incremental progress
logs. Each dataset's result is written to ``chunks/<dataset>.parquet`` as it
completes; the final flat ``<output_filename>`` is produced at the end by
concatenating chunks. A ``materialize_progress.json`` is updated atomically
after every chunk so operators can tail progress mid-run.

Defaults produce the "street-level horizontal" cut sensible for scaffold
imagery: faces F/B/L/R, all 5 boroughs, all columns retained. Override via
CLI flags if a use case needs U/D (sky/ground) or a single borough.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import polars as pl

from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog, DEFAULT_CATALOG_ROOT
from ..filter_facing import (
    DEFAULT_BEARING_TOL_DEG,
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_MAX_DISTANCE_FT,
)

__all__ = ["materialize", "MaterializeResult", "DEFAULT_FACES", "sjoin_dataset_chunk"]

log = logging.getLogger(__name__)

DEFAULT_FACES: tuple[str, ...] = ("F", "B", "L", "R")


@dataclass
class MaterializeResult:
    curation_root: str
    output_path: str
    coverage_path: str
    catalog_root: str
    rows: int
    datasets: list[str]
    faces: list[str]
    elapsed_s: float
    file_size_mb: float
    per_dataset_counts: dict[str, int]
    per_dataset_elapsed_s: dict[str, float]
    manifest_path: str
    progress_path: str
    chunks_dir: str
    chunk_paths: list[str] = field(default_factory=list)
    # Populated when ``facing=True`` — the automatically-produced
    # facing-filtered sibling. ``facing_rows == -1`` means facing was disabled.
    facing_output_path: Optional[str] = None
    facing_rows: int = -1


def _load_coverage(path: str) -> gpd.GeoDataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"coverage GeoJSON not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path} loaded 0 features")
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS set")
    log.info(
        "materialize: loaded coverage from %s — %d features, total area %.2f km²",
        path, len(gdf),
        float(gdf.to_crs("EPSG:2263").area.sum() * (0.3048 ** 2) / 1e6),
    )
    return gdf


def _autodetect_units_parquet(curation_root: str) -> str:
    """Pick the per-unit parquet from a curation dir.

    Recognized files:
      * ``facilities.parquet`` — FacDB
      * ``permits.parquet`` — scaffolding permits
      * ``restaurants_aggregated.parquet`` — DOHMH (after ``aggregate-restaurants``)
      * ``entrances.parquet`` — subway entrances/exits
      * ``open_restaurants.parquet`` — Open Restaurants / Dining Out NYC licenses

    Materialize **prefers** ``restaurants_aggregated.parquet`` over the
    multi-inspection ``restaurants.parquet`` because the latter has
    duplicate ``uid`` values per CAMIS, which would explode the spatial
    join.

    Fail fast if none is present — unit attribution is mandatory so
    downstream consumers have per-image unit attribution baked in. Fail
    *helpfully* if only the unaggregated restaurants.parquet is present —
    the aggregate step is one CLI invocation away.
    """
    candidates = (
        "facilities.parquet",
        "permits.parquet",
        "restaurants_aggregated.parquet",
        "entrances.parquet",
        "open_restaurants.parquet",
    )
    for name in candidates:
        p = os.path.join(curation_root, name)
        if os.path.isfile(p):
            return p

    # Special case: bare restaurants.parquet → require explicit aggregate step.
    bare = os.path.join(curation_root, "restaurants.parquet")
    if os.path.isfile(bare):
        raise FileNotFoundError(
            f"{curation_root} contains restaurants.parquet (inspection-level, "
            f"multiple rows per CAMIS) but no restaurants_aggregated.parquet. "
            f"Run:\n\n"
            f"    python -m dagspaces.common.curation aggregate-restaurants \\\n"
            f"        --parquet {bare}\n\n"
            f"…then re-run materialize-cyclomedia. Or pass --units-path explicitly."
        )

    raise FileNotFoundError(
        f"no units parquet found in {curation_root}; expected one of "
        "facilities.parquet (FacDB), permits.parquet (scaffolding), "
        "restaurants_aggregated.parquet (DOHMH, after aggregate-restaurants), "
        "or entrances.parquet (subway). Re-run the curation build step "
        "that creates it, or pass --units-path explicitly."
    )


def _load_units(path: str) -> gpd.GeoDataFrame:
    """Load a per-unit parquet and return a GeoDataFrame in EPSG:4326 with
    canonical columns ``unit_uid`` / ``unit_name`` / ``geometry``.

    The parquet's primary key + display-name columns depend on the curation
    family:
      * FacDB (``facilities.parquet``): ``uid`` + ``facname``
      * DOHMH restaurants (``restaurants_aggregated.parquet``): ``uid`` + ``facname``
      * Subway entrances (``entrances.parquet``): ``uid`` + ``facname``
      * scaffolding-permits (``permits.parquet``): ``permit_id`` + ``address``

    Geometry comes from the ``geom_wkb`` column (already the buffered
    polygon at curation time).
    """
    from shapely.wkb import loads as wkb_loads

    if not os.path.isfile(path):
        raise FileNotFoundError(f"units parquet not found: {path}")
    df = pl.read_parquet(path)
    cols = set(df.columns)

    if "uid" in cols and "facname" in cols:
        uid_col, name_col = "uid", "facname"
    elif "permit_id" in cols and "address" in cols:
        uid_col, name_col = "permit_id", "address"
    elif "permit_id" in cols:
        uid_col, name_col = "permit_id", "permit_id"
    else:
        raise ValueError(
            f"{path} has no recognizable unit key — expected uid+facname "
            "(FacDB) or permit_id+address (scaffolding). Columns: "
            f"{sorted(cols)[:12]}..."
        )
    if "geom_wkb" not in cols:
        raise ValueError(f"{path} missing geom_wkb column; can't load buffered geometries")

    # Refuse duplicate unit IDs — the spatial join produces one attribution
    # row per match, and duplicate IDs silently inflate the output.
    n_dup = int(df.select(pl.col(uid_col).is_duplicated().sum()).item())
    if n_dup > 0:
        raise ValueError(
            f"{path} has {n_dup} duplicate {uid_col} values — every unit must "
            f"appear exactly once. For DOHMH restaurants, run "
            f"`aggregate-restaurants` first. For other sources, dedup before "
            f"passing to materialize."
        )

    pdf = df.select([uid_col, name_col, "geom_wkb"]).to_pandas()
    geoms = [wkb_loads(b) if b is not None else None for b in pdf["geom_wkb"]]
    gdf = gpd.GeoDataFrame(
        {
            "unit_uid": pdf[uid_col].astype(str),
            "unit_name": pdf[name_col].astype(str),
        },
        geometry=geoms,
        crs="EPSG:4326",
    )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    log.info(
        "materialize: loaded %d units from %s (key=%s name=%s)",
        len(gdf), path, uid_col, name_col,
    )
    return gdf


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def sjoin_dataset_chunk(
    cat: CyclomediaCatalog,
    dataset: str,
    units_gdf: gpd.GeoDataFrame,
    faces: Iterable[str],
    columns: Optional[Iterable[str]] = None,
    projected_crs: str = "EPSG:2263",
) -> tuple[pl.DataFrame, dict]:
    """Spatial-filter a single catalog dataset against ``units_gdf``.

    ``units_gdf`` is the per-unit buffered-polygon frame (one row per
    permit / facility, with columns ``unit_uid``, ``unit_name``, and
    ``geometry`` = the 80-ft-buffered polygon). sjoin'ing against it
    (rather than a dissolved coverage) labels every matched row with the
    unit it falls inside.

    When a recording point sits inside N overlapping buffers, we keep only
    the attribution with the smallest recording → unit-centroid distance
    (Fix B). ``unit_dist_ft`` stores that distance in US feet instead of
    the old placeholder 0.0.

    Returns ``(df, stats)``. ``df`` has the catalog columns plus
    ``unit_uid`` / ``unit_name`` / ``unit_dist_ft``.
    """
    stats: dict = {"dataset": dataset}
    faces = [f.upper() for f in faces]

    # -- scan + project + face filter (Polars, fast, hive-pruned) --
    t = time.monotonic()
    lf = cat.scan(datasets=[dataset]).filter(pl.col("face").cast(pl.Utf8).is_in(faces))
    if columns is not None:
        lf = lf.select(list(columns))
    df = lf.collect()
    stats["scan_collect_s"] = round(time.monotonic() - t, 2)
    stats["scanned_rows"] = int(df.height)
    if df.is_empty():
        log.info("sjoin[%s]: catalog returned 0 rows after face filter", dataset)
        stats["sjoined_rows"] = 0
        return df, stats

    # -- build a lightweight points GDF (lat/lon only; full frame stays in Polars) --
    t = time.monotonic()
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    n = df.height
    pts = gpd.GeoDataFrame(
        {"_row": np.arange(n, dtype=np.int64)},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    stats["points_build_s"] = round(time.monotonic() - t, 2)

    # -- STRtree-backed sjoin against per-unit buffered polygons --
    t = time.monotonic()
    joined = gpd.sjoin(
        pts[["_row", "geometry"]],
        units_gdf[["unit_uid", "unit_name", "geometry"]],
        predicate="within",
        how="inner",
    )
    stats["sjoin_s"] = round(time.monotonic() - t, 2)
    stats["sjoin_matches"] = int(len(joined))

    # -- distance-based disambiguation (Fix B) --
    # Project recording points and unit centroids to a planar CRS (feet), then
    # for each _row keep only the match with smallest distance to centroid.
    t = time.monotonic()
    multi_attr_rows = 0
    if joined.empty:
        result = df.head(0).with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("unit_uid"),
            pl.lit(None, dtype=pl.Utf8).alias("unit_name"),
            pl.lit(0.0, dtype=pl.Float64).alias("unit_dist_ft"),
        )
    else:
        # Centroid per unit_uid (computed once, reused across dataset chunks
        # would be nicer — kept local for isolation).
        units_projected = units_gdf.to_crs(projected_crs)
        unit_centroids = units_projected.geometry.centroid
        centroid_xs = dict(zip(units_projected["unit_uid"].astype(str), unit_centroids.x))
        centroid_ys = dict(zip(units_projected["unit_uid"].astype(str), unit_centroids.y))

        # Recording points in the same projected CRS.
        pts_proj = pts[["_row", "geometry"]].to_crs(projected_crs)
        row_x = dict(zip(pts_proj["_row"].to_numpy(), pts_proj.geometry.x))
        row_y = dict(zip(pts_proj["_row"].to_numpy(), pts_proj.geometry.y))

        j = joined[["_row", "unit_uid", "unit_name"]].copy()
        j["unit_uid"] = j["unit_uid"].astype(str)
        j["unit_name"] = j["unit_name"].astype(str)
        j["_rx"] = j["_row"].map(row_x)
        j["_ry"] = j["_row"].map(row_y)
        j["_cx"] = j["unit_uid"].map(centroid_xs)
        j["_cy"] = j["unit_uid"].map(centroid_ys)
        j["unit_dist_ft"] = np.hypot(j["_cx"] - j["_rx"], j["_cy"] - j["_ry"])

        # How many rows had >1 candidate attribution?
        multi_attr_rows = int(
            (j.groupby("_row").size() > 1).sum()
        )
        # Keep the closest (row, unit) per _row.
        j = j.sort_values(["_row", "unit_dist_ft"], kind="mergesort").drop_duplicates("_row", keep="first")

        match_df = pl.DataFrame({
            "_row": pl.Series(values=j["_row"].to_numpy(), dtype=pl.Int64),
            "unit_uid": pl.Series(values=j["unit_uid"].to_numpy(), dtype=pl.Utf8),
            "unit_name": pl.Series(values=j["unit_name"].to_numpy(), dtype=pl.Utf8),
            "unit_dist_ft": pl.Series(values=j["unit_dist_ft"].to_numpy(), dtype=pl.Float64),
        })
        result = (
            df.with_row_index(name="_row")
              .join(match_df, on="_row", how="inner")
              .drop("_row")
        )
    stats["gather_s"] = round(time.monotonic() - t, 2)
    stats["sjoined_rows"] = int(result.height)
    stats["multi_attr_rows_deduped"] = multi_attr_rows
    stats["unique_images_in_match"] = int(len(set(joined["_row"].tolist()))) if not joined.empty else 0
    return result, stats


def materialize(
    curation_root: str,
    *,
    catalog_root: str = DEFAULT_CATALOG_ROOT,
    faces: Iterable[str] = DEFAULT_FACES,
    datasets: Optional[Iterable[str]] = None,
    output_filename: str = "cyclomedia_near_permits.parquet",
    columns: Optional[Iterable[str]] = None,
    keep_chunks: bool = True,
    facing: bool = True,
    facing_ray_length_m: float = 30.0,
    facing_bearing_tol_deg: float = DEFAULT_BEARING_TOL_DEG,
    facing_max_distance_ft: Optional[float] = DEFAULT_MAX_DISTANCE_FT,
    facing_occlusion: bool = True,
    facing_buildings_path: str = DEFAULT_BUILDINGS_PATH,
    units_path: Optional[str] = None,
) -> MaterializeResult:
    """Query the Cyclomedia catalog against the per-unit buffered polygons
    from ``facilities.parquet`` / ``permits.parquet`` and write parquet.

    The spatial join targets the **per-unit** buffered polygons (one row per
    facility or permit), not a dissolved coverage mask, so every output row
    is labeled with the unit it sits inside (``unit_uid`` / ``unit_name``).
    A catalog point inside N overlapping buffered polygons produces N rows,
    one per `(point, unit)` pair.

    Args:
        curation_root: Curation dir containing ``coverage.geojson`` and
            ``facilities.parquet`` (or ``permits.parquet``).
        catalog_root: Path to the Cyclomedia catalog root.
        faces: Subset of {F,B,L,R,U,D}. Default F/B/L/R (horizontal street faces).
        datasets: Restrict to these dataset names. ``None`` → all available.
        output_filename: Parquet filename written inside ``curation_root``.
        columns: If set, only keep these catalog columns (prior to unit-column
            append).
        keep_chunks: If True (default), the per-dataset chunk parquets remain
            on disk under ``chunks/`` after the final concat.
        facing: If True (default), also produce a facing-filtered sibling
            (in per-unit mode; see ``filter_facing``).
        facing_ray_length_m: Ray length (meters) for the facing filter.
        facing_bearing_tol_deg: Max angular offset (degrees) between a face's
            bearing and the bearing from recording → attributed unit centroid.
            Default = ``filter_facing.DEFAULT_BEARING_TOL_DEG`` (22.5°, i.e.
            the unit must sit within the center 45° of the face's 90° FOV).
        facing_max_distance_ft: Fix D hard cap on recording → unit centroid
            distance, in US feet. Default 200. Pass ``None`` to disable.
        facing_occlusion: Fix F — if True (default), drop rows whose LOS to
            the library's own building is strictly pierced by a non-unit
            NYC building. Disable only for smoke tests / debugging.
        facing_buildings_path: Path to ``nyc_buildings.parquet`` used by the
            Fix F occlusion check.
        units_path: Override the auto-detected units parquet path. Default:
            auto-detect ``facilities.parquet`` → ``permits.parquet``.
    """
    t0 = time.monotonic()
    curation_root = os.path.abspath(curation_root)
    if not os.path.isdir(curation_root):
        raise FileNotFoundError(f"curation_root not a dir: {curation_root}")

    coverage_path = os.path.join(curation_root, "coverage.geojson")
    coverage = _load_coverage(coverage_path)

    units_path = units_path or _autodetect_units_parquet(curation_root)
    units_gdf = _load_units(units_path)

    cat = CyclomediaCatalog(catalog_root)
    all_datasets = cat.datasets()
    use_datasets = list(datasets) if datasets is not None else all_datasets
    missing = [d for d in use_datasets if d not in all_datasets]
    if missing:
        raise ValueError(
            f"requested datasets not in catalog: {missing}; available: {all_datasets}"
        )
    use_faces = [f.upper() for f in faces]
    log.info(
        "materialize: catalog_root=%s  datasets=%s  faces=%s  chunks=%d",
        catalog_root, use_datasets, use_faces, len(use_datasets),
    )

    chunks_dir = os.path.join(curation_root, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    progress_path = os.path.join(curation_root, "materialize_progress.json")
    manifest_path = os.path.join(curation_root, "cyclomedia_materialize_manifest.json")
    output_path = os.path.join(curation_root, output_filename)

    # Prime the progress file so tailers see "in progress" before the first chunk.
    _atomic_write_json(progress_path, {
        "status": "in_progress",
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
        "curation_root": curation_root,
        "catalog_root": os.path.abspath(catalog_root),
        "faces": use_faces,
        "datasets_planned": use_datasets,
        "chunks_total": len(use_datasets),
        "chunks_completed": 0,
        "per_dataset_counts": {},
        "per_dataset_elapsed_s": {},
        "cumulative_rows": 0,
    })

    per_dataset_counts: dict[str, int] = {}
    per_dataset_elapsed_s: dict[str, float] = {}
    per_dataset_stats: dict[str, dict] = {}
    chunk_paths: list[str] = []
    cumulative_rows = 0
    total = len(use_datasets)

    for i, ds in enumerate(use_datasets, start=1):
        t_chunk = time.monotonic()
        log.info("chunk %d/%d: dataset=%s — scan + sjoin...", i, total, ds)
        df, chunk_stats = sjoin_dataset_chunk(
            cat, ds, units_gdf, faces=use_faces, columns=columns,
        )
        n_rows = df.height
        chunk_path = os.path.join(chunks_dir, f"{ds}.parquet")

        t_write = time.monotonic()
        if n_rows > 0:
            df.write_parquet(chunk_path)
            chunk_paths.append(chunk_path)
        elif os.path.isfile(chunk_path):
            os.remove(chunk_path)
        chunk_stats["write_s"] = round(time.monotonic() - t_write, 2)

        elapsed = time.monotonic() - t_chunk
        cumulative_rows += n_rows
        per_dataset_counts[ds] = int(n_rows)
        per_dataset_elapsed_s[ds] = round(elapsed, 2)
        per_dataset_stats[ds] = chunk_stats
        log.info(
            "chunk %d/%d: dataset=%s → %d rows in %.1fs "
            "(scan %.1fs, pts %.1fs, sjoin %.1fs on %d candidates, gather %.1fs, write %.1fs) "
            "cumulative %d across %d/%d",
            i, total, ds, n_rows, elapsed,
            chunk_stats.get("scan_collect_s", 0.0),
            chunk_stats.get("points_build_s", 0.0),
            chunk_stats.get("sjoin_s", 0.0),
            chunk_stats.get("scanned_rows", 0),
            chunk_stats.get("gather_s", 0.0),
            chunk_stats.get("write_s", 0.0),
            cumulative_rows, i, total,
        )

        _atomic_write_json(progress_path, {
            "status": "in_progress",
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "curation_root": curation_root,
            "catalog_root": os.path.abspath(catalog_root),
            "faces": use_faces,
            "datasets_planned": use_datasets,
            "chunks_total": total,
            "chunks_completed": i,
            "per_dataset_counts": per_dataset_counts,
            "per_dataset_elapsed_s": per_dataset_elapsed_s,
            "per_dataset_stats": per_dataset_stats,
            "cumulative_rows": cumulative_rows,
            "last_completed": ds,
            "last_chunk_path": chunk_path if n_rows > 0 else None,
        })

    # -------- concat + final write --------
    log.info("materialize: concatenating %d chunk(s) → %s", len(chunk_paths), output_path)
    if chunk_paths:
        full = pl.concat(
            [pl.read_parquet(p) for p in chunk_paths],
            how="vertical_relaxed",
        )
    else:
        full = pl.DataFrame()
    full.write_parquet(output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    log.info(
        "materialize: wrote %d rows → %s (%.1f MB)",
        full.height, output_path, size_mb,
    )

    if not keep_chunks:
        import shutil
        shutil.rmtree(chunks_dir, ignore_errors=True)
        log.info("materialize: removed chunks dir %s (keep_chunks=False)", chunks_dir)

    # -------- facing filter (default on) --------
    facing_output_path: Optional[str] = None
    facing_rows = -1
    if facing and full.height > 0:
        from ..filter_facing import filter_facing as _filter_facing
        stem, ext = os.path.splitext(output_path)
        facing_output_path = stem + "_facing" + ext
        log.info("materialize: running facing filter → %s", facing_output_path)
        try:
            ff = _filter_facing(
                input_parquet=output_path,
                coverage_geojson=coverage_path,
                output_parquet=facing_output_path,
                ray_length_m=facing_ray_length_m,
                units_parquet=units_path,
                bearing_tol_deg=facing_bearing_tol_deg,
                max_distance_ft=facing_max_distance_ft,
                occlusion=facing_occlusion,
                buildings_path=facing_buildings_path,
                overwrite=True,
            )
            facing_rows = ff.kept_rows
            log.info(
                "materialize: facing filter kept %d/%d rows (%.1f%%)",
                ff.kept_rows, ff.in_rows,
                100 * ff.kept_rows / max(ff.in_rows, 1),
            )
        except Exception as exc:
            log.error("materialize: facing filter failed — %s. Unfiltered parquet "
                      "is still valid at %s.", exc, output_path)
            facing_output_path = None

    elapsed_total = time.monotonic() - t0
    manifest = {
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "coverage_source": coverage_path,
        "coverage_n_features": int(len(coverage)),
        "catalog_root": os.path.abspath(catalog_root),
        "datasets_queried": use_datasets,
        "datasets_available": all_datasets,
        "faces": use_faces,
        "output_path": output_path,
        "rows": int(full.height),
        "per_dataset_counts": per_dataset_counts,
        "per_dataset_elapsed_s": per_dataset_elapsed_s,
        "per_dataset_stats": per_dataset_stats,
        "file_size_mb": round(size_mb, 3),
        "elapsed_s": round(elapsed_total, 3),
        "chunks_kept": keep_chunks,
        "chunks_dir": chunks_dir if keep_chunks else None,
        "spatial_join_backend": "geopandas.sjoin (STRtree via GEOS)",
        "units_path": units_path,
        "units_count": int(len(units_gdf)),
        "facing_enabled": bool(facing),
        "facing_mode": "per_unit" if facing else None,
        "facing_ray_length_m": facing_ray_length_m if facing else None,
        "facing_bearing_tol_deg": facing_bearing_tol_deg if facing else None,
        "facing_max_distance_ft": facing_max_distance_ft if facing else None,
        "facing_output_path": facing_output_path,
        "facing_rows": int(facing_rows) if facing_rows >= 0 else None,
    }
    _atomic_write_json(manifest_path, manifest)
    log.info("materialize: wrote manifest → %s", manifest_path)

    # Flip progress to "done" so tailers see terminal state.
    _atomic_write_json(progress_path, {
        "status": "done",
        "finished_at": datetime.now(tz=timezone.utc).isoformat(),
        "curation_root": curation_root,
        "chunks_total": total,
        "chunks_completed": total,
        "per_dataset_counts": per_dataset_counts,
        "per_dataset_elapsed_s": per_dataset_elapsed_s,
        "cumulative_rows": int(full.height),
        "output_path": output_path,
        "facing_output_path": facing_output_path,
        "facing_rows": int(facing_rows),
        "file_size_mb": round(size_mb, 3),
        "elapsed_s": round(elapsed_total, 3),
    })

    return MaterializeResult(
        curation_root=curation_root,
        output_path=output_path,
        coverage_path=coverage_path,
        catalog_root=os.path.abspath(catalog_root),
        rows=int(full.height),
        datasets=use_datasets,
        faces=use_faces,
        elapsed_s=elapsed_total,
        file_size_mb=size_mb,
        per_dataset_counts=per_dataset_counts,
        per_dataset_elapsed_s=per_dataset_elapsed_s,
        manifest_path=manifest_path,
        progress_path=progress_path,
        chunks_dir=chunks_dir,
        chunk_paths=chunk_paths,
        facing_output_path=facing_output_path,
        facing_rows=int(facing_rows),
    )
