"""CLI for curation sub-dataset builds.

Usage::

    python -m dagspaces.common.curation scaffolding-permits \\
        --cutoff 2025-12-31 \\
        --buffer-ft 80 \\
        --out curation/scaffolding_permits_through_2025/
"""

from __future__ import annotations

import argparse
import logging
import sys

from .dohmh.aggregate import aggregate_restaurants as aggregate_dohmh_restaurants
from .dohmh.cuisines import UnknownCuisineError
from .dohmh.dohmh_restaurants import build as build_dohmh
from .dohmh.validation import DohmhValidationError
from .facdb.categorization import UnknownCategoryError
from .facdb.facdb_facilities import build as build_facdb
from .facdb.validation import FacdbValidationError
from .filter_facing import (
    DEFAULT_BEARING_TOL_DEG,
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_MAX_DISTANCE_FT,
    DEFAULT_RAY_LENGTH_M,
    DEFAULT_RAY_SAMPLES,
    filter_facing,
)
from .open_restaurants.license_types import UnknownLicenseTypeError
from .open_restaurants.open_restaurants import build as build_open_restaurants
from .open_restaurants.validation import OpenRestaurantsValidationError
from .permits.materialize import DEFAULT_FACES, materialize as materialize_cyclomedia
from .permits.scaffolding_permits import build as build_scaffolding_permits
from .permits.validation import PermitValidationError
from .sample import DEFAULT_WORKERS, sample_images
from .subway.entrance_types import UnknownEntranceTypeError
from .subway.subway_entrances import build as build_subway_entrances
from .subway.validation import SubwayValidationError


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_scaffolding_permits(args: argparse.Namespace) -> int:
    try:
        r = build_scaffolding_permits(
            out=args.out,
            cutoff=args.cutoff,
            since=args.since,
            buffer_ft=args.buffer_ft,
            buildings_path=args.buildings,
            refresh=args.refresh,
            bin_match_warn_threshold=args.bin_match_threshold,
            nearest_max_ft=args.nearest_max_ft,
        )
    except PermitValidationError as exc:
        print(f"FATAL: validation failed — {exc}", file=sys.stderr)
        print(f"  See: {args.out}/summary.md", file=sys.stderr)
        return 2

    print()
    print(f"Built scaffolding-permits sub-dataset at {r.output_root}")
    range_str = f"{r.since or '(no lower bound)'} → {r.cutoff}"
    print(f"  issue-date range: {range_str}   buffer: {r.buffer_ft:.0f} ft")
    print(f"  publishable rows: {r.total_publishable:,}")
    print(f"  dob_now: {r.dob_now_rows:,}   bis: {r.bis_rows:,}")
    if r.validation is not None:
        m = r.validation.metrics
        print(f"  polygon match: {m.get('polygon_match_rate_overall_pct', 0):.2f}% "
              f"(bin_exact {m.get('bin_exact_rate_overall_pct', 0):.2f}% + "
              f"nearest {m.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
        print(f"  coverage area: {m.get('coverage_area_km2', 0):.2f} km²")
    print(f"  summary:  {r.summary_path}")
    print(f"  permits:  {r.permits_parquet}")
    print(f"  coverage: {r.coverage_geojson}")
    print(f"  elapsed: {r.elapsed_s:.1f}s")
    return 0


def _cmd_materialize_cyclomedia(args: argparse.Namespace) -> int:
    r = materialize_cyclomedia(
        curation_root=args.curation_root,
        catalog_root=args.catalog_root,
        faces=args.faces,
        datasets=args.datasets,
        output_filename=args.output_filename,
        columns=args.columns,
        keep_chunks=not args.drop_chunks,
        facing=not args.no_facing,
        facing_ray_length_m=args.facing_ray_length_m,
        facing_bearing_tol_deg=args.facing_bearing_tol_deg,
        facing_max_distance_ft=args.facing_max_distance_ft,
        facing_occlusion=not args.no_facing_occlusion,
        facing_buildings_path=args.facing_buildings_path,
        units_path=args.units_path,
    )
    print()
    print(f"Materialized Cyclomedia sub-dataset at {r.output_path}")
    print(f"  rows: {r.rows:,}")
    print(f"  file size: {r.file_size_mb:.1f} MB")
    print(f"  datasets: {r.datasets}")
    print(f"  faces: {r.faces}")
    if r.per_dataset_counts:
        print("  per-dataset row counts:")
        for ds in sorted(r.per_dataset_counts):
            print(f"    {ds}: {r.per_dataset_counts[ds]:,}")
    print(f"  manifest: {r.manifest_path}")
    if r.facing_output_path:
        print(f"  facing:   {r.facing_output_path}  ({r.facing_rows:,} rows, "
              f"{100*r.facing_rows/max(r.rows,1):.1f}% kept — recommended default)")
    print(f"  elapsed: {r.elapsed_s:.1f}s")
    return 0


def _cmd_facdb_facilities(args: argparse.Namespace) -> int:
    try:
        r = build_facdb(
            out=args.out,
            facdomain=args.facdomain,
            facgroup=args.facgroup,
            facsubgrp=args.facsubgrp,
            factype=args.factype,
            buffer_ft=args.buffer_ft,
            buildings_path=args.buildings,
            refresh=args.refresh,
            bin_match_warn_threshold=args.bin_match_threshold,
            nearest_max_ft=args.nearest_max_ft,
        )
    except UnknownCategoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except FacdbValidationError as exc:
        print(f"FATAL: validation failed — {exc}", file=sys.stderr)
        print(f"  See: {args.out}/summary.md", file=sys.stderr)
        return 2

    print()
    print(f"Built FacDB sub-dataset at {r.output_root}")
    print(f"  filters: {r.filters or '(none — full FacDB pull)'}")
    print(f"  buffer: {args.buffer_ft:.0f} ft")
    print(f"  raw rows:          {r.raw_rows:,}")
    print(f"  publishable rows:  {r.total_publishable:,}")
    if r.validation is not None:
        m = r.validation.metrics
        print(f"  polygon match: {m.get('polygon_match_rate_overall_pct', 0):.2f}% "
              f"(bin_exact {m.get('bin_exact_rate_overall_pct', 0):.2f}% + "
              f"nearest {m.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
        print(f"  coverage area: {m.get('coverage_area_km2', 0):.2f} km²")
    print(f"  summary:  {r.summary_path}")
    print(f"  facilities: {r.facilities_parquet}")
    print(f"  coverage: {r.coverage_geojson}")
    print(f"  elapsed:  {r.elapsed_s:.1f}s")
    return 0


def _cmd_dohmh_restaurants(args: argparse.Namespace) -> int:
    try:
        r = build_dohmh(
            out=args.out,
            cuisines=args.cuisine,
            boroughs=args.borough,
            buffer_ft=args.buffer_ft,
            buildings_path=args.buildings,
            refresh=args.refresh,
            bin_match_warn_threshold=args.bin_match_threshold,
            nearest_max_ft=args.nearest_max_ft,
            drop_placeholder_only=args.drop_placeholder_only,
        )
    except UnknownCuisineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except DohmhValidationError as exc:
        print(f"FATAL: validation failed — {exc}", file=sys.stderr)
        print(f"  See: {args.out}/summary.md", file=sys.stderr)
        return 2

    print()
    print(f"Built DOHMH restaurant sub-dataset at {r.output_root}")
    print(f"  filters: {r.filters or '(none — full DOHMH pull)'}")
    print(f"  buffer: {args.buffer_ft:.0f} ft")
    print(f"  raw rows (camis × inspection × violation): {r.raw_rows:,}")
    print(f"  publishable rows (one per camis × inspection_date): {r.total_publishable:,}")
    if r.validation is not None:
        m = r.validation.metrics
        print(f"  unique CAMIS (restaurants): {m.get('unique_camis', 0):,}")
        print(f"  polygon match: {m.get('polygon_match_rate_overall_pct', 0):.2f}% "
              f"(bin_exact {m.get('bin_exact_rate_overall_pct', 0):.2f}% + "
              f"nearest {m.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
        print(f"  coverage area: {m.get('coverage_area_km2', 0):.2f} km²")
        if m.get("camis_only_placeholder"):
            print(f"  CAMIS with only placeholder rows: {m['camis_only_placeholder']:,}")
    print()
    print("  Next: run 'aggregate-restaurants' to collapse to one row per "
          "CAMIS (required before 'materialize-cyclomedia').")
    print(f"  summary:     {r.summary_path}")
    print(f"  restaurants: {r.restaurants_parquet}")
    print(f"  coverage:    {r.coverage_geojson}")
    print(f"  elapsed:     {r.elapsed_s:.1f}s")
    return 0


def _cmd_subway_entrances(args: argparse.Namespace) -> int:
    try:
        r = build_subway_entrances(
            out=args.out,
            entrance_types=args.entrance_type,
            divisions=args.division,
            boroughs=args.borough,
            routes=args.route,
            buffer_ft=args.buffer_ft,
            refresh=args.refresh,
        )
    except UnknownEntranceTypeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except SubwayValidationError as exc:
        print(f"FATAL: validation failed — {exc}", file=sys.stderr)
        print(f"  See: {args.out}/summary.md", file=sys.stderr)
        return 2

    print()
    print(f"Built subway-entrances sub-dataset at {r.output_root}")
    print(f"  filters: {r.filters or '(none — full pull)'}")
    print(f"  buffer: {args.buffer_ft:.0f} ft (around each entrance point)")
    print(f"  raw rows:        {r.raw_rows:,}")
    print(f"  publishable rows: {r.total_publishable:,}")
    if r.validation is not None:
        m = r.validation.metrics
        print(f"  unique stations:  {m.get('unique_stations', 0):,} "
              f"(complexes: {m.get('unique_complexes', 0):,})")
        print(f"  coverage area:    {m.get('coverage_area_km2', 0):.3f} km²")
    print(f"  summary:    {r.summary_path}")
    print(f"  entrances:  {r.entrances_parquet}")
    print(f"  coverage:   {r.coverage_geojson}")
    print(f"  elapsed:    {r.elapsed_s:.1f}s")
    return 0


def _cmd_open_restaurants(args: argparse.Namespace) -> int:
    try:
        r = build_open_restaurants(
            out=args.out,
            license_types=args.license_type,
            boroughs=args.borough,
            buffer_ft=args.buffer_ft,
            buildings_path=args.buildings,
            refresh=args.refresh,
            bin_match_warn_threshold=args.bin_match_threshold,
            nearest_max_ft=args.nearest_max_ft,
        )
    except (UnknownLicenseTypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OpenRestaurantsValidationError as exc:
        print(f"FATAL: validation failed — {exc}", file=sys.stderr)
        print(f"  See: {args.out}/summary.md", file=sys.stderr)
        return 2

    print()
    print(f"Built Open Restaurants sub-dataset at {r.output_root}")
    print(f"  filters: {r.filters or '(none — full issued-license pull)'}")
    print(f"  buffer: {args.buffer_ft:.0f} ft")
    print(f"  raw rows:          {r.raw_rows:,}")
    print(f"  publishable rows:  {r.total_publishable:,}")
    if r.validation is not None:
        m = r.validation.metrics
        print(f"  unique tax lots (BBL): {m.get('unique_bbl', 0):,}")
        print(f"  polygon match: {m.get('polygon_match_rate_overall_pct', 0):.2f}% "
              f"(bin_exact {m.get('bin_exact_rate_overall_pct', 0):.2f}% + "
              f"nearest {m.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
        print(f"  coverage area: {m.get('coverage_area_km2', 0):.3f} km²")
    print()
    print("  Next: 'materialize-cyclomedia' (auto-detects open_restaurants.parquet).")
    print(f"  summary:     {r.summary_path}")
    print(f"  restaurants: {r.restaurants_parquet}")
    print(f"  coverage:    {r.coverage_geojson}")
    print(f"  elapsed:     {r.elapsed_s:.1f}s")
    return 0


def _cmd_aggregate_restaurants(args: argparse.Namespace) -> int:
    try:
        r = aggregate_dohmh_restaurants(
            input_parquet=args.parquet,
            output_parquet=args.out,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Aggregated DOHMH restaurants: {r.input_parquet}")
    print(f"  inspection rows in:    {r.in_rows:,}")
    print(f"  unique CAMIS out:      {r.out_rows:,}")
    print(f"  CAMIS only placeholder: {r.n_only_placeholder:,}")
    print(f"  output:                {r.output_parquet}")
    print(f"  manifest:              {r.manifest_path}")
    print(f"  elapsed:               {r.elapsed_s:.1f}s")
    return 0


def _cmd_filter_facing(args: argparse.Namespace) -> int:
    try:
        r = filter_facing(
            input_parquet=args.parquet,
            coverage_geojson=args.coverage,
            output_parquet=args.out,
            ray_length_m=args.ray_length_m,
            ray_samples=args.ray_samples,
            units_parquet=args.units,
            bearing_tol_deg=args.bearing_tol_deg,
            max_distance_ft=args.max_distance_ft,
            occlusion=not args.no_occlusion,
            buildings_path=args.buildings_path,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Facing-filtered {r.in_rows:,} → {r.kept_rows:,} rows "
          f"({100 * r.kept_rows / max(r.in_rows, 1):.1f}% kept)  [mode={r.mode}]")
    print(f"  ray length: {r.ray_length_m:.0f} m")
    if r.mode == "per_unit":
        print(f"  bearing tolerance: {r.bearing_tol_deg:.0f}°")
        if r.max_distance_ft is not None:
            print(f"  max distance:      {r.max_distance_ft:.0f} ft")
        else:
            print(f"  max distance:      (disabled)")
        print(f"  dropped non-horizontal/null-bearing/unit: "
              f"{r.in_rows - r.with_bearing_rows:,}")
        print(f"  dropped by ray-vs-own-polygon (A): {r.dropped_by_ray:,}")
        print(f"  dropped by bearing tolerance (C):  {r.dropped_by_bearing:,}")
        if r.dropped_by_distance is not None:
            print(f"  dropped by distance cap (D):       {r.dropped_by_distance:,}")
        if r.dropped_by_occlusion is not None:
            print(f"  dropped by occlusion (F):          {r.dropped_by_occlusion:,}"
                  f"  ({r.occlusion_processable or 0:,} had a library BIN; "
                  f"the rest fell through as pass-through)")
        if r.mean_confidence is not None:
            print(f"  attribution_confidence (E): mean={r.mean_confidence:.3f} "
                  f"median={r.median_confidence:.3f}")
    else:
        print(f"  dropped non-horizontal or null-bearing: "
              f"{r.in_rows - r.with_bearing_rows:,}")
        print(f"  dropped by ray-vs-coverage: "
              f"{r.with_bearing_rows - r.kept_rows:,}")
    if r.per_face_kept:
        print(f"  kept by face: {dict(sorted(r.per_face_kept.items()))}")
    print(f"  output:   {r.output_parquet}")
    print(f"  manifest: {r.manifest_path}")
    print(f"  elapsed:  {r.elapsed_s:.1f}s")
    return 0


def _cmd_sample_images(args: argparse.Namespace) -> int:
    try:
        r = sample_images(
            curated_parquet=args.parquet,
            output_dir=args.out,
            k=args.k,
            mode="symlink" if args.symlink else "copy",
            seed=args.seed,
            stratify_by=args.stratify_by,
            image_path_col=args.image_path_col,
            sample_id_col=args.sample_id_col,
            dataset_col=args.dataset_col,
            force=args.force,
            workers=args.workers,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Sampled {r.n_sampled} images from {r.source_parquet}")
    print(f"  mode: {r.mode}   seed: {r.seed}   k: {r.k_requested}")
    if r.stratify_by:
        print(f"  stratified by: {r.stratify_by}")
    print(f"  exported ok: {r.n_exported}   missing: {r.n_missing}   failed: {r.n_failed}")
    print(f"  images dir:  {r.images_dir}")
    print(f"  manifest:    {r.manifest_parquet}")
    print(f"  summary:     {r.manifest_json}")
    print(f"  elapsed:     {r.elapsed_s:.1f}s")
    return 0 if r.n_failed == 0 else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m dagspaces.common.curation")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "scaffolding-permits",
        help="Build a sub-dataset of NYC DOB scaffold/shed permits + 80ft building buffer.",
    )
    sp.add_argument("--out", required=True, help="Output directory")
    sp.add_argument("--cutoff", default="2025-12-31",
                    help="Issue-date upper bound (YYYY-MM-DD). Default: 2025-12-31")
    sp.add_argument("--since", default=None,
                    help="Optional issue-date lower bound (YYYY-MM-DD). "
                         "Default: no lower bound (BIS goes back to 1990s).")
    sp.add_argument("--buffer-ft", type=float, default=80.0,
                    help="Buffer distance in feet. Default: 80")
    sp.add_argument("--buildings",
                    default="/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet",
                    help="Path to nyc_buildings.parquet")
    sp.add_argument("--refresh", action="store_true",
                    help="Ignore Socrata caches and re-fetch")
    sp.add_argument("--bin-match-threshold", type=float, default=0.85,
                    help="Polygon match rate below this raises a warn. Default: 0.85")
    sp.add_argument("--nearest-max-ft", type=float, default=200.0,
                    help="Max distance (ft) for nearest-building fallback when BIN not "
                         "in nyc_buildings.parquet. Beyond this, rows fall back to point. Default: 200")
    sp.set_defaults(func=_cmd_scaffolding_permits)

    mc = sub.add_parser(
        "materialize-cyclomedia",
        help="Query the Cyclomedia catalog against a curation coverage.geojson "
             "and write a curated parquet. Submit as a SLURM job — the spatial "
             "join is heavy for an interactive session.",
    )
    mc.add_argument("--curation-root", required=True,
                    help="Curation dir containing coverage.geojson")
    mc.add_argument("--catalog-root", default="/share/ju/cyclomedia/catalog/v1",
                    help="Cyclomedia catalog root")
    mc.add_argument("--faces", nargs="+", default=list(DEFAULT_FACES),
                    help=f"Face subset of F,B,L,R,U,D. Default: {' '.join(DEFAULT_FACES)}")
    mc.add_argument("--datasets", nargs="+", default=None,
                    help="Restrict to these dataset names. Default: all available.")
    mc.add_argument("--output-filename", default="cyclomedia_near_permits.parquet",
                    help="Parquet filename inside --curation-root")
    mc.add_argument("--columns", nargs="+", default=None,
                    help="Restrict to these catalog columns. Default: keep all.")
    mc.add_argument("--drop-chunks", action="store_true",
                    help="Delete the per-dataset chunks/ dir after the final concat. "
                         "Default: keep chunks on disk for audit + resumability.")
    mc.add_argument("--no-facing", action="store_true",
                    help="Skip the default facing-filter step. Default: after "
                         "writing the unfiltered parquet, also produce a "
                         "<name>_facing.parquet sibling (rows whose face ray "
                         "intersects coverage). Downstream consumers should "
                         "prefer the _facing parquet for sampling / labeling.")
    mc.add_argument("--facing-ray-length-m", type=float, default=30.0,
                    help="Ray length (meters) for the facing filter. "
                         "Default: 30 (matches artifact_gen/raster.py).")
    mc.add_argument("--facing-bearing-tol-deg", type=float,
                    default=DEFAULT_BEARING_TOL_DEG,
                    help="Angular tolerance (degrees) for the facing filter's "
                         "per-unit bearing check. Drop rows where the face's "
                         "bearing differs from the bearing to the attributed "
                         "unit's centroid by more than this. "
                         f"Default: {DEFAULT_BEARING_TOL_DEG:g}° "
                         "(unit sits within the center 45° of the face's 90° FOV).")
    mc.add_argument("--facing-max-distance-ft", type=float,
                    default=DEFAULT_MAX_DISTANCE_FT,
                    help="Hard cap (US feet) on recording → attributed unit "
                         "centroid distance. Drops long-tail across-plaza / "
                         "large-campus shots (Fix D). Pass a large number to "
                         f"effectively disable. Default: {DEFAULT_MAX_DISTANCE_FT:g}.")
    mc.add_argument("--no-facing-occlusion", action="store_true",
                    help="Skip the facing filter's Fix F occlusion check. Default: "
                         "when a row's LOS from recording to its attributed unit's "
                         "own building is strictly pierced by another NYC building "
                         "(BIN != unit's BIN, both segment endpoints outside the "
                         "blocker), drop that row.")
    mc.add_argument("--facing-buildings-path", default=DEFAULT_BUILDINGS_PATH,
                    help=f"Path to nyc_buildings.parquet used by the Fix F "
                         f"occlusion check. Default: {DEFAULT_BUILDINGS_PATH}")
    mc.add_argument("--units-path", default=None,
                    help="Per-unit buffered-polygon parquet. Default: auto-detect "
                         "<curation_root>/{facilities,permits}.parquet. Materialize "
                         "sjoins against these polygons so every output row carries "
                         "unit_uid/unit_name/unit_dist_ft.")
    mc.set_defaults(func=_cmd_materialize_cyclomedia)

    si = sub.add_parser(
        "sample-images",
        help="Sample K images from a curated parquet to an inspection dir "
             "(copy by default; --symlink for fast local materialization).",
    )
    si.add_argument("--parquet", required=True,
                    help="Path to a curated parquet with an image_path column")
    si.add_argument("--out", required=True,
                    help="Inspection output dir (must be empty unless --force)")
    si.add_argument("-k", type=int, required=True, help="Number of images to sample")
    si.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy (fast, but local-only; "
                         "uses absolute source paths)")
    si.add_argument("--seed", type=int, default=0, help="RNG seed. Default: 0")
    si.add_argument("--stratify-by", default=None,
                    help="Column to stratify by (e.g. dataset, face). Default: no stratification")
    si.add_argument("--image-path-col", default="image_path",
                    help="Column name for image paths. Default: image_path")
    si.add_argument("--sample-id-col", default="sample_id",
                    help="Column name for sample IDs (used in destination filenames). Default: sample_id")
    si.add_argument("--dataset-col", default="dataset",
                    help="Column name for dataset (used as filename prefix). Default: dataset")
    si.add_argument("--force", action="store_true",
                    help="Overwrite a non-empty --out dir")
    si.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel worker count. Default: {DEFAULT_WORKERS}")
    si.set_defaults(func=_cmd_sample_images)

    ff = sub.add_parser(
        "filter-facing",
        help="Filter a curated parquet to rows whose face ray intersects coverage. "
             "Matches the `rays` pattern in dagspaces/artifact_gen/stages/raster.py. "
             "Off by default — enable this step when sample-images is returning "
             "too many faces pointed away from the building.",
    )
    ff.add_argument("--parquet", required=True,
                    help="Input curated parquet. Needs latitude/longitude/bearing/face; "
                         "also unit_uid when --units is provided (per-unit mode).")
    ff.add_argument("--coverage", default=None,
                    help="Coverage GeoJSON (typically curation/.../coverage.geojson). "
                         "Used in legacy dissolved-coverage mode. Optional when "
                         "--units is provided.")
    ff.add_argument("--units", default=None,
                    help="Per-unit parquet (facilities.parquet or permits.parquet) "
                         "keyed by unit_uid. Passing this enables per-unit mode "
                         "(Fix A + Fix C): the ray must hit the row's OWN unit "
                         "polygon AND the face must be within --bearing-tol-deg of "
                         "the bearing to that unit's centroid.")
    ff.add_argument("--bearing-tol-deg", type=float, default=DEFAULT_BEARING_TOL_DEG,
                    help=f"Per-unit mode: angular tolerance (deg) between face "
                         f"bearing and bearing to attributed unit's centroid. "
                         f"Default: {DEFAULT_BEARING_TOL_DEG:.0f}°.")
    ff.add_argument("--max-distance-ft", type=float,
                    default=DEFAULT_MAX_DISTANCE_FT,
                    help=f"Per-unit mode: hard cap (US feet) on recording → "
                         f"unit centroid distance (Fix D). Default: "
                         f"{DEFAULT_MAX_DISTANCE_FT:.0f}. Pass a very large "
                         f"number to effectively disable.")
    ff.add_argument("--out", required=True,
                    help="Output parquet path for the filtered sibling")
    ff.add_argument("--ray-length-m", type=float, default=DEFAULT_RAY_LENGTH_M,
                    help=f"Ray length in meters. Default: {DEFAULT_RAY_LENGTH_M:.0f} "
                         "(matches artifact_gen/raster.py)")
    ff.add_argument("--ray-samples", type=int, default=DEFAULT_RAY_SAMPLES,
                    help=f"Number of points sampled along each forward ray "
                         f"(point-in-polygon approximation of line-in-polygon). "
                         f"Default: {DEFAULT_RAY_SAMPLES}")
    ff.add_argument("--no-occlusion", action="store_true",
                    help="Skip the Fix F occlusion check (LOS from recording → "
                         "library's own building polygon strictly pierced by a "
                         "non-unit NYC building). Default: on.")
    ff.add_argument("--buildings-path", default=DEFAULT_BUILDINGS_PATH,
                    help=f"Path to nyc_buildings.parquet used by the Fix F "
                         f"occlusion check. Default: {DEFAULT_BUILDINGS_PATH}")
    ff.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing --out parquet")
    ff.set_defaults(func=_cmd_filter_facing)

    fb = sub.add_parser(
        "facdb-facilities",
        help="Build a curation sub-dataset from NYC DCP Facilities Database (ji82-xba5). "
             "Filter at any of the 4 hierarchy levels (facdomain → facgroup → facsubgrp → factype); "
             "values validated against the frozen FacDB dictionary (25v2).",
    )
    fb.add_argument("--out", required=True, help="Output directory")
    fb.add_argument("--facdomain", nargs="+", default=None,
                    help="One or more facdomain values. Options: see categorization.json")
    fb.add_argument("--facgroup", nargs="+", default=None,
                    help="One or more facgroup values")
    fb.add_argument("--facsubgrp", nargs="+", default=None,
                    help="One or more facsubgrp values")
    fb.add_argument("--factype", nargs="+", default=None,
                    help="One or more factype values (most granular)")
    fb.add_argument("--buffer-ft", type=float, default=80.0,
                    help="Buffer distance (ft). Default: 80")
    fb.add_argument("--buildings",
                    default="/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet",
                    help="Path to nyc_buildings.parquet")
    fb.add_argument("--refresh", action="store_true",
                    help="Ignore Socrata cache and re-fetch")
    fb.add_argument("--bin-match-threshold", type=float, default=0.75,
                    help="Polygon match rate below this raises a warn. Default: 0.75 "
                         "(FacDB has many park/roadway rows with no BIN; lower than permits' 0.85)")
    fb.add_argument("--nearest-max-ft", type=float, default=200.0,
                    help="Max distance (ft) for nearest-building fallback. Default: 200")
    fb.set_defaults(func=_cmd_facdb_facilities)

    dh = sub.add_parser(
        "dohmh-restaurants",
        help="Build a curation sub-dataset of NYC restaurants from the DOHMH "
             "Restaurant Inspection Results dataset (43nn-pn8j). Used as a "
             "proxy for ALL NYC restaurants. Dedupes the inspection-level "
             "rows to one row per CAMIS (restaurant) carrying the most "
             "recent inspection's grade / score / cuisine.",
    )
    dh.add_argument("--out", required=True, help="Output directory")
    dh.add_argument("--cuisine", nargs="+", default=None,
                    help="One or more cuisine_description values "
                         "(e.g. 'Pizza' 'Mexican'). Validated against the "
                         "frozen DOHMH cuisine vocab; case-insensitive. "
                         "Default: no filter (all cuisines).")
    dh.add_argument("--borough", nargs="+", default=None,
                    help="One or more borough names (Manhattan / Bronx / "
                         "Brooklyn / Queens / 'Staten Island'); aliases "
                         "MN/BX/BK/QN/SI accepted. Default: all boroughs.")
    dh.add_argument("--buffer-ft", type=float, default=80.0,
                    help="Buffer distance (ft). Default: 80")
    dh.add_argument("--buildings",
                    default="/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet",
                    help="Path to nyc_buildings.parquet")
    dh.add_argument("--refresh", action="store_true",
                    help="Ignore Socrata cache and re-fetch")
    dh.add_argument("--bin-match-threshold", type=float, default=0.85,
                    help="Polygon match rate below this raises a warn. "
                         "Default: 0.85 (DOHMH restaurants are almost all "
                         "BIN-matched, like permits — higher than FacDB's 0.75)")
    dh.add_argument("--nearest-max-ft", type=float, default=200.0,
                    help="Max distance (ft) for nearest-building fallback. Default: 200")
    dh.add_argument("--drop-placeholder-only", action="store_true",
                    help="Drop CAMIS that only have placeholder "
                         "inspection_date='1900-01-01' rows (registered but "
                         "never inspected). Default: keep them — every "
                         "registered restaurant counts for the 'all NYC "
                         "restaurants' proxy.")
    dh.set_defaults(func=_cmd_dohmh_restaurants)

    orr = sub.add_parser(
        "open-restaurants",
        help="Build a curation sub-dataset of NYC outdoor-dining licenses from "
             "the DCWP Open Restaurants / Dining Out NYC dataset (fpeh-f7ci). "
             "Each row is a restaurant licensed for Sidewalk or Roadway "
             "outdoor dining; geometry is the restaurant's BIN building polygon "
             "(nearest-building + point fallback) buffered by --buffer-ft to "
             "capture the dining setup out front.",
    )
    orr.add_argument("--out", required=True, help="Output directory")
    orr.add_argument("--license-type", nargs="+", default=None,
                     help="One or more license_type values (Sidewalk / Roadway). "
                          "Validated against the frozen vocab; case-insensitive. "
                          "Default: no filter (both types).")
    orr.add_argument("--borough", nargs="+", default=None,
                     help="One or more borough names (Manhattan / Bronx / "
                          "Brooklyn / Queens / 'Staten Island'); aliases "
                          "MN/BX/BK/QN/SI accepted. Default: all boroughs.")
    orr.add_argument("--buffer-ft", type=float, default=80.0,
                     help="Buffer distance (ft). Default: 80")
    orr.add_argument("--buildings",
                     default="/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet",
                     help="Path to nyc_buildings.parquet")
    orr.add_argument("--refresh", action="store_true",
                     help="Ignore Socrata cache and re-fetch")
    orr.add_argument("--bin-match-threshold", type=float, default=0.85,
                     help="Polygon match rate below this raises a warn. "
                          "Default: 0.85 (licenses carry BIN/BBL — mostly "
                          "BIN-matched, like permits/DOHMH).")
    orr.add_argument("--nearest-max-ft", type=float, default=200.0,
                     help="Max distance (ft) for nearest-building fallback. Default: 200")
    orr.set_defaults(func=_cmd_open_restaurants)

    ar = sub.add_parser(
        "aggregate-restaurants",
        help="Collapse an inspection-level restaurants.parquet (multiple "
             "rows per CAMIS, one per inspection date) to one row per "
             "restaurant — required before 'materialize-cyclomedia' since "
             "the spatial join needs unique unit IDs. Writes a sibling "
             "<input_dir>/restaurants_aggregated.parquet by default.",
    )
    ar.add_argument("--parquet", required=True,
                    help="Input restaurants.parquet from 'dohmh-restaurants' build")
    ar.add_argument("--out", default=None,
                    help="Output parquet path. Default: "
                         "<input_dir>/restaurants_aggregated.parquet")
    ar.add_argument("--overwrite", action="store_true",
                    help="Overwrite an existing output parquet")
    ar.set_defaults(func=_cmd_aggregate_restaurants)

    sw = sub.add_parser(
        "subway-entrances",
        help="Build a curation sub-dataset of NYC subway station "
             "entrances/exits from the MTA Permanent Station Entrances/Exits "
             "dataset (data.ny.gov i9wp-a4ja). Subway entrances are points "
             "(sidewalk stairs, elevators, station houses) — geometry is "
             "the entrance lat/lon buffered by --buffer-ft (no BIN match).",
    )
    sw.add_argument("--out", required=True, help="Output directory")
    sw.add_argument("--entrance-type", nargs="+", default=None,
                    help="One or more entrance_type values (e.g. 'Stair' "
                         "'Elevator' 'Station House'). Validated against "
                         "the frozen MTA vocab; case-insensitive. Default: "
                         "no filter (all entrance types).")
    sw.add_argument("--division", nargs="+", default=None,
                    help="One or more division codes (IRT / IND / BMT / "
                         "SIR / IRT/BMT / IND/BMT). Default: all divisions.")
    sw.add_argument("--borough", nargs="+", default=None,
                    help="One or more borough codes (M / B / Bx / Q / SI) "
                         "or full names (Manhattan / Brooklyn / Bronx / "
                         "Queens / 'Staten Island'). Default: all boroughs.")
    sw.add_argument("--route", nargs="+", default=None,
                    help="One or more subway route IDs (e.g. 'L' '4' 'Q'). "
                         "Matched as whole tokens against the space-"
                         "separated daytime_routes column. Default: all routes.")
    sw.add_argument("--buffer-ft", type=float, default=80.0,
                    help="Buffer distance around each entrance point in feet. "
                         "Default: 80 (matches other curation families)")
    sw.add_argument("--refresh", action="store_true",
                    help="Ignore Socrata cache and re-fetch")
    sw.set_defaults(func=_cmd_subway_entrances)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
