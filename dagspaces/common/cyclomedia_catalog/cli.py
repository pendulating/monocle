"""CLI for the Cyclomedia catalog.

Subcommands:
  build       — full rebuild from raw_root
  rejoin-wfs  — re-run the WFS join against existing partitions (no re-walk)
  validate    — run the 11-check validation on an existing catalog
  query       — materialize a query to parquet

Run as: `python -m dagspaces.common.cyclomedia_catalog.cli <subcommand> ...`
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .catalog import CyclomediaCatalog, DEFAULT_CATALOG_ROOT
from .indexer import build_catalog, rejoin_wfs
from .wfs import DEFAULT_CATALOG_GLOB


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_build(args: argparse.Namespace) -> int:
    result = build_catalog(
        raw_root=args.raw_root,
        output_root=args.output,
        datasets=args.datasets or None,
        catalog_globs=args.catalog_csv_glob or DEFAULT_CATALOG_GLOB,
        fd_path=args.fd_path,
    )
    print(f"\nBuilt catalog at {result.output_root}")
    print(f"  datasets: {len(result.datasets)}")
    print(f"  total rows: {result.total_rows:,}")
    print(f"  elapsed: {result.elapsed_s:.1f}s")
    if result.validation_summary_path:
        print(f"  summary: {result.validation_summary_path}")
    return 0


def _cmd_rejoin_wfs(args: argparse.Namespace) -> int:
    result = rejoin_wfs(
        output_root=args.output,
        datasets=args.datasets,
        catalog_globs=args.catalog_csv_glob or DEFAULT_CATALOG_GLOB,
        raw_root=args.raw_root,
    )
    print(f"\nRejoined WFS at {result.output_root}")
    print(f"  datasets: {list(result.row_counts.keys())}")
    print(f"  rows rewritten: {result.total_rows:,}")
    print(f"  elapsed: {result.elapsed_s:.1f}s")
    if result.validation_summary_path:
        print(f"  summary: {result.validation_summary_path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    import polars as pl
    from .validation import run_validation

    cat = CyclomediaCatalog(root=args.output)
    df = cat.scan().collect()
    # raw_root is not stored; best effort: use manifest.json
    raw_root = cat.manifest().get("raw_root", "/share/ju/cyclomedia/raw")
    path = run_validation(df, output_root=args.output, raw_root=raw_root)
    print(f"summary: {path}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    cat = CyclomediaCatalog(root=args.catalog)

    within = None
    if args.within:
        import geopandas as gpd
        within = gpd.read_file(args.within)

    between: Optional[tuple[str, str]] = None
    if args.between:
        between = (args.between[0], args.between[1])

    faces = None
    if args.faces:
        faces = {f.strip().upper() for f in args.faces.split(",") if f.strip()}

    datasets = args.datasets or None

    df = cat.build_inference_parquet(
        output_path=args.output_path,
        within=within,
        between=between,
        faces=faces,
        datasets=datasets,
    )
    print(f"\nwrote {df.height:,} rows to {args.output_path}")
    print(f"  columns: {len(df.columns)}")
    print(df.head(5))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dagspaces.common.cyclomedia_catalog.cli",
        description="Cyclomedia catalog build + query CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    p_build = sub.add_parser("build", help="Full rebuild of the catalog")
    p_build.add_argument("--raw-root", default="/share/ju/cyclomedia/raw")
    p_build.add_argument("--output", default=DEFAULT_CATALOG_ROOT)
    p_build.add_argument("--datasets", nargs="*", default=None,
                         help="Subset of dataset dir names. Default: all top-level dirs under --raw-root.")
    p_build.add_argument("--catalog-csv-glob", nargs="*", default=None,
                         help="Glob patterns for WFS catalog CSVs. Default: /share/ju/cyclomedia/pull/**/recordings_*.csv")
    p_build.add_argument("--fd-path", default=None,
                         help="Absolute path to fd binary (overrides PATH search). Default: auto.")
    p_build.set_defaults(func=_cmd_build)

    # rejoin-wfs
    p_rj = sub.add_parser(
        "rejoin-wfs",
        help="Re-run the WFS join against existing partitions without re-walking",
    )
    p_rj.add_argument("--output", default=DEFAULT_CATALOG_ROOT)
    p_rj.add_argument("--datasets", nargs="+", required=True,
                      help="Dataset names under --output/by_dataset to rejoin.")
    p_rj.add_argument("--catalog-csv-glob", nargs="*", default=None,
                      help="Glob patterns for WFS catalog CSVs. Default: built-in set.")
    p_rj.add_argument("--raw-root", default="/share/ju/cyclomedia/raw",
                      help="Only used for the path-inside-raw_root validation check.")
    p_rj.set_defaults(func=_cmd_rejoin_wfs)

    # validate
    p_val = sub.add_parser("validate", help="Run validation on an existing catalog")
    p_val.add_argument("--output", default=DEFAULT_CATALOG_ROOT)
    p_val.set_defaults(func=_cmd_validate)

    # query
    p_q = sub.add_parser("query", help="Materialize a query to parquet")
    p_q.add_argument("--catalog", default=DEFAULT_CATALOG_ROOT)
    p_q.add_argument("--output-path", required=True)
    p_q.add_argument("--within", default=None, help="Path to a GeoJSON/shapefile with polygon(s).")
    p_q.add_argument("--between", nargs=2, metavar=("START", "END"),
                     help="recordedAt window, ISO strings (e.g. 2025-05-01 2025-08-01)")
    p_q.add_argument("--faces", default=None, help="Comma-separated face letters. Default: all.")
    p_q.add_argument("--datasets", nargs="*", default=None, help="Restrict to these datasets.")
    p_q.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
