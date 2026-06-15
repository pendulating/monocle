"""Build (or rebuild) the urbanroamvqa board-first street graph.

Composes the same graph config the pipelines use (conf/graph/board_25m.yaml +
metadata_parquet), so the config-fingerprinted cache written here is the one
pipeline runs will load.

Usage:
    python scripts/build_roam_board_graph.py [--graph-config board_25m] \
        [--parquet /path/to/metadata.parquet] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from omegaconf import OmegaConf  # noqa: E402

CONF_DIR = os.path.join(REPO_ROOT, "dagspaces", "urbanroamvqa", "conf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-config", default="board_25m",
                        help="Name of a conf/graph/*.yaml config (default: board_25m)")
    parser.add_argument("--parquet", default=None,
                        help="Metadata parquet (default: data config's parquet_path)")
    parser.add_argument("--force", action="store_true",
                        help="Delete the existing cache before building")
    args = parser.parse_args()

    graph_cfg = OmegaConf.load(os.path.join(CONF_DIR, "graph", f"{args.graph_config}.yaml"))

    if args.parquet:
        parquet = os.path.abspath(args.parquet)
    else:
        data_cfg = OmegaConf.load(os.path.join(CONF_DIR, "data", "cyclomedia_manhattan_2025.yaml"))
        parquet = str(data_cfg.parquet_path)
    # Mirror the runtime graph node: config.yaml injects metadata_parquet
    graph_cfg.metadata_parquet = parquet

    precomputed = graph_cfg.get("precomputed_path")
    if args.force and precomputed and os.path.exists(str(precomputed)):
        print(f"--force: removing {precomputed}", flush=True)
        os.remove(str(precomputed))
    if precomputed:
        os.makedirs(os.path.dirname(str(precomputed)), exist_ok=True)
    osm_graphml = str(OmegaConf.select(graph_cfg, "osm_graphml") or "")
    if osm_graphml:
        os.makedirs(os.path.dirname(osm_graphml), exist_ok=True)

    from dagspaces.urbanroamvqa.graph.builder import build_street_graph
    from dagspaces.urbanroamvqa.graph.street_graph import compute_graph_diagnostics

    graph = build_street_graph(parquet, graph_cfg)
    diag = compute_graph_diagnostics(graph)
    print(json.dumps(diag, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
