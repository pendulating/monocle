"""Walk seed selection for roaming VQA."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..graph.street_graph import StreetGraph


def sample_walk_seeds(
    graph: StreetGraph,
    n_walks: int,
    seed: int,
    strategy: str = "random",
    initial_face: str = "F",
    min_neighbors: int = 1,
    manual_seeds: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Select starting recordings for walks.

    Args:
        graph: The street graph.
        n_walks: Number of walks to seed.
        seed: Random seed.
        strategy: "random", "spatial_stratified", or "manual".
        initial_face: Starting face direction (typically "F").
        min_neighbors: Minimum neighbor count for eligible seeds.
        manual_seeds: Explicit recording_ids for "manual" strategy.

    Returns:
        DataFrame with columns: walk_id, seed_recording_id, seed_face, lat, lon.
    """
    if strategy == "manual":
        if not manual_seeds:
            raise ValueError("manual strategy requires manual_seeds list")
        rows = []
        for i, rid in enumerate(manual_seeds):
            if rid not in graph.coords:
                raise ValueError(f"Manual seed recording_id not in graph: {rid}")
            lat, lon = graph.coords[rid]
            rows.append({
                "walk_id": f"walk_{i:06d}",
                "seed_recording_id": rid,
                "seed_face": initial_face,
                "lat": lat,
                "lon": lon,
            })
        return pd.DataFrame(rows)

    # Filter to recordings with enough neighbors
    eligible = [
        rid for rid, nbs in graph.adjacency.items()
        if len(nbs) >= min_neighbors
    ]
    if not eligible:
        raise ValueError(f"No recordings with >= {min_neighbors} neighbors")

    rng = np.random.default_rng(seed)

    if strategy == "spatial_stratified":
        return _spatial_stratified(graph, eligible, n_walks, rng, initial_face)

    # Default: random
    selected = rng.choice(eligible, size=min(n_walks, len(eligible)), replace=n_walks > len(eligible))
    rows = []
    for i, rid in enumerate(selected):
        lat, lon = graph.coords[rid]
        rows.append({
            "walk_id": f"walk_{i:06d}",
            "seed_recording_id": str(rid),
            "seed_face": initial_face,
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(rows)


def _spatial_stratified(
    graph: StreetGraph,
    eligible: list,
    n_walks: int,
    rng: Any,
    initial_face: str,
    grid_size: int = 10,
) -> pd.DataFrame:
    """Grid the bounding box, sample proportionally per cell."""
    lats = np.array([graph.coords[r][0] for r in eligible])
    lons = np.array([graph.coords[r][1] for r in eligible])

    lat_bins = np.linspace(lats.min(), lats.max() + 1e-9, grid_size + 1)
    lon_bins = np.linspace(lons.min(), lons.max() + 1e-9, grid_size + 1)

    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons, lon_bins) - 1

    # Group by cell
    cells: dict = {}
    for i, rid in enumerate(eligible):
        cell = (int(lat_idx[i]), int(lon_idx[i]))
        cells.setdefault(cell, []).append(rid)

    # Sample proportionally
    total = len(eligible)
    rows = []
    walk_idx = 0
    for cell_key, cell_rids in cells.items():
        cell_n = max(1, int(round(n_walks * len(cell_rids) / total)))
        if walk_idx >= n_walks:
            break
        cell_n = min(cell_n, n_walks - walk_idx)
        selected = rng.choice(cell_rids, size=min(cell_n, len(cell_rids)), replace=cell_n > len(cell_rids))
        for rid in selected:
            lat, lon = graph.coords[rid]
            rows.append({
                "walk_id": f"walk_{walk_idx:06d}",
                "seed_recording_id": str(rid),
                "seed_face": initial_face,
                "lat": lat,
                "lon": lon,
            })
            walk_idx += 1

    # Fill remaining if proportional rounding left gaps
    while walk_idx < n_walks:
        rid = rng.choice(eligible)
        lat, lon = graph.coords[rid]
        rows.append({
            "walk_id": f"walk_{walk_idx:06d}",
            "seed_recording_id": str(rid),
            "seed_face": initial_face,
            "lat": lat,
            "lon": lon,
        })
        walk_idx += 1

    return pd.DataFrame(rows)
