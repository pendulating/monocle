"""Build a StreetGraph from Cyclomedia recording metadata + OSMNX street network."""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .street_graph import Neighbor, StreetGraph


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.cos(lat2_r) * math.sin(dlon)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def _build_knn_graph(
    recordings: pd.DataFrame,
    k_neighbors: int,
    max_radius_m: float,
) -> Dict[str, List[Tuple[str, float, float]]]:
    """Build KNN adjacency from recording coordinates.

    Returns dict: recording_id -> list of (neighbor_id, distance_m, bearing_deg).
    """
    from scipy.spatial import cKDTree

    ids = recordings["recording_id"].values
    lats = recordings["latitude"].values.astype(float)
    lons = recordings["longitude"].values.astype(float)

    # Project to approximate meters for KNN (simple equirectangular)
    mean_lat = np.mean(lats)
    cos_lat = math.cos(math.radians(mean_lat))
    xs = np.radians(lons) * 6_371_000.0 * cos_lat
    ys = np.radians(lats) * 6_371_000.0
    coords_m = np.column_stack([xs, ys])

    tree = cKDTree(coords_m)
    # Query k+1 because the first result is the point itself
    dists, indices = tree.query(coords_m, k=min(k_neighbors + 1, len(ids)))

    adjacency: Dict[str, List[Tuple[str, float, float]]] = {}
    for i in range(len(ids)):
        src_id = str(ids[i])
        neighbors: List[Tuple[str, float, float]] = []
        for j in range(1, dists.shape[1]):  # skip self (index 0)
            nb_idx = indices[i, j]
            dist_m = _haversine_m(lats[i], lons[i], lats[nb_idx], lons[nb_idx])
            if dist_m > max_radius_m:
                continue
            bearing = _bearing_deg(lats[i], lons[i], lats[nb_idx], lons[nb_idx])
            neighbors.append((str(ids[nb_idx]), dist_m, bearing))
        adjacency[src_id] = neighbors

    return adjacency


def _build_osmnx_constrained_graph(
    recordings: pd.DataFrame,
    k_neighbors: int,
    max_radius_m: float,
) -> Dict[str, List[Tuple[str, float, float]]]:
    """Build adjacency constrained to OSMNX street network edges.

    Falls back to pure KNN if osmnx is not available.
    """
    try:
        import osmnx as ox
    except ImportError:
        print("[street_graph] osmnx not available, falling back to KNN-only graph", flush=True)
        return _build_knn_graph(recordings, k_neighbors, max_radius_m)

    lats = recordings["latitude"].values.astype(float)
    lons = recordings["longitude"].values.astype(float)

    # Build OSMNX graph for bounding box with buffer
    buffer = 0.005  # ~500m buffer
    north, south = float(np.max(lats)) + buffer, float(np.min(lats)) - buffer
    east, west = float(np.max(lons)) + buffer, float(np.min(lons)) - buffer

    G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type="drive")

    # Snap each recording to nearest OSMNX edge
    X = lons.tolist()
    Y = lats.tolist()
    nearest_edges = ox.nearest_edges(G, X, Y)

    ids = recordings["recording_id"].values
    # Map each recording to its snapped edge (as frozenset of nodes for undirected comparison)
    rec_to_edge = {}
    edge_to_recs: Dict[Any, List[int]] = {}
    for i, (u, v, _key) in enumerate(nearest_edges):
        edge = frozenset([u, v])
        rec_to_edge[i] = edge
        edge_to_recs.setdefault(edge, []).append(i)

    # Build adjacency of OSMNX edges (edges that share a node)
    edge_adjacency: Dict[Any, set] = {}
    for edge in edge_to_recs:
        edge_adjacency.setdefault(edge, set())
        nodes = list(edge)
        for node in nodes:
            for neighbor in G.neighbors(node):
                for n2 in G.neighbors(node):
                    nb_edge = frozenset([node, n2])
                    if nb_edge != edge and nb_edge in edge_to_recs:
                        edge_adjacency[edge].add(nb_edge)
                nb_edge = frozenset([node, neighbor])
                if nb_edge != edge and nb_edge in edge_to_recs:
                    edge_adjacency[edge].add(nb_edge)

    # For each recording, candidate neighbors are on the same or adjacent OSMNX edges
    adjacency: Dict[str, List[Tuple[str, float, float]]] = {}
    for i in range(len(ids)):
        src_id = str(ids[i])
        src_edge = rec_to_edge[i]
        candidate_indices: set = set()
        # Same edge
        for j in edge_to_recs.get(src_edge, []):
            if j != i:
                candidate_indices.add(j)
        # Adjacent edges
        for adj_edge in edge_adjacency.get(src_edge, set()):
            for j in edge_to_recs.get(adj_edge, []):
                if j != i:
                    candidate_indices.add(j)

        neighbors: List[Tuple[str, float, float]] = []
        for j in candidate_indices:
            dist_m = _haversine_m(lats[i], lons[i], lats[j], lons[j])
            if dist_m > max_radius_m:
                continue
            bearing = _bearing_deg(lats[i], lons[i], lats[j], lons[j])
            neighbors.append((str(ids[j]), dist_m, bearing))

        # Sort by distance, keep top k
        neighbors.sort(key=lambda x: x[1])
        adjacency[src_id] = neighbors[:k_neighbors]

    return adjacency


def build_street_graph(
    metadata_parquet: str,
    graph_cfg: Any,
    use_osmnx: bool = True,
) -> StreetGraph:
    """Build a StreetGraph from recording metadata.

    Args:
        metadata_parquet: Path to parquet with recording_id, latitude, longitude, yaw_deg columns.
        graph_cfg: Config object with k_neighbors, max_radius_m, precomputed_path.
        use_osmnx: Whether to use OSMNX for street-network-constrained adjacency.
    """
    # Dispatch by graph_type
    graph_type = str(getattr(graph_cfg, "graph_type", "knn"))
    if graph_type == "osm":
        return build_osm_projected_graph(metadata_parquet, graph_cfg)
    if graph_type == "h3":
        return build_h3_graph(metadata_parquet, graph_cfg)
    if graph_type == "intersection":
        return build_intersection_graph(metadata_parquet, graph_cfg)

    precomputed_path: Optional[str] = getattr(graph_cfg, "precomputed_path", None)
    if precomputed_path and os.path.exists(precomputed_path):
        print(f"[street_graph] Loading precomputed graph from {precomputed_path}", flush=True)
        with open(precomputed_path, "rb") as f:
            return pickle.load(f)

    k_neighbors = int(getattr(graph_cfg, "k_neighbors", 10))
    max_radius_m = float(getattr(graph_cfg, "max_radius_m", 50.0))

    print(f"[street_graph] Loading metadata from {metadata_parquet}", flush=True)
    meta_df = pd.read_parquet(metadata_parquet)

    # Normalize coordinate column names.
    # Handle cases where canonical column exists but is empty (all None)
    # while the alias column has actual data.
    col_aliases = {
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "yawDegrees": "yaw_deg",
        "yaw": "yaw_deg",
    }
    rename_map = {}
    for alias, canonical in col_aliases.items():
        if alias not in meta_df.columns:
            continue
        if canonical not in meta_df.columns:
            rename_map[alias] = canonical
        elif meta_df[canonical].isna().all() and not meta_df[alias].isna().all():
            # Canonical column exists but is empty; use alias instead
            meta_df = meta_df.drop(columns=[canonical])
            rename_map[alias] = canonical
    if rename_map:
        meta_df = meta_df.rename(columns=rename_map)
        print(f"[street_graph] Normalized columns: {rename_map}", flush=True)

    # Deduplicate to unique recordings
    required_cols = ["recording_id", "latitude", "longitude"]
    for col in required_cols:
        if col not in meta_df.columns:
            raise ValueError(f"Metadata parquet missing required column: {col}")

    # Get unique recordings with their coords and yaw
    has_yaw = "yaw_deg" in meta_df.columns
    if not has_yaw:
        print("[street_graph] WARNING: no yaw column found — all yaw values will be 0. "
              "Face-to-neighbor resolution will be unreliable.", flush=True)
    group_cols = ["recording_id", "latitude", "longitude"]
    if has_yaw:
        group_cols.append("yaw_deg")

    recordings = meta_df.drop_duplicates(subset=["recording_id"])[group_cols].reset_index(drop=True)

    # Drop recordings with NaN/inf coordinates
    n_before = len(recordings)
    recordings["latitude"] = pd.to_numeric(recordings["latitude"], errors="coerce")
    recordings["longitude"] = pd.to_numeric(recordings["longitude"], errors="coerce")
    coord_mask = (
        recordings["latitude"].notna()
        & recordings["longitude"].notna()
        & np.isfinite(recordings["latitude"].values.astype(float))
        & np.isfinite(recordings["longitude"].values.astype(float))
    )
    recordings = recordings[coord_mask].reset_index(drop=True)
    n_dropped = n_before - len(recordings)
    if n_dropped > 0:
        print(f"[street_graph] Dropped {n_dropped} recordings with NaN/inf coordinates", flush=True)

    print(f"[street_graph] Building graph for {len(recordings)} unique recordings (k={k_neighbors}, radius={max_radius_m}m)", flush=True)

    if use_osmnx:
        raw_adj = _build_osmnx_constrained_graph(recordings, k_neighbors, max_radius_m)
    else:
        raw_adj = _build_knn_graph(recordings, k_neighbors, max_radius_m)

    # Build StreetGraph
    adjacency: Dict[str, List[Neighbor]] = {}
    for src_id, nbs in raw_adj.items():
        adjacency[src_id] = [
            Neighbor(recording_id=nb_id, distance_m=dist, bearing_deg=bearing)
            for nb_id, dist, bearing in nbs
        ]

    coords: Dict[str, Tuple[float, float]] = {}
    yaw_degrees: Dict[str, float] = {}
    for _, row in recordings.iterrows():
        rid = str(row["recording_id"])
        coords[rid] = (float(row["latitude"]), float(row["longitude"]))
        if has_yaw:
            yaw_degrees[rid] = float(row["yaw_deg"])
        else:
            yaw_degrees[rid] = 0.0

    graph = StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees)

    if precomputed_path:
        os.makedirs(os.path.dirname(precomputed_path), exist_ok=True)
        print(f"[street_graph] Saving precomputed graph to {precomputed_path}", flush=True)
        with open(precomputed_path, "wb") as f:
            pickle.dump(graph, f)

    print(f"[street_graph] Graph built: {len(adjacency)} nodes, "
          f"avg {np.mean([len(v) for v in adjacency.values()]):.1f} neighbors", flush=True)
    return graph


# ---------------------------------------------------------------------------
# OSM-backed intersection graph builder
# ---------------------------------------------------------------------------

def _normalize_yaw_from_recorder_direction(meta_df: pd.DataFrame, yaw_column: str = "recorderDirection") -> np.ndarray:
    """Extract reliable yaw in [0, 360) from recorderDirection.

    Falls back to yawDegrees with radians-detection heuristic if
    recorderDirection is unavailable.
    """
    if yaw_column in meta_df.columns:
        raw = pd.to_numeric(meta_df[yaw_column], errors="coerce").fillna(0.0).values
        return raw % 360.0

    # Fallback: try yawDegrees with unit detection
    for col in ("yaw_deg", "yawDegrees"):
        if col in meta_df.columns:
            raw = pd.to_numeric(meta_df[col], errors="coerce").fillna(0.0).values
            # Values in [0, 2*pi) are radians; values ~20600 are rad * (180/pi)
            deg_per_rad = 180.0 / math.pi
            yaw = np.where(raw < 7.0, raw * deg_per_rad, raw / deg_per_rad)
            print(f"[street_graph] WARNING: using {col} with heuristic unit fix", flush=True)
            return yaw % 360.0

    print("[street_graph] WARNING: no heading column found, all yaw = 0", flush=True)
    return np.zeros(len(meta_df))


def _project_to_meters(lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection to approximate meters."""
    mean_lat = np.mean(lats)
    cos_lat = math.cos(math.radians(mean_lat))
    xs = np.radians(lons) * 6_371_000.0 * cos_lat
    ys = np.radians(lats) * 6_371_000.0
    return xs, ys


# ---------------------------------------------------------------------------
# OSM projection-based graph builder (recommended)
# ---------------------------------------------------------------------------

def _get_edge_geom(G: Any, u: int, v: int, k: int) -> Any:
    """Get the LineString geometry for an OSM edge."""
    from shapely.geometry import LineString
    d = G.edges[u, v, k]
    if "geometry" in d:
        return d["geometry"]
    return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                       (G.nodes[v]["x"], G.nodes[v]["y"])])


def build_osm_projected_graph(
    metadata_parquet: str,
    graph_cfg: Any,
) -> StreetGraph:
    """Build a street graph by projecting recordings onto OSM edge geometry.

    Pipeline:
    1. Load OSM drive network for the area
    2. Snap each recording to its nearest OSM edge
    3. Project recordings onto edge geometry to get position along the edge
    4. Subsample at regular intervals along each edge (one random pick per bin)
    5. Connect sequential recordings along each edge
    6. Connect edge tips at shared OSM nodes (intersections)
    """
    import osmnx as ox
    from shapely.geometry import Point
    from collections import defaultdict

    precomputed_path: Optional[str] = getattr(graph_cfg, "precomputed_path", None)
    if precomputed_path and os.path.exists(precomputed_path):
        print(f"[osm_graph] Loading precomputed graph from {precomputed_path}", flush=True)
        with open(precomputed_path, "rb") as f:
            return pickle.load(f)

    yaw_column = str(getattr(graph_cfg, "yaw_column", "recorderDirection"))
    target_spacing = float(getattr(graph_cfg, "target_spacing_m", 25.0))
    subsample_seed = int(getattr(graph_cfg, "subsample_seed", 42))
    osm_place = str(getattr(graph_cfg, "osm_place", "Manhattan, New York, USA"))
    osm_graphml = str(getattr(graph_cfg, "osm_graphml", ""))
    osm_network_type = str(getattr(graph_cfg, "osm_network_type", "drive"))

    # --- Load & normalize recordings ---
    print(f"[osm_graph] Loading metadata from {metadata_parquet}", flush=True)
    meta_df = pd.read_parquet(metadata_parquet)

    col_aliases = {"lat": "latitude", "lon": "longitude", "lng": "longitude"}
    for alias, canonical in col_aliases.items():
        if alias not in meta_df.columns:
            continue
        if canonical not in meta_df.columns:
            meta_df = meta_df.rename(columns={alias: canonical})
        elif meta_df[canonical].isna().all() and not meta_df[alias].isna().all():
            meta_df = meta_df.drop(columns=[canonical])
            meta_df = meta_df.rename(columns={alias: canonical})

    for col in ("recording_id", "latitude", "longitude"):
        if col not in meta_df.columns:
            raise ValueError(f"Metadata parquet missing required column: {col}")

    recs = meta_df.drop_duplicates(subset=["recording_id"]).reset_index(drop=True)
    recs["latitude"] = pd.to_numeric(recs["latitude"], errors="coerce")
    recs["longitude"] = pd.to_numeric(recs["longitude"], errors="coerce")
    recs = recs[recs["latitude"].notna() & recs["longitude"].notna()].reset_index(drop=True)

    yaw_all = _normalize_yaw_from_recorder_direction(recs, yaw_column)
    lats = recs["latitude"].values.astype(float)
    lons = recs["longitude"].values.astype(float)
    rec_ids_all = recs["recording_id"].values.astype(str)
    print(f"[osm_graph] {len(recs):,} unique recordings", flush=True)

    # --- Load OSM network ---
    if osm_graphml and os.path.exists(osm_graphml):
        print(f"[osm_graph] Loading OSM graph from {osm_graphml}", flush=True)
        G = ox.load_graphml(osm_graphml)
    else:
        print(f"[osm_graph] Fetching OSM {osm_network_type} network for {osm_place}...", flush=True)
        G = ox.graph_from_place(osm_place, network_type=osm_network_type)
        if osm_graphml:
            os.makedirs(os.path.dirname(osm_graphml) or ".", exist_ok=True)
            ox.save_graphml(G, osm_graphml)
    print(f"[osm_graph] OSM: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    # --- Heading-aware snap to nearest edges ---
    # Pure geometric snapping (ox.nearest_edges) assigns ~35% of recordings
    # to cross-street edges when they're at intersections. We use a combined
    # distance + heading score to prefer edges aligned with the recording's
    # driving direction.
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    print("[osm_graph] Building heading-aware edge index...", flush=True)
    edge_geom_list: List[Any] = []
    edge_key_list: List[Tuple] = []
    edge_bearing_list: List[float] = []

    for u, v, k, d in G.edges(keys=True, data=True):
        geom = _get_edge_geom(G, u, v, k)
        edge_geom_list.append(geom)
        edge_key_list.append((u, v, k))
        coords = list(geom.coords)
        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        edge_bearing_list.append(math.degrees(math.atan2(dx, dy)) % 360)

    edge_tree = STRtree(edge_geom_list)
    edge_bearing_arr = np.array(edge_bearing_list)
    search_radius = 0.001  # ~100m in degrees

    print("[osm_graph] Snapping recordings to OSM edges (heading-aware)...", flush=True)
    nearest_edges: List[Tuple] = []
    for i in range(len(recs)):
        pt = Point(lons[i], lats[i])
        yaw = yaw_all[i]

        candidates = edge_tree.query(pt.buffer(search_radius))
        if len(candidates) == 0:
            candidates = edge_tree.query(pt.buffer(search_radius * 5))
        if len(candidates) == 0:
            # Absolute fallback: nearest edge by pure distance
            candidates = edge_tree.query(pt.buffer(0.01))

        best_idx = candidates[0] if len(candidates) > 0 else 0
        best_score = float("inf")
        for idx in candidates:
            dist = edge_geom_list[idx].distance(pt)
            brg = edge_bearing_arr[idx]
            hdiff = abs(brg - yaw) % 360
            hdiff = min(hdiff, 360 - hdiff)
            hdiff = min(hdiff, abs(hdiff - 180))  # bidirectional
            heading_penalty = (hdiff / 90.0) * 0.0005
            score = dist + heading_penalty
            if score < best_score:
                best_score = score
                best_idx = idx

        nearest_edges.append(edge_key_list[best_idx])

    edge_to_recs: Dict[Tuple, List[int]] = defaultdict(list)
    for i, edge in enumerate(nearest_edges):
        edge_to_recs[edge].append(i)
    print(f"[osm_graph] Recordings on {len(edge_to_recs)}/{G.number_of_edges()} edges", flush=True)

    # --- Project onto edge geometry and subsample ---
    rng = np.random.default_rng(subsample_seed)
    selected: List[int] = []

    for (u, v, key), rec_indices in edge_to_recs.items():
        geom = _get_edge_geom(G, u, v, key)
        edge_len = float(G.edges[u, v, key].get("length", 50))

        # Project each recording onto the edge's LineString
        projections = sorted(
            (geom.project(Point(lons[i], lats[i]), normalized=True), i)
            for i in rec_indices
        )

        # Divide edge into equal bins, pick one recording per bin
        n_bins = max(1, int(edge_len / target_spacing))
        bin_edges = np.linspace(0, 1, n_bins + 1)
        pi = 0
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            bin_recs: List[int] = []
            while pi < len(projections) and projections[pi][0] <= hi:
                if projections[pi][0] >= lo:
                    bin_recs.append(projections[pi][1])
                pi += 1
            if bin_recs:
                selected.append(int(rng.choice(bin_recs)))

    print(f"[osm_graph] Subsampled: {len(recs):,} -> {len(selected):,} "
          f"({100 * len(selected) / len(recs):.1f}%)", flush=True)

    sub_rids = rec_ids_all[selected]
    sub_lats = lats[selected]
    sub_lons = lons[selected]
    sub_yaws = yaw_all[selected]

    # Edge -> local subsampled indices
    edge_to_local: Dict[Tuple, List[int]] = defaultdict(list)
    for li, gi in enumerate(selected):
        edge_to_local[nearest_edges[gi]].append(li)

    # --- Along-edge connections (sequential along geometry) ---
    adjacency_raw: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)

    for (u, v, key), local_indices in edge_to_local.items():
        if len(local_indices) <= 1:
            continue
        geom = _get_edge_geom(G, u, v, key)
        projs = sorted(
            (geom.project(Point(sub_lons[li], sub_lats[li]), normalized=True), li)
            for li in local_indices
        )
        for j in range(len(projs) - 1):
            _, li_a = projs[j]
            _, li_b = projs[j + 1]
            rid_a, rid_b = sub_rids[li_a], sub_rids[li_b]
            d = _haversine_m(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
            b_ab = _bearing_deg(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
            b_ba = _bearing_deg(sub_lats[li_b], sub_lons[li_b], sub_lats[li_a], sub_lons[li_a])
            adjacency_raw[rid_a].append((rid_b, d, b_ab))
            adjacency_raw[rid_b].append((rid_a, d, b_ba))

    n_along = sum(len(v) for v in adjacency_raw.values())

    # --- Intersection connections via shared OSM nodes ---
    # For each edge, find the recording closest to each endpoint.
    # Group by OSM node; connect tips pairwise at each node.
    node_tips: Dict[int, Dict[Tuple, int]] = defaultdict(dict)

    for (u, v, key), local_indices in edge_to_local.items():
        if not local_indices:
            continue
        u_pt = Point(G.nodes[u]["x"], G.nodes[u]["y"])
        v_pt = Point(G.nodes[v]["x"], G.nodes[v]["y"])

        best_u: Optional[int] = None
        best_u_d = float("inf")
        best_v: Optional[int] = None
        best_v_d = float("inf")
        for li in local_indices:
            pt = Point(sub_lons[li], sub_lats[li])
            du = pt.distance(u_pt)
            dv = pt.distance(v_pt)
            if du < best_u_d:
                best_u, best_u_d = li, du
            if dv < best_v_d:
                best_v, best_v_d = li, dv

        if best_u is not None:
            node_tips[u][(u, v, key)] = best_u
        if best_v is not None:
            node_tips[v][(u, v, key)] = best_v

    n_intersection = 0
    for osm_node, edge_tip_dict in node_tips.items():
        tips = list(edge_tip_dict.values())
        if len(tips) < 2:
            continue
        for a in range(len(tips)):
            for b in range(a + 1, len(tips)):
                li_a, li_b = tips[a], tips[b]
                rid_a, rid_b = sub_rids[li_a], sub_rids[li_b]
                if rid_a == rid_b:
                    continue
                existing = {nb[0] for nb in adjacency_raw.get(rid_a, [])}
                if rid_b in existing:
                    continue
                d = _haversine_m(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
                b_ab = _bearing_deg(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
                b_ba = _bearing_deg(sub_lats[li_b], sub_lons[li_b], sub_lats[li_a], sub_lons[li_a])
                adjacency_raw[rid_a].append((rid_b, d, b_ab))
                adjacency_raw[rid_b].append((rid_a, d, b_ba))
                n_intersection += 2

    # Ensure all selected recordings have an adjacency entry
    for li in range(len(sub_rids)):
        if sub_rids[li] not in adjacency_raw:
            adjacency_raw[sub_rids[li]] = []

    # --- Assemble StreetGraph ---
    adjacency: Dict[str, List[Neighbor]] = {}
    for rid, nbs in adjacency_raw.items():
        adjacency[rid] = [
            Neighbor(recording_id=nb_id, distance_m=dist, bearing_deg=brg)
            for nb_id, dist, brg in nbs
        ]

    coords: Dict[str, Tuple[float, float]] = {}
    yaw_degrees_map: Dict[str, float] = {}
    for i in range(len(sub_rids)):
        rid = sub_rids[i]
        coords[rid] = (float(sub_lats[i]), float(sub_lons[i]))
        yaw_degrees_map[rid] = float(sub_yaws[i])

    graph = StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees_map)

    total_edges = sum(len(v) for v in adjacency.values())
    avg_deg = total_edges / max(1, len(adjacency))
    n_ix_nodes = sum(1 for v in adjacency.values() if len(v) > 2)
    all_dists = [nb.distance_m for nbs in adjacency.values() for nb in nbs]
    med_dist = float(np.median(all_dists)) if all_dists else 0.0

    print(
        f"[osm_graph] Built: {len(adjacency):,} nodes, {total_edges:,} edges "
        f"(avg deg {avg_deg:.1f}), {n_along} along-edge + {n_intersection} intersection, "
        f"median edge {med_dist:.0f}m, {n_ix_nodes:,} intersection nodes",
        flush=True,
    )

    if precomputed_path:
        os.makedirs(os.path.dirname(precomputed_path) or ".", exist_ok=True)
        print(f"[osm_graph] Saving to {precomputed_path}", flush=True)
        with open(precomputed_path, "wb") as f:
            pickle.dump(graph, f)

    return graph


def build_h3_graph(
    metadata_parquet: str,
    graph_cfg: Any,
) -> StreetGraph:
    """Build a street graph using an H3 hexagonal spatial index.

    Each occupied H3 cell becomes a node (one randomly selected recording).
    Edges connect cells that share a hex boundary (k-ring 1 adjacency).
    Because recordings exist only on streets, block interiors are empty,
    and the hex grid naturally produces a connected street topology with
    no cross-block shortcuts.
    """
    import h3

    precomputed_path: Optional[str] = getattr(graph_cfg, "precomputed_path", None)
    if precomputed_path and os.path.exists(precomputed_path):
        print(f"[h3_graph] Loading precomputed graph from {precomputed_path}", flush=True)
        with open(precomputed_path, "rb") as f:
            return pickle.load(f)

    resolution = int(getattr(graph_cfg, "h3_resolution", 12))
    yaw_column = str(getattr(graph_cfg, "yaw_column", "recorderDirection"))
    subsample_seed = int(getattr(graph_cfg, "subsample_seed", 42))

    # --- Load & normalize ---
    print(f"[h3_graph] Loading metadata from {metadata_parquet}", flush=True)
    meta_df = pd.read_parquet(metadata_parquet)

    # Column normalization
    col_aliases = {"lat": "latitude", "lon": "longitude", "lng": "longitude"}
    for alias, canonical in col_aliases.items():
        if alias not in meta_df.columns:
            continue
        if canonical not in meta_df.columns:
            meta_df = meta_df.rename(columns={alias: canonical})
        elif meta_df[canonical].isna().all() and not meta_df[alias].isna().all():
            meta_df = meta_df.drop(columns=[canonical])
            meta_df = meta_df.rename(columns={alias: canonical})

    for col in ("recording_id", "latitude", "longitude"):
        if col not in meta_df.columns:
            raise ValueError(f"Metadata parquet missing required column: {col}")

    recs = meta_df.drop_duplicates(subset=["recording_id"]).reset_index(drop=True)
    recs["latitude"] = pd.to_numeric(recs["latitude"], errors="coerce")
    recs["longitude"] = pd.to_numeric(recs["longitude"], errors="coerce")
    recs = recs[recs["latitude"].notna() & recs["longitude"].notna()].reset_index(drop=True)

    yaw_all = _normalize_yaw_from_recorder_direction(recs, yaw_column)
    lats = recs["latitude"].values.astype(float)
    lons = recs["longitude"].values.astype(float)
    rec_ids_all = recs["recording_id"].values.astype(str)
    print(f"[h3_graph] {len(recs):,} unique recordings, resolution {resolution}", flush=True)

    # --- Assign H3 cells ---
    h3_cells = np.array([
        h3.latlng_to_cell(lats[i], lons[i], resolution)
        for i in range(len(recs))
    ])

    # --- One recording per cell (random selection) ---
    rng = np.random.default_rng(subsample_seed)
    cell_to_indices: Dict[str, List[int]] = {}
    for i, cell in enumerate(h3_cells):
        cell_to_indices.setdefault(cell, []).append(i)

    cell_to_selected: Dict[str, int] = {}
    for cell, indices in cell_to_indices.items():
        cell_to_selected[cell] = int(rng.choice(indices))

    occupied = set(cell_to_selected.keys())
    n_cells = len(occupied)
    print(f"[h3_graph] Occupied cells: {n_cells:,} (from {len(recs):,} recordings)", flush=True)

    # --- Build adjacency from hex neighbors (two passes) ---
    # Pass 1: collect all candidate edges, classifying each as
    #   along-road, lateral (cross-lane), or intersection.
    # Pass 2: drop lateral edges only when the node already has
    #   at least one along-road edge (preserving connectivity).
    CandidateEdge = Tuple[str, float, float, bool]  # (nb_rid, dist, bearing, is_lateral)
    raw_adj: Dict[str, List[CandidateEdge]] = {}

    for cell, idx in cell_to_selected.items():
        rid = rec_ids_all[idx]
        lat_i, lon_i = lats[idx], lons[idx]
        yaw_i = yaw_all[idx]
        candidates: List[CandidateEdge] = []

        for nb_cell in h3.grid_disk(cell, 1):
            if nb_cell == cell or nb_cell not in occupied:
                continue
            nb_idx = cell_to_selected[nb_cell]
            nb_rid = rec_ids_all[nb_idx]
            yaw_j = yaw_all[nb_idx]
            dist = _haversine_m(lat_i, lon_i, lats[nb_idx], lons[nb_idx])
            brg = _bearing_deg(lat_i, lon_i, lats[nb_idx], lons[nb_idx])

            # Classify edge
            yaw_diff = abs(yaw_i - yaw_j) % 360
            yaw_diff = min(yaw_diff, 360 - yaw_diff)
            same_road = yaw_diff < 45

            is_lateral = False
            if same_road:
                brg_vs_yaw = abs(brg - yaw_i) % 360
                brg_vs_yaw = min(brg_vs_yaw, 360 - brg_vs_yaw)
                along_road = brg_vs_yaw < 60 or abs(brg_vs_yaw - 180) < 60
                if not along_road:
                    is_lateral = True

            candidates.append((nb_rid, dist, brg, is_lateral))

        raw_adj[rid] = candidates

    # Pass 2: build final adjacency, dropping lateral edges when safe
    adjacency: Dict[str, List[Neighbor]] = {}
    n_edges = 0
    n_lateral_dropped = 0

    for rid, candidates in raw_adj.items():
        has_non_lateral = any(not c[3] for c in candidates)
        neighbors: List[Neighbor] = []
        for nb_rid, dist, brg, is_lateral in candidates:
            if is_lateral and has_non_lateral:
                n_lateral_dropped += 1
                continue
            neighbors.append(Neighbor(recording_id=nb_rid, distance_m=dist, bearing_deg=brg))
        adjacency[rid] = neighbors
        n_edges += len(neighbors)

    print(f"[h3_graph] Dropped {n_lateral_dropped:,} lateral cross-lane edges", flush=True)

    # --- Build coords and yaw dicts ---
    coords: Dict[str, Tuple[float, float]] = {}
    yaw_degrees: Dict[str, float] = {}
    for cell, idx in cell_to_selected.items():
        rid = rec_ids_all[idx]
        coords[rid] = (float(lats[idx]), float(lons[idx]))
        yaw_degrees[rid] = float(yaw_all[idx])

    graph = StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees)

    # --- Diagnostics ---
    degrees = [len(v) for v in adjacency.values()]
    avg_deg = n_edges / max(1, n_cells)
    n_isolated = sum(1 for d in degrees if d == 0)
    n_intersection = sum(1 for d in degrees if d > 2)

    all_dists = [nb.distance_m for nbs in adjacency.values() for nb in nbs]
    median_dist = float(np.median(all_dists)) if all_dists else 0.0

    print(
        f"[h3_graph] Built: {n_cells:,} nodes, {n_edges:,} edges "
        f"(avg degree {avg_deg:.1f}, median edge {median_dist:.0f}m), "
        f"{n_intersection:,} intersection nodes, {n_isolated} isolated",
        flush=True,
    )

    if precomputed_path:
        os.makedirs(os.path.dirname(precomputed_path) or ".", exist_ok=True)
        print(f"[h3_graph] Saving to {precomputed_path}", flush=True)
        with open(precomputed_path, "wb") as f:
            pickle.dump(graph, f)

    return graph


def build_intersection_graph(
    metadata_parquet: str,
    graph_cfg: Any,
) -> StreetGraph:
    """Build a street graph using the OSM road network as backbone.

    Pipeline:
    1. Load metadata, normalize yaw from recorderDirection
    2. Fetch the OSM drive network for the area (or load cached graphml)
    3. Snap each recording to its nearest OSM edge
    4. Subsample to ~target_spacing_m per edge (random selection)
    5. Build along-edge sequential connections
    6. Build intersection connections via shared OSM nodes
    """
    import osmnx as ox
    from collections import defaultdict

    precomputed_path: Optional[str] = getattr(graph_cfg, "precomputed_path", None)
    if precomputed_path and os.path.exists(precomputed_path):
        print(f"[intersection_graph] Loading precomputed graph from {precomputed_path}", flush=True)
        with open(precomputed_path, "rb") as f:
            return pickle.load(f)

    # --- Config ---
    yaw_column = str(getattr(graph_cfg, "yaw_column", "recorderDirection"))
    target_spacing = float(getattr(graph_cfg, "target_spacing_m", 25.0))
    subsample_seed = int(getattr(graph_cfg, "subsample_seed", 42))
    osm_place = str(getattr(graph_cfg, "osm_place", "Manhattan, New York, USA"))
    osm_graphml = str(getattr(graph_cfg, "osm_graphml", ""))
    osm_network_type = str(getattr(graph_cfg, "osm_network_type", "drive"))

    # --- Load & normalize recordings ---
    print(f"[intersection_graph] Loading metadata from {metadata_parquet}", flush=True)
    meta_df = pd.read_parquet(metadata_parquet)

    # Column normalization
    col_aliases = {"lat": "latitude", "lon": "longitude", "lng": "longitude"}
    for alias, canonical in col_aliases.items():
        if alias not in meta_df.columns:
            continue
        if canonical not in meta_df.columns:
            meta_df = meta_df.rename(columns={alias: canonical})
        elif meta_df[canonical].isna().all() and not meta_df[alias].isna().all():
            meta_df = meta_df.drop(columns=[canonical])
            meta_df = meta_df.rename(columns={alias: canonical})

    for col in ("recording_id", "latitude", "longitude"):
        if col not in meta_df.columns:
            raise ValueError(f"Metadata parquet missing required column: {col}")

    recs = meta_df.drop_duplicates(subset=["recording_id"]).reset_index(drop=True)
    recs["latitude"] = pd.to_numeric(recs["latitude"], errors="coerce")
    recs["longitude"] = pd.to_numeric(recs["longitude"], errors="coerce")
    recs = recs[recs["latitude"].notna() & recs["longitude"].notna()].reset_index(drop=True)

    yaw_deg = _normalize_yaw_from_recorder_direction(recs, yaw_column)
    lats = recs["latitude"].values.astype(float)
    lons = recs["longitude"].values.astype(float)
    rec_ids_all = recs["recording_id"].values.astype(str)
    print(f"[intersection_graph] {len(recs)} unique recordings with valid coords", flush=True)

    # --- Load OSM network ---
    if osm_graphml and os.path.exists(osm_graphml):
        print(f"[intersection_graph] Loading OSM graph from {osm_graphml}", flush=True)
        G = ox.load_graphml(osm_graphml)
    else:
        print(f"[intersection_graph] Fetching OSM {osm_network_type} network for {osm_place}...", flush=True)
        G = ox.graph_from_place(osm_place, network_type=osm_network_type)
        # Cache for next time
        if osm_graphml:
            os.makedirs(os.path.dirname(osm_graphml) or ".", exist_ok=True)
            ox.save_graphml(G, osm_graphml)
            print(f"[intersection_graph] Cached OSM graph to {osm_graphml}", flush=True)
    print(f"[intersection_graph] OSM network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    # --- Snap recordings to nearest edges ---
    print("[intersection_graph] Snapping recordings to OSM edges...", flush=True)
    nearest_edges = ox.nearest_edges(G, lons.tolist(), lats.tolist())

    edge_to_recs: Dict[Tuple, List[int]] = defaultdict(list)
    for i, (u, v, key) in enumerate(nearest_edges):
        edge_to_recs[(u, v, key)].append(i)
    print(f"[intersection_graph] Recordings snapped to {len(edge_to_recs)}/{G.number_of_edges()} edges", flush=True)

    # --- Subsample per edge (evenly spaced, random within each bin) ---
    rng = np.random.default_rng(subsample_seed)
    selected: List[int] = []
    for (u, v, key), rec_indices in edge_to_recs.items():
        edge_len = float(G.edges[u, v, key].get("length", 50))
        n_keep = max(1, int(edge_len / target_spacing))
        if len(rec_indices) <= n_keep:
            selected.extend(rec_indices)
            continue

        # Sort recordings by projection along the edge direction
        u_data, v_data = G.nodes[u], G.nodes[v]
        u_lat, u_lon = float(u_data["y"]), float(u_data["x"])
        v_lat, v_lon = float(v_data["y"]), float(v_data["x"])
        cos_lat_local = math.cos(math.radians((u_lat + v_lat) / 2))
        edge_dx = (v_lon - u_lon) * cos_lat_local
        edge_dy = v_lat - u_lat

        projs = []
        for gi in rec_indices:
            dx = (lons[gi] - u_lon) * cos_lat_local
            dy = lats[gi] - u_lat
            projs.append(dx * edge_dx + dy * edge_dy)

        order = np.argsort(projs)
        sorted_indices = [rec_indices[o] for o in order]

        # Divide into n_keep bins and pick one randomly from each
        bin_size = len(sorted_indices) / n_keep
        for b in range(n_keep):
            start = int(b * bin_size)
            end = int((b + 1) * bin_size)
            if start >= len(sorted_indices):
                break
            end = min(end, len(sorted_indices))
            chosen = rng.choice(sorted_indices[start:end])
            selected.append(chosen)

    print(f"[intersection_graph] Subsampled: {len(recs)} -> {len(selected)} "
          f"({100 * len(selected) / len(recs):.1f}%)", flush=True)

    sub_rids = rec_ids_all[selected]
    sub_lats = lats[selected]
    sub_lons = lons[selected]
    sub_yaws = yaw_deg[selected]
    sub_edges = [nearest_edges[i] for i in selected]

    # Local index -> edge, edge -> local indices
    edge_to_local: Dict[Tuple, List[int]] = defaultdict(list)
    for li, gi in enumerate(selected):
        edge_to_local[nearest_edges[gi]].append(li)

    # --- Build along-edge edges ---
    adjacency_raw: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)

    for (u, v, key), local_indices in edge_to_local.items():
        if len(local_indices) <= 1:
            continue
        # Project onto edge direction from OSM node coordinates
        u_data, v_data = G.nodes[u], G.nodes[v]
        u_lat, u_lon = float(u_data["y"]), float(u_data["x"])
        v_lat, v_lon = float(v_data["y"]), float(v_data["x"])
        cos_lat = math.cos(math.radians((u_lat + v_lat) / 2))
        edge_dx = (v_lon - u_lon) * cos_lat
        edge_dy = v_lat - u_lat

        projs = []
        for li in local_indices:
            dx = (sub_lons[li] - u_lon) * cos_lat
            dy = sub_lats[li] - u_lat
            proj = dx * edge_dx + dy * edge_dy
            projs.append((proj, li))
        projs.sort()

        for k in range(len(projs) - 1):
            _, li_a = projs[k]
            _, li_b = projs[k + 1]
            rid_a, rid_b = sub_rids[li_a], sub_rids[li_b]
            d = _haversine_m(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
            b_ab = _bearing_deg(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
            b_ba = _bearing_deg(sub_lats[li_b], sub_lons[li_b], sub_lats[li_a], sub_lons[li_a])
            adjacency_raw[rid_a].append((rid_b, d, b_ab))
            adjacency_raw[rid_b].append((rid_a, d, b_ba))

    n_along = sum(len(v) for v in adjacency_raw.values())

    # --- Build intersection edges via shared OSM nodes ---
    # For each OSM node, find the single closest recording from each
    # incident edge. Then connect those tips pairwise.
    # This gives exactly one tip per edge per node, avoiding the mesh
    # of cross-connections that occurs when multiple recordings per edge
    # all get linked.
    node_tip_map: Dict[int, Dict[Tuple, int]] = defaultdict(dict)  # osm_node -> {edge: local_idx}

    for (u, v, key), local_indices in edge_to_local.items():
        if not local_indices:
            continue
        u_data, v_data = G.nodes[u], G.nodes[v]
        u_lat, u_lon = float(u_data["y"]), float(u_data["x"])
        v_lat, v_lon = float(v_data["y"]), float(v_data["x"])

        # Find recording closest to u
        best_u: Optional[int] = None
        best_u_d = float("inf")
        for li in local_indices:
            du = (sub_lats[li] - u_lat) ** 2 + (sub_lons[li] - u_lon) ** 2
            if du < best_u_d:
                best_u, best_u_d = li, du
        if best_u is not None:
            node_tip_map[u][(u, v, key)] = best_u

        # Find recording closest to v
        best_v: Optional[int] = None
        best_v_d = float("inf")
        for li in local_indices:
            dv = (sub_lats[li] - v_lat) ** 2 + (sub_lons[li] - v_lon) ** 2
            if dv < best_v_d:
                best_v, best_v_d = li, dv
        if best_v is not None:
            node_tip_map[v][(u, v, key)] = best_v

    # Connect tips at each OSM node. Each edge contributes exactly one
    # tip per node (already enforced by node_tip_map). All tips at a
    # node are connected pairwise — this creates the intersection edges.
    n_intersection = 0
    for osm_node, edge_tip_dict in node_tip_map.items():
        tips = list(edge_tip_dict.values())  # one per edge
        if len(tips) < 2:
            continue
        for a in range(len(tips)):
            for b in range(a + 1, len(tips)):
                li_a, li_b = tips[a], tips[b]
                rid_a, rid_b = sub_rids[li_a], sub_rids[li_b]
                if rid_a == rid_b:
                    continue
                existing = {nb[0] for nb in adjacency_raw.get(rid_a, [])}
                if rid_b in existing:
                    continue
                d = _haversine_m(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
                b_ab = _bearing_deg(sub_lats[li_a], sub_lons[li_a], sub_lats[li_b], sub_lons[li_b])
                b_ba = _bearing_deg(sub_lats[li_b], sub_lons[li_b], sub_lats[li_a], sub_lons[li_a])
                adjacency_raw[rid_a].append((rid_b, d, b_ab))
                adjacency_raw[rid_b].append((rid_a, d, b_ba))
                n_intersection += 2

    # Ensure all selected rids have an entry
    for li in range(len(sub_rids)):
        if sub_rids[li] not in adjacency_raw:
            adjacency_raw[sub_rids[li]] = []

    # --- Assemble StreetGraph ---
    adjacency: Dict[str, List[Neighbor]] = {}
    for rid, nbs in adjacency_raw.items():
        adjacency[rid] = [
            Neighbor(recording_id=nb_id, distance_m=dist, bearing_deg=brg)
            for nb_id, dist, brg in nbs
        ]

    coords: Dict[str, Tuple[float, float]] = {}
    yaw_degrees_map: Dict[str, float] = {}
    for i in range(len(sub_rids)):
        rid = sub_rids[i]
        coords[rid] = (float(sub_lats[i]), float(sub_lons[i]))
        yaw_degrees_map[rid] = float(sub_yaws[i])

    graph = StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees_map)

    total_edges = sum(len(v) for v in adjacency.values())
    avg_deg = total_edges / max(1, len(adjacency))
    n_intersect_nodes = sum(1 for v in adjacency.values() if len(v) > 2)
    print(
        f"[intersection_graph] Built: {len(adjacency)} nodes, {total_edges} edges "
        f"(avg degree {avg_deg:.1f}), {n_along} along-edge + {n_intersection} intersection, "
        f"{n_intersect_nodes} intersection nodes (degree > 2)",
        flush=True,
    )

    if precomputed_path:
        os.makedirs(os.path.dirname(precomputed_path) or ".", exist_ok=True)
        print(f"[intersection_graph] Saving to {precomputed_path}", flush=True)
        with open(precomputed_path, "wb") as f:
            pickle.dump(graph, f)

    return graph
