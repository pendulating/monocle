"""Board-first street graph builder (graph_type: board).

Inverts the legacy construction order: the board comes from the OSM street
network FIRST — uniform node spacing, consolidated intersections, single
connected component by construction — and imagery is attached to board
positions afterwards. The legacy builders did the opposite (nodes = image
coordinates, edges inferred), which inherited every coverage gap and
duplicate-lane artifact of the capture vehicle.

Pipeline:
1. Load the OSM network (place query or cached graphml), project, consolidate
   nearby intersections (merges divided roads / dual carriageways), convert to
   undirected, keep the largest connected component.
2. Discretize every edge geometry at a fixed target spacing: board nodes are
   OSM junctions plus interpolated points along each street.
3. Attach recordings: globally-greedy nearest assignment (distance plus a
   heading-alignment penalty) of at most one recording per board node, within
   a snap radius.
4. Contract imageless board nodes: neighbors are reconnected through them
   with summed street distance, so connectivity survives coverage gaps while
   every playable position has imagery. Reconnections longer than
   max_contracted_edge_m are not created (prevents teleport edges across
   large unimaged regions); the board is then re-trimmed to its largest
   component.
5. QA gate: the final graph must be a single connected component, and must
   retain at least min_main_component_frac of the image-attached nodes.

The resulting StreetGraph is keyed by recording_id (so the roaming stage's
image lookup works unchanged), but coordinates are the BOARD positions and
neighbor bearings follow the street geometry, not the raw capture points.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import cfg_fingerprint, load_cached_graph, save_cached_graph
from .street_graph import Neighbor, StreetGraph, compute_graph_diagnostics

TAG = "[board_graph]"

# Internal board node ids: ("j", osm_node) for junctions,
# ("e", u, v, key, i) for interpolated points along an edge.
BoardNode = Tuple


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon = lon2_r - lon1_r
    x = math.cos(lat2_r) * math.sin(dlon)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


# ---------------------------------------------------------------------------
# Step 1: OSM backbone
# ---------------------------------------------------------------------------

def _load_osm_backbone(graph_cfg: Any) -> Any:
    """Load the OSM network and prepare it as an undirected, consolidated,
    fully connected lat/lon MultiGraph."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError(
            f"{TAG} osmnx is required for the board graph builder. "
            "Install it or choose a different graph_type."
        ) from exc

    osm_place = str(getattr(graph_cfg, "osm_place", "Manhattan, New York, USA"))
    osm_graphml = str(getattr(graph_cfg, "osm_graphml", "") or "")
    network_type = str(getattr(graph_cfg, "osm_network_type", "drive"))
    consolidate_tolerance_m = float(getattr(graph_cfg, "consolidate_tolerance_m", 15.0))

    if osm_graphml and os.path.exists(osm_graphml):
        print(f"{TAG} Loading OSM graph from {osm_graphml}", flush=True)
        G = ox.load_graphml(osm_graphml)
    else:
        print(f"{TAG} Fetching OSM {network_type} network for {osm_place}...", flush=True)
        G = ox.graph_from_place(osm_place, network_type=network_type)
        if osm_graphml:
            os.makedirs(os.path.dirname(osm_graphml) or ".", exist_ok=True)
            ox.save_graphml(G, osm_graphml)
            print(f"{TAG} Cached OSM graph to {osm_graphml}", flush=True)
    print(f"{TAG} OSM raw: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges", flush=True)

    if consolidate_tolerance_m > 0:
        Gp = ox.projection.project_graph(G)
        Gp = ox.simplification.consolidate_intersections(
            Gp, tolerance=consolidate_tolerance_m, rebuild_graph=True, dead_ends=True,
        )
        G = ox.projection.project_graph(Gp, to_latlong=True)
        print(
            f"{TAG} Consolidated intersections (tol={consolidate_tolerance_m}m): "
            f"{G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges",
            flush=True,
        )

    import networkx as nx

    G = ox.convert.to_undirected(G)
    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    print(
        f"{TAG} Undirected largest component: {G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges",
        flush=True,
    )
    return G


# ---------------------------------------------------------------------------
# Step 2: discretize edges into uniform board nodes
# ---------------------------------------------------------------------------

def _discretize_edges(G: Any, spacing_m: float) -> Any:
    """Turn an OSM MultiGraph into a simple board graph with ~spacing_m pitch.

    Nodes carry lat/lon attrs; edges carry "dist" (meters along the street).
    Parallel OSM edges shorter than ~1.5x spacing collapse into one board
    edge (shortest distance wins).
    """
    import networkx as nx
    from shapely.geometry import LineString

    board = nx.Graph()
    for node, data in G.nodes(data=True):
        board.add_node(("j", node), lat=float(data["y"]), lon=float(data["x"]))

    for u, v, key, data in G.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([
                (G.nodes[u]["x"], G.nodes[u]["y"]),
                (G.nodes[v]["x"], G.nodes[v]["y"]),
            ])
        else:
            # to_undirected leaves geometry direction arbitrary relative to
            # the (u, v) order that edges() yields; interpolation must be
            # anchored at u or the chain's junction hops span the whole edge.
            c0 = geom.coords[0]
            d0u = _haversine_m(c0[1], c0[0], G.nodes[u]["y"], G.nodes[u]["x"])
            d0v = _haversine_m(c0[1], c0[0], G.nodes[v]["y"], G.nodes[v]["x"])
            if d0v < d0u:
                geom = geom.reverse()
        length = float(data.get("length", 0.0)) or geom.length * 111_000.0
        n_seg = max(1, int(round(length / spacing_m)))
        if u == v and n_seg < 2:
            continue  # degenerate self-loop shorter than the board pitch

        chain: List[BoardNode] = [("j", u)]
        for i in range(1, n_seg):
            pt = geom.interpolate(i / n_seg, normalized=True)
            nid: BoardNode = ("e", u, v, key, i)
            board.add_node(nid, lat=float(pt.y), lon=float(pt.x))
            chain.append(nid)
        chain.append(("j", v))

        for a, b in zip(chain[:-1], chain[1:]):
            if a == b:
                continue
            d = _haversine_m(
                board.nodes[a]["lat"], board.nodes[a]["lon"],
                board.nodes[b]["lat"], board.nodes[b]["lon"],
            )
            if board.has_edge(a, b):
                board[a][b]["dist"] = min(board[a][b]["dist"], d)
            else:
                board.add_edge(a, b, dist=d)

    return board


# ---------------------------------------------------------------------------
# Step 3: attach recordings to board nodes
# ---------------------------------------------------------------------------

def _attach_recordings(
    board: Any,
    rec_lats: np.ndarray,
    rec_lons: np.ndarray,
    rec_yaws: np.ndarray,
    max_snap_dist_m: float,
    heading_penalty_m: float = 10.0,
    k_candidates: int = 6,
) -> Dict[BoardNode, int]:
    """Globally-greedy unique assignment of recordings to board nodes.

    Candidate pairs are scored by distance plus a heading-misalignment
    penalty (a recording whose yaw is parallel to one of the node's incident
    streets gives cleaner face-to-street mapping). Each recording is used at
    most once; each node gets at most one recording.
    """
    from scipy.spatial import cKDTree

    nodes = list(board.nodes)
    if len(rec_lats) == 0 or not nodes:
        return {}

    mean_lat = float(np.mean(rec_lats))
    cos_lat = math.cos(math.radians(mean_lat))

    def _project(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        xs = np.radians(lons) * 6_371_000.0 * cos_lat
        ys = np.radians(lats) * 6_371_000.0
        return np.column_stack([xs, ys])

    rec_xy = _project(rec_lats, rec_lons)
    node_lats = np.array([board.nodes[n]["lat"] for n in nodes])
    node_lons = np.array([board.nodes[n]["lon"] for n in nodes])
    node_xy = _project(node_lats, node_lons)

    tree = cKDTree(rec_xy)
    k = min(k_candidates, len(rec_lats))
    dists, idxs = tree.query(node_xy, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    candidates: List[Tuple[float, float, int, int]] = []
    for ni, node in enumerate(nodes):
        incident_bearings = [
            _bearing_deg(node_lats[ni], node_lons[ni],
                         board.nodes[nb]["lat"], board.nodes[nb]["lon"])
            for nb in board.neighbors(node)
        ]
        for j in range(k):
            d = float(dists[ni, j])
            if not math.isfinite(d) or d > max_snap_dist_m:
                continue
            ri = int(idxs[ni, j])
            penalty = 0.0
            if incident_bearings and heading_penalty_m > 0:
                yaw = float(rec_yaws[ri])
                mis = 90.0
                for brg in incident_bearings:
                    diff = abs(yaw - brg) % 360.0
                    diff = min(diff, 360.0 - diff)
                    diff = min(diff, 180.0 - diff)  # street alignment is bidirectional
                    mis = min(mis, diff)
                penalty = (mis / 90.0) * heading_penalty_m
            candidates.append((d + penalty, d, ni, ri))

    candidates.sort(key=lambda c: c[0])
    assigned: Dict[BoardNode, int] = {}
    used_recs: set = set()
    taken_nodes: set = set()
    for _score, _d, ni, ri in candidates:
        if ni in taken_nodes or ri in used_recs:
            continue
        assigned[nodes[ni]] = ri
        taken_nodes.add(ni)
        used_recs.add(ri)
    return assigned


# ---------------------------------------------------------------------------
# Step 4: contract imageless board nodes
# ---------------------------------------------------------------------------

def _contract_imageless(
    board: Any,
    assigned: Dict[BoardNode, int],
    max_contracted_edge_m: float,
) -> int:
    """Remove board nodes with no attached recording, reconnecting their
    neighbors pairwise with summed street distance.

    Reconnections longer than max_contracted_edge_m (when > 0) are skipped —
    a step on the board should never teleport across a large unimaged area.
    Returns the number of contracted nodes. Mutates board in place.
    """
    imageless = [n for n in board.nodes if n not in assigned]
    for node in imageless:
        nbrs = list(board.neighbors(node))
        weights = {nb: board[node][nb]["dist"] for nb in nbrs}
        board.remove_node(node)
        for ai in range(len(nbrs)):
            for bi in range(ai + 1, len(nbrs)):
                a, b = nbrs[ai], nbrs[bi]
                d = weights[a] + weights[b]
                if max_contracted_edge_m > 0 and d > max_contracted_edge_m:
                    continue
                if board.has_edge(a, b):
                    board[a][b]["dist"] = min(board[a][b]["dist"], d)
                else:
                    board.add_edge(a, b, dist=d)
    return len(imageless)


# ---------------------------------------------------------------------------
# Recording metadata loading
# ---------------------------------------------------------------------------

def _load_recordings(metadata_parquet: str, yaw_column: str) -> pd.DataFrame:
    """Load unique recordings with normalized latitude/longitude/yaw_deg."""
    from .builder import _normalize_yaw_from_recorder_direction

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
            raise ValueError(f"{TAG} Metadata parquet missing required column: {col}")

    recs = meta_df.drop_duplicates(subset=["recording_id"]).reset_index(drop=True)
    recs["latitude"] = pd.to_numeric(recs["latitude"], errors="coerce")
    recs["longitude"] = pd.to_numeric(recs["longitude"], errors="coerce")
    recs = recs[recs["latitude"].notna() & recs["longitude"].notna()].reset_index(drop=True)
    recs["yaw_deg"] = _normalize_yaw_from_recorder_direction(recs, yaw_column)
    return recs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_board_graph(metadata_parquet: str, graph_cfg: Any) -> StreetGraph:
    """Build a board-first StreetGraph. See module docstring for the pipeline."""
    fingerprint = cfg_fingerprint(graph_cfg, metadata_parquet)
    precomputed_path: Optional[str] = getattr(graph_cfg, "precomputed_path", None)
    cached = load_cached_graph(precomputed_path, fingerprint, TAG)
    if cached is not None:
        return cached

    spacing_m = float(getattr(graph_cfg, "spacing_m", 25.0))
    max_snap_dist_m = float(getattr(graph_cfg, "max_snap_dist_m", 30.0))
    heading_penalty_m = float(getattr(graph_cfg, "heading_penalty_m", 10.0))
    max_contracted_edge_m = float(getattr(graph_cfg, "max_contracted_edge_m", 4 * spacing_m))
    min_main_component_frac = float(getattr(graph_cfg, "min_main_component_frac", 0.9))
    yaw_column = str(getattr(graph_cfg, "yaw_column", "recorderDirection"))

    print(f"{TAG} Loading recordings from {metadata_parquet}", flush=True)
    recs = _load_recordings(metadata_parquet, yaw_column)
    rec_ids = recs["recording_id"].values.astype(str)
    rec_lats = recs["latitude"].values.astype(float)
    rec_lons = recs["longitude"].values.astype(float)
    rec_yaws = recs["yaw_deg"].values.astype(float)
    print(f"{TAG} {len(recs):,} unique recordings", flush=True)

    G = _load_osm_backbone(graph_cfg)
    board = _discretize_edges(G, spacing_m)
    n_board_total = board.number_of_nodes()
    print(
        f"{TAG} Discretized board: {n_board_total:,} nodes, "
        f"{board.number_of_edges():,} edges at ~{spacing_m:.0f}m pitch",
        flush=True,
    )

    assigned = _attach_recordings(
        board, rec_lats, rec_lons, rec_yaws, max_snap_dist_m, heading_penalty_m,
    )
    coverage = len(assigned) / max(1, n_board_total)
    print(
        f"{TAG} Imagery attached to {len(assigned):,}/{n_board_total:,} board nodes "
        f"({100 * coverage:.1f}%, snap radius {max_snap_dist_m:.0f}m)",
        flush=True,
    )

    n_contracted = _contract_imageless(board, assigned, max_contracted_edge_m)
    if n_contracted:
        print(
            f"{TAG} Contracted {n_contracted:,} imageless nodes "
            f"(max reconnect {max_contracted_edge_m:.0f}m)",
            flush=True,
        )

    # Contraction with a reconnect limit can fragment: keep largest component.
    import networkx as nx

    components = sorted(nx.connected_components(board), key=len, reverse=True)
    n_attached = board.number_of_nodes()
    if len(components) > 1:
        main = components[0]
        dropped = n_attached - len(main)
        board = board.subgraph(main).copy()
        print(
            f"{TAG} Trimmed {len(components) - 1} disconnected fragments "
            f"({dropped:,} nodes, {100 * dropped / max(1, n_attached):.1f}% of attached)",
            flush=True,
        )

    main_frac = board.number_of_nodes() / max(1, n_attached)
    if main_frac < min_main_component_frac:
        raise RuntimeError(
            f"{TAG} QA gate failed: largest component holds only "
            f"{100 * main_frac:.1f}% of image-attached board nodes "
            f"(min_main_component_frac={min_main_component_frac}). Imagery coverage "
            f"is too fragmented for this OSM extent — widen max_snap_dist_m / "
            f"max_contracted_edge_m, or narrow osm_place to the covered area."
        )

    # --- Assemble StreetGraph keyed by recording_id, positioned at board nodes ---
    adjacency: Dict[str, List[Neighbor]] = {}
    coords: Dict[str, Tuple[float, float]] = {}
    yaw_degrees: Dict[str, float] = {}
    node_rid: Dict[BoardNode, str] = {n: rec_ids[assigned[n]] for n in board.nodes}

    for node in board.nodes:
        rid = node_rid[node]
        lat, lon = board.nodes[node]["lat"], board.nodes[node]["lon"]
        coords[rid] = (float(lat), float(lon))
        yaw_degrees[rid] = float(rec_yaws[assigned[node]])
        neighbors: List[Neighbor] = []
        for nb in board.neighbors(node):
            nb_lat, nb_lon = board.nodes[nb]["lat"], board.nodes[nb]["lon"]
            neighbors.append(Neighbor(
                recording_id=node_rid[nb],
                distance_m=float(board[node][nb]["dist"]),
                bearing_deg=_bearing_deg(lat, lon, nb_lat, nb_lon),
            ))
        adjacency[rid] = neighbors

    graph = StreetGraph(adjacency=adjacency, coords=coords, yaw_degrees=yaw_degrees,
                        face_frame=str(getattr(graph_cfg, "face_frame", "absolute")))

    # --- QA gate ---
    diag = compute_graph_diagnostics(graph)
    n_components = int(diag.get("graph/n_components", 0))
    if n_components != 1:
        raise RuntimeError(
            f"{TAG} QA gate failed: final graph has {n_components} connected "
            f"components (expected exactly 1)."
        )
    print(
        f"{TAG} Built board: {len(adjacency):,} nodes, "
        f"{int(diag['graph/n_edges']):,} edges, 1 component, "
        f"degree mean {diag['graph/degree_mean']:.2f}, "
        f"pitch median {diag.get('graph/edge_m_median', 0):.0f}m "
        f"(p10 {diag.get('graph/edge_m_p10', 0):.0f}m / p90 {diag.get('graph/edge_m_p90', 0):.0f}m), "
        f"{int(diag['graph/n_dead_ends'])} dead ends, "
        f"{int(diag['graph/n_intersections'])} intersections",
        flush=True,
    )

    save_cached_graph(precomputed_path, graph, fingerprint, TAG)
    return graph
