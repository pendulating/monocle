"""StreetGraph dataclass for recording-level street network adjacency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Face bearings (degrees). Interpretation depends on StreetGraph.face_frame:
#   "absolute" — bearings are compass-fixed: F=North, R=East, B=South, L=West.
#       Cyclomedia NYC cube faces are rendered in this globally-oriented frame
#       (camera orientation is ~0° on 100% of NYC rows; verified 2026-04-22,
#       see dagspaces.common.cyclomedia_catalog.indexer._compute_bearing).
#   "relative" — bearings are offsets from the recording yaw (vehicle heading),
#       for datasets whose panoramas are vehicle-oriented.
FACE_BEARING_DEG: Dict[str, float] = {
    "F": 0.0,
    "R": 90.0,
    "B": 180.0,
    "L": 270.0,
}

HORIZONTAL_FACES = ("F", "R", "B", "L")


def _normalize_bearing(deg: float) -> float:
    """Normalize bearing to [0, 360)."""
    return deg % 360.0


def _bearing_diff(a: float, b: float) -> float:
    """Smallest angular difference between two bearings (0-180)."""
    diff = abs(_normalize_bearing(a) - _normalize_bearing(b))
    return min(diff, 360.0 - diff)


def _face_for_bearing(bearing_deg: float, yaw_deg: float) -> str:
    """Return the face whose absolute bearing is closest to bearing_deg."""
    best_face = "F"
    best_diff = 999.0
    for face, offset in FACE_BEARING_DEG.items():
        abs_bearing = _normalize_bearing(yaw_deg + offset)
        diff = _bearing_diff(abs_bearing, bearing_deg)
        if diff < best_diff:
            best_diff = diff
            best_face = face
    return best_face


@dataclass
class Neighbor:
    recording_id: str
    distance_m: float
    bearing_deg: float  # absolute bearing from source to this neighbor


@dataclass
class StreetGraph:
    adjacency: Dict[str, List[Neighbor]] = field(default_factory=dict)
    coords: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # recording_id -> (lat, lon)
    yaw_degrees: Dict[str, float] = field(default_factory=dict)  # recording_id -> recorder yaw
    face_frame: str = "absolute"  # "absolute" (compass-fixed faces) | "relative" (yaw-offset)

    def neighbors(self, recording_id: str) -> List[Neighbor]:
        return self.adjacency.get(recording_id, [])

    def _face_reference_yaw(self, recording_id: str) -> Optional[float]:
        """Yaw added to face offsets: 0 in the absolute frame, recorder yaw otherwise."""
        if self.face_frame == "absolute":
            return 0.0
        return self.yaw_degrees.get(recording_id)

    def face_bearing(self, recording_id: str, face: str) -> Optional[float]:
        """Absolute compass bearing of a face at a recording, or None if unknown."""
        yaw = self._face_reference_yaw(recording_id)
        offset = FACE_BEARING_DEG.get(face)
        if yaw is None or offset is None:
            return None
        return _normalize_bearing(yaw + offset)

    def resolve_face_to_neighbor(
        self,
        recording_id: str,
        face: str,
        bearing_tolerance_deg: float = 45.0,
    ) -> Optional[Neighbor]:
        """Map a face choice to the best matching neighbor.

        Computes the face's absolute bearing (see face_frame), then returns
        the closest neighbor within bearing_tolerance_deg.
        """
        target_bearing = self.face_bearing(recording_id, face)
        if target_bearing is None:
            return None

        neighbors = self.adjacency.get(recording_id, [])
        if not neighbors:
            return None

        best: Optional[Neighbor] = None
        best_diff = 999.0
        for nb in neighbors:
            diff = _bearing_diff(target_bearing, nb.bearing_deg)
            if diff < best_diff and diff <= bearing_tolerance_deg:
                best_diff = diff
                best = nb
        return best

    def arrival_face(self, from_id: str, to_id: str) -> str:
        """Compute the backtrack face at to_id: the face pointing back toward from_id.

        Choosing this face at to_id would retrace the move from_id -> to_id.
        Walk mechanics exclude it from the choices shown (no backtracking)
        unless it is the only legal move (dead-end turnaround).

        Returns "" when either node is unknown (no face excluded).
        """
        from_coords = self.coords.get(from_id)
        to_coords = self.coords.get(to_id)
        to_yaw = self._face_reference_yaw(to_id)
        if to_yaw is None:
            to_yaw = 0.0

        if from_coords is None or to_coords is None:
            return ""

        # Bearing from to -> from: the direction looking back where we came from
        lat1, lon1 = math.radians(to_coords[0]), math.radians(to_coords[1])
        lat2, lon2 = math.radians(from_coords[0]), math.radians(from_coords[1])
        dlon = lon2 - lon1
        x = math.cos(lat2) * math.sin(dlon)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        back_bearing = math.degrees(math.atan2(x, y)) % 360.0

        return _face_for_bearing(back_bearing, to_yaw)

    def available_faces(self, arrival_face: str) -> List[str]:
        """Return horizontal faces excluding the backtrack face.

        An empty arrival_face (walk seed, unknown origin) excludes nothing.
        """
        return [f for f in HORIZONTAL_FACES if f != arrival_face]

    def legal_faces(
        self,
        recording_id: str,
        exclude_face: str = "",
        bearing_tolerance_deg: float = 45.0,
        exclude_ids: Optional[set] = None,
    ) -> List[str]:
        """Return the faces that constitute legal moves from recording_id.

        A face is legal when it resolves to a neighbor within the bearing
        tolerance, the neighbor is not in exclude_ids, and the face is not
        exclude_face (the backtrack face). If that leaves nothing but the
        backtrack face would itself be legal, it is returned alone so a
        walk can turn around at a dead end instead of dying.
        """
        exclude_ids = exclude_ids or set()

        def _resolves(face: str) -> bool:
            nb = self.resolve_face_to_neighbor(recording_id, face, bearing_tolerance_deg)
            return nb is not None and nb.recording_id not in exclude_ids

        legal = [
            f for f in HORIZONTAL_FACES
            if f != exclude_face and _resolves(f)
        ]
        if not legal and exclude_face and _resolves(exclude_face):
            return [exclude_face]
        return legal

    def connected_components(self) -> List[set]:
        """Return connected components (sets of recording_ids), largest first.

        Adjacency is treated as undirected; nodes referenced only as
        neighbors are included.
        """
        node_ids = set(self.adjacency.keys())
        for nbs in self.adjacency.values():
            node_ids.update(nb.recording_id for nb in nbs)

        undirected: Dict[str, set] = {rid: set() for rid in node_ids}
        for rid, nbs in self.adjacency.items():
            for nb in nbs:
                undirected[rid].add(nb.recording_id)
                undirected[nb.recording_id].add(rid)

        seen: set = set()
        components: List[set] = []
        for start in node_ids:
            if start in seen:
                continue
            comp = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in comp:
                    continue
                comp.add(node)
                stack.extend(undirected[node] - comp)
            seen.update(comp)
            components.append(comp)

        components.sort(key=len, reverse=True)
        return components


def compute_graph_diagnostics(graph: StreetGraph) -> Dict[str, float]:
    """Board-quality metrics: connectivity, degree, and pitch uniformity."""
    diagnostics: Dict[str, float] = {}
    n_nodes = len(graph.adjacency)
    diagnostics["graph/n_nodes"] = float(n_nodes)
    if n_nodes == 0:
        return diagnostics

    degrees = sorted(len(v) for v in graph.adjacency.values())
    n_edges_directed = sum(degrees)
    diagnostics["graph/n_edges"] = float(n_edges_directed) / 2.0
    diagnostics["graph/degree_mean"] = n_edges_directed / n_nodes
    diagnostics["graph/degree_median"] = float(degrees[n_nodes // 2])
    diagnostics["graph/degree_max"] = float(degrees[-1])
    diagnostics["graph/n_isolated"] = float(sum(1 for d in degrees if d == 0))
    diagnostics["graph/n_dead_ends"] = float(sum(1 for d in degrees if d == 1))
    diagnostics["graph/n_intersections"] = float(sum(1 for d in degrees if d > 2))

    components = graph.connected_components()
    diagnostics["graph/n_components"] = float(len(components))
    diagnostics["graph/largest_component_frac"] = (
        len(components[0]) / float(n_nodes) if components else 0.0
    )

    dists = sorted(nb.distance_m for nbs in graph.adjacency.values() for nb in nbs)
    if dists:
        n = len(dists)
        diagnostics["graph/edge_m_median"] = float(dists[n // 2])
        diagnostics["graph/edge_m_p10"] = float(dists[int(n * 0.10)])
        diagnostics["graph/edge_m_p90"] = float(dists[min(n - 1, int(n * 0.90))])

    return diagnostics
