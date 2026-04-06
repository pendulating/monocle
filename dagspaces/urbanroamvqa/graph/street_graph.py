"""StreetGraph dataclass for recording-level street network adjacency."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Face bearing offsets (degrees relative to recording yaw).
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

    def neighbors(self, recording_id: str) -> List[Neighbor]:
        return self.adjacency.get(recording_id, [])

    def resolve_face_to_neighbor(
        self,
        recording_id: str,
        face: str,
        bearing_tolerance_deg: float = 45.0,
    ) -> Optional[Neighbor]:
        """Map a face choice to the best matching neighbor.

        Computes absolute_bearing = (yaw + FACE_BEARING[face]) % 360,
        then returns the closest neighbor within bearing_tolerance_deg.
        """
        yaw = self.yaw_degrees.get(recording_id)
        if yaw is None:
            return None
        offset = FACE_BEARING_DEG.get(face)
        if offset is None:
            return None
        target_bearing = _normalize_bearing(yaw + offset)

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
        """Compute the face at to_id that represents the forward direction (away from from_id).

        reverse_bearing = bearing(to -> from)
        forward_bearing = (reverse_bearing + 180) % 360
        Return face whose absolute bearing is closest to forward_bearing.
        """
        from_coords = self.coords.get(from_id)
        to_coords = self.coords.get(to_id)
        to_yaw = self.yaw_degrees.get(to_id, 0.0)

        if from_coords is None or to_coords is None:
            return "F"

        # Compute bearing from to -> from
        lat1, lon1 = math.radians(to_coords[0]), math.radians(to_coords[1])
        lat2, lon2 = math.radians(from_coords[0]), math.radians(from_coords[1])
        dlon = lon2 - lon1
        x = math.cos(lat2) * math.sin(dlon)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        reverse_bearing = math.degrees(math.atan2(x, y)) % 360.0

        # Forward = opposite of reverse
        forward_bearing = (reverse_bearing + 180.0) % 360.0

        return _face_for_bearing(forward_bearing, to_yaw)

    def available_faces(self, arrival_face: str) -> List[str]:
        """Return the 3 horizontal faces excluding the arrival face."""
        return [f for f in HORIZONTAL_FACES if f != arrival_face]
