"""Cube-face geometry for Cyclomedia panoramas.

Each Cyclomedia recording is stored as six 90 degree-hfov rectilinear renders
(``faces/{F,R,B,L,U,D}.jpg``), produced by the StreetSmart PanoramaRendering
API at::

    F: yaw=0    R: yaw=90   B: yaw=180   L: yaw=270   (pitch=0)
    U: pitch=+90            D: pitch=-90              (yaw=0)

The render yaw is an absolute compass bearing, so **F faces true north** --
the cube is north-referenced, not vehicle-referenced. The catalog confirms
this: its ``bearing`` column is exactly 0/90/180/270 for F/R/B/L and NULL for
U/D, regardless of the vehicle's heading (``recorderDirection``).

World frame used throughout: ``x = East, y = North, z = Up`` (right-handed).

Face orientations are *derived* from the render parameterisation rather than
guessed. A camera at (yaw t, pitch p) has::

    fwd   = ( sin t cos p,  cos t cos p,  sin p )
    right = ( cos t,       -sin t,        0     )      # = normalize(fwd x z_up)
    up    = (-sin t sin p, -cos t sin p,  cos p )      # = right x fwd

which stays well-defined in the pitch = +/-90 limit (``right`` is independent
of pitch), and yields the ray for normalised face coords (u right, v up),
u,v in [-1, 1] -- the hfov=90 image plane sits at unit distance, so::

    ray = fwd + u * right + v * up

Substituting each face's (yaw, pitch) gives FACE_RAYS below. Note this makes
the top of the D (down) face point north and the top of the U (up) face point
south -- a consequence of the shared (yaw, pitch) convention, not an arbitrary
choice. ``verify_face_orientations()`` checks the result empirically by
measuring seam continuity in the stitched panorama.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "FACES",
    "HORIZONTAL_FACES",
    "FACE_ANGLES",
    "FACE_DESCRIPTIONS",
    "face_ray",
    "load_faces",
    "cube_cross",
    "cube_to_equirect",
    "verify_face_orientations",
]

FACES: Tuple[str, ...] = ("F", "R", "B", "L", "U", "D")
HORIZONTAL_FACES: Tuple[str, ...] = ("F", "R", "B", "L")

# (yaw, pitch) in degrees, matching the downloader's face_to_angles().
FACE_ANGLES: Dict[str, Tuple[int, int]] = {
    "F": (0, 0),
    "R": (90, 0),
    "B": (180, 0),
    "L": (270, 0),
    "U": (0, 90),
    "D": (0, -90),
}

FACE_DESCRIPTIONS: Dict[str, str] = {
    "F": "Front (north)",
    "R": "Right (east)",
    "B": "Back (south)",
    "L": "Left (west)",
    "U": "Up",
    "D": "Down",
}


def face_ray(face: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ray direction(s) in world coords (East, North, Up) for face coords (u, v).

    ``u`` runs left->right across the face, ``v`` runs bottom->top, both in
    [-1, 1]. The returned rays are *not* normalised (magnitude is irrelevant
    for the axis-dominance test used when resampling).
    """
    if face not in FACE_ANGLES:
        raise ValueError(f"Unknown face {face!r}; expected one of {FACES}.")
    t, p = np.deg2rad(FACE_ANGLES[face][0]), np.deg2rad(FACE_ANGLES[face][1])
    fwd = np.array([np.sin(t) * np.cos(p), np.cos(t) * np.cos(p), np.sin(p)])
    right = np.array([np.cos(t), -np.sin(t), 0.0])
    up = np.array([-np.sin(t) * np.sin(p), -np.cos(t) * np.sin(p), np.cos(p)])
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return (
        fwd[None, :]
        + u[..., None] * right[None, :]
        + v[..., None] * up[None, :]
    )


def _face_uv_from_ray(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Invert :func:`face_ray`: map world rays to (face, u, v).

    Returns ``(face_index, masks, uv)`` where ``masks[face]`` selects the rays
    landing on that face (by dominant axis) and ``uv[face] = (u, v)``.
    """
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    eps = 1e-12

    masks = {
        "F": (ay >= ax) & (ay >= az) & (y > 0),
        "B": (ay >= ax) & (ay >= az) & (y <= 0),
        "R": (ax > ay) & (ax >= az) & (x > 0),
        "L": (ax > ay) & (ax >= az) & (x <= 0),
        "U": (az > ax) & (az > ay) & (z > 0),
        "D": (az > ax) & (az > ay) & (z <= 0),
    }
    # Each expression is the algebraic inverse of the corresponding FACE ray:
    #   F: (u, 1, v)     R: (1, -u, v)    B: (-u, -1, v)
    #   L: (-1, u, v)    U: (u, -v, 1)    D: (u, v, -1)
    uv = {
        "F": (x / (ay + eps), z / (ay + eps)),
        "B": (-x / (ay + eps), z / (ay + eps)),
        "R": (-y / (ax + eps), z / (ax + eps)),
        "L": (y / (ax + eps), z / (ax + eps)),
        "U": (x / (az + eps), -y / (az + eps)),
        "D": (x / (az + eps), y / (az + eps)),
    }
    return masks, uv


def load_faces(
    recording_dir: str,
    kind: str = "rgb",
    faces: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load a recording's cube faces from disk.

    Args:
        recording_dir: ``.../{dataset}/{group}/{recording_id}``.
        kind: ``"rgb"`` -> ``faces/{F}.jpg``; ``"depth"`` -> ``depthmaps_faces/{F}.png``.
        faces: Subset to load. Defaults to all six.

    Missing faces are skipped rather than raising, so a partially-downloaded
    recording still renders.
    """
    if kind == "rgb":
        subdir, ext = "faces", ".jpg"
    elif kind == "depth":
        subdir, ext = "depthmaps_faces", ".png"
    else:
        raise ValueError(f"kind must be 'rgb' or 'depth', got {kind!r}.")

    out: Dict[str, np.ndarray] = {}
    for face in faces or FACES:
        path = os.path.join(recording_dir, subdir, f"{face}{ext}")
        if not os.path.isfile(path):
            continue
        with Image.open(path) as im:
            out[face] = np.asarray(im.convert("RGB"))
    return out


def cube_cross(faces: Dict[str, np.ndarray], pad: int = 4) -> np.ndarray:
    """Lay the six faces out in the standard unfolded-cube cross.

            [ U ]
        [ L ][ F ][ R ][ B ]
            [ D ]

    Missing faces render as mid-grey. Returns an (H, W, 3) uint8 array.
    """
    present = [f for f in FACES if f in faces]
    if not present:
        raise ValueError("No faces supplied.")
    h, w = faces[present[0]].shape[:2]

    def tile(face: str) -> np.ndarray:
        if face in faces:
            return faces[face]
        return np.full((h, w, 3), 128, dtype=np.uint8)

    rows, cols = 3, 4
    canvas = np.full(
        (rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 32, dtype=np.uint8
    )

    def place(face: str, r: int, c: int) -> None:
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y : y + h, x : x + w] = tile(face)

    place("U", 0, 1)
    for i, face in enumerate(("L", "F", "R", "B")):
        place(face, 1, i)
    place("D", 2, 1)
    return canvas


def cube_to_equirect(
    faces: Dict[str, np.ndarray],
    width: int = 2048,
    height: Optional[int] = None,
    fill: float = 0,
) -> np.ndarray:
    """Resample cube faces into an equirectangular panorama.

    The output is north-referenced and reads like a compass: column 0 is bearing
    0 degrees (north), increasing eastward to 360; row 0 is +90 degrees elevation
    (zenith), the middle row is the horizon, the last row is nadir.

    Works on any face dtype and channel count -- (H, W, 3) uint8 RGB faces give
    an RGB panorama, and (H, W) uint16 depth-code faces (see :mod:`.depth`) give
    a single-channel code panorama. Stitching the *codes* and colourising once
    afterwards is what keeps a depth panorama on a single consistent scale;
    colourising each face first would give every face its own stretch.

    Uses nearest-neighbour sampling, which also matters for depth: interpolating
    across a depth discontinuity would invent surfaces that do not exist, and
    would blend the ``0`` no-return sentinel into real distances.

    Rays landing on a missing face are set to ``fill``.
    """
    if not faces:
        raise ValueError("No faces supplied.")
    if height is None:
        height = width // 2

    sample = next(iter(faces.values()))
    tail = sample.shape[2:]  # () for single-channel, (3,) for RGB

    # Pixel centres -> (bearing, elevation) -> unit ray in (East, North, Up).
    bearing = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 * np.pi
    elev = np.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * np.pi
    B, E = np.meshgrid(bearing, elev)
    cos_e = np.cos(E)
    x = cos_e * np.sin(B)  # East
    y = cos_e * np.cos(B)  # North
    z = np.sin(E)  # Up

    masks, uv = _face_uv_from_ray(x, y, z)
    pano = np.full((height, width) + tail, fill, dtype=sample.dtype)

    for face in FACES:
        mask = masks[face]
        if face not in faces or not mask.any():
            continue
        img = faces[face]
        fh, fw = img.shape[:2]
        u, v = uv[face]
        # [-1, 1] -> pixel indices; v is +up, so it flips against the row axis.
        col = np.clip(((u[mask] + 1.0) * 0.5 * fw).astype(np.int64), 0, fw - 1)
        row = np.clip(((1.0 - v[mask]) * 0.5 * fh).astype(np.int64), 0, fh - 1)
        pano[mask] = img[row, col]

    return pano


def verify_face_orientations(faces: Dict[str, np.ndarray], width: int = 1024) -> dict:
    """Measure seam continuity of the stitched panorama.

    A wrong per-face orientation (a rotation or flip of U/D, say) shows up as a
    hard discontinuity where that face abuts its neighbours. This renders the
    panorama and compares the mean absolute row-to-row gradient *across* the
    U/D seams (elevation +/-45 degrees, where the polar faces meet the horizontal
    ones) against the gradient in the horizontal band just inside them.

    A ``ratio`` near 1 means the seam is invisible (orientation correct); a
    ratio of several times 1 means the faces do not line up.
    """
    pano = cube_to_equirect(faces, width=width).astype(np.float64)
    height = pano.shape[0]

    def band_gradient(r0: int, r1: int) -> float:
        r0 = max(1, min(height - 1, r0))
        r1 = max(r0 + 1, min(height, r1))
        return float(np.abs(np.diff(pano[r0 - 1 : r1], axis=0)).mean())

    # Seam rows: elevation +45 (U meets F/R/B/L) and -45 (D meets them).
    seam_u = int(round((90.0 - 45.0) / 180.0 * height))
    seam_d = int(round((90.0 + 45.0) / 180.0 * height))
    span = max(2, height // 64)

    seam = 0.5 * (
        band_gradient(seam_u - span // 2, seam_u + span // 2)
        + band_gradient(seam_d - span // 2, seam_d + span // 2)
    )
    # Reference: the equatorial band, which is entirely interior to F/R/B/L.
    interior = band_gradient(height // 2 - span, height // 2 + span)
    return {
        "seam_gradient": seam,
        "interior_gradient": interior,
        "ratio": seam / interior if interior > 0 else float("nan"),
    }
