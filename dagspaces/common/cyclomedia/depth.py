"""Cyclomedia depth-map decoding.

Depth maps ship as ``depthmaps_faces/{F,R,B,L,U,D}.png`` -- 1024x1024 RGB PNGs
rendered by the same StreetSmart PanoramaRendering call as the RGB faces (with
``depthMap=true``), so they share the cube geometry in :mod:`.cubemap` exactly,
pixel for pixel.

Encoding (reverse-engineered; Cyclomedia publishes no spec)::

    code = R * 256 + G        # 16-bit, big-endian across the R and G channels
    B channel is unused (always 0)
    code == 0 means "no return" (sky, or beyond the depth model)

Evidence: the blue channel is identically zero on every face; the up-face is
98.7% zeros (sky); and binning the down-face's code against the ray geometry of
a flat ground plane gives a clean monotonic curve (R^2 = 0.998). ``code``
increases with distance.

Metric calibration (resolved 2026-07-29, was previously an open question)::

    range_m = (code - 16384) / 250.0      # 4 mm per code step, 0 .. 196.6 m

``range_m`` is the EUCLIDEAN distance from the camera centre along the pixel's
ray, not the perpendicular z-distance to the face plane.

The scale was measured from known camera baselines: when camera B sits on the
ray camera A shoots along, both terminate on the same surface point, so
``range_A - range_B`` equals the catalogued |AB|. Over 893 inlier pairs that
gives 249.86 +/- 0.17 codes/m, and -- decisively -- the value does not drift
with distance (249.7-250.0 from 19 m to 52 m), which is what finally ruled out
the log and inverse-depth curves. The zero-point came from cross-projection
consistency between neighbouring cameras (peak 16388, with 2**14 tied) and is
corroborated by facade planarity, which bottoms out at the same value. See
``docs/depth_maps.md`` and ``calibration/`` in the ``cyclomedia`` repo.

Two traps that produce confident wrong numbers:

* **Do not anchor on ``groundLevelOffset``.** The render places the camera at a
  fixed nominal ~2.18 m above the road for *every* vehicle fleet -- the
  down-face nadir code is ~16930 whether the catalogue says 2.23 m or 2.99 m.
  Anchoring on it yields 245.76 codes/m, which is 24 sigma off.
* **It is not IEEE float16.** Tempting because the down-face then lands at
  3.06-3.88 m, but observed codes exceed 45000, where the float16 sign bit is
  set: ~2% of pixels would decode negative or NaN while staying spatially
  continuous.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

__all__ = [
    "NO_RETURN",
    "ZERO_CODE",
    "SCALE",
    "MAX_RANGE_M",
    "decode_depth",
    "decode_depth_faces",
    "to_metres",
    "colorize_depth",
    "depth_stats",
]

# Sentinel stored in the PNG for pixels with no depth return (sky, out of range).
NO_RETURN = 0

# Metric calibration -- see the module docstring for how these were measured.
ZERO_CODE = 16384.0          # = 2**14; the code corresponding to zero range
SCALE = 250.0                # codes per metre, i.e. a 4 mm quantum
MAX_RANGE_M = (65535 - ZERO_CODE) / SCALE    # 196.6 m at the top of the range


def decode_depth(source) -> np.ndarray:
    """Decode a depth face to its raw 16-bit code.

    Args:
        source: Path to a depth PNG, a PIL image, or an (H, W, 3) uint8 array.

    Returns:
        (H, W) uint16 array. Zero means no return -- use ``code > 0`` as the
        validity mask rather than treating zeros as "very close".
    """
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, Image.Image):
        arr = np.asarray(source.convert("RGB"))
    else:
        with Image.open(source) as im:
            arr = np.asarray(im.convert("RGB"))

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected an (H, W, 3) RGB depth image, got shape {arr.shape}.")

    arr = arr.astype(np.uint16)
    return (arr[..., 0] << 8) | arr[..., 1]


def decode_depth_faces(faces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Decode a dict of RGB depth faces (as returned by ``load_faces(..., 'depth')``)."""
    return {face: decode_depth(img) for face, img in faces.items()}


def to_metres(code: np.ndarray) -> np.ndarray:
    """Convert raw depth codes to metric range.

    Args:
        code: (H, W) uint16 codes from :func:`decode_depth`.

    Returns:
        (H, W) float32 array of **Euclidean range from the camera centre along
        each pixel's ray** -- not perpendicular z-distance to the face plane.
        No-return pixels come back as NaN, so use ``np.nanmedian`` and friends,
        or mask with ``np.isfinite``.
    """
    out = (np.asarray(code).astype(np.float32) - ZERO_CODE) / SCALE
    out[np.asarray(code) == NO_RETURN] = np.nan
    return out


def colorize_depth(
    code: np.ndarray,
    cmap: str = "turbo",
    clip: Tuple[float, float] = (1.0, 99.0),
    invalid_rgb: Tuple[int, int, int] = (0, 0, 0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Render a depth code array as an RGB image using a *relative* scale.

    Valid codes are percentile-stretched (default 1st-99th) and mapped through
    a matplotlib colormap: near = dark/blue, far = bright/red for ``turbo``.
    No-return pixels are painted ``invalid_rgb`` (black by default).

    This is a relative visualisation. The colour scale carries no metric
    meaning -- see the module docstring.

    Args:
        vmin/vmax: Override the stretch with explicit code bounds. Useful for
            holding the scale fixed across faces so they are comparable.
    """
    import matplotlib

    valid = code > NO_RETURN
    out = np.empty(code.shape + (3,), dtype=np.uint8)
    out[...] = np.asarray(invalid_rgb, dtype=np.uint8)
    if not valid.any():
        return out

    vals = code[valid].astype(np.float64)
    lo = float(np.percentile(vals, clip[0])) if vmin is None else float(vmin)
    hi = float(np.percentile(vals, clip[1])) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0

    norm = np.clip((vals - lo) / (hi - lo), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](norm)
    out[valid] = (rgba[:, :3] * 255).astype(np.uint8)
    return out


def depth_stats(code: np.ndarray) -> dict:
    """Summarise a depth face: coverage and the code distribution.

    Percentiles are over valid pixels only. Codes are raw 16-bit values, not
    metres.
    """
    valid = code > NO_RETURN
    n = int(code.size)
    n_valid = int(valid.sum())
    stats = {
        "n_pixels": n,
        "n_valid": n_valid,
        "pct_valid": 100.0 * n_valid / n if n else 0.0,
        "pct_no_return": 100.0 * (n - n_valid) / n if n else 0.0,
    }
    if n_valid:
        vals = code[valid].astype(np.float64)
        for p in (1, 25, 50, 75, 99):
            stats[f"p{p}"] = float(np.percentile(vals, p))
        stats["min"] = float(vals.min())
        stats["max"] = float(vals.max())
    return stats
