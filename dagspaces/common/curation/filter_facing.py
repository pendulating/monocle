"""Filter a curated parquet to rows where the face is oriented toward coverage.

The raw point-in-polygon filter (`materialize-cyclomedia`) keeps every face
whose recording position sits inside the coverage mask. That over-collects:
a face looking **away** from a scaffolded building still has its recording
point within the building's 80-ft buffer, so it passes point-in-polygon even
though the image contains no scaffold.

Two operating modes:

  * **Per-unit mode** (preferred, enabled by passing ``units_parquet``):
    each input row already carries a ``unit_uid`` attribution. We require
    that the ray intersect THAT specific unit's polygon (Fix A), that the
    LOS from recording → unit's own building is not strictly pierced by
    a non-unit NYC building (Fix F, occlusion), and that the angular
    offset between the face's bearing and the bearing from recording to
    unit's centroid be within a configurable tolerance (Fix C, default
    45° — the face's full half-FOV; unit just needs to be inside the
    90° face at all). Drops the "cross-street leakage" failure where a
    face tagged as Library A is actually pointing at Library B, and the
    "physically in-front-of" failure where a building (e.g. a food
    market) occludes the library.

  * **Legacy dissolved-coverage mode** (fallback when ``units_parquet`` is
    None): cast a short ray in the face's absolute bearing direction; keep
    rows whose ray intersects **any** coverage polygon within the ray
    length. Kept for backward compatibility with callers that don't have
    per-unit attribution yet.

Inputs:
- A curated parquet with ``latitude``, ``longitude``, ``bearing``, ``face``
  columns (the Cyclomedia catalog schema). In per-unit mode, also
  ``unit_uid``.
- Either a coverage GeoJSON (legacy) or a per-unit parquet
  (``facilities.parquet`` / ``permits.parquet``) exposing buffered polygons
  keyed by ``unit_uid``.

Output:
- A filtered sibling parquet. Per-unit mode adds diagnostic columns
  ``bearing_to_unit_deg``, ``delta_bearing_deg``, ``distance_to_unit_ft``.
- A ``filter_facing_manifest.json`` with counts + parameters.

U/D (sky/ground) faces drop unconditionally — no horizontal bearing is
defined for them.

Implementation note: a literal line-vs-polygon ``gpd.sjoin(predicate=
'intersects')`` is correct but slow — a 30-m LineString's bounding box
covers many more STRtree candidates than a Point's, so the join runs
orders of magnitude slower than the corresponding point join. We instead
**sample N points along the forward ray** (``ray_samples`` per ray,
evenly spaced from the recording outward to the endpoint, excluding the
recording itself) and keep a row if ANY of its sample points falls inside
a polygon. This is a point-in-polygon join at K×N points and retains the
directional semantics of the line test because the samples only cover
the forward half-line.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import geopandas as gpd
import numpy as np
import polars as pl

__all__ = [
    "filter_facing",
    "FilterFacingResult",
    "DEFAULT_RAY_LENGTH_M",
    "DEFAULT_RAY_SAMPLES",
    "DEFAULT_BEARING_TOL_DEG",
    "DEFAULT_MAX_DISTANCE_FT",
    "DEFAULT_CONFIDENCE_NORMALIZE_FT",
    "DEFAULT_BUILDINGS_PATH",
]

log = logging.getLogger(__name__)

DEFAULT_RAY_LENGTH_M = 30.0
DEFAULT_RAY_SAMPLES = 0   # sentinel: auto-scale from ray_length_m (≤10 m spacing, min 3)
# Widened 2026-04-22 (evening) back to the face's half-FOV. The earlier 22.5°
# cone was paired with a quadratic angular term and made sense when we were
# defending against "unit is a sliver at frame edge" — but after fixing the
# cube to the absolute frame and switching to linear angular falloff, the
# score itself already captures "edge-of-cone = bad attribution" via
# centeredness → 0 at 45°. The tight cone was then dropping genuinely-good
# close-off-axis shots (e.g. Δθ=19° at 69 ft, where the unit fills ~40% of
# the frame). 45° is the face's actual half-FOV: anything inside the face.
DEFAULT_BEARING_TOL_DEG = 45.0
# Fix D: default distance cap. The raw 80-ft buffer + typical building depth
# bound meaningful recordings at ~120 ft to centroid; 200 ft cuts the long tail
# (across-plaza / large-campus) without being aggressive.
DEFAULT_MAX_DISTANCE_FT: Optional[float] = 200.0
# Fix E: denominator for distance normalization when no hard cap is set.
DEFAULT_CONFIDENCE_NORMALIZE_FT = 200.0
# Fix E: characteristic unit half-width (US feet) used to estimate how much of
# the unit gets clipped by the face's FOV. ~25 ft is a reasonable one-size-
# fits-all for NYC libraries, scaffolding permits, etc. The score's angular
# term is the fraction of the unit still visible after edge-clipping; when
# the unit fits entirely in the face (common for D > ~30 ft), angular_term=1.
DEFAULT_UNIT_HALF_WIDTH_FT = 25.0
# Fix F: default NYC buildings polygon source for the occlusion check. Keeping
# this in sync with ``dagspaces.common.curation.geom.DEFAULT_BUILDINGS_PATH``
# but duplicated here so filter_facing has no import-time dependency on geom.
DEFAULT_BUILDINGS_PATH = "/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet"
_MAX_SAMPLE_SPACING_M = 10.0
_MIN_RAY_SAMPLES = 3


def _resolve_ray_samples(ray_samples: int, ray_length_m: float) -> int:
    """Pick the sample count. ``0`` → auto: keep spacing ≤ 10 m, at least 3 samples."""
    if ray_samples > 0:
        return ray_samples
    import math
    return max(_MIN_RAY_SAMPLES, math.ceil(ray_length_m / _MAX_SAMPLE_SPACING_M))
NYC_SP_CRS = "EPSG:2263"   # NY State Plane Long Island (US feet)
WGS84_CRS = "EPSG:4326"
M_PER_FT = 0.3048


@dataclass
class FilterFacingResult:
    input_parquet: str
    output_parquet: str
    coverage_source: str
    ray_length_m: float
    ray_samples: int
    in_rows: int
    horizontal_rows: int
    with_bearing_rows: int
    kept_rows: int
    dropped_rows: int
    per_face_kept: dict
    per_dataset_kept: dict
    manifest_path: str
    elapsed_s: float
    # Per-unit mode extras (all None in legacy mode).
    mode: str = "coverage"            # "coverage" or "per_unit"
    units_source: Optional[str] = None
    bearing_tol_deg: Optional[float] = None
    max_distance_ft: Optional[float] = None
    dropped_by_ray: Optional[int] = None
    dropped_by_bearing: Optional[int] = None
    dropped_by_distance: Optional[int] = None
    dropped_by_occlusion: Optional[int] = None
    occlusion_processable: Optional[int] = None
    buildings_source: Optional[str] = None
    mean_confidence: Optional[float] = None
    median_confidence: Optional[float] = None


def _build_ray_sample_points_gdf(
    df: pl.DataFrame,
    ray_length_m: float,
    ray_samples: int,
    projected_crs: str,
    *,
    attach_unit_uid: bool = False,
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of ray-sample Points in ``projected_crs``.

    For each row, sample ``ray_samples`` points evenly along the forward ray
    at fractions 1/N, 2/N, ..., N/N of the ``ray_length_m``. Sample 0 (the
    recording itself) is excluded on purpose — every row already passed
    point-in-coverage at its recording location, so including it would turn
    the filter into a no-op.

    Returns (N * ray_samples) × {_row, geometry} (+ ``unit_uid`` if
    ``attach_unit_uid``). ``_row`` refers back to the source DataFrame's
    row index, shared across all samples of that row.
    """
    cols = ["latitude", "longitude", "bearing"]
    if attach_unit_uid:
        cols = cols + ["unit_uid"]
    pdf = df.select(cols).to_pandas()
    n = len(pdf)
    pdf["_row"] = np.arange(n, dtype=np.int64)

    pts = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf["longitude"], pdf["latitude"]),
        crs=WGS84_CRS,
    ).to_crs(projected_crs)

    if projected_crs.upper() == NYC_SP_CRS:
        ray_len_units = ray_length_m / M_PER_FT    # EPSG:2263 is US feet
    else:
        ray_len_units = ray_length_m

    bearings_rad = np.radians(pdf["bearing"].to_numpy(dtype=np.float64))
    dx = np.sin(bearings_rad) * ray_len_units
    dy = np.cos(bearings_rad) * ray_len_units

    xs = pts.geometry.x.to_numpy()
    ys = pts.geometry.y.to_numpy()

    # Sample fractions 1/N, 2/N, ..., N/N along the forward ray (exclude origin).
    fracs = np.linspace(1.0 / ray_samples, 1.0, ray_samples)
    all_x = np.concatenate([xs + dx * f for f in fracs])
    all_y = np.concatenate([ys + dy * f for f in fracs])
    all_row = np.tile(pdf["_row"].values, ray_samples)

    out_cols = {"_row": all_row}
    if attach_unit_uid:
        all_uid = np.tile(pdf["unit_uid"].astype(str).values, ray_samples)
        out_cols["unit_uid"] = all_uid
    samples = gpd.GeoDataFrame(
        out_cols,
        geometry=gpd.points_from_xy(all_x, all_y),
        crs=projected_crs,
    )
    return samples


def _load_coverage(coverage_path: str, projected_crs: str) -> gpd.GeoDataFrame:
    if not os.path.isfile(coverage_path):
        raise FileNotFoundError(f"coverage not found: {coverage_path}")
    gdf = gpd.read_file(coverage_path)
    if gdf.empty:
        raise ValueError(f"{coverage_path} loaded 0 features")
    if gdf.crs is None:
        raise ValueError(f"{coverage_path} has no CRS set")
    return gdf.to_crs(projected_crs)


def _load_units_from_parquet(units_parquet: str, projected_crs: str) -> gpd.GeoDataFrame:
    """Load a ``facilities.parquet`` / ``permits.parquet`` into a GeoDataFrame
    keyed by ``unit_uid`` with geometry = the 80-ft buffered polygon.

    Also carries ``bin`` when present in the source parquet (both FacDB and
    scaffolding have it) — the occlusion filter (Fix F) uses it to exempt
    the unit's own BIN from blocker-candidate buildings. Rows without a BIN
    get an empty string and fall through the occlusion check as pass-through.

    Duplicates the schema-detection logic of
    :func:`dagspaces.common.curation.permits.materialize._load_units` but
    stays self-contained so filter_facing has no import-time dependency on
    the materialize module.
    """
    from shapely.wkb import loads as wkb_loads

    if not os.path.isfile(units_parquet):
        raise FileNotFoundError(f"units parquet not found: {units_parquet}")
    df = pl.read_parquet(units_parquet)
    cols = set(df.columns)

    if "uid" in cols and "facname" in cols:
        uid_col, name_col = "uid", "facname"
    elif "permit_id" in cols and "address" in cols:
        uid_col, name_col = "permit_id", "address"
    elif "permit_id" in cols:
        uid_col, name_col = "permit_id", "permit_id"
    else:
        raise ValueError(
            f"{units_parquet} has no recognizable unit key — expected uid+facname "
            f"(FacDB) or permit_id+address (scaffolding). Columns: {sorted(cols)[:12]}..."
        )
    if "geom_wkb" not in cols:
        raise ValueError(f"{units_parquet} missing geom_wkb column")

    select_cols = [uid_col, name_col, "geom_wkb"]
    if "bin" in cols:
        select_cols.append("bin")
    pdf = df.select(select_cols).to_pandas()
    geoms = [wkb_loads(b) if b is not None else None for b in pdf["geom_wkb"]]
    data = {
        "unit_uid": pdf[uid_col].astype(str),
        "unit_name": pdf[name_col].astype(str),
    }
    if "bin" in cols:
        # Normalize BIN to string; null/NaN → "" so equality comparisons are safe.
        data["bin"] = pdf["bin"].astype("string").fillna("").astype(str)
    gdf = gpd.GeoDataFrame(
        data,
        geometry=geoms,
        crs=WGS84_CRS,
    )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    return gdf.to_crs(projected_crs)


def _shortest_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the smallest unsigned angular difference between two bearings
    (degrees), in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _load_buildings_for_occlusion(path: str, projected_crs: str) -> gpd.GeoDataFrame:
    """Load ``nyc_buildings.parquet`` for the occlusion check (Fix F).

    Returns a GeoDataFrame with ``bin`` + ``geometry`` in ``projected_crs``,
    dissolved by BIN (so each BIN maps to exactly one polygon). We only need
    BIN + geometry; the parquet's other columns (`geom_source`, etc.) are
    dropped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"nyc_buildings parquet not found: {path}")
    gdf = gpd.read_parquet(path, columns=["bin", "geometry"])
    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; cannot proceed")
    gdf["bin"] = gdf["bin"].astype(str)
    if gdf["bin"].duplicated().any():
        gdf = gdf.dissolve(by="bin", as_index=False)
    return gdf.to_crs(projected_crs)


def _check_occlusion(
    post_ray: pl.DataFrame,
    units_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    projected_crs: str,
    *,
    lat_col: str,
    lon_col: str,
) -> tuple[np.ndarray, int, int]:
    """Fix F: drop rows whose LOS from recording → attributed library's
    building is strictly pierced by a non-unit building.

    "Strict pierce" = the ray segment intersects the candidate polygon AND
    both segment endpoints lie outside that polygon (i.e., the ray enters
    one side and exits the other — the polygon is fully between camera
    and target).

    Target endpoint is the library's actual building polygon's
    ``representative_point`` (guaranteed inside, unlike centroid which can
    fall outside concave footprints), looked up via the unit's BIN in
    ``buildings_gdf``. Rows whose attributed unit has no BIN (or whose BIN
    is absent from ``buildings_gdf``) pass through as "keep" — the check
    can't run without an anchor building, and these are the rare
    point-fallback / nearest-polygon rows anyway.

    Args:
        post_ray: Polars frame of rows that already passed Fix A
            (ray-vs-own-polygon). Must have ``unit_uid`` + lat/lon columns.
        units_gdf: Per-unit buffered polygons in ``projected_crs`` with
            ``unit_uid`` and (if available) ``bin`` columns.
        buildings_gdf: NYC buildings in ``projected_crs``; columns
            ``bin`` + ``geometry`` (dissolved by BIN).
        projected_crs: A planar CRS — expected to be EPSG:2263 (US feet).
        lat_col, lon_col: Column names in ``post_ray``.

    Returns:
        ``(keep_mask, n_processable, n_dropped)`` where ``keep_mask`` is a
        bool array aligned with ``post_ray`` (True = keep / pass-through /
        no blocker found; False = strictly pierced).
        ``n_processable`` is the count of rows for which the check actually
        ran (had a BIN matched in ``buildings_gdf``); ``n_dropped`` is the
        subset of those that failed.
    """
    from shapely.geometry import LineString, Point

    n = post_ray.height
    keep = np.ones(n, dtype=bool)
    if n == 0 or "bin" not in units_gdf.columns:
        # No BIN column on units → can't identify self-matches; skip.
        return keep, 0, 0

    # Build per-uid → bin lookup (fast dict; beat .loc[] per row)
    bin_by_uid: dict[str, str] = dict(
        zip(
            units_gdf["unit_uid"].astype(str).tolist(),
            units_gdf["bin"].astype(str).tolist(),
        )
    )

    # Buildings BIN → representative_point (guaranteed inside polygon) in
    # projected CRS. `representative_point` is O(1) per call and handles
    # concave polygons correctly.
    bldg_bin_arr = buildings_gdf["bin"].astype(str).to_numpy()
    bldg_geom_arr = buildings_gdf.geometry.to_numpy()
    bldg_poly_by_bin: dict[str, object] = dict(zip(bldg_bin_arr, bldg_geom_arr))

    pdf = post_ray.select(["unit_uid", lat_col, lon_col]).to_pandas()
    pdf["unit_uid"] = pdf["unit_uid"].astype(str)
    pdf["_unit_bin"] = pdf["unit_uid"].map(bin_by_uid).fillna("").astype(str)

    # Which rows can we actually check? Need a non-empty BIN that matches
    # a building polygon.
    processable_mask = np.array(
        [(b != "") and (b in bldg_poly_by_bin) for b in pdf["_unit_bin"].tolist()],
        dtype=bool,
    )
    n_processable = int(processable_mask.sum())
    if n_processable == 0:
        return keep, 0, 0

    # Project recording points all at once (WGS84 → planar).
    rec_gdf = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf[lon_col], pdf[lat_col]),
        crs=WGS84_CRS,
    ).to_crs(projected_crs)
    rxs = rec_gdf.geometry.x.to_numpy()
    rys = rec_gdf.geometry.y.to_numpy()

    # Build segments only for processable rows.
    proc_idx = np.where(processable_mask)[0]
    segments = []
    unit_bins_for_seg = []
    row_idx_for_seg = []
    for i in proc_idx:
        ub = pdf["_unit_bin"].iat[int(i)]
        target = bldg_poly_by_bin[ub].representative_point()
        segments.append(
            LineString([(float(rxs[i]), float(rys[i])), (float(target.x), float(target.y))])
        )
        unit_bins_for_seg.append(ub)
        row_idx_for_seg.append(int(i))

    rays_gdf = gpd.GeoDataFrame(
        {
            "_row": np.asarray(row_idx_for_seg, dtype=np.int64),
            "_unit_bin": np.asarray(unit_bins_for_seg, dtype=object),
        },
        geometry=segments,
        crs=projected_crs,
    )

    # STRtree-backed join: candidate blockers per segment.
    joined = gpd.sjoin(
        rays_gdf[["_row", "_unit_bin", "geometry"]],
        buildings_gdf[["bin", "geometry"]],
        predicate="intersects",
        how="inner",
    )
    if len(joined) == 0:
        return keep, n_processable, 0

    # Drop self-matches (library's own BIN blocking itself).
    joined = joined[joined["_unit_bin"] != joined["bin"]].copy()
    if len(joined) == 0:
        return keep, n_processable, 0

    # Strict-pierce test: both endpoints of the segment must be OUTSIDE the
    # candidate polygon. (sjoin already guarantees intersection, so both-
    # outside + intersects = enters AND exits.)
    seg_geoms = joined.geometry.to_numpy()
    # Matched building polygon, pulled via the sjoin right index.
    bldg_polys = buildings_gdf.geometry.to_numpy()[joined["index_right"].to_numpy()]
    starts = np.array([Point(g.coords[0]) for g in seg_geoms], dtype=object)
    ends = np.array([Point(g.coords[-1]) for g in seg_geoms], dtype=object)
    start_outside = np.array(
        [not p.covers(s) for p, s in zip(bldg_polys, starts)], dtype=bool
    )
    end_outside = np.array(
        [not p.covers(e) for p, e in zip(bldg_polys, ends)], dtype=bool
    )
    pierced = start_outside & end_outside
    pierced_rows = joined.loc[pierced, "_row"].unique()
    if len(pierced_rows) > 0:
        keep[pierced_rows.astype(np.int64)] = False

    n_dropped = int(n_processable - keep[processable_mask].sum())
    return keep, n_processable, n_dropped


def filter_facing(
    input_parquet: str,
    coverage_geojson: Optional[str],
    output_parquet: str,
    *,
    ray_length_m: float = DEFAULT_RAY_LENGTH_M,
    ray_samples: int = DEFAULT_RAY_SAMPLES,
    projected_crs: str = NYC_SP_CRS,
    horizontal_faces: tuple[str, ...] = ("F", "B", "L", "R"),
    bearing_col: str = "bearing",
    face_col: str = "face",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    units_parquet: Optional[str] = None,
    bearing_tol_deg: float = DEFAULT_BEARING_TOL_DEG,
    max_distance_ft: Optional[float] = DEFAULT_MAX_DISTANCE_FT,
    confidence_normalize_ft: float = DEFAULT_CONFIDENCE_NORMALIZE_FT,
    occlusion: bool = True,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    overwrite: bool = False,
) -> FilterFacingResult:
    """Drop rows whose forward ray doesn't actually face the attributed unit.

    Args:
        input_parquet: Curated parquet with ``latitude``/``longitude``/
            ``bearing``/``face`` columns. In per-unit mode, also ``unit_uid``.
        coverage_geojson: Dissolved coverage FeatureCollection. Used only
            in legacy mode; ignored (may be ``None``) when ``units_parquet``
            is provided.
        output_parquet: Where to write the filtered parquet. A sibling
            ``filter_facing_manifest.json`` is written next to it.
        ray_length_m: Ray length in meters. Default 30 (matches
            ``raster.py`` ``ray_length_m`` default).
        projected_crs: CRS used for the spatial math. Default NY State Plane.
        horizontal_faces: Faces to consider. U/D are excluded because they
            have no horizontal bearing.
        units_parquet: If provided, enable **per-unit mode** — the ray must
            hit the row's own ``unit_uid`` polygon, AND the angular offset
            between face and bearing-to-unit-centroid must be within
            ``bearing_tol_deg``.
        bearing_tol_deg: Per-unit mode only. Drop rows where
            ``|face_bearing - bearing_to_unit_centroid| > bearing_tol_deg``.
            Default 45° — the face's half-FOV. The unit just needs to be
            inside the 90° face at all; the continuous confidence score
            (Fix E) then prioritizes closer + more-centered shots with
            linear angular × quadratic distance falloff. Units: degrees.
        max_distance_ft: **Fix D.** Per-unit mode only. Drop rows where the
            recording → attributed unit centroid distance exceeds this cap
            (US feet). Default 200 — cuts the long tail of across-plaza /
            large-campus shots. Pass ``None`` to disable.
        confidence_normalize_ft: **Fix E.** Denominator used to normalize
            distance in the ``attribution_confidence`` score when
            ``max_distance_ft`` is ``None`` (otherwise the hard cap is
            reused). Units: US feet. Default 200.
        occlusion: **Fix F.** Per-unit mode only. If ``True`` (default),
            after Fix A, drop rows whose LOS from recording → attributed
            unit's building representative-point is strictly pierced by
            another NYC building (BIN ≠ unit's BIN). "Strict pierce" =
            ray enters and exits a non-unit building (both endpoints
            outside it). Needs the unit parquet to carry ``bin``; rows
            without a BIN match pass through.
        buildings_path: Path to ``nyc_buildings.parquet``, used only when
            ``occlusion=True``. Default:
            ``/share/pierson/matt/mllmsci/data/geo/nyc_buildings.parquet``.
    """
    t0 = time.monotonic()
    input_parquet = os.path.abspath(input_parquet)
    output_parquet = os.path.abspath(output_parquet)
    if coverage_geojson is not None:
        coverage_geojson = os.path.abspath(coverage_geojson)
    if units_parquet is not None:
        units_parquet = os.path.abspath(units_parquet)
    per_unit = units_parquet is not None
    mode = "per_unit" if per_unit else "coverage"

    if os.path.isfile(output_parquet) and not overwrite:
        raise FileExistsError(
            f"{output_parquet} already exists; pass overwrite=True"
        )
    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)

    if not per_unit and coverage_geojson is None:
        raise ValueError(
            "filter-facing requires either units_parquet (per-unit mode) or "
            "coverage_geojson (legacy dissolved-coverage mode)."
        )

    ray_samples = _resolve_ray_samples(ray_samples, ray_length_m)

    log.info(
        "filter-facing: reading %s (mode=%s ray_length=%.1f m ray_samples=%d bearing_tol=%.1f°)",
        input_parquet, mode, ray_length_m, ray_samples,
        bearing_tol_deg if per_unit else -1.0,
    )
    df = pl.read_parquet(input_parquet)
    n_in = df.height
    log.info("filter-facing: %d input rows", n_in)

    # Required columns
    required = [lat_col, lon_col, bearing_col, face_col]
    if per_unit:
        required.append("unit_uid")
    for c in required:
        if c not in df.columns:
            raise ValueError(f"input parquet missing required column {c!r}")

    # Drop U/D (no bearing) and rows with null bearing.
    horizontal = df.filter(pl.col(face_col).is_in(list(horizontal_faces)))
    n_horiz = horizontal.height
    with_bearing = horizontal.filter(pl.col(bearing_col).is_not_null())
    if per_unit:
        with_bearing = with_bearing.filter(pl.col("unit_uid").is_not_null())
    n_with_bearing = with_bearing.height
    log.info(
        "filter-facing: horizontal faces %d, with-bearing %d (dropped %d non-horizontal + %d null-bearing/unit)",
        n_horiz, n_with_bearing, n_in - n_horiz, n_horiz - n_with_bearing,
    )

    # Defaults updated by whichever branch runs.
    dropped_by_ray = 0
    dropped_by_bearing = 0
    dropped_by_distance = 0
    dropped_by_occlusion = 0
    occlusion_processable = 0
    mean_conf: Optional[float] = None
    median_conf: Optional[float] = None

    if with_bearing.is_empty():
        log.warning("filter-facing: no rows left after horizontal + bearing filter")
        kept = with_bearing

    elif per_unit:
        # -------- per-unit mode: A (ray vs row's own polygon) + C (bearing tol) --------
        t_units = time.monotonic()
        units_gdf = _load_units_from_parquet(units_parquet, projected_crs)
        log.info(
            "filter-facing[per_unit]: loaded %d units from %s in %.1fs",
            len(units_gdf), units_parquet, time.monotonic() - t_units,
        )

        known_uids = set(units_gdf["unit_uid"].astype(str).tolist())
        # Surface attribution mismatches as a warning but keep running — the
        # A test will drop them as "no polygon to hit".
        observed_uids = set(with_bearing["unit_uid"].unique().to_list())
        missing_uids = observed_uids - known_uids
        if missing_uids:
            log.warning(
                "filter-facing[per_unit]: %d unit_uid value(s) in input are absent "
                "from units parquet — those rows will be dropped.",
                len(missing_uids),
            )

        # Fix A: ray samples must land in the row's own unit polygon.
        t_rays = time.monotonic()
        samples = _build_ray_sample_points_gdf(
            with_bearing, ray_length_m, ray_samples, projected_crs,
            attach_unit_uid=True,
        )
        log.info(
            "filter-facing[per_unit]: built %d ray-sample points (N=%d per row) in %.1fs",
            len(samples), ray_samples, time.monotonic() - t_rays,
        )

        t_sj = time.monotonic()
        joined = gpd.sjoin(
            samples[["_row", "unit_uid", "geometry"]],
            units_gdf[["unit_uid", "geometry"]],
            predicate="within",
            how="inner",
            lsuffix="sample",
            rsuffix="unit",
        )
        # Per-unit: drop joins where the sample's attributed unit_uid doesn't
        # match the polygon it landed in.
        self_match = joined[joined["unit_uid_sample"] == joined["unit_uid_unit"]]
        ray_ok_rows = np.sort(self_match["_row"].drop_duplicates().to_numpy())
        log.info(
            "filter-facing[per_unit]: ray-vs-own-polygon kept %d/%d rows in %.1fs "
            "(%d sample hits, %d cross-attribution hits discarded)",
            len(ray_ok_rows), with_bearing.height, time.monotonic() - t_sj,
            len(self_match), len(joined) - len(self_match),
        )
        dropped_by_ray = with_bearing.height - int(len(ray_ok_rows))

        # Restrict to rows that pass A before running C + D + E.
        if len(ray_ok_rows) == 0:
            kept = with_bearing.head(0).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("bearing_to_unit_deg"),
                pl.lit(None, dtype=pl.Float64).alias("delta_bearing_deg"),
                pl.lit(None, dtype=pl.Float64).alias("distance_to_unit_ft"),
                pl.lit(None, dtype=pl.Float64).alias("attribution_confidence"),
            )
        else:
            mask = np.zeros(with_bearing.height, dtype=bool)
            mask[ray_ok_rows] = True
            post_ray = with_bearing.filter(pl.Series(values=mask))

            # Fix F: occlusion check — drop rows whose LOS to the library's
            # own building is strictly pierced by a different BIN. Runs
            # between A and C so the subsequent bearing/distance stats and
            # confidence score only reflect rows with unoccluded LOS.
            if occlusion:
                t_occ = time.monotonic()
                buildings_gdf = _load_buildings_for_occlusion(buildings_path, projected_crs)
                log.info(
                    "filter-facing[per_unit]: loaded %d buildings for occlusion "
                    "check from %s in %.1fs",
                    len(buildings_gdf), buildings_path, time.monotonic() - t_occ,
                )
                t_occ = time.monotonic()
                keep_mask, occlusion_processable, dropped_by_occlusion = _check_occlusion(
                    post_ray,
                    units_gdf=units_gdf,
                    buildings_gdf=buildings_gdf,
                    projected_crs=projected_crs,
                    lat_col=lat_col,
                    lon_col=lon_col,
                )
                post_ray = post_ray.filter(pl.Series(values=keep_mask))
                log.info(
                    "filter-facing[per_unit]: F kept %d/%d rows in %.1fs "
                    "(%d processable, %d strictly-pierced by non-unit building)",
                    post_ray.height, len(keep_mask), time.monotonic() - t_occ,
                    occlusion_processable, dropped_by_occlusion,
                )

            # Fix C: compute bearing and distance from recording → unit centroid
            # (in projected CRS for accuracy within NYC).
            t_bearing = time.monotonic()
            pdf = post_ray.select([lat_col, lon_col, bearing_col, "unit_uid"]).to_pandas()
            pts = gpd.GeoDataFrame(
                pdf,
                geometry=gpd.points_from_xy(pdf[lon_col], pdf[lat_col]),
                crs=WGS84_CRS,
            ).to_crs(projected_crs)
            centroids = units_gdf.set_index("unit_uid").geometry.centroid
            cx = centroids.x.to_dict()
            cy = centroids.y.to_dict()

            uids = pdf["unit_uid"].astype(str).to_numpy()
            cxs = np.array([cx.get(u, np.nan) for u in uids], dtype=np.float64)
            cys = np.array([cy.get(u, np.nan) for u in uids], dtype=np.float64)
            rxs = pts.geometry.x.to_numpy()
            rys = pts.geometry.y.to_numpy()

            dx = cxs - rxs
            dy = cys - rys
            bearing_to_unit = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
            # projected_crs in feet (EPSG:2263). Other CRSs: "distance_to_unit_ft"
            # becomes distance in whatever units_gdf's CRS uses — we don't convert.
            distance_ft = np.hypot(dx, dy)
            face_bearing = pdf[bearing_col].to_numpy(dtype=np.float64)
            delta = _shortest_angle_deg(face_bearing, bearing_to_unit)

            # Fix C + D combined mask.
            valid = np.isfinite(bearing_to_unit)
            bearing_mask = (delta <= bearing_tol_deg) & valid
            if max_distance_ft is not None:
                distance_mask = distance_ft <= float(max_distance_ft)
            else:
                distance_mask = np.ones_like(bearing_mask, dtype=bool)
            keep_mask = bearing_mask & distance_mask

            # Split drop counts so the manifest can attribute them.
            dropped_by_bearing = int((~bearing_mask & valid).sum())
            dropped_by_distance = int((bearing_mask & ~distance_mask).sum())

            # Fix E: confidence ∈ [0, 1] = visibility_fraction × distance³.
            # - visibility_fraction: physical "how much of the unit actually
            #   shows up in the face" — computed as the overlap between the
            #   unit's angular span [Δθ − H_half, Δθ + H_half] (with H_half
            #   = atan(H_REF / D)) and the face's FOV [−bearing_tol, +bearing_tol],
            #   divided by the unit's full angular width. It's 1.0 when the
            #   unit fits entirely in the face, and drops only when parts
            #   are clipped out of frame. Angular position within the
            #   "safe" sub-cone is NOT penalized — proximity alone ranks
            #   there.
            # - distance term: CUBIC (1 − D/D_cap)³. Proximity dominates:
            #   50 ft scores 3.38× 100 ft, 80 ft scores 1.73× 100 ft.
            denom_dist = (
                float(max_distance_ft)
                if max_distance_ft is not None
                else float(confidence_normalize_ft)
            )
            h_half_deg = np.degrees(
                np.arctan(
                    DEFAULT_UNIT_HALF_WIDTH_FT / np.maximum(distance_ft, 1e-6)
                )
            )
            visible_right = np.minimum(delta + h_half_deg, bearing_tol_deg)
            visible_left = np.maximum(delta - h_half_deg, -bearing_tol_deg)
            visible_width = np.maximum(0.0, visible_right - visible_left)
            visibility_fraction = np.clip(
                visible_width / np.maximum(2.0 * h_half_deg, 1e-6),
                0.0, 1.0,
            )
            distance_term = np.clip(1.0 - (distance_ft / max(denom_dist, 1e-6)), 0.0, 1.0) ** 3
            confidence = visibility_fraction * distance_term

            kept = post_ray.with_columns(
                pl.Series("bearing_to_unit_deg", bearing_to_unit, dtype=pl.Float64),
                pl.Series("delta_bearing_deg", delta, dtype=pl.Float64),
                pl.Series("distance_to_unit_ft", distance_ft, dtype=pl.Float64),
                pl.Series("attribution_confidence", confidence, dtype=pl.Float64),
            ).filter(pl.Series(values=keep_mask))

            if kept.height > 0:
                conf_series = kept["attribution_confidence"]
                mean_conf = float(conf_series.mean())
                median_conf = float(conf_series.median())
            log.info(
                "filter-facing[per_unit]: C+D kept %d/%d rows in %.1fs "
                "(dropped %d by bearing, %d by distance)",
                kept.height, post_ray.height,
                time.monotonic() - t_bearing, dropped_by_bearing, dropped_by_distance,
            )
            if mean_conf is not None:
                log.info(
                    "filter-facing[per_unit]: attribution_confidence mean=%.3f median=%.3f",
                    mean_conf, median_conf,
                )

    else:
        # -------- legacy mode: dissolved coverage + ray intersection --------
        t_cov = time.monotonic()
        cov_gdf = _load_coverage(coverage_geojson, projected_crs)
        log.info(
            "filter-facing[coverage]: loaded %d coverage polygons in %.1fs",
            len(cov_gdf), time.monotonic() - t_cov,
        )

        t_rays = time.monotonic()
        samples = _build_ray_sample_points_gdf(
            with_bearing, ray_length_m, ray_samples, projected_crs,
        )
        log.info(
            "filter-facing[coverage]: built %d ray-sample points (N=%d per row) in %.1fs",
            len(samples), ray_samples, time.monotonic() - t_rays,
        )

        t_sj = time.monotonic()
        joined = gpd.sjoin(
            samples[["_row", "geometry"]],
            cov_gdf[["geometry"]],
            predicate="within",
            how="inner",
        )
        matched_rows = np.sort(joined["_row"].drop_duplicates().to_numpy())
        log.info(
            "filter-facing[coverage]: sjoin matched %d/%d rows in %.1fs "
            "(%d candidate sample hits)",
            len(matched_rows), with_bearing.height,
            time.monotonic() - t_sj, len(joined),
        )

        if len(matched_rows) == 0:
            kept = with_bearing.head(0)
        else:
            mask = np.zeros(with_bearing.height, dtype=bool)
            mask[matched_rows] = True
            kept = with_bearing.filter(pl.Series(values=mask))

    # Write the filtered parquet.
    kept.write_parquet(output_parquet)
    size_mb = os.path.getsize(output_parquet) / (1024 ** 2)
    log.info(
        "filter-facing: wrote %d rows → %s (%.1f MB)",
        kept.height, output_parquet, size_mb,
    )

    # Per-face / per-dataset summary
    per_face_kept: dict = {}
    per_dataset_kept: dict = {}
    if face_col in kept.columns and kept.height > 0:
        per_face_kept = {
            r[face_col]: int(r["len"])
            for r in kept.group_by(face_col).len().to_dicts()
        }
    if "dataset" in kept.columns and kept.height > 0:
        per_dataset_kept = {
            r["dataset"]: int(r["len"])
            for r in kept.group_by("dataset").len().to_dicts()
        }

    elapsed = time.monotonic() - t0
    manifest_path = os.path.join(
        os.path.dirname(output_parquet),
        os.path.splitext(os.path.basename(output_parquet))[0]
        + "_filter_facing_manifest.json",
    )
    manifest = {
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_parquet": input_parquet,
        "output_parquet": output_parquet,
        "mode": mode,
        "coverage_source": coverage_geojson,
        "units_source": units_parquet,
        "ray_length_m": ray_length_m,
        "ray_samples": ray_samples,
        "bearing_tol_deg": bearing_tol_deg if per_unit else None,
        "max_distance_ft": max_distance_ft if per_unit else None,
        "confidence_normalize_ft": confidence_normalize_ft if per_unit else None,
        "projected_crs": projected_crs,
        "in_rows": int(n_in),
        "horizontal_rows": int(n_horiz),
        "with_bearing_rows": int(n_with_bearing),
        "dropped_by_ray": int(dropped_by_ray) if per_unit else None,
        "dropped_by_bearing": int(dropped_by_bearing) if per_unit else None,
        "dropped_by_distance": int(dropped_by_distance) if per_unit else None,
        "dropped_by_occlusion": int(dropped_by_occlusion) if per_unit else None,
        "occlusion_processable": int(occlusion_processable) if per_unit else None,
        "buildings_source": buildings_path if (per_unit and occlusion) else None,
        "kept_rows": int(kept.height),
        "dropped_rows": int(n_with_bearing - kept.height),
        "mean_attribution_confidence": mean_conf if per_unit else None,
        "median_attribution_confidence": median_conf if per_unit else None,
        "per_face_kept": per_face_kept,
        "per_dataset_kept": per_dataset_kept,
        "file_size_mb": round(size_mb, 3),
        "elapsed_s": round(elapsed, 3),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(
        "filter-facing: done — kept %d/%d (%.1f%%) in %.1fs",
        kept.height, n_in, 100 * kept.height / max(n_in, 1), elapsed,
    )

    return FilterFacingResult(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        coverage_source=coverage_geojson or "",
        ray_length_m=ray_length_m,
        ray_samples=ray_samples,
        in_rows=int(n_in),
        horizontal_rows=int(n_horiz),
        with_bearing_rows=int(n_with_bearing),
        kept_rows=int(kept.height),
        dropped_rows=int(n_with_bearing - kept.height),
        per_face_kept=per_face_kept,
        per_dataset_kept=per_dataset_kept,
        manifest_path=manifest_path,
        elapsed_s=elapsed,
        mode=mode,
        units_source=units_parquet,
        bearing_tol_deg=bearing_tol_deg if per_unit else None,
        max_distance_ft=max_distance_ft if per_unit else None,
        dropped_by_ray=int(dropped_by_ray) if per_unit else None,
        dropped_by_bearing=int(dropped_by_bearing) if per_unit else None,
        dropped_by_distance=int(dropped_by_distance) if per_unit else None,
        dropped_by_occlusion=int(dropped_by_occlusion) if per_unit else None,
        occlusion_processable=int(occlusion_processable) if per_unit else None,
        buildings_source=buildings_path if (per_unit and occlusion) else None,
        mean_confidence=mean_conf,
        median_confidence=median_conf,
    )
