#!/usr/bin/env python
"""F6: select the best storefront view of each restaurant, and render it.

Version 1 (`f6_build_restaurant_depiction_sample.py`) took a cube face that the
facing filter aimed at the centroid of the building footprint. Version 1.x of
this script (kept as `data/f6_restaurant_depiction/v3/f6_select_storefront_views_v1.2.0.py`)
aimed at the footprint point nearest the DOHMH point P. The advisor review of
2026-10-24 (`data/f6_restaurant_depiction/advisor-review-2026-10-24-math.md`)
showed that P is on the street centerline, so that point is often a building
corner, and the normal there is the 45 degree diagonal. Version 2 finds the
storefront wall from the address street. The rules come from the decisions in
`research/docs/facct-runbook.md` (D46 to D84).

The rules
---------
| Rule | Value | Decision |
|------|-------|----------|
| Images | the `*_2025_1k` datasets: Aug-Nov 2025 (`recordedAt` on or after 2025-07-01); Mar-Jun 2025 only as below | D58, D78 |
| Restaurants | the DOHMH pull of 2026-04-27 (`curation/dohmh_restaurants_inspected_all`) | D59 |
| Footprint | the building of the restaurant's BIN; no BIN footprint, no restaurant | D77 |
| Storefront wall | the footprint edge parallel to the address street (CSCL centerline) that faces it, nearest P | D73 |
| Storefront point S | the point of that wall nearest P; the middle of the wall when P is past an end of it; n is the outward normal of the wall | D73, D80 |
| Camera on the frontage | the camera is at most 80 ft from P | D56 |
| Bad geocode | the restaurant leaves when P is more than 150 ft from S | D75, D82 |
| Viewing angle | theta is at most 45 degrees | D74, D81 |
| Line of sight | the segment from the camera to S runs at most 1 ft inside building footprints | D76 |
| Attribution | one-to-many: 1 recording can serve every restaurant that qualifies | D51 |
| Open at capture | the restaurant has an inspection within 180 days of `recordedAt` | D50 |
| Score | facade size in the frame, `cos(theta) / r ** 2`, r = camera to S, theta = angle of `C - S` to n | D74 |
| Mar-Jun pass | when no Aug-Nov view in the near half of the street qualifies, Mar-Jun views in the near half join and compete on the score | D78, D84 |
| Fallback wall | when no wall faces the address street, the wall that faces P (below) | D83 |
| Shared address | restaurants at 1 address can get the same view | D79 |
| View | a virtual camera at the recording, aimed at S, pitch +10 degrees, 70 degree field of view, 1024 x 1024 | D57, D60, D61 |

The score: the camera is rectilinear and aimed at S. A facade element of width
dl and height dh at S has an image width of `f dl cos(theta) / r` and an image
height of `f dh / r`, thus its area in the frame is proportional to
`cos(theta) / r ** 2`.

Choices of the implementation (itemized to the user):
- Street match: the DOHMH `street` and the CSCL `full_street_name`, both
  normalized (`norm_street`); the nearest segment of the same name within
  150 ft of P.
- Wall: a straight run of the ring, simplified with a tolerance of 1 ft, of 5 ft or more; parallel means within 25 degrees of the street
  direction at P; "faces it" means the street is on the outward side of the
  edge.
- Fallback wall: when the street does not match, or no wall faces it, the
  wall that faces P with the smallest overshoot of P past its ends.
- Near half: the camera is between the wall and the street centerline
  (`dot(C - Q, n) < 0`, Q = the centerline point nearest S; Q = P for the
  fallback wall).
- Ties of the score: the smaller angle theta, then the recording id.
- Duplicate recordings in 2 datasets: the first dataset in name order.

The cube convention (north-referenced faces, ray formulas, U/D orientation)
follows `/share/ju/cyclomedia/cycdb/cubemap.py`.
# ref: /share/ju/cyclomedia/cycdb/cubemap.py (face_ray, _face_uv_from_ray)

Use:
    PY=.venv-mllmsci-vllm025cu129/bin/python
    $PY research/scripts/f6_select_storefront_views.py check-render     # the renderer reproduces F and R
    $PY research/scripts/f6_select_storefront_views.py select            # candidates.parquet (all flags)
    $PY research/scripts/f6_select_storefront_views.py best              # qualifying.parquet, best.parquet
    $PY research/scripts/f6_select_storefront_views.py sample --n 100    # render 100, Label Studio tasks
    sbatch research/scripts/f6_render_best_views.sub                     # render the best view of each restaurant (D63)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
from PIL import Image

ROOT = "/share/ju/matt/mllmsci"
DOHMH = os.path.join(ROOT, "curation/dohmh_restaurants_inspected_all")
BUILDINGS = os.path.join(ROOT, "data/geo/nyc_buildings.parquet")
CATALOG = "/share/ju/cyclomedia/catalog/v1/by_dataset"
DATASETS = ["bronx_2025_1k", "brooklyn_2025_1k", "manhattan_2025_1k", "queens_2025_1k", "si_2025_1k"]
STREETS = os.path.join(ROOT, "data/external/nyc_street_centerline/cscl_roadway_lines_2263.parquet")
# v2 (version 1.0.0, no angle limit) holds the images of Label Studio project 3;
# v3 (version 1.2.0, the corner diagonal) holds project 5.
OUT_DIR = os.path.join(ROOT, "data/f6_restaurant_depiction/v4")
CRS_FT = "EPSG:2263"

EPOCH_START = "2025-07-01"   # D58: no recording in July 2025; Aug-Nov is the second epoch
MAX_CAM_TO_P_FT = 80.0       # D56
OPEN_WINDOW_DAYS = 180       # D50
LOS_MAX_INSIDE_FT = 1.0      # D76: the sightline can run 1 ft inside footprints
STREET_MATCH_FT = 150.0      # the address street must be this near P
STREET_PARALLEL_DEG = 25.0   # a wall parallel to the street
MIN_EDGE_FT = 5.0            # shorter walls are corner notches
WALL_SIMPLIFY_FT = 1.0       # collinear edges join into 1 wall; jogs under 1 ft go
TANGENT_HALF_FT = 5.0        # the street direction from 5 ft each side of the nearest point
VIEW_PX = 1024
VIEW_FOV_DEG = 70.0          # D61
VIEW_PITCH_DEG = 10.0        # D61: removes the roof of the camera car

__version__ = "2.0.0"


# ---------------------------------------------------------------- the renderer
# World frame: x = East, y = North, z = Up. A face of yaw t and pitch p has the
# ray fwd + u * right + v * up, with u to the right and v up, in [-1, 1].

def _basis(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(t) * math.cos(p), math.cos(t) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(t), -math.sin(t), 0.0])
    up = np.array([-math.sin(t) * math.sin(p), -math.cos(t) * math.sin(p), math.cos(p)])
    return fwd, right, up


def _sample_bilinear(img: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    c1 = np.minimum(c0 + 1, w - 1)
    r1 = np.minimum(r0 + 1, h - 1)
    fc = (col - c0)[:, None]
    fr = (row - r0)[:, None]
    top = img[r0, c0] * (1 - fc) + img[r0, c1] * fc
    bot = img[r1, c0] * (1 - fc) + img[r1, c1] * fc
    return top * (1 - fr) + bot * fr


def render_view(faces: dict[str, np.ndarray], yaw_deg: float, pitch_deg: float = 0.0,
                fov_deg: float = VIEW_FOV_DEG, size: int = VIEW_PX) -> np.ndarray:
    """Render a rectilinear view from the 6 cube faces of 1 recording."""
    fwd, right, up = _basis(yaw_deg, pitch_deg)
    half = math.tan(math.radians(fov_deg) / 2)
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords * half, -coords * half)          # v is +1 on the top row
    rays = fwd[None, None, :] + u[..., None] * right + v[..., None] * up
    x, y, z = rays[..., 0].ravel(), rays[..., 1].ravel(), rays[..., 2].ravel()
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
    uv = {
        "F": (x / (ay + eps), z / (ay + eps)),
        "B": (-x / (ay + eps), z / (ay + eps)),
        "R": (-y / (ax + eps), z / (ax + eps)),
        "L": (y / (ax + eps), z / (ax + eps)),
        "U": (x / (az + eps), -y / (az + eps)),
        "D": (x / (az + eps), y / (az + eps)),
    }
    out = np.zeros((x.size, 3), dtype=np.float64)
    for face, m in masks.items():
        if not m.any():
            continue
        if face not in faces:
            raise FileNotFoundError(f"the view needs face {face}, which is missing")
        img = faces[face].astype(np.float64)
        n = img.shape[0]
        fu, fv = uv[face][0][m], uv[face][1][m]
        col = (fu + 1) / 2 * n - 0.5
        row = (1 - fv) / 2 * n - 0.5
        out[m] = _sample_bilinear(img, col, row)
    return np.clip(out.reshape(size, size, 3) + 0.5, 0, 255).astype(np.uint8)


def load_faces(face_dir: str) -> dict[str, np.ndarray]:
    faces = {}
    for f in ("F", "R", "B", "L", "U", "D"):
        p = os.path.join(face_dir, f"{f}.jpg")
        if os.path.isfile(p):
            with Image.open(p) as im:
                faces[f] = np.asarray(im.convert("RGB"))
    return faces


class LazyFaces(dict):
    """The cube faces of 1 recording, read from disk only when a view needs them.

    A view of 70 degrees needs 2 or 3 of the 6 faces. This halves the NFS reads
    of `render-all`, and the faces stay cached for the next view of the same
    recording.
    """

    def __init__(self, face_dir: str):
        super().__init__()
        self.face_dir = face_dir

    def _path(self, face: str) -> str:
        return os.path.join(self.face_dir, f"{face}.jpg")

    def __contains__(self, face: object) -> bool:
        return dict.__contains__(self, face) or os.path.isfile(self._path(str(face)))

    def __missing__(self, face: str) -> np.ndarray:
        with Image.open(self._path(face)) as im:
            arr = np.asarray(im.convert("RGB"))
        self[face] = arr
        return arr


def cmd_check_render(args: argparse.Namespace) -> int:
    """Render yaw 0, 90, 180, 270 at pitch 0 and compare with F, R, B, L."""
    face_dir = args.face_dir
    faces = load_faces(face_dir)
    for face, yaw in (("F", 0), ("R", 90), ("B", 180), ("L", 270)):
        view = render_view(faces, yaw, 0.0, 90.0, faces[face].shape[0])
        diff = np.abs(view.astype(np.int16) - faces[face].astype(np.int16))
        print(f"{face} yaw {yaw}: mean abs diff {diff.mean():.3f}, p99 {np.percentile(diff, 99):.1f}")
    os.makedirs(args.out, exist_ok=True)
    for yaw in (45, 135):
        Image.fromarray(render_view(faces, yaw, 0.0, 90.0)).save(os.path.join(args.out, f"check_yaw{yaw}.jpg"), quality=92)
    print(f"wrote the yaw 45 and 135 views to {args.out}")
    return 0


# ---------------------------------------------------------------- the geometry

# Street words in the DOHMH form and in the CSCL form. Both sides go through the
# same map, thus "SOUTH" and "S" give the same token on both sides.
STREET_WORDS = {
    "STREET": "ST", "AVENUE": "AVE", "AV": "AVE", "BOULEVARD": "BLVD", "ROAD": "RD",
    "PLACE": "PL", "PARKWAY": "PKWY", "PLAZA": "PLZ", "SQUARE": "SQ", "COURT": "CT",
    "DRIVE": "DR", "LANE": "LN", "TERRACE": "TER", "TURNPIKE": "TPKE", "HIGHWAY": "HWY",
    "EXPRESSWAY": "EXPY", "BEACH": "BCH", "CIRCLE": "CIR", "CRESCENT": "CRES",
    "ALLEY": "ALY", "BRIDGE": "BRG", "SAINT": "ST",
    "WEST": "W", "EAST": "E", "NORTH": "N", "SOUTH": "S", "CONCOURSE": "CONC",
    "FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4", "FIFTH": "5",
    "SIXTH": "6", "SEVENTH": "7", "EIGHTH": "8", "NINTH": "9", "TENTH": "10",
}
# Whole names that differ between the 2 sources, after the word map.
STREET_NAMES = {
    "AVE OF THE AMERICAS": "6 AVE", "CROSSBAY BLVD": "CROSS BAY BLVD",
    "MACDOUGAL ST": "MAC DOUGAL ST", "LAGUARDIA PL": "LA GUARDIA PL",
}


def norm_street(s: str | None) -> str | None:
    """Give a street name in 1 form: upper case, short street words, no ordinal suffix."""
    if s is None:
        return None
    toks = []
    for t in re.sub(r"[^A-Z0-9 ]", " ", s.upper()).split():
        t = re.sub(r"^(\d+)(ST|ND|RD|TH)$", r"\1", t)
        toks.append(STREET_WORDS.get(t, t))
    name = " ".join(toks)
    return STREET_NAMES.get(name, name) or None


def _edges(poly) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Give the walls of a footprint: start a, unit direction d, length L, outward normal n.

    A wall is 1 straight run of the exterior ring: the ring is simplified with a
    tolerance of `WALL_SIMPLIFY_FT`, thus collinear edges join and small jogs go.
    The rings are made counterclockwise, thus the inside is on the left of d and
    the outward normal is the right of d. Walls shorter than `MIN_EDGE_FT` are
    corner notches.
    """
    parts = list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly]
    a_all, d_all, l_all = [], [], []
    for part in parts:
        part = shapely.Polygon(part.exterior).simplify(WALL_SIMPLIFY_FT, preserve_topology=True)
        xy = np.asarray(shapely.geometry.polygon.orient(part, 1.0).exterior.coords)[:, :2]
        v = xy[1:] - xy[:-1]
        length = np.hypot(v[:, 0], v[:, 1])
        keep = length >= MIN_EDGE_FT
        a_all.append(xy[:-1][keep])
        d_all.append(v[keep] / length[keep, None])
        l_all.append(length[keep])
    a, d, length = np.concatenate(a_all), np.concatenate(d_all), np.concatenate(l_all)
    n = np.stack([d[:, 1], -d[:, 0]], 1)
    return a, d, length, n


def _foot(a: np.ndarray, d: np.ndarray, length: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give the position t of p along each edge, and the edge point nearest p."""
    t = ((p - a) * d).sum(1)
    return t, a + np.clip(t, 0, length)[:, None] * d


def _aim(poly, a: np.ndarray, d: np.ndarray, length: np.ndarray, t: np.ndarray, k: int) -> tuple[np.ndarray, bool]:
    """Give S on wall k, and whether P is past an end of the wall.

    S is the wall point nearest P. When P is past an end of the wall, S is the
    middle of the wall (D80). S then moves to the nearest point of the true
    footprint ring, because the simplified wall can be up to 1 ft off it.
    """
    past = bool(t[k] < 0 or t[k] > length[k])
    s = a[k] + (0.5 * length[k] if past else t[k]) * d[k]
    return _nearest_on(poly.boundary, s[None])[0], past


def _nearest_on(line, xy: np.ndarray) -> np.ndarray:
    """Give the point of a line nearest each point of xy."""
    near = shapely.shortest_line(line, shapely.points(xy))
    return shapely.get_coordinates(shapely.get_point(near, 0))


def _street_frame(line, p: shapely.Point) -> np.ndarray:
    """Give the unit direction of a street centerline at the point nearest p."""
    if line.geom_type == "MultiLineString":
        line = min(line.geoms, key=lambda g: g.distance(p))
    at = line.project(p)
    q1 = line.interpolate(max(at - TANGENT_HALF_FT, 0.0))
    q2 = line.interpolate(min(at + TANGENT_HALF_FT, line.length))
    t = np.array([q2.x - q1.x, q2.y - q1.y])
    return t / max(np.hypot(*t), 1e-9)


def wall_by_street(poly, p: shapely.Point, street) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """Give S, n, Q and "P past the wall end" on the wall that faces the address street (D73), or None.

    A wall qualifies when it is parallel to the street within
    `STREET_PARALLEL_DEG` and the street is on its outward side. Of these, the
    wall nearest P gives n, and S (`_aim`). Q is the centerline point nearest S.
    """
    a, d, length, n = _edges(poly)
    if not len(a):
        return None
    tdir = _street_frame(street, p)
    parallel = np.abs(d[:, 0] * tdir[1] - d[:, 1] * tdir[0]) <= math.sin(math.radians(STREET_PARALLEL_DEG))
    mid = a + 0.5 * length[:, None] * d
    q_mid = _nearest_on(street, mid)
    faces = ((q_mid - mid) * n).sum(1) > 0
    ok = parallel & faces
    if not ok.any():
        return None
    pv = np.array([p.x, p.y])
    t, foot = _foot(a, d, length, pv)
    dist = np.hypot(*(foot - pv).T)
    k = np.flatnonzero(ok)[np.argmin(dist[ok])]
    s, past = _aim(poly, a, d, length, t, k)
    return s, n[k], _nearest_on(street, s[None])[0], past


def wall_by_p(poly, p: shapely.Point) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """Give S, n, Q and "P past the wall end" on the wall that faces P (the fallback wall), or None.

    A wall qualifies when P is on its outward side. Of these, the wall with the
    smallest overshoot of P past its ends gives n, and S (`_aim`); ties go to
    the nearer wall. Q is P.
    """
    a, d, length, n = _edges(poly)
    pv = np.array([p.x, p.y])
    ok = ((pv - a) * n).sum(1) > 0
    if not ok.any():
        return None
    t, foot = _foot(a, d, length, pv)
    over = np.maximum(np.maximum(-t, t - length), 0.0)
    dist = np.hypot(*(foot - pv).T)
    idx = np.flatnonzero(ok)
    k = idx[np.lexsort((dist[idx], over[idx]))[0]]
    s, past = _aim(poly, a, d, length, t, k)
    return s, n[k], pv, past


def load_restaurants() -> gpd.GeoDataFrame:
    a = pl.read_parquet(os.path.join(DOHMH, "restaurants_aggregated.parquet"),
                        columns=["uid", "camis", "dba", "bin", "street", "raw_latitude", "raw_longitude", "borough"])
    g = gpd.GeoDataFrame(a.to_pandas(), geometry=gpd.points_from_xy(a["raw_longitude"], a["raw_latitude"]),
                         crs="EPSG:4326").to_crs(CRS_FT)
    return g


def load_buildings() -> gpd.GeoDataFrame:
    b = gpd.read_parquet(BUILDINGS, columns=["bin", "geometry"]).to_crs(CRS_FT)
    b["bin"] = b["bin"].astype(str)
    return b


def load_streets() -> tuple[np.ndarray, np.ndarray]:
    """Give the CSCL centerline geometries (EPSG:2263) and their normalized names."""
    s = pl.read_parquet(STREETS, columns=["full_street_name", "geometry"])
    geoms = shapely.from_wkb(s["geometry"].to_numpy())
    names = np.array([norm_street(x) for x in s["full_street_name"].to_list()], dtype=object)
    return geoms, names


def match_streets(rest: gpd.GeoDataFrame, geoms: np.ndarray, names: np.ndarray) -> dict[str, int]:
    """Give, for each restaurant uid, the nearest CSCL segment with its address street name."""
    pts = rest.geometry.values
    ri, si = shapely.STRtree(geoms).query(pts, predicate="dwithin", distance=STREET_MATCH_FT)
    want = np.array([norm_street(x) for x in rest["street"]], dtype=object)
    same = want[ri] == names[si]
    ri, si = ri[same], si[same]
    dist = shapely.distance(pts[ri], geoms[si])
    order = np.lexsort((dist, ri))
    ri, si = ri[order], si[order]
    first = np.unique(ri, return_index=True)[1]
    uids = rest["uid"].to_numpy()
    return {uids[r]: int(s) for r, s in zip(ri[first], si[first])}


def attach_facades(rest: gpd.GeoDataFrame, bld: gpd.GeoDataFrame) -> pl.DataFrame:
    """Give the wall, S, n and Q of each restaurant, with the rule that found the wall."""
    by_bin = bld.assign(_a=bld.area).sort_values("_a", ascending=False).drop_duplicates("bin")
    by_bin = by_bin.set_index("bin").geometry
    sgeoms, snames = load_streets()
    street_of = match_streets(rest, sgeoms, snames)
    rows = []
    for r in rest.itertuples():
        if r.bin is None or str(r.bin) not in by_bin.index:
            rows.append({"uid": r.uid, "footprint_source": "none"})       # D77: no fallback building
            continue
        poly = by_bin.loc[str(r.bin)]
        si = street_of.get(r.uid)
        got, rule = None, None
        if si is not None:
            got, rule = wall_by_street(poly, r.geometry, sgeoms[si]), "street"
        if got is None:
            got, rule = wall_by_p(poly, r.geometry), ("no_street_wall" if si is not None else "no_street_match")
        if got is None:
            rows.append({"uid": r.uid, "footprint_source": "bin", "wall_rule": "none"})
            continue
        s, n, q, past = got
        rows.append({"uid": r.uid, "footprint_source": "bin", "wall_rule": rule, "p_past_wall_end": past,
                     "street_segment": si, "px": r.geometry.x, "py": r.geometry.y,
                     "sx": float(s[0]), "sy": float(s[1]), "nx": float(n[0]), "ny": float(n[1]),
                     "qx": float(q[0]), "qy": float(q[1]),
                     "p_to_s_ft": float(np.hypot(r.geometry.x - s[0], r.geometry.y - s[1])),
                     "p_inside_footprint": bool(poly.contains(r.geometry))})
    return pl.DataFrame(rows, infer_schema_length=None)


def load_recordings() -> pl.DataFrame:
    """Give the 2025 recordings, with the epoch: Aug-Nov (D58) or Mar-Jun (D78)."""
    frames = []
    for ds in DATASETS:
        lf = pl.scan_parquet(os.path.join(CATALOG, f"dataset={ds}", "**", "*.parquet"))
        frames.append(
            lf.filter((pl.col("face") == "F") & pl.col("manifest_ok")
                      & (pl.col("recordedAt").dt.year() == 2025))
            .select("recording_id", "image_path", "latitude", "longitude", "recordedAt",
                    pl.lit(ds).alias("dataset"))
            .collect())
    rec = pl.concat(frames).sort("dataset").unique("recording_id", keep="first", maintain_order=True)
    start = pl.lit(EPOCH_START).str.to_datetime().dt.replace_time_zone("America/New_York")
    return rec.with_columns(
        pl.when(pl.col("recordedAt") >= start).then(pl.lit("aug_nov")).otherwise(pl.lit("mar_jun")).alias("epoch"))


def cmd_select(args: argparse.Namespace) -> int:
    """Write every (restaurant, recording) pair within 80 ft of P, with all the flags.

    The step `best` applies the limits and picks the view; this step decides
    nothing that a limit can change.
    """
    os.makedirs(args.out, exist_ok=True)
    rest = load_restaurants()
    bld = load_buildings()
    fac = attach_facades(rest, bld)
    fac.write_parquet(os.path.join(args.out, "facades.parquet"))
    print(f"[select] restaurants {rest.shape[0]:,}; footprint sources "
          f"{dict(fac['footprint_source'].value_counts().iter_rows())}; wall rules "
          f"{dict(fac['wall_rule'].value_counts().iter_rows())}")
    fac = fac.filter(pl.col("sx").is_not_null())

    rec = load_recordings()
    print(f"[select] 2025 recordings: {dict(rec['epoch'].value_counts().iter_rows())}")
    rg = gpd.GeoDataFrame(rec.select("recording_id").to_pandas(),
                          geometry=gpd.points_from_xy(rec["longitude"], rec["latitude"]),
                          crs="EPSG:4326").to_crs(CRS_FT)
    pts = gpd.GeoDataFrame(fac.select("uid").to_pandas(),
                           geometry=gpd.points_from_xy(fac["px"], fac["py"]), crs=CRS_FT)
    pts["geometry"] = pts.buffer(MAX_CAM_TO_P_FT)
    j = gpd.sjoin(rg, pts, predicate="within", how="inner")
    pairs = pl.DataFrame({"recording_id": j["recording_id"].to_numpy(), "uid": j["uid"].to_numpy(),
                          "cx": j.geometry.x.to_numpy(), "cy": j.geometry.y.to_numpy()})
    pairs = pairs.join(fac.drop("footprint_source"), on="uid").join(
        rec.select("recording_id", "recordedAt", "epoch", "image_path", "dataset"), on="recording_id")
    pairs = pairs.with_columns(
        (pl.col("cx") - pl.col("sx")).alias("_vx"), (pl.col("cy") - pl.col("sy")).alias("_vy"),
    ).with_columns(
        (pl.col("_vx") ** 2 + pl.col("_vy") ** 2).sqrt().alias("cam_to_s_ft"),
        ((pl.col("cx") - pl.col("px")) ** 2 + (pl.col("cy") - pl.col("py")) ** 2).sqrt().alias("cam_to_p_ft"),
        (pl.col("_vx") * pl.col("nx") + pl.col("_vy") * pl.col("ny")).alias("depth_ft"),
    ).with_columns(
        (pl.col("depth_ft") > 0).alias("in_front"),
        pl.arctan2(pl.col("_vx") * pl.col("ny") - pl.col("_vy") * pl.col("nx"), pl.col("depth_ft"))
        .abs().degrees().alias("incidence_deg"),
        # D74: facade area in the frame, cos(theta) / r^2 = depth / r^3.
        (pl.col("depth_ft").clip(0, None) / pl.col("cam_to_s_ft") ** 3).alias("score"),
        # The camera is between the wall and the street centerline.
        (((pl.col("cx") - pl.col("qx")) * pl.col("nx") + (pl.col("cy") - pl.col("qy")) * pl.col("ny")) < 0)
        .alias("near_half"),
        # Compass bearing from the camera to S: 0 = north, 90 = east.
        ((pl.arctan2(-pl.col("_vx"), -pl.col("_vy")).degrees() + 360) % 360).alias("yaw_deg"),
    ).drop("_vx", "_vy")
    print(f"[select] (restaurant, recording) pairs within {MAX_CAM_TO_P_FT:.0f} ft of P: {pairs.height:,}")

    # D76: line of sight. The length of the segment from the camera to S inside
    # all building footprints, the own footprint included.
    segs = shapely.linestrings(np.stack([np.stack([pairs["cx"].to_numpy(), pairs["sx"].to_numpy()], 1),
                                         np.stack([pairs["cy"].to_numpy(), pairs["sy"].to_numpy()], 1)], 2))
    bgeoms = bld.geometry.values
    hit = shapely.STRtree(bgeoms).query(segs, predicate="intersects")
    inside = np.zeros(pairs.height)
    np.add.at(inside, hit[0], shapely.length(shapely.intersection(segs[hit[0]], bgeoms[hit[1]])))
    pairs = pairs.with_columns(pl.Series("los_inside_ft", inside),
                               pl.Series("line_of_sight", inside <= LOS_MAX_INSIDE_FT))

    # D50: open at capture, an inspection within 180 days of the capture date.
    insp = (pl.read_parquet(os.path.join(DOHMH, "restaurants.parquet"), columns=["uid", "inspection_date"])
            .filter(pl.col("inspection_date").dt.year() > 1900).unique())
    cap = pairs.select("uid", "recording_id", pl.col("recordedAt").dt.replace_time_zone(None).alias("_t"))
    gap = (cap.join(insp, on="uid")
           .with_columns(((pl.col("inspection_date") - pl.col("_t")).dt.total_days().abs()).alias("_gap"))
           .group_by("uid", "recording_id").agg(pl.col("_gap").min().alias("nearest_inspection_days")))
    pairs = pairs.join(gap, on=["uid", "recording_id"], how="left").with_columns(
        (pl.col("nearest_inspection_days") <= OPEN_WINDOW_DAYS).fill_null(False).alias("open_at_capture"))
    pairs.write_parquet(os.path.join(args.out, "candidates.parquet"))

    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "select",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": DATASETS, "epoch_start": EPOCH_START, "max_cam_to_p_ft": MAX_CAM_TO_P_FT,
        "open_window_days": OPEN_WINDOW_DAYS, "los_max_inside_ft": LOS_MAX_INSIDE_FT,
        "street_match_ft": STREET_MATCH_FT, "street_parallel_deg": STREET_PARALLEL_DEG,
        "min_edge_ft": MIN_EDGE_FT,
        "restaurants": int(rest.shape[0]),
        "recordings": dict(rec["epoch"].value_counts().iter_rows()),
        "footprint_sources": dict(pl.read_parquet(os.path.join(args.out, "facades.parquet"))
                                  ["footprint_source"].value_counts().iter_rows()),
        "wall_rules": dict(fac["wall_rule"].value_counts().iter_rows()),
        "pairs": int(pairs.height),
        "pairs_by_epoch": dict(pairs["epoch"].value_counts().iter_rows()),
    }
    with open(os.path.join(args.out, "select_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_best(args: argparse.Namespace) -> int:
    """Apply the limits, and pick the best view of each restaurant.

    Writes `qualifying.parquet` (every qualifying view, in the order of the
    pick, with `rank`) and `best.parquet` (rank 0).
    """
    cand = pl.read_parquet(os.path.join(args.out, "candidates.parquet"))
    rules = ["street"] + (["no_street_match", "no_street_wall"] if args.wall_fallback == "p_facing" else [])
    ok = (pl.col("in_front") & pl.col("line_of_sight") & pl.col("open_at_capture")
          & pl.col("wall_rule").is_in(rules))
    if args.max_incidence is not None:
        ok = ok & (pl.col("incidence_deg") <= args.max_incidence)
    if args.max_p_to_s is not None:
        ok = ok & (pl.col("p_to_s_ft") <= args.max_p_to_s)
    q = cand.filter(ok)
    # D78: Mar-Jun views in the near half join only when no Aug-Nov view in the near half qualifies.
    near_aug = q.filter((pl.col("epoch") == "aug_nov") & pl.col("near_half"))["uid"].unique()
    q = q.filter((pl.col("epoch") == "aug_nov")
                 | (pl.col("near_half") & ~pl.col("uid").is_in(near_aug.implode())))
    keys, desc = ["uid", "score", "incidence_deg", "recording_id"], [False, True, False, False]
    if args.mar_jun == "near_first":
        keys, desc = ["uid", "near_half"] + keys[1:], [False, True] + desc[1:]
    q = q.sort(keys, descending=desc).with_columns(pl.int_range(pl.len()).over("uid").alias("rank"))
    q.write_parquet(os.path.join(args.out, "qualifying.parquet"))
    rest = load_restaurants()
    best = q.filter(pl.col("rank") == 0).join(
        pl.from_pandas(rest.drop(columns="geometry")).select("uid", "camis", "dba", "borough"),
        on="uid", how="left")
    best.write_parquet(os.path.join(args.out, "best.parquet"))

    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "best",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_incidence_deg": args.max_incidence, "max_p_to_s_ft": args.max_p_to_s,
        "wall_fallback": args.wall_fallback, "mar_jun": args.mar_jun,
        "restaurants_with_a_view": int(best.height),
        "best_by_epoch": dict(best["epoch"].value_counts().iter_rows()),
        "best_by_wall_rule": dict(best["wall_rule"].value_counts().iter_rows()),
        "best_near_half": int(best["near_half"].sum()),
        "best_cam_to_s_ft_quantiles": {str(x): float(best["cam_to_s_ft"].quantile(x)) for x in (0.05, 0.5, 0.95)},
        "best_incidence_deg_quantiles": {str(x): float(best["incidence_deg"].quantile(x)) for x in (0.05, 0.5, 0.75, 0.95)},
        "best_by_borough": dict(best["borough"].value_counts().iter_rows()),
    }
    with open(os.path.join(args.out, "best_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


# ---------------------------------------------------------------- the sample

LABEL_CONFIG = """<View>
  <Header value="$dba"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <Choices name="verdict" toName="image" choice="single" required="true" showInline="true">
    <Choice value="Pass"/>
    <Choice value="Fail"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Notes" rows="3" editable="true" maxSubmissions="1"/>
</View>
"""


def cmd_sample(args: argparse.Namespace) -> int:
    best = pl.read_parquet(os.path.join(args.out, "best.parquet"))
    cand = pl.read_parquet(os.path.join(args.out, "qualifying.parquet"))
    drawn = best["uid"].unique().sort().sample(n=args.n, seed=args.seed, shuffle=True)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    rows, tasks = [], []
    for uid in drawn.to_list():
        # The best view first; the next one only when a face of the best is missing.
        opts = cand.filter(pl.col("uid") == uid).sort("rank")
        for rank, o in enumerate(opts.iter_rows(named=True)):
            face_dir = os.path.dirname(o["image_path"])
            try:
                view = render_view(load_faces(face_dir), o["yaw_deg"], args.pitch, args.fov)
            except FileNotFoundError:
                continue
            name = f"{uid}_{o['recording_id']}.jpg"
            Image.fromarray(view).save(os.path.join(img_dir, name), quality=92)
            row = dict(o)
            row["rank_used"] = rank
            row["view_file"] = name
            rows.append(row)
            break
    sample = pl.DataFrame(rows, infer_schema_length=None).join(
        best.select("uid", "camis", "dba", "borough"), on="uid", how="left")
    for r in sample.iter_rows(named=True):
        # The Label Studio local-files root is data/f6_restaurant_depiction.
        rel = os.path.relpath(os.path.join(img_dir, r["view_file"]),
                              os.path.join(ROOT, "data/f6_restaurant_depiction"))
        tasks.append({"data": {"image": f"/data/local-files/?d={rel}",
                               "uid": r["uid"], "recording_id": r["recording_id"], "dba": r["dba"]}})
    sample.write_parquet(os.path.join(args.out, "sample.parquet"))
    with open(os.path.join(args.out, "tasks.json"), "w") as fh:
        json.dump(tasks, fh, indent=1)
    with open(os.path.join(args.out, "label_config.xml"), "w") as fh:
        fh.write(LABEL_CONFIG)
    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__, "step": "sample",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_requested": args.n, "n_rendered": sample.height, "seed": args.seed,
        "pitch_deg": args.pitch, "fov_deg": args.fov, "view_px": VIEW_PX,
        "pool_restaurants": int(best.height),
        "rank_used_counts": dict(sample["rank_used"].value_counts().iter_rows()),
        "by_borough": dict(sample["borough"].value_counts().iter_rows()),
    }
    with open(os.path.join(args.out, "sample_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_render_best(args: argparse.Namespace) -> int:
    """Render the best view of every restaurant, for the model check of the pixels.

    The check reads only the best view of a restaurant (D63, D67): a "No"
    removes the restaurant, and no other view of it replaces the best one. As
    in `sample`, the next view is used only when a face of the best view is
    missing. A shard holds a part of the restaurants. It writes
    `views/shard_<k>/<uid>_<recording_id>.jpg` and `views/index_<k>.parquet`.
    A view that exists on disk is not rendered again, so a shard can start
    again.
    """
    cand = (pl.read_parquet(os.path.join(args.out, "qualifying.parquet"))
            .select("uid", "rank", "recording_id", "yaw_deg", "image_path", "score", "incidence_deg")
            .sort("uid", "rank"))
    uids = cand["uid"].unique().sort()
    mine = uids.gather(np.arange(args.shard, uids.len(), args.num_shards))
    part = cand.filter(pl.col("uid").is_in(mine.implode()))
    shard_dir = os.path.join(args.out, "views", f"shard_{args.shard:03d}")
    os.makedirs(shard_dir, exist_ok=True)
    rows = []
    for (uid,), opts in part.group_by(["uid"], maintain_order=True):
        row = {"uid": uid, "recording_id": None, "view_path": None, "rank_used": None,
               "score": None, "incidence_deg": None, "missing_faces": 0}
        for rank, o in enumerate(opts.iter_rows(named=True)):
            path = os.path.join(shard_dir, f"{uid}_{o['recording_id']}.jpg")
            if not os.path.isfile(path):
                try:
                    view = render_view(LazyFaces(os.path.dirname(o["image_path"])),
                                       o["yaw_deg"], args.pitch, args.fov)
                except FileNotFoundError:
                    row["missing_faces"] += 1
                    continue
                Image.fromarray(view).save(path + ".tmp", format="JPEG", quality=92)
                os.replace(path + ".tmp", path)
            row.update(recording_id=o["recording_id"], view_path=path, rank_used=rank,
                       score=o["score"], incidence_deg=o["incidence_deg"])
            break
        rows.append(row)
    index = pl.DataFrame(rows, schema={"uid": pl.Utf8, "recording_id": pl.Utf8, "view_path": pl.Utf8,
                                       "rank_used": pl.Int32, "score": pl.Float64,
                                       "incidence_deg": pl.Float64, "missing_faces": pl.Int32})
    index.write_parquet(os.path.join(args.out, "views", f"index_{args.shard:03d}.parquet"))
    print(f"[render-best] shard {args.shard}/{args.num_shards}: {index.height:,} restaurants, "
          f"{index['view_path'].is_null().sum():,} with no view, "
          f"{(index['rank_used'] > 0).sum():,} on a later view")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-render")
    c.add_argument("--face-dir", default="/share/ju/cyclomedia/raw/bronx_2025_1k/W0CQ3/W0CQ37T8/faces")
    c.add_argument("--out", default=os.path.join(OUT_DIR, "render_check"))
    s = sub.add_parser("select")
    s.add_argument("--out", default=OUT_DIR)
    b = sub.add_parser("best")
    b.add_argument("--out", default=OUT_DIR)
    b.add_argument("--max-incidence", type=float, default=45.0, help="D81: the cap on theta, degrees")
    b.add_argument("--max-p-to-s", type=float, default=150.0, help="D75, D82: the limit on P to S, ft")
    b.add_argument("--wall-fallback", choices=["p_facing", "drop"], default="p_facing",
                   help="D83: when no wall faces the address street, the wall that faces P, or drop the restaurant")
    b.add_argument("--mar-jun", choices=["compete", "near_first"], default="compete",
                   help="D78: Mar-Jun near-half views compete on the score, or near-half views come first")
    m = sub.add_parser("sample")
    m.add_argument("--out", default=OUT_DIR)
    m.add_argument("--n", type=int, default=100)
    m.add_argument("--seed", type=int, default=20261023)
    m.add_argument("--pitch", type=float, default=VIEW_PITCH_DEG)
    m.add_argument("--fov", type=float, default=VIEW_FOV_DEG)
    r = sub.add_parser("render-best")
    r.add_argument("--out", default=OUT_DIR)
    r.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--pitch", type=float, default=VIEW_PITCH_DEG)
    r.add_argument("--fov", type=float, default=VIEW_FOV_DEG)
    args = ap.parse_args()
    return {"check-render": cmd_check_render, "select": cmd_select, "best": cmd_best, "sample": cmd_sample,
            "render-best": cmd_render_best}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
