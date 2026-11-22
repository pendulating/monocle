#!/usr/bin/env python
"""F6: show how the virtual camera makes a facade view, with 1 real example.

The figure of the appendix (`fig:facade-view` in the FAccT paper) shows the
6 cube faces of 1 recording, the part of each face that the view samples,
and the view that the renderer makes from them. This script writes that
composite, and a JSON with the geometry of the example for the TikZ panels
that sit beside it (the plan view, and the face regions in the frame).

The example is the best view of 1 restaurant of the drawn set of F6 v4. The
default is Mesa Azteca (uid 41548430, recording W0E5VP9H): the yaw of 29.5
degrees puts the frame across the north and the east faces, the viewing
angle is 8.3 degrees, and the camera is 32.7 ft from the storefront.

Use:
    PY=.venv-vllm025cu129/bin/python
    $PY research/scripts/f6_facade_view_example.py                 # the default example
    $PY research/scripts/f6_facade_view_example.py --uid 50129273  # another restaurant

Output (`--out`, default `research/notebooks/facct/figures/facade_view/`):

| File | Content |
|------|---------|
| `facade_view_example.png` | the cube cross with the sampled regions, and the view |
| `facade_view_example.json` | the geometry: yaw, pitch, theta, r, the footprint, P, S, n, Q, C, the centerline, and the face regions of the frame |
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import polars as pl
import shapely
from PIL import Image, ImageDraw, ImageFont

ROOT = "/share/ju/matt/mllmsci"
sys.path.insert(0, os.path.join(ROOT, "research", "scripts"))
import f6_select_storefront_views as f6  # noqa: E402

__version__ = "1.0.0"

BEST = os.path.join(f6.OUT_DIR, "best.parquet")
OUT = os.path.join(ROOT, "research/notebooks/facct/figures/facade_view")
FACE_PX = 256                       # each face in the cross
VIEW_PX = 768                       # the view beside the cross
GAP = 40
# The house colours of the paper (sec/4_results.tex): cread, cscene, csense, cstandard, cverdict, cplain.
FACE_COLOURS = {"F": (73, 133, 115), "R": (86, 116, 179), "B": (229, 114, 100),
                "L": (243, 177, 79), "U": (124, 143, 191), "D": (156, 148, 138)}
# The cross: column and row of each face. U above F, D below F.
CROSS = {"U": (0, 0), "F": (0, 1), "R": (1, 1), "B": (2, 1), "L": (3, 1), "D": (0, 2)}


def face_of_rays(yaw_deg: float, pitch_deg: float, fov_deg: float, size: int):
    """Give the face of each pixel of the view, and the face coordinates it samples."""
    fwd, right, up = f6._basis(yaw_deg, pitch_deg)
    half = math.tan(math.radians(fov_deg) / 2)
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords * half, -coords * half)
    rays = fwd[None, None, :] + u[..., None] * right + v[..., None] * up
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    face = np.full((size, size), "", dtype="<U1")
    face[(ay >= ax) & (ay >= az) & (y > 0)] = "F"
    face[(ay >= ax) & (ay >= az) & (y <= 0)] = "B"
    face[(ax > ay) & (ax >= az) & (x > 0)] = "R"
    face[(ax > ay) & (ax >= az) & (x <= 0)] = "L"
    face[(az > ax) & (az > ay) & (z > 0)] = "U"
    face[(az > ax) & (az > ay) & (z <= 0)] = "D"
    eps = 1e-12
    uv = {
        "F": (x / (ay + eps), z / (ay + eps)),
        "B": (-x / (ay + eps), z / (ay + eps)),
        "R": (-y / (ax + eps), z / (ax + eps)),
        "L": (y / (ax + eps), z / (ax + eps)),
        "U": (x / (az + eps), -y / (az + eps)),
        "D": (x / (az + eps), y / (az + eps)),
    }
    return face, uv


def hull(points: np.ndarray) -> list[list[float]]:
    """Give the convex hull of 2-d points (monotone chain), counterclockwise."""
    pts = sorted({(round(float(p[0]), 4), round(float(p[1]), 4)) for p in points})
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1] + upper[:-1]]


def region_masks(face, uv, n: int) -> dict[str, np.ndarray]:
    """Give, for each face, the pixels of the face (n x n) that the view samples."""
    out = {}
    for f in CROSS:
        m = face == f
        if not m.any():
            continue
        fu, fv = uv[f][0][m], uv[f][1][m]
        col = np.clip(np.round((fu + 1) / 2 * n - 0.5).astype(int), 0, n - 1)
        row = np.clip(np.round((1 - fv) / 2 * n - 0.5).astype(int), 0, n - 1)
        mask = np.zeros((n, n), dtype=bool)
        mask[row, col] = True
        # Close the gaps between the sampled pixels: a 3 x 3 dilation, twice.
        for _ in range(2):
            d = mask.copy()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    d |= np.roll(np.roll(mask, dr, 0), dc, 1)
            mask = d
        out[f] = mask
    return out


def overlay(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Tint the masked pixels and draw the edge of the mask in the full colour."""
    out = img.astype(np.float64)
    c = np.array(colour, dtype=np.float64)
    out[mask] = 0.6 * out[mask] + 0.4 * c
    inner = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            inner &= np.roll(np.roll(mask, dr, 0), dc, 1)
    edge = mask & ~inner
    for _ in range(2):
        d = edge.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                d |= np.roll(np.roll(edge, dr, 0), dc, 1)
        edge = d & mask
    out[edge] = c
    return np.clip(out, 0, 255).astype(np.uint8)


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def compose(faces: dict[str, np.ndarray], masks: dict[str, np.ndarray], view: np.ndarray, out_png: str) -> None:
    n = faces["F"].shape[0]
    w = 4 * FACE_PX + GAP + VIEW_PX
    h = 3 * FACE_PX
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fnt = font(22)
    for f, (c, r) in CROSS.items():
        img = faces[f]
        if f in masks:
            img = overlay(img, masks[f], FACE_COLOURS[f])
        tile = Image.fromarray(img).resize((FACE_PX, FACE_PX), Image.LANCZOS)
        x0, y0 = c * FACE_PX, r * FACE_PX
        canvas.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + FACE_PX - 1, y0 + FACE_PX - 1], outline=(255, 255, 255), width=2)
        label = {"F": "F  north", "R": "R  east", "B": "B  south", "L": "L  west", "U": "U  up", "D": "D  down"}[f]
        tw = draw.textlength(label, font=fnt)
        draw.rectangle([x0 + 6, y0 + 6, x0 + 6 + tw + 10, y0 + 36], fill=(255, 255, 255))
        draw.text((x0 + 11, y0 + 9), label, fill=FACE_COLOURS[f], font=fnt)
    vx = 4 * FACE_PX + GAP
    vy = (h - VIEW_PX) // 2
    canvas.paste(Image.fromarray(view).resize((VIEW_PX, VIEW_PX), Image.LANCZOS), (vx, vy))
    draw.rectangle([vx, vy, vx + VIEW_PX - 1, vy + VIEW_PX - 1], outline=(60, 60, 60), width=3)
    # The arrow from the cross to the view.
    ax0, ax1, ay = 4 * FACE_PX + 6, vx - 6, h // 2
    draw.line([ax0, ay, ax1, ay], fill=(60, 60, 60), width=4)
    draw.polygon([(ax1, ay), (ax1 - 14, ay - 9), (ax1 - 14, ay + 9)], fill=(60, 60, 60))
    canvas.save(out_png, quality=92)


def geometry(uid: str) -> dict:
    """Give the plan geometry of the example in feet, relative to the camera C."""
    best = pl.read_parquet(BEST).filter(pl.col("uid") == uid)
    if best.height != 1:
        raise SystemExit(f"uid {uid} is not in {BEST}")
    r = best.row(0, named=True)
    rest = f6.load_restaurants()
    row = rest[rest["uid"] == uid].iloc[0]
    bld = f6.load_buildings()
    poly = bld[bld["bin"] == str(row["bin"])].geometry.iloc[0]
    sgeoms, _ = f6.load_streets()
    street = sgeoms[int(r["street_segment"])]
    cx, cy = r["cx"], r["cy"]

    def rel(xy):
        return [[round(float(x - cx), 2), round(float(y - cy), 2)] for x, y in xy]

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    window = shapely.box(cx - 90, cy - 90, cx + 90, cy + 90)
    street_part = street.intersection(window)
    if street_part.geom_type == "MultiLineString":
        street_part = max(street_part.geoms, key=lambda g: g.length)
    return {
        "uid": uid, "dba": row["dba"], "bin": str(row["bin"]), "recording_id": r["recording_id"],
        "recorded_at": str(r["recordedAt"]), "yaw_deg": round(float(r["yaw_deg"]), 2),
        "pitch_deg": f6.VIEW_PITCH_DEG, "fov_deg": f6.VIEW_FOV_DEG,
        "theta_deg": round(float(r["incidence_deg"]), 2), "r_ft": round(float(r["cam_to_s_ft"]), 2),
        "cam_to_p_ft": round(float(r["cam_to_p_ft"]), 2), "p_to_s_ft": round(float(r["p_to_s_ft"]), 2),
        "near_half": bool(r["near_half"]), "wall_rule": r["wall_rule"],
        "C": [0.0, 0.0], "P": rel([(r["px"], r["py"])])[0], "S": rel([(r["sx"], r["sy"])])[0],
        "Q": rel([(r["qx"], r["qy"])])[0], "n": [round(float(r["nx"]), 4), round(float(r["ny"]), 4)],
        "footprint": rel(np.asarray(poly.exterior.coords)[:, :2]),
        "centerline": rel(np.asarray(street_part.coords)[:, :2]),
        "view_path": r.get("view_path"), "face_dir": os.path.dirname(r["image_path"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", default="41548430")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    g = geometry(args.uid)
    faces = f6.load_faces(g["face_dir"])
    view = f6.render_view(faces, g["yaw_deg"], g["pitch_deg"], g["fov_deg"], f6.VIEW_PX)
    face, uv = face_of_rays(g["yaw_deg"], g["pitch_deg"], g["fov_deg"], f6.VIEW_PX)
    masks = region_masks(face, uv, faces["F"].shape[0])
    compose(faces, masks, view, os.path.join(args.out, "facade_view_example.png"))

    # The face regions of the frame, as polygons in normalized frame coordinates (x right, y up).
    size = face.shape[0]
    coords = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(coords, -coords)
    regions = {}
    for f in CROSS:
        m = face == f
        if m.any():
            regions[f] = {"share": round(float(m.mean()), 4), "hull": hull(np.stack([u[m], v[m]], 1))}
    g["frame_regions"] = regions
    g["faces_used"] = sorted(regions, key=lambda f: -regions[f]["share"])
    g["script"] = os.path.relpath(__file__, ROOT)
    g["version"] = __version__
    g["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(os.path.join(args.out, "facade_view_example.json"), "w") as fh:
        json.dump(g, fh, indent=1)
    print(json.dumps({k: g[k] for k in ("uid", "dba", "yaw_deg", "theta_deg", "r_ft", "faces_used", "P", "S", "Q", "n")}, indent=1))
    print({f: regions[f]["share"] for f in regions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
