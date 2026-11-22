#!/usr/bin/env python
"""F4: draw and render the street photography images (D116 to D120).

The rules:

| Item | Rule | Decision |
|------|------|----------|
| Pass rule | none: any random recording passes, with no pixel check | D116 |
| Frame | `--per` recordings for each residential NTA (`NTAType` 0, 197 NTAs); or `--total N`: `N // 197` for each NTA, and 1 more for `N % 197` NTAs at random | D117, D132 |
| Pool | the 2025 Cyclomedia recordings of the F6 catalog | as F6 |
| Epoch | Aug-Nov 2025 (D58). An NTA with fewer than `--per` such recordings takes Mar-Jun ones | as D78 |
| Faces | a recording that lacks 1 of the F, R, B, L, U faces is skipped; the next one in the rank replaces it | as F6 |
| View | the F6 virtual camera: pitch +10, 70 degree field of view, 1024 px | D118, D61 |
| Direction | to the side of the street: the bearing of the nearest CSCL centerline, plus or minus 90 degrees, the side at random | D118 |
| Seed | 20261023, as F6 and F4 | as D88 |

The manifest column `yawDegrees` holds no heading (its values are near
20,600), thus the street direction comes from CSCL. A recording with no
centerline within `--street-ft` is not drawn.

Batch 2 (D132) draws `--total 10000` with the same seed. The rank of each NTA
does not change, thus the first 5 of each NTA are the 985 canonical
recordings. `--exclude-sample` writes only the other rows, and stops when a
recording of that file is not at the top of its NTA. `--idx-offset` moves
`image_idx`. The NTAs that get 1 more come from a second generator
(`default_rng([seed, 1])`), thus the rank keys stay the same. The script
renders only the rows that it writes.

Output:

| File | Content |
|------|---------|
| `data/f4_traces/f4_street_1k.parquet` | `image_idx`, `sample_id` (the recording id), `image_path`, the NTA, the point, the epoch, the yaw |
| `data/f4_traces/f4_street_1k_manifest.json` | the counts and the settings |
| `data/f4_traces/street_views/<recording_id>.jpg` | the rendered views |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/sample_street_images.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/sample_street_images.py --total 10000 \
        --exclude-sample data/f4_traces/f4_street_1k.parquet --idx-offset 1000 \
        --out data/f4_traces/b2/f4_street_b2.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "research", "scripts"))
import f6_select_storefront_views as f6  # noqa: E402

NTA = os.path.join(ROOT, "data/geo/nynta2020_26b/nynta2020.shp")
OUT = os.path.join(ROOT, "data/f4_traces/f4_street_1k.parquet")
VIEWS = os.path.join(ROOT, "data/f4_traces/street_views")
SEED = 20261023

__version__ = "1.1.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=5, help="recordings for each residential NTA")
    ap.add_argument("--total", type=int, default=None,
                    help="recordings in all; in place of --per, as even as possible over the NTAs (D132)")
    ap.add_argument("--exclude-sample", default=None,
                    help="a sample whose recordings must be at the top of their NTA; they are not written")
    ap.add_argument("--idx-offset", type=int, default=0, help="the first image_idx")
    ap.add_argument("--street-ft", type=float, default=60.0, help="the largest distance to a CSCL centerline")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-render", action="store_true", help="draw only; render nothing")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rec = f6.load_recordings().to_pandas()
    pts = gpd.GeoDataFrame(rec, geometry=gpd.points_from_xy(rec.longitude, rec.latitude), crs=4326).to_crs(f6.CRS_FT)
    nta = gpd.read_file(NTA).to_crs(f6.CRS_FT)
    nta = nta[nta.NTAType.astype(str) == "0"][["NTA2020", "NTAName", "BoroName", "geometry"]]
    pts = gpd.sjoin(pts, nta, how="inner", predicate="within").drop(columns="index_right")

    # The street direction at each recording, from the nearest CSCL centerline.
    geoms, _ = f6.load_streets()
    tree = shapely.STRtree(geoms)
    pi, si = tree.query_nearest(pts.geometry.values, max_distance=args.street_ft, return_distance=False,
                                all_matches=False)
    pts = pts.iloc[pi].assign(_street=si)
    print(f"[street] {len(rec):,} recordings; {len(pts):,} in a residential NTA and within "
          f"{args.street_ft:g} ft of a centerline")

    # Rank the recordings of each NTA: Aug-Nov first, then Mar-Jun, then a
    # random key. Take the first `per` of each NTA that have the 5 cube faces
    # a view can need (F, R, B, L, U; pitch +10 never reaches D).
    pts = pts.sort_values("recording_id")
    pts["_key"] = rng.random(len(pts))
    pts["_epoch_rank"] = (pts.epoch != "aug_nov").astype(int)
    ranked = pts.sort_values(["NTA2020", "_epoch_rank", "_key"])
    ntas = sorted(ranked.NTA2020.unique())
    if args.total is None:
        quota = {k: args.per for k in ntas}
    else:
        base, extra = divmod(args.total, len(ntas))
        more = set(np.random.default_rng([args.seed, 1]).choice(ntas, size=extra, replace=False)) if extra else set()
        quota = {k: base + (k in more) for k in ntas}
    keep, skipped = [], 0
    for nta_id, g in ranked.groupby("NTA2020", sort=True):
        n = 0
        for i, path in zip(g.index, g.image_path):
            d = os.path.dirname(path)
            if all(os.path.isfile(os.path.join(d, f"{f}.jpg")) for f in "FRBLU"):
                keep.append(i)
                n += 1
                if n == quota[nta_id]:
                    break
            else:
                skipped += 1
    drawn = ranked.loc[keep].copy()
    got = drawn.groupby("NTA2020").size()
    short = (pd.Series(quota) - got.reindex(list(quota)).fillna(0)).astype(int)
    short = short[short > 0]
    print(f"[street] skipped {skipped} recordings with a missing cube face")

    rows = []
    for p, s in zip(drawn.geometry.values, drawn._street.values):
        t = f6._street_frame(geoms[s], p)
        bearing = math.degrees(math.atan2(t[0], t[1])) % 360   # x = East, y = North
        side = int(rng.choice([-90, 90]))
        rows.append((bearing, side, (bearing + side) % 360))
    drawn[["street_bearing_deg", "side_deg", "yaw_deg"]] = rows
    drawn["_rank"] = drawn.groupby("NTA2020").cumcount()
    drawn = drawn.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    excluded = 0
    if args.exclude_sample:
        old = set(pd.read_parquet(args.exclude_sample, columns=["recording_id"]).recording_id)
        hit = drawn.recording_id.isin(old)
        if hit.sum() != len(old):
            raise SystemExit(f"{len(old) - int(hit.sum())} recordings of {args.exclude_sample} are not in the draw; "
                             "the pool or the rank changed")
        top = drawn[hit].groupby("NTA2020")._rank.max() + 1
        if (top != drawn[hit].groupby("NTA2020").size()).any():
            raise SystemExit(f"a recording of {args.exclude_sample} is not at the top of its NTA")
        excluded = int(hit.sum())
        drawn = drawn[~hit].reset_index(drop=True)
    drawn["image_idx"] = np.arange(args.idx_offset, args.idx_offset + len(drawn), dtype=np.int64)
    # `image_path` of the catalog is the F face; the view gets its own path.
    drawn = drawn.rename(columns={"image_path": "face_path"})
    drawn["image_path"] = [os.path.join(VIEWS, f"{r}.jpg") for r in drawn.recording_id]

    if not args.no_render:
        os.makedirs(VIEWS, exist_ok=True)
        for i, r in enumerate(drawn.itertuples()):
            if os.path.isfile(r.image_path):
                continue
            faces = f6.LazyFaces(os.path.dirname(r.face_path))
            view = f6.render_view(faces, r.yaw_deg, f6.VIEW_PITCH_DEG, f6.VIEW_FOV_DEG)
            Image.fromarray(view).save(r.image_path, quality=95)
            if (i + 1) % 100 == 0:
                print(f"[street] rendered {i + 1}/{len(drawn)}")

    out = drawn.rename(columns={"NTA2020": "nta2020", "NTAName": "nta_name", "BoroName": "borough"})
    out["sample_id"] = out.recording_id
    out["recordedAt"] = out.recordedAt.astype(str)
    cols = ["image_idx", "sample_id", "image_path", "recording_id", "nta2020", "nta_name", "borough",
            "latitude", "longitude", "recordedAt", "epoch", "face_path", "street_bearing_deg", "side_deg", "yaw_deg"]
    out[cols].to_parquet(args.out, index=False)
    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(out), "per_nta": args.per if args.total is None else None, "total": args.total,
        "quota": {str(k): int(v) for k, v in pd.Series(quota).value_counts().sort_index().items()},
        "exclude_sample": args.exclude_sample, "excluded": excluded, "idx_offset": args.idx_offset,
        "ntas": int(out.nta2020.nunique()),
        "ntas_short": {k: int(v) for k, v in short.items()}, "skipped_missing_face": skipped,
        "seed": args.seed,
        "street_ft": args.street_ft, "pitch_deg": f6.VIEW_PITCH_DEG, "fov_deg": f6.VIEW_FOV_DEG,
        "view_px": f6.VIEW_PX, "by_epoch": out.epoch.value_counts().to_dict(),
        "by_borough": out.borough.value_counts().to_dict(),
    }
    with open(args.out.replace(".parquet", "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
