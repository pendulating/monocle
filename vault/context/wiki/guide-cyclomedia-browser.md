---
title: Cyclomedia Browser (marimo)
category: guide
created: 2026-07-12
updated: 2026-07-12
tags:
  - cyclomedia
  - marimo
  - duckdb
  - notebook
  - depth
  - panorama
sources: []
---

# Cyclomedia Browser (marimo)

Interactive explorer over the materialized [[cyclomedia-catalog]] — click the citywide map, find nearby recordings, view their cube faces, panoramas, and depth maps.

**Notebook:** `notebooks/cyclomedia/browser.py`
**Module:** `dagspaces/common/cyclomedia/` (`catalog.py`, `cubemap.py`, `depth.py`)
**Tests:** `tests/test_cyclomedia_browser.py` (18 tests, ~2 s)

```bash
marimo edit notebooks/cyclomedia/browser.py
```

Scale: **5,239,918 recordings / 31.5M cube faces**, all five boroughs, Mar–Nov 2025, 100% depth coverage.

## Sections

| # | Section | What it does |
|---|---------|--------------|
| 1 | Filter | Borough / month / has-depth → a `WHERE` clause threaded through every downstream query |
| 2 | Citywide map | Recordings binned to a density grid (100 m–1 km cells) on a Carto basemap. Box/lasso-select → query point |
| 3 | Find nearby | k-nearest recordings within a radius of the query point; results map + picker table |
| 4 | Viewer | Metadata stats, unfolded cube cross, equirectangular panorama |
| 5 | Depth | Depth cube cross, depth panorama, per-face code stats, RGB-vs-depth side-by-side |
| 6 | SQL | Free-form DuckDB against `recordings` or the raw catalog parquet |

## Architecture

Thin notebook, testable module. All heavy lifting lives in `dagspaces/common/cyclomedia/`.

### `catalog.py` — DuckDB layer

DuckDB reads the hive-partitioned catalog parquet directly; a full 31.5M-row scan takes ~5 s, so nothing needs preloading.

A **recording-level index** (`build_recording_index`) collapses the face-level catalog to one row per recording and caches it to `data/cyclomedia/browser/recordings_v1.parquet` (5.24M rows, 160 MB, ~10 s to build, then ~1 s to load). This is what the map needs — 6 face rows per point would be pure duplication.

Cross-dataset duplicates (the 128k `(recording_id, face)` pairs from borough bbox overlap, see [[cyclomedia-catalog]]) are collapsed by keeping the alphabetically-first `dataset`, so `recording_id` is unique in the index.

| Function | Purpose |
|----------|---------|
| `build_recording_index` / `load_recording_index` | Build + cache the recording-level index; register as an in-memory table |
| `nearest_recordings(lat, lon, k, radius_m)` | What a map click resolves to. Bbox prefilter → equirectangular distance |
| `recordings_in_bbox` / `count_in_bbox` | Box query with a render cap (**samples**, does not truncate) |
| `overview_grid(cell_m)` | Density grid for the citywide view (250 m → ~11.5k cells) |
| `faces_for_recording` | Face-level rows (paths, bearings, sizes) — partition-scoped by `dataset` |
| `recording_dir(dataset, group, recording_id)` | Path construction — **never** a glob |

> **Never walk the image tree.** Directory listing on this NFS mount is glacial — a `find` over one dataset times out at two minutes. Every path is *constructed* from catalog fields. This is why `recording_dir()` exists.

### `cubemap.py` — cube geometry

Faces are 90°-hfov rectilinear renders at yaw `F=0 R=90 B=180 L=270`, pitch `U=+90 D=-90`. The render yaw is an absolute compass bearing, so the cube is **north-referenced — `F` always points true north**, regardless of vehicle heading. This independently reproduces `FACE_BEARING_DEG` in [[concept-street-graph]].

Face orientations are *derived* from the render parameterization, not guessed:

```
fwd   = ( sin t cos p,  cos t cos p,  sin p )
right = ( cos t,       -sin t,        0     )    # normalize(fwd x z_up)
up    = (-sin t sin p, -cos t sin p,  cos p )    # right x fwd
ray   = fwd + u*right + v*up                      # u,v in [-1,1], hfov=90
```

`right` is independent of pitch, so this stays well-defined at the `p = ±90` poles. A consequence worth knowing: the **top of the `D` face points north, and the top of the `U` face points south**. That falls out of the shared (yaw, pitch) convention — it is not an arbitrary choice.

Verified two ways: `verify_face_orientations()` measures seam continuity and the derived convention beats *every* rotation/flip variant of U/D; and `test_equirect_places_faces_by_compass_bearing` paints each face a solid color and asserts it lands at the right compass bearing.

| Function | Purpose |
|----------|---------|
| `cube_cross(faces)` | Unfolded cross: `U` / `L F R B` / `D` |
| `cube_to_equirect(faces, width)` | Equirectangular panorama. Column 0 = north, increasing east; middle row = horizon |
| `verify_face_orientations(faces)` | Seam-continuity metric (regression guard on the convention) |

`cube_to_equirect` is dtype- and channel-agnostic: `(H,W,3) uint8` RGB gives an RGB panorama, `(H,W) uint16` depth codes give a code panorama. That matters — see below.

### `depth.py` — depth decoding

See [[concept-cyclomedia-depth-maps]] for the encoding and the metric calibration. Short version: `code = R*256 + G`, `0` = no return, and `to_metres()` gives Euclidean range along the pixel ray as `(code - 16384) / 250` — 4 mm per code, 0–196.6 m. The module also keeps the raw code and relative colorization, which is what the viewer renders (a shared percentile stretch reads better than absolute metres across faces).

## Gotchas worth remembering

- **Stitch codes, then colorize — not the reverse.** Colorizing each depth face first gives every face its own percentile stretch, making them mutually incomparable. Build the code panorama, then apply one shared scale. Same reason the notebook computes `vmin/vmax` across *all six* faces at once.
- **Nearest-neighbour sampling is required for depth**, not a shortcut. Interpolating across a depth discontinuity invents surfaces that do not exist, and blends the `0` no-return sentinel into real distances.
- **DuckDB applies `USING SAMPLE` before `WHERE`.** Sampling in the same `SELECT` as a bbox filter draws from all 5.2M rows and leaves only the handful that happen to land in the box. The sample must wrap the already-filtered subquery. (Caught in `recordings_in_bbox`: it returned 20 rows instead of 5000.)
- **`QUALIFY` needs a window function** in DuckDB — a plain computed-column filter has to go in a subquery.
- **`dagspaces` is not installed into the venv.** Notebooks insert the repo root on `sys.path` (same pattern as `notebooks/roaming/network_validation.py`); marimo runs with the notebook's directory as cwd.

## Dependencies

Added to `pyproject.toml`: `duckdb>=1.5.4`, `plotly>=6.9.0` (Scattermap on a Carto basemap needs no API token; `mo.ui.plotly` exposes box/lasso selection back to Python).

## See also

- [[cyclomedia-catalog]] — the catalog this reads
- [[concept-cyclomedia-depth-maps]] — depth encoding + the open metric-calibration question
- [[concept-street-graph]] — the other consumer of the north-referenced face frame
- [[concept-facing-filter]] — uses `face` / `bearing` / lat / lon from the same catalog
