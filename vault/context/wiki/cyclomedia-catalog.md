---
title: Cyclomedia Catalog
category: infrastructure
created: 2026-04-17
updated: 2026-04-21
tags:
  - cyclomedia
  - dataset
  - polars
  - spatial
  - indexing
sources: []
---

# Cyclomedia Catalog

Centralized, queryable index over Cyclomedia panoramic cube-face imagery at `/share/ju/cyclomedia/raw/`. Replaces the per-run NFS walk in `scripts/create_cyclomedia_dataset.py` with a **Polars + `polars-st`** backed catalog that supports spatial (point-in-polygon against NYC community districts), temporal, face, and dataset filters.

**Status: built & QC-cleared (2026-04-21).** Full borough-wide catalog at `/share/ju/cyclomedia/catalog/v1/` — **31,534,741 rows** across `manhattan_2025_1k` (3.24M), `brooklyn_2025_1k` (7.58M), `queens_2025_1k` (11.63M), `bronx_2025_1k` (4.74M), `si_2025_1k` (4.35M). All 4 fatal validation checks pass; WFS catalog hit rate is **100%** on every dataset. Module: `dagspaces/common/cyclomedia_catalog/`. Tests: `tests/test_cyclomedia_catalog.py` (20 tests). Full plan at `docs/plans/cyclomedia-catalog.md`.

Two residual warnings, both benign: (1) 3.49% of rows have `file_size ≤ 50KB` — concentrated almost entirely in `U` (sky) and `D` (ground) faces, which legitimately compress small; F/B/L/R are effectively clean. (2) 128,059 `(recording_id, face)` pairs appear in two adjacent-borough datasets (bbox edge overlap: BK↔Queens 99,563; Bronx↔Manhattan 19,056; BK↔Manhattan 5,252; Manhattan↔Queens 4,188). Within-dataset uniqueness is perfect — scope downstream queries by `dataset` if strict dedup is needed.

## Module layout

| File | Purpose |
|------|---------|
| `schema.py` | `ALL_FACES`, `FACE_BEARING_DEG`, `dataset_to_borough`, `CATALOG_COLUMNS`, `NYC_BBOX`, `SCHEMA_VERSION` |
| `walker.py` | fd-driven directory walk (`/share/ju/matt/.cargo/bin/fd`) with threaded `os.scandir` fallback; emits per-face rows with stat |
| `manifest.py` | Threaded `manifest.json` parser; extracts imageId, label (lat/lon), zoom, tilePx, tileSchema, nameVersion, mode, checkpoint, per-face render + depthmap provenance |
| `wfs.py` | Loads every WFS CSV under `/share/ju/cyclomedia/pull/` (chunked dirs, flat parts, **and** `out_catalog/`), dedupes on imageId, casts types, converts `recordedAt` to `US/Eastern` tz-aware. Loads ~6.3M unique recording_ids covering all five boroughs. |
| `indexer.py` | `build_catalog()` — walks raw tree, parses manifests, calls `_join_wfs_and_derive()` to join + compute `bearing`/`geom_wkb`/`borough`, writes hive-partitioned parquet at `by_dataset/dataset=.../year=.../part-0.parquet`. `rejoin_wfs()` — **fast-path refresh**: reads existing partitions, strips WFS-sourced columns, re-joins against fresh WFS, rewrites. ~4 min for the full 20M-row catalog vs ~10h for a full rebuild. |
| `validation.py` | `run_validation()` — 11 invariants (4 fatal, 7 warn); writes `validation_report.parquet` + `summary.md` |
| `catalog.py` | `CyclomediaCatalog.query(within, between, faces, datasets, years, columns)` — lazy `pl.scan_parquet` + `polars-st` spatial filter |
| `cli.py` | `python -m dagspaces.common.cyclomedia_catalog.cli {build,rejoin-wfs,validate,query}` |

## Why

| Problem in the old script | Fixed here |
|---------------------------|------------|
| Re-walks ~500k recordings across 5 boroughs on every invocation | Walk once at index time; queries hit parquet |
| Re-parses ~97k manifest JSONs on each `--parse_manifests` run | Manifest parsed once, stored as columns |
| Ad-hoc pandas join of WFS catalog CSVs per run | Canonical join baked into the catalog |
| No spatial primitive (callers hand-roll bbox filters) | `within=gdf_cd` on any GeoDataFrame |
| No cross-dataset view | Single `catalog.parquet` unions all boroughs |
| Silent `recording_id ↔ face` mapping drift | 10-check validation report with every build |

See [[urban-vqa]], [[urban-roam-vqa]], [[urban-embed]] — all five dagspaces consume Cyclomedia frames via the old script today.

## Layout

```
/share/ju/cyclomedia/catalog/v1/
  catalog.parquet                  # union across datasets, partitioned
  validation_report.parquet        # sanity-check flags per recording
  manifest.json                    # build metadata: schema_version, built_at, source mtimes
  summary.md                       # human-readable validation summary
  by_dataset/
    dataset=manhattan_2025_1k/year=2025/part-0.parquet
    dataset=brooklyn_2025_1k/year=2025/part-0.parquet
    ...
```

Catalog home is outside `/raw`; can be blown away and rebuilt.

## Catalog schema

One row per `(dataset, recording_id, face)`. All six faces kept (F/B/L/R/U/D); U/D have `bearing=NULL`. **Uniqueness is scoped to the dataset** — a recording whose lat/lon falls inside two borough pull bboxes (~4k recordings along the NYC borough edges) physically exists under both `dataset=X/` and `dataset=Y/` dirs, and the catalog faithfully mirrors that. Queries that should dedupe cross-dataset can call `.unique(subset=["recording_id","face"])`. Cross-dataset overlap count is surfaced by validation warn #11.

Every column from the WFS catalog CSV and every non-trivial field from `manifest.json` is captured — nothing dropped.

**Identity / path**

| Column | Type | Source |
|--------|------|--------|
| `sample_id` | string | `{recording_id}_{face}` |
| `recording_id` | string | manifest `imageId` → fallback dirname |
| `face` | cat | F/B/L/R/U/D |
| `image_path` | string | absolute |
| `dataset`, `group` | cat / string | walk |
| `borough` | cat | **dataset-name map** — see caveat below |

> **Borough caveat:** pulls use a lat/lon bounding rectangle, so recordings near borough edges may actually sit in a neighboring borough. The stored `borough` reflects the pull batch, not geography. Callers who need polygon-accurate borough should reverse-geocode against CD polygons at query time.

**Spatial**

| Column | Type | Source |
|--------|------|--------|
| `latitude`, `longitude` | float64 | manifest `label` → fallback WFS |
| `geom_wkb` | binary | WGS84 point (EPSG:4326); consumed by `polars-st` for point-in-polygon |
| `statePlaneX`, `statePlaneY` | float64 | WFS |
| `locationSRS` | cat | WFS |
| `latitudePrecision`, `longitudePrecision` | float32 | WFS |
| `height`, `heightPrecision`, `groundLevelOffset` | float32 | WFS |
| `heightSystem` | int16 | WFS |

**Temporal / orientation**

| Column | Type | Source |
|--------|------|--------|
| `recordedAt` | timestamp tz=US/Eastern | WFS |
| `year` | int16 | WFS |
| `recorderDirection` | float32 | WFS (vehicle heading — direction of travel, **not** camera yaw) |
| `yawDegrees`, `yawPrecisionDegrees` | float32 | WFS (camera yaw, cumulative) |
| `orientation`, `orientationPrecision` | float32 | WFS (camera yaw in radians; ≈0 across 100% of NYC rows, see caveat below) |
| `bearing` | float32 | `FACE_BEARING_DEG[face]` (absolute compass bearing — F=0°/N, R=90°/E, B=180°/S, L=270°/W). NULL for U/D. |

> **Face bearing caveat (tightened 2026-04-22):** Cyclomedia's NYC cube faces are rendered in a **globally-oriented absolute frame**, not a vehicle-relative one. The `orientation` column (camera yaw) is within ±0.2° of 0° across 100% of rows, confirming every panorama's F-face reference axis is fixed to North. The earlier `(recorderDirection + offset) mod 360` formula treated the cube as vehicle-relative and produced bearings that were wrong by the vehicle heading; the fix lives in `indexer.py::_compute_bearing`. See [[concept-facing-filter]] for the downstream consequences.

**Product metadata**

| Column | Type | Source |
|--------|------|--------|
| `productType`, `panoramaTileSchema`, `tileSchema` | cat | WFS |
| `hasDepthMap`, `isAuthorized` | bool | WFS |

**Manifest-derived render provenance**

| Column | Type | Source |
|--------|------|--------|
| `manifest_zoom`, `manifest_tile_px` | int8 / int16 | manifest |
| `manifest_tile_schema`, `manifest_name_version`, `manifest_mode` | cat | manifest |
| `manifest_checkpoint` | cat | manifest (pull-batch tag — cross-check vs `dataset`) |
| `manifest_no_tiles` | bool | manifest |
| `face_elapsed_s`, `face_used_render` | float32 / bool | manifest `faces[face]` |
| `depthmap_present`, `depthmap_used_render`, `depthmap_stitched` | bool | manifest `depthmaps.faces[face]` |
| `depthmap_render_size`, `depthmap_rgb_render_size` | int16 | manifest |
| `depthmap_downsample_factor` | float32 | manifest |

**Index provenance**

| Column | Type | Source |
|--------|------|--------|
| `file_size`, `file_mtime` | int64 / ts | walk |
| `manifest_ok`, `catalog_hit` | bool | indexer |
| `indexed_at` | ts | build time |

## Query API

```python
from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog
import geopandas as gpd

cat = CyclomediaCatalog()  # defaults to /share/ju/cyclomedia/catalog/v1

cd5 = gpd.read_file("data/geo/nyc_community_districts.geojson").query("boro_cd == 105")
df = cat.query(
    within=cd5,                              # GeoDataFrame/GeoSeries, any CRS
    between=("2025-05-01", "2025-08-01"),    # recordedAt window
    faces={"F", "B", "L", "R"},
    datasets=["manhattan_2025_1k"],
)

# Drop-in replacement for the old script's --output_path mode
cat.build_inference_parquet(
    output_path="data/cyclomedia_cd5.parquet",
    within=cd5,
    faces={"F", "B", "L", "R"},
)
```

`pl.scan_parquet(..., hive_partitioning=True)` drives the query with predicate + projection pushdown; hive-partition pruning on `dataset=.../year=...` skips irrelevant files. The `within=` polygon is unioned into a single WKB and tested via `pl.col("geom_wkb").st.from_wkb().st.within(...)` from `polars-st`.

## Indexer pipeline

Internally split into two functions so the expensive walk/manifest phases can be skipped when only the WFS inputs change:

1. **Walk** `/share/ju/cyclomedia/raw/{dataset}/{group}/{recording}/faces/*.jpg` via `fd` (`fd --type f --extension jpg --absolute-path . $root`), piped into Python. Parse path → `(dataset, group, recording, face)`. Stat in same pass. **(~4h for brooklyn @ 1.26M recordings on NFS.)**
2. **Manifest parse** — threaded read of `manifest.json` per recording; cache `imageId`, `label` → lat/lon, per-face render + depthmap provenance. **(~30–60 min per borough.)**
3. **`_explode_walk_with_manifests`** — join walk × manifests, pick per-face manifest columns, resolve `recording_id`.
4. **`_join_wfs_and_derive`** — load all `/share/ju/cyclomedia/pull/` WFS CSVs (chunks + flat parts + `out_catalog/`), dedup on `imageId`, normalize `recordedAt` TZ. Left-join on `recording_id`; compute `bearing`, `geom_wkb`, `year`, `latitude`/`longitude` coalesce, `catalog_hit`. **(Seconds to low minutes per borough.)**
5. **Write** — pyarrow dataset partitioned by `dataset/year` at `by_dataset/dataset=.../year=.../part-0.parquet`.
6. **Validate** — run sanity checks, emit `validation_report.parquet` + `summary.md`.

**Fast-path refresh (`rejoin_wfs`)**: when only the WFS layer changes (new CSVs land, or the glob is fixed), steps 1–3 are skipped entirely. The function reads existing partitions, strips WFS-derived columns, renames current `latitude`/`longitude` back to the manifest-source columns, and jumps straight into step 4 via `_join_wfs_and_derive`. Full 5-borough / 20M-row rewrite runs in ~4 min on a login node.

**Refresh model: manual only** — no cron, no nightly rebuild. A human runs the CLI after each new Cyclomedia pull lands. Incremental mode (walk + partition diff) is still TODO.

## Sanity checks

Run at build time; failures either fatal (1–4) or warnings (5–11).

| # | Invariant | Severity |
|---|-----------|----------|
| 1 | `manifest.imageId == basename(recording_path)` | fatal |
| 2 | `sample_id == f"{recording_id}_{face}"` and `face ∈ {F,B,L,R,U,D}` | fatal |
| 3 | `basename(image_path)[0] == face` | fatal |
| 4 | One row per `(dataset, recording_id, face)` (unique) | fatal |
| 5 | `catalog_hit` rate ≥ 95% per dataset | warn |
| 6 | NYC lat/lon bbox plausibility | warn |
| 7 | `0 ≤ bearing < 360` (NULL for U/D) | warn |
| 8 | `file_size > 50_000` bytes (truncated JPEG sniff) | warn |
| 9 | Every `(dataset, year)` partition non-empty, schema consistent | warn |
| 10 | `image_path` starts with configured `raw_root` (no symlink escape) | warn |
| 11 | Cross-dataset `(recording_id, face)` overlap count | warn |

Check #4 used to be `unique(recording_id, face)` (no `dataset`), but the pull pipeline uses a lat/lon bounding rectangle per borough, so ~4k recordings legitimately appear in two borough dirs on disk (bronx↔manhattan, brooklyn↔manhattan, brooklyn↔queens — SI has no neighbors, zero overlaps). Scoping uniqueness to `(dataset, recording_id, face)` is honest about what's on disk; check #11 surfaces the overlap count so it stays visible. Callers that want a single copy per recording should `.unique(subset=["recording_id","face"])` at query time.

Results surfaced in `summary.md` (current full-borough build: all fatals PASS, hit rate 100% across all 5 datasets, 24,956 cross-dataset overlap pairs, 3.19% rows with `file_size ≤ 50KB`).

## Dependencies

- **`polars`** — lazy scan over partitioned parquet, predicate pushdown.
- **`polars-st`** — GEOS-backed spatial expressions (`st.from_wkb`, `.st.within`, etc.).
- **`pyarrow`, `geopandas`, `shapely`** — already in env; used to load input polygons and write the partitioned parquet.
- **`fd` binary** — installed at `/share/ju/matt/.cargo/bin/fd` (v10.4.2). Indexer calls it by absolute path (or after prepending `/share/ju/matt/.cargo/bin` to `PATH`). A threaded `os.scandir` fallback exists but the cluster path assumes `fd`.

See [[shared-infrastructure]] for where this fits in `dagspaces/common/`.

## Migration

1. Build `v1/` catalog manually. Full-borough build ran in ~9.7h (2026-04-19 → 2026-04-20) as `sbatch scripts/build_cyclomedia_catalog.sub` on the `ju` partition — dominated by fd walks and manifest parses over NFS.
2. **Still TODO:** rewrite `scripts/create_cyclomedia_dataset.py` as a thin shim calling `CyclomediaCatalog.build_inference_parquet(...)`. Same CLI surface; `--parse_manifests` and `--catalog_csv` become no-ops.
3. **Still TODO:** update pipeline configs that reference script output paths — only the **build command** changes, not the output schema.
4. All five dagspaces ([[urban-vqa]], [[urban-ocr]], [[urban-pair-vqa]], [[urban-roam-vqa]], [[urban-embed]]) consume the same parquet shape as before.

## CLI

```bash
# One-time full build (SLURM)
sbatch scripts/build_cyclomedia_catalog.sub
# …or directly:
python -m dagspaces.common.cyclomedia_catalog.cli build \
    --raw-root /share/ju/cyclomedia/raw \
    --output /share/ju/cyclomedia/catalog/v1 \
    --datasets manhattan_2025_1k brooklyn_2025_1k queens_2025_1k bronx_2025_1k si_2025_1k

# Fast-path refresh: re-run WFS join only (minutes, login-node friendly).
# Use when WFS CSVs change or the glob is expanded — skips all walks and
# manifest parses.
python -m dagspaces.common.cyclomedia_catalog.cli rejoin-wfs \
    --output /share/ju/cyclomedia/catalog/v1 \
    --datasets manhattan_2025_1k brooklyn_2025_1k queens_2025_1k bronx_2025_1k si_2025_1k

# Validate without rebuilding
python -m dagspaces.common.cyclomedia_catalog.cli validate --output /share/ju/cyclomedia/catalog/v1

# Query → parquet
python -m dagspaces.common.cyclomedia_catalog.cli query \
    --within data/geo/manhattan_cd5.geojson \
    --between 2025-05-01 2025-08-01 \
    --faces F,B,L,R \
    --output-path data/cyclomedia_cd5.parquet
```

## Decisions

| # | Decision | Locked |
|---|----------|--------|
| 1 | Catalog path: `/share/ju/cyclomedia/catalog/v1/` | 2026-04-17 |
| 2 | Query engine: **Polars + `polars-st`** — all-Python, no SQL layer; GEOS-backed point-in-polygon | 2026-04-17 |
| 3 | Refresh: manual only, triggered after each Cyclomedia pull | 2026-04-17 |
| 4 | Keep all six faces (U/D `bearing=NULL`) | 2026-04-17 |
| 5 | `borough` from dataset name; polygon-accurate reverse geocode is a caller responsibility | 2026-04-17 |
| 6 | WFS CSV glob includes `/share/ju/cyclomedia/pull/out_catalog/recordings_*.csv` — only place brooklyn + staten island WFS data lives | 2026-04-20 |
| 7 | Uniqueness scoped to `(dataset, recording_id, face)`, not `(recording_id, face)`. Borough bbox edge overlaps (~4k recordings) produce legitimate cross-dataset duplicates; surface them via warn #11 rather than rejecting the build. | 2026-04-20 |
| 8 | Provide `rejoin_wfs()` fast path so WFS-only fixes don't pay the ~10h walk cost. | 2026-04-20 |

## See also

- `docs/plans/cyclomedia-catalog.md` — full implementation plan with milestones
- [[shared-infrastructure]] — where this module lives
- [[file-map]] — project layout
- [[concept-street-graph]] — downstream consumer (uses `recordedAt` + `recorderDirection` for trajectory edges)
- [[concept-facing-filter]] — downstream orientation filter consuming `face`, `bearing`, `latitude`, `longitude`
