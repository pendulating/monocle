# Cyclomedia Catalog — Implementation Plan

**Status:** proposed — awaiting approval before implementation
**Owner:** mllmsci (shared across all dagspaces)
**Target path:** `dagspaces/common/cyclomedia_catalog/`

## Goal

Replace the per-run `scripts/create_cyclomedia_dataset.py` directory walk with a centralized, queryable backend that lets every dagspace assemble an inference-ready DataFrame by **space × time × face × dataset** in sub-second wall time.

## Problem with the current script

| Problem | Impact |
|---------|--------|
| Walks the NFS tree on every run (~500k recordings across 5 boroughs) | Minutes of wall time per pipeline launch, millions of `stat()` calls per run |
| Manifest JSON re-parsed every run | ~97k file opens per run in `--parse_manifests` mode |
| Catalog CSVs joined ad-hoc in pandas | Column drift, silent mismatches, hardcoded chunk paths in docstrings |
| No spatial primitive | Callers bolt on lat/lon bbox filters in notebooks; no point-in-polygon against community districts |
| No cross-dataset view | Brooklyn + Queens + Manhattan queries require multiple script invocations + manual concat |
| No sanity checks | `recording_id ↔ face` drift, missing faces, manifest/dir mismatches go unnoticed |

## Design

### Two layers

1. **Indexer** (`indexer.py`) — walks `/share/ju/cyclomedia/raw/` with `fd`, parses manifests once, joins the WFS catalog CSVs, emits one canonical Parquet per dataset plus a global `catalog.parquet` union.
2. **Query API** (`catalog.py`) — Polars + `polars-st` backed reader. Accepts a GeoDataFrame/GeoSeries (`within=`), time range (`between=`), face set (`faces=`), and dataset filter (`datasets=`). Returns a `pl.DataFrame` by default; `.to_pandas()` is one call away for callers that still want pandas. The schema matches what `create_cyclomedia_dataset` emits today plus the expanded metadata columns.

### Storage layout

```
/share/ju/cyclomedia/catalog/          # new -- outside /raw, can be rebuilt
  v1/
    catalog.parquet                    # union across datasets, partitioned by dataset/year
    validation_report.parquet          # per-recording sanity check flags
    manifest.json                      # {schema_version, built_at, fd_version, source_mtimes, row_count}
    by_dataset/
      manhattan_2025_1k.parquet
      brooklyn_2025_1k.parquet
      queens_2025_1k.parquet
      bronx_2025_1k.parquet
      si_2025_1k.parquet
      plazas_sample.parquet
      ...
```

### Catalog schema (`catalog.parquet`)

One row per `(recording_id, face)`. All six faces (F/B/L/R/U/D) kept — U/D have `bearing=NULL`. Target 4–6 M rows.

Full metadata capture: every column from the WFS catalog CSV + every non-trivial field from `manifest.json`. Nothing dropped.

**Identity / path**

| Column | Type | Source |
|--------|------|--------|
| `sample_id` | string | `{recording_id}_{face}` |
| `recording_id` | string | manifest `imageId` (fallback: dirname) |
| `face` | cat | derived (F/B/L/R/U/D) |
| `image_path` | string | walk (absolute) |
| `dataset` | cat | walk (e.g. `manhattan_2025_1k`) |
| `group` | string | walk (parent dir, e.g. `W0CDN`) |
| `borough` | cat | **dataset name map** (manhattan/brooklyn/queens/bronx/si). **Caveat:** pulls use a bounding rectangle, so edge recordings may actually lie in an adjacent borough. A downstream reverse-geocode against CD polygons can reclassify if needed, but the stored `borough` stays authoritative-to-the-pull. |

**Spatial**

| Column | Type | Source |
|--------|------|--------|
| `latitude` | float64 | manifest `label` → fallback WFS `lat` |
| `longitude` | float64 | manifest `label` → fallback WFS `lon` |
| `geom_wkb` | binary | WGS84 point (EPSG:4326), consumed by `polars-st` for point-in-polygon |
| `statePlaneX`, `statePlaneY` | float64 | WFS catalog |
| `locationSRS` | cat | WFS catalog (usually `urn:x-ogc:def:crs:EPSG:3857`) |
| `latitudePrecision`, `longitudePrecision` | float32 | WFS catalog |
| `height` | float32 | WFS catalog |
| `heightSystem` | int16 | WFS catalog |
| `heightPrecision` | float32 | WFS catalog |
| `groundLevelOffset` | float32 | WFS catalog |

**Temporal / orientation**

| Column | Type | Source |
|--------|------|--------|
| `recordedAt` | timestamp[ns, tz='US/Eastern'] | WFS catalog |
| `year` | int16 | WFS catalog |
| `recorderDirection` | float32 | WFS catalog (vehicle heading) |
| `yawDegrees` | float32 | WFS catalog |
| `yawPrecisionDegrees` | float32 | WFS catalog |
| `orientation` | float32 | WFS catalog |
| `orientationPrecision` | float32 | WFS catalog |
| `bearing` | float32 | derived: `(recorderDirection + FACE_BEARING_DEG[face]) mod 360`; NULL for U/D |

**Product metadata**

| Column | Type | Source |
|--------|------|--------|
| `productType` | cat | WFS (`Cyclorama` etc.) |
| `panoramaTileSchema` | cat | WFS |
| `tileSchema` | cat | WFS (cross-check against manifest) |
| `hasDepthMap` | bool | WFS |
| `isAuthorized` | bool | WFS |

**Manifest-derived (render provenance)**

| Column | Type | Source |
|--------|------|--------|
| `manifest_zoom` | int8 | manifest `zoom` |
| `manifest_tile_px` | int16 | manifest `tilePx` |
| `manifest_tile_schema` | cat | manifest `tileSchema` |
| `manifest_name_version` | cat | manifest `nameVersion` (e.g. `streetsmart_25.2.0`) |
| `manifest_mode` | cat | manifest `mode` (render/tiles) |
| `manifest_checkpoint` | cat | manifest `checkpoint` (pull-batch tag — cross-check vs `dataset`) |
| `manifest_no_tiles` | bool | manifest `no_tiles` |
| `face_elapsed_s` | float32 | manifest `faces[face].elapsed_s` — per-face render time |
| `face_used_render` | bool | manifest `faces[face].used_render` |
| `depthmap_present` | bool | manifest `depthmaps.faces[face].tiles_present > 0 OR used_render` |
| `depthmap_used_render` | bool | manifest `depthmaps.faces[face].used_render` |
| `depthmap_render_size` | int16 | manifest `depthmaps.faces[face].render_size` |
| `depthmap_rgb_render_size` | int16 | manifest `depthmaps.faces[face].rgb_render_size` |
| `depthmap_downsample_factor` | float32 | manifest `depthmaps.faces[face].downsample_factor` |
| `depthmap_stitched` | bool | manifest `depthmaps.stitched_faces` (recording-level) |

**Index provenance**

| Column | Type | Source |
|--------|------|--------|
| `file_size` | int64 | walk |
| `file_mtime` | timestamp | walk |
| `manifest_ok` | bool | indexer — false if manifest missing/unparseable |
| `catalog_hit` | bool | indexer — true if joined a WFS catalog row |
| `indexed_at` | timestamp | indexer build time |

Partitioning: `by_dataset/dataset={dataset}/year={year}/*.parquet` (pyarrow dataset layout).

### Validation report (`validation_report.parquet`)

One row per recording, plus a summary `.json`. Checks run at index time:

| Check | Column | Pass criterion |
|-------|--------|----------------|
| Face set completeness | `faces_present_mask` | bitmask over F/B/L/R/U/D |
| Expected horizontal faces present | `has_all_horizontal` | F/B/L/R all exist |
| Manifest parse | `manifest_ok` | JSON loads, has `imageId` and `label` |
| `imageId` vs dirname | `imageid_matches_dirname` | catches rsync mismatches |
| Catalog join | `catalog_hit` | |
| Coord plausibility | `coord_in_nyc_bbox` | rough bbox sanity |
| Coord plausibility (tight) | `coord_in_borough` | reverse-geocoded via CD polygons |
| File size plausibility | `any_face_size_too_small` | <50 KB → likely truncated |
| mtime drift | `newest_face_mtime`, `manifest_mtime` | for incremental rebuild |

Summary at build time:
```
manhattan_2025_1k: 97,240 recs | 96,840 (99.6%) fully valid
  manifest_ok:               97,240 (100.0%)
  has_all_horizontal:        96,998 ( 99.8%)
  imageid_matches_dirname:   97,240 (100.0%)
  catalog_hit:               97,001 ( 99.8%)
  coord_in_nyc_bbox:         97,240 (100.0%)
  any_face_size_too_small:        3 (  0.003%)  <-- listed below
  - W0CDN0T5: R.jpg=1.2KB
  - ...
```

### Indexer (`indexer.py`)

Stages:

1. **Walk** with `fd` (`fd --type f --extension jpg --absolute-path . {dataset_root}`) piped into Python. Parse paths to `(dataset, group, recording, face)`. Stat in the same pass (file_size, mtime). ~10-20× faster than Python `os.scandir` for the full tree.
2. **Manifest parse** in parallel (threads, NFS-bound). Cache per-recording.
3. **Catalog load** — read + concat all `recordings_*_chunks/*.csv` + `out_catalog/*.csv`. De-dup on `imageId`. Normalize timezone on `recordedAt`. Emit one long table.
4. **Join** — left join walk+manifest rows to catalog on `recording_id`. Compute `bearing`, `geom_wkb`, validation columns.
5. **Write** — per-dataset partitioned parquet + a union `catalog.parquet` (Polars can query either).
6. **Validate + report** — run sanity checks, emit `validation_report.parquet` and a human-readable `summary.md` in the catalog dir.

**Refresh model: manual only.** No cron / nightly rebuild. The catalog is regenerated (or refreshed incrementally) by a human running the CLI *after* a new Cyclomedia pull lands. Incremental mode re-walks only; compares `(path, mtime, size)` against the existing catalog and rebuilds only affected partitions. Catalog-CSV changes force a full rebuild of their borough.

### Query API (`catalog.py`)

```python
import geopandas as gpd
from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog

cat = CyclomediaCatalog()  # resolves /share/ju/cyclomedia/catalog/v1/ by default

# Spatial + temporal
cd = gpd.read_file("data/geo/nyc_community_districts.geojson").query("boro_cd == 105")
df = cat.query(
    within=cd,                                         # GeoDataFrame/GeoSeries, any CRS (reprojected to 4326 internally)
    between=("2025-05-01", "2025-08-01"),              # recordedAt window
    faces={"F", "B", "L", "R"},                        # default: all six
    datasets=["manhattan_2025_1k"],
)
# -> pl.DataFrame; call .to_pandas() if a caller wants pandas

# Convenience: drop-in replacement for the old script's --output_path mode
cat.build_inference_parquet(
    output_path="data/cyclomedia_manhattan_cd5_scaffolding.parquet",
    within=cd,
    faces={"F", "B", "L", "R"},
)
```

Sketch of the implementation:

```python
import polars as pl
import polars_st as st  # registers the .st namespace on pl.Expr

def query(self, within=None, between=None, faces=None, datasets=None) -> pl.DataFrame:
    lf = pl.scan_parquet(f"{self.root}/by_dataset/**/*.parquet", hive_partitioning=True)

    if datasets:
        lf = lf.filter(pl.col("dataset").is_in(list(datasets)))
    if faces:
        lf = lf.filter(pl.col("face").is_in(list(faces)))
    if between is not None:
        t0, t1 = between
        lf = lf.filter(pl.col("recordedAt").is_between(pl.lit(t0), pl.lit(t1)))
    if within is not None:
        poly = _to_wgs84_union_wkb(within)  # shapely union -> WKB bytes
        lf = lf.filter(
            pl.col("geom_wkb").st.from_wkb().st.within(st.from_wkb(pl.lit(poly)))
        )

    return lf.collect()
```

Under the hood: `pl.scan_parquet(...)` over the partitioned dataset with predicate + projection pushdown. Temporal and face/dataset filters translate to native Polars expressions. The spatial predicate uses **`polars-st`**: `geom_wkb` is decoded lazily via `st.from_wkb(...)` and tested against the input polygon(s) with `.st.within(...)`. Hive partition pruning on `dataset=.../year=.../` skips irrelevant files entirely; the spatial check runs only on surviving rows.

### CLI

```bash
# Full rebuild
python -m dagspaces.common.cyclomedia_catalog.cli build \
    --raw-root /share/ju/cyclomedia/raw \
    --catalog-csv-glob "/share/ju/cyclomedia/pull/**/recordings_*.csv" \
    --output /share/ju/cyclomedia/catalog/v1

# Incremental (walk + diff only)
python -m dagspaces.common.cyclomedia_catalog.cli refresh --output /share/ju/cyclomedia/catalog/v1

# Validate an existing catalog without rebuilding
python -m dagspaces.common.cyclomedia_catalog.cli validate --output /share/ju/cyclomedia/catalog/v1

# One-shot query (replaces create_cyclomedia_dataset.py for common cases)
python -m dagspaces.common.cyclomedia_catalog.cli query \
    --catalog /share/ju/cyclomedia/catalog/v1 \
    --within data/geo/manhattan_cd5.geojson \
    --between 2025-05-01 2025-08-01 \
    --faces F,B,L,R \
    --output data/cyclomedia_cd5.parquet
```

## Sanity checks (testable, baked into the indexer)

These are the invariants `scripts/create_cyclomedia_dataset.py` silently violates today:

1. **`recording_id` ↔ directory**: for every recording, `manifest.imageId == basename(recording_path)`. Log violations; don't silently prefer manifest.
2. **Face ↔ `sample_id` mapping**: `sample_id.endswith(f"_{face}")` and `face ∈ {F,B,L,R,U,D}`.
3. **Face filename ↔ face label**: `basename(image_path)[0] == face`.
4. **One row per `(recording_id, face)`**: unique constraint; indexer asserts before writing.
5. **Catalog join sanity**: warn if `catalog_hit` rate < 95% for any dataset.
6. **Coord sanity**: lat ∈ [40.4, 41.0], lon ∈ [-74.3, -73.6] for NYC datasets.
7. **Bearing**: `0 ≤ bearing < 360`; for U/D faces `bearing IS NULL`.
8. **File presence**: `image_path` resolves and `file_size > 50_000` (JPEG sanity).
9. **Partition completeness**: every (dataset, year) partition has at least one row; schema matches union.
10. **No path escape**: `image_path` starts with the configured `raw_root` (defense against symlink surprises).

All checks run as a pytest suite (`tests/test_cyclomedia_catalog.py`) against a fixture catalog built from `plazas_sample` (small, fast) and as runtime assertions in the indexer (fatal on 1–4, warning on 5–10).

## Dependencies

- `polars` — primary query engine (lazy scan over partitioned parquet)
- `polars-st` — spatial extension for `polars` (GEOS-backed point-in-polygon, `st.from_wkb`, `.st.within`)
- `pyarrow` — already used; powers the parquet writer
- `geopandas`, `shapely` — already in env; used for loading/preparing input polygons that feed the `.st.within(...)` call
- `fd` binary — installed at `/share/ju/matt/.cargo/bin/fd` (v10.4.2). Indexer prepends `/share/ju/matt/.cargo/bin` to `PATH` or calls the binary by absolute path. Keep a threaded `os.scandir` fallback for portability, but the cluster path assumes `fd`.

## Migration path for `scripts/create_cyclomedia_dataset.py`

1. Build v1 catalog once (manual rebuild).
2. Rewrite `create_cyclomedia_dataset.py` as a thin shim calling `CyclomediaCatalog.build_inference_parquet(...)`. Keep the CLI flags identical for backward compatibility.
3. Update `dagspaces/urbanvqa/conf/pipeline/vqa_cyclomedia_scaffolding.yaml` + any pipeline configs that reference the old script output to point at the new catalog-built parquet (path is stable).
4. Deprecate `--parse_manifests` and `--catalog_csv` flags (no-ops; catalog already has everything).

## Decisions (2026-04-17 review)

| # | Decision | Notes |
|---|----------|-------|
| 1 | Catalog lives at `/share/ju/cyclomedia/catalog/v1/` | Outside `/raw`, rebuildable |
| 2 | Polars + `polars-st` for queries | All-Python surface, no SQL layer; `polars-st` handles point-in-polygon via GEOS |
| 3 | Manual refresh only | No cron. CLI is run by a human after each new Cyclomedia pull |
| 4 | Keep **all six** faces (F/B/L/R/U/D); U/D stored with `bearing=NULL` | Default query returns all; callers filter explicitly |
| 5 | `borough` comes from the dataset name | **Caveat baked into docs:** pulls use a bounding rectangle, so edge recordings may lie in a neighboring borough. Downstream code that needs polygon-accurate borough should reverse-geocode against CD polygons at query time, not rely on the stored `borough` |

## Milestones

| # | Deliverable | Rough size |
|---|-------------|-----------|
| 1 | `indexer.py` walks one dataset (`plazas_sample`) end-to-end, emits parquet + validation report | 1 day |
| 2 | Catalog CSV loader + join, bearing computation, partitioned output | 0.5 day |
| 3 | `CyclomediaCatalog` Polars + `polars-st` reader with `within`/`between`/`faces`/`datasets` filters | 0.5 day |
| 4 | pytest suite covering the 10 sanity checks on `plazas_sample` fixture | 0.5 day |
| 5 | Full build across all 5 boroughs + validation report review | 0.5 day (wall), 2–4h compute |
| 6 | Rewrite `scripts/create_cyclomedia_dataset.py` as shim + update one pipeline config end-to-end | 0.5 day |
| 7 | Wiki + CLI reference updates | alongside above |

Total: ~3–4 focused days.
