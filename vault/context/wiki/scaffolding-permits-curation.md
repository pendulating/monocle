---
title: Scaffolding Permits Curation
category: curation
created: 2026-04-21
updated: 2026-04-22
tags:
  - curation
  - dob
  - permits
  - scaffolding
  - spatial
  - cyclomedia
  - facing
sources: []
---

# Scaffolding Permits Curation

First concrete "curated sub-dataset" bootstrap: produce a spatial mask of every NYC DOB scaffold/shed permit issued **through** 2025-12-31 — that is, any permit whose issue date is on or before that cutoff, spanning all historical years (earliest in current build: 1989-06-09). All 5 boroughs, regardless of active/expired status. Intersect the resulting coverage mask with [[cyclomedia-catalog]] to carve out a scaffold-adjacent image sub-dataset.

> **Directory naming convention.** `scaffolding_permits_through_<cutoff>/` — the `through_<year>` suffix makes explicit that the year is a **cutoff**, not a year-only filter (i.e. 2024, 2023, 2017, 1995 permits are all included if they were issued on or before the cutoff). When a lower bound is also applied, the dir is `scaffolding_permits_<since>_through_<cutoff>/` (e.g. `scaffolding_permits_2020_through_2025`). CLI flags: `--cutoff YYYY-MM-DD` (upper, default 2025-12-31) and `--since YYYY-MM-DD` (optional lower).

## Built sub-datasets

| Dir | Window | Permits | Polygon match | Coverage | Curated Cyclomedia rows |
|-----|--------|--------:|--------------:|---------:|-----------------------:|
| `scaffolding_permits_through_2025/` | **(none)** → 2025-12-31 (earliest 1989-06-09) | **323,474** | 99.98% | 188.82 km² (24.26% NYC) | 6,296,458 (191 MB) |
| `scaffolding_permits_2020_through_2025/` | **2020-01-01** → 2025-12-31 | **57,492** | 99.97% | 116.23 km² (14.94% NYC) | 3,970,154 (120 MB) |

**`through_2025`** — full historical pull (1989-2025, 37 years). Use when you want maximum recall and are OK with old demolished-building permits contributing to the coverage mask.

**`2020_through_2025`** — modern pull (6 years, post-DOB NOW launch). Use when you want permits for buildings that are plausibly still standing / scaffolds that are plausibly still associated with current street imagery. DOB NOW dominates: 52,985 rows vs 4,516 BIS (BIS mostly pre-2020).

Both sub-datasets pass all 8 fatal validation checks and share the same pipeline / module / test suite. Implementation at `dagspaces/common/curation/permits/`; plan at `docs/plans/scaffolding-permits-curation.md`. See [[guide-compliance-map]] for the sibling DoB cross-reference workflow this builds on.

## Pipeline

```
DOB NOW (w9ak-ipjd) + BIS (ipu4-2q9a)  →  filter on scaffold/shed + issue_date ≤ 2025-12-31
                                       →  normalize to common schema
                                       →  join nyc_buildings.parquet on BIN (polygon) OR fallback point
                                       →  buffer 80 ft in EPSG:2263
                                       →  write permits.parquet + permits.geojson + coverage.geojson
```

`coverage.geojson` is a single dissolved MultiPolygon suitable for `CyclomediaCatalog.query(within=...)`.

## Data sources

| Source | Endpoint | Scaffold filter | Date field |
|--------|----------|-----------------|------------|
| DOB NOW (2020+) | `data.cityofnewyork.us/resource/w9ak-ipjd.json` | `scaffold='1' OR shed='1'` | `first_permit_date` |
| BIS (legacy) | `data.cityofnewyork.us/resource/ipu4-2q9a.json` | `permit_subtype IN ('SH','SD','SF')` | `issuance_date` |

Both fetched unauthenticated (no Socrata app token), paginated at 50k rows, cached to parquet under `curation/scaffolding_permits_through_2025/by_source/`.

## Outputs

| File | Purpose |
|------|---------|
| `permits.parquet` | Flat, one row per permit, WKB geometry in `geom_wkb` column |
| `permits.geojson` | Same rows, WGS84, one feature per permit with all metadata |
| `coverage.geojson` | `unary_union` of every buffered polygon — the consumable spatial mask |
| `by_source/dob_now_raw.parquet`, `bis_raw.parquet` | Raw API responses, post-filter |
| `manifest.json` | Schema version, cutoff date, row counts, git SHA, geom_source breakdown |
| `validation_report.parquet` | One row per permit, boolean column per validation check (drill-downs) |
| `summary.md` | Human-readable validation scoreboard: fatal + warn check pass rates, BIN match rate per source × borough, dropped-permit funnel, top BIN frequencies, coverage area |

## Validation

Modeled on [[cyclomedia-catalog]]'s `validation.py`. Every build runs 8 fatal checks + 12 warn checks against the normalized, buffered frame; fatals refuse to publish, warns log + land in `summary.md`. The headline metric is **BIN → building-polygon match rate** broken out per source × per borough — this is the primary signal for how successful the preprocessing pipeline is at the "100% precision/recall" goal.

| Severity | Checks (abbreviated) |
|----------|----------------------|
| Fatal | Both sources non-empty · no duplicate `(source, permit_id)` · every permit has either BIN-matched polygon or non-null `(lat, lon)` · no null `issue_date` / `scaffold_type` · all buffered geoms valid + inside NYC bbox · `coverage` = `unary_union(permits)` within tolerance |
| Warn | BIN → polygon match rate (per source × borough, ≥85% target) · dropped-permit funnel per preprocessing step · Socrata pagination truncation detection · null `first_permit_date` count · `scaffold_type` + `permit_status` distributions · `issue_date` floor/ceiling outliers · BIN occurrence histogram · cross-source BIN overlap · per-permit buffered-area percentiles · total coverage area + % of NYC land · `geom_source` breakdown |

Full check list and `summary.md` layout: `docs/plans/scaffolding-permits-curation.md`. If a fatal fires, `summary.md` + `validation_report.parquet` are still written so the failure can be diagnosed without re-running the pull.

## Design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Drop DOB NOW filings with null `first_permit_date`** | "Issued through 2025-12-31" means a permit was actually issued. Filings submitted but never issued don't correlate with built scaffolding. |
| 2 | **Include expired + signed-off permits** | Purpose is image retrieval near *any* historical permit, not current compliance. |
| 3 | **Keep BIS and DOB NOW copies separate** | The two sources largely partition by time (pre/post 2020) and carry different metadata. Callers can `.unique(['bin','issue_date'])` if they want. |
| 4 | **80-ft building buffer** | Matches the upper-bound buffer in [[guide-compliance-map]] (~95% camera-to-building match). Chosen for recall on image curation. |
| 5 | **Point fallback for unmatched BIN** | Rows where `permit.bin` is not in `nyc_buildings.parquet` fall back to an 80-ft circle around the API's `(lat, lon)`, tagged `geom_source='point'`. Keeps recall without degrading polygon accuracy elsewhere. |
| 6 | **All 5 boroughs in one file** | Downstream per-borough slicing is a view, not a separate curation output. |
| 7 | **No Socrata app token** | Existing paginated pattern (50k page size) handles throttling. Revisit if 429s appear. |
| 8 | **Client-side date clip for BIS** | The BIS `issuance_date` column is a **plain-text `MM/DD/YYYY` field**, not a Socrata floating timestamp. Server-side `issuance_date <= '2025-12-31T23:59:59'` does lexicographic string comparison and silently passes ~5% of out-of-range rows (`'02/20/2019' < '2025-12-31T23:59:59'` by ASCII order). The pipeline uses a **year-level prune** on the server (`substring(issuance_date, 7, 4) <= 'YYYY'`) to cut payload, then re-clips exactly against the parsed `issue_date` client-side. The dropped-permit funnel in `summary.md` reports how many rows the client-side clip removed. DOB NOW's `first_permit_date` is a real timestamp and does not have this issue. |
| 9 | **`coverage.geojson` as per-polygon features, not a single MultiPolygon** | The dissolved mask can exceed pyogrio's default per-feature size limit (~16 MB) for large runs. Splitting the `unary_union` result's `geoms` into one `Polygon` feature each keeps each feature small without changing semantics — downstream consumers like `CyclomediaCatalog.query(within=...)` re-dissolve via `unary_union` on intake. |
| 10 | **Nearest-building fallback for BIN-miss rows** | Before falling back to an 80-ft-buffered point, try `sjoin_nearest` against `nyc_buildings.parquet` within 200 ft (configurable via `--nearest-max-ft`). Empirically recovers ~99.7% of BIN-miss rows at p50 distance ≈ 36 ft — these are the correct building with a retired or new-construction BIN that DoITT hasn't caught up to yet (5,997 unique missing BINs in the first full build, 100% absent from `nyc_buildings.parquet`). Overall polygon match rate goes from 95.28% (exact BIN only) to ~99.98%. `geom_source` enum expanded to `{bin_polygon, nearest_polygon, point}` and a new `match_dist_ft` column carries the fallback distance for drill-down. Validator's per-source × per-borough table now splits `bin_exact` vs `nearest` so callers can choose stricter filtering if needed. |
| 11 | **Bulk materialization uses `gpd.sjoin`, not `CyclomediaCatalog.query(within=...)`** | `polars-st`'s `st.within` against a literal 5,000+ polygon MultiPolygon has no spatial index — observed 45+ min with zero progress on one borough. The materialize path scans Polars for filters/projection and delegates only the point-in-polygon step to STRtree-backed `gpd.sjoin`, built from the catalog's `latitude`/`longitude` columns (no WKB decode). 38 s end-to-end for all 5 boroughs, 6.3M rows. Catalog's native `query(within=...)` stays the right tool for small-coverage queries (CD polygons in tests); the slow path only hurts at 5K+ polygon coverage scale. Revisit when GeoPolars sjoin ships ([issue #27](https://github.com/geopolars/geopolars/issues/27)). |

## Materialized Cyclomedia sub-dataset

Run via SLURM: `sbatch scripts/materialize_scaffolding_cyclomedia.sub`, wrapping the CLI:

```bash
python -m dagspaces.common.curation materialize-cyclomedia \
    --curation-root curation/scaffolding_permits_through_2025 \
    --faces F B L R
```

**Built 2026-04-21** — `curation/scaffolding_permits_through_2025/cyclomedia_near_permits.parquet`:

| | |
|---|---|
| Rows | **6,296,458** |
| File size | 191 MB |
| Faces | F, B, L, R |
| Elapsed | **38 s** end-to-end on 16 CPUs |
| Chunks kept | `chunks/<dataset>.parquet` per-borough |
| Manifest | `cyclomedia_materialize_manifest.json` |

Per-dataset row counts (hit rate relative to F/B/L/R catalog rows in parens):
- bronx: 1.12M (35%), brooklyn: 2.13M (42%), manhattan: 1.62M (75%), queens: 1.32M (17%), si: 0.11M (4%)

## Why we don't call `CyclomediaCatalog.query(within=...)` for bulk curation

The catalog's ``query(within=...)`` uses ``polars-st.st.within(literal_multipoly)`` which, as of polars-st 0.7.x (and GeoPolars still a prototype; [sjoin issue #27](https://github.com/geopolars/geopolars/issues/27) open since June 2022), has **no spatial index**. For each row it decodes the geom_wkb and does a GEOS containment test against the full 5,388-part coverage MultiPolygon. Observed: a 4.7M-point Bronx chunk made zero progress in 45 minutes.

`dagspaces.common.curation.permits.materialize.sjoin_dataset_chunk` takes a different path: Polars scan + hive partition pruning + face filter (fast, 0.4-1.5s per borough), then GeoDataFrame construction from the catalog's `latitude`/`longitude` columns (skipping WKB decode entirely, 0.8-2.3s), then `gpd.sjoin(..., predicate='within')` which is STRtree-backed via GEOS (1-5s for millions of points × 5,388 polygons). Full 5-borough materialization: 38s. The catalog's `query(within=...)` is still the right tool for small-coverage queries (single CD polygon, etc.) — the slow path only hurts at 5,000+ polygon coverage scale.

Design decision #11 in the table below captures this.

## Orientation filter: keep only faces that look *at* the permit

Full recipe lives in [[concept-facing-filter]]. Summary for this curation family:

**Default behavior** (as of 2026-04-22): `materialize-cyclomedia` automatically runs the new per-unit `filter-facing` after producing the unfiltered parquet, writing a sibling `<name>_facing.parquet`. The filter:

1. Keeps only the closest attribution when a recording sits inside overlapping permit buffers (Fix B).
2. Requires the face's 30-m ray to intersect the row's **own** `unit_uid` polygon, not just any coverage polygon (Fix A — eliminates the "point at a different permit across the street" failure mode).
3. Requires the face to point within 45° of the bearing to the permit's centroid (Fix C, `--facing-bearing-tol-deg`).
4. Caps recording → permit distance at 200 ft (Fix D, `--facing-max-distance-ft`).
5. Emits `attribution_confidence ∈ [0, 1]` per row (Fix E) for downstream weighted sampling.

Pass `--no-facing` to skip the extra step. Both parquets are preserved when facing is on — the unfiltered one is the audit copy, the `_facing` one is the recommended default consumption path.

The filter is also a standalone CLI for post-hoc application:

```bash
python -m dagspaces.common.curation filter-facing \
    --parquet curation/scaffolding_permits_2020_through_2025/cyclomedia_near_permits.parquet \
    --units   curation/scaffolding_permits_2020_through_2025/permits.parquet \
    --out     curation/scaffolding_permits_2020_through_2025/cyclomedia_near_permits_facing.parquet \
    --bearing-tol-deg 45 \
    --max-distance-ft 200
```

The filtered parquet preserves the original schema plus diagnostic columns (`bearing_to_unit_deg`, `delta_bearing_deg`, `distance_to_unit_ft`, `attribution_confidence`) and a sibling `<name>_filter_facing_manifest.json`. U/D faces and null-bearing rows drop unconditionally.

**Legacy mode** (dissolved coverage, no per-unit attribution) is still available: pass `--coverage` without `--units`. Kept for backward compatibility only.

## Sampling images for inspection

Downstream audit / labeling often wants a small inspection folder of actual JPEGs — not just the parquet manifest. `dagspaces.common.curation.sample.sample_images` (CLI: `sample-images`) takes any curated parquet and materializes K images to an inspection dir, with copy or symlink mode.

```bash
# Symlink mode — fast, uses absolute source paths, stays local to this box
python -m dagspaces.common.curation sample-images \
    --parquet curation/scaffolding_permits_2020_through_2025/cyclomedia_near_permits.parquet \
    --out curation/scaffolding_permits_2020_through_2025/inspect_k200_symlink \
    -k 200 --symlink --stratify-by dataset --seed 7

# Copy mode — safe to tar up and move elsewhere
python -m dagspaces.common.curation sample-images \
    --parquet curation/scaffolding_permits_2020_through_2025/cyclomedia_near_permits.parquet \
    --out inspection/scaffolding_2020_k500 \
    -k 500 --stratify-by dataset
```

Output layout:

```
<output_dir>/
  images/
    <dataset>__<sample_id>.jpg     # prefixed to disambiguate cross-dataset dupes
    ...
  manifest.parquet                  # full provenance: every sampled row + export_status
  manifest.json                     # summary: k, seed, mode, counts, elapsed
```

Key behaviors: `--stratify-by COL` splits K evenly across distinct values (common: `dataset`, `face`). `--seed` makes runs reproducible. Missing source files count as `missing` in the manifest (non-fatal) — the tool logs the first few and keeps going. A non-empty `--out` is rejected unless `--force`. Parallel copy/symlink via `--workers N` (default 8). Benchmark: 200 symlinks in 4.5s; copy is I/O-bound on NFS, expect ~10× slower per-file than symlink.

The `manifest.parquet` output has every source column plus `export_filename` + `export_status`, so downstream labeling tools can join labels back into the full curated dataset trivially.

## Related

- [[concept-facing-filter]] — the A–E per-unit facing pipeline (shared with [[facdb-curation]])
- [[cyclomedia-catalog]] — upstream spatial catalog, consumer of `coverage.geojson`
- [[guide-compliance-map]] — sibling DoB cross-reference workflow (scaffold compliance classification)
- [[project-overview]] — overall MLLMSCI framing
