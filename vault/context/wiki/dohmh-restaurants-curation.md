---
title: DOHMH Restaurants Curation
category: infrastructure
created: 2026-04-28
updated: 2026-04-28
tags:
  - curation
  - dohmh
  - restaurants
  - poi
  - cyclomedia
  - facing
sources: []
---

# DOHMH Restaurants Curation

Third curation family after [[scaffolding-permits-curation]] and [[facdb-curation]]. Pulls NYC restaurants from the **DOHMH Restaurant Inspection Results** dataset (Socrata `43nn-pn8j`, ~31k unique CAMIS as of 2026-04-28), used as a proxy for **every restaurant in NYC**. Optionally filter by `cuisine_description` and/or borough.

The raw dataset is at the **(inspection × violation)** granularity (~296k rows). This curation family is **two steps**:

1. **`dohmh-restaurants` build** — collapses violation-level multiplication only. Output is **inspection-level** (one row per `(camis, inspection_date)`). Multi-year inspection history is preserved on purpose so downstream consumers can plot grade trajectories or compute inspection cadence.
2. **`aggregate-restaurants`** — opt-in second step that collapses to **one row per CAMIS**, picking the most-recent non-placeholder inspection's metadata as the row's headline state and emitting `n_inspections`, `n_grade_a/b/c`, `first_inspection_date` aggregates. Required before `materialize-cyclomedia` (which needs unique unit IDs for the spatial join).

Each curation lands in `curation/dohmh_<slug>/` with the same contract as the other curation families: `restaurants.parquet`, `restaurants.geojson`, `coverage.geojson`, `manifest.json`, `summary.md`, `validation_report.parquet`, `by_source/dohmh_raw.parquet`. Aggregation adds `restaurants_aggregated.parquet` + `restaurants_aggregated_manifest.json`.

## CLI

```bash
# Step 1: build → inspection-level rows (every restaurant, every inspection)
python -m dagspaces.common.curation dohmh-restaurants \
    --out curation/dohmh_pizza --cuisine "Pizza"

# Step 2: aggregate → one row per CAMIS (opt-in)
python -m dagspaces.common.curation aggregate-restaurants \
    --parquet curation/dohmh_pizza/restaurants.parquet
# → writes curation/dohmh_pizza/restaurants_aggregated.parquet

# Step 3: materialize Cyclomedia images near restaurants
sbatch --export=ALL,OUTPUT_FILENAME=cyclomedia_near_restaurants.parquet \
    scripts/materialize_scaffolding_cyclomedia.sub \
    curation/dohmh_pizza
```

Other build examples:

```bash
# All NYC restaurants (no filter) — ~31k CAMIS, ~296k inspection rows
python -m dagspaces.common.curation dohmh-restaurants \
    --out curation/dohmh_restaurants_all

# Multi-cuisine + multi-borough (AND across levels, OR within a level)
python -m dagspaces.common.curation dohmh-restaurants \
    --out curation/dohmh_asian_manhattan \
    --cuisine "Chinese" "Japanese" "Korean" "Thai" \
    --borough Manhattan

# Drop placeholder inspection rows (1900-01-01 sentinel for "registered but never inspected")
python -m dagspaces.common.curation dohmh-restaurants \
    --out curation/dohmh_inspected_only --drop-placeholder-only
```

Filter values are **validated against the frozen vocab at startup** (`dagspaces/common/curation/dohmh/cuisines.json`, baked from a 2026-04-28 distinct-values pull). Typos fail fast with suggestions — e.g. `--cuisine "mxican"` → `did you mean: ['Mexican', 'American', 'African']`.

## Pipeline

```
Socrata 43nn-pn8j  →  fetch (cuisine + borough where clause)
                   →  normalize (collapse violation-level only;
                                  one row per (camis, inspection_date);
                                  most-Critical violation row wins)
                   →  drop bad-geocode rows (no BIN AND lat/lon outside NYC)
                   →  drop null facname (DBA missing)
                   →  attach_geometry (shared geom.py: BIN-exact → nearest-200ft → point)
                   →  buffer 80 ft in EPSG:2263 → WGS84
                   →  validate (7 fatal + warn checks)
                   →  write restaurants.parquet / .geojson + coverage.geojson

[optional, opt-in]
restaurants.parquet  →  aggregate (group_by camis, most-recent real inspection wins)
                     →  write restaurants_aggregated.parquet
```

Shares `dagspaces/common/curation/geom.py` (3-stage BIN match) with [[facdb-curation]] and [[scaffolding-permits-curation]] — the same column-name parameters (`id_col=permit_id`, `bin_col=bin`, `lat_col=raw_latitude`, `lon_col=raw_longitude`) let DOHMH reuse the polygon-buffering machinery.

## Output schema

The columns are deliberately **a superset of the FacDB schema** so an aggregated `restaurants_aggregated.parquet` row drops into any tooling that already speaks FacDB.

### Build output (`restaurants.parquet`, inspection-level)

| Column | Notes |
|---|---|
| `uid` | = `camis` — **NOT unique** in this parquet (one row per inspection) |
| `permit_id` | Aliased from `camis` for the shared `geom.attach_geometry` API |
| `sample_id` | `"{camis}_{inspection_date_iso}"` — unique per row |
| `camis`, `dba` | Native DOHMH names |
| `facname` | = `dba` (FacDB-shaped alias) |
| `address`, `building`, `street`, `city`, `zipcode`, `borough` | Identity |
| `cuisine_description` | Native DOHMH column |
| `facdomain`, `facgroup`, `facsubgrp`, `factype` | FacDB-shaped hierarchy aliases — `facdomain="FOOD SERVICE"`, `facgroup="RESTAURANTS"`, `facsubgrp=factype=upper(cuisine_description)` |
| `inspection_date`, `is_placeholder_inspection`, `inspection_type`, `action`, `grade`, `score`, `grade_date`, `critical_flag`, `violation_code`, `violation_description` | This row's inspection. `inspection_date` is null + `grade='NOT YET INSPECTED'` for placeholder rows. |
| `record_date` | DOHMH dataset's own record_date |
| `bin`, `bbl`, `latitude`, `longitude` | Spatial keys |
| `raw_latitude`, `raw_longitude` | Aliases (match shared geom API) |
| `nta`, `community_board`, `council_district`, `census_tract` | Admin geographies |
| `datasource` | Fixed: `"dohmh:43nn-pn8j"` |
| `geom_source` ∈ {`bin_polygon`, `nearest_polygon`, `point`} | Match stage |
| `match_dist_ft` | Distance for nearest fallback |
| `geom_wkb` | Buffered polygon as WKB binary (parquet) |

### Aggregate output (`restaurants_aggregated.parquet`, one row per CAMIS)

Same schema as the build, with these changes:

| Column | Notes |
|---|---|
| `uid` (= `camis`) | Now **unique** per row |
| `last_inspection_date` | Replaces `inspection_date` — most-recent real inspection's date |
| `last_inspection_type`, `last_action`, `last_grade`, `last_score`, `last_critical_flag`, `last_violation_code`, `last_violation_description` | All renamed with `last_` prefix |
| `n_inspections` | Count of non-placeholder inspections seen for this CAMIS |
| `n_placeholder_inspections` | Count of placeholder rows |
| `first_inspection_date` | Earliest non-placeholder inspection date |
| `n_grade_a`, `n_grade_b`, `n_grade_c`, `n_grade_other` | Counts per grade across all real inspections |
| `is_placeholder_inspection`, `sample_id` | Dropped (not meaningful at the camis level) |

## Dedup & placeholder handling

The DOHMH dataset uses `inspection_date='1900-01-01T00:00:00'` as a sentinel meaning **"this CAMIS is registered but has not yet been inspected"** (~3.5k rows / ~1k CAMIS as of 2026-04-28).

Build-time violation-level collapse: within a single `(camis, inspection_date)`, the **most-Critical** violation row wins (Critical > Non-Critical, then alpha-sorted `violation_code`) so the kept row carries a substantive citation. The build does **not** dedup across inspections — that's `aggregate-restaurants`'s job.

Aggregate-time CAMIS-level collapse: rank rows by

1. **Real inspection > placeholder** (so a registered-but-uninspected CAMIS gets a placeholder row only if no real one exists).
2. **Most-recent `inspection_date`** among real rows.
3. **Critical > Non-Critical** within a single inspection — same tie-breaker as the build.

A CAMIS that only has placeholder rows still gets a row in the aggregated parquet (the user explicitly opted into "all NYC restaurants"), but `last_inspection_date` is null and `last_grade='NOT YET INSPECTED'`. Pass `--drop-placeholder-only` on the **build** to filter placeholder rows upstream entirely.

## Validation

Mirrors permits' / FacDB's validation with DOHMH-specific tweaks:

**Fatal (7):** Socrata returned ≥ 1 row · unique `(camis, inspection_date)` (sample_id) · every row has supported `geom_source` · no null DBA / facname · all buffered geoms valid · all inside NYC bbox · `coverage` non-empty + valid.

**Warn notables:** polygon match rate below threshold (default 85%, same as permits — DOHMH restaurants are almost all BIN-matched in real data) · placeholder count + percentage · CAMIS with only placeholder rows · inspections-per-CAMIS distribution (mean / p50 / p95 / max) · per-cuisine, per-borough, per-grade, per-`geom_source` breakdowns · total coverage area + % of NYC land · Socrata pagination truncation.

Fatal failures still write `summary.md` + `validation_report.parquet` + a `FATAL` `manifest.json` so diagnosis doesn't require a second Socrata pull.

## Sample build (Afghan cuisine smoke test)

A small live build at `--cuisine Afghan` (smoke tested 2026-04-28):

| Step | Output | Rows |
|------|--------|-----:|
| Raw inspection rows pulled from Socrata | (in memory) | 140 |
| `restaurants.parquet` (inspection-level) | one row per (camis, inspection_date) | **35** |
| `restaurants_aggregated.parquet` (after aggregate) | one row per CAMIS | **12** |
| BIN match rate | 100.00% | |
| Coverage area | 0.06 km² | |
| Boroughs | 9× Queens, 2× Manhattan, 1× Brooklyn | |

## Integration with the Cyclomedia pipeline

`materialize-cyclomedia` auto-detects `restaurants_aggregated.parquet` (in addition to `facilities.parquet` and `permits.parquet`) and looks up unit_uid + name as `uid` + `facname`.

If only the unaggregated `restaurants.parquet` is present, autodetect **refuses** with a helpful error pointing at `aggregate-restaurants`. Same protection in `_load_units`: it scans for duplicate unit IDs and raises if any exist (an explicit `--units-path` override doesn't bypass this).

```bash
sbatch --export=ALL,OUTPUT_FILENAME=cyclomedia_near_restaurants.parquet \
    scripts/materialize_scaffolding_cyclomedia.sub \
    curation/dohmh_pizza

# sample images for inspection — prefer the _facing parquet
python -m dagspaces.common.curation sample-images \
    --parquet curation/dohmh_pizza/cyclomedia_near_restaurants_facing.parquet \
    --out curation/dohmh_pizza/inspect_k100_symlink \
    -k 100 --symlink --stratify-by dataset --seed 0
```

## Comparing to FacDB

DOHMH overlaps slightly with [[facdb-curation]]'s `factype="EATING AND DRINKING PLACE"` (~140 rows in FacDB) but the two are very different in spirit:

| | DOHMH (`43nn-pn8j`) | FacDB (`ji82-xba5`) |
|---|---|---|
| Coverage | ~31k unique restaurants — every food-service establishment licensed in NYC | ~140 "eating and drinking places" (sparse, accidental) |
| Use case | Proxy for "all restaurants in NYC" | One of 600+ POI categories citywide |
| Granularity | Inspection-level by default (multi-row); aggregate to CAMIS-level | Already restaurant-level |
| Native categorical | `cuisine_description` (91 values) | 4-level hierarchy (`facdomain` → `factype`) |
| Why use this | Restaurant-specific data: grade/score/cuisine + temporal inspection history | Multi-domain POI catalog |

Use DOHMH when the visual question concerns restaurants specifically (storefront recognition, cuisine inference, signage); use FacDB when restaurants are just one category among many.

## Module layout

```
dagspaces/common/curation/dohmh/
  __init__.py
  cuisines.py            — frozen vocab + UnknownCuisineError
  cuisines.json          — 91 cuisine_description values, frozen 2026-04-28
  fetch.py               — Socrata pull (paginated, cached) for 43nn-pn8j
  normalize.py           — raw inspection rows → one row per (camis, inspection_date)
  validation.py          — 7 fatal + warn checks
  dohmh_restaurants.py   — build() orchestrator
  aggregate.py           — aggregate_restaurants() utility (opt-in)
```

## Related

- [[concept-facing-filter]] — the A–E per-unit facing pipeline (shared with [[facdb-curation]] and [[scaffolding-permits-curation]])
- [[facdb-curation]] — sibling curation, same module structure
- [[scaffolding-permits-curation]] — the original curation pattern these all share
- [[cyclomedia-catalog]] — downstream consumer of `coverage.geojson`
- `dagspaces/common/curation/dohmh/cuisines.json` — frozen cuisine vocab checked into repo
- DOHMH data dictionary: <https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j>
