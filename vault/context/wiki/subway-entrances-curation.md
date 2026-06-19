---
title: Subway Entrances Curation
category: infrastructure
created: 2026-04-28
updated: 2026-04-28
tags:
  - curation
  - mta
  - subway
  - transit
  - poi
  - cyclomedia
  - facing
sources: []
---

# Subway Entrances Curation

Fourth curation family after [[scaffolding-permits-curation]], [[facdb-curation]], and [[dohmh-restaurants-curation]]. Pulls every NYC subway station entrance/exit from the NY State Open Data **MTA Permanent Station Entrances/Exits** dataset (Socrata `i9wp-a4ja`, 2,120 rows / 485 stations / 13 entrance types as of 2026-04-28).

> **Different geometry shape from the other curation families.** Subway entrances are mostly **points** — sidewalk stairs, elevators, easements — not building features. So this build skips the BIN-match + nearest-building stages of the shared `geom.py` and **buffers the entrance lat/lon directly** by `--buffer-ft` (default 80 ft). No `nyc_buildings.parquet` lookup, no fallback chain. The output is a sub-dataset of buffered points around every entrance.

Each curation lands in `curation/subway_<slug>/` with the contract familiar from the other families: `entrances.parquet`, `entrances.geojson`, `coverage.geojson`, plus `manifest.json`, `summary.md`, `validation_report.parquet`, and the raw Socrata cache under `by_source/`.

## CLI

```bash
# Every NYC subway entrance — 2,120 rows / ~485 stations
python -m dagspaces.common.curation subway-entrances \
    --out curation/subway_entrances_all

# Just elevators (accessibility audit)
python -m dagspaces.common.curation subway-entrances \
    --out curation/subway_elevators \
    --entrance-type "Elevator"

# Single line (e.g. L train) — matched as a whole token in daytime_routes
python -m dagspaces.common.curation subway-entrances \
    --out curation/subway_l_line \
    --route L

# Multi-filter: Manhattan IRT stations only
python -m dagspaces.common.curation subway-entrances \
    --out curation/subway_irt_manhattan \
    --division IRT --borough Manhattan
```

Filter values are validated up-front:
- `--entrance-type` against the frozen 13-value vocab in `dagspaces/common/curation/subway/entrance_types.json` (typos fail with suggestions).
- `--division` against `{IRT, IND, BMT, SIR, IRT/BMT, IND/BMT}`.
- `--borough` accepts both source codes (`M / B / Bx / Q / SI`) and full names (Manhattan / Brooklyn / etc.).
- `--route` is passed through as user-facing IDs (`L`, `4`, `Q`); matched as a whole space-delimited token via SoQL `LIKE '% <id> %'`.

## Pipeline

```
Socrata data.ny.gov/resource/i9wp-a4ja
  → fetch (entrance_type + division + borough + route filters)
  → normalize (synthesize stable uid from station_id + entrance_type + rounded coords;
                map borough codes; cast YES/NO → bool)
  → drop rows outside NYC bbox
  → buffer the entrance point by --buffer-ft in EPSG:2263 → WGS84
  → validate (8 fatal + warn checks)
  → write entrances.parquet / .geojson + coverage.geojson
```

The build does **not** call `dagspaces.common.curation.geom.attach_geometry` — that path's three-stage BIN→nearest→point fallback is wrong for street-furniture data. The `_buffer_points` helper inside `subway_entrances.py` is a single-stage point buffer; every row gets `geom_source='point'` and `match_dist_ft=0.0`.

## Output schema (`entrances.parquet`)

| Column | Notes |
|---|---|
| `uid` | Synthesized: `"{station_id}_{entrance_type_alpha}_{lat_int}_{lon_int}"` (lat/lon as 7-decimal int strings, with `-` rendered as `n`). Stable across rebuilds. |
| `permit_id` | Aliased from `uid` for the shared geom API |
| `facname` | `"{stop_name} ({entrance_type})"` — used as the displayed unit name in materialize / sample-images |
| `station_id`, `complex_id`, `gtfs_stop_id` | MTA identifiers |
| `stop_name`, `constituent_station_name` | Station display names |
| `line` | Subway trunk line (e.g. "Canarsie", "Broadway - Brighton") |
| `division` | IRT / IND / BMT / SIR / IRT/BMT / IND/BMT |
| `daytime_routes` | Space-separated route IDs running the platform during daytime |
| `entrance_type` | One of 13 values (Stair, Elevator, Station House, Easement - Street, …) |
| `entry_allowed`, `exit_allowed` | Booleans (cast from YES/NO) |
| `address`, `city`, `borough` | Borough-derived only; the source has no street address |
| `facdomain`, `facgroup`, `facsubgrp`, `factype` | FacDB-shaped aliases — `facdomain="TRANSPORTATION"`, `facgroup="SUBWAY ENTRANCES"`, `facsubgrp=factype=upper(entrance_type)` — so the row drops into any tooling already speaking the FacDB schema |
| `latitude`, `longitude`, `raw_latitude`, `raw_longitude` | Spatial keys |
| `datasource` | Fixed: `"mta:i9wp-a4ja"` |
| `geom_source` | Always `"point"` |
| `match_dist_ft` | Always `0.0` |
| `geom_wkb` | Buffered point as WKB binary (parquet) |

## Validation

8 fatal checks + warn metrics:

**Fatal:** Socrata returned ≥1 row · unique entrance `uid` · all rows have non-null lat/lon · all rows have non-null `entrance_type` · all rows have non-null `facname` · all buffered geoms valid · all inside NYC bbox · `coverage` non-empty + valid.

**Warn notables:** coverage area + % of NYC land · entrances-per-station distribution (mean / p50 / p95 / max) · per-`entrance_type`, per-`division`, per-`borough` breakdowns · Socrata pagination truncation.

## Sample build (L-line smoke test)

A small live build at `--route L` (smoke-tested 2026-04-28):

| | |
|---|---|
| Raw rows from Socrata | 131 |
| Publishable entrances | 131 |
| Unique stations / complexes | 30 / 25 |
| Coverage area | 0.171 km² |
| Entrance types | 107× Stair, 9× Elevator, 9× Station House, 3× Walkway, 2× Easement - Street, 1× Easement - Passage |
| Boroughs | 67× Brooklyn, 57× Manhattan, 7× Queens |
| Elapsed | 0.7 s end-to-end |

## Integration with the Cyclomedia pipeline

`materialize-cyclomedia` autodetects `entrances.parquet` (unit-key columns are `uid` + `facname`, same as FacDB and DOHMH-aggregated). No code changes needed in the per-unit attribution path or [[concept-facing-filter]]; the filter just sees a unit polygon (the 80-ft-buffered point) and works as designed.

```bash
sbatch --export=ALL,OUTPUT_FILENAME=cyclomedia_near_subway.parquet \
    scripts/materialize_scaffolding_cyclomedia.sub \
    curation/subway_l_line

# inspection — prefer the _facing parquet
python -m dagspaces.common.curation sample-images \
    --parquet curation/subway_l_line/cyclomedia_near_subway_facing.parquet \
    --out curation/subway_l_line/inspect_k100_symlink \
    -k 100 --symlink --stratify-by entrance_type --seed 0
```

## Comparing to the other curation families

| | Subway entrances | DOHMH restaurants | FacDB | Scaffolding permits |
|---|---|---|---|---|
| Source | data.ny.gov `i9wp-a4ja` | data.cityofnewyork.us `43nn-pn8j` | `ji82-xba5` | `w9ak-ipjd` + `ipu4-2q9a` |
| Granularity | one row per entrance | one row per (camis, inspection); `aggregate-restaurants` collapses to camis | one row per facility | one row per permit |
| Geometry | **point buffer** | BIN → nearest → point | BIN → nearest → point | BIN → nearest → point |
| Native categorical | `entrance_type` (13 values) | `cuisine_description` (91 values) | 4-level hierarchy (`facdomain` → `factype`) | `scaffold_type` |
| Use case | Locate subway entrances visually | "All NYC restaurants" proxy | Multi-domain POI catalog | Permit-adjacent imagery |

Use subway when the visual question concerns transit (entrance accessibility, subway signage, station condition); the geometry shape is **fundamentally different** from the others and the build path reflects that.

## Module layout

```
dagspaces/common/curation/subway/
  __init__.py
  entrance_types.py       — frozen vocab + UnknownEntranceTypeError
  entrance_types.json     — 13 entrance_type values, frozen 2026-04-28
  fetch.py                — Socrata pull (paginated, cached) for i9wp-a4ja
  normalize.py            — raw rows → canonical schema with synthetic stable uid
  validation.py           — 8 fatal + warn checks
  subway_entrances.py     — build() orchestrator + _buffer_points() helper
```

## Related

- [[concept-facing-filter]] — the A–E per-unit facing pipeline (works transparently against subway entrance polygons)
- [[dohmh-restaurants-curation]] — sibling POI curation
- [[facdb-curation]] — sibling POI curation
- [[scaffolding-permits-curation]] — original curation pattern
- [[cyclomedia-catalog]] — downstream consumer of `coverage.geojson`
- MTA dataset page: <https://data.ny.gov/Transportation/MTA-Subway-Entrances-and-Exits-2024/i9wp-a4ja>
