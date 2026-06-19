---
title: Open Restaurants Curation
category: infrastructure
created: 2026-06-01
updated: 2026-06-01
tags:
  - curation
  - open-restaurants
  - dining-out-nyc
  - restaurants
  - poi
  - cyclomedia
  - facing
sources: []
---

# Open Restaurants Curation

Fifth curation family, after [[scaffolding-permits-curation]], [[facdb-curation]], [[dohmh-restaurants-curation]], and [[subway-entrances-curation]]. Pulls NYC restaurants licensed for **outdoor dining** from the DCWP **Open Restaurants / Dining Out NYC** dataset (Socrata `fpeh-f7ci`) — the permanent licensing program that succeeded the COVID-era emergency Open Restaurants. Each row is one license to operate a `Sidewalk` or `Roadway` dining setup. Optionally filter by `license_type` and/or borough.

Distinct from [[dohmh-restaurants-curation]] (which proxies *all* ~31k inspected restaurants): this is the much smaller set (~1.3k) of restaurants that specifically hold an **outdoor-dining license** — the right population when the visual question is about sidewalk cafés, roadway dining sheds, or streetscape dining furniture.

Lands in `curation/open_restaurants_all/` with the standard curation contract: `open_restaurants.parquet`, `open_restaurants.geojson`, `coverage.geojson`, `manifest.json`, `summary.md`, `validation_report.parquet`, `by_source/open_restaurants_raw.parquet`.

## CLI

```bash
# Full pull — every issued outdoor-dining license (~1.3k)
python -m dagspaces.common.curation open-restaurants \
    --out curation/open_restaurants_all

# Filter by license type and/or borough (AND across, OR within)
python -m dagspaces.common.curation open-restaurants \
    --out curation/open_restaurants_roadway_mn \
    --license-type Roadway --borough Manhattan

# Materialize Cyclomedia images near licensed restaurants (SLURM)
sbatch scripts/materialize_open_restaurants_cyclomedia.sub \
    curation/open_restaurants_all
```

`--license-type` is validated against the frozen vocab (`Sidewalk`, `Roadway`) at startup; typos fail fast with suggestions (`--license-type sidewlk` → `did you mean: ['Sidewalk']`). `--borough` accepts full names or `MN/BX/BK/QN/SI` aliases.

## Pipeline

```
Socrata fpeh-f7ci  →  fetch (license_type + borough where clause; latitude IS NOT NULL)
                   →  normalize (synthesize unique uid; facname = DBA name else legal name;
                                  borough canonical uppercase, fallback to BBL boro digit)
                   →  drop null facname
                   →  drop bad-geocode rows (no BIN AND lat/lon outside NYC bbox)
                   →  attach_geometry (shared geom.py: BIN-exact → nearest-200ft → point)
                   →  buffer 80 ft in EPSG:2263 → WGS84
                   →  validate (8 fatal + warn checks)
                   →  write open_restaurants.parquet / .geojson + coverage.geojson
```

Shares `dagspaces/common/curation/geom.py` (3-stage BIN match) with [[facdb-curation]] / [[dohmh-restaurants-curation]] / [[scaffolding-permits-curation]] via the same column-name parameters (`id_col=permit_id`, `bin_col=bin`, `lat_col=raw_latitude`, `lon_col=raw_longitude`).

## Synthetic primary key

The dataset has **no native primary key** (no license/objectid column), and a single business can hold multiple rows of the same `license_type` at different locations. `normalize.py` synthesizes a stable `uid`:

```
{bbl}_{LICENSE_TYPE}_{lat7}_{lon7}        # base key (coords rounded to 7 dp, digits-only)
{base}_{i}                                # i = within-base occurrence index, only when base collides
```

This guarantees uniqueness (required by `materialize-cyclomedia`, which refuses duplicate unit IDs) and stays stable across rebuilds unless DCWP moves the location or relicenses it. Same approach as [[subway-entrances-curation]].

## Output schema

`open_restaurants.parquet` — one row per license, keyed `uid` + `facname` (the contract `materialize-cyclomedia` auto-detects).

| Column | Notes |
|---|---|
| `uid` | Synthesized unique license UID |
| `permit_id` | Aliased from `uid` for the shared `geom.attach_geometry` API |
| `facname` | Public-facing name — `assumed_name_s` (DBA) if present, else `business_legal_name` |
| `business_legal_name`, `assumed_name_s` | Native DCWP name columns |
| `address` (= `street`), `city`, `borough`, `postcode` | Identity |
| `license_type` | `Sidewalk` \| `Roadway` |
| `license_status` | `Issued` (all rows as of 2026-06) |
| `license_issue_date`, `license_expiration_date` | ISO timestamps |
| `bin`, `bbl`, `latitude`, `longitude` | Spatial keys |
| `raw_latitude`, `raw_longitude` | Aliases (shared geom API) |
| `council_district`, `community_board`, `nta2020`, `ct2020` | Admin geographies |
| `datasource` | Fixed: `"dcwp:fpeh-f7ci"` |
| `geom_source` ∈ {`bin_polygon`, `nearest_polygon`, `point`} | Match stage |
| `match_dist_ft` | Distance for nearest fallback |
| `geom_wkb` | Buffered polygon as WKB binary |

## Validation

Mirrors FacDB's validation (BIN-polygon geometry):

**Fatal (8):** Socrata returned ≥1 row · unique `uid` · every row has supported `geom_source` · no null `facname` · no null `license_type` · all buffered geoms valid · all inside NYC bbox · `coverage` non-empty + valid.

**Warn notables:** polygon match rate below threshold (default 85% — licenses carry BIN/BBL so are nearly all BIN-matched) · per-`license_type`, per-`license_status`, per-borough, per-`geom_source` breakdowns · coverage area + % of NYC land · Socrata pagination truncation.

Fatal failures still write `summary.md` + `validation_report.parquet` + a `FATAL` `manifest.json`.

## Build result (`open_restaurants_all`, 2026-06-01)

| Metric | Value |
|--------|------:|
| Raw rows pulled | 1,313 |
| Publishable licenses | **1,309** (4 bad geocodes dropped) |
| Unique tax lots (BBL) | 1,043 |
| Polygon match rate | 100.00% (99.16% BIN-exact + 0.84% nearest) |
| Coverage area | 4.028 km² (0.52% of NYC land) |
| License type | Sidewalk 656 / Roadway 653 |
| Borough | Manhattan 807, Brooklyn 389, Queens 84, Bronx 28, Staten Island 1 |

## Integration with the Cyclomedia pipeline

`materialize-cyclomedia` auto-detects `open_restaurants.parquet` (added to the candidate list alongside `facilities.parquet` / `permits.parquet` / `restaurants_aggregated.parquet` / `entrances.parquet`) and reads `uid` + `facname` as the per-unit attribution. The default run also produces the `_facing` sibling via the per-unit facing filter ([[concept-facing-filter]]).

Materialized 2026-06-01 (job under `.slurm_jobs/curation_open_restaurants/`):

| Output | Rows |
|--------|-----:|
| `cyclomedia_near_open_restaurants.parquet` (F/B/L/R, all boroughs) | 191,964 |
| `cyclomedia_near_open_restaurants_facing.parquet` (per-unit facing) | **36,284** across 1,039 restaurants |

Per-borough catalog matches: Manhattan 120,492 · Brooklyn 52,098 · Queens 14,858 · Bronx 4,400 · Staten Island 116. Elapsed ~165 s.

Downstream consumers use the Hydra data config `cyclomedia_near_open_restaurants_facing.yaml` (datasets group) — same column contract as `cyclomedia_near_restaurants_facing.yaml`, including `attribution_confidence` for `pair_sampler.weight_column` weighted draws.

```bash
# sample images for inspection — prefer the _facing parquet
python -m dagspaces.common.curation sample-images \
    --parquet curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing.parquet \
    --out curation/open_restaurants_all/inspect_k100_symlink \
    -k 100 --symlink --stratify-by dataset --seed 0
```

## Preview / QA reports

Two PDF report scripts (both reusable for any curation family's `*_facing.parquet`):

```bash
# Bird's-eye image-distribution preview: counts, per-borough/license_type/face
# bars, images-per-unit + facing-diagnostic histograms, a borough-coloured
# point map, and a stratified image montage.
python scripts/image_distribution_report.py \
    --parquet curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing.parquet \
    --units-parquet curation/open_restaurants_all/open_restaurants.parquet \
    --category-col license_type --unit-label restaurant \
    --title "Open Restaurants (Dining Out NYC) · image distribution"
# → curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing_distribution.pdf

# Deep per-unit audit: one thumbnail page per restaurant, captioned with the
# per-row facing diagnostics, sorted by attribution_confidence.
python scripts/facing_audit_report.py \
    --parquet curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing.parquet \
    --out curation/open_restaurants_all/facing_audit.pdf \
    --unit-label restaurant --unit-label-plural restaurants
```

`image_distribution_report.py` reuses the matplotlib polish + accent colours from `pairwise_vqa_report.py` and the thumbnail loader from `facing_audit_report.py`; pass `--units-parquet` to unlock the per-borough / per-`license_type` breakdowns (joined `unit_uid` → `uid`) and the coverage ratio. The `open_restaurants_all` preview (2026-06-01): 36,284 images / 1,039 restaurants (79.4% coverage), 62% Manhattan, Sidewalk 65% / Roadway 35%, median 33 images/restaurant, median facing distance 93 ft.

## Module layout

```
dagspaces/common/curation/open_restaurants/
  __init__.py
  license_types.py       — frozen Sidewalk/Roadway vocab + UnknownLicenseTypeError
  fetch.py               — Socrata pull (paginated, cached) for fpeh-f7ci
  normalize.py           — raw rows → one row per license, synthetic unique uid
  validation.py          — 8 fatal + warn checks
  open_restaurants.py    — build() orchestrator
```

Tests: `tests/test_open_restaurants_curation.py` (vocab validation, fetch clause, normalize uid/facname/borough logic, end-to-end build incl. bad-geocode drop + fatal duplicate-uid).

## Related

- [[concept-facing-filter]] — the per-unit facing pipeline (shared with the other BIN-polygon curations)
- [[dohmh-restaurants-curation]] — sibling restaurant curation (all inspected restaurants vs. just outdoor-dining licensees)
- [[facdb-curation]] — same module structure / BIN-polygon geometry
- [[scaffolding-permits-curation]] — the original curation pattern these all share
- [[cyclomedia-catalog]] — downstream consumer of `coverage.geojson`
- Dataset: <https://data.cityofnewyork.us/resource/fpeh-f7ci>
