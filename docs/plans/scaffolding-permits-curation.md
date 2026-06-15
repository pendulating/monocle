# Scaffolding Permits Curation — Implementation Plan

**Status:** proposed — awaiting approval before implementation
**Owner:** mllmsci (first of several "curated sub-dataset" bootstraps)
**Target path:** `dagspaces/common/curation/` (module) + `curation/` (data)
**Downstream consumer:** `CyclomediaCatalog.query(within=coverage_gdf)`

## Goal

Produce a reusable, versioned spatial mask of every NYC DOB scaffold/shed permit **issued through 2025-12-31**, regardless of current active/expired status, so downstream stages can carve a curated Cyclomedia sub-dataset by intersecting the catalog with the permit footprint (building polygon buffered 80 ft).

This is the first concrete bootstrap of the broader "curated sub-dataset" pattern — a permit geojson is the spatial side; a `CyclomediaCatalog.query(within=...)` call + parquet write is the image side. Each sub-dataset plugs into the same contract.

## Scope & constraints

| Axis | Decision |
|------|----------|
| Boroughs | All 5 (no borough filter) |
| Permit types | Scaffold + shed (sidewalk shed and proper scaffold) |
| Time | `issue_date <= 2025-12-31` (includes expired/signed-off) |
| Sources | Union of **BIS** (legacy, pre-2020) and **DOB NOW** (modern, 2020+) |
| Auth | Unauthenticated Socrata (no app token); existing pattern is 50k-row paginated GET with parquet cache |
| Spatial unit | Per-permit building footprint buffered **80 ft** in EPSG:2263 |
| Fallback when BIN not in building footprint file | 80-ft circle around `(latitude, longitude)` from the API, tagged `geom_source='point'` |

## Data sources

### DOB NOW (`data.cityofnewyork.us/resource/w9ak-ipjd.json`)

- Fields we pull: `job_filing_number, filing_status, filing_date, first_permit_date, current_status_date, signoff_date, latitude, longitude, scaffold, shed, borough, house_no, street_name, block, lot, bin, initial_cost, job_type`
- Filter: `(scaffold='1' OR shed='1') AND latitude IS NOT NULL AND first_permit_date IS NOT NULL AND first_permit_date <= '2025-12-31T23:59:59'`
- Date field used: `first_permit_date` (when the permit was issued). Filings never issued (null `first_permit_date`) are **dropped** — documented in the wiki.
- Derives scaffold_type: `scaffold='1' AND shed='1'` → `both`; `scaffold='1'` → `scaffold`; `shed='1'` → `shed`.

### BIS (`data.cityofnewyork.us/resource/ipu4-2q9a.json`)

- Fields: `job__, bin__, permit_type, permit_subtype, permit_status, issuance_date, expiration_date, borough, house__, street_name, block, lot, job_type, permit_si_no, work_type, initial_cost, lat, lng, gis_nta_name, owner_s_business_name`
- Filter: `permit_subtype IN ('SH','SD','SF') AND issuance_date IS NOT NULL AND issuance_date <= '2025-12-31T23:59:59'`
- `issuance_date` is a Socrata floating timestamp (ISO string server-side even though some cached parquets store it as `MM/DD/YYYY` — confirm during implementation; fall back to client-side filter if server-side comparison fails).
- Derives scaffold_type: `SH` → `shed`; `SD` → `scaffold`; `SF` → `scaffold` (Supported Scaffold).

### Building footprints

- `data/geo/nyc_buildings.parquet` (1.08 M polygons, BIN-keyed). Join: `permit.bin → buildings.bin` left.
- Rows with matched BIN → use building polygon.
- Rows with unmatched BIN (deleted building, bad data, or missing BIN) → use `Point(longitude, latitude)` from the API. Tagged `geom_source='point'`.

## Module layout

```
dagspaces/common/curation/
  __init__.py
  socrata.py                 # fetch_socrata(...) — paginated, parquet-cached, no-token
  permits/
    __init__.py
    fetch.py                 # fetch_dob_now(), fetch_bis(), schemas, date filters
    normalize.py             # both APIs → common flat schema
    buffer.py                # BIN → building polygon or point; buffer 80 ft; project
    validation.py            # post-pull sanity checks (fatal + warn), emits summary.md + report.parquet
    scaffolding_permits.py   # top-level orchestration: fetch → union → buffer → validate → write
  cli.py                     # `python -m dagspaces.common.curation scaffolding-permits ...`
```

```
curation/                    # new repo-root data dir (alongside datasets/, vault/)
  scaffolding_permits_through_2025/
    manifest.json            # {schema_version, built_at, source_rows, final_rows, cutoff_date}
    permits.parquet          # one row per permit, flat schema (below)
    permits.geojson          # same rows, WGS84, one feature per permit (80-ft buffered poly)
    coverage.geojson         # unary_union of all buffered polygons → single MultiPolygon
    by_source/
      dob_now_raw.parquet    # raw Socrata response, post-filter
      bis_raw.parquet        # raw Socrata response, post-filter
```

## Pipeline

1. **Fetch DOB NOW** (`socrata.fetch_socrata`) with above where clause → `by_source/dob_now_raw.parquet`.
2. **Fetch BIS** same pattern → `by_source/bis_raw.parquet`.
3. **Normalize** both to the common schema (see below). Cast dates, infer scaffold_type, uppercase borough.
4. **Concat** (do not dedupe — copies kept intentionally; `(source, permit_id)` is the natural primary key, and downstream can `.unique(['bin', 'issue_date'])` if it cares).
5. **Attach geometry:**
   - Left-join on `bin` to `nyc_buildings.parquet` (columns: `bin`, `geometry`).
   - For rows with a polygon match: use it directly.
   - For rows without: construct `Point(longitude, latitude)` in EPSG:4326.
   - Reproject everything to EPSG:2263 (NY State Plane Long Island, US ft).
6. **Buffer** 80 ft (polygons and points alike — an 80-ft circle for point-fallback rows).
7. **Reproject** buffered geometry back to EPSG:4326. Set `geom_source ∈ {'bin_polygon','point'}`.
8. **Validate** (see `## Validation` below). Refuse to publish outputs if any fatal check fails.
9. **Write outputs:**
   - `permits.parquet` — flat schema, includes WKB geometry in `geom_wkb` column.
   - `permits.geojson` — one feature per permit, properties = all non-geometry columns.
   - `coverage.geojson` — `unary_union(all buffered polygons)` as a single MultiPolygon (consumed by `CyclomediaCatalog.query(within=...)`).
   - `manifest.json` — schema version, cutoff date, source row counts, final row count, geom_source breakdown, git SHA.
   - `validation_report.parquet` — per-permit boolean columns for every check (drill-down for failures).
   - `summary.md` — human-readable scoreboard of every fatal + warn check, pass rates, per-source and per-borough breakdowns.

## Output schema (`permits.parquet` / feature properties)

| Column | Type | Source |
|--------|------|--------|
| `source` | str | `"dob_now"` / `"bis"` |
| `permit_id` | str | DOB NOW `job_filing_number` / BIS `job__` + `permit_si_no` |
| `bin` | str | API |
| `borough` | str | uppercase: `MANHATTAN, BROOKLYN, QUEENS, BRONX, STATEN ISLAND` |
| `issue_date` | datetime (UTC) | DOB NOW `first_permit_date` / BIS `issuance_date` |
| `expiration_date` | datetime or null | BIS `expiration_date` (DOB NOW: null) |
| `signoff_date` | datetime or null | DOB NOW `signoff_date` (BIS: null) |
| `scaffold_type` | enum | `scaffold, shed, both` (see derivation rules) |
| `permit_status` | str | raw status string from API |
| `permit_subtype` | str or null | BIS only (`SH/SD/SF`) |
| `address` | str | `{house_no} {street_name}` |
| `block`, `lot` | str | API |
| `job_type` | str | API |
| `initial_cost` | float | API |
| `raw_latitude`, `raw_longitude` | float | API (pre-buffer point) |
| `geom_source` | enum | `bin_polygon, point` |
| `geom_wkb` | binary | 80-ft-buffered polygon in EPSG:4326 (parquet only; GeoJSON carries geometry natively) |

## Design decisions (explicit, for the wiki)

1. **Drop filings with null `first_permit_date`.** DOB NOW filings that were submitted but never issued a permit are out of scope — "issued through 2025-12-31" means a permit was actually issued. Rationale: the purpose of this sub-dataset is to find Cyclomedia images near built scaffolding, and a non-issued filing correlates poorly with built work.
2. **Include expired and signed-off permits.** Per explicit user instruction — we want all imagery near any historical permit, not just currently-active work.
3. **Keep BIS + DOB NOW copies.** No dedupe across sources. DOB NOW mostly replaced BIS around 2020, so the datasets are largely disjoint in time; when they overlap, both copies carry different metadata and should both be preserved. Callers can dedupe downstream.
4. **80 ft building buffer.** Matches the upper bound in the rerank compliance notebook (`BUFFER_FT = 50` was the default for 85% camera match; 80 ft gives ~95% — the right choice for a recall-biased image curation pass).
5. **Point fallback for unmatched BINs.** Keeps a handful of permits with bad / demolished BINs in the sub-dataset, tagged `geom_source='point'` so recall-sensitive downstream uses can exclude them.
6. **All 5 boroughs in one file.** Coverage geojson is a single MultiPolygon. If a future sub-dataset needs per-borough splits, write them as views, not as separate curation outputs.
7. **No Socrata app token.** Existing fetch pattern handles throttling fine at 50k page size. Revisit if we start hitting 429s.

## Integration with `CyclomediaCatalog`

Next task (separate PR) wires the curated mask to the catalog:

```python
from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog
import geopandas as gpd

cat = CyclomediaCatalog()
coverage = gpd.read_file("curation/scaffolding_permits_through_2025/coverage.geojson")
df = cat.query(
    within=coverage,
    faces={"F", "B", "L", "R"},       # drop U/D for street-level scaffold imagery
    datasets=["manhattan_2025_1k", "brooklyn_2025_1k", "queens_2025_1k",
              "bronx_2025_1k", "si_2025_1k"],
)
df.write_parquet("curation/scaffolding_permits_through_2025/cyclomedia_near_permits.parquet")
```

Rough expected volume: current compliance notebook shows ~45k unique BINs citywide with scaffold/shed filings. With 80-ft buffers these footprints likely cover ~10-15% of NYC linear street frontage near buildings, which at ~31.5M catalog rows × 4/6 horizontal faces ≈ ~2-3M curated rows. Precise count is a build-time output.

## Validation

Modeled on `dagspaces/common/cyclomedia_catalog/validation.py`. Runs at the tail of `scaffolding_permits.build(...)` against the normalized + buffered frame, before any output is written. Produces two artifacts per run:

- **`validation_report.parquet`** — one row per permit with a boolean column per check. Used for drill-downs.
- **`summary.md`** — human-readable scoreboard. Every check lists severity, pass rate, per-source breakdown, per-borough breakdown where applicable.

If any **fatal** check fails, refuse to write `permits.parquet` / `permits.geojson` / `coverage.geojson` and raise `PermitValidationError` (fatal list is stuffed into the exception message). `validation_report.parquet` + `summary.md` are still written so the operator can diagnose.

### Checks

**Fatal (refuse to publish):**

| # | Check | Rationale |
|---|-------|-----------|
| 1 | Both sources returned ≥ 1 row after Socrata pagination | Catches silent API outage or a where-clause that accidentally excluded everything. |
| 2 | No duplicate `(source, permit_id)` | `permit_id` is our primary key within a source; dupes break downstream joins. |
| 3 | Every permit has either `bin` present in `nyc_buildings.parquet` **OR** non-null `(latitude, longitude)` | No geometry source = no way to produce a polygon. |
| 4 | No null `issue_date` (defense in depth — fetch filters upstream) | Without a date we can't satisfy the "issued through 2025-12-31" contract. |
| 5 | No null `scaffold_type` | Empty classification means the normalization rule failed. |
| 6 | All buffered geometries `.is_valid` | Self-intersections in the buffered polygon break later spatial joins. |
| 7 | All buffered geometries fully inside NYC bbox | Catches projection bugs (e.g. wrong CRS applied before buffer → buffered polygon on the wrong continent). |
| 8 | `coverage.geojson` equals `unary_union(permits.geometry)` within tolerance (1e-6 sq deg) | Ensures the dissolved mask is a faithful summary of the per-permit geometries. |

**Warn (log + record, do not block):**

| # | Check | What it surfaces |
|---|-------|------------------|
| 9 | **BIN → building polygon match rate** (per source, per borough) | The headline recall metric the user called out. Expected ≥ 85% polygon match; below that, flag loudly in `summary.md`. |
| 10 | **Dropped permits** counted at each filter step (from raw → filtered → normalized → joined → buffered) | Measures how much recall we leak per preprocessing component. Single table in `summary.md`. |
| 11 | **Socrata pagination** — for each source, was the final page size < `limit`? | Detects truncation (server cut off mid-result). If the last page returned exactly `limit` we warn hard — we almost certainly truncated. |
| 12 | **Null `first_permit_date` counts** in raw DOB NOW fetch | Documents how many filings were intentionally dropped by design decision #1. Separate from check #10 because this is expected, not leakage. |
| 13 | **`scaffold_type` distribution** per source | Sanity: each source should produce a mix. A 100%-shed or 100%-scaffold breakdown suggests a filter bug. |
| 14 | **`permit_status` distribution** per source | Mostly a visibility metric — we keep all statuses, but we surface the split so readers can sanity-check "expired permits included." |
| 15 | **`issue_date` floor check** — permits before 1990-01-01 or after cutoff | Bad dates upstream. Doesn't exclude them (by design — we trust the API), but counts. |
| 16 | **BIN occurrence histogram** — permits per BIN, top 20 BINs by permit count | One BIN with 200 permits is a data-quality signal worth eyeballing. |
| 17 | **Cross-source BIN overlap** — size of `bis_bins ∩ dob_now_bins` and example rows | Informs whether the "don't dedupe" decision is producing lots of duplicated footprints. Not fatal — caller can dedupe. |
| 18 | **Per-permit buffered-area distribution** (p50, p95, p99 in m²) | Catches a buffer-distance bug: if p50 area ≠ expected area for the typical building × 80 ft, something went wrong in projection. |
| 19 | **Coverage footprint** — total `coverage.geojson` area in km² + % of NYC land area | Gives a one-glance sense of how much of the city the mask actually spans. |
| 20 | **`geom_source` breakdown** — count + pct of `bin_polygon` vs `point` rows | Same number as check #9 reframed — headline in the summary. |

### Summary.md layout

```
# Scaffolding permits curation — validation summary

- Cutoff date: 2025-12-31
- Sources: BIS (60,xxx rows) + DOB NOW (40,xxx rows) = 100,xxx permits
- Buffer: 80 ft (EPSG:2263)
- Built at: 2026-mm-dd hh:mm UTC

## Fatal checks
| # | Check | Status |
|---|-------|--------|
| 1 | ...   | PASS   |

## Warn checks
| # | Check | Value | Threshold |
...

## BIN match rate by source × borough
| source | borough | total | bin_polygon | point | polygon % |
...

## Dropped-permit funnel
| step | source | rows in | rows out | dropped | reason |
...

## Top BIN frequencies
| bin | permits | borough | address example |
...
```

### Unit tests (`tests/test_scaffolding_permits_curation.py`)

- `test_normalize_dob_now_scaffold_type` — both flags → `both`, etc.
- `test_normalize_bis_permit_subtype` — `SH` → shed; `SD/SF` → scaffold.
- `test_drops_null_first_permit_date` — DOB NOW filing with null `first_permit_date` excluded.
- `test_point_fallback_for_unmatched_bin` — permit with BIN not in buildings → `geom_source='point'` with 80-ft circle.
- `test_coverage_is_unary_union_of_permits` — matches sum of individual buffered geometries (check #8 in isolation).
- `test_validation_fatal_blocks_output` — inject a duplicate `(source, permit_id)`, assert `PermitValidationError` is raised and `permits.parquet` is NOT written but `summary.md` IS.
- `test_validation_warn_surfaces_in_summary` — inject low BIN match rate, assert the warn shows up in `summary.md` without blocking the build.

## CLI

```bash
python -m dagspaces.common.curation scaffolding-permits \
    --cutoff 2025-12-31 \
    --buffer-ft 80 \
    --out curation/scaffolding_permits_through_2025/
```

Defaults baked in match this plan. Versioned output dir name (`scaffolding_permits_through_2025`) is a parameter so future re-runs with later cutoffs go to a new dir (`scaffolding_permits_2026Q2`, etc.).

## Testing

`tests/test_scaffolding_permits_curation.py`:
- `test_normalize_dob_now_scaffold_type` — both flags → `both`, etc.
- `test_normalize_bis_permit_subtype` — `SH` → shed; `SD/SF` → scaffold
- `test_drops_null_first_permit_date` — DOB NOW filing with null `first_permit_date` excluded
- `test_point_fallback_for_unmatched_bin` — permit with bin not in buildings → `geom_source='point'` with 80-ft circle
- `test_coverage_is_unary_union_of_permits` — matches sum of individual buffered geometries

## Open items to resolve during implementation (non-blocking for plan approval)

- Confirm Socrata accepts ISO-timestamp comparison on `issuance_date` in `ipu4-2q9a`. If it's Plain Text, filter server-side via `starts_with(issuance_date, '2024')` / client-side date parsing.
- Confirm BIS `permit_si_no` is the right secondary key for a stable `permit_id` (one `job__` can have multiple permits).
- Exact borough normalization: DOB NOW returns `"MANHATTAN"`; BIS returns title case. Normalize to uppercase.
