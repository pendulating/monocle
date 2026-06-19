---
title: FacDB Curation
category: curation
created: 2026-04-22
updated: 2026-04-22
tags:
  - curation
  - facdb
  - facilities
  - poi
  - cyclomedia
  - facing
sources: []
---

# FacDB Curation

Second curation family after [[scaffolding-permits-curation]]. Pulls NYC POIs from the City Planning **Facilities Database** (Socrata `ji82-xba5`, ~34.7k rows as of 25v2), filtered at any of the 4 hierarchy levels from the `facilities_data_dictionary.xlsx` Categorization sheet:

```
facdomain   (7)    — e.g. "HEALTH AND HUMAN SERVICES"
  facgroup  (25)   — e.g. "LIBRARIES"
    facsubgrp (72) — e.g. "BRANCH LIBRARIES"
      factype (609) — e.g. "PUBLIC LIBRARY"
```

Each sub-dataset lands in `curation/facdb_<slug>/` with the same contract as permits: `facilities.parquet`, `facilities.geojson`, `coverage.geojson` (per-polygon features, ready to feed `CyclomediaCatalog.query(within=...)`), plus `manifest.json`, `summary.md`, `validation_report.parquet`, and the raw Socrata cache under `by_source/`.

## CLI

```bash
# All libraries citywide (255 → 253 publishable, 100% polygon match, 1.53 km²)
python -m dagspaces.common.curation facdb-facilities \
    --out curation/facdb_libraries \
    --facgroup "LIBRARIES"

# All health & human services facilities
python -m dagspaces.common.curation facdb-facilities \
    --out curation/facdb_health_human_services \
    --facdomain "HEALTH AND HUMAN SERVICES"

# Multi-level filter (AND across levels, OR within a level)
python -m dagspaces.common.curation facdb-facilities \
    --out curation/facdb_schools_and_colleges \
    --facgroup "SCHOOLS (K-12)" "HIGHER EDUCATION"
```

Filter values are **validated against the frozen dictionary at startup** (`dagspaces/common/curation/facdb/categorization.json`, baked from the xlsx Categorization sheet). Typos fail fast with suggestions — e.g. `--facdomain "healht"` → `did you mean: ['HEALTH AND HUMAN SERVICES']`.

## The 7 `facdomain` values

| `facdomain` | rows in full FacDB |
|---|---:|
| EDUCATION, CHILD WELFARE, AND YOUTH | 15,787 |
| HEALTH AND HUMAN SERVICES | 5,863 |
| CORE INFRASTRUCTURE AND TRANSPORTATION | 5,347 |
| PARKS, GARDENS, AND HISTORICAL SITES | 3,656 |
| LIBRARIES AND CULTURAL PROGRAMS | 2,648 |
| ADMINISTRATION OF GOVERNMENT | 963 |
| PUBLIC SAFETY, EMERGENCY SERVICES, AND ADMINISTRATION OF JUSTICE | 444 |

Full 4-level hierarchy browsable via:

```python
from dagspaces.common.curation.facdb import load_categorization
h = load_categorization()
# h["hierarchy"][facdomain][facgroup][facsubgrp] → list of factype
```

## Pipeline

```
Socrata ji82-xba5 → fetch (dict-filtered where clause)
                  → normalize (canonical schema + upper-case hierarchy levels)
                  → drop bad-geocode rows (bin='0' AND lat/lon outside NYC)
                  → drop null facdomain
                  → attach_geometry (shared geom.py: BIN-exact → nearest-200ft → point)
                  → buffer 80 ft in EPSG:2263 → WGS84
                  → validate (7 fatal + warn checks)
                  → write facilities.parquet / .geojson + coverage.geojson
```

Shares `dagspaces/common/curation/geom.py` (3-stage BIN match) with [[scaffolding-permits-curation]] — extracted during this build so both sub-dataset families use the same polygon-buffering machinery. Column-name parameters (`id_col`, `bin_col`, `lat_col`, `lon_col`) let the two modules reuse despite slightly different schemas.

## Output schema

| Column | Notes |
|---|---|
| `uid` | FacDB primary key (stable across versions) |
| `permit_id` | Aliased from `uid` for the shared `geom.attach_geometry` API |
| `facname`, `address`, `city`, `zipcode`, `borough`, `borocode` | Identity |
| `facdomain`, `facgroup`, `facsubgrp`, `factype` | Hierarchy (uppercased) |
| `bin`, `bbl`, `latitude`, `longitude`, `xcoord`, `ycoord` | Spatial keys |
| `raw_latitude`, `raw_longitude` | Aliases (match shared geom API) |
| `capacity`, `captype`, `opname`, `overagency`, `overlevel`, `servarea` | Ops metadata |
| `cd`, `council`, `nta2020`, `ct2020` | Admin geographies |
| `datasource` | Which upstream source file this row came from |
| `geom_source` ∈ {`bin_polygon`, `nearest_polygon`, `point`} | Match stage |
| `match_dist_ft` | Distance for nearest fallback |
| `geom_wkb` | Buffered polygon as WKB binary (parquet) |

## Validation

Mirrors permits' validation with FacDB-specific tweaks:

**Fatal (7):** Socrata returned ≥ 1 row · unique `uid` · every row has supported `geom_source` · no null `facdomain` · all buffered geoms valid · all inside NYC bbox · `coverage` non-empty + valid.

**Warn notables:** polygon match rate below threshold (default 75%, lower than permits' 85% because FacDB has many park/roadway rows with no BIN) · null `facname` count (kept, some legit) · per-`facdomain`, per-`facgroup`, per-`borough`, per-`geom_source` breakdowns · total coverage area + % of NYC land.

Fatal failures still write `summary.md` + `validation_report.parquet` + a `FATAL` `manifest.json` so diagnosis doesn't require a second Socrata pull.

## Built sub-datasets

| Dir | Filter | Raw → Publishable | Polygon match | Coverage |
|-----|--------|------------------:|--------------:|---------:|
| `facdb_libraries/` | `facgroup=LIBRARIES` | 255 → 253 | 100.00% (96.44% bin + 3.56% near) | 1.53 km² |

## Integration with the Cyclomedia pipeline

Downstream consumers use `coverage.geojson` + `facilities.parquet` (keyed by `unit_uid`) the same way permits are consumed:

```bash
# materialize: sjoin Cyclomedia catalog points vs per-unit buffered polygons.
# As of 2026-04-22, the materialize path also
#   (a) keeps only the *closest* unit attribution when a point sits inside
#       overlapping buffers (Fix B in [[concept-facing-filter]]),
#   (b) runs the new per-unit filter-facing automatically: the face must face
#       its attributed unit (A), within a 45° cone (C), within 200 ft (D),
#       and emits an attribution_confidence score ∈ [0, 1] (E).
sbatch --export=ALL,OUTPUT_FILENAME=cyclomedia_near_libraries.parquet \
    scripts/materialize_scaffolding_cyclomedia.sub \
    curation/facdb_libraries

# sample images for inspection — prefer the _facing parquet
python -m dagspaces.common.curation sample-images \
    --parquet <facdb-sub-dataset>/cyclomedia_near_libraries_facing.parquet \
    --out <facdb-sub-dataset>/inspect_k100_symlink \
    -k 100 --symlink --stratify-by dataset --seed 0
```

Empirical: on the pre-Fix-B libraries materialization, the new per-unit filter shrinks 59,080 → 13,632 rows (23%) while fixing the "face tagged to library X but depicts library Y" failure the old dissolved-coverage filter didn't catch. See [[concept-facing-filter]] for the recipe, knobs, and diagnostic columns (`bearing_to_unit_deg`, `delta_bearing_deg`, `distance_to_unit_ft`, `attribution_confidence`).

For standalone post-hoc filtering (e.g. to re-run with a different bearing tolerance or distance cap), `filter-facing` is still available as its own CLI subcommand — now with `--units` / `--bearing-tol-deg` / `--max-distance-ft`.

## Related

- [[concept-facing-filter]] — the A–E facing pipeline (per-unit attribution + confidence score)
- [[scaffolding-permits-curation]] — sibling curation, same module structure
- [[cyclomedia-catalog]] — downstream consumer of `coverage.geojson`
- [[urban-pair-vqa]] — sampler that will weight pair draws by `attribution_confidence`
- `curation/facilities_data_dictionary.xlsx` — upstream reference (25v2)
- `dagspaces/common/curation/facdb/categorization.json` — frozen hierarchy checked into repo
