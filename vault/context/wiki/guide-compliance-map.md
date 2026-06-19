---
title: Scaffolding Permit Compliance Map
category: guide
created: 2026-04-12
updated: 2026-04-12
tags: [scaffolding, compliance, dob, permits, folium, geospatial, mvp]
---

# Scaffolding Permit Compliance Map

Interactive maps classifying detected scaffolding as permitted, expired, or unpermitted by cross-referencing VLM-based embedding detections with NYC DoB permit data. Two approaches, same DoB cross-reference.

## Two Approaches

| | Rerank-based | Raster-based |
|--|--------------|--------------|
| | Rerank-based | Raster-based |
|--|--------------|--------------|
| **Notebook** | `scaffolding_compliance_map.ipynb` | `scaffolding_compliance_raster.ipynb` |
| **Input** | `outputs/rerank/*.parquet` + building footprints | `outputs/raster/` GeoTIFFs (ray-accumulated heatmaps) |
| **Matching** | Building footprint buffer (configurable, default 50 ft) | KD-tree point-to-point (100m radius) |
| **Detection unit** | Camera position matched to building polygon | Hotspot centroid from converging rays |
| **Count** | 2,019 unique camera locations | ~1,059 hotspots (area >= 3 pixels) |
| **Spatial accuracy** | Building-shape-aware (buffered footprint polygon) | Estimated scaffold position (ray triangulation) |
| **Type info** | Implicit (which rerank run) | Explicit (green/white from signed_diff raster) |
| **Building-centric view** | Yes — buildings with permits vs. detection presence | No |
| **Best for** | Building-level compliance auditing | Geospatially accurate scaffold location mapping |

## File Inventory

| File | Purpose |
|------|---------|
| `notebooks/scaffolding/scaffolding_compliance_map.ipynb` | Rerank-based notebook |
| `notebooks/scaffolding/scaffolding_compliance_raster.ipynb` | Raster-based notebook |
| `notebooks/scaffolding/cache/` | Cached DoB API responses (parquet, shared) |
| `notebooks/scaffolding/scaffolding_compliance_map.html` | Rerank folium map (generated) |
| `notebooks/scaffolding/scaffolding_compliance_raster.html` | Raster folium map (generated) |
| `notebooks/scaffolding/scaffolding_compliance_classified.parquet` | Rerank classified detections |
| `notebooks/scaffolding/scaffolding_compliance_raster_classified.parquet` | Raster classified hotspots |

## Raster Approach Details

The raster notebook uses GeoTIFFs from the `artifact_gen` dagspace:
- `green_scaffolding.tif` — ray-accumulated relevance for "green scaffolding" query
- `white_arched_fancy_light_scaffolding.tif` — same for "white arched fancy light scaffolding"
- `scaffolding_types_signed_diff.tif` — green minus white (positive = green dominant)

**Hotspot extraction:**
1. Compute `combined = max(green, white)` per pixel (type-agnostic detection)
2. Threshold at configurable percentile (default p95)
3. scipy `ndimage.label()` for connected component extraction
4. Filter by minimum area (default 3 pixels = 300m2)
5. Classify type from `signed_diff` at centroid: green (>0.005), white (<-0.005), ambiguous

**Raster specs:** 987x2233 grid, 10m resolution, EPSG:4326, 1,038,932 input images via 30m ray casting with linear decay.

## Data Sources (shared)

| Source | Records | Key Fields |
|--------|---------|------------|
| NYC Building Footprints (`data/geo/nyc_buildings.parquet`) | 1,082,872 | `bin`, `geometry` (MultiPolygon), `height_roof` |
| DoB NOW Filings (`w9ak-ipjd`) | ~50-100K citywide | `scaffold`, `shed`, `first_permit_date`, `signoff_date`, `bin` |
| DoB Permit Issuance (`ipu4-2q9a`) | Batch by BIN | `expiration_date`, `permit_subtype` |

### Building-Centric View (rerank notebook)

The rerank notebook also provides an inverted perspective:

| Building Status | Detection? | Interpretation |
|-----------------|-----------|----------------|
| Active permit | Yes | Confirmed — scaffold detected + permitted |
| Active permit | No | Permitted but not detected (removed? occluded?) |
| Expired permit | Yes | Lingering — scaffold still standing |
| Expired permit | No | Resolved — scaffold removed |

## Classification Logic

**Rerank notebook:** Building footprint polygons buffered by 50 ft (configurable) in EPSG:2263, spatial join via `gpd.sjoin(predicate='within')`. Detection falls within buffered footprint of a scaffold-permitted building → matched.

**Raster notebook:** KD-tree nearest-neighbor on EPSG:2263, 100m radius (point-to-point).

| Status | Color | Rule |
|--------|-------|------|
| **Permitted** | Green | Nearest filing has active permit or recently-issued permit without expiration data |
| **Expired** | Orange | Nearest filing has expired permit (`expiration_date < today`) or completed work (`signoff_date` set) |
| **Unpermitted** | Red | No scaffold/shed filing within 100m, or filing never received a permit |

**Permit lifecycle derivation (priority order):**
1. `expiration_date` from Dataset 2 (BIN join) — `active` if future, `expired` if past
2. `signoff_date` from Dataset 1 — `completed` (scaffold should be removed)
3. `first_permit_date` from Dataset 1 without expiration — `active_no_expiry_data`
4. No permit date — `no_permit`

## DoB API Join

The two DoB datasets use different job numbering systems:
- **Dataset 1** (DOB NOW): alphanumeric `job_filing_number` (e.g., `B00047864-I1`)
- **Dataset 2** (legacy BIS): numeric `job__` (e.g., `321496519`)

Join via **BIN** (Building Identification Number), filtering to scaffold-related `permit_subtype` values: `SH` (sidewalk shed), `SD`, `SF`.

## Caveats

- Scaffolding under 40 ft is exempt from permit requirements (NYCBC 3314.2)
- 100m match radius is approximate
- Multiple scaffolds can coexist near the same location
- "Permitted" with `active_no_expiry_data` may include technically-expired permits
- Detection scope: Manhattan only (Cyclomedia 2025 embeddings); DoB permits shown citywide

## Related

- [[urban-embed]] — Embedding pipeline producing rerank detections and embeddings for rasterization
- [[artifact-gen]] — GeoTIFF raster generation from embeddings (input to raster notebook)
- [[guide-validation-pipeline]] — Evaluation methodology (DOB cross-reference is Phase 6)
- `viz/scaffolding_map/` — deck.gl visualization of green/white scaffolding rasters + DoB overlay
