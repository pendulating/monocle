---
title: Facing filter — per-unit attribution + confidence
category: concept
created: 2026-04-22
updated: 2026-04-23
tags:
  - concept
  - curation
  - cyclomedia
  - facing
  - attribution
  - confidence
  - sampling
  - occlusion
sources: []
---

# Facing filter — per-unit attribution + confidence

Deciding *which* building-adjacent street-level face actually depicts the target building. Used by any curation family that materializes Cyclomedia images against per-unit buffered polygons — [[facdb-curation]] (libraries, schools, health, …) and [[scaffolding-permits-curation]] (DoB scaffolds).

The materialize step is permissive by design: it uses an 80-ft buffered polygon per unit and point-in-polygon against the recording position. That over-collects — a face on a sidewalk inside the buffer but pointing **away** from the building passes, and a recording inside N overlapping buffers is tagged N times. The facing filter is the corrective post-pass.

The pipeline is six fixes stacked, labeled A–F in the implementation. A, B, C, D, F are hard gates; E is a continuous score.

| Fix | Where | What it does |
|---|---|---|
| **A** — per-unit ray test | `filter_facing.py` | Ray from recording in face's bearing must land inside the row's **own** `unit_uid` polygon, not just any coverage polygon |
| **B** — distance-dedup at attribution | `permits/materialize.py` (`sjoin_dataset_chunk`) | When a point sits inside N overlapping unit buffers, keep only the attribution with smallest recording → unit centroid distance. Populates `unit_dist_ft` |
| **C** — bearing cone | `filter_facing.py` | Drop rows where \|face_bearing − bearing(recording → unit centroid)\| > tolerance. Default tolerance **45°** (the face's half-FOV — unit just needs to be inside the 90° face; the confidence score does the fine-grained prioritization) |
| **D** — distance cap | `filter_facing.py` | Drop rows where `distance_to_unit_ft` > cap. Default cap 200 ft |
| **E** — `attribution_confidence` | `filter_facing.py` | Continuous score ∈ [0, 1] = **visibility_fraction** × **cubic** distance. Visibility fraction is 1 when the unit fits entirely in the face; drops only when parts are clipped out of frame (physical model with H_REF≈25 ft unit half-width). Within the safe sub-cone, proximity alone ranks: 50 ft scores 3.4× 100 ft, 80 ft scores 1.7× 100 ft |
| **F** — line-of-sight occlusion | `filter_facing.py` | Drop rows whose segment (recording → unit's own BIN polygon's `representative_point`) is **strictly pierced** by another NYC building (BIN ≠ unit's BIN). "Strict pierce" = segment enters AND exits the non-unit polygon (both endpoints outside). Uses `nyc_buildings.parquet` |

## Quick summary of the flow

```
catalog  →  materialize-cyclomedia (sjoin + Fix B dedup) → cyclomedia_near_*.parquet
                                                        →  filter-facing (A + F + C + D, emits E)
                                                        →  cyclomedia_near_*_facing.parquet
```

## The five fixes in detail

### Fix A — ray hits the row's own polygon

Cast a 30-m ray from the recording in the face's absolute bearing; sample 3 points along it; keep the row if any sample lies inside the polygon of **this row's attributed `unit_uid`**, not the dissolved coverage.

Old (legacy coverage mode) kept a row as long as the ray hit *some* unit — which leaks cross-street images where the tagged library is on the wrong side and the ray is actually aimed at a different library. New self-match sjoin resolves that.

### Fix B — one attribution per point, the closest

`sjoin_dataset_chunk` no longer emits one row per (point, unit) match. When a recording sits inside overlapping buffers, compute distance to each candidate unit's centroid (EPSG:2263, feet) and keep only the smallest. `unit_dist_ft` now holds that distance instead of the old hardcoded 0.0.

### Fix C — the face must point at the unit

For each kept row, compute bearing from recording → unit centroid (projected CRS, atan2) and compare to `face_bearing`. Drop if \|Δbearing\| > `bearing_tol_deg` (default **45°**).

45° matches the face's actual half-FOV — anything inside the 90° face qualifies. The fine-grained "is the unit the focus of the image?" question is answered by Fix E's continuous score, not by a hard gate. The iteration history: we started at 45°, tightened to 22.5° when the score was all-quadratic (the tight cone compensated for a too-flat score at the sweet spot), and widened back to 45° once the score moved to linear angle × cubic distance (aggressive proximity weighting makes the tight cone redundant and unnecessarily dropped good close-off-axis shots).

### Fix D — the unit must be close enough

Hard cap on `distance_to_unit_ft`. Default 200 ft drops the long tail of across-plaza / large-campus shots. Configurable via `--facing-max-distance-ft` (materialize) or `--max-distance-ft` (filter-facing).

### Fix E — continuous confidence score

```python
# Physical: angular half-width of the unit in the face (degrees)
H_half = degrees(atan(H_REF / D))           # H_REF = 25 ft (characteristic)

# Fraction of the unit still visible after face-edge clipping
vis_right = min(Δθ + H_half, +bearing_tol_deg)
vis_left  = max(Δθ − H_half, −bearing_tol_deg)
visibility_fraction = max(0, vis_right − vis_left) / (2 × H_half)  # ∈ [0, 1]

# Cubic distance falloff within the 200-ft cap
distance_term = clip(1 − D / d_norm, 0, 1) ** 3

attribution_confidence = visibility_fraction × distance_term
```

Why this shape: **within the cone, physical proximity should dominate — an angular factor should kick in only when the unit is partially cropped from the frame**. The visibility-fraction term is 1.0 whenever the unit fits entirely inside the face (i.e., Δθ + H_half ≤ bearing_tol), so close-off-axis shots that sit fully in frame are ranked purely by distance. When Δθ pushes the unit past the face edge, the score drops only by the fraction of the unit clipped. Cubic distance makes proximity dominate: 50 ft scores 3.4× 100 ft; 80 ft scores 1.7× 100 ft.

Reference values under the current defaults (tol=45°, H_REF=25 ft, d_norm=200 ft):

| Δθ | dist_ft | H_half | visibility | dist³ | conf |
|---:|---:|---:|---:|---:|---:|
| 0° | 10 | 68.2° | 0.66 (building bigger than face) | 0.857 | 0.563 |
| 0° | 50 | 26.6° | 1.00 | 0.422 | 0.422 |
| 0° | 100 | 14.0° | 1.00 | 0.125 | 0.125 |
| 0° | 200 | 7.1° | 1.00 | 0.000 | 0.000 |
| 19° | 69 | 19.9° | 1.00 (19+20=39 ≤ 45) | 0.289 | 0.289 |
| 23° | 65 | 21.0° | 1.00 (23+21=44 ≤ 45) | 0.304 | 0.304 |
| **29°** | **65** | **21.0°** | **0.87** (5° clipped) | 0.309 | **0.269** |
| 42° | 65 | 21.0° | 0.56 (18° clipped) | 0.305 | 0.171 |
| 44° | 200 | 7.1° | 0.57 (6° clipped) | 0.000 | 0.000 |

### Fix F — line-of-sight occlusion (added 2026-04-23)

A+C+D+E are all either ray-vs-own-polygon or angle/distance tests; none of them know whether *another building* sits between the camera and the target. The empirical failure mode: side-street shots where the geometry is correct (face points toward the library's unit polygon, angle is acute, distance is short) but the image depicts the food market or residential building *in front of* the library.

Castle Hill Library (BIN 2022944) was the motivating case — four L-face / bearing=270° rows from the side street that passed A+C+D but showed Top Banana Food Market occluding the library.

Mechanism:
1. Load `nyc_buildings.parquet` (~1.08M polygons, dissolved by BIN, projected to EPSG:2263).
2. For each post-A row, look up the attributed unit's `bin` in the units parquet.
3. Build a `LineString` from recording → the unit building's `representative_point` (not centroid — representative_point is guaranteed inside even for concave footprints).
4. STRtree `sjoin(segment, buildings, predicate='intersects')`.
5. Drop matches whose `bin == unit_bin` (library blocking itself).
6. **Strict pierce test**: retain a match only if both segment endpoints lie outside the candidate polygon. sjoin already guarantees the segment intersects the polygon, so "both endpoints outside" ⇒ "enters AND exits" ⇒ polygon is fully between camera and target.
7. Row is dropped if any non-unit BIN strictly pierces the segment.

Rows whose attributed unit has no BIN (or whose BIN isn't in `nyc_buildings.parquet`) **pass through** as keep — the check can't anchor the target endpoint without a known building. In practice this is the rare point-fallback / nearest-polygon row.

Why strict (entry + exit) and not lenient (any intersection): a building that merely clips a corner of the segment is typically in the image's foreground frame (the camera is beside it, not looking through it), not an occluder. Enforcing entry + exit means the polygon is fully between camera and target.

Why `representative_point` and not centroid: centroids of L-shaped / courtyard / concave footprints can fall outside the polygon, which would break the "endpoint is inside library building" guarantee that keeps the strict-pierce check unambiguous.

### Performance

Loading `nyc_buildings.parquet` (~1.08M rows) takes ~33 s. The occlusion sjoin itself is ~2 s on 44 k post-A rows. Net cost of Fix F on `facdb_libraries`: +35 s per run.

## Diagnostic columns

After `filter-facing --units <path>` runs in per-unit mode, each output row carries:

| Column | Meaning |
|---|---|
| `unit_uid`, `unit_name`, `unit_dist_ft` | From materialize (Fix B); recording → unit centroid distance in US feet |
| `bearing_to_unit_deg` | Compass bearing from recording to unit centroid (0° = N, CW) |
| `delta_bearing_deg` | Unsigned angular difference vs face bearing, in [0°, 180°] |
| `distance_to_unit_ft` | Same as `unit_dist_ft`, recomputed in filter stage |
| `attribution_confidence` | Score ∈ [0, 1], higher = more confidently facing |

## Empirical behavior on `facdb_libraries`

Four re-materializations, same curation root. The bearing-frame flip (afternoon → evening) came from a catalog-side fix — see [[cyclomedia-catalog]]'s face-bearing caveat. The cone/score rebalance (evening → night) came from the audit showing close-off-axis rows scoring unreasonably low:

| Run | Bearing frame | Cone | Angular | Distance | Occlusion | Kept rows | % kept | Units |
|---|---|---:|---|---|---|---:|---:|---:|
| 2026-04-22 morning | vehicle-relative (buggy) | 45° | linear | linear | — | 13,632 | 23.1% | 246 / 253 |
| 2026-04-22 afternoon | vehicle-relative (buggy) | 22.5° | quadratic | quadratic | — | 5,497 | 10.0% | 235 / 236 |
| 2026-04-22 evening | **absolute (F=N)** | 22.5° | quadratic | quadratic | — | 6,376 | 11.6% | 234 / 236 |
| 2026-04-22 night | absolute (F=N) | **45°** | **linear** | **cubic** | — | 12,947 | 23.5% | 236 / 236 |
| 2026-04-23 (current) | absolute (F=N) | 45° | linear | cubic | **strict pierce** | **11,721** | **21.2%** | **236 / 236** |

Rebalance notes:
- Fix F drops another ~3,265 rows (7.8% of the 41,653 post-A rows that have a library BIN) — physically occluded shots where the camera's LOS to the library's own BIN polygon is pierced by another NYC building. Castle Hill Library (BIN 2022944) lost its 4 L-face/bearing=270° side-street shots that were pointing at Top Banana Food Market.
- Opening the cone 22.5° → 45° doubled the kept row count and restored coverage to 236/236 libraries. The angular cone was over-protective given that the continuous score already penalizes edge-of-frame rows.
- Angular linear (not quadratic) + distance cubic (not quadratic) is the "proximity is what matters once you're in the cone" stance — close-at-moderate-angle beats far-and-centered.
- Per-face distribution on the 11,721 kept rows after Fix F is balanced (F=3040, R=3031, B=2851, L=2799) — the L-face (leftward) count drops slightly more than F/R/B because left-of-vehicle shots more often come from side streets where occlusion is present.
- Confidence distribution on the 11,721 kept rows: mean 0.096, median 0.081. Absolute values shifted up vs the pre-F 12,947-row run (mean 0.050) because Fix F preferentially drops the lower-confidence side-street occluded rows.

## CLI

**Per-unit mode** (recommended, requires a units parquet keyed by `unit_uid`):

```bash
python -m dagspaces.common.curation filter-facing \
    --parquet  curation/facdb_libraries/cyclomedia_near_libraries.parquet \
    --units    curation/facdb_libraries/facilities.parquet \
    --out      curation/facdb_libraries/cyclomedia_near_libraries_facing.parquet \
    --bearing-tol-deg  45 \
    --max-distance-ft  200 \
    --overwrite
```

**`materialize-cyclomedia`** runs the above automatically after producing the unfiltered parquet. Flags:

- `--facing-bearing-tol-deg` (default **45°**)
- `--facing-max-distance-ft` (default 200, pass a large number to disable)

Both defaults are sourced from `filter_facing.DEFAULT_BEARING_TOL_DEG` / `DEFAULT_MAX_DISTANCE_FT` so there's one source of truth.

**Legacy dissolved-coverage mode** (no per-unit attribution) still works by passing `--coverage` without `--units`. Kept for backward compatibility but prefer per-unit.

## Knobs and tuning

| Knob | Default | What it trades |
|---|---:|---|
| `bearing_tol_deg` | **45°** | Tighter (e.g. 22.5°) kills close-off-axis shots the cubic distance term would otherwise rescue; looser has no meaning (a face only sees 90° anyway) |
| `max_distance_ft` | 200 | Tighter (e.g. 120) = only near-sidewalk; looser = keeps campus-far shots |
| `ray_length_m` | 30 m (= ~98 ft) | Shorter = stricter "can I actually see it"; longer = forgives distant but on-axis shots |
| `confidence_normalize_ft` | 200 ft | Only affects E's score normalization when `max_distance_ft` is None |
| `occlusion` | **True** | Disable only for smoke tests. Disabling restores the pre-F behavior where side-street occluded shots pass A+C+D |
| `buildings_path` | `data/geo/nyc_buildings.parquet` | Path to the NYC buildings parquet (BIN + geometry); same file `geom.py` uses for the materialize stage |

## Known edge cases

- **Libraries with zero kept rows.** Under the current 45°/linear-angle/cubic-distance defaults, **0 / 236** attributed libraries end up empty. Prior tighter configurations left 1-7 empty; usually a symptom of: BIN polygon doesn't cover the actual facade (retired / wrong BIN), building footprint is very small vs its 80-ft buffer, or the library occupies the top floors of a mixed-use building so the catalog face bearings rarely intersect the polygon vertically.
- **Historical bug: bearings were vehicle-relative, should have been absolute.** Until the 2026-04-22 evening rematerialize, `cyclomedia_catalog.indexer::_compute_bearing` added `recorderDirection` to `FACE_BEARING_DEG[face]`. That treated the cube as rotating with the vehicle; empirically Cyclomedia's NYC panoramas are always rendered F=N/R=E/B=S/L=W regardless of which way the van was driving. Symptom in the audit: recordings captured in opposite directions on the same street showed the same scaffolded building in F and the same library in B, which is impossible if F were vehicle-forward. Fix: drop the `recorder_direction` term in `_compute_bearing`; see [[cyclomedia-catalog]].
- **Dark / night-time faces.** A+C+D is purely geometric — an unviewable image (near-black, heavily occluded) passes the filter if its geometry is correct. Image-quality filtering is a separate concern not handled here.
- **`distance_to_unit_ft` is to *centroid*, not edge.** For a 40-ft-deep building, the centroid is ~20 ft inside the footprint, so "distance to building edge" ≈ `distance_to_unit_ft − 20`. Callers that want polygon-edge distance need to store the unbuffered footprint — not a schema the curation parquets expose today.
- **Fix F is BIN-anchored.** Units without a BIN (geom_source ∈ {`nearest_polygon`, `point`}) can't supply a target building polygon, so the occlusion check falls through as "keep". In practice this is the rare FacDB row with no BIN match; most libraries have a real BIN.
- **Fix F uses `representative_point`, not centroid.** A concave library footprint (L-shape, courtyard) can have its centroid outside the polygon. `representative_point` is guaranteed inside and gives a clean "endpoint-outside-candidate" test.

## Weighted pair sampling (E's downstream) — implemented 2026-04-22

Fix E's `attribution_confidence` now feeds [[urban-pair-vqa]]'s `build_unit_random_pairs` via `weight_column: Optional[str]`. When set, for each unit the sampler builds a per-unit weight vector (clip NaN/negative to 0, then normalize); image draws within that unit use `rng.choice(indices, p=weights)` instead of uniform. Unit-level pair selection stays uniform so per-library coverage for TrueSkill is preserved.

- **Hydra**: `pair_sampler.weight_column: attribution_confidence` in `conf/pipeline/pairwise_libraries_mvp.yaml`; default `null` in `conf/config.yaml`.
- **Fallbacks**: missing column → uniform + warning; per-unit all-zero → uniform for that unit only; negative/NaN → clipped to 0.
- **Tests**: `tests/test_pairwise_unit_sampler.py::TestWeightedWithinUnit` covers the weighted-bias, missing-column, per-unit-all-zero, and negative/NaN-clipped paths.

Variants with the same surface but deferred:
- **Top-K per unit** (`weight_top_k: int`) — keep top K by weight, then uniform. Stricter than continuous weighting.
- **Confidence floor** (`min_confidence: float`) — hard pre-filter; cleaner than relying on `weight=0`.

Expected effect on the MVP libraries run (45° + linear angle + cubic distance): mean confidence ≈ 0.050 (max 0.51). The weighted sampler biases strongly toward close, centered rows — the cubic distance term makes this explicit (50 ft scores 3.4× 100 ft).

## Related

- [[facdb-curation]] — one of the two curation families that feeds this filter
- [[scaffolding-permits-curation]] — the other curation family
- [[cyclomedia-catalog]] — upstream source of every face; defines `face`, `bearing`, recording geometry
- [[urban-pair-vqa]] — the dagspace whose sampler Fix E's score is about to feed
