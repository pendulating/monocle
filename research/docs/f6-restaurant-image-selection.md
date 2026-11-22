# F6: how we select 1 storefront image for each restaurant

A one-page record of the method and its development, for the methods section of
the FAccT paper. It records facts and numbers; it is not paper text. Details,
decisions (D28, D41 to D84), and job ids: `facct-runbook.md`, section F6.
Status: complete (2026-10-24). The final set is
`data/f6_restaurant_depiction/v4/facade_check_b300/drawn_10000.parquet`.

## The goal

Each restaurant image of the single-image experiments (F4, F5) must show the
restaurant's storefront, sharp and from close by (D28). The target is 10,000
different restaurants, 1 image each (D66, D68).

## The inputs

| Input | Value |
|-------|-------|
| Restaurants | NYC DOHMH inspection data, pull of 2026-04-27: 26,973 restaurants with a geocode P, a BIN and a street address |
| Buildings | NYC building footprints, joined by BIN |
| Streets | NYC CSCL street centerlines (103,429 segments, with names) |
| Images | Cyclomedia 360-degree captures of 2025, about every 17 ft along each driven lane: 2,582,805 (Aug-Nov) and 2,651,759 (Mar-Jun) |

## The final method (v4)

| Step | Rule |
|------|------|
| 1. Storefront wall | The footprint wall that is parallel to the address street (within 25 degrees) and faces it, nearest P. Fallback (1,028 restaurants): the wall that faces P. No BIN footprint: the restaurant leaves (337) |
| 2. Aim point S | The wall point nearest P. P is on the street centerline and is often past the end of the building, so in that case S is the middle of the wall (52% of restaurants) |
| 3. Candidate cameras | All captures within 80 ft of P, in front of the wall, with a clear sightline to S (at most 1 ft inside any building footprint) |
| 4. Open at capture | DOHMH has no closure flag. The restaurant has an inspection within 180 days of the capture date |
| 5. Limits | Angle theta between the camera direction and the wall normal at most 45 degrees; P at most 150 ft from S (a bad geocode) |
| 6. Score | Facade area in the frame, from the camera model: `cos(theta) / r^2`, r = camera to S |
| 7. Epoch | Aug-Nov 2025. Mar-Jun 2025 captures join only on the near side of the street, when Aug-Nov has none there |
| 8. Render | A virtual rectilinear camera at the best capture, reprojected from the 6 cube faces: aimed at S, pitch +10 degrees, 70 degree field of view, 1024 x 1024 |
| 9. Pixel check | Gemma-4-12B-it answers "Is a restaurant facade or food restaurant signage visible in this image?" (verbatim), structured `Yes` / `No`, greedy, thinking off. A "No" removes the restaurant (D63 to D69) |
| 10. Draw | 10,000 restaurants at random from the "Yes" pool, seed 20261023 |

The pixel-check model is also a model of the study. We accept this circularity
(D64).

## The funnel (v4)

| Step | Restaurants |
|------|-------------|
| DOHMH pull | 26,973 |
| With a BIN footprint and a wall | 26,594 |
| With a qualifying view | 18,242 |
| Rendered (all cube faces present) | 18,237 |
| Gemma "Yes" | 13,724 (75.3%) |
| Drawn | 10,000 (9,228 distinct captures; Manhattan 3,614, Brooklyn 2,778, Queens 2,215, Bronx 1,090, Staten Island 303) |

Drawn set: angle to the wall median 7.6 degrees (p95 32.3); camera to S median
36.5 ft (p95 59.5); 76% from the near side of the street; 2,056 from Mar-Jun.

## How the method developed

Each version was checked by hand: 100 random restaurants in Label Studio,
Pass or Fail on "the image truly depicts a restaurant, in high definition,
from close proximity".

| Version | What changed | Why | Hand check (labelled so far) |
|---------|--------------|-----|------------------------------|
| v1 | Existing filter: the cube face that points to the footprint centroid, top 10% of a distance score | the existing pipeline | 18 Pass of 21 |
| Review 1 (Fable advisor) | Found: the centroid is often on another side of the building; 1 capture went to 1 restaurant only; the ray test hit an 80 ft buffer | 3 fails were side walls or the wrong shop | |
| v2 | Aim at the footprint point nearest P; one-to-many attribution; open at capture; a virtual camera in place of a fixed cube face; best view per restaurant | fixes of review 1 | not labelled |
| v3 | Angle to the facade normal at most 20 degrees | side views in v2 | 51 Pass of 58 |
| v3 + model check | Gemma pixel check, then the draw of 10,000 | trucks and trees in front of storefronts | 65 Pass of 66 |
| Review 2 (Fable advisor) | Found: at a corner the normal was the 45 degree diagonal (46% of the drawn set), so oblique views passed the 20 degree limit; no limit on camera distance (max 645 ft) | the user saw side views and far views | |
| v4 | Wall from the address street; aim at the wall middle when P is past its end; area score from the camera model; limits as above; Mar-Jun near-side pass | fixes of review 2 | not labelled (project 6) |

Effect of v4 on the drawn set: the angle median 13.6 to 7.6 degrees; views over
30 degrees 3,330 to 578; far side of the street 33% to 24%.

## Known limits

| Limit | Size |
|-------|------|
| Restaurants at 1 address share P, so they can share 1 image (D79) | 1,388 drawn in v3 |
| The selection keeps restaurants with a usable capture; the pixel check keeps clear storefronts. Both can shift the set from the DOHMH population | v4 pool 18,242 of 26,973 |
| The pixel-check model is a model of the study (D64) | all 10,000 |

## Files

| What | Path |
|------|------|
| Selection, render | `research/scripts/f6_select_storefront_views.py` (2.0.0), `research/scripts/f6_render_best_views.sub` |
| Pixel check, draw, review | `research/scripts/f6_vlm_facade_check.py`, `research/scripts/f6_vlm_facade_check_g3.sub` |
| The figure of the virtual camera (1 example: the 6 faces, the sampled regions, the view, and the geometry JSON for the TikZ panels of `fig:facade-view`) | `research/scripts/f6_facade_view_example.py` -> `research/notebooks/facct/figures/facade_view/` |
| Outputs | `data/f6_restaurant_depiction/v4/` |
| Advisor reviews | `data/f6_restaurant_depiction/advisor-review-2026-10-23.md`, `advisor-review-2026-10-24-math.md` |
| Label exports | `data/f6_restaurant_depiction/exports/` |
