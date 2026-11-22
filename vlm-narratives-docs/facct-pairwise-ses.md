# The pairwise results against the area features (FAccT)

`research/notebooks/facct/_pairwise_ses.py` 1.1.0. Written 2026-10-30.

## What it is

The FAccT paper brings back 3 pairwise cases of the CVPR battery. The CVPR
figure (`research/notebooks/cvpr/_proxy_overview.py`, see `proxy-overview.md`) read
each case against outside measurements and showed the white-box lens read
beside it. This figure reads the same area scores against the 5 ACS features
of the F4, F8 and F9 figures. It has no white-box read and no ablation panel,
and it is 1 column wide (3.33 in).

| Case | Question | Area score |
|------|----------|------------|
| Subway safety | which station entrance looks safer | `research/notebooks/cvpr/subway/outputs/subway_aggregated_v1.0.0.parquet` |
| Street photography | which block is a more appealing location for a street photography photoshoot | `_maps.image_scores_by_layer` (5 images minimum), cached in `data/facct_pairwise/` |
| Restaurants | which restaurant would you rather eat at | `research/notebooks/cvpr/restaurants/outputs/restaurants_aggregated_v1.0.0.parquet` |

The scores come from the registered 1,000,000-pair runs of Gemma-4-12B and
Qwen3.5-9B (no thinking). Each run has 1,100,000 rows: 1,000,000 unique pairs
and 100,000 repeats for the retest. The old figure used the same 2 exports.

## The features

`research/scripts/f4/nta_acs.py` 1.2.0 writes the 5 features (% white, % Black and %
Asian, all non-Hispanic; % bachelor's; median household income) at 3 layers,
by the same method (ACS 2024 5-year tract counts, summed over the area):

| Layer | File | Key | Areas with residents |
|-------|------|-----|----------------------|
| Census tract | `data/f4_traces/tract_acs.parquet` | `boroct2020` | 2,243 |
| NTA | `data/f4_traces/nta_acs.parquet` (no change) | `nta2020` | 214 |
| Community district | `data/f4_traces/cd_acs.parquet` | `boro_cd` | 59 |

A district is the union of the tracts of its CDTA ("CD equivalent"). A CDTA
follows the tract edges, thus it is near the district and not equal to it.
The NTA file is identical to the file before 1.2.0 (checked).

## The test

| Item | Value |
|------|-------|
| Statistic | Kendall tau-b of the area score with 1 feature, over the areas with residents |
| Floor | 20 areas (`_results_table.MIN_AREAS`) |
| Correction | Bonferroni over the 90 tests of the figure (3 cases x 5 features x 2 models x 3 layers), alpha 0.05 |
| Sign | positive: the model picks the area with more of the feature |
| Areas (tract, NTA, CD) | subway 656, 142, 57; street photography 2,242, 197, 59; restaurants 1,495, 197, 59 |

## Result (2026-10-30)

48 of the 90 cells are significant. In all 3 cases, both models pick the
areas with more white residents (tau 0.13 to 0.38) and more college
graduates. % bachelor's gives the largest tau: 0.35 to 0.48 for subway
safety, 0.21 to 0.41 for street photography, 0.09 to 0.44 for restaurants.
Income is strong for subway safety (0.30 to 0.39) and weaker for the other 2
cases. Every % Black tau is negative (-0.26 to -0.05), but only 6 of the 18
cells pass Bonferroni (4 restaurants, 2 street photography, 0 subway safety).
% Asian is small (0.02 to 0.19).

The tau grows from the tract to the district, as in the CVPR granularity
result, but n falls. With 57 to 59 districts, a district cell needs a tau
of about 0.31 to pass Bonferroni (the smallest district tau that passes is
0.35; the largest that does not is 0.31). Thus some district markers are
hollow near 0.3.

## Outputs

| File | Content |
|------|---------|
| `research/notebooks/facct/figures/pairwise/pairwise_ses.png`, `.pdf`, `.json` | the figure and its caption sidecar |
| `research/notebooks/facct/figures/pairwise/pairwise_ses_cells.csv` | 1 row for each cell: n, tau, p, p_bonf, significant |
| `research/notebooks/facct/figures/pairwise/pairwise_ses_restaurants_f6.csv` | the 30 restaurant cells in the 3 variants, with `tau_all_same` |
| `research/notebooks/facct/figures/pairwise/pairwise_ses_restaurants_f6_share.csv` | the tau of the `Yes` share and the `No` share of an area with each feature |

```bash
.venv-mllmsci-vllm025cu129/bin/python research/notebooks/facct/_pairwise_ses.py
```

The first run scores 1,100,000 street photography pairs for each model and
layer (about 2 minutes on a CPU node). Later runs read the cache.

## The word blocks of the same cases

`research/scripts/export_wordblocks.py` 1.3.0 adds restaurants and street
photography. The block reads the version 2 trace corpus
(`data/trace_extractions_v2_gemma12b_by_qwen27b`). The photographs need a
ranking of units, and the trace run (about 11,000 pairs) is too thin for
these 2 cases:

| Case | Problem in the trace run | Fix |
|------|--------------------------|-----|
| Street photography | it pairs images; each image is judged about 1 time | `unit_win_rates(proxy_rank=True)`: each image is a unit, ranked by the 1,000,000-pair run of the same judge |
| Restaurants | only 3 restaurants reach 8 decided comparisons, thus the top and the bottom rows showed the same 3 | the same option: the restaurants are ranked by the 1,000,000-pair run |

The 2 runs ask the same question with the same image layout; the proxy run
does not think. The caption says where the ranking comes from.
Outputs: `research/notebooks/cvpr/<case>/figures/<case>_win_block.png` and
`_win_block_column.png`, with `.json` sidecars.

### The restaurant photographs are F6 views

**Warning:** the pairwise runs (`cyclomedia_near_restaurants_facing`) show raw
cube faces, not the F6 views. The F6 reprojection came later, for the
single-image experiments (`f6-restaurant-image-selection.md`). The capture
vehicle shows in many raw faces, and many raw faces show no storefront.

The photographs of the restaurant blocks are therefore NOT the images that
the model judged. `f6_units` keeps the ranking and the win rate, and takes the
photograph from the F6 set, with these rules:

| Rule | Value |
|------|-------|
| Pixel check | `Yes` (`facade_check_b300/part_*.parquet`) |
| Camera to storefront | 35 ft or less |
| Angle to the wall normal | 20 degrees or less |
| Screened by eye | 8 low-ranked restaurants whose view does not show the storefront (`F6_SCREENED`, with the reason for each; the names go in the sidecar) |

The rules are the same at both ends, but the screen acts on the low end only.
The top 6 needed no screen. The caption states all of this.

### The street photography photographs are cropped

F6 covers restaurants only, thus this block shows raw faces, and the capture
vehicle fills the bottom of many of them. `CROP_BOTTOM` in
`research/scripts/export_wordblocks.py` removes the bottom 30% of the height.
`_photo_row` then cuts the same width at the 2 sides in equal parts, thus the
photograph stays square. The roof of the vehicle reaches about 0.71 of the
height in a forward or back face, thus 0.30 is the smallest round value that
removes it. We looked at the 12 uncropped faces and at both exported forms: no
part of the vehicle remains. The model judged the full image, and the caption
says this.

### The bottom of the ranking is partly an image effect

The F6 pixel check gives a test of what the lowest ranks hold (Gemma-4-12B,
18,488 restaurants, 10 bins of the shrunk win rate):

| Bin | F6 `Yes` | F6 `No` | No F6 view |
|-----|----------|---------|------------|
| Lowest | 28.0% | 31.5% | 40.5% |
| Highest | 68.5% | 4.3% | 27.3% |

The check reads the F6 view, not the raw faces that the model judged, thus
this is indirect evidence. It says that a low rank can mean "no storefront
in the image" as well as "a storefront that the model does not like".

### The gradients hold on the `Yes` restaurants

`python research/notebooks/facct/_pairwise_ses.py --f6` repeats the 30 restaurant cells
in 3 variants, with the same steps and the same Bonferroni count (90):

| Variant | Restaurants | Comparisons | Cells that pass | Change of tau on the same areas (mean, range) |
|---------|-------------|-------------|-----------------|-----------------------------------------------|
| `all` | all | all | 16 of 30 | the figure; all 30 taus agree exactly |
| `yes` | 10,046 F6 `Yes` | all | 21 of 30 | +0.030 (-0.012 to +0.138) |
| `yes_pairs` | 10,046 F6 `Yes` (Qwen3.5-9B 10,044) | both restaurants `Yes` | 20 of 30 | +0.033 (-0.026 to +0.161) |

No cell loses significance, and no sign changes. 26 of 30 taus (`yes`) and 25
of 30 (`yes_pairs`) move away from 0. % bachelor's moves most, +0.07 on
average. Thus the raw-face problem does not make the gradients. It hides a
part of them, and the figure with all the restaurants is the conservative form.

The path that we feared also goes the other way. The share of `Yes` restaurants
falls with % white, % bachelor's and income (tau -0.10 to -0.34) and rises with
% Black (0.05 to 0.27). 12 of the 15 `Yes` cells pass Bonferroni over the 30
share tests. The share of `No` is flat at the NTA and the district; 2 tract
cells pass. Thus the better areas do not win because more of their restaurants
show a storefront.

**Warning:** the pixel check read the F6 view, not the raw faces that the model
judged. A `Yes` restaurant can still have raw faces with no storefront. Thus
this check removes a part of the problem, not all of it. A pixel check on the
raw faces is the direct test.

Decision: the figure keeps all the restaurants, because those are the images
and the runs of record. Report the `Yes` result as a robustness check.

## 2 repairs to the CVPR pipeline (2026-10-30)

The project moved from `/share/pierson/matt/mllmsci` to
`/share/ju/matt/mllmsci`. 2 records still named the old root:

| Record | Symptom | Repair |
|--------|---------|--------|
| The 243 links of `research/notebooks/cvpr/canonical_data/` | the gate `_canonical.verify_or_raise` failed: "the link moved" | each link now names the same file under `/share/ju`; all 81 runs exist there with the recorded size, and the gate passes |
| `source_results_path` in the extraction corpora | `_extractions.load(require_canonical=True)` failed | `_extractions.relocated` (1.3.0) reads an old-root path under the repo root; some nodes still mount the old root, thus the map applies always |

The 14 links that still do not resolve are old SLURM submission files inside
the run folders. No reader uses them.
