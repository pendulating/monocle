---
title: "Pairwise Regression Testing (covariate regressions over urbanpairvqa ratings)"
category: guide
created: 2026-06-11
updated: 2026-06-11
tags:
  - guide
  - pairwise
  - statistics
  - regression
  - registry
  - wandb
---

# Pairwise Regression Testing

Targeted regressions over a [[urban-pair-vqa]] run: "how much of the VLM's per-school rating is explained by poverty rate?" Implemented in `scripts/pairwise_vqa_regression_report.py`; the regression-shaped sibling of [[guide-pairwise-difference-testing]], sharing its registry, W&B mirror, and join machinery via `scripts/pairwise_analysis_common.py`.

## What it does

| Piece | Detail |
|---|---|
| Response | Per-unit TrueSkill μ (or μ−3σ via `--y ts_conservative`) from the full run ([[concept-trueskill]]) |
| Unit-level fit | OLS (HC3 robust SEs) of μ on x; `--controls` adds covariates (categoricals auto-dummied) with focal **partial R²**; `--wls` weights by 1/σ²; standardized β; Spearman ρ |
| Pair-level validation | `relative_score` regressed on Δx = x_a − x_b over direct pairs (repeats collapsed) — immune to TrueSkill coupling |
| Screen mode | `--x-list a,b,c` — one regression per covariate, BH-corrected, ranked bar chart |
| Multi-model | `--aggregation-dir` — per-model fits, forest plot of standardized β |
| Diagnostics | Scatter + fit + decile means, residuals, named top-Cook's-d units, attenuation note (mean σ vs sd(μ)) |
| Registry / W&B | Shared registry (`mode: regression`/`screen`), W&B `job_type=regression` in `URBANPAIRVQA-ANALYSIS` |

## Covariate sources

Same resolution as the difference tool: surfaced `<col>_a/_b` pair metadata, else `--unit-metadata-parquet` joined on `unit_uid`. For schools use **`curation/external/school_covariates.parquet`** (id column `uid`), built by `scripts/build_school_covariates.py`:

- FacDB facilities → BIN/BBL/borough/cd/sector/capacity
- DOE locations 2019-20 → BIN→DBN (million-BIN placeholders dropped)
- DOE demographic snapshot 2017-22 → latest year per DBN, enrollment-weighted per building: `pct_poverty`, `economic_need_index`, `pct_swd`, `pct_ell`, `total_enrollment`
- PLUTO 26v1 via BBL: `yearbuilt`/`building_age`, `bldgclass`, `numfloors`, `assesstot`, `assess_per_bldg_sqft`, `log_assesstot`

DOE fields are null for most non-public schools — regressions on them describe the DOE sector only. Restaurants need no builder: `restaurants_aggregated.parquet` has `last_score` etc. (join on `camis`).

## Usage

```bash
# The motivating question
python scripts/pairwise_vqa_regression_report.py <schools_run>.parquet \
    --x pct_poverty \
    --unit-metadata-parquet curation/external/school_covariates.parquet \
    --attribute "rather send your child to" --unit-label school --pdf

# Controlling for borough (focal partial R² reported)
... --x pct_poverty --controls borough

# Screen
... --x-list pct_poverty,economic_need_index,building_age,log_assesstot

# Multi-model
... --aggregation-dir <sweep_dir> --x pct_poverty ...
```

## Validated findings (2026-06-11, good June 8 Qwen3.5-9B runs)

- **Schools:** preference declines with poverty — β* = −0.156, unit p = 1e-7; pair-level slope −0.0029/poverty-point, p = 3e-34 (n = 23,682 direct pairs). Survives borough control (partial R² = 0.022). Screen: ENI ≈ poverty ≫ everything else; building_age weakly −, log_assesstot weakly +.
- **Restaurants:** preference does **not** track DOHMH inspection score (R² = 0.0000, both levels null) — inspection results aren't visible from the street.

## Caveats (printed in every report)

- μ's are coupled → unit-level p approximate; pair-level slope is the clean check
- Noise in μ attenuates R²; reports quote mean σ / sd(μ)
- Coverage selection (e.g. DOE-only poverty); association ≠ causation
- Screen: BH within screen only

## Tests

`tests/test_pairwise_regression.py` (17): planted slope/R² recovery, null, Δx orientation, categorical-control partial-R² collapse, WLS downweighting, value-map paths, screen BH, registry id sensitivity, end-to-end CLI (single / controls / screen).
