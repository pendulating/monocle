---
title: "Pairwise Difference Testing (t-tests over urbanpairvqa groups)"
category: guide
created: 2026-06-11
updated: 2026-06-11
tags:
  - guide
  - pairwise
  - statistics
  - difference-testing
  - registry
  - wandb
---

# Pairwise Difference Testing

On-the-fly group-difference tests over a [[urban-pair-vqa]] run: "would the VLM rather eat at *Chinese* restaurants than *Italian* ones?" Implemented in `scripts/pairwise_vqa_difference_report.py`; companion to the single-run report (`scripts/pairwise_vqa_report.py`, see [[concept-trueskill]]).

## What it does

| Piece | Detail |
|---|---|
| Group assignment | A metadata column maps each unit to a group. Two sources: (1) columns already surfaced on `pairs.parquet` as `<col>_a` / `<col>_b` (see [[concept-counterbalancing]] for the pair manifest), or (2) an external unit-metadata parquet joined on `unit_uid` |
| Head-to-head test | Direct A-vs-B pairs only; `relative_score` oriented so + = group A preferred; repeats collapsed by `canonical_pair_id`; one-sample t-test vs 0, Wilcoxon, binomial sign test, win rate + Wilson CI |
| Rating-level test | Per-unit TrueSkill μ over the **full** run ([[concept-trueskill]]); Welch's t + Mann-Whitney U between the groups' μ distributions; Cohen's d, Cliff's δ |
| Matrix mode | `--all-pairs`: every group pair, Benjamini-Hochberg corrected, Kruskal-Wallis omnibus, annotated heatmaps |
| Multi-model mode | `--aggregation-dir <dir>`: per-model runs discovered via the aggregation-report layouts; tests run per model (BH within model), cross-model replication summary, forest plots (pair mode) / k-of-N significance heatmaps (matrix mode) |
| Outputs | Markdown (+ optional PDF via pandoc), tidy `*.tests.parquet` of all test rows |
| Registry | Append-only JSONL at `machine-beholder/difference_tests/registry.jsonl`; deterministic `experiment_id`; reruns skipped unless `--force`; `--list` to browse |
| W&B mirror | Separate analysis project **`URBANPAIRVQA-ANALYSIS`** (entity `urbanekg`), `job_type=difference_test`; summary metrics, results table, report artifacts. Non-fatal on failure; `--no-wandb` to skip |

## Usage

```bash
# Single comparison — cuisine is NOT on the restaurants pairs.parquet, so it
# is joined from the DOHMH aggregation (camis == unit_uid); see
# [[dohmh-restaurants-curation]].
python scripts/pairwise_vqa_difference_report.py \
    <run>/outputs/pairwise/restaurants_mvp_*.parquet \
    --group-column cuisine_description \
    --unit-metadata-parquet curation/dohmh_restaurants_inspected_all/restaurants_aggregated.parquet \
    --unit-metadata-id-column camis \
    --group-a Chinese --group-b Italian \
    --attribute "rather eat at" --unit-label restaurant --pdf

# All-pairs matrix over the 12 biggest cuisines
python scripts/pairwise_vqa_difference_report.py <output.parquet> \
    --group-column cuisine_description \
    --unit-metadata-parquet .../restaurants_aggregated.parquet \
    --unit-metadata-id-column camis \
    --all-pairs --top-k-groups 12 --pdf

# Multi-model: same comparison across every model in a sweep aggregation dir
# (schools example: public vs private from FacDB facsubgrp; see [[facdb-curation]])
python scripts/pairwise_vqa_difference_report.py \
    --aggregation-dir machine-beholder/aggregations/schools_preferences/may1_sweep \
    --group-column facsubgrp \
    --unit-metadata-parquet curation/facdb_schools_k_12/facilities.parquet \
    --unit-metadata-id-column uid \
    --group-a "PUBLIC K-12 SCHOOLS" --group-b "NON-PUBLIC K-12 SCHOOLS" \
    --attribute "rather send your child to" --unit-label school --pdf

# Registry
python scripts/pairwise_vqa_difference_report.py --list
```

Group names match case-insensitively against observed values; a miss prints the top-20 observed values.

## Registry record

One JSON line per completed experiment: `experiment_id` (sha1 of source parquet + group column + groups + join source + stats knobs, 12 hex chars), timestamps, group/source provenance, `model_label` (from the run's resolved Hydra config), a results summary (p-values, effect sizes, Ns), report/results paths, and `wandb_url`. Latest record wins on id collision (`--force` reruns append). Override the path with `--registry` or `$MLLMSCI_DIFFTEST_REGISTRY`.

## Statistical caveats (also printed in every report)

- Head-to-head observations share units when `allow_replacement=true` — not fully independent; repeats *are* collapsed by canonical pair.
- TrueSkill μ values are coupled through the comparison graph → rating-level p-values are approximate; prefer the effect sizes.
- When the tests disagree: trust head-to-head direction if direct pairs are plentiful; the rating-level test borrows strength from comparisons against all other groups.

## Multi-model caveat

Models in a sweep judge the **same pair set**, so per-model results are correlated through the shared images; replication counts (k/N models significant) overstate independence. BH correction is applied within each model, never across the pooled table.

## Tests

`tests/test_pairwise_difference.py` — orientation sign, repeat collapse, planted effect / null, both group-resolution paths, registry determinism/dedupe/corrupt-line tolerance, and three end-to-end CLI runs on synthetic data (single-pair, matrix, and multi-model with planted opposite preferences).
