# F9: review of the H9 analysis

An independent methods review of the test of H9 (2026-10-30). The claims
under review are in `f9-subjective-prompt.md`, sections 1 and 4. The plan
is in `facct-runbook.md`, section F9, "The tests, set before the run". The
code is `research/scripts/f9/analyze_subjective.py` (1.1.0) with
`research/scripts/f8/method_b.py`.

Every number in this review comes from a check that ran on
`research/figures/f9/pairs.parquet` and the CSV outputs, unless the row says
"reasoning only". The checks did not change any file and did not submit a
job.

## 1. Summary

| # | Finding | Severity | Verified by |
|---|---------|----------|-------------|
| 1 | The code implements the pre-registered test. No bug, no wrong join, no silent drop that changes a result | - | code |
| 2 | Every number in sections 4.1 to 4.7 of the record matches the output files | - | code |
| 3 | The extractor reads the F9 text under a judgment framing. The valence precision of the `visual_evidence` spans is not labelled yet (record, section 5). A precision difference between the 2 prompts biases `d` directly | High | reasoning only |
| 4 | The F4 baseline depends on the schema: under the new schema the F4 restaurant gradient is 1.4 to 2.3 times the method B gradient of record on the same images | High | code |
| 5 | The F4 restaurant net valence is exactly 0 for 65% to 86% of the texts that enter. The raw test compares a near-constant outcome with a spread one. The standardized form is sound (bootstrap agrees with CR1), but it is post hoc | High | code |
| 6 | Entry correlates with the features in 7 of 60 cells. The result is robust to it (missing to 0: restaurant 24 to 24, street 10 to 13) | Medium | code |
| 7 | % white, % bachelor's and log income correlate 0.75 to 0.86 over the images. "4 of 5 features" is 1 axis and its complement, not 4 findings | Medium | code |
| 8 | 4 of the 10 street H9 rows rest on the sign of an F4 slope with p of 0.10 or more. The label is correct but fragile | Low | code |
| 9 | The stratification weights of the restaurant pool change nothing (H9 24 to 24; median slope ratio 1.03) | Low | code |
| 10 | Section 4.2 reports the pairs that enter (9,479), but the fit uses 9,470 (9 images in NTAs with no ACS value). `share_kept` is dropped from the CSV and the record computes it by hand | Low | code |
| 11 | "The street result does not depend on the scale" overstates: the count is 10 against 11, but 5 rows change | Low | code |

## 2. Correctness (question 1)

| Item of the plan | Code | Status |
|------------------|------|--------|
| Outcome: (good - bad) / `visual_evidence` spans with a valence | `text_valence`, lines 207 to 215: `valence` in (good, bad, neutral); NaN with 0 such spans | correct. 102 of 614,000 F9 restaurant VE spans have no valence and leave the count |
| Pairing on the same image | `paired`, lines 232 to 240: inner join on (model, case, field, image_idx), `validate="one_to_one"` | correct. Pairs per cell 9,920 to 10,000; the losses are copy suspects. `image_idx` 0 to 9,999 (restaurant) and 0 to 10,014 (street), as the pool |
| Images that enter: both texts have 1 span or more | `d` is NaN when 1 outcome is NaN; `h9_table` drops NaN `d` | correct. The record says "with a valence"; the plan says "1 `visual_evidence` span"; the 2 rules differ by the 102 spans above |
| OLS, CR1 by NTA, t with G - 1 df | `method_b.ols_cluster`, lines 97 to 139; `c = G/(G-1) * (N-1)/(N-K)` | correct. G = 197 to 199 |
| Bonferroni over 60 | `fit_all` multiplies by the rows of the call; `h9_table(pair, ["response"])` has 60 rows; `tests` = 60 in the CSV | correct |
| H9 sign rule | `h9 = sig_d & (sign(beta_d) == sign(beta_f4))`, lines 288 to 290 | correct. Algebra: sign(d) = sign(f4) is the same as "same sign and abs(f9) > abs(f4)" |
| Copy suspects | `load` drops every row of a copy-suspect text, line 192 | correct; 0.17% or fewer |
| Secondary rows | the `reasoning` field gets Bonferroni over 40 (the plan says no correction); the record gives both counts (31 at p < 0.05, 21 after Bonferroni) | acceptable |
| Standardized form (D143) | `standardize`, lines 253 to 267: z-score within (model, case, field) over the images that enter, ddof 1 | correct; post hoc |

Small points:

- `build_table` joins the NTA with `how="inner"` and the ACS features with
  `how="left"`; `fit_all` then drops rows with no feature value. Thus the
  fit `n` is 9,470 for gemma-4-12b restaurant and the record's section
  4.2 says 9,479. The log income fit has 1 row fewer again (1 NTA with no
  income). The street case loses 0.
- `h9_table` drops `share_kept` (line 297). The record's "median share
  kept 0.95 / 0.77" is `beta1_d / beta_d` by hand; the values are 0.946
  and 0.765.
- The vocabulary is clean: 0 `verdict` spans with a label outside the 5
  labels; VE spans not quotable 0.4%, duplicate 0.1%.

## 3. Validity (question 2)

### 3.1 The spread of the 2 outcomes

| Case | Share of F4 net valence exactly 0 | Share of F9 net valence exactly 0 | SD F4 | SD F9 |
|------|-----------------------------------|-----------------------------------|-------|-------|
| restaurant | 0.65 to 0.86 | 0.19 to 0.58 | 0.09 to 0.19 | 0.37 to 0.45 |
| street | 0.34 to 0.79 | 0.15 to 0.30 | 0.26 to 0.58 | 0.35 to 0.63 |

The F4 restaurant outcome sits on a floor: most texts carry no valence at
all. A raw slope on such an outcome is small for that reason alone. Thus
the raw test answers "does the net valence move more, in net-valence
units"; it does not answer "is the gradient steeper relative to the
spread". Both readings are legitimate forms of H9. The pre-registered
form is the raw one.

Checks of the standardized form:

| Check | Result |
|-------|--------|
| Does z-scoring punish F9 for its own gradient (the between-NTA part of the SD)? | No. The largest R-squared of 1 outcome on 1 feature is 0.09 (restaurant F9), 0.06 (restaurant F4), 0.02 (street) |
| Does the fixed SD understate the SE? | No. A cluster bootstrap by NTA (200 draws, z-scores computed inside each draw) gives SE ratios bootstrap / CR1 of 0.76 to 1.13, median 0.98. The same 23 rows are significant |

Is there a better scale-free test? The standardized slope of `d` is a
paired comparison of 2 standardized coefficients on the same images with
NTA clusters. It is the textbook form. A rank form (the NTA Spearman of
section 4.5) agrees in direction. One more description helps the reader:
the good share and the bad share as 2 separate outcomes, because the F9
gradient can come from more good spans in rich NTAs, from more bad spans
in poor NTAs, or both.

### 3.2 Selection by the entry rule

Entry needs a valenced `visual_evidence` span in both texts. The share
that enters is 0.89 to 0.99 (restaurant) and 0.55 to 0.94 (street). The
missing side is the F4 text in the street case (0.06 to 0.44 missing) and
in the 2 Llama restaurant cells (0.06).

| Check | Result |
|-------|--------|
| Entry rate on each feature, CR1 by NTA, 60 cells | 7 of 60 significant after Bonferroni |
| Largest: llama-3.2-90b restaurant, the F4 text has no valenced span | +4.0 points per SD of % bachelor's, p 2e-19 (richer NTAs lose more images) |
| Largest street: gemma-4-12b, the F4 text has no valenced span | -2.5 points per SD of % bachelor's, p 2e-5 |
| Sensitivity: a text with no valenced span gets net valence 0, thus every pair enters | raw H9: restaurant 24 to 24, street 10 to 13; standardized: restaurant 12 to 12, street 11 to 13; 0 flatter rows. Median slope ratio (all enter / entered) 1.00 (restaurant), 1.03 (street) |
| Rows that change (raw) | gemma-4-12b street gains % white, % Black, % bachelor's; gemma-4-31b street gains % bachelor's; llama-3.2-11b street loses % Black |

The conclusion holds under both rules. The street count is sensitive to
the rule (10 against 13).

### 3.3 The stratification weights

The restaurant pool is stratified by NTA (`weight` = capacity / n, 1.0 to
5.07, median 1.0). A weighted fit (WLS with the same CR1 by NTA) gives
H9 in 24 of 30 restaurant rows, as the unweighted fit, with a median slope
ratio of 1.03. The street weights are 1.

### 3.4 Correlated features

| Pair, image level | Restaurant | Street |
|-------------------|------------|--------|
| % white, % bachelor's | 0.81 | 0.72 |
| % white, log income | 0.75 | 0.65 |
| % bachelor's, log income | 0.86 | 0.80 |
| % Black, % white | -0.56 | -0.61 |
| % Asian, any | -0.44 to 0.16 | -0.47 to 0.16 |

Bonferroni stays valid under this dependence (it is conservative). The
claim "4 of the 5 features" is 1 socioeconomic axis plus its complement.
Also, the 6 models answer the same images, thus the 6 rows of 1 feature
are not 6 independent replications.

### 3.5 The F4 side under the new schema (D142)

The method B record (`research/figures/f8/method_b/valence_fits.csv`) fits the
same F4 outcome under the x2 schema of record. On the same images:

| Model, restaurant, % white | F4 slope, new schema | F4 slope, x2 schema | Ratio |
|----------------------------|----------------------|---------------------|-------|
| gemma-4-12b | 0.0375 | 0.0215 | 1.74 |
| gemma-4-31b | 0.0200 | 0.0123 | 1.62 |
| llama-3.2-11b | 0.0231 | 0.0110 | 2.09 |
| llama-3.2-90b | 0.0220 | 0.0113 | 1.95 |
| qwen3.5-9b | 0.0172 | 0.0091 | 1.89 |
| qwen3.8-27b | 0.0223 | 0.0134 | 1.66 |

Over the 4 socioeconomic features the ratio is 1.4 to 2.3. The new schema
assigns more valence to descriptive text (its examples keep the F9
valence mix, D142 item 2). Thus the F4 baseline moves by about 2 times
with the schema, and the F9 side has no second schema. The ratio "F9
gradient about 4 times the F4 gradient" is a property of this schema,
not of the texts alone. The shared schema (option C) is the correct
choice; the caveat is about the magnitude, not the direction. In the
street case the new schema gives the F4 texts half the VE spans, and the
F4 street slope is near 0 under both schemas.

### 3.6 The extractor under a judgment framing (reasoning only)

The F9 text argues a verdict. Words such as "inviting" or "run-down" are
visual evidence with a valence in both prompts, but the extractor may
read a valence into a neutral cue more often when the text around it
argues. This bias goes into `d` directly, and it can follow the NTA if
the models argue more in some neighbourhoods. The record (section 5)
draws a precision sheet by (case, prompt, class, valence) with 30 spans
per stratum; it has no label yet. This is the main open check. The
schema comparison of 3.5 shows that the extractor's valence rate is
schema-sensitive, which makes the check more necessary, not less.

### 3.7 Other points

| Point | Result |
|-------|--------|
| The sign rule in the street case | 4 of the 10 street H9 rows have an F4 slope with p of 0.10 or more (llama-3.2-11b % white p 0.25 and % Black p 0.90; llama-3.2-90b income p 0.10; qwen3.8-27b % white p 0.54). The label rests on the sign of a noise-level slope. A flip would make the row "flatter", which would be wrong (F9 has a gradient where F4 has none). All 24 restaurant H9 rows have a significant F4 slope |
| Clustering | NTA is the level of the predictor, thus it is the correct minimum cluster. G = 197 to 199. The bootstrap agrees with CR1 |
| Copy suspects | 0.17% or fewer of the texts; no check by NTA (reasoning only: too few to move a slope) |
| "The prompt is the only change" | the seed is the same, but the 2 prompts give different token sequences, thus the sampling noise is not paired. The configuration is paired; the draw is not (reasoning only) |
| The 6 rows where the F4 and F9 slopes have different signs | all have a non-significant `d` (5 are % Asian). No row is mislabelled |

## 4. Do the conclusions follow (question 3)

| Claim (section 4.8 and 4.4) | Status |
|-----------------------------|--------|
| 1. Steeper in all 6 models (restaurant) and 4 models (street) | holds for the raw test only. Standardized: gemma-4-12b has 0 restaurant rows, qwen3.8-27b has 1. Say "in net-valence units" |
| 1. % Asian gives no effect anywhere | holds (0 rows in either form) |
| 2. Most of the raw restaurant effect is the wider spread; 12 of 30 rows stay significant, about 1.4 times | holds (median standardized ratio 1.41; raw 4.10). Add the floor: 65% to 86% of F4 texts have net valence 0 |
| 3. The controls do not remove the effect | holds: 34 of 34 H9 rows keep p < 0.05 with the same sign under model 1 |
| 4. The verdict follows the features; a description | holds; the signs and the 12 significant cells match `verdict_fits.csv` |
| 4.4 "The street result does not depend on the scale" | overstated. The count is 10 against 11, but 5 rows change: gemma-4-12b gains 3 rows, llama-3.2-90b drops from 3 to 1 |
| 4.2 "images that enter" | the fit uses 9 fewer restaurant images (NTAs with no ACS value) |
| Runbook status: "H9 holds in 34 of the 60 primary tests ... 23 of 60 standardized; 0 flatter" | matches `meta.json` |

No number in the record disagrees with the output files.

## 5. What must change before the paper (question 4)

| Rank | Change | Why |
|------|--------|-----|
| 1 | Label the precision sheet for the `visual_evidence` valence strata of both prompts, at least the restaurant case (section 5 of the record; 30 spans per stratum, 8 strata). Report the F4 and F9 floors side by side. If the F9 precision of `good` or `bad` is below the F4 precision by more than the gradient difference, the claim does not stand | the main open threat (3.6) |
| 2 | State the schema dependence of the F4 baseline: under the new schema the F4 restaurant slope is 1.4 to 2.3 times the slope of record. Do not quote "4 times" without it | 3.5 |
| 3 | Report the raw test as the pre-registered test and the standardized test as post hoc (D143). The headline "all 6 models" is raw only. Give the floor share (65% to 86% zeros) next to the SDs | 3.1 |
| 4 | Present the 4 features as 1 socioeconomic axis. Count the rows, but do not count them as independent findings. Say that the 6 models share the images | 3.4 |
| 5 | Add the 2 sensitivity rows to the record: missing to 0 (restaurant 24 to 24, street 10 to 13) and the weighted fit (24 to 24). Name the 4 street rows whose F4 slope is not significant, and read the street case as "F9 has a gradient where F4 has none" | 3.2, 3.3, 3.7 |
| 6 | Correct section 4.2 (entered against the fit `n`), keep `share_kept` in `h9_primary.csv`, and soften "does not depend on the scale" | section 2, section 4 |
| 7 | Add the good share and the bad share as 2 outcomes, as a description | 3.1 |

Ranks 1 to 3 are necessary before the claim goes into the paper. Ranks 4
to 7 are edits to the text and the record.

## 6. The checks that ran

All checks used `.venv-mllmsci-vllm025cu129/bin/python` and read
`research/figures/f9/pairs.parquet`, the CSV outputs, `research/figures/f8/method_b/valence_fits.csv`,
`data/f4_traces/pooled/f4_restaurants_nta.parquet`, and 1 merged span file.
The functions came from `research/scripts/f9/analyze_subjective.py` and
`research/scripts/f8/method_b.py`.

| Check | Method |
|-------|--------|
| Record against outputs | pivot of `h9_primary.csv`, `h9_standardized.csv`, `h9_reasoning.csv`, `nta_spearman.csv`, `verdict_*.csv`, `span_mix.csv`, `coverage.csv`, `entered.csv` |
| Pair counts, index ranges, zero shares | groupby on `pairs.parquet` |
| Entry against the features | OLS of the entry flag (and of each side's missing flag) on each z-scored feature, CR1 by NTA |
| Missing to 0 | `h9_table` and `standardize` on the pairs with NaN outcomes set to 0 |
| Weights | WLS (rows scaled by the square root of `weight`) with `ols_cluster` |
| Feature correlations | Pearson over the entering images and over the NTAs |
| Schema dependence | `beta_f4` of `h9_primary.csv` against `beta0` of the method B record, same (model, case, feature), `response` field |
| R-squared | squared Pearson of each outcome with each feature |
| Bootstrap | 200 draws of NTAs with replacement, z-scores inside the draw, OLS slope of `d`; SE against `se_d` of `h9_standardized.csv` |
| Vocabulary | `valence` and `label` values of `f4_restaurant_subj_x2_extractions.parquet` |
