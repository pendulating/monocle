# F9: the subjective prompt (H9)

F9 asks the 6 F4 models a subjective, yes-or-no question about the same
10,000 images of each F4 case. This record gives the extraction run, the
test of H9, and the results. The plan is in `facct-runbook.md` (F9), and
the extraction design is in `f9-extraction-design.md`.

| Item | Value |
|------|-------|
| Decisions | D139 (plan), D140 (generation), D141 (not F9), D142 (extraction) |
| Prompts | restaurant: `Does this look like a nice restaurant to eat at?`; street: `Would this street make a good spot for taking a photo?` |
| F4 pair | restaurant: `Look and tell me more about this restaurant.`; street: `Look and tell me a good angle to take a picture from.` |
| Answers | 6 models x 2 cases x 10,000 images = 120,000 (`data/f4_traces/subj/f9_<case>_traces.parquet`) |
| Seeds | the F4 seeds; thus the prompt is the only change |

## 1. The hypothesis and the test

| # | Hypothesis |
|---|------------|
| H9 | With a subjective prompt, the net valence of the answer changes more along the NTA socioeconomic features than with the F4 prompt, on the same images |

| Item | Rule |
|------|------|
| Text outcome | net valence = (good - bad) / `visual_evidence` spans with a valence (method B, F8.1) |
| Unit | 1 image, for 1 (model, case, field) |
| `d` | net valence under F9 minus net valence under F4, on the same image |
| Images that enter | both texts have 1 `visual_evidence` span or more with a valence |
| Test | OLS of `d` on 1 z-scored ACS feature; standard errors clustered by NTA (CR1); t test with G - 1 df |
| Features | `pct_white_nh`, `pct_black_nh`, `pct_asian_nh`, `pct_bachelors`, log `median_household_income` |
| Primary | the `response` field: 6 models x 2 cases x 5 features = 60 tests, Bonferroni |
| H9 holds for 1 row | the slope of `d` is significant and has the sign of the F4 gradient (model 0 on the F4 outcome, same images) |
| Flatter | the slope of `d` is significant and has the other sign |
| Secondary (no correction) | the `reasoning` field (4 models, 40 rows); model 1 of method B on `d` (restaurant: cuisine, DOHMH grade and borough fixed effects; street: borough) |

The slope of `d` equals the F9 gradient minus the F4 gradient, because
both outcomes are on the same images.

## 2. The extraction

Both sides come from 1 schema (D142, Q4 option C). The F9 answers and the
F4 answers of the same images go through `urban_cues_single_subj_v1`
(restaurant) and `urban_cues_street_subj_v1` (street). These schemas are
the x2 schemas (`single-v2`, `street-v3`) plus 1 class, `verdict`. The F4
x2 spans of record (`data/f4_traces/x2/`) do not enter the F9 test.

| Item | Value |
|------|-------|
| Extractor | DeepSeek-V4-Flash with langextract, on 1 B300 for each job (`research/scripts/f4/x2_queue.sbatch`) |
| Arms | `x2_cues_subj`, `x2_cues_street_subj`; retry arms `x2r_cues_subj`, `x2r_cues_street_subj` |
| W&B | `F9X2_CUES_SUBJ`, `F9X2_CUES_STREET_SUBJ`, and the `_RETRY` experiments |
| Inputs | F9: `extract_input/f4_{restaurant,street}_subj_x2.parquet` (100,000 texts each). F4: `f4_{case}_x2.parquet` and `f4_{case}_b2x2.parquet` |
| Smoke | job 105550: PASS for both arms (`f9-extraction-design.md`, section 6.4) |
| Pass 1 | 80 steps in 12 parts, jobs 105565 to 105576 (`logs/f4/f9x2_pass1_jobs.txt`) |
| Pass 2 | `research/scripts/f4/plans/subj_pass2.txt`: flag the copy suspects, retry them, merge to `data/f4_traces/subj/x2/` |
| Empty response | qwen3.8-27b restaurant 8826: the response text is the draft from its trace (`response_from_trace`, D142 Q5) |

### 2.1 The verdict of 1 text

The analysis gives 1 label to each text from its quotable `verdict` spans
in text order (`text_verdict` in `research/scripts/f9/analyze_subjective.py`):

| Spans | Label |
|-------|-------|
| none with a declared label | `none` |
| only `not_applicable` | `not_applicable` |
| the last other span is `mixed` or `graded` | that label |
| the last other span is `yes` or `no`, and the text also holds the other one | `mixed` |
| the last other span is `yes` or `no`, and the text holds no span of the other one | that label |

**Caution:** on the smoke run, a reader agreed with the label of the rule
for 19 of 24 restaurant responses and 23 of 24 street responses. The design
rule asks for 90%. The errors are at the yes/graded/mixed borders. The
verdict is a description only; the test does not use it.

## 3. Scripts and outputs

| Script | Output |
|--------|--------|
| `research/scripts/f9/analyze_subjective.py` | `research/figures/f9/`: `h9_primary.csv`, `h9_reasoning.csv`, `verdict_fits.csv`, `span_mix.csv`, `verdict_shares.csv`, `verdict_agreement.csv`, `nta_spearman.csv`, `verdict_by_nta.csv`, `verdict_nta_spearman.csv`, `coverage.csv`, `entered.csv`, `meta.json`, `pairs.parquet`, `text_verdicts.parquet`; figures `f9_gradients`, `f9_verdicts`, `f9_span_mix` (PDF, PNG, caption JSON) |
| `research/scripts/f9/compare_f4_schema.py` | `research/figures/f9/f4_schema_compare.json` (section 6.2 of the design) |

## 4. Results (2026-10-30)

### 4.1 The extraction (B10 rules)

| Case | Prompt | Texts | Missing | Quotable | Copy suspects after the retry |
|------|--------|-------|---------|----------|-------------------------------|
| restaurant | F9 | 100,000 | 0 | 0.996 | 4 |
| restaurant | F4 (canonical + batch 2) | 9,956 + 89,811 | 0 | 0.993 | 10 + 123 |
| street | F9 | 100,000 | 0 | 0.996 | 168 |
| street | F4 (canonical + batch 2) | 9,836 + 90,054 | 0 | 0.997 | 0 + 4 |

All 6 cases pass the B10 rules (0 missing; quotable 99% or more; copy
suspects 0.17% or fewer). The analysis drops the copy suspects. The 3
design checks of the verdict rule pass (gemma-4-31b 7880 `mixed`,
qwen3.8-27b 8244 `graded`, llama-3.2-11b 7530 `no`). The copied response
(qwen3.8-27b restaurant 8826) gets `mixed`, F9 net valence -0.4, F4 net
valence 0.0.

### 4.2 The images that enter (response field)

| Model | Restaurant | Street |
|-------|------------|--------|
| gemma-4-12b | 9,479 | 7,174 |
| gemma-4-31b | 9,418 | 6,580 |
| qwen3.5-9b | 9,938 | 8,949 |
| qwen3.8-27b | 9,924 | 9,409 |
| llama-3.2-11b | 8,948 | 5,522 |
| llama-3.2-90b | 8,922 | 7,462 |

Fewer street images enter because many F4 street answers have no
`visual_evidence` span under the new schema (section 6.4 of the design).

### 4.3 The primary test (H9)

The slope of `d` on each z-scored feature (net valence units per 1 SD of
the feature). `*`: significant after Bonferroni over 60 tests, with the
sign of the F4 gradient (H9 holds). No row has a significant slope with
the other sign.

| Case | Model | % white | % Black | % Asian | % bachelor's | log income |
|------|-------|---------|---------|---------|--------------|------------|
| restaurant | gemma-4-12b | +0.080 * | -0.052 * | +0.006 | +0.073 * | +0.069 * |
| restaurant | gemma-4-31b | +0.078 * | -0.050 * | +0.008 | +0.069 * | +0.074 * |
| restaurant | llama-3.2-11b | +0.046 * | -0.032 * | +0.005 | +0.037 * | +0.030 * |
| restaurant | llama-3.2-90b | +0.076 * | -0.061 * | +0.006 | +0.066 * | +0.060 * |
| restaurant | qwen3.5-9b | +0.120 * | -0.076 * | -0.004 | +0.111 * | +0.096 * |
| restaurant | qwen3.8-27b | +0.077 * | -0.057 * | +0.008 | +0.074 * | +0.063 * |
| street | gemma-4-12b | +0.027 | -0.031 | +0.017 | +0.027 | +0.020 |
| street | gemma-4-31b | +0.002 | +0.008 | -0.005 | +0.016 | +0.021 |
| street | llama-3.2-11b | +0.066 * | -0.042 * | +0.019 | +0.069 * | +0.058 * |
| street | llama-3.2-90b | +0.041 * | -0.013 | -0.003 | +0.072 * | +0.045 * |
| street | qwen3.5-9b | +0.022 | -0.009 | -0.001 | +0.036 * | +0.016 |
| street | qwen3.8-27b | +0.026 * | -0.015 | -0.003 | +0.036 * | +0.016 |

**H9 holds in 34 of 60 rows** (restaurant 24 of 30, street 10 of 30).
In the restaurant case the F4 gradients are 0.01 to 0.04 per SD and the
F9 gradients are 0.05 to 0.14 per SD (`f9_gradients`).

### 4.4 The scale of the 2 outcomes (D143)

**Caution:** the F9 net valence spreads wider than the F4 net valence.
Restaurant SD: F4 0.09 to 0.19, F9 0.37 to 0.45. Street SD: F4 0.26 to
0.58, F9 0.35 to 0.63. The reason is the span mix (section 4.6): an F9
restaurant response has fewer `visual_evidence` spans, and more of them
are good or bad. A raw slope grows with the spread of its outcome, even
when the correlation does not change.

Thus the scale-free form of the test z-scores each outcome within
(model, case) over the images that enter, then tests `d` = z(F9) - z(F4)
in the same way. This form was not set before the run (D143).

| Case | Model | % white | % Black | % Asian | % bachelor's | log income |
|------|-------|---------|---------|---------|--------------|------------|
| restaurant | gemma-4-12b | +0.015 | -0.022 | +0.022 | +0.001 | +0.011 |
| restaurant | gemma-4-31b | +0.070 * | -0.050 | +0.018 | +0.044 | +0.068 * |
| restaurant | llama-3.2-11b | +0.062 * | -0.048 | +0.009 | +0.032 | +0.023 |
| restaurant | llama-3.2-90b | +0.105 * | -0.100 * | +0.022 | +0.072 * | +0.071 * |
| restaurant | qwen3.5-9b | +0.135 * | -0.080 * | -0.006 | +0.116 * | +0.095 * |
| restaurant | qwen3.8-27b | +0.038 | -0.050 * | +0.027 | +0.033 | +0.018 |
| street | gemma-4-12b | +0.064 * | -0.073 * | +0.032 | +0.072 * | +0.049 |
| street | gemma-4-31b | +0.016 | +0.002 | -0.019 | +0.060 | +0.044 |
| street | llama-3.2-11b | +0.103 * | -0.069 | +0.036 | +0.095 * | +0.082 * |
| street | llama-3.2-90b | +0.048 | -0.007 | +0.001 | +0.095 * | +0.058 |
| street | qwen3.5-9b | +0.065 * | -0.029 | -0.003 | +0.105 * | +0.047 |
| street | qwen3.8-27b | +0.063 * | -0.038 | -0.007 | +0.090 * | +0.040 |

With standardized outcomes, **H9 holds in 23 of 60 rows** (restaurant 12,
street 11), and 0 rows are flatter. In the restaurant case the median F9
gradient is about 1.4 times the F4 gradient in SD units; gemma-4-12b is
almost equal (0.25 against 0.26 on % white). The raw F9 gradient is
about 4 times the F4 gradient (median), thus most of the raw restaurant
difference is the wider spread of the F9 outcome, and a smaller part is a
steeper gradient. The street result does not depend on the
scale (`f9_gradients_standardized`).

**Paper (2026-11-03):** section 4.3 gives the raw form as the pre-registered test and the standardized form beside it, with the floor (65% to 86% of the descriptive restaurant answers have net valence 0). The appendix paragraph before `tab:h9-summary` gives the schema note of `f9-h9-review.md` section 3.5.

### 4.5 The secondary rows

| Row | Result |
|-----|--------|
| Reasoning field (40 rows, raw `d`) | p < 0.05 in 31 rows; with Bonferroni over 40, H9 in 21 (restaurant 16, street 5), flatter in 2 |
| Model 1 on `d` (fixed effects) | all 34 H9 rows keep p < 0.05 with the same sign; median share kept 0.95 (restaurant), 0.77 (street) |
| NTA Spearman of the mean net valence (NTAs with 20 images or more) | the absolute rho is larger under F9 in 26 of 30 restaurant rows and 24 of 30 street rows. Median rho on % white: restaurant 0.67 (F4) and 0.73 (F9); street 0.08 and 0.28 |

### 4.6 The span mix (responses)

| Case | Model | VE spans per text, F4 / F9 | Inference spans per text, F4 / F9 | Good share of VE, F4 / F9 | Bad share of VE, F4 / F9 |
|------|-------|----------------------------|-----------------------------------|---------------------------|--------------------------|
| restaurant | gemma-4-12b | 6.19 / 2.72 | 3.80 / 4.27 | 0.06 / 0.27 | 0.01 / 0.11 |
| restaurant | gemma-4-31b | 4.23 / 2.54 | 3.21 / 3.48 | 0.03 / 0.21 | 0.01 / 0.15 |
| restaurant | qwen3.5-9b | 8.13 / 4.97 | 5.22 / 6.51 | 0.03 / 0.30 | 0.01 / 0.15 |
| restaurant | qwen3.8-27b | 9.39 / 5.48 | 6.23 / 3.90 | 0.04 / 0.14 | 0.02 / 0.32 |
| restaurant | llama-3.2-11b | 4.57 / 2.57 | 3.54 / 2.56 | 0.09 / 0.16 | 0.01 / 0.11 |
| restaurant | llama-3.2-90b | 2.84 / 2.29 | 3.18 / 2.30 | 0.07 / 0.12 | 0.01 / 0.18 |
| street | gemma-4-12b | 2.28 / 4.31 | 0.12 / 1.69 | 0.24 / 0.53 | 0.09 / 0.28 |
| street | gemma-4-31b | 1.86 / 4.00 | 0.06 / 2.10 | 0.16 / 0.42 | 0.22 / 0.40 |
| street | qwen3.5-9b | 3.67 / 5.18 | 0.18 / 1.56 | 0.17 / 0.54 | 0.11 / 0.21 |
| street | qwen3.8-27b | 3.61 / 4.59 | 0.02 / 0.77 | 0.26 / 0.48 | 0.13 / 0.39 |
| street | llama-3.2-11b | 1.11 / 2.64 | 0.04 / 0.68 | 0.15 / 0.46 | 0.01 / 0.19 |
| street | llama-3.2-90b | 1.65 / 2.32 | 0.03 / 0.74 | 0.12 / 0.34 | 0.00 / 0.30 |

The caution of the runbook (a yes-or-no question moves text from
`visual_evidence` to `inference`) holds for the restaurant case only in
part: the F9 responses have fewer `visual_evidence` spans, but the
`inference` count does not rise in every model. In the street case both
counts rise. In both cases the F9 spans carry a valence far more often
(`f9_span_mix`).

### 4.7 The verdict (a description)

**Caution:** the hand-read agreement of the verdict rule on the smoke is
19 of 24 (restaurant) and 23 of 24 (street).

| Model | Case | yes | no | mixed | graded | not_applicable | none |
|-------|------|-----|----|-------|--------|----------------|------|
| gemma-4-12b | restaurant | 0.110 | 0.068 | 0.649 | 0.079 | 0.048 | 0.046 |
| gemma-4-31b | restaurant | 0.048 | 0.076 | 0.722 | 0.088 | 0.054 | 0.011 |
| qwen3.5-9b | restaurant | 0.243 | 0.086 | 0.526 | 0.133 | 0.011 | 0.001 |
| qwen3.8-27b | restaurant | 0.030 | 0.131 | 0.635 | 0.185 | 0.018 | 0.000 |
| llama-3.2-11b | restaurant | 0.249 | 0.066 | 0.180 | 0.034 | 0.071 | 0.400 |
| llama-3.2-90b | restaurant | 0.123 | 0.163 | 0.150 | 0.054 | 0.220 | 0.290 |
| gemma-4-12b | street | 0.135 | 0.030 | 0.815 | 0.017 | 0.002 | 0.000 |
| gemma-4-31b | street | 0.066 | 0.060 | 0.851 | 0.020 | 0.002 | 0.000 |
| qwen3.5-9b | street | 0.452 | 0.022 | 0.400 | 0.125 | 0.000 | 0.000 |
| qwen3.8-27b | street | 0.106 | 0.102 | 0.591 | 0.200 | 0.000 | 0.000 |
| llama-3.2-11b | street | 0.458 | 0.115 | 0.383 | 0.007 | 0.007 | 0.032 |
| llama-3.2-90b | street | 0.248 | 0.242 | 0.467 | 0.015 | 0.007 | 0.020 |

- The thinking models seldom take a side: `mixed` is 40% to 85% of their
  responses ("it depends on what you are looking for").
- The share of `yes` rises with % white, % bachelor's and log income, and
  falls with % Black, in every (model, case) cell. The 3 positive
  features are significant in all 12 cells (Bonferroni over the 60 rows
  of 1 outcome). Example: qwen3.5-9b restaurant, +12.6 points of `yes`
  per SD of % bachelor's (`verdict_fits.csv`).
- The response label equals the reasoning label in 51% to 71% of the
  answers of the 4 thinking models (`verdict_agreement.csv`).
- The F4 responses also hold verdict spans: 14% (restaurant) and 23%
  (street) of them have 1 or more. The F4 street answers often say "this
  is a great spot".

### 4.8 What the result says

1. A subjective prompt makes the valence of the answer follow the
   socioeconomic features of the neighborhood more steeply, on the same
   images, with the same seeds. The effect is in all 6 models for the
   restaurant case, and in 4 models for the street case. % Asian gives no
   effect anywhere.
2. In the restaurant case, most of the raw effect is that the subjective
   prompt makes the valence wider. A smaller part is a steeper gradient
   in SD units (12 of 30 rows, about 1.4 times the F4 gradient). Report
   both forms.
3. The image-content controls (cuisine, DOHMH grade, borough) do not
   remove the effect.
4. The verdict follows the same features, but its label quality on the
   restaurant case is below the design rule. Use it as a description.

## 5. The accuracy of the extraction

Notebook `research/notebooks/facct/f9_extraction_accuracy.py`, helper
`research/notebooks/facct/_xacc.py` (1.0.0). `python research/notebooks/facct/_xacc.py build`
(about 2.5 minutes, 8 GB) writes the tables to `research/figures/f9/xacc/` and draws
the label samples into `data/f9_xacc/` (seed 20261030; `--redraw` draws
them again).

| Level | Source | Status (2026-10-30) |
|-------|--------|---------------------|
| 1. Grounding | the aligner, every span | done: 89.5% to 99.8% of the spans of each class are verbatim; 98.6% or more are quotable |
| 1b. Verbatim precision and recall | `python research/notebooks/facct/_xacc.py verbatim` writes `verbatim_summary.csv` (1.1.0, 2026-11-02) | done: precision (the share of the spans that are verbatim) 96.3% over 6,637,869 spans, 94.4% to 98.1% by (case, prompt, field); recall at the text level (texts with 1 or more verbatim span) 99.5% over 399,657 texts, lowest cell 96.4% (street, F9, reasoning); the weakest class is `photo_advice` at 93.7%. The paper reads these as `\stat{xacc.*}` through `results/facct/f9_xacc_verbatim.csv` |
| Coverage | the spans against the text lengths | done: the spans cover 29% to 56% of a reasoning trace and 64% to 92% of a response; the span density in the last tenth of a long trace is 47% to 72% of its peak |
| Performance | the logs and `sacct` | done: 428,987 prompts, 31.8 GPU hours in `generate`, 3.7 prompts per second on 1 B300, 19 preemptions |
| 2. Class precision | a person labels 30 random spans of each of 34 strata (case x prompt x class, `visual_evidence` by valence) | the sheet is drawn (`data/f9_xacc/precision_sheet.csv`); no label yet |
| 3. Class and attribute precision | the same labels | no label yet |
| Recall | a person marks every span of 40 texts (`recall_sample.parquet`) | drawn; no label yet |

**The floor.** For `k` correct of `n` labelled spans of a stratum, the
floor is the one-sided 95% Clopper-Pearson lower bound. The floor of a
class pools its strata with the corpus weights, with alpha / (strata) for
each stratum (Bonferroni). The floor times the spans of the class gives
"at least N spans of this class are correct". 30 labels with 0 errors
give a floor of 0.905; a floor of 0.95 needs about 60 labels with 0
errors.

**The H9 check.** The valence strata of `visual_evidence` have their own
floors for F4 and F9. A valence error that differs between the 2 prompts
would bias `d`; section 6.1 of the notebook compares them.
