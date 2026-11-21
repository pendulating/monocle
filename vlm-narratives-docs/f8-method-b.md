# F8.1 method B: the NTA effects with image-content controls

Run on 2026-10-28 (D134). The spec is in `facct-runbook.md`, section F8.1.
The method comes from `caption-method-review-2026-10-28.md`, method B.

## The question

Section 4.12 of the traces notebook finds that the NTA net valence of the
`visual_evidence` spans goes with the ACS features of the NTA. For example,
gemma-4-12b has rho +0.74 with % white (restaurant responses). But the
images also differ by NTA: the cuisine, the storefront, the borough. Thus a
reviewer can say that the image content causes the result. Method B adds
fixed effects for the image content and asks: does the NTA coefficient stay?

## What we ran

| Item | Value |
|------|-------|
| Script | `scripts/f8/method_b.py` 1.0.0 (module and CLI) |
| Test | `tests/test_f8_method_b.py` (7 tests, all pass) |
| Data | the pool (`data/f4_traces/pooled/`), the x2 spans, read with `_f4.load_extractions` and `_f4.spans` |
| Unit | 1 text: 1 (model, case, field, image) |
| Texts | the complete images of section 4.8 (`_f4._complete_texts`) |
| Outcome | the net valence of 1 text: (good - bad) / spans, over its `visual_evidence` spans with a valence. This is the 4.12 rule (`_f4.valence_by_stratum`) at the text level |
| Predictor | 1 ACS feature at a time, z-scored: `pct_white_nh`, `pct_black_nh`, `pct_asian_nh`, `pct_bachelors`, `log_median_household_income` |
| Model 0 | OLS: outcome ~ 1 + z(feature) |
| Model 1, restaurant | model 0 + cuisine FE + DOHMH grade FE + borough FE (36 columns) |
| Model 1, street | model 0 + borough FE (6 columns) |
| Standard errors | clustered by NTA, CR1 |
| Test | the t test of the z(feature) coefficient, with G - 1 degrees of freedom (G = the NTAs) |
| Correction | Bonferroni over all 100 rows (20 (model, case, field) x 5 features), alpha 0.05 |
| Cases, fields | restaurant and street; `response` (6 models) and `reasoning` (4 thinking models; the Llama models write no trace) |
| Run time | about 80 s on 1 CPU node |

The CR1 variance (Cameron and Miller 2015):

    V = c * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1
    c = G / (G - 1) * (N - 1) / (N - K)

N is the count of texts, K the count of columns, and u the OLS residuals.
The code computes it with numpy. A test compares it with statsmodels
0.14.6 (`cov_type="cluster"`, `use_t=True`): the coefficients, the SEs and
the p values agree.

## The choices that Claude made

| Choice | Value | Reason |
|--------|-------|--------|
| A text with no `visual_evidence` span with a valence | no outcome; the text leaves the fit | (good - bad) / 0 has no value. The counts are in `valence_counts.csv` (for example, llama-3.2-90b restaurant responses: 8,492 of 9,771 texts have an outcome) |
| Weights | none: each text has the weight 1 | 4.12 is not weighted. The pool samples by NTA, and the NTA is the regressor level, thus OLS stays fair without weights |
| The z-score | over all 214 NTAs of the ACS table with residents (208 for income), sample SD | 1 unit is then 1 SD over New York City, the same for each case and model |
| Income | the natural log of the ACS median; the value stays top-coded at 200,000 dollars | the spec asks for the log |
| NTAs that enter | all NTAs with residents and with 1 image or more (199 restaurant, 197 street; 198 for income) | the clustered SE uses small NTAs correctly. A check with 20 images or more is below |
| Cuisine FE | `_f4.image_cuisines(100)`: 26 cuisines, plus `other (small)` for cuisines with fewer than 100 images | the 4.12 rule |
| Grade FE | the DOHMH grade with the `grade_date` closest to the capture time of the image, before or after | see the next table |
| Borough FE | the first 2 letters of the NTA code (MN, BX, BK, QN, SI) | the NTA sits inside 1 borough, thus the borough is the same for each image of 1 NTA |
| Output folder | `figures/f8/method_b/` at the repository root | the brief of the run gave this path |

The DOHMH grade. The source is the inspection table
`curation/dohmh_restaurants_inspected_all/restaurants.parquet` (1 row for
each `camis` and inspection date, with `grade` and `grade_date`). The
aggregated table of `_f4.DOHMH_PATH` holds only the last grade, thus it
cannot give the grade at the capture time.

| Step | Rule |
|------|------|
| Capture time | `recordedAt` of the F6 v4 `best.parquet`, matched on (camis, recording_id) from the image file name. 10 images are not in `best`; `candidates.parquet` gives their time. All 10,000 images get a time (2025-03 to 2025-10) |
| Graded rows | `grade` in A, B, C, N, Z, P; the date is `grade_date`, else `inspection_date` |
| The grade of 1 image | the graded row with the smallest gap to the capture time. Median gap: 105 days |
| Levels | A (7,617 images), B (915), C (377), `pending` (N, Z and P together: 1,088), `missing` (no graded row: 3) |

## Sanity checks

| Check | Result |
|-------|--------|
| Row counts | the complete images are 9,972 (restaurant, reasoning), 9,771 (restaurant, response), 9,355 (street, reasoning), 9,890 (street, response). They agree with `f4_model_differences.csv` of section 4.8. No mismatch |
| 4.12 again, with `_f4` | gemma-4-12b, % white: rho +0.744 (159 NTAs); gemma-4-12b, % bachelor's: +0.717; qwen3.8-27b, % white: +0.666. They agree with the runbook (+0.74, +0.72, +0.67) |
| Sign and size at the text level | gemma-4-12b, % white, model 0: beta0 = +0.0215 for each SD (SE 0.0015). The NTA-level OLS slope of the 4.12 net on the same z(feature) is +0.0190. Same sign, about the same size |
| Size against the outcome | the outcome has a mean of 0.032 and an SD of 0.128 (gemma-4-12b restaurant responses). Thus 1 SD of % white moves the net valence by about 0.17 SD of the outcome |
| External outcome | a parquet of random `c_yes` values with `sample_id` gives 0 significant rows of 20 |

## Results

The full table is `figures/f8/method_b/valence_fits.csv`. A summary for
each (model, case, field) is `valence_summary.csv`; the totals for each
case are `valence_cases.csv`. `share_kept` = beta1 / beta0.

Totals for each case:

| Case | Rows | Significant, model 0 | Significant, model 1 | Of these, also in model 0 with the same sign | Median share kept, rows significant in model 0 | Median share kept, all rows |
|------|------|----------------------|----------------------|----------------------------------------------|------------------------------------------------|-----------------------------|
| restaurant | 50 | 40 | 37 | 37 | 0.89 | 0.90 |
| street | 50 | 9 | 4 | 3 | 0.80 | 0.92 |

For each field:

| Case, field | Rows | Significant, model 0 | Significant, model 1 |
|-------------|------|----------------------|----------------------|
| restaurant, response | 30 | 24 | 21 |
| restaurant, reasoning | 20 | 16 | 16 |
| street, response | 30 | 7 | 4 |
| street, reasoning | 20 | 2 | 0 |

Restaurant, response, for each model (model 0 / model 1 significant of 5
features; median share kept of the model-0 rows):

| Model | Model 0 | Model 1 | Median share kept |
|-------|---------|---------|-------------------|
| gemma-4-12b | 4 | 4 | 0.87 |
| gemma-4-31b | 4 | 4 | 0.85 |
| qwen3.5-9b | 4 | 4 | 0.93 |
| qwen3.8-27b | 4 | 4 | 0.90 |
| llama-3.2-11b | 4 | 2 | 0.73 |
| llama-3.2-90b | 4 | 3 | 0.70 |

The selected rows (restaurant, response):

| Model, feature | beta0 (SE) | beta1 (SE) | Share kept | p1, Bonferroni |
|----------------|------------|------------|------------|----------------|
| gemma-4-12b, % white | +0.0215 (0.0015) | +0.0191 (0.0018) | 0.89 | < 1e-10 |
| gemma-4-12b, % bachelor's | +0.0202 (0.0015) | +0.0200 (0.0018) | 0.99 | < 1e-10 |
| gemma-4-12b, % Black | -0.0125 (0.0020) | -0.0092 (0.0021) | 0.74 | 0.001 |
| qwen3.8-27b, % white | +0.0134 (0.0010) | +0.0121 (0.0011) | 0.90 | < 1e-10 |
| qwen3.8-27b, log income | +0.0119 (0.0010) | +0.0107 (0.0011) | 0.90 | < 1e-10 |

What the results show:

1. In the restaurant case, the image-content controls remove about 10% of
   the NTA coefficient (median share kept 0.89). 37 of the 40 model-0
   effects stay significant, all with the same sign. % Asian is not
   significant for any model, in either model.
2. The 3 restaurant effects that do not stay are all Llama responses:
   llama-3.2-11b % Black and log income, llama-3.2-90b % Black. For the
   Llama models, the controls remove 27% to 30% of the effect.
3. In 7 restaurant rows of the reasoning field (gemma-4-31b and qwen3.5-9b),
   the controls make the effect larger (share kept 1.03 to 1.27).
4. In the street case, the NTA effects are weak before the controls (9 of
   50). 4 are significant in model 1. 3 of these are also in model 0:
   qwen3.5-9b % white and % Black, qwen3.8-27b % Black. 1 is new:
   gemma-4-31b % Black (beta1 -0.0107, Bonferroni p 0.002; raw p 1.7e-5).
5. In the street case, the borough FE takes some effects away. qwen3.8-27b
   % bachelor's falls from +0.0098 to +0.0019 (share 0.19). For others, the
   coefficient stays but the SE grows. For example, gemma-4-12b % bachelor's
   has a share of 0.98, but its SE goes from 0.0025 to 0.0039.

Figure: `figures/f8/method_b/valence_betas.png`. 1 panel for each (case,
field); beta1 against beta0, with the identity line. A hollow marker is a
row that is not significant in model 1.

### Check: only the NTAs with 20 images or more

`--min-nta-images 20` gives the 4.12 NTA set (files `valence_nta20_*`).

| Case | Significant, model 0 | Significant, model 1 | Median share kept, model-0 rows |
|------|----------------------|----------------------|---------------------------------|
| restaurant | 39 | 38 | 0.91 |
| street | 9 | 4 | 0.80 |

The result does not change.

## Limits

1. The NTA is an ecological unit (Robinson 1950). Write "the valence shifts
   with the resident mix of the area". Do not write "the model treats
   people differently".
2. Spatial correlation. Neighbour NTAs are alike, and the clustered SE
   treats 2 NTAs as independent. Thus the p values can be too small. A
   cluster at a larger unit (the CDTA, or the borough) gives fewer clusters
   and wider SEs; this run does not do it.
3. Cuisine and NTA go together (section 4.12: 64 of 135 cuisine shares
   correlate with an ACS feature). With the cuisine FE, the coefficient of
   the NTA feature uses only the variation inside a cuisine. When a cuisine
   sits mostly in 1 type of NTA, that variation is small, and the SE grows.
4. The controls are coarse. Cuisine, grade and borough do not hold the
   pixels fixed: the storefront, the light, the season and the street stay
   free. Thus a share kept of 0.89 is an upper bound on what the image
   content cannot explain, for the content that the controls measure. F8.2
   (the counterfactual cue) holds the pixels fixed.
5. The cuisine and the grade can also be mediators of a true NTA effect.
   Then model 1 removes a part of a real effect.
6. The outcome is a ratio of spans of 1 text. A text with few spans gives a
   noisy value, and it has the same weight as a text with many spans. 4.12
   pools the spans in an NTA instead. The 2 rules give about the same slope
   for gemma-4-12b (above).
7. Bonferroni over 100 rows treats the rows as independent. The 5 features
   correlate, and the models read the same images. Thus Bonferroni is
   conservative here.
8. The models read the same images. The rows of 2 models are not
   independent; the method compares them only as a description.

## How to run it again

The default outcome (about 80 s, mostly to read the spans):

    .venv-mllmsci-vllm025cu129/bin/python scripts/f8/method_b.py

With an external per-text outcome, for example the R3 readout `c` of F8.3:

    .venv-mllmsci-vllm025cu129/bin/python scripts/f8/method_b.py \
        --outcome-parquet PATH.parquet --outcome-col c_yes --tag r3_c_yes

The parquet needs the columns `model`, `case`, the outcome column, and
`image_idx` or `sample_id` (the F4 traces map `sample_id` to `image_idx`).
A `field` column is optional; without it, each row gets the field `answer`.
The external path uses every row that the file gives; it does not apply the
complete-image rule of section 4.8. More flags: `--cases`, `--min-cuisine`,
`--min-nta-images`, `--alpha`, `--out-dir`.

The tests:

    .venv-mllmsci-vllm025cu129/bin/python -m pytest tests/test_f8_method_b.py -q

## Files

| File | Content |
|------|---------|
| `valence_fits.csv` | 1 row for each (model, case, field, feature): n, n_nta, beta0, se0, p0, beta1, se1, p1, share_kept, the Bonferroni p values and `sig0`, `sig1` |
| `valence_summary.csv` | 1 row for each (model, case, field): the significant counts and the median share kept |
| `valence_cases.csv` | the totals for each case |
| `valence_counts.csv` | the texts, the images, the texts with an outcome and the NTAs, for each (case, field, model) |
| `valence_meta.json` | the run settings, the grade levels and the 4.12 checks |
| `valence_betas.png` | the figure |
| `valence_nta20_*` | the same, NTAs with 20 images or more |
