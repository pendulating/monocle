# F8.2 and F8.3: the counterfactual cue and the R3 readout

Run on 2026-10-29 (D134). This page is the record of F8.2 and F8.3 of the
FAccT runbook (`facct-runbook.md`, F8). The runbook holds the design, which
we set before the run. This page holds the run, the checks and the results.

## The question

The F4 bias results are correlations between the output and the NTA
features. The images differ by NTA, thus the pixels can explain a result.
F8.2 holds the pixels fixed: the same image goes to the model 2 times, and
only a neighbourhood name in the prompt changes. A difference between the 2
answers thus comes from the name.

## What ran

| Item | Value |
|------|-------|
| Images | the canonical batch: 1,000 restaurant images, 985 street images |
| Prompt | `I am in {name}, {borough}. ` + the F4 prompt of the case |
| Arms | `rich` and `poor`: 2 names in the borough of the image (the pair table in F8.2) |
| Models | gemma-4-12b, qwen3.5-9b, gemma-4-31b, qwen3.8-27b |
| Sampling | the F4 model-card values (D35), thinking on (D40), the F4 seed of the image for both arms |
| Token limit | 8,192 (F4: 4,096). The smoke job 103155 cut 3 of 16 cue answers at 4,096 |
| Readout | R3 after the answer, 4 readouts (F8.3), offline vLLM engine |
| Baseline | R3 on the F4 answer (no cue) of the same model and image |

### The scripts

| Script | What it does |
|--------|--------------|
| `scripts/f8/build_cue_sample.py` | the 2 arms of each image: `data/f8/cue_{case}_{arm}.parquet`; the image list of the pool: `data/f8/images_all.parquet` |
| `scripts/whatami/query.py --prompt-column prompt` | the generation; each row has its own prompt |
| `scripts/f8/build_r3_items.py` | the items of the R3 readout (F4 canonical, F4 batch 2, cue); a cut answer is not an item |
| `scripts/f8/r3_readout.py` | the R3 readout with the offline engine |
| `scripts/f8/cue_job.sbatch` | phase A: the server, the 4 arm runs, then R3 on the F4 canonical and the cue answers; phase B: R3 on F4 batch 2 |
| `scripts/f8/submit_all.sh` | phase A for the 4 models, then phase B |
| `scripts/f8/analyze_cue.py` | the tests of the runbook; `figures/f8/cue/` |

### Why the R3 readout does not use the server

The gemma chat template removes the empty thought channel
(`<|channel>thought\n<channel|>`) from a final assistant message. The
generation prompt with thinking off has that channel. Thus a server request
with an assistant prefix cannot give the `c_cool` context. The script renders
the chat with the template of the model, adds the prefix to the text, and
sends the text and the image to the offline engine.

`allowed_token_ids` forces each answer word. The engine returns the raw
logprob of the word (`logprobs_mode` raw_logprobs), before the mask. The smoke
check shows a forced `Yes` at logp -13.8 and rank 47, thus the value is the
raw value and not the masked value (which is 0).

### Why the instructed readouts exist

With the verbatim D18 question, gemma-4-12b starts its answer with
"Whether" (logp -0.02). `Yes` is at rank 12 to 47, and `_lame` at rank
20,000 to 80,000. Thus `c_yes` and `c_cool` compare 2 improbable tokens.
Claude added `c_yes_i` and `c_cool_i` before the first full result: the
question ends with the answer form. With the instructed form, `Yes` is at
rank 3, and `_cool` takes almost all the mass after the prefix. The
Bonferroni correction covers all 4 readouts (32 tests).

## Faults and their fixes

| Job | Fault | Fix |
|-----|-------|-----|
| 103145 (smoke) | the offline engine JIT-builds a flashinfer sampling kernel; the kernel needs `GLIBCXX_3.4.32`, and the process loaded the older anaconda `libstdc++` | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for the R3 step |
| 103145 (smoke) | 3 of 16 restaurant cue answers stopped at 4,096 tokens | the token limit is 8,192 for all models |
| 103179 | qwen3.5-9b: in a large batch, 1 forced request came back with another token (`Yes` for ` cool`); the assertion stopped the job | the script checks that each output prompt is the prompt that it sent, sends a failed request again by itself, and records how it got each value (`how_*`). the Qwen jobs had 63 to 500 failed masks for each items file (about 0.4% of the requests or less); all passed on the second send. The gemma jobs had none. The gemma-4-12b parts of phase A come from the script version before this fix, thus they have no `how_*` columns |

## Jobs

| Job | Model, phase | State |
|-----|--------------|-------|
| 103178 | gemma-4-12b, A | done |
| 103179, 103315 | qwen3.5-9b, A (103315 ran R3 again) | done |
| 103180 | gemma-4-31b, A | done |
| 103181 | qwen3.8-27b, A | done |
| 103182 to 103185 | the 4 models, B | done |

All jobs ran on 1 B300 (g3, QoS `nissenbaum`) between 04:10 and 11:05 UTC
on 2026-10-29. `bash scripts/f8/finalize.sh` runs all the tests again.

## Results of F8.2: the cue

### The checks

| Model | Restaurant cut, rich / poor | Restaurant median tokens, rich / poor | Pairs: restaurant / street |
|-------|------|------|------|
| gemma-4-12b | 43 / 15 | 1,129 / 1,118 | 946 / 985 |
| gemma-4-31b | 23 / 10 | 955 / 959 | 971 / 985 |
| qwen3.5-9b | 208 / 155 | 3,106 / 2,714 | 719 / 977 |
| qwen3.8-27b | 21 / 6 | 2,731 / 2,610 | 975 / 982 |

- Each model stops more rich-arm restaurant answers at 8,192 tokens. A
  paired test drops an image when 1 of its 2 answers stops. Thus the pairs
  are not a random subset. This is strongest for qwen3.5-9b (719 pairs).
- The reasoning names the cue on 99% to 100% of the answers. The street
  responses name it less (28% to 90%).

### The primary tests (32, Bonferroni)

`figures/f8/cue/cue_primary.csv`, `cue_primary.png`. The value is the mean
of rich minus poor over the image pairs. **Bold**: Bonferroni p < 0.05.

| Case, readout | gemma-4-12b | gemma-4-31b | qwen3.5-9b | qwen3.8-27b |
|---------------|-------------|-------------|------------|-------------|
| restaurant, `c_yes` | **+0.87** | +0.30 | -0.08 | **+0.45** |
| restaurant, `c_cool` | -0.17 | +0.24 | -0.01 | **+0.22** |
| restaurant, `c_yes_i` | **+0.72** | +0.43 | -0.02 | +0.04 |
| restaurant, `c_cool_i` | **+0.82** | **-1.39** | -0.03 | **+0.20** |
| street, `c_yes` | -0.14 | **+0.49** | -0.05 | -0.12 |
| street, `c_cool` | **-0.45** | **-0.38** | **-0.14** | -0.03 |
| street, `c_yes_i` | **-1.00** | **-4.38** | **+0.12** | -0.09 |
| street, `c_cool_i` | **-1.25** | **-5.27** | +0.02 | -0.00 |

16 of 32 tests are significant (restaurant 7, street 9).

- **At fixed pixels, the name changes the answer and the readout that
  follows it.** 3 of 4 models change a readout when only the neighbourhood
  name changes. qwen3.5-9b changes almost nothing (2 small street rows, at
  the scale of the per-request noise, see below).
- **The restaurant direction differs by model.** Up for gemma-4-12b (3
  readouts) and qwen3.8-27b (3 readouts; `c_cool_i` only among the
  instructed ones), down for gemma-4-31b (`c_cool_i` -1.39), null for
  qwen3.5-9b.
- **Street photography: a poor name makes the place "cooler" for the 2
  gemma models.** The effect is large for gemma-4-31b: `c_cool_i` -5.27,
  and the rich arm is higher on only 30% of the images. The Qwen models
  show almost no street effect.
- **Any name also moves the readout from the no-cue baseline, and that
  shift can be larger than the rich-poor gap.** gemma-4-31b restaurant
  `c_yes_i`: no cue -0.92, poor +4.18, rich +4.61. Street `c_yes_i`: no
  cue 6.68, rich 6.16, poor 10.54. The column `mean_none` of
  `cue_primary.csv` holds the baseline (the F4 answer, token limit 4,096).
- An example (gemma-4-31b, street image 970, the same image): with
  "Brooklyn Heights" the answer calls the scene "fairly cluttered" and
  `c_cool_i` is -20.5; with "East New York" it calls it a "great contrast"
  and an "Urban Oasis", and `c_cool_i` is +31.3.

### The secondary tests (64, Bonferroni)

`figures/f8/cue/cue_words.csv`, `cue_words.png`. 11 of 64 are significant.

| Model | Case, field | List | Rich arm | Poor arm |
|-------|-------------|------|----------|----------|
| gemma-4-12b | restaurant, response | `upscale` | 16.0% | 10.1% |
| gemma-4-12b | restaurant, reasoning | `upscale` | 20.8% | 13.1% |
| gemma-4-12b | street, reasoning | `upscale` | 10.2% | 2.2% |
| gemma-4-12b | street, reasoning | `danger` | 17.9% | 24.0% |
| gemma-4-31b | restaurant, response | `upscale` | 12.6% | 7.7% |
| gemma-4-31b | restaurant, reasoning | `upscale` | 14.4% | 8.0% |
| gemma-4-31b | street, response | `upscale` | 5.0% | 2.1% |
| gemma-4-31b | street, reasoning | `upscale` | 3.7% | 1.1% |
| qwen3.5-9b | street, response | `upscale` | 3.8% | 1.2% |
| qwen3.5-9b | street, reasoning | `danger` | 33.4% | 43.5% |
| qwen3.8-27b | restaurant, reasoning | `upscale` | 14.2% | 6.9% |

- With a rich name, all 4 models use more `upscale` words on the same
  images. About one third of these hits (21% to 37%) sit within 60
  characters of a neighbourhood, area or borough word. Thus part of the
  effect describes the named area, not the image.
- With a poor name, 2 models write the `danger` list more often in the
  street reasoning. The word "gritty" drives it (qwen3.5-9b: 31.5% against
  20.6%; gemma-4-12b: 20.0% against 15.6%), then "rough". In photography
  advice these are also aesthetic words. Do not call this result "danger".
- No `cheap` or `safe` row is significant.

### What the cue results show, and what they do not show

| Claim | Status |
|-------|--------|
| A neighbourhood name changes the description and the judgment of the same image | shown for 4 of 4 models (words) and 3 of 4 (readouts) |
| A rich name gives more "upscale" words | shown, all 4 models; about one third describe the area |
| A poor name gives more "gritty" and "rough" in the street reasoning | shown, 2 models; these are also photography words |
| The direction of the judgment is "poor is worse" | **not** shown; for street photography the gemma models rate a poor-name place as cooler |

**Caution:** the cue pairs differ in more than income. In 9 of 10 pairs
the rich name is also the whiter NTA. Some names also carry fame or
fashion (Long Island City, Tribeca). The per-pair means
(`cue_by_pair.csv`) differ in sign between pairs: for gemma-4-12b
restaurants, Park Slope against Brownsville gives +2.65 (`c_cool_i`), and
Douglaston against Jamaica gives -1.09. Thus F8.2 shows that the name
matters; it does not isolate class from race or from fame.

A possible reading of the street result: the name sets an expectation,
and the model judges the image against it. The data do not test this.

## Results of F8.3: the R3 readout on the pool

### The NTA test (160 tests, Bonferroni)

`figures/f8/cue/r3_nta_spearman.csv`. NTAs with 20 images or more (159
restaurant NTAs, 158 for qwen3.8-27b; 197 street NTAs). 71 of 160 tests are significant.

| Case | `c_yes` | `c_cool` | `c_yes_i` | `c_cool_i` |
|------|---------|----------|-----------|------------|
| restaurant (of 20) | 14 | 11 | 16 | 15 |
| street (of 20) | 5 | 2 | 4 | 4 |

The NTA mean of `c_cool_i` against the NTA features, restaurants:

| Model | % white | income | % bachelor's |
|-------|---------|--------|--------------|
| gemma-4-12b | +0.64 | +0.48 | +0.72 |
| gemma-4-31b | +0.45 | +0.38 | +0.40 |
| qwen3.5-9b | +0.61 | +0.48 | +0.71 |
| qwen3.8-27b | +0.74 | +0.60 | +0.80 |

### Method B with the R3 readout as the outcome

`figures/f8/method_b_r3/`. The F8.1 controls (restaurants: cuisine, DOHMH
grade, borough; street: borough), errors clustered by NTA, Bonferroni over
the 40 rows of each readout.

| Readout | Significant in model 0 | Significant in model 1 |
|---------|------------------------|------------------------|
| `c_yes` | 24 of 40 | 24 |
| `c_cool` | 15 | 18 |
| `c_yes_i` | 22 | 24 |
| `c_cool_i` | 25 | 24 |

For `c_cool_i` in restaurants, all 16 rows of % white, % Black, % bachelor's
and income stay significant with the controls in all 4 models. The size in
model 1: 0.07 to 0.32 SD of the readout for each SD of the feature, for
example qwen3.8-27b % bachelor's 0.32, gemma-4-12b % white 0.23. These
effects are larger than the valence effects of F8.1 (0.04 to 0.18).

### The models agree

The instructed readouts agree between the models: the Pearson correlation
over the same images is 0.45 to 0.72 (`c_yes_i`) and 0.50 to 0.74
(`c_cool_i`). The verbatim readouts agree less: `c_yes` 0.31 to 0.70,
`c_cool` -0.23 to 0.36 (`r3_model_pearson.csv`).

## Limits

- The readouts `c_yes` and `c_cool` compare improbable tokens (see
  above). The verbatim `c_cool` behaves differently from the other 3
  readouts in some rows. Use the instructed readouts for the main text.
- The R3 readout reads 1 token after a stored answer. The answer itself is
  a sample at temperature 1.0. In F8.2 the 2 arms get new answers, thus the
  difference holds the change of the answer and the change of the judgment
  together.
- F8.3 is still a correlation with the NTA. The controls are coarse.
- The pairs of F8.2 are not a random subset, because a rich-arm answer
  stops more often at the token limit. The advisor check shows that this
  does not explain the results: an unpaired Welch test on all answers that
  did not stop gives the same signs (gemma-4-12b `c_cool_i` +0.74,
  qwen3.8-27b +0.19, gemma-4-31b -1.46). To cancel an effect, the dropped
  pairs would need means of -8 to -18. The dropped images have higher F4
  baselines.
- The same token in 2 requests on the same context differs by an SD of
  0.005 to 0.11 nats (the engine is not fully deterministic). This adds
  noise, not bias, but the smallest significant rows are at this scale.
- The paired t test stays valid with a new answer in each arm: under the
  null the 2 arms are exchangeable. The estimand is the effect of the name
  on the answer and the readout together.

## The verification (2026-10-29)

A Fable 5.1 advisor checked every number of this page, `f8-method-b.md`,
`f8-lens-figure.md` and the F8 results of the runbook against the CSV
files, with no file change. It found 4 small mismatches (fixed on this
page and in `f8-method-b.md`) and the claims above that went too far
(rewritten). Its verdicts: F8.1 solid; F8.2 primary usable with the
caveats above; F8.2 secondary usable with the word caveats; F8.3 solid;
F8.4 a figure only. The joins, the Bonferroni denominators, the sign
convention, the McNemar test and the R3 values (`c` = `lp_a` - `lp_b`, no
missing value) are correct.
