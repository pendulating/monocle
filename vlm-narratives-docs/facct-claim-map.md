# FAccT claim map: the 6 claims and their results

Written 2026-10-31. Updated 2026-11-01 with the F10 results. This page maps the 6
claims of the FAccT results section to the results in `facct-runbook.md` and the
other pages of this folder. Use it to choose the evidence for each claim.

The section [Final pass](#final-pass-add-cut-strengthen) gives the plan for the last
edit of the paper. The rows C1.13 to C1.16, C2.11 to C2.16, C3.16 to C3.19, C4.13 to
C4.18, C5.19 to C5.22 and C6.14 to C6.26 are new. Their full tables are in
[f10-judgment-mechanism.md](f10-judgment-mechanism.md).

| Item | Value |
|------|-------|
| Draft of the rows of 2026-10-31 | `papers/cvpr27_machine-perception-public-spaces/sec/4_results.tex`, commit `1a0575a` (2026-10-31) |
| Draft of the rows of 2026-11-01 | `artificial-aesthetic/paper/sec/4_results.tex`, commit `846aec8` |
| Earlier sort | [facct-rq-evidence.md](facct-rq-evidence.md) sorts the results under RQ1 to RQ3. This page sorts them under the 6 claims |
| Line numbers | L numbers are the lines of each page on 2026-10-31. The link goes to the section, thus it stays correct when the lines move |
| Lines of the draft | A `4_results.tex:n` in a row of 2026-10-31 gives the old line. A row of 2026-11-01 gives the line of `846aec8`, as `now:n`. See the table below |
| Deadline | Confirmed on 2026-11-01 from the FAccT 2027 call: 2026-11-03, 23:59 AoE. The Nov 15 comment in `1_intro.tex` is wrong |
| Page limit | 14 single-column pages, with figures and tables. References do not count. Endmatter statements get 1 more page. Appendices do not count, but reviewers need not read them |

**Note:** [facct-runbook.md](facct-runbook.md) and [paper-repo-sync.md](paper-repo-sync.md) call the clone in `papers/` stale. On 2026-10-31 it held 1 Overleaf commit that `artificial-aesthetic/paper` (`6b5b07e`) did not hold. The update of 2026-11-01 reads `artificial-aesthetic/paper` at `846aec8`.

| Section of the draft | Line on 2026-10-31 | Line now |
|----------------------|--------------------|----------|
| Claim 1 | 16 | 16 |
| Claim 2 | 35 | 47 |
| Claim 3 | 40 | 59 |
| Claim 4 | 65 | 94 |
| Claim 5 | 85 | 117 |
| Claim 6 | 105 | 142 |

## How to read the tables

| Mark | Meaning |
|------|---------|
| `*` | We read the value from the named data file on 2026-10-31 |
| Pairs of values | A pair such as 0.82, 0.75 gives Gemma-4-31B, then Qwen3.5-9B |
| review 1.2 | The row of [facct-rq-evidence.md](facct-rq-evidence.md) that has this result. That page has the main threat of each result |
| strong, moderate, weak | The scale of [facct-rq-evidence.md](facct-rq-evidence.md). A result that the image, the prompt, the extractor or the sampling can explain is weak |

| Short name | Path |
|------------|------|
| `pool/` | `notebooks/facct/figures/f4_x2_pooled/` |
| `rq/` | `notebooks/facct/figures/rq/` |
| `blocks/` | `notebooks/facct/figures/f9_blocks/` |
| `mech/` | `figures/f10/mechanism/` |
| `f10/` | `figures/f10/` |

| Data set | Prompt | Input | Models | Extraction |
|----------|--------|-------|--------|------------|
| F4 | a description or advice | 1 image, 10,000 for each case | 6 | x2 (D131) |
| F9 | a subjective yes-or-no question | the same images | 6 | F9 schemas (D142) |
| Parent | a pairwise comparison | 2 images | Gemma-4-12B, Qwen3.5-9B | version 2 |
| F10 | "Is this place cool?", and 53 other adjectives | 1 image, 591 images in 197 NTAs | Gemma-4-31B, Qwen3.5-9B | F10.11 schemas; 7,092 answers, 3 repeats, 2 prompts |

**Caution:** do not compare an F4 rate with a parent rate. The prompt, the input and the extractor all differ ([runbook L652](facct-runbook.md#the-f4-span-extraction-d93-to-d96-2026-10-24)).

This page reads the "descriptive" tasks of the draft as F4 and the "prescriptive" tasks as F9. The note at `4_results.tex:82` ("prescriptive tasks from yday") matches the F9 blocks.

## Summary

**Update (2026-11-02):** F10.26 tests the F10 results on 14 prompts in 3 models (42 cells). The rows below carry a note "F10.26:" with the count of cells in which a result holds. See `f10-judgment-mechanism.md`, section "14 prompts: results (F10.26)".

| Claim | Evidence on 2026-10-31 | Evidence on 2026-11-01 | Main limit now |
|-------|------------------------|------------------------|----------------|
| 1. Cues and inferences of place | moderate | strong for "not always correct" | The probe tests 1 fact, on 591 images |
| 2. Awareness of the task | moderate | strong on Gemma-4-31B, moderate on Qwen3.5-9B | The 2 models differ. The `cool` prompts only |
| 3. Outsized inferences | strong as counts | strong as counts | No run ties an inference to a cue, thus "unsupported" has no test |
| 4. Internal hierarchy | strong for the cue order | strong for the cue order and the value order | The same restaurants rank weakly across paradigms |
| 5. Link to privileged areas | strong | strong; F10 adds a borough result only | No SES test on the F10 images |
| 6. Interpretability | weak | strong for the process; weak for specific objects | A fixed prompt gives no causal order |

The section "Open analyses" lists the 14 analyses that are still open.

## Claim 1. Cues and inferences of place

> "Models invoke aesthetic judgments using visual cues and inferences of place that are not always correct" (`4_results.tex:5`, `:16`)

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C1.1 | `location` inferences sit in 24.9% to 79.4% of restaurant answers and 69.0% to 89.0% of traces `*` | [runbook L1489](facct-runbook.md#the-model-as-a-hyperparameter-traces-notebook-250-section-48) | `pool/f4_model_differences.csv` | moderate; review 1.1 |
| C1.2 | `brand_knowledge` sits in 17.8% to 39.5% of restaurant answers, `institution_knowledge` in 2.8% to 21.5% `*` | the same | the same | moderate; review 1.1 |
| C1.3 | The sentence at `4_results.tex:18`. NTA Spearman on restaurant answers against % white, % bachelor's and income, 159 NTAs: see the table below `*` | [runbook L1581](facct-runbook.md#the-model-and-the-nta-findings-traces-notebook-250-section-410) | `pool/f4_nta_feature_correlations.csv` | moderate; review 2.5 |
| C1.4 | The F9 answers give the same location row: 8 of 60 restaurant cells pass, 0 of 60 street cells. Qwen3.8-27B goes the other way | [f9-inference-valence.md L94](f9-inference-valence.md#result-2026-10-30) | `notebooks/facct/figures/trace_ses/trace_ses_cells.csv` | moderate |
| C1.5 | The city map of the draft (`4_results.tex:22`). 3,372 of 368,271 street captions (0.9%) name NYC; 11 of 197 NTAs have none `*` | hashtags: [runbook L939](facct-runbook.md#the-caption-notebook-f4_captionspy-100-_f4cap-100-2026-10-28); fill: `notebooks/facct/_f4cap.py` | `pool/f4_street_caption_nta_city.json` | weak; no test against the features |
| C1.6 | The other-city map (`4_results.tex:28`). 96 of 10,000 street images (1.0%) get a caption that gives another city; 119 of 197 NTAs have none. Philadelphia 27, Washington DC 19, Chicago 16 `*` | `notebooks/facct/_f4cap.py`, `other_city_top` | `pool/f4_street_caption_nta_other_city.json` | weak; the only direct measure of a wrong place |
| C1.7 | Gemma-4-12B holds a latent place that it does not say. On Manhattan images, "Manhattan" is the top token of the logit lens in 47% of the answers (late layers). Only 17.5% of those answers write it | [whatami-lens-study.md L184](whatami-lens-study.md#a-latent-borough-against-the-true-borough); [runbook L2431](facct-runbook.md#the-advisor-review-2026-10-28) | `multirun/whatami/lens_report/gemma-4-12b/borough_check_late.csv` | moderate |
| C1.8 | At fixed pixels, a neighbourhood name changes the answer: 16 of 32 tests. The direction differs by model and case | [runbook L2573](facct-runbook.md#results-of-f82-and-f83-2026-10-29); [f8-counterfactual-cue.md](f8-counterfactual-cue.md) | `figures/f8/cue/cue_primary.csv` | strong for "the name matters"; review 3.1 |
| C1.9 | With race and income apart, 9 of 32 tests pass. 1 name (East New York) drives the large Gemma-4-31B effect | [runbook L2628](facct-runbook.md#f85-the-decoupled-cue-d136) | `figures/f8/decoupled/` | moderate |
| C1.10 | The name and the image act together. All 8 `cross` tests pass; on both Gemma models the true borough name scores higher. The float32 run holds this | [runbook L3200](facct-runbook.md#f101-result-2026-10-30); [runbook L3539](facct-runbook.md#f101-float32-result-2026-10-31) | `figures/f10/names_fp32/` | strong for the crossing |
| C1.11 | The verdict moves with a cue that has no relation to the question: the north camera face wins in 11 of 13 cells | [camera-face-effect.md](camera-face-effect.md); review 1.8 | `notebooks/cvpr/master/outputs/face_effect_grid_v1.0.0.csv` | strong as a measurement fact |
| C1.12 | Parent traces: 28 cues that the extractor calls neutral move the judgment by more than 1 logit | [cue-hierarchy.md L290](cue-hierarchy.md#the-model-judges-by-words-it-calls-neutral) | see review M8: the file now gives 12 | moderate; review 1.11 |
| C1.13 | The direct test of "not always correct". All 591 images are in New York City. The model says that the image is in New York on 61.6%, 88.5% and 88.8% of images (Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B). See the table below `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `f10/probe/probe_place.csv` | strong as a measure |
| C1.14 | The city is also not sure outside the core: `in a city` is 0.75, 0.94 on Staten Island, `urban` is 0.60, 0.58. Manhattan is 0.99, 0.99 and 0.99, 0.98 `*` | the same | the same | strong as a measure |
| C1.15 | Reasoning traces state a location in 37% and 60% of traces. Manhattan 0.57, 0.74; Staten Island 0.16, 0.47 `*`. The share follows the margin: r 0.32, 0.20 | the same | `f10/reason/reason_labels.csv` | moderate |
| C1.16 | The read ties location to the image. At `place`, `location` and `nearby` rank 3 and 4 only with the image (`rank_added`). With no image they rank about 20,000 on Gemma-4-31B (`now:176` holds) `*` | the same | `f10/stage/stage_words.csv` | moderate; 1 model |

The test of C1.13, share of images with a positive `in New York` margin:

| Model | All | Manhattan | Brooklyn | Bronx | Queens | Staten Island |
|-------|-----|-----------|----------|-------|--------|---------------|
| Gemma-4-31B | 0.616 | 0.833 | 0.780 | 0.694 | 0.412 | 0.208 |
| Qwen3.5-9B | 0.885 | 0.948 | 0.931 | 0.919 | 0.831 | 0.729 |
| Qwen3.6-27B | 0.888 | 0.948 | 0.918 | 0.901 | 0.870 | 0.708 |

The Qwen3.6-27B row is from 2026-11-02 (F10.26, task 5). The draft at `4_results.tex:269` names 2 models; see P45 in [f10-manuscript-proposals.md](f10-manuscript-proposals.md).

The test of C1.3, restaurant answers:

| Kind | Rows with a positive rho (of 18) | rho range | Rows that pass Bonferroni (of 18) | Families that pass |
|------|----------------------------------|-----------|-----------------------------------|--------------------|
| `location` | 15 | -0.19 to +0.59 | 7 | Gemma, Llama |
| `institution_knowledge` | 17 | -0.03 to +0.53 | 5 | Gemma, Qwen |
| `brand_knowledge` | 18 | +0.18 to +0.57 | 10 | Gemma, Qwen, Llama |

Limits:

- The sentence at `4_results.tex:18` holds in direction for restaurants. All 3 families pass the test for `brand_knowledge` only.
- No row of these 3 kinds passes in the street answers. The caption case gives 1 row. The street case cannot support the sentence.
- A location or brand inference from a sign is grounded in the sign (review 1.1). More chains in an area can also give the `brand_knowledge` result. No result controls for this.

**Result on 2026-11-01:** "Not always correct" now rests on C1.13, not on C1.6 and single quotes. The probe asks a direct question, and all images give the same true answer.

**No result exists for:** the accuracy of `location` spans in free text. C1.13 reads a forced one-word probe.

## Claim 2. Awareness of the task

> "Models demonstrate an awareness of the task requirements for aesthetic evaluation, but not nearly enough" (`4_results.tex:6`, `:35`)

The table below fills the TODO at `4_results.tex:37`. It gives the percent of texts with 1 `uncertainty` span or more, on the complete images `*` (`pool/f4_model_differences.csv`, category `uncertainty: any`).

| Model | Restaurant trace | Caption trace | Street trace | Restaurant answer | Caption answer | Street answer |
|-------|------------------|---------------|--------------|-------------------|----------------|---------------|
| Gemma-4-12B | 35.0 | 6.5 | 0.2 | 19.0 | 0.5 | 0.0 |
| Gemma-4-31B | 37.0 | 5.0 | 0.2 | 24.8 | 0.5 | 0.0 |
| Qwen3.5-9B | 60.9 | 28.0 | 1.5 | 24.0 | 2.9 | 0.1 |
| Qwen3.8-27B | 94.8 | 37.3 | 5.6 | 79.8 | 1.1 | 0.1 |
| Llama-3.2-11B | no trace | no trace | no trace | 7.1 | 0.6 | 0.2 |
| Llama-3.2-90B | no trace | no trace | no trace | 12.0 | 1.9 | 0.1 |

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C2.1 | The table above. All 4 thinking models write the most uncertainty on the restaurant case, as the draft says `*` | [runbook L1489](facct-runbook.md#the-model-as-a-hyperparameter-traces-notebook-250-section-48) | `pool/f4_model_differences.csv` | moderate; review 1.5 |
| C2.2 | The uncertainty falls from the trace to the answer in all 4 thinking models: 35.0 to 19.0, 37.0 to 24.8, 60.9 to 24.0, 94.8 to 79.8 `*` | [runbook L1388](facct-runbook.md#the-notebooks-on-x2-d131-2026-10-27) | the same | moderate; 2 extractions |
| C2.3 | The reasons in the restaurant answers: illegible 3.2 to 50.8, knowledge 1.4 to 38.4, out_of_view 2.2 to 32.4, identity 0.7 to 8.7 `*` | the same | the same | moderate |
| C2.4 | F9: `out_of_view` is 20,158 of 26,562 uncertainty spans in the restaurant answers. Without it, 2.4% to 25.8% of the answers hold a span | [f9-inference-valence.md L81](f9-inference-valence.md#the-out_of_view-filter) | `data/f4_traces/subj/inference_valence/uncertainty_reasons.parquet` | moderate |
| C2.5 | The thinking models seldom take a side under F9: `mixed` is 40% to 85% of their answers ("it depends on what you are looking for") | [f9-subjective-prompt.md L209](f9-subjective-prompt.md#47-the-verdict-a-description); [runbook L3039](facct-runbook.md#results-of-the-generation-2026-10-29) | `figures/f9/verdict_shares.csv` | moderate; label agreement 19 of 24 on restaurants |
| C2.6 | Parent: the uncertainty count separates `NotSure` from a decision at AUC 0.60 to 0.93 in all 4 arms. `cannot_see` gives 0.75 to 0.95 | [trace-channels.md L187](trace-channels.md#the-reading) | `notebooks/cvpr/master/figures/trace_channels_abstention_arms.csv` | moderate; review 3.14 |
| C2.7 | Parent abstention share differs by task: restaurants 0.28 and 0.58, schools 0.52 and 1.00, street photography 0.00 (Gemma and Qwen) `*` | [proxy-overview.md](proxy-overview.md); `tables/cross-model.tex` | `artificial-aesthetic/results/cvpr/cross_model.csv` | moderate |
| C2.8 | Abstention hides no judgment: forced labels lean to 0, and the score delta is 0.000 on all 14 rows | [prompt-ablations.md](prompt-ablations.md); [quantile-shift.md](quantile-shift.md) | `notebooks/cvpr/master/outputs/ablations_v1.0.0.csv` | strong; review 3.5 |
| C2.9 | "Not nearly enough": an `inference` span sits in 99% to 100% of restaurant texts, 4 to 8 for each `visual_evidence` group. The models infer in spite of the doubt | [runbook L1489](facct-runbook.md#the-model-as-a-hyperparameter-traces-notebook-250-section-48) | `pool/f4_response_inference_kind.csv` | moderate; review 1.1 |
| C2.10 | F9: the verdict of the answer equals the verdict of the trace in only 51% to 71% of the answers | [f9-subjective-prompt.md L209](f9-subjective-prompt.md#47-the-verdict-a-description) | `figures/f9/verdict_agreement.csv` | weak |
| C2.11 | The refusal of the verdict. `cool_word` asks for yes or no: Gemma-4-31B gives neither on 33.3%, Qwen3.5-9B on 0.4%. `cool_open`: no yes or no on 91.9% and 52.1% `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `f10/reason/reason_verdict.csv` | strong for Gemma-4-31B |
| C2.12 | The direct evidence of taste. Share of reasoning traces with each note: `subjective` 0.664, 0.172; `no_taste` 0.415, 0.028; `forced_choice` 0.349, 0.061; `safe_default` 0.120, 0.193 `*` | the same | `f10/reason/reason_labels.csv` | strong; model-labelled |
| C2.13 | The awareness falls from the trace to the answer. `subjective` goes from 0.664 to 0.222, and from 0.172 to 0.001 `*`. On Gemma-4-31B the answer moves it to a persona: holder `persona` 0.318 to 0.851 | the same | the same | moderate; 2 fields of 1 run |
| C2.14 | "Not nearly enough". Qwen3.5-9B says yes on 75.4% of `cool_word`, and 19.3% of its traces call yes the safe default. Traces still argue a side: for in 0.42, 0.91 of traces, against in 0.80, 0.68 | the same | `f10/reason/` | moderate |
| C2.15 | The read holds the same awareness before any text. At `?`, `subjective` ranks 2 and 1 with the image (`rank_added`). With no image it ranks 22,500 and 942 `*` | the same | `f10/stage/stage_words.csv` | moderate |
| C2.16 | The models state the ambiguity of the word. 23% and 26% of reasoning traces name temperature or a close word. A typical line: "Is it cool in terms of temperature? No, it's a photo." | C6.17 | `f10/sense/sense_links.csv` | moderate; a regex, 14 lines read by hand |

Limits:

- The reasons in C2.3 are about what the image shows or what the model knows. They are not about aesthetic standards. On 2026-11-01 the `cool` schema adds a class for this (`decision_note`). F4 and F9 have no such class, thus C2.12 holds for the `cool` prompts only.
- The closest evidence for an aesthetic hedge in F9 is C2.5, and the verdict rule misses the design rule on restaurants.
- The case contrast mixes 2 things. The restaurant prompt names 1 object with a name to read. The caption and street prompts do not. Thus the interested and disinterested contrast is not isolated.
- The rate differs 11 times by model (7.1 to 79.8). It can measure style and not accuracy (review 1.5).
- No result tests the difference between the cases for each model. The pooled table has the counts for a chi-square test.

## Claim 3. Outsized inferences

> "Models make outsized inferences about aesthetic properties and values in both descriptive and (even moreso) prescriptive tasks." (`4_results.tex:7`, `:40`)

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C3.1 | `popularity`: 7.9% to 43.2% of restaurant answers (the draft gives 8% to 43%) `*` | [runbook L1489](facct-runbook.md#the-model-as-a-hyperparameter-traces-notebook-250-section-48) | `pool/f4_model_differences.csv` | moderate; review 1.2 |
| C3.2 | `atmosphere`: 33.7% to 70.9% of restaurant answers `*` | the same | the same | moderate |
| C3.3 | Wealth, class, demographic or crime: 4.3% to 10.4% of restaurant answers; wealth 1.8% to 5.9%; crime 0.0%; street under 1% `*` | the same | the same | strong as a count; review 1.3 |
| C3.4 | The trace holds more than the answer. A person in the photograph: 26% to 54% of caption traces, 0.5% to 13% of caption answers | [runbook L1388](facct-runbook.md#the-notebooks-on-x2-d131-2026-10-27) | `pool/f4_trace_against_answer.csv` | moderate; review 1.4 |
| C3.5 | Parent traces: 27.5% hold a social inference (schools 70%, restaurants 46%, road quality 0.9%) | review 1.10 | `data/trace_extractions_v2_gemma12b_by_qwen27b/` | moderate as a floor |
| C3.6 | Parent traces: a `wealth` inference pays for the image in every case (lift +0.23 to +0.44). 96.7% run up: "almost no vocabulary for poverty" | [trace-channels.md L251](trace-channels.md#3-what-the-model-reads-into-a-photograph) | `notebooks/cvpr/master/figures/trace_channels_inference_summary.csv` | moderate; review 1.9 |
| C3.7 | Order and disorder, F9 inference block. Top: `pride`, `owners care`, `well-run`, `safe`. Bottom: `neglect`, `not maintained`, `run-down`, `vacant`, `closed` `*` | [f9-inference-valence.md L139](f9-inference-valence.md#the-inference-block) | `blocks/restaurant_inference_block_cues.csv` | moderate |
| C3.8 | Order and disorder, F9 evidence block. Bottom: `litter`, `significant graffiti`, `debris`, `graffiti`, `scaffolding`, `trash` `*` | [f9-inference-valence.md L125](f9-inference-valence.md#the-f9-win-blocks-d147) | `blocks/restaurant_valence_block_cues.csv` | moderate |
| C3.9 | Parent ladder: `construction site` -0.41, `graffiti` -0.38, `scaffolding` -0.27 over 5 facility cases | [cue-hierarchy.md L151](cue-hierarchy.md#the-2-word-figures) | `notebooks/cvpr/master/figures/cue_words_facility_ladder.png` | moderate |
| C3.10 | The same cue flips with the frame, as the 2 graffiti quotes show. `graffiti on wall` is +0.83 for street photography and -1.71 to -4.33 elsewhere | [cue-hierarchy.md L270](cue-hierarchy.md#street-photography-is-the-contrarian) | `notebooks/cvpr/master/figures/cue_words_outliers.csv` | moderate; review 3.11 |
| C3.11 | "Even moreso", valence. In restaurant answers the good share of evidence goes from 0.03-0.09 to 0.12-0.30, and the bad share from 0.01-0.02 to 0.11-0.32 (F4 to F9) | [f9-subjective-prompt.md L185](f9-subjective-prompt.md#46-the-span-mix-responses) | `figures/f9/span_mix.csv` | strong |
| C3.12 | "Even moreso", street. Inference spans for each answer go from 0.02-0.18 to 0.68-2.10 | the same | the same | strong |
| C3.13 | F9 social inferences: 65,266 spans. Restaurant wealth 34,072, restaurant upkeep 20,126, street safety 4,490 (80% bad) | [f9-inference-valence.md L31](f9-inference-valence.md#result) | `data/f4_traces/subj/inference_valence/full.parquet` | moderate; retest kappa 0.889 |
| C3.14 | A name alone adds wealth words: with a rich name, all 4 models write more `upscale` words (16.0% against 10.1%, Gemma-4-12B) | [runbook L2573](facct-runbook.md#results-of-f82-and-f83-2026-10-29) | `figures/f8/cue/cue_words.csv` | moderate; review 3.2 |
| C3.15 | Words that only 1 model writes: 486 pass; 419 are Qwen3.8-27B words (`mediocre`, `unglamorous`, `lifeless`) | [runbook L1410](facct-runbook.md#words-that-only-1-model-uses-descriptor-notebook-140-spelling-key-170) | `notebooks/facct/figures/f4_descriptors_x2_pooled/` | moderate for vocabulary; review 1.6 |
| C3.16 | F10 inferences, share of reasoning traces: atmosphere 0.67, 0.87; neighborhood character 0.64, 0.83; upkeep 0.07, 0.16; safety 0.02, 0.12; class 0.03, 0.09; wealth 0.02, 0.06 `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `f10/reason/reason_labels.csv` | moderate |
| C3.17 | A stated inference holds less of the margin than a stated value: R2 0.235, 0.198 against 0.572, 0.468 `*` | the same | `f10/reason/reason_fit.csv` | moderate |
| C3.18 | Reads show social content that holds on held-out NTAs. Qwen3.5-9B Brooklyn at `cool` L20 to L24: neighborhood, barrio, unemployment, landlords, homeless. See the limit at C6.23 | the same | `f10/borough/borough_words.csv` | moderate |
| C3.19 | C3.10 with a test. The value gives the sign of 1 object in 18 of 22 and 20 of 25 objects (chi-square, Holm). Graffiti: 0.86, 0.97 for under edge; 0.00, 0.05 for under order `*` | the same | `f10/reason/argument_cells.csv`, `argument_flips.csv` | strong; `mech/mech_objects.png` |

The 4 quotes with a TODO at `4_results.tex:45` to `:51`. All 4 are final answers of the canonical batch in `data/f4_traces/pooled/f4_traces_1k.parquet` `*`. The last column is `nearest_inspection_days` of `data/f6_restaurant_depiction/v4/best.parquet` `*`.

| Line | NTA | Model | `doc_id` | Restaurant, CAMIS | Days to an inspection |
|------|-----|-------|----------|-------------------|-----------------------|
| 45 | Greenpoint (BK0101) | Qwen3.5-9B | `qwen3.5-9b\|restaurant\|455\|response` | CASANOVA, 50176146 | 129 |
| 47 | Murray Hill-Kips Bay (MN0603) | Qwen3.8-27B | `qwen3.8-27b\|restaurant\|962\|response` | VATAN, 50124382 | 149 |
| 49 | Bushwick West (BK0401) | Gemma-4-12B | `gemma-4-12b\|restaurant\|512\|response` | KNICKERBOCKER BAGEL, 50171899 | 8 |
| 51 | Co-op City (BX1004) | Gemma-4-31B | `gemma-4-31b\|restaurant\|879\|response` | CAPRI IV DELI, 50013830 | 165 |

The closure claims at lines 45 and 51 are a correctness test. The open rule ([D50, runbook L108](facct-runbook.md#decisions-of-2026-10-23-f6-version-2)) keeps only restaurants with an inspection within 180 days of capture. Both restaurants pass that rule.

Limits:

- "Unsupported" (`4_results.tex:54`) has no test. "Ungrounded" is not a class of the schema, and no run ties an inference to a cue (review section 2.2). The paper can say that the models infer. It cannot say that a given inference has no cue.
- "Even moreso" does not hold for the restaurant inference count. It rises in 3 of 6 models and falls in 3 (Gemma-4-12B 3.80 to 4.27; Qwen3.8-27B 6.23 to 3.90).
- Most F9 restaurant wealth spans deny wealth. A keyword rule finds a negation in 88.4% of the spans with a word of wealth ([f9-inference-valence.md L162](f9-inference-valence.md#negation)).
- Every inference rate is a floor: extractor recall is 0.26 and 0.07 on 10 human-read traces, in the parent corpus (review section 2.2).
- The street safety spans are about the photographer more than the area. 60.2% hold words of the shoot or the road ([f9-inference-valence.md L162](f9-inference-valence.md#negation)).
- The `crime` spans of the parent all name a cue (review 1.12). They are counter-evidence for "unsupported".

## Claim 4. An internal hierarchy

> "Model judgments evidence an internal aesthetic hierarchy" (`4_results.tex:8`, `:65`)

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C4.1 | One standard across prompts: cross-case Spearman of the `category_valence` lift is 0.91 (0.76 to 0.94) over 7 prompts; 16 of 22 keys hold 1 sign | [cue-hierarchy.md L66](cue-hierarchy.md#the-unit) | `notebooks/cvpr/master/figures/cue_hierarchy_agreement.csv` | moderate; review 3.11 |
| C4.2 | The model's own words give the same ladder: 0.85 (0.80 to 0.94) over 24 words, 5 facility cases | [cue-hierarchy.md L151](cue-hierarchy.md#the-2-word-figures) | `notebooks/cvpr/master/figures/cue_words_facility_ladder.png` | moderate |
| C4.3 | The cue list of `4_results.tex:75` (Gemma-4-12B, lift). Up: `outdoor seating` +0.31, `clear signage` +0.31, `large glass window` +0.28, `greenery` +0.16, `bike lane` +0.14, `tree` +0.12, `high rise building` +0.06. Down: `graffiti` -0.34, `construction site` -0.32, `elevated train track` -0.28, `small storefront` -0.24, `older brick building` -0.22, `scaffolding` -0.16, `laundromat` -0.11 `*` | [cue-hierarchy.md L308](cue-hierarchy.md#what-each-case-values) | `notebooks/cvpr/master/figures/cue_words_per_case.csv` | moderate |
| C4.4 | The win block of `4_results.tex:68`: Gemma-4-12B judge, Qwen3.8-27B extractor, 10,992 traces | [facct-pairwise-ses.md L83](facct-pairwise-ses.md#the-word-blocks-of-the-same-cases) | `notebooks/cvpr/restaurants/figures/restaurants_win_block.png` | moderate |
| C4.5 | The new analysis at `4_results.tex:82`: 4 F9 blocks. Evidence blocks use Gemma-4-12B. Inference blocks use 6 models | [f9-inference-valence.md L125](f9-inference-valence.md#the-f9-win-blocks-d147); [runbook L2998](facct-runbook.md#the-planned-figures) | `blocks/` | moderate |
| C4.6 | Cross-paradigm order: the cue order holds from pairwise to F9, rho 0.80 to 0.87 | [f11-paradigm-consistency.md L99](f11-paradigm-consistency.md#32-the-order-of-the-cues) | `data/f11/f11_cues.parquet` | strong |
| C4.7 | Cross-model, parent, restaurants: area tau 0.46, 0.51, 0.51 (tract, NTA, district). Pair by pair, kappa is only 0.23 `*` | `tables/cross-model.tex`; [proxy-overview.md](proxy-overview.md) | `artificial-aesthetic/results/cvpr/cross_model.csv` | moderate |
| C4.8 | Cross-model, street photography: label kappa 0.48, side kappa 0.65 `*` | the same | the same | moderate |
| C4.9 | Cross-model, F4: the models agree on the R3 readout over the same images, Pearson 0.50 to 0.74 | [f8-counterfactual-cue.md](f8-counterfactual-cue.md) | `figures/f8/cue/r3_model_pearson.csv` | moderate; review 3.3 |
| C4.10 | Cross-model, F4: 157 of 193 significant NTA rows have the same sign in all models; 0 rows have 2 significant models with opposite signs | [runbook L1581](facct-runbook.md#the-model-and-the-nta-findings-traces-notebook-250-section-410) | `pool/f4_nta_feature_signs.csv` | moderate; review 3.8 |
| C4.11 | The model prices a modifier, not a noun: `large glass` +0.29 and `smaller` -0.39 on `window`; every `modern` form pays and every `older` form costs | [cue-hierarchy.md L170](cue-hierarchy.md#why-a-row-is-wide-the-modifier) | `notebooks/cvpr/master/figures/cue_words_modifiers.png` | moderate |
| C4.12 | Contestability of the standard: street photography inverts the upkeep rungs (`clean` -0.70, `fire escape` +1.42) | [cue-hierarchy.md L270](cue-hierarchy.md#street-photography-is-the-contrarian) | `notebooks/cvpr/master/figures/cue_words_outliers.csv` | moderate |
| C4.13 | The value order of "cool". Partial r given `good`, `nice` and the borough: interesting 0.87, 0.56; unique 0.60, 0.43; stylish 0.54, 0.46; trendy 0.44, 0.49; ordinary -0.44, -0.56; boring -0.71, -0.35; clean -0.38, -0.29; safe -0.26, -0.34 `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `f10/probe/probe_links.csv` | strong; `f10/probe/probe_links.png` |
| C4.14 | Values read cool better than properties do: R2 0.924, 0.810 against 0.840, 0.749. Neutral words give 0.018, 0.021 `*` | the same | `f10/probe/probe_model.csv` | strong |
| C4.15 | Stated standards in the traces. Utility against r -0.59, -0.61; beauty against -0.52, -0.52; beauty for 0.37, 0.33; heritage for 0.49, 0.20 `*` | the same | `f10/reason/reason_labels.csv` | moderate; `mech/mech_values.png` |
| C4.16 | Object and value together give the sign of an argument: AUC 0.921, 0.922. Value alone 0.822, 0.864. Object alone 0.853, 0.856 `*` | the same | `f10/reason/argument_sign.csv` | strong |
| C4.17 | Repeats of 1 image agree more on the object than on the value: ICC 0.551, 0.551 against 0.312, 0.278 (`cool_open`) `*`. The hierarchy is firmer in what the model cites than in the standard it applies | the same | `f10/reason/argument_repeat.csv` | moderate |
| C4.18 | Cross-model, probes: the cool margin agrees at r 0.72. The median over 53 adjectives is 0.76, and the range is 0.09 to 0.90 `*` | the same | `f10/probe/probe_cross.csv` | moderate |

Limits:

- **The photographs of the block are not the images that the model judged.** The run read raw cube faces. The block shows F6 views, and we removed 8 low restaurants by hand ([facct-pairwise-ses.md L101](facct-pairwise-ses.md#the-restaurant-photographs-are-f6-views)). Thus the argument at `4_results.tex:75` about bottom restaurants with clear signage does not follow from this figure. The photographs of the F9 blocks are images that the models judged (`blocks/restaurant_valence_block_images.csv`), thus they can carry that argument.
- The bottom of the ranking is partly an image effect. The lowest bin holds 28.0% F6 `Yes`, the highest 68.5% ([facct-pairwise-ses.md L134](facct-pairwise-ses.md#the-bottom-of-the-ranking-is-partly-an-image-effect)).
- "Especially if they are brick" (`4_results.tex:75`) does not hold. `brick building` has a lift of -0.01 and `red brick building` -0.06. `older` carries the cost. Under F9, `red brick` is +0.83 and `classic brick` +0.85 `*`. C3.19 gives the reason: brick is 0.91, 0.98 for under heritage, and 0.00, 0.09 for under utility.
- The value order of C4.13 is for `cool`, on street images. It is not the restaurant ladder of C4.3. Clean and neat work against `cool` after the control for approval, and they pay in the ladder. This supports the contestability of C4.12.
- `laundromat` is weak: -0.11 in the pairwise traces and -0.02 under F9 `*`. `McDonald's` has 241 spans but no row in the per-case file `*`.
- The level of the valence does not hold across paradigms. H11 fails in 11 of 12 cells ([f11-paradigm-consistency.md L166](f11-paradigm-consistency.md#5-the-reading)).
- The same restaurants rank weakly across paradigms: rho 0.04 to 0.23, on different views ([f11-paradigm-consistency.md L122](f11-paradigm-consistency.md#33-the-same-restaurant-u5-no-kl)).
- All cue prices rest on Gemma-4-12B traces. No hierarchy exists yet for the other 5 models, except the pooled inference blocks.
- A cue follows a win and need not cause it. When 1 facility holds 2 valences, the good one sits in the win 0.988 of the time (review section 2.2). The F9 block "says how the answers frame a cue" ([runbook L2998](facct-runbook.md#the-planned-figures)).

## Claim 5. A link to privileged areas

> "Privileged aesthetic properties and values correlate with privileged areas of living" (`4_results.tex:9`, `:85`)

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C5.1 | Pairwise: 48 of 90 cells pass. % bachelor's tau is 0.35-0.48 (subway), 0.21-0.41 (street), 0.09-0.44 (restaurants). % white tau is 0.13-0.38 | [facct-pairwise-ses.md L50](facct-pairwise-ses.md#result-2026-10-30) | `notebooks/facct/figures/pairwise/pairwise_ses_cells.csv` | moderate; aggregation raises tau |
| C5.2 | The pairwise gradients hold on the F6 `Yes` restaurants: 21 of 30 cells pass, no cell loses significance, no sign changes | [facct-pairwise-ses.md L148](facct-pairwise-ses.md#the-gradients-hold-on-the-yes-restaurants) | `notebooks/facct/figures/pairwise/pairwise_ses_restaurants_f6.csv` | strong as a check |
| C5.3 | Method B: 37 of 40 restaurant effects stay with image-content controls; median share kept 0.89; 0.04 to 0.18 SD | [runbook L2471](facct-runbook.md#results-of-f81-2026-10-29); [f8-method-b.md](f8-method-b.md) | `figures/f8/method_b/valence_cases.csv` | strong; review 2.2 |
| C5.4 | The R3 readout: `c_cool_i` rho +0.45 to +0.74 with % white, +0.40 to +0.80 with % bachelor's; 16 of 16 rows stay with the controls; 0.07 to 0.32 SD | [runbook L2573](facct-runbook.md#results-of-f82-and-f83-2026-10-29) | `figures/f8/cue/r3_nta_spearman.csv`; `figures/f8/method_b_r3/c_cool_i_cases.csv` | strong; review 2.3, 2.4 |
| C5.5 | The figure of `4_results.tex:99` is the plot of C5.3 and C5.4 | [facct-rq-evidence.md L361](facct-rq-evidence.md#6-headline-results) | `rq/rq2_method_b.json` | strong |
| C5.6 | H9: the subjective prompt steepens the gradient in 34 of 60 rows (restaurant 24, street 10), 0 flatter; 23 of 60 with standardized outcomes | [runbook L3073](facct-runbook.md#results-of-the-test-2026-10-30); [f9-subjective-prompt.md L114](f9-subjective-prompt.md#43-the-primary-test-h9) | `figures/f9/h9_primary.csv` | strong; report both forms |
| C5.7 | The F9 verdict: the share of `yes` rises with % white, % bachelor's and income in all 12 cells; +12.6 points per SD of % bachelor's (Qwen3.5-9B restaurant) | [f9-subjective-prompt.md L209](f9-subjective-prompt.md#47-the-verdict-a-description) | `figures/f9/verdict_fits.csv` | moderate; a description |
| C5.8 | `trace_ses`, 124 of 480 cells: see the table below. This figure is in the paper folder, but the draft does not use it | [f9-inference-valence.md L94](f9-inference-valence.md#result-2026-10-30) | `notebooks/facct/figures/trace_ses/trace_ses_cells.csv` | strong for restaurants |
| C5.9 | NTA net valence: Gemma-4-12B rho +0.74 with % white, +0.72 with % bachelor's; Qwen3.8-27B +0.67 and +0.64 with income | [runbook L1801](facct-runbook.md#strata-valence-uncertainty-the-nta-and-the-cuisine-traces-notebook-280-sections-412-and-413-2026-10-28) | `figures/f8/method_b/valence_meta.json` | strong as a correlation; review 2.1 |
| C5.10 | Cue rates by NTA: 408 of 7,620 tests pass (% bachelor's 176, % white 115, income 94, % Black 15, % Asian 8) | [runbook L1581](facct-runbook.md#the-model-and-the-nta-findings-traces-notebook-250-section-410) | `pool/f4_nta_feature_correlations.csv` | moderate; review 2.5 |
| C5.11 | Lighting (`4_results.tex:94`), restaurant answers: rho +0.27 to +0.56 with % bachelor's (4 of 6 pass) and +0.25 to +0.44 with income (3 of 6 pass) `*` | the same | the same | moderate |
| C5.12 | Greenery, restaurant answers: rho +0.20 to +0.44 with % white (4 of 6 pass). With % Black: -0.08 to -0.26, 0 of 6 pass `*` | the same | the same | moderate; see the draft checks |
| C5.13 | Parent income: tract r 0.35 to 0.47 under paraphrase, persona, order swap and no city name | [robustness-battery.md](robustness-battery.md); [prompt-ablations.md](prompt-ablations.md) | `notebooks/cvpr/master/outputs/robustness_estimand_v1.0.0.csv` | strong at the area layer; review 3.4 |
| C5.14 | Popularity in the 5 richest NTAs: 30.2% of 2,731 answers; in the 5 poorest: 20.3% of 1,693 | [runbook L1675](facct-runbook.md#quotes-from-the-top-k-and-the-bottom-k-ntas-traces-notebook-290-section-4101-2026-10-28) | `pool/` | weak; a description |
| C5.15 | Pair-level design at fixed pixels. A name changes the judgment, but the direction differs by model and case. Only 1 of 9 significant F8.5 tests favours the whiter name | [runbook L2573](facct-runbook.md#results-of-f82-and-f83-2026-10-29); [runbook L2628](facct-runbook.md#f85-the-decoupled-cue-d136) | `figures/f8/cue/cue_primary.csv`; `figures/f8/decoupled/` | strong against a simple "poor is worse" prior |
| C5.16 | Lens, income: 7 of 32 status-word tests pass the controls with the expected sign, as a description. The gated test gives 0 of 90 | [runbook L3317](facct-runbook.md#f102-result-2026-10-30) | `figures/f10/reads/reads_ses.csv` | weak |
| C5.17 | The refused judgment: Gemma-4-12B-it refuses "rich or poor" on 62% to 72% of images, and a readout on those images separates rich from poor NTAs at AUC 0.73 | [lens-refusal-probe-results.md](lens-refusal-probe-results.md) | `outputs/_monocle/probe_nta/group_test/class.csv` | moderate; review 2.10 |
| C5.18 | Caption entropy: 9 of 70 tests on restaurants, 3 of 70 on street. A homogeneity result, not a bias result | [runbook L939](facct-runbook.md#the-caption-notebook-f4_captionspy-100-_f4cap-100-2026-10-28) | `pool/f4_caption_entropy_nta_features.csv` | weak; review 2.7 |
| C5.19 | Cool by borough, in SD of the NTA means: Manhattan +0.95, +1.16; Brooklyn 0.0, 0.0; Bronx -0.42, -0.37; Queens -0.20, -0.31; Staten Island -0.26, -0.28. F 11.9, 18.7; eta2 0.20, 0.28 `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `f10/probe/probe_borough.csv` | strong for boroughs. F10.26: the order depends on the adjective. Manhattan leads on `cool`, `fancy` and `urban` in 18 of 18 prompt cells, and sits last on `ordinary` in 6 of 6. Staten Island leads on `nice` (6 of 6) and `pretty` (4 of 6). The Bronx leads on `ugly` (5 of 6). `f10/prompts/t8_borough_order.csv` |
| C5.20 | Status carries the top of the order. `trendy`, `rich`, `famous`, `upscale`, `expensive` and `fashionable` give the largest borough effects (Manhattan about +1.4). 34 and 38 of 53 ANOVAs pass Holm `*` | the same | the same | strong for boroughs. F10.26: `fancy` gives the largest Manhattan t of the 7 adjectives (8.4 to 10.5) in all 6 cells. `f10/prompts/t8_manhattan.csv` |
| C5.21 | Yes share under `cool_word`: Manhattan 0.448, 0.927; Staten Island 0.382, 0.583; Bronx 0.225, 0.661 `*` | the same | `f10/reason/reason_verdict.csv` | moderate |
| C5.22 | The text gives each borough a stable register. Manhattan: corporate, sterile, upscale. Brooklyn: gritty, authentic, lived-in. Bronx: institutional, generic, mundane. Queens and Staten Island: suburban, quiet, peaceful `*` | the same | `f10/reason/reason_words.csv` | moderate; log odds on text |

The table of C5.8, cells that pass of 60:

| Outcome (F9 answers) | Restaurants | Street photography |
|----------------------|-------------|--------------------|
| Net valence | 48 | 18 |
| Social net valence (the draft note at `4_results.tex:92`) | 29 | 1 |
| Location inference | 8 | 0 |
| Uncertainty, in view | 20 | 0 |

Limits:

- The street case is near null for F4 value: 3 effects stay of 50, and 1 new effect comes in. Do not extend the restaurant sentence to street without F9 (review section 3.2).
- % Asian is never significant in method B, and it gives no H9 row.
- The NTA is an ecological unit. The paper cannot claim that a model treats people differently.
- 96% of the F4 spans are neutral, and the effects are small (0.04 to 0.32 SD).
- The controls are coarse, and a control can be a mediator.
- The raw restaurant H9 effect is mostly scale. The F9 gradient is about 1.4 times the F4 gradient in SD units.
- Parent unit-layer signs can differ from area signs: restaurants unit tau -0.04 against CD +0.10 ([facct-rq-evidence.md L146](facct-rq-evidence.md#32-counter-evidence); [unit-validation.md](unit-validation.md)).
- bfloat16 can move a Gemma-4-12B mean by 0.3 to 0.4 nats. The F8 readouts use bfloat16 and did not run again ([runbook L3539](facct-runbook.md#f101-float32-result-2026-10-31)).
- The street panel of the social row is about the shoot, and it has low power. Do not read it as a null result for the area.
- C5.19 to C5.22 are borough contrasts. A borough is not an SES measure, and F10 has no SES test. Do not move these rows to an SES sentence without that test.

## Claim 6. Interpretability

> "Interpretability methods reveal how models judge specific visual objects." (`4_results.tex:10`, `:105`)

| ID | Result | Source | Data | Strength |
|----|--------|--------|------|----------|
| C6.1 | The teaser figure: R-lens on Gemma-4-12B, layer 42, 8 images, 16 blocks. Most tokens name the scene | [runbook L2718](facct-runbook.md#run-status) | `rq/rq1_patch_words.json`; `data/rq1/patch_words/rlens/` | weak; a figure |
| C6.2 | The R-lens names the content of an image token at layers 36 to 46 only. 95% of tokens are uninterpretable at layers 12 to 24 | [r-lens.md L11](r-lens.md#summary-of-results) | `rq/rq1_patch_layers.json` | moderate |
| C6.3 | Decay words by income on 65 images: 6 of 34 low-income images against 2 of 31 high-income images; Fisher p = 0.26, not significant | [runbook L2718](facct-runbook.md#run-status) | `data/rq1/seeds/all_patch_words.parquet` | weak |
| C6.4 | With the positions fixed, the question changes 80% of blocks at layer 42. A neutral filler alone changes 68%. Thus the question adds only about 12 points | [runbook L2718](facct-runbook.md#run-status) (D138, D141) | `data/rq1/patch_words/` | moderate |
| C6.5 | "homelessness" and "illegal" on the Norwood image appear only with the image first. The word family occurs in 3 models, but no layout, layer or rank holds across them | [runbook L2341](facct-runbook.md#the-j-replication-d146-2026-10-30) | `data/rq1/patch_words/` | weak; do not claim a stable read |
| C6.6 | The J++ lenses name the scene on Gemma-4-31B and Qwen3.5-9B. They share the R-lens top-1 token on 0.0% to 1.6% of blocks | [runbook L2341](facct-runbook.md#the-j-replication-d146-2026-10-30) | `data/rq1/patch_words/` | moderate; 3 models differ |
| C6.7 | Lens against text, logit lens, Gemma-4-31B, 200 answers: `upscale` rises at layers 39 to 58 in the rich group. 22% against 4% of the traces already say it | [runbook L2603](facct-runbook.md#results-of-f84-2026-10-29); [f8-lens-figure.md](f8-lens-figure.md) | `figures/f8/lens/lens_rates.png` | weak; adds little to the text |
| C6.8 | Reads of evaluative words on patches: 9 of 45 tests read, 0 of 90 area tests pass. `yummy` reads on the street case, thus a read can follow general valence | [runbook L3317](facct-runbook.md#f102-result-2026-10-30) | `figures/f10/reads/reads_validity.csv` | moderate as a negative |
| C6.9 | The coordinate swap: 2 of 16 cells succeed, both Qwen3.5-9B | [runbook L3487](facct-runbook.md#f103-result-2026-10-30) | `figures/f10/swap/` | moderate as a negative |
| C6.10 | Held-out test on 400 new images: removal of the name holds on Qwen3.5-9B and fails on Gemma-4-31B. Recovery holds: 0.60 held-out against 0.64 in F10.3 | [runbook L3617](facct-runbook.md#the-held-out-result-2026-10-31) | `figures/f10/heldout/heldout_primary.csv` | moderate; 1 model |
| C6.11 | The lens is a readout, not a site. Removal of the label subspace changes the judgment by 0.043 logits | [label-ablation.md](label-ablation.md); [steer-coord.md](steer-coord.md) | review section 4.2 | moderate |
| C6.12 | The R-lens does not predict the causal effect of an image token, and its directions do not steer | [r-lens.md L11](r-lens.md#summary-of-results); [causal-lens-retrospective.md](causal-lens-retrospective.md) | | strong as a negative |
| C6.13 | Latent place and class: C1.7 and C5.17. These are the 2 strongest lens results, and neither is about a specific object | [whatami-lens-study.md L184](whatami-lens-study.md#a-latent-borough-against-the-true-borough); [lens-refusal-probe-results.md](lens-refusal-probe-results.md) | see C1.7, C5.17 | moderate |
| C6.14 | Scene first. A ridge decoder with no lens reads the margin from the question-free mean patch: R2 0.820, 0.753. Layer 0 alone gives 0.601, 0.698 `*` | [f10-judgment-mechanism.md](f10-judgment-mechanism.md) | `data/f10/hop_free/*/decoder_grid1.csv`; `f10/free/patch_depth.csv` | strong. F10.26: holds in 39 of 42 prompt cells (R2 at least .50). Fails on Gemma `ugly` (20 and 83 answered images) and Qwen3.6-27B `scene ordinary` (.47). `f10/prompts/t2_grid.csv` |
| C6.15 | The question adds a step. Gain over the mean patch: `this` 0.043, 0.051; `place` 0.067, 0.078; `cool` 0.124, 0.178. Then flat until the last tokens (0.177, 0.238). 130 of 375 and 99 of 192 cells pass Holm `*` | the same | the same | strong; `mech/mech_gain.png`. The t is somewhat liberal. F10.26: the step at the adjective holds in 41 of 42 cells, and the gain at `?` in 41 of 42. The gain at the noun is .00 to .03 for `urban` and `fancy`. `f10/prompts/t2_counts.csv` |
| C6.16 | The patch read is not specific to `cool`. All adjectives read well, for example `beautiful` 0.88, `green` 0.87 on Gemma-4-31B. The patches hold a scene code, not a stored answer | the same | `f10/free/patch_words.csv` | strong |
| C6.17 | The sense of the word. Axis 1 at `cool` L24 holds 63% and 49% of NTA variance. It runs from temperature words to social words. A fixed-list score gives r 0.99 with the axis, r 0.10, -0.11 with the margin, r 0.70, 0.62 with the `urban` probe, and r -0.60, -0.61 with traces that name temperature. The score agrees across models at r 0.67 `*` | the same | `f10/sense/`; `f10/axes/axes_words.csv` | strong; `mech/mech_sense.png`. The lists share words with the axis. F10.26: a property of `cool`. The sense band holds in 6 of 6 `cool` cells and 0 of 36 other cells. For the other adjectives, axis 1 at the adjective follows the verdict (29 of 36 cells) and its ends name kinds of place, not senses. `f10/prompts/t3_sense_tests.csv` |
| C6.18 | Appraisal comes before the word `cool`. The aesthetic pair of sets gives r 0.755, 0.715 with the margin at `this` L32 and `place` L16. Status peaks later: 0.707, 0.646 at `?` `*` | the same | `f10/sets/sets_pairs.csv` | strong; pairs only, not single sets |
| C6.19 | Word lists predict held-out margins: 367 of 375 and 192 of 192 cells pass Holm. `this` L32 up: illustrious, delighted, romantic. Down: contempt, indictment `*` | the same | `f10/logit/` | strong |
| C6.20 | `rank_added` separates source. `place` and `?` words come from the image. `cool` and end-token words come from the prompt or the lens `*` | the same | `f10/stage/stage_words.csv` | moderate; Gemma-4-31B words. F10.26: the embedded conception at the adjective holds in 42 of 42 cells (at least 50 of the top 100 words). `f10/prompts/t2_embedded_by_layer.csv` |
| C6.21 | The read of a one-word prompt holds the words of a separate argument, beyond the borough: 194 of 375 and 73 of 192 cells pass Holm. The strongest are `thought`, `<\|channel>` and `<\|im_start\|>`. The link at `cool` and `?` is weak `*` | the same | `f10/link/link_cells.csv` | moderate; shared image, not a cause |
| C6.22 | Named pairs link: criterion edge with `p_grit` 0.74, 0.74; heritage with `p_old` 0.66, 0.73; greenery with `p_nature` 0.63, 0.69 `*` | the same | `f10/reason/reason_sets.csv` | moderate; largest r over cells |
| C6.23 | The stark `cool` words of the draft sit on axis 1. Bronx and Brooklyn words at L20 to L24 have a median axis rank of 43 to 69 of 53,999 at the social end (`boutique` 3, `fucking` 5, `prostit` 7). Staten Island and Queens words rank 53,837 to 53,998 at the temperature end. Without axis 1, Brooklyn t falls from at most 6.5 to at most 2.9 `*` | [hop_claim_axis.py](../scripts/f10/hop_claim_axis.py) | `f10/axes/axes_claim_words.csv` | strong for Gemma-4-31B; no Holm test after the removal |
| C6.24 | List stability over 40 splits, Gemma-4-31B then Qwen3.5-9B: Manhattan 0.88, 0.84; Queens 0.85, 0.79; Brooklyn 0.76, 0.71; Staten Island 0.76, 0.73; Bronx 0.22, 0.10 `*` | the same | `f10/sense/sense_splits.csv` | strong; `mech/mech_borough.png`. F10.26: stability over 40 splits is .59 to .78 in all 42 cells (mean over boroughs, at the adjective, mid depth). `f10/prompts/t8_stability.csv` |
| C6.25 | Held-out borough lists that pass Holm: Manhattan 344 of 375, 183 of 192; Queens 89, 23; the others 0. Stable lists: Manhattan `place` L36 dazzling, ballroom, gala; Staten Island `place` L36 carrots, canola, mulch `*` | the same | `f10/borough/` | moderate; 1 split. F10.26: the top-50 C3 words at the adjective are specific to the adjective (Jaccard .01 at most between adjectives, .09 to .21 between the 2 nouns of 1 adjective). `f10/prompts/t8_jaccard.csv` |
| C6.26 | Second-order RSA: split-half r 0.99, 0.99, but the scene and margin structure relate at rho 0.069, 0.047 only. At `cool`, the unique margin share of a cell is 0.022, 0.032 and the unique scene share is 0.219, 0.327 `*` | the same | `data/f10/hop_free/*/commonality.csv` | moderate |

Limits:

- **Claim as written:** "how models judge specific visual objects" still has weak support in the reads. The strong object result (C3.19) comes from traces. The strong read results are about the process. The heading at `now:142` ("how models reach aesthetic judgements") fits the evidence. Use it in the list at `now:10` too.
- The prompt is fixed, thus all variation between images comes from the image. A flexible decoder must reach R2 = 1 from the patches. C6.14 and C6.15 show how simple the map is. They show no causal order.
- 2 Bronx words stay after the removal of axis 1 in C6.23: `illegal` (L24, t 3.49, rank 42) and `vandalism` (L20, t 3.44, rank 148). We did not test them with Holm, and the Bronx list as a whole fails. Do not cite them without a test.
- C6.17 does not say that the temperature end is a choice in every image. It can also hold a low-signal image with no street content. The images at that end are trees and roadside green.
- The caption of the teaser says: "The lens names what a patch holds; it does not show which patches move the answer." This conflicts with the claim as written.
- [D135, runbook L206](facct-runbook.md#decisions-of-2026-10-28-the-f8-round) says that the lens is a figure and a smoke test, not the harm claim. F10.2 ends: the lens "does not give a validated read of evaluative words on patches".
- The comparison at `4_results.tex:108` has no result. No run measures the overlap of lens tokens with trace tokens. C6.1 shows them side by side on 8 images, and C6.7 is the nearest test.
- "Sequential process" (`4_results.tex:108`, now `now:191`) has only a figure (`rq1_patch_layers`). Below layer 30 that figure repeats word pieces. C6.15 now gives a tested order on 591 images. Onset of word sets does not separate stages (F10.13).
- The results supported by the records are about objects named in the image and about names in the prompt (C6.1, C6.10). The claim "how models judge" goes further.
- Comments in `4_results.tex:257` to `:261` give the token-ablation map and the layer flow to Paper 2. Do not move them back here without a check with the author.

## Draft checks

These lines of the draft do not match the data or the records.

| # | Line on 2026-10-31 | Line now | Draft says | The record says | Source |
|---|--------------------|----------|------------|-----------------|--------|
| 1 | `4_results.tex:23` | 35 | Colour is the rate of correct geolocation in traces or outputs | The fill is P(a caption names NYC). It reads captions only, with no trace, no hashtag and no borough name | `pool/f4_street_caption_nta_city.json` |
| 2 | `4_results.tex:29` | 40 | Names of "areas" that images are misidentified as | They are 42 cities and regions. The rate is 96 of 10,000 images, at the image level | `pool/f4_street_caption_nta_other_city.json` |
| 3 | `4_results.tex:18` | 18 | All 3 families, higher in whiter, educated, richer areas | Passes for all 3 families for `brand_knowledge` only, and only in restaurants. Qwen3.8-27B location is negative | C1.3 |
| 4 | `4_results.tex:54`, `:59` | 83 | Rates of 8% to 43%, with the figure as support | The figure gives spans for each 100 answers (11 to 51 on popularity). The text gives percent of answers. Both are correct, but the units differ | `rq/rq1_inference_kinds.json` |
| 5 | `4_results.tex:54` | 61 | "unsupported inferences" | No run tests this | review section 2.3 |
| 6 | `4_results.tex:75` | 105 | Bottom photographs show clear signage and outdoor seating | The model did not judge these photographs | C4 limits |
| 7 | `4_results.tex:75` | 105 | "older buildings (especially if they are brick)" | `older` costs. `brick` alone does not | C4 limits |
| 8 | `4_results.tex:89` | 124 | The figure shows a TrueSkill2 ordering | The area score is the mean ordinal score of `_provenance.score_units`. We found no TrueSkill step in the notebooks of this figure | [facct-pairwise-ses.md](facct-pairwise-ses.md) |
| 9 | `4_results.tex:94` | 130 | Greenery falls with % Black | The sign is correct, but 0 of 6 rows pass. % white is the row that passes | C5.12 |
| 10 | `4_results.tex:100` | 137 | "Inference valence" | The figure reads the valence of visual evidence and the `cool` readout, in the restaurant case. Inference valence is in `trace_ses` | `rq/rq2_method_b.json` |
| 11 | `4_results.tex:106` | 143 | Chunks that are "inferrable by the model" | The 16 blocks are 4 x 4 pools that we set (`--pool 4`). The model does not make them | [runbook L2718](facct-runbook.md#run-status) |

The draft at `846aec8` still holds all 11 lines. Line 61 now says "unreasonable inferences", and no run tests this word either.

**The J-space section.** These lines are new in `846aec8`. They use Gemma-4-31B unless a row says otherwise.

| # | Line now | Draft says | The record says | Source |
|---|----------|------------|-----------------|--------|
| 12 | 147 | All cited vectors are top 10 at the position and layer | True for the capped raw read. Softcap saturates, and raw lists favour high-variance words. Raw and t lists overlap only 4% to 7% | `f10/borough/borough_cells.csv` |
| 13 | 174 | `own` shows model egocentrism or couples place and image | No test. `own` sits on both ends of several axes (for example `?` L44 axis 1 low, axis 3 high). It looks like a high-variance word. We did not check the 68% in the TikZ note | `f10/axes/axes_words.csv` |
| 14 | 176 | `place` surfaces localization | Holds. `location` and `nearby` rank 3 and 4 by `rank_added`. `metropolitan` and `beachfront` do not: `rank_added` 18,390 and 48,576 | `f10/stage/stage_words.csv` |
| 15 | 178 | `cool` surfaces synonyms, then `vibe` | The words are real, but they rank the same with no image. They come from the prompt or the lens, not the image | the same |
| 16 | 180 | `?` surfaces evaluative words | Holds, and it is the best of this passage. All 5 depend on the image | the same |
| 17 | 182, 184 | `model` token shows values in the top 10 | `model` words rank 330 to 11,460 in the mean read. They rank 4 to 19 in the C2 contrast. Cite C2 for this list | the same |
| 18 | 187 | The last token shows uncertainty | `inconclusive`, `idk` and `cannot` rank about the same with no image (4 and 6, 8 and 14, 26 and 32). They come from the prompt or the lens | the same |
| 19 | 191 | Order: localization, standard, application, judgment | Onset does not support this order. Use C6.15 and C6.17: scene, sense, standard. The sentence also lacks a word ("we do not observe any in top-k") | C6.15 |
| 20 | 198 | Bronx `place` list, in Spanish and Portuguese; "no other borough" | 0 of 12 words hold on held-out NTAs, and none hold on Qwen3.5-9B. The Spanish and Portuguese set peaks in Staten Island and Queens (+1.25, +0.91 at `place` L8). The Bronx is +0.18 | `f10/borough/borough_claims.csv`, `borough_sets.csv` |
| 21 | 200 | Bronx `cool` list: `porn`, `gangs`, `illegal` at L20 to L24 | 0 of 13 hold. Raw rank about 6, t rank about 3,000. These words are not specific to the Bronx. See C6.23 | the same; C6.23 |
| 22 | 202 | "Alarmed" by the Bronx, then Brooklyn | The Brooklyn list holds (16 of 16), but it sits on axis 1. Delete "alarmed". The aim is the process | C6.23 |
| 23 | 205 | Staten Island and Queens show hydrology, and "no harmful content" | These are the temperature end of axis 1. The stark words are absent because the image took the other sense | C6.17, C6.23 |
| 24 | 207 | Manhattan `cool` list | Holds (16 of 16). Stability stays at 0.89 with axis 1 out. This is the only list that does not rest on the sense axis | C6.24 |
| 25 | 209 | Urban recognition gates aesthetic properties at L20 to L24 | Direction holds: r 0.70, 0.62 with `urban`. Replace "aesthetic properties" with "the social sense of the word". The older gate test was not specific (F10.10) | C6.17 |
| 26 | 113 | C3 ranks by the raw difference of means | Use the t on NTA means, with held-out NTAs. The unit has df 196 | C6.24 |

## Open analyses

The author listed these 14 items on 2026-10-31. The status comes from a search of the records and the data folders on 2026-10-31. The column "Update of 2026-11-01" gives the change from F10. A blank cell means no change.

| Status | Meaning |
|--------|---------|
| result exists | The numbers exist. The item needs a sentence, a table or a figure |
| part exists | A result answers a part of the item |
| designed | The design and the script exist, but no result exists |
| no result | The records hold nothing that answers the item |

**Note:** [runbook L3910](facct-runbook.md#open-questions) gives the FAccT paper date as 2026-11-03. The order below puts the work that needs no GPU and no new labels first.

| Order | Item | Claim | Status | What exists | What the item needs | Run type | Update of 2026-11-01 |
|-------|------|-------|--------|-------------|---------------------|----------|----------------------|
| 1 | Pairwise spans against H9 spans. Is the order about monotonic? | 4 | result exists | C4.6: rho 0.80 to 0.87 on 2 models, 4 cells ([f11-paradigm-consistency.md L99](f11-paradigm-consistency.md#32-the-order-of-the-cues)). Figure: `notebooks/facct/figures/f11/f11_cues.png` | A check that this figure shows what the author wants. The level does not hold (H11 fails in 11 of 12 cells) | none |  |
| 2 | Readouts change when the prompt goes before the image | 2, 6 | result exists | C6.4: the question changes 80% of blocks on Gemma-4-12B and Gemma-4-31B, 73% to 75% on Qwen3.5-9B. A filler changes 68% | **Caution:** any text changes the read, and the question adds about 12 points. Thus this shows the effect of text, not awareness of the task | none |  |
| 3 | Uncertainty by SES | 2, 5 | result exists | F4: 1 of 180 tests pass ([runbook L1801](facct-runbook.md#strata-valence-uncertainty-the-nta-and-the-cuisine-traces-notebook-280-sections-412-and-413-2026-10-28)). F9 in view: 20 of 60 restaurant, 0 of 60 street (C5.8) | A choice of which result to show. Neither is a figure yet | CPU, small |  |
| 4 | What the model says in uncertainty spans | 2 | part exists | C2.3, C2.4 give counts of the reasons. C2.5 gives "it depends on what you are looking for" | A read of the span text for hedges about taste. This is the direct test of the first limit of claim 2. Files: `data/f4_traces/subj/inference_valence/uncertainty_reasons.parquet`, and the F4 traces | CPU, read by hand | Result exists for `cool`: C2.12. Decision notes hold `subjective` and `no_taste`. F4 and F9 still need a read |
| 5 | Prescriptive tasks and the global ordering | 3, 5 | part exists | Disparity: C5.6 (H9) and C5.8 (net valence). Order: no result. [f11-paradigm-consistency.md L122](f11-paradigm-consistency.md#33-the-same-restaurant-u5-no-kl) compares F9 with pairwise, not F4 with F9 | Spearman of the NTA or image order, F4 against F9, for each model and case. F4 and F9 share the images, thus the pairs are direct | CPU, small |  |
| 6 | Uncertainty by type of social inference | 2 | no result | F9 social labels in `data/f4_traces/subj/inference_valence/full.parquet`. Uncertainty reasons by `doc_id` | A rate of uncertainty in texts that hold each social kind, with a chi-square test. It gives co-occurrence in a text, not a link from a span to a span | CPU, small |  |
| 7 | Transplant rate as an SES correlate | 5 | no result | The word is not in the records. This page reads it as residents who come from another US state. `data/f4_traces/nta_acs.parquet` has no such column, and the stored B05002 columns give only the total and the foreign-born count | A new Census pull: B05002 (born in another state) or B07003 (moved from another state). Then 1 more feature in the existing tests. **Caution:** a sixth feature changes the Bonferroni family | CPU; Census API |  |
| 8 | Small annotation sample | all | part exists | Parent extractor, 10 traces: precision 0.86, recall 0.26 (review section 2.2). Retest kappa 0.889 is consistency, not truth. The 100-trace audit never ran. No gold labels exist for F4 or F9 | Human labels. Sample by SES group, and compare precision with a chi-square test. **Caution:** the same precision across groups does not show the same recall, thus rates stay floors | human labels | 34 spans of the `cool` extraction read by hand, with few value errors. No gold labels |
| 9 | Aesthetic and judgment tokens in the J space | 2, 6 | part exists | C6.8: 9 of 45 reads, `yummy` fails the negative check. C6.7: `upscale`. Text-first blocks show `Good` and `gritty` ([runbook L2718](facct-runbook.md#run-status)) | A count of the words on patches over the stored reads. **Caution:** patch positions come before the question, thus they cannot hold a judgment for that prompt ([runbook L3691](facct-runbook.md#f104-the-pilot-of-the-hop-read-d158)) | CPU on stored reads | Result exists: C6.18, C6.19. Use pairs and t lists. The caution stays: patches come before the question |
| 10 | "Jumps" in the J space, and reads at answer positions | 6 | designed | F10.4 (D158), `scripts/f10/hop_read.py`. Only `data/f10/hops-smoke/` is on disk. The spider case: on Gemma-4-31B at L45, the text control reads `cardiac` and `Heart`, and `spider` ranks 2nd ([runbook L2341](facct-runbook.md#the-j-replication-d146-2026-10-30)) | The full pilot run: 2 models, 200 images. **Warning:** the design has no preset test. It can only suggest a hypothesis | GPU | Result exists: the store reads all tokens on 591 images (F10.6). C6.15 gives a preset test, and C6.20 gives the source of each word |
| 11 | Geo-location token rate, as 2 maps | 1, 6 | part exists | C1.7 for Gemma-4-12B on the whatami images. No lens read covers the 10,000 images of the caption maps | A lens read over NTA-tagged images, then the count of NYC and other-city tokens. It needs the same token rule as `flag_city_names` | GPU | Part replaced: C1.13 gives a direct place measure with no lens |
| 12 | "Task flow" by layer, across models | 6 | part exists | `rq/rq1_patch_layers.json` for 1 model. J++ reads on 8 images for 2 more models. [runbook L2341](facct-runbook.md#the-j-replication-d146-2026-10-30) finds that no layout, layer or rank holds across the models | A definition of a "capability" that you can count, before a run. **Caution:** the 3 models differ, thus the tables cannot separate the lens from the model | GPU; design first | Part exists for 2 models: C6.15 is a countable step on the same prompt |
| 13 | Multilingual J spaces | 6 | part exists | Stored notes: "multilingual nonsense tokens" in [causal-lens-retrospective.md](causal-lens-retrospective.md); Qwen reads `路灯` and `斑马` ([runbook L2341](facct-runbook.md#the-j-replication-d146-2026-10-30)) | A count of the non-English share of reads, if the point needs a number. The records hold no citation for the prior work | CPU; a search for the citation | Part exists: the Spanish and Portuguese set by borough (check 20). No citation yet |
| 14 | Separate aesthetics from harmful biases | 3, 5, discussion | no result | Material: `homelessness` on 1 of 65 images (C6.5), the refused judgment (C5.17), F9 wealth and upkeep labels (C3.13) | A definition from the authors of the 2 classes. After that, the F9 labels can give counts. This is a question of framing first, not a run | author decision | Part exists: C6.23 shows that the stark `cool` words are a shared axis, not a borough harm. The author decision stays |

One new item comes from F10:

| Item | Claim | Status | What the item needs | Run type |
|------|-------|--------|---------------------|----------|
| The cool margin and the probes against NTA SES on the 591 images | 5 | no result | NTA correlation with the 5 features, with df 196. The test inside a borough stays deferred. Keep the aim on process in the F10 text | CPU, small |

## Final pass: add, cut, strengthen

This is our recommendation for the last edit. We did not edit the paper.

### The paper's argument in 1 paragraph

In token order, the reads hold the scene first, then the sense of "cool", then a standard. Each part holds a contestable choice. The models often know this, and they answer anyway. The same object takes opposite signs under different values, and the values that win follow status. Thus aesthetic testimony carries a hierarchy that a user cannot see in the answer.

### Pare 6 claims to 5

| New claim | Merges | Claim sentence | Lead evidence | Figure or table |
|-----------|--------|----------------|---------------|-----------------|
| A. Place | Claim 1 | Models ground a judgment in a place, and the place is often wrong | C1.13, C1.15, C1.3 | Table of C1.13, plus 1 map |
| B. Awareness | Claim 2 | Models state that the judgment is a matter of taste, and they answer anyway | C2.11, C2.12, C2.13, C2.1 | 1 table |
| C. Standards | Claims 3 and 4 | One object takes opposite signs, because the value gives the sign | C3.19, C4.16, C4.13, C4.1, C3.11 | `mech_objects`, `probe_links` |
| D. Status | Claim 5 | The standards that win follow privileged areas | C5.1, C5.3, C5.4, C5.6, C5.19 | `pairwise_ses`, `rq2_method_b` |
| E. Process | Claim 6 | The reads show scene, sense and standard as separate parts | C6.15, C6.17, C6.18, C6.23 | `mech_gain`, `mech_sense` |

Claim 3 merges with claim 4 because its strongest new result (C3.19) is a result about standards. Its counts stay as 1 paragraph (C3.1, C3.3, C3.11, C3.12).

### Add

| Item | Where | Rows |
|------|-------|------|
| The `in New York` table | After `now:20` | C1.13 |
| The decision notes and the fall from trace to answer | The empty end of `now:51` | C2.12, C2.13. This closes the `\matt` note there |
| The object and value figure | After `now:105` | C3.19, C4.16. It replaces the graffiti anecdote as the proof |
| The value order of `cool` | After the figure above | C4.13, C4.14 |
| The gain figure and the sense figure | At the start of the J-space section | C6.15, C6.17 |
| Methods for F10 | `3_methods.tex`, after line 113 | Decoder, gain, probes, sense score, 40 splits |

### Cut

**Note:** The author rejected the held-out filter for borough lists on 2026-11-01. The 2 Bronx rows and the first half of the "Borough lists" row of "Strengthen" are withdrawn. Line numbers below are from `846aec8`.

| Item | Line now | Reason |
|------|----------|--------|
| ~~The Bronx `place` paragraph~~ | 198 | Withdrawn. The paragraph stays, in a shorter form |
| ~~The Bronx `cool` paragraph~~ | 200 | Withdrawn. The paragraph stays |
| "We, alarmed by" | 202 | Tone, and the words sit on a shared axis |
| The `own` hypothesis | 174 | No test |
| The `cool` synonyms sentence as an image effect | 178 | Same rank with no image |
| The end-token sentence | 187 | Same rank with no image |
| The word-by-word `model` list | 182 | Keep 1 sentence on C2, and cite it as C2 |
| The list of 21 chain-of-thought words | 184 | Use the t list or cut it |
| The TikZ prompt note | 151 to 167 | Replace it with `mech_gain` |
| C1.5, C1.6 as the proof of error | 20 | Keep 1 map as a picture. C1.13 is the proof |
| `laundromat` and McDonald's | 103, 105 | Weak rows. See the limits of claim 4 |

### Strengthen

| Weak point | Fix |
|------------|-----|
| "Not always correct" rests on quotes | C1.13 |
| "Awareness" rests on visibility, not taste | C2.12. Say that the `cool` prompts test taste, and F4 and F9 test visibility |
| "Used differently" rests on 2 graffiti quotes | C3.19 with Holm |
| Borough lists rest on raw top words | Keep only t lists that hold: Manhattan, Queens, and the Staten Island `place` list. Report 40-split stability |
| Layers 20 to 24 rest on a hypothesis | Restate it as the sense of the word (C6.17), with the traces as a separate check |
| The lens rests on a claim about objects | Claim the process (claim E) and give objects to the traces |
| A reader may read the R2 as causal | State the fixed-prompt limit once, in the methods |

### Discussion

| Item at `5_discussion.tex` | Evidence to add |
|----------------------------|-----------------|
| Less-entrenched standards | C6.17: the word itself is open, and the image settles it. C4.13: clean works against cool |
| Ontological collapse | C4.17: objects repeat more than values, so the words of the standard vary more than the objects |
| Models are hyperparameters | C2.11: Gemma-4-31B declines, and Qwen3.5-9B answers yes. C4.13: they disagree on `old` and `gritty` |

## Appendix pass of 2026-11-03

We did a pass over the appendix and the body on 2026-11-03 (paper commits `5087b33` to `905085c`). The table gives the status of the draft checks above.

| Check | Status |
|-------|--------|
| 4 (rates against spans) | Closed. The caption of `fig:model-inference-rates-by-type` gives the unit: spans for each 100 answers |
| 5 ("unsupported", "unreasonable") | Closed. The heading of 4.3 and the facade sentence do not use the words. Appendix C states the limit |
| 8 (TrueSkill2) | Closed. Appendix A and the caption of `fig:pairwise-ses` give the mean ordinal score of `score_units` |
| 10 ("inference valence") | Closed. The caption gives the valence of the visual evidence and the cool readout, with fixed effects |
| Greenery, "4 of 6 pairs" | Closed. The unit is 6 models in 1 case |
| 75.9% | Closed. The body gives the out-of-view share of the prescriptive restaurant answers |
| "53 other adjectives" | Closed. The list holds 52 words other than `cool` |

The appendix now has this order:

| Part | Contents |
|------|----------|
| A | TikTok search, images, models and prompts and samples and compute, J-space readouts, shared direction, neighborhood statistics, probes of place, span extraction and its validation, pairwise study and robustness |
| B | J-space, hierarchies and ACS features, inferences and uncertainty, pairwise results |
| C | Limitations |

Changes to tools:

| Tool | Change |
|------|--------|
| `notebooks/cvpr/_reliability.py` | New option `--cases`. It keeps only the named cases in the tables and the CSV files. The model names print as in the paper. The paper tables use `--cases subway_safety restaurants street_photography`. Run the script again without the option to write the 7-case files |
| `tables_export/build_tables.py` | The captions do not use run names (F4, F9). The caption of `h9-summary` does not say "with controls" |

Open items for the author (the paper shows a red TODO):

- The search terms and the models of the TikTok search
- The IRB status, and the blurring of faces and plates in the imagery
- The generative AI statement
- The link to the anonymized repository

## Related

- [facct-runbook.md](facct-runbook.md), [facct-rq-evidence.md](facct-rq-evidence.md), [paper-repo-sync.md](paper-repo-sync.md)
- [f10-judgment-mechanism.md](f10-judgment-mechanism.md)
- [f8-method-b.md](f8-method-b.md), [f8-counterfactual-cue.md](f8-counterfactual-cue.md), [f9-subjective-prompt.md](f9-subjective-prompt.md), [f9-inference-valence.md](f9-inference-valence.md), [f11-paradigm-consistency.md](f11-paradigm-consistency.md)
- [facct-pairwise-ses.md](facct-pairwise-ses.md), [cue-hierarchy.md](cue-hierarchy.md), [trace-channels.md](trace-channels.md), [r-lens.md](r-lens.md)
