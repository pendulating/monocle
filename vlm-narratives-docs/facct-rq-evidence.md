# FAccT evidence review: RQ1 to RQ3

Written 2026-10-29 by a research-advisor pass over `vlm-narratives-docs/`.
It reads every experiment record and runbook, and sorts the results under
the 3 research questions of the FAccT paper. It is a review. It adds no
new experiment. The F8 results were checked by an earlier advisor
(F8.1 solid, F8.2 usable with caveats, F8.3 solid, F8.4 a figure only).
This page does not check F8 again in depth.

| RQ | Question |
|----|----------|
| RQ1 | Do VLMs make aesthetic inferences that specific observable cues in the image do not back up? |
| RQ2 | Do the interpretations of the pixels (what the model says it sees, and how it values it) correlate with the socioeconomic features of the neighbourhood? |
| RQ3 | To what extent are the ungrounded inferences and the biases unavoidable: in the model, and not a product of the prompt, the sampling, the image content or our measurement? |

The rule of this review: a result that the image content, the prompt
wording, the extractor or the sampling can explain is weak evidence for
RQ3. The strength column uses 3 values: strong, moderate, weak.

## 1. What this review read

| Group | Docs | Use |
|-------|------|-----|
| Main tracker | `facct-runbook.md` (D1 to D135, F1 to F8) | all RQs |
| F8 records | `f8-method-b.md`, `f8-counterfactual-cue.md`, `f8-lens-figure.md`, `caption-method-review-2026-10-28.md` | RQ2, RQ3 |
| F4 and F6 plans | `f4-batch-2-reproduction-plan.md` (part C1, the batch effect), `f6-restaurant-image-selection.md` | RQ3 (sampling) |
| The pairwise audit (the parent) | `restaurant-proxies.md`, `proxy-overview.md`, `master-results-table.md`, `proxy-arms.md`, `proxy-audit-fixes.md`, `quantile-shift.md`, `unit-validation.md`, `building-condition-proxies.md`, `road-proxies.md`, `parks-plazas-proxies.md`, `subway-station-proxy.md`, `pole-projection-proxy.md`, `vintage-proxies.md`, `robustness-battery.md`, `robustness-battery-rerun-plan.md`, `positional-bias-and-reliability.md`, `camera-face-effect.md`, `modality-order.md`, `do-the-colours-mean-anything.md`, `looks-like-rerun.md`, `aesthetic-hierarchy.md`, `prompt-ablations.md`, `canonical-run-registry.md`, `million-pair-battery.md`, `paper-plan-cvpr-audit.md` | RQ1, RQ2 (income only), RQ3 |
| The trace extraction | `reasoning-trace-analysis.md`, `trace-channels.md`, `cue-hierarchy.md`, `judge-vocabulary.md`, `langextract-trace-extraction.md`, `ic-ingredient-extraction.md`, `ic-metadata-findings.md`, `ic-v3-analysis.md`, `terminology.md`, `thinking-100k.md` | RQ1, RQ3 (measurement) |
| The white-box work (paused, D1) | `lens-refusal-probe-plan.md`, `lens-refusal-probe-results.md`, `whatami-probe.md`, `whatami-lens-study.md`, `single-image-steering.md`, `single-image-lens-path.md`, `steering-weak-alpha.md`, `where-the-judgment-lives.md`, `label-ablation.md`, `answer-token-readout.md`, `lens-corpus-faithfulness.md`, `jacobian-lens-review.md`, `jlens-47-layer-fit.md`, `workspace-depth-47.md`, `shared-negative-evidence.md`, `causal-lens-retrospective.md`, `steer-coord.md`, `r-lens.md`, `patch-attribution.md`, `causal-patch-reader-tools.md`, `lens-registry.md`, `canonical-lens-prompt-path.md`, `sae-plan.md`, `paper-plan-interp-posttraining.md`, `white-box-related-work.md` | RQ3 (support only) |

Skipped as infrastructure, with no result for the RQs:
`blackwell-sm120-flashinfer.md`, `gemma4-12b-architecture.md`,
`gemma-4-best-practices.md`, `gpu-census-tool.md`, `gpu-partition-census.md`,
`nfs-load-pairvqa-sweeps.md`, `review-refactor-2026-08.md`,
`scratch-mirrors.md`, `trashbot-annotator.md`, `vllm-025-upgrade.md`.
`vault/context/` was not read (stale).

Two facts frame every table below:

- The pairwise audit (the parent) reads income only. No parent doc reads
  % white, % Black, % Asian or % bachelor's. The race and education
  results all come from F4 and F8 (2026-10-25 to 2026-10-29).
- F4 and F8 use 1 image, 6 models and the x2 extraction (D131). The parent
  uses 2 images, 2 models and the version-2 extraction. The runbook says:
  do not compare an F4 rate with a parent rate.

## 2. RQ1: ungrounded aesthetic inferences

### 2.1 Evidence table

| # | Finding | Source | Strength | Main threat |
|---|---------|--------|----------|-------------|
| 1.1 | In the restaurant answers, an `inference` span sits in 99% to 100% of the texts, and the models write 4 to 8 inference spans for each visual_evidence span group: `location` 29 to 125 spans and `brand_knowledge` 30 to 88 spans for each 100 answers, from 1 storefront photo | runbook, "The model as a hyperparameter"; `notebooks/facct/figures/f4_x2_pooled/f4_response_inference_kind.csv` (checked) | moderate | the extractor sets the class border; a location or brand inference from a sign is grounded in the sign |
| 1.2 | `popularity` inferences ("a neighborhood favorite") appear in 11 to 51 of each 100 restaurant answers (llama-3.2-11b 51, gemma-4-31b 11). A still photo of a storefront shows no popularity | the same CSV (checked) | moderate | a crowd or a queue in the photo can ground it; the F4 schema does not tie an inference to a cue |
| 1.3 | Social inferences (wealth, class, demographic, crime) are rare in single-image answers: 4.3% to 10.4% of the restaurant answers hold 1 or more; wealth 1.8% to 5.9%; crime 0.0%; street answers under 1% | runbook, "The model as a hyperparameter"; `f4_model_differences.csv`, rows `inference: any of the 4` (checked) | strong as a count | this is a count of what the extractor calls a social inference under `single-v2`; a "trendy", "upscale" descriptor is not in these 4 kinds |
| 1.4 | The trace holds social and person claims that the final answer drops: a person in the photograph is in 26% to 54% of the caption traces and in 0.5% to 13% of the caption answers (trace only: 26% to 42%); wealth, class, demographic or crime: 5% to 14% of the restaurant traces, 5% to 11% of the answers | runbook 4.7; `f4_trace_against_answer.csv` (checked; the runbook gives the canonical values 24 to 53, 1 to 13, 23 to 41) | moderate | the trace and the answer are 2 extractions; a gap can be a miss in 1 of them. The runbook says: read the quotes first |
| 1.5 | `uncertainty` spans mark that the model cannot read a cue, and the rate differs 11x by model: 7.1% (llama-3.2-11b) to 79.8% (qwen3.8-27b) of the restaurant answers | `f4_model_differences.csv`, row `uncertainty: any` (checked) | moderate | the rate measures the writing style of the model, not its accuracy |
| 1.6 | The exclusive-word test: 486 descriptor words pass (Bonferroni binomial) as words that only 1 model writes on the same images; 419 of them are qwen3.8-27b words ("mediocre" in 40 traces, "handsome", "weakest") | runbook, "Words that only 1 model uses"; `figures/f4_descriptors_x2_pooled/exclusive_heads_*.csv` (checked: 50, 172, 103, 161 pass; qwen3.8-27b 419) | moderate for RQ3, weak for RQ1 | an exclusive word shows a model vocabulary, not an ungrounded claim; qwen3.8-27b writes the longest texts |
| 1.7 | The parent audit: the raters prefer a library with more building violations (unit rho -0.37 gemma, -0.48 qwen, n 228), a bigger park (gemma +0.285), a busier station as safer (tau +0.45 and +0.57), and a restaurant against its inspection record (-0.10 after the reliability correction) | `building-condition-proxies.md`, `parks-plazas-proxies.md`, `subway-station-proxy.md`, `restaurant-proxies.md`; the unit parquet files (checked) | moderate | each odd association has a visible cue next to it: scaffolding, size, crowds, outdoor seating. The docs call them confounders, not inferences |
| 1.8 | The north camera face wins in 11 of 13 cells (p 4.4e-5): the verdict moves with the light, a grounded cue with no relation to the question | `camera-face-effect.md`; `face_effect_grid_v1.0.0.csv` (checked) | strong as a measurement fact | the swing is 0.08 ordinal units at most and cancels in the unit mean |
| 1.9 | Parent traces (gemma-4-12b, 76,980 traces, 2 images): a `wealth` inference pays for the image in every case, mean lift +0.23 to +0.44 across 4 arms, 2 extractor families and 2 schema versions; 96.7% of the pole-readable wealth inferences run up ("the model has almost no vocabulary for poverty") | `trace-channels.md` section 3, `ic-metadata-findings.md` F3; `trace_channels_inference_summary.csv` (checked on the v3 file: wealth +0.231, 6 pays, 0 costs) | moderate | an in-trace win association, not a bias against a place; the model narrates while it decides, so the span can follow the verdict |
| 1.10 | About 27.5% of the parent gemma traces hold a social inference (wealth 11.6%, class 20.5%, demographic 2.3%, crime 1.0%); by case: schools 70%, restaurants 46%, road quality 0.9% | recount of `data/trace_extractions_v2_gemma12b_by_qwen27b/` by the trace reader | moderate as a lower bound | the extractor recall on `inference` is 0.26 (27B) to 0.07 (31B) against 10 human-read traces; the number is a floor |
| 1.11 | The model judges by cues the extractor calls neutral: 28 neutral cues move the judgment by more than 1 logit (plain -3.65, small window -3.07, chain-link fence -2.20, large arched window +2.24) | `cue-hierarchy.md`, "The model judges by words it calls neutral" | moderate | `neutral` is the extractor's label; the file on disk no longer agrees with the doc counts (section 7, M8) |
| 1.12 | The `crime` inferences (454 of 302,784 inference spans, 440 in subway safety) all name a cue in the quoted excerpts: bars, graffiti, fences, a police van | `trace-channels.md`, "Where a rare kind goes" | weak | n is small; 1 case; this is counter-evidence for "ungrounded" |
| 1.13 | 4 image statistics explain 4% to 26% of the single-image judgment, and 54% for street photography: dark, edge-rich images are "beautiful" | `steer-coord.md` | weak | gemma only; a description of the pixels, not a test of an inference |

### 2.2 Counter-evidence

- The one clean "model against ground truth" null: `school_kept_clean`
  (a teacher survey) gives rho +0.019, p 0.52 (`building-condition-proxies.md`).
  The survey is from 2017-18.
- The strongest unit agreement in the parent audit is grounded: station
  crime for each rider, rho +0.21 and +0.29 (`subway-station-proxy.md`).
  Plaza violations agree with both raters (+0.16, +0.14).
- "Ungrounded" is not a class of the F4 schema. The schema has
  `visual_evidence`, `inference`, `uncertainty`, `person_reference`,
  `image_artifact` (and `photo_advice` on street). No run ties an inference
  span to the cue that supports it, or marks an inference as unsupported.
  The counts in 1.1 to 1.4 show that the models infer. They do not show
  that a given inference has no cue.
- The class rates moved 2x to 10x when the extractor changed (parent, `trace-channels.md`;
  runbook, "The F4 span extraction"). Inside F4, the round-1 to x2 change
  moved the span count of 1 (model, field) by 2x (qwen3.5-9b street
  traces, 27 to 72 cue spans in the smoke). The x2 tables are canonical
  (D131); compare F4 rows with F4 rows only.
- The caption-case `other` bucket of `inference.kind` held about half of
  the inference spans under `single-v1` (24,865 of 50,790). The
  descriptor run (D109) exists because the kind list was too coarse.
- The grounded/ungrounded line is the line the extractor draws worst.
  Against 10 human-read traces (540 human spans), the Qwen3.8-27B
  extractor has `inference` precision 0.86 and recall 0.26 (F1 0.40); the
  Gemma-4-31B extractor has precision 1.00 and recall 0.07. The main
  confusion is a human `inference` read as `visual_evidence` (14 to 33
  spans for each arm). No second annotator exists, and the planned
  100-trace audit never ran (`langextract-trace-extraction.md`, the gold
  set; `gold_alignment_positional_by_extractor_v1.2.0.csv`, checked by the
  trace reader).
- Only 23% of the inference judge contrasts survive an extractor swap
  (Spearman 0.69 between extractors; `judge-vocabulary.md`, checked).
- In the parent traces the valence of a described thing follows the vote:
  when 1 facility appears in a win and a loss with the same dimension and a
  different valence, the good valence sits in the win 0.988 (598 of 605;
  `ic-metadata-findings.md`, checked). The description is post-hoc
  narration, or the extractor reads the vote; either way a span is not an
  independent read of the pixels.
- The `class` lift changed sign between the v2 doc and the v3 file on disk
  (section 7, M9).

### 2.3 Verdict

The paper can claim: "From 1 storefront photo, every model writes
inferences about the business, the location and the character of the
area. Popularity and atmosphere claims appear in 11 to 51 and 47 to 119 of
each 100 answers." It can also claim that the thinking trace holds more
claims about people and about wealth than the final answer. It cannot
claim "ungrounded" in the strict sense: no experiment ties an inference
to a missing cue. The counterfactual cue (F8.2) is the only design that
holds the pixels fixed, and it tests a prompt cue, not a pixel cue. The
social inferences (wealth, class, demographic, crime) are rare (under
11% of answers), so a harm claim built on their count is thin. The
parent traces give the best RQ1 line: "a wealth inference pays for the
image in every case and almost never runs down", and this holds across
2 extractors and 2 schemas. But every inference rate is a floor
(extractor recall 0.26), and the grounded/ungrounded border is the one
the extractor draws worst.

## 3. RQ2: the interpretations correlate with the neighbourhood

### 3.1 Evidence table

| # | Finding | Source | Strength | Main threat |
|---|---------|--------|----------|-------------|
| 2.1 | The NTA net valence of the `visual_evidence` spans goes with the resident mix: gemma-4-12b rho +0.74 with % white and +0.72 with % bachelor's; qwen3.8-27b +0.67 with % white and +0.64 with income (restaurant answers, 159 NTAs) | runbook 4.12; `figures/f8/method_b/valence_meta.json` (checked: 0.744, 0.717, 0.666) | strong as a correlation | the storefronts and the cuisines differ by NTA; 96% of the spans are neutral; the NTA is an ecological unit |
| 2.2 | The same effect at the text level with the image-content controls (cuisine, DOHMH grade, borough; SE clustered by NTA): 37 of 40 restaurant effects stay significant, median share kept 0.89; size 0.04 to 0.18 SD of the outcome for each SD of the feature. % Asian is never significant | `f8-method-b.md`; `figures/f8/method_b/valence_cases.csv` (checked: 40, 37, 0.89) | strong | the controls are coarse and can be mediators; the street case gives 9 of 50 before and 4 after the controls |
| 2.3 | The R3 readout "This place is cool / lame" after the stored answer: the NTA mean of `c_cool_i` has rho +0.45 to +0.74 with % white and +0.40 to +0.80 with % bachelor's in all 4 thinking models (restaurants); 71 of 160 NTA tests significant | `f8-counterfactual-cue.md`, F8.3; `figures/f8/cue/r3_nta_spearman.csv` (checked) | strong as a correlation | the readout reads 1 token after a sampled answer; the same pixel confound as 2.1 |
| 2.4 | With the F8.1 controls, all 16 restaurant rows of `c_cool_i` (4 models x % white, % Black, % bachelor's, income) stay significant; 0.07 to 0.32 SD of the readout for each SD of the feature | the same; `figures/f8/method_b_r3/c_cool_i_cases.csv` (checked: 16 of 16 restaurant; street 9 then 8) | strong | as 2.2 |
| 2.5 | The cue rates by NTA: 408 of 7,620 Spearman tests significant after Bonferroni (% bachelor's 176, % white 115, income 94, % Black 15, % Asian 8). In the 193 significant rows, the sign of rho is the same for every model in 157; no row has 2 models significant with opposite signs. Example: signage evidence against % bachelor's, rho -0.28 (qwen3.8-27b, not significant) to -0.68 (gemma-4-12b) | runbook 4.10; `f4_nta_feature_correlations.csv`, `f4_nta_feature_signs.csv` (checked) | moderate | the test counts "what the model mentions", not a value judgment; image content differs by NTA; strict Bonferroni leaves most rows null |
| 2.6 | The 4.10 rows with the largest rho are value rows: `visual_evidence: valence = good` against % bachelor's +0.70 and % white +0.68 (gemma-4-12b); `inference: kind = atmosphere` against % white +0.67 (gemma-4-31b) | `f4_nta_feature_correlations.csv` (checked) | moderate | as 2.5 |
| 2.7 | The caption diversity (rarefied entropy) correlates with the NTA mix in 9 of 70 tests: hashtags with % Asian +0.28 to +0.36 (3 models and the pool); qwen3.5-9b caption words with % bachelor's -0.42. On the street captions, 3 of 70; only gemma-4-31b hashtags with % Asian holds in both cases (+0.36, +0.35) | runbook, "The caption notebook"; recomputed from `data/f4_traces/pooled/captions/nta_analysis/nta_measures.csv` and `nta_acs.parquet` (checked: 9 of 70, same rho) | weak | the measure is lexical diversity, not meaning; the restaurant-case entropy still tracks the NTA size (rho -0.46) and the answer format (rho -0.66); the hashtags name the place (#chinatown, #bronxeats) |
| 2.8 | The parent audit: the area-layer income association is positive in every case for both raters (tract r 0.39 to 0.47 for the restaurants battery; CD tau +0.42 and +0.37 for subway) | `robustness-battery.md`, `subway-station-proxy.md`, `proxy-overview.md` (checked by the parent reader) | moderate | shown at the area layer only; the median tau rises with the layer (tract 0.08, NTA 0.15, CD 0.19), so the aggregation inflates it; no unit-level income row exists |
| 2.9 | The popularity inferences in the 5 richest NTAs: 30.2% of 2,731 restaurant answers; in the 5 poorest: 20.3% of 1,693 | runbook 4.10.1 | weak | a description from 10 NTAs, not a test |
| 2.10 | White-box (paused): gemma-4-12b-it refuses "rich or poor" on 62% to 72% of images, and a logit-lens readout on the refused images still separates rich from poor NTAs at AUC 0.73 [0.68, 0.79] (density-matched 0.75); the base model states the same judgment at AUC 0.72 on the same images | `lens-refusal-probe-results.md`, "Experiment 2 in 5 lines"; `probe_nta/group_test/class.csv`, `probe_base/base_on_refused.md` (checked by the white-box reader) | moderate | 1 model family, 1 prompt family; a lens readout is a decodable quantity, not a stated output; the control scene words also track the census at 0.4 to 0.6 over 197 NTAs |
| 2.11 | Under an open prompt that names no word, the output words `ghetto`, `poor`, `poverty`, `graffiti` track NTA income at AUC 0.69 to 0.73 (extreme groups), replicated on 500 new images for each NTA (0.74 against 0.73) | the same, `probe_words_all/*/dose_test/dose.csv` | moderate | over all 197 NTAs the `ghetto` association is rho_partial 0.45 but raw rho 0.07 (checked; section 7, M10); a poor block looks different, and a model that names what it sees correlates |
| 2.12 | Parent traces: a trace with a wealth inference agrees with district median income more (+0.17 [0.11, 0.24], 204 pairs, gemma) and with the facility record less (-0.08); +0.08 to +0.17 on 4 arms | `ic-metadata-findings.md` section 4; `reasoning_vs_proxy_*_contrasts.csv` (checked) | weak | 204 pairs; a community district is coarse; 5 to 12 of 38 to 67 cells clear 0; agreement with income is not a bias measure |

### 3.2 Counter-evidence

- The street case is near null for value: 9 of 50 method-B rows before
  the controls, 4 after; the borough removes some. The R3 readout gives 15
  of 80 street NTA tests. The F4 street views are random recordings in
  residential NTAs (D116), and the prompt asks for a camera angle; the
  answers hold social inferences in under 1% of texts.
- The NTA rates of a cue category are noisy: the split-half `within`
  reliability has a median of 0.06 to 0.20; only 62 of 316 categories
  reach 0.3 (runbook 4.10, `f4_nta_rank_agreement.csv`, checked).
- The cuisine and the NTA mix go together: 64 of 135 cuisine shares
  correlate with an ACS feature (Caribbean with % Black +0.71). The 4.12
  chi-square tests cannot separate them; only method B tries, with coarse
  controls.
- The caption review (`caption-method-review-2026-10-28.md`) rates the
  entropy result as a homogeneity result, not a semantic result, and ranks
  the image content as the second threat to validity.
- In the parent, the unit-layer and the area-layer signs differ:
  restaurants unit tau -0.04 against CD +0.10; roads gemma unit +0.06
  against CD -0.11 (`unit-validation.md`). The area rows win by variance
  alone (proxy CV over areas 0.02 to 0.07, against 0.4 to 0.95 for SES).
- % Asian is never significant in method B, and only 8 of 408 cue-rate
  tests name it. % Black gives 15.
- White-box race: no residual-stream race judgment survives the controls
  in the instruction-tuned model (race AUC 0.46, migration 0.52). The
  Experiment 1 race result was the class readout under colour tokens. The
  base-model `Black` association (0.34 to 0.47 after the luminance and
  income controls) is the strongest race fact, and it is in the base
  model only (`lens-refusal-probe-results.md`, checked).
- Parent, pair level: the road-quality judgment on 1 pair agrees with
  median income at chance (0.543, about 9,000 pairs); the area correlation
  is an aggregate (`ic-ingredient-extraction.md`).

### 3.3 Verdict

The paper can claim, in these words: "In the restaurant case, the
valence of what the model says it sees, and its answer to 'is this a cool
place', shift with the share of white residents and of residents with a
bachelor's degree in the area, in all 6 models, and the cuisine, the
grade and the borough remove only about 10% of the shift." The sizes are
small (0.04 to 0.32 SD for each SD of the feature), and the paper must
say so. It cannot claim that the model treats people differently (the
NTA is an ecological unit), and it cannot claim the same for street
photography, where the effect is near null. The caption entropy result
should be reported as homogeneity, not as bias. The paused white-box
work adds 1 sentence for the paper: the wealth judgment that the model
refuses to state is still decodable on the refused images (AUC 0.73),
and the base model states it (0.72); no race judgment survives the
controls in the instruction-tuned model.

## 4. RQ3: how much is unavoidable

### 4.1 Evidence table

| # | Finding | Source | Strength | Main threat |
|---|---------|--------|----------|-------------|
| 3.1 | At fixed pixels, a neighbourhood name in the prompt changes the answer and the readout: 16 of 32 paired t tests significant; gemma-4-12b restaurant `c_cool_i` +0.82 (rich minus poor), gemma-4-31b -1.39, gemma-4-31b street -5.27 (rich higher on only 30% of the images) | `f8-counterfactual-cue.md`; `figures/f8/cue/cue_primary.csv` (checked) | strong for "the name matters"; weak for "poor is worse" | the direction differs by model and by case; the rich name is also the whiter NTA in 9 of 10 pairs; any name moves the readout from the no-cue baseline by more than the rich-poor gap in some cells |
| 3.2 | With a rich name, all 4 models write more `upscale` words on the same images (16.0% against 10.1%, gemma-4-12b restaurant answers); 11 of 64 McNemar tests | the same; `cue_words.csv` | moderate | about one third of the hits describe the named area, not the image; "gritty" and "rough" in the `danger` list are photography words |
| 3.3 | The valence and the R3 effects of RQ2 hold in all 4 thinking models and in both Llama models (valence), with the same sign; the models agree on the readout over the same images (Pearson 0.50 to 0.74 for `c_cool_i`) | `f8-method-b.md`, `f8-counterfactual-cue.md`; `r3_model_pearson.csv` | moderate | 6 models from 3 families, all post-trained on similar data; agreement can be shared training, not an image truth |
| 3.4 | The parent income association survives every elicitation change measured: paraphrase (tract r 0.39 and 0.39 for gemma; 0.41 and 0.35 for qwen), persona (0.47 and 0.39; 0.22 and 0.35), the order halves (0.421 and 0.423), no city name, no abstention, a batch retest, 2 raters | `robustness-battery.md`, `prompt-ablations.md`, `proxy-arms.md`; `robustness_estimand_v1.0.0.csv`, `ablations_v1.0.0.csv` (checked by the parent reader) | strong at the area layer | the area layer inflates every correlation; 1 case in Tier B; the persona moves the 2 raters in opposite directions |
| 3.5 | The abstention hides no judgment: forced labels on held pairs lean 0 (gemma 94.8% Same on schools), score delta 0.000 on all 14 rows; `noabstain` compresses the tails and does not reorder | `prompt-ablations.md`, `quantile-shift.md`; `ablations_v1.0.0.csv` (checked) | strong | gemma rows re-ran on 2026-09-04; the quantile-shift gemma cells in the doc are stale (see section 7) |
| 3.6 | The single-image logit gap `c` tracks the same proxies as the pairwise rating (roads 0.23 against 0.09; libraries -0.22 against -0.37; schools 0.18 against 0.17); the block-42 lens read takes the sign of the pairwise cell in 22 of 27 rows | `pole-projection-proxy.md`, `proxy-overview.md`; `proxy_correlations.parquet` (checked) | moderate | gemma only; 3 images for each unit |
| 3.7 | The model is a hyperparameter: 291 of 316 chi-square tests on "the text holds the category" differ by model after Bonferroni; 150 categories differ by 10 pp or more; the largest range is 86.6 pp (street `image_artifact`, llama-3.2-11b 7.7% against qwen3.8-27b 94.3%) | runbook 4.8; `f4_model_differences.csv` (checked: 316, 291, 268, 187, 150) | strong | the unit is 1 completion; the texts differ in length (the secondary length-band check keeps the result: 26 to 45 significant categories in each band with 4 models or more) |
| 3.8 | The direction of the NTA associations does not depend on the model: 157 of 193 significant rows have the same sign of rho in all models, and 0 rows have 2 significant models with opposite signs | runbook 4.10; `f4_nta_feature_signs.csv` (checked) | moderate | only 16 rows are significant in every model; "not significant" in 1 model is not a difference |
| 3.9 | The batch-effect check: 1 of 115 cue rates differs between the canonical batch and batch 2 after Bonferroni (caption, qwen3.5-9b traces, `image_artifact` +12.7%); the large relative changes come from a few long texts | `f4-batch-2-reproduction-plan.md`, C1 | strong as a sampling check | the 2 restaurant draws have different designs |
| 3.10 | The F8.1 effect does not depend on the NTA floor: with NTAs of 20 images or more, 39 then 38 restaurant rows, median share 0.91 | `f8-method-b.md`, "Check" | moderate | as 2.2 |
| 3.11 | One standard across prompts: the cross-case Spearman of the `category_valence` lift is 0.91 (0.76 to 0.94) over 7 pairwise prompts; 16 of 22 keys hold 1 sign in every case; `visual_evidence` is the class that holds across extractors (1.05x) | `cue-hierarchy.md`; `cue_hierarchy_agreement.csv` (checked: 0.912, 0.764, 0.944) | moderate | the 7 prompts share a scaffold; an in-trace win association; street photography inverts the upkeep standard (graffiti +0.83 there, -1.71 to -4.33 elsewhere) |
| 3.12 | The wealth judgment precedes post-training: the base model forms it at L30 to L36 and states it; the IT model forms it by L30 and the abstention overrides it at L42; the final MLP cancels the affluence words while a probe on h47 keeps AUC 0.89. Post-training masks the readout, it does not remove the information | `lens-refusal-probe-results.md`; `paper-plan-interp-posttraining.md`; `parts_test.md` (checked: 0.891) | moderate | 1 model family; under a completion-style prompt the IT model shows NO association at any layer (class 0.50 to 0.59): the prompt template carries the judgment |
| 3.13 | Activation addition of the lens direction flips all 7 pairwise cases (restaurants 60 of 60 to `gross` at alpha -1, L42; control 0) | `single-image-steering.md`; `coverage.json` (checked) | weak | "the vector can elicit the token", not "the model held the judgment"; the direction is orthogonal to the principal judgment direction (cos 0.001 to 0.019) and carries about 1% of the variance |
| 3.14 | The abstention count in a trace separates NotSure from a decision at AUC 0.60 to 0.93 in all 4 arms; the AUC moves at most 0.08 under an extractor swap while the raw count moves 2x | `trace-channels.md` section 2; `trace_channels_abstention_arms.csv` (checked: subway 0.930) | moderate | 1,000 traces for each case; street photography has 0 abstentions |

### 4.2 Counter-evidence

- The counterfactual cue shows that a 2-word prompt cue moves the
  judgment by more than the pixel-correlated effect: gemma-4-31b restaurant
  `c_yes_i` is -0.92 with no cue, +4.18 with the poor name and +4.61 with
  the rich name. The model answers the prompt as much as the image. This
  cuts both ways: it is prompt sensitivity (not unavoidable), and it is a
  belief about a named area at fixed pixels (not the image).
- The direction of the cue effect flips between models (gemma-4-12b up,
  gemma-4-31b down in restaurants) and between cases (a poor name makes a
  street "cooler" for both gemma models). Per pair, the sign also flips
  (Park Slope against Brownsville +2.65; Douglaston against Jamaica
  -1.09). A single "poor is worse" prior is not in the data.
- The text-first layout on qwen reverses the subway income association
  (tau +0.37 to -0.42, p 5e-6) and halves the spread (`proxy-arms.md`,
  checked). The layout alone moves gemma abstention by 10 to 17 points
  (`prompt-ablations.md`). Positional bias is large (qwen restaurants
  pick the first image 99% of the time; sign agreement under a swap 4%).
  All cancel in the unit means, by the docs' own accounting, but the
  single verdict is not a measurement.
- The nocity headline of the parent ("the city name carries real
  information", 17 of 28 cells) is stale: the current CSV gives a median
  d_tau of -0.006 (section 7). The city name does not move the ranking.
- The extractor is a large part of the measurement: the class rates moved
  2x to 10x between extractors; the round-1 extractor copied example spans
  into 161 texts; 131 texts (1%) remain copy suspects after x2 and are
  dropped. A rate difference between models can be a difference in how
  well the extractor reads each model's style (the Llama answers are
  short; the qwen traces are long and loop).
- The R3 verbatim readouts compare improbable tokens (`Yes` at rank 12 to
  47; `_lame` at rank 20,000 to 80,000). Only the instructed readouts are
  usable, and they were added after the smoke test (before any full
  result).
- The lens (F8.4) adds little to what the text says: `upscale` is in the
  top 5 of layers 39 to 58 at 2.14 against 0.08 for each 1,000 positions,
  and 22% against 4% of the texts already say an `upscale` word. The
  control cities also rise in the rich group. The groups hold 2 NTAs each.
- The parent abstention rate tracks the proxies (r up to 0.58, libraries
  income), but weighting moves tau by under 0.04 on 25 of 30 rows
  (`proxy-audit-fixes.md`).
- The prompt template carries the judgment for the IT model: under a
  completion-style prompt, no layer shows the class association (0.50 to
  0.59), and with the word in the question, `crime` and `whore` become
  significant (0.68, 0.63), a prompt artefact. For 4 of the refusal
  tasks (prostitution, intelligence, trust, migration) the negative term
  leads at L40 on 100% of images in BOTH groups: a prompt prior, not a
  judgment (`lens-refusal-probe-results.md`, checked).
- The layout drives the abstention: gemma-4-12b in thinking mode under
  `images_then_text` says "only one image" in 15 of 18 traces (NotSure
  83%); `interleaved_labels` gives 0 of 18 (5.6%)
  (`reasoning-trace-analysis.md`). The layout alone moves the gemma
  abstention by 10 to 17 points (`prompt-ablations.md`).
- The lens is a readout, not a site: removing the rank-5 label subspace
  at L42 changes the judgment by +0.043 logits (about 1% of a 5-logit
  scale; `label-ablation.md`, checked), and a random direction correlates
  with the judgment at |r| up to 0.98 because the last-token state is
  near one-dimensional (`steer-coord.md`, checked). No lens predicts the
  token-ablation map (`causal-lens-retrospective.md`,
  `lens-corpus-faithfulness.md`). The SAE plan states it: "the study did
  not show that the judgment is a structure of the model and not a state
  that the prompt template induces".
- The parent traces: the image artifacts drive the verdict (occlusion
  costs in 7 of 7 cases; weather and time of day pay, +0.21, +0.15), and
  the position drives the verdict and the narration (gemma picks image A
  0.545 and its praise leans to A; `trace-channels.md`,
  `ic-metadata-findings.md`).
- Moving the `dismissal` field from position 3 to 5 in the extraction
  schema dropped its rate from 85.2% to 64.2% with no text change: "an
  ingredient rate is a property of the schema version"
  (`ic-ingredient-extraction.md`).

### 4.3 Verdict

The paper can claim: "The NTA shift is not a product of 1 prompt, 1
model, 1 sample or the abstention option: it holds in 6 models, in 2
extraction rounds, in 2 image batches, under the image-content controls,
and the parent audit shows the income association under paraphrase,
persona, no city name and order swap." It can also claim: "At fixed
pixels, a neighbourhood name alone changes what 3 of 4 models say and
how they rate the place." It cannot claim that the bias is a fixed
direction in the model: the cue direction depends on the model and the
case, the text layout can reverse an area association, and the name
effect can be larger than the pixel effect. The honest form is: the
models carry a place prior that the prompt can switch on, and the image
content explains only about 10% of the area shift that we can measure.
"Unavoidable" is too strong; "present under every elicitation we tried,
and not removed by the controls we have" is what the data support. The
white-box docs say the same in their own words: the judgment precedes
post-training, but the prompt template carries it in the tuned model,
and the lens is a readout, not a site.

## 5. The gaps

### RQ1 (ungrounded inferences)

1. Tie each inference span to a cue. On a random 300 restaurant answers
   (50 for each model), a human (or a second LLM pass checked on 100
   human labels) marks each `inference` span as "a cue in the image
   supports it" or "no cue". Report the unsupported share for each model
   and each kind, with a chi-square test by model. No new GPU run: the
   texts, the spans and the images are on disk (`data/f4_traces/viewer/`).
2. A pixel counterfactual for 1 cue. Take 200 restaurant images, remove or
   blur the sign (or swap the awning colour), and ask the F4 prompt again.
   McNemar on "holds a popularity or upscale inference". 1 short GPU run
   (about 1,600 answers for 4 models).
3. A ground-truth check for 1 inference kind. The `popularity` claims
   against a public footfall or review-count proxy for the restaurant
   (the DOHMH join exists). A point-biserial or a t test of the proxy
   between the answers with and without the claim. No new GPU run if a
   proxy table exists; otherwise drop it.

### RQ2 (the neighbourhood correlation)

1. Method B with `c_cool_i` and the net valence at the unit layer with a
   finer image control: add the F6 facade-check P(Yes), the capture
   month and the camera distance as covariates, and a storefront
   descriptor (the OCR DBA match). This is a re-run of
   `scripts/f8/method_b.py` with more columns. No new GPU run.
2. Report the % Black row as the main race row next to % white, with a
   paired comparison of the 2 betas, and say that % Asian is null. The
   numbers exist in `valence_fits.csv`. No new GPU run.
3. For the street case, a positive control: the same R3 readout on the
   100 richest and 100 poorest NTAs with 5 images each, a t test. The
   pooled F8.3 street data already hold this; a t test on the 2 ends is a
   notebook cell. No new GPU run.

### RQ3 (unavoidable or not)

1. Separate class from race in the cue. Add 2 pairs for each borough
   where the poorer name is the whiter NTA (or the 2 names have the same
   % white and differ in income), and run the same F8.2 design on the
   same 1,000 images and 4 models. Paired t test, Bonferroni. 1 GPU run
   of the size of F8.2 (about 8,000 answers). This is the single most
   useful run.
2. A no-name control arm. Run the F8.2 images with a neutral cue of the
   same form ("I am in New York City.") so that the baseline has the same
   prompt length and shape. A paired t test of rich against neutral and
   poor against neutral. 1 short GPU run (4,000 answers). This tells the
   paper whether the name moves the judgment up, down or both.
3. A noise floor for 4.8. Run 1 model 2 times on the same 1,000 images and
   repeat the 4.8 chi-square table. The 2 qwen3.5-9b reruns (D91, D92) and
   the canonical against batch-2 check (C1) give a partial floor already
   (1 of 115 rates differs). No new GPU run for the partial answer.

## 6. Headline results

| # | Result, with the number | Source | Honest caveat |
|---|-------------------------|--------|---------------|
| H1 | After the model answers "Look and tell me more about this restaurant", its answer to "Is this a cool place to eat at?" rises with the share of white residents (rho +0.45 to +0.74) and of residents with a bachelor's degree (+0.40 to +0.80) in the area, in all 4 thinking models; the cuisine, the DOHMH grade and the borough do not remove it (16 of 16 rows stay) | `f8-counterfactual-cue.md` F8.3; `r3_nta_spearman.csv`, `method_b_r3/c_cool_i_*` | an NTA correlation; the controls are coarse; the size is 0.07 to 0.32 SD for each SD of the feature |
| H2 | The valence of what the model says it sees shifts with the resident mix in all 6 models: 37 of 40 restaurant effects survive the image-content controls, median share kept 0.89 | `f8-method-b.md`; `valence_cases.csv` | small effects (0.04 to 0.18 SD); 96% of the spans are neutral; ecological unit |
| H3 | The same image with "I am in Park Slope" or "I am in Brownsville" gets a different answer: 16 of 32 readout tests and 11 of 64 word tests are significant; all 4 models write more "upscale" words with the rich name | `f8-counterfactual-cue.md` F8.2; `cue_primary.csv`, `cue_words.csv` | the direction differs by model and case; class and race are mixed in 9 of 10 pairs; about one third of the upscale hits describe the named area |
| H4 | The choice of the model changes what the text holds: 291 of 316 categories differ by model, 150 by 10 pp or more, up to 86.6 pp; but the direction of the NTA association is the same across models in 157 of 193 rows | runbook 4.8, 4.10; `f4_model_differences.csv`, `f4_nta_feature_signs.csv` | the texts differ in length; the extractor reads each style with a different recall |
| H5 | The parent audit: the area-income association holds under paraphrase, persona, order swap, no city name and no abstention (tract r 0.35 to 0.47; order halves 0.421 and 0.423) | `robustness-battery.md`, `prompt-ablations.md`; `robustness_estimand_v1.0.0.csv` | area layer only; the unit-layer sign can differ; income only, no race row in the parent |

## 7. Mismatches and stale numbers

| # | Claim | Where | What the file says | Note |
|---|-------|-------|--------------------|------|
| M1 | nocity lowers the proxy agreement in 17 of 28 CD cells, median d_tau -0.033; "naming the city carries real information" | `proxy-arms.md`, `proxy-audit-fixes.md`, the closing line of `prompt-ablations.md` ("19 of 32 cells") | `proxy_arms_community_district_v1.0.0.csv` (73 cells, v1.2.0 proxies, gemma reruns): median d_tau -0.006, 23 fall and 14 rise at a 0.02 band | stale after the 2026-09-02 and 09-04 gemma reruns. Do not quote the headline |
| M2 | quantile shift, gemma d05: schools +0.654, subway +0.158, libraries +0.228 | `quantile-shift.md` | +0.515, +0.040, +0.044 | the doc says it predates the gemma noabstain rerun; the qwen cells agree |
| M3 | "largest abstention-proxy r is +0.58 (libraries, qwen, income)" | `proxy-audit-fixes.md` | the file now also holds schools building rows at -0.70 and +0.74 | the doc is older than the schools rows |
| M4 | 4.7 trace against answer, caption persons: 24 to 53 (trace), 1 to 13 (answer), 23 to 41 (trace only) | runbook 4.7, "first numbers" | the pooled CSV: 26 to 54, 0.5 to 13, 26 to 42 | the runbook gives the canonical values; not an error, but the paper should cite the pooled CSV |
| M5 | 4.8 examples: uncertainty llama-3.2-11b 7.4 and qwen3.8-27b 79.5; street image_artifact 7.8 and 94.5; caption person_reference 0.7 and 25.2 | runbook 4.8, "Examples" | the pooled CSV: 7.1 and 79.8; 7.7 and 94.3; 0.4 and 23.8 | canonical against pooled; the same note as M4 |
| M6 | "All descriptors, x2" table: restaurant response 17 pass, all qwen3.8-27b | runbook, "Words that only 1 model uses" | the pooled files: 172 pass, qwen3.8-27b 168 and qwen3.5-9b 4 | the table is the canonical batch; the pooled table in the same section (50, 172, 103, 161) agrees with the files |
| M7 | `proxy-overview.md`: median tau at the CD layer 0.19 for gemma | `proxy-overview.md` | the long CSV gives 0.14 over all rows | the doc value is over the validation rows only; not a clean check |
| M8 | Cue flags: `strong_and_consistent` 47, `reverses` 56, `neutral_but_decisive` 28, and "`contradicts_valence` is EMPTY ... 53 of 53 agree" | `cue-hierarchy.md`, the flag table and the text after it | `notebooks/cvpr/master/figures/cue_words_outliers.csv` (2026-09-22): 40, 26, 12, and 1 `contradicts_valence` row ("lot of tree", good, 0 pays, 3 costs) | checked. The "empty" claim is false on the file as it stands |
| M9 | `class` inferences pay (table: 4 pays, 2 costs; text: "class pays in 5 of 6"; lift "+0.354 above" where the table says +0.297) | `trace-channels.md` section 3 | `trace_channels_inference_summary.csv` on disk (written 2026-10-01 from the v3 corpus): class lift -0.092, 1 pays, 6 costs; wealth +0.231, 6 pays, 0 costs | checked. The doc is internally inconsistent, and the figure files were overwritten by the v3 run; the v2 numbers are no longer on disk. Wealth holds in direction; class flipped sign with the schema |
| M10 | `ghetto` tracks income across 197 NTAs at rho 0.45 (p_pool 0.05) | `lens-refusal-probe-results.md`, the dose test | `outputs/_monocle/probe_words_all/it_open/dose_test/dose.csv`, L47: `rho_partial` 0.448, raw `rho` 0.072 | checked. The doc gives the partial only; say which when you quote it. The extreme-groups AUC (0.74) does not depend on this |
| M11 | The whatami control-city test: Chicago 33% at top 1 | `whatami-lens-study.md` (a 30-answer tool test) | the full replay: latent Chicago 10.7%, latent Manhattan 8.8% | the runbook F8 advisor table states this correction already |

Numbers checked and found to agree (the file is named in each table
row): the F8 counts (16 of 32; 40, 37, 0.89; 9, 4, 3, 0.80; 71 of 160; 16
of 16; the `c_cool_i` rho values; the 4.12 rho +0.744, +0.717, +0.666);
the 4.8 counts (316, 291, 268, 187, 150, the medians and the largest
ranges); the 4.10 counts (408 of 7,620; 176, 115, 94, 15, 8; 193 rows; 16
all-model rows; 62 and 39; the test-1 medians; the weather gap 0.62 and
0.47); the 4.11 cross-case table (all cells); the exclusive-head counts
(50, 172, 103, 161; 8, 76, 34, 71; qwen3.8-27b 419 of 486); the caption
entropy (9 of 70, recomputed); the parent unit correlations named in
section 2.1 and 3.1; the parent robustness and ablation values of 3.4 and
3.5; the white-box values of 2.10, 2.11, 3.12 and 3.13; and the trace
values of 1.9, 2.12, 3.11 and 3.14 (the 3 readers checked these against
the files that each doc names).

## The notebook

`notebooks/facct/rq_evidence.py` (helpers in `_rq.py`) shows the camera-ready
figures and tables of this review: 1 section for each RQ, with its verdict,
and the headline table. It reads the result files and does no test. When
F8.5 (D136) ends, its section shows the decoupled cue. Its button writes the
figures to `notebooks/facct/figures/rq/`.

## Related

- `facct-runbook.md`, `f8-method-b.md`, `f8-counterfactual-cue.md`,
  `f8-lens-figure.md`, `caption-method-review-2026-10-28.md`
- `robustness-battery.md`, `prompt-ablations.md`, `proxy-arms.md`,
  `unit-validation.md`
- `trace-channels.md`, `cue-hierarchy.md`, `langextract-trace-extraction.md`
