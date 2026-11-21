# Review of "Test residuals and word sets first" (J space readout analysis methods)

Date: 2026-10-31. Reviewer: an independent advisor session. Scope: the report at
`reports/J space readout analysis methods.md` (line numbers below refer to that file), its 4
source notes, the runbook F10.1 to F10.6, the lens code, and the F10.6 store.

Evidence labels in this review:

- **Store check**: a new calculation on `data/f10/hop_all/` for this review. Scripts:
  `/tmp/claude-1643544/-share-ju-matt-mllmsci/46773041-7225-4a8a-a948-787bb670957c/scratchpad/store_check.py`
  and `softcap_check.py` (session scratch; the numbers are in this file).
- **Code**: read in `sub/jacobian-lens/jlens/hf.py`, `jlens/lens.py`, `scripts/f10/hop_all.py`,
  `scripts/f10/hop_store.py`, `monocle/jlens_read.py`, and the `safety-research/jpp_lens` README
  and `base_lens.py` (by `gh api`).
- **Opened**: I read the source page or the extracted PDF text myself.
- **Agent**: a research subagent opened the page and returned a summary. Quotes from an agent
  are "as returned" and not checked against the raw text.
- **Snippet**: a search result only.

## 1. Verdict

The report is sound on statistics and on the count-method critique, and its store facts are
correct. Trust it for the inference plan (NTA clusters, Holm, split sample, power). Do not trust
its logit algebra on Gemma as written: the saved Gemma reads are softcap-saturated at the top
of the list, so set scores, slopes, PCA and RDMs on raw Gemma reads measure a compressed scale.
Trust the lens-class table, but not the claim that the crossed read separates a lens artefact
from a model change. The method rank puts two lens-bound word methods above the residual
decoder that the report's own conclusion says to run first; the rank should follow the
conclusion. The report misses the most direct logit method for the mechanism question: the
per-cell `Yes` minus `No` lens margin, computed from the saved residuals with no GPU.

## 2. Errors

### 2a. Definite errors

| # | Claim | Line | What is wrong | Severity | Fix |
|---|---|---|---|---|---|
| E1 | "The final norm and the softcap are non-linear, so check this algebra at the top of the list." The report then treats the saved reads as logits in every method. | 46, 83 to 98 | **Store check.** On Gemma the saved read is `30·tanh(z/30)`. The cap saturates the top of every list. At `final` L40 the median top-1 is 29.98 and the median top-30 is 29.33: the whole top-30 sits inside 0.65 of the cap. At `after_place` L56 the top-30 sits inside 0.94. On average 160 to 270 words per read are above 25 at L40 to L56 (raw value above 36). A saved value of 29.9 is a raw value of about 96; 29.99 is about 130. Thus a difference of 0.1 near the top hides a raw difference of tens. Set-minus-control, slopes, PCA and Euclidean RDMs on raw Gemma reads measure a compressed scale, and the compression differs by image and layer. Qwen has no cap (maximum saved read 22.5). | High, Gemma only | Invert the cap before any algebra: `z = 30·atanh(x/30)`. Mark values above about 29.99 as censored (float32 loses the raw value there). Or compute the pre-cap logits from the saved residuals with `lm_head(final_norm(J h))` and no `tanh`. Ranks do not change (the cap is monotone), so the top-30 lists stand. |
| E2 | "J++ and the R-lens share the top-1 token on only 0.0% to 1.6% of image blocks (F7)." Used as evidence that the lens drives the words. | 116 | The runbook F7 table compares J++ on Gemma-4-31B and Qwen3.5-9B with the R-lens on **Gemma-4-12B**: 3 different models. The runbook says on the same lines: "The 3 models differ, thus these tables cannot separate the lens from the model." The report drops that caveat. | Moderate | Cite the fact with its caveat, or drop it. The lens-dependence claim has better local support: `jacobian-lens-review.md` F2 (r = 0.67 between lenses at L42, all positions) and the SRP paper. |
| E3 | "The lens read is linear in `h` before the norm step, so all set scores are projections of one state"; "a contrast removes [the bias] to first order"; "a set-minus-control contrast cancels `log Z`". | 46, 106, 140 | **Code.** `HFLensModel.unembed` is `lm_head(final_norm(h))`, then `cap·tanh(·/cap)` on Gemma. The final RMSNorm makes every read `c_i · W diag(g) J h_i` with `c_i = 1/rms(J h_i)`, a scale that differs by image and by layer. A set-minus-control contrast does cancel `log Z` and the omitted tokens, but it carries the factor `c_i`. Across images the contrast is `c_i · (linear contrast)`, so a slope of the contrast on the margin also picks up any link between the residual norm and the margin. The report nowhere accounts for `c_i`. On Gemma the softcap sits on top (E1). | Moderate | State the read as `cap·tanh(c_i · W_axis diag(g) J_l h_i / cap)`. Standardize each read (divide by its SD over the axis) before a contrast, or compute contrasts on `J h / rms(J h)`. The report proposes the per-read SD only for cross-layer scale (line 52); apply it within layer as well. |
| E4 | "Cameron and Miller say that G = 50 is enough for state-year panels." | 144 | **Opened.** The paper says: "there is no clear-cut definition of 'few'; depending on the situation 'few' may range from less than 20 to less than 50 clusters in the balanced case, and even more clusters in the unbalanced case." It also reports (MacKinnon and Webb) that "even with G = 50, tests based on the standard CRVE with T(G−1) critical values can over-reject considerably if the clusters are unbalanced." The report's paraphrase inverts the caution. The conclusion for 197 balanced NTAs still holds. | Low | Quote the "less than 20 to less than 50" sentence. Our 197 balanced clusters of 3 are above that range. |
| E5 | "Whole-block permutation tests a link between NTAs, and within-block permutation tests a link inside NTAs." | 159 | When the regressor is an image-level margin, a whole-block exchange gives NTA A's 3 margins to NTA B's 3 images. That breaks the within-NTA pairing as well as the between-NTA pairing. Whole-block permutation thus tests the joint null (no link at all) under arbitrary within-block dependence. It is the valid null for the cluster OLS row; it is not a test of "a link between NTAs" only. Within-block permutation tests the within-NTA link conditional on the NTA means. | Low to moderate | Rewrite: whole-block exchange preserves within-NTA dependence and tests any link; within-block exchange tests the within-NTA link. The between-NTA link is the Pearson row on NTA means. |
| E6 | Fact 2 treats the case of the answer token on Qwen only. | 26 | **Store check.** On Gemma the first token is `no` 238, `Yes` 208, `No` 131, `yes` 14. The model says yes in upper case and no in lower case. Thus `yes_no` (capital pair) agrees with the answer on 97.0% and `yes_no_lc` on 93.4%; 18 images have the wrong sign on the primary margin. | Low to moderate | On both models use a case-pooled margin: `logsumexp(Yes, yes) − logsumexp(No, no)` from the saved final logits, or recompute it from the saved final residual with the 4 `W_U` rows. Report `yes_no` as the check. |
| E7 | "In random splits of texts, chi-square rejected 57.6% of words and G2 rejected 65%. The t, Wilcoxon and bootstrap tests rejected 3.6% to 4.8%." | 40 | **Opened.** Lijffijt et al. report the share of words whose p-value distribution over 500 random splits fails a Kolmogorov-Smirnov test of uniformity. The numbers are correct (57.6%, 65%, 3.6 to 4.8%, and 16.3% for inter-arrival time), but they are not per-test false positive rates. The report's sentence reads as a false positive rate. | Low | Write "57.6% of words had non-uniform p-values under the null" or similar. The point of the citation stands. |

### 2b. Disputed points

| # | Claim | Line | Dispute | How serious | Suggestion |
|---|---|---|---|---|---|
| D1 | The crossed read splits a word change into "a model term and a lens term"; "RSA alone cannot settle that case, and the crossed read can." | 127 to 134 | The algebra is right: the two terms add to `z(l2,l2) − z(l1,l1)` (I checked). But each term holds half of the interaction `z(l2,l2) − z(l2,l1) − z(l1,l2) + z(l1,l1)`, and the report does not name the interaction. More important: `J_l1 ≠ J_l2` because the computation after layer l1 differs from the computation after l2. That difference is a property of the model, not an artefact of the fit. So the "lens term" mixes a real model fact (how a state at l1 is used later) with fit noise and corpus bias. The crossed read is a descriptive 2-way table, not an artefact detector. `J_b h_a` is also off the fit domain, as the report says. | Moderate | Keep it as a description. For the artefact question use the standard comparators: (a) one fixed lens for all layers (one row of the cross), (b) the plain logit lens `J = I` (the report's "Shared" class, which no ranked method uses), (c) a lens fit on a second corpus, and (d) the KL-to-final trajectory (section 5). Agreement across (a) to (c) is the usual evidence that the model, not the lens, moved. |
| D2 | "If the unembed is linear, a read RDM is a residual RDM under the metric `J_lᵀ Wᵀ W J_l`." | 134 | The condition fails for both models (final norm) and twice for Gemma (softcap). With the norm, a Euclidean read RDM is the RDM of `c_i J h_i`, so a per-image scale enters. Two images with the same direction and different norms are far in the read RDM and near in the residual RDM. The inference "residuals stable and reads changed means a lens change or a low-variance model change" has a third cause: norm or cap. | Moderate | Use correlation distance (1 − r over the axis) for read RDMs; it removes the per-read scale and shift. State that the metric argument then holds only in the pre-cap, normalized space. |
| D3 | "Softmax probabilities do not stay valid, because the omitted mass is unknown." | 46 | Too strong. A softmax over the saved axis gives `p_w / P_i(axis)`. Within one read, every log-ratio between two words is exact (this is subcompositional coherence in the Aitchison sense). Only the absolute level moves by `log P_i(axis)`, which differs by image. So within-read log-ratios of the partial softmax are valid; cross-image levels are not. | Low | Rewrite as: within-read log-ratios are exact on the partial axis; absolute probabilities and cross-image levels are not. |
| D4 | The rank puts two lens-bound word methods (rank 1, 2) above the residual decoder (rank 3) and RSA (rank 4). | 81 to 98, 224 | The report's own logic: "only residual methods can show that a stage exists" (line 3) and "Run the decoder curve first" (line 224). Rank 2 is the most exposed method in the table: lexicon loading over the whole axis is dominated by the shared word profile (store check: the image-specific part of a read is 0.1% to 2.6% of its variance; see section 3), and on Gemma the top of the axis is capped. Rank 1 is sound as a test but it names content that a stage must first exist for. | Moderate | Rank the decoder curve and the lens margin trajectory (section 5, method 1) first, RSA second, set-minus-control third, lexicon axes after the crossed-lens checks. See section 6. |
| D5 | "Divide by the per-read SD over the fixed axis, or use ranks." The double-centring section calls per-word centring a trap. | 48, 52 | The published fix for the fixed bias is per-token z-scoring across contexts (Panchal 2026, LessWrong; J++ post calls z-scoring a "mixed influence" on recall). Panchal finds that subtracting the per-token offset alone **hurts** recall 3.5 to 7.3 times, and that dividing by the per-token SD helps. The report's per-read SD is a different axis (per image, over words). Both are valid for different questions; the report does not say which axis, and it does not cite the per-token result. | Low to moderate | Say: per-read standardization fixes layer and image scale; per-token standardization (mean and SD over training-NTA images, or over the prefix and patch reads) fixes the frequency offset. The one-sample test is then on a held-out split, which removes the "trap". |
| D6 | "A critique says that swaps can pass by construction." | 114 | **Agent.** The GPT-2 post does not use those words. It says the tests "may mostly follow from how they and the J-lens are constructed" and "the verbal swap is nearly guaranteed". The paraphrase is fair but should not be in quotation form. | Low | Quote the exact words. |
| D7 | The "low prior" paragraph cites one outside critique (the GPT-2 post). | 36, 114 | **Agent, Opened.** There are more, and some bear directly on our models: Bandularatne 2026 (J-lens loses to the logit lens on GPT-2 at 11 of 11 layers; the shared bias correlates with log frequency at r = 0.66); Panchal 2026 (the J-lens offset is token frequency, Spearman 0.48 on Qwen3.5-4B, r = 0.70 to 0.73 on GPT-2 and Pythia); Zeisler 2026 (swaps flip the top-1 on 6% to 11% of items on Qwen3.6-27B and Gemma-3-27B, against 54% to 70% on Claude); Nanda 2026 review ("a crude approach that will find noisy directions"; English tokens have higher unembedding norm, so top-k favours high-variance tokens); a pre-registered Qwen2.5-7B detector that "does not clear a TF-IDF baseline (0.690)" (comment on the Nanda review). | Low to moderate | Add these to the prior. The frequency-offset results support the report's analysis 1 and the control match on the text-only read. |
| D8 | "Holm ... (Shaffer 1986)." | 161 | Holm's procedure and its proof for any joint distribution are Holm 1979. Shaffer 1986 extends it with logical constraints. | Trivial | Cite Holm 1979; keep Shaffer for the modified form. |
| D9 | "Liu 2026: answer before criteria." | 213 | **Opened** (abstract). The paper is "Layer-Order Inversion": later-hop answer entities become decodable earlier than bridge entities. It is not about an answer before its criteria. The prediction in the row (answer decodes before a criteria word) does fit the finding. | Low | Rename the row "later hop before bridge (layer-order inversion)". |
| D10 | "No source validates a J-lens on image tokens." | 116, 216 | True for the J-lens. But the plain logit lens on visual tokens has a literature: Neo et al. 2024 (object words at late layers in LLaVA-1.5 and Qwen2-VL), Jiang et al. 2024 (middle layers align best), LatentLens 2026 (snippet: the tuned lens adds little over the logit lens on visual tokens). The report should name these when it treats the patch reads. | Low | Add the three citations to the patch section. |

## 3. Items checked that hold

**Store facts (store check, all 10 rows of the table at lines 11 to 22).** Tail cells 25 × 15 and 24 × 8; axis 53,999 and 43,563; patches 256 and 1,024; `d` 5,376 and 4,096; `yes_no` mean (SD) −2.39 (8.61) and +1.57 (0.59); answer yes 37.6% and 94.9%; sign agreement 97.0 / 93.4 and 95.9 / 99.8; ICC 0.270 and 0.196; design effect 1.54 and 1.39; effective n 384 and 425. The correlation of the two margins is 0.996 and 0.978; `yes_no` is above 0 on 99.0% of Qwen images; the cross-model correlation is 0.734 on images and 0.791 on NTA means. The axis holds ` yes`, ` que`, ` own`, ` con` and lacks ` de`, ` la`, ` no`. The text-only read is NaN at tail rows 0 and 1 on Gemma and row 0 on Qwen. The within-NTA share of the margin sum of squares is 0.49 and 0.54. Array sizes (4.8, 1.9, 49, 79 GB) and `d/n` (9.1, 6.9) are right.

**New store facts the report should add.** The image-specific part of a read is small: after per-read centring, the word means over images explain 97.4% to 99.9% of the variance, and the image-specific residual is 0.1% to 2.6% (Gemma `final` L56: 2.6%; Qwen `final` L20: 1.4%). The word mean over 50 images correlates at r = 0.98 with the text-only read at the same cell. At a cell, the 591 images share 1 to 5 distinct top-1 tokens. Thus the report's analysis 1 ("measure the raw bias") has its answer already: the fixed profile is nearly the whole read, and the image signal lives in the residual after per-word centring.

**Code and lens form.** The GW paper §2.1 gives `softmax(W_U norm(J_l h_l))` (Opened). The J++ README gives `top-k over V_sem of W_U J_l h`, "where W_U is the unembedding (the model's final norm and LM head)"; `base_lens.apply` calls `model.unembed(transport(h))` (Code). The local `HFLensModel.unembed` adds the softcap when the config has one (Code). `hop_all.py` applies `lens.transport` then `lens_model.unembed`, keeps the ` [a-z]{3,}` axis, and defines `yes_no` from the model's own final logits (Code). The crossed-read algebra adds up (checked by hand). The J++ transport is one linear map per layer at inference (README).

**Statistics.** Power table (recomputed in SciPy): t cutoffs 1.97, 3.90, 4.47, 5.05, 6.24; detectable d at 80% power 0.20, 0.34, 0.38, 0.42, 0.51; r cutoffs 0.14, 0.27, 0.30, 0.34, 0.41 (the report's 0.31 at 3,750 tests and 0.50 at 18.75 million are rounding). Split sample: 99 NTAs, 5 sets d = 0.35, 10 sets d = 0.37. Tercile case: 66 against 66 NTAs, one word of 54,000, d = 0.90 at 50% power. Degrees of freedom in the table at lines 148 to 157 are right (196 for one-sample and paired t and for cluster t with G − 1; 195 for Pearson and two-sample t). The double-centring argument (a one-sample test after per-word centring is 0 by construction) is right. "A logit has no sample size" and the count = N·softmax objection are right. The hybrid-test reading of rank 1 is right (Goeman and Bühlmann, Opened: "a hybrid form between gene sampling and subject sampling"). Holm, max-t, BH and BY statements agree with Rousselet 2024 and Winkler 2014 (Opened: sign flipping needs "symmetry (i.e., non-skewness)" and independence; blocks are "exchanged with the others on their entirety"; the max statistic "requires that the statistic has a distribution that is spatially homogeneous"). Murtaugh 2007 (Opened): "the assumptions behind the two analyses are identical". The "2 models are 2 cases" rule is right.

**Logit algebra that holds.** `log p_w = l_w − log Z`, so a within-read logit difference equals the Monroe log-odds-ratio point estimate for a rare word apart from `log Z` (and on Gemma, with the saved capped value, this holds for the model's own output distribution form). A set-minus-control contrast does not depend on the omitted tokens. GW §A.21 uses mean log-probability of 2 target words minus 100 control words over the workspace layers (Opened); a log-probability difference within a read equals the logit difference. The answer-confound statement (deep `final` cells decode the margin by construction; jspace AUC 1.0 on constant-answer slices, 0.55 to 0.78 with varying truth) is right (Opened).

**Project record.** F10.2: 9 of 45 read, 0 of 90 area tests pass. F10.3 and the held-out test: the swap replicates on Qwen and is not readable on Gemma. F10.4: themes chosen by eye, no swap. F10.6: no preset test. HF `neuronpedia/jacobian-lens` holds `oracle/` adapters for qwen3-4b, qwen3.5-0.8b, qwen3.5-2b and qwen3.6-27b only, and a J-lens plus J++ for gemma-4-31b and a J++ only for qwen3.5-9b (Opened). So "no adapter for our 2 models" and "the Gemma J-lens as well as J++" are right.

## 4. Citation spot-check

| Claim in the report | Source | How read | Result |
|---|---|---|---|
| TF-IDF "breaks down in a collection of two documents" | Monroe et al. 2008, p. 384 | Opened (PDF text) | Holds, verbatim. |
| Count-free variance `1/(y_i+a_i) + 1/(y_j+a_j)` | Monroe et al. 2008 | Source note says eq. 17 to 20, full text | Not re-checked; the note read the full text. |
| chi-square 57.6%, G2 65%, others 3.6 to 4.8% | Lijffijt et al. 2016, §5.1 | Opened (journal HTML) | Numbers hold; they are KS rejections of p-value uniformity, not false positive rates (E7). |
| Gene resampling 37 to 42%, CAMERA 4.8% at set size 100, correlation 0.05 | Wu and Smyth 2012 | Opened (PMC) | Holds: 41.6, 41.4, 38.8, 37.1% against 4.8 and 4.7%. |
| "may greatly understate the true P-values"; "a hybrid form" | Goeman and Bühlmann 2007, §4.2, §4.4 | Opened (journal HTML) | Holds, verbatim. |
| 2 target words minus 100 control words, averaged over band layers | GW paper §A.21 | Opened (arXiv HTML, chunk 300k to 400k) | Holds: "fictional" and "simulation" against 100 concrete nouns ("recipe, bicycl, guitar"), mean over workspace layers. |
| "artifacts of the J-lens methodology" | Workspace paper | Opened (transformer-circuits) | Holds, verbatim: "it is possible that these layer-wise effects are artifacts of the J-lens methodology". |
| Swap success 54% to 70% | GW §3.3 | Opened | Holds: 54% Haiku 4.5, 70% Sonnet 4.5, 70% Opus 4.5. |
| Mean-difference patches that need no lens, §A.24.1 | GW paper | Opened (chunk 400k to 500k, section start cut) | Partly: the patch is a mean-difference vector added at one layer. The chunk did not show whether the lens enters its construction. The transformer-circuits page says the intermediate-concept probe is a mean-difference probe that does not use the lens. |
| "mixed influence" of z-scoring; a few tokens lead many reads because the map is linear and not affine | J++ post | Agent (GreaterWrong, near full) | Holds as returned: "We may also consider z-scoring the lens readouts ... We find these approaches to have a mixed influence on performance"; "the Jacobian transport is a homogeneous linear map rather than an affine map". |
| Vocabulary-axis centering improves interventions, not ranks | GPT-2 post | Agent (full) | Holds: "This centering has no effect on the readouts, but makes the interventions much more effective." The post centres the J-lens vectors. |
| "swaps can pass by construction" | GPT-2 post | Agent (full) | Paraphrase; not verbatim (D6). |
| 39 of 80 prompts differ in script between English-fit and Chinese-fit lenses on Qwen3.5-9B | SRP paper (arXiv 2609.01936) | Opened (first 100k chars) | Holds; it is a Jacobian lens over layers 21, 24, 26 and 29. |
| jspace: AUC about 1.0 on constant-answer slices; 0.55 to 0.78 honest | solarkyle/jspace README | Opened | Holds, verbatim. |
| Logit lens bias; tuned lens affine translator; KL to final | Belrose et al. 2023 | Agent (ar5iv, full) | Holds as returned: "systematically puts more probability mass on certain vocabulary items than the final layer does". |
| Squared norm linear in log probability; WEAT non-zero for unbiased sets unless frequencies match; "frequency effects can flip the sign" | Ethayarajh et al. 2019 | Opened (ar5iv) | First two hold. The sign-flip claim does not: the paper's sign reversals (Table 1) come from the choice of attribute sets, not from frequency. Line 69 should drop "flip the sign". |
| "G = 50 is enough for state-year panels" | Cameron and Miller 2015 | Opened (PDF text) | Does not hold as written (E4). |
| t test on means and balanced nested ANOVA: identical assumptions | Murtaugh 2007 | Opened (PDF text) | Holds. |
| Sign flip needs independence and symmetry; whole-block exchange; max statistic needs homogeneity | Winkler et al. 2014 | Opened (PMC) | Holds. |
| Max-t strong FWER with pointwise thresholds; BH "limited claims about the location of effects" | Rousselet 2024 (Eur J Neurosci 2025) | Opened (PMC) | Holds; the paper also calls MAX "overly conservative". |
| "typically only interpretable in cases where the true model is extremely sparse" | Gentzkow et al. 2019 | Opened (PDF text) | Holds, verbatim (penalized regression section). |
| SemAxis two-word poles AUC 67.2 to 92.4 | An et al. 2018 | Opened (PDF text) | Holds: good/bad 92.4, a random pair 67.2; expansion raises the mean to 80.6. |
| NRC VAD v2 covers 44,928 words | Mohammad 2025 | Opened (ar5iv) | Holds: 44,928 unigrams plus 10,073 multiword terms. |
| Conwell 2025: linear read of scene features | PMC11789064 | Opened | Holds: PNAS 2025, ridge from frozen vision features predicts valence, arousal and beauty. |
| Liu 2026 "answer before criteria" | arXiv 2601.03542 | Opened (abstract) | Partly (D9). |
| NLA: text hypothesis, confabulation risk | transformer-circuits 2026/nla | Opened | Holds: "NLA explanations can contain claims about the target model's input context that are verifiably false." |
| Kozlowski: 6 to 10 pairs better than 1; rich-poor, luxury-cheap | Kozlowski et al. 2018 | Opened (abstract only; ar5iv redirects) | The abstract lists rich-poor. The 6-to-10 claim and luxury-cheap are not verified. |

## 5. New methods that work directly with logits

Type: **LW** = LessWrong or Alignment Forum post (not peer reviewed); **PR** = peer-reviewed venue;
**arXiv** = preprint. Read: who opened it and how far. "Covered" says whether the report has it.

| Method | Source (type, read) | What it computes | Applies to our data | What it answers for the mechanism | Failure mode | Covered |
|---|---|---|---|---|---|---|
| Lens margin trajectory (logit difference per cell) | Wang et al. 2022 IOI (PR ICLR 2023, agent full); Heimersheim and Nanda 2024 (arXiv, agent full); Belrose et al. 2023 prediction trajectory (arXiv, agent full) | `m(t,l) = [unembed(J_l h_{t,l})]_Yes − [·]_No`, and the same with `J = I`. Logit difference is "the cleanest" patching metric; it cancels the free constant. | Yes: saved residuals, saved `J_l`, 4 rows of `W_U`, final norm weights. No GPU. | The first token and layer where the answer log-odds appears, its depth profile, and its per-cell correlation with the final margin. The direct test of the answer confound. | Lens-bound; trivial at the last cell; the logit difference "may be driven by either" side. | No |
| Softcap inversion and censoring | Gemma 2 report and `jlens/hf.py` (Code); no analysis source | `z = c·atanh(x/c)`, `c = 30`; censor `x ≥ 29.99`. | Yes, Gemma reads. | A precondition: restores a scale on which differences mean something. | Float32 loses the raw value near the cap; recompute from residuals there. | Flagged, not applied |
| Per-token z-scoring of reads over contexts | Panchal 2026 (LW, opened); J++ post (LW, agent); Bandularatne 2026 (LW, agent) | Subtract the per-token mean over contexts and divide by the per-token SD. The offset correlates with log frequency (Spearman 0.48 on Qwen3.5-4B; r = 0.66 to 0.73 on GPT-2 and Pythia). Subtraction alone made recall 3.5 to 7.3 times worse; z-scoring improved the median rank (72 to 58 at L27, p = 0.002). | Yes, with means and SDs from training-NTA images or from the prefix and patch reads. | Removes the frequency profile that is 97% to 99.9% of a read in our store. | Reads of a rare target can get worse at some layers (L23 in Panchal). | Partly (per-read SD only) |
| KL or JS to the final read per cell | Belrose et al. 2023 (arXiv, agent full); DoLa, Chuang et al. 2023 (PR ICLR 2024, agent full) | `KL(final ‖ lens_l)` or `JSD(q_final, q_l)` per image and cell; "prediction depth" is the first layer where the top-1 stops changing; DoLa picks the layer of maximum JSD. | Yes on the full vocabulary from residuals; on the saved axis only as a relative curve. | Where the read settles; which cells carry the largest change. A lens-bound stage finder that needs no word list. | On a partial axis the JSD is biased; both lenses must share the softcap state. | No |
| Centred unembedding and causal inner product | Park et al. 2023 (PR ICML 2024, agent partial); Park et al. 2024 (PR ICLR 2025, agent partial); TransformerLens `center_unembed` (snippet) | Subtract the mean row of `W_U` (no change to softmax); whiten with `Cov(γ)^{-1/2}` before any cosine in unembedding space. | Yes, for the report's cosines between set vectors and `u_yes − u_no`. | Makes the cosine checks at lines 106 and 138 canonical. | Whitening is derived for the final layer only. | No |
| Direct logit attribution per source | Elhage et al. 2021 (Transformer Circuits, agent partial); DLA adversarial example (LW, agent full) | Project each component's output onto the `Yes − No` unembedding direction through the final norm. | Needs head and MLP outputs: a new GPU run. | Which patches and heads write the answer direction at the final position. | "DLA does not account for the fact that later nodes depend on the output of earlier nodes"; the final norm scale. | No (step 8 is a swap, not an attribution) |
| Activation patching with a logit-difference metric | Heimersheim and Nanda 2024 (arXiv, agent full); Conmy et al. 2023 (PR NeurIPS, agent partial) | Patch a cell between an image pair; measure the change in `yes_no`. KL is "the most robust metric to optimize"; logit difference "is not always positive". | New GPU run. | The report's step 8. | Dose and placebo choice; the metric choice changes the ranking of cells. | Yes (step 8) |
| Attribution patching | Nanda 2023 (LW, agent truncated); Syed et al. 2023 (arXiv, agent partial) | First-order Taylor estimate of patching from one backward pass. | New GPU run. | A cheap map over all 375 cells before real patches. | Fails where the gradient is 0 (KL at the clean model); R² = 0.27 against real patches. | No |
| Calibration by a content-free input | Zhao et al. 2021 (PR ICML, agent abstract); Holtzman et al. 2021 PMI_DC (PR EMNLP, agent full) | Divide the answer probability by its probability under a content-free prompt. | Yes: the text-only read is the content-free input. | Separates the judgment from the answer-token prior; a baseline for the margin. | PMI_DC fails on some tasks (BoolQ). | Partly (text-only as a word baseline) |
| Centred log-ratio (Aitchison) on the axis | Aitchison 1986 (not opened); UPF working paper 1106 (source note, abstract) | `clr(p) = log p − mean log p` over the axis. Equal to per-read centring of logits. | Yes. | Makes within-read log-ratios the unit of analysis; subcompositional coherence justifies the partial axis. | Needs the softcap inversion on Gemma first. | Implicitly (per-image centring) |
| Logit lens on visual tokens | Neo et al. 2024 (arXiv, agent full); Jiang et al. 2024 (arXiv, agent full); LatentLens 2026 (snippet) | Project patch residuals to the vocabulary; take the top object words per patch and layer. | Yes, from the saved patch residuals and top-30. | Validates that patch reads name the scene before any set score on patches. | "the accuracy of vocabulary projections substantially decreases in early layers" (Patchscopes); the tuned lens adds little on visual tokens. | No |
| Patchscopes and back-patching | Ghandeharioun et al. 2024 (PR ICML, agent full); Biran et al. 2024 (PR EMNLP, agent full) | Patch a residual into a target prompt that asks the model to describe it. | New GPU run. | A text read of a cell that does not depend on a fitted lens. | Prompt design; the target prompt can confabulate. | Partly (back-patch named) |
| Logit standardization | Sun et al. 2024 (PR CVPR, agent full) | Z-score logits per sample before a shared temperature. | Yes. | Same as per-read standardization. | Treats the shift as a nuisance; no inference. | Partly |
| J-lens replication results on open models | Zeisler 2026 (LW, agent full); Bandularatne 2026 (LW, agent full); Mehta 2026 (LW, agent full); Nanda 2026 review (LW, opened); Ratnaditya comment (LW, opened) | Swap flip rates 6% to 11% on Qwen3.6-27B and Gemma-3-27B; J-lens below the logit lens on GPT-2; a pre-registered detector that does not beat TF-IDF. | Context, not a method. | Sets the prior for any lens claim on our 2 models. | Small prompt sets (25 background prompts in Zeisler). | No |
| Local KL metric for lens directions | Briggs 2026 (LW, agent full; author says 30% finished) | Replaces the Euclidean logit norm of a direction with `sqrt(local KL)`; "a direction is only actually influential if its energy is focused on tokens that actually exist". On Gemma-4-31B. | Yes, for any direction analysis of our Gemma lens. | Guards against directions that spread mass over rare tokens. | Single-author draft; no comments. | No |
| Frequency and entropy neurons; the null space of `W_U` | Stolfo et al. 2024 (PR NeurIPS, agent partial); Gurnee et al. 2024 (PR JMLR, agent partial) | A token-frequency direction from centred log frequencies; entropy neurons act through the final-norm scale. | Yes for the frequency direction as a control regressor. | Names the direction that the fixed bias likely lies along. | Model-specific. | No |

### Notes on the top 5 that the report misses

**1. The lens margin trajectory.** The mechanism question is where the answer forms. The standard
metric for that in circuit work is the logit difference of the answer pair (IOI; Heimersheim and
Nanda: "we personally trust logit difference (or equivalently logprob difference) the most"). The
report never computes it inside the lens. From the saved residuals, `m(t,l) = (u_Yes − u_No)ᵀ diag(g)
J_l h / rms(J_l h)` costs one matrix product per cell, and on Gemma it goes through the softcap after
the product. Plot it over the 375 and 192 cells, correlate it with the final margin across images,
and compare with `J = I` and with a ridge probe (the report's rank 3). This is the one Bound method
that speaks to the answer and not to word content. It also gives the answer-confound check at line
138 a number per cell. Caution: it is trivial at the last cell, and a lens that cannot carry the
direction (the `jacobian-lens-review.md` P8 result: 5% reconstruction) gives a false null.

**2. The softcap on Gemma.** The store check shows hundreds of words per read inside the compressed
zone at L40 to L56. Nothing linear should run on the saved Gemma reads before `30·atanh(x/30)`.
Top-30 lists and ranks are safe. The viewer's percent values (softmax over the store words) are not
affected in order, but their spread is.

**3. Per-token standardization.** Three independent posts find that the J-lens offset tracks log
token frequency, and the one that tests the fix finds that z-scoring works where subtraction does
not. Our store agrees: the word profile shared across images is 97% to 99.9% of a read, and it
correlates at 0.98 with the text-only read. The report's set-minus-control with matched controls is
one way to remove it; per-token z-scoring on a training split is the other, and it gives a word map
that the control match cannot.

**4. KL or JS to the final read.** The tuned-lens literature measures a per-layer read by its
divergence from the final distribution, and DoLa uses the layer of maximum JSD as the point where
content enters. This is a stage finder on logits that needs no word list and no set. Compute it on
the full vocabulary from the residuals, per image, and summarize by NTA. On the saved axis it is
only a relative curve.

**5. Centred unembedding and whitened cosines.** The report checks cosines between set vectors and
between a set contrast and `u_yes − u_no`. Park et al. show that the Euclidean inner product in
unembedding space is not canonical, because the softmax is invariant to a shift and to an affine
map; the causal inner product whitens by the covariance of the unembedding rows. Subtract the mean
row of `W_U` (TransformerLens `center_unembed`) and whiten before any cosine claim.

## 6. Changes to the rank and to the first analyses

**Rank.** Move the residual decoder curve (rank 3) and the lens margin trajectory (new) to the top,
RSA with correlation distance to second, set-minus-control to third. Reasons: the report's own
conclusion ("Run the decoder curve first"); the store check (the image signal is at most 2.6% of a
read); the softcap (E1). Move lexicon loading and semantic axes (rank 2) below the crossed-lens
checks, and run them only on standardized, cap-inverted reads. Keep the crossed read (rank 5) as a
description and add the three standard comparators (fixed lens, `J = I`, second corpus). Promote
the top-and-bottom-images check (rank 15) to a routine step after any named word: it is cheap and
it is the only check that a word tracks visible content. Add a "Shared" row for the plain logit lens.

**First analyses.** Keep the order with these changes:

1. Measure the raw bias. The store check gives the first numbers (section 3). Add the frequency
   direction as a regressor and report the share of the word-mean profile it explains.
2. Before any logit method on Gemma, invert the softcap and standardize each read.
3. Plot the margin decoder over all cells (as written), and beside it the lens margin trajectory
   with `J_l` and with `J = I`. State the scene block: which patch layer, which pooling. A nested
   model (patches alone against patches plus cell) with NTA folds is the uniqueness test; a block
   permutation of the margin tests only the overall link.
4. RSA with correlation distance; the generalization matrix (as written).
5. Preset sets on standardized reads (as written), with controls matched on the text-only read and
   on log frequency.
6. Crossed read at block edges, with the fixed-lens row and the `J = I` row.
7. Word search on 98 NTAs, test on 99 (as written).
8. Neutral question on the same images (as written). Also run the "Think of ..." style no-image
   control: our text-only read already gives it for the words, not for the margin.
9. Swap a named stage (as written), with the logit-difference metric and KL as the second metric.

**The first-analysis logic.** The margin is a deterministic function of the image, and so is every
cell. "Unique information beyond the scene" therefore means "beyond a linear read of the chosen
patch features", and the report says so (line 224). Two more limits: with `n = 591` and `d = 5,376`,
a ridge decoder from any late cell will reach a high cross-validated R² because the final margin is
near-linear in the last residual, so the curve shows where linear decodability starts, not where the
model uses the code (the project's own `jacobian-lens-review.md` F1 makes this point); and the
Qwen margin has SD 0.59 with 30 no answers, so its curve will be flat for lack of spread, not for
lack of a stage. Report the Qwen curve on `yes_no_lc` and with the case-pooled margin (E6).

## 7. What I could not verify

- The J++ post equations (the agent's fetch did not render them). The readout order comes from the
  README and `base_lens.py`, which I read.
- GW §A.24.1 in full: the chunk I read starts after the section opens, so whether the lens enters
  the construction of the mean-difference patch is unconfirmed. The transformer-circuits page says
  the intermediate-concept probe is lens-free.
- Monroe et al. equations 17 to 20 (I checked the TF-IDF sentence only; the source note read the
  full text).
- Kozlowski et al.: the "6 to 10 pairs" claim and the luxury-cheap pair (abstract only).
- Iigaya 2021 and 2023, Imai 2010, arXiv 1410.7217, Schultheiss et al. 2021 ("p-value lottery" is
  a phrase from Meinshausen, Meier and Bühlmann 2009), Kriegeskorte 2009, Egami 2022, Hebart 2018,
  Dupré la Tour 2022, Semedo 2019, Williams 2018, Chang 2009, Paulo 2024, Sato 2022, Hamilton
  2016, Kornblith 2019, Murphy 2024, Dubey 2016, Brysbaert 2014, Shaffer 1986, Augustin 2012: not
  opened in this review. The claims about them are standard and I have no reason to doubt them.
- Every "Agent" row in section 5: the quotes are as a subagent's fetch returned them, from a small
  model's page summary. Check a quote against the page before it goes in a paper.
- LatentLens 2026, MMNeuron, "Over-alignment / geometric debiasing" 2026, OEC centred unembedding
  (arXiv 2605.18106): snippets only.
- Whether Qwen3.5-9B's config has a softcap: the store maximum of 22.5 says no, and the code reads
  `final_logit_softcapping` from the config; I did not open the config file.
- The size of `c_i = 1/rms(J_l h_i)` across images at a cell (E3): not measured. One line on the
  saved residuals gives it.
