# Paper 2 plan: what post-training does to a judgment (arXiv, then ICML 2027)

Written 2026-09-17. One of 2 paired plans. The other is
`paper-plan-cvpr-audit.md`. The demarcation table is the same in both
files. Change it in both.

| Item | Value |
|------|-------|
| Repo | to make: `papers/icml27_posttraining-judgment` (same layout as the CVPR repo) |
| Working title | Suppressed, Removed, or Preserved: What Post-Training Does to a Vision-Language Model's Judgment of Neighborhoods |
| Venue and date | arXiv version 1 near 2026-11-15, as the pair of Paper 1; ICML 2027 (deadlines were in late January in recent years; check the call) |
| Reader | an interpretability or alignment reviewer |
| Record | the artifact "Inside the Judgment"; `lens-refusal-probe-results.md`; `sae-plan.md` |

## The claim

Post-training has 3 different effects on the social judgments a
vision-language model forms from a street image. The layer at which a
judgment is decodable tells the effects apart.

| Effect | Evidence | Words |
|--------|----------|-------|
| Suppressed at the output | the model refuses, and the layer-44 logit-lens readout still separates rich from poor neighborhoods (AUC 0.73); the base model states it (0.72 against 0.74) | the wealth contrast task |
| Suppressed in the final layer | both models form the word at layers 36 to 42; layer 47 of the IT model removes it, in the unnormalized logit and not in the normalization | `rich`, `smart`, `luxury`, `trustworthy` |
| Removed | the base model forms the word from layer 30; the IT model forms it at no layer | `Black`, `abandoned`, `dirty`, `filthy`, `neglected`, `posh`, `trash`, `violent` |
| Preserved | the word follows the neighborhood at the output of the IT model | `ghetto`, `poor`, `poverty`, `immigrant` |

**Warning:** do not write that the early layers hold the harmful
features. The early-layer single-word logit-lens analysis shows the
opposite: layers 6 to 24 show a significant association for 0 to 4 of 59
words at the last token position, in both models. The judgments form
from layer 30. The suppression is late: at the output and in layer 47.

## Contributions

1. A Jacobian lens for image tokens in a decoder with no vision tower, with validity criteria that say when a logit-lens readout is valid.
2. The information flow: the positions at which the judgment is decodable across layers (image tokens, question tokens, last token position).
3. A refused judgment that stays linearly decodable, shown against census ground truth, with the base model as the control.
4. The suppressed, removed and preserved split, per word and per layer, in both models, at the last token position and at the image tokens.
5. The final-layer direct logit attribution: both models replace most of the residual stream at layer 47; only the final-layer transformation of the IT model drops the gaps of the affluence terms.
6. The sparse-autoencoder analysis (ICML version): sparse autoencoders at the image tokens with latents shared across 2 seeds, and what they show about contributions 3 and 4.

## Demarcation: which paper owns which result

Rule: a result is a contribution of 1 paper only. The other paper cites
it in the third person and gives no new number for it.

| Result | Owner | In the other paper |
|--------|-------|--------------------|
| 14 million pairwise judgments, TrueSkill2 rankings, 7 cases, 2 models | Paper 1 | cited as the source of the task and the proxies |
| Validation by proxy (unit, tract, NTA, CD); the overview figure | Paper 1 | cited |
| Prompt ablations, the robustness battery, positional bias, the camera face effect | Paper 1 | not used |
| Reasoning traces: the cue hierarchy, the inference channels, uncertainty and abstention, point of view | Paper 1 | 1 sentence of motivation |
| The single-image logit difference `c` and the layer-42 logit-lens readout against the proxies (`tables/single_image_read.tex`) | Paper 1 | cited as the external validity of the readout |
| Activation addition moves the answer to each pole in 6 of 7 cases (the headline only) | Paper 1 | Paper 2 owns the geometry (the principal judgment direction, the pseudo-inverse steering direction, the gradient) and the image-token condition |
| The refused wealth judgment stays linearly decodable (AUC 0.73 on refused images); the base model states it (0.72 against 0.74) | Paper 2 | Paper 1 states it in 1 paragraph with the citation, no table |
| The Jacobian lens on image tokens, its validity criteria, the estimand, the registry | Paper 2 | Paper 1 gives the 4-sentence method and cites |
| The information flow across layers (image tokens 0-22, question tokens 23-30, last token 31-47) | Paper 2 | not used |
| The token-ablation map, the failure of gradient-based attribution, the fitted ablation predictor | Paper 2 | Paper 1 keeps 2 sentences: no gradient-based map finds the image tokens |
| Experiments 2 and 3 of the contrast tasks, the single-word logit-lens analysis, the correlation across all NTAs | Paper 2 | not used |
| The final-layer direct logit attribution; suppression against removal across layers; the early-layer single-word analysis | Paper 2 | not used |
| The sparse autoencoders | Paper 2 | not used |

## Section plan

| Section | Pages | Content | Source | State |
|---------|-------|---------|--------|-------|
| 1 Introduction | 1.0 | the question; the 3 effects; the pair with Paper 1 | this plan | to write |
| 2 Related work | 0.7 | shallow safety alignment; hidden bias; logit lens and activation patching on VLMs; SAEs | `white-box-related-work.md` | list done |
| 3 Setting | 0.6 | the model (no vision tower, 256 image tokens, 48 layers), the 2 models and their prompt templates, the images, the census variables, the partial correlation that controls for population density | `gemma4-12b-architecture.md` | done |
| 4 The method | 1.0 | the Jacobian lens, the estimand, the validity criteria, the control-word threshold, the pitfalls (massive activations, bare-or-space token) | `jacobian-lens-review.md`, `lens-registry.md` | done |
| 5 Where the judgment is decodable | 0.8 | the information flow; the token-ablation map; the failure of gradient-based attribution | `where-the-judgment-lives.md`, `patch-attribution.md` | done |
| 6 The refused judgment stays decodable | 1.2 | Experiments 1 and 2, the primary readout, the pseudo-inverse steering direction at layer 45, the base-model condition | `lens-refusal-probe-results.md` | done |
| 7 Suppressed, removed, preserved | 1.7 | the single-word logit-lens analysis, the correlation across all NTAs, layers 6 to 47 in 4 conditions, the final-layer direct logit attribution | the same, section "Early layers" | done on 1 model |
| 8 The sparse-autoencoder analysis | 0.6 | the P40 SAEs; the census of latents; which latents layer 47 drops | `sae-plan.md` | stage 2 done; 3 to 6 open |
| 9 Limits | 0.4 | 1 city; token choice; census variables; 1 model family until the second model family runs | | to write |
| Appendix | | every validity criterion, every pitfall, the full word tables, the activation-addition geometry, the fitted ablation predictor | the artifact | material exists |

## Figures and tables

| Number | Content | Exists |
|--------|---------|--------|
| Fig. 1 | the 3 effects on one layer axis (the per-layer chart, reduced) | draw |
| Fig. 2 | the information flow: the fraction of the effect recovered, by position and across layers | data yes |
| Fig. 3 | refused images: the logit-lens readout against NTA income, IT model and base model | data yes |
| Fig. 4 | words above the control-word threshold per layer, 4 conditions (the artifact chart) | yes |
| Fig. 5 | per-word layer profiles: `Black`, `dirty`, `rich`, `smart`, `ghetto`, `immigrant`, IT model against base model | data yes |
| Fig. 6 | layers 46 to 47: the gap per word, both models | data yes |
| Tab. 1 | the lens validity criteria | yes |
| Tab. 2 | the contrast tasks: AUC, density-matched AUC, density gap, 10 contrast tasks, 2 models | yes |
| Tab. 3 | the shared latents of each SAE (ICML version) | yes |

## Work that remains

| Item | Needed for | Size | State |
|------|------------|------|-------|
| The `Black` token control (a partial correlation that also controls for mean image luminance, or a same-image colour control) | arXiv v1; the removal claim rests on it | small | DONE 2026-09-17: the base-model association survives the luminance control at layers 36 to 47 (0.39 to 0.47); the layer-30 value is confounded with colour vocabulary |
| Which component of layer 47 writes the new residual stream (attention or MLP); a linear probe on h47 with held-out neighborhoods (is the information removed, or does it stay linearly decodable) | arXiv v1; it separates "suppressed" from "moved to a different direction" | medium | DONE 2026-09-17: the MLP output of layer 47 cancels the affluence terms in the IT model (attention near 0; the base model has no such term); a linear probe on h47 keeps AUC 0.89 (IT) and 0.90 (base), equal to h46 |
| The test of independence from the wealth readout (are trust and intelligence the wealth readout?) | arXiv v1, 1 paragraph | small | DONE 2026-09-17: housing type, intelligence and trust are the wealth readout (own AUC 0.39 to 0.54 with the class readout regressed out); `immigrant`, `safe`, `lawful`, `housed` and the base-model `Black` are independent of it |
| Your decision on the shared-latent rule | stage 3 and later | none | OPEN with the user |
| SAE stages 3 and 4 (the census of latents; selection on held-out NTAs) | v1 only if clear by 2026-10-31; if not, ICML | medium | not started |
| SAE stages 5 and 6 (causal tests; weight attribution) | ICML | large | not started |
| A second model family: lens fits, the single-word logit-lens analysis, base model against IT model | ICML; reviewers will ask | large | not started; the Qwen lens is open in the summary of claims |
| The paper repo, the figure scripts under `notebooks/` | v1 | small | not started |

## Schedule

| Date | Milestone |
|------|-----------|
| 2026-09-24 | the repo; the outline; Fig. 1 and Fig. 4 from the existing data; the shared-latent decision |
| 2026-10-10 | the `Black` control, the layer-47 direct logit attribution and the linear probe on h47 done |
| 2026-10-31 | sections 3 to 7 drafted; decide if section 8 goes in v1 |
| 2026-11-10 | arXiv v1 posted, so that Paper 1 can cite it |
| 2026-12-15 | the second model family done; SAE stages 5 and 6 done or cut |
| late January 2027 | ICML submission (arXiv v2) |
| May 2027 | NeurIPS, if ICML says no; no change to the argument of the paper |

## Risks

| Risk | Answer |
|------|--------|
| "Removed" is only "not decodable by this lens" | the linear probe on h47 with held-out neighborhoods, the image-token condition and the SAEs are 3 independent measurements; say "not formed under any measurement we have" |
| The `Black` result is a colour-word effect | run the control before v1; if it fails, the removal list still holds 7 words |
| 1 model family | the ICML version adds the second; v1 says so in the limits |
| The street-image setting looks narrow to ICML | put the census ground truth first: few interpretability papers have an external truth for a social judgment |
| Overlap with Paper 1 | the demarcation table; no shared table or figure |
| The preprint removes the anonymity of Paper 1 | check the CVPR 2027 and ICML 2027 preprint rules before the post |

## Related

- `paper-plan-cvpr-audit.md`, `white-box-related-work.md`
- `lens-refusal-probe-results.md`, `lens-refusal-probe-plan.md`, `sae-plan.md`
- `jacobian-lens-review.md`, `lens-registry.md`, `where-the-judgment-lives.md`, `patch-attribution.md`, `steer-coord.md`, `single-image-steering.md`
