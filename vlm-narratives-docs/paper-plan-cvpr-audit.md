# Paper 1 plan: the audit (CVPR 2027)

Written 2026-09-17. One of 2 paired plans. The other is
`paper-plan-interp-posttraining.md`. The demarcation table is the same
in both files. Change it in both.

| Item | Value |
|------|-------|
| Repo | `papers/cvpr27_machine-perception-public-spaces` |
| Working title | AI Judgments of Public Spaces: From Aesthetic Appearance to Neighborhood Reputation |
| Venue and date | CVPR 2027, deadline 2026-11-15, 8 pages without references |
| Reader | a vision or fairness reviewer; an urban-science reader |
| Priority | first: it has the hard deadline |

## The claim

Vision-language models judge public places by appearance. At city
scale, those judgments follow neighborhood reputation and its
statistical proxies. They rest on one aesthetic hierarchy across
unrelated questions, and an abstention does not remove them. This is a
representational and a distributive fairness problem.

The paper is a black-box audit with one white-box check. The white-box
check says: the judgment that the proxies validate is a quantity inside the
model that we can decode and steer, and the refusal suppresses it at the
output.

## Demarcation: which paper owns which result

Rule: a result is a contribution of 1 paper only. The other paper cites
it in the third person and gives no new number for it.

| Result | Owner | In the other paper |
|--------|-------|--------------------|
| 14 million pairwise judgments, TrueSkill2 rankings, 7 cases, 2 models | Paper 1 | cited as the source of the task and the proxies |
| Validation by proxy (unit, tract, NTA, CD); the overview figure | Paper 1 | cited |
| Prompt ablations, the robustness battery, positional bias, the effect of the camera face (compass direction) | Paper 1 | not used |
| Reasoning traces: the cue hierarchy, the inference categories, uncertainty and abstention, point of view | Paper 1 | 1 sentence of motivation |
| The single-image logit difference `c` and the layer-42 logit-lens readout against the proxies (`tables/single_image_read.tex`) | Paper 1 | cited as the external validity of the readout |
| Activation addition moves the answer to each end of the judgment in 6 of 7 cases (the headline result only) | Paper 1 | Paper 2 owns the geometry (the principal judgment direction, the pseudo-inverse steering direction, the gradient) and the image-token condition |
| The refused wealth judgment stays linearly decodable (AUC 0.73 on refused images); the base model expresses it (0.72 against 0.74) | Paper 2 | Paper 1 states it in 1 paragraph with the citation, no table |
| The Jacobian lens on image tokens, its criteria, the estimand, the registry | Paper 2 | Paper 1 gives the 4-sentence method and cites |
| The information flow across layers (image tokens 0-22, question tokens 23-30, last token 31-47) | Paper 2 | not used |
| The token-ablation map, the failure of gradient-based attribution, the fitted ablation predictor | Paper 2 | Paper 1 keeps 2 sentences: no first-order attribution map recovers the image tokens |
| Experiment 2 and Experiment 3 of the contrast tasks, the single-word logit-lens analysis, the correlation across all NTAs | Paper 2 | not used |
| The final-layer direct logit attribution; suppressed against removed across layers; the early-layer logit-lens analysis | Paper 2 | not used |
| The sparse autoencoders | Paper 2 | not used |

## Section plan and page budget

| Section | Pages | Content | State |
|---------|-------|---------|-------|
| Abstract | 0.2 | new; the commented abstract describes an older study | EMPTY |
| 1 Introduction | 1.0 | deployment, how-questions, the 2 fairness types, contributions | draft; 2 TODO |
| 2 Related work | 0.9 | cut from about 1.4 pages; add 1 paragraph of technical prior work (VLM urban perception; bias suppressed at the output after alignment) | draft; too long |
| 3 Methods | 1.6 | cases, data, models, task validity, TrueSkill2, proxies, ablations, traces, the white-box readout | draft; 2 TODO |
| 4.1 Validation by proxy and prompt sensitivity | 1.5 | cross-model agreement, the overview figure, abstention, ablations, robustness | draft; 4 TODO |
| 4.2 Reasoning traces | 1.3 | hierarchy, uncertainty, inference, point of view | draft; 3 captions |
| 4.3 The white-box check | 0.8 | the readout against the proxies; activation addition moves the judgment; the refused wealth judgment stays decodable (cited); 2 sentences on the image tokens | draft; revise |
| 5 Discussion | 0.7 | what a deployer must report; limits (1 city, 2 models, proxies are not truth) | EMPTY |

## Changes to the draft

1. Write the abstract and the discussion.
2. Correct the counts. The intro says 7 cases and 14 million pairs. The macros say `five` and `5,000,000`.
3. Change the title of 4.3. "One internal coordinate produces the judgment" is too strong. The vector `J^+ d` is the pseudo-inverse steering direction, a narrow direction. A separate principal judgment direction encodes the judgment (see `steer-coord.md`). Write "A decodable, steerable direction encodes the judgment".
4. Cut the image-token attribution paragraph to 2 sentences. Move the rest to the appendix, or drop it and cite Paper 2.
5. Close the "it remains unclear" sentence in the abstention paragraph with the cited result from Paper 2.
6. Fill the "Motivated by" column of the case table, or remove the column.
7. Explain the positional-bias adjustment (the 10 percent inverted pairs) and the camera-face result (the 2 open `\autoref{}`).
8. Write the 3 figure captions that say TODO.
9. Add technical related work: Gebru et al. and Naik et al. (PNAS 2017), Dubey et al. (ECCV 2016), Manvi et al. (ICML 2024), Hofmann et al. (Nature 2024), Bai et al. (PNAS 2025), Nwatu et al. (EMNLP 2023). See `white-box-related-work.md`.
10. Appendix: the targeting algorithm, the robustness tables, the word blocks, the CD table.

## What stays out

The logit-lens criteria, the information flow across layers, the fitted
ablation predictors, the 3 steering directions, the single-word logit-lens analysis,
the final-layer direct logit attribution, the early-layer logit-lens
analysis and the sparse autoencoders. A sentence that needs one of them
cites Paper 2.

## Schedule

| Date | Milestone |
|------|-----------|
| 2026-09-30 | counts, title of 4.3, the image-token paragraph cut, the TODO items closed |
| 2026-10-15 | abstract, discussion, related-work cut; full draft to the co-author |
| 2026-10-31 | figures final (captions, the overview figure, the forest plot); appendix complete |
| 2026-11-08 | the citation to Paper 2 resolves (the arXiv id, or an anonymous supplement) |
| 2026-11-15 | submit |

## Risks

| Risk | Answer |
|------|--------|
| The preprint of Paper 2 shows the same images and names the authors | cite it in the third person; check the CVPR 2027 preprint policy before the post |
| A reviewer asks for human ratings | the proxies and the Place Pulse line of work are the answer; say so in the limits |
| 2 models only | the smaller models fail the robustness battery; the table is in the appendix |
| The page budget | the related work and 4.3 are the 2 sections that can give space |

## Related

- `paper-plan-interp-posttraining.md`
- `master-results-table.md`, `proxy-overview.md`, `robustness-battery.md`, `trace-channels.md`, `cue-hierarchy.md`
- `pole-projection-proxy.md`, `single-image-steering.md`
