# A shared aesthetic direction (F10.27)

This report looks for 1 direction in the J++ Lens readout that gives the judgment of many
aesthetic prompts. The author asks for a shared axis of the aesthetic sense, with the
temperature sense of `cool` removed. The work belongs to Task 7 of the F10.26 plan
(`f10-14-prompt-analysis-plan.md`), but it uses its own script and output folder.

**Status (2026-11-02):** All 3 models have all 14 prompts (script 1.2.0). F10.28 adds the words and the ACS test.

**Changes:** Version 1.2.0 uses Bonferroni in place of Holm. With the first 30 held-out tests, this
changed 1 side result (Qwen3.5-9B `place ordinary`, p .012 to .055).

**Correction (script 1.1.0):** Version 1.0.0 also removed the sense score from the `cool` readout
at the noun token. That step has no purpose there: the model has not read `cool`, and the 4
`place` readouts are the same. Version 1.1.0 removes the score only after the noun. The noun
results below are from 1.1.0. The 1.0.0 output is in `figures/f10/shared/{model}_place_v1.0.0/`.

## Main result

| Token | What the shared direction is | Signs |
|-------|------------------------------|-------|
| Noun (`place`, `scene`), before the adjective | An aesthetic axis of the image. Up: flowers, blossoms, beautiful, elegant, exquisite, splendid. Down: crude, rusted, scrap, wreck, dumpster, worse, appalling | + for `cool`, `nice`, `fancy`, `pretty`. - for `ugly`. About 0 (Gemma) or weak + (both Qwen) for `ordinary`. - for `urban` |
| Adjective | A forecast of the answer. Up: yes, confirm, indeed, excellent. Down: none, nope, neither, nowhere, useless, meaningless | + for all prompts, `ordinary`, `urban` and `ugly` too |

1. A shared aesthetic direction exists. It is in the readout at the noun, before the model
   reads the adjective. Thus it is a property of the image, not of the word.
2. The direction generalizes to prompts that the fit did not use, with the noun `scene`.
   It reverses for `ugly` in all 3 models (r -.38 to -.64).
3. `ordinary` is not the opposite of the aesthetic words. Its r is about 0 in Gemma-4-31B, and
   weak and positive in the 2 Qwen models (.18 to .38).
4. `urban` loads against the axis in all 3 models (r -.28 to -.51). An urban scene is less
   "aesthetic" on this axis.
5. At the adjective, the shared direction is mostly the forecast of the answer. It predicts
   yes to `ordinary`, `urban` and, in both Qwen models, `ugly` (r .45 to .86). When we remove a fixed yes-minus-no word score, the
   r falls from .8 to .9 to about 0 to .3.
6. The temperature sense of `cool` is not on the noun-token axis. The word `cool` comes after
   the noun, thus it cannot enter. At the adjective, the temperature score hides the judgment
   of `cool`. When we remove the score, the r of `cool` goes up (see "Temperature").

## Method

Script: `scripts/f10/hop_shared.py` (version 1.2.0). Output: `figures/f10/shared/{model}_{position}/`.
Input: the NTA mean readouts of F10.26 stage 1 (`data/f10/prompts/{store}/nta_{position}.npy`).

| Step | Rule |
|------|------|
| Judgment vector | For each of the about 50,000 lens words: r over the 197 NTAs of the readout with the answer margin |
| Fit set | `cool`, `nice`, `fancy`, `pretty`, all with `place` |
| Shared direction | The mean of the 4 unit judgment vectors |
| Folds | 5 folds of NTAs. Each vector is fit on 4 folds and scores the 5th fold |
| Fit prompts | Each is scored by the direction of the other 3 (leave 1 prompt out) |
| Held-out prompts | `ordinary`, `ugly`, `urban` with `place`, and the 7 prompts with `scene` |
| Sense removal | In the `cool` readouts only, and not at the noun token: regress each word on the NTA sense score (social words minus temperature words, F10.17) |
| Polarity removal | In all readouts: regress each word on a fixed score of yes words minus no words |
| Test layer | The layer with the best mean r of the 4 fit prompts. The held-out prompts do not choose it |
| Test | r over NTAs, t test, Bonferroni over the held-out prompts at the test layer (p times the number of tests, 10 for each model and token) |

The variants are `base` (no removal), `main` (sense removal) and `main_polarity` (both removals).

**Note:** At the noun token, all `place` stores hold the same readout, because attention is
causal. Thus for these prompts, 1 image score is tested against 7 different answers.

**Note:** Gemma-4-31B gives yes or no to `ugly` in only a few images. The test uses the NTAs
with a margin: 19 for `place ugly` and 66 for `scene ugly`.

**Note:** An earlier run used the Gemma `place pretty` summary before it was complete, with
`--allow-partial`. The script takes such a store only when its noun readout equals that of `cool`
(gap 0.0). The final runs use complete summaries only.

## Results at the noun token

The values are r at the test layer, in the `main_polarity` variant. The p values are Bonferroni
p over the 10 held-out prompts. They are below .001 where the table gives no p.

| Prompt | Role | Gemma-4-31B, L8 | Qwen3.5-9B, L28 | Qwen3.6-27B, L60 |
|--------|------|-----------------|-----------------|------------------|
| place cool | fit, leave 1 out | .56 | .42 | .22 |
| place nice | fit, leave 1 out | .51 | .52 | .53 |
| place fancy | fit, leave 1 out | .29 | .33 | .42 |
| place pretty | fit, leave 1 out | .62 | .63 | .45 |
| scene cool | held out | .48 | .12 (p .88) | .14 (p .55) |
| scene nice | held out | .60 | .62 | .66 |
| scene fancy | held out | .37 | .33 | .40 |
| scene pretty | held out | .65 | .66 | .71 |
| place ugly | held out | -.48 (19 NTAs, p .36) | -.52 | -.62 |
| scene ugly | held out | -.38 (66 NTAs, p .02) | -.45 | -.64 |
| place ordinary | held out | -.12 (p .99) | .19 (p .06) | .18 (p .09) |
| scene ordinary | held out | .05 (p 1.00) | .36 | .38 |
| place urban | held out | -.28 | -.32 | -.29 |
| scene urban | held out | -.36 | -.47 | -.51 |

The signs hold at most layers:
- Gemma-4-31B: `ugly` is negative from L4 to L56. `place ordinary` stays between -.15 and .15.
- Qwen3.5-9B: `scene ugly` is negative at all 8 layers (-.27 to -.59). `ordinary` is between 0 and .53.
- Qwen3.5-9B has 2 test layers of almost the same score: L28 (mean leave-1-out r .499) and L12 (.495).
- `scene cool` is weak in both Qwen models. With the noun `scene`, `cool` follows the axis less.

The removal of the yes-no score changes little at the noun token, because the model has not
read the question yet.

Words at the 2 ends of the direction (`words.csv`):

| Model, layer | Up | Down |
|--------------|----|------|
| Gemma-4-31B, L24 | flowers, blossoms, petals, beautiful, lavender, cherished, elegant, garland | crud, pulp, warring, disorganized, carcasses, crude, unreliable, defunct |
| Gemma-4-31B, L32 | ornaments, exquisite, beautiful, gracefully, blossoms, heavenly, luscious, magnolia | rusted, scrap, scraping, welded, disorganized, rusty, struggling, unfriendly |
| Qwen3.5-9B, L16 | exquisite, wonderful, elegant, beautifully, splendid, stunning, majestic, gorgeous | worse, undesirable, worst, unhealthy, wreck, dumpster, inefficient, poorly |
| Qwen3.5-9B, L24 | magnificent, verdure, majestic, golden, bouquet, triumph | salvage, disposal, negligence, dumpster, dysfunctional, condemned, appalling, useless |
| Qwen3.5-9B, L28 (test layer) | medal, elite, priceless, ambassadors, majestic, extraordinary | incompetent, offenders, faulty, salvage, misuse, arson, asbestos, downgrade |
| Qwen3.6-27B, L24 | elegance, beautifully, wonderful, elegant, enchant | recycl, industrielle, warehouse, warehouses, supervisor, factory, worthless |
| Qwen3.6-27B, L40 | wonderful, enchant, marvelous, attractive, splendid, adore | junk, offenders, obscene, worthless, garbage, bad, factories, salvage |
| Qwen3.6-27B, L60 (test layer) | admir, magnifique, merits, cultiv | hopeless, defective, misguided, broken, faulty, unreliable, contaminated, dumpster |

At the Qwen3.5-9B test layer, the axis is closer to status and order than to beauty.

**Limit:** The down end also holds words of work, industry and income in all 3 models (Gemma:
`trabalhadores`, `miners`, `welder`, `laboral`. Qwen3.5-9B: `unemployment`, `poorer`, `jobs`.
Qwen3.6-27B: `factory`, `warehouse`, `supervisor`).
Qwen3.5-9B L16 also holds `negro` (Spanish and Italian for "black") at the down end.
This report does not test these words. The aim is the mechanism, not a harm audit, but a
reviewer can ask about them. Do not quote a word list in the paper without this context.

## Results at the adjective token

The values are r at the test layer: `main`, then after polarity removal (`main_polarity`).

| Prompt | Gemma-4-31B, L36 | Qwen3.5-9B, L16 | Qwen3.6-27B, L32 |
|--------|------------------|-----------------|------------------|
| place cool | .82, .29 | .87, -.15 | .66, .08 |
| place nice | .81, .15 | .96, -.05 | .92, .21 |
| place fancy | .78, .08 | .97, .04 | .89, .24 |
| place pretty | .89, .32 | .96, -.02 | .95, .18 |
| place ordinary (expected -) | .39, .30 | .92, .31 | .64, .42 |
| place ugly (expected -) | .29 (19 NTAs), .04 | .76, .46 | .45, .45 |
| place urban | .79, .03 | .97, .19 | .93, -.02 |
| scene nice | .90, .18 | .96, -.05 | .91, .11 |
| scene ordinary (expected -) | .56, .40 | .92, .18 | .50, .36 |
| scene ugly (expected -) | -.06 (66 NTAs), -.32 | .86, .37 | .59, .36 |
| scene urban | .83, .05 | .98, .20 | .95, .03 |

In both Qwen models, the direction predicts yes to `ugly` (r .45 to .86), the wrong sign for an
aesthetic axis. After polarity removal, `ugly` and `ordinary` stay positive (.18 to .46): the
fixed word lists do not remove all of the forecast. At L16, the shared score correlates .93 to .99 with the yes-minus-no word score.
The ends of the direction are answer words in many languages: `conferma`, `yes`, `oui`, `excellent`
against `aucune`, `nope`, `nenhuma`, `meaningless`, `worthless`.

## Temperature

| Test | Result |
|------|--------|
| Noun token | The model has not read `cool`, thus no sense removal is done (1.1.0). The 4 fit prompts share 1 readout there, and averaging cannot add a word sense |
| Why not let the average remove it | At the adjective, the problem is in the scoring step, not in the fit. The sense axis holds most of the NTA variance of the `cool` readout (63% at Gemma L24). A small weight of the direction on those words gives a score that follows the axis, not the answer. The average dilutes the weights, but it cannot change the readout that is scored |
| Adjective token, first fit (`cool`, `nice`, `ordinary`; Gemma-4-31B, L24) | The shared score in the `cool` readout has r -.88 with the sense score. Leave-1-out r of `cool` is .03. After removal it is .53 |
| Adjective token, same fit, L36 and L48 | .61 to .73, and .08 to .29 |
| Removal in all prompts (not only `cool`) | `fancy` falls from .78 to .43 at L44. The same score carries real scene content for the other words |

Thus remove the sense score from `cool` only. For the other words, the social-to-temperature
score is the urban-to-green content of the scene. The `nice` margin has r -.42 with it, and the
`ugly` margin +.55 (Gemma-4-31B, L44).

## An axis of the visual evidence (F10.27b)

This section asks if a readout holds the reasons for the judgment, not only the verdict.

Script: `scripts/f10/hop_evidence.py` (version 1.0.0). Output: `figures/f10/shared/evidence_{model}/`.
The traces exist for Gemma-4-31B and Qwen3.5-9B only.

| Item | Rule |
|------|------|
| Target | For each NTA: the mean net valence of the `visual_evidence` spans that the same model cites in its `cool` traces with thinking on (good +1, bad -1, neutral 0) |
| Readout | The one-word `cool` prompt, at the noun, the adjective, `?` and the last token |
| Direction | The method of `hop_shared.py`, with the evidence valence in place of the margin, out of fold |
| Controls | The one-word answer margin, and the verdict of the trace itself |
| Test | t test of the partial r, Bonferroni over all cells of 1 model (60 for Gemma, 32 for Qwen) |

The evidence valence has r .71 (Gemma) and .69 (Qwen) with the one-word margin. Thus most of
it is the verdict. The table gives the partial r of the evidence score given both verdicts.

| Token | Gemma-4-31B | Qwen3.5-9B |
|-------|-------------|------------|
| Noun | .15 to .31 from L8 to L56 | .17 to .21 from L8 to L28 |
| Adjective | .31 to .41 from L8 to L56 (13 of 15 layers pass) | .33 to .41 from L4 to L28 (7 of 8 layers pass) |
| `?` | .17 to .36 from L8 to L56 | .37 to .41 from L12 to L28 |
| Last token | .16 to .39 from L4 to L56 | .09 to .32 from L4 to L28 |

1. At the adjective token, the readout holds a part of the evidence valence that neither verdict
   explains. It is the "why" part: how good the cited evidence looks, apart from the conclusion.
2. The words of this direction are not readable. In Gemma at L44 the ends are `indeed`, `sufficiently`
   against `aucune`, `tragedy`, `dangerous`, `troubling`. No end names a visual object.
3. The readable aesthetic words are on the noun-token shared direction. That direction has r .30
   to .60 with the evidence valence, but its partial r given the margin is only .05 to .23.

**Limit:** The traces are separate runs with thinking on. An extraction model gives the valence labels.

F10.12 found the content of the arguments (the words of observations, evidence, criteria) in the
readout, with fixed word lists. It peaked at the tokens before free text (`thought`, `<|channel>`),
which the F10.26 summaries do not hold. A test of the evidence axis at those tokens needs the full stores.

## The words of the direction, and the ACS gradients (F10.28)

Script: `scripts/f10/hop_hierarchy.py` (1.2.0). Output: `figures/f10/hierarchy/` and
`notebooks/facct/figures/f10/shared_direction_words{,_stacked}.{pdf,png,json}`.

### Words

We z-score the direction at each layer and take the mean over the layers. Thus no person
selects a layer. The 20 words at each end are in `shared_direction_words.csv`.

| Model | Up (examples) | Down (examples) |
|-------|---------------|-----------------|
| Gemma-4-31B | flower, beautiful, bouquet, cherished, ornaments | industrial, welding, smelting, trabajadores, contaminants |
| Qwen3.5-9B | stunning, exquisite, majestic, elegant, queen, regal | unemployed, warehouses, dumpsters, abuses, slaughter, `hitam` |
| Qwen3.6-27B | exquisite, beautifully, stunning, elegant, splendid | factory, warehouses, gang, thugs, dumpster, job |

`hitam` is Indonesian for "black". The paper quotes it.

**Rejected:** We also compared the direction with the net valence of the restaurant cues of
F9 (Spearman over about 150 cues that are 1 lens word). rho is -.01 to .27 over the layers, and
-.01 for Gemma at its test layer. This is weak, thus the paper does not use it.

### ACS gradients of the answers (14 prompts)

The margin of an `unable` answer is empty (D167). Thus the margin cannot order all of the images
for Gemma-4-31B. For each image, the script uses the probabilities of the 4 yes and the 4 no forms
over the full vocabulary (`data/f10/hop_logit/{store}/answers.parquet`):

| Score | Formula | Meaning |
|-------|---------|---------|
| E | P(yes) - P(no) | The expected answer, yes = +1, unsure = 0, no = -1. An `unable` answer sits near 0 |
| P_unsure | 1 - P(yes) - P(no) | The mass on all other tokens. For `place nice`, 278 of 295 `unable` answers start with `Subject` |

Checks (Gemma): the median P_unsure is .85 to 1.00 for `unable` answers and about 0 for the other
answers. Among the answered images, rho(E, margin) is .96 (`nice`, `pretty`) and 1.00 (`cool`).

Test: Spearman rho over NTAs of the NTA mean with 4 ACS features. E for 3 models and 14 prompts,
and P_unsure for Gemma where 5% or more of the answers are `unable` (`place nice`, `place pretty`,
`place ugly`, `scene nice`, `scene ugly`). Bonferroni over 188 tests: 81 pass.
Output: `ses_prompts.csv` and the table `notebooks/facct/tables/f10_prompt_ses.tex`.

| Result | Values |
|--------|--------|
| E of `cool`, `nice`, `pretty`, `fancy` rises with income | 23 of 24 model and prompt pairs pass, rho .16 to .46 |
| The same, with bachelor's and White share | 16 and 14 of 24 pass |
| E of `ugly` falls with income | All 6 pairs, rho -.44 to -.34 |
| `ordinary`, `urban` | No test passes |
| Gemma P_unsure falls with income | `nice` -.36 (scene -.38), `ugly` -.39 (scene -.44) |
| Gemma P_unsure of `place ugly` rises with Black share | rho +.28. The only Black-share test that passes |
| Black share and the aesthetic E | All negative (-.19 to -.04). No test passes |

On `scene ugly`, Gemma puts almost no mass on yes. Thus E = -P(no) and P_unsure = 1 - P(no), and
the 2 rows have the same rho.

### ACS gradients of the readout and the traces

Measures: `shared_direction` (the NTA mean readout at `place`, test layer, projected on the
direction) and `evidence_valence` (the net valence of the `visual_evidence` spans of the `cool`
traces, Gemma and Qwen3.5-9B). Spearman rho, Bonferroni over 20 tests: 10 pass. Version 1.2.0
removes the yes rates from this family, because the 14-prompt table holds them.

| Result | Values |
|--------|--------|
| `shared_direction` rises with income | Gemma .32, Qwen3.5-9B .25. Gemma bachelor's .30, White .23 |
| `evidence_valence` rises with income, bachelor's, White share | Gemma .32, .26, .24. Qwen3.5-9B .25, .28, .26 |
| Black share | All negative. No test passes |

**Limit:** Neighboring NTAs are not independent. The p values assume that they are.

### Paper

Section 4.2 uses Figure 3a (the words), the table `tab:prompt-ses`, and the readout and trace
results. Section 4.3 starts with the table. The rank figure of Gemma (`fig:ranks`) is in Appendix B.
Methods gives E, P_unsure and the 2 test families.

## What this changes for the paper

| Paper text now | Result |
|----------------|--------|
| "At 'cool' the image picks the sense of the word" (Results, part 3 of the summary; Figure 1, callout 4) | The sense axis is a property of `cool` only. The noun-token readout holds an aesthetic axis that is shared by the aesthetic words, and that is free of temperature |
| "'nice' and 'ordinary' pass the same test at no layer" | That test uses the `cool` word lists. It cannot find a sense of another word |
| "Appraisal, before 'cool'" (callout 2: aesthetic words track the verdict at the noun) | Holds, and generalizes: 1 direction fit on 3 words predicts the 4th word and the `scene` prompts, and reverses for `ugly` |

## Commands

```bash
.venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py fit --model gemma-4-31b --position place
.venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py fit --model gemma-4-31b --position after_place
.venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py figure
```

| File in `figures/f10/shared/{model}_{position}/` | Content |
|--------------------------------------------------|---------|
| `heldout.csv` | r, p and Bonferroni p for each layer, variant and prompt. Also r of the score with the sense and the polarity scores |
| `cosine.csv` | Cosines of the judgment vectors of all prompts, at each layer |
| `words.csv` | The 30 words at each end of the direction, at each layer |
| `direction.npz` | The shared direction at each layer, and the vocab |
| `sense_margin.csv` | r of each margin with its own sense score |
| `summary.json` | Test layer, fit set, held-out, missing and partial stores, the yes and no words |

The figure is `figures/f10/shared/shared_direction.png`, with the caption in `shared_direction.json`.

## Limits

| Limit | Effect |
|-------|--------|
| NTAs, not images | 197 NTA means. The images inside an NTA are not independent |
| Same images | The held-out prompts use the same 591 images. A held-out prompt is a new question, not a new sample |
| Margins correlate | The `cool`, `nice` and `pretty` margins correlate at r .65 to .81. A held-out r shows a shared direction. It does not show a shared mechanism |
| Fixed word lists | The yes and no lists, and the temperature and social lists, are our choice |
| Test layer | 1 layer for each model and token, chosen on the fit prompts |
