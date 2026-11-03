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

Section 4.2 used the words as Figure 3 until 2026-11-03; now Figure 3 is the restaurant cue block, and the words figure is in Appendix B (`app:direction-words`) with the full word description. Section 4.2 also uses the readout and trace results. The table
`tab:prompt-ses` and the rank figure of Gemma (`fig:ranks`) are in Appendix B. Methods gives E,
P_unsure and the 2 test families. Appendix A ("The shared direction and its tests") gives the fit,
the transfer, the partial test and the reliability.

## Borough words that 3 models share (F10.30)

The old Figure 2 (`fig:borough-layers`) gave the C3 words of Gemma-4-31B only. The top-50 words
of C3 do not agree between models (Jaccard about 0). F10.30 keeps only the words that all 3
models raise.

| Step | Rule |
|------|------|
| Contrast | C3 (`$C_L$` in the paper) as a Welch t over the 197 NTA means at the `cool` token: 1 borough against the other 4, for each word and saved layer |
| Bands | The mean t over the saved layers in each quarter of relative depth (layer / number of layers: 60, 32, 64) |
| Words | Join the 3 vocabularies on the word string. Keep lowercase words of 4 or more letters (33,965) |
| Score | z-score the t in each model. The score of a word is its lowest z of the 3 models |

The full t vectors agree between models. Median Spearman rho over pairs and boroughs: .15 (depth
0 to .25), .36 (.25 to .50), .26 (.50 to .75), .21 (.75 to 1).

| Borough | Shared words (from the figure) |
|---------|--------------------------------|
| Manhattan | marketed, secured, allure, venues, fama |
| Brooklyn | comunidade, socially, sociale, barrio, sidewalk, street, marijuana, prostitution |
| Bronx | music, playground, recreation, then disappointed, unsafe, dangerous, berbahaya (Indonesian "dangerous") |
| Queens, Staten Island | Few shared words. Science words and word pieces in other languages |

**Limits.** We did not test single words: a word is in the figure because it ranks high in each
model, not because it passes a test. The rho values between models are descriptive, because the
words are not independent.

| Item | Path |
|------|------|
| Script | `scripts/f10/plot_borough_shared.py` 1.0.0 |
| Figure | `figures/f10/mechanism/mech_borough_shared.{pdf,png,json}` |
| Words | `figures/f10/mechanism/borough_shared_words.csv` (top 20 for each borough and band, with the t of each model) |
| Agreement | `figures/f10/mechanism/borough_shared_agreement.csv` |
| Paper copy | `paper/figures/facct/mech_borough_shared.pdf` (manifest entry in `artificial-aesthetic/manifest/assets.toml`) |

## Each standard in the J-space (F10.31, Figure 2)

Figure 2 (`fig:standards`) gives 1 row for each of the 7 adjectives (noun `place`), in words that
the 3 models share. It takes the place of the borough figure (F10.30), which is now in Appendix B
(`app:borough-words`).

**Figure 1** (`fig:prompt-readout`) is the TikZ prompt figure again (author decision, 2026-11-02).
Its 8 callouts now give results that hold in the 3 models, with values in the order Gemma-4-31B
(Qwen3.5-9B, Qwen3.6-27B) and counts over the 42 cells of F10.26. They have no layer tags and no
Gemma-only word lists. Sources:

| Callout | Result | Source |
|---------|--------|--------|
| 1 Scene | Mean-patch R2 .82 (.75, .56), at least .50 in 39 of 42 cells | F10.24, F10.26 task 2 |
| 2 One aesthetic direction | Words that the 3 models share at the noun (up: elegant, exquisite, majestic; down: crude, salvage, garbage). Leave-1-out r and the reversal for `ugly` | F10.27; the words are a new computation with the F10.30 rule on `direction.npz` (not in a CSV) |
| 3 The standard, embedded | Conception words with no image; embedded 42 of 42 | F10.31, F10.26 task 2 |
| 4 A forecast of the answer | r .82 (.87, .66) at the adjective; yes to `ordinary` and `urban` (.39 to .97) | F10.27 |
| 5 The question adds a step | Gain .07 to .12 to .13 (.08, .18, .18; .07, .19, .21); 41 of 42 cells | F10.24, F10.26 task 2 |
| 6 The instruction adds little | Gain flat through the instruction, up at the last token | F10.7 |
| 7 One direction for three models | At `place`, the direction of the other 2 models reads out the answers: r .47, .49 (.74, .79; .67, .62) on `nice`, `pretty`, near or above the own direction. Passes Bonferroni in 6 (14, 11) of 14 prompts. No lead line, because the callout sits far from `place` | F10.29 Q2, `transfer_test_layer.csv` |
| 8 Verdict | r .99 (.98, .92) on `cool`; r at least .92 in 41 of 42 cells | `t2_grid.csv`, F10.29 Q6 |

The Qwen3.5-9B verdict value is .98 with the 8-form margin (the old figure gave .96, from the
4-form margin).

| Column | Rule |
|--------|------|
| Conception, no image | The no-image readout at the adjective token, minus the mean no-image readout of the 7 adjectives. The figure leaves out the adjective and its forms |
| Toward yes, toward no | At the `place` token, before the model reads the adjective, r over the 197 NTAs of the NTA mean readout of each word with the NTA mean E = P(yes) - P(no) of that prompt. The 7 prompts have the same activations at this token. Only the answer differs |
| Grey numbers | `embedded_mid` of F10.26 (`t2_grid.csv`): of the top 100 words at the adjective, the number that are also in the no-image top 100 |

All values are means over the saved layers at relative depth .25 to .75. Words are joined on the
string, z-scored in each model, and ranked by the lowest z of the 3 models (the F10.30 rule).

| Agreement between 2 models, Spearman rho over 33,965 words | Min | Median | Max |
|------------------------------------------------------------|-----|--------|-----|
| Conception | .49 | .54 | .79 |
| Scene words (toward yes or no) | .22 | .46 | .63 |

`ordinary` has the lowest scene agreement (median .23). The other 6 adjectives are .41 to .50.

**Why not the adjective token for the yes and no words.** At the adjective, the words that go with
E are answer words (`nope`, `none`) and function words in other languages. This agrees with
F10.27: at the adjective, the direction is a forecast of the answer.

**Limits.** We did not test single words. Gemma-4-31B declines `ugly` on 97% of the images, thus
its E on `ugly` is near 0 for most images. The `cool` temperature sense is not in the figure or in
the body of the paper (author decision, 2026-11-02).

| Item | Path |
|------|------|
| Script | `scripts/f10/plot_standards.py` 1.0.0 |
| Figure | `figures/f10/mechanism/mech_standards.{pdf,png,json}` |
| Words | `figures/f10/mechanism/standards_words.csv` |
| Agreement | `figures/f10/mechanism/standards_agreement.csv` |

## Paper edits for the 14-page limit (2026-11-02)

The body ends at the bottom of page 14 (manuscript class). No lines are free.

| Change | Where now |
|--------|-----------|
| Figure 1 panel b (the layer bands) | Appendix B, `fig:prompt-layers`, with the band rules from Methods |
| Figure 1 panel a | Qwen3.6-27B bars at `place`, `cool`, `?` (gain .072, .191, .206) and its values in the callouts |
| Figure 2 | F10.31 (each standard, 3 models). F10.30 (borough words) is in Appendix B |
| Figure 3 (restaurant cue block) | Section 4.2, `fig:restaurant-inference-block`, a side figure since 2026-11-03. The words figure moved to Appendix B |
| Figure 4 (inference kinds) | Appendix B |
| Table 1 (`tab:prompt-ses`) | Appendix B, with the reliability result (F10.29 Q3) |
| Added to Results | F10.29 Q1 (partial test), F10.29 Q2 (transfer), C1.10 (name x image), C5.3 (image controls), C5.6 (H9) |
| Removed from Results | Greenery with % Black (now % White), the inference kinds other than brand knowledge, the 34,072 wealth spans, the Gemma-only Bronx and Brooklyn lists, the layer numbers of "Along the prompt", the `model` C2 list |
| Methods fixes | The prompt reads "Answer with one word", and the adjective list holds `fancy` |

The removed text is in `paper/sec/scratch.tex`, word for word.

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

## Words along the prompt for 3 models (F10.32)

Appendix B ("Along the prompt") now gives words that the 3 models share, not Gemma words. Script:
`scripts/f10/prompt_words_shared.py`. Output: `figures/f10/mechanism/prompt_shared_words.csv`.
The script uses the `cool` store, the tokens `place`, `?` and the last token, and quarters of
relative depth. It gives 2 scores: the mean readout, and the mean readout minus the no-image
readout. It ranks by the lowest z over the 3 models (the F10.30 rule).

| Token | Quarter | Result |
|-------|---------|--------|
| `place` | 1 | Function words (still, even, here) |
| `place` | 2 | Place words: nearby, downtown, surroundings, location, town |
| `place` | 3 | The image adds the setting: residential, urban, metropolitan, vicinity, suburbs. None is in the no-image top 100 of a model |
| `?` | 3 | The image adds words of judgment: attractiveness, aesthetic, pleasant, fashionable, trendy, artistic |
| `?` | 4 | subjective, depends |
| last | 2 to 3 | answer, answers, opinion |
| last | 4 | yeah, nope, affirmative |

The image score in the first quarter gives multilingual noise, thus the paper does not use it.

The appendix layer figure (`fig:prompt-layers`) has no "sense split" row now, and the text has
no sense rule. The figure gives Gemma-4-31B and Qwen3.5-9B only, because F10 did not compute
the scene band for Qwen3.6-27B.

## The teaser figure: J++ Lens words on image patches (F10.33)

The paper has a teaser figure again (`fig:teaser`, in `main.tex`). It has the form of the old
R-lens figure (`notebooks/facct/_rq.py`, `plot_patch_words`), in 1 row of 5 images. The old
figure used the R-lens on Gemma-4-12B. The new figure uses the J++ Lens on Gemma-4-31B.

| Item | Value |
|------|-------|
| Script | `scripts/f10/plot_patch_teaser.py` (1.3.0) |
| Output | `figures/f10/mechanism/mech_patch_teaser.{pdf,png,json}`, `patch_teaser_words.csv`, `patch_teaser_images.csv` |
| Data | `data/f10/hop_all/gemma-4-31b/patches/ids-*.npy`: the top 30 token ids of each of 256 patches (16 x 16, row-major), 15 layers |
| Pole words | The 300 words (lowercase, 4 or more letters) at each end of the Gemma-4-31B direction at `place` (F10.27 `direction.npz`, z-scored at each layer, then the mean over layers, as in F10.28) |
| Hit | A patch holds a pole word in its top 30 at a layer of relative depth .25 to .75 (L16 to L44) |
| Images | In each borough, rank the images with the answer of the panel by the signed (up - down) patch count. Yes: Manhattan, Staten Island. No: Bronx, Queens, Brooklyn. The panel shows a street scene from the top 15, picked by eye (`PICKS` in the script). The top images of 4 boroughs were a park or warehouse yards (author decisions, 2026-11-03) |
| Word of a block | Score: the share of the (patch, layer) top-30 lists of the block that hold the word, minus the share over all images. Pole words go to blocks greedily by score. A word shows at most 2 times in an image, and only words of 11 letters or fewer show |

**Result.** The patches do not see the question, but their pole words follow the answer. Over the
591 images, the number of patches that hold an up word correlates with the answer margin at
r = .46 (p < 1e-31), and the number that hold a down word at r = -.47 (Pearson). The caption
gives these values.

| Borough | Image | NTA | Answer | Margin | Up patches | Down patches | Rank |
|---------|-------|-----|--------|--------|------------|--------------|------|
| Manhattan | W0ECNCTD | Upper West Side (Central) | yes | 15.9 | 97 | 11 | 2 of 64 |
| Staten Island | W0EL0RUQ | West New Brighton-Silver Lake-Grymes Hill | yes | 17.0 | 122 | 3 | 1 of 15 |
| Bronx | W0EDUW80 | Allerton | no | -4.4 | 53 | 88 | 11 of 84 |
| Queens | W0EA3B36 | Astoria (North)-Ditmars-Steinway | no | -15.2 | 35 | 83 | 14 of 118 |
| Brooklyn | W0EH7YSV | Sheepshead Bay-Manhattan Beach-Gerritsen Beach | no | -9.6 | 29 | 80 | 10 of 103 |

The first version (1.2.0) showed the top image of each borough: a park in SoHo, and warehouse
or truck yards in Mott Haven, Ozone Park and Carroll Gardens. The street scenes hold fewer
pole words, and their "no" images also hold some up words.

The pole words that patches hold most are concrete. Up: gardens, ornament, balkon, jardins,
pastel, decorations, arched, hydrangea. Down: truck, trucks, forklift, warehouse, rusted, tanker,
bulldozer, filth. Thus the down end of the direction reads trucks and industry in the scene.

The raw top-1 word of a patch is not usable. It is "own" or a punctuation id in most patches,
the same problem that the R-lens had. A test on 1 image (`W0EFVJ3D`) shows that the grid is
row-major: the sky blocks give "clouds", and the bicycle block gives "bicycle".

The selection is for a teaser, and it does not give a typical image. The figure is drawn at the
text width (430 pt), so it prints at 100%.

**Warning:** the 8 images of the old figure are not in the 591. To use them, run the lens again
on a GPU.

Qwen3.5-9B and Qwen3.6-27B have the same patch files (1024 patches, 32 x 32).

### The pole words by layer (F10.33b, 2026-11-03)

Script `scripts/f10/patch_layers.py` (1.0.0). Outputs `figures/f10/mechanism/patch_layers.csv`,
`patch_layer_words.csv`, `patch_layer_teaser.csv`. The teaser counts a hit at any layer of the
band L16 to L44. This table gives each saved layer on its own: the mean number of patches (of 256)
with an up or a down word, the Pearson r of the per-image counts with the `cool` margin, and the
words that the patches hold most.

| Layer | Up patches | Down patches | r up | r down | Up words | Down words |
|-------|------------|--------------|------|--------|----------|------------|
| 8 | 2 | 5 | .16 | -.27 | antlers, floral, alpine | trucker, truck, tanker |
| 12 | 8 | 9 | .23 | -.39 | balkon, hydrangea, antlers, ornament | truck, trucker, tanker, warehouse |
| 16 | 12 | 16 | .31 | -.40 | balkon, antlers, gardens, jardins | truck, trucks, tanker, warehouse |
| 24 | 14 | 21 | .36 | -.42 | gardens, jardins, balkon | truck, trucks, forklift, warehouse |
| 32 | 34 | 8 | .44 | -.37 | gardens, ornament, decorations, pastel | truck, trucks, warehouse, filth |
| 36 | 25 | 8 | .43 | -.36 | gardens, balkon, ornament, pastel | truck, rusted, warehouse, forklift |
| 40 | 14 | 7 | .49 | -.39 | pastel, arched, gardens, turquoise | forklift, rusted, truck, tarp |
| 44 | 11 | 4 | .44 | -.33 | arched, hydrangea, turquoise, emerald | truck, metal, warehouse |
| 48 | 5 | 6 | .40 | -.33 | leaf, balkon, arched | metal, industrial, truck |
| 56 | 8 | 5 | .40 | -.33 | leaf, arched, turquoise | metal, truck, industrial |
| band L16-44 | 58 | 33 | .46 | -.47 | | |

1. The words move with the depth. Up to L24 the patches hold objects: trucks, tankers and
   forklifts at the down end, antlers, balconies and gardens at the up end. At L32 to L36 the
   up end turns to ornament and decoration, and the up count peaks (34 patches). From L40 the
   words are colours and materials: pastel, arched, turquoise, leaf against metal, rusted, tarp,
   industrial.
2. The down count peaks early (L24, 21 patches) and the up count late (L32, 34 patches). Thus
   "middle layers" mixes 2 regimes: the down words of the band come mostly from L16 to L28, and
   the up words from L32 to L36.
3. The hit patches change from layer to layer: the Jaccard overlap of the hit mask with the next
   saved layer is .14 to .47 (up) and .27 to .50 (down). The same patch seldom keeps its word.
4. The correlation with the answer is flat over the band (r up .31 to .49, r down -.33 to -.42),
   and the band count adds little (.46, -.47). One layer gives nearly the same test as the band.
5. In the 5 teaser images, the "no" images hold their down words at L16 to L24 (Bronx 51 to 70
   patches, Brooklyn 50 to 67) and the "yes" images their up words at L32 (Manhattan 67,
   Staten Island 83).

The figure draws the words of the whole band, so a block can show an L16 truck beside an L32
ornament. A figure at 1 layer, or 1 layer for each end, would be easier to state.

### The severe words on innocuous scenes (F10.33c, 2026-11-03)

The author did not want the object words (truck, warehouse) of the top 300 at the down end. The
teaser now shows the severe words: the social words that fire on innocuous scenes.

Script `scripts/f10/patch_severe.py` (1.0.0). It keeps a hand list of 92 severe candidates
(criminal, sewage, filth, slaughter, stench, gangs, trafficking, rape, abuse, trash) and finds 79
of them within rank 2,000 of the down end. For each image it counts the patches with a severe
word at each layer, and it draws a contact sheet for each borough
(`figures/f10/mechanism/severe_sheet_*.png`). Outputs `patch_severe_words.csv` (hits by layer)
and `patch_severe_images.csv` (counts by image).

| Layer | Severe hits | Images with a hit (of 591) | Top severe words |
|-------|-------------|----------------------------|------------------|
| 12 | 660 | 200 | sewage, sludge, criminal, rotten, gangs, stench |
| 16 | 1,305 | 448 | sewage, slaughter, filth, punk, gangs, dumped |
| 24 | 1,184 | 261 | sewage, vandalism, punk, criminal, trash |
| 32 | 1,702 | 357 | filth, vandalism, sewage, trash, criminal |
| 36 | 1,466 | 351 | vandalism, sewage, trash, filth, criminal |
| 40 | 644 | 223 | sewage, vandalism, deteriorating, filth, unclean |
| 52 | 1,263 | 325 | trash, garbage, waste, trafficking, dirty |
| 56 | 1,096 | 351 | trash, garbage, dirty, trafficking, brutality |

1. Severe words fire on 571 of the 591 images. The count of severe patches over all layers
   correlates with the `cool` margin at r = -.28.
2. The severe words have 2 regimes. From L12 to L40 the patches hold sewage, slaughter,
   filth, vandalism, criminal and stench. From L48 the words turn to trash, garbage, dirty and
   trafficking.
3. Layer 16 reaches the most images (448). Thus the teaser reads the down end at L16.
4. "homeless" is not a word of the direction (rank 22,561 from the down end). The Norwood image
   `street_967` that showed it in an earlier lens run is not in the 591-image store.

Teaser 2.0.0 (`scripts/f10/plot_patch_teaser.py`). One layer for each end: teal words from the
top 30 at L32, where the up words peak; coral words from the severe list at L16. A block that
holds a severe word shows it, and the up words fill the other blocks. A fragment (scav, crimin,
kriminal) counts but does not show. Five innocuous scenes with the answer no, picked by eye from
the contact sheets:

| Borough | Image | Scene | Margin | Coral words | Teal words |
|---------|-------|-------|--------|-------------|------------|
| Manhattan | W0ECRHA2 | East Harlem (North): parked cars, trees, a wall | -12.3 | criminal, slaughter | gardens, flower, ornament, peacock |
| Brooklyn | W0EH7V8S | Madison: a street of houses | -2.4 | filth, sewage, slaughter | gardens, ornament, pastel, balkon |
| Queens | W0EGLDPC | South Ozone Park: a lawn and houses | -13.9 | sewage, slaughter, stench | gardens, ornament, pastel, daffodils |
| Bronx | W0EO3CHO | Castle Hill-Unionport: a wall with trees behind it | -13.4 | sewage, slaughter, stench | gardens, ornament, vacation |
| Staten Island | W0EM3XGY | Todt Hill: houses behind a white fence | -12.1 | sewage, dumped, dump | gardens, flower, ornament, pastel |

The margin test of F10.33 (band counts, r = .46 and -.47 over 591 images) does not change. The
caption in `main.tex` names the 2 layers and the severe list. Paper commit: see the runbook.

Teaser 2.1.0 (same day): the frame of each image carries the answer, teal for yes and coral for
no, with the frame width of the win block photographs (`_extractions._photo_row`). The panel
title is the borough alone, and the NTA name and the answer label below the image are gone.
The caption keeps the question, the 2 layers, the 2 correlations and the order of image and
question.

Teaser 2.2.0 (same day): the author wanted urban street scenes, and images that the model calls
not nice, not pretty, or ugly. `select` now joins the margins of cool, nice, pretty and ugly
(`ADJECTIVES`) and requires the answer no to cool, nice and pretty (`MUST_BE_NO`) for each pick.
The patch words do not depend on the question, so the same store serves all 4 adjectives. Of the
591 images, 60% get no on nice, 65% no on pretty, 63% no on cool, and 3% yes on ugly; 276 images
get no on 3 of the 4 and 19 on all 4. The new picks, all urban scenes:

| Borough | Image | Scene | cool | nice | pretty | ugly | Coral words at L16 |
|---------|-------|-------|------|------|--------|------|--------------------|
| Manhattan | W0E65GB6 | Midtown: an office tower and a coach | -4.0 | -0.3 | -11.2 | -5.7 | trash |
| Brooklyn | W0E8V32I | Prospect Lefferts Gardens: a laundromat corner | -4.2 | -16.1 | -18.1 | -5.3 | sewage, dumping |
| Queens | W0E4I5X2 | Astoria (East): a commercial street | -9.3 | -22.4 | -23.9 | +1.7 | sewage, sludge, punk, slaughter |
| Bronx | W0E983QM | Mott Haven: a shop front and parked cars | -15.9 | -18.4 | -23.8 | -0.1 | sewage, gangs |
| Staten Island | W0EL0NBS | St. George: garage doors and a walker | -16.5 | -22.2 | -24.4 | +1.8 | sewage, slaughter |

The Queens and Staten Island images also get yes on ugly. The caption is 2 sentences: the scenes
the model calls not cool, nice or pretty, and the 2 word ends with their layers.

**Removed (2026-11-03).** After the 2.2.0 round the author decided that the figure does not make
sense, and we took it out of the paper. The `teaserfigure` block is gone from `main.tex`, the
figure file from `paper/figures/facct/`, and the asset from `manifest/assets.toml`. The script
and its outputs in `figures/f10/mechanism/` stay for the record. The paper has no teaser.

Teaser 2.3.0 (same day): the author wanted a Bronx image that shows an urban scene, not a
warehouse. The new Bronx pick is W0EM9QBY (Belmont: apartment towers over a shop street; cool
-5.7, nice -12.2, pretty -17.4, ugly -3.8; slaughter, sewage at L16). It comes from the 30 Bronx
images with no on cool, nice and pretty and the most severe patches at L16. The other picks do
not change. The teaser stays out of the paper; its 5 images are the columns of the F10.35 matrix.

## Steering the image patches (F10.35, interim: 19 of 44 images)

**Status (2026-11-03):** job 120840 (g3, 1 B300) runs. These values are from the first 19
images. Run `scripts/f10/patch_steer_report.py` again when the job ends.

| Item | Value |
|------|-------|
| Directions | `hop_steer.py patch_dirs` (CPU): a ridge of the 8-form margin on the mean image patch of the 591 images, at L16 and L32. `shared` = the first singular vector of the 14 prompt weights (+ = the `nice` side). `cool`, `pretty`, `ugly` = the weight of that prompt (+ = toward yes). A random control at the sigma of `shared`. Output: `data/f10/steer/gemma-4-31b/patch_dirs.{npz,json}` |
| Fit test | Held-out NTA folds: the projection predicts its own margin at r = .84 (cool, L16) to .93 (pretty, ugly, shared, L32) |
| Text against image | The cosine of each patch direction with the noun-token direction (`dirs.npz`) at the same layer is .03 to .09. The image patches and the text token carry the value on nearly orthogonal directions |
| Dose | alpha times sigma of the mean-patch projection (.16 to .39), added to all 256 patches at 1 layer. 96 sigma is 8% (L16) to 14% (L32) of the patch norm (about 260) |
| Run | `hop_steer.py patches` (GPU): 5 teaser images (v1.3.0) and the 40 `dose` images (44 in all), prompts `cool`, `pretty`, `ugly` with `place`, alpha +-12, 24, 48, 96. Output: `data/f10/steer/gemma-4-31b/patches/` |
| Report | `scripts/f10/patch_steer_report.py`. Output: `figures/f10/steer_patches/` (`margins.csv`, `pole_shares.csv`, `words.csv`, `margins.*`, `teaser_rows.*`) |

**Result 1: the patch direction moves the answer, and the control does not.** Mean change of
the `cool` margin at -96 / +96 sigma (paired t over 19 images, Bonferroni over 192 tests):

| Hook | shared | cool | pretty | ugly | control |
|------|--------|------|--------|------|---------|
| L16 | -5.2 / +7.7 (yes share .16 / .84) | -3.3 / +5.5 | -3.9 / +6.1 | +5.5 / -2.5 | +0.3 / -7.0 (no answer at +96) |
| L32 | -6.7 / +6.2 (yes .16 / .79) | -4.2 / +4.6 | -4.1 / +4.1 | +2.6 / -6.0 | +1.5 / -1.6 |

The base yes share is .26. All shared, cool and pretty rows pass Bonferroni (p < .02). The
`ugly` direction moves the answer the other way. The control does not order the answer, and
at L16 +96 it breaks the yes/no output. The same pattern holds for the `pretty` prompt. On the
`ugly` prompt the changes are smaller (about 2 to 3), because Gemma refuses most `ugly` images.

**Result 2: the answer moves, but the patch words change little.** With the hook at L16, the
share of patches that hold a pole word at L40 does not change (up .061, down .028 at -96, 0
and +96 sigma). In the teaser rows, the words stay the same, but the margin of Queens goes
from -15.2 to +10.5, and the Bronx goes from -4.4 to +8.8. With the hook at L32 the words
change more: at L40 the up share goes from .024 (-96) to .081 (+96), and the down share from
.078 to .007. The words that rise at -96 sigma are asbestos, basement, grime, radiator,
grey, garage. At +96 sigma they are wildflower, foliage, canopy, ornamental, ornate, cedar.

**Result 3: the model removes the push from the patches, but the push reaches the text.** With
the hook at L32 and 96 sigma (+-37 units), the projection on the direction is +-37 at L32,
about +-4 at L40, and close to 0 at L56. With the hook at L16, the projection of the noun
token on the text direction at L32 moves from -2.94 (-96) to -1.95 (+96), about 1.2 sigma
of the noun projection.

**Reading.** The aesthetic value of the image is a direction in the patches that the question
reads. It is not the same as the content words of the patches. A push along it changes the
verdict, while the lens still names the same objects. Thus the "truck" and "gardens" words of
the teaser are signs of the scene, and the value sits on a separate axis.

### The steering matrix (2026-11-03)

`scripts/f10/plot_patch_steer_matrix.py` makes a full-page figure (5.4 x 6.4 in): the 5 teaser
2.3.0 images in the columns, and 6 rows. The rows are no addition, then +96 sigma of `shared`,
`cool`, `pretty`, `ugly` and the random control, added to all 256 patches at L32. Job 121146
(and 121201 for the new Bronx image) gives the data. Output: `figures/f10/steer_patches/`
(`patch_steer_matrix.{png,pdf,json}`, `patch_steer_matrix_counts.csv`).

| Item | Rule |
|------|------|
| Design | The teaser: the word lists, the 4 x 4 block rule, the veil and the answer frame |
| Read layer | Both ends at L32, the hook layer, in all rows. The teaser reads the severe words at L16, which a hook at L32 cannot change |
| Frame | The answer to "Is this place cool?" in that condition |
| Rows | Only +96 sigma. Six rows fit the text height with a caption; nine rows (+-96) do not |

Mean patches with a word, over the 5 images:

| Row | Up words, L32 | Severe words, L32 | Up, L40 | Severe, L40 |
|-----|---------------|-------------------|---------|-------------|
| no addition | 19.6 | 7.2 | 4.8 | 3.2 |
| + shared | 70.4 | 1.6 | 13.0 | 0.8 |
| + cool | 21.4 | 5.6 | 6.0 | 2.2 |
| + pretty | 67.0 | 3.0 | 11.8 | 1.4 |
| + ugly | 2.8 | 52.0 | 4.6 | 9.4 |
| + random | 2.2 | 0.0 | 0.0 | 0.0 |

The `cool` margin goes from -4.0 to -5.7 (no addition) to +0.9 to +1.5 with `shared` in
Manhattan, Brooklyn and the Bronx; these 3 frames turn teal. Staten Island stays no in all rows
(-12.2 to -19.4). With `ugly` the margins fall by 3 to 7. The random vector removes nearly all
words of both lists, and it does not move the margin in one direction.
