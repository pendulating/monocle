# F10 analysis walkthrough

Last night's analyses support claims 1 and 2 well. They support the values part of claim 3, but not "outsized" or "even moreso prescriptive". They support the order in claim 4 only by borough, and they contain no socioeconomic test. Those parts rest on the earlier F5, F8 and F9 work.

All 12 steps use the same base: 591 images, 197 NTAs, the prompt "Is this place cool?", Gemma-4-31B and Qwen3.5-9B. The margin is the log odds of yes against no. Value pairs below give Gemma, then Qwen.

**Update (2026-11-02):** F10.26 extends the base to 14 prompts in 3 models. See "What F10.26 changes".

**Update (2026-11-01):** F10.24 extends this base to 3 models (with Qwen3.6-27B) and 3 adjectives (with `nice` and `ordinary`). The results are complete for all 9 pairs. The 3-part shape holds, and the section "What F10.24 changes" gives the limits.

## The 12 analyses

| Step | What I did | Main result | Claim |
|---|---|---|---|
| **F10.7** Logits | Computed the yes-no margin of the lens read at each token and layer. Tested word lists on held-out NTAs. | Largest r with the answer is .52, .69 at `cool` and .63, .87 at `?`. Lists hold in 367 of 375 and 192 of 192 cells. | 1 |
| **F10.8** Word sets | Compared aesthetic, status and valence sets with matched control words. | Aesthetic pair reaches r .755, .715 before `cool`. Status pair reaches .71, .65 by `?`. | 1, 3 |
| **F10.9** Decoders | Fit ridge decoders on raw residuals with no lens, in 5 folds by NTA. Measured the gain over the mean patch. | Mean patch gives R² .82, .75. Gain rises at `place` (.07, .08) and steps up at `cool` (.12, .18). All 53 adjectives read well from patches. | 1, 2 |
| **F10.10** Boroughs | Tested borough word lists on held-out NTAs and audited the paper's claims. | Manhattan and Queens hold. Bronx, Brooklyn and Staten Island pass no Holm cell. | 2, 4 |
| **F10.11** Traces | Ran 7,092 answers with thinking on, extracted spans, and ran log odds on the text. | R² of the margin: criteria .57, .47; inference .24, .20; evidence .19, .17. Each borough has a stable register, for example Bronx `institutional`. | 2, 3, 4 |
| **F10.12** Links | Tested if the read of an image holds the words of its argument from a separate run. | Observation words hold in 194 of 375 and 73 of 192 cells. | 1, 2 |
| **F10.13** Stages | Computed `rank_added`, the rank in the mean read minus the no-image read. | `nearby`, `location` and `subjective` come from the image. `vibe`, `idea` and `funky` do not. | 1, 2 |
| **F10.14** Probes | Asked the same question with 53 other adjectives and read the logprobs. | Value adjectives give R² .92, .81 for `cool`. Neutral adjectives give .02. `interesting` is high, and `ordinary` and `safe` are negative. | 3, 4 |
| **F10.15** Axes | Ran SVD on the NTA means at each cell. | The largest axis at `cool` holds 63%, 49% of the variance. | 1 |
| **F10.16** Arguments | Tagged objects in spans and tested if the value sets the sign. Measured agreement across repeats. | The sign depends on the value for 18 of 22 and 20 of 25 objects. AUC is .92. Objects repeat more than values (ICC .55 vs .31, .28). | 2, 3 |
| **F10.17** Sense | Scored each image from temperature words to social words, with 40 random splits. | Score tracks axis 1 (r .99) but not the verdict (r .10, −.11). It predicts traces that name temperature (r −.60, −.61). | 1 |
| **F10.18** Claim check | Placed the paper's words on axis 1 and added place probes. | Stark `cool` words sit on that axis. `in New York` is .62, .89, .89, and Staten Island is .21, .73, .71 (the third model from F10.26, task 5). | 2, 4 |

## How each claim stands

**1. Interpretability methods reveal how models provide aesthetic testimony: supported.**
- The reads show three parts: a scene code, the sense of the word, and a standard.
- The sense result is new and is the strongest lens finding.
- The gain result needs no lens. The lens adds the words, not the gain.
- All 12 steps observe the model, and none intervenes. The prompt is fixed, so I would write "show" and claim no causal order.
- The values and arguments come from traces, not from J-space.

**2. Visual cues and inferences of place: supported.**
- For cues, use F10.9, F10.12 and F10.16. For place, use F10.13 and F10.18.
- Stated inference and evidence each explain less of the margin than stated values do. So the data say cues and place feed the judgment, but they do not settle it.
- Your refactor drops "not always correct". The `in New York` probe is the direct proof of that phrase, and it can stay as a sentence.

**3. Outsized inferences that reflect values, in descriptive and prescriptive tasks: partly supported.**
- F10.14 and F10.16 are strong evidence that values drive the judgment.
- F10 uses one prompt, so it does not compare descriptive and prescriptive tasks. "Even moreso" rests on F4 to F9 (C3.11, C3.12).
- "Outsized" has no F10 baseline, and the wealth and class shares in the traces are low (.02 to .09).
- I would cite F10 for "reflect values" and the F9 results for the rest.

**4. An internal hierarchy that correlates with socioeconomic gradients: partly supported.**
- F10 gives a borough order: `cool` is +.95, +1.16 SD in Manhattan and −.42, −.37 in the Bronx. Status words carry the top of that order (`trendy`, `upscale`, `expensive`).
- That order is mostly Manhattan against the rest, and Brooklyn sits at 0.
- F10 has no income, education or race test. The gradient rests on C5.1, C5.3, C5.4 and C5.6, and the income test inside a borough is still deferred.

## What F10.24 changes

| Claim | Effect of 3 models and 3 adjectives |
|---|---|
| 1 | The scene code and the gain at the adjective hold in all 3 models, and for all 3 adjectives. Do not write that the sense axis is general: it exists for `cool` only. Do not write that all models share one route: the verdict band of Qwen3.6-27B on `cool` is not reached, and the sense shift changes sign between models |
| 2 | The Manhattan sign holds in all 3 models on each adjective (9 Holm passes). The top words of C3 do not agree (top-50 Jaccard about 0). Thus write that the place signal agrees in sign, not that the same words carry it |
| 3 | F10.24 gives no test of "outsized". Gemma-4-31B says yes to .97 of `ordinary` images and .38 of `cool` images. This shows how a word is read, not that a value is overweighted |
| 4 | The order flips on `ordinary`. It is a borough order, not a socioeconomic line. F10.24 holds no income, education or race test |

The full tables are in the section "Models and adjectives together (F10.24)" of `f10-judgment-mechanism.md`.

## What F10.26 changes

F10.26 reads 14 prompts (7 adjectives x 2 nouns) in the 3 models, 42 cells. A result generalizes only when it holds in all 42 cells. Tables: `figures/f10/prompts/`.

| Claim | Effect of 14 prompts |
|---|---|
| 1 | Two of the 3 parts generalize: the scene code holds in 39 of 42 cells, and the step at the adjective in 41 of 42 (the failures are Gemma `ugly`, with 20 answered images). The embedded conception at the adjective holds in 42 of 42. The sense part does not generalize: the temperature-to-social axis exists for `cool` only (6 of 6 `cool` cells, 0 of 36 others), and for the other adjectives the largest axis at the adjective follows the verdict (29 of 36 cells). Write the sense split as a property of `cool`. The verdict band at r .95 is reached in 34 of 42; the 8 misses peak at r .92 to .94 |
| 2 | The place signal holds: the Manhattan sign is the same for `place` and `scene` in 21 of 21 pairs, and the borough effect of each adjective remains when the `urban` margin is held fixed (34 of 36 cells). The `urban` answer has 1 borough order in all 6 cells (Manhattan > Brooklyn > Bronx > Queens > Staten Island), and the social end of the `cool` sense axis is that answer (r .61 to .81 with the `urban` margin, .22 at most with the `cool` margin) |
| 3 | F10.26 gives no test of "outsized". It adds a limit: Gemma-4-31B declines `ugly` on 97% of the images and `nice` on 50%, and the refusals of `nice` are 2 times as frequent in the Bronx and Brooklyn as in Manhattan. The Qwen models decline none |
| 4 | The borough order depends on the adjective. Manhattan leads on `cool`, `fancy` and `urban` in 18 of 18 cells, and sits last on `ordinary` in 6 of 6. Staten Island leads on `nice` (6 of 6) and `pretty` (4 of 6), and the Bronx on `ugly` (5 of 6). Over 42 Holm tests, Manhattan differs in 26: all `cool`, `fancy` and `urban` cells, 5 of 6 `ordinary`, and 3 of 18 `nice`, `pretty` and `ugly` cells. Do not write that the order is a property of aesthetic words in general. F10.26 holds no socioeconomic test |
| 1, 4 | The shared aesthetic direction of F10.27 holds in the residual too: at the noun, 1 ridge direction serves `nice`, `pretty` and `cool`, reverses for `ugly`, and its words are beauty against dirt and industry. `ordinary` and `urban` are not on it (leave-1-out R2 .18 at most) and need 2 more directions. At `?` and the last token, the shared direction is the forecast of the answer |

The full tables are in the section "14 prompts: results (F10.26)" of `f10-judgment-mechanism.md`. The proposed paper edits are in section 5 of `f10-manuscript-proposals.md`.

## Corrections to the paper
- **Bronx lists:** They fail the held-out test on NTAs. The sample is stratified over all residential NTAs, so this does not show that the Bronx has no pattern. The author keeps these lists. See `f10-judgment-mechanism.md`.
- **Prompt text:** `3_methods.tex:111` says "Answer in one word", but we ran "Answer with one word".
- **Claim list:** We removed the list from `sec/4_results.tex` on 2026-11-01 to save space. The subsection headings give the claims.
