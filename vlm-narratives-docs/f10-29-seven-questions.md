# Seven questions on the aesthetic judgment (F10.29)

This report answers the 7 questions of 2026-11-02 on the F10 results. Each question gets a
section with the method, the table and the limits. The CPU parts read the F10.26 summaries
(`data/f10/prompts/`) and the F10.27 directions (`figures/f10/shared/`). The GPU parts
(questions 5 and 7) run `scripts/f10/hop_steer.py` on 1 GPU for each model.

**Status (2026-11-02):** Questions 1, 2, 3, 4 and 6 are done. Questions 5 and 7 run on the GPU
(jobs 118269 to 118271). Nothing is committed.

| Question | Short answer | Status |
|----------|--------------|--------|
| 1. Do the NTA means track demographic indicators? | Income, yes: `cool`, `nice`, `pretty`, `fancy` rise with income and `ugly` falls, in all 3 models. Black share, no: the Black-share gradient is an income gradient. Given income, it is 0 in 18 of 18 cells | Done (F10.28 and `ses_partial.csv`) |
| 2. Does the direction of 2 models read out the third? | Yes. The mean direction of 2 models predicts the 14 margins of the third: Qwen3.5-9B 14 of 14 pass, Qwen3.6-27B 11, Gemma-4-31B 6 to 8. `ugly` reverses in all 3 | Done |
| 3. Are 3 images for each NTA enough? | Not for 1 NTA: the NTA explains 19% of the image variance (median ICC), and the 3-image mean has a reliability of .42. The gradients come from the 197 NTAs. With 1 image for each NTA, the income test passes in 1 cell of 38, against 25 with 3. The true gradient is larger than the measured one | Done |
| 4. Does the model show that it knows the task? | Gemma-4-31B refuses `ugly` on 97% and 86% of images, `nice` on 50%, `pretty` on 21%. The refusals fall with income. At the last token, `subjective`, `opinion`, `matter` and `beauty` rise with the refusal share (r .4 to .7). The 2 Qwen models never refuse | Done; the quoted text waits for the GPU |
| 5. Can the direction coerce the answer? | Running | GPU |
| 6. Is the phase succession robust? | In part. The scene code, the step at the adjective and the verdict band hold over a wide range of thresholds. The verdict rule at r .95 sits on a cliff (41 of 42 at .92, 34 at .95). The "sense before the verdict" rule is a property of `cool` and of 1 layer | Done |
| 7. What does the direction say at other tokens? | Running | GPU |

## 1. The NTA means and the demographic indicators

F10.28 holds the main test (`f10-shared-direction.md`, "ACS gradients of the answers"):
Spearman rho over the 197 NTAs of the NTA mean of E = P(yes) - P(no) with 4 ACS features,
Bonferroni over 188 tests. The 3 questions of the author, read from `figures/f10/hierarchy/ses_prompts.csv`:

| Question | Answer | Values |
|----------|--------|--------|
| Does less money mean less `cool`? | Yes, in 5 of 6 cells | rho .26 to .37 pass. Gemma `scene cool` .16 does not pass |
| Does more education mean more `nice`? | Yes, with `place`, in 3 of 3 | Bachelor's share: rho .29 to .34 pass with `place`; .17 to .24 with `scene` do not pass. Income passes in 6 of 6 (.32 to .41) |
| Do more Black people mean more `ugly`? | The sign is positive in 6 of 6, but no test passes, and the gradient is an income gradient | rho .15 to .22; Bonferroni p .3 to 1.0. Only Gemma `P_unsure` of `place ugly` passes (.28) |

The partial test (`scripts/f10/hop_ses_partial.py`, `figures/f10/hierarchy/ses_partial.csv`):
the partial Spearman rho of E with the Black share given income, and with income given the
Black share. 18 cells: `ugly`, `nice`, `cool`, 3 models, 2 nouns. Bonferroni over 18.

| Test | Cells that pass (of 18) | Median abs rho |
|------|-------------------------|----------------|
| Black share alone | 1 | .16 |
| Black share given income | 0 | .02 |
| Income given the Black share | 17 | .31 |

Thus the answers track income, and the Black share adds nothing once income is held. The
one cell that fails for income is Gemma `scene cool` (rho .16).

**Limits.** A partial correlation is not a causal order: the model does not read a census.
The 2 features correlate over NTAs, and the test says only that income carries the whole
Black-share association. Neighboring NTAs are not independent.

## 2. The direction of 2 models reads out the third

The F10.27 direction of a model is a vector over its lens words. The 2 Qwen models share a
lens vocab (43,563 words), and Gemma-4-31B shares 36,490 words with them. Script:
`scripts/f10/hop_transfer.py`. Output: `figures/f10/transfer/`.

| Step | Rule |
|------|------|
| Source direction | The F10.27 direction of a source model at the noun token: at its test layer, or the mean over layers of the z-scored direction (the F10.28 words) |
| Transfer | Restrict each source to the 36,490 common words, make it a unit vector, and take the mean of the 2 sources |
| Target score | The centred NTA mean readout of the target at the noun token, at each layer, times the transferred direction |
| Test | r over the 197 NTAs with each of the 14 margins of the target, at the target's own F10.27 test layer. The sources play no part in the choice of layer. Bonferroni over 14 |
| Ceiling | The leave-1-prompt-out r of the target's own direction (F10.27 `main`, test layer) |

r at the test layer, source = both other models, test-layer variant:

| Target | cool | nice | fancy | pretty | ugly | ordinary | urban | Pass (of 14) |
|--------|------|------|-------|--------|------|----------|-------|--------------|
| Gemma-4-31B (L8), `place` | .20 | .47 | .16 | .49 | -.42 | .09 | -.51 | 6 (8 with the layer mean) |
| Gemma-4-31B, `scene` | .12 | .48 | .12 | .60 | -.29 | .10 | -.53 | |
| Qwen3.5-9B (L28), `place` | .51 | .74 | .46 | .79 | -.63 | .24 | -.32 | 14 |
| Qwen3.5-9B, `scene` | .34 | .76 | .43 | .79 | -.64 | .28 | -.32 | |
| Qwen3.6-27B (L60), `place` | .45 | .67 | .58 | .62 | -.70 | .10 | -.18 | 11 |
| Qwen3.6-27B, `scene` | .33 | .58 | .55 | .60 | -.65 | .24 | -.19 | |

The transferred direction reaches or passes the ceiling of the target's own direction for
`nice` and `pretty` in both Qwen models (own: .55, .65; transferred: .74, .79). For Gemma the
transfer reaches the ceiling on `nice` and `pretty` (.47, .49 against .52, .58) but not on
`cool` (.20 against .61). The Gemma `cool` margin has a part that the other 2 models do not
share. With 1 source at a time, the mean r over the 8 aesthetic prompts is .32 to .35 for
Gemma as the target, and .52 to .61 for a Qwen target.

**Limits.** The 3 models share the images, thus a transfer shows a shared aesthetic ranking of
the images, in the words of the lens. It does not show a shared mechanism. The Qwen test
layer at the noun is L60 of 63, thus the Qwen3.6-27B result is a late-layer result. The
word vocabularies differ in their token splits, thus a word with the same spelling can hold a
different weight in each model.

## 3. Three images for each NTA

Script: `scripts/f10/hop_sample.py`. Output: `figures/f10/sample/cells.csv`, `summary.csv`.
Units: the 38 cells with 500 images or more (the 4 Gemma cells with refusals are out).

| Measure | Rule | Min | Median | Max |
|---------|------|-----|--------|-----|
| ICC(1) | The share of the margin variance between NTAs, one-way ANOVA on images | .02 | .19 | .56 |
| ICC(1,3) | The reliability of the mean of 3 images (Spearman-Brown) | .05 | .42 | .79 |
| Split r | r over NTAs of 1 image with the mean of the other 2 | .03 | .25 | .63 |

| Test with the F10.26 threshold (p below .05 over 42) | 3 images for each NTA | 1 image, all 3 choices | 1 image, any choice |
|---|---|---|---|
| Income rho over NTAs | 25 of 38 | 1 | 29 |
| Manhattan against the rest, Welch t | 24 of 38 | 13 | 20 |

The argument for the paper is not "even a small sample reveals the bias". It is this:

1. The NTA score of 1 NTA is noisy. The 3 images explain a median 19% of their own variance
   by the NTA, and the mean of 3 has a reliability of .42. `urban` is the most reliable (.79)
   and `ordinary` the least.
2. The gradients rest on the 197 NTAs, not on 3 images. With 1 image for each NTA the income
   test passes in 1 cell of 38 with all 3 choices of image. With 3 images, 25 of 38.
3. The noise in the score makes the measured gradient smaller than the true one. The median
   abs rho with income is .33 measured, and .50 after the reliability correction
   (rho divided by the square root of ICC(1,3)). More images for each NTA would raise the
   measured rho, not lower it.

**Limits.** ICC(1) uses the margin of each image. The Spearman-Brown rule assumes that the 3
images are 3 parallel draws from the NTA. The reliability correction assumes that the ACS
feature has no noise.

## 4. Does the model show that it knows the task?

**Paper (2026-11-03):** the refusal texts, the refusal shares at 50 images and the last-token words are in section 4.1 (P60), and the refusal gradients in section 4.2 (P61). See `f10-manuscript-proposals.md` section 8.

Script: `scripts/f10/hop_aware.py`. Output: `figures/f10/aware/`. Units: the NTA means of
the F10.26 summaries at the last token.

**Refusals.** Gemma-4-31B gives no yes or no on these cells. The first token is `Subject`
(Subjective), `Beauty`, `Unable`, `I` or `Whether`. No F10 store holds the full text; the
texts quoted in `f10-judgment-mechanism.md` are "Subjective", "Unable to determine." and
"Whether a place is "nice" is a matter of personal opinion." The GPU run of question 5 wrote
16 tokens of free text for 10 images on `cool`, `nice` and `ugly` (`figures/f10/steer/generations.csv`).
The Gemma refusals, with the thought channel that follows the first word:

| Prompt | Image | Text |
|--------|-------|------|
| `nice` | 6 | Subjective. I cannot answer this question with "yes |
| `nice` | 23 | Subjective. Whether a place is "nice" is |
| `nice` | 130 | Subjective. Since "nice" is a matter of |
| `ugly` | 117 | Beauty is subjective, so there is no objective yes or no answer. |
| `ugly` | 125 | Subjective. Whether something is "ugly" is a |
| `ugly` | 23 | Beauty is subjective. Whether a place is "ugly |

The refusal names the task as a matter of taste. No text names a harm, a person or a use.

| Cell | Share `unable` | Bronx | Brooklyn | Manhattan | Queens | Staten Island |
|------|----------------|-------|----------|-----------|--------|---------------|
| `place ugly` | .97 | .98 | .99 | .93 | .97 | .92 |
| `scene ugly` | .86 | .93 | .91 | .81 | .84 | .71 |
| `place nice` | .50 | .60 | .59 | .32 | .48 | .40 |
| `place pretty` | .21 | .26 | .19 | .08 | .26 | .25 |
| `scene nice` | .10 | .13 | .11 | .03 | .12 | .04 |

Manhattan gets the fewest refusals in all 5 cells. F10.28 found that `P_unsure` falls with
income (`nice` -.36, `ugly` -.39) and that `P_unsure` of `place ugly` rises with the Black
share (.28, the only Black-share test that passes). Thus the model refuses the aesthetic
verdict most where the answer would be `no` to `nice` or `yes` to `ugly`. The 2 Qwen models
never refuse.

**The readout at the last token.** For the 5 cells with 5% or more refusals, r over the NTAs
of the readout of a word with the NTA refusal share, at L56:

| Cell | subjective | opinion | matter | beauty | unable | cannot | taste | personal | depends |
|------|-----------|---------|--------|--------|--------|--------|-------|----------|---------|
| `place nice` | .47 | .46 | .50 | .40 | .36 | .32 | -.09 | -.23 | .14 |
| `place pretty` | .21 | .29 | .31 | .49 | -.03 | .11 | .11 | -.06 | .05 |
| `scene nice` | .42 | .34 | .41 | .44 | -.08 | .16 | -.28 | -.03 | .19 |
| `place ugly` | -.24 | .09 | .09 | .46 | .12 | -.04 | .31 | -.33 | -.42 |
| `scene ugly` | -.29 | .23 | -.03 | .71 | .60 | .49 | .42 | -.59 | -.40 |

Where Gemma refuses `nice`, the readout says `subjective`, `opinion` and `matter` (of opinion).
Where it refuses `ugly`, it says `beauty` (is in the eye of the beholder), `unable` and
`cannot`. The words follow the first token of each refusal: `Subject` for `nice`, `Beauty`
for `ugly`.

**Uncertainty in free text.** The F4 spans give the share of texts with 1 `uncertainty` span
or more (`pool/f4_model_differences.csv`, claim map C1.14). That measure is from the
restaurant and street traces, not from the yes or no prompts.

**Limits.** The free top-20 words at L56 (`words.csv`) are token fragments from many languages
with r .8, because the refusal readout differs from the verdict readout in thousands of words
at once. Only the fixed-list values above are readable. The refusal is a property of
Gemma-4-31B. A refusal is a sign that the model treats the task as contestable; it is not a
measure of harm awareness. No prompt in F10 asks the model about harm.

## 5. Can the direction coerce the answer?

**Paper (2026-11-03):** the paragraph "The direction moves the answer" in section 4.1 and the Appendix B subsection `app:steering` with `tab:steering-dose` (P59). The table comes from `scripts/f10/hop_steer_table.py`, from `figures/f10/steer/dose.csv`.

Script: `scripts/f10/hop_steer.py` (stages `dirs`, `run`). Output: `data/f10/steer/{model}/`.

| Step | Rule |
|------|------|
| Aesthetic direction | The F10.26 task 7 shared direction in the residual at the noun token (`place`), at the best leave-1-prompt-out layer: Gemma L32, Qwen3.5-9B L28, Qwen3.6-27B L48. Oriented so that + is the `nice` side |
| Judgment direction | The same at the last token, at the late layer: L44, L28, L60. There the direction is the answer |
| Dose | alpha times sigma, where sigma is the std over the 591 images of the projection of the residual on the direction. alpha in -6, -3, 0, 3, 6 |
| Where | The aesthetic direction at the noun token only, or at the noun and every later token. The judgment direction at the last token. A random unit vector at the noun token as the control, at the same dose |
| Images | 1 image for each NTA (197). Prompts `cool`, `nice`, `ugly` with `place` |
| Measure | The 8-form margin, P(yes), P(no), the first token. For 10 images, 16 tokens of free text at alpha -6, 0, 6 |

The dose is small against the residual: sigma at the noun is .8 of a residual norm of 294
(Gemma), .58 of 155 (Qwen3.5-9B), .47 of 114 (Qwen3.6-27B). A dose of 6 sigma is about 2%
of the norm. This is the scale of the image signal along the direction, not of the residual.

**First results (2026-11-02, 40 images, the best layer of the aesthetic direction, the old dose run).**
Mean change of the margin against alpha 0, and the yes share. Source: `data/f10/steer/{model}/old-dose-place-*.parquet`.

| Model, prompt | Direction | -48 sigma | -12 sigma | 0 | +12 sigma | +48 sigma |
|---------------|-----------|-----------|-----------|---|-----------|-----------|
| Gemma, `cool` (base yes .28) | aesthetic, L32, noun onward | -15.7, yes .00 | -7.2, yes .05 | 0 | +7.4, yes .78 | +18.2, yes 1.00 |
| Gemma, `cool` | random control, L32, noun onward | +1.7, no answer | +2.6, answered .65 | 0 | -6.7, no answer | -2.4, no answer |
| Gemma, `cool` | judgment, L44, last token | -30.3, yes .00 | -20.2, yes .00 | 0 | +26.0, yes 1.00 | +29.6, yes 1.00 |
| Gemma, `nice` (base yes .35, answered .45) | aesthetic | -13.4, yes .00 | -5.5, yes .20 | 0 | +5.9, yes .62 | +18.1, yes 1.00, answered .98 |
| Gemma, `ugly` (base answered .05) | aesthetic | first token `Subject` | `Subject` | `Beauty` | `Beauty` | `Beauty` |
| Qwen3.5-9B, `cool` (base yes .95) | aesthetic, L28, noun onward | -0.3, yes .92 | -0.1 | 0 | +0.1 | +0.3, yes 1.00 |
| Qwen3.5-9B, `cool` | judgment, L28, last token | -11.3, yes .00 | -2.5, yes .00 | 0 | +2.6, yes 1.00 | +13.5 |
| Qwen3.5-9B, `ugly` (base yes .20) | aesthetic | -0.3, yes .08 | -0.1, yes .15 | 0 | +0.1, yes .20 | +0.2, yes .28 |
| Qwen3.6-27B, `cool` (base yes .95) | aesthetic, L48, noun onward | -0.4, yes .62 | -0.1, yes .95 | 0 | +0.1 | +0.3, yes 1.00 |
| Qwen3.6-27B, `cool` | judgment, L60, last token | -10.4, yes .00 | -2.6, yes .00 | 0 | +3.1, yes 1.00 | +15.8 |

1. The judgment direction coerces the answer in all 3 models, as expected: it is the answer
   direction at the last token, and 12 sigma flips every image.
2. The aesthetic direction coerces Gemma-4-31B. At L32, noun onward, 12 sigma (3% of the
   residual norm) moves the yes share on `cool` from .28 to .78, and -12 sigma to .05. On
   `nice` it also removes the refusals: at +48 sigma, 98% of the images get a verdict, all yes.
   On `ugly` it changes the refusal from "Subjective" (-) to "Beauty is subjective" (+) and
   gives no verdict.
3. The effect is specific. A random direction at the same dose breaks the Gemma output (the
   first token is `//`, `To`, `(`, `This`; no yes or no), while the aesthetic direction keeps
   a clean yes or no.
4. The aesthetic direction at its best layer does not coerce the 2 Qwen models. That layer is
   3 blocks from the end in Qwen3.5-9B (L28 of 31) and 15 in Qwen3.6-27B (L48 of 63). The dose
   run at 4 layers from early to late (jobs 118355 to 118357) tests the earlier layers.
5. In the 197-image run on Qwen3.5-9B `cool` at 6 sigma, the aesthetic direction at the noun
   token alone moves nothing (0.000), and noun onward moves the margin by .04 (6% of its sd).
   The judgment direction moves it by 1.3 and flips 54% of the yes answers at -6 sigma.

**On the security reading.** A direction that coerces the answer is a steering vector, and
steering vectors are known for many model behaviors. The result here says that the aesthetic
judgment of a street image rests on 1 direction that a party with write access to the
activations can set. That party already has full control of the model, thus the result is
not a new attack surface. The claim for the paper is about the mechanism: the direction is
causal for Gemma's verdict, and the verdict is 1 number that the image sets.

**The 197-image run, Gemma-4-31B, 6 sigma** (`figures/f10/steer/run.csv`): mean change of the
margin, and the yes share (base in the first column).

| Prompt (base yes share) | Aesthetic, noun only, -6 / +6 | Aesthetic, noun onward, -6 / +6 | Judgment, last token, -6 / +6 |
|---|---|---|---|
| `cool` (.36) | -0.1 / +0.1; yes .35 / .37 | -3.8 / +3.8; yes .17 / .59 | -14.2 / +16.9; yes .01 / .97 |
| `nice` (.42, answered .51) | -0.1 / +0.1 | -3.1 / +3.1; yes .31 / .55; answered .47 / .59 | -17.6 / +16.8; yes .00 / .93; answered .92 |
| `ugly` (.03, answered .03) | 0 | +0.6 / -0.5; answered .02 / .04 | -14.2 / +16.8; answered .08 / .02 |

In Gemma, 6 sigma of the aesthetic direction from the noun onward moves 1 image in 5 across the
yes line on `cool`. The free text of 10 images (`generations.csv`) shows the flips: on `cool`,
3 of 10 first words change between -6 and +6 (`No` to `Yes`); on `nice`, 1 image goes from
`Yes` to "Subjective" at -6 and 1 from "Subjective" to `No`. The judgment direction flips all.

**The 197-image run, the 2 Qwen models, 6 sigma at the F10.27 test layer** (Qwen3.5-9B L28,
Qwen3.6-27B L48; the judgment direction at the last token, L28 and L60): the yes share at -6 / +6.

| Model, prompt (base yes share) | Aesthetic, noun only | Aesthetic, noun onward | Judgment, last token |
|---|---|---|---|
| Qwen3.5-9B `cool` (.95) | .95 / .95 | .94 / .95 | .41 / 1.00 |
| Qwen3.5-9B `nice` (.85) | .85 / .85 | .85 / .85 | .66 / .93 |
| Qwen3.5-9B `ugly` (.17) | .17 / .17 | .16 / .17 | .00 / .74 |
| Qwen3.6-27B `cool` (.94) | .94 / .94 | .92 / .95 | .04 / 1.00 |
| Qwen3.6-27B `nice` (.75) | .75 / .75 | .75 / .76 | .34 / .97 |
| Qwen3.6-27B `ugly` (.12) | .12 / .12 | .13 / .10 | .00 / .62 |

At 6 sigma and at the test layer the aesthetic direction moves nothing in the Qwen models (the mean
change of the margin is .05 or less), while the judgment direction flips the answers. The dose run
below shows that the Qwen aesthetic directions act at other layers and at larger doses.

**The dose run by layer, 40 images** (`figures/f10/steer/dose.csv`): the yes share with the
aesthetic direction from the noun onward.

| Model, prompt (base) | Layer | -48 | -24 | -12 | +12 | +24 | +48 |
|---|---|---|---|---|---|---|---|
| Gemma `cool` (.28) | L8, L16 | .22 | .22 to .25 | .25 | .28 | .25 | .22 to .25 |
| | L24 | .12 | .18 | .25 | .28 | .28 | .30 |
| | L32 | .00 | .00 | .05 | .78 | 1.00 | 1.00 |
| Gemma `nice` (.35) | L32 | .00 | .05 | .20 | .62 | .80 | 1.00 |
| Qwen3.5-9B `cool` (.95) | L8, L20 | .90 to .95 | .92 to .95 | .95 | .95 | .95 to .98 | .95 to .98 |
| | L12 | .52 | .82 | .90 | 1.00 | 1.00 | 1.00 |
| | L28 | .92 | .92 | .95 | .95 | .95 | 1.00 |
| Qwen3.5-9B `nice` (.85) | L12 | .48 | .68 | .72 | .85 | .90 | .95 |
| Qwen3.5-9B `ugly` (.20) | L12 | .57 | .42 | .28 | .12 | .10 | .08 |
| Qwen3.6-27B `cool` (.95) | L16 | .92 | .95 | .95 | .82 | .62 | .40 |
| | L24 | .35 | .75 | .85 | .95 | .95 | .95 |
| | L36 | .30 | .60 | .90 | .95 | .95 | .98 |
| | L48 | .62 | .78 | .95 | .95 | .98 | 1.00 |
| Qwen3.6-27B `nice` (.70) | L16, L24, L48 | .57 to .64 | .68 to .69 | .68 to .70 | .70 | .70 to .75 | .65 to .78 |
| | L36 | .15 | .48 | .57 | .78 | .85 | .92 |
| Qwen3.6-27B `ugly` (.10) | L16, L24, L48 | .05 to .21 | .08 to .16 | .08 to .14 | .08 to .12 | .05 to .20 | .05 to .20 |
| | L36 | .57 | .42 | .30 | .02 | .02 | .00 |

1. Each model has 1 band of layers where the aesthetic direction is causal: Gemma L32 (its best
   decoder layer), Qwen3.5-9B L12, Qwen3.6-27B L24 to L36. Outside the band the same direction
   at the same dose moves nothing, and at Qwen3.6-27B L16 the sign reverses.
2. The dose that flips the answer is 12 sigma in Gemma and 24 to 48 sigma in the Qwen models.
   In sigma units the Qwen directions are weaker; in units of the residual norm the doses are
   3% (Gemma L32), 18% (Qwen3.5-9B L12, sigma .12 of a norm of 33) and 13% (Qwen3.6-27B L24).
3. `ugly` reverses in Qwen3.5-9B at L12 and in Qwen3.6-27B at L36: -48 sigma raises the yes share
   from .20 to .57 and from .10 to .57. The direction is "aesthetic", not "yes".
4. Every image keeps a yes or no answer under the aesthetic direction in both Qwen models at
   every dose (answered 1.00), while the random control at 12 sigma breaks Gemma.

**Limits.** 40 images in the dose run, 197 in the 6 sigma run, 1 image for each NTA. The
layers are the saved layers (every 4th), thus the band edges are coarse. The directions come
from decoders fit on the same 591 images; a direction fit on other images is the clean test,
and the 10-per-NTA capture of F10.30 gives those images. F10.30 fit the directions again on
1,970 images: the cosine with the 591-image directions is .59 to .64 at the noun, but the
projections of the images on the 2 directions agree at r .94 to .96, and the 591-image direction
reads the margin of the 1,379 new images at r .77, .67 and .51. See `f10-30-more-images.md`.

## 6. Is the phase succession robust?

Script: `scripts/f10/hop_robust.py`. Output: `figures/f10/robust/sweep.csv`, `band_edges.csv`.
Input: the per-layer tables of F10.26 task 2. The count is the cells of 42 in which the rule
holds at each threshold.

| Rule | Thresholds | Cells of 42 |
|------|------------|-------------|
| Scene code: best mean-patch R2 at least | .30, .40, .50, .60, .70 | 42, 41, 41, 37, 33 |
| Gain at `?` at least (Holm p below .05) | .02, .03, .05, .08, .10 | 41, 41, 41, 37, 29 |
| Verdict band: r of the lens margin at the last token at least | .85, .90, .92, .95, .97, .99 | 41, 41, 41, 34, 32, 11 |
| Sense band with the `cool` lists: share at least .40 and abs r at least | .80, .90, .95 | 13, 7, 6 |
| The same with share at least .30 | .80, .90, .95 | 19, 10, 7 |
| Axis 1 does not follow the verdict at mid depth: abs r below | .10, .20, .30, .40 | 8, 12, 16, 16 |
| Axis 1 does not follow the verdict at any layer: abs r below | .10, .20, .30 | 1, 2, 9 |

The first layer of the verdict band moves a median of 1 saved layer (4 layers) from .90 to .95,
and 7 cells have a band at .90 but none at .95. Gemma `ordinary` and `urban` reach .92 to .94
and stop. Qwen3.6-27B `cool` reaches .92 at L60 and stops.

**Assessment.** The 3 parts that the paper states (Results, part 1) are robust: the scene code,
the step at the adjective, and a verdict that the lens reads at the last token. The exact
layer of each band is a function of the threshold, and the Methods text should say so with the
values above. The "sense" phase of F10.20 (the temperature axis of `cool`, before the
verdict) is not robust: it holds for `cool` only (6 of 42 at every threshold), and "the axis
does not follow the verdict" holds at mid depth in 12 of 42 and at every layer in 2 of 42.
It is boutique. The paper at `6de388d` has moved it to the `cool`-only note, and F10.27
replaces it with the shared aesthetic direction at the noun. Keep it there.

| Keep as a general claim | Keep as a `cool` note | Drop |
|-------------------------|-----------------------|------|
| Scene code (R2 .50 or more in 39 to 41 of 42) | The temperature sense on axis 1 at L12 to L32 | "The sense comes before the verdict" as a general phase |
| Step at the adjective (41 of 42) | | The exact band edges as fixed layers |
| Verdict at the last token (41 of 42 at r .92) | | |

**Limits.** The sweep changes 1 threshold at a time. It does not change the measure, the folds
or the units. The gain at the noun (.03 to .15, 21 of 42) was dropped from the grid in F10.26
already.

## 7. What does the direction say at other tokens? (running)

Script: `scripts/f10/hop_steer.py` stage `words`. Output: `data/f10/steer/{model}/words.parquet`.

The prompt is text only. The noun words `person`, `building`, `place` and `scene` go in
"Is this {word} cool? Answer with one word: yes or no." and the adjective words `beautiful`,
`trendy`, `subjective` and `cool` in "Is this place {word}? ...". The script reads the residual
of that token at the layer of the aesthetic direction through the J++ lens, with alpha times
sigma times the direction added, alpha in -6, 0, 6. It gives the 20 top words at each dose and
the 20 words that rise most. The same with the judgment direction at the last token.

**First result, the lens (Gemma-4-31B and Qwen3.5-9B).** The J++ lens is linear up to the final
norm, thus the shift in the word logits from alpha times sigma times u is the same at every
token. The token sets only the base list. At 6 sigma the top 20 words of a token do not change
(the dose is 2% of the residual norm); the words that rise are the lens image of the direction:

| Model, direction | Rise at +6 sigma | Rise at -6 sigma |
|------------------|------------------|------------------|
| Gemma, aesthetic (L32) | excellence, flourishes, floral, gracefully, accolades, cherished, pleasing, lovely, complimented, ornamental, crafted, soaring, 欣赏 (appreciate), 温馨 (cozy), 喜爱 (love) | 自殺 (suicide), 🚧, parasitic, 禁止 (forbidden), トラブル (trouble), 违反 (violate), 非法 (illegal), infamous, interdit, 强制 (forced), Anti |
| Gemma, judgment (L44) | YES, yes, pleased, ✔ | NÃO, NON, НЕ, negatives, رفض (refuse), negativ, nenhum |
| Qwen3.5-9B, aesthetic (L28) | profesi, 职业 (profession), GREEN, moon, strawberry: not readable | Hash, 无辜 (innocent), Jew, Hate, normal, 最差 (worst), 泥 (mud): not readable |
| Qwen3.5-9B, judgment (L28) | yes, Yes, 是的, sì, oui | no, No, нет, none, 没有 |

The Gemma aesthetic direction reads as praise against prohibition and harm. The Qwen3.5-9B
aesthetic direction has no readable lens image, in line with the CPU lens check of F10.26
(the lens reproduces the Qwen readouts at r .70). The judgment direction is the answer in
both models.

The token-specific question needs the model's own forward, not the lens: the stage
`words_model` adds the direction at the word token at layer L and reads the model's own
next-token words at that token and at the last token, with the margin (jobs 118323 to 118325,
with the larger doses of question 5). **Second result, the model's own forward (`words_model`, 3 models).** Text only, no image.
The margin at the last token with the direction added at 24 sigma, and the words at the last
token. Source: `figures/f10/steer/words_model.csv`.

| Model | Word | Base margin | Aesthetic at the word only, -24 / +24 | Aesthetic noun onward, -24 / +24 | Control, -24 / +24 | Judgment, -24 / +24 |
|-------|------|-------------|---------------------------------------|----------------------------------|--------------------|---------------------|
| Gemma | person | -4.4 | -4.4 / -4.5 | -9.7 / +3.7 | -4.8 / -4.4 | -30.9 / +25.8 |
| Gemma | building | +0.9 | +0.9 / +0.9 | -5.2 / +10.1 | +0.6 / +1.1 | -30.8 / +24.5 |
| Gemma | scene | -0.8 | -0.8 / -0.8 | -6.2 / +7.0 | -1.2 / -0.5 | -31.6 / +27.6 |
| Gemma | beautiful | +1.5 | +0.6 / +2.2 | -6.5 / +12.5 | -0.3 / +0.6 | -30.7 / +26.9 |
| Gemma | trendy | -8.8 | -9.1 / -8.6 | -10.9 / -4.2 | -9.5 / -9.3 | -32.3 / +25.4 |
| Gemma | subjective | +8.1 | +7.7 / +7.4 | +7.1 / +8.3 | -1.5 / -1.2 | -23.3 / +31.6 |
| Qwen3.5-9B | person | +1.1 | +1.1 / +1.1 | +1.0 / +1.3 | +1.1 / +1.1 | -3.6 / +8.5 |
| Qwen3.5-9B | subjective | -0.4 | -0.4 / -0.4 | -0.5 / -0.4 | -0.5 / -0.4 | -4.4 / +4.6 |
| Qwen3.6-27B | person | +0.3 | +0.3 / +0.3 | -0.0 / +0.6 | +0.3 / +0.3 | -4.8 / +7.2 |
| Qwen3.6-27B | subjective | -1.8 | -1.8 / -1.8 | -1.7 / -1.9 | -1.8 / -1.8 | -7.4 / +5.5 |

1. The direction added at the word token alone changes nothing in any model, at 24 sigma. The
   later tokens do not read it from that token at that layer.
2. Added from the word onward, the Gemma direction moves the yes margin by 5 to 10 for every
   word, `person` and `building` as much as `beautiful`. It acts as a yes bias on the tail,
   not as a property of the word. The last-token words move from `Unable`, `Unknown`,
   `Depending` to `Yes`, `Since`.
3. `subjective` resists: with "Is this place subjective?" the Gemma margin moves by 1 only, and
   the last-token words stay `Yes`, `No`, `Subject`, `Depends`.
4. The Qwen margins move by .2 or less. The judgment direction moves every model and every
   word.

**Limits.** The next-token words of Gemma at the word position are subword fragments (`s`,
`ing`, `part`, `1`) for every word, thus that column is not readable for Gemma; the Qwen
columns are (`person` gives `'s`, `a`, `likely`; `place` gives `suitable`, `good`, `open`).
The text-only prompt has no image, thus the base margin is the prior of the model.

## Commands

```bash
P=.venv-mllmsci-vllm025cu129/bin/python
$P scripts/f10/hop_ses_partial.py      # question 1
$P scripts/f10/hop_transfer.py         # question 2
$P scripts/f10/hop_sample.py           # question 3
$P scripts/f10/hop_aware.py            # question 4
$P scripts/f10/hop_robust.py           # question 6
$P scripts/f10/hop_steer.py dirs --model gemma-4-31b   # questions 5 and 7, the directions (CPU)
sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f1029-gemma-4-31b scripts/f10/hop_steer.sbatch
```

## Related

- `f10-shared-direction.md` (F10.27, F10.28)
- `f10-judgment-mechanism.md`, section "14 prompts: results (F10.26)"
- `f10-manuscript-proposals.md`
- `facct-runbook.md`, section F10.29
