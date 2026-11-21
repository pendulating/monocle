# F10 mechanism of the "cool" judgment

This report gives the results of the overnight run of 2026-11-01. It covers 2 models
(Gemma-4-31B, Qwen3.5-9B) and 591 street images in 197 NTAs. The prompt is "Is this
place cool?". The aim is the process of the judgment, not evidence of harm.

The sections [Three models](#three-models-f1019), [More adjectives](#more-adjectives-f1023)
and [Models and adjectives together](#models-and-adjectives-together-f1024) extend this
report to 3 models and 3 adjectives. The results are complete for all 9 pairs. Those
sections take priority where the older sections give only 2 models.

## Main result

The 2 models form the judgment in 3 parts. The parts are separate in the data.

| Part | Where it shows | Result |
|------|----------------|--------|
| Scene appraisal | Question-free patches, and the tokens `this` and `place` | The mean patch gives out-of-fold R2 0.82 (Gemma) and 0.75 (Qwen) of the cool margin. Aesthetic words track the margin before the word `cool` arrives |
| Sense of the word | The token `cool`, layers 12 to 28 | The largest axis at `cool` (42% to 63% of NTA variance) runs from temperature words to social words. It does not follow the verdict at L24. It exists for `cool` only (F10.24) |
| Application of a standard | The token `cool` onward, and the traces | The gain over the mean patch steps up at `cool`. In the traces, the image fixes the object and the value gives the sign |

The borough pattern comes from 2 axes. An urban axis runs from Manhattan and Brooklyn to
Staten Island. It matches the sense of the word. A status axis puts Manhattan above all
others. The Bronx sits near 0 on the first axis and low on the second, so its word lists
do not hold.

**Note:** The prompt is fixed, so the image is the only source of variation between
images. Thus a flexible decoder must reach R2 = 1 from the patches. The R2 values here
show how simple the map from the image is. They do not show a causal order.

## Question 1: how the judgment forms

### Decoders with no lens (F10.9)

A ridge decoder reads the margin from raw residuals, with 5 folds by NTA and 3 repeats.

| Input | Gemma R2 | Qwen R2 |
|-------|----------|---------|
| Mean patch, all layers | 0.820 | 0.753 |
| Mean patch, layer 0 only | 0.601 | 0.698 |
| Mean patch, best layer | 0.836 (L32) | 0.758 (L24) |
| 4 by 4 patch grid | 0.701 | 0.685 |
| Image-end token | 0.436 | 0.754 |

The table below gives the largest gain when a cell joins the mean patch. Holm covers
all cells of 1 model, and a paired t on NTA errors gives the test.

| Token | Gemma gain | Qwen gain |
|-------|------------|-----------|
| After the image | -0.008 | 0.027 |
| `Is` | 0.021 | 0.013 |
| `this` | 0.043 | 0.051 |
| `place` | 0.067 | 0.078 |
| `cool` | 0.124 | 0.178 |
| `?` | 0.127 | 0.177 |
| `no` | 0.145 | 0.163 |
| Last token | 0.177 | 0.238 |

| Holm result | Gemma | Qwen |
|-------------|-------|------|
| Cells with a positive gain | 130 of 375 | 99 of 192 |
| Cells with a negative gain | 18 of 375 | 2 of 192 |

The first positive cells are at `this`. The largest step is at `cool` in both models.
The gain stays flat through the instruction tokens. It rises again at the end of the
prompt. Thus a linear read of the scene holds about 3/4 of the margin, and the question
adds 0.12 to 0.18.

**Result of the null:** The plan said that no unique margin information before the answer
is a useful null. The null fails. The gain is small before `cool` and clear at `cool`.

See `figures/f10/mechanism/mech_gain.png`.

### Patches at each depth (F10.9c)

Patches see no question. All 53 adjectives read well from the mean patch, for example
`beautiful` 0.88, `nice` 0.88, `gritty` 0.87 and `green` 0.87 on Gemma. Thus a high patch
R2 is not specific to the word `cool`. The patches do not hold a stored answer to the
question. They hold a scene code from which many judgments are simple to read.

### Word lists and pairs of sets (F10.7, F10.8)

| Measure | Gemma | Qwen |
|---------|-------|------|
| Largest r of the cell margin with the answer margin at `cool` | 0.52 | 0.69 |
| The same at `?` | 0.63 | 0.87 |
| Aesthetic pair of sets, largest r before `cool` | 0.755 (`this` L32) | 0.715 (`place` L16) |
| Status pair, largest r up to `?` | 0.707 (`?` L36) | 0.646 (`?` L16) |
| Valence pair, largest r up to `?` | 0.760 (`?` L36) | 0.595 (`<|vision_end|>` L24) |
| Held-out word lists that pass Holm | 367 of 375 cells | 192 of 192 cells |

Before `cool`, the top words are pure appraisal.

| Cell | Up | Down |
|------|----|------|
| Gemma `this` L32 | illustrious, prince, delighted, gems, romantic | discharge, contempt, indictment |
| Qwen `this` L16 | splendid, stunning, glorious, exquisite | undesirable, poor, inadequate, jail |
| Gemma `place` L32 | ornament, cascading, delightful, exquisite | unclear, landfill |

At and after `cool`, the late layers turn to yes and no words in many languages
(conferma, oui, aucun, kein). Pairs stay valid there, but single sets do not, because
the whole vocabulary collapses onto the answer.

### The stage of each paper word (F10.13)

`rank_added` is the rank in the mean read minus the read with no image.

| Token | Words | Rank in the mean | Rank with no image | `rank_added` | Source |
|-------|-------|------------------|--------------------|--------------|--------|
| `place` | nearby, location | 3 | about 20,000 | 3 to 4 | The image |
| `?` | subjective, attractiveness | 7 | 22,500, 9,562 | 2 | The image |
| `?` | judging, debatable | 13, 11 | 4,165, 3,796 | 19, 20 | The image |
| `cool` | vibe, idea, funky | 2 to 5 | 2 to 6 | 5,300 to 6,800 | The prompt or the lens |
| `model` | renovated, fashionable, ugly | 616 to 4,403 | about 53,000 | 4 to 8 | The image, in the C2 contrast only |

Onset alone does not separate stages. Partial r starts at the first tokens for nearly
all sets.

## Question 2: the sense of the word (F10.15, F10.17)

This result is new. The paper does not state it.

| Measure at `cool`, layer 24 | Gemma | Qwen |
|-----------------------------|-------|------|
| Share of NTA variance on axis 1 | 0.63 | 0.49 |
| r of the fixed-list score with axis 1 | 0.99 | 0.99 |
| r of the score with the cool margin | 0.10 (p 0.051) | -0.11 (p 0.008) |
| r of the score with the `urban` probe | 0.70 | 0.62 |
| r of the score with traces that name temperature | -0.60 | -0.61 |
| Those traces, in the temperature decile of images | 0.525 | 0.559 |
| Those traces, in the social decile of images | 0.065 | 0.088 |

The score of each image agrees between the 2 models (r 0.67). The axis words are:

| End | Gemma L24 | Qwen L24 |
|-----|-----------|----------|
| Temperature | coolant, temperature, humidity, arctic, meteorological | temperature, cooling, weather, breeze, climate |
| Social | boutique, hipster, bodega, bohemian, nightclubs | street, vintage, retro, urban, graffiti |

The images at the temperature end show trees, roadside green and 1 blank truck wall.
The images at the social end show murals, graffiti, billboards, storefronts and a coffee
shop. The traces state the same choice. A typical line is "Is it cool in terms of
temperature? No, it's a photo."

Thus the image selects the sense of the word, and that choice is separate from the
verdict. An image can take the social sense and still get a "no".

**Limit:** The 2 fixed lists share words with the axis. The axis is the evidence, and the
score is only a readable measure of it. The temperature end may also hold a low-signal
image, with no street content for the social sense.

See `figures/f10/mechanism/mech_sense.png`.

## Question 3: aesthetic properties (F10.14, F10.8)

We asked the same model the same question with 53 other adjectives. Each row gives the
partial r with the cool margin, given `good`, `nice` and the borough.

| Adjective | Gemma | Qwen |
|-----------|-------|------|
| beautiful | 0.42 | 0.60 |
| pretty | 0.39 | 0.45 |
| historic | 0.52 | 0.28 |
| old | 0.55 | -0.02 |
| gritty | 0.47 | 0.07 |
| modern | -0.35 | 0.11 |
| clean | -0.38 | -0.29 |
| neat | -0.22 | -0.23 |
| green | -0.34 | -0.02 |
| ugly | 0.20 | -0.37 |

The models disagree on `old`, `gritty` and `ugly`. They agree that neat and clean work
against cool, after the control for general approval.

## Question 4: aesthetic values (F10.14, F10.11)

### From the probes

| Adjective | Gemma | Qwen |
|-----------|-------|------|
| interesting | 0.87 | 0.56 |
| unique | 0.60 | 0.43 |
| stylish | 0.54 | 0.46 |
| trendy | 0.44 | 0.49 |
| fashionable | 0.45 | 0.30 |
| famous | 0.26 | 0.44 |
| charming | 0.65 | 0.25 |
| boring | -0.71 | -0.35 |
| ordinary | -0.44 | -0.56 |
| safe | -0.26 | -0.34 |
| friendly | -0.19 | -0.32 |
| functional | -0.01 | -0.29 |

| Group of adjectives | Gemma R2 for cool | Qwen R2 for cool |
|---------------------|-------------------|------------------|
| Valence | 0.719 | 0.574 |
| Properties | 0.840 | 0.749 |
| Values | 0.924 | 0.810 |
| Neutral | 0.018 | 0.021 |
| All | 0.949 | 0.877 |
| All, without the values | 0.873 | 0.804 |

Thus cool reads as "interesting, not ordinary" more than as "pretty". Safe, friendly and
functional work against it. The neutral group gives a clean null.

### From the traces

We ran 7,092 answers with thinking on (0 errors, 11 cut). A separate model extracted
10,634 texts into spans.

| Family of labels | Gemma R2 of the margin | Qwen R2 of the margin |
|------------------|------------------------|-----------------------|
| Criteria | 0.572 | 0.468 |
| Inference | 0.235 | 0.198 |
| Evidence | 0.192 | 0.167 |
| All | 0.679 | 0.683 |

Stated values explain more of the margin than stated observations do.

| Label in the reasoning | Gemma r | Qwen r |
|------------------------|---------|--------|
| Utility, against | -0.59 | -0.61 |
| Beauty, against | -0.52 | -0.52 |
| Beauty, for | 0.37 | 0.33 |
| Vitality, for | 0.35 | 0.33 |
| Heritage, for | 0.49 | 0.20 |
| Nature, for | 0.33 | 0.32 |

"Only functional" is the strongest reason for a "no". In the overall share, character
leads (0.164), and comfort, edge, heritage and beauty follow (0.088 to 0.096). Edge is
mostly conditional, and it is a persona argument ("if you like urban grit"). Edge does
not raise the margin.

See `figures/f10/mechanism/mech_values.png`.

### The 2 models take a different route

| Measure | Gemma | Qwen |
|---------|-------|------|
| `cool_word`: yes, no, other | 0.329, 0.338, 0.333 | 0.754, 0.243, 0.004 |
| `cool_open`: yes, other | 0.055, 0.919 | 0.465, 0.521 |
| Main decision note | subjective 3,129, no taste 1,935 | safe default 732, subjective 622 |

Gemma often declines. A typical line is "Whether a place is cool is a matter of personal
opinion". Qwen gives an answer, and it often says that "yes" is the safe default.

## Question 5: visual descriptors and observations (F10.11, F10.12, F10.16)

### Words of the arguments

The contrasts use log odds with an informative prior, on generated text only.

| Contrast | Gemma descriptors | Qwen descriptors |
|----------|-------------------|------------------|
| Yes | nice, pleasant, classic, well-maintained, charming | classic, pleasant, peaceful, historic, lived-in, authentic |
| No | mundane, plain, unremarkable, ordinary, generic, utilitarian | mundane, boring, plain, ordinary, utilitarian, functional |
| Bronx | institutional, generic, mundane, typical, plain | institutional, mundane, everyday, functional, boring |
| Brooklyn | gritty, urban, historic, authentic, lived-in | authentic, historic, urban, lived-in, character |
| Manhattan | corporate, sterile, upscale, iconic, prestigious | tall, busy, vibrant, iconic, upscale |
| Queens | cozy, suburban, peaceful, quiet, traditional | suburban, quiet, peaceful, comfortable, cozy |
| Staten Island | suburban, peaceful, quiet, green, safe | suburban, standard, peaceful, quiet, ordinary |

| Contrast | Objects in the evidence, both models |
|----------|--------------------------------------|
| Bronx | school, overpass, bus, stop, institutional buildings |
| Brooklyn | fire escape, brownstone, row house, stoop, graffiti |
| Manhattan | glass, skyscraper, skyline, stone, water |
| Queens | house, lawn, yard, shrubs, siding |
| Staten Island | siding, power lines, driveway, lawn, utility lines |

### One object, 2 signs

The value gives the sign of an object. Each cell gives the share of spans for cool.

| Object | Value | Gemma | Qwen |
|--------|-------|-------|------|
| Brick | heritage | 0.91 | 0.98 |
| Brick | utility | 0.00 | 0.09 |
| Graffiti | edge | 0.86 | 0.97 |
| Graffiti | order | 0.00 | 0.05 |
| Power lines | edge | 0.54 | 0.80 |
| Power lines | order | 0.00 | 0.01 |
| Trees or greenery | nature | 0.92 | 0.99 |
| Trees or greenery | utility | 0.02 | 0.07 |

| Test | Gemma | Qwen |
|------|-------|------|
| Objects where the sign depends on the value (chi-square, Holm) | 18 of 22 | 20 of 25 |
| AUC of the sign from the value | 0.822 | 0.864 |
| AUC of the sign from the object | 0.853 | 0.856 |
| AUC of the sign from both | 0.921 | 0.922 |
| Mean ICC over 3 repeats, objects, `cool_open` | 0.551 | 0.551 |
| Mean ICC over 3 repeats, values, `cool_open` | 0.312 | 0.278 |
| Mean ICC over 3 repeats, objects, `cool_word` | 0.358 | 0.412 |
| Mean ICC over 3 repeats, values, `cool_word` | 0.231 | 0.244 |

Thus repeats of 1 image agree on what they see more than on the standard that they use.
Value and object each carry the sign, and together they give AUC 0.92.

See `figures/f10/mechanism/mech_objects.png`.

### Reads and arguments (F10.12)

The reads come from the one-word prompt. The arguments come from separate runs with
thinking on. A read holds the words of the argument of the same image, beyond the
borough.

| List | Gemma cells, Holm | Qwen cells, Holm |
|------|-------------------|------------------|
| Observations | 194 of 375 | 73 of 192 |
| Evidence | 168 of 375 | 65 of 192 |
| Criteria | 176 of 375 | 68 of 192 |
| Inference | 177 of 375 | 66 of 192 |

The strongest cells are `thought` and `<|channel>` on Gemma, and `<|im_start|>` on Qwen.
These tokens come just before free text. `this` is the strongest early token in both
models. The link at `cool` and `?` is weak, although the margin signal is strong there.
Thus observation content and verdict content peak at different tokens.

Named pairs also link. The table gives the largest r over all cells, Gemma then Qwen.

| Argument label | Read set | r |
|----------------|----------|---|
| Criterion edge | `p_grit` | 0.74, 0.74 |
| Criterion heritage | `p_old` | 0.66, 0.73 |
| Evidence of greenery | `p_nature` | 0.63, 0.69 |

**Limit:** The image holds both the read and the argument. A link shows shared content,
not a cause.

See `figures/f10/link/link_observation.png`.

## Boroughs (F10.10, F10.14, F10.15, F10.17)

### Probes

| Probe | Manhattan | Brooklyn | Bronx | Queens | Staten Island |
|-------|-----------|----------|-------|--------|---------------|
| cool, Gemma | +0.95 | 0.0 | -0.42 | -0.20 | -0.26 |
| cool, Qwen | +1.16 | 0.0 | -0.37 | -0.31 | -0.28 |
| urban, Gemma | +0.62 | +0.53 | +0.19 | -0.51 | -1.56 |
| trendy, Gemma | +1.41 | +0.12 | -0.27 | -0.53 | -0.64 |
| green, Gemma | -0.42 | -0.30 | -0.11 | +0.33 | +0.87 |
| Sense score, Gemma | +0.42 | +0.59 | +0.19 | -0.53 | -1.28 |

Cool varies by borough (F 11.9, 18.7; eta2 0.20, 0.28). The `urban` and `in a city`
probes give direct support to the urban recognition hypothesis of the paper. The older
gate test (F10.10) passed in both directions and for all sets. Thus it did not isolate a
gate, and the paper can cite the probe instead.

### Stability of the lists

We correct a value from earlier in this run. One split gave a negative value for the
Bronx (-0.13). With 40 random splits, the mean over 3 cells is near 0.2, with a wide
range.

| Borough | Gemma all axes | Gemma axis 1 out | Qwen all axes | Qwen axis 1 out |
|---------|----------------|------------------|---------------|-----------------|
| Manhattan | 0.88 | 0.80 | 0.84 | 0.86 |
| Queens | 0.85 | 0.67 | 0.79 | 0.71 |
| Brooklyn | 0.76 | 0.43 | 0.71 | 0.37 |
| Staten Island | 0.76 | 0.42 | 0.73 | 0.47 |
| Bronx | 0.22 | 0.25 | 0.10 | 0.25 |

| Finding | Evidence |
|---------|----------|
| Manhattan lists are independent of the sense axis | The mean stays at 0.8 or more when axis 1 goes |
| Brooklyn and Staten Island `cool` lists sit mostly on the sense axis | At `cool` L24 on Gemma, they drop from 0.91 and 0.89 to 0.37 and 0.20 |
| The Bronx list is weak on all axes | SD over splits is 0.21 to 0.29, and the range is -0.42 to 0.67 |

The word contrast of the Bronx has no stable content in the reads, but the Bronx text is
stable: institutional, generic, mundane, everyday. The Bronx is urban and not cool, so it
sits between the 2 axes.

### Held-out lists

These tests use 1 split of the NTAs. The table gives cells that pass Holm.

| Borough | Gemma | Qwen |
|---------|-------|------|
| Manhattan | 344 of 375 | 183 of 192 |
| Queens | 89 of 375 | 23 of 192 |
| Bronx, Brooklyn, Staten Island | 0 of 375 | 0 of 192 |

Lists that hold in both halves include:

| Cell | Words |
|------|-------|
| Gemma Manhattan `place` L36 | dazzling, sequins, ballroom, gala, champagne, peacock |
| Gemma Staten Island `place` L36 | carrots, blackberries, canola, sunflower, zucchini, mulch |
| Gemma Queens `?` L44 | jokes, marshmallows, popsicle, funny, kiddos |
| Gemma Brooklyn `?` L44 | bourgeois, gauche, hipster, arbitrage |
| Qwen Brooklyn `cool` L20 to L24 | neighborhood, barrio, unemployment, landlords, homeless |

See `figures/f10/mechanism/mech_borough.png` and `figures/f10/probe/probe_borough.png`.

### Paper words on the sense axis (F10.18)

The stark `cool` words of the paper are not specific to 1 borough. They sit on axis 1.

| Words, Gemma-4-31B, L20 to L24 | Axis 1 rank of 53,999 | Borough t | t with axis 1 out |
|--------------------------------|-----------------------|-----------|-------------------|
| Bronx and Brooklyn words | Median 43 to 69, social end | Brooklyn up to 6.5 | Brooklyn up to 2.9 |
| Staten Island and Queens words | 53,837 to 53,998, temperature end | 4.9 to 7.0 | -1.3 to 4.0 |

We did not apply Holm after the removal. `illegal` and `vandalism` stay in the Bronx (t 3.49, 3.44). Do not cite them without a test.

### Place probes (F10.18)

All 591 images are in New York City. Thus the `in New York` probe measures accuracy.

| Model | All | Manhattan | Brooklyn | Bronx | Queens | Staten Island |
|-------|-----|-----------|----------|-------|--------|---------------|
| Gemma-4-31B | 0.616 | 0.833 | 0.780 | 0.694 | 0.412 | 0.208 |
| Qwen3.5-9B | 0.885 | 0.948 | 0.931 | 0.919 | 0.831 | 0.729 |
| Qwen3.6-27B | 0.888 | 0.948 | 0.918 | 0.901 | 0.870 | 0.708 |

**Update (2026-11-02):** the Qwen3.6-27B row comes from a new probe run (F10.26, task 5). The script
`scripts/f10/probe_place.py` makes this table for 3 models on the CPU, and it reproduces the rows of
`hop_claim_axis.py` to 0.0. It also checks each probe run against the answer store of the same question
(`figures/f10/probe/probe_check.csv`): r .986 to .998 for `cool` and `urban` in all 3 models.

## Layers of each phase (F10.20)

`scripts/f10/phase_layers.py` gives the bands of the prompt figure, panel (b). Each band
is the first unbroken run of saved layers that passes a fixed rule. The output is
`figures/f10/mechanism/phase_layers.csv`.

| Phase | Token | Rule | Gemma | Qwen |
|-------|-------|------|-------|------|
| Scene code | Mean patch | From L0 to the peak of R2 | L0 to L32 | L0 to L24 |
| Sense of the word | `cool` | Axis 1 holds at least 0.40, and the score gives abs(r) of at least 0.95 with it | L12 to L32 | L4 to L28 |
| Standard | `?` | Gain is at least half of its peak, up to the verdict | L32 to L44 | L12 to L28 |
| Verdict | Last token | r with the answer margin is at least 0.95 | L44 to L56 | L28 |

**Note:** This rule gives L12 to L32 for the sense on Gemma. The table at the top of this
report gives L12 to L28, the layers where axis 1 holds at least 0.55.

**Limit:** The bands are at different tokens. An overlap of 2 bands does not show 1 shared
step. The thresholds are our choice, and a different threshold moves a band edge by 1 saved
layer or more.

## 2 estimands (2026-11-01)

The paper asks 2 questions. We sort each read into 1 of them.

| Estimand | Question | Evidence | Limit |
|----------|----------|----------|-------|
| Embedded conception | What aesthetic conception does the model hold with no image? | The no-image read, and words with the same rank in both reads | The read depends on the lens, and the lens was fit on WikiText |
| Image-caused change | What does an image change in that conception? | The mean read minus the no-image read (`rank_added`) | The no-image text also has no vision tokens, thus the contrast is image and slot together |
| Image-specific change | What does 1 image change, compared with others? | Contrasts between images, borough lists, the sense score | The image is the cause. A named feature of the image is not isolated |

**Note:** The fixed-prompt limit concerns the order of steps inside the model. It does not
stop the image contrast from being causal: we add the image and we change nothing else in
the text.

**Note:** The row "`cool` words and end-token words" in the table below is evidence for the
embedded conception. It is not a reason to cut those words.

### Words by estimand (F10.21)

`scripts/f10/plot_ranks.py` ranks each store word with an image and with no image. The
table counts words of Gemma-4-31B in the top 100. `rank_cells.csv` holds all words.

| Cell | Embedded (top 100 in both) | Raised by the image (no-image rank past 1,000) | Lowered (the reverse) |
|------|----------------------------|-----------------------------------------------|-----------------------|
| `place` L40 | 2 | 95 | 91 |
| `cool` L36 | 80 | 0 | 0 |
| `?` L40 | 21 | 42 | 36 |
| `model` L44 | 5 | 68 | 71 |

### The sample design and the borough lists (2026-11-01)

The sample holds all 197 residential NTAs, with 3 random images from each. Thus the NTAs are
strata, and the mean contrast C3 is an unbiased estimate of each borough mean.

| Item | Status |
|------|--------|
| Bronx lists | The held-out tests treat the NTA as a random unit. A failure there does not show that the Bronx has no pattern, and it does not make the C3 list wrong. Read "fail" in this report with this limit |
| Bronx concentration | 10 of 111 images hold 56% of the top-word effect at `cool` L24. This can be a true property of a skewed borough mean |
| Selection | C3 picks the top words of 54,000. The sample design does not remove this selection effect |
| "No other borough" at `place` | Still wrong. Queens and Staten Island also show words in many languages |

`scripts/f10/plot_borough_layers.py` draws the C3 lists of `cool`, with no split
and no test.

## Three models (F10.19)

`scripts/f10/hop_compare.py` compares Gemma-4-31B, Qwen3.5-9B and Qwen3.6-27B. Gemma-4-31B and
Qwen3.6-27B are close in parameter size, and the 2 Qwen models are close in family. Value triples
below give G-Q27, G-Q9, Q27-Q9.

| Measure | G-Q27 | G-Q9 | Q27-Q9 | Result |
|---------|-------|------|--------|--------|
| Margin r, image | .58 | .72 | .69 | |
| Margin r, NTA mean | .62 | .76 | .77 | G-Q27 is lower than the other 2 (Williams, each p < .001) |
| Answer kappa | .09 | .06 | .33 | Both Qwen models lean to yes |
| Word r of the mean readout | .28 | .26 | .44 | Family: Holm p .017, .003. Size: Holm p 1 |
| Residual RDM (rsa) | .36 | .37 | .58 | Family: Holm p .010, .010. Size: Holm p 1 |

| Model | Yes rate | Mean-patch R2 | Gain at `place` | Gain at `cool` | Sense at `cool` L24: no image, mean, share above no image |
|-------|----------|---------------|-----------------|----------------|-----------------------------------------------------------|
| Gemma-4-31B | .38 | .82 | .07 | .12 | +0.71, +1.12, .90 |
| Qwen3.5-9B | .95 | .75 | .08 | .18 | -1.66, -1.91, .34 |
| Qwen3.6-27B | .92 | .56 | .07 | .19 | -0.81, -1.67, .16 |

Parameter size does not bring 2 models closer. Family does, for words and residuals. The
mechanism replicates: a scene code, then a step at the adjective. An image moves the sense of
`cool` to the social side in Gemma-4-31B, and to the temperature side in both Qwen models.

**Note:** Each family result rests on 2 Qwen models. Read it as a hypothesis for a wider set
of models. The Qwen3.6-27B margin SD on `cool` is .48.

**Margin fault (corrected):** Qwen3.6-27B writes ` yes` and ` no` with a leading space, and the
first margin read the no-space forms. That run gave a false r of .36, and its files are in
`data/f10/hop_logit/qwen3.6-27b-forms4`. The margin now pools 8 forms. The change for the 2
older models is small, thus the numbers of this report stand:

| Model | r with the old margin | Largest change | Sign changes |
|-------|-----------------------|----------------|--------------|
| Gemma-4-31B | 1.000 | .002 | 0 of 591 |
| Qwen3.5-9B | .998 | .34 | 4 of 591 |

The G-Q9 row uses the old margin. The `-forms8` folders of `data/f10/hop_logit/` hold the new one.

## More adjectives (F10.23)

The J++ lenses stay the same. A new adjective needs only a new read, not a new fit.

| Item | Value |
|------|-------|
| Prompts | "Is this place nice? Answer with one word: yes or no." and the same with `ordinary` |
| Models | Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B |
| Stores | `data/f10/hop_all/{model}-nice`, `data/f10/hop_all/{model}-ordinary` |
| Positions | Text tokens only. `after_place` holds the adjective in each store, and `after_cool` holds `?` |
| Answers | `margin` pools 8 forms: `Yes`, `yes`, ` Yes`, ` yes`, and the same for no |

The stores hold no patches. The image comes before the text and the attention is causal,
thus the patches of each adjective equal the patches of the `cool` store.

Each array task of `hop_all.sbatch` is 1 worker. Many workers can fill 1 store.

| Part | Rule |
|------|------|
| Priority worker | `sbatch --clusters=g3 --qos=nissenbaum --array=0-5%1 scripts/f10/hop_all.sbatch`. This QoS has 1 GPU |
| Preemptible workers | `sbatch --clusters=g3 --qos=g3 --array=6-23 scripts/f10/hop_all.sbatch`. A requeue resumes |
| Claims | `claims/shard-*` holds the job id. A requeued job takes its claim again. A claim is free after 30 minutes |
| Done | The answers file of a shard is the last write |

**Warning:** A worker that does not come back leaves its shard open until a new worker
starts. Count the 12 answers files of each store before you read it.

All 6 stores are complete: 12 shards, 591 images, finite values. A check of each file found
no fault after a race on the shared temporary name. `atomic_save` now adds the process id.

The first token sets the outcome (D167). The rates are yes, no, unable.

| Model | `cool` | `nice` | `ordinary` |
|-------|--------|--------|------------|
| Gemma-4-31B | .38, .62, 0 | .26, .24, .50 | .97, .03, 0 |
| Qwen3.5-9B | .95, .05, 0 | .84, .16, 0 | .76, .24, 0 |
| Qwen3.6-27B | .92, .08, 0 | .74, .26, 0 | .94, .06, 0 |

**Warning:** The Gemma-4-31B answer to `nice` is not yes or no on 295 of 591 images. The margin
of these images compares 2 answers that the model did not give.

The non-answers are refusals, not a fault. The texts are `Subjective`, `Unable to determine.`, and
"Whether a place is "nice" is a matter of personal opinion." The same images get yes or no at `cool`.

| Group | Refusal share |
|-------|---------------|
| Bronx, Brooklyn | .59 |
| Queens | .48 |
| Staten Island | .40 |
| Manhattan | .32 |
| Lowest, middle, highest third of the `cool` margin | .39, .71, .40 |

Borough against answer kind: chi-square p = 1e-07. A refusal is a third outcome: yes, no or unable
(`scripts/f10/hop_outcomes.py`, `figures/f10/outcomes/`). Read the `nice` margin of Gemma-4-31B
only on the 296 yes or no images. On those images, r with the
`cool` margin is .81.

The refusal is a property of Gemma-4-31B on this word. It is not a mark of a hard image. On the
295 refused images, the Qwen margins are positive and not near 0.

| Model | Mean margin on refused images | Yes rate on refused images | Yes rate on the other images |
|-------|-------------------------------|----------------------------|------------------------------|
| Qwen3.5-9B | +1.44 | .94 | .73 |
| Qwen3.6-27B | +0.91 | .81 | .68 |

With no image, the first token of Gemma-4-31B is `Unknown` at .72 for `nice`, `Yes` at 1.00 for
`cool`, and `No` at .999 for `ordinary`. The refusal words of the readout show at the token after
the image, before any adjective, but they are a list without a test. The store vocabulary does
not hold `Subject`, `Unable`, `No` or `Whether`, thus the readout cannot test those tokens.

## Models and adjectives together (F10.24)

Two analyses read all 9 pairs. Code: `scripts/f10/hop_models.py`, `scripts/f10/hop_adjectives.py`.
Tables: `figures/f10/models/`, `figures/f10/adjectives/`. Value triples give Gemma-4-31B,
Qwen3.5-9B, Qwen3.6-27B unless a row says otherwise.

### What holds in all 3 models

| Result | Values | Test |
|--------|--------|------|
| Scene code on `cool` (mean-patch R2) | .82, .75, .56 | Out-of-fold ridge |
| Gain over the mean patch at `place` | .07, .08, .07 | Out-of-fold ridge |
| Gain at `cool` | .12, .18, .19 | Out-of-fold ridge |
| Gain at `?` | .13, .18, .21 | Out-of-fold ridge |
| Largest axis at `cool` L24 | .63, .49, .57 of the NTA variance | SVD |
| Sense band at `cool` | L12 to 32, L4 to 28, L8 to 44 | F10.20 rule |
| Held-out word lists | 367 of 375, 192 of 192, 384 of 384 | Holm |
| Embedded words at `cool` | 80, 84, 65 to 82 of the top 100 | Rank count |
| Manhattan against the rest, `cool` | t 6.75, 8.46, 7.49 | t, Holm over 9 tests |
| Manhattan against the rest, `nice` | t 2.26, 2.91, 2.31 | t, Holm over 9 tests |
| Manhattan against the rest, `ordinary` | t -6.73, -7.10, -6.52 | t, Holm over 9 tests |

### What does not hold in all 3 models

| Result | Values | Reading |
|--------|--------|---------|
| Verdict band on `cool` | L44 to 56, L28, not reached | Qwen3.6-27B reaches r .92 at L60. Cause not found. It reaches .977 on `nice` and .976 on `ordinary` |
| Verdict band on `ordinary`, Gemma-4-31B | Not reached (r .927) | The other 2 models reach it. The yes rate is .97, thus the range is small |
| Image raises words at `place` | 95, 0, 0 | Gemma only |
| Embedded words at `place` | 2, 78, 62 to 75 | Family |
| Sense shift of `cool` | .90, .34, .16 at L24 | Social in Gemma, temperature in Qwen3.6-27B |
| Top words of C3 | Top-50 Jaccard .00 to .01 | The borough sign agrees. The words do not |

The sense shift depends on the layer. The share of images above the no-image value in
Gemma-4-31B is .98, .92, .80, .93, .90, .80, .75, .67, .57, .36, .58, .53, .57 at L8 to L56. Thus
the social side holds from L8 to L32, not at all layers. Qwen3.5-9B changes sides: .93 at L4, .34
at L24, .63 at L28. Qwen3.6-27B stays low at most layers.

"The sense does not follow the verdict" holds at L24 only. In Qwen3.6-27B, r rises to .29 at L52.

### The 3 adjectives

| Measure | Gemma-4-31B | Qwen3.5-9B | Qwen3.6-27B |
|---------|-------------|------------|-------------|
| `cool`-`nice` margin r, NTA | .80 (n 296 images) | .62 | .48 |
| `cool`-`ordinary` margin r, NTA | -.36 | -.40 | -.28 |
| `nice`-`ordinary` margin r, NTA | .25 (n 296 images) | .25 | .21 |

| Measure, between models | G-Q27 | G-Q9 | Q27-Q9 |
|-------------------------|-------|------|--------|
| `cool` margin r | .58 | .72 | .69 |
| `nice` margin r | .90 (n 296) | .89 (n 296) | .87 |
| `ordinary` margin r | .78 | .76 | .75 |
| `cool` agreement after the yes share is matched | .71 | .77 | .90 |

1. `ordinary` is not the mirror of `cool`. The largest negative NTA r is -.40.
2. Only `cool` has a temperature-to-social axis. `nice` and `ordinary` pass at no layer, in any
   model. Thus the sense split is a property of `cool`, not of aesthetic words in general.
3. The scene code and the gain at `?` appear for all 3 adjectives. Their layers follow the model,
   not the adjective.
4. The verdicts on `nice` and `ordinary` agree across models, with no family gap and no size gap.
   The gap between models on `cool` is mostly a yes offset, plus a real difference in rank order.
5. The offset on `cool` does not mean that Qwen has a weak judgment. The margin SD is 7.87, .70,
   .48, thus compare ranks, not raw margins.

### Limits of F10.24

| Limit | Effect |
|-------|--------|
| Gemma `nice` | Margin results use the 296 yes or no images. They are not a random sample |
| New stores | They hold no patches. The results use the `cool` patches, which are equal because attention is causal |
| Socioeconomic tests | This run holds none. It gives a borough order only |
| "Outsized" | This run has no baseline. Do not cite it for that word |
| 3-way borough overlap | Not done. Only pairs of adjectives |
| Depth tests | The 8 matched depths are not independent, thus the family tests are approximate |

## 14 prompts (F10.25)

The set of prompts grows from 3 to 14, to test how far the results generalize (2026-11-01).
The form of the question stays the same: "Is this {noun} {adjective}? Answer with one word: yes or no."

| Item | Value |
|------|-------|
| Adjectives | `cool`, `nice`, `ordinary`, `pretty`, `ugly`, `fancy`, `urban` |
| Nouns | `place`, `scene` |
| Prompts | 7 x 2 = 14. `cool`, `nice` and `ordinary` with `place` are done |
| New stores | 11 prompts x 3 models = 33. Each store has no patches |
| Names | `{model}-{adjective}` for `place`, `{model}-scene-{adjective}` for `scene` |
| Disk | About 113 GB for each prompt, thus about 1.2 TB |
| Positions | `place` holds the noun in each store. `after_place` holds the adjective |

All these words are 1 token in the 3 models, thus the positions of F10.5 hold. `hop_all.py` and
`hop_logit.py` 1.2.0 take `--noun`, and `hop_depth.ask` builds the question.

`hop_all.sbatch` and `hop_logit.sbatch` hold the 33 items in PLAN. Do not read PLAN as the plan of
F10.23, because the older items left the file. `F10_ADJ` and `F10_NOUN` run 1 prompt with no array.

```bash
EX=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02
sbatch --clusters=g3 --qos=nissenbaum --exclude=$EX --array=0-32%1 --job-name=f108p scripts/f10/hop_all.sbatch
sbatch --clusters=g3 --qos=g3 --exclude=$EX --array=33-131%16 --job-name=f108w scripts/f10/hop_all.sbatch
sbatch --clusters=g3 --qos=g3 --exclude=$EX --dependency=afterany:<p>:<w> --array=0-32%12 --job-name=f108l scripts/f10/hop_logit.sbatch
```

`outputs/f10_check/check_14.py` checks the 33 read stores and the 33 answer stores.

The plan of the analysis is in `f10-14-prompt-analysis-plan.md` (F10.26).

A shared aesthetic direction over the prompts is in `f10-shared-direction.md` (F10.27).

### Result of the run (2026-11-02)

All 33 read stores and all 33 answer stores are complete, and they pass the check.

| Check | Result |
|-------|--------|
| Read stores | 12 shards, 591 images, the right question, no patches, the right tokens at `place`, `after_place`, `after_cool` |
| Prefix across the 11 stores of a model | gap 0.0 in the 3 models |
| Text at ` scene` across the 7 `scene` stores | gap 0.0 in the 3 models |
| Answer stores | 591 rows, 591 images, a finite margin for each image |
| `max_output_gap` | 0.0 in all 33 stores |
| `max_capped_gap` | 0.0 for Qwen. Gemma 8e-5 to 2.3e-4, the same size as the `cool` and `nice` stores (float32 and the softcap of 30) |

Only a model with a softcap writes `uncapped/`, thus the Qwen answer stores have no `uncapped/`.
`hop_logit.py` 1.2.0 wrote 11 answer stores, and 1.2.1 wrote 22. The 2 versions give the same output.

### Faults in the run

| Fault | Cause | Fix |
|-------|-------|-----|
| Tasks fail at the start with exit 0:53 and no log | Unknown. It came in bursts on g3-compute-04 (9 tasks) and g3-compute-02 (6 tasks). Both nodes also completed other tasks | Exclude the node. A later array task covers the store |
| 1 task: `MODEL_LOCAL: unbound variable` | 2 workers on 1 node copied the same model to `/scratch` at the same time, and the rsync of 1 failed | None. Another worker covered the store |
| 12 orphan `*.tmp` files | Preempted workers. The final file of each is newer and has the same size | Deleted after all read workers stopped |
| The answer step ran at 0.01 to 0.04 images/s | 1.2.0 read each reads file through a memory map, in 1 strided pass for each layer. This made many small NFS requests, and 24 tasks saturated NFS | 1.2.1 reads each file once, in sequence. The speed went to 0.11 to 0.23 images/s |

One test on a Qwen3.5-9B reads file: 1 strided layer pass took 941 s for 209 MB, and a sequential load took 213 s
for 1,673 MB. A higher task limit did not help with 1.2.0, because each task became slower by the same amount.

**Warning:** a requeue of a running task gives its GPU back to the queue. On a full cluster, other users can
take it. After the requeue, 4 of 21 tasks ran again in the first 10 minutes.

## 14 prompts: results (F10.26)

This section reads the 42 stores of F10.25: 7 adjectives x 2 nouns x 3 models (2026-11-02).
Code: `scripts/f10/hop_prompts.py`. Tables and figures: `figures/f10/prompts/`. Summaries:
`data/f10/prompts/`. The plan is in `f10-14-prompt-analysis-plan.md`. A cell is 1 prompt in 1
model, thus 42 cells. A result generalizes only when it holds in all 42 cells. Value triples give
Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B. Mid depth is the saved layer nearest to relative depth .5:
L28, L16, L32. The margin pools 8 answer forms for all prompts (D171).

### Task 0: the new code reproduces F10.24

| Check | Result |
|-------|--------|
| Checks on the 3 old prompts | 144, in `t0_reproduction.csv` |
| Agreement | 142 checks agree to 1e-8 or better |
| Differences | The verdict r at the last layer of `cool`: .001 (Gemma-4-31B) and .016 (Qwen3.5-9B) |

The 2 differences come from the margin file. F10.24 read a margin with 4 answer forms for these 2
cells, and the new code reads 8 forms. The gate of the plan passed, and the analysis went on.

### Task 1: outcomes

| Result | Values |
|--------|--------|
| Gemma-4-31B gives no verdict (`unable`) | `nice`: 295 of 591 (`place`), 58 (`scene`). `pretty`: 125, 2. `ugly`: 571, 508 |
| The Qwen models give no verdict | 0 images in all 28 cells |
| Borough x outcome, chi-square, Holm over 42 | 25 of 42 cells pass. `fancy` and `urban`: 6 of 6. `nice`: 2 of 6. `ugly`: 1 of 6 |
| Noun x outcome, chi-square, Holm over 21 | 8 of 21 pass. The noun moves the yes share of `cool` from .38 to .72 in Gemma-4-31B, and from .92 to .39 in Qwen3.6-27B |
| Refusals by borough, Gemma `place nice` | `unable` share: Manhattan .32, Staten Island .40, Queens .48, Brooklyn .59, Bronx .60 |

Gemma-4-31B refuses `ugly` on 97% and 86% of the images, thus its 2 `ugly` cells rest on 20 and 83
images, and their values below are not reliable. The refusal of `nice` is not uniform over the
boroughs: the Bronx and Brooklyn get `unable` 2 times as often as Manhattan. The Qwen models
answer every image. Figure: `t1_outcomes.png`.

### Task 2: the mechanism grid

The pass rules are fixed in `RULES` of the script (D174). `t2_grid.csv` holds the values of each
cell, and `t2_counts.csv` the counts. Figure: `t2_grid.png`.

| Result | Rule | Cells of 42 | Cells that fail |
|--------|------|-------------|-----------------|
| Embedded words | At the adjective, at least 50 of the top 100 words are in the no-image top 100, at mid depth | 42 | none |
| Step at the adjective | Gain at the adjective above the gain at the noun, Holm p below .05 | 41 | Gemma `place ugly` (20 images) |
| Gain at `?` | At least .05, Holm p below .05 | 41 | Gemma `place ugly` |
| Image raises no word at the adjective | 0 words from past rank 1,000 into the top 100, at mid depth | 40 | Qwen `scene cool` (3 and 12 words) |
| Scene code | Mean-patch R2 at least .50 | 39 | Gemma `ugly` x 2 (.25, .48); Qwen3.6-27B `scene ordinary` (.47) |
| Verdict band | Lens margin r at least .95 at the last token | 34 | Gemma `ordinary` x 2, `urban` x 2, `scene ugly`; Qwen3.6-27B `cool` x 2, `scene ordinary` |
| Axis 1 share | At least .40 of the NTA variance at the adjective, mid depth | 27 | `ugly` x 5, `ordinary` x 4, `fancy` x 3, `nice` x 3 |
| Manhattan differs from the rest | Either sign, Holm over 42 | 26 | `nice` x 6, `pretty` x 5, `ugly` x 4, Qwen3.6-27B `scene ordinary` |
| Gain at the noun between .03 and .15 | | 21 | `urban` x 6, `fancy` x 5, `scene` cells with .00 to .03 |
| Manhattan above the rest | Positive sign, Holm over 42 | 19 | `ordinary` x 6 (negative), `ugly` x 6, `nice` x 6, `pretty` x 5 |
| Axis 1 does not follow the verdict | abs r with the margin below .20, at mid depth | 12 | 30 cells; see task 3 |
| Sense band with the `cool` lists | Axis 1 share at least .40 and abs r at least .95 with the `cool`-list score | 6 | all 36 cells of the 6 other adjectives |

Three results generalize or almost generalize: the embedded conception at the adjective (42), the
step at the adjective and the gain at `?` (41, the 1 failure has 20 images), and the scene code
(39). The scene code falls to .47 on Qwen3.6-27B `scene ordinary`, where the yes share is .98 and
the range of the margin is small. The verdict band fails in 8 cells, all with a peak r between .92
and .94, thus the rule is tight, not absent. The sense band with the `cool` word lists is a
property of `cool` (6 of 6) and of no other adjective (0 of 36). The borough results depend on the
adjective. The gain at the noun is not a fixed value: `urban` and `fancy` read almost nothing
from the noun residual beyond the mean patch.

| Band layers | Gemma-4-31B | Qwen3.5-9B | Qwen3.6-27B |
|-------------|-------------|------------|-------------|
| Mean-patch R2 peak | L24 to 44 | L8 to 24 | L32 to 52 |
| Largest gain at the adjective | L36 to 40 | L16 to 20 | L44 to 52 |
| Largest gain at `?` | L36 to 56 | L20 to 24 | L44 to 60 |
| Verdict band, where reached | L40 or 44 to 56 | L24 or 28 to 28 | L52 to 60 |
| Sense band, `cool` only | L12 to 32 (`place`), L12 to 24 (`scene`) | L4 to 28, L8 to 28 | L8 to 44, L12 to 40 |

The layers follow the model, not the prompt, as F10.24 found for 3 adjectives.

### Task 3: the sense of each word

Rule: name a sense axis only when the words at its 2 ends name 2 senses of the word. Lists:
`t3_axis_words.csv` (mid depth, L24 and the peak layer). Test: `t3_sense_tests.csv`.

| Adjective | Words at the 2 ends of axis 1, at the adjective, mid depth | Sense axis? |
|-----------|-------------------------------------------------------------|-------------|
| `cool` | temperature, coolant, thermometer, arctic, scientists against bodega, boutique, hipster, graffiti, swagger, jazz, punk | Yes, in 6 of 6 cells |
| `nice` | Qwen: homeowners, suburban, mortgage, township against tourist, glacier, hiking, shoppers. Gemma: tourism, solstice against homeowner, condo | No. The ends name kinds of place, not senses of `nice`. No city Nice |
| `pretty` | flowers, floral, gardens, beautifully against profanity (Gemma) or bleak, negative, useless (Qwen) | No. The ends give beauty against its absence. No adverb sense |
| `fancy` | Qwen3.5-9B: restaurant, dining, hotel against rural, forest. Qwen3.6-27B: homeowners, mansion, estate against restaurant, food, cuisine | No. The ends name kinds of place. No verb or noun sense |
| `urban` | loft, iconic, designers against or, neither, either in 5 languages | No. The down end is function words. No music or style sense |
| `ordinary` | suburban, homeowners, families against tourist, visitors, seafood | No. The yes end is a residential conception |
| `ugly` | city, skyline, subway, aviation against rural, pasture, farm | No. An urban-against-rural axis of the image |

| Test | Result |
|------|--------|
| r of the axis 1 score with the NTA mean margin, t test, Holm over 42 | 29 of 42 pass |
| The 6 `cool` cells | 0 pass. abs r .03 to .19 |
| The 36 other cells | 29 pass. abs r up to .97 (`urban`) and .95 (`scene pretty`) |
| Cells that fail among the 36 | Gemma `nice` x 2, Gemma `ugly` x 2, Gemma `scene ordinary`, Qwen3.6-27B `place ugly`, `place fancy` |

Only `cool` has 2 senses on its largest axis. For the other adjectives, axis 1 at the adjective is
an axis of the image, and in Qwen it is close to the same axis for `nice`, `ordinary` and `fancy`:
residential and suburban against tourist and commercial. For `urban`, `pretty`, `nice` and
`ordinary`, this axis follows the verdict (abs r .4 to .97). Thus the F10.24 picture "the sense
comes before the verdict, and does not follow it" is a property of `cool`, not of the judgment.

### Task 4: `pretty` against `ugly`

Table: `t4_pretty_ugly.csv`. Values give `place`, then `scene`. The distance from -1 is in brackets.

| Measure | Gemma-4-31B | Qwen3.5-9B | Qwen3.6-27B |
|---------|-------------|------------|-------------|
| r of the 2 margins, NTA means | -.86 (.14; 19 NTAs), -.79 (.21; 66 NTAs) | -.88 (.12), -.86 (.14) | -.91 (.09), -.90 (.10) |
| r of the 2 margins, images | -.86 (20 images), -.80 (83) | -.84, -.83 | -.90, -.89 |
| Cosine of the 2 margin decoders at the noun, mid depth | -.09, -.11 | -.37, -.37 | -.60, -.57 |
| Cosine of the 2 C3 contrasts at the adjective, mean over boroughs | +.15, +.14 | +.43, +.41 | +.37, +.33 |
| `cool` against `ordinary`, r of the margins, NTA means | -.36, -.05 | -.39, -.39 | -.28, +.07 |

`ugly` is the mirror of `pretty` in the answer: the NTA means correlate at -.86 to -.91 in the
Qwen models, against -.40 at most for `cool` against `ordinary`. The residual does not mirror in
the same way: the 2 ridge decoders at the noun point in directions with cosine -.09 to -.60. The
borough contrasts C3 have the same sign for both words, because the image sets them. Gemma
answers `ugly` on 20 and 83 images only, thus its values are not reliable.

### Task 5: `urban`

Tables: `t5_urban_borough.csv`, `t5_urban_partial.csv`, `t5_urban_sense.csv`, `t8_place_probes.csv`.

| Question | Result |
|----------|--------|
| Q1: borough order of the `urban` margin | Manhattan > Brooklyn > Bronx > Queens > Staten Island, in 6 of 6 cells |
| Q1: t against the rest, Holm over 30 | Manhattan 4.2 to 6.4, Brooklyn 3.2 to 4.6, Queens -5.2 to -6.4, Staten Island -6.1 to -7.3: all pass. Bronx 1.2 to 1.7: none pass |
| Q2: partial r of Manhattan with each margin, given the `urban` margin, NTA means | `cool` .40 to .52, `fancy` .52 to .61, `ordinary` -.08 to -.39, `nice` .23 to .44, `pretty` .33 to .45, `ugly` -.23 to -.60 |
| Q2: F of the borough on each margin, given `urban` | Passes .05 in 34 of 36 cells. Fails: Gemma `place ugly`, Qwen3.6-27B `scene ordinary` |
| Q3: r of the `cool` sense score (L24, `cool` lists) with the `urban` margin, NTA means | .78, .81; .74, .61; .68, .72 |
| Q3: r of the `cool` margin with the `urban` margin | .07, .15; .12, .22; .21, -.03 |
| Place probes of F10.18, 3 models | r of the `urban` probe with the `urban` store: .988 to .999. `in a city`: .97 to .99. `in New York`: .71 to .79 |

The `urban` margin has the same borough order in all 6 cells, and the Bronx sits with Brooklyn
and Manhattan, not with Queens and Staten Island. The borough effect of each other adjective
remains when the `urban` margin is held fixed. For `nice` and `pretty`, the partial r is larger
than the raw r, because these margins run against `urban` (r -.33 to -.53). The social end of
the `cool` sense axis is the `urban` answer: the sense score predicts the `urban` margin at r .61
to .81, and the `cool` margin at .22 at most. The F10.25 `urban` store reproduces the F10.18
probe (r .988 to .999). The Qwen3.6-27B probe run is new (2026-11-02, GPU job 117965, 53 words, 591
images). Its `in New York` share is .888, with the same gradient: Manhattan .948, Staten Island .708.
See "Place probes (F10.18)" above for the 3-model table.

### Task 6: `place` against `scene`

Tables: `t6_noun_margins.csv`, `t6_noun_mechanism.csv`, `t6_noun_words.csv`.

| Measure | Result |
|---------|--------|
| r of the 2 margins, NTA means, 21 pairs | .61 to .99. Lowest: `ordinary` (.74, .93, .61), `cool` (.89, .86, .78) |
| Paired t of the NTA means, Holm over 21 | 18 of 21 differ. Not: Qwen3.5-9B `pretty`, `urban`; Qwen3.6-27B `ordinary` |
| Manhattan sign | The same for both nouns in 21 of 21 pairs |
| Mechanism measures, `scene` minus `place` | Scene R2 within .07 in 19 of 21. Gains within .06 in 19 of 21. Embedded words within 9, except Qwen `scene cool` (-19, -27) |
| Yes share of `cool` | .38 to .72 (Gemma), .95 to .90 (Qwen3.5-9B), .92 to .39 (Qwen3.6-27B) |
| Embedded conception at the noun, `scene` minus `place` | `scene`: cinematic, film, screenplay, storyboard. `place`: postcode, hostel, landfill (Gemma); tourist, locals, lodging (Qwen3.6-27B) |
| Image-caused change at the noun, `scene` minus `place` | `scene` end: ecosystem, outdoor, daylight, urban, sidewalk. `place` end: screenplay, theatrical, actress, rehearsals |
| r of the 2 image-caused contrasts at the noun | .16 to .55 at L24 to 32. .92 at Gemma L56 |

The noun changes the answer level and keeps the order: the yes share of `cool` moves by .34 and
.53, in opposite directions in the 2 models, but the NTA ranks agree at r .78 to .89 and the
Manhattan sign holds. The mechanism measures change little. The 2 estimands at the noun differ in
a clear way: with no image, `scene` means a film scene; with an image, that sense leaves, and the
`scene` readout gains words of the outdoor setting. The image-caused contrasts of the 2 nouns
agree at r .16 to .55 only at mid depth.

### Task 7: a shared aesthetic direction, in the residual

F10.27 (`f10-shared-direction.md`) finds a shared direction in the readout. This task repeats
the question in the residual, with ridge decoders, and the 2 agree. Tables: `t7_*.csv`. Figure:
`t7_shared.png`. Decoders are fit on z-scored margins in 5 NTA folds (D175). R2 is out of fold.

| Step | Result |
|------|--------|
| 1. Own decoder at the noun, mid depth, R2 | Gemma .17 (`ugly`) to .88; Qwen3.5-9B .68 to .89; Qwen3.6-27B .39 to .88 |
| 2. Cosines at the noun, `place`, mid depth | `nice`-`pretty` .63, .78, .76. `cool`-`nice` .36, .56, .41. `ugly` against `nice` and `pretty` -.09 to -.60. `ordinary` and `urban` against all: abs .33 at most |
| 2. Rank-k fit at the noun | k 1 serves `nice`, `pretty`, `ugly` (R2 .5 to .8). `ordinary` and `urban` need k 3 (R2 .68 to .88 at k 3, .00 to .18 at k 1) |
| 3. Leave 1 prompt out, k 1, noun, mid depth | `nice` .57 to .79, `pretty` .52 to .77, `ugly` .21 to .71, `cool` .16 to .46, `fancy` .10 to .32, `urban` .03 to .18, `ordinary` .01 to .08 |
| 3. Leave 1 out at `?` and the last token, best layer, mean over 14 prompts | `?`: .74, .94, .72. Last token: .95, .98, .97 (the lowest prompt: .67, .96, .93) |
| 3. Share of the first singular value over 14 unit decoders | Noun .20 to .27. Last token .27 to .46 (1 of 14 is .07) |
| 4. Words of the direction at the noun, through the lens on CPU | Gemma: botanical, floral, flower, ornamental against trucks, cargo, profanity. Qwen3.5-9B: delightful, wonderful, elegant against dumpster, garbage, industrial, unemployed. Qwen3.6-27B: exquisite, beautiful, gorgeous, serene against salvage, grim, brutality, dumpsters |
| 4. Words at the adjective and the last token | Adjective: not readable through the lens (waterfowl, candy). Last token: yes, indeed, affirmative against none, nope, neither |
| 5. abs cosine of the direction at the noun with named decoders | `nice` .62 to .71. `pretty` minus `ugly` .58 to .78. `cool` .39 to .64. `urban` .04 to .25. The `cool` sense score decoder: .01 at the adjective |
| Limit: without `nice` | Remove the `nice` direction, and fit the rank-1 direction of the other 5 prompts of the noun again: R2 below .30 in 34 of 36 cells (`scene pretty` .45, .46). At `?`, the forecast survives: .59 to .97 in Qwen3.5-9B |

The noun residual holds 1 aesthetic direction: `nice`, `pretty` and `cool` share it, `ugly` runs
against it, and its words are beauty against dirt and industry, as in F10.27. The direction is
the `nice` direction: when we remove it, the other prompts share almost nothing at the noun.
`ordinary` and `urban` are not on it; they need 2 more directions. At `?` and the last token, the
shared direction is the forecast of the answer, and its words are yes and no. Sign: `ugly` loads
with a negative slope, and the SVD sign is arbitrary. The CPU readout reproduces the stored
Gemma readout at r 1.00 and the Qwen readouts at r .70, thus the Qwen word lists are indicative.

### Task 8: boroughs over 14 prompts

Tables: `t8_manhattan.csv`, `t8_borough_order.csv`, `t8_stability.csv`, `t8_jaccard.csv`. Figure: `t8_manhattan.png`.

| Measure | Result |
|---------|--------|
| Manhattan against the rest, t on NTA means, Holm over 42 | 26 of 42 differ. `cool`, `fancy`, `urban`: 6 of 6 each, positive (t 4.2 to 10.5). `ordinary`: 5 of 6, negative (-3.6 to -7.1). `ugly`: 2 of 6 (Qwen3.5-9B, negative). `pretty`: 1 of 6 (Qwen3.5-9B `place`, p .04). `nice`: 0 of 6 |
| Borough order | 18 distinct orders over 42 cells. Manhattan first on `cool`, `fancy`, `urban` (18 of 18). Staten Island first on `nice` (6 of 6), `ordinary` (6 of 6) and `pretty` (4 of 6). The Bronx first on `ugly` (5 of 6). Manhattan last on `ordinary` (6 of 6) |
| Stability of C3 over 40 splits, at the adjective, mid depth | .59 to .78 in all 42 cells (mean over boroughs). With axis 1 removed: .26 to .73 |
| Top-50 Jaccard of C3 words across prompts | Same adjective, other noun: .09, .17, .21 (mean). Different adjective: .01 at most |
| Place probes | See task 5. All 3 models have a probe run |

The Manhattan sign of F10.24 holds for `cool`, and for `fancy` and `urban`, in every cell. It is
not a property of aesthetic words in general: `nice` and `pretty` put Staten Island first, and the
Manhattan difference passes in 1 of 12 cells. The words of C3 are stable within a cell and
specific to the adjective.

### What generalizes, in short

| Holds in all or almost all 42 cells | Holds for `cool` only, or for some adjectives |
|-------------------------------------|-----------------------------------------------|
| The embedded conception at the adjective (42) | The temperature-to-social sense axis (6 `cool` cells) |
| The step at the adjective and the gain at `?` (41) | "The sense does not follow the verdict" (6 `cool` cells; 29 of 36 other cells follow it) |
| The scene code (39) | Manhattan above the rest (`cool`, `fancy`, `urban`; `ordinary` below) |
| The Manhattan sign is the same for both nouns (21 of 21) | The borough order (18 orders) |
| The layers of each band follow the model (3 of 3) | The verdict band at r .95 (34) |
| C3 is stable within a cell (42) | The words of C3 (specific to the adjective) |

### Limits of F10.26

| Limit | Effect |
|-------|--------|
| Gemma `ugly` | 20 and 83 verdict images. Gemma `nice`: 296 and 533. These cells carry the failures of task 2 |
| Axis orientation | The axis of task 3 is turned toward the verdict. Where r is near 0 (`cool`), the end that we call "yes" is arbitrary |
| Decoders at `?` and the last token | The shared direction there is the answer direction. It is not evidence of a shared aesthetic |
| Lens on CPU | Reproduces the Gemma readout (r 1.00) and the Qwen readouts in part (r .70). Qwen word lists of task 7 are indicative |
| Patches | The stores hold no patches. The mean patch of the `cool` store serves all prompts, because attention is causal |
| Units | NTA means (197). The images inside an NTA are not independent. The 14 prompts use the same images |
| Families | Holm over 42 for the cell tests, over 21 for the noun tests, over 30 for the `urban` boroughs. F10.27 uses Bonferroni |
| Socioeconomic tests | None in this run. F10.28 holds the ACS tests |

## Corrections for the paper

We did not edit the paper. These are for the author to judge.

**Note:** The author rejected held-out NTA filters for borough lists, because the sample is
stratified over all 197 residential NTAs. The rows on the Bronx and on the Brooklyn list below
are not advice to cut. See [The sample design and the borough lists](#the-sample-design-and-the-borough-lists-2026-11-01).

| Item | Text now | Finding |
|------|----------|---------|
| Bronx lists at `place` and `cool` | Reads as a stable contrast | 0 of 25 words hold in Gemma and 0 of 23 in Qwen, on held-out NTAs. Raw rank about 6, t rank about 3,000 |
| Raw lists | Top words by raw mean | Raw lists favour high-variance words. Raw and t lists overlap only 4% to 7% |
| Gemma logits | Capped read | Softcap saturates. Use the uncapped values |
| `model` token words | Top 10 of the read | They rank 330 to 11,460 in the mean read. They rank 4 to 19 in the C2 contrast (`rank_added`) |
| `cool` words and end-token words | Effects of the image | vibe, idea, funky, amazing and inconclusive rank the same with no image. They come from the prompt or the lens |
| Stage order | Onset by token | Onset does not separate stages. Use `rank_added` or the gain |
| Urban hypothesis | Gate test | Use the probes. The gate test is not specific |
| Brooklyn `cool` list | Borough property | The list holds (16 of 16), but it sits mostly on the sense axis |

## Methods

| Step | Script | Method |
|------|--------|--------|
| F10.7 | `scripts/f10/hop_logit.py`, `analyze_logit.py` | Margin at each cell, held-out word lists |
| F10.8 | `hop_sets.py` | Word sets against matched controls, on z-scored logits |
| F10.9 | `hop_free.py`, `hop_scene.py`, `hop_patchdepth.py` | Ridge decoders, nested gain, second-order RSA |
| F10.10 | `hop_borough.py` | Held-out borough lists, audit of paper claims |
| F10.11 | `analyze_cool.py`, `build_cool_*.py`, `merge_cool_spans.py` | Traces, span extraction, log odds |
| F10.12 | `hop_link.py` | Argument words in the read of another pass |
| F10.13 | `hop_stage.py` | Partial r, `rank_added` |
| F10.14 | `probe_query.py`, `analyze_probe.py` | 53 adjective probes with logprobs |
| F10.15 | `hop_axes.py` | SVD of NTA means |
| F10.16 | `analyze_argument.py` | Object, value and stance tables, ICC |
| F10.17 | `hop_sense.py` | Sense score, 40 splits |
| F10.18 | `hop_claim_axis.py`, `probe_place.py` | Paper words on axis 1, place probes (3 models) |
| F10.20 | `phase_layers.py` | Layer band of each phase |
| F10.20b | `plot_layers_tikz.py` | The TikZ layer panel of the paper (Gemma-4-31B and Qwen3.6-27B, the patch words) |
| F10.21 | `plot_ranks.py` | Rank of each word with and with no image |
| Figures | `plot_mechanism.py` | Summary figures with JSON sidecars |

The tests are the textbook defaults: t, F, chi-square, Holm. NTA means are the unit, with
df 196. We used count methods only on generated text. The nested gain t is somewhat
liberal, because folds share training images.

## Limits

- The study has 2 models, 1 question and 1 city.
- The fixed prompt makes all variation depend on the image. No result here shows a
  causal order inside the model.
- Decoders are linear. The gain measures a linear read, not the total information.
- Span labels come from another model. We read 34 spans by hand. We found few value
  errors, and this is a small check.
- We did not run the retry pass of the extraction. The examples in the cue schema are
  street-photo examples.
- The object list in F10.16 and the adjective list in F10.14 are hand lists.
- Holm cell counts in the held-out borough test use 1 split. Only the stability values
  use 40 splits.
- `figures/f10/free/free_gain.png` shows the gain over the 4 by 4 grid. Use
  `mech_gain.png`, which shows the gain over the mean patch.
- Older heatmaps in `figures/f10/sets/` and `figures/f10/logit/` keep faint grid lines.
  The code is fixed, and a rerun removes them.
- NRC lexicon terms apply to the language set. We did not check them in detail.

## Related

- `facct-runbook.md`, sections F10.7 to F10.18
- `f10-manuscript-proposals.md`
- `facct-claim-map.md`
- `neuronpedia-viewer.md`
- `reports/J space readout analysis methods.md`
- `research_notes/J space logit methods review/review.md`
