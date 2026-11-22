# Shared negative evidence across 7 judgments

Started 2026-08-25. This page holds the line of work on a content question.
The question is what the model sees at the image token that lowers a
judgment, and if it is the same content for every question.

The logit lens cannot answer this. See
[causal-lens-retrospective.md](causal-lens-retrospective.md): the 2 extremes
of the token-ablation map carry the same logit-lens words at every layer. The
crop-and-caption method does answer it, because the model itself supplies the
words.

## The method

`research/monocle/patch_caption.py` cuts a square out of the photograph and asks the
model to name its content. The model describes the crop in a new context. It
does not know the judgment, the causal role, or the case.

| Condition | The image token whose patch is cut out |
|---|---|
| `pos` | the image token that the map ranks highest |
| `neg` | the image token that the map ranks lowest |
| `random` | other image tokens of the same photograph |

**Warning:** "the map" is 2 different things, and the difference decides the
result. Read the table below before you read any number on this page.

| Map store | Content |
|---|---|
| `causal_maps60/{case}_absent` | the token-ablation map: 256 measured ablations |
| `reader_cells/{case}_shared` | the predicted map of the fitted ablation predictor |

The fitted ablation predictor is a ridge regression from gradient-times-input
(`g * h`) at layer 20 to the token-ablation effect
(`research/scripts/k1_export_reader_cells.py`, which calls `fp.predict`). The
`shared` mode fits on the other cases, so the fit did not include this case.

The 2 stores agree at a median r of 0.164, and they name the same most
negative image token for 8 per cent of the photographs. They are not the same
measurement.

## Finding 1: the predictor's negative extreme elicits disorder words

Share of crops whose caption holds a word from `LEXICON_V1`
(`research/scripts/disorder_rubric.py`). 60 photographs for each case.

**These crops come from the predicted map of the fitted ablation predictor**,
not from the token-ablation map. We checked it. The `neg` token of this
experiment is the argmin of the predictor's map for 100 per cent of the
photographs. It is the argmin of the token-ablation map for 3 to 25 per cent.
Finding 3 shows why that matters.

| Case | negative extreme | positive extreme | random |
|---|---|---|---|
| restaurants | **0.200** | 0.033 | 0.037 |
| parks and plazas | **0.183** | 0.033 | 0.017 |
| road quality | **0.150** | 0.050 | 0.013 |
| libraries | **0.117** | 0.050 | 0.050 |
| subway safety | **0.117** | 0.067 | 0.017 |
| street photography | 0.050 | 0.033 | 0.029 |
| schools | 0.033 | 0.000 | 0.033 |

The rate at the negative extreme is 3 to 10 times the rate at the positive
extreme and at a random image token. It holds for questions that say nothing
about safety.

**The effect is 1-sided.** The positive extreme scores +0.010 against a
random image token, which is no difference. Only the negative extreme elicits
the words.

### The position control

A negative image token sits lower in the image than a positive one: mean row
8.65 against 6.65, where row 15 is the ground. So the words could come from
the height in the image and not from the ablation effect.

They do not. Inside a row band:

| Row band | neg | random | difference |
|---|---|---|---|
| top (0-5) | 0.000 | 0.002 | -0.002 |
| mid (6-10) | 0.152 | 0.045 | **+0.107** |
| low (11-15) | 0.145 | 0.043 | **+0.102** |

A permutation of the condition label inside each photograph and each row band
holds the scene, the camera, the light and the height fixed. The observed gap
is +0.094 against a null-distribution median of +0.039, **p = 0.00005** over
20000 draws. The null distribution sits above zero because the top band holds
random crops and almost no extremes. The test accounts for that, and the
excess above it is the result.

**Warning:** these numbers are exploratory. We wrote the word list while we
read the same captions. `research/scripts/disorder_rubric.py` freezes a blind scoring
rule for the confirmatory sample at seed 20260827.

## Finding 2: the negative tails coincide, the most negative image token does not

`causal_maps --question-case` measures the 256-token map of one photograph
under each of the 7 questions. The hidden state at the image tokens does not
depend on the question, so only the gradient can make the maps differ.
`research/scripts/xq_pole_coincidence.py` scores it.

30 photographs from the `street_photography` sample, all 7 questions.

### The maps differ, and the differences are plausible

| Question pair | whole-map r |
|---|---|
| libraries and schools | **+0.389** |
| libraries and restaurants | +0.296 |
| parks and schools | +0.281 |
| restaurants and road quality | **+0.056** |

The 2 upkeep questions agree most. Roadway condition and restaurant appeal
agree least.

### The single most negative image token: a null result

The statistic is the rank that question B gives to the image token that
question A ranks most negative. 0 means B agrees, 0.5 is chance.

| Measure | Value |
|---|---|
| observed | 0.3905 |
| a Gaussian null model at the same r | 0.3722 |
| **a copula null model at the same r** | **0.3757** |
| a different photograph | 0.5116 |

The copula null model keeps each map's own values and ties them only by rank
correlation. A token-ablation map has a heavy tail, and 2 correlated heavy
tails agree about the most negative token more often than 2 normal tails. So
the copula null model is the appropriate one.

Observed minus copula is **+0.0148**, with a 95% interval of
[-0.0135, +0.0423] over photographs. That interval includes zero.

**So the 16-fold enrichment of the same most negative token (0.0619 against a
chance of 0.0039) is what the whole-map correlation already predicts.** It is
not evidence of a shared feature.

### The whole tail: a result

One image token of 256 is a weak statistic. The tail is not.

| Tail size k | shared | copula null model | excess | 95% interval |
|---|---|---|---|---|
| 4 | 0.0718 | 0.0466 | **+0.0252** | +0.0087 to +0.0421 |
| 10 | 0.1073 | 0.0852 | **+0.0221** | +0.0074 to +0.0394 |
| 25 | 0.1721 | 0.1602 | +0.0119 | -0.0002 to +0.0245 |

The excess concentrates at the extreme tail: 54 per cent above the null-model
value at k=4, 26 per cent at k=10, and 7 per cent at k=25. The k=4 and the
k=10 rows exclude zero. The k=25 row does not, so the agreement is gone by 25
tokens.

### The weak question does not produce the result

`subway_safety` is the worst fit of the 7, and 16 of its 30 images are weak.
So we ran the same test again without it.

| Tail size k | 7 questions | 6 questions, no subway |
|---|---|---|
| 4 | +0.0252 [+0.0087, +0.0421] | +0.0249 [+0.0071, +0.0443] |
| 10 | +0.0221 [+0.0074, +0.0394] | +0.0263 [+0.0114, +0.0443] |
| 25 | +0.0119 [-0.0002, +0.0245] | +0.0147 [+0.0030, +0.0274] |

The 2 columns agree. The weak question dilutes the k=25 row a little and
changes nothing else. The single-token test stays null in both
(+0.0148 and +0.0113, and both intervals include zero).

**Different questions do not agree on the most negative image token. Their
extreme negative tails overlap more than their whole-map correlation
explains.** The shared negative evidence sits in a small region, not in 1
token, and it is gone by 25 tokens of 256.

### The weak-image count is data

A question asked of the wrong photographs moves the answer less. The count of
images whose whole-image ablation moves the logit difference (`c`) by less
than the threshold:

| Question, asked of street photographs | weak images |
|---|---|
| libraries | 10 of 30 |
| street photography | 8 of 30 |
| schools | 5 of 30 |
| parks and plazas | 5 of 30 |
| road quality | 3 of 30 |
| restaurants | 2 of 30 |
| subway safety | 16 of 30 |

Subway safety is the worst fit, which is the expected result: the
photographs are city blocks and not station entrances.

## Finding 3: the presentation does not matter

A crop of 1 image token is 56 by 56 pixels. The 1st experiment made it 5.3
times larger, and an enlarged image can produce spurious detail. So we ran 4
presentations over 4 cases and 60 photographs for each case. Every condition
cuts its crops from the same token-ablation map, so only the presentation
changes.

| Condition | The crop | neg | random | gap |
|---|---|---|---|---|
| `x5.3` | 1 token, 5.3 times larger | 0.079 | 0.033 | +0.046 |
| `up448` | 1 token, 2.7 times larger | 0.067 | 0.025 | +0.042 |
| `native` | 1 token at its own size | 0.042 | 0.013 | +0.029 |
| `pad2` | 2 tokens, 5.3 times larger | 0.071 | 0.052 | +0.019 |

The 4 gaps sit between +0.02 and +0.05. **The upscale does not produce the
words.** No condition alone reaches p < 0.05, because a gap of about +0.04
needs more than 1440 crops to reach significance. Finding 4 provides that
statistical power.

## Finding 4: the token-ablation extreme holds a small effect, the predictor's extreme a large one

### The token-ablation extreme, on held-out photographs

The `fresh` experiment is the pre-registered test: 7 cases, 60 held-out
photographs for each case, seed 20260827, the token-ablation map,
`LEXICON_V1`. The held-out photographs share 1 image of 420 with the 1st
experiment.

| Case | neg | pos | random | gap |
|---|---|---|---|---|
| parks and plazas | 0.150 | 0.083 | 0.054 | +0.096 |
| road quality | 0.100 | 0.067 | 0.029 | +0.071 |
| libraries | 0.067 | 0.017 | 0.017 | +0.050 |
| subway safety | 0.067 | 0.083 | 0.017 | +0.050 |
| schools | 0.050 | 0.033 | 0.021 | +0.029 |
| restaurants | 0.050 | 0.050 | 0.037 | +0.013 |
| street photography | 0.000 | 0.033 | 0.021 | -0.021 |
| **ALL** | **0.069** | 0.052 | 0.028 | **+0.041** |

The permutation holds the photograph and the row band, over 20000 draws:
observed +0.0411, null median +0.0232, null p95 +0.0351, **p = 0.018**.

**So the token-ablation negative extreme does elicit disorder words. The
effect is real, and it is small.** 2520 crops show it; 1440 do not.

### The predictor's extreme, on the same photographs

Now hold everything fixed except the map. Same 4 cases, same 60 photographs,
same 5.3 times presentation, same word list.

| The map that selects the token | neg | random | gap |
|---|---|---|---|
| the token-ablation map | 0.079 | 0.033 | +0.046 |
| the predictor's map | 0.163 | 0.029 | **+0.133** |

A permutation of the map label inside each (case, photograph, condition)
holds the scene, the camera and the height fixed. The difference is
**+0.0875** against a null mean of -0.0005 and a null sd of 0.0252, so
**p = 0.0004**.

**The image token that the predictor ranks most negative elicits disorder
words about 3 times more than the token whose measured ablation effect is
most negative.**

### Interpretation

Both negative extremes elicit the words, so the token-ablation map holds
content. But the ridge regression on gradient-times-input (`g * h`) at layer
20 finds it far better than the argmin of 1 ablation does.

That fits Finding 2. A single measured token is a weak statistic. Its argmin
agrees with another question no more than the whole-map correlation
predicts, and only the tail of 4 to 10 tokens carries a signal. A predictor
that pools over the whole map is more stable than the argmin of 1 ablation,
so it selects the disorder region more often.

**Warning:** we cannot yet say the predictor is more correct about the model.
It may find the semantic region better, or it may track image content that
the measured extreme does not hold. Finding 5 has to separate those.

### 2 earlier interpretations of this page were wrong

| The interpretation | Why it failed |
|---|---|
| "the upscale produces the words" | the matched `x53` condition gives a null result like the other conditions |
| "the token-ablation map shows nothing" | it does, at p = 0.018, once 2520 crops give the power |

The 1st comparison confounded the map with the presentation, because the 1st
experiment and the control conditions came from different map stores. The
2nd took a null result from a condition that held too few crops. Use the
numbers above.

## Finding 5: the ablation effect does not order the words

Findings 1 and 4 contrast 2 extremes against a random image token. That test
cannot detect a graded relation. So the `dense` experiment cuts 76 crops for
each photograph, over the whole range of the ablation effect: 7 cases, 360
photographs, 27507 crops.

For each photograph we rank its crops by the measured ablation effect, where
rank 0 is the most negative token. Then we take the mean rank of the crops
whose caption holds a `LEXICON_V1` word. A value of 0.5 means no graded
relation.

| Case | photographs | hits | mean rank |
|---|---|---|---|
| parks and plazas | 31 | 95 | 0.407 |
| libraries | 44 | 110 | 0.469 |
| restaurants | 49 | 176 | 0.491 |
| subway safety | 44 | 153 | 0.529 |
| road quality | 37 | 116 | 0.540 |
| schools | 34 | 94 | 0.557 |
| street photography | 36 | 107 | 0.573 |
| **ALL** | **275** | **851** | **0.510** |

A permutation of the caption hits inside each photograph, over 10000 draws,
gives a null mean of 0.4999 and a sd of 0.0097. The observed 0.5099 sits
above it, so **p = 0.85**. The cases fall on both sides of 0.5.

**The disorder words do not follow the ablation effect.** A crop from the
middle of the range is as likely to elicit them as a crop from the bottom.

### Read Finding 4 and Finding 5 together

| Test | Result |
|---|---|
| the most negative token against a random token | +0.041, p = 0.018 |
| the ablation effect as a gradient over 76 tokens | 0.510, p = 0.85 |

The effect sits at the extreme and not along the range. That is the same
shape as Finding 2, where the tail of 4 to 10 tokens carries a signal and the
whole-map agreement does not. A token-ablation map is heavy-tailed, and only
its end holds the content.

**Warning:** do not fit a linear regression from the ablation effect to a
caption score. The data does not support one.

## Finding 6: the predictor's extreme replicates on photographs it never saw

Finding 4 left an open question. Its predictor condition used the same
photographs that the 1st experiment used, so the advantage could belong to
that sample. This experiment resolves it.

### The run

| Item | What it is |
|---|---|
| the photographs | seed 20260827, 60 for each case, 7 cases |
| the regression inputs | `acts60_fresh`, layer 20, `g * h` |
| the ridge weights | `shared` mode: the fit uses the other 6 cases only |
| the crops | 1 token, pad 1, 896 px, as the token-ablation experiment cut them |
| the count | 2520 crops, 6 for each photograph |

The fit did not include this case, and it did not include these
photographs. The export tool rebuilt the original maps first, and they agree
with the stored ones at a median rho of 1.0000 for all 7 cases. The alpha
parameter and the code path thus match the 1st experiment.

### The result

| Case | predictor neg | predictor gap | ablation neg | ablation gap |
|---|---|---|---|---|
| parks and plazas | 0.233 | **+0.192** | 0.150 | +0.096 |
| restaurants | 0.200 | **+0.163** | 0.050 | +0.013 |
| subway safety | 0.133 | +0.113 | 0.067 | +0.050 |
| libraries | 0.117 | +0.096 | 0.067 | +0.050 |
| road quality | 0.117 | +0.083 | 0.100 | +0.071 |
| schools | 0.083 | +0.050 | 0.050 | +0.029 |
| street photography | 0.017 | -0.008 | 0.000 | -0.021 |
| **ALL** | **0.129** | **+0.098** | 0.069 | +0.041 |

The permutation holds the case, the photograph and the row band, over 20000
draws:

| Map | gap | null median | null p95 | p |
|---|---|---|---|---|
| the predictor's map | +0.0982 | +0.0536 | +0.0685 | **0.00005** |
| the token-ablation map | +0.0411 | +0.0262 | +0.0381 | 0.028 |

The paired test permutes only the map label inside each (case, photograph,
condition). The difference is **+0.0571**, the null mean is +0.0002 and the
null sd is 0.0166, thus **p = 0.0007**.

**The predictor's advantage replicates out of sample.** It is smaller than
the +0.0875 of Finding 4, and it points the same way.

### Where the predictor's token ranks in the token-ablation map

| Case | same neg token | predictor neg rank | median rho |
|---|---|---|---|
| parks and plazas | 0.183 | 0.289 | 0.090 |
| road quality | 0.133 | 0.158 | 0.086 |
| libraries | 0.117 | 0.286 | 0.102 |
| schools | 0.117 | 0.279 | 0.104 |
| restaurants | 0.083 | 0.348 | 0.111 |
| street photography | 0.067 | 0.394 | 0.069 |
| subway safety | 0.050 | 0.344 | 0.092 |
| **ALL** | **0.107** | **0.300** | **0.094** |

The rank runs from 0, the measured minimum, to 1, the measured maximum.
Chance is 0.500, and the same token by chance is 0.004. The predictor thus
points low in the token-ablation map, and it does not point at the minimum.

### The predictor is not a denoised copy of the token-ablation map

`outputs/_monocle/truth_retest_report.json` measures how well the
token-ablation map reproduces itself. Under the same batch it is exact. Under
another batch size it reproduces at a Spearman of 0.53 to 0.75, and it names
the same most negative token 8 to 58 percent of the time.

| Pair | Spearman | same most negative token |
|---|---|---|
| the token-ablation map against itself, another batch | +0.53 to +0.75 | 0.08 to 0.58 |
| the predictor against the token-ablation map | +0.09 | 0.11 |

The predictor differs from the token-ablation map far more than the map
differs from itself. A denoised copy would sit near the test-retest level and
not below it. So the ridge regression does not denoise the token-ablation
map. It measures something else, and that something predicts the disorder
words better.

**Warning:** this does not say the predictor is more correct about the model.
The words are a property of the crop. There is no second causal measurement
of the predictor's token.

### A correction to the ablation condition's p value

Finding 4 records p = 0.018 for the ablation condition. That permutation held
the photograph and the row band, but not the case. 60 photographs of 360
carry 2 questions, so 1 stratum could hold crops of 2 different maps. With
the case in the stratum the same test gives **p = 0.028**. The interpretation
does not change: the token-ablation extreme elicits the words, and the effect
is small.

### Steps that ran

| Step | Tool |
|---|---|
| the regression inputs of the held-out photographs | `monocle.input_gate --save-acts 20 --truth-from ...` |
| the predictor's map | `research/scripts/k1_export_reader_cells.py --pred-root` |
| the crops and the captions | `monocle.patch_caption --maps reader_cells_fresh/...` |
| the report | `research/scripts/reader_fresh_report.py` |

## Open questions

Findings 5 and 6 close 3 rows of this table. The remaining rows:

| Question | The experiment |
|---|---|
| is the predictor's token causal by another measurement? | ablate the predictor's token and read the logit difference, then compare it with the measured minimum |
| what does the ridge regression measure, if not the ablation effect? | the predictor's weights against a content measurement of the same crops |
| the rare-event class (a firearm) | 1 event in 4700 captions, and it sits on a crop with no measurable ablation effect. See the protocol in `research/scripts/disorder_rubric.py` |

## 2 defects this line of work found

`causal_maps.py` ran its weak-image criterion before it wrote the parquet. An
experiment that failed the criterion thus discarded every map that it
measured. The criterion now runs after the write, and `--allow-weak` reports
it instead of a stop. A cross-question run needs that flag, because a
question asked of the wrong photographs is weak by design.

### The alias that returned ranks

`transfer_k1.load_maps` takes `--pick causal`. A token-ablation map names
that column `values`, so the loader renames `pick`. A later test then asked
if `pick` was still `causal`, saw that it was not, and routed the map to the
layer code path. That path returns ranks of 1 to 256, not the measured
values.

The captions are correct, because a rank keeps the order and the extremes
are the argmax and the argmin. But each caption row records `value`, and a
job in the window wrote a rank there.

| Item | State |
|---|---|
| the defect | `research/monocle/transfer_k1.py`, 2026-08-25 20:45 to 22:00 |
| the fix | keep the map type in `whole_map` before the rename |
| the repair | `research/scripts/repair_caption_values.py` |
| stores repaired | 12 of 25; the other 13 predate the defect |
| the check | the rank order agrees at r = 1.0000 for all 12 |

Run the repair again after a job that started inside the window.

### The repair's own defect: a guess in place of a test

The 1st repair asked if the numbers looked like a rank: whole numbers in the
range 1 to 256. It then called a store clean unless every photograph looked
that way. A dense store holds 65 crops of 256 tokens, so 1 photograph that
did not fit passed the whole store. The tool reported 0 stores with ranks
while 14 held them.

The tool now reads the map and compares. A store is clean when its `value`
column already equals the measured value, and it is repaired only when the
rank order agrees at r = 1.0000. That guard names the source map as well:

| Store | Its map | Result |
|---|---|---|
| `patch_caption_reader/*_causal` | `reader_cells/{case}_shared` | ok |
| `patch_caption_pres/*` | `causal_maps60/{case}_absent` | ok |
| `patch_caption_fresh/*` | `causal_maps60_fresh/{case}_absent` | ok |
| `patch_caption_dense/*` | `causal_maps60/{case}_absent` | repaired |
| `patch_caption/*_gxa` | not found | skipped, not touched |

**Warning:** do not test a data defect by the shape of the numbers when the
measured value is on disk. Read the measured value and compare.
