# The modality order and the question-blind image

**Date:** 2026-08-26. **Model:** gemma-4-12B-it. **Tools:**
`research/monocle/modality_order.py`, `research/scripts/modality_order_report.py`,
`research/monocle/causal_maps.py --layout`, and `research/scripts/layout_map_compare.py`.
**Cases:** all 7 for the encoding, 3 for the causal map. 20 photographs each,
seed 20260821.

## The answer, first

The gemma-4 card asks for the image BEFORE the text. That rule has a measured
consequence, and it is absolute.

| Question | Answer |
|---|---|
| Does the order decide whether the encoding sees the question? | **Yes, and it is the only thing that does.** |
| With the image first, how blind is the encoding? | **Bit for bit.** 0.0000 at every block |
| Does the patch position matter? | No. Same positions, still exactly 0 |
| Does the order move the verdict? | Yes, by -1.9 to +5.1 logits, and the SIGN is a property of the case |
| Does the question move the encoding evenly? | Yes. r = -0.026 against the causal map |
| Does the question change WHICH cells decide? | **Yes, and by a lot.** r = 0.192 against a floor of 1.0 |
| Is that because it is the QUESTION, or is any text enough? | Both. Half is information, half is relevance |
| Did Google choose the order for objectivity? | The card's audio rule says no |

The last 4 rows carry the result. The question does reach the image when it
comes first. It reaches the whole image block evenly, so it does not steer the
encoding toward the cells that already matter. But it still changes which
cells matter: the measured causal map largely reorganises. **The ordering rule
does protect the evidence.**

A decoy arm splits that protection into 2 nearly equal halves. Any informative
text in front of the image reorganises the map. The fact that the text is the
question reorganises it by as much again. So the rule guards against a general
effect and a question-specific effect together.

The intent behind the rule is a separate question, and the card's audio rule
answers it.

## The mechanism

`blockwise_overlay` in `transformers/masking_utils.py:112` unmasks a pair of
positions only when both sit in the SAME vision island:

```python
return (q_group == kv_group) & (q_group >= 0)
```

Text positions carry `-1`, and the overlay is OR'd with the causal mask. So
one image patch reads:

- every token BEFORE it, causally, and
- every patch of its own island, in both directions.

It never reads text that follows it. Put the question after the image and the
question cannot reach the patch residual. Put it before, and it can.

## The measurement

Each photograph runs 4 times: 2 layouts by 2 questions. The statistic is

    gap = || h_own - h_other || / || h_own ||     over the patch block

**Read the gap, not a raw cosine.** Gemma carries massive activations, so a
raw cosine sits near 1.0 for any 2 residuals. `tests/test_modality_order.py`
proves the trap: add 1 huge shared channel to 2 unrelated patch sets and the
raw cosine goes above 0.99 while the centered one stays below 0.5.

### The control pair, and why it matters

Under `text_first` the question comes first, so 2 questions of different
length move the patches and their rotary phase with them. A length-matched
pair removes that by construction.

`--list-lengths` gives the prompt length of all 7 cases:

| road_quality | restaurants | subway_safety | libraries | street_photography | parks_plazas | schools |
|---|---|---|---|---|---|---|
| 69 | 71 | 72 | **73** | **73** | 75 | 76 |

**libraries and street_photography are the pair.** They agree at 73 tokens,
and they share the labels `Good` and `Poor`, so the contrast `c` stays the
same quantity. Only the subject and the question wording change. The measured
`patch_shift` is 0 under both layouts.

## The 2-arm probe, and the control that made it readable

libraries photographs, the question from street_photography, 20 photographs,
9 blocks, 180 rows:

| Block | 6 | 12 | 18 | 20 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|---|
| `image_first` | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| `text_first` | 0.0135 | 0.0142 | 0.0676 | 0.0711 | 0.1415 | 0.4467 | 0.4044 | 0.6079 | 0.5416 |

**Every one of the 180 `image_first` rows is exactly 0, and the centered
cosine is exactly 1.0.** The 2 questions give the same bits. This is stronger
than the number on record in
[causal-lens-retrospective.md](causal-lens-retrospective.md), which reported a
gap of 0.031 at block 18 and called the residue bf16 arithmetic. That reading
was right, and the matched pair now removes the residue.

Under `text_first` the question reaches the patches at once. The gap is 1.4%
at block 6, it passes 7% at the blocks the reader uses, and it passes 60% by
block 42.

### The length difference was the whole floor

The second pair carries a 1-token difference: subway_safety at 72 against
restaurants at 71.

| Block | 6 | 12 | 18 | 20 | 24 | 30 | 42 |
|---|---|---|---|---|---|---|---|
| `image_first` | 0.0110 | 0.0096 | 0.0090 | 0.0087 | 0.0244 | 0.0912 | 0.1852 |
| `text_first` | 0.0141 | 0.0146 | 0.0729 | 0.0682 | 0.1272 | 0.4093 | 0.5356 |
| ratio | 1.3x | 1.5x | **8.1x** | **7.9x** | 5.2x | 4.5x | 2.9x |

The `image_first` row here is NOT a leak. `patch_shift` is 0, so the patches
sit in the same places; only the sequence length differs, by 1 token. That
changes how the bf16 kernels tile, and the error compounds over 47 blocks:
0.009 at block 18 and 0.185 at block 42.

**Warning: a length-matched pair is mandatory for this probe.** Without it the
floor is 0.009 to 0.185 and the reader cannot tell a leak from arithmetic.

### The gradient is the scale bar

The same statistic on `g`, the backward half, for the matched pair:

| Block | 18 | 20 | 24 | 30 | 42 |
|---|---|---|---|---|---|
| `image_first` | 1.43 | 1.44 | 1.33 | 0.66 | 0.61 |
| `text_first` | 3.43 | 2.09 | 1.95 | 1.11 | 1.08 |

Under `image_first` the forward half moves by exactly 0 and the backward half
moves by 143%. That is the 2-halves result of
[causal-lens-retrospective.md](causal-lens-retrospective.md), now shown in one
table. **The question does select image evidence. It selects it at read-out,
not at encode.**

## The 4-arm battery, 2026-08-26

The 2-arm probe could not say whether the verdict shift came from the question
or from the patches sitting nearer the answer. The 4-arm set separates them.

| Arm | Block order | Question is before the image | First patch | Answer distance |
|---|---|---|---|---|
| `image_first` | image, question | no | 5 | 69 |
| `neutral_first` | filler, image, question | no | 57 | 69 |
| `question_first` | question, image, filler | **yes** | 57 | 69 |
| `text_first` | question, image | **yes** | 57 | 17 |

`neutral_first` and `question_first` are mirror images. The filler matches the
question in TOKENS, so the 2 arms share the patch positions, the distance to
the answer, and the total length. Their difference is the question alone.

The chain has no gap: `image_first` to `neutral_first` to `question_first` to
`text_first` moves 1 thing at a time and ends at the whole reversal.

7 cases, 20 photographs each, seed 20260821.

### 1. The order alone decides whether the encoding is blind

The gap in `h`, over the 2 length-matched cases (libraries and
street_photography, both 52 question tokens, `patch_shift` 0):

| Block | `image_first` | `neutral_first` | `question_first` | `text_first` |
|---|---|---|---|---|
| 6 | **0.0000** | **0.0000** | 0.0134 | 0.0134 |
| 18 | **0.0000** | **0.0000** | 0.0712 | 0.0713 |
| 24 | **0.0000** | **0.0000** | 0.1417 | 0.1424 |
| 30 | **0.0000** | **0.0000** | 0.4520 | 0.4522 |
| 42 | **0.0000** | **0.0000** | 0.6070 | 0.6076 |

**This is a clean 2 by 2, and it has 1 factor.** `neutral_first` puts the
patches at position 57, exactly where `question_first` puts them, and its gap
is still exactly 0. `text_first` moves the answer 52 tokens nearer and its gap
matches `question_first` to 3 decimal places.

So the position of the patches does not matter, and the distance to the answer
does not matter. Only 1 thing matters: does the question come before the image.

### 2. The verdict: 2 effects, and they pull against each other

Paired by photograph, pooled over the 7 cases, 140 photographs:

| Contrast | Isolates | Shift | p |
|---|---|---|---|
| `neutral_first` - `image_first` | the patches move 52 tokens later | +0.077 | 0.620 |
| `question_first` - `neutral_first` | **the question, at fixed positions** | **+0.842** | **0.006** |
| `text_first` - `question_first` | the answer moves nearer the patches | **-0.477** | **0.000** |
| `text_first` - `image_first` | the whole reversal | +0.442 | 0.204 |

**A position change alone does nothing.** Moving the patches 52 tokens later
shifts the verdict by 0.08 logits, at p = 0.62.

**The whole reversal LOOKS null, and it is not.** `text_first` against
`image_first` gives p = 0.20. That is 2 real effects of opposite sign which
partly cancel: the question adds +0.84 and the recency takes -0.48. A 2-arm
probe would have reported no effect and been wrong.

### 3. The direction is a property of the case, not of the model

The decisive contrast, by case, with a BH step over the 7:

| Case | Shift | q | Survives |
|---|---|---|---|
| subway_safety | **+5.08** | 0.0000 | yes |
| parks_plazas | +2.78 | 0.0001 | yes |
| libraries | +2.78 | 0.0001 | yes |
| restaurants | +0.59 | 0.318 | no |
| road_quality | -1.63 | 0.0024 | yes |
| schools | -1.83 | 0.0021 | yes |
| street_photography | -1.88 | 0.0310 | yes |

6 of the 7 survive, and they run in BOTH directions. The pooled +0.84 is thus
a weak mean over large case effects that cancel.

**Warning: do not quote the pooled number as the effect size.** The earlier
2-case run read subway_safety at +6.11 and libraries at +1.68, and it looked
as if the order pushed every judgment toward the positive label. It does not.
subway_safety moves 5 logits toward `Safe`, and street_photography moves 1.9
toward `Poor`.

The label pair does not explain the sign. subway_safety alone uses
`Safe`/`Dangerous`, and the other 6 all use `Good`/`Poor`, yet those 6 split
3 positive and 3 negative.

### 4. The question does NOT land on the evidence

The per-patch gap under `question_first` says how far each patch moves when
the question changes. The measured causal map says what each patch is worth to
the judgment. Block 20, against `causal_maps60`, 20 photographs for each case:

| Case | r | rho |
|---|---|---|
| libraries | -0.042 | -0.032 |
| restaurants | -0.034 | -0.030 |
| subway_safety | -0.001 | +0.003 |

Pooled r = **-0.026**, p = 0.007, n = 60.

The correlation is distinguishable from 0 and it is practically 0, and its
sign points the wrong way for a steering account. **The question changes the
encoding almost evenly over the frame, and if anything it moves the causally
important cells LESS than the rest.**

This is the number that decides what the ordering rule buys. A rule that kept
perception neutral would have to stop the question from reaching the evidence.
The question does not reach the evidence in the first place: it reaches the
whole image block, diffusely, and the judgment still comes from the gradient.

**Warning:** this number reads the SIZE of the per-patch move, not its
direction. An even size does not make an even consequence. Section 5 measures
the consequence, and the consequence is large. Read the 2 sections together.

### 5. The evidence itself moves

Sections 1 to 4 read the encoding. This section reads the judgment. We
measured the causal map again under each layout, and compared the maps.

**The floor is exact.** `research/monocle/truth_retest.py` shows that at a fixed batch
the map repeats to the last bit: r = 1.000000, with a shuffled cell order in a
new process. Only the batch size moves it. Every run here uses `--batch 12`,
so the floor is 1.0 and any drop is a real change.

**The mirror pair holds the geometry fixed.** `neutral_first` and
`question_first` give the same sequence length, the same first patch, and the
same answer distance. Only the question placement moves.

| Case | r | rho | Top cell repeats |
|---|---|---|---|
| subway_safety | 0.120 | 0.103 | 2 of 20 |
| libraries | 0.268 | 0.211 | 7 of 20 |
| street_photography | 0.187 | 0.171 | 2 of 20 |

Pooled r = **0.192**, against a floor of 1.000000. Each case falls below the
floor at p < 1e-10. The top cell repeats 11 times in 60. Chance is 0.9 in 60,
so a trace of the production map stays, but the map is mostly new.

**The map is not weaker, it points somewhere else.** A weak map correlates
badly for a trivial reason, so we checked the strength.

| Case | Arm | Mean image effect | Mean strongest cell |
|---|---|---|---|
| subway_safety | `neutral_first` | 6.16 | 1.83 |
| subway_safety | `question_first` | 6.43 | 2.61 |
| libraries | `neutral_first` | 4.68 | 1.75 |
| libraries | `question_first` | 3.48 | 2.06 |
| street_photography | `neutral_first` | 5.88 | 1.76 |
| street_photography | `question_first` | 8.22 | 4.33 |

The image keeps the same total effect or a larger one, and the strongest cell
gets stronger in all 3 cases. The low correlation is not a weak signal.

#### The control: is it the question, or is it any text?

A repeated filler holds little information. So a map that changes under
`question_first` could change because the leading block carries INFORMATION,
and not because it carries the QUESTION. A third arm separates the 2.

| Arm | Blocks | The leading text | Distinct words |
|---|---|---|---|
| `neutral_first` | filler, image, question | 1 sentence, repeated | 4 |
| `decoy_first` | decoy, image, question | prose about the moons of Jupiter | 37 |
| `question_first` | question, image, filler | the question | — |

The decoy matches the question in tokens for every case, 51 or 52. So the 3
arms share the sequence length, the patch positions, and the answer distance.
The decoy holds no word of the rating scale, and a test guards that. A word
such as `safe` or `clean` would make the decoy a weak question, not a control.

**The result is a ladder, and both steps are real.** Each arm scores against
production, and each adds a block and moves the patches 52 tokens later. So a
paired difference between 2 arms leaves the content of the leading block.

| Which block leads the image | r against production |
|---|---|
| a filler, repeated | +0.446 |
| a decoy, informative but irrelevant | +0.324 |
| the question | +0.192 |

| Step | Difference | t p | wilcoxon p | Lower in |
|---|---|---|---|---|
| filler to decoy | -0.122 | 3.1e-09 | 1.4e-07 | 49 of 60 |
| decoy to question | **-0.132** | 2.0e-05 | 6.6e-05 | 44 of 60 |
| filler to question | -0.254 | 1.4e-12 | 1.5e-09 | 53 of 60 |

Read the paired differences, not the 3 absolute values. Each absolute value
holds the same length change, and the difference removes it.

**The effect splits into 2 nearly equal halves.** Any informative text in
front of the image costs -0.122 of the map. The fact that the text is the
QUESTION costs a further -0.132. So about half the change belongs to
information, and about half belongs to relevance.

The same split appears at fixed geometry, with no length change at all:

| Pair | r |
|---|---|
| `decoy_first` against `neutral_first` | 0.383 |
| `question_first` against `neutral_first` | 0.192 |
| `question_first` against `decoy_first` | 0.198 |

The question arm sits as far from the decoy arm as from the filler arm. It
does not sit between them.

**The decoy map is not weak either.** The mean image effect rises above the
filler arm in all 3 cases: 7.14 against 4.68 for libraries, 7.10 against 5.88
for street_photography, and 6.61 against 6.15 for subway_safety.

#### How section 4 and section 5 fit together

The 2 sections look opposed. They are not.

| Section | What it measures | Result |
|---|---|---|
| 4 | the SIZE of the per-patch move | even over the frame, r = -0.026 |
| 5 | the CONSEQUENCE for the judgment | the map largely changes, r = 0.192 |

A per-patch size says nothing about a direction. Every patch can move by the
same amount and still change `g*h` by a lot, because the gradient reads the
direction. So the question spreads its change evenly over the frame, and the
change still lands on the read-out.

**This corrects the reading in section 4.** The question does not steer the
encoding toward the cells that already matter. But it does change which cells
matter. The ordering rule protects more than section 4 alone suggests.

## The audio rule breaks the intent reading

Section 4 of the model card gives 2 rules, not 1:

> Image content **before** the text in your prompt.
> Audio content **after** the text in your prompt.

Section 6 then prints the audio prompts, and they are task instructions placed
ahead of the audio: "Transcribe the following speech segment in {LANGUAGE}",
and "Transcribe the following speech segment in {SOURCE_LANGUAGE}, then
translate it into {TARGET_LANGUAGE}".

So Google puts the instruction where it DOES condition the encoding for audio,
and where it CANNOT for images. One principle about unbiased perception cannot
give both rules. A task account gives both: audio is ambiguous between
transcribe and translate, so the instruction must reach it, and an image
encoding is generic and gets reused.

Two more facts point the same way.

1. **The architecture enforces nothing.** Text before an image conditions the
   patches, and the template writes a system turn ABOVE the user turn when
   thinking is on, when tools are present, or when a system message exists
   (`chat_template.jinja:179`). A system prompt reaches the patches today.
2. **The card calls the order a tip, not a contract.** "Interleaved
   Multimodal Input — freely mix text and images in any order."

There is also a plain engineering account that the intent reading must beat:
images first make the image KV cache a reusable prefix across many questions
about one photograph.

## What this changes for our instruments

| Claim | Status |
|---|---|
| a reader built on `h` alone is question-blind | **exact**, not approximate |
| a wrong-question control on an `h`-only reader is empty | confirmed, by construction |
| the fitted J-lens cannot gain from the question | confirmed. It reads `h` |
| `g` carries the whole question | confirmed. 143% where `h` gives 0% |
| a map measured under 1 layout transfers to another | **no.** r = 0.19 across the mirror pair |
| a leading text must be the question to move the map | no. A decoy moves it about half as far |
| a cross-layout map comparison needs a noise band | no. The floor is exact at a fixed batch |

## Scope

`text_first` leaves the canonical path. 3 gates demand `images_then_labels`,
so no `text_first` run may enter the registry. Every number here is a probe.

Both cases run 20 photographs at seed 20260821. The verdict shift needs a
position control.

The causal maps use 3 cases, 20 photographs each, group 2 (64 cells), and the
absent reference. The floor holds the batch fixed, not the length. So it is a
true floor for a matched pair only. Read the `image_first` rows as an upper
bound, because they add the length difference to whatever the order does.

The decoy is 1 passage, about 1 subject. It shows that an informative text
alone moves the map, and it measures how much. It does not show that every
informative text moves the map by the same amount. A subject nearer to a
street scene could move it more.

## Commands

```bash
pytest tests/test_modality_order.py -v

sbatch --time=00:20:00 research/monocle/monocle.sub monocle.modality_order \
    --list-lengths

sbatch --time=01:00:00 research/monocle/monocle.sub monocle.modality_order \
    --case libraries --other-case street_photography --n-images 20
```

Output: `outputs/_monocle/modality_order/<case>/gaps.parquet` and
`summary.json`.

The causal map under each layout, and the comparison:

```bash
pytest tests/test_layout_map_compare.py -v

for case in subway_safety libraries street_photography; do
  for layout in image_first neutral_first question_first; do
    sbatch --time=02:00:00 --job-name=cmap-$case-$layout \
        research/monocle/monocle.sub monocle.causal_maps \
        --cases "$case" --n-images 20 --group 2 --batch 12 \
        --reference absent --seed 20260821 --layout "$layout" \
        --out-dir outputs/_monocle/causal_maps_layout
  done
done

# The decoy arm. It keeps the neutral_first layout and changes only the
# content of the leading block, so the geometry does not move.
for case in subway_safety libraries street_photography; do
  sbatch --time=02:00:00 --job-name=cmap-$case-decoy \
      research/monocle/monocle.sub monocle.causal_maps \
      --cases "$case" --n-images 20 --group 2 --batch 12 \
      --reference absent --seed 20260821 \
      --layout neutral_first --filler-kind decoy \
      --out-dir outputs/_monocle/causal_maps_layout
done

python research/scripts/layout_map_compare.py \
    --out outputs/_monocle/layout_map_compare.json
```

**Warning:** keep `--batch 12` in every arm. The batch size moves the map, and
a mixed batch destroys the exact floor.

## See also

- [gemma4-12b-architecture.md](gemma4-12b-architecture.md) — the mask, the 2 block kinds, the encoder-free vision path.
- [causal-lens-retrospective.md](causal-lens-retrospective.md) — the 2 halves, and the reader that uses both.
- [gemma-4-best-practices.md](../../docs/gemma-4-best-practices.md) — why production uses `images_then_labels` and not the card's simple form.
