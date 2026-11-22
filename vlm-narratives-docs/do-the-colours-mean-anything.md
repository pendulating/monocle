# Do the colours mean anything?

A causal map paints a square coral or teal, and a reader reads "the model
finds this square dangerous". This page holds the tests of that claim, run on
the night of 2026-08-23. The running notes are in
`outputs/_monocle/overnight/RESEARCH_LOG.md`.

Read [[patch-attribution]] first for how a map is measured, and
[[where-the-judgment-lives]] for the relay the judgment travels along.

## The short answer

| Claim | Verdict |
|---|---|
| the ORDER of the cells is real | yes, under 2 interventions |
| it is about CONTENT, not about holes | yes |
| the content carries into another photograph | yes, 1-sided p = 0.014 |
| the map answers THIS question, not saliency | yes, control r = 0.032 |
| a single cell is reproducible | **no**, r = 0.33 between methods |
| the cells add up to the judgment | **no**, at any grid size |
| the map is explained by brightness or edges | no, and that is a null with a control |
| the model can say what is in the cells | yes at group 2, p = 0.010. A null at group 1 |
| it works on questions other than safety | yes, 58 photographs of 58, 4 questions |

So a map may be read as an ORDER over squares at its 2 ends. It may not be
read as a budget of logits, and 2 neighbouring cells may not be compared.

## How to read a number on this page

Every test here makes a DIRECTED prediction, so the conventions below hold
everywhere. A paper that quotes this page must state them.

| Convention | Rule |
|---|---|
| p-values | **1-sided**, unless the text names the test 2-sided |
| a "by target" p | a t-test on the PER-TARGET means, not a sign test |
| the fraction beside it | how many targets ran positive. It is not the test |
| a MEDIAN over photographs | sections 5 and 7, the word cells of 9, and the group sweep |
| a MEAN over photographs | the separations of sections 4 and 8 |
| the independent unit | the photograph, never the pair or the cell |

**Warning: a fraction and a p in the same row are 2 different statements.**
`14/17` counts the targets. The p beside it comes from a t-test on those 17
means. A reader who takes the p for a sign test reads the wrong test.

## The tests

### 1. Direction: the deletion test

Rank the cells, then take several away AT ONCE and watch `c`.

| Arm | Prediction |
|---|---|
| the k cells the map calls Safe | `c` FALLS |
| the k cells it calls Dangerous | `c` RISES |
| k random cells | near 0 |

The random arm holds the hole size fixed, so it measures what a hole of that
size does on its own.

Result, 18 images, 2 x 2 cells: the order holds at every k from 1 to 32, and
the 2 targeted arms sit 2.3 to 4.4 random spreads apart.

```bash
sbatch research/monocle/monocle.sub monocle.deletion_curve \
    --cases subway_safety --n-images 20 --group 2
```

### 2. Content, not holes: the substitution test

The same ranking, but the cells take a donor's PIXELS instead of going away.
No slot is removed, no token is dropped, the coordinates stay. The order still
holds, at z of 1.9 to 2.7.

```bash
... monocle.deletion_curve --mode substitute --n-donors 3
```

### 3. Sufficiency: the keep-only test

Invert the edit: keep k cells and take the rest away. Keeping the 4 safest
cells of 64 raises the verdict ABOVE the whole photograph.

```bash
... monocle.deletion_curve --keep-only --ks 4 8 16
```

### 4. Transfer: the test with no circularity

Take the cells image A's map calls Dangerous and put their pixels into image
B. The map is built on A and judged on B, so nothing is circular.

Result, 153 (target, source) pairs over 17 strong photographs: A's Safe
cells raise B's verdict 0.531 logits more than A's Dangerous cells do
(1-sided t p = 0.0144; by target, 14 of 17 positive, 1-sided t on the
per-target means p = 0.0096). The 3 arms are matched for spatial spread, so
clustering does not explain it. Section 8 carries the same row, and the 2
now agree.

**Warning: an earlier 18-photograph run read +0.872 at p = 0.0008.** A later
20-photograph run overwrote its folder. The numbers above come from the run
on disk, and they are smaller.

```bash
sbatch research/monocle/monocle.sub monocle.content_transfer \
    --n-images 20 --n-sources 9 --ks 8
```

**Warning: this test needs about 150 pairs.** At n=24 it read as a null
(p = 0.15). The effect is about +0.6 against a spread of 3.

### 5. Specificity: is it a saliency map?

Read the SAME photograph under 4 prompts and correlate the maps. `daylight`
(Day / Night) is the orthogonal control: it asks nothing about quality, and
the picture is worth MORE to it than to safety, so a null there is not a floor
effect.

12 photographs, group 2. The module writes the medians, so the medians are
what this table reads.

| Prompt | median r against the safety map | median top 1/8 overlap |
|---|---|---|
| road_quality | 0.247 | 0.31 |
| street_photography | 0.194 | 0.25 |
| **daylight** | **0.032** | **0.12** (chance is 0.125) |

The `daylight` mean is **-0.056**, so the control map does not merely
decorrelate from the safety map. It leans the other way.

**Warning: an earlier run of this table read 0.298, 0.276 and 0.046.** Those
numbers are not in `outputs/_monocle/prompt_specificity/pairs.parquet`. The
values above come from the parquet, and the control is CLEANER than the older
table claimed.

```bash
sbatch research/monocle/monocle.sub monocle.prompt_specificity --n-images 12
```

### 6. What is in the square?

No simple image property explains the map: luminance, contrast, edge density,
saturation, and height in frame all read |r| under 0.06 within an image. The
`col` control, which must be 0, is the LARGEST of them at 0.048.

```bash
python -m research.monocle.cell_statistics --maps <maps.parquet> --stats <map_stats.parquet>
```

## The 2 methods that disagree

| Pair | pearson, per cell | bottom-8 overlap |
|---|---|---|
| donor (L20 swap) against absence (input) | +0.092 | 0.125 = chance |
| absence against substitution (both input) | +0.329 | 0.250 |

**The donor map is retired.** A layer-20 residual carries the whole scene,
because gemma-4 gives the image block a bidirectional mask from block 0. A
swap there does not ask about the content of 1 square. Every donor-map figure
made before 2026-08-23 is not a map of what the content contributes.

### What the map reads, and what it does not

The map reads ONE photograph. `--reference absent` removes the patch at the
input and opens no second image, and the 2 control arms use the same removal.
`n_donors` is 0 in every `*_absent` run on disk.

| Where | Reads a 2nd photograph | Role |
|---|---|---|
| `--reference absent` | no | **the map** |
| `--reference substitute` | yes, 3 donors, pixels | a control |
| `--reference donor` | yes, 3 donors, residuals at a layer | RETIRED |
| `deletion_curve --mode ablate` | no | the test |
| `deletion_curve --mode substitute` | yes, 3 donors, pixels | the test |
| `content_transfer` | yes, A's cells into B | generalization |

A pixel substitution is same-index: cell (3, 7) of the target takes cell
(3, 7) of the donor, so the coordinates stay and only the content changes.
The position embedding is factorized per patch, thus the target keeps its own
grid.

**Why the substitution stays a control.** Its zero is the donors, not the
absence of the square, so the value depends on which donors you draw. That
was the first of the 2 defects that killed the L20 swap. An input removal has
a real zero, and only it can answer a parts-against-whole question. Thus
`group_sweep` refuses every other reference.

`--reference` now defaults to `absent`. It defaulted to `donor` while the
swap was the only method, and a reader who left the flag off got the retired
method. `summary.json` also carried a fixed `method` string that named the
donor swap in every folder, `absent` folders included. Both are corrected,
and 5 tests hold them.

### One zero for the parts and the whole

In bf16 the same row gives a `c` that moves by up to 0.67 logits when the
batch size changes. `ablation_map` thus reads every cell against a no-op row
inside a batch of `batch`. `controls` read the whole image alone, at batch 1.

A shared offset cancels between cells, so the MAP was never harmed. The ratio
was: `parts_over_whole` sums 256 cells against one reference and divides by a
whole measured against another, so the sum can carry 256 copies of the
offset.

`controls` now takes `batch`, and `image_worth` is `zero - c_all` at that
size. The `none` arm stays at batch 1, because `c0` is read alone. The offset
is written per image as `batch_offset`.

**Do not compare the group folders written before 2026-08-23.** Their
`image_worth` median reads +8.177 at groups 1, 4, 8 and 16, and +4.109 at
group 2. That pair is the prefix signature, so those folders do not share a
prompt.

### The corrected sweep

Job 475554 measured all 5 group sizes in one model load, so they share a
prompt, a seed and a batch. 20 images, 17 strong.

| group | cells | px of the model image | parts/whole | strongest cell |
|---|---|---|---|---|
| 1 | 256 | 48 | -2.687 | 0.210 |
| 2 | 64 | 96 | +0.365 | 0.287 |
| 4 | 16 | 192 | +0.722 | 0.376 |
| 8 | 4 | 384 | +0.307 | 0.390 |
| 16 | 1 | 768 | +1.000 | 1.000 |

The end point is exact: a 16x16 run holds 1 cell, which IS the whole-image
ablation, and `parts_over_whole` reads 1.000 with a worst error of 0.00000
over 17 images.

**The cells never sum to the verdict.** -2.69, +0.37, +0.72, +0.31. Only the
1-cell grid reads 1.000, and it does so by construction.

The group-2 map reproduced bit for bit against the folder written before the
batch fix: vmax 2.4573058700562 against 2.4573058700561545. That folder was
already on the correct prompt, so this is the proof that a shared batch
offset cancels between cells and never moved a map. It moved the ratio by
about 30%.

## 7. What is in the square, in the model's own words

Keep only the 16 cells one arm names, ask an OPEN question, let the model
write a sentence, then feed that sentence back as TEXT ONLY with the safety
question. No word list of ours enters the loop.

20 photographs, 17 strong, group 2, k = 16. Job 475810 and 475934.

| Arm | median text-only `c` |
|---|---|
| keep the Safe cells | +14.81 |
| keep the whole image | +14.05 |
| keep 16 random cells | +13.70 |
| keep the Dangerous cells | +7.01 |

The paired difference is +5.48 in the mean and +2.47 in the median, positive
for 13 of 17 photographs. A 1-sided t-test gives p = 0.046 and a Wilcoxon
test gives p = 0.010. Read the Wilcoxon, because 2 photographs carry a
difference above +20 and they pull the mean.

**Warning: an earlier 12-photograph run is on record in this page's history,
and its numbers were larger.** It read +15.35, +15.02, +13.94 and +6.87, with
a difference of +8.55 over 9 of 10 photographs. The 17-photograph run
replaces it. The effect holds and it is smaller.

### The grid decides whether the loop closes

The 3 runs keep the SAME quarter of the photograph, 64 patches of 256. Only
the size of a cell changes. The effect does not survive that change.

| Run | A cell | Cells kept | safe - danger | Photographs | wilcoxon p |
|---|---|---|---|---|---|
| group 2 | 2x2 patches | 16 | **+5.48** | 13 of 17 | **0.010** |
| group 4 | 4x4 patches | 4 | +3.07 | 11 of 17 | 0.122 |
| group 1 | 1 patch | 64 | +0.48 | 8 of 17 | 0.555 |

**The group-1 run is a null, and the page predicts it.** A single cell is not
reproducible: 2 input methods rank the cells at r = 0.33. So a group-1 order
is noise, and a caption of its top 64 cells carries no judgment. The captions
still differ from arm to arm, 17 distinct of 17, so the model CAN describe a
scattered quarter. It is the ranking that fails, not the description.

The loop thus closes at the grain where the map is trustworthy, and it opens
at the grain where the map is not. That is a control, and it agrees with the
kill list at the top of this page.

### The words

The word split is weaker than the 12-photograph run reported. Words that
appear in 2 or more captions of 1 arm and never in the other:

| Only in the SAFE captions | Only in the DANGEROUS captions |
|---|---|
| station, subway, entrance, platform, lights, people, pedestrians | interior, ceiling, facade, asphalt, industrial, space |

**Do not quote the older list.** It held `train` on the Safe side, and the
17-photograph run puts `train` on the Dangerous side twice and never on the
Safe side. `stairs`, `doorway` and `edge` appear in neither arm. A word list
from 10 captions is not stable, and the DANGEROUS side of this one still
carries function words that mean nothing.

Read the medians and the paired test. Read the captions for colour, not for a
word list.

```bash
sbatch research/monocle/monocle.sub monocle.caption_loop --n-images 20 --group 2 --k 16
```

**Warning:** the model is describing a photograph with 48 of 64 cells removed,
so part of what it says is invented. Read the captions, do not only read the
number.

## 8. Does it leave subway platforms?

The deletion test on 4 questions, each with its own labels and its own pool of
photographs.

| case | n | k=1 | k=4 | k=16 | k=1 signs |
|---|---|---|---|---|---|
| subway_safety | 17 | +2.91 | +3.61 | +3.82 | 17/17 |
| road_quality | 14 | +4.22 | +8.15 | +9.57 | 14/14 |
| parks_plazas | 13 | +3.44 | +4.85 | +6.20 | 13/13 |
| restaurants | 14 | +4.85 | +5.19 | +6.69 | 14/14 |

Every one of the 12 comparisons sits at p < 0.002, and that holds 2-sided as
well: the worst is 9.97e-04 1-sided and 1.99e-03 2-sided. **58 photographs of
58** move the predicted way when the single strongest cell goes. The safety
question gives the SMALLEST effects of the 4.

```bash
sbatch research/monocle/monocle.sub monocle.deletion_curve \
    --cases road_quality parks_plazas restaurants --n-images 16 --ks 1 4 16
```

The transfer test travels as well:

| case | pairs | mean(safe in - danger in) | p | by target | p |
|---|---|---|---|---|---|
| subway_safety | 153 | +0.531 | 0.0144 | 14/17 | 0.0096 |
| road_quality | 98 | +1.791 | 0.0001 | 11/14 | 0.0013 |
| restaurants | 98 | +1.047 | 0.0051 | 10/14 | 0.0132 |

## 9. Which word of the question holds the picture

The relay puts the judgment in the question tokens over blocks 23 to 31. The
lens cannot read them: 2 lenses fitted on different corpora correlate at 0.46
to 0.82 there. A swap can, and its null control reads 0.000000.

**The words decompose, and the picture does not.** Sum the 66 single-token
swaps against the whole-question swap: 0.67 at L25, 0.73 at L27, 0.88 at L29,
**1.02 at L33**. The image cells sum to between -2.3 and +1.4 times the whole,
at any grid size.

| Word | L22 | L25 | L27 | L29 | L31 | L33 |
|---|---|---|---|---|---|---|
| ` this` (the demonstrative) | **+0.04** | +0.05 | +0.04 | +0.02 | 0.00 | 0.00 |
| ` Dangerous` (a label) | 0.00 | **+0.08** | +0.06 | +0.05 | +0.04 | +0.04 |
| ` Safe` (a label) | +0.01 | +0.04 | +0.03 | +0.02 | +0.01 | +0.02 |
| the newline after `model` | 0.00 | -0.01 | +0.05 | **+0.11** | +0.04 | 0.00 |
| `":` (the last token) | 0.00 | 0.00 | 0.00 | +0.05 | +0.10 | **+0.16** |

A chain: the word that POINTS at the image wakes first, then the 2 label words
themselves, then the turn scaffold, then the last token of the prompt as the
answer position takes over. No word holds much; the largest is 0.19 against a
whole-question 0.61.

```bash
sbatch research/monocle/monocle.sub monocle.question_tokens \
    --cases subway_safety --n-images 6 --layers 22 25 27 29 31 33
```

**Warning: the zero must be a self swap of the SAME positions, in the SAME
batch.** 3 designs were tried. A `c0` from a separate forward gave a null
control of 103.6. A self row at OTHER positions in the same batch gave 0.488,
because a recorded activation comes from a batch-1 forward and is written into
a batched one, so writing ANY recorded value is already a small change. Only a
self swap of the same positions cancels it, and then the null reads 0.000000.

## An instrument fact you must know

**Batch size moves `c` by up to 0.67 logits, with no intervention at all.**
Measured on gemma-4-12b: the same input at batch 1, 2, 4, 8, 12 and 16 gives 6
different answers, spread up to 0.672. Inside one batch, 16 identical rows
agree bit for bit. It is bf16 accumulation order, and it is deterministic
given the size.

| What | Safe? |
|---|---|
| a difference between arms IN ONE BATCH | exact |
| `dc` against a `c0` measured alone | biased by up to 0.67 |
| a map whose last chunk is a different size | a step at that boundary |

So `ablation_map` and `substitution_map` pad every chunk to exactly `batch`
rows and measure the zero in a batch of `batch` no-op rows.

**Warning:** a no-op row cannot ride inside an ablation batch. An ablation
shortens the sequence, so the 2 rows cannot stack. `ablation_batch` refuses a
mixed size, which is what caught the first version of this fix.

## Rules for a figure

1. Read the ends of the map, never a single cell.
2. Never present the cells as a decomposition of the verdict.
3. Name the 2 labels on the colour bar. See [[patch-attribution]].
4. Say which question the map answers. The same photograph gives a different
   map under a different question, which is the point.
5. Mark a weak image. Some photographs are worth under 2 logits to the model,
   and a ratio from them is noise.
6. Compare arms with a PAIRED test over the independent unit. A median-of-z
   summary hid a real transfer effect until a paired test found it.
7. Keep the batch size fixed across a measurement, and measure the zero at
   that size.
