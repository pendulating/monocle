# The valence of the F9 inferences, and the figures that use it

Written 2026-10-30. See `f9-subjective-prompt.md` for the F9 prompts, and
`facct-pairwise-ses.md` for the features and the test.

## Why this pass exists

The F4 schema gives a valence to `visual_evidence` spans only. An `inference`
span has a `kind` and no valence. The row of social inferences in `trace_ses`
needs a valence, thus a second pass labels these spans.

## The pass

`research/scripts/f9/label_inference_valence.py` 1.0.0, with
`research/scripts/f9/label_inference_valence.sbatch`.

| Item | Value |
|------|-------|
| Spans | `inference` spans of kind upkeep, wealth, demographic, class, safety |
| Scope | F9 subjective prompts, 2 cases, 6 models, fields `response` and `reasoning` |
| Count | 65,266 spans (restaurant 58,541, street 6,725) |
| Labeller | DeepSeek-V4-Flash, `guided_json`, 1 B300 |
| Input | the span, with 300 characters of context on each side, and 6 invented few-shot examples |
| Labels | `good`, `bad`, `neutral` |
| Output | `data/f4_traces/subj/inference_valence/full.parquet` |

**Warning:** `prepare` writes `input.parquet` again. `span_id` values in the
label files refer to that file. Do not run `prepare` again, unless you make
sure that the span order stays the same.

### Result

All 65,266 spans have a label, and all finish with `stop`. The distribution is
neutral 29,964, bad 20,904, good 14,398.

| Case | Kind | Spans | Bad | Good | Neutral |
|------|------|-------|-----|------|---------|
| Restaurant | wealth | 34,072 | 0.240 | 0.106 | 0.653 |
| Restaurant | upkeep | 20,126 | 0.374 | 0.422 | 0.204 |
| Restaurant | safety | 1,778 | 0.406 | 0.389 | 0.205 |
| Restaurant | demographic | 1,568 | 0.008 | 0.199 | 0.793 |
| Restaurant | class | 997 | 0.150 | 0.079 | 0.770 |
| Street | safety | 4,490 | 0.804 | 0.077 | 0.119 |
| Street | upkeep | 1,329 | 0.467 | 0.389 | 0.144 |
| Street | demographic | 433 | 0.051 | 0.406 | 0.543 |
| Street | class | 257 | 0.121 | 0.311 | 0.568 |
| Street | wealth | 216 | 0.060 | 0.403 | 0.537 |

Thus the 2 cases mean different things. Restaurant social valence is mostly
wealth and upkeep. Street social valence is mostly safety.

### Test and retest

The retest labels 3,000 spans again, with a reversed label order and seed 1.

| Group | Spans | Agreement | Kappa |
|-------|-------|-----------|-------|
| All | 3,000 | 0.930 | 0.889 |
| Restaurant | 2,673 | 0.929 | 0.886 |
| Street | 327 | 0.936 | 0.879 |
| upkeep | 986 | 0.963 | 0.943 |
| safety | 302 | 0.954 | 0.897 |
| class | 65 | 0.938 | 0.806 |
| demographic | 90 | 0.933 | 0.816 |
| wealth | 1,557 | 0.904 | 0.799 |

No span changes between `good` and `bad`. All 210 changes involve `neutral`.

**Warning:** this test measures stability, not accuracy. We read a stratified
sample by eye: most labels are correct. Negated wealth is inconsistent. "Not
Michelin-star material" is `bad`, and "doesn't look like a fancy, high-end
dining experience" is `neutral`. No human annotation exists for this pass.

## `trace_ses`

`research/notebooks/facct/_trace_ses.py` 1.2.1. It uses the 6 F4 models and 2 layers
(NTA, community district), with the test of `facct-pairwise-ses.md`. The rows,
top to bottom: net valence, social net valence, location inference,
uncertainty in view. Bonferroni counts 480 tests.

### The `out_of_view` filter

`uncertainty_reasons()` writes
`data/f4_traces/subj/inference_valence/uncertainty_reasons.parquet` (`doc_id`,
`reason`, `spans`). A text scores 1 when it has an `uncertainty` span with a
reason other than `out_of_view`. A text with only `out_of_view` spans scores
0 and stays in the denominator.

`out_of_view` is the largest reason in the restaurant answers (20,158 of
26,562 `response` spans), and it is rare in the street answers (143 of 883).
The CSV keeps the old score as `uncertainty_all`. That outcome is not in the
figure or in the Bonferroni count.

### Result (2026-10-30)

| Outcome | Restaurants | Street photography |
|---------|-------------|--------------------|
| Net valence | 48 of 60 | 18 of 60 |
| Social net valence | 29 of 60 | 1 of 60 |
| Location inference | 8 of 60 | 0 of 60 |
| Uncertainty, in view | 20 of 60 | 0 of 60 |
| Uncertainty, all reasons (not in the figure) | 26 of 60 | 0 of 60 |

The filter does not remove the restaurant gradient. At the district, the mean
tau of % bachelor's goes from 0.36 to 0.42, and income stays at 0.36. At the
NTA, most taus fall: income 0.22 to 0.17, % white 0.19 to 0.16. Qwen3.8-27B
goes the other way from the other models on location, in restaurants.

**Warning:** the share of restaurant answers with a filtered uncertainty span
is small and varies by model: 0.024 (Qwen3.5-9B) to 0.258 (Llama-3.2-11B).
The share with all reasons is 0.122 to 0.444.

**Warning:** the social row is thin. It reads only the answers with a social
inference: restaurants 1,977 to 4,865 texts for each model and 170 to 193
NTAs; street 274 to 585 texts for each model and 37 to 101 NTAs. Do not read
the street panel of this row as a null result, because it has low power.

**Paper (2026-11-03):** the figure is `fig:trace-ses` in Appendix B, and the "Popularity" paragraph of section 4.3 reads the in-view uncertainty row (20 of 60 restaurant cells, 0 street cells).

### Outputs

| File | Content |
|------|---------|
| `research/notebooks/facct/figures/trace_ses/trace_ses.png`, `.pdf`, `.json` | the figure and its sidecar |
| `research/notebooks/facct/figures/trace_ses/trace_ses_cells.csv` | 600 cells, with the 120 cells of `uncertainty_all` |

## The F9 win blocks (D147)

`research/scripts/f9/export_valence_blocks.py` 1.4.0 writes the blocks to
`research/notebooks/facct/figures/f9_blocks/`. The cue is a term (a word, or 2 adjacent
content words). The colour is the net valence of the spans with the term. Each
text gives 1 vote.

**Warning:** do not use a leave-cue-out score. It reversed the street block
(`trash` scored +0.53 and its own spans score -0.77). The script uses the own
net valence of the cue.

**Warning:** we did not verify negation in the cues. 2 of the lowest
restaurant images are dark.

### The inference block

`--sources inference` draws the same block from the social inferences, with
the valence of the span itself. The files are
`{restaurant,street}_inference_block{,_column}.png`, with `.json` sidecars, and
`_inference_block_cues.csv` and `_inference_block_images.csv`.

Inference spans are sparse, thus the floors differ from the first block:

| Item | `visual_evidence` block | Inference block | Why |
|------|-------------------------|-----------------|-----|
| Cue model | Gemma-4-12B | 6 models, pooled, 1 vote for each text | Gemma-4-12B alone gives 42 street cues |
| Cues | 300 | restaurant 300, street 147 | all cues with 25 texts or more |
| Image floor | 30 spans, all 6 models | 3 spans, 3 models or more | no image holds 30 of these spans |
| Images | restaurant 214, street 513 | restaurant 2,471, street 118 | |
| Stop words | `FRAME_WORDS` | also `HEDGE_WORDS` | `usually suggests` says how a span is put |
| Text before the span | not used | used for a span that opens as a fragment | `input.parquet` holds it |

On a tie of net valence, good minus bad decides the order, thus the image with
more spans goes to the outer end at both ends. The first form used the span
count, and that put the images with the fewest spans in the bottom row. The
`visual_evidence` photographs do not change with this rule (checked).

The paper uses `restaurant_inference_block.png` as `fig:restaurant-inference-block`. It replaces
the pairwise win block in the section on the aesthetic hierarchy (2026-11-01). The prose cites span
counts: `older` has 43 good, 41 bad and 47 neutral spans of 131.

### Negation

The first form of the inference block scored `upscale` on "not upscale", thus
`upscale dining` and `high-end establishment` sat in the red. A keyword rule
finds a negation in 88.4% of the 9,687 restaurant `response` spans with a word
of wealth, and in 53.6% of the 1,128 spans with a word of decay.

The valence labels were correct: 2.0% of the negated wealth spans are `good`.
The fault was in the cue step, thus `trace_ses` does not change. It reads the
span labels, not the cues.

`terms()` now marks negation (the rule of Pang et al., 2002). A term in the
scope of a negator takes the form `not <term>`, and it is a separate cue. Both
block types use it.

| Rule | Value | Example |
|------|-------|---------|
| Negators | `NEGATORS`, with `less`, `unlikely`, `lack`, `without`; also `rather than`, `opposite of` | "the opposite of upscale" |
| The scope ends at | end of sentence, or a word of `SCOPE_END` (`but`, `though`, `because`) | "not fancy, but clean" gives `clean` |
| The scope does not end at | a comma, a bracket, `of which` | "not a fancy, high-end spot"; "none of which reads as upscale" |
| A standard before the negator | a span that opens with `if you` or `if your`, and holds a negator, is negated from its start | "If you want a high-end meal, this is not it" |
| A cut span | a span that opens with `that`, `to`, `or`, `and`, `like` takes an open negator from the text before it (inference block only) | "[nothing] that would suggest high-end" |

| Cue, restaurant inference block | Plain | Negated |
|---------------------------------|-------|---------|
| `upscale` | +0.50 | -0.43 |
| `high-end` | +0.33 | -0.40 |
| `polished` | +0.33 | -0.66 |
| `fancy` | +0.31 | -0.08 |
| `clean` | +0.83 | -0.82 |
| `run-down` | -0.69 | +0.81 |

109 of the 300 restaurant cues are negated, and 16 of the 147 street cues. The
visual evidence blocks have 6 and 11 of 300. The photographs do not change.

**Warning:** the rule is a heuristic, not a parse. We read samples of plain
`upscale` and `high-end` spans with the label `bad`, and repaired the patterns
that we found. 33 of 494 plain `upscale` spans and 41 of 304 plain `high-end`
spans are still `bad`, and some of these are missed negations. An `if you want
X` span with no negator ("this is too informal") stays plain.

**Note:** a negated cue can read oddly (`not food`, `not place`), because each
word in the scope gets the mark.

**Warning:** the street block is mostly about the photographer, not about the
area. Safety is 1,826 of the 3,081 street `response` spans. A keyword rule
finds words of the shoot or the road in 60.2% of these safety spans, and words
of the area in 18.5%. The lowest cues are `dangerous`, `highway`, `illegal`,
`road`, `standing`, `shoulder`, and the lowest 6 images are highways. This also
applies to the street panel of the social row in `trace_ses`. Do not read that
panel as a judgment of the neighborhood.
