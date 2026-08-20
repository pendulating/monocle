# Integrative Complexity ingredient extraction

The pipeline reads 1 reasoning trace and returns the raw material of an
Integrative Complexity (IC) code. It returns **no code and no score**. A later
step derives those on the CPU, thus a threshold changes without a second run on
the GPU.

| Piece | Where |
|-------|-------|
| Schema and prompt | `dagspaces/common/ic_schema.py` |
| Examples | `dagspaces/common/ic_examples_v2.json` |
| Stage | `dagspaces/urbanpairvqa/stages/ic_extract.py` |
| Pipeline | `dagspaces/urbanpairvqa/conf/pipeline/ic_extract.yaml` |
| Merge | `scripts/merge_trace_extractions.py --schema ic` |

## The rule that makes it work

The IC codebook asks for judgments: is the uncertainty adequately justified? Is
the perspective substantively developed? A model answers a naked yes or no
badly, and no reader can audit that answer.

Thus the prompt never asks for the judgment. It asks for the SPAN, and the code
comes from the span.

| The codebook asks | The model returns | The later step derives |
|-------------------|-------------------|------------------------|
| Uncertainty justified | `justification_quotes` | the list is not empty |
| Perspective developed | `supporting_quotes` | the list holds 2 or more |
| Weighing justified | `justification_quotes` | the list is not empty |
| Context sensitivity | `condition_quotes` | the list is not empty |

`locate_quote` finds each span in the source trace and returns its character
offsets. There is no fuzzy aligner on this path, thus a quote that no search
finds is a defect and not data. Read `quote_found_rate` first on every run.

## The 7 ingredients

`dimension`, `perspective`, `verdict`, `weighing`, `dismissal`,
`reconsideration`, `hedge`. `dimensions` and `verdicts` carry a lower bound of
1, because a trace always names something it sees and always states a decision.
The other 5 may be empty, and an empty list is the correct answer for a simple
trace.

## Why the order of the arrays is not free

Guided decoding fills the arrays in the declared order, thus the token cap eats
the last ones. Measured on the v1 pilot, survival fell in exactly the declared
order and `weighings` died last at 7%.

v2 moved the rare and necessary classes early. It worked, but it also showed
that the position of a class changes how much of it the model reports, and
truncation does not explain the change:

| Ingredient | v1 (position) | v2 (position) | share of traces, untruncated answers only |
|------------|---------------|---------------|--------------------------------------------|
| weighing | 7th | 4th | 50.0% -> 51.4% |
| dismissal | 3rd | 5th | **85.2% -> 64.2%** |
| hedge | 6th | 7th | 90.7% -> 94.3% |
| perspective | 2nd | 2nd | 95.2% -> 94.3% |

The examples and the instruction text of `dismissal` did not change between the
2 versions. Only its position did, and its rate fell 21 points.

**Warning:** thus an ingredient rate is a property of the schema version, not of
the model alone. Never pool 2 versions, and read a rate as a lower bound.

## The pilots

300 traces of the 2026-08-13 subway thinking run, on 1 A6000:

| Arm | Answers cut | Quote located | Weighing | Dismissal |
|-----|-------------|---------------|----------|-----------|
| v1 few-shot | 10.0% | 91.5% | 45.7% | 82.0% |
| v1 zero-shot | 15.7% | 85.1% | 42.3% | 85.0% |
| v2 few-shot | 6.0% | 89.6% | 51.0% | 63.0% |

The few-shot prompt does NOT leak the base rate of its own examples: all 3
examples hold a weighing, and the zero-shot arm found weighings at the same
rate. Few-shot locates quotes better and truncates less, thus it is the default.

## Run it

The source is a THINKING run. The canonical battery runs greedy with
`max_tokens=128`, thus its `model_reasoning` column is empty.

Point the run at the canonical registry, which names 1 trace run for each case
and model:

```bash
export HYDRA_SWEEP_DIR=multirun/ic_<name>
.venv-mllmsci-vllm025cu129/bin/python -m dagspaces.urbanpairvqa.cli -m \
    pipeline=ic_extract \
    ic_extract.results_path=notebooks/cvpr/canonical_data/trace/<case>__gemma-4-12b/results.parquet \
    ic_extract.shard_count=6 ic_extract.shard_index=0,1,2,3,4,5 \
    hydra.launcher.array_parallelism=4
```

The stage follows the symlink before it reads a name from the path. Without
that, every registry link is called `results.parquet`, and the case would be
`results` and the judge model empty.

Then merge:

```bash
python scripts/merge_trace_extractions.py multirun/ic_<name>/... --schema ic
```

## Cost

Measured 2026-08-14: about 1,400 traces in 1 GPU-hour. A case holds 11,000
traces, thus about 8 GPU-hours, or 6 shards of about 1.4 hours.

## The numbers to read after a run

| Metric | What it means | The pilot |
|--------|---------------|-----------|
| `quote_found_rate` | the span is really in the trace | 0.90 |
| `exact_rate` | the span matched word for word | 0.90 |
| `sub_quote_found_rate` | the justifications are grounded | 0.93 |
| `answers_cut` | the token cap cut the answer | 6% |
| `ingredients_per_trace` | the size of a trace | 34 median |

## From ingredients to codes

`dagspaces/common/ic_codes.py` reads the ingredient table and gives each trace a
code. The GPU run is expensive and the thresholds are not settled, thus the run
writes ingredients and the code lives on the CPU. A threshold change costs
seconds.

| Code | What the trace shows | The rule |
|------|----------------------|----------|
| 1 | 1 view | nothing below holds |
| 2 | transitional | a hedge, or an alternative set aside |
| 3 | differentiation | 2 developed perspectives, or 2 distinct dimensions on both images or both valences |
| 4 | unjustified link | differentiation, plus a weighing, a revised verdict, or a reconsideration |
| 5 | integration | a weighing whose justification span is LOCATED |
| 6 | integration + context | a named condition, or 2 distinct weighing mechanisms |

**Warning: the scale runs 1 to 6 here.** Code 7 of the codebook needs an
organizing principle above the integrations, and the schema holds no ingredient
for it. "No trace reached 7" describes the schema, not the model.

Two rules keep a code honest:

- Only a LOCATED span counts. A quote that no search finds is a defect of the
  extractor, thus it lifts no code.
- A trace whose answer the token cap CUT is dropped from every rate. An absence
  inside a cut answer is not a zero.

`Thresholds` holds every number a code depends on: how many located supporting
spans make a perspective "developed" (2), how many distinct dimensions make
differentiation (2), and how many mechanisms stand for context (2).

Pseudo-differentiation has its own flag. A trace that lists 5 cues for image A
and none for image B has described, not differentiated.

## The report

| Piece | Where |
|-------|-------|
| Loader, tables, and figures | `notebooks/cvpr/_ic.py` |
| Notebook | `notebooks/cvpr/master/ic_complexity.py` |
| Headless export | `scripts/export_cvpr_ic_figures.py` |

Both paths hold 2 gates: the canonical registry must match the disk, and the
corpus must come from the REGISTERED thinking runs
(`_ic.registry_mismatch`). The figures are the code mix for each case, the
components under it, and the mean code against the judgment the model gave.

**The first test of the whole pipeline** is that last figure. A `Same` or a
`NotSure` is the hard pair, thus it should cost more reasoning than a
`MuchMore`. A flat line means the code measures nothing the judgment knows
about.

## The word blocks

`notebooks/cvpr/_ic_words.py` counts the words of the located SPANS, not of the
whole trace, and asks what each part of the reasoning sounds like. The weights
use the same `distinctive` score as the trace clouds, thus a word grows when 1
block uses it and the others do not.

| `group_by` | 1 block for each | Question |
|------------|------------------|----------|
| `type` | ingredient type | What does a weighing sound like, against a dismissal? |
| `case` | case | Which words does this case put in its spans? |
| `case_type` | case, inside 1 type | Which cues does this case name? |

### Warning: `distinctive` does not separate the blocks

The `distinctive` score compares 1 block with every OTHER block POOLED, and the
pool is dominated by `dimension`: 1,163,942 spans against 26,286 for
`weighing`. Thus a word that the small blocks share, and the big one does not,
scores high in ALL of them at once. Measured on the corpus of 2026-08-18, **65
of the 198 top-40 words sat in 2 blocks or more**: "usually" led both
`perspective` and `weighing`, and "library" led `dismissal`, `hedge`, and
`reconsideration`.

`exclusive` mode is the default for that reason. It scores every block, then
gives each word to the single block that scores it highest, thus no 2 blocks
share a word and a picture answers "what is THIS part, and not that one". The
word table names the other blocks that also rank a word, in `also_in`.

| Block | Spans | Top exclusive words |
|-------|-------|---------------------|
| dimension | 1,163,942 | building, brick, clean, large, trees, sidewalk, cars |
| hedge | 399,833 | maybe, hard, just, tell, possibly, subjective, similar |
| reconsideration | 256,527 | look, examine, prompt, read, evaluate, closer, double |
| perspective | 164,357 | usually, better, perceived, types, visual, safer, generally |
| verdict | 162,770 | appealing, condition, good, maintained, eat, safe, lean |
| dismissal | 115,910 | actually, plaza, park, library, public, really, subway |
| weighing | 26,286 | difference, wins, means, hand, stark, contrast, leans |

### What `dismissal` really holds

The word block put the unit noun at the top of `dismissal`, and the `target`
field says why. The commonest targets are "library identification" (5,620),
"park classification", "subway entrance visibility", "prompt premise", and
"prompt accuracy". **39% of dismissals target the premise or the referent**,
not a cue: 67% for libraries, 59% for parks and plazas, 7% for street
photography.

The model is not setting a consideration aside. It is doubting that the image
shows the thing the prompt names:

> "it's not really a park or plaza"
> "I don't see a library in Image A"
> "Actually, the prompt says 'Both show New York City public library buildings'."

**This matters for the codes.** Rung 2 rests on a dismissal, thus for the
facing-filtered cases rung 2 largely measures referent doubt and not a
transitional reasoning move. Read the `dismissal` rate of libraries and parks
as a data-quality signal first.

## The link to the proxy

`notebooks/cvpr/_ic_link.py` and `master/ic_linking.py` ask whether a complex
trace agrees with the outside measurement more often than a simple one. The
join happens at the POLYGON, because no per-unit proxy exists:

    proxy_gap = proxy(polygon of A) - proxy(polygon of B)
    agrees    = the sign of the gap matches the sign of `relative_score`

**Control for difficulty or the answer is wrong.** A model reasons longest
about a hard pair, and a hard pair is 2 areas that sit close together. The
tables split the pairs into bands by the size of the gap and compare the codes
INSIDE a band.

Measured 2026-08-18 over the gemma-4-12b corpus, 8 case-proxy contrasts: the
pooled difference between code >= 5 and code < 5 runs from -0.055 to +0.021,
with 4 of each sign and every value inside its Wilson interval. **Reasoning
complexity does not predict agreement with the proxy.**

### The 2 modes of the join

| Mode | Cases | What goes into the polygon |
|------|-------|----------------------------|
| unit | subway, libraries, schools, parks and plazas, restaurants | the FacDB position of the unit |
| image | road quality, street photography | the CAMERA position of each shot |

An image-mode pair holds no unit, thus the camera position is the only position
there is — and it is the true one, unlike in unit mode, where the camera sits up
to 80 ft from the building it looks at. An image-mode case also keeps far more
pairs: 2 random citywide shots rarely share a polygon, thus road quality keeps
82-87% of its traces against 43-63% for the unit cases.

Street photography reaches the test only when a proxy exists for it. It has no
outside measurement today, thus the summary names it with that reason rather
than dropping it.

### Road quality: chance at the level of 1 pair

All 4 road-quality proxies land between 0.506 and 0.543, which is chance:

| Proxy | Usable pairs | Agreement | Complex - simple |
|-------|--------------|-----------|------------------|
| Crime density (negated) | 9,613 | 0.528 | +0.008 |
| DOT pavement rating | 9,217 | 0.506 | -0.034 |
| Median household income | 9,033 | 0.543 | -0.062 |
| Pothole repairs (negated) | 9,543 | 0.509 | +0.024 |

**Warning: this does NOT contradict the area-level table.** The results table
correlates AREA MEANS, where the noise of 100,000 pairs cancels, and it reports
a strong crime row for road quality. This table asks about 1 pair, where the
same signal is thin. A model can order areas well and still call a single pair
of blocks at chance.

## What is still missing

- The 2 image-mode cases in the proxy link. They need the camera position
  joined to the polygon instead of the unit position.
- A second judge. The corpus of 2026-08-18 covers the gemma-4-12b traces alone.
  qwen3.5-9b writes traces 2.5x longer, and whether that is more reasoning or
  more words is exactly what a code can answer.
