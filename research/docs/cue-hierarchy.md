# The aesthetic hierarchy over the urban cues

Does the model carry 1 aesthetic standard from prompt to prompt? Does a
chain-link fence cost a school what it costs a subway entrance?

| Piece | Where |
|-------|-------|
| Module | `research/notebooks/cvpr/_cues_across.py` |
| Notebook | `research/notebooks/cvpr/master/cue_hierarchy.py` |
| Figures and tables | `research/notebooks/cvpr/master/figures/cue_hierarchy_*` |
| Source | The `urban_cues_v2` primary corpus, `data/trace_extractions_v2_gemma12b_by_qwen27b/` (gemma-4-12b traces read by Qwen3.8-27B). The headline numbers come from the rerun of 2026-09-22 (`research/scripts/export_trace_figures.py`); a number marked v1 is from the 2026-08-28 corpus. |

```bash
.venv-mllmsci-vllm025cu129/bin/marimo edit \
    research/notebooks/cvpr/master/cue_hierarchy.py
```

**Note: `visual_evidence` is the class that holds when the extractor
changes.** The review of 2026-09-20 read the same traces with both
extractors. The `visual_evidence` rate stayed within 1.05x, `image` agreed on
99% of shared spans and `valence` on 91%. Thus the hierarchy stands on the
part of the corpus that the 2 extractors agree on. See
`langextract-trace-extraction.md`, "Review of 2026-09-20".

## Why cues, and not words

`master/aesthetic_hierarchy.py` counts words and phrases. It answers a narrower
question — does the model use the same vocabulary — and a word count cannot
separate what a case is ABOUT from what the model VALUES. The top of a word
list is `library`, `entrance`, and `roadway`.

A `visual_evidence` span carries 3 things a word never can.

| Field | What it gives |
|-------|---------------|
| `image` | WHICH photograph the cue sits on |
| `valence` | good, bad, or neutral, in the model's own reading |
| `category` | 1 of 11 urban attribute types |

The `image` field is the one that changes the question. Both photographs live
in 1 trace, so a word count cannot say which one a word described. A cue names
its image, thus the test turns DIRECTIONAL.

## The price of a cue

A cue on image A wins when the label says A. Each cue therefore carries a
price: how often the photograph that holds it wins.

`win_lift` keeps the 3 rules of `_extractions.win_rates`:

1. **1 vote for each comparison.** A trace that names `trash can` 3 times for
   image A still describes 1 comparison.
2. **A key on BOTH images is dropped.** A fence in A and in B separates
   nothing.
3. **The rate is shrunk toward the base rate.** A key seen 4 times cannot take
   the top of the scale.

### Warning: read the lift, never the raw rate

The base rate differs by case. A cue sits on the winning image 59% of the time
in the schools case and 70% in the libraries case, because the model names more
cues for the photograph it prefers. Every number is a **lift** over the base
rate of its own case. A raw rate would carry that difference into every
comparison between cases.

## The unit

| `unit` | 1 key | Shared by every case? |
|--------|-------|-----------------------|
| `category_valence` | `upkeep \| good` | yes, up to 33 keys |
| `category` | `upkeep` | yes, 11 keys |
| `cue` | `chain-link fence` | the head only |
| `cue_valence` | `chain-link fence \| bad` | the head only |
| `cue_norm` | `chain link fence` | 5 facility cases, 50 keys |
| `cue_norm_valence` | `chain link fence \| bad` | the head only |

**`category_valence` is the unit that answers the question.** `category` alone
pools good greenery with bad greenery, and those 2 carry opposite prices, so
the pooled key reads near zero and the hierarchy disappears.

Measured on the version 2 primary corpus on 2026-09-22, over all 7 cases,
for the unit that answers the question:

| Unit | Keys every case scored | Cross-case Spearman of the lift |
|------|------------------------|----------------------------------|
| `category_valence` | 22 | **0.91** median (0.76 to 0.94) |

On the version 1 corpus (2026-08-28, the 9B extractor) the same unit gave
0.90 (0.68 to 0.97), `category` alone 0.46, `cue` 0.71, `cue_valence` 0.53,
and words 0.37. The ranking of the units did not change; the notebook draws
the other units on request.

`category_valence` wins on 7 cases as it won on 4. A category alone cannot
separate what a case is ABOUT from what the model VALUES: every case talks
about upkeep, and only the valence says which way. The 2 weakest case pairs
are libraries against street photography (0.76) and schools against street
photography (0.79): street photography pairs random street images and names
no facility.

16 of the 22 keys hold one sign in every case, against 13 on the version 1
corpus. The top of the ladder is `upkeep | good` +0.38, `architecture |
good` +0.37, and `signage | good` +0.36; the bottom is `upkeep | bad` -0.45,
`other | bad` -0.44, and `cleanliness | bad` -0.41.

## The words the model wrote

A `category_valence` key carries OUR vocabulary. `good upkeep` is a word we
chose, and the 3-way valence is a lossy projection: **16.3% of spans hold an
evaluative word, across 47,835 distinct strings**, and `clean`,
`well maintained`, `well lit`, `tidy`, `pristine` and `intact` all become
`good`.

The `cue_norm` unit keeps the model's word as the key.

### `normalise_cue` joins spelling variants, and nothing more

5 deterministic rules run in order. They read no corpus, thus a key does not
move when the corpus grows.

| Rule | Example |
|------|---------|
| a slash becomes `or`, an ampersand becomes `and` | `debris/dirt` -> `debris or dirt` |
| other punctuation and a hyphen become a space | `well-maintained` -> `well maintained` |
| `X is Y` becomes `Y X` | `sidewalk is clean` -> `clean sidewalk` |
| an article or a quantifier goes | `the tree` -> `tree` |
| a degree adverb goes | `very clean` -> `clean` |
| each word becomes singular | `trees` -> `tree` |

**Warning: `clean`, `well maintained` and `tidy` stay 3 keys.** They are 3
words the model chose, and the unit exists to keep them apart.

### Why a cross-case ladder cannot hold every case

A key must recur in every case to take a row. A cue span does not recur:

| | Distinct keys | Usable in all 7 cases |
|---|---|---|
| exact span | 254,361 | 42 at 25 votes, 26 at 50 |
| `cue_norm` | 225,975 | 46 at 25 votes, 26 at 50 |

Normalisation joins 11% of the strings and buys almost no shared key, thus the
limit is not the spelling. Each case names 78 to 315 words of its own, but the
words that ALL 7 share are the street furniture that every photograph holds:
13 of the 17 name a car.

**Road quality and street photography are the binding cases.** They pair
random street images and name no facility, thus they block 17 of the 23 words
that reach 6 of 7 cases. `FACILITY_CASES` names the other 5, where 50 words
recur.

### The 2 word figures

| File | Cases | Keys | The question |
|------|-------|------|--------------|
| `cue_words_per_case.png` | 7 | 78 to 315 each | What does each case value? |
| `cue_words_facility_ladder.png` | 5 | 50 | Do the words price the same? |

**The shared ladder holds.** On the version 2 corpus the cross-case Spearman
is **0.85** median (0.80 to 0.94) over 24 words, and 16 of the 24 hold one
sign in every case. On the version 1 corpus it was 0.85 (0.79 to 0.91) over
50 words. The large extractor writes fewer distinct cue strings than the 9B,
thus fewer words recur in all 5 facility cases, and the ones that do agree
as well as before. Each case names 44 to 190 words of its own.

| End | Words |
|-----|-------|
| pays most | clean sidewalk +0.33, clear signage +0.32, clean and well maintained +0.31, clean +0.28, clean street +0.28 |
| costs most | construction site -0.41, graffiti -0.38, scaffolding -0.27, fence -0.18, no parking sign -0.18 |

### Why a row is wide: the modifier

A wide row of the shared ladder says the price of that word moved with the
question. `cue_words_modifiers.png` keeps that row and writes the descriptor
along it.

| Mark | What it is |
|------|------------|
| grey line | the spread of the bare word over the cases |
| large dot | the bare word in 1 case |
| faint dot | 1 phrase that ends in the word, in 1 case |
| labelled dot | the same, with the word the model ADDED |

A reader thus sees 2 things at once: how far the bare word moved between the
cases, and which descriptor sits at each place along that line. A label
outside the grey line names a price the bare word never showed.

| Word | Pays | Costs |
|------|------|-------|
| sidewalk | clean, clear, wide | narrow, narrower |
| tree | healthy looking, well maintained, lush green | bare, sparse, dense |
| window | clear, intact, large glass | small, smaller, standard |
| storefront | clean, clear, modern looking | small, smaller |
| building | well maintained, modern looking school | older brick, older |
| scaffolding | `no` scaffolding | extensive, heavy, construction |
| fence | clean, clear, intact | chain link, green construction |

**The modifier explains the spread.** The bare `window` moves over 0.33, but
`large glass` reaches +0.29 and `smaller` falls to -0.39, and both sit outside
the grey line. The model does not price a window; it prices a LARGE one. The
same holds for `building`: every `modern` form pays and every `older` form
costs.

**`tree` says the model prices grooming, not quantity.** `healthy looking`,
`well maintained` and `lush green` all pay, and `dense` AND `sparse` both
cost.

**`car` is the control.** Its descriptors are `black`, `grey`, `blue`,
`silver` and `police` — colours and types, with no evaluative word among
them, and the whole row sits in a narrow band under 0. The model enumerates a
car; it does not judge one.

**The model reads a negation correctly, and it does so every time.** 4 words
have a `no X` form that clears the floor, and all 4 flip the sign:

| Word | bare | `no X` |
|------|------|--------|
| trash | -0.33 | +0.28 |
| scaffolding | -0.26 | +0.26 |
| litter | -0.30 | +0.24 |
| graffiti | -0.31 | +0.16 |

The extractor keeps the negation in the span, thus the cue does not read as
its own opposite. Road quality carries 9 more `no ... pothole or crack` forms
at +0.27 to +0.41.

**Warning: the head is the last word, which is a rule and not a parser.**
`power line` and `elevated train line` both give `line`, and they are not the
same thing. `spread_heads` therefore takes only a 1-word ladder key, where the
head IS the key. `modifier_variants` also drops a modifier that holds a
preposition, because `brick building with large` does not describe a window.

### The outlier table

`cue_words_outliers.csv` isolates the cues a reader should look at. It reads
EVERY case, not the facility set: a cue reverses because 1 case disagrees, and
dropping a case would hide the row that matters.

#### The scale

A raw win rate cannot be compared between cases, because the base rate runs
from 0.47 to 0.60. The table works on the log odds ratio of each case against
the base rate of its OWN case:

    theta = log( odds(win rate) / odds(base rate) )

`theta` is 0 when the cue goes with the winner as often as a cue picked at
random in that case. `lift_mean` gives the same effect on the familiar
win-share scale.

#### The 3 kinds

| `flag` | Rows | The test |
|--------|------|----------|
| `reverses` | 56 | 1 case pays and another costs, each clearing 0 alone |
| `neutral_but_decisive` | 28 | label is `neutral`, \|theta\| > 1.0, no reversal |
| `strong_and_consistent` | 47 | label is `good` or `bad`, \|theta\| > 1.0, no reversal |

A cue needs 3 cases to enter. Over 2 cases a reversal rests on 1 comparison.

**Warning: do not select on heterogeneity.** The median `i2` runs 0.66 to
0.87, thus almost every cue is heterogeneous at this sample size and
Cochran's Q separates nothing. `reverses` needs BOTH sides to clear 0 on
their own, which no amount of data can force.

**Warning: do not read `theta` on a row that reverses.** The pooled value
weights each case by its precision, thus the largest case drives it. `dark`
pools to +0.47 because road quality is large, while subway safety sits at
-4.68. Read `theta_min`, `theta_max`, `cases_pay` and `cases_cost`.

#### Street photography is the contrarian

`minority_case` names the case that stands alone against the others. It is
street photography 21 times of 35, and road quality 7 times.

| Cue | Street photography | The other cases |
|-----|--------------------|-----------------|
| `clean` | -0.70 | +1.94 to +4.26 |
| `clear` | -0.99 | +1.79 to +2.20 |
| `modern building` | -0.66 | +0.82 to +3.72 |
| `graffiti on wall` | +0.83 | -1.71 to -4.33 |
| `older brick building` | +0.53 | -1.59 to -3.53 |
| `fire escape` | +1.42 | -0.24 to -2.46 |

**Street photography inverts the upkeep standard.** Graffiti, old brick and
fire escapes PAY there, and clean, clear and modern COST. The other 6 cases
run the other way on every one of those words. Road quality gives the second
group: `dark`, `shadow` and `concrete barrier` are normal on a road and bad
at a subway entrance.

#### The model judges by words it calls neutral

28 cues carry a `neutral` label and still move the judgment hard:

| Costs | Pays |
|-------|------|
| plain -3.65, small window -3.07, chain link fence -2.20 | large arched window +2.24, architectural detail +2.06, large glass window +1.92 |
| sidewalk shed -2.05, orange barrier -1.81, heavy traffic -1.54 | wide sidewalk +1.90, landscaping +1.50, large window +1.43 |

The extractor calls these descriptive, and the judgments then turn on them.

#### The labels themselves hold

`contradicts_valence` is EMPTY, and that is a result. Over the 53 cues that
carry a `good` or a `bad` label and do not reverse, the sign of the effect
agrees with the label **53 times of 53**. The opinion the extractor wrote
about a cue matches what the cue then did.

### What each case values

The per-case figure is the one the schema ladder cannot give. Each case reads
as its own vocabulary:

| Case | Pays most | Costs most |
|------|-----------|------------|
| Subway safety | well lit, clear visibility, bright | dark, shadow, dimly lit |
| Libraries | pristine, clean facade, tidy | weathered, older, worn |
| Schools | clean facade, large glass window | construction site, chain link fence |
| Road quality | uniform, smooth, smooth asphalt | rougher, uneven surface, cracked |
| Parks and plazas | clean pavement, well kept, tidy | cluttered, dirt, construction site |
| Restaurants | outdoor seating, clear storefront | small storefront, gritty, small shop |
| Street photography | texture, architectural detail, good depth | highway, concrete barrier, guardrail |

**Street photography is the one case that does not price upkeep.** It prices
photographic qualities — texture, depth, and `classic nyc architecture`. The
other 6 cases price cleanliness and repair.

**Restaurants price size.** `small storefront`, `small shop` and `small sign`
all cost, and `large glass window` and `large window` all pay.

## The 4 figures

| File | What it shows |
|------|---------------|
| `_ladder.png` | The hierarchy itself: each key at its price, 1 dot for each case |
| `_price.png` | The scatter: price against how much the price moves |
| `_price_pairs.png` | 1 case against the mean of the others, with rho |
| `_agreement.png` | rho for every case pair |
| `cue_words_per_case.png` | what each case values, in the model's words |
| `cue_words_facility_ladder.png` | the shared word ladder, 5 facility cases |
| `cue_words_modifiers.png` | the widest rows, opened by their modifiers |
| `cue_words_outliers.csv` | the cues that reverse, or judge while called neutral |

The ladder is the headline. A short row says every case gave the key the same
price. A long row says the price moved with the question.

## The 2 gates

A cue figure is 2 steps downstream of a judgment run: the judge wrote the
trace, and the extractor read it. Thus it passes 2 gates.

1. `_canonical.verify_or_raise()` accepts the judgment runs.
2. `_extractions.load` tests whether the corpus comes from those same runs.

**A corpus that fails gate 2 writes every file with a `_stale` suffix.** That
suffix is not decoration: a stale corpus answers an older question, and a file
called `cue_hierarchy_ladder.png` must always come from the registered runs.

## The corpus (2026-09-22)

| | |
|---|---|
| Root | `data/trace_extractions_v2_gemma12b_by_qwen27b/` |
| Judge | gemma-4-12b, thinking, the registered runs of 2026-08-14 |
| Extractor | Qwen3.8-27B, schema `urban_cues_v2` |
| Traces | 77,000, of which 76,980 have an answer and carry a cue |
| Cues | 1,990,202 spans, of which 1.02 million are `visual_evidence` |
| Filters in `load` | quotable rows, no duplicate, judge answered, last decision |

The version 1 corpus (`data/trace_extractions/`, the 9B extractor,
2026-08-28) stays on disk. Its numbers are marked v1 on this page.

### To add a rater or a case

Run `extract_traces` on each registered trace run, then merge:

```bash
python -m dagspaces.pairwisevqa.cli -m pipeline=extract_traces \
    trace_extract.results_path=<the registered results.parquet> \
    trace_extract.shard_count=8 trace_extract.shard_index=0,1,2,3,4,5,6,7 \
    hydra.launcher.array_parallelism=8
python research/scripts/merge_trace_extractions.py
```

1 case costs about **7 GPU-hours**, and sharding buys wall clock, not GPU
hours. 7 cases on 1 rater is about 49 GPU-hours, and both raters about 98.

**Warning: do not launch more than about 5 monitors at a time.** `sbatch`
forwards through ssh, and a wider burst makes sshd refuse the extra
connections. The refused monitor still exits with success, and its stage job
never reaches SLURM. A 56-way multirun lost 21 of 56 jobs this way on
2026-08-28. Run 1 multirun for each case, pause about 120 seconds between
them, and count what really landed:

```bash
for n in $SWEEP/[0-9]*; do
  ls $n/.slurm_jobs/extract/*_submitted.pkl >/dev/null 2>&1 || echo "lost: $n"
done
```

`research/scripts/merge_trace_extractions.py` takes several sweep directories, thus a
re-run in its own sweep merges with the first one.

See `research/docs/langextract-trace-extraction.md` for the throughput
numbers and the shard layout.

## The other channels

This page reads 1 extraction class of 7. `trace-channels.md` reads the 4
others that carry a signal: the axis the model compares on, the reason it
abstains, what it reads INTO a photograph, and the flaws of the camera.
