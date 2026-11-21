# The 4 other channels of the reasoning trace

`cue-hierarchy.md` reads 1 extraction class, `visual_evidence`. That class is
46% of the corpus. This page reads the other 4 classes that carry a signal.

| Piece | Where |
|-------|-------|
| Module | `notebooks/cvpr/_trace_channels.py` |
| Notebook | `notebooks/cvpr/master/trace_channels.py` |
| Figures and tables | `notebooks/cvpr/master/figures/trace_channels_*` |
| Source | The `urban_cues_v2` primary corpus, `data/trace_extractions_v2_gemma12b_by_qwen27b/` (gemma-4-12b traces read by Qwen3.8-27B). Every number on this page comes from the rerun of 2026-09-22 (`scripts/export_trace_figures.py`). | Section 2 reads the 4 version 3 arms (2026-09-30).

```bash
.venv-mllmsci-vllm025cu129/bin/marimo edit \
    notebooks/cvpr/master/trace_channels.py
```

**Warning: the abstention and inference rates depend on the extractor.** The
review of 2026-09-20 read the same gemma-4-12b traces with both extractors.
After the duplicate rows and the unanswered traces leave, `uncertainty`
differed 1.6x in the median case and 2.9x at most, `inference` 1.9x, and
`image_artifact` 1.8x. `visual_evidence`, `comparison`, and `decision` held
within 1.3x in every case. Thus every rate on this page is a fact about the
pair (judge, extractor), and the paper must report it from both extractors
or not at all. The notebook shows the table from `_extractions.rate_stability`.

**The corpus changed on 2026-09-22.** The page read the version 1 corpus
(qwen3.5-9b as the extractor) until then. It now reads the version 2 primary
corpus, and `load` drops the duplicate spans, the unanswered traces, and
every decision but the last. The largest change is in the axes: the
Qwen3.8-27B extractor sets `direction` in agreement with the final label on
96% of the decided comparisons, where the 9B reached 58%. The hedge marker
no longer costs agreement. Read the sections below with the new extractor
in mind.

## What the corpus holds

76,980 traces, 10,992 to 11,000 for each of the 7 cases, after the
unanswered traces leave. 1,990,202 spans, thus 25.9 spans for each trace.

| Channel | Class | Spans | Share |
|---------|-------|-------|-------|
| Cue (`cue-hierarchy.md`) | `visual_evidence` | 1,022,669 | 51.4% |
| Inference | `inference` | 402,508 | 20.2% |
| Abstention | `uncertainty` | 246,797 | 12.4% |
| Axes | `comparison` | 193,528 | 9.7% |
| Decision | `decision` | 74,006 | 3.7% |
| Artifact | `image_artifact` | 25,804 | 1.3% |
| Person | `person_reference` | 24,890 | 1.3% |

**Warning: `pair_id` does not name a trace.** The 7 cases draw from 1 pair
table, thus the same `pair_id` appears once in each case. A group by `pair_id`
alone pools 7 traces into 1, and it gave 24,491 traces where there are 76,271.
`trace_id()` builds the key that is correct.

## 1. The axes the model compares on

A `comparison` span carries a free phrase, `dimension`, and a `direction` that
names the winning image. Thus the model writes the AXIS of the comparison in
its own words. There are 18,520 distinct phrases over 193,528 comparisons,
127,315 of them on a decided pair, and they are a second vocabulary beside
the cue words. A cue says what the model
saw; an axis says what it compared on.

The top phrases: `condition`, `maintenance`, `type`, `safety`, `cleanliness`,
`appearance`, `appeal`, `style`, `visibility`, `smoothness`, `standardness`,
`vibe`, `visual interest`.

### The measure

`agree` is the share of the DECIDED comparisons where the final label names
the image the axis named. Over every axis it is 0.963 on the version 2
corpus. On the version 1 corpus, read by the 9B, it was 0.584: the small
extractor set `direction` almost at chance, and the large one sets it with
the label.

**Warning: `agree` measures polarity first, and binding second.** `direction`
names the image with MORE of the quality, not the image that wins. Thus a
negative axis word runs the other way:

| Axis | n | `agree` | `bind` |
|------|---|---------|--------|
| `smoothness and maintenance` | 541 | 1.000 | 1.000 |
| `overall quality` | 2,431 | 1.000 | 0.999 |
| `preference` | 1,338 | 0.999 | 0.999 |
| `safety` | 4,591 | 0.999 | 0.998 |
| `maintenance` | 11,808 | 0.999 | 0.997 |
| `condition` | 8,215 | 0.998 | 0.996 |
| ... | | | |
| `openness` | 611 | 0.851 | 0.702 |
| `greenery` | 455 | 0.780 | 0.560 |
| `clutter` | 658 | 0.649 | 0.298 |
| `wear` | 1,330 | 0.514 | 0.027 |
| `wear and tear` | 470 | 0.211 | 0.579 |

Read `bind`, which is `|agree - 0.5| * 2`, for the strength, and `polarity`
for the sign. `wear` and `maintenance` both bind the vote hard; they point
opposite ways because one word is negative.

An axis whose Wilson interval covers chance takes `polarity = "none"`. Of the
37 axes over a 400-comparison floor, 35 bind above 0.4, and 1 (`wear`) names
no side at all.

### The interesting band

The rows near chance are the ones to read first: the model names the axis and
then does not decide on it. `wear` 0.514, `clutter` 0.649, `greenery` 0.780.
`wear and tear` at 0.211 is a negative axis read the right way round: the
image with more wear loses. The model compares on clutter and greenery and
decides on maintenance.

### The hedge marker

`hedged` is a second free field. The model hedges on 21.7% of its
comparisons. On the version 2 corpus the marker costs nothing: a hedged
comparison agrees with the final label as often as a plain one, and
`hedge_cost`, which is `agree_plain - agree_hedged`, sits between -0.062
(road quality) and +0.013 (schools) in every case. On the version 1 corpus
it read +0.13 to +0.27 in 6 of 7 cases. That gap was the 9B extractor's
`direction`, not the model's hedging: when the direction is set right, a
hedge does not move the vote.

## 2. Where an abstention comes from

**Measured on 2026-09-30, on the 4 version 3 arms.** `_trace_channels`
1.1.0, `abstention_across_arms`. Each arm holds the same 1,000 traces for
each case of its judge, after `load` drops the unanswered traces. The
section before this date read the version 2 primary corpus (AUC 0.59 to
0.88) and the version 1 corpus (AUC 0.74 to 0.94).

The count of the `uncertainty` spans in a trace is a score. The AUC asks
whether it separates an abstention (`NotSure`) from a decision. A case with
fewer than 25 abstentions gives no AUC.

### The AUC of the total count, case x arm

| Case | Abstentions, gemma / qwen | gemma by Qwen27 (primary) | gemma by Gemma31 | qwen by Gemma31 (primary) | qwen by Qwen27 |
|------|---------------------------|---------------------------|------------------|---------------------------|----------------|
| Subway safety | 84 / 24 | 0.930 | 0.849 | (0.789) | (0.790) |
| Libraries | 91 / 17 | 0.779 | 0.740 | (0.726) | (0.710) |
| Restaurants | 94 / 35 | 0.739 | 0.758 | 0.792 | 0.794 |
| Parks and plazas | 168 / 14 | 0.678 | 0.599 | (0.744) | (0.716) |
| Schools | 110 / 49 | 0.622 | 0.680 | 0.851 | 0.800 |
| Road quality | 6 / 2 | — | — | — | — |
| Street photography | 0 / 0 | — | — | — | — |

A value in brackets is under the floor of 25. It comes from a floor of 10
(`min_abstain=10`), thus it is noisy. In the cases that abstain at all, the
qwen3.5-9b judge abstains on 0.2% to 5.1% of the traces, and the
gemma-4-12b judge on 0.6% to 16.8%.

### The AUC of each marker and reason, on the primary arms

Each AUC reads 1 count alone. `core` is the total less the
`reconsideration` spans.

| Case | all | core | qualifier | contrastive | explicit_doubt | reconsideration | cannot_see | looks_equal |
|------|-----|------|-----------|-------------|----------------|-----------------|------------|-------------|
| gemma: Subway safety | 0.930 | 0.936 | 0.885 | 0.546 | 0.946 | 0.897 | 0.952 | 0.331 |
| gemma: Libraries | 0.779 | 0.818 | 0.738 | 0.579 | 0.843 | 0.711 | 0.892 | 0.339 |
| gemma: Restaurants | 0.739 | 0.782 | 0.700 | 0.567 | 0.829 | 0.662 | 0.868 | 0.440 |
| gemma: Parks and plazas | 0.678 | 0.700 | 0.626 | 0.576 | 0.731 | 0.624 | 0.749 | 0.414 |
| gemma: Schools | 0.622 | 0.672 | 0.519 | 0.511 | 0.772 | 0.547 | 0.813 | 0.412 |
| qwen: Schools | 0.851 | 0.892 | 0.711 | 0.723 | 0.910 | 0.678 | 0.944 | 0.669 |
| qwen: Restaurants | 0.792 | 0.829 | 0.745 | 0.786 | 0.839 | 0.665 | 0.856 | 0.527 |

### The paper tables

`_trace_channels` 1.2.0 writes the 2 tables above as LaTeX. The export
button of `master/trace_channels.py` writes them into `master/outputs/`, not
into `master/figures/`. The version in the name is the `NOTEBOOK_VERSION`.

| File | `\label` | What it shows |
|------|----------|---------------|
| `trace_channels_abstention_auc_v1.3.0.tex` | `tab:abstention-auc-arms` | The AUC of the total count, case x arm, with the abstentions of each judge |
| `trace_channels_abstention_markers_v1.3.0.tex` | `tab:abstention-auc-markers` | The AUC of each marker and reason, on the 2 primary arms |

A cell under the floor of 25 abstentions is a dash. The tables do not print
the bracketed values of the floor of 10. To write the tables again without
the notebook, use this command. It reads the 4 arms (about a minute) and it
writes no figure:

```bash
.venv-mllmsci-vllm025cu129/bin/python scripts/export_trace_figures.py --only channels --latex
```

### The reading

- **The result reproduces.** The total count separates an abstention in
  every case that abstains enough, in all 4 arms: AUC 0.60 to 0.93. The
  range sits between the version 2 range (0.59 to 0.88) and the version 1
  range (0.74 to 0.94). Subway safety is again the strongest gemma case.
- **The order of the cases mostly holds across the 2 extractors of the
  gemma judge.** Subway safety is first under both. Libraries and
  restaurants swap, and parks and schools swap. The AUC moves by 0.08 at
  most when the extractor changes, but the count itself moves about 2x (the
  median trace holds 10 against 6 spans). Thus the AUC holds better than
  the rate of the class.
- **The qwen judge gives the same result**, on 2 cases only. Its 2
  extractors agree within 0.06.
- **`explicit_doubt` and `cannot_see` separate better than the total**, in
  every primary case (`cannot_see` 0.75 to 0.95). The model that says "I
  cannot tell" abstains.
- **`reconsideration` dilutes the total.** Version 3 marks the "Wait, let me
  look again" span, and that span is the most frequent marker (50,440 of
  99,699 spans on the primary gemma arm). It separates worse than the total
  in every primary case, thus `core` is above `all` in all 7 rows. Compare
  `core` with version 1 and version 2.
- **`looks_equal` goes with a decision** on the gemma arms (AUC 0.26 to
  0.44). The judge that says the 2 photographs look equal answers `Same`,
  not `NotSure`.

### The dose curve

Version 3 marks about 10x the spans of version 1, thus the bins are wider:
0-2, 3-5, 6-10, 11-20, 21-40, 41+. On the primary gemma arm the curve is flat
or falls below 11 spans, and it climbs above 20: 0.11 to 0.28 at 21-40 and
0.20 to 0.44 at 41+. The count works as a threshold more than as a dose.

The figure of the 2 x 2 (`trace_channels_abstention_arms.png`) bins the
count by its quintile INSIDE each arm, because the 2 extractors mark
different counts. In every arm the rate stays below 0.20 to the 4th quintile
and jumps at the 5th: 0.20 to 0.30 for the gemma judge and 0.10 to 0.20 for the
qwen judge.

**Warning: the left end of the curve is a small group.** 28 traces of the
primary gemma arm abstain with 0 to 2 uncertainty spans. They are short
traces: a median of 14 spans in all, against 26 for a deciding trace with so
few uncertainty spans. The judge answers `NotSure` and gives no reason.

**Warning: street photography holds NO abstention,** in both judges. Road
quality holds 6 (gemma) and 2 (qwen). Check an arm before you pool it into an
abstention figure.

### The reason field

The table below is from the version 2 primary corpus (2026-09-22).
`reason` types the span, and `other` takes 28% to 87% of the spans. Read the
3 named reasons against each other, never against the whole.

| Case | `cannot_see` | `out_of_view` | `looks_equal` | spans/trace |
|------|-------------|---------------|---------------|-------------|
| Restaurants | 0.543 | 0.010 | 0.100 | 3.3 |
| Parks and plazas | 0.363 | 0.015 | 0.211 | 2.3 |
| Schools | 0.343 | 0.010 | 0.278 | 5.4 |
| Libraries | 0.313 | 0.002 | 0.403 | 3.2 |
| Subway safety | 0.297 | 0.010 | 0.409 | 4.7 |
| Road quality | 0.165 | 0.010 | 0.309 | 1.9 |
| Street photography | 0.026 | 0.000 | 0.101 | 1.6 |

## 3. What the model reads INTO a photograph

An `inference` span is the model past the pixels. Both `inference` and
`image_artifact` name an image, thus both take the same directional test as a
cue: 1 vote for each comparison, a key on both images is dropped, and the rate
is shrunk toward the base rate of the case.

| `kind` | Cases | n | Mean lift | Range | Pays / costs |
|--------|-------|---|-----------|-------|--------------|
| `wealth` | 7 | 5,556 | **+0.297** | +0.059 to +0.504 | 7 / 0 |
| `quality` | 7 | 12,286 | +0.088 | -0.068 to +0.290 | 6 / 1 |
| `class` | 6 | 6,572 | -0.013 | -0.138 to +0.073 | 4 / 2 |
| `upkeep` | 7 | 12,107 | -0.014 | -0.198 to +0.130 | 3 / 4 |
| `demographic` | 7 | 1,074 | -0.073 | -0.257 to +0.244 | 1 / 6 |
| `crime` | 1 | 390 | -0.126 | — | 0 / 1 |
| `safety` | 4 | 2,221 | -0.151 | -0.286 to -0.036 | 0 / 4 |
| `other` | 7 | 12,712 | -0.234 | -0.418 to +0.001 | 1 / 6 |

**A wealth inference is the most decisive move in the trace.** It pays in 7 of
7 cases, and the lift is larger than any cue word carries. `class` pays in 5
of 6. The 4 risk kinds — wealth, class, crime, demographic — are named in
`RISK_KINDS`.

An `upkeep` or a `safety` inference COSTS the image it sits on. The model
writes those when it finds a fault.

**Warning: 45% of `kind` is `other`, and the model wrote that word itself.**
The vocabulary is not stale. `vocabulary_report` in `_extractions` proves it.

### `kind` names the topic, never the direction

`wealth` is the same key whether the model wrote `upscale` or `lower-income
area`. Thus a lift on `kind` alone cannot say WHICH way the inference runs,
and the +0.354 above is a blend. 2 readings fix that.

**First, drop the restatements.** A span that names the other photograph or
the label restates the vote and infers nothing. `much less rather eat at A
than B` is the 2nd most frequent `wealth` span of the corpus.
`inference_key()` strips an `image A` reference and drops a restatement. It
removes 0.0% of `class` and 4.6% of `wealth`.

**Then read the pole.** `_INFERENCE_HIGH` and `_INFERENCE_LOW` are 2 short
hand-built lists of the words the model repeats. A span that matches neither
takes `none`, and one that matches both takes `mixed`.

| `kind` | high | low | mixed | none | determined | share high |
|--------|------|-----|-------|------|-----------|------------|
| `wealth` | 5,552 | 191 | 322 | 4,008 | 0.570 | **0.967** |
| `class` | 5,852 | 3,347 | 431 | 7,594 | 0.534 | 0.636 |
| `quality` | 2,938 | 2,793 | 62 | 45,603 | 0.112 | 0.513 |
| `upkeep` | 309 | 3,844 | 7 | 43,103 | 0.088 | 0.074 |
| `safety` | 35 | 706 | 3 | 34,657 | 0.021 | 0.047 |
| `demographic` | 7 | 64 | 3 | 1,328 | 0.051 | 0.099 |
| `crime` | 0 | 4 | 0 | 450 | 0.009 | 0.000 |

**The model has almost no vocabulary for inferring poverty.** 94.8% of the
wealth inferences it can be read on run UP. Not one low-pole wealth phrase
reaches 20 votes.

A negator in front of a pole word turns it around: `less upscale` counts LOW
and `no visible crime` counts HIGH.

**Warning: the flip fires only when a negator sits in the 2 tokens BEFORE the
pole word.** A blanket search over the span would break `lower income area`,
where `lower` is part of the pole word, and `less safe due to poor
visibility`, where `less` belongs to `safe` and not to `poor`. The narrow
window misses a long-range negation such as `less likely to have crime`.

The pole ladder separates what the kind ladder blends:

| Key | Cases | n | Mean lift | Pays / costs |
|-----|-------|---|-----------|--------------|
| `wealth \| high` | 6 | 2,703 | **+0.436** | 6 / 0 |
| `wealth \| low` | 1 | 74 | -0.302 | 0 / 1 |
| `class \| high` | 6 | 2,741 | +0.429 | 6 / 0 |
| `class \| low` | 6 | 1,314 | -0.280 | 0 / 6 |
| `quality \| high` | 5 | 1,800 | +0.340 | 5 / 0 |
| `quality \| low` | 7 | 1,522 | -0.230 | 0 / 7 |
| `upkeep \| high` | 3 | 214 | +0.284 | 3 / 0 |
| `upkeep \| low` | 7 | 2,514 | -0.277 | 0 / 7 |

**Warning: `public school` sits on the LOW list.** The model uses it as its
own contrast to `private school`, and 4 of the 5 most frequent low-pole
`class` spans are `standard public school` and its variants. It is the
model's class vocabulary, not a claim about a school.

### The words themselves

`inference_words()` gives the phrase back with a price on it. The lift is
POOLED over the cases, not measured inside 1 of them: an inference span is a
clause, not a noun phrase, and 302,784 spans carry 154,794 distinct keys, so
almost nothing clears a per-case floor. Each vote instead carries the base
rate of its OWN case, and the lift is the mean of `won - base`, shrunk
toward 0.

**Warning: do not take the base over (case, kind).** 97% of the wealth
inferences run 1 way, thus a per-kind base asks "which wealth word pays more
than another wealth word" and the whole panel collapses to 0.

| `kind` | pays | costs |
|--------|------|-------|
| `wealth` | `high end` +0.40, `upscale` +0.30, `well funded` +0.23 | none clears the floor |
| `class` | `high end` +0.42, `prestigious` +0.40, `premium` +0.32 | `standard public school` -0.38, `gritty` -0.30 |
| `safety` | `safer` +0.52, `eye on street` +0.42, `well lit` +0.35 | `less safe` -0.40, `secluded` -0.34, `sketchy` -0.25 |
| `upkeep` | `better maintained` +0.44, `pristine` +0.42 | `older` -0.41, `neglected` -0.38 |
| `other` | `classic street photography setting` +0.32 | `utilitarian` -0.42, `industrial` -0.36 |

The word view also validates the pole lists: of the 52 word keys they
determine, **51 agree with the measured sign**. The 1 exception is
`gritty urban feel` at +0.001.

### The sentences behind the numbers

`excerpts()` quotes the model. It gives 4 sentences for each case and each
inference kind, and it names the price of the phrase, the photograph the span
sits on, and whether that photograph won.

| Piece | How it works |
|-------|--------------|
| Which rows | The phrases whose price is furthest from 0, half paying and half costing. Inside 1 phrase the span is drawn with seed 777. |
| The anchor | A search for the span text, and the stored offsets as the fallback. The offsets land exactly on 92% of the spans; the rest drift by a quote mark the extractor normalised. 98% of the quoted rows anchor exactly, and `offsets_exact` says which. |
| The window | The sentence that holds the span. A sentence shorter than 90 characters takes the sentence before it as well, because `It looks very high-end.` names neither the photograph nor the thing. |
| The source | `model_reasoning` in the run named by `source_results_path`. |

What the model wrote:

> Image A has an AC unit in a window. - Image B shows a Range Rover and an
> Audi, suggesting a **wealthy neighborhood**. *(schools, wealth, +0.16)*

> Image A looks much cleaner, better lit, and more modern. It feels like a
> **high-end** area. *(subway safety, class, +0.42)*

> Is there anything in Image A that makes it notably less safe? It's a bit
> more **gritty** but standard NYC. *(subway safety, class, -0.30)*

**Warning: an excerpt needs the source run on disk.** `trace_texts` skips a
run it cannot read, thus a case can be absent from the table. It is not a
null result. The whole corpus of traces is about 230 MB of text, so
`trace_texts` reads only the pairs the chosen rows name.

160 rows over 44 of the 49 (case, kind) cells.

## Where a rare kind goes

A rare kind can pass this whole page without a row and read as absent.
`inference_coverage()` says where each one survives.

| `kind` | Risk | Spans | Cases | On the ladder | Pole rows | Word keys | Word floor |
|--------|------|-------|-------|---------------|-----------|-----------|------------|
| `wealth` | yes | 10,073 | 7 | 7 | 8 | 10 | 20 |
| `class` | yes | 17,224 | 7 | 6 | 12 | 25 | 20 |
| `crime` | yes | **454** | 4 | **1** | 1 | 4 | **8** |
| `demographic` | yes | 1,402 | 7 | 4 | 0 | 6 | **8** |
| `quality` | no | 51,396 | 7 | 7 | 12 | 68 | 20 |
| `safety` | no | 35,401 | 7 | 4 | 2 | 34 | 20 |
| `upkeep` | no | 47,263 | 7 | 7 | 10 | 114 | 20 |
| `other` | no | 134,820 | 7 | 7 | 11 | 251 | 20 |

**`crime` is 454 spans of 302,784, and 440 of them sit in subway safety.** It
holds 1 dot on the kind ladder: subway safety, 165 decided comparisons, a raw
win rate of 0.218 against a base of 0.329, thus a lift of -0.096. A
photograph the model names crime about wins 1 time in 5.

The phrasing also splits — `high crime area` 23, `higher crime rate` 22,
`higher crime area` 22, `higher crime` 11 — so a floor of 20 gives it 0 word
keys. `RARE_KINDS` therefore takes `RARE_MIN_COUNT = 8`. The shrink does the
work the floor was doing: a key at 8 votes is pulled most of the way to 0 by
`prior_strength`, thus a low floor adds rows without adding claims.

| `crime` phrase | n | Lift |
|----------------|---|------|
| `higher crime` | 8 | -0.072 |
| `higher crime rate` | 12 | -0.083 |
| `higher crime area` | 10 | -0.131 |
| `high crime area` | 15 | -0.136 |

**What the model reads crime off.** Every photograph below LOST:

> The windows have **bars**, which might suggest a **higher-crime area** or a
> need for extra security. *(schools)*

> Image A has a lot of **graffiti**, which can sometimes be associated with
> **higher crime areas**, but not necessarily. *(subway safety)*

> **Fences** can sometimes signal a **high-crime area** or just a way to keep
> people off the tracks. *(subway safety)*

> Image B shows a **police van and an officer**, which might imply a **higher
> crime rate** in the area. *(subway safety)*

**Warning: a lift belongs to the table it sits in.** `keyed_lift` takes the
base rate over the keys that cleared ITS floor, thus the same span reads
-0.096 on the kind ladder and +0.013 on the pole ladder. The pole table of
subway safety holds mostly low-pole keys, so its base is lower. Never carry a
lift from one figure to another. `win_rate` and `base_rate` ride in every
summary CSV for this reason.

## 4. Does the camera move the judgment?

An `image_artifact` span names a flaw of the photograph, not of the place.

| `kind` | Cases | n | Mean lift | Range | Pays / costs |
|--------|-------|---|-----------|-------|--------------|
| `weather` | 4 | 753 | +0.207 | +0.045 to +0.308 | 4 / 0 |
| `time_of_day` | 3 | 686 | +0.148 | +0.133 to +0.169 | 3 / 0 |
| `blur` | 6 | 698 | +0.083 | -0.036 to +0.191 | 5 / 1 |
| `angle` | 7 | 1,796 | -0.017 | -0.100 to +0.039 | 4 / 3 |
| `other` | 7 | 2,290 | -0.023 | -0.154 to +0.051 | 3 / 4 |
| `occlusion` | 7 | 2,332 | -0.080 | -0.121 to -0.007 | 0 / 7 |

**A blocked view costs an image in 7 of 7 cases.** Weather and the time of day
PAY, and blur pays in 5 of 6. Thus the judgment tracks how the photograph was
taken, not only what stands in it. This is a validity threat with a number
attached: 71,401 spans, 0.22 to 2.53 for each trace.

## What each figure holds

Each PNG has a `.json` sidecar with the same stem (for example
`trace_channels_axes.json`). It holds `plot-title`, `plot-caption`,
`plot-latex-label` (`fig:axes-binding`, `fig:abstention-dose-arms`, and so
on), and `plot-tags`, with a `provenance` block. The figure itself carries
no caption text. The `caption_<figure>` functions of `_trace_channels`
build the caption, the notebook shows it under the figure, and
`_figmeta.save_figure` writes it.

| File | What it shows |
|------|---------------|
| `trace_channels_axes.png` | Each axis at its agreement, with the chance band |
| `trace_channels_axes_hedge.png` | Plain against hedged, for each axis |
| `trace_channels_abstention.png` | The dose curve, 1 line for each case |
| `trace_channels_abstention_reasons.png` | The reason mix of each case |
| `trace_channels_abstention_markers.csv` | The AUC of each marker and reason, for each case, on the loaded corpus |
| `trace_channels_abstention_arms.png` | The dose curve of each judge under its 2 extractors, by quintile |
| `trace_channels_abstention_arms.csv` | The abstention table of each of the 4 version 3 arms |
| `trace_channels_inference.png` | The price of each inference kind |
| `trace_channels_inference_pole.png` | The price by DIRECTION, high against low |
| `trace_channels_inference_words.png` | The phrases themselves, 1 panel for each kind |
| `trace_channels_inference_excerpts.csv` | 4 quoted sentences for each case and kind |
| `trace_channels_inference_coverage.csv` | Where each kind survives, and which floor removed it |
| `trace_channels_artifact.png` | The price of each photographic flaw |

## Warnings that apply to every channel

- **This is an association, not a cause.** The model narrates while it
  decides, thus a span can follow the judgment rather than drive it. Read a
  lift as "what goes with a win".
- **A case that never wrote a channel is absent, not null.** `coverage()`
  names what is present, and every figure reports it.
- **The same 2 gates as a cue figure.** `_canonical.verify_or_raise()` accepts
  the judgment runs, and `_extractions.load` tests whether the extraction
  corpus comes from those same runs. A corpus that fails writes every file
  with a `_stale` suffix.

## See also

- `cue-hierarchy.md` — the `visual_evidence` class, the aesthetic hierarchy
- `langextract-trace-extraction.md` — the extraction stage and the schema
- `aesthetic-hierarchy.md` — the word unit, before the cue unit
