# IC version 3 analysis notebooks

Three marimo notebooks read the version 3 extraction corpora. The version 3
classes follow the Integrative Complexity (IC) codebook. See
`langextract-trace-extraction.md` (section "Version 3") for the schema and
the corpora, and `ic-ingredient-extraction.md` for the codebook.

## Files

| File | Purpose |
|------|---------|
| `research/notebooks/cvpr/_ic_v3.py` | The arithmetic, the figures, and the export of the 3 notebooks |
| `research/notebooks/cvpr/master/ic_perspectives.py` | Question 1: perspectives, inferences, and dimensions against the vote |
| `research/notebooks/cvpr/master/ic_grounding.py` | Question 2: what grounds a perspective |
| `research/notebooks/cvpr/master/ic_differentiation.py` | Questions 3 and 4: pseudo-differentiation and dismissals |
| `research/scripts/export_ic_v3_figures.py` | Writes every file of the 3 notebooks for the 2 primary arms, without marimo |
| `tests/test_ic_v3.py` | 15 tests on small synthetic frames. No corpus read |

Each notebook shows the primary arm of the gemma judge by default. An arm
dropdown selects one of the 4 v3 arms. A box reads the other extractor of
the same judge (questions 1 and 2) or all 4 arms (questions 3 and 4).

## Commands

```bash
# Run a notebook headless (silent on success)
.venv-mllmsci-vllm025cu129/bin/python research/notebooks/cvpr/master/ic_perspectives.py

# Write every file for the 2 primary arms (about 2 minutes, 4 arms read)
.venv-mllmsci-vllm025cu129/bin/python research/scripts/export_ic_v3_figures.py \
    --fig-dir research/notebooks/cvpr/master/figures

# Tests
.venv-mllmsci-vllm025cu129/bin/python -m pytest tests/test_ic_v3.py -q
```

## Figure captions

Each PNG has a `.json` sidecar with the same stem. It holds `plot-title`,
`plot-caption`, `plot-latex-label`, and `plot-tags`, with a `provenance`
block. The label follows the table labels: `fig:ic-perspective-ends-<arm>`
beside `tab:ic-perspective-ends-<arm>`. The figure itself carries no caption
text. The `caption_<figure>` functions of `_ic_v3` build the caption, the
notebook shows it under the figure, and `_figmeta.save_figure` writes it.

## Paper tables (LaTeX)

The export button of each notebook writes compact LaTeX tables into
`research/notebooks/cvpr/master/outputs/`, not into `figures/`. The `latex_<table>`
functions of `_ic_v3.py` build them with the shared helper `_latex.py`. The
file name is `<tag>_<table>_v<NOTEBOOK_VERSION>.tex` (version 1.1.0). In the
labels, `<arm>` is the arm name with `-` for `_`, for example
`gemma-by-qwen27-v3`.

| File | `\label` | Content |
|------|----------|---------|
| `ic_perspectives_<arm>_ends_v1.1.0.tex` | `tab:ic-perspective-ends-<arm>` | the 3 perspectives of highest and of lowest lift in each case, with the base rate |
| `ic_perspectives_<arm>_signs_v1.1.0.tex` | `tab:ic-perspective-signs-<arm>` | the lift of each perspective in each case (3 cases or more); a bullet marks 1 sign in every case |
| `ic_grounding_<arm>_nesting_v1.1.0.tex` | `tab:ic-nesting-<arm>` | the percent of the perspective spans in, around, or on an inference or a dimension, by case |
| `ic_grounding_<arm>_strongest_nests_v1.1.0.tex` | `tab:ic-strongest-nests-<arm>` | the 10 perspectives (100 traces or more) that nest most often with 1 partner |
| `ic_grounding_arms_nesting_v1.1.0.tex` | `tab:ic-nesting-arms` | the percent of the perspective spans inside an inference clause, for the 4 arms (2 x 2) |
| `ic_differentiation_<arm>_pseudo_by_case_v1.1.0.tex` | `tab:ic-pseudo-by-case-<arm>` | the shares of question 3, by case |
| `ic_differentiation_<arm>_dismissals_v1.1.0.tex` | `tab:ic-dismissals-<arm>` | the dismissal rate, the `about` shares, and the top target, by case |
| `ic_differentiation_arms_pseudo_v1.1.0.tex` | `tab:ic-pseudo-arms` | the pooled shares of question 3 and the dismissal rate, for the 4 arms (2 x 2) |

The notebook `ic_grounding.py` reads 1 arm, thus only the script writes
`tab:ic-nesting-arms`. The notebook `ic_differentiation.py` writes
`tab:ic-pseudo-arms` when its 4-arm box is on. To write all 14 tables for
the 2 primary arms without marimo (about 2 minutes, no figure, no CSV):

```bash
.venv-mllmsci-vllm025cu129/bin/python research/scripts/export_ic_v3_figures.py --latex
```

## The data path

`_ic_v3.load_arm(arm)` calls `_extractions.load` with the paper filters:
quotable, not duplicate, judge answered, 1 decision for each trace, and the
last verdict promoted when no decision exists. Then `slim` keeps 22 columns
and adds 3:

| Column | Content |
|--------|---------|
| `tid` | `case\|pair_id`, the key of 1 trace |
| `key_name` | the normalised `name` |
| `key_target` | the normalised dismissal `target` |

The normalisation is `_cues_across.normalise_cue`. It lowers the text, makes
a hyphen a space, removes articles and degree adverbs, and makes each word
singular. It joins the spelling variants only. The loader does NOT lower the
free names (the corpus holds "NYC baseline"), thus the normalisation is
necessary.

## Question 1: perspectives and the vote

The test is `_cues_across.keyed_lift`, the directional test of a cue.

| Class | The image the span counts for | The key |
|-------|-------------------------------|---------|
| perspective | `favors` | name |
| inference | `image` | `name \| kind` |
| dimension | `image` | `name \| valence` |

The rules:

1. 1 vote for each (trace, key, image).
2. A key on both images of 1 trace is dropped. `favors = neither` does not
   vote.
3. `Same` and `NotSure` do not vote.
4. The base rate is the win rate of all the votes of the class in the case,
   before the count floor.
5. The win rate is shrunk toward the base rate by 25 pseudo votes. The lift
   is the shrunk rate less the base rate.
6. The interval is the Wilson interval of the raw rate (`X._wilson`).
7. A key needs 25 decided votes (`min_count`).

`key_summary` says whether a key holds 1 sign in every case where it clears
the floor (3 cases at least). `wins_check` tests the `wins` attribute against
the vote. `extractor_agreement` compares the lifts of the 2 extractors of 1
judge.

| File (`ic_perspectives_<arm>_`) | Content |
|---------------------------------|---------|
| `<class>_lift.csv` | every key of every case |
| `<class>_ends.csv`, `.png` | the 5 top and the 5 bottom keys of each case |
| `<class>_summary.csv` | each key over the cases |
| `<class>_consistent.png` | the keys that hold 1 sign |
| `<class>_extractor_agreement.csv`, `.png` | the 2 extractors of the judge |
| `wins_check.csv` | the share of favored images that win, by `wins` |

## Question 2: what grounds a perspective

The partners are the inference and dimension spans, with the keys of
question 1.

| Table | Rule |
|-------|------|
| `cooccurrence` | the 2 keys in the same trace; `lift` = observed / expected |
| `nesting` | 1 span inside the other, by `char_start` and `char_end` |

`expected` sums over the cases: the traces of the case that name the
perspective, times the share of the traces of the case that name the
partner. Thus a link between 2 keys of 1 case does not come only from the
case.

The 3 nesting relations:

| Relation | Rule |
|----------|------|
| `contains_perspective` | the perspective lies inside the partner clause |
| `inside_perspective` | the partner lies inside the perspective clause |
| `same_span` | the 2 spans have the same offsets |

`nest_rate` is the share of the traces that name the perspective where the
partner nests with it.

| File (`ic_grounding_<arm>_`) | Content |
|------------------------------|---------|
| `table.csv` | every pair over the floor, with the co-occurrence and the nested counts |
| `top_by_lift.csv`, `top_by_count.csv`, `top_by_nesting.csv` | the 5 partners of each class for the 15 most named perspectives |
| `nested_<relation>.csv` | the pairs that nest most often |
| `nesting_overview.csv`, `nesting.png` | the share of the perspective spans that nest, by case |
| `heatmap_<class>_cooc.png`, `heatmap_<class>_nested.png` | the perspectives against their partners |

`ic_grounding_arms_nesting_overview.csv` holds the 4 arms side by side.

## Question 3: pseudo-differentiation

`differentiation_flags` gives 1 row for each trace.

| Flag | Rule |
|------|------|
| `has_verdict` | at least 1 verdict that starts before the decision |
| `reconsiders_after` | at least 1 `uncertainty` with marker `reconsideration` after the first verdict and before the decision |
| `decision_unchanged` | the decision label equals the last verdict before it |
| `pseudo` | has_verdict, reconsiders_after, and decision_unchanged |
| `revised` | has_verdict, reconsiders_after, and not decision_unchanged |

The variants: `*_any` uses any uncertainty span; `*_first` uses the first
verdict; `revised_direction` asks whether the winner (A, B, or undecided)
changes. The task said "after the first verdict". We also stop the window at
the decision, because a span after the decision cannot change it.

| File (`ic_differentiation_<arm>_`) | Content |
|------------------------------------|---------|
| `pseudo_by_case.csv`, `.png` | the shares by case |

`ic_differentiation_arms_pooled.csv` and `_by_case.csv` hold the 4 arms.
The 2 figures `_revised_given_recon.png` and `_share_pseudo.png` show them.

## Question 4: what the model discards

`dismissal_frame` gives 1 row for each dismissal. A dismissal matches a
perspective, a dimension, or an inference of the same trace by 1 of 2
rules:

1. Name: the normalised name equals the normalised target.
2. Text: the lowered span text holds the lowered target or the normalised
   target (3 characters at least).

`matched_elsewhere` leaves out a span that overlaps the dismissal clause.
`named_before` keeps only a span that ends before the dismissal starts.

| File (`ic_differentiation_<arm>_`) | Content |
|------------------------------------|---------|
| `dismissal_about.csv`, `.png` | the share of each `about`, by case |
| `dismissal_targets.csv`, `.png` | the 10 most frequent targets of each case (3 at least) |
| `dismissal_match.csv` | the match shares, by case |
| `dismissal_rate.csv` | the dismissals for each 100 traces |

`ic_differentiation_arms_dismissal_pooled.csv` and `_dismissal_about.csv`
hold the 4 arms.

## Warnings

**Warning: the free names are the model's own phrases.** `maintenance`,
`upkeep`, and `well maintained` stay 3 keys. Read a key with its count, and
read the families of keys together.

**Warning: the perspective, inference, and uncertainty rates are facts
about the extractor.** The Qwen3.8-27B extractor marks about 2x the
perspectives and the uncertainty spans of the Gemma-4-31B extractor on the
same traces. Read every count of these classes beside the other extractor.

**Warning: the base rate of a perspective is high** (0.79 to 0.95 on the
gemma primary arm). The model names a standard for the image it then picks.
A negative lift is a standard that the model names and then overrules.

**Warning: the `wins` attribute reads the decision.** On the gemma primary
arm a `wins = yes` perspective favors the winner in 99.9% of its rows. Thus
`wins` is not independent of the vote. Do not use it as a predictor.

**Warning: an inference has no valence.** `maintenance | upkeep` pools
"well maintained" and "poorly maintained".

**Warning: the Gemma-4-31B qwen arm reads a trace in chunks of 6,000
characters.** A chunk that states an answer writes a decision. The loader
keeps the last decision and drops the others, thus an earlier chunk's answer
is not a verdict in question 3. The verdict counts of that arm are low by an
unknown amount.

**Warning: a premise target is the facility of the prompt.** "Library
building" or "park or plaza" match a dimension often because the model also
describes the facility, not because it reasons about it.

**Warning: this is an association, not a cause.** The model narrates while
it decides, thus a span can follow the judgment.

## Headline numbers (2026-09-30)

From `research/scripts/export_ic_v3_figures.py`, `min_count` 25. "Gemma" is
`gemma_by_qwen27_v3`; "qwen" is `qwen_by_gemma31_v3`.

### Question 1

| Number | Gemma | Qwen |
|--------|-------|------|
| Perspective base rate, range over the cases | 0.79 to 0.95 | 0.86 to 0.93 |
| Perspectives in 3 cases or more | 20 | 16 |
| ... that hold 1 sign | 9 | 4 |
| Favored image wins, `wins = yes` | 0.999 | 0.981 |
| Favored image wins, `wins = no` | 0.63 | 0.76 |
| Perspective lift, Spearman with the control extractor, median case | 0.72 | 0.66 |
| Dimension lift, Spearman with the control extractor, median case | 0.96 | 0.97 |

`functional` is the most negative perspective under both judges: about
-0.33 (gemma, 6 cases) and -0.36 (qwen, 4 cases). `standard` (gemma, all
7 cases) and `modern` (qwen, all 7 cases) cost in every case. The dimension
keys follow their valence: 22 of 30 (gemma) and 28 of 40 (qwen) hold 1 sign.

The top 5 perspectives by lift, gemma:

| Case | Top 5 (lift) |
|------|--------------|
| Schools | resource 0.083, aesthetic 0.075, high end 0.075, grand 0.069, upscale 0.066 |
| Restaurants | appeal 0.134, quality 0.121, inviting 0.118, prestige 0.112, aesthetic sentiment 0.110 |
| Street photography | appeal 0.153, street photography criteria 0.142, visual interest 0.141, street photography standard 0.140, photogenic 0.138 |

The bottom of schools (gemma): functional -0.40, classic -0.24, traditional
-0.24, historic -0.19, standard -0.08.

### Question 2

| Number | Gemma | Qwen |
|--------|-------|------|
| Perspective spans inside an inference clause, range over the cases | 68% to 88% | 34% to 69% |
| ... inside a dimension clause | 5% to 9% | 3% to 10% |
| An inference or dimension inside a perspective clause | 0.3% or less | 1.4% or less |
| Co-occurrence lift, 5% to 95% of the pairs | 0.83 to 1.75 | 0.88 to 1.61 |

The perspective is the mode word inside an inference, as the schema teaches.
The strongest nests (gemma): `maintenance` in `maintenance | upkeep` (88% of
its traces), `modern` in `modernity | quality` (63%), `standard` in
`standardness | quality` (57%). The control arms give 36% to 73%
(gemma by Gemma31) and 75% to 91% (qwen by Qwen27), thus the share is a
fact about the extractor.

### Question 3

| Number | Gemma | Qwen |
|--------|-------|------|
| Traces with a decision | 6,941 | 6,754 |
| Share with a verdict | 0.715 | 0.940 |
| Share pseudo | 0.618 | 0.717 |
| Share revised | 0.040 | 0.118 |
| Decision changes, of the traces that reconsider | 0.060 | 0.141 |
| ... against the first verdict | 0.108 | 0.358 |
| ... the winner changes | 0.040 | 0.048 |
| Mean reconsiderations, pseudo / revised | 3.8 / 5.6 | 5.7 / 5.4 |

The controls: pseudo 0.437 (gemma by Gemma31) and 0.776 (qwen by Qwen27).
In every arm, of the traces that reconsider, the winner changes in 3% to 5%.
The rest of a revision is a change of strength (More to MuchMore).

### Question 4

| Number | Gemma | Qwen |
|--------|-------|------|
| Dismissals for each 100 traces | 20 | 113 |
| Share `premise` | 0.89 | 0.73 |
| Share `cue` | 0.10 | 0.22 |
| Share `perspective` | 0.009 | 0.044 |
| Target named by a perspective of the trace | 0.023 | 0.081 |
| Target matched by any named span | 0.318 | 0.438 |
| Target named before the dismissal | 0.209 | 0.349 |

The top targets are the facility of the prompt. Gemma: `park or public
plaza` (288), `park or plaza` (175), `library building` (57), `subway
entrance` (56). Qwen: `park` (465), `park or public plaza` (449), `subway
entrance` (324), `subway station entrance` (199), `restaurant` (143). The
dismissal rate differs about 3x to 4x between the 2 extractors of a judge (gemma
traces: 20 and 64; qwen traces: 113 and 30).
