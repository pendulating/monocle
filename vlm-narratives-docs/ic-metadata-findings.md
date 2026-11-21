# The span attributes of version 3: findings

Date: 2026-09-30.

This page reads the attributes of the version 3 spans that the earlier
analyses did not use: `valence` and `image` of a `dimension`, `kind` of an
`inference`, `source` and `favors` of a `perspective`, `direction` and
`hedged` of a `comparison`, `about` of a `dismissal`. It joins the spans to
the images of each pair and to the proxy arm.

| Piece | Where |
|-------|-------|
| Module | `notebooks/cvpr/_ic_metadata.py` |
| Notebook, questions 1 and 2 | `notebooks/cvpr/master/aesthetic_consistency.py` |
| Notebook, question 3 | `notebooks/cvpr/master/reasoning_vs_proxy.py` |
| Tests | `tests/test_ic_metadata.py` (10 tests, synthetic frames) |
| Tables and figures | `notebooks/cvpr/master/outputs/explore/` |

### The figure captions

Each PNG in `master/outputs/explore/` has a `.json` sidecar with the same
stem. It holds `plot-title`, `plot-caption`, `plot-latex-label`
(`fig:ic-meta-valence-grid-<arm>`, `fig:ic-meta-partner-effect-<arm>`,
`fig:ic-meta-proxy-contrasts-<arm>`), and `plot-tags`. The figure itself
carries no caption text. The `caption_<figure>` functions of `_ic_metadata`
build the caption, and `_figmeta.save_figure` writes it.

### The paper tables (LaTeX)

The module writes 5 compact LaTeX tables (`_ic_metadata` 1.1.0). Each
function takes `{arm: frame}`. The export button of a notebook writes the
tables of its arm as `<tag>_<table>_v<NOTEBOOK_VERSION>.tex`, next to the
CSV files. The 2-arm copies for the 2 primary arms use the tag `<notebook>_primary`.

| Table | File in `master/outputs/explore/` | `\label` | Finding |
|-------|-----------------------------------|----------|---------|
| `latex_valence_flip` | `aesthetic_consistency_primary_valence_flip_v1.1.0.tex` | `tab:ic-meta-valence-flip` | 1 |
| `latex_same_image` | `aesthetic_consistency_primary_same_image_v1.1.0.tex` | `tab:ic-meta-same-image` | 7 |
| `latex_content_hierarchy` | `aesthetic_consistency_primary_content_hierarchy_v1.1.0.tex` | `tab:ic-meta-content-hierarchy` | 5 |
| `latex_proxy_contrasts` | `reasoning_vs_proxy_primary_proxy_contrasts_v1.1.0.tex` | `tab:ic-meta-proxy-contrasts` | 4, 9 |
| `latex_name_markers` | `reasoning_vs_proxy_primary_name_markers_v1.1.0.tex` | `tab:ic-meta-name-markers` | 4 |

The name table needs the other extractor of the judge. Thus the notebook
writes it only when section 5 reads that extractor.

The 4 arms: "gemma P" is `gemma_by_qwen27_v3` (primary), "gemma C" is
`gemma_by_gemma31_v3` (control), "qwen P" is `qwen_by_gemma31_v3` (primary),
"qwen C" is `qwen_by_qwen27_v3` (control). Each arm reads 1,000 traces of
each of the 7 cases. "Both extractors" means that the number holds on the
primary arm and on the control arm of the judge.

## The ranked findings

The rank puts the most interesting and the most robust first.

### 1. The description of a fixed image follows the vote

Take 2 traces that name the same image (or the same facility) with the same
dimension name, where the image wins 1 pair and loses the other. When the 2
valences differ, the good valence sits in the win almost always.

| Arm | Facility: good in the win | Valence differs, 1 win 1 loss | Valence differs, same outcome | Image: good in the win |
|-----|---------------------------|-------------------------------|-------------------------------|------------------------|
| gemma P | 598 / 605 = 0.988 [0.976, 0.994] | 0.67 (900 pairs) | 0.07 (2,016) | 16 / 16 |
| gemma C | 756 / 760 = 0.995 [0.987, 0.998] | 0.73 (1,036) | 0.06 (2,378) | 24 / 24 |
| qwen P | 3,585 / 3,631 = 0.987 [0.983, 0.990] | 0.61 (5,942) | 0.13 (9,204) | 148 / 150 |
| qwen C | 2,620 / 2,673 = 0.980 [0.974, 0.985] | 0.58 (4,644) | 0.13 (7,522) | 105 / 107 |

The same holds for a perspective that favors the image: 0.94 to 0.995 on the
facility level, all 4 arms. Holds on both extractors of both judges.

Examples (qwen P, image `W0CPXJBY_R`, libraries): "the trees are dead/bare"
(`greenery`, bad) in `unit_00004599`, which the image lost; "The trees are
bare, but the landscaping around them looks intentional" (`greenery`, good)
in `unit_00003119`, which it won. Image `W0D9KV8K_L`: "The police cars are
shiny" (`vehicle`, good, a win, `unit_00000997`) against "The cars are
parked haphazardly" (bad, a loss, `unit_00001197_r1`).

Interpretation: the valence is not a property of the image. The model reads the
same scene up or down to fit the verdict of the pair.

### 2. The same pair under 2 prompts: the winner is independent, the description is not

Road quality and street photography read the SAME pairs, the same images in
the same order.

| Judge | Pairs, both decided | Same winner | 2 independent votes | phi | Same name, same image, same valence |
|-------|---------------------|-------------|---------------------|-----|-------------------------------------|
| gemma | 878 | 0.479 [0.447, 0.513] | 0.500 | -0.04 | 0.81 [0.67, 0.90] (43, P); 0.69 [0.61, 0.77] (127, C) |
| qwen | 968 | 0.451 [0.420, 0.483] | 0.497 | -0.09 | 0.68 [0.64, 0.72] (526, P); 0.69 [0.63, 0.74] (275, C) |

The chance level of the valence column is 0.50 to 0.51. The winner is a
judge fact; the valence column holds on both extractors.

Interpretation: the 2 prompts set 2 unrelated standards (for qwen, slightly
opposed), while the model still describes the same features in about the
same way.

### 3. A wealth inference counts for the winner, and a class inference for the loser

The directional lift of `inference.kind` (shrunk, 1 vote for each trace and
side; a case counts when its Wilson interval clears the base rate).

| Arm | wealth: cases that pay / cases | wealth lift, range | class: cases that cost / cases | class lift, range |
|-----|--------------------------------|--------------------|--------------------------------|-------------------|
| gemma P | 6 / 6 | +0.15 to +0.33 | 4 / 7 | -0.20 to 0.00 |
| gemma C | 6 / 6 | +0.18 to +0.33 | 6 / 6 | -0.28 to -0.16 |
| qwen P | 7 / 7 | +0.12 to +0.30 | 6 / 7 | -0.19 to 0.00 |
| qwen C | 6 / 7 | +0.04 to +0.27 | 1 / 7 | -0.07 to +0.08 |

Wealth holds on both extractors of both judges. Class holds on 3 of 4 arms
(not qwen C). Interpretation: the model names affluence for the image it picks,
and names social class for the image it rejects, in every prompt.

### 4. Status words mark agreement with area income and disagreement with the unit record

The band-controlled contrast (question 3, method below): the agreement of
the traces with the feature less the agreement of the traces without it,
inside (cell, gap band) strata.

`wealth_inference` (the trace holds an inference of kind wealth):

| Arm | income | crime density (area) | unit proxy |
|-----|--------|----------------------|------------|
| gemma P | +0.171 [+0.108, +0.235] (204 pairs) | -0.140 [-0.232, -0.048] (89) | -0.082 [-0.150, -0.015] (222) |
| gemma C | +0.173 [+0.114, +0.232] (252) | -0.108 [-0.190, -0.025] (113) | -0.081 [-0.141, -0.021] (314) |
| qwen P | +0.080 [+0.041, +0.119] (734) | -0.092 [-0.142, -0.043] (445) | -0.058 [-0.100, -0.017] (764) |
| qwen C | +0.120 [+0.078, +0.162] (521) | -0.132 [-0.186, -0.078] (331) | -0.053 [-0.100, -0.006] (555) |

The names that clear 0 with the same sign on both extractors of a judge:

| Name (class) | Judge | income | unit proxy |
|--------------|-------|--------|------------|
| prestige (inference) | gemma | +0.168, +0.142 | -0.124, -0.074 |
| prestige (inference) | qwen | +0.079, +0.091 | -0.079, -0.076 |
| grandeur (inference) | gemma | +0.128, +0.143 | -0.173, -0.320 |
| grandeur (inference) | qwen | | -0.220, -0.213 |
| high end (perspective) | gemma | +0.188, +0.090 | |
| high end (perspective) | qwen | +0.137, +0.126 | -0.072, -0.070 |
| grand (perspective) | gemma | +0.115, +0.134 | -0.201, -0.164 |
| grand (perspective) | qwen | | -0.221, -0.154 |
| prestige (perspective) | gemma / qwen | qwen: +0.092, +0.072 | gemma: -0.139, -0.127 |
| construction hazard (perspective) | gemma / qwen | -0.199, -0.141 / -0.107, -0.118 | |
| maintenance (perspective) | qwen | +0.080, +0.073 | +0.100, +0.087 |
| maintenance (inference) | gemma / qwen | qwen: +0.108, +0.060 | +0.058, +0.107 / +0.087, +0.074 |

Each pair of values is (primary, control). An empty cell does not clear 0
on both. Interpretation: a trace that reasons about status picks the image in the
richer district, and it picks against the facility record. A trace that
reasons about maintenance agrees with the facility record.

### 5. One hierarchy of content, with 3 flips that the prompt sets

The valence sign of a property name (not an axis such as "facade
condition") over the cases. "Settled" means the Wilson interval of the good
share clears 0.5.

| Name | Sign, all 4 arms |
|------|------------------|
| construction | bad in 7 of 7 cases, every arm (net -0.69 to -0.99) |
| architecture | good in 6 or 7 cases, never bad, every arm |
| signage | good in 5 or 6 cases; bad in parks and plazas on qwen P only |
| clutter, litter, debris, graffiti | bad in every case where they clear the floor (gemma P, qwen C); qwen P names them under the floor |
| infrastructure | FLIPS on every arm: good in parks, schools, subway; bad in restaurants and street photography |
| vehicle | bad in 4 cases on gemma P; FLIPS on 3 arms (good in schools or street photography) |
| greenery | good in 6 cases; bad in road quality on gemma P and qwen C |

Property names in 3 or more cases that hold 1 sign: 8 of 12 (gemma P), 5 of
9 (gemma C), 6 of 12 (qwen P), 12 of 17 (qwen C).

Perspectives that pay in 1 case and cost in another, on every arm:
`classic` pays in street photography and costs in 3 to 5 facility cases;
`clean` or `cleanliness` pays in 1 to 4 facility cases and costs in street
photography. Flip counts: 3 of 37 (gemma P), 3 of 15 (gemma C), 5 of 23
(qwen P), 5 of 58 (qwen C) perspectives in 2 cases or more.

Interpretation: the content hierarchy (construction and clutter bad, architecture
and signage good) serves every prompt. Street photography inverts the
cleanliness and the age standards.

### 6. Each judge's spans lean with its position bias

The share that falls on Image A. The order of the pair is random.

| Measure | gemma P | gemma C | qwen P | qwen C |
|---------|---------|---------|--------|--------|
| decision for A | 0.545 [0.531, 0.559] | (same judge) | 0.486 [0.473, 0.498] | (same judge) |
| good dimension on A | 0.546 | 0.537 | 0.483 | 0.491 |
| bad dimension on A | 0.496 | 0.502 | 0.551 | 0.552 |
| perspective favors A | 0.491 | 0.508 | 0.460 | 0.438 |
| comparison direction A | 0.576 | 0.542 | 0.469 | 0.491 |

Counts: 4,837 and 6,483 decided traces; 19,565 to 291,364 spans. Holds on
both extractors. Interpretation: gemma writes more praise of the first image and
picks it; qwen writes more criticism of the first image and picks the
second. Libraries, subway, and street photography carry the gemma effect
(0.58 to 0.59).

### 7. The same image keeps its valence more often than chance, but 1 pair in 5 disagrees

`pair_agree`: of the pairs of traces that name 1 image (or 1 facility) with
1 name, the share with the same valence. `chance` reads 2 traces on
different images.

| Arm | Image: agree (trace pairs) | chance | Facility: agree (trace pairs) | chance |
|-----|----------------------------|--------|-------------------------------|--------|
| gemma P | 0.834 [0.795, 0.866] (415) | 0.682 | 0.767 [0.758, 0.777] (8,036) | 0.684 |
| gemma C | 0.839 [0.804, 0.870] (473) | 0.671 | 0.760 [0.751, 0.769] (9,039) | 0.668 |
| qwen P | 0.788 [0.761, 0.813] (938) | 0.600 | 0.686 [0.679, 0.693] (16,666) | 0.598 |
| qwen C | 0.782 [0.750, 0.810] (714) | 0.615 | 0.703 [0.695, 0.710] (13,330) | 0.615 |

The same perspective favors the same image in 0.68 to 0.72 of the trace
pairs (chance 0.50), and the same facility in 0.62 to 0.64. Of the images
in 2 or more decided traces, 24% (gemma, 190 images) and 34% (qwen, 381)
win 1 pair and lose another; of the facilities, 46% (923) and 52% (1,677).
The same good dimension, or the same perspective that favors the image, is in a win
and in a loss of 158 to 204 facilities (gemma) and 514 to 549 (qwen).
Finding 1 says where the disagreement goes.

### 8. Task-convention perspectives are rare, mostly neutral, and almost always win

| Arm | Share of perspectives | Favors neither | Favored image wins | World knowledge: wins |
|-----|-----------------------|----------------|--------------------|-----------------------|
| gemma P | 3.7% | 44% | 0.966 [0.954, 0.974] (1,250) | 0.864 [0.861, 0.868] |
| gemma C | 2.9% | 62% | 0.975 [0.952, 0.988] (326) | 0.920 [0.917, 0.924] |
| qwen P | 4.7% | 67% | 0.945 [0.926, 0.959] (777) | 0.883 [0.880, 0.886] |
| qwen C | 2.6% | 31% | 0.945 [0.934, 0.955] (1,922) | 0.861 [0.859, 0.863] |

A trace that sets a task-convention perspective against a world-knowledge
perspective for opposite images is almost absent: 0 to 5 traces in each
arm. The task-convention perspective does not change the agreement with the
proxy (`tc_for_winner`: -0.010 to +0.018, every interval covers 0 on gemma
P). Interpretation: the model states the prompt's standard and then applies its own
knowledge; the 2 never compete in the text.

### 9. Smaller findings

| Finding | Number | Both extractors |
|---------|--------|-----------------|
| A good dimension sits on the winner | 0.79 to 0.89 pooled; a bad one 0.12 to 0.16 | yes |
| A hedged comparison points at the winner as often as a plain one | gemma 0.93 against 0.92 (P), 0.91 against 0.86 (C); qwen 0.84 against 0.91 (P), 0.90 against 0.94 (C) | yes, a small cost on qwen |
| Hedged share by label strength | Same/NotSure 0.12 to 0.44; More/Less 0.05 to 0.24; Much 0.02 to 0.09 | yes |
| More good-valence names on the winner (margin 2 or more) agree with the proxy more | qwen: unit +0.085 [+0.029, +0.140] (P), +0.073 [+0.023, +0.123] (C); income +0.095, +0.082 | qwen yes; gemma no (+0.04 to +0.05, intervals cover 0) |
| A perspective that favors the loser marks less agreement with income | qwen -0.070 [-0.107, -0.032] (P), -0.041 [-0.074, -0.007] (C) | qwen yes; gemma P -0.054, gemma C -0.001 |
| A premise dismissal marks more agreement | gemma P +0.066 [+0.016, +0.116] pooled | no: gemma C unit -0.031, qwen near 0 |
| A costly perspective (one the model overrules) marks less agreement with low crime | gemma C -0.067 [-0.129, -0.004], qwen P -0.053 [-0.098, -0.007] | no, 2 of 4 arms |
| A trace whose perspectives favor the loser in the majority | 0% to 5.3% of decided traces | yes |
| A property name with good and bad on the same image inside 1 trace | gemma 0.3% to 8% of traces; qwen 5% to 32% | the qwen rate is a fact about the long qwen traces |
| The same canonical pair 2 times in the sample: same winner | 25 of 30 (gemma), 41 of 52 (qwen) | thin |

## The method

### The image join

`pair_meta(judge)` reads the results parquet of the registered run of each
case (`submit_trace_shards.GEMMA_RUNS`, `QWEN_RUNS`) and the `pairs.parquet`
beside it. Image A of a trace is `presented_left_path`. When it equals
`image_path_a`, Image A is `sample_id_a`; when it equals `image_path_b`,
Image A is `sample_id_b`. The path decides, and `is_swapped` is the
fallback. The test `test_path_wins_over_a_wrong_swap_flag` holds the rule.
Every span of every arm found its pair row (0.0% missing).

`attach_images` gives each span the `sample_id` and the `unit_uid` of the
side its attribute names: `image` for a dimension or an inference, `favors`
for a perspective, `direction` for a comparison.

### The tables of questions 1 and 2

| Function | Holds |
|----------|-------|
| `valence_table`, `valence_summary` | good share and `net` of each dimension name in each case; the sign over the cases; `axis` marks a name with "condition", "quality", ... |
| `keyed_win` | the directional lift of any attribute (`_cues_across.keyed_lift`) |
| `source_table`, `source_conflict` | the perspectives by `source` |
| `hedge_table`, `hedge_by_strength` | the comparisons by `hedged` |
| `position_table` | the shares on Image A |
| `perspective_flips` | the perspectives that pay in 1 case and cost in another |
| `trace_valence`, `trace_favor`, `consistency` | the same-image consistency against chance |
| `partner_effect` | where the good valence sits when 1 image wins 1 pair and loses another |
| `outcome_flips`, `contradictions` | the flips with a shared reason, and examples |
| `cross_prompt_winner`, `cross_prompt_valence` | road quality against street photography |
| `within_trace` | the contradictions inside 1 trace |

### The proxy join of question 3

`proxy_pairs` gives 1 row for each pair: `proxy_gap = value(a) - value(b)`,
`agrees = sign(gap) == sign(relative_score)`, and 4 bands of `abs(gap)`
inside the cell. The cells:

| Family | Proxy | Cases | Join |
|--------|-------|-------|------|
| unit | transit crime per rider, building violations, school attendance, PIP acceptable rate, inspection score, DOT pavement rating | subway, libraries, schools, parks, restaurants, road quality | `_unit_validation.unit_proxy`; the station, the DPR property, or the nearest rated segment |
| income | median household income | subway, libraries, parks, road quality | `_ic_link`: the community district polygon of the unit or the camera |
| crime_area | crime density, negated | subway, parks, road quality | the same |

Street photography has no proxy. Schools (100 to 193 pairs) and parks unit
(115 to 150) are thin. A pair drops out when the 2 sides share a station, a
property, or a polygon. The agreement of the cells runs from 0.35 (subway,
crime density) to 0.62 (subway, income; parks, PIP).

`banded_contrast` is the Mantel-Haenszel risk difference over the (cell,
band) strata: the weight of a stratum is n1 n0 / (n1 + n0), and the normal
95% interval comes from the summed variance. A stratum with fewer than 5
pairs in a group drops out. The test
`test_banded_contrast_removes_a_difficulty_confound` shows that a raw
difference of -0.48 becomes 0 when the feature sits in the hard band only.

## The warnings

**Warning: an extractor reads the valence, and it sees the whole trace
with the decision.** Finding 1 holds on 2 extractors of 2 families, but
both can read "cleaner than B" as good for A. Thus finding 1 says that the
narration and the extractor's read of it follow the vote. It does not prove which of the
2 moves.

**Warning: the sample is thin for the same image.** 1,000 traces of each
case out of 11,000 pairs give 17 to 238 repeated images in a case. The
facility rows give 15x to 20x more pairs of traces, but 2 traces of 1 facility can
show 2 different photographs.

**Warning: the perspective and inference counts are facts about the
extractor.** Each finding above names the arms it holds on. The name
lifts of question 3 correlate across the 2 extractors at Spearman 0.76
(gemma perspectives, unit and income), 0.48 to 0.62 (qwen perspectives),
and 0.24 to 0.64 (inferences).

**Warning: question 3 tests many names.** On gemma P, 5 to 12 of the 38 to 67
(name, family) cells of each class and family clear 0. The table of finding 4 keeps only the
names that clear 0 with 1 sign on both extractors of a judge.

**Warning: an area proxy is coarse.** A pair with 2 units in 1 polygon drops
out, and a unit can differ from its polygon. Read the area agreement as a
floor.

**Warning: the cross-prompt test of finding 2 uses the presented winner.**
The 2 cases show each pair in the same order, thus the position effect of
each case enters the expected value, not the observed difference.

**Warning: this is an association, not a cause.**

## What the analysis did not do

- The source conflict (world knowledge against task convention for
  opposite images) has 0 to 5 traces in each arm. No test is possible.
- The weighing class holds 3 spans on gemma P. It is not read.
- The subway unit proxy reads the station; 2 entrances of 1 station carry
  no gap and drop out.
- The polygon join of `_ic_link.pair_table` takes the first registered
  trace run without a model filter. This module reads the results parquet
  of the judge and keeps only the polygon helpers of `_ic_link`.
