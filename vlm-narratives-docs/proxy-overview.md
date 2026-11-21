# The proxy overview

`notebooks/cvpr/_proxy_overview.py` and `notebooks/cvpr/master/proxy_overview.py`.
Added 2026-09-06.

## What it is

The paper has little space, thus 1 full-width figure and 2 one-column tables
replace the 3 validation tables. The figure folds 3 questions into 1 place,
and the tables hold the 2 that a figure cannot.

| Question | Where it sits |
|----------|---------------|
| Does the model agree with the proxy? | Panel A, 1 marker for each layer |
| Does the geography layer change that? | Panel A, the line between the markers |
| Does an ablation align the model more or less with the proxy? | Panel B, the heatmap |
| Do the 2 models agree with each other? | `tables/cross-model.tex` |
| Which value does an ablation move? | `tables/ablation-by-family.tex` |

The cross-model block is no longer part of the validation table. A row of
that block differed from the next row only in the count of areas, thus it
took 4 columns to say 1 thing.

## The figure

One row for each (case, proxy) of the master table, in the order of the
master table. A row must reach the area floor (`R.MIN_AREAS`, 20) at the
community district for at least 1 rater.

| Mark | Meaning |
|------|---------|
| Blue | Gemma-4-12B |
| Amber | Qwen3.5-9B |
| Green diamond | Gemma, the block-42 lens read of a single image, at the community district (see below) |
| Marker size | The layer: tract, NTA, community district, small to large |
| Hollow marker | Kendall tau with p at or above 0.05 |
| Line | Joins the 3 layers of 1 rater; its length is the granularity effect |
| Letter after the proxy | The value family (see below) |
| Row opens with `Fewer` or `Lower` | A bad-is-high proxy that the loader negated; a positive tau says the better-scored area holds less of it |
| Tag `(newer = +)`, `(larger = +)`, `(busier = +)` | A vintage or context row with no orientation; the tag states what a positive value means |
| Heatmap cell | The paired change of tau under the arm, at the community district |
| Teal cell | The arm aligns the model MORE with the proxy |
| Coral cell | The arm aligns the model LESS with the proxy |
| Black outline | A 95% bootstrap interval over areas that excludes 0 |

The single-image series is a read of the SAME rater as the blue markers,
thus it takes a marker shape of its own and not a second blue. The figure
draws it when `master/outputs/single_image_read_v1.0.0_long.csv` exists,
and it falls back to the 2-rater layout when it does not (`--no-single`
forces the fallback). By default it draws the lens read alone, at the
community district alone, 1 diamond for each row: 2 reads at 3 layers made
the panel too dense to read, and the lens read is the white-box one (user
decision 2026-09-06). `--single-reads contrast_L42,c --single-layers
tract,NTA,CD` brings the full set back.

Blue and amber carry the model identity because they pass the colour-vision
checks as a pair. The coral and teal of the house ramp do not (protan
Delta E 3.9), thus the ramp carries the polarity of the heatmap alone.

The row names come from `ROW_LABEL` in `_proxy_overview.py`, not from the
master-table labels: the table writes `(negated)` after a proxy, and the
figure writes the orientation into the name instead (2026-09-06).

## The value families

A proxy stands for an attitude, and the ablation question reads "does the
prompt change align the model more with that attitude?". `FAMILY` in
`_proxy_overview.py` maps each proxy key to a family. Edit it there.

| Glyph | Family | Proxies |
|-------|--------|---------|
| D | Disorder (broken windows) | crime density, transit crime and felonies, building violations, DOB and ECB violations, potholes |
| A | Affluence and investment | median household income, alteration permits, BID membership and partner, economic need |
| R | Official condition rating | DOT pavement rating, resurfacing age, PIP rates, teacher survey, DOHMH score, grade A rate |
| V | Vintage | construction year, acquisition year, POPS completion year |
| c | Context | ridership, park acreage, attendance, outdoor dining |

A context row describes the place and makes no validation claim, thus the
family table leaves it out.

## The cross-model table

One line for each of the 7 cases. The raw columns read the label of each
comparison from the canonical registry and compare the 2 models pair by
pair. The layer columns take the cross-model tau of the master table at
each layer, from the proxy with the most areas.

| Rule | What it does |
|------|--------------|
| Every case stays | Unlike `R.raw_cross_model`, a thin rater does not drop the row. Schools keeps its raw cells. |
| Street photography | Has no proxy join, thus it has raw cells and dashes. |
| Parks and plazas | Come from 1 run and 2 proxy folders. The raw cells print once, the layer cells on 2 lines. |
| Layer floor | A layer cell needs 20 areas. Libraries has 1 tract and 11 NTAs, thus 2 dashes. |
| Side floor | The side kappa needs `R.MIN_BOTH` (1,000) answered pairs, or it prints a dash. |

## The family table

The mean paired change of tau by family, model, and arm, over the cells of
the proxy re-run (`proxy_arms_<layer>_v*.csv`). A thin cell stays out. The
`rows` column counts the (case, proxy) cells of the family.

## Inputs

The module is downstream. It reads:

| Input | Written by |
|-------|------------|
| `master/outputs/results_table_<layer>_v1.0.0_long.csv`, 3 layers | `master/results_table.py` or `scripts/export_cvpr_results_table.py` |
| `master/outputs/proxy_arms_community_district_v1.0.0.csv` | `master/ablations.py` |
| The canonical run parquets | the registry, for the raw rows |
| `master/outputs/single_image_read_v1.0.0_long.csv` | `_single_image_read.py` (optional) |

Panel B reads the layer that `master/ablations.py` exported. It has written
the community district alone. To show another layer, export the ablations
notebook at that layer and pass `--arm-layer`.

## Outputs

The export writes to `master/outputs/` and into the paper repo.

| File | Where |
|------|-------|
| `proxy_overview.pdf`, `proxy_overview.png` | `papers/.../figures/` |
| `proxy_overview.tex` | `papers/.../figures/`, the `figure*` float with the caption |
| `cross-model.tex` | `papers/.../tables/` |
| `ablation-by-family.tex` | `papers/.../tables/` |
| `proxy_overview_v1.1.0.*`, `cross_model_v1.1.0.*`, `ablation_by_family_v1.1.0.*` | `master/outputs/`, the same files and the frames behind them |
| `proxy_overview_v1.1.0_single.csv` | `master/outputs/`, the single-image cells the figure drew |

The paper reads them with `\input{figures/proxy_overview}`,
`\input{tables/cross-model}`, and `\input{tables/ablation-by-family}`.

## The single-image series

`notebooks/cvpr/_single_image_read.py` puts the white-box reads of
`pole-projection-proxy.md` on the footing of the master table. For each
unit the study scored (3 images each; 1,500 restaurants, 3,000 road
images, every joined unit elsewhere) it takes the mean read, joins the unit
to its tract, NTA, and community district with the point layer and the
polygon floor of the case notebook (`MIN_UNITS`), takes the plain mean over
the polygon, and reads Kendall tau against the `<prefix>_proxy_v*.parquet`
export of the case. The proxy side is the file the master table reads,
thus a single-image cell and a pairwise cell differ in the model score
alone. The area floor of 20 applies in `load_single`.

| Read | What |
|------|------|
| `c` | The logit gap between the 2 polar words, one prefill per image |
| `contrast_L42` | The centered projection of the block-42 answer residual on the lens contrast vector |

At the community district, over the 27 rows where both exist, the lens read
takes the sign of the pairwise Gemma cell in 22 and ranks the rows with it at
Spearman 0.52 (the gap `c`: 26 and 0.64). Both sit closer to zero for
libraries and farther from it for schools, where the pairwise run abstains
on half the comparisons. Over all 73 cells of the 3 layers the sign of `c`
agrees in 92% and its rank correlation is 0.73. The units are
a sample, thus a polygon here can hold fewer units than the same polygon
of the master table.

```bash
.venv-mllmsci-vllm025cu129/bin/python notebooks/cvpr/_single_image_read.py
```

## Commands

```bash
# after the master table and the ablations notebook export
.venv-mllmsci-vllm025cu129/bin/python notebooks/cvpr/_proxy_overview.py
# or, for a look first
.venv-mllmsci-vllm025cu129/bin/marimo edit notebooks/cvpr/master/proxy_overview.py
```

The command checks the canonical registry against the disk before it reads
a parquet. Pass `--skip-verify` only when you know why.

## The granularity effect, 2026-09-06

The median tau over the validation rows rises with the layer, for both
models. The share of cells with p under 0.05 does not, because n falls from
about 2,300 tracts to 59 districts.

| Model | tract | NTA | CD |
|-------|-------|-----|----|
| Gemma-4-12B | .08 | .15 | .19 |
| Qwen3.5-9B | .11 | .14 | .18 |

The cross-model tau follows the same path (subway safety .61, .67, .71).
