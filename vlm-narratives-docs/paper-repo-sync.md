# The paper repo and its checks

Started 2026-10-30. This page describes how the FAccT paper gets its figures,
tables and numbers from mllmsci, and how CI checks them.

| Item | Value |
|------|-------|
| Paper repo | `artificial-aesthetic/` (GitHub `pendulating/artificial-aesthetic`, private) |
| LaTeX | `artificial-aesthetic/paper/`, a submodule (`pendulating/cvpr27_machine-perception-public-spaces`, in sync with Overleaf) |
| Tool | `artificial-aesthetic/paperkit/`, Python 3.11 or later, standard library only |
| Use | `artificial-aesthetic/README.md` |

**Warning:** `papers/cvpr27_machine-perception-public-spaces/` is an older clone
of the same LaTeX repo. Its last commit is of 2026-10-01. Do not edit the paper
there. `scripts/curate_notebook_figures.py` still reads it for the
`in_paper_now` column of `figure_index.csv`.

## The data flow

```
mllmsci notebook ──> notebooks/.../outputs/x_v1.1.0.tex|png|csv
                         │  make sync (pulls the paper first)
                         ▼
artificial-aesthetic/ paper/figures, paper/tables     (+ patches)
                      results/*.csv  ──make numbers──> paper/generated/numbers.tex
                      manifest/assets.lock.json       (sha256 of source and dest)
                         │  push
                         ▼
GitHub Actions: paperkit check --ci, LaTeX build, log check
```

## The 2 manifests

| File | 1 entry is | Keys |
|------|------------|------|
| `manifest/assets.toml` | 1 file copied from mllmsci | `dest`, `source`, `producer`, `inputs`, `regen`, `patches` |
| `manifest/numbers.toml` | 1 `\stat{key}` macro | `key`, `source`, `column`, `where`, `agg`, `format`, `also_in` |

A `patches` entry holds an author edit of a generated table as an exact
old -> new pair. Example: the float position `[t]` -> `[h]`, or the
`\resizebox` width. Each old string must occur once in the source. If it does
not, the sync stops. Thus a new export cannot silently drop an author edit.

A number source must be a file in `results/`, which a sync copies. Thus CI
can make `numbers.tex` again without mllmsci.

## The checks

| Check | Where | Level |
|-------|-------|-------|
| `refs`: each `\input` and `\includegraphics` resolves | CI and local | error |
| `coverage`: each used file in `figures/`, `tables/`, `generated/` has a manifest entry | CI and local | error; a `[[manual]]` file is a warning |
| `integrity`: each copied file has the hash of the last sync | CI and local | error |
| `fresh`: the source has the hash of the last sync | local only | error |
| `producer`: the producer or an input is newer than the source; a newer `_vX.Y.Z` sibling exists | local only | warning |
| `numbers`: `numbers.tex` is current; each `\stat` key is defined | CI and local | error |
| `also-in`: the number is in the declared table row | CI and local | error |
| `hardcoded`: a literal number in the prose | CI and local | warning |
| `build`: LaTeX errors, undefined numbers, references, citations; the page count | CI, and local with `--build` | error or warning |

`--strict` makes each warning an error. Use it before the submission.

## Notes on the producers

- A matplotlib PDF holds a `CreationDate`. Thus an export of the same figure
  changes the hash. Set `SOURCE_DATE_EPOCH=0` in a producer run to make the
  PDF bytes stable.
- The check reads the age of a producer from the file time. An edit to a
  notebook that does not change its output also gives a `producer` warning.
  Run the producer again, then `make sync`, to clear it.

## State on 2026-10-30

The first sync locked 20 of 28 paper assets. The other 8 differ from their
current source, and the sync refused them for a decision by the author:

| Paper file | Difference |
|------------|------------|
| `tables/validation-by-proxy-cd.tex` | The paper holds the export of the 2026-08-20 registry (28 runs). The source is of the 2026-09-04 registry (77 runs), with other proxy rows. |
| `figures/wordblocks/{libraries,schools,subway_safety}_win_block.png` | The paper holds the `stale_pre_looks` copies. The source is the current `paper_*` figure. |
| `figures/wordblocks/subway_safety_win_block_column.png` | The source is a newer export |
| `figures/cue_words_facility_ladder.png` | The source is a newer export |
| `figures/facct/rq1_patch_words.pdf` | The paper holds the export of 2026-10-29 13:01. The source is of 17:49. |
| `figures/facct/restaurants_win_block.png` | The source is the export of 2026-10-30 14:48 |

5 numbers are wired: `corpus.images-used`, `f4.restaurant.popularity.{min,max}`
and `crossmodel.tau-tract.{min,max}`.

## Changes of 2026-11-03 (the paper audit)

| Item | Change |
|------|--------|
| `scripts/f10/hop_steer_table.py` 1.1.0 | A dash in a cell where fewer than 5 images answer yes or no (`MIN_ANSWERED`). This removes the Gemma-4-31B 'ugly' cells, which 2 to 4 images gave |
| `scripts/f10/hop_names_table.py` 1.0.0 | New. Writes `tab:borough-name` (the 3 planned contrasts of the borough-name question) from `figures/f10/names/names_primary.csv`; a new asset in `manifest/assets.toml` |
| `notebooks/facct/_rq.py` 1.10.0 | `plot_kinds` draws the percent of answers with the inference (`model_diff`), not spans for each 100 answers, so the figure and the body give the same quantity |
| `tables_export/build_tables.py` | No caption column (the caption case is out of the paper); a new `cue-categories` table (greenery and lighting against 3 ACS features); "prescriptive" is "evaluative", and `descriptive-prescriptive.tex` is `descriptive-evaluative.tex` |

The facct tables that `tables_export/` writes are not in the manifest. Copy
them into `paper/tables/facct/` by hand after a build.

## Changes of 2026-11-03, evening: the CI is green again

| Item | Change |
|------|--------|
| Unused assets | 8 CVPR-era entries left `assets.toml` and the paper: `proxy_overview.*`, `validation-by-proxy-cd.tex`, `cue_words_facility_ladder.png`, the libraries, schools and `subway_safety_win_block.png` word blocks, `rq1_patch_words.pdf` |
| `positional_bias.tex`, `test-retest.tex` | A forced sync took the current export (the interval in the caption is now $\pm0.005$). Patches drop the 4 rows of the cases that are not in the FAccT paper (libraries, schools, road quality, parks) |
| `subway_safety_win_block_column.png` | A forced sync took the current render in place of the `stale_pre_looks` copy |
| `pairwise_ses.pdf` | Now a declared asset (`notebooks/facct/_pairwise_ses.py`) |
| `results/f10/*.csv` | 7 result files behind 31 new `\stat` numbers of the body: the probe $R^2$ and gains, the verdict $r$, the steering yes shares, the subjectivity notes, the ICCs, the 'interesting' partial $r$, the New York shares, and the Manhattan $t$ on 'cool' |
| `paperkit/checks.py` | The `refs` check skips `\input{#1}` inside a macro definition |

The numbers check found one stale literal: the Qwen3.5-9B scene $R^2$ is .76
(raw .7562), where the text and Figure 1 said .75. Both now say .76.

## Changes of 2026-11-03, night: the pairwise study is out of the paper

The author removed the pairwise comparison study from the FAccT paper. It did
not add to the argument, and it took about 2,800 words of appendix.

| Item | Change |
|------|--------|
| Body | No pairwise sentence in the abstract, the introduction, Table 1, the methods, the results, the discussion, or the limitations. The CVPR-era `comment` block of the related work is gone |
| Appendix A | Lane C of the image figure, the cube-faces paragraph, the pairwise roles in `tab:models`, and the subsection "The pairwise comparison study" with its robustness checks are gone. The human check of the extractor stays; it reads "ten reasoning traces from a pilot comparison task" |
| Appendix B | The subsection "Pairwise comparisons" is gone. The crime paragraph of B.3 is now "Sources of the quotations" |
| Paper files | Deleted: `tables/facct/cue-prices.tex`, `tables/facct/paradigm-cue-order.tex`, `tables/positional_bias.tex`, `tables/test-retest.tex`, 6 CVPR tables under `tables/` that nothing input, `figures/facct/pairwise_ses.pdf`, and the `figures/wordblocks/` folder |
| `manifest/assets.toml` | 10 pairwise assets and `results/cvpr/cross_model.csv` are gone. `prompt-ses.tex` has a patch, "share of white and of Black residents", that holds the author's Overleaf edit |
| `manifest/numbers.toml` | `crossmodel.tau-tract.{min,max}` are gone (42 numbers remain) |
| `manifest/assets.lock.json` | The entries of the removed assets are gone |

The paper builds at 44 pages (51 before). The references start on page 15.
The CI check reports 0 errors.

Bib entries that nothing cites now: `naikStreetscorePredictingPerceived2014`,
`naikComputerVisionUncovers2017`, `minkaTrueSkill2Improved`.

## Change of 2026-11-03, night: recurrence in place of the ICC

The body sentence on the stability of objects and standards (the 'cool'
traces) now gives a simpler measure. `scripts/f10/analyze_argument.py`
writes a `recur` column to `argument_repeat.csv`: when 1 repeat of an image
holds an item, the share of the other 2 repeats that hold it. The `share`
column is its base rate.

| Model | Kind | Base rate | Recurrence | ICC(1) |
|-------|------|-----------|------------|--------|
| Gemma-4-31B | object | 9% | 59% | .55 |
| Gemma-4-31B | standard | 47% | 64% | .31 |
| Qwen3.5-9B | object | 16% | 62% | .55 |
| Qwen3.5-9B | standard | 58% | 69% | .28 |

**Warning:** The raw recurrence favors the standards, because a standard
such as 'character' is in 80% to 89% of all traces. State the base rate
with it. The chance-corrected recurrence equals the ICC to 2 decimals.

The 4 `f10.icc.*` numbers are now 8 numbers, `f10.recur.{object,value}.*`
and `f10.base.{object,value}.*`, in percent. Appendix A defines recurrence
and keeps the ICC as a check; Appendix B gives both.

## Change of 2026-11-04: the layers figure is in the body again

Figure 2 of the body (`fig:prompt-layers`) is the layer panel that was Figure B.1. It now
shows Gemma-4-31B (dark bars, top scale) and Qwen3.6-27B (light bars, bottom scale), not
Qwen3.5-9B. The patch-word strip was in the figure for one revision and is out again.
The row "scene code" is "image, mean patch".

| Item | Change |
|------|--------|
| `scripts/f10/phase_layers.py` 1.1.0 | 3 models. Reads the per-layer files of F10.26 (`figures/f10/prompts/t2_*.csv`, prompt `place cool`). The phase `scene` is `image`. A verdict band that no layer reaches is the peak layer with `reached` false (Qwen3.6-27B, r .92 at L60). Gemma-4-31B and Qwen3.5-9B keep their bands |
| `scripts/f10/plot_layers_tikz.py` 1.0.0 | New. Writes `figures/f10/mechanism/layers_panel.tex`, the `tikzpicture` of the figure, from `phase_layers.csv` and `patch_layer_words.csv` (F10.33b) |
| `manifest/assets.toml` | New asset `paper/figures/facct/layers_panel.tex`. The body figure inputs it |
| Appendix B.1.1 | The figure is gone. The text gives the rules, the Qwen3.5-9B bands, and the reading of the patch strip |

Bands of Qwen3.6-27B on `place cool`: image L0 to 44 (onset 28, peak R2 .57), standard
L32 to 60, verdict L60 (r .92, below the .95 rule).
