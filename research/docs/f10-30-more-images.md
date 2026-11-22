# F10.30: more images for each NTA (D185 to D189)

**Status (2026-11-03, 11:00):** both captures are complete and verified for the 3 models: 140,000
answer records for each model with 0 errors, and 42 lean stores of 1,970 images each with 0 faults.
The 10-image summaries (`data/f10/prompts_k10/`) and the readout analyses follow below. D189 (the
word lists, the residual directions, the k = 50 table and the paper proposals) is done. Nothing is
committed.

## The question

The F10 readout analyses (F10.18 to F10.29) use 3 images for each NTA, 591 images in all. F10.29
question 3 gives the limit: the ICC(1) of 1 image is .19 (median), the reliability of the 3-image
mean is .42, and with 1 image for each NTA the income test passes in 1 cell of 38. More images for
each NTA make the NTA mean readouts, the ACS and borough gradients, the refusal shares and the
shared direction more stable. One night of GPU time gives 2 captures:

| Capture | Images | For each NTA | What it keeps | Pipeline |
|---------|--------|--------------|---------------|----------|
| Answers | 10,000 | 50 or 51 | the yes-or-no answer of the 14 prompts, 8 forms and the floor | vLLM server (`research/scripts/f4/serve_and_query.sbatch` with `F4_ANSWERS=1`) |
| Lean | 1,970 | 10 | residuals at 6 positions, lens readouts at 3 positions, the 8-form margin | HF float32 (`research/scripts/f10/hop_lean.py`) |

The answers give the gradient tests at 50 images for each NTA. The lean capture gives the readout
and direction analyses at 10 images for each NTA, 3.3 times the 591 of F10.26.

## The sample (D185)

The user set 3 rules: do not use `data/cyclomedia`; sample from the full citywide catalog at
`/share/ju/cyclomedia`; use the same capture epoch as the other samples. The F4 street sample
meets all 3, and it is the sample the 591 images come from.

| Item | Value |
|------|-------|
| Source files | `data/f4_traces/f4_street_1k.parquet` (batch 1, 985 views, 5 for each NTA) and `data/f4_traces/b2/f4_street_b2.parquet` (batch 2, 9,015 views, 45 or 46 for each NTA) |
| Sampler | `research/scripts/f4/sample_street_images.py` 1.1.0, seed 20261023, catalog `/share/ju/cyclomedia/catalog/v1` (the citywide catalog, `docs/catalog.md`) |
| Epoch | August to November 2025 (2025H2). `recordedAt` from 2025-08-01 to 2025-11-04, both batches |
| Renderer | the F6 renderer, street direction from CSCL, 1024 px |
| Tables | `data/f10/hop_more/images_all.parquet` (10,000 rows), `images_k10.parquet` (1,970 rows), `images_manifest.json` |
| Rank rule | the 591 images of `data/f10/hop_depth/images.parquet` take ranks 1 to 3 in their NTA; the other views follow by `image_idx`. The 10-image table is rank 1 to 10 |

The 591 images of F10.26 are a subset of both tables (`in_591` column), thus the old results
reproduce from the new stores, and the new images are a held-out set for the directions that
F10.26 task 7 fit on the 591.

## The answers capture (D186)

`research/scripts/f10/answer_query.py` sends the F10 prompt `Is this {noun} {adjective}? Answer with one
word: yes or no.` for the 7 adjectives and 2 nouns to a vLLM server, thinking off, 1 token, the
top 2,000 log probabilities (`--max-logprobs 20000` on the server, the F4_PROBE path). The record
keeps the 8 answer forms and the floor of the list; a form that is not in the list takes the floor
(D164). 96 requests at a time, resume from the output file.

| Model | Job | Node | Output |
|-------|-----|------|--------|
| qwen3.5-9b | 118501 | g3-compute-13 | `data/f10/hop_more/answers/qwen3.5-9b/answers.jsonl` |
| gemma-4-31b | 118502 | g3-compute-14 | `data/f10/hop_more/answers/gemma-4-31b/answers.jsonl` |
| qwen3.6-27b | 118503 | pending at 23:10 | `data/f10/hop_more/answers/qwen3.6-27b/answers.jsonl` |

Submit:

```bash
for m in qwen3.5-9b gemma-4-31b qwen3.6-27b; do
  sbatch --job-name=f1030-$m --exclude=g3-compute-17 \
    --export=ALL,F4_MODEL=$m,F4_PROBE=1,F4_ANSWERS=1,F4_PROBE_TOP=2000,F4_SAMPLE=data/f10/hop_more/images_all.parquet \
    research/scripts/f4/serve_and_query.sbatch
done
```

The job copies the model and the 10,000 images to `/scratch` first (`stage_inputs.sh`). 140,000
requests for each model.

## The lean capture (D187)

`research/scripts/f10/hop_lean.py` is the F10.6 capture (`hop_all.py`) with less output. The model loads
once in float32 and runs the 14 prompts in turn. For each image it keeps:

| Array | Shape | Positions |
|-------|-------|-----------|
| `residuals/residuals-*.npy` | [images, 6, layers, d] float16 | the 6 positions of `hop_depth.POSITIONS` |
| `reads/reads-*.npy` | [images, 3, layers, vocab] float16 | `place`, `after_place`, `final` |
| `answers/answers-*.parquet` | 1 row for each image | the 8 `lp_` forms, margin, yes_no, top, tokens |

The vocab is the F10 word list (`" [a-z]{3,}"`). The store name is the F10.26 one
(`store_name(model, adj, noun)`), thus the F10.26 scripts read the new stores with a root change.
Shards of 50 images with claims, as F10.6: many workers fill 1 store, and a requeue resumes.

| Model | Jobs | Time for each image (3 prompts of F10.29) | Estimate for 14 prompts, 1,970 images |
|-------|------|------------------------------------------|---------------------------------------|
| gemma-4-31b | 118504, 118505 | .4 to .56 s | 2 to 4 h with 2 workers; done at 02:20 |
| qwen3.6-27b | 118506, 118507, 118597 (added at 00:05) | 1.3 s | 5 h with 2 workers; done at 03:35 with 3 |
| qwen3.5-9b | 118508, 118598 (added at 00:05) | .5 s | 4 h with 1 worker; done at 02:50 with 2 |

Submit:

```bash
sbatch --export=ALL,F10_MODEL=gemma-4-31b --job-name=f1030-lean-gemma-4-31b-1 research/scripts/f10/hop_lean.sbatch
```

All 5 lean jobs were pending at 23:10 (Priority) behind the 3 answer jobs and the F10.29 runs. The
cluster preempted 3 of the jobs once each; the requeue took the same job id, thus each worker took
its own claims again and lost at most 1 shard of work.

## Checks of the 2 captures against F10.26

| Check | Result |
|-------|--------|
| The vLLM answers against the F10.28 HF answers, Qwen3.5-9B, the 591 images, E for each cell | Spearman rho .98 to 1.00 in the 14 cells (`research/figures/f10/more/check.csv`) |
| The lean store vocab against the F10.6 store vocab, Gemma | the same 53,999 tokens in the same order |
| The lean readouts against the F10.6 store, Gemma `cool place`, shard 1, the 2 shared images | r = 1.00 at `place` and `final` (L32), max difference .008 (float16); the residuals r = 1.00; the margins equal to 3 decimals |
| The 42 lean stores at the end | 40 shards each, 1,970 images, reads [50, 3, layers, vocab] and residuals [50, 6, layers, d] float16, the image ids in the sample order, no empty margin. Layers 15 (Gemma), 8 (Qwen3.5-9B), 16 (Qwen3.6-27B); d 5,376, 4,096, 5,120; vocab 53,999, 43,563, 43,563 |
| The refusals in the lean stores, Gemma | `ugly` 1,904 and 1,653 of 1,970, `nice` 930 and 166, `pretty` 405 and 9; 0 in the Qwen stores. The same shares as F10.26 and as the 10,000-image answers |
| The k = 10 summary against the F10.26 summary, Gemma `cool place` | the same NTA order; the residual rows of the shared images at r = 1.00; the margins of the 591 shared images equal to 1e-6 |
| Speed of the lean capture, Gemma, 1 worker | 1.27 images a second: 26 min for each prompt, 6 h for the 14 prompts; 2 workers share the shards |

The scripts for the analysis stage are `research/scripts/f10/hop_more_answers.py` (the answers at k = 3, 10, 50:
`research/figures/f10/more/{check,ses,partial,manhattan,reliability,unable}.csv`) and
`research/scripts/f10/hop_more_summary.py` (the F10.26 summary layout from a lean store, into
`data/f10/prompts_k10/{store}/`, with `nta_` and `residual_` at `place`, `after_place` and `final`).

## Results of the answers at k = 3, 10 and 50 (D188, part 1)

`research/figures/f10/more/` holds the tables. k is the number of images for each NTA; k = 3 is the 591
images of F10.26 to F10.29, from the same vLLM capture.

**The ACS gradients** (`ses.csv`; 168 E tests at each k, Spearman rho over 197 NTAs, Bonferroni
over the 188 tests of the k, which include the 20 Gemma P_unsure tests):

| k | E tests that pass | median abs rho with income | with % bachelor's | with % White | with % Black |
|---|---|---|---|---|---|
| 3 | 69 of 168 | .34 | .26 | .25 | .12 |
| 10 | 89 of 168 | .43 | .33 | .34 | .20 |
| 50 | 108 of 168 | .47 | .39 | .39 | .22 |

The gradients grow with k, as the reliability of the NTA mean grows. At k = 50 the rho of E with
income is .40 to .64 for `cool`, `nice`, `pretty` and `fancy` in all 3 models, -.30 to -.54 for
`ugly`, and near 0 for `ordinary` and `urban` (-.11 to .02). The Gemma P_unsure tests pass in 12
of 20 at every k: the refusal share of `nice` falls with income (rho -.45 and -.53) and with the
share of White residents (-.47 and -.51), and rises with the Black share (.31 and .32).

**The partial test of F10.29 question 1** (`partial.csv`; 18 cells of `ugly`, `nice`, `cool`):

| k | Black share passes | Black share given income passes | income given Black share passes | median abs partial rho, Black given income |
|---|---|---|---|---|
| 3 | 2 of 18 | 0 of 18 | 17 of 18 | .02 |
| 10 | 12 of 18 | 0 of 18 | 16 of 18 | .04 |
| 50 | 13 of 18 | 0 of 18 | 17 of 18 | .07 |

With 50 images for each NTA the Black share passes on its own in 13 cells (rho -.21 to -.31 for
`cool` and `nice`, .12 to .31 for `ugly`), but given income it passes in 0 cells (partial rho -.12
to .13). Income given the Black share passes in 17. The F10.29 answer holds at 17 times the
images: income carries the gradient, and the Black share adds nothing that the test can see.
This is a statement about the 2 features in this sample, not a causal order.

**Manhattan** (`manhattan.csv`; Welch t of the NTA mean E, 42 cells): 23, 20 and 24 cells pass at
k = 3, 10 and 50; 19, 17 and 18 of them with Manhattan above the rest.

**The reliability** (`reliability.csv`; ICC(1) of E over images in an NTA, and the Spearman-Brown
reliability of the k-image mean, medians over the 14 prompts):

| Model | ICC(1), k = 3 / 10 / 50 | reliability of the mean, k = 3 / 10 / 50 |
|-------|---|---|
| Gemma-4-31B | .18 / .15 / .13 | .40 / .65 / .88 |
| Qwen3.5-9B | .16 / .17 / .16 | .37 / .67 / .90 |
| Qwen3.6-27B | .19 / .19 / .18 | .41 / .69 / .91 |

ICC(1) is stable in k, thus the 3-image estimate of F10.29 question 3 was right. The 50-image mean
has a reliability of .9, and the disattenuated gradient of F10.29 (rho .50) is the observed one
at k = 50 (.47 median with income).

**The refusals** (`unable.csv`, Gemma, k = 50, 10,000 images): `ugly` .97 (`place`) and .85
(`scene`), `nice` .48 and .09, `pretty` .21 and .01; 0 on the other 8 prompts. Manhattan has the
fewest refusals on `nice` (.37 against .50 to .54) and `scene ugly` (.84 against .83 to .89, with
Staten Island at .71). The 2 Qwen models refuse on 0 images.

**The check of the vLLM path** (`check.csv`): on the 591 images the Spearman rho of E against
the F10.28 HF answers is .96 to 1.00 in 38 of the 42 cells. The 4 cells at .36 to .86 are the
saturated Gemma cells (`ordinary` at 97% yes, `fancy` at 5% yes), where E sits at the bound for
almost every image and the rank order rests on the bf16 tail. Mean E and the yes share agree to
2 decimals in all 42 cells.

## Results of the lens readouts at k = 10 (D188, part 2)

The summaries are in `data/f10/prompts_k10/` (42 stores, `research/scripts/f10/hop_more_summary.sbatch`,
jobs 118875 and 118944, 6 min each on 1 CPU node). `research/scripts/f10/hop_more_readouts.py` runs the
F10.27 to F10.29 readout analyses on them with a change of root; the tables are in
`research/figures/f10/more/`. The F10.27 directions of the 591 images stay in `research/figures/f10/shared/`.

**The agreement of the NTA means, k = 3 against k = 10** (`agree.csv`; 14 prompts for each model,
the projection of the centred NTA mean readout at the noun on the F10.27 direction at its test
layer; medians over the prompts):

| Model | r of the projection, k = 3 against k = 10 | r of the NTA margin, k = 3 against k = 10 | r of the projection with the NTA margin, k = 3 | the same at k = 10 | passes at k = 10 (Bonferroni over 14) |
|-------|---|---|---|---|---|
| Gemma-4-31B | 0.68 | 0.75 | 0.5 | 0.54 | 8 of 14 |
| Qwen3.5-9B | 0.61 | 0.74 | 0.38 | 0.41 | 14 of 14 |
| Qwen3.6-27B | 0.6 | 0.76 | 0.47 | 0.42 | 13 of 14 |

The 3-image and the 10-image NTA means agree at r .6 to .7 on the direction (7 of the 10 images
are new), and the direction reads the NTA margin as well at k = 10 as at k = 3.

**The transfer across models** (`transfer/transfer_test_layer.csv`; the direction of 2 models,
fit on the 591 images, reads the k = 10 NTA means of the third; Bonferroni over 14):

| Target | Passes, k = 3 | Passes, k = 10 | Mean r over the 8 aesthetic prompts, k = 3 / k = 10 | r of `ugly`, k = 3 / k = 10 |
|--------|---|---|---|---|
| Gemma-4-31B | 6 of 14 | 9 of 14 | .33 / .33 | -.36 / -.23 |
| Qwen3.5-9B | 14 of 14 | 14 of 14 | .60 / .58 | -.64 / -.69 |
| Qwen3.6-27B | 11 of 14 | 14 of 14 | .55 / .56 | -.68 / -.70 |

The F10.29 question 2 answer holds on the held-out images, and 2 targets gain passes.

**The shared direction against the ACS features** (`ses_readout.csv`; the NTA mean readout at
the noun, projected on the F10.27 direction; Spearman rho over 197 NTAs, Bonferroni over 12):

| Model | income, k = 3 / 10 | % bachelor's, k = 3 / 10 | % White, k = 3 / 10 | % Black, k = 3 / 10 | passes, k = 3 / 10 |
|-------|---|---|---|---|---|
| Gemma-4-31B | .32 / .41 | .30 / .41 | .23 / .40 | -.12 / -.19 | 3 / 3 |
| Qwen3.5-9B | .25 / .25 | .21 / .19 | .16 / .20 | -.12 / -.14 | 1 / 2 |
| Qwen3.6-27B | .19 / .20 | .18 / .18 | .17 / .19 | -.07 / -.10 | 0 / 0 |

The lens gradients are smaller than the answer gradients at the same k (.47 median with income
at k = 50 for the answers), and the Black share passes in no model at either k.

**The shared direction fit again on the 10-image summaries** (`shared/{model}_place/`; the
F10.27 fit, 4 fit prompts, 10 held out):

| Model | Test layer, k = 3 / 10 | Cosine of the 2 directions at the test layers | Held-out passes at the test layer, k = 3 / 10 | Mean abs r on the held-out prompts, k = 3 / 10 |
|-------|---|---|---|---|
| Gemma-4-31B | L8 / L4 | .96 and .95 (.73 to .98 over the layers) | 4 / 5 of 10 | .32 / .25 |
| Qwen3.5-9B | L28 / L12 | .96 and .97 (.93 to .97) | 9 / 8 of 10 | .41 / .48 |
| Qwen3.6-27B | L60 / L60 | .94 (.91 to .96) | 6 / 10 of 10 | .43 / .53 |

The direction is the same at k = 10 (cosine .94 to .97 at the test layers, and above .9 at every
layer but Gemma L0). The test layer, which the leave-1-out r of the 4 fit prompts chooses, moves
in 2 models; the held-out r is flat over the layers, thus the choice is not stable, and the paper
text should name the band and not 1 layer. The word lists at 1 layer differ between the fits
(0 to 14 of 30 shared); the F10.28 word analysis uses the layer mean, and the next block gives it.

**The word lists of the k = 10 direction** (`words.csv`, `words_agree.csv`; the F10.28 rule:
z-score the direction at each layer, take the mean over the layers, and list the 20 words at each
end; job 120293, task 0):

| Model | rho of the layer mean over the vocabulary, k = 3 against k = 10 | Shared words of the top 20, up / down | Shared words of the top 100, up / down |
|-------|---|---|---|
| Gemma-4-31B | .97 | 12 / 13 | 67 / 67 |
| Qwen3.5-9B | .96 | 15 / 12 | 61 / 67 |
| Qwen3.6-27B | .95 | 12 / 10 | 63 / 68 |

The lists hold at k = 10. The top of each list at k = 10: Gemma up `flower`, `wishlist`,
`bridesmaid`, `flute`, `holiday`, `petals`, `cherished`, `lace`, `bouquet`; down `contaminate`,
`crude`, `contaminating`, `carbides`, `contempt`, `incompetent`, `depleted`, `welding`, `smelting`.
Qwen3.5-9B up `exquisite`, `majestic`, `queen`, `elegant`, `beloved`, `inspiring`; down `abusive`,
`industriale`, `industrielle`, `dumpster`, `magazzino`, `abuses`, `slaughter`, `hitam`.
Qwen3.6-27B up `exquisite`, `elegance`, `elegant`, `beautifully`, `magnifique`, `stunning`;
down `dumpster`, `stripper`, `zombie`, `makeshift`, `poorly`, `punk`, `grim`, `gang`, `worthless`.
Of the words that the paper quotes, these leave the top 20 at k = 10 (rank at k = 3, then at
k = 10): Gemma `industrial` (8, 44) and `ornaments` (20, 26); Qwen3.5-9B `warehouses` (3, 39),
`unemployed` (20, 31), `dumpsters` (8, 21) and `regal` (10, 21); Qwen3.6-27B `factory` (15, 89),
`warehouses` (13, 59) and `splendid` (28, 41). `hitam` is at rank 12 in both fits. Proposal P57
gives the words to keep.

**The task-7 residual directions on the k = 10 residuals** (`steer/{model}/dirs.npz`,
`dirs_agree.csv`, `dirs_projection.csv`; `hop_steer.py dirs` with the k = 10 summaries as the
input: 1,970 images, 14 prompts, 1 direction for each cell of F10.29 question 5; job 120293,
tasks 1 to 3; stage `project` on the session node):

| Model | Cell | Cosine of the 2 directions | r of the 2 projections, 1,970 images | r with the margin on the 1,379 new images, 591-image direction / 1,970-image direction | sigma, 591 / 1,970 |
|-------|------|---|---|---|---|
| Gemma-4-31B | `place` L32 | .64 | .95 | .77 / .86 | .82 / .94 |
| Gemma-4-31B | `final` L44 | .75 | 1.00 | 1.00 / 1.00 | 3.65 / 2.44 |
| Qwen3.5-9B | `place` L28 | .59 | .94 | .67 / .74 | .58 / .59 |
| Qwen3.5-9B | `final` L28 | .78 | 1.00 | .99 / 1.00 | 2.54 / 1.91 |
| Qwen3.6-27B | `place` L48 | .62 | .96 | .51 / .52 | .47 / .66 |
| Qwen3.6-27B | `place` L36 | .60 | .96 | .44 / .51 | .40 / .57 |
| Qwen3.6-27B | `final` L60 | .81 | .99 | .98 / .98 | 4.36 / 2.93 |

The cosine of 2 ridge directions in a space of 4,096 to 5,376 dimensions, fit on 591 and on
1,970 images, is .53 to .64 at the noun and .75 to .81 at the last token. But the 2 directions
order the images in the same way: r .93 to .96 at the steering layers (lower at Gemma L16, .68,
and Qwen3.5-9B L20, .84). The part of the direction that changes between the fits lies where the
images have little variance. The 591-image direction reads the margin of the 1,379 new images at
r .77, .67 and .51 (Gemma L32, Qwen3.5-9B L28, Qwen3.6-27B L48), against .81, .74 and .62 on its
own images. Thus the F10.29 steering directions are the directions that a 3.3 times larger fit
gives, up to a component that does not move the images. The limit of question 5 ("decoders fit
on the same 591 images") does not change the order of the images. The dose unit does change:
sigma at the noun is 10% to 40% larger under the 1,970-image direction in Gemma and Qwen3.6-27B,
and smaller at the last token. A rerun of the steering with these directions is optional; if it
runs, use the sigma of the 1,970-image direction.

**The k = 50 table in the paper layout.** `research/notebooks/facct/tables/f10_prompt_ses_k50.tex` is
`tab:prompt-ses` built from `ses.csv` at k = 50 with `hop_hierarchy.prompt_table` (120 of 188
tests pass). It is not in the paper repo. Proposal P52 offers it.

## The borough order of each standard at k = 50 (D191)

Stage `--borough` of `research/scripts/f10/hop_more_answers.py` (1.1.0). Outputs
`research/figures/f10/more/borough_order.csv` (the order of the borough means of the NTA means
of E, 42 cells) and `borough_tests.csv` (Welch t of each borough against the rest,
Bonferroni over the 42 cells of each borough), and the paper table
`research/notebooks/facct/tables/f10_borough_order.tex` (`tab:borough-order`).

| Adjective | First | Last | Passes of the first borough (of 6) |
|-----------|-------|------|------------------------------------|
| `cool` | Manhattan (6) | Bronx (4), Staten Island (2) | 6 |
| `fancy` | Manhattan (6) | Bronx (6) | 6 |
| `urban` | Manhattan (6) | Staten Island (6) | 6 |
| `nice` | Staten Island (6) | Bronx (6) | 6 |
| `pretty` | Staten Island (6) | Bronx (6) | 6 |
| `ordinary` | Staten Island (6) | Manhattan (5), Bronx (1) | 4 |
| `ugly` | Bronx (5), Brooklyn (1) | Staten Island (6) | 5 |

The F10.26 task 8 result at k = 3 holds at k = 50. Manhattan passes on `nice` in 1 of 6
cells and on `pretty` in 0 of 6, thus the Manhattan lead is a property of `cool`,
`fancy` and `urban`, not of aesthetic approval in general. The Bronx passes on `pretty`
and `fancy` in 6 of 6 and on `nice` and `ugly` in 5 of 6. The paper reads this in
section 4.2 and in Appendix B (`app:borough-order`).

## What comes next

| Step | Status |
|------|--------|
| NTA mean readouts at 10 images | done: `data/f10/prompts_k10/` |
| ACS, borough, Manhattan and refusal gradients at 50 images | done: `research/figures/f10/more/{ses,partial,manhattan,unable,reliability}.csv` |
| The transfer and the ACS test of the shared direction on the held-out images | done: `research/figures/f10/more/transfer/`, `ses_readout.csv` |
| The shared direction fit at k = 10 | done: `research/figures/f10/more/shared/` |
| The reliability at k = 10 and k = 50 | done: `reliability.csv` |
| The F10.28 word lists from the layer mean of the k = 10 direction | done: `words.csv`, `words_agree.csv`; rho .95 to .97 over the vocabulary, 10 to 15 of the top 20 shared |
| The task-7 residual direction and the steering directions on the k = 10 residuals (`residual_place.npy`, `residual_final.npy`, float16) | done: `steer/`, `dirs_agree.csv`, `dirs_projection.csv`; cosine about .6 at the noun, but the projections agree at r .94 to .96 |
| A steering rerun with the 1,970-image directions (GPU) | open, optional |
| Proposals for the paper | done and applied (2026-11-03): `f10-manuscript-proposals.md` section 7, P51 to P58 (P54 not applied); the paper at `3b28f30`, the table through the manifest (`hop_more_answers.py --table`) |

Rules: keep the 2 estimands separate (the lens readout and the model answer); no causal order;
no socioeconomic gradient named from boroughs; the 591-image numbers stay in the paper until a
proposal replaces them.

## Faults

| Fault | Cause | Fix |
|-------|-------|-----|
| The answers branch of `serve_and_query.sbatch` ran the probe branch too | no `exit` after the answer query | `exit $?` after the call |
| A different capture epoch would break the match with the 591 images | the b2 batch uses the same seed, catalog and epoch | checked `recordedAt` of both batches: 2025-08-01 to 2025-11-04 |

## 2026-11-03: the partial test for the White share

`research/scripts/f10/hop_more_answers.py` 1.2.0 adds the two partials for the White
share to `partial.csv`: `rho_white_given_income` and `rho_income_given_white`,
with their `p` and Bonferroni columns (18 tests in each family, for each k).
The paper audit asked for them, because the body said that income carries the
gradient with race but the test covered the Black share only.

| Family, k = 50 | Pass of 18 | Range |
|----------------|-----------|-------|
| Black share given income | 0 | -.12 to .13 |
| Income given the Black share | 17 | median abs .44 |
| White share given income | 5, all on `cool` | .25 to .37 on `cool`; -.18 to .17 on `nice` and `ugly` |
| Income given the White share | 14 | median abs .30 |

Thus income carries the association with the Black share, and most of the
association with the White share; `cool` keeps a part that income does not
explain. The paper reads this in Results, "One-word answers".
