# F10.26 plan: the analysis of 14 prompts

Written 2026-11-02 for a Fable agent. The agent follows this plan from start to end.
The stores of F10.25 are complete and checked. This plan says how to read them, what to
test, and what to deliver.

## 1. Aim

The paper claims that a model forms an aesthetic judgment in 3 parts: a scene code, the
sense of the word, and the application of a standard (F10.24). These results come from 3
prompts. The aim of F10.26 is to find which results generalize over 14 prompts, and which
do not.

The aim is the mechanism of the judgment. It is not harm, and it is not bias. Report each
result that does not generalize as clearly as each result that does.

## 2. Read these first, in this order

| File | Why |
|------|-----|
| `CLAUDE.md` | The language rule (ASD-STE100), the documentation rules, SLURM use |
| `vlm-narratives-docs/f10-judgment-mechanism.md` | Main result, Question 2 (sense), F10.19, F10.23, F10.24, F10.25, Methods, Limits |
| `vlm-narratives-docs/f10-analysis-walkthrough.md` | The 12 analyses and how each claim stands |
| `vlm-narratives-docs/f10-manuscript-proposals.md`, section 2 | The canonical terms and what the code computes |
| `vlm-narratives-docs/facct-claim-map.md` | The claims, and the results that support each claim |
| `vlm-narratives-docs/jacobian-lens-review.md`, `lens-registry.md` | The lens method and the lens files |
| `research/scripts/f10/hop_store.py`, `hop_models.py`, `hop_adjectives.py`, `hop_outcomes.py` | The readers and the F10.24 code that you extend |
| `paper/sec/3_methods.tex`, `paper/sec/4_results.tex` in `artificial-aesthetic/` | The voice and the claims of the paper. Read only, do not edit |

## 3. The premise: J++ Lens and readouts

### 3.1 The lens

A Jacobian lens (J-lens) gives a linear map for each layer. The map moves a residual
vector at layer l to the space of the last layer. The model's own final norm and unembedding
then give logits over the vocabulary. We use the J++ Lens files that `koayon/jpp-lenses`
released, 1 for each model, in `outputs/_monocle/jlens/jpp/{model}/lens.pt`.

| Item | Rule |
|------|------|
| Load | `monocle.jlens_read.load_lens(path)`. It holds `parameters`, `source_layers` and `config` |
| Transport | `lens.transport(h, layer)`, then `_lm_head(_final_norm(.))`. Gemma also applies a softcap of 30 |
| Fit | The lens does not depend on the prompt. A new prompt needs a new readout, not a new fit |
| Security | Load any downloaded lens with `torch.load(..., weights_only=True)` only |

### 3.2 The terms

Use these terms in all text. See `f10-manuscript-proposals.md`, section 2.2.

| Term | Meaning |
|------|---------|
| J-space | A region of activation space. It is never the measurement |
| readout | The lens logits over the vocabulary at 1 token position and 1 layer, as a ranked word list |
| mean readout | The mean of readouts over a group of images (usually NTA means) |
| contrast | A difference of mean readouts (C1 to C5 of the paper) |
| word | 1 entry of the filtered store vocabulary (`vocab.parquet`) |
| answer margin | The pooled log odds of yes against no at the first answer token |

Do not use `read` as a noun. Do not write `Jlens` in prose.

### 3.3 The 2 estimands

| Estimand | Measure |
|----------|---------|
| Embedded conception | The readout of the same text with no image (`text_reads.npy`) |
| Image-caused change | The readout of an image minus the no-image readout |

Keep the 2 estimands separate in each result that uses words.

## 4. The data

### 4.1 The 14 prompts

The form is "Is this {noun} {adjective}? Answer with one word: yes or no."

| Noun | Adjectives |
|------|------------|
| `place` | `cool`, `nice`, `ordinary`, `pretty`, `ugly`, `fancy`, `urban` |
| `scene` | the same 7 |

Each prompt has a store for each model: Gemma-4-31B, Qwen3.5-9B and Qwen3.6-27B (42 stores in all).
The images are the same 591 street images in 197 NTAs for all stores.

| Prompt | Read store | Answer store |
|--------|------------|--------------|
| `place cool` | `data/f10/hop_all/{model}` | `data/f10/hop_logit/{model}` |
| `place {adj}`, other adjectives | `data/f10/hop_all/{model}-{adj}` | `data/f10/hop_logit/{model}-{adj}` |
| `scene {adj}` | `data/f10/hop_all/{model}-scene-{adj}` | `data/f10/hop_logit/{model}-scene-{adj}` |

Do not use `qwen3.6-27b-smoke`. It is an old test store.

### 4.2 A read store (`hop_all.py`)

| File | Shape and content |
|------|-------------------|
| `manifest.json` | `question`, `positions` (tail names), `layers`, `tail_tokens`, `vocab` (word count), `d`, `patches_saved` |
| `vocab.parquet` | `index`, `token_id`, `token`: the store words. Gemma 53,999, Qwen3.5-9B 43,563 |
| `reads/reads-NNNNN.npy` | float32 [50 images, tail, layers, words]: the readout (softcapped for Gemma) |
| `residuals/residuals-NNNNN.npy` | float32 [50 images, tail, layers, d]: the raw residuals |
| `topk/` | the top 30 token ids and logits of each cell, over the full vocabulary |
| `answers/answers-NNNNN.parquet` | `image_idx`, `yes_no`, `yes_no_lc`, `margin`, `top`, `tokens` |
| `text_reads.npy`, `text_residuals.npy` | [tail, layers, ...]: the same text with no image. NaN at `after_image` |
| `prefix_reads.npy`, `prefix_residuals.npy` | the chat tokens before the image |
| `patches/` | only in the 3 `cool` stores `data/f10/hop_all/{model}` |
| `data/f10/hop_all/images.parquet` | `image_idx`, `sample_id`, `nta2020`, `nta_name`, `borough`, `latitude`, `longitude` |

There are 12 shards of 50 images, and the last shard has 41. Open a store with
`hop_store.open_store(name, hop_store.ALL_ROOT)`.

The tail positions keep the F10.5 names. The other tokens have the names `t02` and so on.

| Name | Token |
|------|-------|
| `after_image` | the token that closes the image |
| `first_text` | `Is` |
| `place` | the noun: ` place` or ` scene` |
| `after_place` | the adjective |
| `after_cool` | `?` |
| `final` | the last token before the answer |

Layers: Gemma L0 to L56 (15 layers), Qwen3.5-9B L0 to L28 (8), Qwen3.6-27B L0 to L60 (16), each 4th layer.
Compare models at matched relative depth, as `hop_models.py` does.

### 4.3 An answer store (`hop_logit.py`)

| File | Content |
|------|---------|
| `answers.parquet` | `image_idx`, `lp_` of the 8 answer forms, `margin` (pooled over `Yes`, `yes`, ` Yes`, ` yes` and the same for no) |
| `metrics/` | npz of `lens`, `plain` [images, tail, layers, 8 forms], `rms`, `mu`, `sd`, `js`, `js_capped`, `kl_capped`, `agree` |
| `text.npz` | the same metrics for the no-image text |
| `last/` | the last-layer residuals of each shard, and its answers |
| `uncapped/` | the readouts before the softcap. Gemma only, because Qwen has no softcap |

The outcome of an answer is yes, no or unable (D167). The first token decides it, and
`hop_outcomes.py` applies the rule. The margin of an `unable` image is empty.

### 4.4 Facts that constrain the design

1. **The text before the changed word is the same.** The image comes first and the attention
   is causal. Thus, for one image and one model, the residuals up to the noun are equal in all
   7 stores with the same noun. The check of F10.25 shows a gap of 0.0. The residual at `place`
   is 1 vector, and 7 margins depend on it. This is the cleanest test of a shared code.
2. **The patches are shared.** The patches of all 14 prompts equal the patches of the `cool`
   store. The mean-patch kernel of F10.9 is in `data/f10/hop_free/{model}/scene_kernel_1.npy`.
3. **The yes share differs much.** At the first look, Gemma says yes to .01 to .05 of the images
   for `ugly` and `fancy`, and to .89 to .98 for `urban` and `scene ordinary`. A rare outcome
   gives a small range. Compare ranks, use the continuous margin, and report the yes share
   next to each result.
4. **Margins have different scales.** The margin SD differs about 10 times between models
   (F10.24). Do not compare raw margins across models.
5. **Gemma refuses `nice`.** Use the 3 outcomes. Report each `unable` count. Do not put a
   margin on an `unable` image.

## 5. The tasks

Do the tasks in this order. Each task gives 1 table or 1 figure, and 1 short paragraph in the doc.

### Task 0. Reproduce F10.24 first

Before you extend the code, run the F10.24 analyses on the 3 old prompts with your new code.
Your values must equal the tables of F10.24. If they do not, stop and find the cause.

Then make the code take a prompt list, not an adjective list. A prompt is an
(adjective, noun) pair. Put 1 function in 1 place that gives the store name of a prompt.

### Task 1. Outcomes and refusals

| Item | Plan |
|------|------|
| Question | How does each model answer each prompt? |
| Method | `hop_outcomes.py` for 14 prompts and 3 models |
| Tests | Chi-square of borough against outcome for each prompt and model, Holm over the 42 rows. Chi-square of noun against outcome for each adjective and model |
| Output | yes, no and unable shares, by prompt and model, and by borough |

### Task 2. Mechanism core

Repeat the F10.24 table "What holds in all 3 models" for each of the 14 prompts.

| Measure | Method |
|---------|--------|
| Scene code | Mean-patch R2 of the margin, out-of-fold ridge, 5 folds by NTA |
| Gain over the mean patch | At `place`, at the adjective, at `?` |
| Verdict band | The F10.20 rule on the margin |
| Sense band | The F10.20 rule at the adjective |
| Largest axis at the adjective | SVD of NTA mean readouts at mid depth: the share of variance and the words at each end |
| Embedded words | The count of no-image top words in the top 100 at the adjective |

Give the result as a grid of 14 prompts by 3 models. A result generalizes when it holds in all
42 cells. Say for each result how many cells it holds in.

### Task 3. The sense of each word

`cool` has a temperature-to-social axis. `nice` and `ordinary` have no such axis (F10.24).

| Item | Plan |
|------|------|
| Question | Does the image select a sense of the other words? |
| Method | The SVD and sense method of F10.15 and F10.17 at the adjective, for each prompt |
| Candidates | `fancy` (the noun or verb "fancy", or "luxury"), `pretty` (the adverb "pretty", as in "pretty good"), `nice` (the city Nice), `urban` (urban music or style, or the city type). The noun `scene` also has senses: a social scene, a crime scene, a film scene |
| Rule | Name a sense axis only when its end words name 2 senses of the word. Do not name an axis from 1 list |
| Test | r of the sense score with the margin, t test, Holm over the prompts |

### Task 4. `pretty` against `ugly`

| Item | Plan |
|------|------|
| Question | Is `ugly` the reverse of `pretty`? |
| Method | r of the 2 margins, on images and on NTA means. Cosine of the 2 margin directions in residual space at `place` (Task 7). Cosine of the 2 contrasts (C3) at the adjective |
| Reading | A full reversal gives r near -1 and a cosine near -1. `ordinary` against `cool` gave only -.40 (F10.24). Report how far from -1 each value is |

### Task 5. `urban`

The author holds that `urban` is an aesthetic word too. It is also a sign that the model sees
an image or a borough as urban. At first look, the model does not see the mean Staten Island
or Queens image as urban.

| Item | Plan |
|------|------|
| Question 1 | Does the `urban` margin follow the borough? Give the borough order, and the t of each borough against the rest |
| Question 2 | Does the `urban` margin explain the borough pattern of the other adjectives? Partial r of each adjective margin with the borough, given the `urban` margin. Use NTA means |
| Question 3 | Does the `urban` direction match the social end of the `cool` sense axis? F10.17 found r .70 and .62 of the sense score with the `urban` probe |
| Rule | Do not call `urban` a control. Report what it shares with the aesthetic words and what it does not |

### Task 6. `place` against `scene`

| Item | Plan |
|------|------|
| Question | Does the noun change where or how the judgment forms? |
| Margin | For each adjective and model: r of the 2 margins over images, and the paired t of the NTA means |
| Mechanism | The Task 2 measures for the 2 nouns, side by side |
| Readout at the noun | The words at `place` that differ between the 2 nouns, with the 2 estimands kept separate |
| Boroughs | Does the borough sign (Task 8) change with the noun? |

### Task 7. A shared aesthetic direction

The author asks for this task. It is new.

| Item | Plan |
|------|------|
| Question | Is there 1 direction, or a few, in the residual space that gives the judgment in all 14 prompts? |
| Step 1 | At `place`, fit 1 ridge decoder of each margin from the same residual (fact 1). Use 5 folds by NTA, and z-scored margins |
| Step 2 | Give the cosine matrix of the 7 decoder weights for each noun and model. Give the out-of-fold R2 of a rank-k fit for k = 1, 2, 3 (reduced-rank regression). If rank 1 or 2 keeps most of the R2, a shared direction exists |
| Step 3 | Leave 1 prompt out: fit the shared direction on 13 prompts, and test it on the 14th. Do this at the adjective, at `?` and at `final` also, where the text differs |
| Step 4 | Put the shared direction through the J++ Lens to get its words. Report the words at each end. Say if they are aesthetic, valence or status words |
| Step 5 | Compare the direction with the `cool` sense axis, the `urban` direction and the valence of the words. Report the cosines |
| Sign | `ugly` and `ordinary` can load with a negative sign. A shared direction with signs is still 1 direction |
| Limit | A shared direction can be general valence. Test this: remove the `nice` direction, and see what stays |

### Task 8. Boroughs and place probes

| Item | Plan |
|------|------|
| Manhattan sign | t of Manhattan against the rest for 14 prompts and 3 models, Holm over the 42 tests. Report the full family. F10.24 reported Holm over 9 tests only |
| Borough order | The order of the 5 boroughs for each prompt. Count the prompts with the same order |
| Stability | The 40 random NTA splits of F10.17, for each prompt |
| Word overlap | Top-50 Jaccard of the C3 words across prompts. F10.24 found about 0 between adjectives |
| Place probes | The F10.18 place probes. If a probe needs a new GPU readout, do not run it. Write it as an open item |

## 6. Compute and I/O rules

**Warning:** NFS failed in F10.25 because of a bad read pattern. Do not read a large file
through a memory map in many strided passes.

| Rule | Detail |
|------|--------|
| Jobs | CPU SLURM jobs only on g3: `--clusters=g3 --partition=g3 --account=nissenbaum --qos=g3 --requeue`, with no GPU. Use the header of `research/scripts/f10/analyze_logit.sbatch` |
| Concurrency | At most 4 jobs at the same time |
| Exclude | `--exclude=g3-compute-16,g3-compute-10,g3-compute-03,g3-compute-04,g3-compute-02` |
| Reads | Load each shard file once, in sequence (`np.load` with no `mmap_mode`), and take all the cells that you need from it |
| Cache | First, make 1 summary for each store in 1 pass: NTA mean readouts at the 6 named positions, and the residuals at those positions. Put it in `data/f10/prompts/{store}/`. All later steps read only the summary |
| Session node | Use it only for small reads and tables. A wide `find` on NFS stalls. Foreground `sleep` is blocked |
| Python | `.venv-mllmsci-vllm025cu129/bin/python` |
| Tests | Add a test in `tests/` for each new function that computes a value, for example the prompt-to-store map |

A gemma read store is about 48 GB, and the 42 stores are about 1.6 TB. A full pass over all
stores is long, thus make the cache once.

## 7. Statistics rules

- Use the textbook default tests: t, paired t, F, chi-square, Pearson r, Holm. Do not use elaborate methods.
- The unit is the NTA mean (df 196), unless a test needs images.
- Fold by NTA for each decoder. Report out-of-fold values only.
- Give the size of each Holm family. A family is the set of tests that support 1 claim.
- Report n for each test. The Gemma `nice` tests use the yes or no images only.
- Do not split the NTAs by borough as a held-out set. The sample is stratified over all 197
  residential NTAs, thus C3 is unbiased.

## 8. Deliverables

| Item | Place |
|------|-------|
| Code | `research/scripts/f10/hop_prompts.py` (a new name is acceptable), with a docstring in STE |
| Tables | `research/figures/f10/prompts/*.csv` |
| Figures | `research/figures/f10/prompts/*.png` and `.pdf`, each with a JSON sidecar. House style: `~/style/proper-plotting.md`, `research/notebooks/cvpr/_style.py`. No caption text in the image. View each saved image |
| Report | A new section "14 prompts: results (F10.26)" in `f10-judgment-mechanism.md`, with the format of F10.24: what holds in all cells, what does not, limits |
| Walkthrough | A new section "What F10.26 changes" in `f10-analysis-walkthrough.md` |
| Claim map | Update each row of `facct-claim-map.md` that F10.26 makes stronger or weaker |
| Paper edits | A new section in `f10-manuscript-proposals.md`: an itemized list of proposed edits, each with the file, the line, the old text, the new text and the result that supports it |
| Runbook | A decision row for each design choice. Compute the next D number from `facct-runbook.md` before you write it. Another session also writes this file |

The paper edits use the voice, syntax and terms of the paper (section 3.2). The paper prose
does not use STE. All other text uses STE.

Acceptance:

1. Task 0 reproduces the F10.24 values.
2. Each task has its table or figure and its paragraph.
3. Each "generalizes" claim gives the count of cells, of 42, in which it holds.
4. The paper edit list holds no edit that a table does not support.

## 9. Do not

- Do not edit the paper repo. Propose the edits only.
- Do not commit or push. The author asks for each commit.
- Do not write to `vault/context/`.
- Do not send GPU jobs, and do not make new readouts.
- Do not change or delete a store under `data/f10/hop_all/` or `data/f10/hop_logit/`.
- Do not claim a causal order. The prompt is fixed for each store, thus the image is the only source of variation in a store.
- Do not name a socioeconomic gradient from boroughs. F10 has no income, education or race test.

## 10. Open items to know

| Item | State |
|------|-------|
| Qwen3.6-27B `cool` verdict band | It reaches r .92 only. Cause not found. Task 2 shows if `scene cool` has the same fault |
| 3-way borough overlap | Not done in F10.24 |
| `Subject` and `Unable` | They are not in the store vocabulary |
| Answer step versions | 11 answer stores come from `hop_logit.py` 1.2.0, and 22 from 1.2.1. The output is the same |
| Holm family of the Manhattan sign | "All 9 pass" holds over 9 tests only. Task 8 gives the full family |
