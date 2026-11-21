# The whatami lens study

Started 2026-10-21. It follows the "what am I looking at?" probe
(`whatami-probe.md`). For each of the four models, we fit two Jacobian lenses
and read every layer of the model along the 40,000 answers of the probe. The
question: which tokens does the model suggest in the earlier layers, before it
writes the answer?

## The plan

| Step | Script | Output |
|------|--------|--------|
| 1. Fit the lenses | `scripts/whatami/lens_fit.py` | `outputs/_monocle/jlens/whatami/<model>/<lens>/lens.pt` |
| 2. Replay the answers | `scripts/whatami/lens_replay.py` | `multirun/whatami/lens_replay/<model>/<lens>/part-*.parquet` |
| 3. Summaries | `scripts/whatami/lens_report.py` | `multirun/whatami/lens_report/<model>/<lens>/*.csv` |

`scripts/whatami/lens_job.sbatch` runs each step as one job on the nissenbaum
B300. `scripts/whatami/lens_common.py` holds the model table, the HF loaders
and the prompt builder.

## The lenses

A Jacobian lens reads the residual `h_l` at layer `l` as
`unembed(J_l h_l)`, with `J_l = E[dh_last / dh_l]` (see
`jacobian-lens-review.md`). The lens depends on the corpus that fits it. We
use the corpora and estimands of the gemma-4-12B lenses in
`lens-registry.md`, thus the four models are fitted the same way:

| Lens | Corpus | Sources | Old gemma lens |
|------|--------|---------|----------------|
| `text` | the first 128 prompts of `corpus_wikitext_1k.json`, raw text, 128 tokens max | every position after the first 16, minus the last | `wikitext` |
| `mm` | the first 128 images of `mm/fit_images.json`, a describe prompt in rotation | the image patches | `mm_all47` |
| `mm_textafter` | the same images | the text after the image | `mm_textafter` |

The target is the last block. Each lens covers every block below it.

- No fit image is in the probe sample: no recording is in both.
- One fit gives `mm` and `mm_textafter` from the same backward passes. The two
  lenses average the gradient over different source positions.
- The replay also reads `logit`, the plain logit lens `unembed(h_l)`, as a
  baseline.

### The lenses of each model

| Model | Layers | d_model | Lenses |
|-------|--------|---------|--------|
| gemma-4-12b | 48 | 3840 | text, mm, mm_textafter |
| qwen3.5-9b | 32 | 4096 | text, mm, mm_textafter |
| llama-3.2-11b | 40 | 4096 | text, mm_textafter |
| llama-3.2-90b | 100 | 8192 | text, mm_textafter |

Llama 3.2 Vision (Mllama) has no `mm` lens. Its decoder sequence holds one
`<|image|>` token, and the image enters through cross-attention layers. Thus
the decoder has no image-patch positions. Each fit image gives only 8
text-after positions (gemma: 13), thus check how well the Llama image lens
converges.

The Mllama text lens leaves out the cross-attention layers (11B: 8 of them,
90B: 20). With no image, HF skips those layers, and their hooks do not fire.
The replay gives no `text` readout at those layers.

## The replay

The replay teacher-forces each stored answer: the served prompt, the answer
tokens, and the end-of-turn token, in one forward pass. The model is causal,
thus each residual is the residual that the model had when it chose the next
token. We chose this over a new generation, because the text then stays the
same as in the probe.

At each position that predicts an answer token, and at each layer, each lens
gives the top 5 tokens with their log-probabilities. It also gives the rank and
the log-probability of the token that the model wrote. See the docstring of
`lens_replay.py` for the columns.

### The check of the replay

`lens_replay.py --check-only` replays 3 images with the logit lens. Measured on
2026-10-21:

| Model | HF prompt tokens | Served | Token count agrees | Final-layer top-1 = answer token |
|-------|------------------|--------|--------------------|----------------------------------|
| gemma-4-12b | 277 | 277 | 30 of 30 | 87.6% |
| qwen3.5-9b | 1,044 | 1,044 | 30 of 30 | 90.0% |
| llama-3.2-11b | 17 | 6,421 (6,404 are image tiles) | 30 of 30 | 81.2% |

The answers were sampled at temperature 0.6, thus a final-layer top-1 below
100% is expected. A value near 0 would show a prompt that does not agree.

## The summaries

`lens_report.py` reads the replay one file at a time and writes, for each
lens:

| File | Content |
|------|---------|
| `layer_top1.csv` | the tokens most often at top 1 at each layer, over all answer positions |
| `layer_top1_pos0.csv` | the same at the position that chooses the first answer token |
| `settle_layer.csv` | the layer from which the written token stays at top 1 |
| `place_latent.csv` | for each place term: the share of answers that say it, and the share where it is at top 1 or top 5 in a depth band |

### Read a place term against the controls

A term is "latent" in an answer when one of its tokens is at top 1 (or top 5)
at any position in a depth band (the early, middle or late third of the layers
below the last). An answer has many positions, thus a stray hit is common. The
first form of the measure (top 5, any layer) gave Toronto 53% and Brooklyn 60%
on 30 gemma answers: it did not separate a NYC term from a control.

Read each NYC term against the control cities (Philadelphia, Boston, Chicago,
London, Paris, Toronto). The probe images show none of them.

The report ran on the 30 answers of the replay check (3 images, logit lens,
2026-10-21). It is a test of the tool, not a result:

| Term | Said | Top 1 in the late third (layers 31-46) |
|------|------|----------------------------------------|
| Brooklyn | 0% | 53% |
| NYC | 0% | 50% |
| York | 50% | 50% |
| Chicago (control) | 0% | 33% |
| London, Paris (controls) | 0% | 0% |

With 3 images, one image decides a share like 33%. Use the full replay.

## The gemma lenses against the old gemma lenses

The new gemma lenses use the corpora of the old ones, thus they must agree.
They agree at the late layers and not at the early layers. Cosine similarity of
the J matrices, 2026-10-21:

| Pair | L6 | L12 | L18 | L24 | L30 | L46 |
|------|----|-----|-----|-----|-----|-----|
| old `mm` (8 layers) vs old `mm_all47`: 2 old fits, same images | 0.55 | 0.71 | 0.50 | 0.99 | 1.00 | |
| new `text` vs old `wikitext1k` (128 vs 1,000 prompts) | 0.61 | 0.15 | 0.78 | 1.00 | 1.00 | 1.00 |
| new `mm` vs old `mm` (8 layers) | 0.07 | 0.02 | 0.06 | 0.96 | 1.00 | |
| new `mm` vs old `mm_all47` (norm ratio) | 0.07 at L8 (x2.1) | | | 0.97 (x1.01) | | 1.00 (x1.00) |

What we know:

1. Below about L24, two fits on the same corpus do not give the same J. The
   two old image fits agree at cos 0.5 to 0.7 there.
2. The new image lens is further from the old one (cos about 0.05), and its
   early-layer norms are 2 to 5 times larger. The text lens does not show
   this. Thus something in the image path changed between the fits. The old
   fits ran in `.venv-nightly`; the new ones run in the vLLM 0.25 venv
   (transformers 5.13). We did not find the old fit logs, thus we could not
   compare the image-token count.
3. This agrees with `r-lens.md`: on an image token at L6, the raw gradient is
   dominated by a component that is not content (x155 against x12.8 for text),
   and the J-lens reads legibly only above about L24 to L30.

**Warning:** do not report a readout below L24 (gemma) as a stable fact. Show
it next to the logit lens and the other lenses, and say that the J estimate is
not reproducible there. The cause of item 2 is open.

## Gemma results (2026-10-21)

The gemma fits and the replay are complete: 10,000 answers, 1,906,181
positions, 4 lenses. Every re-tokenized answer has the served token count,
and the final-layer top 1 is the answer token at 87 to 88% of positions.

### The J-lenses collapse to `<pad>` on gemma

At many layers the fitted lenses put `<pad>` (id 0) at top 1. Measured on
the first 250 answers (all positions):

| Lens | answer token at top 1: L32 / L40 / L44 / L46 | `<pad>` at top 1: L28 / L36 / L42 |
|------|------|------|
| logit | 0.18 / 0.42 / 0.51 / 0.25 | 0.00 / 0.00 / 0.00 |
| text | 0.00 / 0.01 / 0.24 / 0.39 | 1.00 / 0.97 / 0.84 |
| mm | 0.00 / 0.24 / 0.35 / 0.27 | 0.92 / 0.86 / 0.12 |
| mm_textafter | 0.00 / 0.06 / 0.41 / 0.34 | 0.31 / 0.89 / 0.70 |

This is not a fault of the new fits: above L24 their J matrices agree with the
old gemma lenses at cos 0.9995 or more. `r-lens.md` reports the same on a text
fact prompt: the J-lens gives `<pad>` at L18 and L24 and finds the answer at
L42, 6 layers after the logit lens.

**Thus on gemma, the J-lenses read the middle layers worse than the logit
lens does.** Use the logit lens as the main readout for gemma, and the J-lenses
as a comparison. The first-answer-word position shows no legible J-lens token
below L44.

### A latent borough against the true borough

`scripts/whatami/lens_borough_check.py`: the share of answers where the
borough token is at top 1 at any position in the late third (layers 31 to 46),
by the borough of the image. `multirun/whatami/lens_report/gemma-4-12b/borough_check_late.csv`.

| Lens, token | bronx | brooklyn | manhattan | queens | staten_island |
|-------------|-------|----------|-----------|--------|---------------|
| logit, "Manhattan" | 0.107 | 0.117 | **0.470** | 0.021 | 0.006 |
| logit, "Brooklyn" | 0.218 | **0.305** | 0.258 | 0.107 | 0.045 |
| logit, "Bronx" | 0.096 | 0.080 | 0.111 | 0.047 | 0.014 |
| text, "Manhattan" | 0.008 | 0.016 | **0.142** | 0.001 | 0.001 |
| mm_textafter, "Manhattan" | 0.018 | 0.038 | **0.144** | 0.002 | 0.001 |
| mm, "Bronx" | 0.409 | 0.514 | 0.510 | 0.219 | 0.107 |

Answers SAY "Manhattan" in 2.1% of cases. What the table shows:

1. A latent "Manhattan" follows Manhattan images under 3 lenses (logit,
   text, mm_textafter). This is a recognition that the answer does not state.
2. A latent "Brooklyn" is highest for Brooklyn images, but almost as high for
   the Bronx and Manhattan: it acts as a guess for a dense NYC street.
3. Queens and Staten Island images give few latent borough tokens of any
   kind.
4. The `mm` lens "Bronx" does not follow the Bronx. It is an artifact of that
   lens.

## The gemma R-lenses (queued 2026-10-21)

Because the gemma J-lenses do not read the middle layers, we fit R-lenses on
the same corpora: `lens_fit.py --rules lrp`. The LRP rules of
`monocle/rlens_rules.py` change only the backward pass (norm, identity, half,
softmax). The forward values stay bit-identical. The softmax rule needs eager
attention, thus `--rules lrp` loads the model eager.

| Lens | Corpus | Jobs |
|------|--------|------|
| `r_text` | as `text` | smoke 73671, fit 73672 |
| `r_mm`, `r_mm_textafter` | as `mm`, `mm_textafter` | fit 73673 |
| replay of the 3 R-lenses | | 73674 (`--lenses r_text r_mm r_mm_textafter`) |

A replay with `--lenses` writes only the lens files that are missing. The
J-lens and logit files of the first replay stay as they are.

**Warning:** `r-lens.md` found that the gemma R-lens gives the token `0` for
about 95 of 100 readouts below L30 on images. Expect the same here. The R-lens
was legible on text from L30.

### Related methods to try next

| Method | Source | Relevance |
|--------|--------|-----------|
| Tuned lens: an affine translator per layer, fitted by KL to the final output | [Belrose et al., arXiv 2303.08112](https://arxiv.org/abs/2303.08112); code [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens) | a learned bias removes the constant offset that a linear J-lens cannot remove |
| R-lens | [Alignment Forum post](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9) | what `--rules lrp` fits |
| J++ lens: per-cluster Jacobians, LRP backward pass, no non-semantic tokens in the readout | [LessWrong post](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr); code `github.com/safety-research/jpp_lens` | tested on Gemma 4 31B and Qwen3.5-9B |
| J-lens offset as token frequency; per-token z-scoring | [LessWrong post](https://www.lesswrong.com/posts/Qe3jdphWCg9kny3e5/the-j-lens-offset-is-the-model-s-token-frequency-z-scoring) | a cheap readout change that needs no new fit |

## The cost of a fit

The cost is `ceil(d_model / dim_batch)` backward passes for each item. The
backward work does not shrink with a larger `dim_batch`. On the B300, a large
`dim_batch` was slower, not faster. Measured with `--smoke` (one item):

| Model | Text, s/item | Image, s/item | dim_batch (text, image) |
|-------|--------------|---------------|-------------------------|
| gemma-4-12b | 57 | 115 | 16, 16 |
| qwen3.5-9b | 40 | 158 | 16, 8 |
| llama-3.2-11b | 28 | 59 | 16, 16 |
| llama-3.2-90b | 162 | 112 | 16, 16 |

The first `dim_batch` in a smoke includes the Triton compile. Thus its time
is too high.

All eight fits take about 26 GPU hours. QoS nissenbaum gives the group 1 GPU,
thus they run in one chain.

## Faults and their fixes

| Fault | Cause | Fix |
|-------|-------|-----|
| qwen image fit: out of memory at dim_batch 16; text fit 2.4x slower than gemma | transformers runs the gated delta rule of qwen3.5 as a python loop without `flash-linear-attention` | an overlay `.overlays/fla-0.5.2` (fla-core and flash-linear-attention 0.5.2, `--no-deps`) on `PYTHONPATH` in the qwen jobs only; the shared venv does not change |
| `no end-of-turn token '<end_of_turn>'` | gemma-4 ends a turn with `<turn|>` (id 106), not the gemma-3 token | the model table names `<turn|>` |
| A CPU check of the prompts on g3-compute-15 did not end in 20 min | python imports from the NFS venv on that node | run checks as a GPU job, which uses the `/scratch` venv mirror |

**Warning:** the transformers message "The fast path is not available" also
shows with the overlay. It names `causal-conv1d`, which is not in the
overlay. The gated delta rule still comes from `fla`.

## Run status

See the job chain in `logs/whatami/whatami-lens*-<model>-<job>.out`.
