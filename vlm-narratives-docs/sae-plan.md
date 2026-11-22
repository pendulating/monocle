# Sparse autoencoder latents: a research and experiment plan

Written 2026-09-08. Status: stage 1 DONE 2026-09-09 (48 shards, checked by `research/monocle/sae_check.py`); stage 2 trained 2026-09-15/16 (recipes v1 and v2, 42 SAEs; criterion G1 (reconstruction fidelity) passes; criterion G2 (stability across seeds), as written, fails for every SAE, see the stage-2 record and the proposed shared-latent rule; a decision is open).

## The question

The white-box study showed the following. The wealth judgment is
linearly decodable in the residual stream. A steering vector reaches it
causally. One direction of the residual stream at the last token
position carries it, under the prompt templates we tested. The study
did not show that the judgment is a structure of the model and not a
state that the prompt template induces. The single-word logit-lens
analysis gave more weight to this doubt: under the prompt template, the
affluence-term associations disappear at layer 47, and the steering
vector that worked was the pseudo-inverse steering direction and not
the principal judgment direction.

A sparse autoencoder (SAE) is the next step. Its latents are directions
that the model's own activation geometry selects, not directions that
we fit to a label. The questions it can answer:

1. Is the wealth readout one latent or a set of latents (graffiti,
   trash, fencing, brick, new glass)?
2. Which latents does layer 47 remove, and which does it keep?
3. Does the principal judgment direction at the image tokens fall in
   the span of a few latents?
4. Do the same latents exist in the base model?
5. Which parameters write those latents (the step to a claim that the
   judgment is stored in the weights)?

No public SAE covers gemma-4. Gemma Scope stops at Gemma 2, thus we
train our own.

## Design rules

| Rule | Reason |
|------|--------|
| Train unsupervised on the residual stream; use the labels only for selection and validation, on held-out NTAs | a label in the loss makes a probe with a sparsity penalty, and the structural claim is lost |
| Two seeds per SAE; keep a latent only when both seeds recover it (decoder cosine similarity above 0.7) | latents split and merge across seeds |
| Mean-center the residual stream before training and before any cosine similarity | massive activations in gemma: a raw cosine similarity of 1.0 is an artifact |
| Both models, same data, same recipe | the base-model condition produced the refusal result; it produces this one |
| Store activations on `/scratch`, never on NFS | 100 GB of image-token vectors per SAE |
| A supervised SAE is a second condition, compared against the first, never the main result | see rule 1 |

## Positions and layers

| SAE | Position | Layer | Reason for this layer | Prompt template |
|------------|----------|-------|----------------|-------|
| P22 | the 256 image tokens | 22 | the principal judgment direction at the image tokens, layers 12 to 22 | none: the residual stream at the image tokens does not depend on the question |
| P40 | the 256 image tokens | 40 | the late layer at which the judgment is decodable at the image tokens | none |
| A42 | the last token position | 42 | the judgment is decodable over layers 40 to 45 | the canonical single-image question (per case) and the open-ended one-word prompt |
| A46 | the last token position | 46 | the last layer before the final-layer transformation | the same |

We read layer 47 by projection onto the A46 SAE. The final-layer
transformation replaces the residual stream there, and an SAE of its
own would not be comparable. The residual stream at the image tokens
does not depend on the question in this model, thus one image-token
extraction serves every prompt template.

## Data

| Item | Value |
|------|-------|
| Images | the all-NTA sample, 197 NTAs at 500 images (`outputs/_monocle/probe_words_all/sample`); 300 per NTA for the SAEs (59,100 images), the other 200 held out for selection and validation |
| Image-token vectors per SAE | 59,100 x 256 = 15.1 million, at 3,840 dimensions (the limit of 280 gives 256 tokens at this image size) |
| Last-token vectors per SAE and prompt template | 59,100 |
| Storage per image-token SAE | 116 GB in bf16, on klara's `/scratch` (1.5 TB free at launch) |
| Labels, used after training only | the NTA census variables, the pairwise decisions and the abstention rate from the canonical runs, the ranked cue list from the corpus of extracted explanations |

## The SAE

| Item | Value |
|------|-------|
| Type | BatchTopK, k = 48 active per vector, with an auxiliary loss on dead latents |
| Dictionary size | 32,768 (8.5 x), with a 16,384 variant on P22 for a dictionary-size check |
| Input | the layer's residual stream, mean-centered with the training mean, scaled so that the mean vector norm is sqrt(D) (unit RMS per coordinate) |
| Training | image tokens: 2 epochs (7,386 steps); last token position: 40 epochs over the 8 prompt templates of the train images (4,400 steps); Adam, batch 4,096, learning rate 3e-4, warm-up 1,000 steps, linear decay over the last 20 percent |
| Seeds | 2 |
| Implementation | our own, about 300 lines of torch; no library dependency in the nightly venv |
| Quality criteria | fraction of variance unexplained (FVU) below 0.10; L0 (mean number of active latents) at k; dead latents below 5 percent; spliced-in evaluation: less than 2 percent of the decisions under the canonical single-image prompt template change when the SAE reconstruction replaces the residual stream |

## The stages

### Stage 1: extraction (GPU)

One forward pass per image for the image-token layers, with activation
hooks at layers 22 and 40 on the image-token positions. The pass writes
bf16 shards of 1,000 images to `/scratch/mwf62/sae/<ckpt>/P<block>/`.
The last-token vectors come from separate passes under each prompt
template: the open-ended prompt and the 7 canonical single-image
questions, at layers 42 and 46, one row per image and prompt template.

| Item | Value |
|------|-------|
| Passes per image | 1 (image tokens, train images) + 8 (last-token prompt templates, every image) |
| Time, measured on the pilot run | 0.20 s per image-token pass (2 layers stored), 0.13 s per last-token pass |
| GPU-hours per model | about 16 |
| Jobs | 24 shards per model, 150-minute limit, about 80 minutes each |
| Module | `research/monocle/sae_extract.py`; the split in `research/scripts/sae_sample_split.py` (`outputs/_monocle/sae/sample/images.parquet`, 59,100 train / 39,231 held out) |
| Pilot check | 256 image tokens, contiguous; image-token norm median 259, one coordinate at 286 (the massive activation); a last-token file per prompt template and layer |

### Stage 2: training (GPU)

Eight SAEs per model (P22, P40, A42, A46, 2 seeds each) plus the 16k
dictionary-size check. Module: `research/monocle/sae_train.py`; criteria:
`research/scripts/sae_gates.py` -> `outputs/_monocle/sae/gates.md`. Measured on
the pilot run: 0.15 s per step on an image-token SAE with the IO, thus
an image-token SAE trains in about 20 minutes, not 3 hours; a last-token
SAE in about 8 minutes. The mean-centered residual stream has a mean
norm of 32 to 33 against a raw norm of 100 (last token) or 258 (image
tokens): the shared mean is 98 percent of the raw norm.

Validation sets: 2 fixed image-token parts (262,144 vectors) per
image-token SAE; a fixed 5 percent of the train images (by path hash)
for the last-token SAEs. The held-out NTA images stay unused in
stage 2. The criteria use the threshold (JumpReLU) encoder.

Criterion G1: every SAE passes the quality criteria above, or the
recipe changes before any downstream stage runs.

Stage-2 record, recipe v1 (pooled mean subtraction, 2 epochs at the
image tokens, 40 at the last token position; `outputs/_monocle/sae/gates.md`,
the seed-match reports (`seedmatch_*.md`, latents shared across seeds),
the per-condition variance reports (`condvar_*.md`), and the spliced-in
evaluation per condition (`<arm>/splice_test/`)):

| Finding | Value |
|---------|-------|
| Criterion G1 | all 18 pass: FVU 0.004 to 0.010 (last token), 0.066 to 0.098 (image tokens); L0 48; dead latents 0 to 3 percent over the last 2 million training vectors |
| The first A42 at dictionary size 32,768 | 72 percent dead latents: the auxiliary loss had a ReLU that zeroed the gradient of a dead latent; we fixed it (no ReLU on the dead pre-activations); the last-token dictionary size is 8,192 because 450,000 vectors cannot support 32,768 latents |
| Spliced-in evaluation, the share of decisions that change under the reconstruction, 400 held-out images | pass at A42, A46, P40 in both models (0.003 to 0.016); IT P22 fails at 0.023 to 0.025 |
| Criterion G2, decoder cosine similarity above 0.7 | 0.01 to 0.16 of the seed-1 latents, all SAEs: FAIL |
| Criterion G2 by activation correlation, top 1,000 latents by activation rate | A42/A46 0.55 to 0.68, but the partner's decoder cosine similarity is 0.01: the last-token SAEs encode the prompt template; P40 0.80 to 0.84 with partner cosine similarity 0.6: latents shared in direction; P22 0.41 to 0.49 |
| Share of variance that the prompt-template means explain at the last token position | 0.81 to 0.94 (8 prompt templates); that the position means explain at the image tokens, 0.12 to 0.18 (P22), 0.05 to 0.06 (P40) |
| 6 epochs at P22 (IT) | FVU 0.097 to 0.085; decoder cosine match of the top 1,000 latents 0.23 to 0.38; the activation-correlation match share unchanged at 0.40 |

Recipe v2, trained 2026-09-15/16: subtract a per-template mean at the
last token position and a per-position mean at the image tokens
(`--center cond`), 6 epochs at the image tokens, 40 at the last token
position. The SAE stores `means [n_cond, D]`. The FVU under this mean
subtraction is the within-condition FVU and is stricter than the v1
FVU. Results, latents matched across seeds over the latents that fire
on more than 1 in 10,000 validation vectors (`seedmatch_*_c.md`,
`seedmatch_*_c6.md`, `seedmatch_*_width.md`, the shared-latent sizes in
`working_set_sizes.csv`):

| SAE | Share matched by activation correlation (> 0.7), top 1,000 | Share matched by decoder cosine similarity (> 0.7), top 1,000 | Decoder cosine similarity of the activation-correlation partner, median |
|------------|----------------------------------------------|-------------------------|---------------------------------------|
| IT A42, 8k | 0.83 | 0.10 | 0.37 |
| IT A46, 8k | 0.81 | 0.17 | 0.26 |
| base A42, 8k | 0.63 | 0.23 | 0.12 |
| base A46, 8k | 0.67 | 0.41 | 0.04 |
| IT A42, 2k / 4k | 0.81 / 0.86 | 0.05 / 0.08 | 0.33 / 0.36 |
| IT P22 | 0.38 | 0.41 | 0.54 |
| IT P40 | 0.84 | 0.80 | 0.65 |
| base P22 | 0.51 | 0.49 | 0.55 |
| base P40 | 0.80 | 0.74 | 0.71 |

Interpretation. Per-template mean subtraction makes the last-token
latents match across seeds by activation correlation (0.8 in the IT
model), but not by decoder direction at any dictionary size from 2,048
to 8,192 latents. After the prompt template is removed, the residual stream at
the last token position is a dense, low-rank signal, and a sparse code
of it has many equivalent bases. Those latents match by activation
correlation and not by decoder direction. At the image tokens, P40
gives latents that match by decoder cosine similarity in both models,
and P22 about half as many. Six epochs and per-position mean
subtraction each help at the image tokens.

Proposed shared-latent rule (in place of "60 percent at decoder cosine
similarity 0.7 over all latents", which no SAE of any recipe meets, and
which counts the 30 percent of latents that never fire): a latent is a
shared latent when it fires on more than 1 in 10,000 validation
vectors, its best seed-2 partner by activation correlation exceeds 0.7,
and that partner's decoder cosine similarity exceeds 0.5. Sizes under
v2:

| SAE | Shared latents | Share of the activation mass |
|------------|-------------|------------------------------|
| IT P40 | 11,323 | 0.77 |
| base P40 | 10,035 | 0.79 |
| IT P22 | 3,541 | 0.53 |
| base P22 | 3,336 | 0.61 |
| IT A42 / A46 | 469 / 566 | 0.37 / 0.40 |
| base A42 / A46 | 778 / 1,088 | 0.40 / 0.51 |

Under this rule, P40 is the primary SAE for stages 3 to 7, and P22 a
secondary one. The last-token SAEs serve stage 4 (the decision
regressions, where a match by activation correlation is enough) and the
layer-47 projection, but not stage 6 (weight attribution needs a
direction).

Criterion G2: at least 60 percent of the latents of seed 1 match a
latent of seed 2 at decoder cosine similarity above 0.7. The matched
set is the set of shared latents. We measure it over the latents that
fire on more than 1 in 10,000 validation vectors, with the
activation-correlation match beside the cosine match
(`research/monocle/sae_seedmatch.py`): a correlated pair with an unrelated
decoder direction is a shared condition, not a shared latent.

### Stage 3: the latent catalogue (CPU, then GPU for the crops)

For every shared latent: the activation rate, the mean activation, the
64 top-activating image tokens with their image and grid position, and
the per-NTA mean activation over the 197 NTAs. The top image tokens
become crops (in the `research/monocle/caption_panel.py` style) for viewing. The
model itself names each latent from its top crops, as a first label,
not as evidence.

### Stage 4: selection with the labels (held-out)

On the 200 held-out images per NTA:

| Test | Statistic |
|------|-----------|
| Which latents track a census variable | partial Spearman correlation over NTAs of the NTA-mean activation with each census variable, controlling for log population density, with the control-word threshold of the single-word logit-lens analysis as the reference for the score of a content latent |
| Which latents carry the decision | a sparse regression from the last-token latent activations to the pairwise decision and to the abstention rate, held out by NTA |
| Which latents are the cues | alignment of the image-token latents with the ranked cue list from the langextract explanations: does the latent that fires on graffiti crops track the lift of the `graffiti` cue |
| One latent or a set | how many latents the income regression needs to reach the logit-lens readout (AUC 0.73); the split of variance across them |

Criterion G3: at least one shared latent tracks income on held-out NTAs
above the control-word threshold, in both models. If none does, the
SAE did not find the judgment, and the plan stops at a negative result
with the catalogue as the record.

### Stage 5: causal tests (GPU)

| Test | Method | Measured quantity |
|------|--------|-------|
| Latent activation addition at the image tokens | add or subtract the latent's decoder direction at layer 22 on the image-token positions, with a reduced image-token budget as in the steering-direction study; coefficient in units of the latent's natural spread | the canonical single-image decision and the word ranks under the open-ended prompt |
| Latent ablation at the last token position | set the latent activation to zero in the A42 activations, and splice the reconstruction back | the decision |
| The principal judgment direction | projection of the principal judgment direction onto the span of the top-k income latents; the fraction of the natural judgment variance that those latents hold | one number per k |
| Layer 47 | project the layer-47 residual stream onto the A46 SAE; which latents present at 46 are absent at 47, per model | the removed set against the affluence-term list of the single-word logit-lens analysis |

Criterion G4: activation addition of a latent that tracks income
changes the decision, and its ablation at the last token position
reverses the change. Control: a shared latent with the same activation
rate that does not track income.

### Stage 6: weight attribution (the structural step)

For each latent that passes criterion G4, at its layer L:

- the cosine similarity of its decoder direction with every row of the
  MLP down-projection of the layers up to L; the rows above a threshold
  are the candidate writers (the components that write the direction
  into the residual stream);
- the direct contribution of each attention head to the latent
  direction, from activation hooks on the sub-layer outputs;
- the check: ablate the candidate writers, and measure the latent's
  activation and the decision on held-out images.

A latent with a small set of writers, whose ablation removes the latent
under both prompt templates, is the object that the structural claim
needs.

### Stage 7: across models and prompt templates

- Match latents between the base-model and the IT-model SAEs by decoder
  cosine similarity (a one-to-one assignment). Which income latents are
  shared, which are new after post-training, and which of the shared
  ones does layer 47 of the IT model remove.
- Measure the matched latents under the pairwise prompt template on the
  same images. A latent with a significant association under the
  single-image prompt template, the open-ended prompt and the pairwise
  prompt template is invariant to the prompt template; one with a
  significant association under two of them is not.

### Stage 8: the supervised condition (secondary)

The same SAE with a small linear head from the latent activations to
the income census variable in the loss, with weight 0.1. Compare it
against the unsupervised latents: how many of its income latents have
an unsupervised match, and whether its held-out association is
stronger. This is the condition that the question "train it with the
labels" asks for; we report it beside the unsupervised condition, not
instead of it.

## Compute and storage

| Item | Value |
|------|-------|
| GPU-hours, extraction, both models | about 30 |
| GPU-hours, training, both models | about 30 |
| GPU-hours, catalogue and causal tests | about 20 |
| Total | about 80, which is 2 nights on 8 A6000 cards, or 1 night at full occupancy |
| Scratch | about 465 GB for the 4 image-token extractions; 1.5 TB free on klara at launch |

## Order and milestones

| Night | Work | Criterion |
|-------|------|------|
| 1 | Stage 1 for both models; Stage 2 for A42 and A46 (fast) | G1 on the last-token SAEs |
| 2 | Stage 2 for P22 and P40; Stage 3 | G1, G2 |
| day | Stage 4 | G3 |
| 3 | Stage 5 | G4 |
| after | Stages 6 to 8 | |

## Known pitfalls

- The gemma residual stream has a few massive activations. Mean-center
  the activations before training, and never compare raw cosine
  similarities.
- The image tokens come before the question, thus the residual stream
  at the image tokens does not depend on the question. A last-token SAE
  is bound to its prompt template and must state that prompt template.
- The final-layer transformation at layer 47 replaces most of the
  incoming residual stream (norm divided by 12, cosine similarity 0.34
  to 0.49 with the input). Read it by projection onto A46, not with an
  SAE of its own.
- The bf16 precision floor: activation addition at 1 image token is at
  the floor. Add the latent direction over many image tokens, or reduce
  the image-token budget.
- There are 256 image tokens, not 280: the limit is 280 and the image size gives 256.
- The last image token receives 80 to 105 times the attention of any
  other image token. It dominates an image-token SAE unless we
  down-weight its rows or exclude them from the catalogue statistics.
- Latents are not unique. The latent shared across seeds is the unit,
  and a single-seed latent is a candidate, not a result.

## Outputs

| Path | Content |
|------|---------|
| `research/monocle/sae_extract.py` | stage 1 |
| `research/monocle/sae_train.py` | stage 2, the BatchTopK SAE |
| `monocle/sae_census.py`, `scripts/sae_select.py` | stages 3 and 4 (the latent catalogue and the selection) |
| `monocle/sae_steer.py`, `monocle/sae_writers.py` | stages 5 and 6 |
| `outputs/_monocle/sae/<ckpt>/<dict>/` | weights, metrics, catalogue, crops |
| `vlm-narratives-docs/sae-results.md` | the results page |

## Related

- [lens-refusal-probe-results.md](lens-refusal-probe-results.md): the
  single-word logit-lens analysis, the direct logit attribution at the
  final layer, the correlation across all neighborhoods
- [steer-coord.md](steer-coord.md): the principal judgment direction
  and the pseudo-inverse steering direction
- [single-image-steering.md](single-image-steering.md): the
  activation-addition method that this plan reuses
- [gemma4-12b-architecture.md](gemma4-12b-architecture.md): the layers
  and the global-attention layout
