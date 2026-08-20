---
title: "Jacobian-Lens Monocle — Depth-Resolved Patch Readout"
category: concept
created: 2026-07-23
updated: 2026-07-24
tags:
  - concept
  - interpretability
  - jacobian-lens
  - gemma-4
  - monocle
  - workspace
---

# Jacobian-Lens Monocle

Extends [[concept-monocle]] with a **layer axis** using the Jacobian lens
(Anthropic, "Verbalizable Representations Form a Global Workspace in Language
Models", transformer-circuits 2026-07; reference code vendored at
`sub/jacobian-lens`, installed in `.venv-nightly`). The lens transports a
residual at layer l into the final-layer basis with the corpus-averaged
input-output Jacobian, then decodes with the model's own unembedding:

    lens_l(h) = unembed(J_l @ h),  J_l = E[dh_final / dh_l]

Applied at every image-patch position of encoder-free gemma-4-12B, this shows
**what each patch is disposed to make the model say at every depth** — a
word-cloud stack through the network, rendered as per-layer overlays, a depth
GIF, and an interactive HTML layer scrubber.

## Fitted lens

| Property | Value |
|---|---|
| Checkpoint | `outputs/_monocle/jlens/gemma4_12b_lens.pt` (225 MB fp16) |
| Corpus | 128 WikiText-103 prompts (`corpus_wikitext_1k.json`, prepared on the login node — compute nodes are offline) |
| Source layers | {6, 12, 18, 24, 30, 36, 42, 46}, target 47 (final) |
| Cost | ~180 s/prompt at `dim_batch=16` (peak 40.6 GiB; 32 OOMs on A6000) — ~6.9 GPU-h across 4 shards, merged with `JacobianLens.merge` |
| Fit driver | `monocle/jlens_fit.py` (`--smoke` calibration / `--shard i/n` / `--merge`) |

`jlens.from_hf` needed **zero adapter code**: it auto-detects the
`model.language_model` layout inside `Gemma4UnifiedForConditionalGeneration`
(48 layers, d_model 3840) and handles the 30.0 logit softcap.

## Readout path (`monocle/jlens_read.py`)

One multimodal forward with jlens's own `ActivationRecorder` forward hooks on
`lens_model.layers` — the SAME modules the fitting estimator hooks — so the
layer convention matches fitting **by construction**. Fitted layers:
`unembed(J_l @ h_patch)`. Final layer: `out.logits` verbatim. Scoring /
pooling / rendering are unchanged monocle machinery, applied per layer.
CLI: `python -m monocle.cli ... --jlens CKPT [--layers 6,12] [--gif] [--scrubber]`;
outputs `<id>.jlens.parquet` (+`layer` column), per-layer PNGs,
`<id>.depth.gif`, `<id>.scrubber.html`.

## Validation (`monocle/jlens_validate.py`, jobs 148170 / 256790)

- **Stage A** — unembed(block-47 hook output) vs `out.logits` at patch
  positions: max diff **0.00e+00** (exact).
- **B1 text** (BOS-prefixed "capital of the state containing Dallas"):
  J-lens reads *Seville/Spain/France* at L30 → *Wisconsin/Texas/Illinois* at
  L36 → *Texas* at L42; plain logit lens is `<unused*>` garbage until L36.
  Replicates the paper's headline on gemma-4-12B.
- **B2 depth-resolved quadrant test** (synthetic: "DOG" text TL, red fill BR):

| Layer | dog TL mass | red right-half |
|---|---|---|
| 6–30 | 0.00–0.25 (absent) | 0.32–0.98 (from **L12**) |
| **36** | **0.95 (ignition)** | 0.94 |
| 42 / 46 / 47 | 0.91 / 0.98 / 0.91 | 1.00 / 1.00 / 0.99 |

**First scientific readout:** color localizes **early** (L12); reading
rendered *text* as a verbalizable token ignites **late and sharply** (L36 =
75% depth) — the paper's workspace-onset "ignition", spatially resolved on an
image. L47 = 0.91 exactly matches the direct-logits monocle validation.

## ⚠️ Gotchas (both bit during bring-up)

> [!danger] Do not read HF `output_hidden_states` for block outputs
> In transformers 5.x the tuple's final entry is NOT the bare final-block
> output — unembedding it double-applies the final norm (stage A failed by
> 5e+01). Record block outputs with forward hooks on the layer modules
> (jlens's `ActivationRecorder`), which is what fitting does.

> [!danger] `from_hf(force_bos=True)` mutates the shared tokenizer
> gemma-4 ships `add_bos_token=False` (the chat template emits `<bos>`).
> The default `force_bos=True` double-BOSed every subsequent multimodal
> forward — dog-TL collapsed 0.91 → 0.18 *on the model's own logits*.
> `wrap_for_unembed` passes `force_bos=False`; the **fitting** path keeps the
> default (raw WikiText genuinely needs its single BOS). Text prompts through
> `lens.apply` must prepend `<bos>` manually.

## Urban-corpus lens + corpus-dependence result (2026-07-23)

Second lens fitted on a **domain-aligned corpus**:
`outputs/_monocle/jlens/urban/gemma4_12b_lens.pt`, same recipe, first 128 of
`corpus_urban_1k.json` — an agent-crawled 1,000-prompt mixture (provenance in
`corpus_staging/*.stats.json`): 400 Wikipedia urban/NYC leads (50/50
NYC/general), 350 Localized-Narratives scene descriptions (CC-BY; 57%
street-filtered), 250 NYC street-level (43 Wikivoyage district pages + 103
digests from ~1,130 real 311 records, street-visible complaint types only).
Crime-ranking / safety-evaluation content excluded by construction (keeps the
lens independent of the pairvqa judgment tasks).

Head-to-head on identical recorded activations
(`monocle/jlens_compare.py`, jobs 258331/258332):

| Layer | top-10 Jaccard | JS div | dog TL (wiki/urban) |
|---|---|---|---|
| 6 / 12 | 0.00 | ~0.69 (≈ln 2, max) | 0.19/0.20, 0.00/0.00 |
| 18–30 | 0.22–0.32 | 0.11–0.31 | both ≤0.11 |
| **36** | 0.46 | 0.13 | **0.95 / 0.76 (both ignite)** |
| 42 / 46 | 0.70 / 0.58 | 0.07 / 0.09 | 0.91/0.93, 0.98/0.95 |

Two findings:
1. **The ignition result is corpus-robust.** Both lenses agree on the L36
   text-reading onset and the late-layer localization — the depth story
   survives swapping the fitting corpus.
2. **Early-layer readouts are corpus-dominated.** At L6–L12 the two lenses
   emit near-disjoint token sets (Jaccard ≈ 0, JS at its ln 2 ceiling) while
   both fail localization — below the workspace, the transport mostly reads
   out its fitting corpus, not the residual. **Do not interpret early-layer
   J-lens word clouds as image content.** (This corpus-dependence measurement
   is absent from the paper.)

The urban lens is not systematically sharper; the WikiText lens remains the
default reference instrument, the urban lens the robustness check.

## Tier-2: multimodal (patch-position) lens — fitter built 2026-07-23

`monocle/jlens_fit_mm.py` fits `J_l = E[dh_target/dh_patch]` over **multimodal
prompts** (cyclomedia face + one of 5 rotating neutral describe-contexts):
source positions = the 256 patch tokens, targets = first-patch-onward
(final position excluded). Core estimator `jacobian_for_inputs` is
dependency-injectable and **CPU-verified against a toy per-position-linear
model with a closed-form Jacobian** (recovers the block weight matrix exactly;
disjoint target mask → exactly 0) — see `tests/test_jlens_mm.py`.
Fit images: `monocle/sample_fit_images.py` — 256 faces from the cached
recording index (equal per borough-dataset, one random F/B/L/R face per
recording, seed 777, 0% missing; respects the never-walk-NFS rule and the
DuckDB SAMPLE-before-WHERE trap).
Calibration (job 259517): **417 s/image at `dim_batch=8`** (peak 43.4 GiB;
12+ OOM) — multimodal seq ~271 tokens vs 128 for text roughly halves the
feasible dim_batch. 128 images ≈ 14.8 GPU-h, sharded 4×.
Lens lands at `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt` (standard
`JacobianLens` format — all downstream tooling unchanged).

### Result (jobs 285686/285687): a different instrument, not a sharper one

Fit completed 128/128 images, validation PASS. But the patch-source lens is a
**genuinely different transport**, divergent from the WikiText lens at every
layer (top-10 Jaccard ≤ 0.19 even at L42 — contrast wikitext-vs-urban which
converges to 0.70 there):

| Effect | Evidence |
|---|---|
| Text recall degraded | mid-layer factual readouts are noise; "Texas" only at L42 |
| L36 text-reading ignition weakened | dog TL 0.25 vs 0.95 (wikitext); clean only from L42 (0.83) |
| Early/mid **color** transport improved | red right-half 0.91 @ L18, 0.90 @ L30 (wikitext: 0.55, 0.32) |
| No early-layer image-driven readout | dog TL ≤ 0.25 through L30 for all three lenses |

**Reading.** The mm estimator averages patch influence on *later* positions —
a cross-position, captioning-oriented transport — while monocle's readout is
*same-position* ("what would this patch say next"). Text-corpus gradients are
same-position-dominant, so the text-fitted lens better matches the readout
semantics for symbolic content; the mm lens instead captures patch→downstream
verbalization pathways (hence the color-transport gain).

**Instrument selection guidance:**
- Per-patch word clouds / depth maps → **WikiText lens** (reference), urban
  lens as robustness check. The L36 ignition is a text-lens-robust result.
- "Which patches feed the eventual *answer*" (phase-4 safety-workspace reads
  the answer position) → the **mm lens** is arguably the right transport —
  its targets are exactly later-position influence.
- Early layers (≤L30) remain uninterpretable as image content under ALL three
  lenses — the corpus/estimator dominates below the workspace.

## Phase 4 — rung A: emergence-depth maps (2026-07-23, jobs 296939/298876)

`monocle/emergence.py` — per patch, at what layer does the J-lens readout
"lock on" to the final-layer readout? Metrics per (patch, fitted layer) vs
L47: top-k Jaccard (k=10) and JS divergence, saved long
(`outputs/_monocle/emergence/<id>.emergence.parquet`) so the ignition
threshold tau is a **post-hoc analysis choice**; heatmap renderer + legend is
PIL-only. Eval set: 24 faces, seed 778, explicitly disjoint from the mm-fit
images. Tests: `tests/test_emergence.py` (25); `jlens_compare` now imports
its Jaccard/JS from here (single source of truth).

**Result: on natural street imagery, per-patch readouts do not converge to
the final readout until the last two layers.** Corpus means (24 images,
6,144 patches): Jaccard vs final ≤0.05 through L42, 0.21 at L46; JS ~0.63–0.69
(≈ln 2 ceiling) through L42, 0.48 at L46. Tau sweep: at tau=0.3, 77% of
patches never ignite; even at tau=0.1, median ignition = L46. The synthetic
L36 ignition (quadrant tests) is about *specific verbalizable content*
arriving — the full token-identity distribution keeps churning to the end.
Spatial structure at tau=0.1: uniform low-texture regions (roadbed, sky, car
hood) lock on from ~L30; buildings/foliage only at L42–46; the most complex
texture never. Re-rendered maps: `*.emergence_tau0.1.png`.

## Phase 4 — rung B: the safety-workspace experiment (2026-07-24, job 299306)

`monocle/safety_workspace.py` — depth-resolved answer emergence on 300 subway
pairs (probe recipe verbatim: prod / axis "brightly lit" / neutral, forced
`{"answer": "` prefix, label/last read positions) under all three lenses,
plus per-patch answer-feeding maps (mm lens, 8 pairs) with prod−neutral
difference panels. Label first-token collision confirmed at runtime:
MuchLess/MuchMore both start with "Much" (id 46003) → 4-way collapsed classes
{Much*, Less, Same, More}; primary readouts are p(prod first token) and 4-way
restricted-argmax agreement (constant-"More" baseline = 0.613 — early-layer
"agreement" at 0.61/0.28 is a constant-class artifact, not signal). Tests:
`tests/test_safety_workspace.py` (28). Outputs:
`outputs/_monocle/safety_workspace/{answer_depth.parquet,answer_depth_summary.json,maps/}`.

**Result — the judgment enters the workspace at L42 under prod, and NEVER
enters under neutral:**

| Condition | p(prod token) @label, L6–36 | L42 | L46 | L47 | agree > 0.613 baseline |
|---|---|---|---|---|---|
| prod (wikitext lens) | 0.000 | 0.066 | 0.254 | **0.770** | from **L42** (0.77) |
| axis "brightly lit" | 0.000 | 0.067 | 0.269 | 0.520 | L42 borderline |
| neutral | 0.000 | 0.012 | 0.043 | 0.128 | **never** (0.07 @L47) |

1. **Decodable ≠ in the workspace, depth-resolved.** The activation probe
   decodes the judgment at 0.877 from these same neutral-prompt forwards; the
   J-lens readout carries it at no depth whatsoever — not even the final
   layer. Between L24 (linearly decodable) and L42 (sayable, prod only), the
   information lives outside the verbalizable channel; without the prompt's
   permission it stays outside forever. This is the mechanistic completion of
   [[concept-activation-probe]].
2. **The workspace carries the decision, not the concept.** Safety- and
   brightness-vocab mass at the answer position ≈ 0 at every layer in every
   condition — the channel transmits the label token, never "safe"/"dark".
3. **Prompt-gated, spatially sparse safety transport at L42 (part b).** Under
   the mm lens, 7/8 pairs show a few individual patches carrying real
   safety-vocab mass (up to 0.78) in their answer-feeding readout — only at
   L42 and only under prod (identically 0 under neutral; per-patch means:
   prod 0.0007 vs neutral 0.0000 at L42, 0 elsewhere). The prompt gates which
   patches may speak safety into the transport.

## Phase 4 — rung C: J-vector steering (2026-07-24, jobs 310077/310078)

`monocle/jlens_steer.py` — causal rung: steering direction
`d = ∂(logit_More − logit_Less)/∂h_final` by autograd through the unembed
(norm + softcap handled by linearization), pulled back `v_l = J_l^T d`
(orientation guarded by the invariant `⟨d, J J^T d⟩ = ‖J^T d‖²`, enforced at
runtime), injected under the NEUTRAL prompt with a dose-response alpha sweep
and a random-direction control; answer-position arm uses the wikitext lens,
patches arm the mm lens. Tests: `tests/test_jlens_steer.py` (28).

> [!danger] Scale injections by the CENTERED residual norm
> First full run (jobs 303437/303438) scaled alpha by raw ``median ||h_l||``
> ≈ 254 — but Gemma's massive-activation component is ~150× the pair-specific
> signal (~1.7), so even the random CONTROL at alpha=2 rewrote the label
> distribution wholesale (exact 0.03 → 0.69) and every steered cell pinned
> p(More)−p(Less) at 0. Uninterpretable by construction. `scale_l` is now the
> median centered norm ``||h_l − mean(h_l)||`` at the injection positions —
> the same lesson as the probe's raw-cosine artifact, in causal form.

Bring-up gotchas: callers put `use_cache` inside the inputs dict — merge,
don't pass twice; separate `--out-dir` per arm or the resume checkpoints
collide.

**Result (centered scales; answer arm L24=1.9, L36/42≈22; patches arm
42–55): no clean causal implant — a structured null.** Under the neutral
prompt no cell shows a sign-consistent, control-separated dose-response:
- L24 answer arm: the strongest effect is **sign-inverted** (−8α drives
  p(More)−p(Less) to +0.80; positive doses pin both label probs to 0), and
  the random control also produces large class-flips at higher doses (exact
  0.03 → 0.69 at −8α ≈ a 15-norm nudge, ~6% of the residual).
- L36/L42 answer arm and the patches arm: label-prob gaps ≈ 0 or move
  identically for signal and control (L42 patches: +0.075 signal vs +0.094
  control — no specificity).

**Reading.** The forced-answer position under a prompt that never asked for
a label is *fragile*: moderate perturbations in any direction snap it
between degenerate modes instead of moving a judgment continuously. This
coheres with rung B — the workspace channel under neutral is gated shut, and
a single corpus-averaged J-vector does not pry it open. A causal result
likely needs a different design: modulate under the PROD prompt (channel
already open) or use per-prompt local Jacobians instead of E[J].

Artifacts: `outputs/_monocle/steer/{answer_wikitext,patches_mm}/`
(`steer_results.parquet`, `steer_summary.json`); the invalid raw-scale run
survives only in `outputs/monocle_30343{7,8}.out`.

## See also

- [[concept-monocle]] — base per-patch logit lens, geometry, scoring
- [[concept-activation-probe]] — the "represented but not sayable" result this targets
- [[slurm-deployment]] — klara / `.venv-nightly` / LD_PRELOAD recipe
