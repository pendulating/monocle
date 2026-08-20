---
title: "Monocle — Per-Patch Logit Lens over Cyclomedia Imagery"
category: concept
created: 2026-07-20
updated: 2026-07-20
tags:
  - concept
  - interpretability
  - visualization
  - gemma-4
  - cyclomedia
  - transformers
---

# Monocle

Per-patch **logit-lens word clouds** for gemma-4-12B, after Matt Henderson's
July 2026 video visualization, adapted to static street imagery. Because
`gemma4_unified` is **encoder-free**, each 48×48-px image block occupies a real
sequence position — so the ordinary forward pass yields next-token logits *at
every patch*. Monocle renders, atop each [[cyclomedia-catalog]] image, the
top-k (default 3) tokens each patch would emit, after dividing out the
image-global token distribution so a patch shows what *it* predicts that the
rest of the image doesn't.

On a Brooklyn BP-station face the clouds are strikingly localized: sky patches
say *city / clear / day / skyline*, buildings *Tall / windows / steel /
scaffolding*, the green BP signage *green / signs / teal / lime*, red banners
*red / flags / poles*, roadbed *car / taxi / bikes / curb / graffiti* — and it
volunteers *MTA / boroughs / York*, i.e. it knows it's NYC.

## Package layout (`monocle/`)

| File | Role |
|---|---|
| `extract.py` | model load (same recipe as [[concept-activation-probe]]), input build, patch-position logits, runtime grid inference |
| `scoring.py` | global-distribution division, vocab filtering, case/piece dedupe, parquet output |
| `render.py` | PNG + hoverable-SVG word-cloud overlay (PIL only, CPU) |
| `cli.py` | `python -m monocle.cli --image P` or `--recording-id ID` (faces via the cyclomedia catalog) |
| `validate.py` | phase-0 geometry checks — run before trusting any render on a new model |
| `monocle.sub` | generic klara runner: `sbatch monocle/monocle.sub <module> [args...]` |

Environment: `.venv-nightly` + `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`,
klara, 1×A6000 — identical to the activation-probe recipe (see [[slurm-deployment]]).
Tests: `tests/test_monocle.py` (CPU-only, 15 tests).

## Verified geometry (job 144418, 2026-07-20)

| Property | Value |
|---|---|
| Patch | one soft token per **48×48 px** block (`patch_size 16 × pooling 3`), cap **280**/image |
| 1024×1024 face | **256 tokens = 16×16 grid**, contiguous at seq 5..260 |
| `pixel_values` | `(1, 280, 6912)` — pre-patchified raw pixels, `6912 = 48·48·3` (encoder-free confirmed) |
| Order | **row-major** — synthetic "DOG" drawn top-left puts 0.91 of exact dog-token mass in the TL quadrant, and its mass spreads horizontally (wide text ⇒ col_std > row_std), ruling out a transposed grid |
| Grid inference | never hard-code 16×16 — `extract.infer_grid()` derives it (`pixel_values` dims → perfect square → aspect-closest factor pair) |

## Scoring

`score(patch, t) = α·log p_patch(t) + [log p_patch(t) − log p_global(t)]`,
`p_global` = mean over the image's patches, α = 0.3 default. The bracketed PMI
term is Henderson's "divide out the global distribution"; the α term stops
rare-token noise from winning. Vocab filter drops specials, `<...>` control /
byte-fallback tokens, and tokens with <2 chars or no letter; case/piece
variants (`▁Dog` vs `dog`) dedupe within a patch. Output: long parquet
`(patch_idx, rank, token, token_id, score, p_patch, p_global, patch_row,
patch_col)` + sidecar `.meta.json` with grid geometry.

**Pooling** (`--pool N`, CLI default 2): the model grid is architecture-fixed,
so larger display cells come from averaging the per-patch *probability
distributions* over N×N blocks before the PMI division (`scoring.pool_probs`).
16×16 → 8×8 at pool 2, 4×4 at pool 4; edge blocks average their actual
members. Pooled runs get an `_pN` image-id suffix; `.meta.json` keeps the
model grid as `model_n_rows/model_n_cols` while `n_rows/n_cols` (what the
renderer reads) refer to the pooled grid. `--pool 1` = full model resolution.

## Gotchas

> [!danger] Substring token matching is a trap
> 858 gemma tokens contain "red" (`▁required`, `▁considered`, `ered`, …) vs 22
> for "dog". The first validation run "failed" red-quadrant localization purely
> because of this contamination. Always match on exact display form
> (`validate.vocab_ids_exact`).

- **Off-by-one anticipation is real and measurable**: logits at position *i*
  predict token *i+1*. On the synthetic quadrant image, exact-"red" mass splits
  **TR=0.51 / BR=0.48** (left half 0.01) — the raster-order *predecessors* of
  the bottom-right red block sit directly above it in TR and fire "red is
  coming" before the block itself fires "red is here". Validation therefore
  checks right-half dominance for red, argmax-TL plus a wide-vs-tall
  mass-spread (transpose) test for dog — never naive argmax-quadrant alone.
- The `mm_token_type_ids` input marks the image block — image tokens likely get
  bidirectional attention within the block (as in Gemma 3), so patch
  predictions are not strictly causal-local. Localization is still strong
  (0.90 quadrant mass), but don't interpret a single patch as seeing only
  pixels before it.
- Softcapping (30.0) is applied inside the model — use `out.logits` as-is.
- System prompt precedes the patches → `--system` **conditions** the per-patch
  predictions (the phase-3 research knob for the [[concept-self-explanation-ladder]]
  storyline); user text after the image cannot.
- The [[concept-activation-probe]] massive-activation trap does not bite here
  (logits, not raw residual cosines).

## Usage

```bash
# phase-0 geometry validation (any new model / transformers bump)
sbatch monocle/monocle.sub monocle.validate

# lens a cyclomedia recording (F B L R faces) with SVG hover overlays
sbatch monocle/monocle.sub monocle.cli \
  --recording-id W0D0M3OU --dataset brooklyn_2025_1k --svg

# arbitrary images, prompt-conditioned
sbatch monocle/monocle.sub monocle.cli --image a.jpg b.jpg \
  --system "You are a real-estate appraiser." --k 5
```

Outputs land in `outputs/_monocle/{validate,runs}/` as
`<image_id>.parquet` + `.meta.json` + `.overlay.png` (+ `.overlay.svg`).
Rendering is CPU-only: `render.load_run()` → `render.render_overlay()`.

## See also

- [[concept-jlens-monocle]] — the depth axis: Jacobian-lens readout at every fitted layer
- [[concept-activation-probe]] — shared extraction recipe; "represented but not sayable"
- [[guide-cyclomedia-browser]] — candidate host for an interactive monocle view
- [[urban-pair-vqa]] — the judgment tasks whose prompts can condition the lens
- [[vllm-inference]] — `_gemma4_unified_chat_template()` production prompt path
