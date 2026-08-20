"""Monocle — per-patch logit-lens visualization for gemma-4-12B (gemma4_unified).

The unified (encoder-free) architecture embeds 48x48-px image blocks directly
as sequence positions, so the ordinary forward pass yields next-token logits
at every patch. Monocle extracts the top-k tokens each patch would emit,
divides out the image-global token distribution (so a patch shows what *it*
predicts that the rest of the image doesn't), and renders the result as a
word-cloud overlay on the source image.

Modules:
    extract  — model load, input build, patch-position logits
    scoring  — global-distribution division, token filtering, top-k
    render   — PNG/SVG overlay
    cli      — image path or cyclomedia recording -> overlay + parquet
    validate — phase-0 geometry checks (run this first on any new model)
"""

MODEL_DIR_DEFAULT = "/share/pierson/matt/zoo/models/gemma-4-12B-it"
MODEL_PATCH_PX = 48  # vision patch_size 16 x pooling_kernel_size 3
