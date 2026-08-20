"""Layer-resolved patch readout: Jacobian-lens monocle.

Applies a fitted Jacobian lens (monocle.jlens_fit -> sub/jacobian-lens) to the
residual stream at every image-patch position, at every fitted layer:

    logits_l[patch] = unembed( J_l @ h_l[patch] )

giving a per-LAYER stack of the per-patch vocab distributions monocle already
knows how to score and render.

Layer convention: residuals are captured with jlens's own ActivationRecorder
forward hooks on the SAME block modules the fitting estimator hooks
(lens_model.layers, i.e. model.model.language_model.layers), during the
multimodal forward. "Layer l" is therefore the output of block l by
construction — identical to fitting, with no dependence on HF's
output_hidden_states tuple layout. (The first implementation used
``hidden_states[l+1]`` and failed stage-A validation by 5e+01: in
transformers 5.x the tuple's final entry is NOT the bare block output.)

The final layer is read straight from ``out.logits`` (no transport, no
unembed) — exact by definition. ``final_layer_consistency`` instead validates
the path the transported layers actually use: unembed(block-47 hook output)
must reproduce ``out.logits`` at patch positions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from monocle import extract  # noqa: E402


def load_lens(path: str):
    from jlens import JacobianLens

    return JacobianLens.load(path)


def wrap_for_unembed(model: Any, tokenizer: Any):
    """jlens's HFLensModel over the already-loaded multimodal model — reused
    for its unembed (final norm + lm_head + softcap) and its .layers handle
    (the hook targets). Holds references, copies nothing.

    force_bos=False is load-bearing: gemma-4 ships add_bos_token=False because
    the chat template emits <bos> itself. from_hf's default (True) mutates the
    SHARED processor tokenizer and double-BOSes every subsequent multimodal
    build_inputs — measured as dog-TL quadrant mass collapsing 0.91 -> 0.18 at
    the final layer. The fitting path (jlens_fit) keeps the default: raw
    WikiText has no template <bos>, so forcing exactly one there is correct.
    """
    import jlens

    return jlens.from_hf(model, tokenizer, force_bos=False)


def record_patch_activations(
    model: Any,
    lens_model: Any,
    inputs: dict,
    layers: list[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """One multimodal forward with jlens hooks on the requested blocks.

    Returns (activations {layer: [B, seq, d] block output}, patch positions,
    final logits [n_patches, vocab] fp32). Hooking lens_model.layers keeps the
    layer convention identical to jlens.fitting by construction.
    """
    from jlens.hooks import ActivationRecorder

    image_token_id = int(model.config.image_token_id)
    pos = extract.image_token_positions(inputs, image_token_id)
    with ActivationRecorder(lens_model.layers, at=sorted(set(layers))) as rec:
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
        activations = {i: rec.activations[i].detach() for i in rec.activations}
    final_logits = out.logits[0, pos, :].float()
    if not torch.isfinite(final_logits).all():
        raise RuntimeError("non-finite final logits at patch positions")
    return activations, pos, final_logits


def lens_patch_logits(
    lens: Any,
    lens_model: Any,
    activations: dict[int, torch.Tensor],
    positions: torch.Tensor,
    final_logits: torch.Tensor,
    layers: Optional[list[int]] = None,
) -> dict[int, torch.Tensor]:
    """Per-layer patch logits: {layer: [n_patches, vocab] fp32 on GPU}.

    ``layers`` defaults to the lens's fitted layers plus the final layer.
    Fitted layers are transported (J_l @ h) then unembedded; the final layer
    is ``final_logits`` verbatim (the model's own head output).
    """
    final = lens_model.n_layers - 1
    if layers is None:
        layers = [*lens.source_layers, final]
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer == final:
            out[layer] = final_logits
            continue
        if layer not in lens.jacobians:
            raise ValueError(
                f"layer {layer} not fitted; lens has {lens.source_layers}")
        if layer not in activations:
            raise ValueError(f"layer {layer} was not recorded")
        h = activations[layer][0, positions, :].float()
        logits = lens_model.unembed(lens.transport(h, layer)).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"non-finite lens logits at layer {layer}")
        out[layer] = logits
    return out


def lens_image_layers(
    proc: Any,
    model: Any,
    tmpl: str,
    lens: Any,
    lens_model: Any,
    image: Image.Image,
    system: Optional[str] = None,
    layers: Optional[list[int]] = None,
    device: str = "cuda:0",
) -> tuple[dict[int, torch.Tensor], extract.PatchGrid]:
    """Full layer-stack extraction for one PIL image.

    Returns ({layer: [n_patches, vocab] logits}, grid). Grid geometry is
    layer-independent (same forward pass).
    """
    final = lens_model.n_layers - 1
    if layers is None:
        layers = [*lens.source_layers, final]
    record_at = [l for l in layers if l != final]
    inputs = extract.build_inputs(proc, tmpl, image, system=system, device=device)
    activations, positions, final_logits = record_patch_activations(
        model, lens_model, inputs, record_at or [final])
    per_layer = lens_patch_logits(
        lens, lens_model, activations, positions, final_logits, layers)
    grid = extract.infer_grid(inputs, positions.numel(), image.size)
    return per_layer, grid


def final_layer_consistency(
    model: Any, lens_model: Any, inputs: dict,
) -> float:
    """Max abs diff between unembed(block-final hook output) and the model's
    own logits at patch positions. This validates the exact path transported
    layers use (hook -> unembed); should be dtype noise (< ~0.1)."""
    final = lens_model.n_layers - 1
    activations, pos, final_logits = record_patch_activations(
        model, lens_model, inputs, [final])
    via_unembed = lens_model.unembed(
        activations[final][0, pos, :].float()).float()
    return float((final_logits - via_unembed).abs().max())
