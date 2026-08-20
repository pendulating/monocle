"""Model loading, input construction, and patch-position logit extraction.

Reuses the exact production prompt path from the activation-probe work
(`_gemma4_unified_chat_template` + `Gemma4UnifiedProcessor`) so the model we
lens is the same one the pairvqa runs serve. See
vault/context/wiki/concept-activation-probe.md for the verified recipe.

Off-by-one semantics: logits at sequence position i are the model's prediction
for token i+1. We read logits *at* each patch position — "what would come next
if you sampled right after this patch" — matching Henderson's framing.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from monocle import MODEL_DIR_DEFAULT, MODEL_PATCH_PX


@dataclasses.dataclass
class PatchGrid:
    """Geometry of one image's patch-token grid, plus how it was inferred."""

    n_rows: int
    n_cols: int
    orig_w: int
    orig_h: int
    resized_w: Optional[int]  # None when inferred without pixel dims
    resized_h: Optional[int]
    strategy: str  # "pixel_values" | "square" | "aspect_factor"

    @property
    def n_patches(self) -> int:
        return self.n_rows * self.n_cols

    def to_meta(self) -> dict:
        return {**dataclasses.asdict(self), "model_patch_px": MODEL_PATCH_PX}


def load_model(model_dir: str = MODEL_DIR_DEFAULT, device: str = "cuda:0"):
    """Load processor + model + production chat template. ~13 s on klara."""
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

    from dagspaces.common.vllm_inference import _gemma4_unified_chat_template

    is_g4u, tmpl = _gemma4_unified_chat_template(model_dir)
    if not is_g4u or not tmpl:
        raise RuntimeError(
            f"{model_dir} is not gemma4_unified or chat_template.jinja missing")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(model_dir)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=device)
    model.eval()
    print(f"[monocle] model loaded in {time.time() - t0:.0f}s "
          f"({type(model).__name__})", flush=True)
    return proc, model, tmpl


def build_inputs(
    proc: Any,
    tmpl: str,
    image: Image.Image,
    system: Optional[str] = None,
    user_text: Optional[str] = None,
    device: str = "cuda:0",
) -> dict:
    """Single-image chat inputs. Default is minimal context (image only).

    `system` text precedes the patches in the sequence, so it *conditions*
    the per-patch predictions (the phase-3 research knob). `user_text` comes
    after the patches and cannot affect them under causal attention — it only
    matters if you also generate.
    """
    content: list[dict] = [{"type": "image"}]
    if user_text:
        content.append({"type": "text", "text": user_text})
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=tmpl)
    # processors differ on image nesting; try batched-nested then flat
    # (same fallback the activation-probe smoke test needed)
    last: Exception | None = None
    for kw in ({"images": [[image]]}, {"images": [image]}):
        try:
            return proc(text=[text], return_tensors="pt", **kw).to(device)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"processor rejected both image nestings: {last}")


def image_token_positions(inputs: dict, image_token_id: int) -> torch.Tensor:
    """1-D LongTensor of sequence positions holding image tokens.

    Raises if the block is not contiguous — a non-contiguous block would mean
    the patch->position mapping assumption is wrong and every downstream
    render is garbage.
    """
    ids = inputs["input_ids"][0]
    pos = (ids == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError("zero image tokens in input_ids — image never entered")
    if int(pos[-1] - pos[0]) != pos.numel() - 1:
        raise RuntimeError(
            f"image-token block not contiguous: span {int(pos[0])}..{int(pos[-1])} "
            f"but only {pos.numel()} tokens")
    return pos


def infer_grid(inputs: dict, n_patches: int, orig_size: tuple[int, int]) -> PatchGrid:
    """Recover (rows, cols) of the patch grid. Never hard-code 16x16 —
    the 280-soft-token cap makes the grid aspect-dependent.

    Strategies, in order of trust:
      1. pixel_values is [B, C, H, W]: rows = H/48, cols = W/48.
      2. n_patches is a perfect square (processor made the image square).
      3. Integer factor pair of n_patches closest to the original aspect.
    """
    orig_w, orig_h = orig_size

    pv = inputs.get("pixel_values")
    if pv is not None and pv.dim() == 4:
        h, w = int(pv.shape[-2]), int(pv.shape[-1])
        rows, cols = h // MODEL_PATCH_PX, w // MODEL_PATCH_PX
        if rows * cols == n_patches:
            return PatchGrid(rows, cols, orig_w, orig_h, w, h, "pixel_values")

    root = math.isqrt(n_patches)
    if root * root == n_patches:
        return PatchGrid(root, root, orig_w, orig_h, None, None, "square")

    aspect = orig_w / orig_h
    best: tuple[float, int, int] | None = None
    for rows in range(1, n_patches + 1):
        if n_patches % rows:
            continue
        cols = n_patches // rows
        err = abs(math.log((cols / rows) / aspect))
        if best is None or err < best[0]:
            best = (err, rows, cols)
    assert best is not None
    return PatchGrid(best[1], best[2], orig_w, orig_h, None, None, "aspect_factor")


def forward_patch_logits(
    model: Any, inputs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward pass; returns (logits [n_patches, vocab] fp32 on GPU,
    patch positions). Softcapping is already applied inside the model."""
    image_token_id = int(model.config.image_token_id)
    pos = image_token_positions(inputs, image_token_id)
    with torch.inference_mode():
        out = model(**inputs, use_cache=False)
    logits = out.logits[0, pos, :].float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite patch logits")
    return logits, pos


def lens_image(
    proc: Any,
    model: Any,
    tmpl: str,
    image: Image.Image,
    system: Optional[str] = None,
    device: str = "cuda:0",
) -> tuple[torch.Tensor, PatchGrid]:
    """Full extraction for one PIL image -> (patch logits, grid)."""
    inputs = build_inputs(proc, tmpl, image, system=system, device=device)
    logits, _pos = forward_patch_logits(model, inputs)
    grid = infer_grid(inputs, logits.shape[0], image.size)
    return logits, grid


def open_image(path: str | Path) -> Image.Image:
    return Image.open(str(path)).convert("RGB")
