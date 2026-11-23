"""A SET-aware causal map: learn which cells hold up the judgment.

Why the linear instruments fail
-------------------------------
A lens, an Rlens, a mean gradient and a grad-times-activation map are all
linear in the patch residual. Each scores a cell on its own, and the score of a
SET is the sum of the parts. The measured data says the judgment does not work
that way: a 2x2 block moves `c` about 5 times less than the sum of its 4
patches, and at 8x8 the sum carries no information at all.

So the correct unit is a set. This module searches for the set directly.

The method
----------
A mask `m` in [0,1], 1 value for each cell, blends the patch residual toward
the mean patch of the same image:

    h_p' = m_c * h_p + (1 - m_c) * baseline

The blend keeps the length of the sequence, so the whole path stays
differentiable, and the gradient goes to `m`. Adam moves `m`, and an L1 term
keeps the set small.

The link to the failing instruments
-----------------------------------
At `m = 1` the gradient of `c` for cell `c` is

    dc/dm_c = sum over p in c of <g_p, h_p - baseline>

which IS the centered grad-times-activation instrument. So 1 step of this
search is the instrument that fails the deletion test. If many steps pass it,
the failure comes from the additive form and from nothing else.

The 2 directions
----------------
| Run | Objective | The set it finds |
|---|---|---|
| `down` | make `c` fall | the cells that hold up the positive label |
| `up` | make `c` rise | the cells that hold it down |

The score of a cell is `m_up - m_down`. A positive score means the cell holds
up the positive label, which is the same sign as the measured map.

Usage:
    python -m research.monocle.mask_search --cases subway_safety --n-images 20 \
        --group 2 --steps 60 --tag mask
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_attribution as pa  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import patch_map_causal as pmc  # noqa: E402
from research.monocle import single as sg  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


class MaskBlender:
    """Blend the image slots toward a baseline, under a differentiable mask.

    `weights` holds 1 value for each SLOT, in the order of `slots`. The hook
    keeps the graph, so `torch.autograd.grad` reaches the mask.
    """

    def __init__(self, slots: list[int], weights: torch.Tensor) -> None:
        self.slots = slots
        self.weights = weights

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        idx = torch.as_tensor(self.slots, device=h.device)
        cur = h[:, idx, :]
        # The baseline is the mean over the patches of the SAME image, which
        # is the baseline the input ablation uses.
        base = cur.mean(dim=1, keepdim=True)
        w = self.weights.to(h.device).to(h.dtype).view(1, -1, 1)
        new = h.clone()
        new[:, idx, :] = w * cur + (1.0 - w) * base
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


class AttnMasker:
    """Hide chosen key positions from EVERY query, by a soft amount.

    A blend toward the mean patch is not what an ablation does. An ablation
    takes the token out of the sequence, so no position can read it. Adding
    `log(m)` to the attention logits of key `p` does the same thing as `m`
    goes to 0, and it keeps the length of the sequence, so the path stays
    differentiable.

    Warning: the block must use eager attention, or it gets no
    `attention_mask` and there is nothing to edit.
    """

    def __init__(self, slots: list[int], weights: torch.Tensor,
                 floor: float = 1e-6) -> None:
        self.slots = slots
        self.weights = weights
        self.floor = floor

    def __call__(self, module: Any, args: Any, kwargs: Any) -> Any:
        m = kwargs.get("attention_mask", None)
        if m is None:
            raise RuntimeError(
                "this block got no attention_mask, so there is nothing to "
                "hide — the model is probably not using eager attention")
        if m.dtype == torch.bool:
            raise RuntimeError(
                "the attention_mask is boolean, so a SOFT mask cannot be "
                "added. Load the model with a float mask.")
        idx = torch.as_tensor(self.slots, device=m.device)
        w = self.weights.to(m.device).clamp_min(self.floor)
        add = torch.log(w).to(m.dtype)
        m = m.clone()
        # `m[..., :, idx]` ends in the key axis, so a 1-D `add` broadcasts on
        # it. Never reshape to 4 dims: a block may hand over a 2-D or 3-D mask.
        m[..., :, idx] = m[..., :, idx] + add
        kwargs["attention_mask"] = m
        return args, kwargs


def attn_masked_contrast(
    model: Any, lens_model: Any, inputs: dict, blocks: list[int],
    slots: list[int], weights: torch.Tensor, pos_id: int, neg_id: int,
) -> torch.Tensor:
    """`c` when the patches are hidden by `weights`, at every block."""
    handles = [lens_model.layers[b].register_forward_pre_hook(
        AttnMasker(slots, weights), with_kwargs=True) for b in blocks]
    try:
        h = pc.final_residual(model, lens_model, inputs, graph=True)
        return pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0]
    finally:
        for hd in handles:
            hd.remove()


def masked_contrast(
    model: Any, lens_model: Any, inputs: dict, layer: int,
    slots: list[int], weights: torch.Tensor, pos_id: int, neg_id: int,
) -> torch.Tensor:
    """`c` of row 0, with the mask applied at the output of `layer`."""
    handle = lens_model.layers[layer].register_forward_hook(
        MaskBlender(slots, weights))
    try:
        h = pc.final_residual(model, lens_model, inputs, graph=True)
        return pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0]
    finally:
        handle.remove()


def search(
    model: Any, lens_model: Any, inputs: dict, layer: int,
    slots: list[int], groups: list[list[int]], pos_id: int, neg_id: int, *,
    direction: int, steps: int, lr: float, l1: float,
    mode: str = "blend", n_blocks: int = 48,
) -> tuple[np.ndarray, list[float]]:
    """Find the smallest set of cells that moves `c` in `direction`.

    `direction` is -1 to make `c` fall and +1 to make it rise. Returns the
    mask over CELLS and the trace of `c`.
    """
    n_cell = len(groups)
    # `expand` maps a cell to its slots, so 1 value covers a whole block.
    slot_of_cell = torch.zeros(n_cell, len(slots))
    for ci, g in enumerate(groups):
        for p in g:
            slot_of_cell[ci, p] = 1.0
    slot_of_cell = slot_of_cell.to(
        next(lens_model.layers[layer].parameters()).device)
    theta = torch.zeros(n_cell, requires_grad=True,
                        device=slot_of_cell.device, dtype=torch.float32)
    opt = torch.optim.Adam([theta], lr=lr)
    trace = []
    for _ in range(steps):
        m_cell = torch.sigmoid(theta)
        w = m_cell @ slot_of_cell
        if mode == "attn":
            c = attn_masked_contrast(model, lens_model, inputs,
                                     list(range(n_blocks)), slots, w,
                                     pos_id, neg_id)
        else:
            c = masked_contrast(model, lens_model, inputs, layer, slots, w,
                                pos_id, neg_id)
        # Move `c` in `direction`, and keep the removed set small.
        loss = -direction * c + l1 * (1.0 - m_cell).abs().sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace.append(float(c.detach()))
    return torch.sigmoid(theta).detach().float().cpu().numpy(), trace


def run_case(
    case: str, *, proc: Any, model: Any, lens_model: Any, tmpl: str,
    n_images: int, layer: int, group: int, device: str, out_dir: Path,
    seed: int, steps: int, lr: float, l1: float, batch: int,
    mode: str = "blend",
) -> pd.DataFrame:
    from PIL import Image

    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rows, maps = [], []
    t0 = time.time()
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(inputs, image_token_id)
        groups = (pmc.patch_groups(len(slots), group) if group > 1
                  else [[j] for j in range(len(slots))])
        c0 = pmc.batched_baseline(model, lens_model, inputs, pos_id, neg_id,
                                  batch=batch)
        m_dn, tr_dn = search(model, lens_model, inputs, layer, slots, groups,
                             pos_id, neg_id, direction=-1, steps=steps,
                             lr=lr, l1=l1, mode=mode,
                             n_blocks=len(lens_model.layers))
        m_up, tr_up = search(model, lens_model, inputs, layer, slots, groups,
                             pos_id, neg_id, direction=+1, steps=steps,
                             lr=lr, l1=l1, mode=mode,
                             n_blocks=len(lens_model.layers))
        score = m_up - m_dn
        maps.append({"image_id": str(r["image_id"]),
                     "values": score.astype(float).tolist(),
                     "mask_down": m_dn.astype(float).tolist(),
                     "mask_up": m_up.astype(float).tolist()})
        rows.append({"case": case, "image_id": str(r["image_id"]),
                     "layer": layer, "mode": mode, "c0": c0,
                     "c_down": tr_dn[-1], "c_up": tr_up[-1],
                     "drop": tr_dn[-1] - tr_dn[0],
                     "rise": tr_up[-1] - tr_up[0],
                     "removed_down": float((m_dn < 0.5).sum()),
                     "removed_up": float((m_up < 0.5).sum())})
        log(f"  {i + 1}/{n_images} {r['image_id']} c0={c0:+.2f} "
            f"down {tr_dn[0]:+.2f}->{tr_dn[-1]:+.2f} "
            f"up {tr_up[0]:+.2f}->{tr_up[-1]:+.2f} "
            f"cut {int((m_dn < 0.5).sum())}/{int((m_up < 0.5).sum())} "
            f"{time.time() - t0:.0f}s")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "summary.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)
    # A ranking parquet, so the deletion test reads it with no more work.
    pd.DataFrame({"image_id": [m["image_id"] for m in maps],
                  "values": [m["values"] for m in maps]}).to_parquet(
        case_dir / "ranking.parquet", index=False)
    log(f"wrote {case_dir}")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"])
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--layer", type=int, default=0,
                    help="the block whose output the mask edits")
    ap.add_argument("--mode", default="blend", choices=["blend", "attn"],
                    help="blend mixes the residual toward the mean patch; "
                         "attn hides the patch from every reader, which is "
                         "much closer to what an ablation does.")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--l1", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--tag", default="mask")
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from research.monocle import attn_route as ar
    from research.monocle import jlens_read

    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    out_dir = Path("outputs/_monocle/mask_search") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    for case in args.cases:
        run_case(case, proc=proc, model=model, lens_model=lens_model,
                 tmpl=tmpl, n_images=args.n_images, layer=args.layer,
                 group=args.group, device=args.device, out_dir=out_dir,
                 seed=args.seed, steps=args.steps, lr=args.lr, l1=args.l1,
                 batch=args.batch, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
