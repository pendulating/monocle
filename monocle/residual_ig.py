"""Integrated gradients in the RESIDUAL space, because the judgment saturates.

Why the plain gradient is the wrong tool here
---------------------------------------------
The deletion curve shows that 1 cell of 64 gives 25 to 46 per cent of what
half the photograph gives. The judgment flattens at once. A gradient reads the
slope AT the real image, and on a flat top that slope says little about what a
removal would do.

Integrated gradients answers the same question along a PATH. It walks from the
mean-patch image to the real one and averages the gradient on the way, so a
direction that mattered early still counts.

Why the residual and not the pixels
-----------------------------------
`patch_causal.integrated_gradients` walks the path in PIXEL space. On
gemma-4-12b that path is dead: the embedder casts to bfloat16 and the pixel
gradient does not survive it. This module walks the path in the residual of a
block, which is downstream of the vision tower, so the gradient is clean.

The definition
--------------
`h` is the residual at the image slots of block L, and `hbar` is the mean over
those slots, which is the baseline the input ablation also uses.

    IG[p] = <h[p] - hbar, mean over k of dc/dz[p] at z = hbar + a_k (h - hbar)>

with `a_k = (k + 0.5) / steps`.

Completeness
------------
The attributions SUM to `c(real) - c(baseline)`. The run reports that residual,
so the instrument checks itself in a way grad-times-activation cannot.

Usage:
    python -m monocle.residual_ig --cases restaurants --n-images 60 \
        --group 1 --layer 18 --steps 24 --tag ig
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import patch_ablate as pab  # noqa: E402
from monocle import patch_attribution as pa  # noqa: E402
from monocle import patch_causal as pc  # noqa: E402
from monocle import patch_map_causal as pmc  # noqa: E402
from monocle import single as sg  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


class SlotWriter:
    """Write a tensor into the image slots of a block output, in the graph."""

    def __init__(self, slots: list[int], z: torch.Tensor) -> None:
        self.slots = slots
        self.z = z

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        h = output[0] if isinstance(output, tuple) else output
        idx = torch.as_tensor(self.slots, device=h.device)
        new = h.clone()
        new[:, idx, :] = self.z.to(h.dtype).to(h.device)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new


def contrast_at(model: Any, lens_model: Any, inputs: dict, layer: int,
                slots: list[int], z: torch.Tensor, pos_id: int, neg_id: int,
                graph: bool) -> torch.Tensor:
    handle = lens_model.layers[layer].register_forward_hook(
        SlotWriter(slots, z))
    try:
        h = pc.final_residual(model, lens_model, inputs, graph=graph)
        return pc.fp32_contrast(lens_model, h, pos_id, neg_id)[0]
    finally:
        handle.remove()


def read_residual(model: Any, lens_model: Any, inputs: dict, layer: int,
                  slots: list[int]) -> torch.Tensor:
    from jlens.hooks import ActivationRecorder

    with torch.no_grad(), ActivationRecorder(lens_model.layers,
                                             at=[layer]) as rec:
        out = model(**inputs, use_cache=False, logits_to_keep=1)
        del out
        return rec.activations[layer][0, slots, :].detach().clone()


def ig_map(model: Any, lens_model: Any, inputs: dict, layer: int,
           slots: list[int], pos_id: int, neg_id: int, *,
           steps: int) -> dict:
    """The IG map, the plain grad-times-activation map, and the check."""
    h = read_residual(model, lens_model, inputs, layer, slots).float()
    hbar = h.mean(dim=0, keepdim=True)
    delta = h - hbar

    total = torch.zeros_like(delta)
    grad_at_one = None
    for k in range(steps):
        a = (k + 0.5) / steps
        z = (hbar + a * delta).detach().requires_grad_(True)
        c = contrast_at(model, lens_model, inputs, layer, slots, z,
                        pos_id, neg_id, graph=True)
        (g,) = torch.autograd.grad(c, z)
        total += g.float()
        del z, c, g
    avg = total / steps

    # The plain gradient, at the real image, for a paired comparison.
    z1 = h.detach().clone().requires_grad_(True)
    c1 = contrast_at(model, lens_model, inputs, layer, slots, z1,
                     pos_id, neg_id, graph=True)
    (g1,) = torch.autograd.grad(c1, z1)
    c_real = float(c1.detach())

    with torch.no_grad():
        z0 = hbar.expand_as(h).contiguous()
        c_base = float(contrast_at(model, lens_model, inputs, layer, slots,
                                   z0, pos_id, neg_id, graph=False))
    ig = (delta * avg).sum(dim=1)
    gxa = (delta * g1.float()).sum(dim=1)
    return {"ig": ig.detach().cpu().numpy(),
            "gxa_centered": gxa.detach().cpu().numpy(),
            "c_real": c_real, "c_base": c_base,
            "completeness": float(ig.sum()) - (c_real - c_base)}


def run_case(case: str, *, proc: Any, model: Any, lens_model: Any, tmpl: str,
             n_images: int, layer: int, group: int, device: str,
             out_dir: Path, seed: int, steps: int,
             truth_from: Optional[str] = None,
             prompt_case: Optional[str] = None,
             paraphrase: Optional[str] = None) -> pd.DataFrame:
    from PIL import Image
    from monocle.input_gate import load_truth, score

    tok = proc.tokenizer
    ask = prompt_case or case
    pos_id, neg_id = sg.contrast_ids(tok, ask)
    if prompt_case:
        log(f"the question comes from {ask}, the images from {case}")
    # The PARAPHRASE control. The judgment and the 2 answer words stay, and
    # only the wording changes. See monocle/input_gate.py.
    overrides = {"question": paraphrase} if paraphrase else None
    if paraphrase:
        log(f"the question is reworded: {paraphrase!r}")
    image_token_id = int(model.config.image_token_id)
    stored = load_truth(truth_from, case, group) if truth_from else None
    rows, maps = [], []
    t0 = time.time()
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, ask,
                                        overrides=overrides,
                                        device=device)
        slots = pab.slots_of(inputs, image_token_id)
        res = ig_map(model, lens_model, inputs, layer, slots,
                     pos_id, neg_id, steps=steps)
        cand = {k: res[k] for k in ("ig", "gxa_centered")}
        if group > 1:
            cand = {k: pmc.pool_by_groups(v, group) for k, v in cand.items()}
        row = {"case": case, "ask": ask, "layer": layer,
               "target": str(r["image_id"]),
               "c_real": res["c_real"], "c_base": res["c_base"],
               "completeness": res["completeness"]}
        if stored is not None and str(r["image_id"]) in stored:
            truth = stored[str(r["image_id"])]
            for name, m in cand.items():
                row.update({f"{name}_{k}": v
                            for k, v in score(truth, m).items()})
            maps.append({"layer": layer, "target": str(r["image_id"]),
                         "causal": np.asarray(truth).tolist(),
                         **{k: np.asarray(v).tolist()
                            for k, v in cand.items()}})
        rows.append(row)
        log(f"  {i + 1}/{n_images} {r['image_id']} c={res['c_real']:+.2f} "
            f"base={res['c_base']:+.2f} "
            f"completeness residual {res['completeness']:+.3f} "
            f"{time.time() - t0:.0f}s")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "agreement.parquet", index=False)
    if maps:
        pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)
    log(f"wrote {case_dir}; mean completeness residual "
        f"{df['completeness'].abs().mean():.3f} against a range of "
        f"{(df['c_real'] - df['c_base']).abs().mean():.2f}")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["restaurants"])
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--truth-from", default=None)
    ap.add_argument("--paraphrase", default=None,
                    help="reword the question, keep the judgment and "
                         "the 2 answer words")
    ap.add_argument("--prompt-case", default=None,
                    help="ask another case's question, as the control")
    ap.add_argument("--tag", default="ig")
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from monocle import attn_route as ar
    from monocle import jlens_read

    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    out_dir = Path("outputs/_monocle/residual_ig") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    for case in args.cases:
        run_case(case, proc=proc, model=model, lens_model=lens_model,
                 tmpl=tmpl, n_images=args.n_images, layer=args.layer,
                 group=args.group, device=args.device, out_dir=out_dir,
                 seed=args.seed, steps=args.steps,
                 truth_from=args.truth_from, prompt_case=args.prompt_case,
                 paraphrase=args.paraphrase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
