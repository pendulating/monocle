"""Per-patch gradients and residuals at the blocks where the patches speak.

One backward pass per image gives, at each block ``l``, the gradient of
the contrast ``c`` with respect to EVERY position's residual. Restricted
to the 256 patch slots, ``g_l[cell]`` is the first-order response of the
answer to any vector ``v`` injected at that cell: ``<g_l[cell], v>``. So
one backward predicts the per-cell injection map of every steering
vector at once, and the recorded residuals ``h_l[cell]`` give every
per-cell read-out (the coordinate of the cell along ``v``).

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``grads.npy`` | ``[n_images, n_layers, n_cells, d_model]`` fp16, the patch gradients |
| ``resid.npy`` | the same shape, the patch residuals |
| ``answer_grads.npy`` | ``[n_images, n_layers, d_model]`` fp16, the gradient at the answer position |
| ``index.parquet`` | image order, the contrast, the slot count |

Usage:
    sbatch --time=01:00:00 monocle/monocle.sub monocle.patch_grads \\
        --case restaurants --layers 12,16,20,22
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import single_steer as ss  # noqa: E402
from monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    parse_int_list,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/patch_grads")
LAYERS_DEFAULT = "12,16,20,22"


def log(msg: str) -> None:
    print(f"[patch-grads] {msg}", flush=True)


def patch_gradients(model, lens_model, inputs: dict, layers: list[int],
                    slots: list[int], tok_pos: int, tok_neg: int):
    """``(grads {l: [n_cells, d]}, resid {l: [n_cells, d]}, answer_grads
    {l: [d]}, c)`` from one forward and one backward."""
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    with torch.enable_grad(), ActivationRecorder(
            lens_model.layers, at=at, start_graph_at=min(layers)) as rec:
        out = model(**inputs, use_cache=False)
        del out
        h_final = rec.activations[final][0, -1]
        logits = lens_model.unembed(h_final).float()
        contrast = logits[tok_pos] - logits[tok_neg]
        sources = [rec.activations[l] for l in layers]
        gs = torch.autograd.grad(contrast, sources, allow_unused=True)
    idx = torch.tensor(slots, device=sources[0].device)
    grads, resid, ans = {}, {}, {}
    for l, src, g in zip(layers, sources, gs):
        if g is None:
            raise RuntimeError(f"no gradient reached block {l}")
        grads[l] = g[0, idx].detach().to(torch.float16).cpu().numpy()
        resid[l] = src[0, idx].detach().to(torch.float16).cpu().numpy()
        ans[l] = g[0, -1].detach().to(torch.float16).cpu().numpy()
    return grads, resid, ans, float(contrast)


def run(args: argparse.Namespace) -> int:
    layers = parse_int_list(args.layers)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    _, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    eval_paths = [r["p"] for r in ev]
    mask_mode = ss.image_mask_mode(args)
    build = ss.make_build_inputs(proc, tmpl, args.case,
                                 causal_image=mask_mode != "bidirectional")
    ss.install_image_mask(model, image_token_id, mask_mode)
    log(f"image mask: {mask_mode}")
    log(f"{args.case}: layers={layers} images={len(eval_paths)}")

    G = R = A = None
    rows = []
    for i, path in enumerate(eval_paths):
        inputs = build(path)
        slots = pab.slots_of(inputs, image_token_id)
        grads, resid, ans, c = patch_gradients(
            model, lens_model, inputs, layers, slots, tok_pos, tok_neg)
        if G is None:
            n, d = len(eval_paths), grads[layers[0]].shape[1]
            G = np.zeros((n, len(layers), len(slots), d), dtype=np.float16)
            R = np.zeros_like(G)
            A = np.zeros((n, len(layers), d), dtype=np.float16)
        for j, l in enumerate(layers):
            G[i, j] = grads[l]
            R[i, j] = resid[l]
            A[i, j] = ans[l]
        rows.append({"image_path": path, "c": c, "n_cells": len(slots)})
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(eval_paths)} ({time.time() - t0:.0f}s)")
    np.save(out_dir / "grads.npy", G)
    np.save(out_dir / "resid.npy", R)
    np.save(out_dir / "answer_grads.npy", A)
    pd.DataFrame(rows).to_parquet(out_dir / "index.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "case": args.case, "words": ss.CASE_WORDS[args.case],
        "layers": layers, "n_images": len(rows),
        "grad_norm_median": {str(l): float(np.median(np.linalg.norm(
            G[:, j].astype(np.float32), axis=-1))) for j, l in enumerate(layers)},
        "seconds": round(time.time() - t0)}, indent=2))
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layers", default=LAYERS_DEFAULT)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--causal-image", action="store_true",
                    help="strip mm_token_type_ids: a causal mask over the "
                         "image tokens (gemma-4-12b ablation)")
    ap.add_argument("--image-mask", choices=list(ss.IMAGE_MASK_MODES),
                    default=None,
                    help="how the image tokens attend to each other; "
                         "see single_steer.install_image_mask")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
