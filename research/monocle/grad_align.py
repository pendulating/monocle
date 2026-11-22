"""The model's own sensitivity direction at the answer position.

For each image, one backward pass gives ``g_l = d(logit_pos - logit_neg) /
d h_l[answer]`` at each block ``l``. That is the direction in which the
rest of the network is most sensitive, so an injection along it has the
highest first-order gain of any unit direction.

The module records ``g_l`` on the calibration images (to build the
``grad`` steering arm, the mean unit gradient) and on the eval images (to
compare with the axis and the pinv vector, image by image). It prints the
cosines against the vectors of the 4-arm sweep when they exist.

Outputs, under ``<out-dir>/<case>/``:

| File | Content |
|---|---|
| ``grad_dirs.npz`` | ``v_grad_L<l>``: the mean unit gradient over the calibration images |
| ``eval_grads.npy`` | ``[n_eval, n_layers, d_model]`` fp16, the per-image unit gradients |
| ``cosines.parquet`` | per image, per block: cos of ``g`` with the axis, the pinv vector, and the mean gradient |

Usage:
    sbatch --time=00:30:00 research/monocle/monocle.sub monocle.grad_align \\
        --case restaurants --n-images 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    parse_int_list,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/grad_align")
VECTORS_DEFAULT = str(REPO / "outputs/_monocle/single_steer_axis")


def log(msg: str) -> None:
    print(f"[grad-align] {msg}", flush=True)


def answer_gradients(
    model, lens_model, inputs: dict, layers: list[int],
    tok_pos: int, tok_neg: int,
) -> tuple[dict[int, torch.Tensor], float]:
    """``g_l`` at the answer position for each block, plus the judgment.

    One forward with the graph rooted at the lowest block, then one
    backward from the contrast. The gradient at every recorded block comes
    from that single backward.
    """
    try:
        from jlens.hooks import ActivationRecorder
    except ModuleNotFoundError:
        sys.path.insert(0, str(REPO / "sub/jacobian-lens"))
        from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    root = min(layers)
    at = sorted(set(layers) | {final})
    with torch.enable_grad(), ActivationRecorder(
            lens_model.layers, at=at, start_graph_at=root) as rec:
        out = model(**inputs, use_cache=False)
        del out
        h_final = rec.activations[final][0, -1]
        logits = lens_model.unembed(h_final).float()
        contrast = logits[tok_pos] - logits[tok_neg]
        sources = [rec.activations[l] for l in layers]
        grads = torch.autograd.grad(contrast, sources, allow_unused=True)
    out_g: dict[int, torch.Tensor] = {}
    for l, g in zip(layers, grads):
        if g is None:
            raise RuntimeError(f"no gradient reached block {l}")
        out_g[l] = g[0, -1].detach().float().cpu()
    return out_g, float(contrast)


def run(args: argparse.Namespace) -> int:
    layers = parse_int_list(args.layers)
    out_dir = Path(args.out_dir) / args.case
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    calib, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    build = ss.make_build_inputs(proc, tmpl, args.case)

    def unit(g: torch.Tensor) -> torch.Tensor:
        return g / (g.norm() + 1e-12)

    # Calibration: the mean unit gradient per block is the `grad` arm.
    sums = {l: torch.zeros(lens_model.d_model) for l in layers}
    for r in calib:
        g, _ = answer_gradients(model, lens_model, build(r["p"]), layers,
                                tok_pos, tok_neg)
        for l in layers:
            sums[l] += unit(g[l])
    dirs = {f"v_grad_L{l}": unit(sums[l]).numpy() for l in layers}
    np.savez(out_dir / "grad_dirs.npz", layers=np.array(layers), **dirs)
    log(f"grad arm from {len(calib)} calibration images ({time.time() - t0:.0f}s)")

    vec_file = Path(args.vectors) / args.case / "vectors.npz"
    ref = dict(np.load(vec_file)) if vec_file.exists() else {}
    if ref:
        log("cos(mean grad, axis) " + " ".join(
            f"L{l}={float(dirs[f'v_grad_L{l}'] @ ref[f'v_axis_L{l}']):+.3f}"
            for l in layers))
        log("cos(mean grad, pinv) " + " ".join(
            f"L{l}={float(dirs[f'v_grad_L{l}'] @ ref[f'v_signal_L{l}']):+.3f}"
            for l in layers))

    # Eval: per-image unit gradients and their cosines.
    rows, store = [], []
    for i, r in enumerate(ev):
        g, c = answer_gradients(model, lens_model, build(r["p"]), layers,
                                tok_pos, tok_neg)
        per = []
        for l in layers:
            u = unit(g[l])
            per.append(u.to(torch.float16).numpy())
            row = {"image_path": r["p"], "layer": l, "c": c,
                   "grad_norm": float(g[l].norm()),
                   "cos_mean_grad": float(u.numpy() @ dirs[f"v_grad_L{l}"])}
            if ref:
                row["cos_axis"] = float(u.numpy() @ ref[f"v_axis_L{l}"])
                row["cos_pinv"] = float(u.numpy() @ ref[f"v_signal_L{l}"])
            rows.append(row)
        store.append(np.stack(per))
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(ev)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "cosines.parquet", index=False)
    np.save(out_dir / "eval_grads.npy", np.stack(store))
    for l, g in df.groupby("layer"):
        line = (f"L{l}: |g| median {g['grad_norm'].median():.3f}, "
                f"cos(g, mean grad) {g['cos_mean_grad'].mean():+.3f}")
        if ref:
            line += (f", cos(g, axis) {g['cos_axis'].mean():+.3f}, "
                     f"cos(g, pinv) {g['cos_pinv'].mean():+.3f}")
        log(line)
    log(f"wrote {out_dir} ({time.time() - t0:.0f}s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layers", default="41,42,44,45")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--vectors", default=VECTORS_DEFAULT,
                    help="root of the 4-arm sweep, for the reference vectors")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
