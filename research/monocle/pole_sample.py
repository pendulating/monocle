"""Rank the eval images by the answer residual's projection on each pole.

Each pole gets its own vector: the gradient of that pole's logit with
respect to the final residual (unit, averaged over the calibration
images), pulled back to block l with ``pinv(J_l)``. The contrast vector of
the steering sweeps is the difference of the two. Per image and block the
script records the centered projection of the answer residual on each
pole vector and on the contrast vector, with the measured logit gap.

Outputs under ``<out-dir>/<case>/``: ``projections.parquet`` (image_path,
layer, proj_<pos>, proj_<neg>, proj_contrast, c), ``vectors.npz``,
``summary.json`` (cosines between the vectors, correlations of the
projections). ``research/scripts/pole_sample_figure.py`` draws the contact sheets.

    sbatch research/monocle/monocle.sub monocle.pole_sample --case restaurants
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import (  # noqa: E402
    load_lens,
    load_model_and_lens,
    pull_back_pinv,
    split_calibration_eval,
)

OUT_DIR_DEFAULT = str(REPO / "outputs/_monocle/pole_sample")


def log(m: str) -> None:
    print(f"[pole-sample] {m}", flush=True)


def record(model, lens_model, inputs: dict, layers: list[int]):
    """Answer-position residuals at ``layers`` and at the final block."""
    from jlens.hooks import ActivationRecorder

    final = lens_model.n_layers - 1
    at = sorted(set(layers) | {final})
    inputs["use_cache"] = False
    with ActivationRecorder(lens_model.layers, at=at) as rec, torch.no_grad():
        model(**inputs)
    return {i: rec.activations[i][0, -1].detach().float().clone() for i in at}, final


def pole_grad(lens_model, h_final: torch.Tensor, tok: int) -> torch.Tensor:
    h = h_final.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = lens_model.unembed(h).float()
        logits[tok].backward()
    g = h.grad.detach()
    return g / (g.norm() + 1e-12)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="restaurants")
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--layers", default="41,42,44,45")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--lens", default=ss.LENS_DEFAULT)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out_dir) / args.case
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    lens = load_lens(args.lens)
    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    w_pos, w_neg = ss.CASE_WORDS[args.case]
    paths = ss.load_images(args.case, args.kind)
    calib, ev = split_calibration_eval(
        [{"p": p} for p in paths], args.n_calib, args.n_images)
    build = ss.make_build_inputs(proc, tmpl, args.case)
    log(f"{args.case}: poles {w_pos}/{w_neg}, layers {layers}, "
        f"{len(calib)} calib + {len(ev)} eval")

    # Pole directions in the final basis, averaged over the calibration set.
    g_pos = g_neg = None
    for r in calib:
        acts, final = record(model, lens_model, build(r["p"]), layers)
        a, b = pole_grad(lens_model, acts[final], tok_pos), \
            pole_grad(lens_model, acts[final], tok_neg)
        g_pos = a if g_pos is None else g_pos + a
        g_neg = b if g_neg is None else g_neg + b
    d_pos = (g_pos / g_pos.norm()).cpu()
    d_neg = (g_neg / g_neg.norm()).cpu()
    d_con = d_pos - d_neg
    d_con = d_con / d_con.norm()
    vecs = {"d_pos": d_pos.numpy(), "d_neg": d_neg.numpy(),
            "d_contrast": d_con.numpy(), "layers": np.array(layers)}
    unit = lambda v: v / (v.norm() + 1e-12)  # noqa: E731
    V = {}
    for l in layers:
        J = lens.jacobians[l]
        V[l] = {"pos": unit(pull_back_pinv(J, d_pos)),
                "neg": unit(pull_back_pinv(J, d_neg)),
                "contrast": unit(pull_back_pinv(J, d_con))}
        for k, v in V[l].items():
            vecs[f"v_{k}_L{l}"] = v.cpu().numpy()
    np.savez(out / "vectors.npz", **vecs)

    # Eval residuals and projections.
    H = {l: [] for l in layers}
    cs, ps = [], []
    for r in ev:
        acts, final = record(model, lens_model, build(r["p"]), layers)
        with torch.no_grad():
            logits = lens_model.unembed(acts[final]).float()
        cs.append(float(logits[tok_pos] - logits[tok_neg]))
        ps.append(r["p"])
        for l in layers:
            H[l].append(acts[l].cpu())
    rows = []
    for l in layers:
        h = torch.stack(H[l])
        hc = h - h.mean(0, keepdim=True)
        pp = (hc @ V[l]["pos"].cpu()).numpy()
        pn = (hc @ V[l]["neg"].cpu()).numpy()
        pc = (hc @ V[l]["contrast"].cpu()).numpy()
        for i, p in enumerate(ps):
            rows.append({"image_path": p, "layer": l, "c": cs[i],
                         f"proj_{w_pos}": float(pp[i]),
                         f"proj_{w_neg}": float(pn[i]),
                         "proj_contrast": float(pc[i])})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "projections.parquet", index=False)

    summ = {"case": args.case, "words": [w_pos, w_neg],
            "cos_final_pos_neg": float(d_pos @ d_neg), "per_layer": {}}
    for l in layers:
        x = df[df.layer == l]
        summ["per_layer"][str(l)] = {
            "cos_v_pos_v_neg": float(V[l]["pos"].cpu() @ V[l]["neg"].cpu()),
            "r_proj_pos_neg": float(np.corrcoef(x[f"proj_{w_pos}"], x[f"proj_{w_neg}"])[0, 1]),
            "r_proj_pos_c": float(np.corrcoef(x[f"proj_{w_pos}"], x["c"])[0, 1]),
            "r_proj_neg_c": float(np.corrcoef(x[f"proj_{w_neg}"], x["c"])[0, 1]),
            "r_proj_contrast_c": float(np.corrcoef(x["proj_contrast"], x["c"])[0, 1]),
        }
        log(f"block {l}: " + json.dumps({k: round(v, 3) for k, v in summ["per_layer"][str(l)].items()}))
    summ["seconds"] = int(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(f"wrote {out} ({summ['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
