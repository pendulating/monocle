"""E4 of the lens-refusal probe: steer the refused images (H2).

H2 says the contrast vector at block 42 moves the model from the abstain
label to a pole. For 1 probe, on `--n-images` refused images of the E1
census (seeded draw), the module adds `alpha * scale * v` to the block-42
output at the answer position and reads the top token and `c` at each
dose. Two arms:

| Arm | Vector |
|-----|--------|
| `signal` | the probe contrast vector under lens C (`v_contrast_C_L42`) |
| `control` | a random unit direction in the final basis, pulled back with `pinv(J_42)` under the same lens, unit-normed: the same construction with no content |

The scale is the median centered residual norm at block 42 over the
census (`jlens_steer.scale_from_samples`, mode `centered`), the scale
the CVPR steering used. The doses cover the window that steering found,
alpha in [-2, +2], with alpha = 0 as the baseline through the hook path.

The go rule: at some dose the signal turns more than 50% of the refused
images into a pole, and the control turns under 20%.

Writes `outputs/_monocle/probe/steer/<probe>/{steer.parquet,summary.json}`.

    sbatch monocle/monocle.sub monocle.probe_steer --probe race --n-images 200
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import probe_frame as pf  # noqa: E402
from monocle.jlens_steer import (  # noqa: E402
    load_lens, load_model_and_lens, pull_back_pinv, scale_from_samples)
from monocle.probe_vectors import LENSES, load_scores  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"
DEFAULT_ALPHAS = "-2,-1,-0.5,0,0.5,1,2"


def log(m: str) -> None:
    print(f"[probe-steer] {m}", flush=True)


def make_batch_hook(add_vec: torch.Tensor):
    """Add `add_vec` to the block output at the LAST position of every row."""
    def hook(module, inputs, output):
        is_tuple = not torch.is_tensor(output)
        t = output[0] if is_tuple else output
        new = t.clone()
        new[:, -1, :] = new[:, -1, :] + add_vec.to(new.dtype).to(new.device)
        return (new, *tuple(output[1:])) if is_tuple else new
    return hook


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True, choices=list(pf.PROBES))
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--split", choices=["refused", "answered"], default="refused")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--lens", choices=list(LENSES), default="C")
    ap.add_argument("--tag", default="", help="a suffix for the output directory")
    ap.add_argument("--scores-dir", default=str(ROOT / "scores"))
    ap.add_argument("--vectors-dir", default=str(ROOT / "vectors"))
    ap.add_argument("--out-dir", default=str(ROOT / "steer"))
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    args = ap.parse_args(argv)
    alphas = [float(x) for x in args.alphas.split(",")]
    from PIL import Image

    LAYER = args.layer
    out = Path(args.out_dir) / args.probe / f"{args.split}_{args.lens}_L{LAYER}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, H = load_scores(args.probe, Path(args.scores_dir))
    pool = df[~df["answered"]] if args.split == "refused" else df[df["answered"]]
    rng = np.random.default_rng(args.seed)
    k = min(args.n_images, len(pool))
    paths = pool["image_path"].to_numpy()[rng.choice(len(pool), size=k, replace=False)].tolist()
    base = df.set_index("image_path").loc[paths]
    scale = scale_from_samples(torch.tensor(H[LAYER]), "centered")
    vecs = np.load(Path(args.vectors_dir) / args.probe / "vectors.npz")
    v_sig = torch.tensor(vecs[f"v_contrast_{args.lens}_L{LAYER}"], dtype=torch.float32)

    lens = load_lens(str(LENSES[args.lens]))
    J = lens.jacobians[LAYER].to("cuda:0")
    g = torch.Generator().manual_seed(args.seed)
    d_r = torch.randn(J.shape[0], generator=g)
    d_r = d_r / d_r.norm()
    v_ctl = pull_back_pinv(J, d_r.to("cuda:0")).cpu()
    v_ctl = v_ctl / (v_ctl.norm() + 1e-12)
    del lens, J
    torch.cuda.empty_cache()
    log(f"{args.probe}: {k} {args.split} images, scale {scale:.2f}, "
        f"cos(signal, control) = {float(v_sig @ v_ctl):+.3f}, alphas {alphas}")

    proc, model, tmpl, lens_model, _ = load_model_and_lens(args.model_dir)
    tok = proc.tokenizer
    id_pos, id_neg, id_abs = pf.token_ids(tok, args.probe)
    sets = pf.label_sets(tok, args.probe)
    text = pf.render_chat(tok, tmpl, pf.user_text(args.probe))
    block = lens_model.layers[LAYER]

    rows = []
    n_fwd = 0
    for b0 in range(0, k, args.batch):
        chunk = paths[b0:b0 + args.batch]
        imgs = [[Image.open(q).convert("RGB")] for q in chunk]
        inputs = dict(proc(text=[text] * len(chunk), images=imgs,
                           return_tensors="pt").to("cuda:0"))
        am = inputs.get("attention_mask")
        if am is not None and not bool(am.all()):
            raise RuntimeError("padding inside a batch")
        inputs["use_cache"] = False
        for arm, v in (("signal", v_sig), ("control", v_ctl)):
            for a in alphas:
                handle = block.register_forward_hook(make_batch_hook(a * scale * v))
                try:
                    with torch.no_grad():
                        logits = model(**inputs).logits[:, -1, :].float().cpu()
                finally:
                    handle.remove()
                n_fwd += 1
                top = logits.argmax(dim=1).numpy()
                for i, q in enumerate(chunk):
                    cls = pf.classify_top(int(top[i]), sets)
                    rows.append({"image_path": q, "arm": arm, "alpha": a,
                                 "c": float(logits[i, id_pos] - logits[i, id_neg]),
                                 "logit_abstain": float(logits[i, id_abs]),
                                 "top1": tok.decode([int(top[i])]), "top_class": cls,
                                 "answered": cls in ("pos", "neg"),
                                 "base_c": float(base.loc[q, "c"]),
                                 "base_class": str(base.loc[q, "top_class"])})
        if (b0 // args.batch) % 5 == 0:
            log(f"  {b0 + len(chunk)}/{k} images, {n_fwd} forwards, "
                f"{time.time() - t0:.0f}s")
    res = pd.DataFrame(rows)
    res.to_parquet(out / "steer.parquet", index=False)

    tab = res.groupby(["arm", "alpha"]).agg(
        answered=("answered", "mean"), pos=("top_class", lambda s: (s == "pos").mean()),
        neg=("top_class", lambda s: (s == "neg").mean()),
        abstain=("top_class", lambda s: (s == "abstain").mean()),
        other=("top_class", lambda s: (s == "other").mean()),
        c_mean=("c", "mean")).reset_index()
    go_doses = []
    for a in alphas:
        s = tab[(tab.arm == "signal") & (tab.alpha == a)]["answered"].iloc[0]
        cc = tab[(tab.arm == "control") & (tab.alpha == a)]["answered"].iloc[0]
        if s > 0.5 and cc < 0.2:
            go_doses.append(a)
    summ = {"probe": args.probe, "split": args.split, "lens": args.lens, "layer": LAYER,
            "n_images": int(k),
            "scale": scale, "cos_signal_control": float(v_sig @ v_ctl),
            "table": tab.round(3).to_dict("records"), "go_doses": go_doses,
            "go": bool(go_doses), "seconds": int(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(tab.round(3).to_string())
    log(f"go doses: {go_doses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
