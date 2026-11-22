"""How many directions does the contrast gradient need?

The premise under test
----------------------
A Jlens or an Rlens is ONE constant matrix. It can only work if the per-patch
gradient `d c / d h_p` points a shared way for every patch and every image.
The mean-gradient run reports a cosine of 0.0004 at layer 0 and 0.47 at layer
30, so a single direction clearly does not serve the shallow layers.

The question here is softer, and it opens a route: does a SMALL BASIS serve
them? If 8 directions hold 80 percent of the energy, then the right instrument
is a small mixture of lenses, not one matrix, and not a per-image fit.

The measure
-----------
Take the sampled patch gradients, centre them, and read the eigenvalues of
their covariance. We report:

| Column | Meaning |
|---|---|
| `pr` | the participation ratio, an effective count of directions |
| `r80` | how many directions hold 80 percent of the energy |
| `top1` | the share that the single strongest direction holds |
| `mean_share` | the share that the MEAN direction holds |

`mean_share` is the number that judges the constant lens, because the constant
lens uses the mean and nothing else.

Usage:
    python research/scripts/gradient_rank.py \
        --samples outputs/_monocle/mean_grad_samples/subway_safety/grad_samples.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def spectrum(x: np.ndarray) -> dict:
    """Eigen-summary of the row set `x` of shape [n, d]."""
    n, d = x.shape
    mu = x.mean(axis=0)
    xc = x - mu
    # The Gram matrix is n x n and n is far below d, so it is the cheap route.
    g = xc @ xc.T / max(n - 1, 1)
    ev = np.linalg.eigvalsh(g)
    ev = np.clip(ev[::-1], 0.0, None)
    tot = ev.sum()
    if tot <= 0:
        return {"n": n, "pr": float("nan"), "r80": -1,
                "top1": float("nan"), "mean_share": float("nan")}
    p = ev / tot
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())
    r80 = int(np.searchsorted(np.cumsum(p), 0.80) + 1)
    # How much energy sits along the mean direction? That is all a constant
    # lens can use.
    unit = mu / (np.linalg.norm(mu) + 1e-12)
    proj = x @ unit
    mean_share = float((proj ** 2).sum() / (x ** 2).sum())
    return {"n": n, "pr": pr, "r80": r80, "top1": float(p[0]),
            "mean_share": mean_share}


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    blob = torch.load(args.samples, map_location="cpu", weights_only=False)
    layers = blob["layers"]
    print(f"case={blob['case']} images={blob['n_images']} "
          f"per_image={blob['per_image']}")
    print(f"{'layer':>5} {'n':>6} {'pr':>8} {'r80':>5} {'top1':>7} "
          f"{'mean_share':>11}")
    rows = []
    for layer in layers:
        x = blob["samples"][layer]
        if x is None or x.numel() == 0:
            continue
        s = spectrum(x.double().numpy())
        s["layer"] = int(layer)
        rows.append(s)
        print(f"{layer:>5} {s['n']:>6} {s['pr']:>8.2f} {s['r80']:>5} "
              f"{s['top1']:>7.3f} {s['mean_share']:>11.4f}")
    if args.out:
        import pandas as pd
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
