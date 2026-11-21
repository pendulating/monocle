"""Is the causal effect of a patch linearly readable from its residual?

Why this question decides the lens
----------------------------------
A Jlens, an Rlens, a mean gradient, and a grad-times-activation map are ALL
linear in the patch residual `h_p`. They differ only in the vector they use.
So a supervised probe, fitted on measured ablations, sets the CEILING for every
one of them.

- If the probe fails on held-out images, no linear instrument can work, and a
  better corpus cannot rescue the lens.
- If the probe succeeds, the lens has the wrong vector, and the corpus and the
  fit are worth more work.

The design
----------
Ridge regression from `h_p` to the measured ablation of patch `p`. The folds
split by IMAGE, never by patch, so a train patch and a test patch never share a
photograph. The truth is standardised inside each image, because `c0` and the
scale of the effect change from photograph to photograph.

The controls
------------
| Control | What it shows |
|---|---|
| `shuffled` | the truth is permuted across images; held-out r must fall to 0 |
| `top16` | overlap of the strongest cells, against a chance of 4.0 of 16 |

Usage:
    python scripts/linear_probe.py --acts outputs/_monocle/acts_subway/subway_safety/acts.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path("/share/ju/matt/mllmsci")
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from make_ranking import pool  # noqa: E402


def zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed form, on the smaller side of the two normal equations."""
    n, d = x.shape
    if n >= d:
        a = x.T @ x + alpha * np.eye(d)
        return np.linalg.solve(a, x.T @ y)
    k = x @ x.T + alpha * np.eye(n)
    return x.T @ np.linalg.solve(k, y)


def run(acts: dict, layer: int, alphas: list[float], n_folds: int,
        seed: int, shuffle: bool) -> dict:
    from scipy.stats import pearsonr, spearmanr

    ids = sorted(acts)
    rng = np.random.default_rng(seed)
    key = f"h{layer}"
    ids = [i for i in ids if key in acts[i]]
    if len(ids) < n_folds:
        raise SystemExit(f"only {len(ids)} images hold {key}")

    truth = {i: zscore(np.asarray(acts[i]["truth"], float).ravel()) for i in ids}
    if shuffle:
        order = list(ids)
        rng.shuffle(order)
        truth = {i: truth[j] for i, j in zip(ids, order)}

    folds = np.array_split(rng.permutation(len(ids)), n_folds)
    rs, sps, ovs = [], [], []
    K, N = 16, 64
    for f in folds:
        test = [ids[j] for j in f]
        train = [i for i in ids if i not in set(test)]
        xtr = np.concatenate([acts[i][key] for i in train]).astype(np.float64)
        ytr = np.concatenate([truth[i] for i in train])
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
        xtr = (xtr - mu) / sd
        # Pick alpha on a held-out slice of the TRAIN images only.
        cut = max(1, len(train) // 5)
        inner_te, inner_tr = train[:cut], train[cut:]
        best, best_r = alphas[0], -np.inf
        xi = np.concatenate([acts[i][key] for i in inner_tr]).astype(np.float64)
        xi = (xi - mu) / sd
        yi = np.concatenate([truth[i] for i in inner_tr])
        xv = np.concatenate([acts[i][key] for i in inner_te]).astype(np.float64)
        xv = (xv - mu) / sd
        yv = np.concatenate([truth[i] for i in inner_te])
        for a in alphas:
            w = ridge_fit(xi, yi, a)
            r = pearsonr(xv @ w, yv)[0]
            if np.isfinite(r) and r > best_r:
                best_r, best = r, a
        w = ridge_fit(xtr, ytr, best)
        for i in test:
            p = ((acts[i][key].astype(np.float64) - mu) / sd) @ w
            t = truth[i]
            if p.std() == 0:
                continue
            rs.append(pearsonr(p, t)[0])
            sps.append(spearmanr(p, t).statistic)
            pp, tt = pool(p, 2), pool(t, 2)
            ovs.append(len(set(np.argsort(-pp)[:K]) & set(np.argsort(-tt)[:K])))
    return {"layer": layer, "n": len(rs), "r": float(np.mean(rs)),
            "spearman": float(np.mean(sps)), "top16": float(np.mean(ovs))}



def block_features(h: np.ndarray, group: int) -> np.ndarray:
    """Sum the patch residuals inside each block.

    A linear instrument scores a block by the SUM of its patch scores:
        sum over p in B of <w, h_p>  =  <w, sum over p in B of h_p>
    So the block feature is the summed residual, and a ridge fit on it finds
    the BEST additive instrument for the block test. That is the exact ceiling
    for a lens, an Rlens, a mean gradient and a grad-times-activation map.
    """
    n_patch, d = h.shape
    side = int(round(n_patch ** 0.5))
    if side * side != n_patch:
        raise SystemExit(f"{n_patch} patches is not a square grid")
    if side % group:
        raise SystemExit(f"a {side}x{side} grid does not divide by {group}")
    g = h.reshape(side, side, d)
    n = side // group
    return g.reshape(n, group, n, group, d).sum(axis=(1, 3)).reshape(n * n, d)


def run_blocks(acts: dict, truth_by_image: dict, layer: int, group: int,
               alphas: list[float], n_folds: int, seed: int,
               shuffle: bool) -> dict:
    """The best ADDITIVE instrument, fitted against measured block effects."""
    from scipy.stats import pearsonr, spearmanr

    key = f"h{layer}"
    ids = sorted(i for i in acts if key in acts[i] and i in truth_by_image)
    if len(ids) < n_folds:
        raise SystemExit(f"only {len(ids)} images hold {key} and a block truth")
    rng = np.random.default_rng(seed)

    feats = {i: block_features(np.asarray(acts[i][key], np.float64), group)
             for i in ids}
    y = {i: zscore(np.asarray(truth_by_image[i], float).ravel()) for i in ids}
    for i in ids:
        if feats[i].shape[0] != y[i].size:
            raise SystemExit(f"{i}: {feats[i].shape[0]} blocks against "
                             f"{y[i].size} truth cells")
    if shuffle:
        order = list(ids)
        rng.shuffle(order)
        y = {i: y[j] for i, j in zip(ids, order)}

    folds = np.array_split(rng.permutation(len(ids)), n_folds)
    rs, sps, ovs = [], [], []
    K = max(1, feats[ids[0]].shape[0] // 4)
    for f in folds:
        test = [ids[j] for j in f]
        train = [i for i in ids if i not in set(test)]
        xtr = np.concatenate([feats[i] for i in train])
        ytr = np.concatenate([y[i] for i in train])
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
        xtr = (xtr - mu) / sd
        cut = max(1, len(train) // 5)
        inner_te, inner_tr = train[:cut], train[cut:]
        xi = (np.concatenate([feats[i] for i in inner_tr]) - mu) / sd
        yi = np.concatenate([y[i] for i in inner_tr])
        xv = (np.concatenate([feats[i] for i in inner_te]) - mu) / sd
        yv = np.concatenate([y[i] for i in inner_te])
        best, best_r = alphas[0], -np.inf
        for a in alphas:
            w = ridge_fit(xi, yi, a)
            r = pearsonr(xv @ w, yv)[0]
            if np.isfinite(r) and r > best_r:
                best_r, best = r, a
        w = ridge_fit(xtr, ytr, best)
        for i in test:
            pred = ((feats[i] - mu) / sd) @ w
            t = y[i]
            if pred.std() == 0:
                continue
            rs.append(pearsonr(pred, t)[0])
            sps.append(spearmanr(pred, t).statistic)
            ovs.append(len(set(np.argsort(-pred)[:K])
                           & set(np.argsort(-t)[:K])))
    return {"layer": layer, "n": len(rs), "r": float(np.mean(rs)),
            "spearman": float(np.mean(sps)), "top16": float(np.mean(ovs))}


def main(argv: list[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--block-truth", default=None,
                    help="a parquet of MEASURED block effects. It switches the probe to the block test, which is the exact ceiling for any additive instrument.")
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)

    blob = torch.load(args.acts, map_location="cpu", weights_only=False)
    acts = blob["acts"]
    layers = args.layers or blob["layers"]
    print(f"case={blob['case']} images={len(acts)} layers={layers}")
    print(f"chance top16 = 4.00 of 16\n")
    print(f"{'layer':>5} {'arm':>9} {'n':>4} {'r':>8} {'spearman':>9} {'top16':>7}")
    truth_blocks = None
    if args.block_truth:
        import pandas as pd
        df = pd.read_parquet(args.block_truth)
        idc = "image_id" if "image_id" in df.columns else "target"
        vc = "values" if "values" in df.columns else "causal"
        truth_blocks = {str(r[idc]): np.asarray(r[vc], float)
                        for _, r in df.iterrows()}
        print(f"block truth: {len(truth_blocks)} images, "
              f"{len(next(iter(truth_blocks.values())))} cells")
    for layer in layers:
        for shuffle in (False, True):
            s = (run_blocks(acts, truth_blocks, layer, args.group, args.alphas,
                            args.folds, args.seed, shuffle)
                 if truth_blocks is not None else
                 run(acts, layer, args.alphas, args.folds, args.seed, shuffle))
            print(f"{s['layer']:>5} {'shuffled' if shuffle else 'real':>9} "
                  f"{s['n']:>4} {s['r']:>+8.3f} {s['spearman']:>+9.3f} "
                  f"{s['top16']:>7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
