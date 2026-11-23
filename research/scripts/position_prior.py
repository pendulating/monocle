"""How much of a patch map is only the position of the cell?

The problem
-----------
A measured ablation map is not flat in space. The middle cells of a photograph
change the judgment more than the corner cells. A probe that knows NOTHING
about the photograph can still score above chance, because it can learn this
one fixed spatial pattern.

So every claim about a patch instrument needs this control. The control is the
POSITION-ONLY map: the mean measured truth at each cell, fitted on the training
images and applied unchanged to every test image.

What the script reports
-----------------------
| Column | Meaning |
|---|---|
| `r_pos` | the position-only map against the held-out truth |
| `r_probe` | the fitted additive probe against the same truth |
| `gain` | `r_probe` minus `r_pos`, the image-specific part |

The script also writes the position-only map as a ranking parquet. Use it as a
null arm of the deletion test.

Usage:
    python research/scripts/position_prior.py \
      --truth outputs/_monocle/causal_maps/subway_safety_g2_absent/maps.parquet \
      --acts outputs/_monocle/acts_subway/subway_safety/acts.pt \
      --out outputs/_monocle/rankings/subway_position_prior.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "scripts"))

from research.scripts.linear_probe import block_features, ridge_fit, zscore  # noqa: E402


def load_truth(path: str) -> dict:
    df = pd.read_parquet(path)
    idc = "image_id" if "image_id" in df.columns else "target"
    vc = "values" if "values" in df.columns else "causal"
    return {str(r[idc]): np.asarray(r[vc], float).ravel()
            for _, r in df.iterrows()}


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import pearsonr, spearmanr, wilcoxon

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--acts", default=None)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    truth = {k: zscore(v) for k, v in load_truth(args.truth).items()}
    ids = sorted(truth)
    n_cell = truth[ids[0]].size
    print(f"images={len(ids)} cells={n_cell}")

    feats = None
    if args.acts:
        import torch
        blob = torch.load(args.acts, map_location="cpu", weights_only=False)
        key = f"h{args.layer}"
        acts = blob["acts"]
        ids = [i for i in ids if i in acts and key in acts[i]]
        feats = {i: block_features(np.asarray(acts[i][key], np.float64),
                                   args.group) for i in ids}
        print(f"probe layer={args.layer} images with residuals={len(ids)}")

    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(ids)), args.folds)
    rows, keep = [], {}
    for f in folds:
        test = [ids[j] for j in f]
        train = [i for i in ids if i not in set(test)]
        prior = np.mean([truth[i] for i in train], axis=0)
        w = None
        if feats is not None:
            xtr = np.concatenate([feats[i] for i in train])
            ytr = np.concatenate([truth[i] for i in train])
            mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
            cut = max(1, len(train) // 5)
            inner_te, inner_tr = train[:cut], train[cut:]
            xi = (np.concatenate([feats[i] for i in inner_tr]) - mu) / sd
            yi = np.concatenate([truth[i] for i in inner_tr])
            xv = (np.concatenate([feats[i] for i in inner_te]) - mu) / sd
            yv = np.concatenate([truth[i] for i in inner_te])
            best, best_r = args.alphas[0], -np.inf
            for a in args.alphas:
                ww = ridge_fit(xi, yi, a)
                r = pearsonr(xv @ ww, yv)[0]
                if np.isfinite(r) and r > best_r:
                    best_r, best = r, a
            w = ridge_fit((xtr - mu) / sd, ytr, best)
        for i in test:
            t = truth[i]
            row = {"image_id": i, "r_pos": pearsonr(prior, t)[0],
                   "sp_pos": spearmanr(prior, t).statistic}
            if w is not None:
                p = ((feats[i] - mu) / sd) @ w
                row["r_probe"] = pearsonr(p, t)[0]
                row["sp_probe"] = spearmanr(p, t).statistic
                # The probe MINUS the prior: does the residual add anything?
                # Warning: take the prior out of BOTH sides. If you leave it in
                # the truth, the centre step of Pearson r lets it leak back.
                def strip(v):
                    return v - prior * (np.dot(v, prior) / np.dot(prior, prior))
                q, tq = strip(p), strip(t)
                row["r_orth"] = pearsonr(q, t)[0] if q.std() > 0 else np.nan
                row["r_spec"] = (pearsonr(q, tq)[0]
                                 if q.std() > 0 and tq.std() > 0 else np.nan)
                keep[i] = (q, tq)
            rows.append(row)
    df = pd.DataFrame(rows)
    print()
    for c in df.columns:
        if c == "image_id":
            continue
        v = df[c].dropna().to_numpy()
        print(f"{c:>9}  mean {v.mean():+.4f}   median {np.median(v):+.4f}"
              f"   >0 in {int((v > 0).sum())}/{len(v)}")
    if "r_probe" in df:
        g = (df["r_probe"] - df["r_pos"]).dropna().to_numpy()
        stat = wilcoxon(g) if np.any(g != 0) else None
        print(f"\n     gain  mean {g.mean():+.4f}   median {np.median(g):+.4f}"
              f"   >0 in {int((g > 0).sum())}/{len(g)}"
              + (f"   wilcoxon p={stat.pvalue:.3f}" if stat else ""))

    if len(keep) > 2:
        # The MATCH test. Score each prediction against its own photograph and
        # against every other photograph. A prediction that carries a second
        # fixed spatial pattern scores the same on both. A prediction that
        # reads THIS photograph scores higher on the diagonal.
        names = sorted(keep)
        diag, off = [], []
        for a in names:
            qa = keep[a][0]
            for b in names:
                tb = keep[b][1]
                if qa.std() == 0 or tb.std() == 0:
                    continue
                r = pearsonr(qa, tb)[0]
                (diag if a == b else off).append(r)
        d, o = np.array(diag), np.array(off)
        print(f"\n  MATCH  own {d.mean():+.4f}   other {o.mean():+.4f}"
              f"   difference {d.mean() - o.mean():+.4f}")
        # A rank test needs no assumption about the shape of the spread.
        ranks = []
        for a in names:
            qa = keep[a][0]
            sc = {b: pearsonr(qa, keep[b][1])[0] for b in names}
            order = sorted(names, key=lambda b: -sc[b])
            ranks.append(order.index(a) + 1)
        ranks = np.array(ranks)
        mid = (len(names) + 1) / 2
        print(f"  MATCH  rank of the own photograph: mean {ranks.mean():.2f}"
              f"   chance {mid:.2f}   first place {int((ranks == 1).sum())}"
              f"/{len(ranks)}   chance {len(ranks) / len(names):.1f}")
        boot = np.array([rng.choice(ranks, ranks.size).mean()
                         for _ in range(4000)])
        print(f"  MATCH  bootstrap CI of the mean rank "
              f"[{np.percentile(boot, 2.5):.2f}, "
              f"{np.percentile(boot, 97.5):.2f}]")

    if args.out:
        # Warning: leave the image OUT of its own prior. The grand mean holds
        # the truth of the test image, and that leak makes the null look
        # stronger than it is.
        tot = np.sum([truth[i] for i in ids], axis=0)
        n = len(ids)
        out = pd.DataFrame({
            "image_id": ids,
            "values": [(tot - truth[i]) / (n - 1) for i in ids]})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out)
        print(f"\nwrote the position-only ranking to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
