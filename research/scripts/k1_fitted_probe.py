"""How much of the causal map can a FITTED linear patch reader find?

`gxa` is analytic: 1 gradient, 1 inner product, nothing fitted. A fitted
probe says how much a linear reader of the same features can reach.

Which features, and why it matters
----------------------------------
The image tokens come BEFORE the question in the prompt. A causal decoder
thus gives the SAME residual stream at every image position under any
question. So:

| Feature set | Carries the question | The control |
|---|---|---|
| `h` | no | the gain is 0 by construction |
| `gh`, which is `g * h` | yes, through the backward pass | real |

A probe on `h` alone is a QUESTION-BLIND reader. Its number is still
worth having: it says how much of the map a reader can find with no
knowledge of what was asked. A probe on `g * h` is the fitted form of
`gxa`, whose plain sum over dimensions IS `gxa`.

The rules that keep it honest:

- The split groups BY PHOTOGRAPH. All 256 cells of a photograph go to
  the train side or to the test side together. A split over cells leaks.
- The per-dimension scale comes from the train fold only.
- The control applies the SAME fitted probe to features taken under
  ANOTHER case's question.
- The 2 by 2: a probe fitted on the WRONG question's features tests
  whether the map information is question-blind after all.
- A photograph-permutation null validates the plumbing.
- The final run fits on 60 photographs and tests on 60 more that share
  no image with them.

    python research/scripts/k1_fitted_probe.py --cases restaurants libraries
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from research.scripts.k1_separation import bh, boot_ci, sep_z  # noqa: E402

ALPHAS = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


@lru_cache(maxsize=16)
def _blob(path: str):
    """One read of a 1.8 GB dump, reused over the layers and features."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_acts(root: Path, case: str, layer: int,
              feat: str = "h") -> dict[str, tuple]:
    """The features and the truth of each photograph.

    `feat` is `h` for the residual, `g` for the gradient, or `gh` for
    the product of the 2. The plain sum of `gh` over the dimensions is
    the `gxa` instrument.
    """
    p = root / case / "acts.pt"
    if not p.exists():
        return {}
    blob = _blob(str(p))
    out = {}
    for img, rec in blob["acts"].items():
        hk, gk = f"h{layer}", f"g{layer}"
        if feat == "h" and hk in rec:
            x = np.asarray(rec[hk], dtype=np.float64)
        elif feat == "g" and gk in rec:
            x = np.asarray(rec[gk], dtype=np.float64)
        elif feat == "gh" and hk in rec and gk in rec:
            x = (np.asarray(rec[hk], dtype=np.float64)
                 * np.asarray(rec[gk], dtype=np.float64))
        else:
            continue
        t = np.asarray(rec["truth"], dtype=np.float64).ravel()
        if x.shape[0] == t.size:
            out[str(img)] = (x, t)
    return out


def prep(h: np.ndarray) -> np.ndarray:
    """Centre the cells of 1 photograph.

    Gemma holds massive activations in a few dimensions. Without the
    centring every photograph looks the same and a raw cosine is about
    1.0, which is an artifact and not a signal.
    """
    return h - h.mean(axis=0, keepdims=True)


def zscore(t: np.ndarray) -> np.ndarray:
    sd = t.std(ddof=1)
    return (t - t.mean()) / sd if sd > 0 else t * 0.0


class Gram:
    """The pooled normal equations, with 1 photograph easy to remove.

    A leave-1-out fit over 60 photographs needs 60 fits. Each one only
    drops 256 rows of about 15000, so we add every photograph once and
    then subtract the held-out one. The per-dimension scale comes out of
    the same sums, so it is still a TRAIN quantity: a scale is a
    diagonal, and a diagonal moves through the normal equations.
    """

    def __init__(self, data: dict, keys: list[str]):
        # Warning: never keep 1 d by d block for each photograph. At 3840
        # dimensions that is 118 MB each, and 60 of them fill the machine.
        # The held-out block is cheap to build again, so we build it again.
        self.data = data
        d = prep(data[keys[0]][0]).shape[1]
        self.S = np.zeros((d, d))
        self.b = np.zeros(d)
        self.sq = np.zeros(d)
        self.n = 0
        for k in keys:
            X = prep(data[k][0])
            y = zscore(data[k][1])
            self.S += X.T @ X
            self.b += X.T @ y
            self.sq += (X * X).sum(0)
            self.n += X.shape[0]

    def fit(self, alpha: float, drop=None):
        """`drop` takes 1 key or a list of keys. A list leaves out a case."""
        S, b, sq, n = self.S, self.b, self.sq, self.n
        if drop is not None:
            keys = [drop] if isinstance(drop, str) else list(drop)
            if keys:
                S, b, sq = S.copy(), b.copy(), sq.copy()
            for k in keys:
                X = prep(self.data[k][0])
                y = zscore(self.data[k][1])
                S -= X.T @ X
                b -= X.T @ y
                sq -= (X * X).sum(0)
                n -= X.shape[0]
        sd = np.sqrt(np.maximum(sq / max(n - 1, 1), 1e-12))
        Di = 1.0 / sd
        A = (S * Di[:, None]) * Di[None, :]
        A.flat[:: A.shape[0] + 1] += alpha
        w = np.linalg.solve(A, b * Di)
        return w * Di          # folded back, so it applies to raw X


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return prep(x) @ w


def prior_of(data: dict, keys: list[str], skip: str | None) -> np.ndarray:
    """The average truth map over every photograph but 1.

    Subway safety carries the strongest average layout of any case, so a
    probe could learn where a cell usually matters and score above 0 on
    any photograph. This takes that away.
    """
    got = [zscore(data[k][1]) for k in keys if k != skip]
    return np.mean(got, axis=0) if got else np.zeros(1)


def strip(t: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Take the position prior out of 1 photograph's truth."""
    if prior.size != t.size:
        return t
    A = np.column_stack([np.ones(t.size), prior])
    beta, *_ = np.linalg.lstsq(A, t, rcond=None)
    return t - A @ beta


def pick_alpha(data: dict, keys: list[str], rng, folds: int = 3) -> float:
    """An inner split over photographs, never over cells."""
    ks = list(keys)
    rng.shuffle(ks)
    parts = np.array_split(np.arange(len(ks)), folds)
    best, best_a = -np.inf, ALPHAS[0]
    for a in ALPHAS:
        got = []
        for i in range(folds):
            te = [ks[j] for j in parts[i]]
            g = Gram(data, [k for k in ks if k not in te])
            w = g.fit(a)
            for k in te:
                got.append(sep_z(predict(data[k][0], w), data[k][1]))
        m = float(np.nanmean(got))
        if m > best:
            best, best_a = m, a
    return best_a


def run(case: str, layer: int, feat: str, own: Path, xq: Path, ho: Path,
        ho_xq: Path, rng, n_boot: int, n_null: int,
        resid_prior: bool = False) -> list[dict]:
    A = load_acts(own, case, layer, feat)
    B = load_acts(xq, case, layer, feat)
    keys = sorted(set(A) & set(B))
    if not keys:
        return []
    rows = []
    prior_keys = sorted(A)

    def truth(k: str, data: dict, skip: str | None = None) -> np.ndarray:
        t = data[k][1]
        if not resid_prior:
            return t
        return strip(t, prior_of(A, prior_keys, skip))

    # 0. the analytic baseline, through the SAME pipeline. The plain sum
    # of `g * h` over the dimensions IS `gxa`, so a weight of all ones
    # gives the head-to-head with nothing fitted.
    if feat == "gh":
        d = prep(A[keys[0]][0]).shape[1]
        one = np.ones(d)
        zr, zw, rho, hit = [], [], [], []
        for k in keys:
            t = truth(k, A, skip=k)
            pr = predict(A[k][0], one)
            zr.append(sep_z(pr, t))
            zw.append(sep_z(predict(B[k][0], one), t))
            rho.append(spearmanr(pr, t).statistic)
            hit.append(float(int(np.argmax(pr)) == int(np.argmax(t))))
        rows.append(pack(case, layer, "gxa", "nothing fitted", 0.0,
                         zr, zw, rho, hit, rng, n_boot))

    alpha = pick_alpha(A, keys, rng)
    GA = Gram(A, keys)
    GB = Gram(B, keys)

    # 1. leave 1 photograph out, inside the first set
    zr, zw, rho, hit = [], [], [], []
    for k in keys:
        w = GA.fit(alpha, drop=k)
        t = truth(k, A, skip=k)
        pr = predict(A[k][0], w)
        pw = predict(B[k][0], w)
        zr.append(sep_z(pr, t))
        zw.append(sep_z(pw, t))
        rho.append(spearmanr(pr, t).statistic)
        hit.append(float(int(np.argmax(pr)) == int(np.argmax(t))))
    rows.append(pack(case, layer, feat, "leave 1 out", alpha,
                     zr, zw, rho, hit, rng, n_boot))

    # 2. the 2 by 2: fit on the WRONG question, test on the own truth
    z2, rho2, hit2 = [], [], []
    for k in keys:
        w = GB.fit(alpha, drop=k)
        t = truth(k, A, skip=k)
        pr = predict(B[k][0], w)
        z2.append(sep_z(pr, t))
        rho2.append(spearmanr(pr, t).statistic)
        hit2.append(float(int(np.argmax(pr)) == int(np.argmax(t))))
    rows.append(pack(case, layer, feat, "fitted on wrong", alpha,
                     z2, [0.0] * len(z2), rho2, hit2, rng, n_boot))

    # 3. the null: break the pair of the features and the truth.
    #
    # Warning: `null_mean` is a null for `real`, NOT for the gain. A probe
    # fitted on a shuffled truth learns the AVERAGE map of the case. The
    # average map separates any truth map that resembles it, so `real`
    # keeps a positive null wherever the case has a strong mean layout.
    # Subway safety is the example: its `null_mean` reaches +0.8.
    #
    # The statistic under test is the GAIN. The own features and the
    # wrong-question features both carry that average map, so the gain
    # under the null is 0. `null_gain` is the number to compare against.
    if n_null:
        nreal, ngain = [], []
        sub = keys[:12]
        for _ in range(n_null):
            sh = list(keys)
            rng.shuffle(sh)
            swap = {k: A[v][1] for k, v in zip(keys, sh)}
            fake = {k: (A[k][0], swap[k]) for k in keys}
            G = Gram(fake, keys)
            gr, gg = [], []
            for k in sub:
                w = G.fit(alpha, drop=k)
                zr0 = sep_z(predict(fake[k][0], w), swap[k])
                zw0 = sep_z(predict(B[k][0], w), swap[k])
                gr.append(zr0)
                gg.append(zr0 - zw0)
            nreal.append(float(np.nanmean(gr)))
            ngain.append(float(np.nanmean(gg)))
        sd = float(np.std(ngain, ddof=1)) if len(ngain) > 1 else 0.0
        for r in rows:
            r["null_mean"] = float(np.mean(nreal))
            r["null_gain"] = float(np.mean(ngain))
            r["null_gain_sd"] = sd

    # 4. fit on the first set, test on 60 photographs it never saw
    C = load_acts(ho, case, layer, feat)
    D = load_acts(ho_xq, case, layer, feat)
    hk = sorted(set(C) & set(D))
    if hk:
        w = GA.fit(alpha)
        zr, zw, rho, hit = [], [], [], []
        for k in hk:
            # the prior comes from the TRAIN photographs, which share no
            # image with the held-out set
            t = strip(C[k][1], prior_of(A, prior_keys, None)) \
                if resid_prior else C[k][1]
            pr = predict(C[k][0], w)
            pw = predict(D[k][0], w)
            zr.append(sep_z(pr, t))
            zw.append(sep_z(pw, t))
            rho.append(spearmanr(pr, t).statistic)
            hit.append(float(int(np.argmax(pr)) == int(np.argmax(t))))
        rows.append(pack(case, layer, feat, "held out", alpha,
                         zr, zw, rho, hit, rng, n_boot))
    return rows


def pack(case, layer, feat, split, alpha, zr, zw, rho, hit,
         rng, n_boot) -> dict:
    zr, zw = np.array(zr, dtype=float), np.array(zw, dtype=float)
    d = zr - zw
    lo, hi = boot_ci(d, n_boot, rng, "mean")
    rlo, rhi = boot_ci(zr, n_boot, rng, "mean")
    try:
        pv = float(wilcoxon(d).pvalue)
    except ValueError:
        pv = 1.0
    return {"case": case, "layer": layer, "feature": feat, "split": split,
            "alpha": alpha, "n": int(d.size), "real": float(zr.mean()),
            "real_lo": rlo, "real_hi": rhi,
            "wrong": float(zw.mean()), "gain": float(d.mean()),
            "lo": lo, "hi": hi, "spearman": float(np.nanmean(rho)),
            "top_hit": float(np.nanmean(hit)),
            "p": float(pv)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--xq", default="outputs/_monocle/acts60_xq")
    ap.add_argument("--ho", default="outputs/_monocle/acts60_ho")
    ap.add_argument("--ho-xq", default="outputs/_monocle/acts60_ho_xq")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[18, 20, 42])
    ap.add_argument("--features", nargs="+", default=["h", "gh"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-null", type=int, default=5)
    ap.add_argument("--resid-prior", action="store_true",
                    help="take a leave-1-out position prior out of the truth")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/k1_fitted_probe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for case in args.cases:
        for feat in args.features:
            for layer in args.layers:
                got = run(case, layer, feat, Path(args.own), Path(args.xq),
                          Path(args.ho), Path(args.ho_xq), rng,
                          args.n_boot, args.n_null, args.resid_prior)
                rows.extend(got)
                for r in got:
                    print(f"  {r['case']:>12} {r['feature']:>3} "
                          f"L{r['layer']:<3} {r['split']:>16} "
                          f"own {r['real']:+6.3f} wrong {r['wrong']:+6.3f} "
                          f"rho {r['spearman']:+6.3f} "
                          f"top {r['top_hit']:.3f}")
    if not rows:
        print("no activations found")
        return
    live = [r for r in rows if r["split"] != "fitted on wrong"]
    q = bh(np.array([r["p"] for r in live]))
    for r, v in zip(live, q):
        r["p_bh"] = float(v)
    print(f"\n{'case':>12} {'ft':>3} {'L':>4} {'split':>16} {'own':>7} "
          f"{'own 95% CI':>18} {'wrong':>7} {'gain':>7} {'rho':>7} "
          f"{'top':>6} {'q':>8}")
    for r in rows:
        print(f"{r['case']:>12} {r['feature']:>3} {r['layer']:>4} "
              f"{r['split']:>16} {r['real']:+7.3f} "
              f"[{r['real_lo']:+6.3f},{r['real_hi']:+6.3f}] "
              f"{r['wrong']:+7.3f} {r['gain']:+7.3f} {r['spearman']:+7.3f} "
              f"{r['top_hit']:6.3f} "
              f"{r.get('p_bh', float('nan')):8.4f}")
    nulls = [r["null_mean"] for r in rows if "null_mean" in r]
    ng = [r["null_gain"] for r in rows if "null_gain" in r]
    if nulls:
        print(f"\nthe photograph-permutation null gives real z = "
              f"{np.mean(nulls):+.3f} and gain z = {np.mean(ng):+.3f}")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
