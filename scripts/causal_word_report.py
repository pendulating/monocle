"""Which words sit at the causal patch and not at a random patch.

The counts come from `monocle.causal_words`. Each photograph gives words at
the 2 poles of its measured map, and words at random patches of the SAME
photograph. So the scene is held constant and only the causal role changes.

Two readings:

1. A log-odds ratio with an informative Dirichlet prior. It ranks the words
   that mark a role. The prior is the pooled count, so a rare word does not
   win by 1 event.
2. A permutation test. It shuffles the role labels INSIDE each photograph and
   rebuilds the divergence. It says whether the 2 word sets differ at all.

Usage:
    python scripts/causal_word_report.py \\
        --words outputs/_monocle/causal_words/words/restaurants/words.parquet \\
        --layer 18
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

JUNK = re.compile(r"^[\s\W_]*$")


def clean(w: str) -> str:
    return str(w).strip().lower()


def counts(df: pd.DataFrame, vocab: list[str]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    out = np.zeros(len(vocab), dtype=float)
    for w, n in df["word"].value_counts().items():
        j = idx.get(w)
        if j is not None:
            out[j] = n
    return out


def log_odds(y1: np.ndarray, y2: np.ndarray, a0: float = 100.0):
    """Monroe's log-odds ratio, informative Dirichlet prior, z scored."""
    back = y1 + y2
    alpha = a0 * back / max(back.sum(), 1.0)
    n1, n2 = y1.sum(), y2.sum()
    a_all = alpha.sum()
    p1 = (y1 + alpha) / (n1 + a_all - y1 - alpha)
    p2 = (y2 + alpha) / (n2 + a_all - y2 - alpha)
    d = np.log(p1) - np.log(p2)
    var = 1.0 / (y1 + alpha) + 1.0 / (y2 + alpha)
    return d / np.sqrt(var)


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log(a[ok] / b[ok])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--top-rank", type=int, default=5,
                    help="keep the first N words of each patch")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-show", type=int, default=18)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)

    df = pd.read_parquet(a.words)
    df = df[(df["layer"] == a.layer) & (df["rank"] < a.top_rank)].copy()
    df["word"] = df["word"].map(clean)
    df = df[~df["word"].map(lambda w: bool(JUNK.match(w)))]
    keep = df["word"].value_counts()
    df = df[df["word"].isin(keep[keep >= a.min_count].index)]
    vocab = sorted(df["word"].unique())
    n_img = df["image_id"].nunique()
    print(f"\n{a.label or a.words}")
    print(f"  layer {a.layer}, {n_img} photographs, {len(vocab)} words, "
          f"{len(df)} word events")

    rand = counts(df[df.role == "random"], vocab)
    for role in ("pos", "neg"):
        y = counts(df[df.role == role], vocab)
        z = log_odds(y, rand)
        order = np.argsort(-z)
        side = ("holds the judgment up" if role == "pos"
                else "pushes the judgment down")
        print(f"\n  words that mark the patch that {side}")
        for j in order[:a.n_show]:
            print(f"    {vocab[j]:<22} z={z[j]:>+6.2f} "
                  f"pole={int(y[j]):>4} random={int(rand[j]):>4}")

    rng = np.random.default_rng(a.seed)
    obs = {r: js(counts(df[df.role == r], vocab), rand) for r in ("pos", "neg")}
    null = {r: [] for r in obs}
    by_img = {k: g for k, g in df.groupby("image_id")}
    for _ in range(a.n_perm):
        parts = []
        for g in by_img.values():
            h = g.copy()
            h["role"] = rng.permutation(h["role"].to_numpy())
            parts.append(h)
        sh = pd.concat(parts)
        r0 = counts(sh[sh.role == "random"], vocab)
        for r in obs:
            null[r].append(js(counts(sh[sh.role == r], vocab), r0))
    print("\n  the poles against the random patches of the same photograph")
    print(f"  {'role':<6} {'divergence':>11} {'null median':>12} "
          f"{'p':>8}")
    for r in obs:
        n = np.asarray(null[r])
        p = float((np.sum(n >= obs[r]) + 1) / (n.size + 1))
        print(f"  {r:<6} {obs[r]:>11.4f} {np.median(n):>12.4f} {p:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
