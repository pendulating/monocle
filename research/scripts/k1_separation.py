"""The deletion test at 1 cell, computed from the measured map alone.

Why 1 cell
----------
The saturation curve says the system is not additive. One cell of 64 already
moves the judgment 25 to 46 percent of the way that half the image moves it.
So a set of k cells does not act like the sum of k single cells, and the
measured ceiling itself fails the null bar at k=16 and k=32. Where the
ceiling has no signal, no arm can show one.

At k=1 the target is real, and the statistic needs no additivity. At
group=1 the map holds the measured single-patch ablation of every patch, so
this test needs no new GPU work and no pooling.

The statistic
-------------
Both the truth and the instruments are stored as CONTRIBUTIONS. A high value
means the cell pushes the judgment toward the positive label, so a deletion
of that cell makes the contrast fall.

    safe   = argmax(instrument)
    danger = argmin(instrument)
    sep    = truth[safe] - truth[danger]
    z      = sep / sd(truth over all cells)

`z` is positive when the instrument names the 2 poles correctly. The ceiling
puts the truth in place of the instrument, which gives (max - min) / sd.

The test
--------
The cross-question map is the control. It holds the same images, the same
truth, and the same instrument, and only the question changes. A reader of
the judgment must lose its separation there. A reader of the picture keeps
it. So the headline test is the PAIRED difference, one image at a time:

    d = z(real question) - z(wrong question)

with a Wilcoxon signed-rank test, a bootstrap interval, and a
Benjamini-Hochberg step over the instruments only.

Usage:
    python research/scripts/k1_separation.py --maps <dir>/maps.parquet \
        --xq <dir>/maps.parquet --layer 18 --label "restaurants 256"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SKIP = {"layer", "target", "image_id", "causal", "values"}


def id_col(df: pd.DataFrame) -> str:
    return "target" if "target" in df.columns else "image_id"


def truth_col(df: pd.DataFrame) -> str:
    return "causal" if "causal" in df.columns else "values"


def instruments(df: pd.DataFrame) -> list[str]:
    """The map columns that are not the truth and not a key."""
    out = []
    r = df.iloc[0]
    for c in df.columns:
        if c in SKIP:
            continue
        v = r[c]
        if isinstance(v, (list, np.ndarray)) and np.asarray(v).ravel().size > 1:
            out.append(c)
    return out


def sep_z(values: np.ndarray, truth: np.ndarray) -> float:
    """The 1-cell separation of a ranking, in units of the cell spread."""
    sd = float(np.std(truth, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    safe = int(np.argmax(values))
    danger = int(np.argmin(values))
    return float(truth[safe] - truth[danger]) / sd


def table(df: pd.DataFrame, layer: int) -> tuple[pd.DataFrame, list[str]]:
    """1 row for each image, 1 column for each arm."""
    d = df[df["layer"] == layer] if "layer" in df.columns else df
    ic, tc = id_col(d), truth_col(d)
    cols = instruments(d)
    rows = []
    seen: set[str] = set()
    for _, r in d.iterrows():
        key = str(r[ic])
        if key in seen:
            continue
        seen.add(key)
        t = np.asarray(r[tc], dtype=float).ravel()
        row = {"image_id": key, "ceiling": sep_z(t, t)}
        for c in cols:
            v = np.asarray(r[c], dtype=float).ravel()
            row[c] = sep_z(v, t) if v.size == t.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("image_id"), cols


def position_prior(df: pd.DataFrame, layer: int) -> pd.Series:
    """A model-free null: the mean truth of the OTHER images.

    The prior knows the frame, not the picture. It is the null that a
    position bias would produce, and it leaves out the image it scores.
    """
    d = df[df["layer"] == layer] if "layer" in df.columns else df
    ic, tc = id_col(d), truth_col(d)
    keep = d.drop_duplicates(subset=[ic])
    ids = [str(x) for x in keep[ic]]
    mats = [np.asarray(v, dtype=float).ravel() for v in keep[tc]]
    tot = np.sum(mats, axis=0)
    n = len(ids)
    out = {}
    for i, key in enumerate(ids):
        other = (tot - mats[i]) / (n - 1)
        out[key] = sep_z(other, mats[i])
    return pd.Series(out)


def shuffle_null(df: pd.DataFrame, layer: int, seed: int) -> pd.Series:
    """A within-image permutation of the truth. Its mean must sit at 0."""
    rng = np.random.default_rng(seed)
    d = df[df["layer"] == layer] if "layer" in df.columns else df
    ic, tc = id_col(d), truth_col(d)
    keep = d.drop_duplicates(subset=[ic])
    out = {}
    for _, r in keep.iterrows():
        t = np.asarray(r[tc], dtype=float).ravel()
        out[str(r[ic])] = sep_z(rng.permutation(t), t)
    return pd.Series(out)


def boot_ci(x: np.ndarray, n_boot: int, rng, reduce: str) -> tuple:
    f = np.median if reduce == "median" else np.mean
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    b = f(x[idx], axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def bh(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        val = min(prev, p[i] * n / (n - rank))
        adj[i] = prev = val
    return adj


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import wilcoxon

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--xq", default=None,
                    help="the cross-question maps, the paired control")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--reduce", default="mean", choices=["mean", "median"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    red = np.median if a.reduce == "median" else np.mean
    df = pd.read_parquet(a.maps)
    tab, cols = table(df, a.layer)
    tab["position_prior"] = position_prior(df, a.layer)
    tab["shuffled"] = shuffle_null(df, a.layer, a.seed)
    n_cells = np.asarray(df.iloc[0][truth_col(df)]).ravel().size

    print(f"\n{a.label or a.maps}")
    print(f"  {len(tab)} images, {n_cells} cells, layer {a.layer}, "
          f"reduce={a.reduce}")
    print(f"  {'arm':<16} {'z':>8} {'lo':>8} {'hi':>8} {'p':>10} {'wins':>7}")
    arms = ["ceiling"] + cols + ["position_prior", "shuffled"]
    stat = {}
    for c in arms:
        x = tab[c].dropna().to_numpy(float)
        lo, hi = boot_ci(x, a.n_boot, rng, a.reduce)
        try:
            p = float(wilcoxon(x, alternative="greater").pvalue)
        except ValueError:
            p = float("nan")
        wins = float(np.mean(x > 0))
        stat[c] = {"z": float(red(x)), "lo": lo, "hi": hi, "p": p,
                   "wins": wins, "n": int(x.size)}
        print(f"  {c:<16} {red(x):>+8.3f} {lo:>+8.3f} {hi:>+8.3f} "
              f"{p:>10.4f} {wins:>7.2f}")

    out = {"label": a.label, "maps": a.maps, "layer": a.layer,
           "n_images": int(len(tab)), "n_cells": int(n_cells),
           "reduce": a.reduce, "arms": stat}

    if a.xq:
        xq = pd.read_parquet(a.xq)
        xtab, xcols = table(xq, a.layer)
        both = [c for c in cols if c in xcols]
        j = tab[both].join(xtab[both], how="inner", lsuffix="_r", rsuffix="_x")
        print(f"\n  PAIRED against the wrong question ({len(j)} images)")
        print(f"  {'arm':<16} {'real':>8} {'wrong':>8} {'gain':>8} "
              f"{'lo':>8} {'hi':>8} {'p_bh':>8}")
        raw, rows = [], []
        for c in both:
            d = (j[f"{c}_r"] - j[f"{c}_x"]).dropna().to_numpy(float)
            lo, hi = boot_ci(d, a.n_boot, rng, a.reduce)
            try:
                p = float(wilcoxon(d, alternative="greater").pvalue)
            except ValueError:
                p = float("nan")
            raw.append(p)
            rows.append({"arm": c, "real": float(red(j[f"{c}_r"])),
                         "wrong": float(red(j[f"{c}_x"])),
                         "gain": float(red(d)), "lo": lo, "hi": hi,
                         "p": p, "n": int(d.size)})
        adj = bh(np.asarray(raw, dtype=float))
        for r, q in zip(rows, adj):
            r["p_bh"] = float(q)
            mark = "  PASS" if (q < 0.05 and r["lo"] > 0) else ""
            print(f"  {r['arm']:<16} {r['real']:>+8.3f} {r['wrong']:>+8.3f} "
                  f"{r['gain']:>+8.3f} {r['lo']:>+8.3f} {r['hi']:>+8.3f} "
                  f"{q:>8.4f}{mark}")
        out["paired"] = rows

    if a.json_out:
        p = Path(a.json_out)
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur[a.label or a.maps] = out
        p.write_text(json.dumps(cur, indent=2))
        print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
