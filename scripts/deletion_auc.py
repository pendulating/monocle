"""One number for each image, so the k grid stops multiplying the tests.

The problem with the grid
-------------------------
The deletion test gives a separation at 6 set sizes. With 6 arms and 3 cases
that is about 108 comparisons at a nominal 95%, and about 5 false passes are
expected under a global null. Scattered passes across k and case are what
noise looks like.

The 2 corrections
-----------------
1. THE POWER GATE. The measured map is the ceiling. Where the ceiling itself
   does not beat the null, the test has no power, and no score there can be
   read. The gate keeps only the k where the ceiling passes.

   Warning: the gate reads the data, so report it, and report the curve that
   built it.

2. ONE STATISTIC. For each image, take the mean separation over the k inside
   the gate. The mean over a geometric grid gives each doubling of the set the
   same weight, so the large sets, which carry the least power, do not take
   over the sum.

Then 1 signed-rank test for each arm, and a Benjamini-Hochberg step over the
arms only.

Usage:
    python scripts/deletion_auc.py --ceiling <dir> --arms <dir> ... \
        --nulls <dir> ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STAT = "separation_z"


def load(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "summary.parquet")
    if "keep_only" in df.columns:
        df = df[~df["keep_only"]]
    return df[["image_id", "k", STAT]].copy()


def per_k_gain(arm: pd.DataFrame, nulls: dict, k: int, n_boot: int, rng):
    """Paired gain over the strongest null, the bar chosen in the bootstrap."""
    x = arm[arm.k == k].set_index("image_id")[STAT]
    cols = {n: d[d.k == k].set_index("image_id")[STAT] for n, d in nulls.items()}
    j = pd.concat({"__arm__": x, **cols}, axis=1).dropna()
    if len(j) < 3:
        return None
    a = j["__arm__"].to_numpy(float)
    names = [c for c in j.columns if c != "__arm__"]
    nm = j[names].to_numpy(float)

    def g(rows):
        med = np.median(nm[rows], axis=0)
        b = int(np.argmax(np.abs(med)))
        flip = -1.0 if med[b] < 0 else 1.0
        return float(np.median(a[rows] - flip * nm[rows, b]))

    full = g(np.arange(a.size))
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([g(idx[t]) for t in range(n_boot)])
    return {"gain": full, "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "n": int(a.size)}


def gated_mean(df: pd.DataFrame, ks: list[int],
               reduce: str = "mean") -> pd.Series:
    """1 value for each image, over the gated k.

    Warning: `separation_z` divides by the spread of the random draws, and
    that spread runs 4x below its median on some images. So a single k can
    reach |z| = 23. The `median` reduction resists that; the `mean` does not.
    Read both before you believe a result.
    """
    d = df[df.k.isin(ks)]
    g = d.groupby("image_id")[STAT]
    return g.median() if reduce == "median" else g.mean()


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
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
    ap.add_argument("--ceiling", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--nulls", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--reduce", default="mean", choices=["mean", "median"],
                    help="how to join the gated k into 1 value per image")
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default=None,
                    help="append the table to this file, keyed by --label")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    ceil = load(Path(args.ceiling))
    nulls = {Path(p).name: load(Path(p)) for p in args.nulls
             if (Path(p) / "summary.parquet").exists()}
    arms = {Path(p).name: load(Path(p)) for p in args.arms
            if (Path(p) / "summary.parquet").exists()}
    if not nulls or not arms:
        print("need at least 1 arm and 1 null")
        return 2

    print(f"=== {args.label or 'deletion'} ===")
    print("the power gate: the ceiling against the strongest null at each k")
    gate = []
    for k in args.ks:
        r = per_k_gain(ceil, nulls, k, args.n_boot, rng)
        if r is None:
            continue
        ok = r["lo"] > 0
        gate.append(k) if ok else None
        print(f"  k={k:>2}  ceiling gain {r['gain']:+.3f} "
              f"[{r['lo']:+.3f}, {r['hi']:+.3f}]  {'IN' if ok else 'out'}")
    if not gate:
        print("  the ceiling passes at no k: this case carries no power")
        return 0
    print(f"  gate = {gate}")

    rows = []
    for name, d in arms.items():
        a = gated_mean(d, gate, args.reduce)
        cols = {n: gated_mean(x, gate, args.reduce) for n, x in nulls.items()}
        j = pd.concat({"__arm__": a, **cols}, axis=1).dropna()
        av = j["__arm__"].to_numpy(float)
        names = [c for c in j.columns if c != "__arm__"]
        nm = j[names].to_numpy(float)
        med = np.median(nm, axis=0)
        b = int(np.argmax(np.abs(med)))
        flip = -1.0 if med[b] < 0 else 1.0
        d_i = av - flip * nm[:, b]
        w = wilcoxon(d_i) if np.any(d_i != 0) else None
        idx = rng.integers(0, d_i.size, size=(args.n_boot, d_i.size))

        def g(rows_):
            m = np.median(nm[rows_], axis=0)
            bb = int(np.argmax(np.abs(m)))
            fl = -1.0 if m[bb] < 0 else 1.0
            return float(np.median(av[rows_] - fl * nm[rows_, bb]))

        boot = np.array([g(idx[t]) for t in range(args.n_boot)])
        rows.append({"arm": name, "n": int(d_i.size),
                     "gain": float(np.median(d_i)),
                     "lo": float(np.percentile(boot, 2.5)),
                     "hi": float(np.percentile(boot, 97.5)),
                     "wins": int((d_i > 0).sum()),
                     "bar": names[b].split("saliency_")[-1],
                     "p": float(w.pvalue) if w else 1.0})
    t = pd.DataFrame(rows).sort_values("gain", ascending=False)
    t["p_bh"] = bh(t["p"].to_numpy())
    t["passes"] = (t["p_bh"] < 0.05) & (t["lo"] > 0)
    t["arm"] = t["arm"].str.replace(r"^[a-z_]*_g2_", "", regex=True)
    print("\nthe gated mean, 1 test for each arm, BH over the arms")
    print(t[["arm", "n", "gain", "lo", "hi", "wins", "p", "p_bh",
             "bar", "passes"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    if args.json_out:
        import json
        f = Path(args.json_out)
        blob = json.loads(f.read_text()) if f.exists() else {}
        key = args.label or "case"
        blob[key] = t.assign(passes=t["passes"].astype(bool)).to_dict(
            orient="records")
        blob[key + "__gate"] = gate
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(blob, indent=1))
        print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
