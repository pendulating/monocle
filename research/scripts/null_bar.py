"""Test each instrument against the STRONGEST model-free null, at each k.

Why the sign is free
--------------------
A model-free feature has no natural direction. Rank by edge density, or by
MINUS edge density, and the deletion test swaps `top_safe` with `top_danger`,
so the separation flips sign exactly. Thus a null whose median z is -0.84 is
really a null of strength +0.84. We read the SIZE of z, never its sign.

Why the strongest
-----------------
A null that is weak at one set size is not a control. Edge density separates
well at k=16 and hardly at all at k=8; luminance does the opposite. If we pick
one feature and keep it, we choose the bar that flatters the instrument.

So the bar is the best null AT EACH k, and the comparison is paired on the
image. An instrument passes only when it beats the null that beats it hardest.

Usage:
    python research/scripts/null_bar.py --arms <dir> ... --nulls <dir> ...
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
    df = df.copy()
    df["arm"] = run.name
    return df[["arm", "image_id", "k", STAT]]


def paired(a: pd.DataFrame, b: pd.DataFrame, k: int, n_boot: int, rng):
    x = a[a.k == k].set_index("image_id")[STAT]
    y = b[b.k == k].set_index("image_id")[STAT]
    j = pd.concat({"a": x, "b": y}, axis=1).dropna()
    d = (j["a"] - j["b"]).to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return None
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    m = np.median(d[idx], axis=1)
    return {"gain": float(np.median(d)),
            "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)),
            "wins": int((d > 0).sum()), "n": int(d.size)}


def paired_select(arm: pd.DataFrame, nulls: dict, k: int, n_boot: int, rng):
    """The same test, but the bar is CHOSEN inside each bootstrap sample.

    Why this matters
    ----------------
    The plain test picks the strongest null on the full data, then bootstraps
    the arm against that one fixed null. So the choice of bar sits outside the
    interval, and the interval is too narrow.

    Here each resample picks its own strongest null. The interval then covers
    both the noise of the images and the noise of the choice.
    """
    x = arm[arm.k == k].set_index("image_id")[STAT]
    cols = {}
    for name, d in nulls.items():
        cols[name] = d[d.k == k].set_index("image_id")[STAT]
    j = pd.concat({"__arm__": x, **cols}, axis=1).dropna()
    if len(j) < 3:
        return None
    a = j["__arm__"].to_numpy(dtype=float)
    names = [c for c in j.columns if c != "__arm__"]
    n_mat = j[names].to_numpy(dtype=float)

    def gain_of(rows):
        med = np.median(n_mat[rows], axis=0)
        b = int(np.argmax(np.abs(med)))
        flip = -1.0 if med[b] < 0 else 1.0
        return float(np.median(a[rows] - flip * n_mat[rows, b])), b

    full, b_full = gain_of(np.arange(a.size))
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.empty(n_boot)
    for t in range(n_boot):
        boot[t] = gain_of(idx[t])[0]
    return {"gain": full, "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "bar": names[b_full], "n": int(a.size)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--nulls", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--select-in-bootstrap", action="store_true",
                    help="choose the bar inside each bootstrap sample. It is "
                         "slower and the interval is honest about the choice.")
    ap.add_argument("--boot-select", type=int, default=3000,
                    help="bootstrap draws for --select-in-bootstrap")
    args = ap.parse_args(argv)

    arms = {Path(p).name: load(Path(p)) for p in args.arms
            if (Path(p) / "summary.parquet").exists()}
    nulls = {Path(p).name: load(Path(p)) for p in args.nulls
             if (Path(p) / "summary.parquet").exists()}
    if not arms or not nulls:
        print("need at least 1 arm and 1 null")
        return 2
    rng = np.random.default_rng(args.seed)

    for k in args.ks:
        med = {n: float(np.nanmedian(d[d.k == k][STAT])) for n, d in nulls.items()}
        # The sign of a model-free ranking is a free choice. Rank by MINUS a
        # feature and the test swaps its 2 directed arms, so the separation
        # flips sign exactly. Thus the strength of a null is |z|, and a null
        # with a negative median is used negated.
        best = max(med, key=lambda n: abs(med[n]))
        flip = -1.0 if med[best] < 0 else 1.0
        bar = nulls[best].copy()
        bar[STAT] = bar[STAT] * flip
        print(f"\n=== k={k}: the bar is {best} (|z|={abs(med[best]):.3f}"
              f"{', negated' if flip < 0 else ''}) ===")
        print("   all nulls by |z|: " + ", ".join(
            f"{n.split('saliency_')[-1].replace('_ablate','')}={abs(v):.2f}"
            for n, v in sorted(med.items(), key=lambda t: -abs(t[1]))))
        rows = []
        for name, d in arms.items():
            r = paired(d, bar, k, args.n_boot, rng)
            if r is None:
                continue
            r["arm"] = name
            rows.append(r)
        if not rows:
            continue
        t = pd.DataFrame(rows).sort_values("gain", ascending=False)
        t["passes"] = t["lo"] > 0
        short = t["arm"].str.replace("subway_safety_g2_", "", regex=False)
        t = t.assign(arm=short)
        print(t[["arm", "n", "gain", "lo", "hi", "wins", "passes"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))

        if args.select_in_bootstrap:
            srows = []
            for name, d in arms.items():
                r = paired_select(d, nulls, k, args.boot_select, rng)
                if r is None:
                    continue
                r["arm"] = name.replace("subway_safety_g2_", "")
                srows.append(r)
            if srows:
                st = pd.DataFrame(srows).sort_values("gain", ascending=False)
                st["passes"] = st["lo"] > 0
                print("   -- the bar chosen inside the bootstrap --")
                print(st[["arm", "n", "gain", "lo", "hi", "bar",
                          "passes"]].to_string(
                    index=False,
                    float_format=lambda v: f"{v:+.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
