"""Where in depth a patch instrument reads the judgment.

The 1-cell separation, layer by layer, with the wrong-question control at
every layer. The truth does not change with the layer, so the profile shows
only what the instrument gains or loses with depth.

Usage:
    python research/scripts/k1_layer_profile.py --root outputs/_monocle/gateB60_g1L \
        --xq-root outputs/_monocle/gateB60_g1L_xq --cases restaurants
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.k1_separation import bh, boot_ci, sep_z, table  # noqa: E402

ARMS = ["gxa", "cheap", "meang_own", "meangp_own", "lens_own"]


def pooled_ranks(path: Path, arm: str) -> dict[str, tuple]:
    """The mean rank of each cell over EVERY layer the file holds.

    A single layer that passes could be the layer we happened to pick. The
    pool uses them all, so no layer is chosen, and 1 test is left for each
    case and arm. Ranks, not values, because the scale of an instrument
    changes with depth.
    """
    from scipy.stats import rankdata

    df = pd.read_parquet(path)
    tc = "causal" if "causal" in df.columns else "values"
    ic = "target" if "target" in df.columns else "image_id"
    out: dict[str, tuple] = {}
    for key, g in df.groupby(ic):
        t = np.asarray(g.iloc[0][tc], dtype=float).ravel()
        rk = []
        for _, r in g.iterrows():
            v = r.get(arm)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size == t.size and np.isfinite(v).all():
                rk.append(rankdata(v))
        if rk:
            out[str(key)] = (np.mean(rk, axis=0), t)
    return out


def pool_report(root: Path, xq_root: Path, cases: list[str], n_boot: int,
                seed: int) -> list[dict]:
    """1 test for each case and arm, over the layer-pooled ranking."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows = []
    for case in cases:
        a = root / case / "maps.parquet"
        b = xq_root / case / "maps.parquet"
        if not (a.exists() and b.exists()):
            continue
        for arm in ARMS:
            A, B = pooled_ranks(a, arm), pooled_ranks(b, arm)
            keys = sorted(set(A) & set(B))
            if not keys:
                continue
            zr = np.array([sep_z(*A[k]) for k in keys])
            zw = np.array([sep_z(*B[k]) for k in keys])
            d = zr - zw
            if not np.isfinite(d).all() or np.allclose(d, 0):
                continue
            lo, hi = boot_ci(d, n_boot, rng, "mean")
            rows.append({"case": case, "arm": arm, "n": int(d.size),
                         "real": float(zr.mean()), "wrong": float(zw.mean()),
                         "gain": float(d.mean()), "lo": lo, "hi": hi,
                         "p": float(wilcoxon(d, alternative="greater").pvalue)})
    if rows:
        q = bh(np.asarray([r["p"] for r in rows], dtype=float))
        for r, qq in zip(rows, q):
            r["p_bh"] = float(qq)
    return rows


def profile(maps: Path, xq: Path | None, n_boot: int, seed: int) -> dict:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    df = pd.read_parquet(maps)
    xdf = pd.read_parquet(xq) if xq and xq.exists() else None
    layers = sorted(int(x) for x in df["layer"].unique())
    out = {"layers": layers, "n_cells": int(np.asarray(
        df.iloc[0]["causal" if "causal" in df.columns else "values"]
    ).ravel().size), "arms": {}, "ceiling": {}}
    for L in layers:
        tab, cols = table(df, L)
        out["ceiling"][L] = float(tab["ceiling"].mean())
        xtab = None
        if xdf is not None:
            xtab, _ = table(xdf, L)
        for c in ARMS:
            rec = out["arms"].setdefault(c, {})
            if c not in tab.columns:
                continue
            x = tab[c].dropna()
            if x.size < 3:
                continue
            r = {"real": float(x.mean()), "n": int(x.size)}
            if xtab is not None and c in xtab.columns:
                j = pd.concat({"r": tab[c], "x": xtab[c]}, axis=1).dropna()
                d = (j["r"] - j["x"]).to_numpy(float)
                lo, hi = boot_ci(d, n_boot, rng, "mean")
                try:
                    p = float(wilcoxon(d, alternative="greater").pvalue)
                except ValueError:
                    p = float("nan")
                r.update({"wrong": float(j["x"].mean()), "gain": float(d.mean()),
                          "lo": lo, "hi": hi, "p": p})
            rec[L] = r
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", default=None)
    ap.add_argument("--cases", nargs="+",
                    default=["subway_safety", "libraries", "restaurants"])
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--pool", action="store_true",
                    help="pool every layer by rank, 1 test for each arm")
    a = ap.parse_args(argv)

    if a.pool:
        rows = pool_report(Path(a.root), Path(a.xq_root), a.cases,
                           a.n_boot, a.seed)
        print("every layer pooled by rank, 1 test for each case and arm")
        print(f"{'case':<15} {'arm':<12} {'real':>7} {'wrong':>7} "
              f"{'gain':>7} {'lo':>7} {'hi':>7} {'q':>8}")
        for r in rows:
            mark = "  PASS" if r["p_bh"] < 0.05 and r["lo"] > 0 else ""
            print(f"{r['case']:<15} {r['arm']:<12} {r['real']:>+7.3f} "
                  f"{r['wrong']:>+7.3f} {r['gain']:>+7.3f} {r['lo']:>+7.3f} "
                  f"{r['hi']:>+7.3f} {r['p_bh']:>8.4f}{mark}")
        if a.json_out:
            Path(a.json_out).write_text(json.dumps(rows, indent=2))
            print(f"\nwrote {a.json_out}")
        return 0

    all_out = {}
    for case in a.cases:
        m = Path(a.root) / case / "maps.parquet"
        if not m.exists():
            print(f"skip {case}: no maps")
            continue
        x = Path(a.xq_root) / case / "maps.parquet" if a.xq_root else None
        d = profile(m, x, a.n_boot, a.seed)
        all_out[case] = d
        print(f"\n{case}: {d['n_cells']} cells, "
              f"{max(v.get('n', 0) for r in d['arms'].values() for v in r.values())}"
              f" images")
        head = f"{'L':>3} {'ceil':>7} " + " ".join(f"{c:>19}" for c in ARMS)
        print(head)
        print(f"{'':>3} {'':>7} " + " ".join(f"{'real / gain':>19}"
                                             for _ in ARMS))
        raw, where = [], []
        for L in d["layers"]:
            cells = []
            for c in ARMS:
                r = d["arms"].get(c, {}).get(L)
                if not r:
                    cells.append(f"{'-':>19}")
                    continue
                if "gain" in r:
                    raw.append(r["p"])
                    where.append((c, L))
                    cells.append(f"{r['real']:>+8.2f} /{r['gain']:>+8.2f} ")
                else:
                    cells.append(f"{r['real']:>+8.2f} /{'-':>8} ")
            print(f"{L:>3} {d['ceiling'][L]:>+7.2f} " + " ".join(cells))
        if raw:
            q = bh(np.asarray(raw, dtype=float))
            hits = [(c, L, p, qq) for (c, L), p, qq in zip(where, raw, q)
                    if qq < 0.05 and d["arms"][c][L]["lo"] > 0]
            print(f"  Benjamini-Hochberg over {len(raw)} layer-by-arm tests:")
            if hits:
                for c, L, p, qq in sorted(hits, key=lambda t: t[3]):
                    r = d["arms"][c][L]
                    print(f"    L{L:<3} {c:<12} gain {r['gain']:+.3f} "
                          f"[{r['lo']:+.3f}, {r['hi']:+.3f}] q={qq:.4f}")
            else:
                print("    none")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(all_out, indent=2))
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
