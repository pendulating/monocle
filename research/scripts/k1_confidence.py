"""Can the reader say when to trust it, with no measurement?

Why
---
The reader gets the sign right on 62 to 65 photographs of every 100. A user
with 1 photograph needs to know which kind they hold. A confidence score must
use only what the instrument itself gives, because the measured map is the
thing the user does not have.

Three candidate scores, all free at read time:

| Score | What it counts |
|---|---|
| `layer_agree` | how many of the 9 layers name the pooled top cell |
| `margin` | the gap in pooled rank between the top cell and the next |
| `concentration` | the share of the absolute instrument in its top cell |

Each is correlated with the per-photograph separation, inside a case and
after the case means come out. A score that works lets a user drop the
photographs the reader cannot read.

Usage:
    python research/scripts/k1_confidence.py --root outputs/_monocle/gateB60_g1L
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_separation import sep_z  # noqa: E402


def scores(path: Path, arm: str) -> pd.DataFrame:
    from scipy.stats import rankdata
    df = pd.read_parquet(path)
    ic = "target" if "target" in df.columns else "image_id"
    tc = "causal" if "causal" in df.columns else "values"
    rows = []
    for key, g in df.groupby(ic):
        truth = np.asarray(g.iloc[0][tc], dtype=float).ravel()
        raw, ranks = [], []
        for _, r in g.iterrows():
            v = r.get(arm)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size != truth.size or not np.isfinite(v).all():
                continue
            raw.append(v)
            ranks.append(rankdata(v))
        if len(ranks) < 2:
            continue
        pooled = np.mean(ranks, axis=0)
        order = np.sort(pooled)[::-1]
        top = int(np.argmax(pooled))
        bot = int(np.argmin(pooled))
        votes_hi = sum(int(np.argmax(v)) == top for v in raw)
        votes_lo = sum(int(np.argmin(v)) == bot for v in raw)
        conc = float(np.mean([np.max(np.abs(v)) / np.sum(np.abs(v))
                              for v in raw]))
        rows.append({
            "image_id": str(key),
            "layer_agree": (votes_hi + votes_lo) / (2 * len(raw)),
            "margin": float(order[0] - order[1]) / len(raw),
            "concentration": conc,
            "sep": sep_z(pooled, truth)})
    return pd.DataFrame(rows)


def report(root: Path, xq_root: Path, cases: list[str], arm: str) -> dict:
    from scipy.stats import spearmanr
    parts = []
    for case in cases:
        a = root / case / "maps.parquet"
        b = xq_root / case / "maps.parquet"
        if not (a.exists() and b.exists()):
            continue
        sa, sb = scores(a, arm), scores(b, arm)
        m = sa.merge(sb[["image_id", "sep"]], on="image_id",
                     suffixes=("", "_xq"))
        m["gain"] = m["sep"] - m["sep_xq"]
        m["case"] = case
        parts.append(m)
    if not parts:
        return {}
    d = pd.concat(parts, ignore_index=True)
    out: dict = {"n": int(len(d)), "cases": {}, "pooled": {}}
    for name in ("layer_agree", "margin", "concentration"):
        d[f"{name}_z"] = d.groupby("case")[name].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=1) or 1.0))
        r = spearmanr(d[f"{name}_z"], d["gain"])
        out["pooled"][name] = {"rho": float(r.statistic),
                               "p": float(r.pvalue)}
        for case, g in d.groupby("case"):
            rc = spearmanr(g[name], g["gain"])
            out["cases"].setdefault(case, {})[name] = {
                "rho": float(rc.statistic), "p": float(rc.pvalue)}
    # what a user gains by keeping only the confident half
    for name in ("layer_agree", "margin", "concentration"):
        keep = d[d[f"{name}_z"] > 0]
        out["pooled"][name]["gain_all"] = float(d.gain.mean())
        out["pooled"][name]["gain_top_half"] = float(keep.gain.mean())
        out["pooled"][name]["n_top_half"] = int(len(keep))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", required=True)
    ap.add_argument("--cases", nargs="+",
                    default=["restaurants", "libraries", "subway_safety"])
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    r = report(Path(args.root), Path(args.xq_root), args.cases, args.arm)
    if not r:
        print("no data")
        return 1
    print(f"{r['n']} photographs")
    print(f"{'score':>15} {'rho':>7} {'p':>8} {'gain all':>9} "
          f"{'gain top half':>14} {'n':>4}")
    print("-" * 62)
    for name, v in r["pooled"].items():
        print(f"{name:>15} {v['rho']:>+7.3f} {v['p']:>8.4f} "
              f"{v['gain_all']:>+9.3f} {v['gain_top_half']:>+14.3f} "
              f"{v['n_top_half']:>4}")
    print()
    for case, d in r["cases"].items():
        for name, v in d.items():
            print(f"  {case:>14} {name:>15} rho={v['rho']:>+6.3f} "
                  f"p={v['p']:.4f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
