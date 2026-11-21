"""Is the map of holes the same map as the map of content?

The absent map removes a cell. The substitute map gives the cell another
photograph's pixels and removes nothing. If the 2 agree, the map ranks
content. If they do not, it ranks holes.

The comparison is inside a photograph, over the 256 cells.

Usage:
    python scripts/hole_vs_content_map.py \
        --absent outputs/_monocle/causal_maps60 \
        --substitute outputs/_monocle/causal_maps60_sub
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path: Path) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    col = "values" if "values" in df.columns else "causal"
    idc = "image_id" if "image_id" in df.columns else "target"
    return {str(k): np.asarray(g.iloc[0][col], dtype=float).ravel()
            for k, g in df.groupby(idc)}


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import pearsonr, spearmanr
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--absent", required=True)
    ap.add_argument("--substitute", required=True)
    ap.add_argument("--cases", nargs="+",
                    default=["restaurants", "libraries", "subway_safety"])
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = []
    for case in args.cases:
        a = Path(args.absent) / f"{case}_absent" / "maps.parquet"
        b = Path(args.substitute) / f"{case}_substitute" / "maps.parquet"
        if not (a.exists() and b.exists()):
            continue
        A, B = load(a), load(b)
        keys = sorted(set(A) & set(B))
        if not keys:
            continue
        r = np.array([pearsonr(A[k], B[k]).statistic for k in keys])
        rho = np.array([spearmanr(A[k], B[k]).statistic for k in keys])
        top = np.array([int(np.argmax(A[k])) == int(np.argmax(B[k]))
                        for k in keys], dtype=float)
        bot = np.array([int(np.argmin(A[k])) == int(np.argmin(B[k]))
                        for k in keys], dtype=float)
        sd_a = np.array([A[k].std(ddof=1) for k in keys])
        sd_b = np.array([B[k].std(ddof=1) for k in keys])
        rows.append({"case": case, "n": len(keys),
                     "pearson": float(r.mean()), "spearman": float(rho.mean()),
                     "top1_same": float(top.mean()),
                     "bottom1_same": float(bot.mean()),
                     "sd_absent": float(sd_a.mean()),
                     "sd_substitute": float(sd_b.mean())})
    hdr = (f"{'case':>14} {'n':>4} {'r':>7} {'rho':>7} {'top1':>6} "
           f"{'bot1':>6} {'sd hole':>8} {'sd content':>11}")
    print(hdr)
    print("-" * len(hdr))
    for x in rows:
        print(f"{x['case']:>14} {x['n']:>4} {x['pearson']:>+7.3f} "
              f"{x['spearman']:>+7.3f} {x['top1_same']:>6.2f} "
              f"{x['bottom1_same']:>6.2f} {x['sd_absent']:>8.3f} "
              f"{x['sd_substitute']:>11.3f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
