"""Write the fitted reader's map in the shape `monocle.patch_caption` reads.

The content test asked what sits in the square that the ANALYTIC reader
picks. The fitted reader picks other squares, so the test must run again
on those. This script writes 1 map for each photograph, in the same
parquet shape as a truth map, so the caption tool needs no change.

| Mode | The fit | What it never saw |
|---|---|---|
| `self` | the case's own photographs, 1 left out | the target photograph |
| `shared` | the OTHER cases only | the target case and its photographs |

The `shared` mode is the strong one. Its weight never saw this judgment.

Fresh photographs
-----------------
`--pred-root` reads the target case's features from a SECOND activation
store. The fit stays in `--own`, so the weight never saw the photographs
that it now predicts. Then `self` needs no leave-1-out, because the 2
photograph sets share no image.

    python research/scripts/k1_export_reader_cells.py --cases restaurants \\
        libraries subway_safety schools street_photography --layer 20

    python research/scripts/k1_export_reader_cells.py --cases restaurants \\
        --pred-root outputs/_monocle/acts60_fresh --modes shared \\
        --out-dir outputs/_monocle/reader_cells_fresh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts import k1_fitted_probe as fp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--feature", default="gh")
    ap.add_argument("--alpha", type=float, default=1e5)
    ap.add_argument("--alpha-from", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=["self", "shared"])
    ap.add_argument("--out-dir",
                    default="outputs/_monocle/reader_cells")
    ap.add_argument("--pred-root", default=None,
                    help="a 2nd activation store. The target case's "
                         "features come from here, so the weight "
                         "predicts photographs it never saw.")
    ap.add_argument("--check-against", default=None,
                    help="an earlier reader_cells folder. The run "
                         "reports the rank agreement with it, which "
                         "proves that alpha and the plumbing match.")
    args = ap.parse_args()

    import json
    alpha = {c: args.alpha for c in args.cases}
    for f in args.alpha_from:
        for r in json.load(open(f)):
            if (r["feature"] == args.feature and r["layer"] == args.layer
                    and r["split"] == "leave 1 out"):
                alpha[r["case"]] = float(r["alpha"])

    def read(root: str) -> dict:
        """The features of every case under 1 activation root.

        Warning: `fp.load_acts` keeps the whole 1.8 GB dump in a cache.
        Seven of them fill the machine, so the cache goes after each case.
        """
        got = {}
        for c in args.cases:
            a = fp.load_acts(Path(root), c, args.layer, args.feature)
            fp._blob.cache_clear()
            if not a:
                print(f"skip {c}: no activations under {root}", flush=True)
                continue
            for k in sorted(a):
                got[f"{c}::{k}"] = a[k]
            print(f"loaded {c} from {root}: {len(a)} photographs",
                  flush=True)
        return got

    A = read(args.own)
    keys = sorted(A)
    # The prediction set. Without `--pred-root` it IS the fit set, and the
    # `self` mode then needs the leave-1-out that the fit set demands.
    P = read(args.pred_root) if args.pred_root else A
    fresh = args.pred_root is not None

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 1 pooled Gram serves both modes. `fit(drop=...)` takes a list, so a
    # case leaves the fit without a rebuild.
    GA = fp.Gram(A, keys)

    for mode in args.modes:
        for c in args.cases:
            ck = [k for k in keys if k.startswith(f"{c}::")]
            pk = [k for k in sorted(P) if k.startswith(f"{c}::")]
            if not ck or not pk:
                continue
            rows = []
            if mode == "self" and not fresh:
                G = fp.Gram(A, ck)
                for k in ck:
                    w = G.fit(alpha[c], drop=k)  # noqa: E501
                    rows.append((k.split("::", 1)[1],
                                 fp.predict(A[k][0], w)))
            elif mode == "self":
                # The 2 photograph sets share no image, so all 60 of the
                # case's own photographs stay in the fit.
                w = fp.Gram(A, ck).fit(alpha[c])
                for k in pk:
                    rows.append((k.split("::", 1)[1],
                                 fp.predict(P[k][0], w)))
            else:
                if len(ck) == len(keys):
                    continue
                w = GA.fit(alpha[c], drop=ck)
                for k in pk:
                    rows.append((k.split("::", 1)[1],
                                 fp.predict(P[k][0], w)))
            d = out / f"{c}_{mode}"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "image_id": [r[0] for r in rows],
                "causal": [np.asarray(r[1], dtype=float).tolist()
                           for r in rows],
            }).to_parquet(d / "maps.parquet")
            print(f"wrote {d}/maps.parquet ({len(rows)} photographs)",
                  flush=True)
            if args.check_against:
                check(Path(args.check_against) / f"{c}_{mode}", rows)


def check(old_dir: Path, rows: list) -> None:
    """Compare a new map against an earlier one, photograph by photograph.

    The alpha and the feature are not written in the output, so a later
    run cannot read them back. This check reads them instead: a median
    rank agreement of 1.00 says that the 2 runs used the same settings.
    """
    from scipy.stats import spearmanr

    path = old_dir / "maps.parquet"
    if not path.exists():
        print(f"  check: {path} does not exist", flush=True)
        return
    ref = pd.read_parquet(path)
    have = {str(i): np.asarray(v, dtype=float)
            for i, v in zip(ref.image_id, ref["causal"])}
    rho = [spearmanr(np.asarray(v, dtype=float), have[i]).statistic
           for i, v in rows if i in have]
    if not rho:
        print(f"  check: {path} shares no photograph", flush=True)
        return
    print(f"  check against {old_dir.name}: {len(rho)} photographs, "
          f"median rho = {float(np.median(rho)):+.4f}", flush=True)


if __name__ == "__main__":
    main()
