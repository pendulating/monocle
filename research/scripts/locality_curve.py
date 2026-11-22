"""The locality curve: where does a slot still mean its patch?

The question
------------
A residual swap at layer L asks "what if this SLOT held another photo's
vector?". An input ablation asks "what does this PATCH contribute?". The two
agree only while the slot still carries its own patch. Attention mixes the
slots, so the agreement must fall with depth.

This script measures that fall. It joins the donor-slot maps against the input
ablation maps on the same images, and it reports the agreement for each layer.

Why it matters
--------------
Gate A (the donor swap) was retired because the layer-20 map agreed with the
input map at only r=+0.09. That result names ONE layer. It does not say that
the donor swap fails at every layer. The curve answers the general question,
and it names the deepest layer that a causal claim can use.

The null
--------
A photo has shared structure. The sky is at the top and the road is at the
bottom in every image. Thus two unrelated maps correlate a little. The
`shuffled` column pairs each donor map with the input map of a DIFFERENT
image. A real locality signal must beat it.

Usage:
    python research/scripts/locality_curve.py \
        --donor outputs/_monocle/locality_donor/subway_safety/maps.parquet \
        --input outputs/_monocle/locality_input/subway_safety/maps.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle.lens_bias import DEFAULT_K, as_map, nanmean, top_k_score  # noqa: E402


def full_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over every patch, not only the top k."""
    from scipy.stats import pearsonr

    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    x, y = a[ok], b[ok]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def input_truth(df_in: pd.DataFrame) -> dict[str, np.ndarray]:
    """One input ablation map for each image.

    The input map does not depend on a layer, so the layer rows repeat it. We
    keep the first row that carries a real array.
    """
    truth: dict[str, np.ndarray] = {}
    for _, r in df_in.iterrows():
        tgt = str(r["target"])
        if tgt in truth:
            continue
        m = as_map(r.get("causal"))
        if m is not None:
            truth[tgt] = m
    return truth


def curve(df_don: pd.DataFrame, truth: dict[str, np.ndarray], k: int,
          seed: int = 777) -> pd.DataFrame:
    """The agreement for each layer, with the shuffled-image null."""
    rng = np.random.default_rng(seed)
    keys = sorted(truth)
    rows = []
    for layer, grp in df_don.groupby("layer"):
        r_all, r_top, s_top, r_null = [], [], [], []
        for _, r in grp.iterrows():
            tgt = str(r["target"])
            donor_map = as_map(r.get("causal"))
            if donor_map is None or tgt not in truth:
                continue
            gold = truth[tgt]
            if gold.size != donor_map.size:
                continue
            r_all.append(full_r(donor_map, gold))
            sign, rr = top_k_score(gold, donor_map, k)
            r_top.append(rr)
            s_top.append(sign)
            # The null: the same donor map against ANOTHER image's truth.
            others = [key for key in keys if key != tgt
                      and truth[key].size == donor_map.size]
            if others:
                pick = truth[others[int(rng.integers(len(others)))]]
                r_null.append(full_r(donor_map, pick))
        rows.append({
            "layer": int(layer),
            "n_pairs": len(r_all),
            "r_full": nanmean(r_all),
            "r_top": nanmean(r_top),
            "sign_top": nanmean(s_top),
            "r_null": nanmean(r_null),
        })
    out = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    out["excess"] = out["r_full"] - out["r_null"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--donor", required=True, help="the donor-slot maps parquet")
    ap.add_argument("--input", required=True, help="the input ablation parquet")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--out", default=None, help="write the table to this path")
    args = ap.parse_args(argv)

    df_don = pd.read_parquet(args.donor)
    df_in = pd.read_parquet(args.input)
    truth = input_truth(df_in)
    if not truth:
        print("no input ablation map is available", file=sys.stderr)
        return 2
    tab = curve(df_don, truth, args.k)

    print(f"input maps: {len(truth)} images")
    print(f"donor rows: {len(df_don)}")
    print()
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print()
    good = tab[tab["excess"] > 0.10]
    if len(good):
        print(f"the slot still means the patch down to layer "
              f"{int(good['layer'].max())} (excess > 0.10)")
    else:
        print("no layer clears excess > 0.10; the donor swap is not local")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
