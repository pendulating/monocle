"""Pool the modality-order arms over the cases, and score WHERE the question lands.

`research/monocle/modality_order.py` writes one folder for each case. This script reads
them together and answers 3 questions:

1. **Does the question reach the image encoding?** The gap in `h` under each
   arm, at every block. Only a case pair of EQUAL question length may carry
   this number, so the script checks `patch_shift` and drops a case that fails.
2. **Does the order move the verdict, and which part does it?** The 3 paired
   contrasts, pooled over the cases, with a BH step over the family.
3. **Where does the question land on the image?** The per-patch gap under
   `question_first`, scored against the measured causal map of the same
   photographs.

Question 3 is the one that decides what the ordering rule is worth. If the
question lands on the cells the judgment already uses, then the order changes
which evidence the encoding carries. If it lands anywhere else, the order
changes the encoding without changing what the encoding is FOR.

Usage:
    python research/scripts/modality_order_report.py
    python research/scripts/modality_order_report.py --patch-truth \\
        outputs/_monocle/causal_maps60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "outputs/_monocle/modality_order4"
ARMS = ("image_first", "neutral_first", "question_first", "text_first")
#: Each contrast, and the one thing it isolates.
CONTRASTS = (
    ("image_first", "neutral_first", "the patches move later, no question"),
    ("neutral_first", "question_first", "the question, at fixed positions"),
    ("question_first", "text_first", "the answer moves nearer the patches"),
    ("image_first", "text_first", "the whole reversal"),
)


def load(root: Path, name: str = "gaps.parquet") -> pd.DataFrame:
    """Every case folder under `root`, concatenated."""
    frames = []
    for path in sorted(root.glob(f"*/{name}")):
        df = pd.read_parquet(path)
        df["case"] = path.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit(f"no {name} under {root}")
    return pd.concat(frames, ignore_index=True)


def matched_cases(df: pd.DataFrame) -> list[str]:
    """Cases whose cross-question arm holds the patch positions fixed.

    A non-zero `patch_shift` means the 2 questions differ in length, so the
    patches move and a rotary phase shift enters the gap. Such a case may
    carry the verdict result and must not carry the encoding result.
    """
    ok = []
    for case, sub in df.groupby("case"):
        if int(sub.patch_shift.abs().max()) == 0:
            ok.append(str(case))
    return sorted(ok)


def encoding_table(df: pd.DataFrame, cases: list[str]) -> pd.DataFrame:
    """The gap in `h` by block and arm, over the length-matched cases."""
    sub = df[df.case.isin(cases)]
    return (sub.pivot_table(index="layer", columns="layout", values="h_gap",
                            aggfunc="mean")
            .reindex(columns=[a for a in ARMS if a in set(sub.layout)]))


def verdict_table(df: pd.DataFrame) -> pd.DataFrame:
    """One paired contrast for each (case, contrast), plus the pooled row."""
    from scipy import stats

    one = (df[df.layer == df.layer.min()]
           [["case", "layout", "target", "c_own"]].drop_duplicates())
    rows = []
    for case, sub in one.groupby("case"):
        for base, arm, meaning in CONTRASTS:
            a = sub[sub.layout == base].set_index("target")["c_own"]
            b = sub[sub.layout == arm].set_index("target")["c_own"]
            common = sorted(set(a.index) & set(b.index))
            if len(common) < 3:
                continue
            d = b.loc[common].to_numpy() - a.loc[common].to_numpy()
            t = stats.ttest_1samp(d, 0.0)
            rows.append({"case": case, "contrast": f"{arm} - {base}",
                         "isolates": meaning, "shift": float(d.mean()),
                         "p": float(t.pvalue), "n": len(common)})
    return pd.DataFrame(rows)


def pooled_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Each contrast pooled over the cases, one photograph one row."""
    from scipy import stats

    one = (df[df.layer == df.layer.min()]
           [["case", "layout", "target", "c_own"]].drop_duplicates())
    rows = []
    for base, arm, meaning in CONTRASTS:
        diffs = []
        for _, sub in one.groupby("case"):
            a = sub[sub.layout == base].set_index("target")["c_own"]
            b = sub[sub.layout == arm].set_index("target")["c_own"]
            common = sorted(set(a.index) & set(b.index))
            diffs.append(b.loc[common].to_numpy() - a.loc[common].to_numpy())
        d = np.concatenate(diffs) if diffs else np.array([])
        if d.size < 3:
            continue
        t = stats.ttest_1samp(d, 0.0)
        rows.append({"contrast": f"{arm} - {base}", "isolates": meaning,
                     "shift": float(d.mean()), "p": float(t.pvalue),
                     "n": int(d.size)})
    return pd.DataFrame(rows)


def patch_landing(
    root: Path, truth_root: Path, layer: int, arm: str = "question_first",
) -> pd.DataFrame:
    """Does the question land on the cells the judgment uses?

    The per-patch gap says how far each patch moves when the question changes.
    The stored causal map says how much each patch is worth to the judgment.
    A positive correlation means the question reaches the evidence. A zero
    means it changes the encoding somewhere the judgment never reads.
    """
    from scipy.stats import pearsonr, spearmanr

    patches = load(root, "patch_gaps.parquet")
    patches = patches[(patches.layout == arm) & (patches.layer == layer)]
    rows = []
    for case, sub in patches.groupby("case"):
        truth_path = truth_root / f"{case}_absent" / "maps.parquet"
        if not truth_path.exists():
            continue
        truth = pd.read_parquet(truth_path)
        # `values` collides with `DataFrame.values`, so index the column by
        # name and zip it. `itertuples` renames it and the rename is silent.
        tmap = {str(i): np.asarray(v, dtype=float)
                for i, v in zip(truth["image_id"], truth["values"])}
        for r in sub.itertuples():
            if r.target not in tmap:
                continue
            gap = np.asarray(r.gap, dtype=float)
            t = tmap[r.target]
            if gap.size != t.size:
                continue
            ok = np.isfinite(gap) & np.isfinite(t)
            if ok.sum() < 10:
                continue
            rows.append({
                "case": case, "target": r.target,
                "r": float(pearsonr(gap[ok], np.abs(t[ok]))[0]),
                "rho": float(spearmanr(gap[ok], np.abs(t[ok]))[0]),
                "r_signed": float(pearsonr(gap[ok], t[ok])[0]),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--patch-truth", default=None,
                    help="A causal_maps root, to score where the question lands.")
    ap.add_argument("--patch-layer", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    df = load(root)
    cases = sorted(df.case.unique())
    ok = matched_cases(df)
    print(f"cases: {', '.join(cases)}")
    print(f"length-matched (patch_shift 0 everywhere): "
          f"{', '.join(ok) if ok else 'none'}\n")

    print("1. Does the question reach the image encoding?")
    print("   the gap in h, mean over the length-matched cases\n")
    print(encoding_table(df, ok).round(4).to_string())

    print("\n2. Does the order move the verdict?\n")
    pooled = pooled_contrasts(df)
    print(pooled.round(4).to_string(index=False))

    per_case = verdict_table(df)
    key = "question_first - neutral_first"
    print(f"\n   the decisive contrast, {key}, by case")
    sub = per_case[per_case.contrast == key].sort_values("shift")
    from statsmodels.stats.multitest import multipletests
    if len(sub):
        rej, q, _, _ = multipletests(sub.p.to_numpy(), alpha=0.05,
                                     method="fdr_bh")
        sub = sub.assign(q=q, survives=rej)
        print(sub[["case", "shift", "p", "q", "survives", "n"]]
              .round(4).to_string(index=False))

    landing = pd.DataFrame()
    if args.patch_truth:
        print("\n3. Where does the question land on the image?")
        print(f"   per-patch gap at block {args.patch_layer} against the "
              f"measured causal map\n")
        landing = patch_landing(root, Path(args.patch_truth),
                                args.patch_layer)
        if landing.empty:
            print("   no photograph matched a stored map")
        else:
            g = landing.groupby("case").agg(
                r=("r", "mean"), rho=("rho", "mean"),
                r_signed=("r_signed", "mean"), n=("r", "size"))
            print(g.round(4).to_string())
            from scipy import stats
            t = stats.ttest_1samp(landing.r.to_numpy(), 0.0)
            print(f"\n   pooled r = {landing.r.mean():+.4f}  "
                  f"p = {t.pvalue:.4f}  n = {len(landing)}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cases": cases, "matched": ok,
            "encoding": encoding_table(df, ok).reset_index()
            .to_dict(orient="records"),
            "pooled": pooled.to_dict(orient="records"),
            "per_case": per_case.to_dict(orient="records"),
            "landing": landing.to_dict(orient="records"),
        }, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
