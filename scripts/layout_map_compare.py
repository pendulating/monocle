"""Does the ORDER change which image cells cause the judgment?

`monocle/modality_order.py` showed that a question placed before the image
changes the patch residual, and that the change does not land on the cells the
judgment uses. That test scored the residual change against a map measured
under the PRODUCTION order. This script asks the harder question directly:
measure the causal map under each order, and compare the maps.

The 4 arms
----------
An arm is a (layout, filler kind) pair. Three of them hold 3 blocks and match
the question in tokens, so they share the patch positions, the distance to the
answer, and the sequence length.

| Arm | Blocks | The leading text |
|---|---|---|
| `image_first` | image, question | none. This is production |
| `neutral_first` | filler, image, question | 1 sentence, repeated |
| `decoy_first` | decoy, image, question | prose about an unrelated subject |
| `question_first` | question, image, filler | the question |

The comparisons that count
--------------------------
| Pair | Shares the geometry | Reads |
|---|---|---|
| `question_first` against `neutral_first` | **yes**, exactly | the question, at fixed geometry |
| `decoy_first` against `neutral_first` | **yes**, exactly | an informative text, at fixed geometry |
| `neutral_first` against `image_first` | no | position, and a length change |

The measured prompts confirm the match. For each case the 3 filler arms give
the same sequence length, the same first patch, and the same answer distance.
`image_first` is 52 tokens shorter and starts the patches at index 5.

**Read a matched pair.** The `image_first` rows carry a length difference,
and a length difference alone moves a bf16 forward pass.

The decoy answers a limit the repeated filler leaves. A map that changes under
`question_first` could change because the leading block carries INFORMATION,
and not because it carries the QUESTION. The decoy separates the 2.

The floor
---------
`monocle/truth_retest.py` measured the map twice. At the SAME batch size it
returns the stored map to the last bit: r = 1.000000, with a shuffled cell
order in a new process. Only the batch size moves it. Every run here uses
`--batch 12`, so the floor for a matched pair is 1.000000 and any drop is a
real difference.

The floor holds the batch fixed, not the length. So it is a true floor for a
matched pair only. Read the `image_first` rows as an upper bound on the change,
because they add the length difference to whatever the order does.

Usage:
    python scripts/layout_map_compare.py \\
        --root outputs/_monocle/causal_maps_layout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs/_monocle/causal_maps_layout"
#: An arm is a (layout, filler kind) pair. `decoy_first` keeps the layout of
#: `neutral_first` and changes only what the leading block SAYS.
ARMS = {
    "image_first": ("image_first", "neutral"),
    "neutral_first": ("neutral_first", "neutral"),
    "question_first": ("question_first", "neutral"),
    "decoy_first": ("neutral_first", "decoy"),
}
#: (base, arm, what the difference isolates, whether the geometry matches).
PAIRS = (
    ("neutral_first", "question_first",
     "the question, at fixed geometry", True),
    ("neutral_first", "decoy_first",
     "an informative text, at fixed geometry", True),
    ("decoy_first", "question_first",
     "the question against an informative text", True),
    ("image_first", "neutral_first",
     "the position, and a length change", False),
    ("image_first", "question_first", "the whole change", False),
)


def map_dir(root: Path, case: str, arm: str, group: int = 2) -> Path:
    """The folder `causal_maps.case_dir_for` writes for an absence run."""
    layout, kind = ARMS[arm]
    name = case if group == 1 else f"{case}_g{group}"
    name = f"{name}_absent"
    name = name if layout == "image_first" else f"{name}_{layout}"
    return root / (name if kind == "neutral" else f"{name}_{kind}")


def load_maps(root: Path, case: str, arm: str, group: int = 2) -> dict:
    """`image_id -> the cell values`, or an empty dict when the run is absent."""
    path = map_dir(root, case, arm, group) / "maps.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    # `values` collides with `DataFrame.values`, so take the column by name.
    return {str(i): np.asarray(v, dtype=float)
            for i, v in zip(df["image_id"], df["values"])}


def compare_pair(a: dict, b: dict) -> pd.DataFrame:
    """One row per photograph the 2 maps share."""
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for image_id in sorted(set(a) & set(b)):
        x, y = a[image_id], b[image_id]
        if x.size != y.size:
            continue
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        x, y = x[ok], y[ok]
        rows.append({
            "image_id": image_id,
            "r": float(pearsonr(x, y)[0]),
            "rho": float(spearmanr(x, y)[0]),
            "top1_same": bool(int(np.argmax(x)) == int(np.argmax(y))),
            "bottom1_same": bool(int(np.argmin(x)) == int(np.argmin(y))),
            # The map is in logits, so the size of the move is readable.
            "max_abs_diff": float(np.max(np.abs(x - y))),
            "range_a": float(x.max() - x.min()),
        })
    return pd.DataFrame(rows)


def leading_block_test(root: Path, cases: list[str],
                       group: int = 2) -> pd.DataFrame:
    """Does the QUESTION reorganise the map, or would any text do it?

    Each arm below scores against production. Each adds a block, and each
    pushes the patches 52 tokens later. They differ in 1 thing: which block
    leads the image. So a paired difference between 2 arms removes the
    position change and the length change, and leaves the content of the
    leading block.

    The 3 arms make a ladder. The filler holds little information, the decoy
    holds information but no relevance, and the question holds both. So the
    2 steps split the effect into an information part and a relevance part.
    """
    rows = []
    for tag, arm in (("filler leads", "neutral_first"),
                     ("decoy leads", "decoy_first"),
                     ("question leads", "question_first")):
        for case in cases:
            df = compare_pair(load_maps(root, case, "image_first", group),
                              load_maps(root, case, arm, group))
            rows.append(df.assign(case=case, arm=tag))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--cases", nargs="+",
                    default=["subway_safety", "libraries",
                             "street_photography"])
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    print("the floor: at a matched batch the map repeats at r = 1.000000")
    print("(truth_retest, shuffled cell order, a new process)\n")

    out: list[dict] = []
    for base, arm, meaning, matched in PAIRS:
        head = f"{arm} against {base}"
        flag = "" if matched else "   [the lengths DIFFER; read with care]"
        print(f"{head:<44} {meaning}{flag}")
        print(f"  {'case':<20} {'r':>8} {'rho':>8} {'top1':>7} "
              f"{'max|d|':>8} {'range':>8} {'n':>4}")
        for case in args.cases:
            a = load_maps(root, case, base, args.group)
            b = load_maps(root, case, arm, args.group)
            if not a or not b:
                print(f"  {case:<20} (a run is missing)")
                continue
            df = compare_pair(a, b)
            if df.empty:
                print(f"  {case:<20} (no shared photograph)")
                continue
            print(f"  {case:<20} {df.r.mean():>8.4f} {df.rho.mean():>8.4f} "
                  f"{df.top1_same.mean():>7.2f} "
                  f"{df.max_abs_diff.mean():>8.3f} "
                  f"{df.range_a.mean():>8.3f} {len(df):>4}")
            out.append({"base": base, "arm": arm, "case": case,
                        "isolates": meaning, "lengths_match": matched,
                        "r": float(df.r.mean()), "rho": float(df.rho.mean()),
                        "top1_same": float(df.top1_same.mean()),
                        "max_abs_diff": float(df.max_abs_diff.mean()),
                        "range": float(df.range_a.mean()), "n": len(df)})
        print()

    print("pooled, against a floor of 1.0000 (a value near 1 means the "
          "arm does")
    print("NOT move the evidence; a value well below 1 means it does)")
    for base, arm, meaning, matched in PAIRS:
        if not matched:
            continue
        got = [o["r"] for o in out if o["base"] == base and o["arm"] == arm]
        if got:
            print(f"  {arm + ' against ' + base:<44} r = {np.mean(got):.4f}"
                  f"   {meaning}")

    ctrl = leading_block_test(root, args.cases, args.group)
    if not ctrl.empty:
        from scipy import stats
        wide = ctrl.pivot_table(index=["case", "image_id"], columns="arm",
                                values="r")
        wide = wide.dropna()
        print("\nthe control: which block leads, all scored against "
              "production")
        print("  (every arm adds a block and moves the patches 52 tokens "
              "later,")
        print("   so a paired difference is the content of the leading "
              "block)")
        for tag in ("filler leads", "decoy leads", "question leads"):
            if tag in wide:
                print(f"  {tag:<18} r = {wide[tag].mean():+.4f}")
        steps = (("filler leads", "decoy leads"),
                 ("filler leads", "question leads"),
                 ("decoy leads", "question leads"))
        for base, tag in steps:
            if tag not in wide or base not in wide:
                continue
            a, b = wide[base], wide[tag]
            t = stats.ttest_rel(b, a)
            w = stats.wilcoxon(b, a)
            print(f"  {tag} - {base}: {(b - a).mean():+.4f}  "
                  f"t p = {t.pvalue:.2e}  wilcoxon p = {w.pvalue:.2e}  "
                  f"lower in {int((b < a).sum())} of {len(wide)}")
            out.append({"test": "leading_block", "base": base, "arm": tag,
                        "r_base": float(a.mean()), "r_arm": float(b.mean()),
                        "difference": float((b - a).mean()),
                        "p_ttest": float(t.pvalue),
                        "p_wilcoxon": float(w.pvalue), "n": int(len(wide))})

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
