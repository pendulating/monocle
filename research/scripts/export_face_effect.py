#!/usr/bin/env python
"""Write the camera-face report, without marimo.

The question: a photograph in the battery is 1 face of a Cyclomedia cube, and
the face is a compass direction. `F` looks north, `R` east, `B` south, `L`
west. The face is a camera decision, thus a model that reads the PLACE must
ignore it. This script measures how much it does not.

What it writes, into `research/notebooks/cvpr/master/`:

| File | What it holds |
|------|---------------|
| `outputs/face_effect_grid_v*.parquet` | 1 row for each case, rater and design |
| `outputs/face_effect_grid_v*.csv` | the same, for a reader |
| `outputs/face_effect_table_v*.tex` | the camera-ready table |
| `outputs/face_head_to_head_v*.parquet` | the 4 x 4 matrix of each cell |
| `outputs/face_by_hour_v*.parquet` | the sun test, for the 2 image cases |
| `outputs/face_abstention_v*.parquet` | the face effect on the `NotSure` rate |
| `outputs/face_rank_shift_v*.parquet` | how far the correction moves a place |
| `figures/face_effect_grid.png` | the coefficient figure |
| `figures/face_head_to_head_<cell>.png` | the matrix of the 2 strongest cells |
| `figures/face_by_hour_<cell>.png` | the sun test |

It holds the gate: `_canonical.verify_or_raise()` runs before it writes a file.

Usage:
    python research/scripts/export_face_effect.py
    python research/scripts/export_face_effect.py --cases libraries road_quality
    python research/scripts/export_face_effect.py --arm nocity

Warning: run it from the canonical venv, `.venv-mllmsci-vllm025cu129`.
Warning: it reads 14 runs of 1,100,000 rows. It needs about 10 minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"
sys.path.insert(0, str(CVPR))

import pandas as pd  # noqa: E402

import _canonical as C  # noqa: E402
import _faces as FX  # noqa: E402

__version__ = "1.0.0"

OUT_DIR = CVPR / "master" / "outputs"
FIG_DIR = CVPR / "master" / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[faces] wrote {path.relative_to(REPO_ROOT)}")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _write(df: pd.DataFrame, stem: str, also_csv: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}_v{__version__}.parquet"
    df.to_parquet(path, index=False)
    print(f"[faces] wrote {path.relative_to(REPO_ROOT)}  ({len(df)} rows)")
    if also_csv:
        csv = path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        print(f"[faces] wrote {csv.relative_to(REPO_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(FX.CASES))
    ap.add_argument("--models", nargs="*", default=list(FX.MODELS))
    ap.add_argument("--arm", default="canonical")
    ap.add_argument("--min-place-comparisons", type=int, default=5)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    print("[faces] gate: the canonical registry")
    C.verify_or_raise()
    print(f"[faces] {C.summary()}")

    grid_rows, h2h_rows, hour_rows, abst_rows, shift_rows = [], [], [], [], []

    for case in args.cases:
        for model in args.models:
            frame = FX.load_face_frame(case, model, args.arm)
            if frame is None:
                print(f"[faces] {case}/{model}: the registry holds no such cell")
                continue
            fe = FX.default_fe(case)
            eff = FX.fit(frame, fe=fe)
            if eff is None:
                print(f"[faces] {case}/{model}: too few usable comparisons")
                grid_rows.append({"case": case, "model": model, "arm": args.arm,
                                  "n": int(frame["y"].notna().sum()),
                                  "notes": "too few usable comparisons"})
                continue
            grid_rows.append(eff.row())
            print(f"[faces] {case}/{model}: fe={fe} n={eff.n:,} "
                  f"swing={eff.swing:+.4f} (t={eff.swing_t:+.1f}) "
                  f"best={eff.best_face} worst={eff.worst_face} "
                  f"place var.={100 * eff.composition_share:.1f}%")

            tab = FX.head_to_head(frame, fe=fe)
            long = tab.stack().rename("mean_label").reset_index()
            long.insert(0, "case", case)
            long.insert(1, "model", model)
            h2h_rows.append(long)

            abst = FX.fit(frame, fe=fe, outcome="abstain")
            if abst is not None:
                abst_rows.append(abst.row())

            hours = FX.by_hour(frame)
            if not hours.empty:
                hours.insert(0, "case", case)
                hours.insert(1, "model", model)
                hour_rows.append(hours)

            shift = FX.corrected_place_scores(
                frame, eff, min_comparisons=args.min_place_comparisons)
            shift_rows.append(pd.DataFrame({
                "case": [case], "model": [model],
                "n_places": [len(shift)],
                "median_abs_rank_shift": [shift["rank_shift"].abs().median()],
                "p90_abs_rank_shift": [shift["rank_shift"].abs().quantile(0.9)],
                "max_abs_rank_shift": [shift["rank_shift"].abs().max()],
                "share_moved_over_5pct": [(shift["rank_shift"].abs() > 0.05).mean()],
                "rank_correlation": [shift["mean_score"].corr(
                    shift["corrected_score"], method="spearman")],
            }))

    grid = pd.DataFrame(grid_rows)
    if grid.empty:
        print("[faces] nothing to write")
        return 1
    grid["case_label"] = grid["case"].map(FX.CASE_LABEL)
    _write(grid, "face_effect_grid", also_csv=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tex = OUT_DIR / f"face_effect_table_v{__version__}.tex"
    tex.write_text(FX.to_latex(grid))
    print(f"[faces] wrote {tex.relative_to(REPO_ROOT)}")

    if h2h_rows:
        _write(pd.concat(h2h_rows, ignore_index=True), "face_head_to_head")
    if hour_rows:
        _write(pd.concat(hour_rows, ignore_index=True), "face_by_hour",
               also_csv=True)
    if abst_rows:
        _write(pd.DataFrame(abst_rows), "face_abstention")
    if shift_rows:
        _write(pd.concat(shift_rows, ignore_index=True), "face_rank_shift",
               also_csv=True)

    print()
    print("[faces] " + FX.summary(grid))
    st = FX.sign_test(grid)
    if st:
        print(f"[faces] sign test: {st}")

    if args.skip_figures:
        return 0

    _save(FX.plot_grid(grid), "face_effect_grid.png")

    # The matrix of the 2 cells with the largest swing. A matrix for each of
    # the 14 cells gives a reader nothing that the grid figure does not.
    top = grid.dropna(subset=["swing"]).nlargest(2, "swing")
    for _, r in top.iterrows():
        frame = FX.load_face_frame(r["case"], r["model"], args.arm)
        tab = FX.head_to_head(frame, fe=r["fe"])
        fig = FX.plot_head_to_head(
            tab, title_value=f", {FX.CASE_LABEL[r['case']]} / {r['model']}")
        _save(fig, f"face_head_to_head_{r['case']}_{r['model']}.png")

    for chunk in hour_rows:
        case, model = chunk["case"].iloc[0], chunk["model"].iloc[0]
        _save(FX.plot_by_hour(chunk), f"face_by_hour_{case}_{model}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
