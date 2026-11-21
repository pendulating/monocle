#!/usr/bin/env python
"""Write the word-block figures of the paper from the current traces.

The word block sets the cues of 1 case as a block of text, the best win rate
first, with photographs of the units the model ranks highest above it and
lowest below it (`_extractions.plot_win_block`). The paper uses 3 forms:
the wide block, the 1-column block, and the win bars of road quality.

Until 2026-10-01 the paper held copies from the `stale_pre_looks/` run,
which read the traces of the prompts BEFORE the "looks like" wording of
2026-08-14. This script draws them again from the version 2 primary corpus
(`data/trace_extractions_v2_gemma12b_by_qwen27b`: gemma-4-12b traces of the
registered "looks" runs, read by Qwen3.8-27B). Version 2 keeps the
`visual_evidence` class with its `valence`, which the block reads; version 3
renames the class and moves the evaluative cues to `inference` and
`perspective`, thus the block stays on version 2, as the cue hierarchy does.

Each figure gets a `.json` caption sidecar (`_figmeta.save_figure`). Run
`scripts/curate_notebook_figures.py --apply` after this script to swap the
new files into the `paper_` copies.

Use:
    python scripts/export_wordblocks.py
    python scripts/export_wordblocks.py --cases libraries schools --min-count 25
    python scripts/export_wordblocks.py --cases restaurants street_photography
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CVPR = HERE.parent / "notebooks" / "cvpr"
sys.path.insert(0, str(CVPR))

__version__ = "1.3.0"

ROOT_V2 = "data/trace_extractions_v2_gemma12b_by_qwen27b"

# case -> (folder, figures to write)
CASES = {
    "libraries": ("libraries", ("win_block", "win_block_column")),
    "schools": ("schools", ("win_block", "win_block_column")),
    "subway_safety": ("subway", ("win_block", "win_block_column")),
    "road_quality": ("road", ("win_bars",)),
    "restaurants": ("restaurants", ("win_block", "win_block_column")),
    "street_photography": ("street_photography", ("win_block", "win_block_column")),
}

# The photographs of these cases come from the ranking of the registered proxy
# run of the same judge (`_extractions.unit_win_rates(proxy_rank=True)`): the
# trace run is too thin to rank them. Street photography pairs IMAGES, thus
# its photographs are images; the restaurants case has so many units that
# only 3 reach 8 decided comparisons in the trace run.
PROXY_RANK_CASES = {"street_photography", "restaurants"}
IMAGE_CASES = {"street_photography"}
# A raw face shows the capture vehicle at the bottom. F6 covers restaurants only, thus these photographs are cropped.
CROP_BOTTOM = {"street_photography": 0.30}

# The pairwise runs show raw cube faces, and the capture vehicle and non-storefront walls are in many of them.
# The restaurant photographs are the F6 views (`f6-restaurant-image-selection.md`): a virtual camera aimed at the
# storefront, with the pitch +10 degrees. Only the win rate comes from the raw faces.
F6 = HERE.parent / "data" / "f6_restaurant_depiction" / "v4"
F6_MAX_CAMERA_FT = 35.0  # about the median of the set (36.2 ft)
F6_MAX_ANGLE_DEG = 20.0
# Low-ranked restaurants that pass the rule, but whose view does not show their storefront. Checked by eye, 2026-10-30.
F6_SCREENED = {
    "50051888": "Wingstop: the frame shows the sign of another shop",
    "41538728": "Lucky Strike: a truck fills the frame",
    "50133551": "Purple Dough: a tree fills the frame",
    "50146300": "El Perro Colombian Fast Food: a shutter and no sign",
    "50067247": "Starbucks #49952: a construction hoarding covers the storefront",
    "41543866": "Dallas BBQ: a tree fills the frame",
    "50015128": "Alidoro: scaffolding covers the storefront",
    "41039810": "Starbucks Coffee: the frame shows the sign of another shop",
}

CASE_LABEL = {"libraries": "libraries", "schools": "schools", "subway_safety": "subway entrances",
              "road_quality": "road condition", "restaurants": "restaurants",
              "street_photography": "street photography locations"}


def caption(kind: str, case: str, n_traces: int, min_count: int) -> str:
    what = CASE_LABEL[case]
    if kind == "win_bars":
        return (f"How often the image a cue describes wins, for {what}. Each bar is 1 cue "
                f"(a visual-evidence span, by its text), with at least {min_count} decided "
                "comparisons; the bar length is its win rate with the Wilson interval, the colour "
                "is the rate shrunk toward the base rate (teal above, coral below), and the dashed "
                "line marks the base rate of the case. A cue named on both images of a pair "
                f"is dropped. Gemma-4-12B judge, Qwen3.8-27B extractor, {n_traces:,} traces "
                "(schema v2).")
    width = "1 column" if kind == "win_block_column" else "2 columns"
    photos = "3" if kind == "win_block_column" else "6"
    if case == "restaurants":
        shown = (f"The {photos} photographs above are restaurants the model ranks highest, the {photos} below are "
                 "restaurants it ranks lowest, by the 1,000,000-pair run of the same model and question, with no "
                 "thinking (the trace run has too few comparisons of each restaurant). The percentages come from "
                 "that run. The photographs are NOT the images the model judged: the run shows raw cube faces, and "
                 "many of them show no storefront. Each photograph is the storefront view of the restaurant from "
                 f"the F6 selection: within {F6_MAX_CAMERA_FT:g} ft and {F6_MAX_ANGLE_DEG:g} degrees of the wall "
                 "normal, with a Yes from the pixel check. We also removed by hand, from the low end, "
                 f"{len(F6_SCREENED)} restaurants whose view does not show the storefront. They show UNITS, the "
                 "words show CUES, and the 2 scales have different middles.")
    elif case in PROXY_RANK_CASES:
        what_unit = "images" if case in IMAGE_CASES else "units"
        why = ("This case pairs images, and the trace run judges each image about 1 time"
               if case in IMAGE_CASES else "Few units reach 8 decided comparisons in the trace run")
        shown = (f"The {photos} photographs above are {what_unit} the model ranks highest, the {photos} below "
                 f"are {what_unit} it ranks lowest. {why}, thus the ranking comes from the 1,000,000-pair run "
                 "of the same model and question, with no thinking (8 or more decided comparisons; win rate "
                 f"shrunk toward the base rate). They show {what_unit.upper()}, the words show CUES, and the 2 "
                 "scales have different middles.")
        if case in CROP_BOTTOM:
            shown += (f" The photographs are cropped: the bottom {CROP_BOTTOM[case]:.0%} of the height is removed to "
                      "hide the capture vehicle, and the model judged the full image.")
    else:
        shown = (f"The {photos} photographs above are units the model ranks highest, the {photos} "
                 "below are units it ranks lowest: they show UNITS, the words show CUES, and the 2 "
                 "scales have different middles.")
    return (f"The cues of {what}, set as text in the order of their win rate, best first "
            f"({width}). The colour repeats the order: teal cues sit on the image that wins more "
            "often than the base rate, coral cues on the image that loses, and grey cues do not "
            f"separate the 2 images; the bar gives the scale. Each cue has at least {min_count} decided comparisons. "
            f"{shown} Gemma-4-12B judge, Qwen3.8-27B extractor, "
            f"{n_traces:,} traces (schema v2, the \"looks like\" prompts).")


def f6_units(units: "pd.DataFrame") -> "pd.DataFrame":
    """Keep the ranked restaurants with a close, frontal F6 view that passes the pixel check, and show that view.

    `units` is the table of `_extractions.unit_win_rates`. The order, the win
    rate and `n` stay; `image_path` becomes the F6 view, and
    `image_distance_ft` becomes the camera distance of that view.
    """
    import pandas as pd

    check = pd.concat([pd.read_parquet(p, columns=["uid", "view_path", "answer"])
                       for p in sorted(glob.glob(str(F6 / "facade_check_b300" / "part_*.parquet")))])
    check = check[check["answer"] == "Yes"].drop_duplicates("uid").assign(uid=lambda d: d["uid"].astype(str))
    best = pd.read_parquet(F6 / "best.parquet", columns=["uid", "cam_to_s_ft", "incidence_deg"])
    best = best.drop_duplicates("uid").assign(uid=lambda d: d["uid"].astype(str))
    out = (units.assign(unit=units["unit"].astype(str)).merge(check, left_on="unit", right_on="uid")
           .merge(best, on="uid"))
    out = out[(out["cam_to_s_ft"] <= F6_MAX_CAMERA_FT) & (out["incidence_deg"].abs() <= F6_MAX_ANGLE_DEG)
              & ~out["unit"].isin(F6_SCREENED)]
    out = out.assign(image_path=out["view_path"], image_distance_ft=out["cam_to_s_ft"].round(0))
    return out.sort_values("shrunk_rate", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES))
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--root", default=ROOT_V2)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import _canonical as C
    import _extractions as X
    import _figmeta as FM
    import _style as S

    C.verify_or_raise()
    S.apply_house_style()
    root = (HERE.parent / args.root).resolve()
    df = X.load(root=root, require_canonical=True)
    print(f"[wordblocks] corpus {args.root}: {len(df):,} rows, {df.case.nunique()} cases", flush=True)

    for case in args.cases:
        folder, kinds = CASES[case]
        part = df[df.case == case]
        n_traces = int(part.pair_id.nunique())
        units = f6_units(X.unit_win_rates(part, case, proxy_rank=True)) if case == "restaurants" else None
        out_dir = CVPR / folder / "figures"
        for kind in kinds:
            if kind == "win_block":
                fig, _ = X.plot_win_block(part, case, min_count=args.min_count, unit_photos=6,
                                          proxy_rank=case in PROXY_RANK_CASES, units=units,
                                          crop_bottom=CROP_BOTTOM.get(case, 0.0))
            elif kind == "win_block_column":
                fig, _ = X.plot_win_block_column(part, case, min_count=args.min_count,
                                                 proxy_rank=case in PROXY_RANK_CASES, units=units,
                                                 crop_bottom=CROP_BOTTOM.get(case, 0.0))
            else:
                fig = X.plot_win_bars(part, case, min_count=args.min_count)
            path = out_dir / f"{case}_{kind}.png"
            FM.save_figure(
                fig, path, dpi=300,
                title=f"Cue win rates, {CASE_LABEL[case]}",
                caption=caption(kind, case, n_traces, args.min_count),
                label=f"fig:{case.replace('_', '-')}-{kind.replace('_', '-')}",
                tags=["wordblocks", case, kind, "gemma_by_qwen27_v2", "v2", "win_rate"],
                provenance={"script": "scripts/export_wordblocks.py", "version": __version__,
                            "corpus": args.root, "min_count": args.min_count,
                            **({"f6_screened": F6_SCREENED} if case == "restaurants" else {})},
            )
            plt.close(fig)
            print(f"[wordblocks] {path.relative_to(CVPR)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
