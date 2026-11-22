"""The figure that carries the verdict: no instrument clears the null bar.

Each panel is 1 case. Each row is an instrument. The point is the median
paired gain over the strongest model-free null, inside the power gate. The
whisker is a bootstrap interval that chooses the bar inside each resample.

A bar to the RIGHT of zero would be an instrument that reads causal patches.

Usage:
    python research/scripts/plot_lens_verdict.py --json <verdict.json> --out <png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/share/ju/matt/mllmsci")

#: The palette of the camera-ready figures. See research/notebooks/cvpr/_style.py.
TEAL = "#498573"
CORAL = "#e57264"
GREY = "#7a6f63"

NICE = {
    "lens_own": "Jacobian lens",
    "gxa": "grad x activation",
    "cheap": "centered grad x act",
    "meang_own": "mean gradient",
    "meangp_own": "mean gradient, per position",
    "self": "measured map (ceiling)",
    "ceil": "measured map (ceiling)",
    "ceil60": "measured map (ceiling)",
    "mask": "mask search",
    "subset": "subset-fitted map",
    "position_prior": "position only",
}


def pretty(arm: str) -> str:
    a = arm.replace("_ablate", "")
    for pre in ("gbJ_", "gblibraries_", "gbrestaurants_", "n60_"):
        a = a.replace(pre, "")
    a = a.replace("_L18", "").replace("_L30", "")
    for case in ("subway_safety_g2_", "libraries_g2_", "restaurants_g2_"):
        a = a.replace(case, "")
    return NICE.get(a, a)


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True,
                    help="a file of {case: [{arm, gain, lo, hi, passes}]}")
    ap.add_argument("--out", default="outputs/figures/lens_verdict.png")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.json).read_text())
    # The file also holds a `<case>__gate` list. It is not a table.
    cases = [k for k in data if not k.endswith("__gate")]
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    n_row = max(len(v) for v in data.values())
    fig, axes = plt.subplots(1, len(cases), figsize=(3.9 * len(cases),
                                                     0.36 * n_row + 1.3),
                             sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        rows = sorted(data[case], key=lambda r: r["gain"])
        y = range(len(rows))
        for i, r in enumerate(rows):
            ceiling = "ceiling" in pretty(r["arm"])
            col = TEAL if r.get("passes") else (GREY if not ceiling else CORAL)
            ax.plot([r["lo"], r["hi"]], [i, i], color=col, lw=1.6,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["gain"]], [i], "o", color=col, ms=4.5, zorder=3)
        ax.axvline(0.0, color="#1a1a1a", lw=0.9, zorder=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels([pretty(r["arm"]) for r in rows])
        if ax is axes[len(cases) // 2]:
            ax.set_xlabel("gain over the strongest model-free null")
        ax.text(0.02, 1.02, case.replace("_", " "), transform=ax.transAxes,
                ha="left", va="bottom", color="#1a1a1a", fontsize=9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
