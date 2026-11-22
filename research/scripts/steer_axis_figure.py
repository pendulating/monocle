"""Dose-response figure of the 4-arm steering sweep.

One row per case, one panel per block. Each panel draws the mean
``logit_diff`` over the 60 eval images against the dose, one line per arm.
A filled marker is a cell that holds the answer contract (90 percent or
more of forwards emit a pole word); a hollow marker is off the contract.
The axis arm's contract window is shaded.

A second figure shows block 41 alone, one panel per case.

No titles: the caption owns that text.

Usage:
    python research/scripts/steer_axis_figure.py
    python research/scripts/steer_axis_figure.py --root outputs/_monocle/single_steer_axis_ext
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path("/share/ju/matt/mllmsci")
ROOT_DEFAULT = REPO / "outputs/_monocle/single_steer_axis"
ROOT = ROOT_DEFAULT
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
ARMS = {
    "axis": ("judgment axis", "#2a9d8f", "-"),
    "signal": ("pinv vector", "#e76f51", "-"),
    "control": ("pinv random", "#8d99ae", "--"),
    "control_raw": ("raw random", "#8d99ae", ":"),
}
CONTRACT = 0.90


def cells_of(case: str) -> tuple[dict, list[int], list[float]]:
    s = json.loads((ROOT / case / "summary.json").read_text())
    return s["per_cell"], s["layers"], s["alphas"]


def draw_panel(ax, cells: dict, layer: int, alphas: list[float],
               arms: list[str], ylabel: bool) -> None:
    lo = hi = 0.0
    for arm in arms:
        if arm not in ARMS:
            continue
        label, col, ls = ARMS[arm]
        xs, ys, on = [], [], []
        for a in alphas:
            c = cells.get(f"{arm}/L{layer}/a{a:+g}")
            if c is None:
                continue
            xs.append(a)
            ys.append(c["mean_logit_diff"])
            on.append(c.get("frac_on_contract", 1.0) >= CONTRACT)
        xs, ys, on = np.array(xs), np.array(ys), np.array(on)
        ax.plot(xs, ys, ls, color=col, lw=1.4, label=label, zorder=2)
        ax.scatter(xs[on], ys[on], s=14, color=col, zorder=3)
        ax.scatter(xs[~on], ys[~on], s=14, facecolors="white",
                   edgecolors=col, zorder=3)
        if arm == "axis":
            # The contiguous on-contract window around 0.
            i0 = int(np.argmin(np.abs(xs)))
            j = k = i0
            while j - 1 >= 0 and on[j - 1]:
                j -= 1
            while k + 1 < len(xs) and on[k + 1]:
                k += 1
            lo, hi = float(xs[j]), float(xs[k])
    ax.axvspan(lo, hi, color="#2a9d8f", alpha=0.07, lw=0, zorder=1)
    ax.axhline(0, color="black", lw=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.5, zorder=1)
    ax.set_xlabel("dose (alpha, centered-scale units)")
    if ylabel:
        ax.set_ylabel("mean logit(pos) - logit(neg)")
    ax.set_facecolor("white")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT_DEFAULT),
                    help="a sweep root with <case>/summary.json")
    ap.add_argument("--out-dir", default=None,
                    help="default: <root>/figures")
    args = ap.parse_args(argv)
    global ROOT
    ROOT = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else ROOT / "research" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASES if (ROOT / c / "summary.json").exists()]

    # Figure 1: every case, every block.
    layers = cells_of(cases[0])[1]
    fig, axes = plt.subplots(len(cases), len(layers),
                             figsize=(3.4 * len(layers), 2.6 * len(cases)),
                             squeeze=False)
    for r, case in enumerate(cases):
        cells, _, alphas = cells_of(case)
        arms = sorted({k.split("/")[0] for k in cells})
        for c, layer in enumerate(layers):
            ax = axes[r, c]
            draw_panel(ax, cells, layer, alphas, arms, ylabel=(c == 0))
            ax.text(0.02, 0.96, f"{case}  L{layer}", transform=ax.transAxes,
                    fontsize=8, va="top")
            if r != len(cases) - 1:
                ax.set_xlabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "dose_response_all.png", dpi=150, transparent=True)
    plt.close(fig)

    # Figure 2: one block, one panel per case (41 when the sweep has it,
    # else the middle block of the sweep).
    layer = 41 if 41 in layers else layers[len(layers) // 2]
    fig, axes = plt.subplots(1, len(cases), figsize=(3.0 * len(cases), 3.0),
                             squeeze=False)
    for c, case in enumerate(cases):
        cells, _, alphas = cells_of(case)
        arms = sorted({k.split("/")[0] for k in cells})
        ax = axes[0, c]
        draw_panel(ax, cells, layer, alphas, arms, ylabel=(c == 0))
        ax.text(0.02, 0.96, case, transform=ax.transAxes, fontsize=8,
                va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / f"dose_response_L{layer}.png", dpi=150,
                transparent=True)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
