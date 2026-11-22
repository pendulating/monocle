"""The gain on the first 60 photographs and on 60 new ones, side by side.

One row for each case. Each row holds 2 intervals: the set the study was
built on, and a set with 0 overlap that no part of the study has seen.

The figure carries no title. The caption of the paper owns that text.

Usage:
    python research/scripts/plot_holdout_replication.py \
        --train outputs/_monocle/k1_pool_paraphrase.json \
        --holdout outputs/_monocle/k1_pool_holdout.json \
        --out outputs/figures/k1_holdout.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUND = "#ffffff"
INK = "#1b1d21"
WARM = "#e8735a"
COOL = "#2f7f8c"
FAINT = "#c9ccd2"
CASES = ("restaurants", "libraries", "subway_safety")
NICE = {"restaurants": "restaurants", "libraries": "libraries",
        "subway_safety": "subway safety"}


def pick(rows: list, case: str, arm: str) -> dict | None:
    for r in rows:
        if r.get("case") == case and r.get("arm") == arm:
            return r
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    tr = json.loads(Path(args.train).read_text())
    ho = json.loads(Path(args.holdout).read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    fig.patch.set_alpha(0)
    ax.set_facecolor(GROUND)
    ax.axvline(0, color=FAINT, lw=1.2, zorder=1)

    ticks, labels = [], []
    y = 0.0
    for case in CASES:
        for src, rows, colour, name in (("first 60", tr, COOL, "first 60"),
                                        ("held out", ho, WARM, "60 new")):
            r = pick(rows, case, args.arm)
            y -= 1.0
            if r is None:
                continue
            ax.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=2.4,
                    solid_capstyle="round", zorder=3)
            ax.plot([r["gain"]], [y], "o", color=colour, ms=7, zorder=4)
            ax.text(r["hi"] + 0.09, y, f"{r['gain']:+.2f}", va="center",
                    ha="left", fontsize=9, color=colour)
            ticks.append(y)
            labels.append(f"{NICE[case]}   {name}")
        y -= 0.55
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlabel("gain over the other judgment  (z of the measured map)",
                  fontsize=10, color=INK)
    ax.tick_params(colors=INK, labelsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.set_xlim(-1.2, 2.6)
    ax.set_ylim(y - 0.3, -0.4)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, transparent=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
