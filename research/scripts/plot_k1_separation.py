"""The 1-cell deletion test, and the control that makes it causal.

Top row: for each case, the directed separation that an instrument reaches
under the real question and under another case's question.
Bottom row: the paired difference, with a bootstrap interval, over every arm
and every case.

Usage:
    python research/scripts/plot_k1_separation.py --json outputs/figures/k1_separation_L18.json \
        --out outputs/figures/k1_separation.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEAL = "#498573"
CORAL = "#e57264"
GREY = "#7a6f63"
INK = "#1a1a1a"
NICE = {"gxa": "grad x activation", "cheap": "centered grad x act",
        "meangp_own": "mean gradient, per position",
        "meang_own": "mean gradient", "lens_own": "Jacobian lens",
        "position_prior": "position prior", "shuffled": "shuffled truth",
        "ceiling": "the measured map"}
ORDER = ["gxa", "cheap", "meang_own", "meangp_own", "lens_own"]


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="outputs/figures/k1_separation.png")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.json).read_text())
    if isinstance(data, list):
        # The pooled layout is a flat list of rows. Fold it to the nested one.
        nested: dict = {}
        for r in data:
            blk = nested.setdefault(r["case"], {"arms": {}, "paired": []})
            blk["paired"].append(r)
        for case, blk in nested.items():
            blk["arms"]["ceiling"] = {"z": float("nan")}
        data = nested
    cases = [c for c in ("subway_safety", "libraries", "restaurants")
             if c in data]

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    fig = plt.figure(figsize=(3.3 * len(cases), 5.6))
    gs = fig.add_gridspec(2, len(cases), height_ratios=[1.0, 1.25],
                          hspace=0.55, wspace=0.18)

    top = [fig.add_subplot(gs[0, i]) for i in range(len(cases))]
    for i, (ax, case) in enumerate(zip(top, cases)):
        d = data[case]
        pair = {r["arm"]: r for r in d.get("paired", [])}
        arms = [x for x in ORDER if x in pair]
        y = np.arange(len(arms), dtype=float)
        ax.barh(y + 0.19, [pair[x]["real"] for x in arms], height=0.34,
                color=TEAL, zorder=2, label="the case's own question")
        ax.barh(y - 0.19, [pair[x]["wrong"] for x in arms], height=0.34,
                color=CORAL, zorder=2, label="another case's question")
        ax.axvline(0.0, color=INK, lw=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([NICE.get(x, x) for x in arms] if i == 0
                           else [""] * len(arms))
        ax.invert_yaxis()
        ax.set_xlim(-0.7, 2.6)
        ceil = d["arms"].get("ceiling", {}).get("z", float("nan"))
        head = case.replace("_", " ")
        if np.isfinite(ceil):
            head = f"{head}   ceiling {ceil:+.1f}"
        ax.text(0.0, 1.03, head, transform=ax.transAxes, ha="left",
                va="bottom", color=INK)
        if i == len(cases) // 2:
            ax.set_xlabel("separation of the 2 poles, in cell spreads")


    h, lab = top[0].get_legend_handles_labels()
    top[len(cases) // 2].legend(h, lab, frameon=False, ncol=2,
                                fontsize=8, loc="upper center",
                                bbox_to_anchor=(0.5, -0.36))

    rows = []
    for case in cases:
        for r in data[case].get("paired", []):
            if np.isfinite(r["p"]):
                rows.append((case, r))
    rows.sort(key=lambda t: t[1]["gain"])
    bot = fig.add_subplot(gs[1, :])
    y = np.arange(len(rows), dtype=float)
    for j, (case, r) in enumerate(rows):
        hit = r["p_bh"] < 0.05 and r["lo"] > 0
        col = TEAL if hit else GREY
        bot.plot([r["lo"], r["hi"]], [j, j], color=col, lw=1.6, zorder=2,
                 solid_capstyle="butt")
        bot.plot([r["gain"]], [j], "o", color=col, ms=5, zorder=3)
        if hit:
            bot.text(r["hi"] + 0.08, j, f"q={r['p_bh']:.4f}", va="center",
                     color=TEAL, fontsize=8)
    bot.axvline(0.0, color=INK, lw=0.8, zorder=1)
    bot.set_yticks(y)
    bot.set_yticklabels([f"{NICE.get(r['arm'], r['arm'])}  ·  "
                         f"{c.replace('_', ' ')}" for c, r in rows])
    bot.set_xlabel("gain over the same instrument under another question")
    lo = min(r["lo"] for _, r in rows)
    hi = max(r["hi"] for _, r in rows)
    bot.set_xlim(lo - 0.15, hi + 0.55)
    fig.savefig(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
