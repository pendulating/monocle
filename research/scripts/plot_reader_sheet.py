"""What the reader sees: the measured map next to the instrument map.

One row for each photograph. The left panel is the photograph. The middle
panel is the MEASURED effect of the deletion of every one of the 256 image
tokens. The right panel is the instrument, which needs 1 backward pass.

The white square marks the cell the panel itself calls strongest. The black
ring marks the cell the MEASURED map calls strongest. A reader that works
puts its square on the ring.

Usage:
    python research/scripts/plot_reader_sheet.py --case restaurants --arm gxa \
        --out outputs/figures/reader_sheet_restaurants.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts.k1_separation import sep_z, table  # noqa: E402
from research.scripts.k1_layer_profile import pooled_ranks  # noqa: E402

TEAL = "#498573"
CORAL = "#e57264"
INK = "#1a1a1a"


def ramp():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "coral_teal", [CORAL, "#f6f2ec", TEAL])


def cell_box(ax, idx, side, kind, weak=False):
    """`ring` marks the measured strongest cell. `box` marks the panel's own."""
    import matplotlib.patches as mp
    r, c = divmod(int(idx), side)
    if weak:
        ax.add_patch(mp.Circle((c + 0.5, r + 0.5), 0.70, fill=False,
                               ec="#ffffff", lw=1.6, zorder=6, alpha=0.9))
        ax.add_patch(mp.Circle((c + 0.5, r + 0.5), 0.70, fill=False,
                               ec=INK, lw=0.7, zorder=7, ls=(0, (2, 2))))
        return
    if kind == "ring":
        ax.add_patch(mp.Circle((c + 0.5, r + 0.5), 0.70, fill=False,
                               ec="#ffffff", lw=2.6, zorder=6))
        ax.add_patch(mp.Circle((c + 0.5, r + 0.5), 0.70, fill=False,
                               ec=INK, lw=1.2, zorder=7))
    else:
        for ec, lw, z in ((INK, 3.0, 8), ("#ffffff", 1.4, 9)):
            ax.add_patch(mp.Rectangle((c, r), 1.0, 1.0, fill=False,
                                      ec=ec, lw=lw, zorder=z))


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from research.monocle import single as sg

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", default=None)
    ap.add_argument("--case", default="restaurants")
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--pool", action="store_true",
                    help="rank by the mean rank over every layer in the file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    default = (f"outputs/_monocle/gateB60_g1L/{a.case}/maps.parquet" if a.pool
               else f"outputs/_monocle/gateB60_g1/{a.case}/maps.parquet")
    maps = a.maps or default
    out = a.out or f"outputs/figures/reader_sheet_{a.case}.png"
    df = pd.read_parquet(maps)
    if a.pool:
        pooled = pooled_ranks(Path(maps), a.arm)
        rows_ = [{"target": k, "causal": list(t), a.arm: list(v)}
                 for k, (v, t) in pooled.items()]
        d = pd.DataFrame(rows_).set_index("target")
        tab = pd.DataFrame({a.arm: {k: sep_z(v, t)
                                    for k, (v, t) in pooled.items()}})
        cols = [a.arm]
    else:
        tab, cols = table(df, a.layer)
        d = df[df["layer"] == a.layer].drop_duplicates(subset=["target"])
        d = d.set_index("target")
    if a.arm not in cols:
        raise SystemExit(f"{a.arm} is absent; the file holds {cols}")

    paths = {r["image_id"]: r["path"]
             for r in sg.load_single_rows(a.n_images, a.case, seed=a.seed)}
    ok = [i for i in tab.index if i in paths and i in d.index]
    order = tab.loc[ok, a.arm].sort_values(ascending=False)
    pick = list(order.index[:a.rows - 1]) + [order.index[-1]]

    plt.rcParams.update({
        "font.size": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    cmap = ramp()
    fig, axes = plt.subplots(len(pick), 3, figsize=(7.6, 2.6 * len(pick)))
    if len(pick) == 1:
        axes = np.array([axes])
    for row, key in enumerate(pick):
        t = np.asarray(d.loc[key, "causal"], float).ravel()
        v = np.asarray(d.loc[key, a.arm], float).ravel()
        side = int(round(np.sqrt(t.size)))
        img = Image.open(paths[key]).convert("RGB")
        for col, (panel, title) in enumerate((
                (None, "the photograph"),
                (t, "the measured deletion of every token"),
                (v, f"{a.arm}, 1 backward pass"))):
            ax = axes[row, col]
            ax.imshow(img, extent=(0, side, side, 0), aspect="auto")
            if panel is not None:
                show = panel
                if col == 2 and a.pool:
                    # A pooled rank runs from 1 up, so a raw scale would paint
                    # every cell teal. Show the order instead.
                    from scipy.stats import rankdata
                    show = rankdata(panel) / (panel.size + 1.0) - 0.5
                m = float(np.max(np.abs(show))) or 1.0
                ax.imshow(show.reshape(side, side), cmap=cmap, vmin=-m,
                          vmax=m, alpha=0.55, extent=(0, side, side, 0),
                          interpolation="nearest", aspect="auto")
                for j in np.argsort(-t)[1:3]:
                    cell_box(ax, int(j), side, "ring", weak=True)
                cell_box(ax, int(np.argmax(t)), side, "ring")
                if col == 2:
                    cell_box(ax, int(np.argmax(panel)), side, "box")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=8, color=INK, pad=4)
        z = sep_z(v, t)
        axes[row, 0].set_ylabel(f"{key}\nseparation {z:+.2f}", fontsize=7,
                                color=INK)
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    import matplotlib.patches as mp
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], marker="o", mfc="none", mec=INK, mew=1.4, ms=11,
               ls="none", label="the measured strongest token"),
        Line2D([0], [0], marker="o", mfc="none", mec=INK, mew=0.7, ms=11,
               ls="none", label="the next 2 measured tokens"),
        mp.Rectangle((0, 0), 1, 1, fill=False, ec=INK, lw=1.4,
                     label="the token the instrument calls strongest"),
        mp.Patch(color=TEAL, label="the token holds the judgment up"),
        mp.Patch(color=CORAL, label="the token pushes the judgment down"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 0.055))
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
