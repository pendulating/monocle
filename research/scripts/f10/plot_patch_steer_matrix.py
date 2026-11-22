"""F10.35: a full-page matrix. The teaser images, with each patch direction added to all patches at 1 layer.

Columns: the 5 teaser images (`research/figures/f10/mechanism/patch_teaser_images.csv`, as the teaser has them now).
Rows: no addition, then ALPHA sigma of each patch direction (`hop_steer.py patch_dirs`) added to all 256
patches at layer HOOK: shared (+ = the `nice` side), cool, pretty, ugly (+ = toward yes), and the random
control at the dose of shared. The data is from `hop_steer.py patches` (`data/f10/steer/gemma-4-31b/patches/`).

Words. The teaser word lists (`plot_patch_teaser.pole_words`, up end; `severe_words`), read from the top 30 of
each patch at layer READ. The teaser block rule: the score of a word in a block is the number of its 16 patches
that hold the word; a block with a severe word shows it (coral), the up words fill the other blocks (teal); a
word shows at most MAX_REPEAT times in an image; a block with no word stays empty.

The design is the teaser (`plot_patch_teaser.py` 2.3.0): the same images, word lists, block rule, veil and frame.
One change: the teaser reads the severe words at L16, but a vector added at L32 cannot change L16. Thus here both
ends come from the same layer READ = HOOK, in all rows, also the first row. The J++ Lens is linear up to the
final norm (F10.29 item 7), thus at the hook layer the readout of a patch is its own readout plus the lens image
of the vector. The counts CSV also gives the hits at READ_LATER, 8 layers on, where the model has processed the
vector, and the answer margins of the 3 prompts.

The frame of each image carries the answer to "Is this place cool?" in that condition (teal yes, coral no).
Six rows fit the text height with the caption; thus the figure shows only +ALPHA, not -ALPHA.

Use:
    python research/scripts/f10/plot_patch_steer_matrix.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

__version__ = "1.1.0"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
import plot_patch_teaser as T  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

SRC = ROOT / "data/f10/steer" / T.MODEL / "patches"
OUT = ROOT / "research/figures/f10/steer_patches/patch_steer_matrix"
HOOK, READ, READ_LATER = 32, 32, 40
ALPHA = 96.0
ROWS = (("none", "no addition"), ("shared", "+ shared"), ("cool", "+ cool"), ("pretty", "+ pretty"),
        ("ugly", "+ ugly"), ("control", "+ random"))
PANEL = 1.0  # inches
LABEL = 0.2  # inches, the row labels
GAP = 0.04  # inches


def key(name: str) -> str:
    return "none|-1|0.0" if name == "none" else f"{name}|{HOOK}|{ALPHA}"


def grid_of(x: np.ndarray, up: dict, severe: dict) -> list:
    """Give the 4 x 4 (word, end) grid of 1 image from its top ids [patches, top], with the teaser rule."""
    k = T.GRID // T.POOL
    g = x.reshape(k, T.POOL, k, T.POOL, -1).transpose(0, 2, 1, 3, 4).reshape(k, k, T.POOL * T.POOL, -1)
    grid, cand = [[("", 0)] * k for _ in range(k)], []
    for words in (up, severe):
        for i in range(k):
            for j in range(k):
                for t in np.unique(g[i, j]):
                    if int(t) in words:
                        w, end = words[int(t)]
                        if len(w) <= T.MAX_LEN and w not in T.FRAGMENTS:
                            cand.append((int((g[i, j] == t).any(-1).sum()), w, end, i, j))
    used, done = Counter(), set()
    for s, w, end, i, j in sorted(cand, key=lambda c: (c[2], -c[0], c[1])):  # severe (-1) first
        if (i, j) not in done and used[w] < T.MAX_REPEAT:
            grid[i][j] = (w, end)
            done.add((i, j))
            used[w] += 1
    return grid


def main() -> None:
    import _style as S
    from PIL import Image

    OUT.parent.mkdir(parents=True, exist_ok=True)
    teaser = pd.read_csv(ROOT / "research/figures/f10/mechanism/patch_teaser_images.csv")
    missing = [s for s in teaser.sample_id if not (SRC / f"{s}.npz").exists()]
    if missing:
        raise SystemExit(f"no steering data for {missing}: run `hop_steer.py patches --images teaser --hook-layers {HOOK}`")
    store = open_store(T.MODEL, ALL_ROOT)
    vocab = pd.read_parquet(store.path / "vocab.parquet")
    up = {t: v for t, v in T.pole_words(vocab).items() if v[1] > 0}
    severe = T.severe_words(vocab)
    up_ids, sev_ids = np.array(list(up)), np.array(list(severe))
    images = store.images.set_index("sample_id")

    S.apply_house_style()
    ink = {1: S.PAL["teal"], -1: S.PAL["coral"], 0: S.EDGE}
    k = T.GRID // T.POOL
    n_r, n_c = len(ROWS), len(teaser)
    head = 0.2  # inches, the column titles
    width, height = LABEL + n_c * PANEL + (n_c - 1) * GAP, head + n_r * PANEL + (n_r - 1) * GAP
    fig = plt.figure(figsize=(width, height))
    renderer = fig.canvas.get_renderer()
    counts = []
    for c, r in enumerate(teaser.itertuples()):
        z = np.load(SRC / f"{r.sample_id}.npz")
        conds, reads = list(z["conditions"]), list(z["read_layers"])
        rows = pd.read_parquet(SRC / f"{r.sample_id}.parquet")
        img = np.asarray(Image.open(images.loc[r.sample_id, "image_path"]).convert("RGB"))
        for i, (name, label) in enumerate(ROWS):
            x0 = LABEL + c * (PANEL + GAP)
            y0 = height - head - (i + 1) * PANEL - i * GAP
            ax = fig.add_axes([x0 / width, y0 / height, PANEL / width, PANEL / height])
            ids = z["ids"][conds.index(key(name))]
            sel = (rows.direction == name) & (rows.alpha == (0.0 if name == "none" else ALPHA)) & (rows.layer == (-1 if name == "none" else HOOK))
            mg = rows[sel].set_index("prompt").margin
            for l in (READ, READ_LATER):
                a = ids[:, reads.index(l)]
                counts.append({"sample_id": r.sample_id, "borough": r.borough, "row": name, "read": l,
                               "up_patches": int(np.isin(a, up_ids).any(-1).sum()), "severe_patches": int(np.isin(a, sev_ids).any(-1).sum()),
                               **{f"m_{q.split()[1]}": round(float(v), 3) for q, v in mg.items()}})
            ax.imshow(img, extent=(0, k, k, 0))
            ax.imshow(np.ones((1, 1, 3)), extent=(0, k, k, 0), alpha=0.22)
            for t in range(1, k):
                ax.axhline(t, color="white", lw=0.4, alpha=0.6)
                ax.axvline(t, color="white", lw=0.4, alpha=0.6)
            grid = grid_of(ids[:, reads.index(READ)], up, severe)
            for a_ in range(k):
                for b_ in range(k):
                    w, end = grid[a_][b_]
                    if not w:
                        continue
                    txt = ax.text(b_ + .5, a_ + .5, w, ha="center", va="center", fontsize=T.WORD_PT, color=ink[end],
                                  fontweight="bold", path_effects=[pe.withStroke(linewidth=.9, foreground="white")])
                    box = ax.transData.transform([(b_, 0), (b_ + 1, 0)])
                    room, used = .92 * (box[1, 0] - box[0, 0]), txt.get_window_extent(renderer).width
                    if used > room:
                        txt.set_fontsize(max(T.MIN_PT, T.WORD_PT * room / used))
            ax.set_xlim(0, k)
            ax.set_ylim(k, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            m = rows[sel & (rows.prompt == "place cool")].iloc[0]
            yes = m.p_yes > m.p_no
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(ink[1] if yes else ink[-1])
                side.set_linewidth(T.FRAME_LW)
            if c == 0:
                fig.text((LABEL - 0.05) / width, (y0 + PANEL / 2) / height, label, rotation=90, ha="right", va="center", fontsize=6.5)
            if i == 0:
                fig.text((x0 + PANEL / 2) / width, (height - head + 0.05) / height, r.borough, ha="center", va="bottom",
                         fontsize=6.5, fontweight="bold")
    for ext in ("png", "pdf"):
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {"dpi": 300}
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.01, facecolor="white", **kw)
    plt.close(fig)
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT.with_name("patch_steer_matrix_counts.csv"), index=False)
    print(counts.pivot_table(index=["row"], columns="read", values=["up_patches", "severe_patches"], aggfunc="mean").round(1).to_string())
    print(counts[counts.read == READ].pivot_table(index="row", columns="borough", values="m_cool").round(1).to_string())
    print(f"[f10.35] figure {width:.2f} x {height:.2f} in")
    OUT.with_suffix(".json").write_text(json.dumps({
        "plot-title": "Patch words when a direction is added",
        "plot-caption": (f"The teaser images, with {ALPHA:.0f} sigma of each patch direction of {T.MODEL} added to all 256 image "
                         f"patches at layer {HOOK}. Directions are fit on the mean patch: shared over 14 prompts (toward 'nice'), or "
                         "the cool, pretty and ugly prompt alone (toward yes); random is a control at the dose of shared. Words: "
                         f"the teaser lists in the top 30 of each patch at layer {READ} (coral: severe words of the down end of the "
                         "aesthetic direction; teal: up end). Frame: the answer to 'Is this place cool?' in that condition (teal yes, coral no)."),
        "plot-latex-label": "fig:patch-steer", "plot-tags": ["F10.35", T.MODEL, "steering", "patch words"],
        "hook": HOOK, "read": READ, "alpha": ALPHA, "version": __version__}, indent=2) + "\n")


if __name__ == "__main__":
    main()
