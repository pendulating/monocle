"""F10.33: the teaser figure. J++ Lens words on the image patches of Gemma-4-31B, 1 image per borough.

The form of the old R-lens figure (`notebooks/facct/_rq.py`, `plot_patch_words`), in 1 row of 5.
The image comes before the question, thus the patch words do not depend on the question.

Pole words. The shared aesthetic direction of Gemma-4-31B at `place` (F10.27,
`figures/f10/shared/gemma-4-31b_place/direction.npz`), z-scored at each layer and then averaged
over the layers (the F10.28 rule). The N_POLE lowercase words of 4 or more letters at each end
are the up words and the down words. The band counts of these words give the r with the margin.

Severe words. The figure shows the down end with the severe words of `patch_severe.py`: the
social words (sewage, criminal, slaughter, stench, gangs) within rank RANK_MAX of the down end,
not the object words (truck, forklift) of the top N_POLE. Author decision (2026-11-03).

One layer for each end (F10.33b). The up words peak at layer UP_LAYER (L32: gardens, ornament,
pastel). The severe words reach the most images at layer DOWN_LAYER (L16: sewage, slaughter,
stench, criminal; 448 of 591 images). Thus the teal words come from the top 30 at L32, and the
coral words from the top 30 at L16.

Selection. PICKS gives one image for each borough, by eye from the contact sheets of
`patch_severe.py` and the candidate list of the 4 adjectives: an urban street scene with the
answer no to cool, nice and pretty (MUST_BE_NO), on which the severe words fire. The images csv
records the margin of each adjective of ADJECTIVES. The margin test (r of the band counts with the answer margin over
the 591 images) stays as in F10.33.

Words of a block. The patches form a 16 x 16 grid (row-major), pooled into 4 x 4 blocks of 4 x 4
patches. The score of a word in a block is the number of its 16 patches that hold the word at the
layer of its end. A block that holds a severe word shows the severe word with the top score; the
up words fill the other blocks by score. A word shows at most MAX_REPEAT times in an image, only
a word of MAX_LEN letters or fewer shows, and a fragment (FRAGMENTS) does not show. A block with
no word of either end stays empty. The frame of each image carries the answer (teal yes, coral
no), as in the win block figures. The panel title is the borough alone.

Use:
    python scripts/f10/plot_patch_teaser.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

__version__ = "2.3.0"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from hop_hierarchy import answer_store  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

MODEL, N_LAYERS = "gemma-4-31b", 60
DEPTH = (0.25, 0.75)
N_POLE = 300
MAX_REPEAT = 2
UP_LAYER, DOWN_LAYER = 32, 16
# Author decision (2026-11-03): each panel is an urban street scene with the answer no to cool,
# nice and pretty, on which the severe words fire. Picks by eye from the contact sheets of
# `patch_severe.py` and of the 4-adjective candidate list.
PICKS = {
    "Manhattan": "W0E65GB6",  # Midtown-Times Square: an office tower and a coach; trash
    "Bronx": "W0EM9QBY",  # Belmont: apartment towers over a shop street (author: an urban scene, not a warehouse); slaughter, sewage
    "Queens": "W0E4I5X2",  # Astoria (East)-Woodside (North): a commercial street; sludge, sewage, punk; ugly yes
    "Brooklyn": "W0E8V32I",  # Prospect Lefferts Gardens-Wingate: a laundromat corner; sewage, dumping
    "Staten Island": "W0EL0NBS",  # St. George-New Brighton: garage doors and a walker; sewage, slaughter; ugly yes
}
ADJECTIVES = ("cool", "nice", "pretty", "ugly")  # the answers that the images csv records
MUST_BE_NO = ("cool", "nice", "pretty")  # each pick has the answer no to these
MAX_LEN = 11  # a longer word does not fit a block at a size that prints; it still counts as a hit
FRAGMENTS = {"scav", "crimin", "kriminal"}  # severe fragments that count as hits but do not show
GRID, POOL = 16, 4
PANELS = (("Manhattan", -1), ("Brooklyn", -1), ("Queens", -1), ("Bronx", -1), ("Staten Island", -1))
YES = ["lp_Yes", "lp_yes", "lp_ Yes", "lp_ yes"]
NO = ["lp_No", "lp_no", "lp_ No", "lp_ no"]
WORD = re.compile(r"[a-z]{4,}")
DIRECTION = ROOT / "figures/f10/shared/gemma-4-31b_place/direction.npz"
OUT = ROOT / "figures/f10/mechanism/mech_patch_teaser"
WIDTH = 430 / 72.27  # the text width of the acmart manuscript format, thus the figure prints at 100%
WORD_PT, MIN_PT = 5.0, 3.0
FRAME_LW = 1.8  # the frame width of the win block photographs


def pole_words(vocab: pd.DataFrame) -> dict[int, tuple[str, int]]:
    """Give {token id: (word, +1 or -1)} for the N_POLE words at each end of the direction."""
    z = np.load(DIRECTION)
    d = z["direction"].astype(np.float64)
    if len(z["vocab"]) != len(vocab) or not (z["vocab"].astype(str) == vocab.token.str.strip().to_numpy()).all():
        raise ValueError("the direction and the store do not have the same vocabulary")
    score = ((d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)).mean(0)
    words = vocab.token.str.strip().to_numpy()
    ok = np.array([bool(WORD.fullmatch(w)) for w in words])
    order = np.argsort(-score)
    up = [i for i in order if ok[i]][:N_POLE]
    down = [i for i in order[::-1] if ok[i]][:N_POLE]
    tid = vocab.token_id.to_numpy()
    return {int(tid[i]): (words[i], 1) for i in up} | {int(tid[i]): (words[i], -1) for i in down}


def severe_words(vocab: pd.DataFrame) -> dict[int, tuple[str, int]]:
    """Give {token id: (word, -1)} for the severe words within rank RANK_MAX of the down end."""
    from patch_severe import RANK_MAX, SEVERE  # here, because patch_severe imports this module

    z = np.load(DIRECTION)
    d = z["direction"].astype(np.float64)
    score = ((d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)).mean(0)
    words = vocab.token.str.strip().to_numpy()
    ok = np.array([bool(WORD.fullmatch(w)) for w in words])
    order = [i for i in np.argsort(score) if ok[i]][:RANK_MAX]
    tid = vocab.token_id.to_numpy()
    return {int(tid[i]): (words[i], -1) for i in order if words[i] in SEVERE}


def load_ids(store) -> tuple[np.ndarray, list[int]]:
    """Give the top-30 token ids of all saved layers [images, patches, layers, 30], and the layers."""
    layers = store.manifest["layers"]
    ids = np.concatenate([np.asarray(np.load(f, mmap_mode="r")) for f in sorted((store.path / "patches").glob("ids-*.npy"))])
    if ids.shape[:2] != (len(store.images), GRID * GRID) or ids.shape[2] != len(layers):
        raise ValueError(f"patch ids have shape {ids.shape}")
    return ids, list(layers)


def band(layers: list[int]) -> list[int]:
    """Give the indexes of the layers of the band of relative depth DEPTH."""
    return [k for k, l in enumerate(layers) if DEPTH[0] <= l / N_LAYERS < DEPTH[1]]


def hits(ids: np.ndarray, pole: dict) -> tuple[np.ndarray, np.ndarray]:
    """Give the (patch, layer) hit masks of the up and the down words [images, patches, layers]."""
    up_ids = np.array([t for t, (_, s) in pole.items() if s > 0])
    down_ids = np.array([t for t, (_, s) in pole.items() if s < 0])
    return np.isin(ids, up_ids).any(-1), np.isin(ids, down_ids).any(-1)


def margins(store) -> pd.DataFrame:
    """Give the answer margin (log P(yes) - log P(no)) of each image for each adjective of ADJECTIVES."""
    im = store.images[["image_idx", "sample_id", "image_path", "nta_name", "borough"]].reset_index(drop=True)
    for adj in ADJECTIVES:
        a = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(MODEL, adj, "place") / "answers.parquet")
        a = a.assign(**{f"m_{adj}": logsumexp(a[YES], axis=1) - logsumexp(a[NO], axis=1)})
        im = im.merge(a[["image_idx", f"m_{adj}"]], on="image_idx", validate="1:1")
    return im.assign(margin=im.m_cool)


def select(store, up: np.ndarray, down: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Give the image of each panel, and the r of the patch counts with the answer margin."""
    d = margins(store).assign(n_up=up.any(-1).sum(1), n_down=down.any(-1).sum(1))
    d = d.assign(row=np.arange(len(d)))
    test = {}
    for col in ("n_up", "n_down"):
        r = stats.pearsonr(d[col], d["margin"])
        test[col] = {"r": round(float(r.statistic), 3), "p": float(r.pvalue), "n": len(d)}
    rows = []
    for borough, sign in PANELS:
        g = d[(d.borough == borough) & (np.sign(d.margin) == sign)]
        g = g.assign(score=sign * (g.n_up - g.n_down)).sort_values(["score", "sample_id"], ascending=[False, True])
        g = g.assign(rank=np.arange(1, len(g) + 1))
        pick = g[g.sample_id == PICKS[borough]]
        if len(pick) != 1:
            raise ValueError(f"{borough}: the pick {PICKS.get(borough)} is not an image with the answer {sign:+d}")
        bad = [adj for adj in MUST_BE_NO if float(pick[f"m_{adj}"].iloc[0]) >= 0]
        if bad:
            raise ValueError(f"{borough}: the pick {PICKS[borough]} does not answer no to {bad}")
        rows.append(pick.iloc[0].to_dict() | {"direction": "yes" if sign > 0 else "no", "of": len(g)})
    return pd.DataFrame(rows), test


def block_words(ids: np.ndarray, layers: list[int], up: dict, severe: dict, rows) -> dict[int, list]:
    """Give the 4 x 4 grid of (word, end) of each image row; end is +1, -1, or 0 for an empty block."""
    ends = ((up, layers.index(UP_LAYER)), (severe, layers.index(DOWN_LAYER)))
    k = GRID // POOL
    out = {}
    for r in rows:
        grid = [[("", 0)] * k for _ in range(k)]
        cand = []  # (score, word, end, i, j): the number of the 16 patches of the block that hold the word
        for words, layer in ends:
            g = ids[r, :, layer].reshape(k, POOL, k, POOL, -1).transpose(0, 2, 1, 3, 4).reshape(k, k, POOL * POOL, -1)
            for i in range(k):
                for j in range(k):
                    n = Counter()
                    for t in np.unique(g[i, j]):
                        if int(t) in words:
                            n[int(t)] = int((g[i, j] == t).any(-1).sum())
                    cand += [(c, words[t][0], words[t][1], i, j) for t, c in n.items()
                             if len(words[t][0]) <= MAX_LEN and words[t][0] not in FRAGMENTS]
        used, done = Counter(), set()
        # A block that holds a severe word shows it; the up words fill the other blocks.
        for s, w, end, i, j in sorted(cand, key=lambda c: (c[2], -c[0], c[1])):
            if (i, j) not in done and used[w] < MAX_REPEAT:
                grid[i][j] = (w, end)
                done.add((i, j))
                used[w] += 1
        out[r] = grid
    return out


def figure(sel: pd.DataFrame, grids: dict) -> None:
    import _style as S
    from PIL import Image

    S.apply_house_style()
    ink = {1: S.PAL["teal"], -1: S.PAL["coral"], 0: S.EDGE}
    k = GRID // POOL
    fig, axes = plt.subplots(1, len(sel), figsize=(WIDTH, 1.30), gridspec_kw={"wspace": 0.04})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, r in zip(axes, sel.itertuples()):
        img = Image.open(r.image_path).convert("RGB")
        ax.imshow(np.asarray(img), extent=(0, k, k, 0))
        ax.imshow(np.ones((1, 1, 3)), extent=(0, k, k, 0), alpha=0.22)  # a light veil, so the words read
        for t in range(1, k):
            ax.axhline(t, color="white", lw=0.4, alpha=0.6)
            ax.axvline(t, color="white", lw=0.4, alpha=0.6)
        for i in range(k):
            for j in range(k):
                w, end = grids[r.row][i][j]
                txt = ax.text(j + 0.5, i + 0.5, w, ha="center", va="center", fontsize=WORD_PT, color=ink[end],
                              fontweight="bold" if end else "normal",
                              path_effects=[pe.withStroke(linewidth=0.9, foreground="white")])
                # Shrink a long word to fit its block: measure it, then scale the font.
                box = ax.transData.transform([(j, 0), (j + 1, 0)])
                room = 0.92 * (box[1, 0] - box[0, 0])
                used = txt.get_window_extent(renderer).width
                if used > room:
                    txt.set_fontsize(max(MIN_PT, WORD_PT * room / used))
        ax.set_xlim(0, k)
        ax.set_ylim(k, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        # The frame carries the answer, as in the win block figures (`_extractions._photo_row`):
        # teal for yes, coral for no.
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_edgecolor(ink[1 if r.direction == "yes" else -1])
            side.set_linewidth(FRAME_LW)
        ax.text(0.5, 1.03, r.borough, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    for ext in ("png", "pdf"):
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {"dpi": 300}
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.01, facecolor="white", **kw)
    plt.close(fig)


def main() -> None:
    from patch_severe import RANK_MAX

    OUT.parent.mkdir(parents=True, exist_ok=True)
    store = open_store(MODEL, ALL_ROOT)
    vocab = pd.read_parquet(store.path / "vocab.parquet")
    pole = pole_words(vocab)
    severe = severe_words(vocab)
    up_words = {t: ws for t, ws in pole.items() if ws[1] > 0}
    ids, layers = load_ids(store)
    up, down = hits(ids[:, :, band(layers)], pole)
    sel, test = select(store, up, down)
    ku, kd = layers.index(UP_LAYER), layers.index(DOWN_LAYER)
    sel = sel.assign(n_up_layer=[int(np.isin(ids[r, :, ku], list(up_words)).any(-1).sum()) for r in sel.row],
                     n_severe_layer=[int(np.isin(ids[r, :, kd], list(severe)).any(-1).sum()) for r in sel.row])
    grids = block_words(ids, layers, up_words, severe, sel.row.to_numpy())
    rows = [{"borough": r.borough, "sample_id": r.sample_id, "block_row": i, "block_col": j,
             "word": grids[r.row][i][j][0], "end": grids[r.row][i][j][1]}
            for r in sel.itertuples() for i in range(GRID // POOL) for j in range(GRID // POOL)]
    pd.DataFrame(rows).to_csv(OUT.with_name("patch_teaser_words.csv"), index=False)
    sel.drop(columns=["row"]).to_csv(OUT.with_name("patch_teaser_images.csv"), index=False)
    figure(sel, grids)
    cols = ["borough", "sample_id", "nta_name", "direction"] + [f"m_{a}" for a in ADJECTIVES] + ["n_up", "n_down", "n_up_layer", "n_severe_layer"]
    print(sel[cols].round(2).to_string(index=False))
    print(json.dumps(test))
    OUT.with_suffix(".json").write_text(json.dumps({
        "plot-title": "What the image patches hold",
        "plot-caption": (f"One street image for each borough, with the answer of {MODEL} to 'Is this place cool?'. Each "
                         "image is an urban street scene with the answer no to cool, nice and pretty. On each 4x4 block of patches: a word of the "
                         f"shared aesthetic direction in the top 30 of its patches. Teal: a word of the yes end at layer {UP_LAYER}. "
                         f"Coral: a severe word of the no end (within rank {RANK_MAX}) at layer {DOWN_LAYER}. "
                         "The image comes before the question, so the words do not depend on it."),
        "plot-latex-label": "fig:teaser", "plot-tags": ["F10.33", MODEL, "J++ Lens", "patch words", "shared direction", "severe words"],
        "margin_test": test, "n_pole": N_POLE, "rank_max": RANK_MAX, "up_layer": UP_LAYER, "down_layer": DOWN_LAYER,
        "version": __version__}, indent=2) + "\n")


if __name__ == "__main__":
    main()
