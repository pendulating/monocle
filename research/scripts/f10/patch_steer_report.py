"""F10.35: the report of the patch steering run (`hop_steer.py patches`). CPU only.

Inputs: `data/f10/steer/gemma-4-31b/patches/{sample_id}.parquet` and `.npz`.

Outputs in `research/figures/f10/steer_patches/`:

| File | Content |
|------|---------|
| `margins.csv` | For each prompt, direction, layer and alpha: the mean change of the 8-form margin against alpha 0, its sd, the yes share (p_yes > p_no), and a paired t test over the images (Bonferroni over all the rows of a non-control direction) |
| `pole_shares.csv` | For each condition and read layer: the mean share of patches with an up word, and with a down word, of the teaser lists (`plot_patch_teaser.pole_words`), in the top 30 |
| `words.csv` | For the shared direction at +-ALPHA_SHOW, at each hook layer and read layer L40: the words whose share of patches rises most, and falls most, against alpha 0 |
| `margins.{pdf,png,json}` | The mean margin change against alpha, 1 panel for each prompt and hook layer |
| `teaser_rows.{pdf,png,json}` | The 5 teaser images at -ALPHA_SHOW, 0, +ALPHA_SHOW of the shared direction, hook L16, words at L40 |

The block rule of the teaser rows is the rule of `plot_patch_teaser.py` at 1 layer: the score of a word in a block
is its share of the block's 16 patches minus its share over the 591 unsteered images at the same layer (the F10.6
store). Pole words go first, greedily, at most MAX_REPEAT times in an image.

Use:
    python research/scripts/f10/patch_steer_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from research.scripts.f10 import plot_patch_teaser as T  # noqa: E402
from research.scripts.f10.hop_store import ALL_ROOT, open_store  # noqa: E402

MODEL = "gemma-4-31b"
SRC = ROOT / "data/f10/steer" / MODEL / "patches"
OUT = ROOT / "research/figures/f10/steer_patches"
ALPHA_SHOW = 96.0
READ_SHOW = 40
HOOK_SHOW = 16
N_WORDS = 25
COLOR = {"shared": "teal", "cool": "blue", "pretty": "amber", "ugly": "coral", "control": "periwinkle"}


def load() -> tuple[pd.DataFrame, dict]:
    rows, ids = [], {}
    for f in sorted(SRC.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        ids[f.stem] = {"ids": z["ids"], "conditions": list(z["conditions"]), "read": list(z["read_layers"])}
        rows.append(pd.read_parquet(f.with_suffix(".parquet")))
    if not rows:
        raise SystemExit(f"no output in {SRC}")
    return pd.concat(rows, ignore_index=True), ids


def margins(d: pd.DataFrame) -> pd.DataFrame:
    base = d[d.direction == "none"].set_index(["sample_id", "prompt"])["margin"]
    s = d[d.direction != "none"].copy()
    s["change"] = s["margin"] - base.reindex(pd.MultiIndex.from_frame(s[["sample_id", "prompt"]])).to_numpy()
    s["yes"] = s["p_yes"] > s["p_no"]
    g = s.groupby(["prompt", "direction", "layer", "alpha"])
    out = g.agg(n=("change", "size"), change=("change", "mean"), sd=("change", "std"), yes_share=("yes", "mean")).reset_index()
    out["t"] = g["change"].apply(lambda x: stats.ttest_1samp(x, 0).statistic).to_numpy()
    out["p"] = g["change"].apply(lambda x: stats.ttest_1samp(x, 0).pvalue).to_numpy()
    real = out.direction != "control"
    out["p_bonferroni"] = np.minimum(1.0, out["p"] * np.where(real, real.sum(), (~real).sum()))
    base_yes = (d[d.direction == "none"].assign(yes=lambda x: x.p_yes > x.p_no).groupby("prompt")["yes"].mean())
    out["base_yes_share"] = out["prompt"].map(base_yes)
    return out


def pole_shares(ids: dict, pole: dict) -> pd.DataFrame:
    up = np.array([t for t, (_, s) in pole.items() if s > 0])
    down = np.array([t for t, (_, s) in pole.items() if s < 0])
    rows = []
    for sid, z in ids.items():
        x = z["ids"]  # [conditions, patches, read layers, top]
        hu, hd = np.isin(x, up).any(-1).mean(1), np.isin(x, down).any(-1).mean(1)  # [conditions, read layers]
        for c, cond in enumerate(z["conditions"]):
            name, layer, alpha = cond.split("|")
            for j, l in enumerate(z["read"]):
                rows.append({"sample_id": sid, "direction": name, "layer": int(layer), "alpha": float(alpha), "read": int(l),
                             "up_share": float(hu[c, j]), "down_share": float(hd[c, j])})
    r = pd.DataFrame(rows)
    return r.groupby(["direction", "layer", "alpha", "read"])[["up_share", "down_share"]].mean().reset_index()


def word_shift(ids: dict, word_of: dict, layer: int, read: int) -> pd.DataFrame:
    """Give the change of the share of patches that hold each word, at +-ALPHA_SHOW of the shared direction."""
    rows = []
    for alpha in (-ALPHA_SHOW, ALPHA_SHOW):
        share = {}
        for key in ("none|-1|0.0", f"shared|{layer}|{alpha}"):
            c = Counter()
            n = 0
            for z in ids.values():
                x = z["ids"][z["conditions"].index(key), :, z["read"].index(read)]
                n += len(x)
                for p in x:
                    c.update({word_of[int(t)] for t in p if int(t) in word_of})
            share[key] = {w: k / n for w, k in c.items()}
        a, b = share["none|-1|0.0"], share[f"shared|{layer}|{alpha}"]
        delta = {w: b.get(w, 0) - a.get(w, 0) for w in set(a) | set(b)}
        order = sorted(delta, key=lambda w: (-delta[w], w))
        for end, words in (("rise", order[:N_WORDS]), ("fall", order[::-1][:N_WORDS])):
            rows += [{"hook_layer": layer, "read": read, "alpha": alpha, "end": end, "rank": i + 1, "word": w,
                      "delta": round(delta[w], 4), "base": round(a.get(w, 0), 4)} for i, w in enumerate(words)]
    return pd.DataFrame(rows)


def block_grid(x: np.ndarray, base: Counter, n_base: int, word_of: dict, pole_end: dict) -> list:
    """Give the 4 x 4 (word, end) grid of 1 image from its top ids [patches, top] at 1 layer."""
    k = T.GRID // T.POOL
    g = x.reshape(k, T.POOL, k, T.POOL, -1).transpose(0, 2, 1, 3, 4).reshape(k, k, T.POOL * T.POOL, -1)
    grid, cand = [[("", 0)] * k for _ in range(k)], []
    for i in range(k):
        for j in range(k):
            c = Counter(w for p in g[i, j] for w in {word_of[int(t)] for t in p if int(t) in word_of})
            score = {w: n / (T.POOL * T.POOL) - base[w] / n_base for w, n in c.items() if len(w) <= T.MAX_LEN}
            cand += [(s, w, pole_end[w], i, j) for w, s in score.items() if w in pole_end]
            if score:
                grid[i][j] = (max(score, key=lambda w: (score[w], w)), 0)
    used, done = Counter(), set()
    for s, w, end, i, j in sorted(cand, key=lambda c: (-c[0], c[1])):
        if (i, j) not in done and used[w] < T.MAX_REPEAT:
            grid[i][j] = (w, end)
            done.add((i, j))
            used[w] += 1
    return grid


def plot_margins(m: pd.DataFrame) -> None:
    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    prompts = ["place cool", "place pretty", "place ugly"]
    layers = sorted(m.layer.unique())
    fig, axes = plt.subplots(len(layers), len(prompts), figsize=(6.0, 3.6), sharex=True, sharey=True, squeeze=False)
    for r, layer in enumerate(layers):
        for c, prompt in enumerate(prompts):
            ax = axes[r][c]
            for name, g in m[(m.layer == layer) & (m.prompt == prompt)].groupby("direction"):
                g = pd.concat([g, pd.DataFrame({"alpha": [0.0], "change": [0.0]})]).sort_values("alpha")
                ax.plot(g.alpha, g.change, marker="o", ms=2.5, lw=1.0, color=S.PAL[COLOR[name]], label=name)
            ax.axhline(0, color="#999999", lw=.5)
            ax.set_title(f"'{prompt.split()[1]}', hook L{layer}", fontsize=7)
            if c == 0:
                ax.set_ylabel("margin change")
            if r == len(layers) - 1:
                ax.set_xlabel("dose (σ of the patch projection)")
    axes[0][-1].legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save(fig, OUT / "margins", "The answer when a direction is added to the patches",
         "Mean change of the 8-form answer margin of Gemma-4-31B against no addition, over the steered images, when a "
         "patch direction times the dose is added to all 256 image patches at 1 layer. Directions: fit on the mean patch "
         "(shared over 14 prompts, or the cool, pretty, ugly prompt alone), and a random control at the dose of the shared "
         "direction.", ["F10.35", MODEL, "steering", "patches"])


def plot_rows(ids: dict, d: pd.DataFrame, store, word_of: dict, pole: dict) -> None:
    from research.notebooks.cvpr import _style as S
    from PIL import Image

    S.apply_house_style()
    pole_end = {w: s for w, s in pole.values()}
    li = store.manifest["layers"].index(READ_SHOW)
    base, n_base = Counter(), 0
    for f in sorted((store.path / "patches").glob("ids-*.npy")):
        x = np.asarray(np.load(f, mmap_mode="r")[:, :, li])
        n_base += x.shape[0] * x.shape[1]
        for img in x:
            for p in img:
                base.update({word_of[int(t)] for t in p if int(t) in word_of})
    sel = d[d.teaser & (d.direction == "none") & (d.prompt == "place cool")].drop_duplicates("sample_id")
    # The run's own teaser images (v1.3.0 of the teaser), in the panel order of the teaser. The teaser CSV can change.
    rank = {b: i for i, (b, _) in enumerate(T.PANELS)}
    order = sel.assign(o=sel.borough.map(rank)).sort_values("o").sample_id.tolist()
    keys = [(f"shared|{HOOK_SHOW}|{-ALPHA_SHOW}", f"−{ALPHA_SHOW:.0f}σ"), ("none|-1|0.0", "none"), (f"shared|{HOOK_SHOW}|{ALPHA_SHOW}", f"+{ALPHA_SHOW:.0f}σ")]
    ink = {1: S.PAL["teal"], -1: S.PAL["coral"], 0: S.EDGE}
    k = T.GRID // T.POOL
    fig, axes = plt.subplots(len(keys), len(order), figsize=(T.WIDTH, 1.25 * len(keys) + .3), gridspec_kw={"wspace": .04, "hspace": .22}, squeeze=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    images = pd.read_parquet(ROOT / "data/f10/hop_depth/images.parquet").set_index("sample_id")
    for c, sid in enumerate(order):
        img = np.asarray(Image.open(images.loc[sid, "image_path"]).convert("RGB"))
        z = ids[sid]
        for r, (key, label) in enumerate(keys):
            ax = axes[r][c]
            ax.imshow(img, extent=(0, k, k, 0))
            ax.imshow(np.ones((1, 1, 3)), extent=(0, k, k, 0), alpha=0.22)
            grid = block_grid(z["ids"][z["conditions"].index(key), :, z["read"].index(READ_SHOW)], base, n_base, word_of, pole_end)
            for i in range(k):
                for j in range(k):
                    w, end = grid[i][j]
                    txt = ax.text(j + .5, i + .5, w, ha="center", va="center", fontsize=T.WORD_PT, color=ink[end],
                                  fontweight="bold" if end else "normal", path_effects=[pe.withStroke(linewidth=.9, foreground="white")])
                    box = ax.transData.transform([(j, 0), (j + 1, 0)])
                    room, used = .92 * (box[1, 0] - box[0, 0]), txt.get_window_extent(renderer).width
                    if used > room:
                        txt.set_fontsize(max(T.MIN_PT, T.WORD_PT * room / used))
            ax.set_xlim(0, k)
            ax.set_ylim(k, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            m = d[(d.sample_id == sid) & (d.prompt == "place cool")]
            name, layer, alpha = key.split("|")
            mm = m[(m.direction == name) & (m.alpha == float(alpha))]["margin"].iloc[0]
            ax.text(.5, -.04, f"cool margin {mm:+.1f}", transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
            if c == 0:
                ax.text(-.06, .5, label, transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=6.5)
            if r == 0:
                ax.set_title(images.loc[sid, "borough"], fontsize=6.5, pad=2)
    save(fig, OUT / "teaser_rows", "Teaser images with the shared direction added",
         f"The 5 teaser images with the shared patch direction added to all patches at L{HOOK_SHOW} at -{ALPHA_SHOW:.0f}, 0 "
         f"and +{ALPHA_SHOW:.0f} sigma. Words: the teaser block rule at L{READ_SHOW} (teal: up end of the aesthetic "
         "direction, coral: down end, dark: other). Under each image: the 8-form margin to 'Is this place cool?'.",
         ["F10.35", MODEL, "steering", "patch words"])


def save(fig, path: Path, title: str, caption: str, tags: list) -> None:
    for ext in ("png", "pdf"):
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {"dpi": 300}
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.01, facecolor="white", **kw)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({"plot-title": title, "plot-caption": caption, "plot-tags": tags, "version": __version__}, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d, ids = load()
    print(f"[f10.35] {d.sample_id.nunique()} images", flush=True)
    store = open_store(MODEL, ALL_ROOT)
    vocab = pd.read_parquet(store.path / "vocab.parquet")
    pole = T.pole_words(vocab)
    word_of = {int(t): w for t, w in zip(vocab.token_id.to_numpy(), vocab.token.str.strip().to_numpy()) if T.WORD.fullmatch(w)}
    m = margins(d)
    m.to_csv(OUT / "margins.csv", index=False)
    p = pole_shares(ids, pole)
    p.to_csv(OUT / "pole_shares.csv", index=False)
    w = pd.concat([word_shift(ids, word_of, layer, READ_SHOW) for layer in sorted(d.layer[d.layer > 0].unique())])
    w.to_csv(OUT / "words.csv", index=False)
    plot_margins(m)
    plot_rows(ids, d, store, word_of, pole)
    pd.set_option("display.width", 200)
    show = m[m.alpha.abs() == ALPHA_SHOW].pivot_table(index=["prompt", "layer", "direction"], columns="alpha", values=["change", "yes_share"]).round(2)
    print(show.to_string())
    print(p[(p.read == READ_SHOW) & (p.alpha.abs().isin([0, ALPHA_SHOW]))].round(3).to_string(index=False))
    for (layer, alpha, end), g in w.groupby(["hook_layer", "alpha", "end"]):
        print(f"L{layer} {alpha:+.0f} {end}: " + ", ".join(g.sort_values("rank").word.head(15)))


if __name__ == "__main__":
    main()
