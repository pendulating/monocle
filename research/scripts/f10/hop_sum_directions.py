#!/usr/bin/env python
"""F10.34: the element-wise sums and differences of prompt directions, and the words at their ends. CPU only.

Each prompt has a direction in the lens-word space: the unit vector of the r, over the 197 NTAs, of each word's
NTA mean readout at the noun token with the prompt's NTA answer margin (`hop_shared.py`, variant `main`, which at
`place` is the centred readout with no removal). The shared direction of F10.27 is the element-wise mean of the
`cool`, `nice`, `fancy` and `pretty` directions. This script adds the `urban` direction to the `nice`, `cool` and
`ordinary` directions, one at a time and all together, and lists the words at the 2 ends of each sum with the
F10.28 rule: z-score at each layer, mean over the layers, top 20 at each end.

| Output | Content |
|--------|---------|
| `words.csv` | model, direction, end, rank, word, z, and the rank of the word in each single direction |
| `cosine.csv` | model, layer, a, b, cosine of the single and summed directions, and with the F10.27 shared direction |
| `overlap.csv` | model, direction, end, the number of the top 20 words shared with each single direction |
| `pairs.csv` | model, layer, op (`sum`, `difference`), a, b, c, cosine of a + b or a - b with the single direction c, for the 21 pairs of the 7 adjectives |
| `pair_words.csv` | the 10 words at each end of each pair sum and difference (layer mean) |
| `pair_cosines.{pdf,png,json}` | the 2 heatmaps for the appendix: the layer-mean cosine of each pair sum (left) and difference (right) with each single direction, 3 models |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_sum_directions.py [--k 10] [--plot-only]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from research.scripts.f10 import hop_shared as hs  # noqa: E402
from research.scripts.f10.hop_prompts import MODELS, store_name  # noqa: E402

__version__ = "1.1.0"
SINGLE = ("urban", "nice", "cool", "ordinary")
ADJECTIVES = ("cool", "nice", "pretty", "fancy", "ugly", "ordinary", "urban")
MODEL_NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
SUMS = {"nice+urban": ("nice", "urban"), "cool+urban": ("cool", "urban"), "ordinary+urban": ("ordinary", "urban"),
        "nice+cool+ordinary+urban": SINGLE}
N_WORDS = 20
OUT = ROOT / "research/figures/f10/sum"


def directions(model: str, root: Path, adjectives: tuple = SINGLE) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    """Give the unit direction of each single prompt at each layer [L, V], the vocabulary and the layers."""
    hs.DATA = root
    data = {a: hs.load(model, a, "place", "place") for a in adjectives}
    order = pd.factorize(data["cool"][1]["nta2020"])[1].tolist()
    for a, (_, fr, _) in data.items():
        if pd.factorize(fr["nta2020"])[1].tolist() != order:
            raise ValueError(f"NTA order differs in {a}")
    layers = data["cool"][2]["layers"]
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{store_name(model, 'cool', 'place')}/vocab.parquet")["token"]
    vocab = vocab.str.strip().to_numpy().astype(str)
    margins = {a: fr.groupby("nta2020")["margin"].mean().reindex(order).to_numpy(float) for a, (_, fr, _) in data.items()}
    g = {a: np.zeros((len(layers), vocab.size), np.float32) for a in adjectives}
    for b in range(len(layers)):
        for a in adjectives:
            m = data[a][0][:, b, :].astype(np.float64)
            g[a][b] = hs.unit(hs.corr_cols(m - m.mean(0, keepdims=True), margins[a]))
    return g, vocab, layers


def pairs(root: Path, out: Path) -> pd.DataFrame:
    """The cosine of each pair sum and difference with each single direction, by layer; and the words of each."""
    rows, words = [], []
    for model in MODELS:
        g, vocab, layers = directions(model, root, ADJECTIVES)
        for a, c in itertools.combinations(ADJECTIVES, 2):
            for op, v in (("sum", g[a] + g[c]), ("difference", g[a] - g[c])):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                for b, layer in enumerate(layers):
                    for s in ADJECTIVES:
                        rows.append({"model": model, "layer": layer, "op": op, "a": a, "b": c, "c": s, "cosine": float(v[b] @ g[s][b])})
                lm = layer_mean(v.astype(np.float64))
                for end, idx in (("up", np.argsort(-lm)[:10]), ("down", np.argsort(lm)[:10])):
                    words.append({"model": model, "op": op, "a": a, "b": c, "end": end, "words": " ".join(vocab[idx])})
        print(f"[f10.34] pairs {model}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(words).to_csv(out / "pair_words.csv", index=False)
    return df


def plot_pairs(out: Path) -> None:
    """2 heatmaps: the layer-mean cosine of each pair sum (left) and difference (right) with each single direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from research.notebooks.cvpr import _figmeta as FM
    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    df = pd.read_csv(out / "pairs.csv")
    mean = df.groupby(["model", "op", "a", "b", "c"]).cosine.mean().reset_index()
    pair_list = list(itertools.combinations(ADJECTIVES, 2))
    cols = [(m, s) for m in MODELS for s in ADJECTIVES]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.15))
    for ax, (op, sign) in zip(axes, (("sum", "+"), ("difference", "−"))):
        M = np.full((len(pair_list), len(cols)), np.nan)
        member = np.zeros_like(M, dtype=bool)
        sub = mean[mean.op == op].set_index(["model", "a", "b", "c"]).cosine
        for i, (a, b) in enumerate(pair_list):
            for j, (m, s) in enumerate(cols):
                M[i, j] = sub[(m, a, b, s)]
                member[i, j] = s in (a, b)
        shown = np.where(member, np.nan, M)
        ax.imshow(shown, cmap=S.CMAP_DIV, vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
        ax.imshow(np.where(member, 0.5, np.nan), cmap="Greys", vmin=0, vmax=1, aspect="equal", interpolation="nearest", alpha=0.35)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not member[i, j] and abs(M[i, j]) >= 0.5:
                    ax.text(j, i, f"{M[i, j]:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=3.6, color=S.text_on(S.CMAP_DIV((M[i, j] + 1) / 2)))
        ax.set_yticks(range(len(pair_list)))
        ax.set_yticklabels([f"{a} {sign} {b}" for a, b in pair_list], fontsize=5.2)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([s for _, s in cols], fontsize=5.0, rotation=90)
        for k, m in enumerate(MODELS):
            ax.text(k * len(ADJECTIVES) + (len(ADJECTIVES) - 1) / 2, -1.0, MODEL_NAMES[m], ha="center", va="bottom", fontsize=6.0, color=S.EDGE)
            if k:
                ax.axvline(k * len(ADJECTIVES) - 0.5, color=S.EDGE, lw=0.6)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.set_title("a + b" if op == "sum" else "a − b", fontsize=7.5, pad=12)
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1, 1), cmap=S.CMAP_DIV)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.09, aspect=40, shrink=0.5)
    cb.set_label("cosine of the combined direction with the single direction (mean over layers)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    caption = ("The cosine of the sum (left) and the difference (right) of 2 prompt directions with each single prompt "
               "direction, at the noun token, as a mean over the saved layers, for 3 models. A prompt direction is the "
               "unit vector of the r over the 197 NTAs of each lens word's NTA mean readout with the prompt's answer margin. "
               "Rows: the 21 pairs of the 7 adjectives; columns: the 7 single directions of each model. A grey cell is a member "
               "of its pair. Values of .5 or more are printed.")
    FM.save_figure(fig, out / "pair_cosines.pdf", title="Sums and differences of the prompt directions", caption=caption,
                   label="fig:pair-cosines", tags=["f10", "shared-direction", "pairs"],
                   provenance={"script": "research/scripts/f10/hop_sum_directions.py", "version": __version__})
    FM.save_figure(fig, out / "pair_cosines.png", dpi=300, title="Sums and differences of the prompt directions", caption=caption,
                   label="fig:pair-cosines", tags=["f10", "shared-direction", "pairs"],
                   provenance={"script": "research/scripts/f10/hop_sum_directions.py", "version": __version__})
    plt.close(fig)
    print(f"[f10.34] {out / 'pair_cosines.pdf'}", flush=True)


def layer_mean(w: np.ndarray) -> np.ndarray:
    z = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    return z.mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="3 = the F10.26 summaries (591 images); 10 = the F10.30 summaries")
    ap.add_argument("--plot-only", action="store_true", help="draw the heatmaps from pairs.csv")
    args = ap.parse_args()
    root = ROOT / ("data/f10/prompts" if args.k == 3 else f"data/f10/prompts_k{args.k}")
    out = OUT if args.k == 3 else OUT / f"k{args.k}"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_pairs(out)
        return
    pairs(root, out)
    plot_pairs(out)
    words, cos, overlap = [], [], []
    for model in MODELS:
        g, vocab, layers = directions(model, root)
        for name, parts in SUMS.items():
            s = np.sum([g[a] for a in parts], 0)
            g[name] = (s / np.linalg.norm(s, axis=1, keepdims=True)).astype(np.float32)
        shared = np.load(ROOT / f"research/figures/f10/shared/{model}_place/direction.npz")
        g["shared (F10.27)"] = shared["direction"].astype(np.float32)
        names = list(g)
        for b, layer in enumerate(layers):
            for a, c in itertools.combinations(names, 2):
                cos.append({"model": model, "layer": layer, "a": a, "b": c, "cosine": float(g[a][b] @ g[c][b])})
        lm = {a: layer_mean(g[a].astype(np.float64)) for a in names}
        rank = {a: {"up": pd.Series(np.argsort(np.argsort(-lm[a])) + 1, index=vocab),
                    "down": pd.Series(np.argsort(np.argsort(lm[a])) + 1, index=vocab)} for a in SINGLE}
        for a in names:
            for end, idx in (("up", np.argsort(-lm[a])[:N_WORDS]), ("down", np.argsort(lm[a])[:N_WORDS])):
                top = set(vocab[idx])
                for k, i in enumerate(idx):
                    words.append({"model": model, "direction": a, "end": end, "rank": k + 1, "word": vocab[i], "z": float(lm[a][i]),
                                  **{f"rank_{s}": int(rank[s][end][vocab[i]]) for s in SINGLE}})
                overlap.append({"model": model, "direction": a, "end": end,
                                **{f"shared_with_{s}": len(top & set(vocab[np.argsort(-lm[s] if end == "up" else lm[s])[:N_WORDS]])) for s in SINGLE}})
        print(f"[f10.34] {model}: {len(layers)} layers, {vocab.size} words", flush=True)
    w, c, o = pd.DataFrame(words), pd.DataFrame(cos), pd.DataFrame(overlap)
    w.to_csv(out / "words.csv", index=False)
    c.to_csv(out / "cosine.csv", index=False)
    o.to_csv(out / "overlap.csv", index=False)
    pd.set_option("display.width", 250)
    print(c.groupby(["model", "a", "b"]).cosine.mean().unstack("model").round(2).to_string())
    print(o.to_string())
    for (m, d, e), sub in w.groupby(["model", "direction", "end"], sort=False):
        print(m, d, e, ", ".join(sub.word))


if __name__ == "__main__":
    main()
