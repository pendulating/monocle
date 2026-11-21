"""F10.33b: how the pole words on the image patches change with the layer (Gemma-4-31B, 591 images).

For each saved layer: the share of patches that hold an up or a down word of the shared direction
in their top 30, the Pearson r of the per-image counts with the `cool` answer margin, the Jaccard
overlap of the hit patches with the next saved layer, and the pole words that the patches hold
most. Also the counts of the 5 teaser images at each layer.

Outputs: `figures/f10/mechanism/patch_layers.csv`, `patch_layer_words.csv`, `patch_layer_teaser.csv`.

Use:
    python scripts/f10/patch_layers.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10")]
import plot_patch_teaser as T  # noqa: E402
from hop_hierarchy import answer_store  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "figures/f10/mechanism"


def main() -> None:
    store = open_store(T.MODEL, ALL_ROOT)
    vocab = pd.read_parquet(store.path / "vocab.parquet")
    pole = T.pole_words(vocab)
    layers = store.manifest["layers"]
    ids = np.concatenate([np.asarray(np.load(f, mmap_mode="r")) for f in sorted((store.path / "patches").glob("ids-*.npy"))])
    up, down = T.hits(ids, pole)  # [images, patches, layers]
    a = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(T.MODEL, "cool", "place") / "answers.parquet")
    a = a.assign(margin=logsumexp(a[T.YES], axis=1) - logsumexp(a[T.NO], axis=1)).set_index("image_idx").margin
    margin = a.reindex(store.images.image_idx.to_numpy()).to_numpy()
    word_of = {t: w for t, (w, _) in pole.items()}
    band = [k for k, l in enumerate(layers) if T.DEPTH[0] <= l / T.N_LAYERS < T.DEPTH[1]]
    rows, words = [], []
    for k, layer in enumerate(layers):
        n_up, n_down = up[:, :, k].sum(1), down[:, :, k].sum(1)
        row = {"layer": layer, "depth": round(layer / T.N_LAYERS, 2), "in_band": k in band,
               "share_up": float(up[:, :, k].mean()), "share_down": float(down[:, :, k].mean()),
               "mean_n_up": float(n_up.mean()), "mean_n_down": float(n_down.mean()),
               "r_up": float(stats.pearsonr(n_up, margin).statistic) if n_up.std() > 0 else np.nan,
               "r_down": float(stats.pearsonr(n_down, margin).statistic) if n_down.std() > 0 else np.nan,
               "r_signed": float(stats.pearsonr(n_up - n_down, margin).statistic) if (n_up - n_down).std() > 0 else np.nan}
        if k + 1 < len(layers):
            for name, h in (("up", up), ("down", down)):
                a_, b_ = h[:, :, k], h[:, :, k + 1]
                inter, union = (a_ & b_).sum(), (a_ | b_).sum()
                row[f"jaccard_next_{name}"] = float(inter / union) if union else np.nan
        rows.append(row)
        for name, mask in (("up", np.isin(ids[:, :, k], [t for t, (_, s) in pole.items() if s > 0])),
                           ("down", np.isin(ids[:, :, k], [t for t, (_, s) in pole.items() if s < 0]))):
            c = Counter(word_of[int(t)] for t in ids[:, :, k][mask].ravel())
            words.append({"layer": layer, "end": name, "n_hits": int(mask.sum()),
                          "top": ", ".join(f"{w} ({n})" for w, n in c.most_common(8))})
    # the band count, as in the teaser
    nb_up, nb_down = up[:, :, band].any(-1).sum(1), down[:, :, band].any(-1).sum(1)
    rows.append({"layer": "band L16-44", "depth": "", "in_band": True, "share_up": float(up[:, :, band].any(-1).mean()),
                 "share_down": float(down[:, :, band].any(-1).mean()), "mean_n_up": float(nb_up.mean()), "mean_n_down": float(nb_down.mean()),
                 "r_up": float(stats.pearsonr(nb_up, margin).statistic), "r_down": float(stats.pearsonr(nb_down, margin).statistic),
                 "r_signed": float(stats.pearsonr(nb_up - nb_down, margin).statistic)})
    all_k = list(range(len(layers)))
    na_up, na_down = up.any(-1).sum(1), down.any(-1).sum(1)
    rows.append({"layer": "all L0-56", "depth": "", "in_band": False, "share_up": float(up.any(-1).mean()), "share_down": float(down.any(-1).mean()),
                 "mean_n_up": float(na_up.mean()), "mean_n_down": float(na_down.mean()),
                 "r_up": float(stats.pearsonr(na_up, margin).statistic), "r_down": float(stats.pearsonr(na_down, margin).statistic),
                 "r_signed": float(stats.pearsonr(na_up - na_down, margin).statistic)})
    d = pd.DataFrame(rows); d.to_csv(OUT / "patch_layers.csv", index=False)
    w = pd.DataFrame(words); w.to_csv(OUT / "patch_layer_words.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    print(d.round(3).to_string(index=False))
    print(w.to_string(index=False))
    # the 5 teaser images
    im = store.images.reset_index(drop=True)
    t = []
    for borough, sign in T.PANELS:
        sid = T.PICKS.get(borough)
        if sid is None:
            sel = pd.read_csv(OUT / "patch_teaser_images.csv"); sid = sel[sel.borough == borough].sample_id.iloc[0]
        r = int(im.index[im.sample_id == sid][0])
        for k, layer in enumerate(layers):
            t.append({"borough": borough, "sample_id": sid, "layer": layer, "n_up": int(up[r, :, k].sum()), "n_down": int(down[r, :, k].sum()),
                      "up_words": ", ".join(f"{w} ({n})" for w, n in Counter(word_of[int(x)] for x in ids[r, :, k][np.isin(ids[r, :, k], [q for q, (_, s) in pole.items() if s > 0])].ravel()).most_common(5)),
                      "down_words": ", ".join(f"{w} ({n})" for w, n in Counter(word_of[int(x)] for x in ids[r, :, k][np.isin(ids[r, :, k], [q for q, (_, s) in pole.items() if s < 0])].ravel()).most_common(5))})
    t = pd.DataFrame(t); t.to_csv(OUT / "patch_layer_teaser.csv", index=False)
    print(t.pivot_table(index="layer", columns="borough", values=["n_up", "n_down"]).to_string())
    print(t[t.layer.isin([8, 16, 24, 32, 40, 48, 56])][["borough", "layer", "up_words", "down_words"]].to_string(index=False))


if __name__ == "__main__":
    main()
