"""F10.33c: the severe words of the down end on the image patches (Gemma-4-31B, 591 images).

A severe word is a social or valence word of the down end of the shared direction (crime, waste,
abuse, poverty), not an object word (truck, forklift). SEVERE lists the candidates by hand, and
the script keeps those within rank RANK_MAX of the down end. For each severe word and layer: the
number of (image, patch) top-30 lists that hold it. For each image: the number of patches with
a severe word at each layer, and the answer margin. Writes the candidate images of each borough
and a contact sheet of them.

Outputs: `figures/f10/mechanism/patch_severe_words.csv`, `patch_severe_images.csv`, and the
contact sheets `figures/f10/mechanism/severe_sheet_{borough}.png`.

Use:
    python scripts/f10/patch_severe.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.special import logsumexp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10")]
import plot_patch_teaser as T  # noqa: E402
from hop_hierarchy import answer_store  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "figures/f10/mechanism"
RANK_MAX = 2000
SEVERE = """criminals criminal criminality crimin kriminal crimes crime unemployed brutality trafficking robbery rape abusive
filth filthy stench trash garbage thieves decay sewage violence violent thugs gangs gang dirty poor vandalism prostitution
murderers assaulting scavenging scav vultures desperate struggling contempt negligent incompetent incompetence distrust
corrupted perverse betray cursing disgusted useless failures dysfunctional unclean waste wastes dump dumping dumped
coerce coerced coercive forcibly adversary enemigos entrap prosecuting prosecute punk retarded worse slaughter
deteriorating contaminate contaminating contaminants contaminated contamination crude disposable discard
drugs addicts beggars menacing rotten sludge illegal unsafe dangerous neglected dilapidated hostile grim bleak""".split()
N_SHOW = 12


def main() -> None:
    store = open_store(T.MODEL, ALL_ROOT)
    vocab = pd.read_parquet(store.path / "vocab.parquet")
    layers = store.manifest["layers"]
    z = np.load(T.DIRECTION); d = z["direction"].astype(np.float64)
    score = ((d - d.mean(1, keepdims=True)) / d.std(1, keepdims=True)).mean(0)
    words = vocab.token.str.strip().to_numpy(); tid = vocab.token_id.to_numpy()
    ok = np.array([bool(T.WORD.fullmatch(w)) for w in words])
    order = [i for i in np.argsort(score) if ok[i]]
    rank = {words[i]: r + 1 for r, i in enumerate(order)}
    severe = {int(tid[i]): words[i] for i in order[:RANK_MAX] if words[i] in SEVERE}
    print(f"[severe] {len(severe)} of {len(set(SEVERE))} candidates within rank {RANK_MAX}:",
          ", ".join(f"{w} ({rank[w]})" for w in sorted(severe.values(), key=lambda w: rank[w])))
    ids = np.concatenate([np.asarray(np.load(f, mmap_mode="r")) for f in sorted((store.path / "patches").glob("ids-*.npy"))])
    sev_ids = np.array(list(severe))
    hit = np.isin(ids, sev_ids)  # [images, patches, layers, 30]
    # words by layer
    rows = []
    for k, layer in enumerate(layers):
        c = Counter(severe[int(t)] for t in ids[:, :, k][hit[:, :, k]].ravel())
        rows.append({"layer": layer, "n_hits": sum(c.values()), "n_images": int(hit[:, :, k].any(-1).any(-1).sum()),
                     "top": ", ".join(f"{w} ({n})" for w, n in c.most_common(12))})
    w = pd.DataFrame(rows); w.to_csv(OUT / "patch_severe_words.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 220)
    print(w.to_string(index=False))
    # images
    a = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(T.MODEL, "cool", "place") / "answers.parquet")
    a = a.assign(margin=logsumexp(a[T.YES], axis=1) - logsumexp(a[T.NO], axis=1)).set_index("image_idx").margin
    im = store.images[["image_idx", "sample_id", "image_path", "nta_name", "borough"]].reset_index(drop=True)
    im["margin"] = a.reindex(im.image_idx).to_numpy()
    patch_hit = hit.any(-1)  # [images, patches, layers]
    im["sev_any"] = patch_hit.any(-1).sum(1)
    for k, layer in enumerate(layers):
        im[f"sev_L{layer}"] = patch_hit[:, :, k].sum(1)
    pole = T.pole_words(vocab)
    up, down = T.hits(ids, pole)
    band = [k for k, l in enumerate(layers) if T.DEPTH[0] <= l / T.N_LAYERS < T.DEPTH[1]]
    im["n_up_band"] = up[:, :, band].any(-1).sum(1); im["n_down_band"] = down[:, :, band].any(-1).sum(1)
    obj = np.isin(ids, np.array([t for t, (w_, s) in pole.items() if s < 0 and w_ in {"truck", "trucks", "trucker", "truckers", "tanker", "tankers", "forklift", "forkl", "bulldozer", "warehouse", "warehouses", "depot", "depots", "loader", "loaders", "machinery", "factories", "refinery"}])).any(-1)
    im["n_object_any"] = obj.any(-1).sum(1)
    def sev_words(r):
        c = Counter(severe[int(t)] for t in ids[r][hit[r]].ravel())
        return ", ".join(f"{w_} ({n})" for w_, n in c.most_common(6))
    im["severe_words"] = [sev_words(r) for r in range(len(im))]
    r = stats.pearsonr(im.sev_any, im.margin)
    print(f"[severe] r of the severe patch count (any layer) with the margin: {r.statistic:.3f} (p {r.pvalue:.1e}); images with a severe word: {(im.sev_any > 0).sum()} of {len(im)}")
    im.to_csv(OUT / "patch_severe_images.csv", index=False)
    for borough in ("Bronx", "Brooklyn", "Queens", "Manhattan", "Staten Island"):
        g = im[im.borough == borough].sort_values(["sev_any", "n_object_any"], ascending=[False, True]).head(N_SHOW)
        print(f"\n== {borough}")
        print(g[["sample_id", "nta_name", "margin", "sev_any", "n_object_any", "n_up_band", "n_down_band", "severe_words"]].round(1).to_string(index=False))
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        for ax, (_, row) in zip(axes.ravel(), g.iterrows()):
            try:
                ax.imshow(Image.open(row.image_path).convert("RGB"))
            except Exception as e:  # noqa: BLE001
                ax.text(0.5, 0.5, str(e)[:40], ha="center")
            ax.set_title(f"{row.sample_id} {row.nta_name[:22]} m={row.margin:.1f}\nsev {row.sev_any} obj {row.n_object_any}: {row.severe_words[:60]}", fontsize=7)
            ax.axis("off")
        for ax in axes.ravel()[len(g):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(OUT / f"severe_sheet_{borough.replace(' ', '_')}.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
