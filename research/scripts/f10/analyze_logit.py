#!/usr/bin/env python
"""F10.7: analyze the logit metrics of `hop_logit.py`. CPU only.

| Method | Output in `research/figures/f10/logit/` | Unit and test |
|--------|--------------------------------|---------------|
| Margin at each cell, J++ and `J = I` | `logit_margin.csv`, `logit_margin.png` | r with the answer margin over images, NTA-clustered t, df 196, Holm over the cells of 1 lens |
| Distance to the model output at the same token | `logit_settle.csv`, `logit_settle.png` | NTA means; a description, no test |
| Per-token z-scoring | `logit_words.csv`, `logit_cells.csv` | word lists from 98 NTAs. A test on the other 99 NTAs, df 98, Holm over the cells |
| Cosine with the answer direction | `logit_cosine.csv` | raw and whitened, against random word sets |

Each read is first standardized over the whole vocabulary: (read - mu) / sd. The softcap
is not in these reads. The words are the store words, thus the map has no non-Latin
tokens and no words under 3 letters.

A cell is a (token, layer) pair. The margin of a cell is log p(yes, any case) minus log
p(no, any case) in the read. The answer margin is the same value in the model output.
The margin at the last tokens is trivial, because the model output is near. The test on
a word list is a search result on held-out images. It is not a causal result.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_logit.py [--models gemma-4-31b qwen3.5-9b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b")
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
LOGIT_ROOT = ROOT / "data/f10/hop_logit"
OUT = ROOT / "research/figures/f10/logit"
SEED = 20261101
LIST_SIZE = 50
N_RANDOM = 200


def split_ntas(ntas: pd.Series, seed: int = SEED) -> np.ndarray:
    """Give True for the images of the discovery NTAs: 98 of the 197, by a fixed seed."""
    names = np.array(sorted(ntas.unique()))
    picked = np.random.default_rng(seed).permutation(names)[: len(names) // 2]
    return ntas.isin(picked).to_numpy()


def cluster_r(x: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> tuple[float, float, float]:
    """Give r, the t value and the p value, from OLS on z-scores with NTA-clustered errors (CR1)."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    r = float(x @ y / n)
    e = y - r * x
    codes = pd.factorize(cluster)[0]
    g = codes.max() + 1
    meat = float(np.sum(np.bincount(codes, x * e, g) ** 2))
    var = meat / (x @ x) ** 2 * (g / (g - 1)) * ((n - 1) / (n - 2))
    t = r / np.sqrt(var) if var > 0 else 0.0
    return r, float(t), float(2 * stats.t.sf(abs(t), g - 1))


def holm(p: np.ndarray) -> np.ndarray:
    """Give Holm-adjusted p values."""
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(p)
    out[order] = adj
    return out


def margin_of(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return logsumexp(x[..., :half], axis=-1) - logsumexp(x[..., half:], axis=-1)


def load(model: str) -> dict:
    root = LOGIT_ROOT / model
    head = json.loads((root / "manifest.json").read_text())
    files = sorted((root / "metrics").glob("metrics-*.npz"))
    parts = [np.load(f) for f in files]
    metrics = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    store = open_store(model, ALL_ROOT)
    images = store.images.merge(pd.read_parquet(root / "answers.parquet"), on="image_idx", validate="1:1", sort=False)
    return {"root": root, "head": head, "metrics": metrics, "store": store, "images": images, "text": np.load(root / "text.npz")}


def margin_table(d: dict, model: str) -> pd.DataFrame:
    m, im, head = d["metrics"], d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    y, cl = im["margin"].to_numpy(), im["nta2020"].to_numpy()
    text = {"lens": margin_of(d["text"]["lens"]), "plain": margin_of(d["text"]["plain"])}
    rows = []
    for kind in ("lens", "plain"):
        cell = margin_of(m[kind])
        block = []
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                x = cell[:, a, b]
                r, t, p = cluster_r(x, y, cl) if x.std() > 0 else (0.0, 0.0, 1.0)
                block.append({"model": model, "lens": kind, "position": pos, "token": tokens[a], "index": a, "layer": layer, "mean": x.mean(),
                              "text": text[kind][a, b], "r": r, "t": t, "p": p})
                if kind == "lens":
                    block[-1]["r_scaled"] = cluster_r(x / m["sd"][:, a, b], y, cl)[0]
        block = pd.DataFrame(block)
        block["p_holm"] = holm(block["p"].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def settle_table(d: dict, model: str) -> pd.DataFrame:
    m, im, head = d["metrics"], d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    codes = pd.factorize(im["nta2020"])[0]
    nta = lambda x: np.stack([np.bincount(codes, x[:, a, b]) / np.bincount(codes) for a in range(x.shape[1]) for b in range(x.shape[2])]).reshape(x.shape[1], x.shape[2], -1)
    means = {k: nta(m[k]) for k in ("js", "js_capped", "kl_capped", "agree")}
    rows = []
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            row = {"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer}
            for k, v in means.items():
                row[k], row[f"{k}_se"] = v[a, b].mean(), v[a, b].std(ddof=1) / np.sqrt(v.shape[2])
            row["rms_cv"] = m["rms"][:, a, b].std() / m["rms"][:, a, b].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def shard_reads(d: dict) -> list[Path]:
    capped = sorted((d["root"] / "uncapped").glob("reads-*.npy"))
    return capped if capped else sorted((d["store"].path / "reads").glob("reads-*.npy"))


def word_maps(d: dict, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Give the held-out test of the word lists, the words of each cell, and the top rows for the cosines."""
    m, im, head = d["metrics"], d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    discovery = split_ntas(im["nta2020"])
    y = im["margin"].to_numpy()
    y_d, y_t = y - y[discovery].mean(), y - y[~discovery].mean()
    files = shard_reads(d)
    shape = (len(head["positions"]), len(head["layers"]), len(words))
    n = np.zeros(2)
    s = {k: np.zeros(shape) for k in ("sum_d", "sq_d", "sum_t", "xy_d", "xy_t")}
    start = 0
    for f in files:
        x = np.load(f)
        k = len(x)
        mask = discovery[start:start + k]
        x = (x - m["mu"][start:start + k, ..., None]) / m["sd"][start:start + k, ..., None]
        dx, tx = x[mask].astype(np.float64), x[~mask].astype(np.float64)
        n += [len(dx), len(tx)]
        s["sum_d"] += dx.sum(0)
        s["sq_d"] += (dx * dx).sum(0)
        s["sum_t"] += tx.sum(0)
        s["xy_d"] += np.einsum("i,ijkl->jkl", y_d[start:start + k][mask], dx)
        s["xy_t"] += np.einsum("i,ijkl->jkl", y_t[start:start + k][~mask], tx)
        start += k
        print(f"[f10.7] {model}: pass 1, {start} images", flush=True)
    mean_d, mean_t = s["sum_d"] / n[0], s["sum_t"] / n[1]
    sd_d = np.sqrt((s["sq_d"] / n[0] - mean_d ** 2).clip(min=1e-12))
    slope_d, slope_t = s["xy_d"] / n[0] / sd_d, s["xy_t"] / n[1] / sd_d
    top = np.argsort(-slope_d, axis=-1)[..., :LIST_SIZE]
    bottom = np.argsort(slope_d, axis=-1)[..., :LIST_SIZE]
    score = np.zeros((len(y), *shape[:2]), np.float32)
    start = 0
    for f in files:
        x = np.load(f)
        k = len(x)
        x = (x - m["mu"][start:start + k, ..., None]) / m["sd"][start:start + k, ..., None]
        z = (x - mean_d) / sd_d
        for a in range(shape[0]):
            for b in range(shape[1]):
                score[start:start + k, a, b] = z[:, a, b, top[a, b]].mean(-1) - z[:, a, b, bottom[a, b]].mean(-1)
        start += k
        print(f"[f10.7] {model}: pass 2, {start} images", flush=True)
    held = ~discovery
    cl = im["nta2020"].to_numpy()
    rows, cells = [], []
    pool_mean, pool_var = mean_d.mean(0, keepdims=True), (sd_d ** 2 + (mean_d - mean_d.mean(0, keepdims=True)) ** 2).mean(0, keepdims=True)
    cell_d = (mean_d - pool_mean) / np.sqrt(pool_var)
    cell_t = (mean_t - mean_t.mean(0, keepdims=True)) / np.sqrt(pool_var)
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            r, t, p = cluster_r(score[held, a, b], y[held], cl[held]) if score[held, a, b].std() > 0 else (0.0, 0.0, 1.0)
            rows.append({"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer, "r_test": r, "t": t, "p": p,
                         "slope_stability": np.corrcoef(slope_d[a, b], slope_t[a, b])[0, 1],
                         "top_sign_held": (slope_t[a, b, top[a, b]] > 0).mean(), "bottom_sign_held": (slope_t[a, b, bottom[a, b]] < 0).mean(),
                         "up": ", ".join(words[top[a, b, :12]]), "down": ", ".join(words[bottom[a, b, :12]])})
            order = np.argsort(-cell_d[a, b])[:15]
            cells.append({"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer,
                          "stability": np.corrcoef(cell_d[a, b], cell_t[a, b])[0, 1], "words": ", ".join(words[order])})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    lists = {"top": top, "bottom": bottom}
    return table, pd.DataFrame(cells), lists


def cosine_table(d: dict, model: str, lists: dict) -> pd.DataFrame:
    """Give the cosine of each list contrast with the yes - no direction, raw and whitened."""
    head, root = d["head"], d["root"]
    tokens = d["store"].manifest["tail_tokens"]
    rows_w = np.load(root / "unembed_store.npy", mmap_mode="r")
    answer = np.load(root / "answer_rows.npy").astype(np.float64)
    cov = np.load(root / "unembed_cov.npy").astype(np.float64)
    value, vector = np.linalg.eigh(cov)
    value = value.clip(min=value.max() * 1e-8)
    white = (vector / np.sqrt(value)) @ vector.T
    target = answer[:len(answer) // 2].mean(0) - answer[len(answer) // 2:].mean(0)
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    rng = np.random.default_rng(SEED)
    null = {"raw": [], "white": []}
    for _ in range(N_RANDOM):
        pick = rng.choice(rows_w.shape[0], 2 * LIST_SIZE, replace=False)
        v = np.asarray(rows_w[np.sort(pick[:LIST_SIZE])], np.float64).mean(0) - np.asarray(rows_w[np.sort(pick[LIST_SIZE:])], np.float64).mean(0)
        null["raw"].append(cos(v, target))
        null["white"].append(cos(white @ v, white @ target))
    sd = {k: float(np.std(v)) for k, v in null.items()}
    rows = []
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            v = np.asarray(rows_w[np.sort(lists["top"][a, b])], np.float64).mean(0) - np.asarray(rows_w[np.sort(lists["bottom"][a, b])], np.float64).mean(0)
            rows.append({"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer, "raw": cos(v, target),
                         "white": cos(white @ v, white @ target), "null_sd_raw": sd["raw"], "null_sd_white": sd["white"]})
    return pd.DataFrame(rows)


def label(token: str) -> str:
    return token.replace("\n", "\\n").replace("$", r"\$").replace("_", r"\_").strip() or "_"


def plot_grid(tables: dict, column: str, path: Path, title: str, caption: str, tags: list[str], vmin: float, vmax: float, label_bar: str, div: bool) -> None:
    import matplotlib.pyplot as plt
    import _style as S

    S.apply_house_style()
    keys = list(tables)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.45 * len(keys), 4.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        t = tables[key]
        grid = t.pivot(index="index", columns="layer", values=column).to_numpy()
        im = ax.imshow(grid, aspect="auto", cmap=S.CMAP_DIV if div else S.CMAP_SEQ, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.grid(False)
        layers = sorted(t["layer"].unique())
        ax.set_xticks(range(len(layers))[::2 if len(layers) > 10 else 1], layers[::2 if len(layers) > 10 else 1], fontsize=6)
        names = t.drop_duplicates("index").sort_values("index")["token"]
        ax.set_yticks(range(len(names)), [label(n) for n in names], fontsize=5.5, family="monospace")
        ax.set_xlabel("layer")
        ax.set_title(key, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.7, label=label_bar)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": title, "plot-caption": caption, "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": tags}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = {k: [] for k in ("margin", "settle", "words", "cells", "cosine")}
    for model in args.models:
        d = load(model)
        frames["margin"].append(margin_table(d, model))
        frames["settle"].append(settle_table(d, model))
        words, cells, lists = word_maps(d, model)
        frames["words"].append(words)
        frames["cells"].append(cells)
        frames["cosine"].append(cosine_table(d, model, lists))
        print(f"[f10.7] {model}: finished", flush=True)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
    for key, name in (("margin", "logit_margin"), ("settle", "logit_settle"), ("words", "logit_words"), ("cells", "logit_cells"), ("cosine", "logit_cosine")):
        tables[key].to_csv(out / f"{name}.csv", index=False)
    m, s = tables["margin"], tables["settle"]
    plot_grid({f"{NAMES[x]}, {n}": m[(m["model"] == x) & (m["lens"] == k)] for x in args.models for k, n in (("lens", "J++"), ("plain", "J = I"))}, "r",
              out / "logit_margin.png", "Margin at each cell against the answer margin",
              "Pearson r over 591 images between the yes-minus-no margin of a read and the same margin in the model output. Each row is a token of "
              "the prompt after the image, and each column is a layer. The first columns of each pair use the J++ lens, and the others use the plain logit lens. "
              "The margin at the last tokens is near the answer by construction.",
              ["F10.7", "logit margin"], -1, 1, "r with the answer margin", True)
    plot_grid({NAMES[x]: s[s["model"] == x] for x in args.models}, "js_capped", out / "logit_settle.png", "Distance of each read from the model output",
              "Mean Jensen-Shannon divergence in nats, over NTA means, between the J++ read at a token and layer and the model output at the same token. "
              "The softcap is on both, as in the model. Low values show that the read is near the output.",
              ["F10.7", "Jensen-Shannon"], 0, float(np.log(2)), "JS divergence (nats)", False)
    print("[f10.7] finished", flush=True)


if __name__ == "__main__":
    main()
