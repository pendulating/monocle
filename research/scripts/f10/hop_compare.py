#!/usr/bin/env python
"""F10.19: compare the J-space of 3 models, to separate model size from model family.

Gemma-4-31B and Qwen3.6-27B have about the same size and different families. Qwen3.6-27B
and Qwen3.5-9B have the same family and different sizes. Each pair uses the same 591
images, the same prompt and the same lens recipe.

| Output in `--out` | Question | Method |
|-------------------|----------|--------|
| `verdict.csv` | Do 2 models give the same verdict for an image | Clustered r of the margins, kappa of the answers |
| `verdict_tests.csv` | Is 1 pair closer than a second pair | Williams t on NTA means |
| `cross_grid.csv` | Which layers of 2 models agree | r over the shared words at each layer pair |
| `cross_depth.csv` | How close are 2 models at the same relative depth | The rows of the grid at matched depth, with the split-half ceiling |
| `cross_tests.csv` | Is 1 pair closer than a second pair, over tokens | Paired t of Fisher z, with 1 unit for each token |
| `sense.csv` | Does each model split `cool` by sense | SVD of the NTA means, and the fixed-list score of F10.17 |

Columns of a grid row:

| Column | Content |
|--------|---------|
| `r_mean` | r of the mean reads over the shared words |
| `r_added` | r of the mean read minus the read with no image |
| `same`, `other` | Mean r of the borough t vectors, for the same borough and for different boroughs |
| `rsa` | Spearman r of the image RDMs, from the residuals. It needs no lens and no shared words |
| `r_sense` | r over images of the sense score of F10.17 |

The layers of 2 models do not align, thus the largest cell of a grid is a selection. Use
`cross_depth.csv` for a test. `--solo` runs the per-model scripts on 1 model and writes
to subfolders of `--out`, thus the tables of the first 2 models do not change.

Use:
    python research/scripts/f10/hop_compare.py --solo qwen3.6-27b
    python research/scripts/f10/hop_compare.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from analyze_logit import ALL_ROOT, cluster_r, holm, load, open_store, shard_reads  # noqa: E402
from hop_borough import BOROUGHS, split_within, two_sample_t  # noqa: E402
from hop_depth import N_LAYERS  # noqa: E402
from hop_free import load_tail, rdm_ranks  # noqa: E402
from hop_sense import CAP, SOCIAL, TEMPERATURE  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.6-27b", "qwen3.5-9b")
OUT = ROOT / "research/figures/f10/compare"
SPLITS = 5
GEMMA = {"end_image": "<image|>", "model": "model", "channel": "<|channel>", "end": "<channel|>"}
QWEN = {"end_image": "<|vision_end|>", "model": "assistant", "channel": "<think>", "end": "</think>"}
TEXT = {"Is": "Is", "this": " this", "place": " place", "cool": " cool", "?": "?", "Answer": " Answer", "yes": " yes", "no": " no"}
ORDER = ("end_image", "Is", "this", "place", "cool", "?", "Answer", "yes", "no", "model", "channel", "end")


def token_index(model: str, tokens: list[str]) -> dict[str, int]:
    """Give the last tail index of each compared token. A late template token can repeat."""
    names = TEXT | (GEMMA if model.startswith("gemma") else QWEN)
    return {k: len(tokens) - 1 - tokens[::-1].index(names[k]) for k in ORDER}


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Give r of each row of a and each row of b."""
    return zscore(a) @ zscore(b).T / a.shape[-1]


def gather(model: str, shared: list[str]) -> dict:
    """Give the NTA means of the compared tokens over the shared words, the sense score and the residual RDMs."""
    d = load(model)
    im, head = d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    words = pd.Series(d["store"].vocab["token"].str.strip().str.lower().to_numpy())
    first = ~words.duplicated()
    index = pd.Series(np.flatnonzero(first), index=words[first].to_numpy())
    pick = index[shared].to_numpy()
    hot, soc = np.flatnonzero(words.isin(TEMPERATURE)), np.flatnonzero(words.isin(SOCIAL))
    where = token_index(model, tokens)
    at = [where[k] for k in ORDER]
    codes = pd.factorize(im["nta2020"])[0]
    groups = codes.max() + 1
    n_layer = len(head["layers"])
    m = np.zeros((groups, len(at), n_layer, len(pick)), np.float32)
    score = np.zeros((len(im), len(at), n_layer), np.float32)
    start = 0
    for f in shard_reads(d):
        x = np.load(f, mmap_mode="r")
        rows = np.arange(start, start + len(x))
        for t, a in enumerate(at):
            z = zscore(np.asarray(x[:, a], dtype=np.float32))
            score[rows, t] = z[..., soc].mean(-1) - z[..., hot].mean(-1)
            z = z[..., pick]
            for g in np.unique(codes[rows]):
                m[g, t] += z[codes[rows] == g].sum(0)
        start += len(x)
    m /= np.bincount(codes)[:, None, None, None]
    text = np.load(d["store"].path / "text_reads.npy")[at]
    if head["softcap"]:
        text = CAP * np.arctanh((text.astype(np.float64) / CAP).clip(-1 + 1e-7, 1 - 1e-7))
    text = zscore(text.astype(np.float32))[..., pick]
    x = load_tail(d)[:, at]
    rdm = np.stack([rdm_ranks(x[:, t, b].astype(np.float64)) for t in range(len(at)) for b in range(n_layer)])
    borough = im.groupby(codes)["borough"].first().to_numpy()
    print(f"[f10.19] {model}: {m.shape}", flush=True)
    return {"model": model, "m": m, "score": score, "text": text, "rdm": rdm.reshape(len(at), n_layer, -1), "layers": list(head["layers"]), "borough": borough,
            "codes": codes, "margin": im["margin"].to_numpy(), "ids": im["image_idx"].tolist(), "nta": im["nta2020"].to_numpy()}


def borough_t(d: dict) -> np.ndarray:
    """Give the t vectors [boroughs, tokens, layers, words], and the split-half r [tokens, layers]."""
    m, b = d["m"], d["borough"]
    t = np.stack([np.stack([two_sample_t(m[:, a, c], b == k) for c in range(m.shape[2])]) for a in range(m.shape[1]) for k in BOROUGHS])
    t = t.reshape(m.shape[1], len(BOROUGHS), m.shape[2], -1).transpose(1, 0, 2, 3)
    half = np.zeros(m.shape[1:3])
    for s in range(SPLITS):
        h = split_within(b, s)
        for a in range(m.shape[1]):
            for c in range(m.shape[2]):
                half[a, c] += np.mean([np.corrcoef(two_sample_t(m[h, a, c], b[h] == k), two_sample_t(m[~h, a, c], b[~h] == k))[0, 1] for k in BOROUGHS]) / SPLITS
    return t.astype(np.float32), half


def williams(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Give Williams t for 2 dependent correlations that share variable 1."""
    det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    bar = (r12 + r13) / 2
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) / (2 * (n - 1) / (n - 3) * det + bar ** 2 * (1 - r23) ** 3))
    return float(t), float(2 * stats.t.sf(abs(t), n - 3))


def verdict(got: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    nta = {k: pd.Series(v["margin"]).groupby(v["codes"]).mean().to_numpy() for k, v in got.items()}
    for a, b in itertools.combinations(got, 2):
        x, y = got[a], got[b]
        ya, yb = x["margin"] > 0, y["margin"] > 0
        po = np.mean(ya == yb)
        pe = ya.mean() * yb.mean() + (1 - ya.mean()) * (1 - yb.mean())
        r, t, p = cluster_r(x["margin"], y["margin"], x["nta"])
        rows.append({"a": a, "b": b, "r": r, "t": t, "p": p, "r_nta": np.corrcoef(nta[a], nta[b])[0, 1], "rho": stats.spearmanr(x["margin"], y["margin"])[0],
                     "yes_a": ya.mean(), "yes_b": yb.mean(), "agree": po, "kappa": (po - pe) / (1 - pe)})
    for pivot in got:
        others = [k for k in got if k != pivot]
        if len(others) == 2:
            u, v = others
            r12, r13, r23 = (np.corrcoef(p, q)[0, 1] for p, q in ((nta[pivot], nta[u]), (nta[pivot], nta[v]), (nta[u], nta[v])))
            t, p = williams(r12, r13, r23, len(nta[pivot]))
            tests.append({"pivot": pivot, "first": u, "second": v, "r_first": r12, "r_second": r13, "r_others": r23, "n": len(nta[pivot]), "t": t, "p": p})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def cross(x: dict, y: dict, tx: np.ndarray, ty: np.ndarray) -> pd.DataFrame:
    rows = []
    ma, mb = x["m"].mean(0), y["m"].mean(0)
    n_b = len(BOROUGHS)
    for t, name in enumerate(ORDER):
        ra, rb = x["layers"], y["layers"]
        mean = corr(ma[t], mb[t])
        added = corr(ma[t] - x["text"][t], mb[t] - y["text"][t])
        c = corr(tx[:, t].reshape(n_b * len(ra), -1), ty[:, t].reshape(n_b * len(rb), -1)).reshape(n_b, len(ra), n_b, len(rb))
        same = np.mean([c[i, :, i] for i in range(n_b)], 0)
        other = np.mean([c[i, :, j] for i in range(n_b) for j in range(n_b) if i != j], 0)
        rsa = x["rdm"][t] @ y["rdm"][t].T / x["rdm"].shape[-1]
        sense = corr(x["score"][:, t].T, y["score"][:, t].T)
        for i, la in enumerate(ra):
            for j, lb in enumerate(rb):
                rows.append({"a": x["model"], "b": y["model"], "token": name, "layer_a": la, "layer_b": lb, "r_mean": mean[i, j], "r_added": added[i, j], "same": same[i, j],
                             "other": other[i, j], "rsa": rsa[i, j], "r_sense": sense[i, j]})
    return pd.DataFrame(rows)


def depth_rows(grid: pd.DataFrame, got: dict, half: dict) -> pd.DataFrame:
    """Keep the matched cells. The model with the fewest saved layers gives the depths."""
    anchor = min(got, key=lambda k: len(got[k]["layers"]))
    depths = [(q + 1) / (N_LAYERS[anchor] + 1) for q in got[anchor]["layers"]]
    near = lambda model, f: min(got[model]["layers"], key=lambda q: abs((q + 1) / (N_LAYERS[model] + 1) - f))
    rows = []
    for (a, b), g in grid.groupby(["a", "b"], sort=False):
        for f in depths:
            la, lb = near(a, f), near(b, f)
            part = g[(g["layer_a"] == la) & (g["layer_b"] == lb)].assign(depth=round(f, 3))
            for _, r in part.iterrows():
                t = ORDER.index(r["token"])
                h_a, h_b = half[a][t, got[a]["layers"].index(la)], half[b][t, got[b]["layers"].index(lb)]
                ceiling = np.sqrt(max(h_a, 0.0) * max(h_b, 0.0))
                rows.append(dict(r) | {"depth": round(f, 3), "half_a": h_a, "half_b": h_b, "same_ceiling": r["same"] / ceiling if ceiling > 0 else np.nan})
    return pd.DataFrame(rows)


def paired(depth: pd.DataFrame) -> pd.DataFrame:
    """Give the paired t of 2 pairs. The unit is 1 token, with the Fisher z mean over depths."""
    rows = []
    pairs = list(depth[["a", "b"]].drop_duplicates().itertuples(index=False, name=None))
    for measure in ("r_mean", "r_added", "same", "rsa", "r_sense"):
        z = depth.assign(z=np.arctanh(depth[measure].clip(-0.999, 0.999))).groupby(["a", "b", "token"])["z"].mean().unstack("token")
        for p, q in itertools.combinations(pairs, 2):
            gap = z.loc[p] - z.loc[q]
            t, value = stats.ttest_1samp(gap, 0.0)
            rows.append({"measure": measure, "first": "/".join(p), "second": "/".join(q), "r_first": np.tanh(z.loc[p].mean()), "r_second": np.tanh(z.loc[q].mean()),
                         "tokens": len(gap), "t": t, "p": value})
    out = pd.DataFrame(rows, columns=["measure", "first", "second", "r_first", "r_second", "tokens", "t", "p"])
    return out.assign(p_holm=holm(out["p"].to_numpy()))


def sense(got: dict) -> pd.DataFrame:
    """Give the first axis of the NTA means and the sense score at the compared tokens."""
    rows = []
    for model, d in got.items():
        codes = d["codes"]
        for t, name in enumerate(ORDER):
            for b, layer in enumerate(d["layers"]):
                c = d["m"][:, t, b] - d["m"][:, t, b].mean(0, keepdims=True)
                w, u = np.linalg.eigh(c @ c.T)
                v = d["score"][:, t, b]
                nta = np.bincount(codes, v) / np.bincount(codes)
                rows.append({"model": model, "token": name, "layer": layer, "share1": w[-1] / w.sum(), "r_axis1": abs(np.corrcoef(nta, u[:, -1])[0, 1]),
                             "r_margin": cluster_r(v, d["margin"], d["nta"])[0], "sd": v.std()})
    return pd.DataFrame(rows)


def solo(model: str, out: Path) -> None:
    """Run the per-model scripts on 1 model. Each script writes to a subfolder of `out`."""
    import analyze_logit
    import hop_axes
    import hop_borough
    import hop_sets
    import hop_stage

    argv = sys.argv
    steps = (("logit", analyze_logit), ("sets", hop_sets), ("borough", hop_borough), ("axes", hop_axes), ("stage", hop_stage))
    for name, mod in steps:
        if name == "logit" and (out / name / "logit_margin.csv").exists():
            continue
        mod.OUT = out / name
        (out / name).mkdir(parents=True, exist_ok=True)
        sys.argv = [name, "--models", model] + (["--out", str(out / name)] if name == "logit" else [])
        mod.main()
    sys.argv = argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--solo", help="run the per-model scripts on this model, then stop")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.solo:
        solo(args.solo, out)
        return
    vocab = {m: set(open_store(m, ALL_ROOT).vocab["token"].str.strip().str.lower()) for m in args.models}
    shared = sorted(set.intersection(*vocab.values()))
    print(f"[f10.19] {len(shared)} shared words", flush=True)
    got = {m: gather(m, shared) for m in args.models}
    if len({tuple(d["ids"]) for d in got.values()}) != 1:
        raise ValueError("the models do not hold the same image order")
    table, tests = verdict(got)
    table.to_csv(out / "verdict.csv", index=False)
    tests.to_csv(out / "verdict_tests.csv", index=False)
    sense(got).to_csv(out / "sense.csv", index=False)
    pairs = {m: borough_t(got[m]) for m in args.models}
    half = {m: pairs[m][1] for m in args.models}
    print("[f10.19] borough vectors", flush=True)
    grid = pd.concat([cross(got[a], got[b], pairs[a][0], pairs[b][0]) for a, b in itertools.combinations(args.models, 2)], ignore_index=True)
    grid.to_csv(out / "cross_grid.csv", index=False)
    depth = depth_rows(grid, got, half)
    depth.to_csv(out / "cross_depth.csv", index=False)
    paired(depth).to_csv(out / "cross_tests.csv", index=False)
    print("[f10.19] finished", flush=True)


if __name__ == "__main__":
    main()
