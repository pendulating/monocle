#!/usr/bin/env python
"""F10.9: lens-free analysis of the residuals. CPU only.

No step here uses a lens. The margin is the pooled yes-minus-no value of the model output.
The prompt puts the image first, thus the patches cannot see the question.

| Analysis | What it asks | Method |
|----------|--------------|--------|
| Decoder map | Which cells hold margin information | Dual ridge, 5 folds by NTA, 3 repeats of the folds, out-of-fold R2, NTA bootstrap interval |
| Scene test | Does a cell hold margin information beyond the scene block | Scene block alone against scene plus cell, on the same folds. Paired t on the NTA error difference, df 196, Holm over the cells |
| Generalization | Does a decoder from 1 cell work at another cell | Train at each cell, test at all cells, out-of-fold r |
| RSA | Which cells share 1 geometry of the images | Correlation-distance RDMs, Spearman r between all cells, split by NTA |
| Commonality | What the RDM of a cell shares with the margin and the scene | Rank regression on 2 model RDMs |
| 2 models | Which layers of the 2 models match | Spearman r across models, then 1 monotone path of layers |

The scene block is the patch residuals of all saved layers, pooled to a 4 x 4 grid. It is
1 linear read of question-free features. "Beyond the scene" means beyond this block only.
All cells are functions of the image. Near the answer, a cell decodes the margin by
construction, and a decoder shows that a code is present and not that the model uses it.
The folds share training images, thus the paired t is somewhat too liberal. Read a small
p value near the Holm limit with caution.

Output: `data/f10/hop_free/{model}/`, `data/f10/hop_free/cross.npz`, `figures/f10/free/`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_free.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import MODELS, NAMES, SEED, holm, load, plot_grid, split_ntas  # noqa: E402

__version__ = "1.0.0"

DATA = ROOT / "data/f10/hop_free"
OUT = ROOT / "figures/f10/free"
ALPHAS = 10.0 ** np.arange(-4, 5.0)
MIX = (0.25, 0.5, 0.75)
REPEATS = 3
N_BOOT = 1000
GRID = 4


def folds(groups: np.ndarray, k: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Give k folds that keep each NTA whole. The seed sets the order of the NTAs."""
    names = np.unique(groups)
    code = dict(zip(names, np.random.default_rng(seed).permutation(len(names))))
    return list(GroupKFold(k).split(groups, groups=[code[g] for g in groups]))


def ridge_fit(K: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Give the dual weights, with the penalty from generalized cross-validation, and the GCV score."""
    w, U = linalg.eigh(K, check_finite=False)
    w = w.clip(min=0)
    b = U.T @ y
    scale = w.mean()
    best, score = 1.0, np.inf
    for a in ALPHAS * scale:
        f = w / (w + a)
        rss = np.sum(((1 - f) * b) ** 2)
        gcv = rss / max((len(y) - f.sum()) ** 2, 1e-12)
        if gcv < score:
            best, score = a, gcv
    return U @ (b / (w + best)), float(score)


def oof_predict(K: np.ndarray, y: np.ndarray, splits: list) -> np.ndarray:
    pred = np.zeros_like(y)
    for tr, te in splits:
        mean = y[tr].mean()
        alpha, _ = ridge_fit(K[np.ix_(tr, tr)], y[tr] - mean)
        pred[te] = K[np.ix_(te, tr)] @ alpha + mean
    return pred


def oof_mixed(A: np.ndarray, B: np.ndarray, y: np.ndarray, splits: list) -> np.ndarray:
    """Predict from 2 kernels. Inside each fold, GCV picks the mix on the training images only."""
    pred = np.zeros_like(y)
    for tr, te in splits:
        mean = y[tr].mean()
        best, score, keep = None, np.inf, None
        for m in MIX:
            K = m * A + (1 - m) * B
            alpha, gcv = ridge_fit(K[np.ix_(tr, tr)], y[tr] - mean)
            if gcv < score:
                best, score, keep = K, gcv, alpha
        pred[te] = best[np.ix_(te, tr)] @ keep + mean
    return pred


def r2(y: np.ndarray, pred: np.ndarray, splits: list) -> float:
    sse = np.sum((y - pred) ** 2)
    sst = sum(np.sum((y[te] - y[tr].mean()) ** 2) for tr, te in splits)
    return float(1 - sse / sst)


def kernel(x: np.ndarray) -> np.ndarray:
    """Give a linear kernel with trace n, on centred features."""
    x = x - x.mean(0, keepdims=True)
    K = x @ x.T
    return K * (len(x) / np.trace(K))


def grid_pool(p: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Pool [images, patches, layers, d] to [images, layers, cells, d]. The patches are a square raster."""
    side = int(round(np.sqrt(p.shape[1])))
    if side * side != p.shape[1]:
        raise ValueError("the patches are not a square")
    row = np.arange(side) * grid // side
    cell = np.repeat(row, side) * grid + np.tile(row, side)
    return np.stack([p[:, cell == c].mean(1) for c in range(grid * grid)], axis=2)


def load_tail(d: dict) -> np.ndarray:
    return np.concatenate([np.load(f) for f in sorted((d["store"].path / "residuals").glob("residuals-*.npy"))])


def scene_kernel(d: dict, model: str, grid: int = GRID) -> np.ndarray:
    path = DATA / model / ("scene_kernel.npy" if grid == GRID else f"scene_kernel_{grid}.npy")
    if path.exists():
        return np.load(path)
    got = []
    for f in sorted((d["store"].path / "patches").glob("residuals-*.npy")):
        got.append(grid_pool(np.load(f, mmap_mode="r"), grid).astype(np.float32))
        print(f"[f10.9] {model}: {f.name}", flush=True)
    x = np.concatenate(got)
    K = sum(kernel(x[:, b].reshape(len(x), -1)) for b in range(x.shape[1])) / x.shape[1]
    np.save(path, K)
    return K


def rdm_ranks(x: np.ndarray, pick: np.ndarray | None = None) -> np.ndarray:
    """Give the z-scored ranks of the upper triangle of a correlation-distance RDM."""
    x = x if pick is None else x[pick]
    x = x - x.mean(0, keepdims=True)
    x = x - x.mean(1, keepdims=True)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    v = (1 - x @ x.T)[np.triu_indices(len(x), 1)]
    r = stats.rankdata(v)
    return ((r - r.mean()) / r.std()).astype(np.float32)


def monotone_path(score: np.ndarray) -> list[int]:
    """Give the column for each row that gives the highest sum, with columns that never fall."""
    n, m = score.shape
    best = np.zeros((n, m))
    best[0] = score[0]
    back = np.zeros((n, m), int)
    for i in range(1, n):
        run = np.maximum.accumulate(best[i - 1])
        where = np.array([int(np.argmax(best[i - 1, :j + 1])) for j in range(m)])
        best[i], back[i] = run + score[i], where
    path = [int(np.argmax(best[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def analyze(model: str, grid: int = GRID, gain_only: bool = False) -> dict:
    d = load(model)
    (DATA / model).mkdir(parents=True, exist_ok=True)
    head, im = d["head"], d["images"]
    tokens = d["store"].manifest["tail_tokens"]
    x = load_tail(d)
    n, n_tail, n_layer, width = x.shape
    print(f"[f10.9] {model}: tail {x.shape}", flush=True)
    y = im["margin"].to_numpy(dtype=np.float64)
    groups = im["nta2020"].to_numpy()
    codes = pd.factorize(groups)[0]
    repeats = [folds(groups, seed=SEED + i) for i in range(REPEATS)]
    splits = repeats[0]
    scene = scene_kernel(d, model, grid)
    scene_pred = np.mean([oof_predict(scene, y, s) for s in repeats], axis=0)
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, codes.max() + 1, (N_BOOT, codes.max() + 1))
    members = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]

    def interval(e: np.ndarray, base: np.ndarray) -> tuple[float, float]:
        got = []
        for b in boot:
            ix = np.concatenate([members[c] for c in b])
            got.append(1 - np.sum(e[ix] ** 2) / np.sum(base[ix] ** 2))
        return tuple(np.percentile(got, [2.5, 97.5]))

    base = y - np.mean(y)
    rows, weights = [], np.zeros((n_tail * n_layer, len(splits), width), np.float32)
    cell_pred = np.zeros((n_tail, n_layer, n))
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            c = x[:, a, b].astype(np.float64)
            mean = c.mean(0, keepdims=True)
            c = c - mean
            K = c @ c.T
            scale = n / np.trace(K)
            K *= scale
            pred = np.mean([oof_predict(K, y, s) for s in repeats], axis=0)
            joint = np.mean([oof_mixed(scene, K, y, s) for s in repeats], axis=0)
            gap = np.bincount(codes, (y - scene_pred) ** 2 - (y - joint) ** 2) / np.bincount(codes)
            t, p = stats.ttest_1samp(gap, 0.0)
            lo, hi = interval(y - pred, base)
            rows.append({"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer, "r2": r2(y, pred, splits), "r2_lo": lo, "r2_hi": hi,
                         "r2_scene": r2(y, scene_pred, splits), "r2_joint": r2(y, joint, splits), "gain": r2(y, joint, splits) - r2(y, scene_pred, splits),
                         "t": t, "p": p, "d": gap.mean() / gap.std(ddof=1)})
            cell_pred[a, b] = pred
            for f, (tr, te) in enumerate(splits):
                y0 = y[tr].mean()
                alpha, _ = ridge_fit(K[np.ix_(tr, tr)], y[tr] - y0)
                weights[a * n_layer + b, f] = (scale * (c[tr].T @ alpha)).astype(np.float32)
        print(f"[f10.9] {model}: decoder {a + 1}/{n_tail}", flush=True)
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    table.to_csv(DATA / model / ("decoder.csv" if grid == GRID else f"decoder_grid{grid}.csv"), index=False)
    if gain_only:
        return {"table": table}

    flat = x.reshape(n, n_tail * n_layer, width)
    flat = flat - flat.mean(0, keepdims=True)
    pred = np.zeros((n, n_tail * n_layer, n_tail * n_layer), np.float32)
    for f, (tr, te) in enumerate(splits):
        pred[te] = flat[te] @ weights[:, f].T
    z = pred - pred.mean(0, keepdims=True)
    z /= z.std(0, keepdims=True).clip(min=1e-9)
    yz = ((y - y.mean()) / y.std()).astype(np.float32)
    gen = np.einsum("i,ijk->jk", yz, z) / n
    np.save(DATA / model / "generalization.npy", gen)

    discovery = split_ntas(im["nta2020"])
    print(f"[f10.9] {model}: RDMs", flush=True)
    full = np.stack([rdm_ranks(flat[:, i]) for i in range(flat.shape[1])])
    half_a = np.stack([rdm_ranks(flat[:, i], discovery) for i in range(flat.shape[1])])
    half_b = np.stack([rdm_ranks(flat[:, i], ~discovery) for i in range(flat.shape[1])])
    m = full.shape[1]
    rsa, a_m, b_m = full @ full.T / m, half_a @ half_a.T / half_a.shape[1], half_b @ half_b.T / half_b.shape[1]
    np.save(DATA / model / "rsa.npy", rsa)
    upper = np.triu_indices(len(rsa), 1)
    reliability = float(np.corrcoef(a_m[upper], b_m[upper])[0, 1])
    margin = stats.rankdata(np.abs(y[:, None] - y[None, :])[np.triu_indices(n, 1)])
    margin = (margin - margin.mean()) / margin.std()
    diag = np.sqrt(np.diag(scene))
    sim = scene / np.outer(diag, diag)
    sc = stats.rankdata((1 - sim)[np.triu_indices(n, 1)])
    sc = (sc - sc.mean()) / sc.std()
    rho = float(margin @ sc / len(sc))
    com = []
    for i, cell in enumerate(full):
        r_m, r_s = float(cell @ margin / m), float(cell @ sc / m)
        both = (r_m ** 2 + r_s ** 2 - 2 * r_m * r_s * rho) / (1 - rho ** 2)
        com.append({"model": model, "position": head["positions"][i // n_layer], "token": tokens[i // n_layer], "index": i // n_layer, "layer": head["layers"][i % n_layer],
                    "r_margin": r_m, "r_scene": r_s, "unique_margin": both - r_s ** 2, "unique_scene": both - r_m ** 2, "common": r_m ** 2 + r_s ** 2 - both})
    com = pd.DataFrame(com)
    com.to_csv(DATA / model / "commonality.csv", index=False)
    (DATA / model / "summary.json").write_text(json.dumps({"rsa_split_half_r": reliability, "scene_margin_rdm_rho": rho, "r2_scene": r2(y, scene_pred, splits)}, indent=1) + "\n")
    print(f"[f10.9] {model}: RSA split-half r {reliability:.3f}, scene R2 {r2(y, scene_pred, splits):.3f}", flush=True)
    return {"ids": im["image_idx"].tolist(), "full": full, "table": table, "gen": gen, "tokens": tokens, "head": head, "commonality": com}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--grid", type=int, default=GRID, help="the side of the scene grid; 1 is the mean patch, the best block of hop_scene.py")
    ap.add_argument("--gain-only", action="store_true", help="write the decoder table and stop")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    got = {m: analyze(m, args.grid, args.gain_only) for m in args.models}
    if args.gain_only:
        return
    if len(args.models) == 2:
        a, b = (got[m] for m in MODELS)
        if a["ids"] != b["ids"]:
            raise ValueError("the 2 models do not hold the same image order")
        cross = a["full"] @ b["full"].T / a["full"].shape[1]
        np.savez(DATA / "cross.npz", cross=cross)
        ga, gb = len(a["head"]["layers"]), len(b["head"]["layers"])
        keep = [i for i, t in enumerate(a["tokens"]) if i != 1]
        rows = []
        for j, i in enumerate(keep):
            block = cross[i * ga:(i + 1) * ga, j * gb:(j + 1) * gb]
            path = monotone_path(block)
            rows += [{"gemma_token": a["tokens"][i], "qwen_token": b["tokens"][j], "gemma_layer": a["head"]["layers"][k], "qwen_layer": b["head"]["layers"][q],
                      "r": block[k, q]} for k, q in enumerate(path)]
        pd.DataFrame(rows).to_csv(DATA / "cross_path.csv", index=False)
    t = {NAMES[m]: got[m]["table"] for m in args.models}
    plot_grid(t, "r2", OUT / "free_decoder.png", "Margin decoder at each cell",
              "Out-of-fold R2 of a ridge decoder from the residual at a token and layer to the answer margin, with 5 folds by NTA, over 591 images. No lens is used.",
              ["F10.9", "decoder"], 0, 1, "out-of-fold R2", False)
    plot_grid(t, "gain", OUT / "free_gain.png", "Margin information beyond the scene block",
              "Out-of-fold R2 of scene plus cell minus R2 of the scene block alone. The scene block is the patch residuals at all saved layers, pooled to a 4 x 4 grid.",
              ["F10.9", "scene block"], -0.5, 0.5, "gain in out-of-fold R2", True)
    plot_grid({NAMES[m]: got[m]["commonality"] for m in args.models}, "unique_margin", OUT / "free_commonality.png", "Unique margin share of each cell RDM",
              "Rank variance of the image RDM at a token and layer that the margin RDM explains and the scene RDM does not.", ["F10.9", "RSA"], -0.05, 0.05, "unique margin R2", True)
    print("[f10.9] finished", flush=True)


if __name__ == "__main__":
    main()
