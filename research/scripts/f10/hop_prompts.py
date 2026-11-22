#!/usr/bin/env python
"""F10.26: the judgment across 14 prompts (7 adjectives x 2 nouns) in 3 models. CPU only.

A prompt is a pair (adjective, noun). The question is "Is this {noun} {adjective}? Answer with
one word: yes or no." The 42 stores of F10.25 hold the same 591 images in 197 NTAs.

The script has 3 stages. Stage 1 and 2 run on SLURM. Stage 3 reads only the summaries.

| Stage | Command | Output |
|-------|---------|--------|
| 1 | `summary --model M --adjective A --noun N` | `data/f10/prompts/{store}/`: the NTA mean readouts at the 6 named positions, the residuals at those positions, the no-image readout, the sense score, the lens margin at the last token, and the answers |
| 2 | `patches --model M` | `data/f10/prompts/{model}-patches/mean_patch.npy`: the mean patch at each layer, from the `cool` store |
| 3 | `analyze` | `research/figures/f10/prompts/*.csv`, `*.png`, `*.pdf`, `*.json` |

Stage 1 loads each shard file one time, in sequence, with no memory map (the NFS rule of F10.25).

| File in `data/f10/prompts/{store}/` | Shape | Content |
|-------------------------------------|-------|---------|
| `nta_{position}.npy` | [NTAs, layers, vocab] | NTA mean of the z-scored readout. Gemma readouts are uncapped first |
| `residual_{position}.npy` | [images, layers, d] | the raw residual |
| `full_mean.npy`, `text_reads.npy` | [6, layers, vocab] | the mean readout over images, and the readout with no image. NaN at `after_image` |
| `text_residuals.npy` | [6, layers, d] | the residuals with no image |
| `sense.npy`, `sense_text.npy` | [images, 6, layers], [6, layers] | the F10.17 score: social words minus temperature words |
| `lens_margin_final.npy` | [images, layers] | the yes-no margin of the lens readout at the last token (from the answer store) |
| `frame.parquet` | 1 row for each image | `image_idx`, `nta2020`, `borough`, `top`, `outcome`, `margin`, `margin_f1024` |

The positions keep the manifest names: `after_image`, `first_text`, `place` (the noun), `after_place`
(the adjective), `after_cool` (`?`) and `final`. The tables use the labels `noun`, `adjective`, `qm`, `final`.

The margin pools 8 answer forms. `margin_f1024` is the margin of the file that F10.24 read, which pooled
4 forms for the `cool` prompt of Gemma-4-31B and Qwen3.5-9B. Task 0 uses it to reproduce F10.24.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py summary --model gemma-4-31b --adjective ugly --noun scene
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py patches --model gemma-4-31b
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py analyze
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_prompts.py figures
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from analyze_logit import SEED, cluster_r, holm  # noqa: E402
from hop_borough import BOROUGHS, split_within, two_sample_t  # noqa: E402
from hop_free import REPEATS, folds, kernel, oof_mixed, oof_predict, r2, ridge_fit  # noqa: E402
from hop_outcomes import outcome  # noqa: E402
from hop_sense import CAP, SOCIAL, TEMPERATURE  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
SHORT = {"gemma-4-31b": "G", "qwen3.5-9b": "Q9", "qwen3.6-27b": "Q27"}
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
ADJECTIVES = ("cool", "nice", "ordinary", "pretty", "ugly", "fancy", "urban")
NOUNS = ("place", "scene")
PROMPTS = tuple((a, n) for n in NOUNS for a in ADJECTIVES)
POSITIONS = ("after_image", "first_text", "place", "after_place", "after_cool", "final")
LABEL = {"after_image": "after_image", "first_text": "first_text", "place": "noun", "after_place": "adjective", "after_cool": "qm", "final": "final"}
DECODE_POSITIONS = ("place", "after_place", "after_cool", "final")
DATA = ROOT / "data/f10/prompts"
FIG = ROOT / "research/figures/f10/prompts"
HOP_LOGIT = ROOT / "data/f10/hop_logit"
HOP_FREE = ROOT / "data/f10/hop_free"
F1024_MODELS = ROOT / "research/figures/f10/models"
F1024_ADJ = ROOT / "research/figures/f10/adjectives"
ZOO = {"gemma-4-31b": "Gemma-4-31B-it", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
FORMS = ["Yes", "yes", " Yes", " yes", "No", "no", " No", " no"]
N_LAYERS = {"gemma-4-31b": 59, "qwen3.5-9b": 31, "qwen3.6-27b": 63}  # as in hop_depth.py, without its torch import
TOP, FAR = 100, 1000
N_WORDS = 15
N_SPLITS = 40
LIST = 50
WORKERS = 6


# ---------------------------------------------------------------- the prompt and its stores


def prompt_name(adjective: str, noun: str) -> str:
    return f"{noun} {adjective}"


def store_name(model: str, adjective: str, noun: str) -> str:
    """Give the store of a prompt. This is the 1 place that maps a prompt to a store."""
    if adjective not in ADJECTIVES or noun not in NOUNS or model not in MODELS:
        raise ValueError(f"unknown prompt {model} {noun} {adjective}")
    if noun == "place":
        return model if adjective == "cool" else f"{model}-{adjective}"
    return f"{model}-scene-{adjective}"


def answer_store(model: str, adjective: str, noun: str) -> str:
    """Give the answer store with the 8-form margin. The `cool` answers of 2 models are in a `-forms8` folder."""
    if (adjective, noun) == ("cool", "place") and model in ("gemma-4-31b", "qwen3.5-9b"):
        return f"{model}-forms8"
    return store_name(model, adjective, noun)


def answer_store_f1024(model: str, adjective: str, noun: str) -> str:
    """Give the answer store that F10.24 read. It differs from `answer_store` for 2 `cool` prompts only."""
    return store_name(model, adjective, noun)


def margin_from(path: Path, forms: int) -> pd.DataFrame:
    a = pd.read_parquet(path)
    have = [f"lp_{f}" for f in FORMS if f"lp_{f}" in a.columns]
    if len(have) != forms:
        raise ValueError(f"{path}: {len(have)} answer forms, expected {forms}")
    return a[["image_idx", "margin"]]


def frame(model: str, adjective: str, noun: str) -> pd.DataFrame:
    """Give 1 row for each image, in the order of `images.parquet`. The margin is empty for `unable` (D167)."""
    images = pd.read_parquet(ALL_ROOT / "images.parquet")[["image_idx", "nta2020", "borough"]]
    top = pd.concat(map(pd.read_parquet, sorted((ALL_ROOT / store_name(model, adjective, noun) / "answers").glob("answers-*.parquet"))))[["image_idx", "top"]]
    m8 = margin_from(HOP_LOGIT / answer_store(model, adjective, noun) / "answers.parquet", 8)
    old = HOP_LOGIT / answer_store_f1024(model, adjective, noun) / "answers.parquet"
    m_old = pd.read_parquet(old)[["image_idx", "margin"]].rename(columns={"margin": "margin_f1024"})
    out = images.merge(top, on="image_idx", validate="1:1").merge(m8, on="image_idx", validate="1:1").merge(m_old, on="image_idx", validate="1:1")
    out["outcome"] = out["top"].map(outcome)
    out.loc[out["outcome"] == "unable", ["margin", "margin_f1024"]] = np.nan
    if len(out) != 591:
        raise ValueError(f"{len(out)} images, expected 591")
    return out.reset_index(drop=True)


def softcap_of(model: str) -> float | None:
    return json.loads((HOP_LOGIT / model / "manifest.json").read_text()).get("softcap")


def uncap(x: np.ndarray, cap: float | None) -> np.ndarray:
    """Give the readout before the softcap. A Gemma store holds cap * tanh(read / cap)."""
    if cap is None:
        return x
    return cap * np.arctanh(np.clip(x / cap, -1 + 1e-7, 1 - 1e-7))


def zscore(x: np.ndarray) -> np.ndarray:
    """Give the z score of each readout over the whole vocabulary (the last axis)."""
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def margin_of(x: np.ndarray) -> np.ndarray:
    """Give the pooled log odds of the 4 yes forms against the 4 no forms, on the last axis."""
    half = x.shape[-1] // 2
    return logsumexp(x[..., :half], axis=-1) - logsumexp(x[..., half:], axis=-1)


def check_positions(manifest: dict, adjective: str, noun: str) -> list[int]:
    """Give the tail index of each named position, after a check of the tokens."""
    positions, tail = manifest["positions"], manifest["tail_tokens"]
    pos = [positions.index(p) for p in POSITIONS]
    want = {"place": f" {noun}", "after_place": f" {adjective}", "after_cool": "?"}
    for name, token in want.items():
        got = tail[positions.index(name)]
        if got != token:
            raise ValueError(f"{manifest.get('model')}: {name} holds {got!r}, expected {token!r}")
    return pos


# ---------------------------------------------------------------- stage 1: the summary of 1 store


def summary(model: str, adjective: str, noun: str, shards: int | None = None) -> None:
    name = store_name(model, adjective, noun)
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    st = open_store(name, ALL_ROOT)
    man = st.manifest
    pos = check_positions(man, adjective, noun)
    layers = man["layers"]
    cap = softcap_of(model)
    fr = frame(model, adjective, noun)
    codes = pd.factorize(fr["nta2020"])[0]
    n_nta = codes.max() + 1
    words = st.vocab["token"].str.strip().str.lower()
    hot = np.flatnonzero(words.isin(TEMPERATURE).to_numpy())
    soc = np.flatnonzero(words.isin(SOCIAL).to_numpy())
    reads = sorted((st.path / "reads").glob("reads-*.npy"))
    resid = sorted((st.path / "residuals").glob("residuals-*.npy"))
    if len(reads) != 12 or len(resid) != 12:
        raise ValueError(f"{name}: {len(reads)} reads files and {len(resid)} residual files, expected 12")
    if shards:
        reads, resid = reads[:shards], resid[:shards]
    n_pos, n_lay, n_voc, d = len(POSITIONS), len(layers), len(words), int(man["d"])
    full = np.zeros((n_pos, n_lay, n_voc), np.float64)
    nta = np.zeros((n_nta, n_pos, n_lay, n_voc), np.float32)
    sense = np.zeros((len(fr), n_pos, n_lay), np.float32)
    res = np.zeros((len(fr), n_pos, n_lay, d), np.float32)
    row = 0
    for f, g in zip(reads, resid):
        x = np.load(f)  # 1 sequential load of the whole shard
        k = x.shape[0]
        rows = np.arange(row, row + k)
        z = zscore(uncap(np.asarray(x[:, pos], np.float64), cap))
        del x
        full += z.sum(0)
        for c in np.unique(codes[rows]):
            nta[c] += z[codes[rows] == c].sum(0).astype(np.float32)
        sense[rows] = z[..., soc].mean(-1) - z[..., hot].mean(-1)
        del z
        r = np.load(g)
        res[rows] = r[:, pos]
        del r
        row += k
        print(f"[f10.26] {name}: {row} images", flush=True)
    if row != len(fr) and not shards:
        raise ValueError(f"{name}: read {row} images, expected {len(fr)}")
    counts = np.bincount(codes, minlength=n_nta)
    text = np.load(st.path / "text_reads.npy")[pos]
    text = zscore(uncap(np.asarray(text, np.float64), cap))
    text[0] = np.nan
    text_res = np.load(st.path / "text_residuals.npy")[pos]
    # The lens margin at the last token, from the answer store (F10.7 metrics).
    logit = HOP_LOGIT / answer_store(model, adjective, noun)
    parts = [np.load(p) for p in sorted((logit / "metrics").glob("metrics-*.npz"))]
    lens = np.concatenate([p["lens"] for p in parts])
    if len(lens) != len(fr):
        raise ValueError(f"{logit}: {len(lens)} images in the metrics")
    lens_margin = margin_of(lens[:, -1].astype(np.float64)).astype(np.float32)
    tag = "_smoke" if shards else ""
    for i, p in enumerate(POSITIONS):
        np.save(out / f"nta_{p}{tag}.npy", (nta[:, i] / counts[:, None, None]).astype(np.float32))
        np.save(out / f"residual_{p}{tag}.npy", res[:, i])
    np.save(out / f"full_mean{tag}.npy", (full / max(row, 1)).astype(np.float32))
    np.save(out / f"text_reads{tag}.npy", text.astype(np.float32))
    np.save(out / f"text_residuals{tag}.npy", np.asarray(text_res, np.float32))
    np.save(out / f"sense{tag}.npy", sense)
    np.save(out / f"sense_text{tag}.npy", (text[..., soc].mean(-1) - text[..., hot].mean(-1)).astype(np.float32))
    np.save(out / f"lens_margin_final{tag}.npy", lens_margin)
    fr.assign(codes=codes).to_parquet(out / f"frame{tag}.parquet", index=False)
    (out / f"meta{tag}.json").write_text(json.dumps({"version": __version__, "store": name, "model": model, "adjective": adjective, "noun": noun,
                                                     "positions": list(POSITIONS), "tokens": [man["tail_tokens"][i] for i in pos], "layers": layers,
                                                     "n_images": int(row), "vocab": n_voc, "d": d, "softcap": cap, "answer_store": answer_store(model, adjective, noun),
                                                     "complete": bool(not shards)}, indent=1) + "\n")
    print(f"[f10.26] {name}: saved to {out}", flush=True)


def patches(model: str, shards: int | None = None) -> None:
    """Save the mean patch at each layer of the `cool` store, [images, layers, d]. The patches are equal for all prompts."""
    st = open_store(model, ALL_ROOT)
    files = sorted((st.path / "patches").glob("residuals-*.npy"))
    if len(files) != 12:
        raise ValueError(f"{model}: {len(files)} patch files, expected 12")
    if shards:
        files = files[:shards]
    got = []
    for f in files:
        x = np.load(f)  # [images, patches, layers, d]
        got.append(x.mean(1).astype(np.float32))
        del x
        print(f"[f10.26] {model}: {f.name}", flush=True)
    out = DATA / f"{model}-patches"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if shards else ""
    np.save(out / f"mean_patch{tag}.npy", np.concatenate(got))
    print(f"[f10.26] {model}: saved the mean patch to {out}", flush=True)


# ---------------------------------------------------------------- stage 3: the readers


class Summary:
    """The summary of 1 store. Each array loads on its first use."""

    def __init__(self, model: str, adjective: str, noun: str):
        self.model, self.adjective, self.noun = model, adjective, noun
        self.name = store_name(model, adjective, noun)
        self.dir = DATA / self.name
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if not self.meta.get("complete"):
            raise ValueError(f"{self.name}: the summary is not complete")
        self.layers: list[int] = self.meta["layers"]
        self.frame = pd.read_parquet(self.dir / "frame.parquet")
        self.codes = self.frame["codes"].to_numpy()
        self.n_nta = int(self.codes.max() + 1)
        self.borough_nta = self.frame.groupby("codes")["borough"].first().to_numpy()
        self._cache: dict = {}

    @property
    def prompt(self) -> str:
        return prompt_name(self.adjective, self.noun)

    def _load(self, key: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = np.load(self.dir / f"{key}.npy")
        return self._cache[key]

    def nta(self, position: str) -> np.ndarray:
        return self._load(f"nta_{position}")

    def residual(self, position: str) -> np.ndarray:
        return self._load(f"residual_{position}")

    @property
    def full(self) -> np.ndarray:
        return self._load("full_mean")

    @property
    def text(self) -> np.ndarray:
        return self._load("text_reads")

    @property
    def sense(self) -> np.ndarray:
        return self._load("sense")

    @property
    def sense_text(self) -> np.ndarray:
        return self._load("sense_text")

    @property
    def lens_margin(self) -> np.ndarray:
        return self._load("lens_margin_final")

    def drop(self, *keys: str) -> None:
        for k in keys:
            self._cache.pop(k, None)

    def margin(self, old: bool = False) -> np.ndarray:
        return self.frame["margin_f1024" if old else "margin"].to_numpy(float)

    def nta_margin(self, old: bool = False) -> np.ndarray:
        """Give the NTA mean margin over the verdict images. NaN where an NTA has no verdict."""
        y = self.margin(old)
        ok = np.isfinite(y)
        s = np.bincount(self.codes[ok], y[ok], self.n_nta)
        n = np.bincount(self.codes[ok], minlength=self.n_nta)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / n


def vocab_of(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Give the store words and the mask of plain alphabetic words of 3 letters or more."""
    v = open_store(model, ALL_ROOT).vocab["token"].str.strip()
    return v.to_numpy().astype(str), (v.str.isalpha() & (v.str.len() >= 3)).to_numpy()


def mid_layer(model: str, layers: list[int]) -> int:
    """Give the saved layer nearest to relative depth 0.5, as `hop_models.near` does."""
    return min(layers, key=lambda q: abs((q + 1) / (N_LAYERS[model] + 1) - 0.5))


def depth_of(model: str, layer: int) -> float:
    return round((layer + 1) / (N_LAYERS[model] + 1), 3)


def first_run(layers: list[int], ok: list[bool]) -> list[int]:
    run: list[int] = []
    for layer, passed in zip(layers, ok):
        if passed:
            run.append(layer)
        elif run:
            break
    return run


def load_all() -> dict[tuple[str, str, str], Summary]:
    return {(m, a, n): Summary(m, a, n) for m in MODELS for a, n in PROMPTS}


def mean_patch(model: str) -> np.ndarray:
    return np.load(DATA / f"{model}-patches" / "mean_patch.npy")


# ---------------------------------------------------------------- the measures


def axis1(c: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Give the share of variance on axis 1, the NTA scores and the word loadings, from NTA means [NTAs, words]."""
    c = c - c.mean(0, keepdims=True)
    w, u = np.linalg.eigh(c @ c.T)
    w = w.clip(min=0)
    share = float(w[-1] / max(w.sum(), 1e-12))
    score = u[:, -1] * np.sqrt(w[-1])
    load = (u[:, -1] @ c) / max(np.sqrt(w[-1]), 1e-9)
    return share, score, load


def embedded_counts(full: np.ndarray, text: np.ndarray) -> tuple[int, int, int]:
    """Give the words in the top 100 of both readouts, the words the image raises from past rank 1,000, and the reverse (F10.21)."""
    order_i = (-full).argsort().argsort() + 1
    order_t = (-text).argsort().argsort() + 1
    emb = int(((order_i <= TOP) & (order_t <= TOP)).sum())
    raised = int(((order_i <= TOP) & (order_t >= FAR)).sum())
    lowered = int(((order_t <= TOP) & (order_i >= FAR)).sum())
    return emb, raised, lowered


def subset_kernel(K: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Give the kernel of a subset of images, centred again on the subset and scaled to trace n."""
    k = K[np.ix_(idx, idx)]
    n = len(idx)
    H = np.eye(n) - 1.0 / n
    k = H @ k @ H
    return k * (n / np.trace(k))


def gain_rows(S: Summary, patch: np.ndarray, old: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the gain over the mean patch at the noun, the adjective and `?` at each layer (F10.9), and the mean-patch R2 at each layer."""
    y_all = S.margin(old)
    idx = np.flatnonzero(np.isfinite(y_all))
    y = y_all[idx]
    groups = S.frame["nta2020"].to_numpy()[idx]
    codes = pd.factorize(groups)[0]
    reps = [folds(groups, seed=SEED + i) for i in range(REPEATS)]
    splits = reps[0]
    r = lambda pred: r2(y, pred, splits)
    kern = [kernel(patch[idx, b].astype(np.float64)) for b in range(len(S.layers))]
    depth = []
    for b, layer in enumerate(S.layers):
        pred = np.mean([oof_predict(kern[b], y, s) for s in reps], axis=0)
        depth.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "n": len(idx), "layer": layer, "r2": r(pred)})
    scene = sum(kern) / len(kern)
    scene_pred = np.mean([oof_predict(scene, y, s) for s in reps], axis=0)
    r2_scene = r(scene_pred)
    rows = []
    for p in ("place", "after_place", "after_cool"):
        res = S.residual(p)
        for b, layer in enumerate(S.layers):
            K = kernel(res[idx, b].astype(np.float64))
            pred = np.mean([oof_predict(K, y, s) for s in reps], axis=0)
            joint = np.mean([oof_mixed(scene, K, y, s) for s in reps], axis=0)
            gap = np.bincount(codes, (y - scene_pred) ** 2 - (y - joint) ** 2) / np.bincount(codes)
            t, pv = stats.ttest_1samp(gap, 0.0)
            rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "n": len(idx), "position": LABEL[p], "layer": layer,
                         "r2_cell": r(pred), "r2_scene": r2_scene, "r2_joint": r(joint), "gain": r(joint) - r2_scene, "t": float(t), "p": float(pv)})
        S.drop(f"residual_{p}")
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())  # the family is the cells of 1 store
    return out, pd.DataFrame(depth)


def axis_rows(S: Summary, position: str = "after_place") -> pd.DataFrame:
    """Give axis 1 at each layer: its share, its r with the cool-list score, its r with the margin, and the cool-list sense shift."""
    rows = []
    pi = POSITIONS.index(position)
    nta = S.nta(position)
    y_nta = S.nta_margin()
    ok = np.isfinite(y_nta)
    for b, layer in enumerate(S.layers):
        share, score, _ = axis1(nta[:, b].astype(np.float64))
        v = S.sense[:, pi, b].astype(np.float64)
        s_nta = np.bincount(S.codes, v, S.n_nta) / np.bincount(S.codes, minlength=S.n_nta)
        r_list = float(np.corrcoef(s_nta, score)[0, 1])
        r_m, p_m = stats.pearsonr(score[ok], y_nta[ok])
        t0 = float(S.sense_text[pi, b])
        rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "position": LABEL[position], "layer": layer,
                     "depth": depth_of(S.model, layer), "share1": share, "r_axis1_coollist": abs(r_list), "r_axis1_margin": r_m, "p_axis1_margin": p_m,
                     "n_nta": int(ok.sum()), "sense_no_image": t0, "sense_mean": float(v.mean()), "sense_share_above": float((v > t0).mean()),
                     "r_coollist_margin": cluster_r(v[np.isfinite(S.margin())], S.margin()[np.isfinite(S.margin())], S.frame["nta2020"].to_numpy()[np.isfinite(S.margin())])[0]})
    return pd.DataFrame(rows)


def axis_words(S: Summary, position: str, layer: int, words: np.ndarray, clean: np.ndarray, n: int = N_WORDS) -> tuple[pd.DataFrame, float]:
    """Give the top words at each end of axis 1, with the axis turned so that the high end goes with yes."""
    b = S.layers.index(layer)
    share, score, load = axis1(S.nta(position)[:, b].astype(np.float64))
    y = S.nta_margin()
    ok = np.isfinite(y)
    r = float(np.corrcoef(score[ok], y[ok])[0, 1])
    if r < 0:
        score, load, r = -score, -load, -r
    v = np.where(clean, load, np.nan)
    hi = np.argsort(-np.nan_to_num(v, nan=-np.inf))[:n]
    lo = np.argsort(np.nan_to_num(v, nan=np.inf))[:n]
    rows = []
    for rank, i in enumerate(hi, 1):
        rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "position": LABEL[position], "layer": layer, "end": "yes end", "rank": rank, "word": words[i], "loading": float(load[i])})
    for rank, i in enumerate(lo, 1):
        rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "position": LABEL[position], "layer": layer, "end": "no end", "rank": rank, "word": words[i], "loading": float(load[i])})
    return pd.DataFrame(rows), r


def embedded_rows(S: Summary) -> pd.DataFrame:
    rows = []
    for pi, p in enumerate(POSITIONS):
        if p == "after_image":
            continue
        for b, layer in enumerate(S.layers):
            e, ra, lo = embedded_counts(S.full[pi, b], S.text[pi, b])
            rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "position": LABEL[p], "layer": layer,
                         "depth": depth_of(S.model, layer), "embedded": e, "raised": ra, "lowered": lo})
    return pd.DataFrame(rows)


def verdict_rows(S: Summary) -> pd.DataFrame:
    """Give r of the lens margin at the last token with the answer margin at each layer, on the verdict images."""
    y = S.margin()
    keep = np.isfinite(y)
    rows = []
    for b, layer in enumerate(S.layers):
        r, t, p = cluster_r(S.lens_margin[keep, b].astype(np.float64), y[keep], S.frame["nta2020"].to_numpy()[keep])
        rows.append({"model": S.model, "adjective": S.adjective, "noun": S.noun, "prompt": S.prompt, "n": int(keep.sum()), "layer": layer, "r_last_token": r, "t": t, "p": p})
    return pd.DataFrame(rows)


def borough_t(nta: np.ndarray, borough: np.ndarray) -> dict[str, np.ndarray]:
    """Give the C3 t vector of each borough: the NTAs of the borough against the others, by word."""
    return {b: two_sample_t(nta, borough == b) for b in BOROUGHS}


def top_set(v: np.ndarray, clean: np.ndarray, n: int = LIST) -> set[int]:
    return set(np.argsort(-np.where(clean, v, -np.inf))[:n].tolist())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


# ---------------------------------------------------------------- decoders (task 7)


def dual_weights(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a ridge of y on X over `rows`. Give the dual weights, the unit primal weight, and the mean of y."""
    Xc = X[rows] - X[rows].mean(0, keepdims=True)
    K = Xc @ Xc.T
    K *= len(rows) / np.trace(K)
    mean = y[rows].mean()
    alpha, _ = ridge_fit(K, y[rows] - mean)
    w = Xc.T @ alpha
    return alpha, w / max(np.linalg.norm(w), 1e-12), mean


def prompt_decoders(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, splits: list) -> dict:
    """Fit 1 ridge decoder for each column of Y (NaN allowed) on the rows of X, in NTA folds.

    Gives the out-of-fold prediction of each column, the unit weight of each column on each training fold,
    the unit weight of the fit on all rows, and the out-of-fold R2.
    """
    n, p = Y.shape
    pred = np.full((n, p), np.nan)
    fold_w = np.zeros((len(splits), X.shape[1], p))
    full_w = np.zeros((X.shape[1], p))
    for j in range(p):
        ok = np.isfinite(Y[:, j])
        for f, (tr, te) in enumerate(splits):
            rows = tr[ok[tr]]
            alpha, w, mean = dual_weights(X, Y[:, j], rows)
            fold_w[f, :, j] = w
            Xc_tr = X[rows] - X[rows].mean(0, keepdims=True)
            Kte = (X[te] - X[rows].mean(0, keepdims=True)) @ Xc_tr.T * (len(rows) / np.trace(Xc_tr @ Xc_tr.T))
            pred[te, j] = Kte @ alpha + mean
        _, full_w[:, j], _ = dual_weights(X, Y[:, j], np.flatnonzero(ok))
    r2s = []
    for j in range(p):
        ok = np.isfinite(Y[:, j])
        sse = np.nansum((Y[ok, j] - pred[ok, j]) ** 2)
        sst = sum(np.sum((Y[te[ok[te]], j] - Y[tr[ok[tr]], j].mean()) ** 2) for tr, te in splits)
        r2s.append(float(1 - sse / sst))
    return {"pred": pred, "fold_w": fold_w, "full_w": full_w, "r2": np.array(r2s)}


def shared_direction(W: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Give the first k left singular vectors of the unit weights [d, prompts], and the share of each singular value."""
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    return u[:, :k], s ** 2 / (s ** 2).sum()


def loo_rows(X: np.ndarray, Y: np.ndarray, names: list[str], dec: dict, splits: list, ks: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Leave 1 prompt out: the shared direction of the other prompts, fit in the training fold, predicts the held-out margin."""
    n, p = Y.shape
    rows = []
    for j in range(p):
        others = [q for q in range(p) if q != j]
        ok = np.isfinite(Y[:, j])
        for k in ks:
            pred = np.full(n, np.nan)
            for f, (tr, te) in enumerate(splits):
                u, _ = shared_direction(dec["fold_w"][f][:, others], k)
                tr_ok = tr[ok[tr]]
                s_tr, s_te = X[tr_ok] @ u, X[te] @ u
                A = np.column_stack([np.ones(len(tr_ok)), s_tr])
                beta = np.linalg.lstsq(A, Y[tr_ok, j], rcond=None)[0]
                pred[te] = np.column_stack([np.ones(len(te)), s_te]) @ beta
            sse = np.nansum((Y[ok, j] - pred[ok]) ** 2)
            sst = sum(np.sum((Y[te[ok[te]], j] - Y[tr[ok[tr]], j].mean()) ** 2) for tr, te in splits)
            r = float(np.corrcoef(Y[ok, j], pred[ok])[0, 1])
            rows.append({"prompt": names[j], "k": k, "n": int(ok.sum()), "r2_shared": float(1 - sse / sst), "r_shared": r, "r2_own": float(dec["r2"][j])})
    return pd.DataFrame(rows)


def rank_rows(Y: np.ndarray, dec: dict, splits: list, ks: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Give the out-of-fold R2 of a rank-k fit of all columns of Y (reduced-rank regression on the ridge fits). Y must be complete."""
    n, p = Y.shape
    rows = []
    for k in list(ks) + [p]:
        pred = np.zeros((n, p))
        for f, (tr, te) in enumerate(splits):
            # The fitted values on the training rows give the rank-k subspace.
            fit_tr = np.column_stack([dec["pred_train"][f][:, j] for j in range(p)])
            mean = fit_tr.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(fit_tr - mean, full_matrices=False)
            V = vt[:k].T
            pred[te] = mean + (dec["pred"][te] - mean) @ V @ V.T
        for j in range(p):
            sse = np.sum((Y[:, j] - pred[:, j]) ** 2)
            sst = sum(np.sum((Y[te, j] - Y[tr, j].mean()) ** 2) for tr, te in splits)
            rows.append({"column": j, "k": k, "r2": float(1 - sse / sst)})
    return pd.DataFrame(rows)


def train_fits(X: np.ndarray, Y: np.ndarray, splits: list) -> list[np.ndarray]:
    """Give the fitted values on the training rows of each fold, for each complete column of Y."""
    out = []
    for tr, te in splits:
        fit = np.zeros((len(tr), Y.shape[1]))
        for j in range(Y.shape[1]):
            alpha, w, mean = dual_weights(X, Y[:, j], tr)
            Xc = X[tr] - X[tr].mean(0, keepdims=True)
            K = Xc @ Xc.T
            K *= len(tr) / np.trace(K)
            fit[:, j] = K @ alpha + mean
        out.append(fit)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


# ---------------------------------------------------------------- the lens on CPU (task 7, step 4)


def norm_weight(model: str) -> np.ndarray:
    """Give the weight of the final norm of the language model, from the safetensors files."""
    from safetensors import safe_open

    d = Path("/share/ju/matt/zoo/models") / ZOO[model]
    index = next(d.glob("*.safetensors.index.json"))
    wm = json.loads(index.read_text())["weight_map"]
    key = "model.language_model.norm.weight"
    with safe_open(d / wm[key], framework="pt") as f:
        return f.get_tensor(key).float().numpy()  # the file holds bfloat16, which numpy cannot read


def lens_words(model: str, layer: int, u: np.ndarray, words: np.ndarray, clean: np.ndarray, n: int = N_WORDS) -> tuple[pd.DataFrame, np.ndarray]:
    """Put a residual direction through the J++ lens, the final norm weight and the unembedding rows of the store words."""
    import torch

    head = torch.load(ROOT / f"outputs/_monocle/jlens/jpp/{model}/lens.pt", map_location="cpu", weights_only=True, mmap=True)
    J = head["parameters"]["jacobians"][layer].float().numpy()
    t = J @ u
    g = norm_weight(model)
    W = np.load(HOP_LOGIT / model / "unembed_store.npy")
    logits = W @ (t * g)
    v = np.where(clean, logits, np.nan)
    hi = np.argsort(-np.nan_to_num(v, nan=-np.inf))[:n]
    lo = np.argsort(np.nan_to_num(v, nan=np.inf))[:n]
    rows = [{"end": "high", "rank": r, "word": words[i], "logit": float(logits[i])} for r, i in enumerate(hi, 1)]
    rows += [{"end": "low", "rank": r, "word": words[i], "logit": float(logits[i])} for r, i in enumerate(lo, 1)]
    return pd.DataFrame(rows), logits


def lens_check(model: str, S: Summary, layer: int) -> float:
    """Give r between the CPU readout of the no-image residual at `?` and the stored no-image readout at that cell."""
    import torch

    head = torch.load(ROOT / f"outputs/_monocle/jlens/jpp/{model}/lens.pt", map_location="cpu", weights_only=True, mmap=True)
    J = head["parameters"]["jacobians"][layer].float().numpy()
    pi, b = POSITIONS.index("after_cool"), S.layers.index(layer)
    h = np.load(S.dir / "text_residuals.npy")[pi, b].astype(np.float32)
    t = J @ h
    t = t / np.sqrt(np.mean(t ** 2) + 1e-6)
    logits = np.load(HOP_LOGIT / model / "unembed_store.npy") @ (t * norm_weight(model))
    return float(np.corrcoef(logits, S.text[pi, b])[0, 1])


# ---------------------------------------------------------------- stage 3: the tasks


def task0(A: dict, gains: pd.DataFrame, depth: pd.DataFrame, axes: pd.DataFrame, emb: pd.DataFrame, ver: pd.DataFrame, gains_old: pd.DataFrame, models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """Reproduce the F10.24 tables on the 3 old prompts with the new code."""
    rows = []

    def add(check, ref, val, note=""):
        rows.append({"check": check, "reference": float(ref), "value": float(val), "abs_diff": abs(float(ref) - float(val)), "note": note})

    old_adj = ("cool", "nice", "ordinary")
    corr = pd.read_csv(F1024_ADJ / "q1_margin_corr.csv")
    shares = pd.read_csv(F1024_ADJ / "q1_outcome_shares.csv")
    bor = pd.read_csv(F1024_MODELS / "q5_borough_margins.csv")
    sense = pd.read_csv(F1024_ADJ / "q2_sense_axis.csv")
    gain_ref = pd.read_csv(F1024_ADJ / "q7_gain.csv")
    gain_ref = gain_ref[gain_ref["file"] == "gain.csv"]
    wc = pd.read_csv(F1024_MODELS / "q1_word_counts_by_layer.csv")
    bands = pd.read_csv(F1024_ADJ / "q6_phase_bands.csv")
    q8 = pd.read_csv(F1024_ADJ / "q8_verdict_lens.csv")
    for m in models:
        for a, b in itertools.combinations(old_adj, 2):
            Sa, Sb = A[(m, a, "place")], A[(m, b, "place")]
            x, y = Sa.nta_margin(old=True), Sb.nta_margin(old=True)
            ok = np.isfinite(x) & np.isfinite(y)
            ref = corr[(corr["model"] == m) & (corr["pair"] == f"{a}-{b}")]["r_nta"].iloc[0]
            # F10.24 took the NTA mean over the images with a verdict in both stores.
            xi, yi = Sa.margin(old=True), Sb.margin(old=True)
            oki = np.isfinite(xi) & np.isfinite(yi)
            nta = pd.DataFrame({"n": Sa.codes, "x": xi, "y": yi})[oki].groupby("n").mean()
            add(f"q1 margin r NTA {a}-{b} {m}", ref, np.corrcoef(nta["x"], nta["y"])[0, 1], "q1_margin_corr.csv r_nta")
        for a in old_adj:
            S = A[(m, a, "place")]
            ref = shares[(shares["model"] == m) & (shares["adjective"] == a)]["yes"].iloc[0]
            add(f"q1 yes share {a} {m}", ref, (S.frame["outcome"] == "yes").mean(), "q1_outcome_shares.csv")
            y = S.nta_margin(old=True)
            ok = np.isfinite(y)
            t = two_sample_t(y[ok][:, None], S.borough_nta[ok] == "Manhattan")[0]
            ref = bor[(bor["model"] == m) & (bor["adjective"] == a) & (bor["borough"] == "Manhattan")]["t_vs_rest"].iloc[0]
            add(f"q5 Manhattan t {a} {m}", ref, t, "q5_borough_margins.csv t_vs_rest")
            g = gains_old[(gains_old["model"] == m) & (gains_old["adjective"] == a) & (gains_old["noun"] == "place")]
            gr = gain_ref[(gain_ref["model"] == m) & (gain_ref["adjective"] == a)]
            add(f"q7 r2 scene {a} {m}", gr["r2_scene"].iloc[0], g["r2_scene"].iloc[0], "q7_gain.csv r2_scene, with the F10.24 margin file")
            for pos, tok in (("noun", " place"), ("adjective", f" {a}"), ("qm", "?")):
                add(f"q7 max gain at {pos} {a} {m}", gr[gr["token"] == tok]["gain"].max(), g[g["position"] == pos]["gain"].max(), "q7_gain.csv max over layers")
            s = sense[(sense["model"] == m) & (sense["adjective"] == a) & (sense["position"] == "adj") & (sense["layer"] == 24)].iloc[0]
            ax = axes[(axes["model"] == m) & (axes["adjective"] == a) & (axes["noun"] == "place") & (axes["layer"] == 24)].iloc[0]
            add(f"q2 axis 1 share adj L24 {a} {m}", s["share1"], ax["share1"], "q2_sense_axis.csv")
            add(f"q2 r axis 1 with cool-list score L24 {a} {m}", s["r_axis1"], ax["r_axis1_coollist"], "q2_sense_axis.csv")
            add(f"q2 sense share above no image L24 {a} {m}", s["share_above"], ax["sense_share_above"], "q2_sense_axis.csv")
            sv = axes[(axes["model"] == m) & (axes["adjective"] == a) & (axes["noun"] == "place")].sort_values("layer")
            run = first_run(sv["layer"].tolist(), ((sv["share1"] >= 0.40) & (sv["r_axis1_coollist"] >= 0.95)).tolist())
            br = bands[(bands["model"] == m) & (bands["adjective"] == a) & (bands["phase"] == "sense")].iloc[0]
            add(f"q6 sense band first layer {a} {m}", np.nan_to_num(br["first"], nan=-1), run[0] if run else -1, "q6_phase_bands.csv (-1 means no layer passes)")
            add(f"q6 sense band last layer {a} {m}", np.nan_to_num(br["last"], nan=-1), run[-1] if run else -1, "q6_phase_bands.csv")
            v = ver[(ver["model"] == m) & (ver["adjective"] == a) & (ver["noun"] == "place")]
            q = q8[(q8["model"] == m) & (q8["adjective"] == a)]
            add(f"q8 verdict r at the last saved layer {a} {m}", q.sort_values("layer")["r_last_token"].iloc[-1], v.sort_values("layer")["r_last_token"].iloc[-1], "q8_verdict_lens.csv (8-form margin in both)")
        for pos, lab, layer in (("noun", "place", 40), ("adjective", "cool", 36), ("qm", "question_mark", 40)):
            if m == "qwen3.5-9b":
                layer = {"noun": 16, "adjective": 24, "qm": 20}[pos]
            w = wc[(wc["model"] == m) & (wc["token"] == lab) & (wc["layer"] == layer)]
            e = emb[(emb["model"] == m) & (emb["adjective"] == "cool") & (emb["noun"] == "place") & (emb["position"] == pos) & (emb["layer"] == layer)]
            if len(w) and len(e):
                for col in ("embedded", "raised", "lowered"):
                    add(f"q1 {col} words at {pos} L{layer} cool {m}", w[col].iloc[0], e[col].iloc[0], "q1_word_counts_by_layer.csv (hop_models used the uncapped Gemma file)")
    out = pd.DataFrame(rows)
    return out


def task1(A: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Outcome shares by prompt and model, and by borough; chi-square of borough and of noun against the outcome."""
    t = pd.concat([S.frame.assign(model=S.model, adjective=S.adjective, noun=S.noun, prompt=S.prompt) for S in A.values()], ignore_index=True)
    keys = ["model", "noun", "adjective", "prompt"]
    rates = pd.concat([t.groupby(keys)["outcome"].value_counts(normalize=True).unstack(fill_value=0).assign(borough="all").reset_index(),
                       t.groupby(keys + ["borough"])["outcome"].value_counts(normalize=True).unstack(fill_value=0).reset_index()], ignore_index=True)
    rates = rates.reindex(columns=keys + ["borough", "yes", "no", "unable"], fill_value=0.0)
    counts = t.groupby(keys)["outcome"].value_counts().unstack(fill_value=0).reindex(columns=["yes", "no", "unable"], fill_value=0).reset_index()
    rates = rates.merge(counts.rename(columns={"yes": "n_yes", "no": "n_no", "unable": "n_unable"}), on=keys)
    rows = []
    for key, g in t.groupby(keys):
        table = pd.crosstab(g["borough"], g["outcome"])
        table = table.loc[:, table.sum() > 0]
        if table.shape[1] < 2:
            chi, p, dof = np.nan, 1.0, 0
        else:
            chi, p, dof, _ = stats.chi2_contingency(table)
        rows.append(dict(zip(keys, key)) | {"chi2": chi, "dof": dof, "p": p, "n_unable": int((g["outcome"] == "unable").sum()), "yes_share": (g["outcome"] == "yes").mean()})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    tests["family"] = len(tests)
    rows = []
    for (m, a), g in t.groupby(["model", "adjective"]):
        table = pd.crosstab(g["noun"], g["outcome"])
        table = table.loc[:, table.sum() > 0]
        if table.shape[1] < 2:
            chi, p, dof = np.nan, 1.0, 0
        else:
            chi, p, dof, _ = stats.chi2_contingency(table)
        yes = g.groupby("noun")["outcome"].apply(lambda s: (s == "yes").mean())
        rows.append({"model": m, "adjective": a, "chi2": chi, "dof": dof, "p": p, "yes_place": yes.get("place", np.nan), "yes_scene": yes.get("scene", np.nan),
                     "unable_place": int(((g["noun"] == "place") & (g["outcome"] == "unable")).sum()), "unable_scene": int(((g["noun"] == "scene") & (g["outcome"] == "unable")).sum())})
    noun = pd.DataFrame(rows)
    noun["p_holm"] = holm(noun["p"].to_numpy())
    noun["family"] = len(noun)
    return rates, tests, noun


RULES = {
    "scene code: mean-patch R2 at least .50": lambda r: r["r2_scene"] >= 0.50,
    "gain at the noun between .03 and .15": lambda r: 0.03 <= r["gain_noun"] <= 0.15,
    "step at the adjective: gain at the adjective above the gain at the noun, and Holm p below .05": lambda r: (r["gain_adjective"] > r["gain_noun"]) and (r["p_holm_adjective"] < 0.05),
    "gain at ? at least .05 and Holm p below .05": lambda r: (r["gain_qm"] >= 0.05) and (r["p_holm_qm"] < 0.05),
    "verdict band reached: lens margin r at least .95 at the last token": lambda r: r["verdict_reached"],
    "sense band (cool lists): axis 1 at least .40 and the cool-list score abs r at least .95, at the adjective": lambda r: r["sense_band_reached"],
    "axis 1 at the adjective holds at least .40 of the NTA variance, at mid depth": lambda r: r["share1_mid"] >= 0.40,
    "axis 1 does not follow the verdict: abs r with the margin below .20, at mid depth": lambda r: abs(r["r_axis1_margin_mid"]) < 0.20,
    "embedded: at least 50 of the top 100 words at the adjective are in the no-image top 100, at mid depth": lambda r: r["embedded_mid"] >= 50,
    "image raises no word at the adjective from past rank 1,000, at mid depth": lambda r: r["raised_mid"] == 0,
    "Manhattan above the rest on the margin, Holm over 42": lambda r: (r["manhattan_t"] > 0) and (r["manhattan_p_holm"] < 0.05),
    "Manhattan differs from the rest on the margin, either sign, Holm over 42": lambda r: r["manhattan_p_holm"] < 0.05,
}


def task2(A: dict, gains: pd.DataFrame, axes: pd.DataFrame, emb: pd.DataFrame, ver: pd.DataFrame, man: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The mechanism grid: 14 prompts x 3 models, and the count of cells in which each result holds."""
    rows = []
    for (m, a, n), S in A.items():
        g = gains[(gains["model"] == m) & (gains["adjective"] == a) & (gains["noun"] == n)]
        row = {"model": m, "adjective": a, "noun": n, "prompt": S.prompt, "n_verdict": int(np.isfinite(S.margin()).sum()), "yes_share": (S.frame["outcome"] == "yes").mean(),
               "unable": int((S.frame["outcome"] == "unable").sum()), "r2_scene": g["r2_scene"].iloc[0]}
        for pos in ("noun", "adjective", "qm"):
            h = g[g["position"] == pos]
            best = h.loc[h["gain"].idxmax()]
            row[f"gain_{pos}"] = best["gain"]
            row[f"gain_{pos}_layer"] = int(best["layer"])
            row[f"p_holm_{pos}"] = best["p_holm"]
        v = ver[(ver["model"] == m) & (ver["adjective"] == a) & (ver["noun"] == n)].sort_values("layer")
        run = first_run(v["layer"].tolist(), (v["r_last_token"] >= 0.95).tolist())
        row["verdict_reached"] = bool(run)
        row["verdict_first"] = run[0] if run else np.nan
        row["verdict_last"] = run[-1] if run else np.nan
        row["verdict_r_last_layer"] = v["r_last_token"].iloc[-1]
        row["verdict_r_peak"] = v["r_last_token"].max()
        ax = axes[(axes["model"] == m) & (axes["adjective"] == a) & (axes["noun"] == n)].sort_values("layer")
        run = first_run(ax["layer"].tolist(), ((ax["share1"] >= 0.40) & (ax["r_axis1_coollist"] >= 0.95)).tolist())
        row["sense_band_reached"] = bool(run)
        row["sense_first"] = run[0] if run else np.nan
        row["sense_last"] = run[-1] if run else np.nan
        mid = mid_layer(m, S.layers)
        row["mid_layer"] = mid
        am = ax[ax["layer"] == mid].iloc[0]
        row["share1_mid"] = am["share1"]
        row["share1_peak"] = ax["share1"].max()
        row["share1_peak_layer"] = int(ax.loc[ax["share1"].idxmax(), "layer"])
        row["r_axis1_margin_mid"] = am["r_axis1_margin"]
        row["r_axis1_coollist_mid"] = am["r_axis1_coollist"]
        e = emb[(emb["model"] == m) & (emb["adjective"] == a) & (emb["noun"] == n) & (emb["position"] == "adjective") & (emb["layer"] == mid)].iloc[0]
        row["embedded_mid"], row["raised_mid"], row["lowered_mid"] = int(e["embedded"]), int(e["raised"]), int(e["lowered"])
        mm = man[(man["model"] == m) & (man["adjective"] == a) & (man["noun"] == n)].iloc[0]
        row["manhattan_t"], row["manhattan_p_holm"] = mm["t"], mm["p_holm"]
        rows.append(row)
    grid = pd.DataFrame(rows)
    for name, rule in RULES.items():
        grid[name] = grid.apply(lambda r: bool(rule(r)), axis=1)
    counts = []
    for name in RULES:
        held = grid[name]
        counts.append({"result": name, "cells_held": int(held.sum()), "cells": len(grid), "generalizes": bool(held.all()),
                       "fails_in": "; ".join(f"{SHORT[r.model]} {r.prompt}" for r in grid[~held].itertuples()) if not held.all() else ""})
    return grid, pd.DataFrame(counts)


def task3(A: dict, axes: pd.DataFrame, words: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The words at each end of axis 1 at the adjective, and the test of the axis score against the margin."""
    lists, tests = [], []
    for (m, a, n), S in A.items():
        mid = mid_layer(m, S.layers)
        ax = axes[(axes["model"] == m) & (axes["adjective"] == a) & (axes["noun"] == n)]
        peak = int(ax.loc[ax["share1"].idxmax(), "layer"])
        for layer, which in ((mid, "mid"), (24, "L24"), (peak, "peak")):
            w, r = axis_words(S, "after_place", layer, *words[m])
            lists.append(w.assign(which=which))
        am = ax[ax["layer"] == mid].iloc[0]
        tests.append({"model": m, "adjective": a, "noun": n, "prompt": S.prompt, "layer": mid, "share1": am["share1"], "r": am["r_axis1_margin"],
                      "n_nta": int(am["n_nta"]), "t": am["r_axis1_margin"] * np.sqrt((am["n_nta"] - 2) / max(1 - am["r_axis1_margin"] ** 2, 1e-12)), "p": am["p_axis1_margin"],
                      "yes_share": (S.frame["outcome"] == "yes").mean()})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    tests["family"] = len(tests)
    return pd.concat(lists, ignore_index=True), tests


def pair_margin_rows(Sa: Summary, Sb: Summary, label: str) -> dict:
    """Give r of 2 margins on images and on NTA means, and the paired t of the NTA means, over the images with a verdict in both."""
    x, y = Sa.margin(), Sb.margin()
    ok = np.isfinite(x) & np.isfinite(y)
    r_img, p_img = stats.pearsonr(x[ok], y[ok])
    nta = pd.DataFrame({"n": Sa.codes, "x": x, "y": y})[ok].groupby("n").mean()
    r_nta, p_nta = stats.pearsonr(nta["x"], nta["y"])
    t_pair, p_pair = stats.ttest_rel(nta["x"], nta["y"])
    return {"pair": label, "n_images": int(ok.sum()), "r_image": r_img, "p_image": p_img, "n_nta": len(nta), "r_nta": r_nta, "p_nta": p_nta,
            "mean_a": float(nta["x"].mean()), "mean_b": float(nta["y"].mean()), "t_paired": t_pair, "p_paired": p_pair, "yes_a": (Sa.frame["outcome"] == "yes").mean(), "yes_b": (Sb.frame["outcome"] == "yes").mean()}


def task4(A: dict, dec_place: dict, words: dict) -> pd.DataFrame:
    """Is `ugly` the reverse of `pretty`? r of the margins, the cosine of the decoders at the noun, and the cosine of C3 at the adjective."""
    rows = []
    for m in MODELS:
        for n in NOUNS:
            for a, b in (("pretty", "ugly"), ("cool", "ordinary"), ("nice", "ugly")):
                Sa, Sb = A[(m, a, n)], A[(m, b, n)]
                row = {"model": m, "noun": n} | pair_margin_rows(Sa, Sb, f"{a}-{b}")
                mid = mid_layer(m, Sa.layers)
                d = dec_place[(m, n)]
                row["layer"] = mid
                row["cos_decoder_noun"] = cosine(d["w"][mid][a], d["w"][mid][b])
                bi = Sa.layers.index(mid)
                ta = borough_t(Sa.nta("after_place")[:, bi].astype(np.float64), Sa.borough_nta)
                tb = borough_t(Sb.nta("after_place")[:, bi].astype(np.float64), Sb.borough_nta)
                for bo in BOROUGHS:
                    row[f"cos_c3_{bo.replace(' ', '_')}"] = cosine(ta[bo], tb[bo])
                row["cos_c3_mean"] = float(np.mean([row[f"cos_c3_{bo.replace(' ', '_')}"] for bo in BOROUGHS]))
                rows.append(row)
    return pd.DataFrame(rows)


def task5(A: dict, dec_place: dict, dec_adj: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The `urban` margin by borough, the borough pattern of the other adjectives given `urban`, and `urban` against the `cool` sense axis."""
    bor_rows, part_rows, sense_rows = [], [], []
    for m in MODELS:
        for n in NOUNS:
            U = A[(m, "urban", n)]
            yu = U.nta_margin()
            ok = np.isfinite(yu)
            order = pd.Series({b: yu[ok][U.borough_nta[ok] == b].mean() for b in BOROUGHS}).sort_values(ascending=False)
            for rank, (b, v) in enumerate(order.items(), 1):
                t = two_sample_t(yu[ok][:, None], U.borough_nta[ok] == b)[0]
                bor_rows.append({"model": m, "noun": n, "borough": b, "rank": rank, "mean_nta_margin": v, "sd_units": (v - yu[ok].mean()) / yu[ok].std(ddof=1),
                                 "t_vs_rest": t, "p": 2 * stats.t.sf(abs(t), ok.sum() - 2), "yes_share": (U.frame.loc[U.frame["borough"] == b, "outcome"] == "yes").mean(), "n_nta": int((U.borough_nta[ok] == b).sum())})
            manh = (U.borough_nta == "Manhattan").astype(float)
            for a in ADJECTIVES:
                if a == "urban":
                    continue
                S = A[(m, a, n)]
                y = S.nta_margin()
                k = ok & np.isfinite(y)
                r_b = np.corrcoef(y[k], manh[k])[0, 1]
                r_u = np.corrcoef(y[k], yu[k])[0, 1]
                r_bu = np.corrcoef(manh[k], yu[k])[0, 1]
                partial = (r_b - r_u * r_bu) / np.sqrt((1 - r_u ** 2) * (1 - r_bu ** 2))
                # The borough as a factor: F before and after `urban` enters as a covariate.
                df = pd.DataFrame({"y": y[k], "u": yu[k], "b": U.borough_nta[k]})
                F0, p0 = stats.f_oneway(*[df.loc[df["b"] == b, "y"] for b in BOROUGHS])
                # residual of y on u, then F of the borough on the residual (ANCOVA with 1 covariate, approximate df)
                beta = np.polyfit(df["u"], df["y"], 1)
                res = df["y"] - np.polyval(beta, df["u"])
                F1, p1 = stats.f_oneway(*[res[df["b"] == b] for b in BOROUGHS])
                part_rows.append({"model": m, "noun": n, "adjective": a, "n_nta": int(k.sum()), "r_manhattan": r_b, "r_urban": r_u, "r_manhattan_urban": r_bu,
                                  "partial_r_manhattan_given_urban": partial, "F_borough": F0, "p_borough": p0, "F_borough_given_urban": F1, "p_borough_given_urban": p1})
            C = A[(m, "cool", n)]
            pi = POSITIONS.index("after_place")
            b24 = C.layers.index(24)
            s = C.sense[:, pi, b24].astype(np.float64)
            s_nta = np.bincount(C.codes, s, C.n_nta) / np.bincount(C.codes, minlength=C.n_nta)
            r_s, t_s, p_s = cluster_r(s_nta[ok], yu[ok], np.arange(ok.sum()))
            yc = C.nta_margin()
            kk = ok & np.isfinite(yc)
            r_cu = np.corrcoef(yc[kk], yu[kk])[0, 1]
            mid = mid_layer(m, C.layers)
            dp = dec_place[(m, n)]
            da = dec_adj[m]
            sense_rows.append({"model": m, "noun": n, "r_sense_score_urban_margin_L24": r_s, "p": p_s, "r_cool_urban_margin": r_cu, "n_nta": int(ok.sum()), "layer": mid,
                               "cos_urban_vs_sense_decoder_at_noun": cosine(dp["w"][mid]["urban"], dp["sense_w"][mid]),
                               "cos_urban_vs_cool_decoder_at_noun": cosine(dp["w"][mid]["urban"], dp["w"][mid]["cool"]),
                               "cos_urban_vs_sense_decoder_at_adjective": cosine(da["w"][("after_place", mid)][prompt_name("urban", n)], da["sense_w"][(mid, n)]),
                               "cos_urban_vs_cool_decoder_at_adjective": cosine(da["w"][("after_place", mid)][prompt_name("urban", n)], da["w"][("after_place", mid)][prompt_name("cool", n)]),
                               "r2_sense_decoder_at_noun": dp["sense_r2"][mid]})
    bor = pd.DataFrame(bor_rows)
    bor["p_holm"] = holm(bor["p"].to_numpy())
    bor["family"] = len(bor)
    return bor, pd.DataFrame(part_rows), pd.DataFrame(sense_rows)


def task6(A: dict, grid: pd.DataFrame, words: dict, man: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`place` against `scene`: the margins, the mechanism measures side by side, and the words at the noun."""
    rows = []
    for m in MODELS:
        for a in ADJECTIVES:
            Sp, Ss = A[(m, a, "place")], A[(m, a, "scene")]
            row = {"model": m, "adjective": a} | pair_margin_rows(Sp, Ss, "place-scene")
            mp = man[(man["model"] == m) & (man["adjective"] == a)].set_index("noun")
            row["manhattan_t_place"], row["manhattan_t_scene"] = mp.loc["place", "t"], mp.loc["scene", "t"]
            row["manhattan_sign_same"] = bool(np.sign(mp.loc["place", "t"]) == np.sign(mp.loc["scene", "t"]))
            rows.append(row)
    margins = pd.DataFrame(rows)
    margins["p_paired_holm"] = holm(margins["p_paired"].to_numpy())
    cols = ["r2_scene", "gain_noun", "gain_adjective", "gain_qm", "verdict_r_last_layer", "share1_mid", "r_axis1_margin_mid", "embedded_mid", "manhattan_t", "yes_share"]
    mech = grid.pivot(index=["model", "adjective"], columns="noun", values=cols)
    mech.columns = [f"{c}_{n}" for c, n in mech.columns]
    for c in cols:
        mech[f"{c}_diff"] = mech[f"{c}_scene"] - mech[f"{c}_place"]
    mech = mech.reset_index()
    word_rows = []
    pi = POSITIONS.index("place")
    for m in MODELS:
        vocab, clean = words[m]
        Sp, Ss = A[(m, "cool", "place")], A[(m, "cool", "scene")]
        for layer in sorted({mid_layer(m, Sp.layers), Sp.layers[-1], 24}):
            b = Sp.layers.index(layer)
            embedded = Ss.text[pi, b] - Sp.text[pi, b]
            caused = (Ss.full[pi, b] - Ss.text[pi, b]) - (Sp.full[pi, b] - Sp.text[pi, b])
            for est, v in (("embedded (no-image readout, scene minus place)", embedded), ("image-caused (C2 of scene minus C2 of place)", caused)):
                vv = np.where(clean, v, np.nan)
                for end, order in (("scene end", np.argsort(-np.nan_to_num(vv, nan=-np.inf))[:N_WORDS]), ("place end", np.argsort(np.nan_to_num(vv, nan=np.inf))[:N_WORDS])):
                    for rank, i in enumerate(order, 1):
                        word_rows.append({"model": m, "layer": layer, "estimand": est, "end": end, "rank": rank, "word": vocab[i], "z_diff": float(v[i])})
            word_rows.append({"model": m, "layer": layer, "estimand": "r of the 2 C2 contrasts at the noun (scene vs place)", "end": "", "rank": 0, "word": "",
                              "z_diff": float(np.corrcoef(Ss.full[pi, b] - Ss.text[pi, b], Sp.full[pi, b] - Sp.text[pi, b])[0, 1])})
    return margins, mech, pd.DataFrame(word_rows)


def decoders_at_noun(A: dict, splits_of: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Task 7, steps 1 and 2 at the noun: 1 residual, 7 margins, for each noun and model, at each layer."""
    out, dec_rows, cos_rows, rank_rows_ = {}, [], [], []
    for m in MODELS:
        for n in NOUNS:
            base = A[(m, "cool", n)]
            layers = base.layers
            Y = np.column_stack([A[(m, a, n)].margin() for a in ADJECTIVES])
            Yz = (Y - np.nanmean(Y, 0)) / np.nanstd(Y, 0)
            pi = POSITIONS.index("after_place")
            sense = A[(m, "cool", n)].sense[:, pi, base.layers.index(24)].astype(np.float64)
            groups = base.frame["nta2020"].to_numpy()
            splits = splits_of[m]
            store = {"w": {}, "r2": {}, "sense_w": {}, "sense_r2": {}}
            complete = [j for j in range(len(ADJECTIVES)) if np.isfinite(Yz[:, j]).all()]
            res = base.residual("place")
            for b, layer in enumerate(layers):
                X = res[:, b].astype(np.float64)
                X = X - X.mean(0, keepdims=True)
                dec = prompt_decoders(X, Yz, groups, splits)
                store["w"][layer] = {a: dec["full_w"][:, j] for j, a in enumerate(ADJECTIVES)}
                store["r2"][layer] = dict(zip(ADJECTIVES, dec["r2"]))
                sd = prompt_decoders(X, (sense - sense.mean())[:, None] / sense.std(), groups, splits)
                store["sense_w"][layer] = sd["full_w"][:, 0]
                store["sense_r2"][layer] = float(sd["r2"][0])
                for j, a in enumerate(ADJECTIVES):
                    dec_rows.append({"model": m, "noun": n, "position": "noun", "layer": layer, "prompt": prompt_name(a, n), "n": int(np.isfinite(Yz[:, j]).sum()), "r2_oof": dec["r2"][j]})
                dec_rows.append({"model": m, "noun": n, "position": "noun", "layer": layer, "prompt": f"{n} cool sense score L24", "n": len(sense), "r2_oof": float(sd["r2"][0])})
                for a, c in itertools.combinations(ADJECTIVES, 2):
                    cos_rows.append({"model": m, "noun": n, "position": "noun", "layer": layer, "a": a, "b": c, "cosine": cosine(store["w"][layer][a], store["w"][layer][c])})
                if len(complete) >= 2:
                    Yc = Yz[:, complete]
                    dc = {"pred": dec["pred"][:, complete], "pred_train": train_fits(X, Yc, splits)}
                    rr = rank_rows(Yc, dc, splits)
                    rr["prompt"] = [prompt_name(ADJECTIVES[complete[j]], n) for j in rr["column"]]
                    rr = rr.assign(model=m, noun=n, position="noun", layer=layer, n_prompts=len(complete))
                    rank_rows_.append(rr)
                print(f"[f10.26] decoders at the noun: {m} {n} L{layer}", flush=True)
            base.drop("residual_place")
            out[(m, n)] = store
    return out, pd.DataFrame(dec_rows), pd.DataFrame(cos_rows), pd.concat(rank_rows_, ignore_index=True)


def decoders_at_positions(A: dict, splits_of: dict, words: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Task 7, steps 3 to 5 at the noun, the adjective, `?` and the last token: each prompt has its own residual."""
    out, dec_rows, loo_rows_, share_rows, cos_rows, nonice_rows = {}, [], [], [], [], []
    for m in MODELS:
        layers = A[(m, "cool", "place")].layers
        splits = splits_of[m]
        groups = A[(m, "cool", "place")].frame["nta2020"].to_numpy()
        names = [prompt_name(a, n) for a, n in PROMPTS]
        Y = np.column_stack([A[(m, a, n)].margin() for a, n in PROMPTS])
        Yz = (Y - np.nanmean(Y, 0)) / np.nanstd(Y, 0)
        store = {"w": {}, "u": {}, "share": {}, "r2": {}, "sense_w": {}}
        for pos in DECODE_POSITIONS:
            for b, layer in enumerate(layers):
                Xs = []
                for a, n in PROMPTS:
                    X = A[(m, a, n)].residual(pos)[:, b].astype(np.float64)
                    Xs.append(X - X.mean(0, keepdims=True))
                # Each prompt has its own residual at this position, thus each decoder reads its own X.
                fold_w = np.zeros((len(splits), Xs[0].shape[1], len(PROMPTS)))
                full_w = np.zeros((Xs[0].shape[1], len(PROMPTS)))
                r2s = np.zeros(len(PROMPTS))
                for j in range(len(PROMPTS)):
                    d = prompt_decoders(Xs[j], Yz[:, [j]], groups, splits)
                    fold_w[:, :, j], full_w[:, j], r2s[j] = d["fold_w"][:, :, 0], d["full_w"][:, 0], d["r2"][0]
                    dec_rows.append({"model": m, "position": LABEL[pos], "layer": layer, "prompt": names[j], "n": int(np.isfinite(Yz[:, j]).sum()), "r2_oof": r2s[j]})
                u, share = shared_direction(full_w, 3)
                store["w"][(pos, layer)] = {names[j]: full_w[:, j] for j in range(len(PROMPTS))}
                store["u"][(pos, layer)] = u
                store["share"][(pos, layer)] = share
                store["r2"][(pos, layer)] = dict(zip(names, r2s))
                share_rows.append({"model": m, "position": LABEL[pos], "layer": layer, "depth": depth_of(m, layer), "share_sv1": share[0], "share_sv2": share[1], "share_sv3": share[2],
                                   "mean_r2_own": float(r2s.mean())} | {f"load_{names[j]}": float(full_w[:, j] @ u[:, 0]) for j in range(len(PROMPTS))})
                # Leave 1 prompt out, with the shared direction of the other 13 from the training fold.
                for j in range(len(PROMPTS)):
                    others = [q for q in range(len(PROMPTS)) if q != j]
                    ok = np.isfinite(Yz[:, j])
                    for k in (1, 2, 3):
                        pred = np.full(len(Yz), np.nan)
                        for f, (tr, te) in enumerate(splits):
                            uu, _ = shared_direction(fold_w[f][:, others], k)
                            tr_ok = tr[ok[tr]]
                            s_tr, s_te = Xs[j][tr_ok] @ uu, Xs[j][te] @ uu
                            beta = np.linalg.lstsq(np.column_stack([np.ones(len(tr_ok)), s_tr]), Yz[tr_ok, j], rcond=None)[0]
                            pred[te] = np.column_stack([np.ones(len(te)), s_te]) @ beta
                        sse = np.nansum((Yz[ok, j] - pred[ok]) ** 2)
                        sst = sum(np.sum((Yz[te[ok[te]], j] - Yz[tr[ok[tr]], j].mean()) ** 2) for tr, te in splits)
                        loo_rows_.append({"model": m, "position": LABEL[pos], "layer": layer, "depth": depth_of(m, layer), "prompt": names[j], "k": k, "n": int(ok.sum()),
                                          "r2_shared": float(1 - sse / sst), "r_shared": float(np.corrcoef(Yz[ok, j], pred[ok])[0, 1]), "r2_own": float(r2s[j])})
                # The limit: remove the `nice` direction of each noun from the residuals, and fit the rank-1 shared direction again.
                for n in NOUNS:
                    jn = names.index(prompt_name("nice", n))
                    members = [q for q in range(len(PROMPTS)) if PROMPTS[q][1] == n and q != jn]
                    for j in members:
                        ok = np.isfinite(Yz[:, j])
                        pred = np.full(len(Yz), np.nan)
                        for f, (tr, te) in enumerate(splits):
                            wn = fold_w[f][:, jn]
                            Wo = fold_w[f][:, [q for q in members if q != j]]
                            Wo = Wo - np.outer(wn, wn @ Wo)
                            uu, _ = shared_direction(Wo, 1)
                            Xj = Xs[j] - np.outer(Xs[j] @ wn, wn)
                            tr_ok = tr[ok[tr]]
                            beta = np.linalg.lstsq(np.column_stack([np.ones(len(tr_ok)), Xj[tr_ok] @ uu]), Yz[tr_ok, j], rcond=None)[0]
                            pred[te] = np.column_stack([np.ones(len(te)), Xj[te] @ uu]) @ beta
                        sse = np.nansum((Yz[ok, j] - pred[ok]) ** 2)
                        sst = sum(np.sum((Yz[te[ok[te]], j] - Yz[tr[ok[tr]], j].mean()) ** 2) for tr, te in splits)
                        nonice_rows.append({"model": m, "position": LABEL[pos], "layer": layer, "noun": n, "prompt": names[j], "n": int(ok.sum()), "r2_shared_without_nice": float(1 - sse / sst)})
                print(f"[f10.26] shared direction: {m} {LABEL[pos]} L{layer}", flush=True)
            for a, n in PROMPTS:
                A[(m, a, n)].drop(f"residual_{pos}")
        # The cool sense score decoded at the adjective of the cool store, for the cosines of task 5 and step 5.
        for n in NOUNS:
            C = A[(m, "cool", n)]
            pi = POSITIONS.index("after_place")
            sense = C.sense[:, pi, C.layers.index(24)].astype(np.float64)
            res = C.residual("after_place")
            for b, layer in enumerate(layers):
                X = res[:, b].astype(np.float64)
                X = X - X.mean(0, keepdims=True)
                sd = prompt_decoders(X, ((sense - sense.mean()) / sense.std())[:, None], groups, splits)
                store["sense_w"][(layer, n)] = sd["full_w"][:, 0]
            C.drop("residual_after_place")
        out[m] = store
        np.savez(DATA / f"shared_u_{m}.npz", **{f"{pos}_L{layer}": u for (pos, layer), u in store["u"].items()})
        for (pos, layer), u in store["u"].items():
            W = store["w"][(pos, layer)]
            row = {"model": m, "position": LABEL[pos], "layer": layer, "depth": depth_of(m, layer)}
            for n in NOUNS:
                row[f"cos_u1_sense_{n}"] = cosine(u[:, 0], store["sense_w"][(layer, n)]) if pos == "after_place" else np.nan
                row[f"cos_u1_urban_{n}"] = cosine(u[:, 0], W[prompt_name("urban", n)])
                row[f"cos_u1_nice_{n}"] = cosine(u[:, 0], W[prompt_name("nice", n)])
                row[f"cos_u1_pretty_minus_ugly_{n}"] = cosine(u[:, 0], W[prompt_name("pretty", n)] - W[prompt_name("ugly", n)])
                row[f"cos_u1_cool_{n}"] = cosine(u[:, 0], W[prompt_name("cool", n)])
            cos_rows.append(row)
    return out, pd.DataFrame(dec_rows), pd.DataFrame(loo_rows_), pd.DataFrame(share_rows), pd.DataFrame(cos_rows), pd.DataFrame(nonice_rows)


def shared_u_at(A: dict, m: str, pos: str, layer: int, splits: list) -> np.ndarray:
    """Give the shared direction [d, 3] at 1 cell again, from the 14 decoders fit on all images. Equal to the saved one."""
    groups = A[(m, "cool", "place")].frame["nta2020"].to_numpy()
    b = A[(m, "cool", "place")].layers.index(layer)
    cols = []
    for a, n in PROMPTS:
        S = A[(m, a, n)]
        X = S.residual(pos)[:, b].astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        y = S.margin()
        y = (y - np.nanmean(y)) / np.nanstd(y)
        _, w, _ = dual_weights(X, y, np.flatnonzero(np.isfinite(y)))
        cols.append(w)
        S.drop(f"residual_{pos}")
    u, _ = shared_direction(np.column_stack(cols), 3)
    return u


def task7_words(A: dict, dec_pos: dict | None, words: dict, loo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The words of the shared direction, through the lens on CPU and through the readout across NTAs."""
    rows, checks = [], []
    for m in MODELS:
        vocab, clean = words[m]
        S = A[(m, "cool", "place")]
        mid = mid_layer(m, S.layers)
        best = loo[(loo["model"] == m) & (loo["k"] == 1)].groupby(["position", "layer"])["r2_shared"].mean().idxmax()
        pos_of = {v: k for k, v in LABEL.items()}
        for pos_label, layer, which in ((LABEL["place"], mid, "noun, mid depth"), (LABEL["after_place"], mid, "adjective, mid depth"), (best[0], int(best[1]), "best LOO cell")):
            pos = pos_of[pos_label]
            if dec_pos is not None:
                u = dec_pos[m]["u"][(pos, layer)][:, 0]
            else:
                saved = DATA / f"shared_u_{m}.npz"
                u = np.load(saved)[f"{pos}_L{layer}"][:, 0] if saved.exists() else shared_u_at(A, m, pos, layer, None)[:, 0]
            lw, logits = lens_words(m, layer, u, vocab, clean)
            rows.append(lw.assign(model=m, position=pos_label, layer=layer, which=which, method="lens"))
            # The readout check: r across NTAs between the projection on u and the NTA mean readout of each word.
            proj = np.zeros(S.n_nta)
            counts = np.zeros(S.n_nta)
            res = S.residual(pos)[:, S.layers.index(layer)].astype(np.float64)
            p = (res - res.mean(0)) @ u
            proj = np.bincount(S.codes, p, S.n_nta) / np.bincount(S.codes, minlength=S.n_nta)
            nta = S.nta(pos)[:, S.layers.index(layer)].astype(np.float64)
            c = nta - nta.mean(0)
            pz = (proj - proj.mean()) / proj.std()
            r = (pz @ c) / (len(pz) * c.std(0).clip(min=1e-9))
            v = np.where(clean, r, np.nan)
            for end, order in (("high", np.argsort(-np.nan_to_num(v, nan=-np.inf))[:N_WORDS]), ("low", np.argsort(np.nan_to_num(v, nan=np.inf))[:N_WORDS])):
                for rank, i in enumerate(order, 1):
                    rows.append(pd.DataFrame([{"end": end, "rank": rank, "word": vocab[i], "logit": float(r[i]), "model": m, "position": pos_label, "layer": layer, "which": which, "method": "readout r across NTAs"}]))
            S.drop(f"residual_{pos}")
            checks.append({"model": m, "position": pos_label, "layer": layer, "r_lens_logits_vs_readout_r": float(np.corrcoef(logits[clean], r[clean])[0, 1])})
        checks.append({"model": m, "position": "qm, no image", "layer": mid, "r_lens_logits_vs_readout_r": lens_check(m, S, mid)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(checks)


def task8(A: dict, words: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Manhattan against the rest (Holm over 42), the borough order, the 40-split stability, the C3 Jaccard, and the place probes."""
    man_rows, order_rows, stab_rows, jac_rows = [], [], [], []
    for (m, a, n), S in A.items():
        y = S.nta_margin()
        ok = np.isfinite(y)
        t = two_sample_t(y[ok][:, None], S.borough_nta[ok] == "Manhattan")[0]
        man_rows.append({"model": m, "adjective": a, "noun": n, "prompt": S.prompt, "n_nta": int(ok.sum()), "t": t, "p": 2 * stats.t.sf(abs(t), ok.sum() - 2),
                         "yes_share": (S.frame["outcome"] == "yes").mean()})
        means = pd.Series({b: y[ok][S.borough_nta[ok] == b].mean() for b in BOROUGHS}).sort_values(ascending=False)
        order_rows.append({"model": m, "adjective": a, "noun": n, "prompt": S.prompt, "order": " > ".join(means.index)} | {f"rank_{b.replace(' ', '_')}": i for i, b in enumerate(means.index, 1)}
                          | {f"sd_{b.replace(' ', '_')}": (means[b] - y[ok].mean()) / y[ok].std(ddof=1) for b in BOROUGHS})
    man = pd.DataFrame(man_rows)
    man["p_holm"] = holm(man["p"].to_numpy())
    man["family"] = len(man)
    order = pd.DataFrame(order_rows)
    for m in MODELS:
        vocab, clean = words[m]
        lists = {}
        for (mm, a, n), S in A.items():
            if mm != m:
                continue
            mid = mid_layer(m, S.layers)
            bi = S.layers.index(mid)
            cell = S.nta("after_place")[:, bi].astype(np.float64)
            tv = borough_t(cell, S.borough_nta)
            lists[(a, n)] = {b: top_set(tv[b], clean) for b in BOROUGHS}
            members = {b: S.borough_nta == b for b in BOROUGHS}
            k0 = {b: [] for b in BOROUGHS}
            k1 = {b: [] for b in BOROUGHS}
            for s in range(N_SPLITS):
                half = split_within(S.borough_nta, SEED + s)
                ca = cell[half] - cell[half].mean(0)
                cb = cell[~half] - cell[~half].mean(0)
                _, sa, la = axis1(ca)
                _, sb, lb = axis1(cb)
                for b in BOROUGHS:
                    k0[b].append(np.corrcoef(two_sample_t(ca, members[b][half]), two_sample_t(cb, members[b][~half]))[0, 1])
                    k1[b].append(np.corrcoef(two_sample_t(ca - np.outer(sa, la), members[b][half]), two_sample_t(cb - np.outer(sb, lb), members[b][~half]))[0, 1])
            for b in BOROUGHS:
                stab_rows.append({"model": m, "adjective": a, "noun": n, "prompt": S.prompt, "layer": mid, "borough": b, "splits": N_SPLITS, "stability_all_axes": float(np.mean(k0[b])),
                                  "sd_all_axes": float(np.std(k0[b])), "stability_axis1_out": float(np.mean(k1[b])), "sd_axis1_out": float(np.std(k1[b]))})
            S.drop("nta_after_place")
            print(f"[f10.26] task 8: {m} {S.prompt}", flush=True)
        for (a, n), (c, d) in itertools.combinations(PROMPTS, 2):
            for b in BOROUGHS:
                jac_rows.append({"model": m, "prompt_a": prompt_name(a, n), "prompt_b": prompt_name(c, d), "same_noun": n == d, "same_adjective": a == c, "borough": b,
                                 "jaccard_top50": jaccard(lists[(a, n)][b], lists[(c, d)][b])})
    probes = place_probe_rows(A)
    return man, order, pd.DataFrame(stab_rows), pd.DataFrame(jac_rows), probes


def place_probe_rows(A: dict) -> pd.DataFrame:
    """Relate the `urban` margin to the F10.18 place probes, where a probe exists. Qwen3.6-27B has none: an open item."""
    from analyze_probe import margins as probe_margins

    rows = []
    for m in MODELS:
        if not (ROOT / "data/f10/hop_probe" / m / "probes.jsonl").exists():
            rows.append({"model": m, "noun": "", "probe": "", "note": "no probe run exists for this model; a new readout would need a GPU (open item)"})
            continue
        pm = probe_margins(m)
        for n in NOUNS:
            U = A[(m, "urban", n)]
            idx = U.frame["image_idx"].to_numpy()
            for probe in ("in New York", "in a city", "urban"):
                if probe not in pm.columns:
                    continue
                v = pm[probe].reindex(idx).to_numpy(float)
                y = U.margin()
                ok = np.isfinite(v) & np.isfinite(y)
                nta = pd.DataFrame({"n": U.codes, "v": v, "y": y})[ok].groupby("n").mean()
                r, p = stats.pearsonr(nta["v"], nta["y"])
                rows.append({"model": m, "noun": n, "probe": probe, "n_nta": len(nta), "r_nta_with_urban_margin": r, "p": p, "probe_yes_share": float((v[ok] > 0).mean()),
                             "urban_yes_share": (U.frame["outcome"] == "yes").mean(), "note": ""})
    return pd.DataFrame(rows)


def store_measures(key: tuple[str, str, str]) -> tuple[pd.DataFrame, ...]:
    """Give the gain, the mean-patch depth curve, the axis, the embedded counts and the verdict tables of 1 store."""
    m, a, n = key
    S = Summary(m, a, n)
    patch = mean_patch(m)
    g, dpt = gain_rows(S, patch)
    if n == "place" and a in ("cool", "nice", "ordinary") and not np.array_equal(np.nan_to_num(S.margin()), np.nan_to_num(S.margin(old=True))):
        g_old = gain_rows(S, patch, old=True)[0]
    else:
        g_old = g
    out = (g, g_old, dpt, axis_rows(S), embedded_rows(S), verdict_rows(S))
    print(f"[f10.26] measures: {m} {S.prompt}", flush=True)
    return out


def task0_only(models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """Run task 0 on the stores that exist. The gate of the plan: stop when a value differs from F10.24."""
    FIG.mkdir(parents=True, exist_ok=True)
    A = {}
    for m in models:
        for a in ("cool", "nice", "ordinary"):
            if (DATA / store_name(m, a, "place") / "meta.json").exists() and (DATA / f"{m}-patches" / "mean_patch.npy").exists():
                A[(m, a, "place")] = Summary(m, a, "place")
    gains, gains_old, depth, axes, emb, ver = [], [], [], [], [], []
    for (m, a, n), S in A.items():
        patch = mean_patch(m)
        g, dpt = gain_rows(S, patch)
        gains.append(g)
        depth.append(dpt)
        gains_old.append(gain_rows(S, patch, old=True)[0] if not np.array_equal(np.nan_to_num(S.margin()), np.nan_to_num(S.margin(old=True))) else g)
        axes.append(axis_rows(S))
        emb.append(embedded_rows(S))
        ver.append(verdict_rows(S))
        print(f"[f10.26] task 0 measures: {m} {S.prompt}", flush=True)
    gains, gains_old, depth, axes, emb, ver = (pd.concat(x, ignore_index=True) for x in (gains, gains_old, depth, axes, emb, ver))
    out = task0(A, gains, depth, axes, emb, ver, gains_old, models=tuple(sorted({m for m, _, _ in A})))
    out.to_csv(FIG / "t0_reproduction.csv", index=False)
    return out


def probes_only() -> None:
    """Run the place-probe step of task 8 alone, after a new probe run (Qwen3.6-27B, 2026-11-02). It reads the frames only."""
    A = {(m, "urban", n): Summary(m, "urban", n) for m in MODELS for n in NOUNS}
    place_probe_rows(A).to_csv(FIG / "t8_place_probes.csv", index=False)
    print(f"[f10.26] place probes written to {FIG}", flush=True)


def words_only() -> None:
    """Run the word step of task 7 alone, after the tables exist. It fits the 14 decoders of the 3 named cells again."""
    A = load_all()
    words = {m: vocab_of(m) for m in MODELS}
    loo = pd.read_csv(FIG / "t7_loo.csv")
    w7, chk = task7_words(A, None, words, loo)
    w7.to_csv(FIG / "t7_words.csv", index=False)
    chk.to_csv(FIG / "t7_lens_check.csv", index=False)
    print(f"[f10.26] words written to {FIG}", flush=True)


def analyze() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    A = load_all()
    words = {m: vocab_of(m) for m in MODELS}
    splits_of = {m: folds(A[(m, "cool", "place")].frame["nta2020"].to_numpy(), seed=SEED) for m in MODELS}
    # The per-store measures, 1 process for each store (the stores are independent).
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        parts = list(pool.map(store_measures, list(A)))
    gains, gains_old, depth, axes, emb, ver = (pd.concat([p[i] for p in parts], ignore_index=True) for i in range(6))
    gains.to_csv(FIG / "t2_gain_by_layer.csv", index=False)
    depth.to_csv(FIG / "t2_patch_depth.csv", index=False)
    axes.to_csv(FIG / "t2_axis_by_layer.csv", index=False)
    emb.to_csv(FIG / "t2_embedded_by_layer.csv", index=False)
    ver.to_csv(FIG / "t2_verdict_by_layer.csv", index=False)
    task0(A, gains, depth, axes, emb, ver, gains_old, models=MODELS).to_csv(FIG / "t0_reproduction.csv", index=False)
    rates, tests, noun = task1(A)
    rates.to_csv(FIG / "t1_outcomes.csv", index=False)
    tests.to_csv(FIG / "t1_outcome_tests.csv", index=False)
    noun.to_csv(FIG / "t1_noun_tests.csv", index=False)
    man, order, stab, jac, probes = task8(A, words)
    man.to_csv(FIG / "t8_manhattan.csv", index=False)
    order.to_csv(FIG / "t8_borough_order.csv", index=False)
    stab.to_csv(FIG / "t8_stability.csv", index=False)
    jac.to_csv(FIG / "t8_jaccard.csv", index=False)
    probes.to_csv(FIG / "t8_place_probes.csv", index=False)
    grid, counts = task2(A, gains, axes, emb, ver, man)
    grid.to_csv(FIG / "t2_grid.csv", index=False)
    counts.to_csv(FIG / "t2_counts.csv", index=False)
    lists, stests = task3(A, axes, words)
    lists.to_csv(FIG / "t3_axis_words.csv", index=False)
    stests.to_csv(FIG / "t3_sense_tests.csv", index=False)
    margins, mech, nwords = task6(A, grid, words, man)
    margins.to_csv(FIG / "t6_noun_margins.csv", index=False)
    mech.to_csv(FIG / "t6_noun_mechanism.csv", index=False)
    nwords.to_csv(FIG / "t6_noun_words.csv", index=False)
    dec_place, drows, cos7, rank7 = decoders_at_noun(A, splits_of)
    drows.to_csv(FIG / "t7_decoders_noun.csv", index=False)
    cos7.to_csv(FIG / "t7_cosines_noun.csv", index=False)
    rank7.to_csv(FIG / "t7_rank_noun.csv", index=False)
    dec_pos, drows2, loo, share, cos5, nonice = decoders_at_positions(A, splits_of, words)
    drows2.to_csv(FIG / "t7_decoders.csv", index=False)
    loo.to_csv(FIG / "t7_loo.csv", index=False)
    share.to_csv(FIG / "t7_shared_share.csv", index=False)
    cos5.to_csv(FIG / "t7_compare.csv", index=False)
    nonice.to_csv(FIG / "t7_without_nice.csv", index=False)
    task4(A, dec_place, words).to_csv(FIG / "t4_pretty_ugly.csv", index=False)
    bor, part, sense5 = task5(A, dec_place, dec_pos)
    bor.to_csv(FIG / "t5_urban_borough.csv", index=False)
    part.to_csv(FIG / "t5_urban_partial.csv", index=False)
    sense5.to_csv(FIG / "t5_urban_sense.csv", index=False)
    w7, chk = task7_words(A, dec_pos, words, loo)
    w7.to_csv(FIG / "t7_words.csv", index=False)
    chk.to_csv(FIG / "t7_lens_check.csv", index=False)
    print(f"[f10.26] tables written to {FIG}", flush=True)


# ---------------------------------------------------------------- figures


def save(fig, name: str, title: str, caption: str, tags: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    (FIG / f"{name}.json").write_text(json.dumps({"plot-title": title, "plot-caption": caption, "plot-latex-label": f"fig:{name.replace('_', '-')}", "plot-tags": tags}, indent=2) + "\n")


def prompt_order() -> list[str]:
    return [prompt_name(a, n) for a, n in PROMPTS]


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import _style as S

    S.apply_house_style()
    order = prompt_order()
    # Figure 1: outcomes.
    rates = pd.read_csv(FIG / "t1_outcomes.csv")
    rates = rates[rates["borough"] == "all"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, m in zip(axs, MODELS):
        g = rates[rates["model"] == m].set_index("prompt").reindex(order)
        y = np.arange(len(order))
        ax.barh(y, g["yes"], color=S.PAL["teal"], label="yes")
        ax.barh(y, g["no"], left=g["yes"], color=S.PAL["coral"], label="no")
        ax.barh(y, g["unable"], left=g["yes"] + g["no"], color=S.PAL["warmgrey"], label="unable")
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_title(NAMES[m])
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of 591 images")
        ax.invert_yaxis()
        ax.grid(False)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.14))
    save(fig, "t1_outcomes", "Outcomes of the 14 prompts in 3 models",
         "The share of yes, no and unable answers for each prompt (rows) and model (panels), on 591 images. The outcome is the first token of the answer. Gemma-4-31B gives no verdict on about half of the images for nice, with both nouns.",
         ["f10.26", "outcomes"])
    # Figure 2: the mechanism grid.
    grid = pd.read_csv(FIG / "t2_grid.csv")
    measures = [("r2_scene", "scene code R2", 0, 1), ("gain_noun", "gain at the noun", 0, 0.45), ("gain_adjective", "gain at the adjective", 0, 0.45), ("gain_qm", "gain at ?", 0, 0.45),
                ("verdict_r_last_layer", "verdict r, last layer", 0.5, 1), ("share1_mid", "axis 1 share, mid depth", 0, 1), ("r_axis1_margin_mid", "axis 1 r with margin", -1, 1),
                ("embedded_mid", "embedded words of 100", 0, 100), ("manhattan_t", "Manhattan t", -9, 9)]
    fig, axs = plt.subplots(1, len(measures), figsize=(13.5, 4.2), sharey=True)
    for ax, (col, label, lo, hi) in zip(axs, measures):
        g = grid.pivot(index="prompt", columns="model", values=col).reindex(order)[list(MODELS)]
        div = lo < 0
        im = ax.imshow(g.to_numpy(float), cmap=S.CMAP_DIV if div else S.CMAP_SEQ, vmin=lo, vmax=hi, aspect="auto")
        for i in range(len(order)):
            for j in range(len(MODELS)):
                v = g.iloc[i, j]
                txt = f"{v:.0f}" if col == "embedded_mid" else (f"{v:.1f}" if col == "manhattan_t" else f"{v:.2f}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=S.text_on(matplotlib.colors.to_hex(im.cmap(im.norm(v)))) if np.isfinite(v) else "black")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([SHORT[m] for m in MODELS])
        ax.set_title(label, fontsize=7)
        ax.grid(False)
    axs[0].set_yticks(range(len(order)))
    axs[0].set_yticklabels(order)
    save(fig, "t2_grid", "The mechanism measures over 14 prompts and 3 models",
         "Each panel gives 1 measure of the F10.24 table for each prompt (rows) and model (columns: G Gemma-4-31B, Q9 Qwen3.5-9B, Q27 Qwen3.6-27B). The gain is the largest out-of-fold R2 that the residual of a token adds to the mean patch, over the saved layers. Mid depth is the saved layer nearest to relative depth 0.5. Values are in t2_grid.csv.",
         ["f10.26", "mechanism"])
    # Figure 3: the shared direction.
    cos = pd.read_csv(FIG / "t7_cosines_noun.csv")
    loo = pd.read_csv(FIG / "t7_loo.csv")
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4))
    for j, m in enumerate(MODELS):
        layers = sorted(cos[cos["model"] == m]["layer"].unique())
        mid = mid_layer(m, layers)
        g = cos[(cos["model"] == m) & (cos["noun"] == "place") & (cos["layer"] == mid)]
        M = pd.DataFrame(np.eye(len(ADJECTIVES)), index=ADJECTIVES, columns=ADJECTIVES)
        for r in g.itertuples():
            M.loc[r.a, r.b] = M.loc[r.b, r.a] = r.cosine
        ax = axs[0, j]
        im = ax.imshow(M.to_numpy(), cmap=S.CMAP_DIV, vmin=-1, vmax=1)
        ax.set_xticks(range(len(ADJECTIVES)))
        ax.set_xticklabels(ADJECTIVES, rotation=90)
        ax.set_yticks(range(len(ADJECTIVES)))
        ax.set_yticklabels(ADJECTIVES if j == 0 else [])
        ax.set_title(f"{NAMES[m]}, place, L{mid}", fontsize=7)
        ax.grid(False)
        for a in range(len(ADJECTIVES)):
            for b in range(len(ADJECTIVES)):
                ax.text(b, a, f"{M.iloc[a, b]:.2f}", ha="center", va="center", fontsize=5, color=S.text_on(matplotlib.colors.to_hex(im.cmap(im.norm(M.iloc[a, b])))))
        ax = axs[1, j]
        l1 = loo[(loo["model"] == m) & (loo["k"] == 1) & (loo["position"] == "noun") & (loo["layer"] == mid)].set_index("prompt").reindex(order)
        y = np.arange(len(order))
        ax.barh(y - 0.2, l1["r2_own"], height=0.4, color=S.PAL["slate"], label="own decoder")
        ax.barh(y + 0.2, l1["r2_shared"], height=0.4, color=S.PAL["blue"], label="shared direction, leave 1 out")
        ax.set_yticks(y)
        ax.set_yticklabels(order if j == 0 else [])
        ax.invert_yaxis()
        ax.set_xlim(-0.1, 1)
        ax.set_xlabel("out-of-fold R2")
        ax.grid(axis="y", visible=False)
    fig.legend(*axs[1, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(wspace=0.15)
    save(fig, "t7_shared", "A shared aesthetic direction at the noun",
         "Top: the cosine between the ridge decoders of the 7 adjective margins, fit from the same residual at the noun place at mid depth, for each model. Bottom: the out-of-fold R2 of each prompt's own decoder (grey) against the R2 of the first singular direction of the 13 other decoders, fit in the training folds and tested on the held-out prompt (blue), at the noun at mid depth. Margins are z-scored. Folds keep each NTA whole.",
         ["f10.26", "shared direction"])
    # Figure 4: Manhattan against the rest.
    man = pd.read_csv(FIG / "t8_manhattan.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    y = np.arange(len(order))
    off = {"gemma-4-31b": -0.25, "qwen3.5-9b": 0.0, "qwen3.6-27b": 0.25}
    col = {"gemma-4-31b": S.PAL["blue"], "qwen3.5-9b": S.PAL["teal"], "qwen3.6-27b": S.PAL["coral"]}
    for m in MODELS:
        g = man[man["model"] == m].set_index("prompt").reindex(order)
        pas = g["p_holm"] < 0.05
        ax.scatter(g["t"][pas], y[pas] + off[m], color=col[m], s=18, label=NAMES[m], zorder=3)
        ax.scatter(g["t"][~pas], y[~pas] + off[m], facecolor="white", edgecolor=col[m], s=18, zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("t of Manhattan NTAs against the other NTAs, on the answer margin")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "t8_manhattan", "Manhattan against the rest on 14 prompts",
         "The two-sample t of the Manhattan NTA mean margins against the other NTAs, for each prompt and model (df 195 or fewer where an NTA has no verdict). A filled marker passes Holm over the family of 42 tests at .05; an open marker does not.",
         ["f10.26", "boroughs"])
    print(f"[f10.26] figures written to {FIG}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--model", required=True, choices=MODELS)
    s.add_argument("--adjective", required=True, choices=ADJECTIVES)
    s.add_argument("--noun", required=True, choices=NOUNS)
    s.add_argument("--shards", type=int, default=None)
    p = sub.add_parser("patches")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--shards", type=int, default=None)
    sub.add_parser("task0")
    sub.add_parser("words")
    sub.add_parser("probes")
    sub.add_parser("analyze")
    sub.add_parser("figures")
    args = ap.parse_args()
    if args.cmd == "summary":
        summary(args.model, args.adjective, args.noun, args.shards)
    elif args.cmd == "patches":
        patches(args.model, args.shards)
    elif args.cmd == "task0":
        print(task0_only().to_string())
    elif args.cmd == "words":
        words_only()
    elif args.cmd == "probes":
        probes_only()
    elif args.cmd == "analyze":
        analyze()
    else:
        figures()


if __name__ == "__main__":
    main()
