#!/usr/bin/env python
"""F10.24: the judgment across 3 adjectives (`cool`, `nice`, `ordinary`), inside each of 3 models.

The `cool` store is in `data/f10/hop_all/{model}`. The other 2 stores are in `{model}-nice` and
`{model}-ordinary`. All stores hold the same 591 images. The image comes first and the attention is
causal, thus the image part of each read is equal across the adjectives.

The tail positions differ by model. Thus each token is found by its text, not by its index or name.
The manifest names are shifted: `after_place` holds the adjective and `after_cool` holds `?`.

`job` runs on SLURM for 1 store and writes `data/f10/adjectives/{store}/`:

| File | Content |
|------|---------|
| `reads.npz` | Z-scored means of the reads by borough and by outcome, the no-image read, the sense score and the probe values of each image |
| `svd.csv` | Share of the NTA-mean variance on axis 1, and abs r of axis 1 with the sense score |
| `sense.csv` | Sense score with no image, the image mean, and the share of images above the no-image value |
| `gain.csv` | Out-of-fold R2 of each cell, the gain over the mean patch, NTA t test and Holm p |
| `patch_depth.csv` | Out-of-fold R2 of the mean patch at each layer, for the margin of the store |
| `frame.parquet` | The answer of each image, in the order of `images.parquet` |

`report` is CPU only. It reads the job outputs and writes `research/figures/f10/adjectives/*.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_adjectives.py job --model gemma-4-31b --adjective nice
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_adjectives.py job --model gemma-4-31b --adjective cool --gain-only --mask-from nice
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_adjectives.py report
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

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from analyze_logit import holm  # noqa: E402
from hop_free import REPEATS, SEED, folds, kernel, oof_mixed, oof_predict, r2  # noqa: E402
from hop_sense import CAP, SOCIAL, TEMPERATURE  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
ADJECTIVES = ("cool", "nice", "ordinary")
POSITIONS = ("after_image", "place", "adj", "qm", "final")
NTA_POSITIONS = (1, 2, 3, 4)  # index into POSITIONS, for the SVD
PROBES = ("Subject", "Unable", "Yes", "No", "yes", "no", "Whether")
OUTCOMES = ("yes", "no", "unable")
DATA = ROOT / "data/f10/adjectives"
FIG = ROOT / "research/figures/f10/adjectives"
HOP_LOGIT = ROOT / "data/f10/hop_logit"
MECH = ROOT / "research/figures/f10/mechanism"


def store_name(model: str, adjective: str) -> str:
    return model if adjective == "cool" else f"{model}-{adjective}"


def outcome(top: str) -> str:
    """Give yes, no or unable for a first token. The rule is the rule of F10.23."""
    word = str(top).strip().lower()
    return word if word in ("yes", "no") else "unable"


def softcap(model: str) -> float | None:
    return json.loads((HOP_LOGIT / model / "manifest.json").read_text()).get("softcap")


def frame(model: str, adjective: str) -> pd.DataFrame:
    """Give 1 row for each image, in the order of `images.parquet`. The margin is empty for unable."""
    images = pd.read_parquet(ALL_ROOT / "images.parquet")[["image_idx", "nta2020", "borough"]]
    answers = pd.concat(map(pd.read_parquet, sorted((ALL_ROOT / store_name(model, adjective) / "answers").glob("answers-*.parquet"))))
    if adjective == "cool":
        # The cool margin comes from the lens run. Its 8 forms are checked against the first token.
        margin = pd.read_parquet(HOP_LOGIT / model / "answers.parquet")[["image_idx", "margin"]]
        answers = answers[["image_idx", "top"]].merge(margin, on="image_idx", validate="1:1")
    else:
        answers = answers[["image_idx", "top", "margin"]]
    out = images.merge(answers, on="image_idx", how="left", validate="1:1", sort=False)
    out["outcome"] = out["top"].map(outcome)
    out.loc[out["outcome"] == "unable", "margin"] = np.nan
    if len(out) != 591:
        raise ValueError(f"{len(out)} images, expected 591")
    return out.reset_index(drop=True)


def tail_index(tokens: list[str], adjective: str) -> list[int]:
    """Give the index of each compared position: after the image, place, adjective, `?` and the last token."""
    place = tokens.index(" place")
    adj = tokens.index(f" {adjective}", place)
    if tokens[adj + 1] != "?":
        raise ValueError(f"the token after {adjective} is {tokens[adj + 1]!r}")
    return [0, place, adj, adj + 1, len(tokens) - 1]


def uncap(x: np.ndarray, cap: float | None) -> np.ndarray:
    """Give the read before the softcap. The stored reads hold cap * tanh(read / cap)."""
    if cap is None:
        return x
    return cap * np.arctanh(np.clip(x / cap, -1 + 1e-7, 1 - 1e-7))


def zscore(x: np.ndarray) -> np.ndarray:
    """Give the z score of each read over the whole vocabulary."""
    return (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)


def job(model: str, adjective: str, gain_only: bool = False, mask_from: str | None = None) -> None:
    name = store_name(model, adjective)
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    store = open_store(name, ALL_ROOT)
    man = store.manifest
    tokens = man["tail_tokens"]
    layers = man["layers"]
    fr = frame(model, adjective)
    fr.to_parquet(out / "frame.parquet", index=False)
    cap = softcap(model)
    tag = "" if mask_from is None else f"-mask-{mask_from}"
    if not gain_only:
        reads(model, adjective, store, fr, tokens, layers, cap, out)
    mask = None
    if mask_from is not None:
        # Keep the images that have a yes or no answer for the other adjective (D167).
        mask = np.isfinite(frame(model, mask_from)["margin"].to_numpy(float))
    gain(model, adjective, store, fr, tokens, layers, out, mask, tag)


def reads(model: str, adjective: str, store, fr: pd.DataFrame, tokens: list[str], layers: list[int], cap: float | None, out: Path) -> None:
    """Read the 591 images once. Write the group means, the sense scores, the probes and the SVD."""
    pos = tail_index(tokens, adjective)
    codes = pd.factorize(fr["nta2020"])[0]
    n_nta = codes.max() + 1
    boroughs = sorted(fr["borough"].unique())
    bcode = fr["borough"].map({b: i for i, b in enumerate(boroughs)}).to_numpy()
    ocode = fr["outcome"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    vocab = store.vocab["token"].str.strip().to_numpy().astype(str)
    words = pd.Series(vocab).str.lower()
    hot = np.flatnonzero(words.isin(TEMPERATURE).to_numpy())
    soc = np.flatnonzero(words.isin(SOCIAL).to_numpy())
    probe_at = [int(np.flatnonzero(vocab == w)[0]) for w in PROBES if (vocab == w).any()]
    probe_names = [w for w in PROBES if (vocab == w).any()]

    text = np.load(store.path / "text_reads.npy", mmap_mode="r")[pos]
    text = zscore(uncap(np.asarray(text, np.float64), cap))
    text[0] = np.nan  # no no-image read exists after the image

    n_img, n_pos, n_lay, n_voc = len(fr), len(pos), len(layers), vocab.size
    z_sum = np.zeros((n_pos, n_lay, n_voc))
    z_border = np.zeros((len(boroughs), n_pos, n_lay, n_voc))
    z_outcome = np.zeros((len(OUTCOMES), n_pos, n_lay, n_voc))
    nta_sum = np.zeros((n_nta, len(NTA_POSITIONS), n_lay, n_voc))
    sense = np.zeros((n_img, n_pos, n_lay))
    probes = np.zeros((n_img, n_pos, n_lay, len(probe_at)))
    row = 0
    for f in sorted((store.path / "reads").glob("reads-*.npy")):
        x = np.load(f, mmap_mode="r")
        k = x.shape[0]
        rows = np.arange(row, row + k)
        z = zscore(uncap(np.asarray(x[:, pos], np.float64), cap))  # [images, positions, layers, vocab]
        z_sum += z.sum(0)
        for g in range(len(boroughs)):
            z_border[g] += z[bcode[rows] == g].sum(0)
        for g in range(len(OUTCOMES)):
            z_outcome[g] += z[ocode[rows] == g].sum(0)
        for g in np.unique(codes[rows]):
            sel = codes[rows] == g
            nta_sum[g] += z[sel][:, list(NTA_POSITIONS)].sum(0)
        sense[rows] = z[..., soc].mean(-1) - z[..., hot].mean(-1)
        probes[rows] = z[..., probe_at]
        row += k
        print(f"[f10.24] {model}-{adjective}: {row} images", flush=True)
    if row != n_img:
        raise ValueError(f"read {row} images, expected {n_img}")

    counts_b = np.bincount(bcode, minlength=len(boroughs))
    counts_o = np.bincount(ocode, minlength=len(OUTCOMES))
    np.savez(
        out / "reads.npz",
        z_all=(z_sum / n_img).astype(np.float32),
        z_borough=(z_border / counts_b[:, None, None, None]).astype(np.float32),
        z_outcome=(z_outcome / np.maximum(counts_o, 1)[:, None, None, None]).astype(np.float32),
        counts_borough=counts_b,
        counts_outcome=counts_o,
        text=text.astype(np.float32),
        sense=sense.astype(np.float32),
        probes=probes.astype(np.float32),
        probe_names=np.array(probe_names),
        positions=np.array(POSITIONS),
        token=np.array([tokens[i] for i in pos]),
        layers=np.array(layers),
        boroughs=np.array(boroughs),
        outcomes=np.array(OUTCOMES),
    )

    # SVD of the NTA means of each position (F10.15), and the sense score of each position (F10.17).
    counts_nta = np.bincount(codes, minlength=n_nta)
    svd_rows, sense_rows = [], []
    for j, p in enumerate(NTA_POSITIONS):
        for b, layer in enumerate(layers):
            m = nta_sum[:, j, b, :] / counts_nta[:, None]
            c = m - m.mean(0, keepdims=True)
            w, u = np.linalg.eigh(c @ c.T)
            share1 = float(w[-1] / w.clip(min=0).sum())
            nta_sense = np.bincount(codes, sense[:, p, b]) / counts_nta
            r_axis1 = abs(float(np.corrcoef(nta_sense, u[:, -1])[0, 1]))
            svd_rows.append({"model": model, "adjective": adjective, "position": POSITIONS[p], "token": tokens[pos[p]], "layer": layer,
                             "share1": share1, "r_axis1": r_axis1, "sd_sense": float(sense[:, p, b].std())})
            t0 = float(text[p, b, soc].mean() - text[p, b, hot].mean())
            sense_rows.append({"model": model, "adjective": adjective, "position": POSITIONS[p], "token": tokens[pos[p]], "layer": layer,
                               "no_image": t0, "image_mean": float(sense[:, p, b].mean()), "share_above": float((sense[:, p, b] > t0).mean())})
    pd.DataFrame(svd_rows).to_csv(out / "svd.csv", index=False)
    pd.DataFrame(sense_rows).to_csv(out / "sense.csv", index=False)


def gain(model: str, adjective: str, store, fr: pd.DataFrame, tokens: list[str], layers: list[int], out: Path, mask: np.ndarray | None, tag: str) -> None:
    """Give the gain of each cell over the mean patch (F10.9), with the margin of the store as the target."""
    y_all = fr["margin"].to_numpy(float)
    keep = np.isfinite(y_all) if mask is None else (np.isfinite(y_all) & mask)
    idx = np.flatnonzero(keep)
    y = y_all[idx]
    groups = fr["nta2020"].to_numpy()[idx]
    codes = pd.factorize(groups)[0]
    reps = [folds(groups, seed=SEED + i) for i in range(REPEATS)]
    splits = reps[0]
    r = lambda pred: r2(y, pred, splits)

    # The patches are the same for all stores, thus the cool store holds them.
    cool = open_store(model, ALL_ROOT)
    mean_patch = np.concatenate([np.asarray(np.load(f, mmap_mode="r"), np.float32).mean(1) for f in sorted((cool.path / "patches").glob("residuals-*.npy"))])
    if len(mean_patch) != len(fr):
        raise ValueError("the patches do not match the images")
    kern = [kernel(mean_patch[idx, b].astype(np.float64)) for b in range(len(layers))]
    depth = []
    for b, layer in enumerate(layers):
        pred = np.mean([oof_predict(kern[b], y, s) for s in reps], axis=0)
        depth.append({"model": model, "adjective": adjective, "mask": mask is not None, "n": len(idx), "layer": layer, "r2": r(pred)})
    pd.DataFrame(depth).to_csv(out / f"patch_depth{tag}.csv", index=False)

    scene = sum(kern) / len(kern)  # the mean patch, with the kernel of each layer averaged (F10.9)
    if len(idx) == len(fr):
        full = np.load(ROOT / "data/f10/hop_free" / model / "scene_kernel_1.npy")
        print(f"[f10.24] {model}: scene kernel max diff to the F10.9 file {np.abs(full - scene).max():.2e}", flush=True)
    scene_pred = np.mean([oof_predict(scene, y, s) for s in reps], axis=0)

    res = np.concatenate([np.load(f) for f in sorted((store.path / "residuals").glob("residuals-*.npy"))])
    rows = []
    for a, tok in enumerate(tokens):
        for b, layer in enumerate(layers):
            c = res[idx, a, b].astype(np.float64)
            c = c - c.mean(0, keepdims=True)
            K = c @ c.T
            K *= len(idx) / np.trace(K)
            pred = np.mean([oof_predict(K, y, s) for s in reps], axis=0)
            joint = np.mean([oof_mixed(scene, K, y, s) for s in reps], axis=0)
            gap = np.bincount(codes, (y - scene_pred) ** 2 - (y - joint) ** 2) / np.bincount(codes)
            t, p = stats.ttest_1samp(gap, 0.0)
            rows.append({"model": model, "adjective": adjective, "mask": mask is not None, "n": len(idx), "index": a, "token": tok, "layer": layer,
                         "r2_cell": r(pred), "r2_scene": r(scene_pred), "r2_joint": r(joint), "gain": r(joint) - r(scene_pred), "t": float(t), "p": float(p)})
        print(f"[f10.24] {model}-{adjective}{tag}: gain token {a + 1}/{len(tokens)}", flush=True)
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].to_numpy())
    table.to_csv(out / f"gain{tag}.csv", index=False)
    (out / f"done{tag}.json").write_text(json.dumps({"version": __version__, "n": int(len(idx)), "mask": mask is not None, "tokens": len(tokens)}, indent=1) + "\n")


# ---------------------------------------------------------------- report (CPU)


def load(model: str, adjective: str) -> dict:
    d = DATA / store_name(model, adjective)
    z = np.load(d / "reads.npz")
    return {"model": model, "adjective": adjective, "dir": d, "z": {k: z[k] for k in z.files}, "frame": pd.read_parquet(d / "frame.parquet"),
            "svd": pd.read_csv(d / "svd.csv"), "sense": pd.read_csv(d / "sense.csv"), "vocab": pd.read_parquet(ALL_ROOT / store_name(model, adjective) / "vocab.parquet")}


def clean_words(vocab: pd.DataFrame) -> np.ndarray:
    """Give the mask of words that are plain alphabetic words of 3 letters or more. Used for word lists only."""
    t = vocab["token"].str.strip()
    return (t.str.isalpha() & (t.str.len() >= 3)).to_numpy()


def report() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    stores = {(m, a): load(m, a) for m in MODELS for a in ADJECTIVES}
    q1(stores)
    q2(stores)
    q3(stores)
    q4(stores)
    q5(stores)
    q6(stores)
    q7(stores)
    q8()
    print("[f10.24] report written to", FIG, flush=True)


def q1(stores: dict) -> None:
    """Pearson r of the margins between adjectives, at image level and at NTA level, and the yes rates."""
    rows, rates = [], []
    for m in MODELS:
        for a in ADJECTIVES:
            f = stores[(m, a)]["frame"]
            c = f["outcome"].value_counts(normalize=True)
            rates.append({"model": m, "adjective": a, "n": len(f), "yes": c.get("yes", 0.0), "no": c.get("no", 0.0), "unable": c.get("unable", 0.0)})
        for a, b in itertools.combinations(ADJECTIVES, 2):
            fa, fb = stores[(m, a)]["frame"], stores[(m, b)]["frame"]
            x = fa["margin"].to_numpy(float)
            y = fb["margin"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            r_img, p_img = stats.pearsonr(x[ok], y[ok])
            nta = pd.DataFrame({"nta": fa["nta2020"], "x": x, "y": y})[ok].groupby("nta").mean()
            r_nta, p_nta = stats.pearsonr(nta["x"], nta["y"])
            rows.append({"model": m, "pair": f"{a}-{b}", "n_images": int(ok.sum()), "r_image": r_img, "p_image": p_img,
                         "n_nta": len(nta), "r_nta": r_nta, "p_nta": p_nta})
    pd.DataFrame(rows).to_csv(FIG / "q1_margin_corr.csv", index=False)
    pd.DataFrame(rates).to_csv(FIG / "q1_outcome_shares.csv", index=False)


def q2(stores: dict) -> None:
    """No-image top words at the adjective and `?`, the sense axis of the NTA means, and the no-image sense score."""
    rows, sense_rows = [], []
    for (m, a), s in stores.items():
        clean = clean_words(s["vocab"])
        vocab = s["vocab"]["token"].str.strip().to_numpy().astype(str)
        text = s["z"]["text"]
        for p in (2, 3):
            for b, layer in enumerate(s["z"]["layers"]):
                v = np.where(clean, text[p, b], -np.inf)
                top = np.argsort(-v)[:10]
                for rank, i in enumerate(top, 1):
                    rows.append({"model": m, "adjective": a, "position": POSITIONS[p], "layer": int(layer), "rank": rank, "token": vocab[i], "z_no_image": float(text[p, b, i])})
        sense_rows.append(s["svd"].merge(s["sense"], on=["model", "adjective", "position", "token", "layer"], validate="1:1"))
    pd.DataFrame(rows).to_csv(FIG / "q2_no_image_top_words.csv", index=False)
    pd.concat(sense_rows, ignore_index=True).to_csv(FIG / "q2_sense_axis.csv", index=False)


def q3(stores: dict) -> None:
    """C2 = mean read minus the no-image read, and the r of C2 between adjectives. Sanity: equal reads before the adjective."""
    rows, sanity = [], []
    for m in MODELS:
        for a, b in itertools.combinations(ADJECTIVES, 2):
            za, zb = stores[(m, a)]["z"], stores[(m, b)]["z"]
            for p in (1, 2, 3, 4):
                for k, layer in enumerate(za["layers"]):
                    ca = za["z_all"][p, k] - za["text"][p, k]
                    cb = zb["z_all"][p, k] - zb["text"][p, k]
                    rows.append({"model": m, "pair": f"{a}-{b}", "position": POSITIONS[p], "token": str(za["token"][p]), "layer": int(layer),
                                 "r_c2_all_words": float(np.corrcoef(ca, cb)[0, 1])})
        for p in (0, 1):
            for k, layer in enumerate(stores[(m, "cool")]["z"]["layers"]):
                diffs = {k2: float(np.abs(stores[(m, k2)]["z"]["z_all"][p, k] - stores[(m, "cool")]["z"]["z_all"][p, k]).max()) for k2 in ADJECTIVES}
                sanity.append({"model": m, "position": POSITIONS[p], "layer": int(layer), "max_abs_diff_vs_cool_nice": diffs["nice"],
                               "max_abs_diff_vs_cool_ordinary": diffs["ordinary"]})
    pd.DataFrame(rows).to_csv(FIG / "q3_c2_correlation.csv", index=False)
    pd.DataFrame(sanity).to_csv(FIG / "q3_sanity_before_adjective.csv", index=False)


def q4(stores: dict) -> None:
    """C3 at the adjective: the mean of each borough minus the mean of the other images. Shared words, by top-30 overlap."""
    rows, words = [], []
    top_n = 30
    for m in MODELS:
        for layer_i, layer in enumerate(stores[(m, "cool")]["z"]["layers"]):
            lists = {}
            for a in ADJECTIVES:
                z = stores[(m, a)]["z"]
                clean = clean_words(stores[(m, a)]["vocab"])
                vocab = stores[(m, a)]["vocab"]["token"].str.strip().to_numpy().astype(str)
                n_total = z["counts_borough"].sum()
                for g, borough in enumerate(z["boroughs"]):
                    n_b = z["counts_borough"][g]
                    gap = z["z_borough"][g, 2, layer_i] - (n_total * z["z_all"][2, layer_i] - n_b * z["z_borough"][g, 2, layer_i]) / (n_total - n_b)
                    lists[(a, borough)] = (gap, clean, vocab)
            for borough in stores[(m, "cool")]["z"]["boroughs"]:
                for a, b in itertools.combinations(ADJECTIVES, 2):
                    ga, clean, vocab = lists[(a, borough)]
                    gb = lists[(b, borough)][0]
                    top_a = set(np.argsort(-np.where(clean, ga, -np.inf))[:top_n])
                    top_b = set(np.argsort(-np.where(clean, gb, -np.inf))[:top_n])
                    k = len(top_a & top_b)
                    p = stats.hypergeom.sf(k - 1, int(clean.sum()), top_n, top_n)
                    rows.append({"model": m, "borough": borough, "layer": int(layer), "pair": f"{a}-{b}", "r_c3_clean": float(np.corrcoef(ga[clean], gb[clean])[0, 1]),
                                 "top30_overlap": k, "top30_expected": top_n * top_n / clean.sum(), "p_hypergeom": float(p)})
                for a in ADJECTIVES:
                    ga, clean, vocab = lists[(a, borough)]
                    for rank, i in enumerate(np.argsort(-np.where(clean, ga, -np.inf))[:top_n], 1):
                        words.append({"model": m, "adjective": a, "borough": borough, "layer": int(layer), "rank": rank, "token": vocab[i], "gap_z": float(ga[i])})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_hypergeom"].to_numpy())
    out.to_csv(FIG / "q4_borough_c3.csv", index=False)
    pd.DataFrame(words).to_csv(FIG / "q4_borough_top_words.csv", index=False)


def q5(stores: dict) -> None:
    """Refusal on `nice`: the readout of unable images against yes and no images, and the probes with Welch t."""
    rows, probe_rows, words = [], [], []
    for (m, a), s in stores.items():
        if a != "nice":
            continue
        z = s["z"]
        clean = clean_words(s["vocab"])
        vocab = s["vocab"]["token"].str.strip().to_numpy().astype(str)
        yes, no, unable = (0, 1, 2)
        frame_ = s["frame"]
        for p in range(len(POSITIONS)):
            for k, layer in enumerate(z["layers"]):
                g = {o: z["z_outcome"][i, p, k] for i, o in enumerate(OUTCOMES)}
                for other in ("yes", "no"):
                    rows.append({"model": m, "adjective": a, "position": POSITIONS[p], "layer": int(layer), "contrast": f"unable-{other}",
                                 "r_mean_readout_clean": float(np.corrcoef(g["unable"][clean], g[other][clean])[0, 1]),
                                 "mean_diff_clean": float((g["unable"] - g[other])[clean].mean()), "sd_diff_clean": float((g["unable"] - g[other])[clean].std())})
                diff = np.where(clean, g["unable"] - (g["yes"] * z["counts_outcome"][yes] + g["no"] * z["counts_outcome"][no]) / (z["counts_outcome"][yes] + z["counts_outcome"][no]), -np.inf)
                for rank, i in enumerate(np.argsort(-diff)[:15], 1):
                    words.append({"model": m, "adjective": a, "position": POSITIONS[p], "layer": int(layer), "direction": "unable-high", "rank": rank, "token": vocab[i], "diff_z": float(diff[i])})
                diff_low = np.where(clean, diff, np.inf)
                for rank, i in enumerate(np.argsort(diff_low)[:15], 1):
                    words.append({"model": m, "adjective": a, "position": POSITIONS[p], "layer": int(layer), "direction": "yes-no-high", "rank": rank, "token": vocab[i], "diff_z": float(diff[i])})
                for j, name in enumerate(z["probe_names"]):
                    v = z["probes"][:, p, k, j].astype(float)
                    is_u = frame_["outcome"].to_numpy() == "unable"
                    t, pv = stats.ttest_ind(v[is_u], v[~is_u], equal_var=False)
                    probe_rows.append({"model": m, "adjective": a, "position": POSITIONS[p], "layer": int(layer), "probe": str(name),
                                       "mean_unable": float(v[is_u].mean()), "mean_yes_no": float(v[~is_u].mean()), "welch_t": float(t), "p": float(pv)})
    out = pd.DataFrame(probe_rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    out.to_csv(FIG / "q5_refusal_probe_tests.csv", index=False)
    pd.DataFrame(rows).to_csv(FIG / "q5_refusal_readout.csv", index=False)
    pd.DataFrame(words).to_csv(FIG / "q5_refusal_top_words.csv", index=False)


def first_run(layers: list[int], ok: list[bool]) -> list[int]:
    run: list[int] = []
    for layer, passed in zip(layers, ok):
        if passed:
            run.append(layer)
        elif run:
            break
    return run


def q6(stores: dict) -> None:
    """Phase bands of F10.20 for each store. The verdict band needs the lens margin and is empty where it is not run."""
    rows = []
    for (m, a), s in stores.items():
        d = s["dir"]
        patch = pd.read_csv(d / "patch_depth.csv") if (d / "patch_depth.csv").exists() else None
        if patch is not None:
            peak = patch.loc[patch["r2"].idxmax()]
            onset = int(patch[patch["r2"] >= 0.9 * peak["r2"]]["layer"].min())
            rows.append({"model": m, "adjective": a, "phase": "scene", "first": 0, "onset": onset, "last": int(peak["layer"]), "peak": float(peak["r2"]),
                         "rule": "mean-patch R2 at least 0.9 of peak, to the peak"})
        sv = s["svd"].query("position == 'adj'").sort_values("layer")
        run = first_run(sv["layer"].tolist(), ((sv["share1"] >= 0.40) & (sv["r_axis1"] >= 0.95)).tolist())
        if run:
            rows.append({"model": m, "adjective": a, "phase": "sense", "first": run[0], "onset": run[0], "last": run[-1], "peak": float(sv["share1"].max()),
                         "rule": "axis 1 share at least 0.40 and abs r at least 0.95, at the adjective"})
        else:
            rows.append({"model": m, "adjective": a, "phase": "sense", "first": np.nan, "onset": np.nan, "last": np.nan, "peak": float(sv["share1"].max()),
                         "rule": "no layer passes"})
        gain_path = d / "gain.csv"
        if gain_path.exists():
            g = pd.read_csv(gain_path).query("token == '?'").sort_values("layer")
            onset = int(g[g["gain"] >= 0.5 * g["gain"].max()]["layer"].min())
            rows.append({"model": m, "adjective": a, "phase": "standard", "first": onset, "onset": onset, "last": np.nan, "peak": float(g["gain"].max()),
                         "rule": "gain over the mean patch at least half of its peak, at ?"})
        rows.append({"model": m, "adjective": a, "phase": "verdict", "first": np.nan, "onset": np.nan, "last": np.nan, "peak": np.nan,
                     "rule": "not run: needs the lens margin (job 114592)"})
    pd.DataFrame(rows).to_csv(FIG / "q6_phase_bands.csv", index=False)


def q7(stores: dict) -> None:
    """Gain over the mean patch for each store, and the check against the F10.9 file for cool."""
    frames = []
    for (m, a), s in stores.items():
        for name in ("gain.csv", "gain-mask-nice.csv"):
            path = s["dir"] / name
            if path.exists():
                frames.append(pd.read_csv(path).assign(file=name))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(FIG / "q7_gain.csv", index=False)
    check = []
    for m in MODELS:
        ref_path = ROOT / "data/f10/hop_free" / m / "decoder_grid1.csv"
        if not ref_path.exists() or not (DATA / m / "gain.csv").exists():
            continue
        ref = pd.read_csv(ref_path).query("token == '?'")
        mine = pd.read_csv(DATA / m / "gain.csv").query("token == '?'")
        j = ref.merge(mine, on=["token", "layer"], suffixes=("_ref", "_new"))
        check.append({"model": m, "n_layers": len(j), "max_abs_diff_gain": float((j["gain_ref"] - j["gain_new"]).abs().max())})
    pd.DataFrame(check).to_csv(FIG / "q7_gain_check_vs_f10_9.csv", index=False)


def q8() -> None:
    """Lens margin of the last token against the answer margin (the verdict band of F10.20), on the store images.

    Gemma `nice` uses only the yes or no images (D167). The other stores hold a margin for every image.
    """
    sys.path.insert(0, str(ROOT / "research/scripts/f10"))
    from analyze_logit import LOGIT_ROOT, cluster_r, load, margin_of  # noqa: E402

    rows = []
    for m in MODELS:
        for a in ADJECTIVES:
            name = store_name(m, a)
            if not (LOGIT_ROOT / name / "metrics").exists() or not any((LOGIT_ROOT / name / "metrics").glob("metrics-*.npz")):
                continue
            d = load(name)
            fr = frame(m, a)  # same image order as the lens run; its margin is the store margin
            cell = margin_of(d["metrics"]["lens"])
            y = fr["margin"].to_numpy(float)
            keep = np.isfinite(y)
            last = cell.shape[1] - 1
            for b, layer in enumerate(d["head"]["layers"]):
                x = cell[keep, last, b]
                r, t, p = cluster_r(x, y[keep], fr["nta2020"].to_numpy()[keep])
                rows.append({"model": m, "adjective": a, "n": int(keep.sum()), "layer": layer, "r_last_token": r, "t": t, "p": p})
    out = pd.DataFrame(rows)
    out.to_csv(FIG / "q8_verdict_lens.csv", index=False)
    print("[f10.24] q8 rows", len(out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--model", required=True, choices=MODELS)
    j.add_argument("--adjective", required=True, choices=ADJECTIVES)
    j.add_argument("--gain-only", action="store_true")
    j.add_argument("--mask-from", choices=ADJECTIVES)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "job":
        job(args.model, args.adjective, args.gain_only, args.mask_from)
    else:
        report()


if __name__ == "__main__":
    main()
