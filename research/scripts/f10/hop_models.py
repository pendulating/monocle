#!/usr/bin/env python
"""F10.24: compare 3 models on 3 adjectives (`cool`, `nice`, `ordinary`). CPU only.

Stage 1, `build`: read one store (1 model and 1 adjective) and save the reads that the
comparison needs. Stage 2, `analyze`: read the 9 saved stores and write the tables of
`research/figures/f10/models/`.

Positions. The adjective token is `after_place` in every store (the token ` cool`, ` nice`
or ` ordinary`). `after_cool` is the verdict-free `?` token. `place` is the token ` place`.

Softcap. The `hop_all` reads of Gemma are capped at 30. The `cool` reads of Gemma come from
`hop_logit/gemma-4-31b/uncapped`, as in `hop_compare.py`. The other Gemma reads are inverted with
CAP * arctanh(x / CAP). The inverse loses precision for the top words, so the `nice` and
`ordinary` reads of Gemma carry a small error. The `cool` check in `analyze` measures it.

| Saved in `data/f10/hop_models/{store}/` | Shape | Content |
|------------------------------------------|-------|---------|
| `full_mean.npy` | [3 positions, layers, vocab] | mean over images of the z-scored read |
| `text_full.npy` | [3, layers, vocab] | z-scored read of the same text with no image |
| `nta_shared.npy` | [NTAs, 3, layers, shared words] | NTA means of the z-scored read |
| `text_shared.npy` | [3, layers, shared words] | no-image read, shared words |
| `sense.npy` | [images, 3, layers] | sense score of F10.17 (social minus temperature) |
| `sense_text.npy` | [3, layers] | sense score with no image |
| `resid_adj.npy` | [images, layers, d] | residual at the adjective token |
| `images.parquet` | 1 row for each image | `image_idx`, `nta2020`, `borough` |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_models.py build --model gemma-4-31b --adj nice
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_models.py analyze
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from analyze_logit import cluster_r, holm  # noqa: E402
from hop_borough import BOROUGHS, two_sample_t  # noqa: E402
from hop_free import rdm_ranks  # noqa: E402
from hop_sense import CAP, SOCIAL, TEMPERATURE  # noqa: E402
from hop_store import ALL_ROOT, open_store  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
SHORT = {"gemma-4-31b": "G", "qwen3.5-9b": "Q9", "qwen3.6-27b": "Q27"}
ADJECTIVES = ("cool", "nice", "ordinary")
SUFFIX = {"cool": "", "nice": "-nice", "ordinary": "-ordinary"}
POS = ("place", "after_place", "after_cool")
ADJ_POS = 1  # index of `after_place` in POS
WORK = ROOT / "data/f10/hop_models"
HOP_ALL = ROOT / "data/f10/hop_all"
HOP_LOGIT = ROOT / "data/f10/hop_logit"
OUT = ROOT / "research/figures/f10/models"


def store_name(model: str, adj: str) -> str:
    return model + SUFFIX[adj]


def invert(x: np.ndarray) -> np.ndarray:
    """Give the logits before the softcap of Gemma. The inverse of CAP * tanh(x / CAP)."""
    return (CAP * np.arctanh((x.astype(np.float64) / CAP).clip(-1 + 1e-7, 1 - 1e-7))).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    """Give the z-score over the whole vocabulary, which is the last axis."""
    x = x.astype(np.float64)
    return ((x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)).astype(np.float32)


def shared_words() -> list[str]:
    """Give the lower-case words of all 3 models, from the `hop_all` vocabularies of the cool stores."""
    sets = [set(open_store(m, ALL_ROOT).vocab["token"].str.strip().str.lower()) for m in MODELS]
    return sorted(set.intersection(*sets))


def first_index(vocab: np.ndarray, words: list[str]) -> np.ndarray:
    """Give the first index of each word in the vocabulary, as `hop_compare.gather` does."""
    s = pd.Series(np.arange(len(vocab)), index=vocab)
    return s[~s.index.duplicated()].loc[words].to_numpy()


def build(model: str, adj: str, shards: int | None = None) -> None:
    """Save the reads of 1 store. `shards` limits the run for a smoke test."""
    name = store_name(model, adj)
    st = open_store(name, ALL_ROOT)
    man = st.manifest
    positions = man["positions"]
    tail = man["tail_tokens"]
    pos_idx = [positions.index(p) for p in POS]
    want = " " + adj
    assert tail[positions.index("after_place")].strip() == adj, (name, tail[positions.index("after_place")])
    assert tail[positions.index("place")] == " place", name
    layers = man["layers"]
    vocab = st.vocab["token"].str.strip().str.lower().to_numpy()
    words = shared_words()
    pick = first_index(vocab, words)
    images = st.images.reset_index(drop=True)
    codes = pd.factorize(images["nta2020"])[0]
    n_nta = codes.max() + 1
    counts = np.bincount(codes)
    hot = np.flatnonzero(pd.Series(vocab).isin(TEMPERATURE).to_numpy())
    soc = np.flatnonzero(pd.Series(vocab).isin(SOCIAL).to_numpy())

    if model == "gemma-4-31b" and adj == "cool":
        reads = sorted((HOP_LOGIT / model / "uncapped").glob("reads-*.npy"))
        cap = False
    else:
        reads = sorted((HOP_ALL / name / "reads").glob("reads-*.npy"))
        cap = model == "gemma-4-31b"
    resid = sorted((HOP_ALL / name / "residuals").glob("residuals-*.npy"))
    if shards:
        reads, resid = reads[:shards], resid[:shards]
    n = sum(np.load(f, mmap_mode="r").shape[0] for f in reads)
    n_pos, n_layer = len(POS), len(layers)
    full = np.zeros((n_pos, n_layer, len(vocab)), np.float64)
    nta = np.zeros((n_nta, n_pos, n_layer, len(words)), np.float32)
    sense = np.zeros((n, n_pos, n_layer), np.float32)
    res = np.zeros((n, n_layer, np.load(resid[0], mmap_mode="r").shape[-1]), np.float32)
    start = 0
    for f, g in zip(reads, resid):
        x = np.asarray(np.load(f, mmap_mode="r")[:, pos_idx], np.float32)
        if cap:
            x = invert(x)
        z = zscore(x)
        k = len(z)
        rows = np.arange(start, start + k)
        full += z.sum(0)
        sense[rows] = z[..., soc].mean(-1) - z[..., hot].mean(-1)
        zp = z[..., pick]
        for c in np.unique(codes[rows]):
            nta[c] += zp[codes[rows] == c].sum(0)
        res[rows] = np.load(g, mmap_mode="r")[:, positions.index("after_place")]
        start += k
        print(f"[f10.24] {name}: {start} images", flush=True)
    done = start
    text = np.asarray(np.load(HOP_ALL / name / "text_reads.npy", mmap_mode="r")[pos_idx], np.float32)
    if model == "gemma-4-31b":  # the text reads are capped for each adjective
        text = invert(text)
    text = zscore(text)
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if shards else ""
    np.save(out / f"full_mean{tag}.npy", (full[:, :, :] / done).astype(np.float32))
    np.save(out / f"text_full{tag}.npy", text)
    np.save(out / f"nta_shared{tag}.npy", (nta / counts[:, None, None, None]).astype(np.float32))
    np.save(out / f"text_shared{tag}.npy", text[..., pick])
    np.save(out / f"sense{tag}.npy", sense)
    np.save(out / f"sense_text{tag}.npy", text[..., soc].mean(-1) - text[..., hot].mean(-1))
    np.save(out / f"resid_adj{tag}.npy", res)
    images.assign(codes=codes)[["image_idx", "nta2020", "borough", "codes"]].to_parquet(out / "images.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"store": name, "positions": list(POS), "layers": layers, "words": len(words), "n_images": done,
                                               "cap_inverted": cap, "uncapped_source": not cap and model == "gemma-4-31b"}, indent=1) + "\n")
    if not shards:
        (WORK / "shared_words.json").write_text(json.dumps(words) + "\n")
    print(f"[f10.24] {name}: saved {done} images to {out}", flush=True)


def load_store(name: str) -> dict:
    """Give the saved arrays of 1 store."""
    d = WORK / name
    meta = json.loads((d / "meta.json").read_text())
    im = pd.read_parquet(d / "images.parquet")
    return {"name": name, "meta": meta, "full": np.load(d / "full_mean.npy", mmap_mode="r"), "text_full": np.load(d / "text_full.npy", mmap_mode="r"),
            "nta": np.load(d / "nta_shared.npy"), "text": np.load(d / "text_shared.npy"), "sense": np.load(d / "sense.npy"),
            "sense_text": np.load(d / "sense_text.npy"), "resid": np.load(d / "resid_adj.npy", mmap_mode="r"), "images": im,
            "codes": im["codes"].to_numpy(), "layers": meta["layers"]}


def outcomes() -> pd.DataFrame:
    """Give 1 row for each model, adjective and image, from `hop_outcomes.py`. The margin is NaN for a refusal."""
    o = pd.read_parquet(ROOT / "research/figures/f10/outcomes/outcomes.parquet")
    return o


def wide(o: pd.DataFrame, adj: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the margins and the outcomes, 1 column for each model, in the order of `images.parquet`."""
    g = o[o["adjective"] == adj]
    base = open_store("gemma-4-31b", ALL_ROOT).images[["image_idx", "nta2020", "borough"]].reset_index(drop=True)
    m = base.copy()
    t = base.copy()
    for model in MODELS:
        s = g[g["model"] == model].set_index("image_idx")
        m[SHORT[model]] = s.loc[base["image_idx"], "margin"].to_numpy()
        t[SHORT[model]] = s.loc[base["image_idx"], "outcome"].to_numpy()
    return m, t


def verdict_pairs(m: pd.DataFrame, t: pd.DataFrame, adj: str) -> pd.DataFrame:
    """Give r, rho, AUC, kappa and the matched-prevalence agreement of each pair, on the images with a verdict in both models."""
    rows = []
    for a, b in (("G", "Q27"), ("G", "Q9"), ("Q27", "Q9")):
        ok = m[a].notna() & m[b].notna()
        x, y = m.loc[ok, a].to_numpy(), m.loc[ok, b].to_numpy()
        nta = m.loc[ok, "nta2020"].to_numpy()
        r, tt, p = cluster_r(x, y, nta)
        ya = (x > 0)
        yb = (y > 0)
        po = np.mean(ya == yb)
        pe = ya.mean() * yb.mean() + (1 - ya.mean()) * (1 - yb.mean())
        # Matched prevalence: call the top share of b `yes`, with the share of yes in a.
        cut = np.quantile(y, 1 - ya.mean())
        agree_matched = np.mean(ya == (y >= cut))
        auc_b_for_a = stats.mannwhitneyu(y[ya], y[~ya]).statistic / (ya.sum() * (~ya).sum()) if ya.any() and (~ya).any() else np.nan
        rows.append({"adjective": adj, "a": a, "b": b, "n": int(ok.sum()), "r": r, "t_cluster": tt, "p_cluster": p,
                     "rho": stats.spearmanr(x, y)[0], "auc_b_for_a_verdict": auc_b_for_a, "yes_a": ya.mean(), "yes_b": yb.mean(),
                     "kappa": (po - pe) / (1 - pe), "agree_own_threshold": po, "agree_matched_prevalence": agree_matched,
                     "sd_a": x.std(ddof=1), "sd_b": y.std(ddof=1)})
    return pd.DataFrame(rows)


def verdict_nta_tests(m: pd.DataFrame, adj: str) -> pd.DataFrame:
    """Give the Williams test of the 3 pivots, on the NTA means of the margins. Only NTAs with a verdict in all 3 models count."""
    ok = m[["G", "Q27", "Q9"]].notna().all(axis=1)
    mm = m[ok].groupby("nta2020")[["G", "Q27", "Q9"]].mean()
    mm = mm.dropna()
    n = len(mm)
    rows = []
    for pivot, u, v in (("G", "Q27", "Q9"), ("Q27", "G", "Q9"), ("Q9", "G", "Q27")):
        r12 = np.corrcoef(mm[pivot], mm[u])[0, 1]
        r13 = np.corrcoef(mm[pivot], mm[v])[0, 1]
        r23 = np.corrcoef(mm[u], mm[v])[0, 1]
        det = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
        bar = (r12 + r13) / 2
        t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) / (2 * (n - 1) / (n - 3) * det + bar ** 2 * (1 - r23) ** 3))
        rows.append({"adjective": adj, "pivot": pivot, "first": u, "second": v, "r_first": r12, "r_second": r13, "r_others": r23, "n_nta": n,
                     "williams_t": t, "p": 2 * stats.t.sf(abs(t), n - 3)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out


def yes_lean(o: pd.DataFrame) -> pd.DataFrame:
    """Give the margin SD, the yes share and the margin mean of each model and adjective, with the verdict images only."""
    rows = []
    for (model, adj), g in o.groupby(["model", "adjective"]):
        v = g["margin"].dropna()
        rows.append({"model": model, "adjective": adj, "n_verdict": len(v), "yes_share_all": (g["outcome"] == "yes").mean(),
                     "margin_mean": v.mean(), "margin_sd": v.std(ddof=1), "margin_mean_abs": v.abs().mean()})
    return pd.DataFrame(rows)


def refusal() -> pd.DataFrame:
    """Give the Qwen margins on the images that Gemma refuses, against the images with a Gemma verdict, for each adjective."""
    o = outcomes()
    rows = []
    for adj in ADJECTIVES:
        mm, tt = wide(o, adj)
        ref = tt["G"].eq("unable").to_numpy()
        if ref.sum() < 2:
            continue
        for col in ("Q27", "Q9"):
            x = mm[col].to_numpy()
            a = x[ref]
            a = a[~np.isnan(a)]
            b = np.abs(x[~ref & ~np.isnan(x)])
            t_one = stats.ttest_1samp(a, 0.0)
            t_two = stats.ttest_ind(np.abs(a), b, equal_var=False)
            yes = tt[col].eq("yes").to_numpy()
            tab = pd.crosstab(ref, yes).to_numpy()
            chi_p = stats.chi2_contingency(tab, correction=False)[1] if tab.shape == (2, 2) else np.nan
            rows.append({"adjective": adj, "qwen": col, "n_gemma_refused": int(ref.sum()), "n_margin": len(a), "margin_mean_refused": a.mean(),
                         "margin_sd_refused": a.std(ddof=1), "mean_abs_refused": np.abs(a).mean(), "mean_abs_gemma_verdict": b.mean(),
                         "t_margin_vs_0": t_one.statistic, "p_margin_vs_0": t_one.pvalue, "t_abs_refused_vs_verdict": t_two.statistic,
                         "p_abs_refused_vs_verdict": t_two.pvalue, "yes_share_refused": yes[ref].mean(), "yes_share_verdict": yes[~ref].mean(),
                         "p_chi2_yes_by_refusal": chi_p, "share_abs_over_2_refused": (np.abs(a) > 2).mean()})
    out = pd.DataFrame(rows)
    out["p_holm_margin_vs_0"] = holm(out["p_margin_vs_0"].to_numpy())
    return out


def borough_margins(o: pd.DataFrame) -> pd.DataFrame:
    """Give the NTA mean margin of each borough against the other 4, for each model and adjective (verdict images)."""
    rows = []
    nta_of = open_store("gemma-4-31b", ALL_ROOT).images[["image_idx", "nta2020"]]
    o = o.merge(nta_of, on="image_idx", validate="m:1")
    for (model, adj), g in o.groupby(["model", "adjective"]):
        g = g.dropna(subset=["margin"])
        nta = g.groupby("nta2020").agg(margin=("margin", "mean"), borough=("borough", "first"))
        ps = []
        for b in BOROUGHS:
            mask = (nta["borough"] == b).to_numpy()
            t = two_sample_t(nta["margin"].to_numpy()[:, None], mask)[0]
            ps.append(2 * stats.t.sf(abs(t), len(nta) - 2))
            rows.append({"model": model, "adjective": adj, "borough": b, "n_nta": int(mask.sum()), "mean_margin": g.loc[g["borough"] == b, "margin"].mean(),
                         "mean_of_nta_means": nta.loc[mask, "margin"].mean(), "t_vs_rest": t, "p": ps[-1]})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for (model, adj), idx in out.groupby(["model", "adjective"]).groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p"].to_numpy())
    return out


def margin_tables() -> None:
    """Write the tables that need only the margins and the outcomes."""
    OUT.mkdir(parents=True, exist_ok=True)
    o = outcomes()
    pairs, nta_tests = [], []
    for adj in ADJECTIVES:
        m, t = wide(o, adj)
        pairs.append(verdict_pairs(m, t, adj))
        nta_tests.append(verdict_nta_tests(m, adj))
    pairs = pd.concat(pairs, ignore_index=True)
    pairs.to_csv(OUT / "q2_verdict_pairs.csv", index=False)
    tests = pd.concat(nta_tests, ignore_index=True)
    tests["p_holm"] = holm(tests["p"].to_numpy())
    tests.to_csv(OUT / "q2_verdict_williams.csv", index=False)
    yes_lean(o).to_csv(OUT / "q3_margin_scale.csv", index=False)
    refusal().to_csv(OUT / "q6_refusal.csv", index=False)
    borough_margins(o).to_csv(OUT / "q5_borough_margins.csv", index=False)
    print("[f10.24] margin tables written", flush=True)


def reproduction(S: dict | None = None) -> pd.DataFrame:
    """Give the checks of the new loader against the published values of F10.19, F10.17 and F10.21."""
    rows = []
    cd = pd.read_csv(COMPARE / "cross_depth.csv")
    words = pd.read_csv(OUT / "q2_word_rsa_by_depth.csv")
    cool_w = words[words["adjective"] == "cool"]
    for pair, lab in (("G-Q27", "G/Q27"), ("G-Q9", "G/Q9"), ("Q27-Q9", "Q27/Q9")):
        ref = cd[(cd["token"] == "cool") & (cd["a"] + "/" + cd["b"] == {"G-Q27": "gemma-4-31b/qwen3.6-27b", "G-Q9": "gemma-4-31b/qwen3.5-9b", "Q27-Q9": "qwen3.6-27b/qwen3.5-9b"}[pair])]
        mine = cool_w[cool_w["pair"] == pair]
        rows.append({"check": f"word r of the mean readout at cool, {lab}, per depth (max abs diff)", "reference": "cross_depth.csv r_mean",
                     "value": float(np.max(np.abs(mine["word_mean"].to_numpy() - ref.groupby("depth")["r_mean"].mean().reindex(mine["f"]).to_numpy()))),
                     "note": "0 means equal; the reference is the mean over depths in the file"})
    sl = pd.read_csv(OUT / "q4_sense_by_layer.csv")
    for model, ref in (("gemma-4-31b", .90), ("qwen3.5-9b", .34), ("qwen3.6-27b", .16)):
        v = sl[(sl["model"] == model) & (sl["layer"] == 24)]["share_above_no_image"]
        rows.append({"check": f"sense share above no image at cool L24, {model}", "reference": ref, "value": float(v.iloc[0]) if len(v) else np.nan, "note": "F10.19 table"})
    sn = pd.read_csv(COMPARE / "sense.csv")
    for model in MODELS:
        g = sn[(sn["model"] == model) & (sn["token"] == "cool") & (sn["layer"] == 24)]
        mine = sl[(sl["model"] == model) & (sl["layer"] == 24)]["share_axis1"]
        rows.append({"check": f"axis 1 share at cool L24, {model}", "reference": float(g["share1"].iloc[0]), "value": float(mine.iloc[0]) if len(mine) else np.nan,
                     "note": "sense.csv (hop_compare) vs NTA means of this script"})
    rc = pd.read_csv(FIG / "mechanism/rank_cells.csv", keep_default_na=False)
    wc = pd.read_csv(OUT / "q1_word_counts_by_layer.csv")
    for model, tok, layer in (("gemma-4-31b", " place", 40), ("gemma-4-31b", " cool", 36), ("gemma-4-31b", "?", 40), ("qwen3.5-9b", " place", 16),
                              ("qwen3.5-9b", " cool", 24), ("qwen3.5-9b", "?", 20)):
        g = rc[(rc["model"] == model) & (rc["token"] == tok) & (rc["layer"] == layer)]
        ref = {"embedded": int((g["kind"] == "embedded").sum()), "raised": int((g["kind"] == "raised").sum()), "lowered": int((g["kind"] == "lowered").sum())}
        lab = {" place": "place", " cool": "cool", "?": "question_mark"}[tok]
        mine = wc[(wc["model"] == model) & (wc["token"] == lab) & (wc["layer"] == layer)]
        if len(mine):
            rows.append({"check": f"word counts at {lab} L{layer}, {model} (embedded/raised/lowered)", "reference": f"{ref['embedded']}/{ref['raised']}/{ref['lowered']}",
                         "value": f"{int(mine['embedded'].iloc[0])}/{int(mine['raised'].iloc[0])}/{int(mine['lowered'].iloc[0])}", "note": "rank_cells.csv"})
    return pd.DataFrame(rows)





def analyze() -> None:
    """Write the tables of F10.24. Needs the 9 stores of `build`, except for the margin tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    margin_tables()
    o = outcomes()
    analyze_reads(o)
    reproduction().to_csv(OUT / "q0_reproduction.csv", index=False)
    sl = sense_l24_table()
    rep = replication(None, sl)
    rep.to_csv(OUT / "q1_replication.csv", index=False)
    print("[f10.24] analysis written", flush=True)


def sense_l24_table() -> pd.DataFrame:
    """Give the sense values at L24 that `replication` needs, from `sense.csv`."""
    s = sense_l24()
    return s



# ---- F10.24 question 1: the replication of each F10 finding for the Qwen3.6-27B `cool` analyses. ----
FIG = ROOT / "research/figures/f10"
COMPARE = FIG / "compare"


def tail_index(name: str, token: str) -> int:
    """Give the last tail index of a token, as `hop_compare.token_index` does."""
    tail = open_store(name, ALL_ROOT).manifest["tail_tokens"]
    return len(tail) - 1 - tail[::-1].index(token)


def sets_question_window() -> pd.DataFrame:
    """Give the best |r| of each pair in the question window (tokens up to `?`), with `p_holm`, for the 3 models."""
    rows = []
    sources = {"gemma-4-31b": FIG / "sets/sets_pairs.csv", "qwen3.5-9b": FIG / "sets/sets_pairs.csv", "qwen3.6-27b": COMPARE / "sets/sets_pairs.csv"}
    for model in MODELS:
        p = pd.read_csv(sources[model]).query("model == @model")
        end = tail_index(model, "?")
        for pair in ("aesthetic", "status", "valence"):
            g = p[(p["pair"] == pair) & (p["index"] <= end)]
            best = g.loc[g["r"].abs().idxmax()]
            rows.append({"model": model, "measure": f"set pair {pair}, question window", "value": best["r"], "token": best["token"], "layer": best["layer"],
                         "p_holm": best["p_holm"], "n_cells": len(g)})
    return pd.DataFrame(rows)


def gain_table() -> pd.DataFrame:
    """Give the largest gain over the mean patch at `place`, at ` cool` and at `?`, with the layer."""
    rows = []
    for model in MODELS:
        d = pd.read_csv(WORK_FREE / model / "decoder_grid1.csv")
        for token, label in ((" place", "place"), (" cool", "cool"), ("?", "question_mark")):
            g = d[d["token"] == token]
            best = g.loc[g["gain"].idxmax()]
            rows.append({"model": model, "measure": f"gain over mean patch at {label}", "value": best["gain"], "layer": best["layer"],
                         "p_holm": best["p_holm"], "r2_scene": d["r2_scene"].iloc[0]})
    return pd.DataFrame(rows)


WORK_FREE = ROOT / "data/f10/hop_free"


def sense_l24() -> pd.DataFrame:
    """Give the axis-1 share and r of the sense score with the margin, at `cool` L24, for the 3 models (`sense.csv`)."""
    rows = []
    for path in (COMPARE / "sense.csv",):
        s = pd.read_csv(path)
        for model in MODELS:
            g = s[(s["model"] == model) & (s["token"] == "cool") & (s["layer"] == 24)]
            if len(g):
                r = g.iloc[0]
                rows.append({"model": model, "measure": "axis 1 share at cool L24", "value": r["share1"], "r_axis1": r["r_axis1"], "r_margin": r["r_margin"]})
    return pd.DataFrame(rows)


def margin_r_cells() -> pd.DataFrame:
    """Give the largest r of a cell margin with the answer margin at ` cool` and `?`, lens read, and the Holm count of held-out word lists."""
    rows = []
    lm = pd.concat([pd.read_csv(FIG / "logit/logit_margin.csv"), pd.read_csv(COMPARE / "logit/logit_margin.csv")], ignore_index=True)
    lw = pd.concat([pd.read_csv(FIG / "logit/logit_words.csv"), pd.read_csv(COMPARE / "logit/logit_words.csv")], ignore_index=True)
    for model in MODELS:
        g = lm[(lm["model"] == model) & (lm["lens"] == "lens")]
        for token in (" cool", "?"):
            h = g[g["token"] == token]
            best = h.loc[h["r"].idxmax()]
            rows.append({"model": model, "measure": f"largest r of cell margin with answer margin at {token.strip()}", "value": best["r"],
                         "layer": best["layer"]})
        w = lw[lw["model"] == model]
        rows.append({"model": model, "measure": "held-out word lists passing Holm (cells)", "value": int((w["p_holm"] < 0.05).sum()), "n_cells": len(w)})
    return pd.DataFrame(rows)


# ---- F10.24: the read-level analyses, on the saved stores of `build`. ----
from hop_compare import williams  # noqa: E402  (same Williams t as F10.19)
from hop_depth import N_LAYERS  # noqa: E402
from analyze_logit import cluster_r as _cluster_r  # noqa: E402
from hop_free import rdm_ranks  # noqa: E402

MSET = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
LABEL = {"G": "gemma-4-31b", "Q9": "qwen3.5-9b", "Q27": "qwen3.6-27b"}
PAIRS = (("G", "Q27"), ("G", "Q9"), ("Q27", "Q9"))


def near(model: str, layers: list[int], f: float) -> int:
    """Give the saved layer at the nearest relative depth (q + 1) / (N + 1), as `hop_compare.depth_rows` does."""
    return min(layers, key=lambda q: abs((q + 1) / (N_LAYERS[model] + 1) - f))


def depth_units(stores: dict) -> list[tuple[float, dict]]:
    """Give the matched depths of the 3 models. The model with the fewest saved layers (Qwen3.5-9B) sets the depths."""
    anchor = stores["cool"]["Q9"]["layers"]
    out = []
    for q in anchor:
        f = round((q + 1) / (N_LAYERS["qwen3.5-9b"] + 1), 3)
        out.append((f, {k: near(LABEL[k], stores["cool"][k]["layers"], f) for k in ("G", "Q9", "Q27")}))
    return out


def load_all() -> dict:
    """Give the saved stores, by adjective and model key."""
    out = {}
    for adj in ADJECTIVES:
        out[adj] = {}
        for k, model in LABEL.items():
            out[adj][k] = load_store(store_name(model, adj))
    return out


def fisher_tests(df: pd.DataFrame, measure: str, adj: str) -> list[dict]:
    """Give the paired t of Fisher z over the matched depths: family (Q27-Q9 against each G pair) and size (G-Q27 against G-Q9)."""
    z = df.assign(z=np.arctanh(df[measure].clip(-0.999, 0.999))).pivot_table(index="f", columns="pair", values="z")
    rows = []
    for name, a, b in (("family: Q27-Q9 vs G-Q27", "Q27-Q9", "G-Q27"), ("family: Q27-Q9 vs G-Q9", "Q27-Q9", "G-Q9"), ("size: G-Q27 vs G-Q9", "G-Q27", "G-Q9")):
        gap = z[a] - z[b]
        t, p = stats.ttest_1samp(gap, 0.0)
        rows.append({"adjective": adj, "measure": measure, "test": name, "r_first": np.tanh(z[a].mean()), "r_second": np.tanh(z[b].mean()),
                     "depths": len(gap), "t": t, "p": p})
    return rows


def word_and_rsa(S: dict, units: list) -> tuple[pd.DataFrame, list]:
    """Give the word r (mean, added, no image) and the residual RDM r at the adjective token, at each matched depth, for each adjective."""
    rows, tests = [], []
    for adj in ADJECTIVES:
        per = []
        for f, lay in units:
            vec, text, rdm = {}, {}, {}
            for k in ("G", "Q9", "Q27"):
                st = S[adj][k]
                li = st["layers"].index(lay[k])
                mean_w = st["nta"][:, ADJ_POS, li].mean(0)
                vec[k] = mean_w
                text[k] = st["text"][ADJ_POS, li]
                rdm[k] = rdm_ranks(np.asarray(st["resid"][:, li], np.float64))
            for a, b in PAIRS:
                key = f"{a}-{b}"
                ra = np.corrcoef(vec[a], vec[b])[0, 1]
                rad = np.corrcoef(vec[a] - text[a], vec[b] - text[b])[0, 1]
                rt = np.corrcoef(text[a], text[b])[0, 1]
                rs = rdm[a] @ rdm[b] / len(rdm[a])
                per.append({"adjective": adj, "f": f, "pair": key, "word_mean": ra, "word_added": rad, "word_no_image": rt, "rsa_residual": rs,
                            "layer_" + a: lay[a], "layer_" + b: lay[b]})
        df = pd.DataFrame(per)
        rows.append(df)
        for measure in ("word_mean", "word_added", "word_no_image", "rsa_residual"):
            tests.extend(fisher_tests(df, measure, adj))
    out = pd.concat(rows, ignore_index=True)
    return out, tests


def sense_by_layer(S: dict, o: pd.DataFrame) -> pd.DataFrame:
    """Give the sense score at `cool` for every layer of each model: the no-image value, the mean, the share above no image, and r with the margin."""
    rows = []
    margin = wide(o, "cool")[0]
    for k in ("G", "Q9", "Q27"):
        st = S["cool"][k]
        m = margin[k].to_numpy()
        for li, lay in enumerate(st["layers"]):
            s = st["sense"][:, ADJ_POS, li]
            t0 = float(st["sense_text"][ADJ_POS, li])
            share = float((s > t0).mean())
            p = stats.binomtest(int((s > t0).sum()), len(s)).pvalue
            r, tt, pc = _cluster_r(s, m, st["images"]["nta2020"].to_numpy())
            # Share of NTA variance on axis 1, from the NTA means of the shared words (as in `hop_compare.sense`).
            c = st["nta"][:, ADJ_POS, li]
            c = c - c.mean(0, keepdims=True)
            w = np.linalg.eigh(c @ c.T)[0]
            rows.append({"model": LABEL[k], "layer": lay, "depth": round((lay + 1) / (N_LAYERS[LABEL[k]] + 1), 3), "no_image": t0, "mean_score": float(s.mean()),
                         "mean_minus_no_image": float(s.mean()) - t0, "share_above_no_image": share, "binomial_p": p, "r_margin": r, "p_r_margin": pc,
                         "share_axis1": float(w[-1] / w.sum())})
    return pd.DataFrame(rows)


def c3_words(S: dict, units: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the agreement of the borough t vectors (C3 at the adjective token), and the top 15 words of Manhattan and Bronx at depth .5."""
    agree, top = [], []
    for adj in ADJECTIVES:
        t_by = {}
        for k in ("G", "Q9", "Q27"):
            st = S[adj][k]
            nta_borough = pd.Series(st["images"]["borough"].to_numpy()).groupby(st["codes"]).first().to_numpy()
            layers = st["layers"]
            t_by[k] = {}
            for li, lay in enumerate(layers):
                x = st["nta"][:, ADJ_POS, li]
                t_by[k][lay] = {b: two_sample_t(x, nta_borough == b) for b in BOROUGHS}
            t_by[k]["borough"] = nta_borough
        for f, lay in units:
            for b in ("Manhattan", "Bronx"):
                for a, c in PAIRS:
                    ta, tc = t_by[a][lay[a]][b], t_by[c][lay[c]][b]
                    ra = np.corrcoef(ta, tc)[0, 1]
                    top_a = set(np.argsort(-ta)[:50])
                    top_c = set(np.argsort(-tc)[:50])
                    agree.append({"adjective": adj, "f": f, "borough": b, "pair": f"{a}-{c}", "r_t_vectors": ra,
                                  "jaccard_top50": len(top_a & top_c) / len(top_a | top_c)})
        for k in ("G", "Q9", "Q27"):
            st = S[adj][k]
            lay = min(st["layers"], key=lambda q: abs((q + 1) / (N_LAYERS[LABEL[k]] + 1) - 0.5))
            words = np.array(json.loads((WORK / "shared_words.json").read_text()))
            for b in ("Manhattan", "Bronx"):
                tt = t_by[k][lay][b]
                for w, v in zip(words[np.argsort(-tt)[:15]], np.sort(tt)[::-1][:15]):
                    top.append({"adjective": adj, "model": LABEL[k], "layer": lay, "borough": b, "word": w, "t": v})
    return pd.DataFrame(agree), pd.DataFrame(top)


def word_counts(S: dict) -> pd.DataFrame:
    """Give the embedded, raised and lowered word counts at `place`, ` cool` and `?` for every layer (the `rank_cells.csv` rule, all layers)."""
    TOP, FAR = 100, 1000
    rows = []
    for k in ("G", "Q9", "Q27"):
        st = S["cool"][k]
        full = np.asarray(st["full"])
        text = np.asarray(st["text_full"])
        for pi, tok in ((0, "place"), (1, "cool"), (2, "question_mark")):
            for li, lay in enumerate(st["layers"]):
                order_i = (-full[pi, li]).argsort().argsort() + 1
                order_t = (-text[pi, li]).argsort().argsort() + 1
                emb = int(((order_i <= TOP) & (order_t <= TOP)).sum())
                rai = int(((order_i <= TOP) & (order_t >= FAR)).sum())
                low = int(((order_t <= TOP) & (order_i >= FAR)).sum())
                rows.append({"model": LABEL[k], "token": tok, "layer": lay, "depth": round((lay + 1) / (N_LAYERS[LABEL[k]] + 1), 3),
                             "embedded": emb, "raised": rai, "lowered": low})
    return pd.DataFrame(rows)


def phase_q27() -> pd.DataFrame:
    """Give the layer bands of the sense, standard and verdict phases for the 3 models, with the rule of `phase_layers.py`. Scene is not computed for Qwen3.6-27B."""
    def first_run(layers, ok):
        run = []
        for layer, passed in zip(layers, ok):
            if passed:
                run.append(layer)
            elif run:
                break
        return run
    rows = []
    sense = pd.read_csv(COMPARE / "sense.csv")
    margin = pd.concat([pd.read_csv(FIG / "logit/logit_margin.csv"), pd.read_csv(COMPARE / "logit/logit_margin.csv")], ignore_index=True)
    for model in MODELS:
        s = sense[(sense["model"] == model) & (sense["token"] == "cool")].sort_values("layer")
        run = first_run(s["layer"].tolist(), ((s["share1"] >= 0.40) & (s["r_axis1"].abs() >= 0.95)).tolist())
        gain = pd.read_csv(WORK_FREE / model / "decoder_grid1.csv").query("token == '?'").sort_values("layer")
        onset = int(gain[gain["gain"] >= 0.5 * gain["gain"].max()]["layer"].min())
        mg = margin[(margin["model"] == model) & (margin["lens"] == "lens")]
        last = mg[mg["index"] == mg["index"].max()].sort_values("layer")
        verdict = first_run(last["layer"].tolist(), (last["r"] >= 0.95).tolist())
        rows.append({"model": model, "phase": "sense", "first": run[0] if run else None, "last": run[-1] if run else None})
        rows.append({"model": model, "phase": "standard (gain at ? at least half of peak, from onset)", "first": onset, "last": verdict[0] if verdict else None})
        rows.append({"model": model, "phase": "verdict (last token r at least .95)", "first": verdict[0] if verdict else None, "last": verdict[-1] if verdict else None})
    return pd.DataFrame(rows)


def replication(counts: pd.DataFrame, sense_layers: pd.DataFrame) -> pd.DataFrame:
    """Give the replication table of question 1: each F10 finding, its value in each model, and the label of the rule."""
    rows = []
    sets = sets_question_window()
    for pair in ("aesthetic", "status", "valence"):
        g = sets[sets["measure"].str.contains(pair)]
        vals = dict(zip(g["model"], g["value"]))
        ok = {k: abs(vals[LABEL[k]]) >= 0.55 for k in ("G", "Q9", "Q27")}
        rows.append({"finding": f"F10.8 pair {pair}, largest |r| before the `?` token", "G": vals[LABEL["G"]], "Q9": vals[LABEL["Q9"]], "Q27": vals[LABEL["Q27"]],
                     "rule": "|r| at least .55 with p Holm below .001", "pass_G": ok["G"], "pass_Q9": ok["Q9"], "pass_Q27": ok["Q27"]})
    gain = gain_table()
    for token, label in (("place", "place"), ("cool", "cool")):
        g = gain[gain["measure"].str.endswith(label)]
        vals = dict(zip(g["model"], g["value"]))
        if token == "cool":
            ratio = {k: vals[LABEL[k]] / gain[(gain["model"] == LABEL[k]) & (gain["measure"].str.endswith("place"))]["value"].iloc[0] for k in ("G", "Q9", "Q27")}
            ok = {k: vals[LABEL[k]] >= 0.10 and ratio[k] >= 1.5 for k in ("G", "Q9", "Q27")}
            rows.append({"finding": "F10.9 gain over mean patch at cool (step at the adjective)", "G": vals[LABEL["G"]], "Q9": vals[LABEL["Q9"]], "Q27": vals[LABEL["Q27"]],
                         "rule": "gain at least .10 and at least 1.5 times the gain at place", "pass_G": ok["G"], "pass_Q9": ok["Q9"], "pass_Q27": ok["Q27"]})
        else:
            ok = {k: 0.03 <= vals[LABEL[k]] <= 0.15 for k in ("G", "Q9", "Q27")}
            rows.append({"finding": "F10.9 gain over mean patch at place", "G": vals[LABEL["G"]], "Q9": vals[LABEL["Q9"]], "Q27": vals[LABEL["Q27"]],
                         "rule": "gain between .03 and .15", "pass_G": ok["G"], "pass_Q9": ok["Q9"], "pass_Q27": ok["Q27"]})
    sl = sense_layers
    vals = dict(zip(sl["model"], sl["value"]))
    rr = dict(zip(sl["model"], sl["r_margin"]))
    rows.append({"finding": "F10.15 axis 1 share at cool L24", "G": vals[LABEL["G"]], "Q9": vals[LABEL["Q9"]], "Q27": vals[LABEL["Q27"]],
                 "rule": "share at least .40", "pass_G": vals[LABEL["G"]] >= .4, "pass_Q9": vals[LABEL["Q9"]] >= .4, "pass_Q27": vals[LABEL["Q27"]] >= .4})
    rows.append({"finding": "F10.17 sense score does not follow the verdict (r with margin, cool L24)", "G": rr[LABEL["G"]], "Q9": rr[LABEL["Q9"]],
                 "Q27": rr[LABEL["Q27"]], "rule": "|r| below .2", "pass_G": abs(rr[LABEL["G"]]) < .2, "pass_Q9": abs(rr[LABEL["Q9"]]) < .2,
                 "pass_Q27": abs(rr[LABEL["Q27"]]) < .2})
    lm = margin_r_cells()
    for token, label, rule in (("cool", "cool", 0.5), ("?", "question_mark", 0.5)):
        g = lm[lm["measure"].str.endswith(f"at {token}")]
        vals = dict(zip(g["model"], g["value"]))
        rows.append({"finding": f"F10.7 largest cell r of the margin with the answer, at {token}", "G": vals[LABEL["G"]], "Q9": vals[LABEL["Q9"]], "Q27": vals[LABEL["Q27"]],
                     "rule": "r at least .50", "pass_G": vals[LABEL["G"]] >= rule, "pass_Q9": vals[LABEL["Q9"]] >= rule, "pass_Q27": vals[LABEL["Q27"]] >= rule})
    g = lm[lm["measure"].str.startswith("held-out")]
    vals = dict(zip(g["model"], g["value"]))
    nvals = dict(zip(g["model"], g["n_cells"]))
    rows.append({"finding": "F10.7 held-out word lists pass Holm (cells)", "G": f"{int(vals[LABEL['G']])} of {int(nvals[LABEL['G']])}",
                 "Q9": f"{int(vals[LABEL['Q9']])} of {int(nvals[LABEL['Q9']])}", "Q27": f"{int(vals[LABEL['Q27']])} of {int(nvals[LABEL['Q27']])}",
                 "rule": "share of cells passing at least .90", "pass_G": vals[LABEL["G"]] / nvals[LABEL["G"]] >= .9,
                 "pass_Q9": vals[LABEL["Q9"]] / nvals[LABEL["Q9"]] >= .9, "pass_Q27": vals[LABEL["Q27"]] / nvals[LABEL["Q27"]] >= .9})
    out = pd.DataFrame(rows)
    out["label"] = out.apply(lambda r: _label(r), axis=1)
    return out


def _label(r) -> str:
    """Give the label of a row: replicates when the 2 other models pass, in family only when Q9 and Q27 pass, else fails."""
    g, q9, q27 = bool(r["pass_G"]), bool(r["pass_Q9"]), bool(r["pass_Q27"])
    if q27 and g and q9:
        return "replicates"
    if q27 and q9 and not g:
        return "replicates in family only"
    if q27 and g and not q9:
        return "replicates against Gemma only (fails against Qwen3.5-9B)"
    if q27:
        return "replicates in Qwen3.6-27B only"
    return "fails"


def analyze_reads(o: pd.DataFrame) -> None:
    """Write the tables that need the saved stores."""
    S = load_all()
    units = depth_units(S)
    words, tests = word_and_rsa(S, units)
    words.to_csv(OUT / "q2_word_rsa_by_depth.csv", index=False)
    pd.DataFrame(tests).assign(p_holm=lambda d: holm(d["p"].to_numpy())).to_csv(OUT / "q2_word_rsa_tests.csv", index=False)
    sense_by_layer(S, o).to_csv(OUT / "q4_sense_by_layer.csv", index=False)
    agree, top = c3_words(S, units)
    agree.to_csv(OUT / "q5_c3_agreement.csv", index=False)
    top.to_csv(OUT / "q5_c3_top_words.csv", index=False)
    word_counts(S).to_csv(OUT / "q1_word_counts_by_layer.csv", index=False)
    phase_q27().to_csv(OUT / "q1_phase_bands.csv", index=False)
    print("[f10.24] read tables written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True, choices=MODELS)
    b.add_argument("--adj", required=True, choices=ADJECTIVES)
    b.add_argument("--shards", type=int, default=None)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.model, args.adj, args.shards)
    else:
        analyze()
