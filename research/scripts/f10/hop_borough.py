#!/usr/bin/env python
"""F10.10: borough contrasts of the reads, with held-out NTAs. CPU only.

The paper reads the contrast C3 (borough mean minus the mean of the other images) by hand,
on the top words of 1 list. A raw list favours words with a large variance, and on Gemma the
softcap compresses it. This script keeps C3, adds a per-word t value, and asks if each
result holds on NTAs that did not make the list. Each NTA has 3 images, thus an NTA mean
is the unit and the design is balanced.

| Output in `research/figures/f10/borough/` | Question | Method |
|----------------------------------|----------|--------|
| `borough_cells.csv` | Does a borough list hold on other NTAs | List of 50 up and 50 down words from 1 half of the NTAs in each borough. Score the other half, two-sample t. Both directions, the larger p, Holm over the cells |
| `borough_words.csv` | Which words separate a borough at a cell | t over all NTAs, and the sign in both halves. A description |
| `borough_claims.csv` | Which words of the paper hold | Rank in the capped raw contrast, rank by t, t in each half. Holm over the rows |
| `borough_sets.csv` | Do the word sets differ by borough | One-way ANOVA on NTA means, df 4 and 192, Holm over set and cell. Each borough against the rest |
| `borough_family.csv` | How much of the margin the sets explain at a cell | OLS, 5 folds by NTA, out-of-fold R2, for properties, values and both |
| `borough_gate.csv` | Does the urban context open the aesthetic words | r between the urban-rural score at an early cell and the property and value sets at `cool`, with and without borough effects |
| `borough_cross.csv` | Do the 2 models split the boroughs the same way | r of the word t values on shared words, same borough against other boroughs |

A list test is a search result on held-out NTAs. It is not a causal result. A borough
effect can come from what is in the images, for example from more commerce in the view.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_borough.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from research.scripts.f10.analyze_logit import MODELS, SEED, cluster_r, holm, load, shard_reads  # noqa: E402

__version__ = "1.0.0"

DATA = ROOT / "data/f10/hop_borough"
SETS = ROOT / "data/f10/hop_sets"
OUT = ROOT / "research/figures/f10/borough"
BOROUGHS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
LIST = 50
SHOW = 15
# The words that the paper cites (section 4), with the token and the layers it names.
CLAIMS = [
    ("Bronx", " place", (20, 24, 28), "longitude geographic latitude"),
    ("Bronx", " place", (32, 36, 40), "geografia coordenadas mapas"),
    ("Bronx", " place", (40,), "miasta"),
    ("Bronx", " place", (44, 48, 52, 56), "identificar reconhecimento ninguna recognizes familiarity"),
    ("Staten Island", " place", None, "landfall coastal acreage homestead drought farmer"),
    ("Brooklyn", " place", None, "storefront pharmacies businesses"),
    ("Queens", " place", None, "homeowners vivienda homeschooling"),
    ("Manhattan", " place", None, "gastronomy attractions tourists busiest"),
    ("Bronx", " cool", (20,), "graffiti fucking shopping gallery homeless"),
    ("Bronx", " cool", (24,), "porn anarch illegal vandalism gangs tattoo"),
    ("Bronx", " cool", (28, 32, 36, 40), "borough bairro"),
    ("Brooklyn", " cool", (20,), "fucking boutique instagram bodega social nightclubs"),
    ("Brooklyn", " cool", (24,), "bodega slums boutiques prostit slum market traff marijuana prostitutes bistro"),
    ("Staten Island", " cool", (20, 24), "seawater scientist hydrological"),
    ("Queens", " cool", (20, 24), "equatorial humidity"),
    ("Manhattan", " cool", (8,), "swiss nord german european"),
    ("Manhattan", " cool", (12,), "boutiques jewellery instagram feminist"),
    ("Manhattan", " cool", (20,), "lingerie femme couture housewife salon"),
    ("Manhattan", " cool", (32,), "boulevard prestigious metropolitan"),
]


def balanced_means(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Give the group mean of each row group. `x` has 1 row for each image."""
    return np.stack([x[codes == g].mean(0) for g in range(codes.max() + 1)])


def split_within(borough: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Give True for half of the NTAs of each borough, by a fixed seed. `borough` has 1 row for each NTA."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(borough), bool)
    for b in sorted(set(borough)):
        ix = rng.permutation(np.flatnonzero(borough == b))
        out[ix[: len(ix) // 2]] = True
    return out


def two_sample_t(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Give the pooled two-sample t of each column: rows in `mask` against the others."""
    a, b = m[mask], m[~mask]
    n1, n2 = len(a), len(b)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    pooled = ((n1 - 1) * va + (n2 - 1) * vb) / (n1 + n2 - 2)
    return (a.mean(0) - b.mean(0)) / np.sqrt(pooled * (1 / n1 + 1 / n2)).clip(min=1e-9)


def held_out(m: np.ndarray, t: np.ndarray, member: np.ndarray) -> tuple[float, float]:
    """Score the up list minus the down list of `t` on held-out rows `m`, and give t and p."""
    k = np.argpartition(t, [LIST, -LIST])
    up, down = k[-LIST:], k[:LIST]
    s = m[:, up].mean(1) - m[:, down].mean(1)
    return tuple(float(v) for v in stats.ttest_ind(s[member], s[~member]))


def read_all(d: dict, files: list[Path]) -> np.ndarray:
    """Load the reads of the store words, each standardized over the words."""
    out = []
    for f in files:
        x = np.load(f)
        out.append((x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True))
    return np.concatenate(out)


def demean(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return x - pd.Series(x).groupby(codes).transform("mean").to_numpy()


def family_r2(x: np.ndarray, y: np.ndarray, splits: list) -> float:
    """Give the out-of-fold R2 of OLS with an intercept."""
    pred = np.zeros_like(y)
    for tr, te in splits:
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), x[tr]], y[tr], rcond=None)[0]
        pred[te] = np.c_[np.ones(len(te)), x[te]] @ b
    sst = sum(np.sum((y[te] - y[tr].mean()) ** 2) for tr, te in splits)
    return float(1 - np.sum((y - pred) ** 2) / sst)


def word_tables(d: dict, model: str, m: np.ndarray, capped: np.ndarray, member: dict, half: np.ndarray, words: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    head, tokens = d["head"], d["store"].manifest["tail_tokens"]
    n_tail, n_layer = len(head["positions"]), len(head["layers"])
    cells, lists, claims, full = [], [], [], {}
    for name in BOROUGHS:
        mask = member[name]
        ta, tb = two_sample_t(m[half], mask[half]), two_sample_t(m[~half], mask[~half])
        tf = two_sample_t(m, mask)
        raw = capped[mask].mean(0) - capped[~mask].mean(0)
        full[name] = tf.astype(np.float32)
        for a in range(n_tail):
            for b in range(n_layer):
                t1, p1 = held_out(m[~half, a, b], ta[a, b], mask[~half])
                t2, p2 = held_out(m[half, a, b], tb[a, b], mask[half])
                rho = stats.spearmanr(ta[a, b], tb[a, b]).statistic
                by_t, by_raw = np.argsort(-tf[a, b])[:SHOW], np.argsort(-raw[a, b])[:SHOW]
                agree = (ta[a, b] > 0) & (tb[a, b] > 0)
                both = np.argsort(-np.where(agree, tf[a, b], -np.inf))[:SHOW]
                cells.append({"model": model, "borough": name, "position": head["positions"][a], "token": tokens[a], "index": a, "layer": head["layers"][b],
                              "t_ab": t1, "t_ba": t2, "p": max(p1, p2) if min(t1, t2) > 0 else 1.0, "stability": rho,
                              "overlap_raw_t": len(set(by_t) & set(by_raw)) / SHOW})
                lists.append({"model": model, "borough": name, "token": tokens[a], "index": a, "layer": head["layers"][b], "by_t": ", ".join(words[by_t]),
                              "by_raw": ", ".join(words[by_raw]), "both_halves": ", ".join(words[both])})
    cells = pd.DataFrame(cells)
    cells["p_holm"] = holm(cells["p"].to_numpy())
    at = {w: i for i, w in enumerate(words)}
    for name, token, layers, text in CLAIMS:
        if token not in tokens:
            continue
        a = tokens.index(token)
        mask = member[name]
        tf, ta, tb = two_sample_t(m[:, a], mask), two_sample_t(m[half, a], mask[half]), two_sample_t(m[~half, a], mask[~half])
        raw = capped[mask, a].mean(0) - capped[~mask, a].mean(0)
        use = [head["layers"].index(v) for v in layers or () if v in head["layers"]] or list(range(n_layer))
        for w in text.split():
            if w not in at:
                claims.append({"model": model, "borough": name, "token": token, "word": w, "in_store": False})
                continue
            i = at[w]
            b = use[int(np.argmax(raw[use, i]))]
            claims.append({"model": model, "borough": name, "token": token, "word": w, "in_store": True, "layer": head["layers"][b],
                           "rank_raw": int((raw[b] > raw[b, i]).sum()) + 1, "rank_t": int((tf[b] > tf[b, i]).sum()) + 1, "t": float(tf[b, i]),
                           "p": float(2 * stats.t.sf(abs(tf[b, i]), m.shape[0] - 2)), "t_a": float(ta[b, i]), "t_b": float(tb[b, i])})
    claims = pd.DataFrame(claims)
    ok = claims["in_store"]
    claims.loc[ok, "p_holm"] = holm(claims.loc[ok, "p"].to_numpy())
    return cells, pd.DataFrame(lists), claims, full


def set_tables(d: dict, model: str, codes: np.ndarray, borough: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test the set scores across boroughs, the fit of the set families, and the urban gate."""
    z = np.load(SETS / model / "scores.npz")
    names, score = z["names"].tolist(), z["score"]
    head, tokens = d["head"], d["store"].manifest["tail_tokens"]
    im = d["images"]
    y, cl = im["margin"].to_numpy(), im["nta2020"].to_numpy()
    boro = im["borough"].to_numpy()
    bcode = pd.factorize(boro)[0]
    rows = []
    for s, name in enumerate(names):
        nta = balanced_means(score[s], codes)
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                v = nta[:, a, b]
                groups = [v[borough == k] for k in BOROUGHS]
                f, p = stats.f_oneway(*groups)
                ss_between = sum(len(g) * (g.mean() - v.mean()) ** 2 for g in groups)
                row = {"model": model, "set": name, "position": pos, "token": tokens[a], "index": a, "layer": layer, "F": f, "p": p,
                       "eta2": ss_between / np.sum((v - v.mean()) ** 2), "r_within": cluster_r(demean(score[s, :, a, b], bcode), demean(y, bcode), cl)[0]}
                sd = np.sqrt(sum((len(g) - 1) * g.var(ddof=1) for g in groups) / (len(v) - 5))
                row |= {k: (v[borough == k].mean() - v[borough != k].mean()) / sd for k in BOROUGHS}
                rows.append(row)
    sets = pd.DataFrame(rows)
    sets["p_holm"] = holm(sets["p"].to_numpy())
    pick = lambda p: [i for i, n in enumerate(names) if n.startswith(p)]
    families = {"properties": pick("p_") + [names.index(n) for n in ("aesthetic_pos", "aesthetic_neg")], "values": pick("v_") + [names.index(n) for n in ("status_high", "status_low")]}
    families["both"] = families["properties"] + families["values"]
    splits = list(GroupKFold(5).split(y, groups=np.random.default_rng(SEED).permutation(codes.max() + 1)[codes]))
    fam = []
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            row = {"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer}
            for k, ix in families.items():
                row[k] = family_r2(stats.zscore(score[ix, :, a, b], axis=1).T, y, splits)
                row[f"{k}_within"] = family_r2(stats.zscore(score[ix, :, a, b], axis=1).T, demean(y, bcode), splits)
            fam.append(row)
    fam = pd.DataFrame(fam)
    setting = score[names.index("s_urban")] - score[names.index("s_rural")]
    gate = []
    place, cool = tokens.index(" place"), tokens.index(" cool")
    for early in (12, 16, 20):
        if early not in head["layers"]:
            continue
        x = setting[:, place, head["layers"].index(early)]
        for late in (20, 24):
            if late not in head["layers"]:
                continue
            b = head["layers"].index(late)
            for s in families["both"]:
                v = score[s, :, cool, b]
                r, t, p = cluster_r(x, v, cl)
                r2, t2, p2 = cluster_r(demean(x, bcode), demean(v, bcode), cl)
                gate.append({"model": model, "early_layer": early, "late_layer": late, "set": names[s], "r": r, "t": t, "p": p, "r_within": r2, "t_within": t2, "p_within": p2})
    gate = pd.DataFrame(gate)
    gate["p_holm"], gate["p_within_holm"] = holm(gate["p"].to_numpy()), holm(gate["p_within"].to_numpy())
    return sets, fam, gate


def cross_table(t_a: dict, t_b: dict, words_a: np.ndarray, words_b: np.ndarray, tokens_a: list[str], tokens_b: list[str], layers_a: list[int], layers_b: list[int]) -> pd.DataFrame:
    shared = sorted(set(words_a) & set(words_b))
    ia, ib = pd.Series(np.arange(len(words_a)), index=words_a)[shared].to_numpy(), pd.Series(np.arange(len(words_b)), index=words_b)[shared].to_numpy()
    rows = []
    for token in (" place", " cool", "?", " Answer", " yes", " no"):
        a, b = tokens_a.index(token), tokens_b.index(token)
        for la, u in enumerate(layers_a):
            for lb, v in enumerate(layers_b):
                pa = np.stack([t_a[k][a, la, ia] for k in BOROUGHS])
                pb = np.stack([t_b[k][b, lb, ib] for k in BOROUGHS])
                r = np.corrcoef(pa, pb)[:5, 5:]
                rows.append({"token": token, "gemma_layer": u, "qwen_layer": v, "same": float(np.mean(np.diag(r))), "other": float(np.mean(r[~np.eye(5, dtype=bool)])), "words": len(shared)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tables, kept = {k: [] for k in ("cells", "words", "claims", "sets", "family", "gate")}, {}
    for model in args.models:
        d = load(model)
        im = d["images"]
        codes = pd.factorize(im["nta2020"])[0]
        borough = im.groupby(codes)["borough"].first().to_numpy()
        words = d["store"].vocab["token"].str.strip().to_numpy()
        m = balanced_means(read_all(d, shard_reads(d)), codes)
        print(f"[f10.10] {model}: NTA means {m.shape}", flush=True)
        stored = sorted((d["store"].path / "reads").glob("reads-*.npy"))
        capped = balanced_means(read_all(d, stored), codes) if d["head"]["softcap"] else m
        member = {k: borough == k for k in BOROUGHS}
        half = split_within(borough)
        cells, lists, claims, full = word_tables(d, model, m, capped, member, half, words)
        sets, fam, gate = set_tables(d, model, codes, borough)
        for key, value in zip(("cells", "words", "claims", "sets", "family", "gate"), (cells, lists, claims, sets, fam, gate)):
            tables[key].append(value)
            pd.concat(tables[key], ignore_index=True).to_csv(OUT / f"borough_{key}.csv", index=False)
        kept[model] = (full, words, d["store"].manifest["tail_tokens"], d["head"]["layers"])
        print(f"[f10.10] {model}: {(cells['p_holm'] < 0.05).sum()} of {len(cells)} borough cells hold on held-out NTAs", flush=True)
        del m, capped
    if len(args.models) == 2:
        a, b = (kept[k] for k in MODELS)
        cross_table(a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]).to_csv(OUT / "borough_cross.csv", index=False)
    print("[f10.10] finished", flush=True)


if __name__ == "__main__":
    main()
