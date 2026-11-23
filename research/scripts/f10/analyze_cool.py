#!/usr/bin/env python
"""F10.11: what the models say when they decide if a place is cool. CPU only.

The input is the 2 runs of `build_cool_input.py` and the spans of 3 extractions. Count
methods are valid here: unlike a logit, a text has a sample size.

| Output in `research/figures/f10/reason/` | Question | Method |
|---------------------------------|----------|--------|
| `reason_verdict.csv` | Do the traces agree with the read margin | Yes share of each image against the margin, NTA-clustered r, df 196 |
| `reason_labels.csv` | Which evidence, inference, values and notes occur, and where | Share of traces with a label. One-way ANOVA over the boroughs on NTA means, df 4 and 192. r with the yes share and with the read margin. Holm over the labels of 1 model |
| `reason_words.csv` | Which words separate yes from no, and a borough from the rest | Monroe log-odds z with an informative Dirichlet prior, 1 count for each trace, and the sign in 2 halves of the NTAs |
| `reason_observations.csv` | Which observations support which value | The `evidence` attribute of criterion spans, by value and stance |
| `reason_fit.csv` | How much of the margin the argument explains | OLS on image-level label shares, 5 folds by NTA, out-of-fold R2 |
| `reason_sets.csv` | Does a read set predict the matching label of the argument | Pre-set pairs. r over images at each cell, NTA-clustered, Holm over the cells of the pair |

The 3 repeats of an image are not independent. Thus every test uses image or NTA values.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_cool.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/scripts/f9"), str(ROOT / "research/notebooks/cvpr")]
from research.scripts.f10.analyze_logit import LOGIT_ROOT, MODELS, SEED, cluster_r, holm  # noqa: E402
from research.scripts.f9.analyze_subjective import text_verdict  # noqa: E402
from research.scripts.f10.build_cool_schemas import example_ids  # noqa: E402
from research.scripts.f10.hop_borough import BOROUGHS, family_r2, split_within  # noqa: E402

__version__ = "1.0.0"

REASON = ROOT / "data/f10/hop_reason"
OUT = ROOT / "research/figures/f10/reason"
WORD = re.compile(r"[a-z][a-z'-]+")
TOP = 40
MIN_COUNT = 5
# The read set and the label of the argument that name the same thing. Fixed before the results.
PAIRS = [("p_order", "evidence:cleanliness"), ("p_decay", "evidence:upkeep:bad"), ("p_nature", "evidence:greenery"), ("p_color", "evidence:color"),
         ("o_commerce", "evidence:commerce"), ("o_built", "evidence:architecture"), ("o_street", "evidence:traffic"), ("o_people", "evidence:people"),
         ("p_busy", "evidence:activity"), ("v_safety", "inference:safety"), ("status_high", "inference:wealth"), ("v_community", "inference:neighborhood_character"),
         ("p_old", "criterion:heritage"), ("p_new", "criterion:modernity"), ("p_grit", "criterion:edge"), ("v_authentic", "criterion:character"),
         ("v_utility", "criterion:utility"), ("aesthetic_pos", "criterion:beauty:for"), ("aesthetic_neg", "criterion:beauty:against"),
         ("s_uncertain", "note:subjective")]


def attrs(text) -> dict:
    return json.loads(text) if isinstance(text, str) and text else {}


def load_spans() -> pd.DataFrame:
    d = pd.read_parquet(REASON / "spans.parquet")
    d = d[d["is_quotable"].fillna(False) & ~d["is_duplicate"].fillna(False) & ~d["doc_id"].isin(set(example_ids()))].copy()
    parts = d["doc_id"].str.split("|", expand=True)
    d["run"], d["image"], d["field"] = parts[1], parts[2], parts[3]
    d["attr"] = [attrs(a) for a in d["attributes_json"]]
    return d


def label_of(row) -> list[str]:
    """Give the labels of 1 span."""
    a, c = row.attr, row.extraction_class
    if c == "visual_evidence":
        return [f"evidence:{a.get('category')}", f"evidence:{a.get('category')}:{a.get('valence')}", f"valence:{a.get('valence')}"]
    if c == "inference":
        return [f"inference:{a.get('kind')}"]
    if c == "uncertainty":
        return [f"uncertainty:{a.get('reason')}"]
    if c == "criterion":
        return [f"criterion:{a.get('value')}", f"criterion:{a.get('value')}:{a.get('stance')}", f"holder:{a.get('holder')}", f"stance:{a.get('stance')}"]
    if c == "decision_note":
        return [f"note:{a.get('kind')}"]
    return []


def verdicts(t: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Give the verdict of each answer. The one-word run reads the answer, the open run reads the spans of the answer."""
    t = t.copy()
    t["image"] = t["image_idx"].astype(str) + "." + t["repeat"].astype(str)
    word = t["response"].fillna("").str.strip().str.lower().str.extract(r"^(yes|no)\b")[0]
    v = spans[(spans["extraction_class"] == "verdict") & (spans["run"] == "cool_open") & (spans["field"] == "response")].sort_values("char_start")
    got = v.groupby(["model", "image"]).apply(lambda g: text_verdict([a.get("label") for a in g["attr"]]), include_groups=False).rename("open_verdict").reset_index()
    t = t.merge(got, on=["model", "image"], how="left")
    t["verdict"] = np.where(t["case"] == "cool_word", word, t["open_verdict"])
    t["verdict"] = t["verdict"].fillna("none")
    return t


def image_labels(s: pd.DataFrame) -> pd.DataFrame:
    """Give [images, labels]: the share of traces of an image that hold the label."""
    rows = [(r.image_idx, r.doc_id, lab) for r in s.itertuples() for lab in label_of(r)]
    d = pd.DataFrame(rows, columns=["image_idx", "doc_id", "label"]).drop_duplicates()
    n = s.groupby("image_idx")["doc_id"].nunique()
    return d.pivot_table(index="image_idx", columns="label", values="doc_id", aggfunc="count", fill_value=0).div(n, axis=0).fillna(0.0)


def log_odds(a: pd.Series, b: pd.Series, n_a: int, n_b: int) -> pd.DataFrame:
    """Give the log-odds z of Monroe et al. 2008, with the pooled counts as the prior."""
    words = a.index.union(b.index)
    y1, y2 = a.reindex(words, fill_value=0).to_numpy(float), b.reindex(words, fill_value=0).to_numpy(float)
    prior = (y1 + y2) / (y1 + y2).sum() * 500.0
    a0 = prior.sum()
    d = np.log((y1 + prior) / (n_a + a0 - y1 - prior)) - np.log((y2 + prior) / (n_b + a0 - y2 - prior))
    return pd.DataFrame({"z": d / np.sqrt(1 / (y1 + prior) + 1 / (y2 + prior)), "n_a": y1, "n_b": y2}, index=words)


def word_counts(rows: pd.DataFrame) -> pd.Series:
    """Count each word 1 time for each document."""
    c = {}
    for _, g in rows.groupby("doc_id"):
        for w in set(WORD.findall(" ".join(g["text"].astype(str)).lower())):
            c[w] = c.get(w, 0) + 1
    return pd.Series(c, dtype=float)


def contrast(rows: pd.DataFrame, flag: pd.Series, half: pd.Series, source: str, model: str, name: str) -> pd.DataFrame:
    def z_of(p: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
        u, v = word_counts(p), word_counts(q)
        return log_odds(u, v, u.sum(), v.sum())

    a, b = rows[flag], rows[~flag]
    z = z_of(a, b)
    z = z[(z["n_a"] + z["n_b"]) >= MIN_COUNT]
    z["z_a"] = z_of(a[half.loc[a.index]], b[half.loc[b.index]])["z"].reindex(z.index)
    z["z_b"] = z_of(a[~half.loc[a.index]], b[~half.loc[b.index]])["z"].reindex(z.index)
    held = z[np.sign(z["z_a"]) == np.sign(z["z_b"])]
    top = pd.concat([held.nlargest(TOP, "z"), held.nsmallest(TOP, "z")])
    return top.reset_index(names="word").assign(model=model, source=source, contrast=name)


def analyze(model: str, trace: pd.DataFrame, spans: pd.DataFrame) -> dict[str, pd.DataFrame]:
    margin = pd.read_parquet(LOGIT_ROOT / model / "answers.parquet").set_index("image_idx")["margin"]
    t = trace[trace["model"] == model].copy()
    s = spans[spans["model"] == model].copy()
    images = t.groupby("image_idx").agg(nta=("nta2020", "first"), borough=("borough", "first"))
    images["margin"] = margin.reindex(images.index)
    ntas = images.groupby("nta")["borough"].first()
    half_of = dict(zip(ntas.index, split_within(ntas.to_numpy())))
    codes = pd.factorize(images["nta"])[0]
    cl = images["nta"].to_numpy()
    y = images["margin"].to_numpy()
    out = {}

    rows = []
    t["yes"] = (t["verdict"] == "yes").astype(float)
    t["no"] = (t["verdict"] == "no").astype(float)
    for case, g in t.groupby("case"):
        share = g.groupby("image_idx")["yes"].mean().reindex(images.index)
        r, tv, p = cluster_r(share.to_numpy(), y, cl)
        groups = [share[images["borough"] == k].groupby(images["nta"]).mean() for k in BOROUGHS]
        f, pf = stats.f_oneway(*groups)
        rows.append({"model": model, "case": case, "n": len(g), "yes": g["yes"].mean(), "no": g["no"].mean(), "other": 1 - g["yes"].mean() - g["no"].mean(), "r_margin": r, "t": tv, "p": p, "F_borough": f, "p_borough": pf}
                    | {f"yes_{k}": float(g.loc[g["borough"] == k, "yes"].mean()) for k in BOROUGHS})
    out["verdict"] = pd.DataFrame(rows)
    yes = t.groupby("image_idx")["yes"].mean().reindex(images.index).to_numpy()

    tables, fit = [], []
    use = {"reasoning": s[(s["field"] == "reasoning")], "response": s[(s["field"] == "response")]}
    splits = list(GroupKFold(5).split(y, groups=np.random.default_rng(SEED).permutation(codes.max() + 1)[codes]))
    for field, part in use.items():
        wide = image_labels(part).reindex(images.index).fillna(0.0)
        for lab in wide.columns:
            v = wide[lab].to_numpy()
            if v.std() == 0:
                continue
            nta = pd.Series(v).groupby(codes).mean().to_numpy()
            b = images.groupby("nta")["borough"].first().reindex(pd.unique(cl)).to_numpy()
            f, p = stats.f_oneway(*[nta[b == k] for k in BOROUGHS])
            r, tv, q = cluster_r(v, y, cl)
            r2, t2, q2 = cluster_r(v, yes, cl)
            tables.append({"model": model, "field": field, "label": lab, "share": v.mean(), "F": f, "p": p, "r_margin": r, "t_margin": tv, "p_margin": q, "r_yes": r2, "t_yes": t2, "p_yes": q2}
                          | {k: nta[b == k].mean() for k in BOROUGHS})
        for fam, prefix in (("evidence", "evidence:"), ("inference", "inference:"), ("criteria", "criterion:"), ("notes", "note:"), ("all", "")):
            cols = [c for c in wide.columns if c.startswith(prefix) and c.count(":") == 1]
            if cols:
                fit.append({"model": model, "field": field, "family": fam, "columns": len(cols), "r2_margin": family_r2(stats.zscore(wide[cols].to_numpy(), axis=0, nan_policy="omit").clip(-9, 9).astype(float), y, splits),
                            "r2_yes": family_r2(np.nan_to_num(stats.zscore(wide[cols].to_numpy(), axis=0)), yes, splits)})
    labels = pd.DataFrame(tables)
    for k in ("p", "p_margin", "p_yes"):
        labels[f"{k}_holm"] = holm(labels[k].fillna(1.0).to_numpy())
    out["labels"], out["fit"] = labels, pd.DataFrame(fit)

    verdict = t.assign(image=t["image_idx"].astype(str) + "." + t["repeat"].astype(str)).set_index(["case", "image"])["verdict"]
    half_doc = lambda g: g["nta"].map(half_of).astype(bool)
    sources = {"descriptor": s[s["extraction_class"] == "descriptor"], "evidence": s[s["extraction_class"] == "visual_evidence"], "criterion": s[s["extraction_class"] == "criterion"], "inference": s[s["extraction_class"] == "inference"]}
    words = []
    for name, g in sources.items():
        g = g.assign(text=g["extraction_text"], nta=g["image_idx"].map(images["nta"]), borough=g["image_idx"].map(images["borough"]))
        g["verdict"] = [verdict.get((c, i), "none") for c, i in zip(g["run"], g["image"])]
        pair = g[g["verdict"].isin(["yes", "no"])]
        if len(pair) and pair["verdict"].nunique() == 2:
            words.append(contrast(pair, pair["verdict"] == "yes", half_doc(pair), name, model, "yes-no"))
        for k in BOROUGHS:
            words.append(contrast(g, g["borough"] == k, half_doc(g), name, model, k))
    out["words"] = pd.concat(words, ignore_index=True)

    c = s[s["extraction_class"] == "criterion"]
    c = c.assign(value=[a.get("value") for a in c["attr"]], stance=[a.get("stance") for a in c["attr"]], observation=[str(a.get("evidence") or "").lower().strip() for a in c["attr"]],
                 borough=c["image_idx"].map(images["borough"]))
    seen = c[c["observation"] != ""]
    obs = seen.groupby(["value", "stance", "observation"]).agg(n=("doc_id", "nunique"), images=("image_idx", "nunique")).reset_index().sort_values("n", ascending=False)
    out["observations"] = obs.groupby(["value", "stance"]).head(25).assign(model=model)
    return out


def set_links(model: str, spans: pd.DataFrame) -> pd.DataFrame:
    from research.scripts.f10.analyze_logit import load

    d = load(model)
    z = np.load(ROOT / "data/f10/hop_sets" / model / "scores.npz")
    names, score = z["names"].tolist(), z["score"]
    wide = image_labels(spans[spans["model"] == model]).reindex(d["images"]["image_idx"]).fillna(0.0)
    cl = d["images"]["nta2020"].to_numpy()
    tokens, head = d["store"].manifest["tail_tokens"], d["head"]
    rows = []
    for read, lab in PAIRS:
        if lab not in wide or read not in names:
            continue
        s = score[names.index(read)]
        v = wide[lab].to_numpy()
        block = []
        for a in range(len(tokens)):
            for b, layer in enumerate(head["layers"]):
                r, t, p = cluster_r(s[:, a, b], v, cl)
                block.append({"model": model, "set": read, "label": lab, "token": tokens[a], "index": a, "layer": layer, "r": r, "t": t, "p": p})
        block = pd.DataFrame(block)
        block["p_holm"] = holm(block["p"].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--no-sets", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spans = load_spans()
    trace = verdicts(pd.read_parquet(REASON / "traces.parquet"), spans)
    trace.drop(columns=["sampling", "chat_template_kwargs"], errors="ignore").to_parquet(REASON / "verdicts.parquet", index=False)
    got = [analyze(m, trace, spans) for m in args.models]
    for key in got[0]:
        pd.concat([g[key] for g in got], ignore_index=True).to_csv(OUT / f"reason_{key}.csv", index=False)
    if not args.no_sets:
        pd.concat([set_links(m, spans) for m in args.models], ignore_index=True).to_csv(OUT / "reason_sets.csv", index=False)
    print("[f10.11] finished", flush=True)


if __name__ == "__main__":
    main()
