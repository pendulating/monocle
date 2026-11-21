#!/usr/bin/env python
"""F10.2: the tests of the read of the word list (D155).

The tests are set in vlm-narratives-docs/facct-runbook.md, F10.2, before the run.
The score of a word on 1 image is its pooled lens logit minus the mean over the
band of control words, at the primary layer of the model. A word reads when its Pearson r
has the expected sign, passes Bonferroni over the `valence` and `word` tests, and
exceeds the 95th percentile of the band control words on the same outcome.

| File in `figures/f10/reads/` | Content |
|------------------------------|---------|
| `reads_validity.csv`, `.png`, `.json` | the primary tests and the control thresholds |
| `reads_block.csv` | the same tests on the `block` score: a description |
| `reads_layers.csv` | the same tests at all layers: a description |
| `reads_ses.csv`, `.png`, `.json` | Kendall tau of the area score, with the 5 ACS features, all words |
| `reads_senses.csv` | the correlation of each word with the words of its second meaning |
| `reads_homeless.csv` | the 12 images with the highest `homeless` score, for each model and case |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/analyze_reads.py [--reads data/f10/reads]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts/f10"), str(REPO / "scripts/f8"), str(REPO / "notebooks/cvpr"), str(REPO / "notebooks/facct")]

from analyze_cue import MODEL_COLORS, MODEL_MARKERS, S  # noqa: E402
from word_read import LAYERS, SENSES, TARGETS  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b")
CASES = ("restaurant", "street")
PRIMARY = {"gemma-4-31b": 45, "qwen3.5-9b": 24}
SIGN = {"good": 1, "nice": 1, "yummy": 1, "stylish": 1, "cool": 1, "bad": -1, "ugly": -1,
        "rich": 1, "wealthy": 1, "poor": -1, "homeless": -1}
EVALUATIVE = ("good", "nice", "yummy", "stylish", "cool", "bad", "ugly")
INCOME_SIGN = {"rich": 1, "wealthy": 1, "poor": -1, "homeless": -1}
SENSE_PAIRS = {"cool": ("cold", "warm"), "rich": ("flavor", "delicious"), "poor": ("quality",)}
BAND, MIN_BAND, ALPHA, MIN_SIDE = 1.0, 20, 0.05, 20
OUT = Path("figures/f10/reads")


def load(root: Path, model: str) -> tuple[pd.DataFrame, list[str]]:
    d = pd.concat(pd.read_parquet(p) for p in sorted((root / model).glob("part-*.parquet")))
    words = [c[5:] for c in d.columns if c.startswith("mean_")]
    return d, [w for w in words if w not in TARGETS and w not in SENSES]


def band_of(means: pd.Series, word: str, pool: list[str]) -> list[str]:
    near = [w for w in pool if abs(means[w] - means[word]) <= BAND]
    return near if len(near) >= MIN_BAND else sorted(pool, key=lambda w: abs(means[w] - means[word]))[:MIN_BAND]


def raw(part: pd.DataFrame, form: str) -> pd.DataFrame:
    """Give 1 column for each word, with no control removed."""
    x = part[[c for c in part.columns if c.startswith(f"{form}_")]]
    return x.rename(columns=lambda c: c[len(form) + 1:])


def scored(x: pd.DataFrame, word: str, pool: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Give the score of a word and of each control word of its band, all minus the mean of the band."""
    band = band_of(x.mean(), word, pool)
    base = x[band].mean(axis=1)
    return x[word] - base, x[band].sub(base, axis=0)


@lru_cache(maxsize=None)
def answers(model: str, case: str) -> pd.Series:
    t = pd.read_parquet(REPO / f"data/f4_traces/subj/f9_{case}_traces.parquet", columns=["image_idx", "response", "model"],
                        filters=[("model", "==", model)])
    return t.set_index("image_idx")["response"]


@lru_cache(maxsize=None)
def valence(model: str, case: str) -> pd.Series:
    t = pd.read_parquet(REPO / "data/f4_traces/subj/inference_valence/texts.parquet",
                        columns=["model", "case", "field", "image_idx", "outcome"])
    t = t[(t["model"] == model) & (t["case"] == case) & (t["field"] == "response")]
    return t.dropna(subset=["outcome"]).set_index("image_idx")["outcome"]


def r_test(x: pd.Series, y: pd.Series) -> dict:
    ok = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    r = stats.pearsonr(ok["x"], ok["y"])
    ci = r.confidence_interval(0.95)
    return {"n": len(ok), "r": r.statistic, "p": r.pvalue, "ci_lo": ci.low, "ci_hi": ci.high}


def validity(root: Path, form: str = "mean", models: tuple = MODELS, layer: int | None = None) -> pd.DataFrame:
    rows = []
    for m in models:
        d, pool = load(root, m)
        l = layer or PRIMARY[m]
        for case in CASES:
            part = d[(d["case"] == case) & (d["layer"] == l)].set_index("image_idx")
            x = raw(part, form)
            texts = answers(m, case).reindex(part.index).fillna("")
            val = valence(m, case)
            for w in TARGETS:
                score, ctrl = scored(x, w, pool)
                outcomes = {"word": texts.str.contains(rf"\b{re.escape(w)}\b", case=False).astype(float)}
                if w in EVALUATIVE:
                    outcomes["valence"] = val
                for name, y in outcomes.items():
                    base = {"model": m, "case": case, "layer": l, "form": form, "word": w, "test": name,
                            "expect": 1 if name == "word" else SIGN[w]}
                    side = int(y.sum()) if name == "word" else np.nan
                    if name == "word" and min(y.sum(), (1 - y).sum()) < MIN_SIDE:
                        rows.append({**base, "n": len(y), "side": side})
                        continue
                    null = [abs(r_test(ctrl[c], y)["r"]) for c in ctrl]
                    rows.append({**base, **r_test(score, y), "side": side, "null_q95": float(np.percentile(null, 95)), "band_n": len(null)})
    out = pd.DataFrame(rows)
    tested = out["p"].notna()
    out["p_bonf"] = (out["p"] * tested.sum()).clip(upper=1.0)
    out["reads"] = tested & (out["p_bonf"] < ALPHA) & (np.sign(out["r"]) == out["expect"]) & (out["r"].abs() > out["null_q95"])
    return out


def ses(root: Path, passing: set) -> pd.DataFrame:
    import _pairwise_ses as P
    import _trace_ses as T

    keys = T.image_keys()
    rows = []
    for m in MODELS:
        d, pool = load(root, m)
        for case in CASES:
            part = d[(d["case"] == case) & (d["layer"] == PRIMARY[m])].set_index("image_idx")
            x = raw(part, "mean")
            k = keys[keys["case"] == case].set_index("image_idx")
            for layer, (short, _, _, _) in T.LAYERS.items():
                feats = P.features(layer)
                areas = k[layer].reindex(part.index)
                size = areas.value_counts()
                keep = areas[areas.isin(size[size >= T.MIN_TEXTS].index)]
                for w in TARGETS:
                    score, ctrl = scored(x, w, pool)
                    g = ctrl.assign(_word=score).reindex(keep.index).groupby(keep.astype(str)).mean()
                    g = g.rename_axis("key").reset_index().merge(feats, on="key", how="inner")
                    if len(g) < P.MIN_AREAS:
                        continue
                    for f in P.FEATURES:
                        tau = stats.kendalltau(g["_word"], g[f], nan_policy="omit")
                        null = [abs(stats.kendalltau(g[c], g[f], nan_policy="omit").statistic) for c in ctrl]
                        rows.append({"model": m, "case": case, "layer": short, "feature": f, "word": w, "areas": len(g),
                                     "tau": tau.statistic, "p": tau.pvalue, "null_q95": float(np.percentile(null, 95)),
                                     "p_pool": float(np.mean(np.array(null) >= abs(tau.statistic))),
                                     "gated": (m, case, w) in passing})
    out = pd.DataFrame(rows)
    n = int(out["gated"].sum())
    out["p_bonf"] = (out["p"] * max(n, 1)).clip(upper=1.0)
    out["reads"] = out["gated"] & (out["p_bonf"] < ALPHA) & (out["tau"].abs() > out["null_q95"])
    out["income_sign"] = [INCOME_SIGN.get(w, 0) * np.sign(t) > 0 if "income" in f else False
                          for w, t, f in zip(out["word"], out["tau"], out["feature"])]
    income = out["word"].isin(INCOME_SIGN) & out["feature"].str.contains("income")
    out["p_bonf_income"] = (out["p"] * int(income.sum())).clip(upper=1.0).where(income)
    out.attrs["tests"] = n
    return out


def senses(root: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d, pool = load(root, m)
        for case in CASES:
            x = raw(d[(d["case"] == case) & (d["layer"] == PRIMARY[m])], "mean")
            for w, others in SENSE_PAIRS.items():
                a = scored(x, w, pool)[0]
                rows.extend({"model": m, "case": case, "word": w, "other": o, "r": a.corr(scored(x, o, pool)[0])} for o in others)
    return pd.DataFrame(rows)


def homeless(root: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d, pool = load(root, m)
        for case in CASES:
            part = d[(d["case"] == case) & (d["layer"] == PRIMARY[m])]
            top = part.assign(score=scored(raw(part, "block"), "homeless", pool)[0]).nlargest(12, "score")
            rows.append(top[["case", "image_idx", "sample_id", "score"]].assign(model=m))
    return pd.concat(rows, ignore_index=True)


def legend(fig, colors: dict) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker=MODEL_MARKERS[m], mfc=colors[m], mec=S.EDGE, mew=S.EDGE_LW, ms=5, label=m) for m in MODELS]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0))


def plot_validity(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    colors = MODEL_COLORS
    tests = [("valence", "net valence of the answer"), ("word", "the word in the answer")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for i, (test, label) in enumerate(tests):
        for j, case in enumerate(CASES):
            ax = axes[i, j]
            g = df[(df["test"] == test) & (df["case"] == case)]
            for k, m in enumerate(MODELS):
                for y, w in enumerate(TARGETS):
                    r = g[(g["model"] == m) & (g["word"] == w)]
                    if r.empty or pd.isna(r["r"].iloc[0]):
                        continue
                    r = r.iloc[0]
                    yy = y + (k - 0.5) * 0.36
                    ax.plot([-r["null_q95"], r["null_q95"]], [yy, yy], color="0.8", lw=3, solid_capstyle="butt", zorder=1)
                    ax.scatter(r["r"], yy, s=20, marker=MODEL_MARKERS[m], color=colors[m] if r["reads"] else "white",
                               edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3)
            ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
            ax.set_title(f"{case}: {label}", fontsize=8)
    axes[0, 0].set_yticks(range(len(TARGETS)), TARGETS)
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("Pearson r of the read")
    legend(fig, colors)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": "Do the lens reads track the answers?",
            "plot-caption": "Pearson r of the lens read of a word on 1,000 images for each case, against the answer of the same "
                            "model to the F9 prompt. Filled: the word reads (expected sign, Bonferroni, above the control "
                            "threshold). The grey bar marks the 95th percentile of the absolute r of the band control words.",
            "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": ["F10.2", "read"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def plot_ses(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    import _pairwise_ses as P

    S.apply_house_style()
    colors = MODEL_COLORS
    g = df[df["layer"] == "CD"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    f = "median_household_income"
    for ax, case in zip(axes, CASES):
        for k, m in enumerate(MODELS):
            for y, w in enumerate(TARGETS):
                r = g[(g["case"] == case) & (g["model"] == m) & (g["word"] == w) & (g["feature"] == f)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = y + (k - 0.5) * 0.36
                ax.plot([-r["null_q95"], r["null_q95"]], [yy, yy], color="0.8", lw=3, solid_capstyle="butt", zorder=1)
                ax.scatter(r["tau"], yy, s=20, marker=MODEL_MARKERS[m], color=colors[m] if abs(r["tau"]) > r["null_q95"] else "white",
                           edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3)
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(case, fontsize=8)
        ax.set_xlabel(r"Kendall $\tau$ with income, community district")
    axes[0].set_yticks(range(len(TARGETS)), TARGETS)
    axes[0].invert_yaxis()
    legend(fig, colors)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": "The lens reads and area income",
            "plot-caption": "Kendall tau of the mean read over the images of a community district with its median household "
                            "income: a description. Filled: the absolute tau exceeds the grey bar, the 95th percentile of the "
                            "absolute tau of the band control words. No word passed the gated area test.",
            "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": ["F10.2", "read", "acs"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default="data/f10/reads")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.reads), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    val = validity(root)
    val.to_csv(out / "reads_validity.csv", index=False)
    validity(root, "block").to_csv(out / "reads_block.csv", index=False)
    layers = pd.concat([validity(root, "mean", (m,), l) for m in MODELS for l in LAYERS[m]], ignore_index=True)
    layers.to_csv(out / "reads_layers.csv", index=False)
    print(val["p"].notna().sum(), "tests;", int(val["reads"].sum()), "read")
    print(val[["model", "case", "word", "test", "n", "side", "r", "null_q95", "p_bonf", "reads"]].round(3).to_string())
    passing = set(map(tuple, val.loc[val["reads"], ["model", "case", "word"]].drop_duplicates().to_numpy()))
    s = ses(root, passing)
    s.to_csv(out / "reads_ses.csv", index=False)
    print(s.attrs["tests"], "gated area tests;", int(s["reads"].sum()), "pass")
    print(s[s["gated"]].round(3).to_string())
    inc = s[s["word"].isin(INCOME_SIGN) & (s["feature"].str.contains("income"))]
    print(inc[["model", "case", "layer", "word", "areas", "tau", "p", "null_q95", "income_sign"]].round(3).to_string())
    sn = senses(root)
    sn.to_csv(out / "reads_senses.csv", index=False)
    print(sn.round(3).to_string())
    homeless(root).to_csv(out / "reads_homeless.csv", index=False)
    plot_validity(val, out / "reads_validity.png")
    plot_ses(s, out / "reads_ses.png")


if __name__ == "__main__":
    main()
