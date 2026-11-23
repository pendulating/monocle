#!/usr/bin/env python
"""F10.14: what the "cool" judgment shares with other judgments of the same image. CPU only.

The input is `probe_query.py`. Each image has a pooled yes-minus-no margin for each
adjective. The cool margin must agree with F10.7, and the script stops if it does not.

| Output in `research/figures/f10/probe/` | Question | Method |
|--------------------------------|----------|--------|
| `probe_links.csv` | Which adjectives move with cool | r with cool over images, NTA-clustered t, df 196; the same given `good` and `nice`, and given the borough. Holm |
| `probe_model.csv` | Which adjectives explain cool together | OLS on z-scored margins with NTA-clustered errors, Holm; out-of-fold R2 of each group; loss when a group leaves |
| `probe_borough.csv` | Which judgments differ by borough | One-way ANOVA on NTA means, df 4 and 192, Holm; borough means in NTA SD units |
| `probe_scene.csv` | How much of each judgment the scene block holds | Out-of-fold R2 from the patch kernel of F10.9 |
| `probe_cross.csv` | Do the 2 models agree | r of the 2 margins for each adjective over images |
| `probe_links.png`, `probe_borough.png` | The 2 main tables | |

A shared yes bias can tie all adjectives together. Thus the partial values matter more than
the raw r.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from research.scripts.f10.analyze_logit import LOGIT_ROOT, MODELS, NAMES, SEED, cluster_r, holm  # noqa: E402
from research.scripts.f10.hop_borough import BOROUGHS, family_r2  # noqa: E402
from research.scripts.f10.hop_free import folds, oof_predict, r2  # noqa: E402
from research.scripts.f10.hop_stage import residual  # noqa: E402
from research.scripts.f10.probe_query import GROUPS  # noqa: E402

__version__ = "1.0.0"

DATA = ROOT / "data/f10/hop_probe"
OUT = ROOT / "research/figures/f10/probe"
GROUP_OF = {w: g for g, v in GROUPS.items() for w in v}


def margins(model: str) -> pd.DataFrame:
    """Give [images, words] of pooled margins. A form outside the top 20 takes the lowest listed value."""
    d = pd.read_json(DATA / model / "probes.jsonl", lines=True).drop_duplicates(["image_idx", "word"], keep="last")
    forms = d[["lp_Yes", "lp_yes", "lp_No", "lp_no"]].to_numpy(dtype=float)
    miss = np.isnan(forms)
    print(f"{model}: {len(d)} rows, words {d['word'].nunique()}, Yes or No in the top 20: {(~miss).any(1).mean():.4f}; each side seen: {((~miss[:, :2]).any(1) & (~miss[:, 2:]).any(1)).mean():.4f}")
    forms = np.where(miss, d[["floor"]].to_numpy(dtype=float), forms)
    d["margin"] = logsumexp(forms[:, :2], axis=1) - logsumexp(forms[:, 2:], axis=1)
    return d.pivot(index="image_idx", columns="word", values="margin")


def analyze(model: str, m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    a = pd.read_parquet(LOGIT_ROOT / model / "answers.parquet").set_index("image_idx")["margin"]
    im = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet").set_index("image_idx").loc[m.index]
    gap = np.corrcoef(m["cool"], a.reindex(m.index))[0, 1]
    print(f"{model}: r of cool with F10.7 {gap:.5f}; max gap {np.abs(m['cool'] - a.reindex(m.index)).max():.3f}")
    if gap < 0.98:
        raise SystemExit("the probe does not agree with F10.7")
    cl, boro = im["nta2020"].to_numpy(), im["borough"].to_numpy()
    y = m["cool"].to_numpy()
    fe = pd.get_dummies(boro).to_numpy(float)
    gen = np.c_[fe, stats.zscore(m[["good", "nice"]].to_numpy(), axis=0)]
    words = [w for w in m.columns if w != "cool"]
    rows = []
    for w in words:
        v = m[w].to_numpy()
        r, t, p = cluster_r(v, y, cl)
        rows.append({"model": model, "word": w, "group": GROUP_OF[w], "mean": v.mean(), "sd": v.std(), "yes_share": float((v > 0).mean()), "r": r, "t": t, "p": p}
                    | dict(zip(("r_within", "t_within", "p_within"), cluster_r(residual(v, fe), residual(y, fe), cl))))
        if w not in ("good", "nice"):
            rows[-1] |= dict(zip(("r_partial", "t_partial", "p_partial"), cluster_r(residual(v, gen), residual(y, gen), cl)))
    links = pd.DataFrame(rows)
    links["p_holm"] = holm(links["p"].to_numpy())
    ok = links["p_partial"].notna()
    links.loc[ok, "p_partial_holm"] = holm(links.loc[ok, "p_partial"].to_numpy())

    z = stats.zscore(m[words].to_numpy(), axis=0)
    zy = stats.zscore(y)
    x = np.c_[np.ones(len(z)), z]
    b = np.linalg.lstsq(x, zy, rcond=None)[0]
    e = zy - x @ b
    codes = pd.factorize(cl)[0]
    g = codes.max() + 1
    bread = np.linalg.inv(x.T @ x)
    score = np.stack([np.bincount(codes, x[:, j] * e, g) for j in range(x.shape[1])], axis=1)
    cov = bread @ (score.T @ score) @ bread * (g / (g - 1)) * ((len(zy) - 1) / (len(zy) - x.shape[1]))
    se = np.sqrt(np.diag(cov))[1:]
    t = b[1:] / se
    p = 2 * stats.t.sf(np.abs(t), g - 1)
    coef = pd.DataFrame({"model": model, "word": words, "group": [GROUP_OF[w] for w in words], "beta": b[1:], "se": se, "t": t, "p": p, "p_holm": holm(p)})
    splits = list(GroupKFold(5).split(y, groups=np.random.default_rng(SEED).permutation(g)[codes]))
    fam = {k: [w for w in words if GROUP_OF[w] == k] for k in ("valence", "property", "value", "neutral")}
    fits = [{"model": model, "set": k, "r2": family_r2(stats.zscore(m[v].to_numpy(), axis=0), y, splits)} for k, v in fam.items()]
    fits.append({"model": model, "set": "all", "r2": family_r2(z, y, splits)})
    for k, v in fam.items():
        rest = [w for w in words if w not in v]
        fits.append({"model": model, "set": f"all minus {k}", "r2": family_r2(stats.zscore(m[rest].to_numpy(), axis=0), y, splits)})
    coef = pd.concat([coef, pd.DataFrame(fits)], ignore_index=True)

    nta = m.groupby(cl).mean()
    b_of = im.groupby("nta2020")["borough"].first().reindex(nta.index).to_numpy()
    rows = []
    for w in m.columns:
        v = nta[w].to_numpy()
        f, q = stats.f_oneway(*[v[b_of == k] for k in BOROUGHS])
        rows.append({"model": model, "word": w, "group": GROUP_OF[w], "F": f, "p": q, "eta2": 1 - sum(((v[b_of == k] - v[b_of == k].mean()) ** 2).sum() for k in BOROUGHS) / ((v - v.mean()) ** 2).sum()}
                    | {k: (v[b_of == k].mean() - v.mean()) / v.std() for k in BOROUGHS})
    bt = pd.DataFrame(rows)
    bt["p_holm"] = holm(bt["p"].to_numpy())

    # The kernel follows the store order of the images, and `m` is sorted by image.
    store = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet", columns=["image_idx"])["image_idx"]
    pos = pd.Series(np.arange(len(store)), index=store.to_numpy()).loc[m.index].to_numpy()
    K = np.load(ROOT / "data/f10/hop_free" / model / "scene_kernel.npy")[np.ix_(pos, pos)]
    reps = [folds(cl, seed=SEED + i) for i in range(3)]
    scene = []
    for w in m.columns:
        v = m[w].to_numpy(dtype=float)
        scene.append({"model": model, "word": w, "group": GROUP_OF[w], "r2_scene": r2(v, np.mean([oof_predict(K, v, s) for s in reps], axis=0), reps[0])})
    return {"links": links, "model": coef, "borough": bt, "scene": pd.DataFrame(scene)}


def figures(links: pd.DataFrame, boro: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.2), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        t = links[(links["model"] == model) & links["r_partial"].notna()].sort_values("r_partial")
        colors = [S.PAL["blue"] if g == "property" else S.PAL["coral"] if g == "value" else S.PAL["amber"] for g in t["group"]]
        ax.barh(t["word"], t["r_partial"], color=colors, edgecolor=S.EDGE, linewidth=S.EDGE_LW)
        ax.axvline(0, color=S.EDGE, linewidth=0.6)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel("partial r with cool")
        ax.set_title(NAMES[model], fontsize=8.5)
    keys = [("property", "blue"), ("value", "coral"), ("valence or neutral", "amber")]
    axes[1].legend(handles=[Patch(facecolor=S.PAL[c], edgecolor=S.EDGE, linewidth=S.EDGE_LW, label=k) for k, c in keys], loc="lower right", fontsize=7, frameon=False)
    fig.savefig(OUT / "probe_links.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (OUT / "probe_links.json").write_text(json.dumps({"plot-title": "Judgments that move with cool", "plot-caption": "Partial r over 591 images between the cool margin and each other adjective margin, given the good and nice margins and the borough. Colour gives the group: property, value or neutral.",
                                                      "plot-latex-label": "fig:probe-links", "plot-tags": ["F10.14", "probe"]}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 7.4), constrained_layout=True)
    order = boro[boro["model"] == MODELS[0]].sort_values("Manhattan")["word"].tolist()
    for ax, model in zip(axes, MODELS):
        g = boro[boro["model"] == model].set_index("word").loc[order, list(BOROUGHS)].to_numpy()
        im = ax.imshow(g, aspect="auto", cmap=S.CMAP_DIV, vmin=-1.5, vmax=1.5, interpolation="nearest")
        ax.set_xticks(range(5), ["Bronx", "Bklyn", "Manh", "Queens", "SI"], fontsize=7)
        ax.set_yticks(range(len(order)), order, fontsize=6.5)
        ax.set_title(NAMES[model], fontsize=8.5)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.5, label="borough mean, in NTA SD")
    fig.savefig(OUT / "probe_borough.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (OUT / "probe_borough.json").write_text(json.dumps({"plot-title": "Judgments by borough", "plot-caption": "Mean margin of each adjective in each borough, as a distance from the mean of all NTAs, in SD of the NTA means. The rows follow the Manhattan column of Gemma-4-31B.",
                                                        "plot-latex-label": "fig:probe-borough", "plot-tags": ["F10.14", "borough"]}, indent=2) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = {m: margins(m) for m in MODELS}
    got = [analyze(m, wide[m]) for m in MODELS]
    tables = {k: pd.concat([g[k] for g in got], ignore_index=True) for k in got[0]}
    for k, v in tables.items():
        v.to_csv(OUT / f"probe_{k}.csv", index=False)
    a, b = (wide[m] for m in MODELS)
    pd.DataFrame([{"word": w, "group": GROUP_OF[w], "r": np.corrcoef(a[w], b[w].reindex(a.index))[0, 1]} for w in a.columns]).to_csv(OUT / "probe_cross.csv", index=False)
    figures(tables["links"], tables["borough"])
    print("[f10.14] finished", flush=True)


if __name__ == "__main__":
    main()
