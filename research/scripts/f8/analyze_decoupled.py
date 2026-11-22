#!/usr/bin/env python
"""F8.5: the tests of the decoupled cue (D136).

The tests are set in vlm-narratives-docs/facct-runbook.md, F8.5, before the
run. The primary tests are 32 paired t tests (Bonferroni): 4 models x 2
cases x 2 instructed readouts x 2 contrasts.

| Contrast | Value |
|----------|-------|
| `race` | `w_poor` - `n_match`: the whiter name against a non-white name with the same income |
| `reversed` | `w_poor` - `n_rich`: the whiter, poorer name against a non-white, richer name |

| File in `research/figures/f8/decoupled/` | Content |
|---------------------------------|---------|
| `dc_primary.csv`, `dc_primary.png` | the 32 primary tests |
| `dc_contrasts.csv` | all contrasts (also `class` = `n_rich` - `n_match`, and each named arm - `city`), all 4 readouts, by set (`all`, `Brooklyn`, `Queens`, `own_borough`); paired t, no correction: a description |
| `dc_arm_means.csv`, `dc_arm_means.png` | the mean of each arm on the images with all 4 readouts, and the no-cue F4 mean |
| `dc_words.csv` | the word lists of F8.2: exact McNemar for `race` and `reversed`; Bonferroni over the rows |
| `dc_cut.csv` | the answers of each arm that stop at the token limit |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f8/analyze_decoupled.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cue import (ALPHA, CASES, MODEL_COLORS, MODEL_MARKERS, MODELS, READOUTS, WORD_RE,  # noqa: E402
                         S, bonferroni, load_r3, read_parts)

ARMS = ("w_poor", "n_match", "n_rich", "city")
PRIMARY_READOUTS = ("c_yes_i", "c_cool_i")
CONTRASTS = {"race": ("w_poor", "n_match"), "reversed": ("w_poor", "n_rich"), "class": ("n_rich", "n_match"),
             "w_poor_vs_city": ("w_poor", "city"), "n_match_vs_city": ("n_match", "city"),
             "n_rich_vs_city": ("n_rich", "city")}
OUT = Path("research/figures/f8/decoupled")


def design() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f"data/f8/dc_{c}_w_poor.parquet",
                                      columns=["image_idx", "cue_boro", "own_borough"]).assign(case=c)
                      for c in CASES], ignore_index=True)


def paired(w: pd.DataFrame, a: str, b: str) -> dict:
    if a not in w or b not in w:  # an empty subset, for example Queens in a smoke run
        return {"n": 0}
    x = w[[a, b]].dropna()
    d = x[a] - x[b]
    if len(d) < 3:
        return {"n": len(d)}
    t = stats.ttest_rel(x[a], x[b])
    ci = t.confidence_interval(0.95)
    return {"n": len(d), "mean_a": x[a].mean(), "mean_b": x[b].mean(), "diff": d.mean(), "ci_lo": ci.low,
            "ci_hi": ci.high, "sd_diff": d.std(ddof=1), "share_a_higher": (d > 0).mean(), "t": t.statistic,
            "p": t.pvalue}


def contrasts(r3: pd.DataFrame, des: pd.DataFrame) -> pd.DataFrame:
    rows = []
    r3 = r3.merge(des, on=["case", "image_idx"], how="left", validate="m:1")
    subsets = {"all": lambda g: g, "Brooklyn": lambda g: g[g["cue_boro"] == "Brooklyn"],
               "Queens": lambda g: g[g["cue_boro"] == "Queens"], "own_borough": lambda g: g[g["own_borough"]]}
    for (m, case), g in r3.groupby(["model", "case"]):
        for sub, f in subsets.items():
            w = f(g).pivot_table(index="image_idx", columns="arm", values=list(READOUTS))
            for rd in READOUTS:
                for name, (a, b) in CONTRASTS.items():
                    wr = w[rd] if rd in w.columns.get_level_values(0) else pd.DataFrame()
                    rows.append({"model": m, "case": case, "subset": sub, "readout": rd, "contrast": name,
                                 "arm_a": a, "arm_b": b, **paired(wr, a, b)})
    return pd.DataFrame(rows)


def arm_means(r3: pd.DataFrame, f4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, case), g in r3.groupby(["model", "case"]):
        for rd in READOUTS:
            w = g.pivot_table(index="image_idx", columns="arm", values=rd).dropna(subset=list(ARMS))
            base = f4[(f4["model"] == m) & (f4["case"] == case)].set_index("image_idx")[rd]
            row = {"model": m, "case": case, "readout": rd, "n": len(w),
                   "none": base.reindex(w.index).mean()}
            row.update({arm: w[arm].mean() for arm in ARMS})
            rows.append(row)
    return pd.DataFrame(rows)


def answers(runs: Path) -> pd.DataFrame:
    out = []
    for m in MODELS:
        for case in CASES:
            for arm in ARMS:
                p = runs / m / f"{case}_{arm}" / "responses.parquet"
                if p.exists():
                    out.append(pd.read_parquet(p).assign(model=m, case=case, arm=arm))
    return pd.concat(out, ignore_index=True)


def cut_table(ans: pd.DataFrame) -> pd.DataFrame:
    return (ans.groupby(["model", "case", "arm"])
            .agg(n=("image_idx", "size"), errors=("error", lambda s: int(s.notna().sum())),
                 cut=("finish_reason", lambda s: int((s == "length").sum())),
                 completion_tokens_p50=("completion_tokens", "median")).reset_index())


def words(ans: pd.DataFrame) -> pd.DataFrame:
    ans = ans[ans["error"].isna() & (ans["finish_reason"] != "length")]
    rows = []
    for (m, case), g in ans.groupby(["model", "case"]):
        for field in ("response", "reasoning"):
            for lst, rx in WORD_RE.items():
                hit = g.assign(h=g[field].fillna("").map(lambda t: bool(rx.search(t))))
                w = hit.pivot_table(index="image_idx", columns="arm", values="h", aggfunc="first")
                for name in ("race", "reversed"):
                    a, b = CONTRASTS[name]
                    x = w[[a, b]].dropna().astype(bool)
                    n10, n01 = int((x[a] & ~x[b]).sum()), int((~x[a] & x[b]).sum())
                    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
                    rows.append({"model": m, "case": case, "field": field, "list": lst, "contrast": name,
                                 "n": len(x), "rate_a": x[a].mean(), "rate_b": x[b].mean(), "a_only": n10,
                                 "b_only": n01, "p": p})
    return bonferroni(pd.DataFrame(rows))


def caption(path: Path, title: str, text: str) -> None:
    """Write the caption data next to the image (no caption text in the image)."""
    meta = {"plot-title": title, "plot-caption": text, "plot-latex-label": "fig:" + path.stem.replace("_", "-"),
            "plot-tags": ["F8.5", "decoupled cue"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def plot_primary(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    cols = [(c, rd) for c in ("race", "reversed") for rd in PRIMARY_READOUTS]
    fig, axes = plt.subplots(1, len(cols), figsize=(7.2, 2.4), sharey=True)
    for ax, (con, rd) in zip(axes, cols):
        g = df[(df["contrast"] == con) & (df["readout"] == rd)]
        for j, case in enumerate(CASES):
            for i, m in enumerate(MODELS):
                r = g[(g["case"] == case) & (g["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(MODELS) + 1) + i
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                            ecolor=S.EDGE, elinewidth=0.7, capsize=1.5)
                ax.scatter(r["diff"], y, s=26, marker=MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                           zorder=3, color=MODEL_COLORS[m] if r["significant"] else "white")
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(f"{con}, {rd}")
        ax.set_xlabel("whiter name minus other (nats)")
    ticks = [j * (len(MODELS) + 1) + (len(MODELS) - 1) / 2 for j in range(len(CASES))]
    axes[0].set_yticks(ticks, CASES)
    axes[0].invert_yaxis()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    caption(path, "The decoupled cue: the whiter name minus the other name",
            "Paired mean, 95% CI. race: w_poor - n_match (the same income); reversed: w_poor - n_rich (the whiter "
            "name is the poorer name). Filled: Bonferroni p < 0.05 over 32 tests; hollow: not significant.")


def plot_arm_means(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    style = {"none": (S.PAL["warmgrey"], "D"), "city": (S.PAL["slate"], "s"), "w_poor": (S.PAL["amber"], "o"),
             "n_match": (S.PAL["coral"], "o"), "n_rich": (S.PAL["teal"], "o")}
    fig, axes = plt.subplots(len(PRIMARY_READOUTS), len(CASES), figsize=(6.4, 3.8), sharey=True)
    for i, rd in enumerate(PRIMARY_READOUTS):
        for j, case in enumerate(CASES):
            ax = axes[i][j]
            g = df[(df["readout"] == rd) & (df["case"] == case)].set_index("model")
            for k, m in enumerate(MODELS):
                if m not in g.index:
                    continue
                vals = [g.loc[m, c] for c in style]
                ax.plot([np.nanmin(vals), np.nanmax(vals)], [k, k], color=S.EDGE, lw=0.5, zorder=1)
                for c, (col, mk) in style.items():
                    ax.scatter(g.loc[m, c], k, s=22, color=col, marker=mk, edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                               zorder=3, label=c if (i == 0 and j == 0 and k == 0) else None)
            ax.set_title(f"{case}, {rd}")
            ax.set_yticks(range(len(MODELS)), MODELS)
    axes[0][0].invert_yaxis()
    fig.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    caption(path, "The decoupled cue: the mean readout of each arm",
            "The mean readout of each arm over the images with all 4 arms, and the no-name F4 mean (none).")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r3 = load_r3(Path("data/f8/r3"), Path("data/f8/items"), "dc")
    f4 = load_r3(Path("data/f8/r3"), Path("data/f8/items"), "f4")
    des = design()
    con = contrasts(r3, des)
    con.to_csv(OUT / "dc_contrasts.csv", index=False)
    prim = con[(con["subset"] == "all") & con["contrast"].isin(["race", "reversed"])
               & con["readout"].isin(PRIMARY_READOUTS)].copy()
    assert len(prim) == 32, len(prim)
    prim = bonferroni(prim.reset_index(drop=True))
    prim.to_csv(OUT / "dc_primary.csv", index=False)
    plot_primary(prim, OUT / "dc_primary.png")
    means = arm_means(r3, f4)
    means.to_csv(OUT / "dc_arm_means.csv", index=False)
    plot_arm_means(means, OUT / "dc_arm_means.png")
    ans = answers(Path("data/f8/runs_dc"))
    cut_table(ans).to_csv(OUT / "dc_cut.csv", index=False)
    words(ans).to_csv(OUT / "dc_words.csv", index=False)
    pd.set_option("display.width", 200)
    print(prim[["model", "case", "readout", "contrast", "n", "diff", "ci_lo", "ci_hi", "share_a_higher",
                "p_bonf", "significant"]].round(3).to_string(index=False))
    print(f"significant: {int(prim['significant'].sum())} of {len(prim)} (alpha {ALPHA})")


if __name__ == "__main__":
    main()
