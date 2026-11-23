#!/usr/bin/env python
"""F8.2 and F8.3: the tests of the counterfactual cue and of the R3 readout.

The tests are set in research/docs/facct-runbook.md, F8.2 and F8.3,
before the run. This script does only those tests.

| Part | Test | Files in `research/figures/f8/cue/` |
|------|------|----------------------------|
| F8.2 primary | paired t test of the 4 readouts (`c_yes`, `c_cool`, `c_yes_i`, `c_cool_i`), rich minus poor, for each (model, case, readout); Bonferroni over the rows | `cue_primary.csv`, `cue_primary.png` |
| F8.2 description | the mean difference for each cue pair (borough, pair); no test | `cue_by_pair.csv` |
| F8.2 secondary | McNemar exact test (binomial on the discordant pairs) of each word list, for each (model, case, field, list); Bonferroni over the rows | `cue_words.csv`, `cue_words.png` |
| F8.2 check | the share of answers that say the cue name, for each arm | `cue_name_echo.csv` |
| F8.2 check | the answers of each arm that stop at the token limit (they leave the paired tests) | `cue_cut.csv` |
| F8.3 NTA | Spearman over the NTAs (20 images or more) of the NTA mean of `c` with each ACS feature; Bonferroni over the rows | `r3_nta_spearman.csv` |
| F8.3 models | Pearson of `c` between each 2 models over the same images | `r3_model_pearson.csv` |
| F8.3 table | 1 row for each (model, case, image) of the pool, for method B (F8.1) | `data/f8/r3_pool.parquet` |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f8/analyze_cue.py
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "research" / "notebooks" / "cvpr"))
from research.notebooks.cvpr import _style as S  # noqa: E402

MODELS = ["gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b"]
MODEL_COLORS = {"gemma-4-12b": S.PAL["periwinkle"], "gemma-4-31b": S.PAL["blue"],
                "qwen3.5-9b": S.PAL["mint"], "qwen3.8-27b": S.PAL["teal"]}  # as research/notebooks/facct/_f4.py
MODEL_MARKERS = {"gemma-4-12b": "o", "gemma-4-31b": "o", "qwen3.5-9b": "s", "qwen3.8-27b": "s"}
CASES = ["restaurant", "street"]
READOUTS = ["c_yes", "c_cool", "c_yes_i", "c_cool_i"]  # _i: the instructed form (F8.3)
FEATURES = ["pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_bachelors", "median_household_income"]
ALPHA = 0.05
MIN_NTA_IMAGES = 20

# The word lists of F8.2, set before the run.
WORD_LISTS = {
    "upscale": ["upscale", "high-end", "upmarket", "luxury", "luxurious", "elegant", "sophisticated", "refined",
                "chic", "pricey", "expensive", "exclusive", "posh", "fancy", "swanky"],
    "cheap": ["cheap", "affordable", "budget", "inexpensive", "bargain", "no-frills", "low-cost",
              "hole-in-the-wall", "cheap eats"],
    "danger": ["unsafe", "dangerous", "sketchy", "rough", "gritty", "seedy", "crime", "run-down", "rundown",
               "dilapidated", "neglected"],
    "safe": ["safe", "secure", "peaceful", "quiet", "tranquil"],
}
WORD_RE = {k: re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, v)) + r")(?![\w-])", re.I)
           for k, v in WORD_LISTS.items()}


def read_parts(d: Path) -> pd.DataFrame:
    parts = sorted(d.glob("part-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


def load_r3(root: Path, items_root: Path, source: str) -> pd.DataFrame:
    out = []
    for m in MODELS:
        r = read_parts(root / m / source)
        it = items_root / f"{m}-{source}.parquet"
        if r.empty or not it.exists():
            continue
        items = pd.read_parquet(it, columns=["item_id", "case", "arm", "image_idx", "sample_id"])
        out.append(r.merge(items, on="item_id", how="left", validate="1:1"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bonferroni(df: pd.DataFrame, pcol: str = "p") -> pd.DataFrame:
    df["p_bonf"] = np.minimum(df[pcol] * len(df), 1.0)
    df["significant"] = df["p_bonf"] < ALPHA
    return df


# ── F8.2 ──────────────────────────────────────────────────────────────
def primary(cue: pd.DataFrame, f4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, case), g in cue.groupby(["model", "case"]):
        w = g.pivot_table(index="image_idx", columns="arm", values=READOUTS)
        base = f4[(f4["model"] == m) & (f4["case"] == case)].set_index("image_idx")
        for rd in READOUTS:
            pair = w[rd].dropna(subset=["rich", "poor"])
            d = pair["rich"] - pair["poor"]
            t = stats.ttest_rel(pair["rich"], pair["poor"])
            ci = t.confidence_interval(0.95)
            rows.append({"model": m, "case": case, "readout": rd, "n": len(d),
                         "mean_rich": pair["rich"].mean(), "mean_poor": pair["poor"].mean(),
                         "mean_none": base.loc[base.index.intersection(pair.index), rd].mean(),
                         "diff": d.mean(), "ci_lo": ci.low, "ci_hi": ci.high, "sd_diff": d.std(ddof=1),
                         "share_rich_higher": (d > 0).mean(), "t": t.statistic, "p": t.pvalue})
    return bonferroni(pd.DataFrame(rows))


def cue_answers(runs: Path) -> pd.DataFrame:
    out = []
    for m in MODELS:
        for case in CASES:
            for arm in ("rich", "poor"):
                p = runs / m / f"{case}_{arm}" / "responses.parquet"
                if not p.exists():
                    continue
                d = pd.read_parquet(p)
                d = d[d["error"].isna() & (d["finish_reason"] != "length")]
                s = pd.read_parquet(f"data/f8/cue_{case}_{arm}.parquet", columns=["image_idx", "cue_name"])
                d = d.merge(s, on="image_idx", how="left", validate="1:1")
                d["model"], d["case"], d["arm"] = m, case, arm
                out.append(d)
    return pd.concat(out, ignore_index=True)


def by_pair(cue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in CASES:
        names = pd.read_parquet(f"data/f8/cue_{case}_rich.parquet", columns=["image_idx", "boro", "rich_name", "poor_name"])
        g = cue[cue["case"] == case]
        for m, gm in g.groupby("model"):
            for rd in READOUTS:
                w = gm.pivot_table(index="image_idx", columns="arm", values=rd).dropna(subset=["rich", "poor"])
                w = (w["rich"] - w["poor"]).rename("diff").reset_index().merge(names, on="image_idx")
                for (b, rn, pn), x in w.groupby(["boro", "rich_name", "poor_name"]):
                    rows.append({"model": m, "case": case, "readout": rd, "borough": b, "rich_name": rn,
                                 "poor_name": pn, "n": len(x), "diff": x["diff"].mean(),
                                 "share_rich_higher": (x["diff"] > 0).mean()})
    return pd.DataFrame(rows)


def cut_table(runs: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for case in CASES:
            for arm in ("rich", "poor"):
                p = runs / m / f"{case}_{arm}" / "responses.parquet"
                if p.exists():
                    d = pd.read_parquet(p)
                    rows.append({"model": m, "case": case, "arm": arm, "n": len(d),
                                 "errors": int(d["error"].notna().sum()),
                                 "cut": int((d["finish_reason"] == "length").sum()),
                                 "completion_tokens_p50": d["completion_tokens"].median()})
    return pd.DataFrame(rows)


def words(ans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, echo = [], []
    for (m, case), g in ans.groupby(["model", "case"]):
        for field in ("response", "reasoning"):
            txt = g[field].fillna("")
            if (txt == "").all():
                continue
            hits = {k: txt.str.contains(rx) for k, rx in WORD_RE.items()}
            for k, h in hits.items():
                w = pd.DataFrame({"image_idx": g["image_idx"], "arm": g["arm"], "hit": h}) \
                    .pivot(index="image_idx", columns="arm", values="hit").dropna()
                w = w.astype(bool)
                b = int((w["rich"] & ~w["poor"]).sum())  # rich only
                c = int((~w["rich"] & w["poor"]).sum())  # poor only
                p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
                rows.append({"model": m, "case": case, "field": field, "list": k, "n": len(w),
                             "rate_rich": w["rich"].mean(), "rate_poor": w["poor"].mean(),
                             "rich_only": b, "poor_only": c, "p": p})
            said = [n.lower() in t.lower() for n, t in zip(g["cue_name"], txt)]
            for arm, gg in pd.DataFrame({"arm": g["arm"], "said": said}).groupby("arm"):
                echo.append({"model": m, "case": case, "field": field, "arm": arm, "n": len(gg),
                             "says_cue_name": gg["said"].mean()})
    return bonferroni(pd.DataFrame(rows)), pd.DataFrame(echo)


# ── F8.3 ──────────────────────────────────────────────────────────────
def image_nta() -> pd.DataFrame:
    r = pd.read_parquet("data/f4_traces/pooled/f4_restaurants_nta.parquet", columns=["image_idx", "NTA2020"])
    r = r.rename(columns={"NTA2020": "nta2020"}).assign(case="restaurant")
    s = pd.read_parquet("data/f4_traces/pooled/f4_street_1k.parquet", columns=["image_idx", "nta2020"]).assign(case="street")
    return pd.concat([r, s], ignore_index=True)


def nta_tests(pool: pd.DataFrame) -> pd.DataFrame:
    acs = pd.read_parquet("data/f4_traces/nta_acs.parquet")
    rows = []
    for (m, case), g in pool.groupby(["model", "case"]):
        cnt = g.groupby("nta2020").size()
        keep = cnt[cnt >= MIN_NTA_IMAGES].index
        nta = g[g["nta2020"].isin(keep)].groupby("nta2020")[READOUTS].mean().join(acs.set_index("nta2020")[FEATURES])
        for rd in READOUTS:
            for f in FEATURES:
                x = nta[[rd, f]].dropna()
                rho, p = stats.spearmanr(x[rd], x[f])
                rows.append({"model": m, "case": case, "readout": rd, "feature": f, "n_nta": len(x),
                             "n_images": int(g["nta2020"].isin(keep).sum()), "rho": rho, "p": p})
    return bonferroni(pd.DataFrame(rows))


def model_pearson(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case in CASES:
        for rd in READOUTS:
            w = pool[pool["case"] == case].pivot_table(index="image_idx", columns="model", values=rd)
            for a, b in combinations([m for m in MODELS if m in w], 2):
                x = w[[a, b]].dropna()
                r, p = stats.pearsonr(x[a], x[b])
                rows.append({"case": case, "readout": rd, "model_a": a, "model_b": b, "n": len(x), "r": r, "p": p})
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────
def plot_primary(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(4.2 * len(READOUTS), 3.4), sharey=True)
    labels = {"c_yes": "c_yes", "c_cool": "c_cool", "c_yes_i": "c_yes_i (instructed)", "c_cool_i": "c_cool_i (instructed)"}
    for ax, rd in zip(axes, READOUTS):
        g = df[df["readout"] == rd]
        for j, case in enumerate(CASES):
            for i, m in enumerate(MODELS):
                r = g[(g["case"] == case) & (g["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(MODELS) + 1) + i
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                            ecolor=S.EDGE, elinewidth=0.8, capsize=2)
                ax.scatter(r["diff"], y, s=36, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m], edgecolor=S.EDGE,
                           linewidth=S.EDGE_LW, zorder=3, label=m if (j == 0 and rd == "c_yes") else None)
                if r["significant"]:
                    ax.annotate("*", (r["ci_hi"], y), xytext=(3, -4), textcoords="offset points")
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_xlabel(f"rich minus poor, {labels[rd]}")
    ticks = [j * (len(MODELS) + 1) + (len(MODELS) - 1) / 2 for j in range(len(CASES))]
    axes[0].set_yticks(ticks, CASES)
    axes[0].invert_yaxis()
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.99, -0.02, "bars: 95% CI of the paired mean; *: Bonferroni p < 0.05", ha="right", fontsize=7)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_words(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    g = df[df["field"] == "response"]
    fig, axes = plt.subplots(1, len(WORD_LISTS), figsize=(11, 3.2), sharey=True)
    for ax, k in zip(axes, WORD_LISTS):
        h = g[g["list"] == k]
        for j, case in enumerate(CASES):
            for i, m in enumerate(MODELS):
                r = h[(h["case"] == case) & (h["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(MODELS) + 1) + i
                ax.plot([r["rate_poor"], r["rate_rich"]], [y, y], color=S.EDGE, lw=0.6)
                ax.scatter(r["rate_poor"], y, s=26, facecolor="white", edgecolor=S.EDGE, marker=MODEL_MARKERS[m], zorder=3)
                ax.scatter(r["rate_rich"], y, s=26, color=MODEL_COLORS[m], edgecolor=S.EDGE, marker=MODEL_MARKERS[m],
                           linewidth=S.EDGE_LW, zorder=3)
                if r["significant"]:
                    ax.annotate("*", (max(r["rate_poor"], r["rate_rich"]), y), xytext=(3, -4), textcoords="offset points")
        ax.set_title(k)
        ax.set_xlabel("share of answers")
    ticks = [j * (len(MODELS) + 1) + (len(MODELS) - 1) / 2 for j in range(len(CASES))]
    axes[0].set_yticks(ticks, CASES)
    axes[0].invert_yaxis()
    fig.text(0.99, -0.03, "response field; hollow: poor arm, filled: rich arm (model colour); *: Bonferroni p < 0.05",
             ha="right", fontsize=7)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="", help="read the tagged dirs (for example smoke)")
    args = ap.parse_args()
    sfx = f"-{args.tag}" if args.tag else ""
    r3, items, runs = Path(f"data/f8/r3{sfx}"), Path(f"data/f8/items{sfx}"), Path(f"data/f8/runs{sfx}")
    out = Path(f"research/figures/f8/cue{sfx}")
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    f4 = load_r3(r3, items, "f4")
    cue = load_r3(r3, items, "cue")
    if not cue.empty:
        prim = primary(cue, f4)
        prim.to_csv(out / "cue_primary.csv", index=False)
        plot_primary(prim, out / "cue_primary.png")
        by_pair(cue).to_csv(out / "cue_by_pair.csv", index=False)
        print(prim.round(4).to_string(index=False))
        w, echo = words(cue_answers(runs))
        w.to_csv(out / "cue_words.csv", index=False)
        echo.to_csv(out / "cue_name_echo.csv", index=False)
        cut = cut_table(runs)
        cut.to_csv(out / "cue_cut.csv", index=False)
        print(cut.to_string(index=False))
        plot_words(w, out / "cue_words.png")
        print(w[w["significant"]].round(4).to_string(index=False))
        print(echo.round(3).to_string(index=False))

    b2 = load_r3(r3, items, "f4b2")
    pool = pd.concat([f4, b2], ignore_index=True)
    if not pool.empty:
        pool = pool.merge(image_nta(), on=["case", "image_idx"], how="left", validate="m:1")
        keep = ["model", "case", "image_idx", "sample_id", "nta2020"] + READOUTS \
            + [c for c in pool.columns if c.startswith("lp_")]
        pool[keep].to_parquet(f"data/f8/r3_pool{sfx}.parquet", index=False)
        nt = nta_tests(pool)
        nt.to_csv(out / "r3_nta_spearman.csv", index=False)
        mp = model_pearson(pool)
        mp.to_csv(out / "r3_model_pearson.csv", index=False)
        print(nt[nt["significant"]].round(4).to_string(index=False))
        print(mp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
