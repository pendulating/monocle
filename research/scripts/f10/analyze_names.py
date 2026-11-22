#!/usr/bin/env python
"""F10.1: the tests of the text swap of the place name (D148).

The tests are set in research/docs/facct-runbook.md, F10.1, before the
run. The primary tests are 24 t tests on `c_yes_i` (Bonferroni): 4 models x 2
cases x 3 contrasts.

| Contrast | Value |
|----------|-------|
| `name` | paired t over the images: `manhattan` - `bronx` |
| `placebo` | paired t over the images: `brooklyn` - `queens` |
| `cross` | two-sample t: `manhattan` - `bronx` on Manhattan images against the same on Bronx images |

| File in `research/figures/f10/names/` | Content |
|------------------------------|---------|
| `names_primary.csv`, `.png`, `.json` | the 24 primary tests |
| `names_gate.csv` | the gate for step 3, for each model and case |
| `names_contrasts.csv` | all 4 readouts, all image sets, each named arm - `none`, the pixels, and `name_bare` (`manhattan` - `bare`) when the arm exists; no correction: a description |
| `names_top.csv` | the first token of the free answer, for each model, case and readout |
| `names_noise.csv` | the SD of the repeat of each readout |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_names.py [--runs data/f10/runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f8"))
from analyze_cue import ALPHA, CASES, MODEL_COLORS, MODEL_MARKERS, MODELS, S, bonferroni, read_parts  # noqa: E402
from analyze_decoupled import paired  # noqa: E402

__version__ = "1.1.0"

ARMS = ("manhattan", "bronx", "brooklyn", "queens", "none")
READOUTS = ("c_yes_i", "c_yes", "c_lc_i", "c_lc")
PRIMARY = "c_yes_i"
PAIRS = {"name": ("manhattan", "bronx"), "placebo": ("brooklyn", "queens")}
BARE = {"name_bare": ("manhattan", "bare")}
VS_NONE = {f"{a}_vs_none": (a, "none") for a in ARMS[:4]}
IMAGE_SETS = ("Manhattan", "Bronx")
OUT = Path("research/figures/f10/names")


def load(runs: Path, bare: Path | None = None) -> pd.DataFrame:
    cols = ["item_id", "case", "image_idx", "borough", "arm"]
    items = pd.concat(pd.read_parquet(p, columns=cols) for p in ("data/f10/name_items.parquet", "data/f10/name_items_bare.parquet"))
    r = [read_parts(runs / m) for m in MODELS] + ([read_parts(bare / m) for m in MODELS] if bare else [])
    return pd.concat(x for x in r if not x.empty).merge(items, on="item_id", how="left", validate="m:1")


def wide(d: pd.DataFrame, readout: str) -> pd.DataFrame:
    w = d.pivot(index="image_idx", columns="arm", values=readout)
    w = w.reindex(columns=[a for a in (*ARMS, "bare") if a in w.columns])
    return w.join(d.drop_duplicates("image_idx").set_index("image_idx")["borough"])


def cross(w: pd.DataFrame) -> dict:
    """Compare `name` between the 2 image boroughs, with a two-sample t test."""
    x = w.dropna(subset=["manhattan", "bronx"])
    a = (x["manhattan"] - x["bronx"])[x["borough"] == "Manhattan"]
    b = (x["manhattan"] - x["bronx"])[x["borough"] == "Bronx"]
    if min(len(a), len(b)) < 3:
        return {"n": len(a) + len(b)}
    t = stats.ttest_ind(a, b)
    ci = t.confidence_interval(0.95)
    return {"n": len(a) + len(b), "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean(), "ci_lo": ci.low,
            "ci_hi": ci.high, "t": t.statistic, "p": t.pvalue}


def pixels(w: pd.DataFrame) -> dict:
    """Describe `none` on Manhattan images minus `none` on Bronx images."""
    a, b = (w.loc[w["borough"] == s, "none"].dropna() for s in IMAGE_SETS)
    return {"n": len(a) + len(b), "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean()}


def tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prim, desc = [], []
    for (m, case), d in df.groupby(["model", "case"]):
        for rd in READOUTS:
            w = wide(d, rd)
            key = {"model": m, "case": case, "readout": rd}
            for con, (a, b) in PAIRS.items():
                r = {**key, "contrast": con, **paired(w, a, b)}
                desc.append({**r, "images": "all"})
                if rd == PRIMARY:
                    prim.append(r)
                for s in IMAGE_SETS:
                    desc.append({**key, "contrast": con, "images": s, **paired(w[w["borough"] == s], a, b)})
            for con, (a, b) in BARE.items():
                for s in ("all", *IMAGE_SETS):
                    part = w if s == "all" else w[w["borough"] == s]
                    desc.append({**key, "contrast": con, "images": s, **paired(part, a, b)})
            for con, (a, b) in VS_NONE.items():
                desc.append({**key, "contrast": con, "images": "all", **paired(w, a, b)})
            r = {**key, "contrast": "cross", **cross(w)}
            desc.append({**r, "images": "all"})
            if rd == PRIMARY:
                prim.append(r)
            desc.append({**key, "contrast": "pixels", "images": "all", **pixels(w)})
    return bonferroni(pd.DataFrame(prim)), pd.DataFrame(desc)


def gate(prim: pd.DataFrame) -> pd.DataFrame:
    p = prim.pivot(index=["model", "case"], columns="contrast", values="diff")
    sig = prim[prim["contrast"] == "name"].set_index(["model", "case"])["significant"]
    return pd.DataFrame({"name": p["name"], "placebo": p["placebo"], "name_significant": sig,
                         "passes": sig & (p["name"].abs() > p["placebo"].abs())}).reset_index()


def noise(runs: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        p = runs / m / "retest.parquet"
        if not p.exists():
            continue
        a, b = read_parts(runs / m).set_index("item_id"), pd.read_parquet(p).set_index("item_id")
        for rd in READOUTS:
            d = (a[rd] - b[rd]).dropna()
            rows.append({"model": m, "readout": rd, "n": len(d), "mean": d.mean(), "sd": d.std(ddof=1),
                         "max_abs": d.abs().max(), "sd_between_items": a.loc[d.index, rd].std(ddof=1)})
    return pd.DataFrame(rows)


def top_tokens(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rd, col in (("instructed", "top_i"), ("question", "top_q")):
        g = df.groupby(["model", "case", col]).size().rename("n").reset_index()
        g["share"] = g["n"] / g.groupby(["model", "case"])["n"].transform("sum")
        rows.append(g.rename(columns={col: "token"}).assign(readout=rd))
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["readout", "model", "case", "n"], ascending=[True, True, True, False]) \
        .groupby(["readout", "model", "case"]).head(6)


def plot_primary(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    S.apply_house_style()
    labels = {"name": "Manhattan - the Bronx", "placebo": "Brooklyn - Queens", "cross": "name, by image borough"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, con in zip(axes, labels):
        g = df[df["contrast"] == con]
        for j, case in enumerate(CASES):
            for i, m in enumerate(MODELS):
                r = g[(g["case"] == case) & (g["model"] == m)]
                if r.empty or "diff" not in r or pd.isna(r["diff"].iloc[0]):
                    continue
                r = r.iloc[0]
                y = j * (len(MODELS) + 1) + i
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                            ecolor=S.EDGE, elinewidth=0.7, capsize=1.5)
                ax.scatter(r["diff"], y, s=26, marker=MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                           zorder=3, color=MODEL_COLORS[m] if r["significant"] else "white")
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(labels[con])
        ax.set_xlabel("difference (nats)")
    ticks = [j * (len(MODELS) + 1) + (len(MODELS) - 1) / 2 for j in range(len(CASES))]
    axes[0].set_yticks(ticks, CASES)
    axes[0].invert_yaxis()
    shown = [m for m in MODELS if m in set(df["model"])]
    handles = [Line2D([], [], ls="", marker=MODEL_MARKERS[m], mfc=MODEL_COLORS[m], mec=S.EDGE, mew=S.EDGE_LW, ms=5, label=m) for m in shown]
    fig.legend(handles=handles, frameon=False, fontsize=7, ncol=len(shown), loc="upper center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": "The text swap of the place name",
            "plot-caption": "Mean difference of c_yes_i, 95% CI. Left and centre: paired over the images. Right: "
                            "the name difference on Manhattan images minus the same on Bronx images. Filled: "
                            f"Bonferroni p < {ALPHA} over the primary tests; hollow: not significant.",
            "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": ["F10.1", "text swap"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="data/f10/runs")
    ap.add_argument("--bare", default=None, help="the runs of the `bare` arm, when they are in another dir")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(runs, Path(args.bare) if args.bare else None)
    missing = df[list(READOUTS)].isna().sum().to_dict()
    print(df.groupby(["model", "case"])["item_id"].nunique().to_string(), "\nmissing", missing)
    prim, desc = tests(df)
    prim.to_csv(out / "names_primary.csv", index=False)
    desc.to_csv(out / "names_contrasts.csv", index=False)
    g = gate(prim)
    g.to_csv(out / "names_gate.csv", index=False)
    top = top_tokens(df)
    top.to_csv(out / "names_top.csv", index=False)
    noi = noise(runs)
    noi.to_csv(out / "names_noise.csv", index=False)
    plot_primary(prim, out / "names_primary.png")
    pd.set_option("display.width", 200)
    print(len(prim), "primary tests")
    print(desc[desc["contrast"] == "name_bare"].pivot_table(index=["model", "case", "images"], columns="readout", values="diff").round(2).to_string())
    print(prim[["model", "case", "contrast", "n", "diff", "ci_lo", "ci_hi", "t", "p_bonf", "significant"]].round(4).to_string())
    print(g.round(3).to_string())
    print(noi.round(3).to_string())
    print(top.round(3).to_string())
    arms = df.groupby(["model", "case", "arm"])[PRIMARY].mean().unstack("arm")[list(ARMS)]
    print(np.round(arms, 3).to_string())


if __name__ == "__main__":
    main()
