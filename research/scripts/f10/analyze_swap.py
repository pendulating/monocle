#!/usr/bin/env python
"""F10.3: the tests of the coordinate swap of the place names (D152, D153).

The tests are set in vlm-narratives-docs/facct-runbook.md, F10.3, before the run.
The primary tests are 64 paired t tests on `c_yes_i` (Bonferroni): 16 cells
(model, case, image borough, prompt) x 4 tests. Each test reads with the expected
sign: the sign of (`clean` of the other prompt - `clean` of this prompt).

| Test | Value |
|------|-------|
| `shift` | `swap` - `clean` |
| `vs_null` | `swap` - `null` |
| `vs_logit` | `swap` - `swap_logit` |
| `vs_random` | `swap` - `random` |

| File in `research/figures/f10/swap/` | Content |
|-----------------------------|---------|
| `swap_primary.csv` | the 64 primary tests |
| `swap_cells.csv` | the text effect, the recovery and the success rule of each cell |
| `swap_conditions.csv` | the mean shift of each condition in each cell: a description |
| `swap_agreement.csv` | the clean values against the F10.1 run |
| `swap_coords.csv` | the clean coordinates of the 2 names at the token of the name |
| `swap_shifts.png`, `.json` | the shifts against the text effect |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_swap.py [--swap data/f10/swap]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f8"))
from analyze_cue import ALPHA, CASES, MODEL_COLORS, S, bonferroni, read_parts  # noqa: E402
from analyze_decoupled import paired  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b")
IMAGES = ("Manhattan", "Bronx")
PROMPTS = {"manhattan": "bronx", "bronx": "manhattan"}
TESTS = {"shift": "clean", "vs_null": "null", "vs_logit": "swap_logit", "vs_random": "random"}
CONDS = ("swap", "swap2", "swap_name", "swap_logit", "null", "random", "placebo")
READOUT = "c_yes_i"
OUT = Path("research/figures/f10/swap")


def load(root: Path) -> pd.DataFrame:
    return pd.concat([read_parts(root / m).assign(model=m) for m in MODELS if (root / m).exists()], ignore_index=True)


def tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prim, cells, conds = [], [], []
    for (m, case, boro), d in df.groupby(["model", "case", "borough"]):
        for arm, other in PROMPTS.items():
            w = d[d["arm"] == arm].pivot(index="image_idx", columns="cond", values=READOUT)
            base = d[(d["arm"] == other) & (d["cond"] == "clean")].set_index("image_idx")[READOUT].rename("other")
            w = w.join(base)
            effect = paired(w, "other", "clean")
            sign = float(np.sign(effect["diff"]))
            key = {"model": m, "case": case, "images": boro, "prompt": arm}
            for name, ref in TESTS.items():
                r = paired(w, "swap", ref)
                r["signed_diff"] = sign * r["diff"]
                r["one_sign"] = sign * r["diff"] > 0
                prim.append({**key, "test": name, "sign": sign, **r})
            shift = {c: w[c].sub(w["clean"]).mean() for c in CONDS}
            cells.append({**key, "text_effect": effect["diff"], "text_p": effect["p"], "n": effect["n"],
                          "recovery": shift["swap"] / effect["diff"], "recovery_scale2": shift["swap2"] / effect["diff"],
                          "recovery_name": shift["swap_name"] / effect["diff"],
                          "recovery_logit": shift["swap_logit"] / effect["diff"]})
            for c in CONDS:
                r = paired(w, c, "clean")
                conds.append({**key, "cond": c, **{k: r[k] for k in ("n", "diff", "ci_lo", "ci_hi", "t", "p")},
                              "edit_ratio": d[(d["arm"] == arm) & (d["cond"] == c)]["edit_ratio"].median(),
                              "share_changed": (d[(d["arm"] == arm) & (d["cond"] == c)].set_index("image_idx")["top"]
                                                != d[(d["arm"] == arm) & (d["cond"] == "clean")].set_index("image_idx")["top"]).mean()})
    prim = bonferroni(pd.DataFrame(prim))
    prim["passes"] = prim["significant"] & prim["one_sign"]
    cells = pd.DataFrame(cells)
    keys = ["model", "case", "images", "prompt"]
    cells = cells.merge(prim.groupby(keys)["passes"].all().rename("success").reset_index(), on=keys)
    cells["text_passes"] = cells["text_p"] * len(cells) < ALPHA
    return prim, cells, pd.DataFrame(conds)


def agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the clean values with the F10.1 run of the same items."""
    rows = []
    for m in df["model"].unique():
        p = Path("data/f10/runs") / m
        if not p.exists():
            continue
        v = read_parts(p)[["item_id", READOUT]].rename(columns={READOUT: "vllm"})
        c = df[(df["model"] == m) & (df["cond"] == "clean") & (df["arm"] != "bare")]
        c = c.assign(item_id=c["case"] + "_" + c["image_idx"].astype(str) + "_" + c["arm"]).merge(v, on="item_id")
        rows.append({"model": m, "n": len(c), "pearson": c[READOUT].corr(c["vllm"]),
                     "mean_diff": (c[READOUT] - c["vllm"]).mean(), "sd_diff": (c[READOUT] - c["vllm"]).std()})
    return pd.DataFrame(rows)


def coords(root: Path) -> pd.DataFrame:
    parts = [pd.concat(pd.read_parquet(p) for p in sorted((root / m).glob("coords-*.parquet"))).assign(model=m)
             for m in MODELS if (root / m).exists()]
    c = pd.concat(parts, ignore_index=True)
    return c.groupby(["model", "arm", "layer"])[["first", "second", "mean_abs_first", "mean_abs_second"]].mean().reset_index()


def plot_shifts(conds: pd.DataFrame, cells: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    marks = {"swap": ("o", S.PAL["blue"]), "swap_logit": ("s", "white"), "null": ("^", "white"), "random": ("D", "white")}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharey=True)
    labels = [f"{b} images, {p}" for b in IMAGES for p in PROMPTS]
    for i, m in enumerate(MODELS):
        for j, case in enumerate(CASES):
            ax = axes[i, j]
            for y, (b, p) in enumerate((b, p) for b in IMAGES for p in PROMPTS):
                c = cells[(cells["model"] == m) & (cells["case"] == case) & (cells["images"] == b) & (cells["prompt"] == p)]
                g = conds[(conds["model"] == m) & (conds["case"] == case) & (conds["images"] == b) & (conds["prompt"] == p)]
                if c.empty:
                    continue
                ax.scatter(c["text_effect"], y, marker="|", s=120, color=S.EDGE, zorder=2)
                for k, (cond, (mk, col)) in enumerate(marks.items()):
                    r = g[g["cond"] == cond].iloc[0]
                    yy = y + (k - 1.5) * 0.16
                    ax.errorbar(r["diff"], yy, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                                ecolor=S.EDGE, elinewidth=0.6)
                    ax.scatter(r["diff"], yy, marker=mk, s=18, edgecolor=S.EDGE, linewidth=S.EDGE_LW, color=col, zorder=3,
                               label=cond if y == 0 else None)
            ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
            ax.set_title(f"{m}, {case}")
            if i == 1:
                ax.set_xlabel("change in the readout (nats)")
    axes[0, 0].set_yticks(range(len(labels)), labels)
    axes[0, 0].invert_yaxis()
    axes[0, 1].legend(frameon=False, fontsize=7)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": "The coordinate swap against the text swap",
            "plot-caption": "Mean change of c_yes_i from the clean prompt, paired over 100 images, 95% CI. The bar marks the "
                            "effect of the text swap on the same images. swap: the lens coordinates of Manhattan and Bronx. "
                            "swap_logit: the same with the rows of the unembedding. null: the name against an unrelated "
                            "noun. random: a pair of random vectors.",
            "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": ["F10.3", "coordinate swap"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swap", default="data/f10/swap")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.swap), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(root)
    prim, cells, conds = tests(df)
    agree, co = agreement(df), coords(root)
    prim.to_csv(out / "swap_primary.csv", index=False)
    cells.to_csv(out / "swap_cells.csv", index=False)
    conds.to_csv(out / "swap_conditions.csv", index=False)
    agree.to_csv(out / "swap_agreement.csv", index=False)
    co.to_csv(out / "swap_coords.csv", index=False)
    plot_shifts(conds, cells, out / "swap_shifts.png")
    pd.set_option("display.width", 220)
    print(len(prim), "primary tests")
    print(agree.round(3).to_string())
    print(cells.drop(columns=["text_p"]).round(3).to_string())
    print(prim.pivot_table(index=["model", "case", "images", "prompt"], columns="test", values="signed_diff").round(2).to_string())
    print(prim.pivot_table(index=["model", "case", "images", "prompt"], columns="test", values="passes").to_string())
    print(conds.pivot_table(index=["model", "case", "images", "prompt"], columns="cond", values="diff").round(2).to_string())
    print(conds.groupby(["model", "cond"])[["edit_ratio", "share_changed"]].mean().round(4).to_string())
    print(co.round(3).to_string())


if __name__ == "__main__":
    main()
