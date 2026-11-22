#!/usr/bin/env python
"""F10.3: the held-out test of the removal hypothesis (D157).

The tests are set in research/docs/facct-runbook.md, before the run: 80 paired
t tests on `c_yes_i` (Bonferroni), 16 cells x 5 tests, on 400 images that F10.3 did
not use. Each test reads with the expected sign of F10.3. The hypothesis covers the 4
cells with Bronx images and the prompt `manhattan`.

| File in `research/figures/f10/heldout/` | Content |
|--------------------------------|---------|
| `heldout_primary.csv` | the 80 tests |
| `heldout_cells.csv` | the text effect and the mean shift of each condition |
| `heldout_shifts.png`, `.json` | the shifts against the text effect |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_heldout.py [--runs data/f10/swap-heldout]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "f8"))
from analyze_cue import CASES, S, bonferroni, read_parts  # noqa: E402
from analyze_decoupled import paired  # noqa: E402

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b")
IMAGES = ("Bronx", "Manhattan")
PROMPTS = {"manhattan": "bronx", "bronx": "manhattan"}
TESTS = {"delete_shift": ("delete", "clean"), "delete_vs_match": ("delete", "rand_match"), "delete_vs_add": ("delete", "add"),
         "swap_vs_match": ("swap", "rand_match"), "swap_vs_null": ("swap", "null")}
CONDS = ("swap", "swap2", "delete", "add", "null", "null2", "rand_match", "rand_match2")
READOUT = "c_yes_i"
OUT = Path("research/figures/f10/heldout")


def tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prim, cells = [], []
    for (m, case, boro), d in df.groupby(["model", "case", "borough"]):
        for arm, other in PROMPTS.items():
            w = d[d["arm"] == arm].pivot(index="image_idx", columns="cond", values=READOUT)
            w = w.join(d[(d["arm"] == other) & (d["cond"] == "clean")].set_index("image_idx")[READOUT].rename("other"))
            effect = paired(w, "other", "clean")
            sign = float(np.sign(effect["diff"]))
            key = {"model": m, "case": case, "images": boro, "prompt": arm, "hypothesis": boro == "Bronx" and arm == "manhattan"}
            for name, (a, b) in TESTS.items():
                r = paired(w, a, b)
                prim.append({**key, "test": name, "sign": sign, **r, "signed_diff": sign * r["diff"]})
            row = {**key, "text_effect": effect["diff"], "n": effect["n"]}
            for c in CONDS:
                r = paired(w, c, "clean")
                row[c] = r["diff"]
                row[f"{c}_lo"], row[f"{c}_hi"] = r["ci_lo"], r["ci_hi"]
                row[f"{c}_edit"] = d[(d["arm"] == arm) & (d["cond"] == c)]["edit_ratio"].median()
            cells.append(row)
    prim = bonferroni(pd.DataFrame(prim))
    prim["passes"] = prim["significant"] & (prim["signed_diff"] > 0)
    return prim, pd.DataFrame(cells)


def plot(cells: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    S.apply_house_style()
    marks = {"swap": ("o", S.PAL["blue"]), "delete": ("v", S.PAL["periwinkle"]), "add": ("^", "white"), "null": ("s", "white"),
             "rand_match": ("D", "white")}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=True, gridspec_kw={"hspace": 0.4})
    pairs = [(b, p) for b in IMAGES for p in PROMPTS]
    for i, m in enumerate(MODELS):
        for j, case in enumerate(CASES):
            ax = axes[i, j]
            for y, (b, p) in enumerate(pairs):
                c = cells[(cells["model"] == m) & (cells["case"] == case) & (cells["images"] == b) & (cells["prompt"] == p)]
                if c.empty:
                    continue
                c = c.iloc[0]
                ax.scatter(c["text_effect"], y, marker="|", s=120, color=S.EDGE, zorder=2)
                for k, (cond, (mk, col)) in enumerate(marks.items()):
                    yy = y + (k - 2) * 0.14
                    ax.errorbar(c[cond], yy, xerr=[[c[cond] - c[f"{cond}_lo"]], [c[f"{cond}_hi"] - c[cond]]], fmt="none",
                                ecolor=S.EDGE, elinewidth=0.6)
                    ax.scatter(c[cond], yy, marker=mk, s=18, edgecolor=S.EDGE, linewidth=S.EDGE_LW, color=col, zorder=3,
                               label=cond if y == 0 else None)
            ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
            ax.set_title(f"{m}, {case}")
            if i == 1:
                ax.set_xlabel("change in the readout (nats)")
    axes[0, 0].set_yticks(range(len(pairs)), [f"{b} images, {p}" for b, p in pairs])
    axes[0, 0].invert_yaxis()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.03))
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    meta = {"plot-title": "The held-out test of removal",
            "plot-caption": "Mean change of c_yes_i from the clean prompt, paired over 100 new images, 95% CI. The bar marks the "
                            "effect of the text swap on the same images. delete: the name coordinate goes to 0. add: the other "
                            "name takes the coordinate of the name, and the name stays. swap: both. null: the name against an "
                            "unrelated noun. rand_match: the swap edit along random vectors, with the same size at each position.",
            "plot-latex-label": "fig:" + path.stem.replace("_", "-"), "plot-tags": ["F10.3", "coordinate swap", "held-out"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="data/f10/swap-heldout")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([read_parts(root / m).assign(model=m) for m in MODELS], ignore_index=True)
    used = pd.concat(read_parts(Path("data/f10/swap") / m)[["case", "image_idx"]] for m in MODELS).drop_duplicates()
    overlap = df.merge(used, on=["case", "image_idx"]).shape[0]
    print("images", df.groupby(["model", "case", "borough"])["image_idx"].nunique().to_dict(), "overlap with F10.3:", overlap,
          "missing", int(df[READOUT].isna().sum()))
    prim, cells = tests(df)
    prim.to_csv(out / "heldout_primary.csv", index=False)
    cells.to_csv(out / "heldout_cells.csv", index=False)
    plot(cells, out / "heldout_shifts.png")
    pd.set_option("display.width", 220)
    print(len(prim), "tests")
    keys = ["model", "case", "images", "prompt"]
    print(cells[keys + ["n", "text_effect", *CONDS]].round(2).to_string())
    print(prim.pivot_table(index=keys, columns="test", values="signed_diff").round(2).to_string())
    print(prim.pivot_table(index=keys, columns="test", values="passes").astype(int).to_string())
    print(cells.groupby("model")[[f"{c}_edit" for c in CONDS]].median().round(5).T.to_string())
    y = df.assign(yes=df["top"].str.strip().str.lower() == "yes").groupby(["model", "arm", "borough", "cond"])["yes"].mean().unstack("cond")
    print(y.round(2).to_string())
    print(df.groupby(["model", "cond"])["top"].agg(lambda s: s.value_counts().head(3).to_dict()).to_string())


if __name__ == "__main__":
    main()
