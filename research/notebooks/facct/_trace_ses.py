"""The content of the F9 answers against the 5 area features, in 1 FAccT column.

This is the companion of `_pairwise_ses.py`: the same features and test. It
reads the 6 F4 models at the NTA and the community district: 1 tract holds too
few images. The pairwise figure reads a judgment. This figure reads what the
answers of the F9 prompts say, for the 2 cases of F9:

| Row | Outcome of 1 text | Text enters when |
|-----|-------------------|------------------|
| Net valence | (good - bad) / `visual_evidence` spans with a valence (`analyze_subjective.text_valence`) | it has 1 such span or more |
| Net valence of social inferences | (good - bad) / labelled inference spans of upkeep, wealth, demographic, class, safety | it has 1 such span or more |
| Location inference | 1 when the text has an `inference` span of kind `location`, else 0 | always |
| Uncertainty, in view | 1 when the text has an `uncertainty` span with a reason other than `out_of_view`, else 0 | always |

`out_of_view` is about what the frame hides (the interior, the food), not
about doubt on what the image shows. It is the most common reason in the
restaurant case, thus the row leaves it out. `uncertainty_all` keeps all the
reasons for that comparison: it goes in the CSV, not in the figure or in the
Bonferroni count.

The score of an area is the mean over its texts, for 1 model (the `response`
field), with 3 texts or more. An image falls in an area by its point: the
facade point (restaurant), or the camera point (street). The labels of the
social spans come from `research/scripts/f9/label_inference_valence.py`. The figure
leaves out the row of social inferences until the labels are on disk.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/facct/_trace_ses.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"
_HERE = Path(__file__).resolve().parent
for _p in (_CVPR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research.notebooks.facct import _f4 as F  # noqa: E402
from research.notebooks.facct import _pairwise_ses as P  # noqa: E402
from research.notebooks.cvpr import _style as S  # noqa: E402

__version__ = "1.2.1"

OUT = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "trace_ses"
VALENCE = REPO_ROOT / "data" / "f4_traces" / "subj" / "inference_valence"
POOLED = REPO_ROOT / "data" / "f4_traces" / "pooled"
FIELD = "response"
MODELS = F.MODELS
MODEL_SHORT = {"gemma-4-12b": "Gemma-4-12B", "gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B",
               "qwen3.8-27b": "Qwen3.8-27B", "llama-3.2-11b": "Llama-3.2-11B", "llama-3.2-90b": "Llama-3.2-90B"}
LAYERS = {k: v for k, v in P.LAYERS.items() if v[0] != "tract"}
CASES = {"restaurant": "Restaurants", "street": "Street photography"}
# The order of the rows, top to bottom.
OUTCOMES: Dict[str, str] = {
    "valence": "Net valence",
    "social": "Social net valence",
    "location": "Location inference",
    "uncertainty": "Uncertainty, in view",
}
EXTRA = ("uncertainty_all",)
DROP_REASON = "out_of_view"
MIN_TEXTS = 3
ALPHA = P.ALPHA


def uncertainty_reasons() -> pd.DataFrame:
    """Give the `uncertainty` spans of each text by reason: `doc_id`, `reason`, `spans`. The result goes to a cache."""
    path = VALENCE / "uncertainty_reasons.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sys.path.insert(0, str(REPO_ROOT / "research" / "scripts" / "f9"))
    from research.scripts.f9 import analyze_subjective as A
    from research.scripts.f9 import label_inference_valence as L

    parts = []
    for case, (arm, name, _, _) in L.CASES.items():
        raw = pd.read_parquet(A.X2 / arm / f"f4_{name}_extractions.parquet", filters=[("extraction_class", "==", "uncertainty")])
        sp = A.spans_of(raw[~raw.copy_suspect.astype(bool)].assign(case=case, prompt="f9"))
        parts.append(sp.assign(reason=sp["reason"].fillna("none")).groupby(["doc_id", "reason"]).size().rename("spans").reset_index())
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def text_outcomes(labels: str = "full") -> pd.DataFrame:
    """Give the 4 outcomes of each text of `FIELD`: `case`, `model`, `image_idx` and 1 column each."""
    t = pd.read_parquet(VALENCE / "texts.parquet")
    t = t[(t["field"] == FIELD) & t["model"].isin(MODELS)]
    out = t.assign(social=np.nan)
    if (VALENCE / f"{labels}.parquet").exists():
        spans = pd.read_parquet(VALENCE / "input.parquet", columns=["span_id", "doc_id"])
        lab = spans.merge(pd.read_parquet(VALENCE / f"{labels}.parquet"), on="span_id").dropna(subset=["valence"])
        n = lab.groupby(["doc_id", "valence"]).size().unstack("valence", fill_value=0)
        n = n.reindex(columns=["good", "bad", "neutral"], fill_value=0)
        social = ((n["good"] - n["bad"]) / n.sum(axis=1)).rename("social")
        out = t.merge(social, left_on="doc_id", right_index=True, how="left")
    why = uncertainty_reasons()
    in_view = set(why.loc[why["reason"] != DROP_REASON, "doc_id"])
    return pd.DataFrame({"case": out["case"], "model": out["model"], "image_idx": out["image_idx"],
                         "valence": out["outcome"], "social": out["social"],
                         "location": (out["n_location"] > 0).astype(float),
                         "uncertainty": out["doc_id"].isin(in_view).astype(float),
                         "uncertainty_all": (out["n_uncertainty"] > 0).astype(float)})


def image_keys() -> pd.DataFrame:
    """Give the NTA and the district of each image: `case`, `image_idx`, 1 column for each CVPR layer."""
    import geopandas as gpd

    from research.notebooks.cvpr import _geography as geo

    r = pd.read_parquet(POOLED / "f4_restaurants_nta.parquet", columns=["image_idx", "px", "py"])
    s = pd.read_parquet(POOLED / "f4_street_1k.parquet", columns=["image_idx", "latitude", "longitude"])
    pts = pd.concat([
        gpd.GeoDataFrame({"case": "restaurant", "image_idx": r["image_idx"]},
                         geometry=gpd.points_from_xy(r["px"], r["py"]), crs=geo.WORKING_CRS),
        gpd.GeoDataFrame({"case": "street", "image_idx": s["image_idx"]},
                         geometry=gpd.points_from_xy(s["longitude"], s["latitude"]), crs=4326).to_crs(geo.WORKING_CRS),
    ], ignore_index=True)
    out = pts[["case", "image_idx"]].copy()
    for layer in LAYERS:
        key = geo.LAYERS[layer]["key"]
        poly = geo.load_layer(layer)[[key, "geometry"]]
        got = gpd.sjoin(pts, poly, how="left", predicate="within")
        got = got[~got.index.duplicated()]
        out[layer] = got[key].astype("string").reindex(out.index)
    return out


def cells(texts: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each (outcome, case, feature, model, layer): n, tau, p, and the Bonferroni p over the figure."""
    from scipy.stats import kendalltau

    frame = texts.merge(keys, on=["case", "image_idx"], how="inner")
    rows = []
    for layer, (short, _, _, _) in LAYERS.items():
        feats = P.features(layer)
        for (case, model), part in frame.groupby(["case", "model"]):
            for outcome in [o for o in (*OUTCOMES, *EXTRA) if frame[o].notna().any()]:
                ok = part[[layer, outcome]].dropna()
                g = ok.groupby(layer)[outcome].agg(["mean", "size"])
                g = g[g["size"] >= MIN_TEXTS].rename(columns={"mean": "score"})
                area = g.reset_index().rename(columns={layer: "key"})
                area = area.assign(key=area["key"].astype(str)).merge(feats, on="key", how="inner")
                for f in P.FEATURES:
                    v = area[["score", f]].dropna()
                    row = {"outcome": outcome, "case": case, "feature": f, "model": model, "layer": short,
                           "n": len(v), "texts": int(ok.shape[0]), "tau": np.nan, "p": np.nan}
                    if len(v) >= P.MIN_AREAS and v["score"].nunique() > 1:
                        t = kendalltau(v["score"], v[f])
                        row.update(tau=float(t.statistic), p=float(t.pvalue))
                    rows.append(row)
    out = pd.DataFrame(rows)
    m = int(out.loc[out["outcome"].isin(OUTCOMES), "p"].notna().sum())
    out["p_bonf"] = (out["p"] * m).clip(upper=1.0)
    out["significant"] = out["p_bonf"] < ALPHA
    out.attrs["tests"] = m
    return out


def plot(df: pd.DataFrame, width: float = 3.33):
    """Draw 2 panels (restaurants, street photography), with 4 outcome blocks of 5 features top to bottom."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    size = {"NTA": 7, "CD": 18}
    off = dict(zip(MODELS, np.linspace(-0.33, 0.33, len(MODELS))))
    ys: Dict[Tuple[str, str], float] = {}
    heads = []
    y = 0.0
    for i, outcome in enumerate(o for o in OUTCOMES if o in set(df["outcome"])):
        if i:
            y += 0.35
        heads.append((y, OUTCOMES[outcome]))
        y += 1
        for f in P.FEATURES:
            ys[(outcome, f)] = y
            y += 1
    row_h = 0.2
    height = row_h * y + 1.15
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True, sharex=True)
    for ax, (case, title) in zip(axes, CASES.items()):
        sub = df[df["case"] == case]
        for (outcome, f), yy in ys.items():
            for m in MODELS:
                part = sub[(sub["outcome"] == outcome) & (sub["feature"] == f) & (sub["model"] == m)
                           & sub["tau"].notna()]
                xs = []
                for short in ("NTA", "CD"):
                    r = part[part["layer"] == short]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    xs.append(r["tau"])
                    ax.scatter(r["tau"], yy + off[m], s=size[short], marker=F.MODEL_MARKERS[m],
                               facecolors=F.MODEL_COLORS[m] if r["significant"] else "white",
                               edgecolors=S.EDGE if r["significant"] else F.MODEL_COLORS[m],
                               linewidths=0.35 if r["significant"] else 0.7, zorder=4)
                if len(xs) == 2:
                    ax.plot(xs, [yy + off[m]] * 2, color=F.MODEL_COLORS[m], lw=0.6, alpha=0.7, zorder=3)
        for h, _ in heads[1:]:
            ax.axhline(h - 0.68, color=S.EDGE, lw=0.4)
        ax.axvline(0, color=S.EDGE, lw=0.8, zorder=2)
        ax.set_title(title, fontsize=7, pad=3)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", lw=0.4, alpha=0.6)
        ax.grid(axis="y", visible=False)
    ax = axes[0]
    labels = [P.FEATURE_LABELS[f] for _, f in ys]
    ax.set_yticks(list(ys.values()) + [h for h, _ in heads])
    ax.set_yticklabels(labels + [t for _, t in heads], fontsize=6.5)
    for tick in ax.get_yticklabels()[len(labels):]:
        tick.set_fontweight("bold")
        tick.set_ha("left")
        tick.set_x(-0.78)
    ax.set_ylim(y - 0.4, -0.6)
    lo, hi = min(0.0, float(np.nanmin(df["tau"]))), max(0.0, float(np.nanmax(df["tau"])))
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    fig.supxlabel(r"Kendall $\tau$ of the area score with the feature", fontsize=7, y=0.55 / height, x=0.63,
                  va="bottom")
    grey = "grey"
    handles = [Line2D([], [], marker=F.MODEL_MARKERS[m], ls="", ms=3, color=F.MODEL_COLORS[m], label=MODEL_SHORT[m])
               for m in MODELS]
    handles += [Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=grey, label="not significant"),
                Line2D([], [], marker="o", ls="", ms=2.4, mfc=grey, mec=grey, label="NTA"),
                Line2D([], [], marker="o", ls="", ms=3.8, mfc=grey, mec=grey, label="community district")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6, handletextpad=0.3,
               columnspacing=0.8)
    fig.subplots_adjust(left=0.27, right=0.985, top=1 - 0.2 / height, bottom=0.87 / height, wspace=0.08)
    return fig


def caption(df: pd.DataFrame) -> str:
    n = {(r.case, r.layer): int(r.n) for r in df[df.outcome == "location"].drop_duplicates(["case", "layer"]).itertuples()}
    areas = "; ".join(f"{CASES[c].lower()} {n[(c, 'NTA')]} and {n[(c, 'CD')]}" for c in CASES)
    social = ("the net valence of the social inferences (upkeep, wealth, demographic, class, safety), which the "
              "extractor labelled in a second pass; " if "social" in set(df["outcome"]) else "")
    thin = ""
    if "social" in set(df["outcome"]):
        rows = df[df.outcome == "social"]
        parts = [f"{CASES[c].lower()} {g['texts'].min():,} to {g['texts'].max():,} texts for each model and {g[g.layer == 'NTA']['n'].min()} "
                 f"to {g[g.layer == 'NTA']['n'].max()} NTAs" for c, g in rows.groupby("case")]
        thin = (" The social row has fewer areas, because it reads only the answers with a social inference: "
                + "; ".join(parts) + ".")
    return ("What the answers to the subjective prompts say, against the socioeconomic features of the area. Each "
            "row is the Kendall tau-b of the area score of each of the 6 models with 1 ACS 2024 feature, over the areas with "
            "residents and 3 or more texts. The area score is the mean over the final answers of the images in the "
            f"area, for {df.outcome.nunique()} outcomes: the net valence of the visual evidence; {social}the share "
            "of answers with a location inference; and the share of answers with an uncertainty span about the image "
            "itself, without the spans about what the frame hides (the reason `out_of_view`). A positive "
            "value says that the outcome is higher in the area with more of the feature. The marker size grows with "
            "the layer, NTA and community district, and a line joins the 2 layers of 1 model. A hollow "
            f"marker is not significant after Bonferroni over the {df.attrs['tests']} tests of the figure (alpha "
            f"{ALPHA}). Areas (NTA, community district), the other rows: {areas}.{thin}")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    from research.notebooks.cvpr import _figmeta as FM

    S.apply_house_style()
    OUT.mkdir(parents=True, exist_ok=True)
    labels = sys.argv[1] if len(sys.argv) > 1 else "full"
    full = cells(text_outcomes(labels), image_keys())
    full.to_csv(OUT / "trace_ses_cells.csv", index=False)
    df = full[full["outcome"].isin(OUTCOMES)].copy()
    df.attrs["tests"] = full.attrs["tests"]
    fig = plot(df)
    FM.save_figure(fig, OUT / "trace_ses.png", dpi=300, title="The content of the answers and the area features",
                   caption=caption(df), label="fig:trace-ses",
                   tags=["f9", "restaurants", "street_photography", "valence", "inference", "uncertainty", "acs", "kendall"],
                   provenance={"script": "research/notebooks/facct/_trace_ses.py", "version": __version__, "labels": labels})
    fig.savefig(OUT / "trace_ses.pdf", bbox_inches="tight")
    print(df.pivot_table(index=["outcome", "case", "feature"], columns=["model", "layer"], values="tau").round(2).to_string())
    print(df.groupby(["outcome", "case", "model", "layer"])[["n", "texts"]].first().to_string())
    print(df.groupby(["outcome", "case"])["significant"].sum().unstack().to_string())
    both = full[full["outcome"].isin(["uncertainty", "uncertainty_all"])]
    print(both.pivot_table(index=["case", "feature"], columns=["outcome", "layer"], values="tau").round(2).to_string())
    print(both.groupby(["outcome", "case"])["significant"].sum().unstack().to_string())
    print("significant", int(df.significant.sum()), "of", df.attrs["tests"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
