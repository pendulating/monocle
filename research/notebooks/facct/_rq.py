"""The helpers of the research-question notebook (`rq_evidence.py`).

The notebook puts the evidence for the 3 research questions of the FAccT
paper in one place, for the camera-ready figures and tables. The review
that sorts the results is `vlm-narratives-docs/facct-rq-evidence.md`.

This module does no test. It reads the result files that the analysis
scripts and notebooks wrote, and it uses `_f8` for the F8 figures.

| Source | Files |
|--------|-------|
| F4 (the 6-model single-image corpus, pooled, x2 extraction) | `research/notebooks/facct/figures/f4_x2_pooled/*.csv` |
| F8 (method B, the cue, the R3 readout) | `research/figures/f8/` (through `_f8`) |
| F8.5 (the decoupled cue, D136) | `research/figures/f8/decoupled/` (when it exists) |
| The parent audit (2 images, 2 judges) | `research/notebooks/cvpr/master/outputs/robustness_estimand_v1.0.0.csv`, `research/notebooks/cvpr/master/figures/trace_channels_inference_summary.csv` |
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import _f8 as F8
from _f8 import S

__version__ = "1.10.0"

REPO_ROOT = F8.REPO_ROOT
F4 = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "f4_x2_pooled"
PARENT = REPO_ROOT / "research" / "notebooks" / "cvpr" / "master"
DC = REPO_ROOT / "research" / "figures" / "f8" / "decoupled"
OUT = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "rq"
RQ1 = REPO_ROOT / "data" / "rq1"  # the per-patch R-lens words (research/scripts/rq1/)

MODELS = F8.ALL_MODELS
KINDS = ("business_type", "location", "atmosphere", "neighborhood_character", "brand_knowledge", "popularity",
         "quality", "time", "upkeep", "wealth", "demographic")
MAIN_FEATURES = ("pct_white_nh", "pct_black_nh", "pct_bachelors", "median_household_income")


def load() -> Dict[str, pd.DataFrame]:
    """Give each source table, keyed by a short name. `dc_primary` is empty before F8.5 ends."""
    out = {
        "kinds": pd.read_csv(F4 / "f4_response_inference_kind.csv", header=[0, 1], index_col=0),
        "model_diff": pd.read_csv(F4 / "f4_model_differences.csv"),
        "trace": pd.read_csv(F4 / "f4_trace_against_answer.csv"),
        "nta_cor": pd.read_csv(F4 / "f4_nta_feature_correlations.csv"),
        "robust": pd.read_csv(PARENT / "outputs" / "robustness_estimand_v1.0.0.csv"),
        "parent_inf": pd.read_csv(PARENT / "figures" / "trace_channels_inference_summary.csv"),
    }
    p = DC / "dc_primary.csv"
    out["dc_primary"] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    out.update(F8.load_results())
    return out


# ── tables ────────────────────────────────────────────────────────────
def social_rates(md: pd.DataFrame) -> pd.DataFrame:
    """Give the % of restaurant answers with a social inference, a wealth inference, and an uncertainty span."""
    rows = {"any social inference (wealth, class, demographic, crime)": "inference: any of the 4",
            "wealth inference": "inference: kind = wealth",
            "popularity inference": "inference: kind = popularity",
            "uncertainty (the model says it cannot tell)": "uncertainty: any"}
    g = md[(md["case"] == "restaurant") & (md["field"] == "response")].set_index("category")
    t = pd.DataFrame({k: g.loc[v, list(MODELS)] for k, v in rows.items()}).T
    t.index.name = "% of restaurant answers"
    return t


def trace_gap(tr: pd.DataFrame) -> pd.DataFrame:
    """Give the % of texts with a claim in the trace and in the answer, for the 4 thinking models."""
    rows = [("restaurant", "inference: any of the 4", "a social inference"), ("caption", "person_reference: any",
                                                                               "a person")]
    out = []
    for case, claim, label in rows:
        g = tr[(tr["case"] == case) & (tr["claim"] == claim)].set_index("model")
        for field in ("trace", "answer"):
            out.append(pd.Series((g[field] * 100).round(1), name=f"{case}, {label}: {field}"))
    t = pd.DataFrame(out)[[m for m in F8.CUE_MODELS]]
    t.index.name = "% of texts"
    return t


def sign_agreement(cor: pd.DataFrame) -> Dict[str, int]:
    """Count the (case, field, category, feature) rows that are significant in 1 model or more, and their signs."""
    g = cor.groupby(["case", "field", "category", "feature"])
    sig = g["sign"].apply(lambda s: (s != 0).any())
    same = g["rho"].apply(lambda r: bool((np.sign(r.dropna()) == np.sign(r.dropna()).iloc[0]).all()))
    opp = g["sign"].apply(lambda s: bool((s == 1).any() and (s == -1).any()))
    return {"tests": len(cor), "significant tests": int((cor["sign"] != 0).sum()), "rows": int(sig.sum()),
            "same sign in all models": int((same & sig).sum()), "2 significant models, opposite signs": int(opp.sum())}


def parent_income(rob: pd.DataFrame) -> pd.DataFrame:
    """Give the tract-level r of the parent rating with median income, for each judge and elicitation."""
    g = rob[(rob["level"] == "area") & (rob["proxy"] == "median_household_income")]
    g = g.assign(elicitation=np.where(g["side"] == "control", "base prompt", g["arm"]))
    t = g.pivot_table(index="elicitation", columns="model", values="r", aggfunc="first").round(2)
    return t.reindex(["base prompt", "paraphrase", "persona"])


def method_b_counts(R: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Give the significant rows without (model 0) and with the controls (model 1), for the 2 outcomes."""
    rows = []
    for name, f in (("net valence of what the model sees (6 models)", R["mb_fits"]),
                    ("c_cool_i: 'is this a cool place' (4 models)",
                     R["r3_fits"][R["r3_fits"]["readout"] == "c_cool_i"])):
        for case in F8.CASES:
            g = f[f["case"] == case]
            kept = (g["beta1"] / g["beta0"])[g["sig0"]]
            std = (g["beta1"] / g["outcome_sd"]).abs()[g["sig1"]]
            rows.append({"outcome": name, "case": case, "rows": len(g), "significant, no controls": int(g["sig0"].sum()),
                         "significant, with controls": int(g["sig1"].sum()),
                         "median share kept": round(float(kept.median()), 2) if len(kept) else np.nan,
                         "size range (SD / SD)": f"{std.min():.2f} to {std.max():.2f}" if len(std) else "-"})
    return pd.DataFrame(rows).set_index(["outcome", "case"])


# ── figures ───────────────────────────────────────────────────────────
# No caption text in the images (~/style/proper-plotting.md). Each plot
# attaches its caption with `F8.describe`; `save` writes it as JSON.
CONTRAST_TITLES = {"race": "same income", "reversed": "whiter is poorer"}


def plot_kinds(md: pd.DataFrame, case: str = "restaurant"):
    """Draw the share of answers with 1 or more inference spans of each kind, 1 marker for each model.

    Since 1.10.0 the input is `model_diff` (the percent of answers), not `kinds` (spans for each 100 answers),
    so that the figure gives the quantity that the body of the paper reports.
    """
    import matplotlib.pyplot as plt

    g = md[(md["case"] == case) & (md["field"] == "response")].set_index("category")
    k = pd.DataFrame({kind: g.loc[f"inference: kind = {kind}", list(MODELS)].astype(float) for kind in KINDS}).T
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for i, kind in enumerate(KINDS):
        vals = k.loc[kind, list(MODELS)]
        ax.plot([vals.min(), vals.max()], [i, i], color=S.EDGE, lw=0.5, zorder=1)
        for n, m in enumerate(MODELS):  # a small offset for each model, so that equal values stay visible
            ax.scatter(vals[m], i + (n - (len(MODELS) - 1) / 2) * 0.09, s=20, color=F8.MODEL_COLORS[m],
                       marker=F8.MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3)
    ax.set_yticks(range(len(KINDS)), [s.replace("_", " ") for s in KINDS])
    ax.invert_yaxis()
    ax.set_xlabel(f"% of {case} answers with the inference")
    ax.legend(handles=F8._model_handles(MODELS), loc="lower right", frameon=False, fontsize=6)
    F8.describe(fig, "The kinds of inference in the answers",
                f"The percent of {case} answers with 1 or more inference spans of each kind (F4, 6 models, the "
                "final answer). The line spans the lowest and the highest model.",
                ["F4", "inference kinds", "RQ1"])
    return fig


def plot_trace_gap(tr: pd.DataFrame):
    """Draw the share of texts with a claim: the trace (hollow) against the final answer (filled)."""
    import matplotlib.pyplot as plt

    panels = [("caption", "person_reference: any", "a person\n(caption case)"),
              ("restaurant", "inference: any of the 4", "a social inference\n(restaurant case)")]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    for ax, (case, claim, title) in zip(axes, panels):
        g = tr[(tr["case"] == case) & (tr["claim"] == claim)].set_index("model")
        for i, m in enumerate(F8.CUE_MODELS):
            t, a = g.loc[m, "trace"], g.loc[m, "answer"]
            ax.annotate("", (a, i), (t, i), arrowprops=dict(arrowstyle="->", color=S.EDGE, lw=0.6))
            ax.scatter(t, i, s=22, facecolor="white", edgecolor=S.EDGE, marker=F8.MODEL_MARKERS[m], zorder=3)
            ax.scatter(a, i, s=22, color=F8.MODEL_COLORS[m], edgecolor=S.EDGE, marker=F8.MODEL_MARKERS[m],
                       linewidth=S.EDGE_LW, zorder=3)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_xlabel("share of texts")
    axes[0].set_yticks(range(len(F8.CUE_MODELS)), F8.CUE_MODELS)
    axes[0].invert_yaxis()
    F8._top_legend(fig, F8._fill_handles("final answer", "thinking trace"), ncol=2)
    F8.describe(fig, "The thinking trace says more than the final answer",
                "The share of texts that mention a person (caption case) or make a social inference (restaurant "
                "case), in the thinking trace (hollow) and in the final answer (filled, model colour). The arrow "
                "goes from the trace to the answer.",
                ["F4", "thinking trace", "RQ1"])
    return fig


def plot_rq2(R: Dict[str, pd.DataFrame], case: str = "restaurant"):
    """Draw the NTA coefficient with the image-content controls: 1 panel for each outcome.

    Left: the net valence of the `visual_evidence` spans (F8.1, 6 models, the
    response). Right: the R3 readout `c_cool_i` (F8.3, 4 models). The rows
    are the features, with 1 row for each model; both panels use the same
    rows and the same x axis. The size is in SD of the outcome for each SD of
    the feature, with the 95% CI (t, G - 1 df). A hollow marker is not
    significant after the Bonferroni correction.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    panels = [("net valence of the answer", R["mb_fits"][R["mb_fits"]["field"] == "response"]),
              ("cool over lame, instructed", R["r3_fits"][R["r3_fits"]["readout"] == "c_cool_i"])]
    feats = {"median_household_income": ("median_household_income", "log_median_household_income")}
    slots = len(MODELS) + 1  # 1 empty row between the feature groups
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), sharex=True, sharey=True)
    for ax, (title, f) in zip(axes, panels):
        g = f[f["case"] == case]
        for j, feat in enumerate(MAIN_FEATURES):
            h = g[g["feature"].isin(feats.get(feat, (feat,)))]
            for i, m in enumerate(MODELS):
                x = h[h["model"] == m]
                if x.empty:
                    continue
                x = x.iloc[0]
                y = j * slots + i
                q = stats.t.ppf(0.975, x["n_nta"] - 1)
                b, se = x["beta1"] / x["outcome_sd"], x["se1"] / x["outcome_sd"]
                ax.errorbar(b, y, xerr=q * se, fmt="none", ecolor=S.EDGE, elinewidth=0.6, capsize=1)
                ax.scatter(b, y, s=20, marker=F8.MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3,
                           color=F8.MODEL_COLORS[m] if x["sig1"] else "white")
            if j:
                ax.axhline(j * slots - 1, color=S.EDGE, lw=0.4, alpha=0.4)
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    F8._model_rows(axes[0], MODELS, [F8.FEATURE_LABELS[f] for f in MAIN_FEATURES], len(MODELS))
    axes[0].set_ylim(len(MAIN_FEATURES) * slots - 1.5, -0.5)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    F8._top_legend(fig, F8._model_handles(MODELS), ncol=3, xlabel="SD of the outcome for each SD of the feature")
    F8.describe(fig, "The judgments follow the NTA demographics, with the image-content controls",
                f"The OLS coefficient of 1 NTA feature, with the image-content controls (cuisine, DOHMH grade, "
                f"borough), {case} case. Left: the net valence of the visual-evidence spans in the final answer "
                "(F8.1, 6 models). Right: the log-odds of 'cool' over 'lame' after the answer, with the answer "
                "form in the question (F8.3, the 4 models of the readout). Both panels use the same x axis: the "
                "size is in SD of the outcome for each SD of the feature. " + F8.SIG_TEXT + " Standard errors "
                "clustered by NTA; t with G - 1 degrees of freedom (G = the NTAs).",
                ["F8.1", "F8.3", "method B", "RQ2"])
    return fig


def plot_decoupled(dc: pd.DataFrame):
    """Draw the 32 primary tests of F8.5: the whiter name minus the other name, with the 95% CI."""
    import matplotlib.pyplot as plt

    cols = [(c, rd) for c in ("race", "reversed") for rd in ("c_yes_i", "c_cool_i")]
    fig, axes = plt.subplots(1, len(cols), figsize=(7.0, 2.3), sharey=True)
    for ax, (con, rd) in zip(axes, cols):
        g = dc[(dc["contrast"] == con) & (dc["readout"] == rd)]
        for j, case in enumerate(F8.CASES):
            for i, m in enumerate(F8.CUE_MODELS):
                r = g[(g["case"] == case) & (g["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(F8.CUE_MODELS) + 1) + i
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                            ecolor=S.EDGE, elinewidth=0.7, capsize=1.5)
                ax.scatter(r["diff"], y, s=24, marker=F8.MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                           zorder=3, color=F8.MODEL_COLORS[m] if r["significant"] else "white")
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(f"{CONTRAST_TITLES[con]}:\n{F8.readout_name(rd).split(',')[0]}")
    F8._model_rows(axes[0], F8.CUE_MODELS, F8.CASES, len(F8.CUE_MODELS))
    F8._top_legend(fig, F8._model_handles(F8.CUE_MODELS), ncol=4, xlabel="whiter name minus the other name (nats)")
    F8.describe(fig, "The decoupled cue: race and class names that go apart",
                "Each marker is the paired mean of the readout with the whiter neighbourhood name minus the "
                "readout with a non-white name, for the same image. Left 2 panels: the 2 names have the same "
                "income (Borough Park against East New York; Maspeth against Hollis). Right 2 panels: the whiter "
                "name is the poorer name (against Canarsie; against Laurelton). Both readouts name the answer form in "
                "the question (instructed). " + F8.SIG_TEXT +
                f" Paired t tests, Bonferroni over {len(dc)} tests.",
                ["F8.5", "decoupled cue", "RQ3"])
    return fig


PATCH_LAYER = 42  # the R-lens names the content of an image token at layers 36 to 46 (r-lens.md)
DICT_PATH = Path("/usr/share/dict/american-english")


def _english() -> set:
    """Give the lower-case English words of the system word list, 3 letters or more."""
    return {w.strip().lower() for w in DICT_PATH.read_text(errors="ignore").split() if len(w.strip()) >= 3
            and w.strip().isalpha()}


TOKEN_FONTS = REPO_ROOT / "data" / "fonts" / "noto"  # Noto fonts for the scripts that CMU Serif lacks
_TOKEN_FAMILY: list = []


def _token_family() -> list:
    """Give the font list for raw lens tokens: CMU Serif first, then a font for each other script.

    matplotlib takes each glyph from the first font in the list that holds
    it, and shapes Arabic and the Indic scripts. The Noto files are in
    `TOKEN_FONTS` (Arabic, Armenian, Bengali, Devanagari, Georgian, Gujarati,
    Hebrew, Korean, Malayalam, Tamil, Thai); Droid Sans Fallback gives Chinese and Japanese.
    """
    from matplotlib import font_manager as fm

    if not _TOKEN_FAMILY:
        files = sorted(TOKEN_FONTS.glob("*.ttf")) + sorted(TOKEN_FONTS.glob("*.otf"))
        for f in files:
            fm.fontManager.addfont(str(f))
        _TOKEN_FAMILY.extend(["CMU Serif", "DejaVu Sans"]
                             + sorted({fm.FontProperties(fname=str(f)).get_name() for f in files})
                             + ["Droid Sans Fallback"])
    return _TOKEN_FAMILY


def _token_size(token: str, base: float, fit: float) -> float:
    """Give the font size at which `token` fits a block that holds `fit` lower-case letters at `base`."""
    import unicodedata

    # A CJK glyph is 2 letters wide; the fallback fonts of the other scripts are wider than CMU Serif.
    wide = sum(2.0 if unicodedata.east_asian_width(c) in "WF" else 1.5 if c.isupper() or ord(c) > 127 else 1.0
               for c in token)
    return base * min(1.0, fit / max(wide, 1.0))


def _top_tokens(d: pd.DataFrame, keys) -> pd.DataFrame:
    """Give the raw top-1 R-lens token of each block, with the spaces stripped (`␣` if nothing is left)."""
    top = d.sort_values("rank").groupby(keys).head(1)
    return top.assign(word=top["token"].str.strip().replace("", "␣"))


def load_patch_words(pool: int = 4, layer: int = PATCH_LAYER, root: Path = RQ1) -> Dict[str, pd.DataFrame]:
    """Give the RQ1 sample, the answer spans, and the raw top-1 R-lens token of each patch block.

    The token is the raw readout: word pieces and tokens of other scripts
    stay (`_top_tokens`). `research/scripts/rq1/patch_words_sample.py` draws the images, and
    `monocle.patch_word_gif --sample` reads the patches. `root` holds the
    sample of 1 draw (for example `data/rq1/seeds/s3`); the lens files of
    all the draws are in `RQ1/patch_words/rlens`.
    """
    sample = pd.read_parquet(root / "patch_words_sample.parquet")
    spans = pd.read_parquet(root / "patch_words_spans.parquet")
    words, cells = [], []
    for r in sample.itertuples():
        stem = f"{r.photo_id}_{r.lens_case}" + (f"_p{pool}" if pool > 1 else "")
        d = pd.read_parquet(RQ1 / "patch_words" / "rlens" / f"{stem}.jlens.parquet")
        d = d[d["layer"] == layer]
        cells.append(d[["patch_row", "patch_col"]].drop_duplicates().assign(photo_id=r.photo_id))
        words.append(_top_tokens(d, ["patch_row", "patch_col"]).assign(photo_id=r.photo_id))
    return {"sample": sample, "spans": spans, "words": pd.concat(words, ignore_index=True),
            "cells": pd.concat(cells, ignore_index=True)}


def _answer_lines(spans: pd.DataFrame, n: int = 2, width: int = 40) -> str:
    """Give up to `n` spans of different kinds, in text order, as quoted and wrapped lines."""
    import textwrap

    s = spans.sort_values("char_start").drop_duplicates("kind").head(n)
    out = []
    for t in s["extraction_text"]:
        t = t.strip().rstrip(".")
        t = t if len(t) <= 80 else t[:77].rsplit(" ", 1)[0] + "…"
        out.append(textwrap.fill(f"“{t}”", width=width))
    return "\n".join(out)


def plot_patch_words(P: Dict[str, pd.DataFrame], pool: int = 4):
    """Draw the 8 images with the R-lens top token of each patch block, and 2 spans of the answer under each."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from PIL import Image

    sample, words, spans = P["sample"], P["words"], P["spans"]
    family = _token_family()
    n = 16 // pool
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 5.6), gridspec_kw={"hspace": 0.28, "wspace": 0.06})
    for i, case in enumerate(F8.CASES):
        rows = pd.concat([sample[(sample["case"] == case) & (sample["group"] == g)] for g in ("low", "high")])
        for ax, r in zip(axes[i], rows.itertuples()):
            img = Image.open(r.image_path).convert("RGB")
            ax.imshow(np.asarray(img), extent=(0, n, n, 0))
            ax.imshow(np.ones((1, 1, 3)), extent=(0, n, n, 0), alpha=0.22)  # a light veil, so the words read
            for k in range(1, n):
                ax.axhline(k, color="white", lw=0.4, alpha=0.6)
                ax.axvline(k, color="white", lw=0.4, alpha=0.6)
            for w in words[words["photo_id"] == r.photo_id].itertuples():
                ax.text(w.patch_col + 0.5, w.patch_row + 0.5, w.word, ha="center", va="center",
                        fontsize=_token_size(w.word, 6.0, 8), family=family, color=S.EDGE,
                        path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
            ax.set_xlim(0, n)
            ax.set_ylim(n, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            title = f"{r.nta_name.split('-')[0]}, {r.boro}"
            ax.set_title(title, fontsize=6.5 * min(1.0, 28 / len(title)), pad=3)  # a long name must fit its panel
            sp = spans[(spans["case"] == r.case) & (spans["image_idx"] == r.image_idx)]
            ax.text(0.0, -0.04, _answer_lines(sp), transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
                    style="italic", linespacing=1.15)
        axes[i][0].text(-0.08, 0.5, case, transform=axes[i][0].transAxes, rotation=90, ha="right", va="center",
                        fontsize=plt.rcParams["axes.titlesize"])
    fig.canvas.draw()
    for j, label in ((0, "low-income NTAs (bottom fifth)"), (2, "high-income NTAs (top fifth)")):
        a, b = axes[0][j].get_position(), axes[0][j + 1].get_position()
        fig.text((a.x0 + b.x1) / 2, a.y1 + 0.045, label, ha="center", va="bottom",
                 fontsize=plt.rcParams["axes.titlesize"])
    seed_text = f"Draw with seed {int(sample['seed'].iloc[0])}. " if "seed" in sample else ""
    F8.describe(fig, "What the image patches carry, and what the answer says",
                f"{seed_text}8 F4 images: 2 restaurant and 2 street images from NTAs in the bottom fifth of median household "
                f"income, and 2 of each from the top fifth (a random draw among the images whose gemma-4-12b answer "
                f"holds 2 social or aesthetic inference spans or more). On each image: the top token that the R-lens "
                f"reads from the gemma-4-12b image tokens at layer {PATCH_LAYER}, for each {pool}x{pool} block of "
                "patches, as the raw readout (word pieces and tokens of other scripts included). The image comes before the question, so the words do not depend on the question. "
                "Under each image: 2 inference spans of the final answer of gemma-4-12b to the same image (F4, x2 "
                "extraction). The lens names what a patch holds; it does not show which patches move the answer.",
                ["F4", "R-lens", "patch words", "RQ1"])
    return fig


R_LENS_MIN = 30  # r-lens.md: do not read the R-lens below layer 30
GROUP_TITLES = {"low": "low income", "high": "high income"}


def load_patch_layers(photo_ids=None, pool: int = 4, root: Path = RQ1,
                      tag: str = "rlens") -> Dict[str, pd.DataFrame]:
    """Give 4 images of the RQ1 sample and the raw top-1 token of each patch block at every lens layer.

    The default takes the first image of each (case, income group) cell of
    the sample, in sample order. The token rule is the one of `load_patch_words`.
    `tag` picks the lens run (`LAYOUT_TEXT` names each): `rlens` (the image
    before the question), `rlens_text_first` (the question before the image),
    or the pair with the same patch positions, `rlens_question_first` and
    `rlens_neutral_first`.
    """
    sample = pd.read_parquet(root / "patch_words_sample.parquet")
    if photo_ids is None:
        sample = sample.groupby(["case", "group"], sort=False).head(1)
    else:
        sample = sample.set_index("photo_id").loc[list(photo_ids)].reset_index()
    words, cells = [], []
    for r in sample.itertuples():
        stem = f"{r.photo_id}_{r.lens_case}" + (f"_p{pool}" if pool > 1 else "")
        d = pd.read_parquet(RQ1 / "patch_words" / tag / f"{stem}.jlens.parquet").assign(photo_id=r.photo_id)
        cells.append(d[["layer", "patch_row", "patch_col", "photo_id"]].drop_duplicates())
        words.append(_top_tokens(d, ["layer", "patch_row", "patch_col"]))
    return {"sample": sample, "words": pd.concat(words, ignore_index=True),
            "cells": pd.concat(cells, ignore_index=True), "tag": tag}


def patch_changes(P: Dict[str, pd.DataFrame], base: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Give the share of blocks whose top token differs from `base`, for each layer and image."""
    keys = ["photo_id", "layer", "patch_row", "patch_col"]
    m = P["words"][keys + ["word"]].merge(base["words"][keys + ["word"]], on=keys, suffixes=("", "_base"))
    m = m.assign(changed=m["word"] != m["word_base"])
    return m.pivot_table(index="layer", columns="photo_id", values="changed", aggfunc="mean")


def patch_change_table(pairs: Dict[str, tuple]) -> pd.DataFrame:
    """Give the mean share (%) of changed blocks for each layer, 1 column for each (run, base) pair."""
    return pd.DataFrame({k: (patch_changes(a, b).mean(axis=1) * 100).round(0).astype(int)
                         for k, (a, b) in pairs.items()})


_QUESTIONS = ("\"How appealing is this restaurant to eat at?\"; "
              "\"How appealing is this block for a street photography photoshoot?\"")
_FILLER = "\"This is a photograph.\", repeated to the token count of the question"
# The caption text of each lens run (the `--tag` of `monocle.patch_word_gif`).
LAYOUT_TEXT = {
    "rlens": "The image comes before the question, so the words do not depend on the question.",
    "rlens_text_first": f"The question comes before the image ({_QUESTIONS}), so the patches can read it.",
    "rlens_question_first": (f"The order is question, image, filler ({_QUESTIONS}; filler: {_FILLER}), "
                             "so the patches can read the question."),
    "rlens_neutral_first": (f"The order is filler, image, question (filler: {_FILLER}), so the patches read "
                            "the filler but not the question. The patches have the same positions as in the "
                            "order question, image, filler."),
}
LAYOUT_BASE = {
    "rlens": "the image comes first",
    "rlens_text_first": "the question comes first",
    "rlens_question_first": "the question comes first and the filler last",
    "rlens_neutral_first": "the filler comes first and the question last (the same patch positions)",
}

PATCH_ROWS = (6, 18, 30, 36, 42, 46)  # 2 noisy layers for contrast, and all the layers the R-lens reads


def plot_patch_layers(P: Dict[str, pd.DataFrame], pool: int = 4, panel_in: float = 1.3,
                      layers=PATCH_ROWS, base: Optional[Dict[str, pd.DataFrame]] = None):
    """Draw 4 images as columns and 1 row for each of `layers`: the raw R-lens top token of each patch block.

    Full-page figure for the appendix. The row label is grey below
    `R_LENS_MIN`, where the R-lens is not reliable. With `base` (an other
    lens run of the same images), a token that differs from `base` is coral.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from PIL import Image

    sample, words = P["sample"], P["words"]
    if base is not None:
        keys = ["photo_id", "layer", "patch_row", "patch_col"]
        words = words.merge(base["words"][keys + ["word"]].rename(columns={"word": "word_base"}), on=keys, how="left")
        words = words.assign(changed=words["word"] != words["word_base"])
    else:
        words = words.assign(changed=False)
    family = _token_family()
    layers = sorted(layers)
    n, gap, left, top = 16 // pool, 0.05, 0.42, 0.42
    w = left + len(sample) * panel_in + (len(sample) - 1) * gap + 0.05
    h = top + len(layers) * panel_in + (len(layers) - 1) * gap + 0.05
    fig, axes = plt.subplots(len(layers), len(sample), figsize=(w, h), squeeze=False)
    fig.subplots_adjust(left=left / w, right=1 - 0.05 / w, top=1 - top / h, bottom=0.05 / h,
                        wspace=gap / panel_in, hspace=gap / panel_in)
    halo = [pe.withStroke(linewidth=1.5, foreground="white")]
    for j, r in enumerate(sample.itertuples()):
        img = np.asarray(Image.open(r.image_path).convert("RGB"))
        for i, layer in enumerate(layers):
            ax = axes[i][j]
            ax.imshow(img, extent=(0, n, n, 0))
            ax.imshow(np.ones((1, 1, 3)), extent=(0, n, n, 0), alpha=0.22)  # a light veil, so the words read
            for k in range(1, n):
                ax.axhline(k, color="white", lw=0.4, alpha=0.6)
                ax.axvline(k, color="white", lw=0.4, alpha=0.6)
            for t in words[(words["photo_id"] == r.photo_id) & (words["layer"] == layer)].itertuples():
                ax.text(t.patch_col + 0.5, t.patch_row + 0.5, t.word, ha="center", va="center",
                        fontsize=_token_size(t.word, 4.8 * panel_in, 7), family=family,
                        color=S.PAL["coral"] if t.changed else S.EDGE, path_effects=halo)
            ax.set_xlim(0, n)
            ax.set_ylim(n, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(False)
            if j == 0:
                ax.text(-0.06, 0.5, f"layer {layer}", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=6.5, color=S.EDGE if layer >= R_LENS_MIN else S.PAL["warmgrey"])
        axes[0][j].set_title(f"{r.case}, {GROUP_TITLES[r.group]}\n{r.nta_name.split('-')[0]}, {r.boro}",
                             fontsize=6.5, pad=3)
    names = ", ".join(f"{r.nta_name.split('-')[0]} ({r.case})" for r in sample.itertuples())
    order = LAYOUT_TEXT[P.get("tag", "rlens")]
    if base is not None:
        order += f" Coral: the top token differs from the run where {LAYOUT_BASE[base.get('tag', 'rlens')]}."
    F8.describe(fig, "What the image patches carry, layer by layer",
                f"4 F4 images of the RQ1 sample, 1 for each case and income group ({names}). Each row gives the "
                f"top token that the R-lens reads from the gemma-4-12b image tokens at 1 layer, for each {pool}x{pool} "
                "block of patches, as the raw readout (word pieces and tokens of other scripts included). "
                "The rows are layers "
                f"{', '.join(map(str, layers))} of the 8 fitted layers "
                f"({', '.join(map(str, sorted(P['cells']['layer'].unique())))}). Grey row labels: below layer {R_LENS_MIN}, where the R-lens "
                "does not read the content of a token reliably. " + order + " The main text uses layer "
                f"{PATCH_LAYER}.",
                ["F4", "R-lens", "patch words", "RQ1", "appendix"])
    return fig


def save(fig, name: str):
    """Write 1 figure as PDF, PNG and caption JSON to `research/notebooks/facct/figures/rq/`."""
    return F8.save(fig, name, OUT)
