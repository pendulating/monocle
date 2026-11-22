#!/usr/bin/env python
"""F10.28: the words of the shared aesthetic direction, and the direction against the ACS features. CPU only.

Two parts:

| Part | Question | Method |
|------|----------|--------|
| `words` | Does the shared direction of F10.27 put the words in an aesthetic order? | z-score the direction at each layer, take the mean over the layers, and give the 20 words at each end. 2 figures: `wide` (1 column for each model) and `stacked` (1 row for each model) |
| `prompts` | Do the answers to the 14 prompts follow the ACS gradients? | For each image, E = P(yes) - P(no) and, for Gemma where 5% or more of the answers are `unable`, P_unsure = 1 - P(yes) - P(no). Spearman rho over NTAs of the NTA means with each ACS feature. Bonferroni over all cells. A table |
| `ses` | Do the one-word answers, the shared direction and the trace evidence follow the ACS gradients of the neighborhoods? | Spearman rho over NTAs of each measure with each ACS feature. Bonferroni over all cells |

Measures in `ses` (1 value for each NTA, the mean over its images):

| Measure | Source |
|---------|--------|
| `yes_{adjective}` | The share of yes answers to the one-word prompt (`cool`, `nice`, `ordinary`), `data/f10/prompts/{store}/frame.parquet` |
| `shared_direction` | The NTA mean readout at the noun token, at the test layer of F10.27, projected on the shared direction (`research/figures/f10/shared/{model}_place/direction.npz`) |
| `evidence_valence` | The net valence of the `visual_evidence` spans of the `cool` traces (as `hop_evidence.py`). Gemma-4-31B and Qwen3.5-9B only |

The ACS features come from `data/f4_traces/nta_acs.parquet` (`research/scripts/f4/nta_acs.py`): the shares of
White and Black residents (not Hispanic), the share of adults with a bachelor's degree, and the median
household income.

Limit: neighboring NTAs are not independent. The p values assume that they are.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_hierarchy.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
import _style as S  # noqa: E402
from hop_evidence import nta_targets  # noqa: E402
from hop_prompts import HOP_LOGIT, answer_store, store_name  # noqa: E402

__version__ = "1.2.0"

MODELS = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
TRACES = ("gemma-4-31b", "qwen3.5-9b")
ADJECTIVES = ("cool", "nice", "ordinary")
FEATURES = {"pct_white_nh": "% White", "pct_black_nh": "% Black", "pct_bachelors": "% bachelor's",
            "median_household_income": "median income"}
N_WORDS = 20
DATA = ROOT / "data/f10/prompts"
SHARED = ROOT / "research/figures/f10/shared"
OUT = ROOT / "research/figures/f10/hierarchy"
FIG = ROOT / "research/notebooks/facct/figures/f10"


def store(model: str, adjective: str) -> Path:
    return DATA / (model if adjective == "cool" else f"{model}-{adjective}")


def layer_mean(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Give the vocabulary and the direction weights, z-scored at each layer and averaged over the layers."""
    d = np.load(SHARED / f"{model}_place" / "direction.npz")
    w = d["direction"].astype(np.float64)
    z = (w - w.mean(1, keepdims=True)) / w.std(1, keepdims=True)
    return d["vocab"], z.mean(0)


def words() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        vocab, w = layer_mean(model)
        order = np.argsort(-w)
        for end, idx in (("up", order[:N_WORDS]), ("down", order[::-1][:N_WORDS])):
            rows += [{"model": model, "end": end, "rank": k + 1, "word": vocab[i], "z": float(w[i])} for k, i in enumerate(idx)]
    return pd.DataFrame(rows)


def nta_measures(model: str) -> pd.DataFrame:
    """Give 1 row for each NTA with the measures of 1 model."""
    base = pd.read_parquet(store(model, "cool") / "frame.parquet")
    order = pd.factorize(base["nta2020"])[1].tolist()
    out = pd.DataFrame(index=pd.Index(order, name="nta2020"))
    for adj in ADJECTIVES:
        f = pd.read_parquet(store(model, adj) / "frame.parquet")
        f = f[f.outcome.isin(["yes", "no"])]
        out[f"yes_{adj}"] = f.assign(y=(f.outcome == "yes").astype(float)).groupby("nta2020").y.mean()
    s = json.loads((SHARED / f"{model}_place" / "summary.json").read_text())
    meta = json.loads((store(model, "cool") / "meta.json").read_text())
    b = meta["layers"].index(s["test_layer"])
    x = np.load(store(model, "cool") / "nta_place.npy")[:, b, :].astype(np.float64)
    x -= x.mean(0, keepdims=True)
    direction = np.load(SHARED / f"{model}_place" / "direction.npz")["direction"][b].astype(np.float64)
    out["shared_direction"] = x @ direction
    if model in TRACES:
        y, _, _, _ = nta_targets(model, base, order)
        out["evidence_valence"] = y
    return out


def ses() -> pd.DataFrame:
    acs = pd.read_parquet(ROOT / "data/f4_traces/nta_acs.parquet").set_index("nta2020")
    rows = []
    for model in MODELS:
        m = nta_measures(model).join(acs[list(FEATURES)], how="left")
        for measure in [c for c in m.columns if c not in FEATURES and not c.startswith("yes_")]:  # the yes rates: see `prompts`
            for feat in FEATURES:
                ok = m[[measure, feat]].dropna()
                rho, p = stats.spearmanr(ok[measure], ok[feat])
                rows.append({"model": model, "measure": measure, "feature": feat, "rho": float(rho), "p": float(p), "n_nta": len(ok)})
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df.p * len(df)).clip(upper=1.0)
    return df


LAYOUTS = {
    # name: (rows, columns, figure size in inches, words at each end, font size)
    "wide": (1, 3, (7.0, 2.55), N_WORDS, 7.2),
    "stacked": (3, 1, (3.35, 3.9), N_WORDS, 6.4),
}


def draw_words(ax, sub: pd.DataFrame, name: str, n_words: int, fs: float, renderer) -> None:
    """Draw the top words, a rule and the bottom words of 1 model, centred, in the order of the direction."""
    ink = {"up": S.PAL["teal"], "down": S.PAL["coral"]}
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = ax.get_window_extent(renderer)
    step = fs * 1.5 / 72.0 * ax.figure.dpi / box.height  # 1 line, in axes units
    gap = fs * 0.45 / 72.0 * ax.figure.dpi / box.width
    ax.text(0.5, 1.0, name, ha="center", va="top", fontsize=fs + 1.2)
    up = sub[(sub.end == "up") & (sub["rank"] <= n_words)].sort_values("rank")
    down = sub[(sub.end == "down") & (sub["rank"] <= n_words)].sort_values("rank", ascending=False)
    y = 1.0 - 1.6 * step
    for part, rows in (("up", up), ("down", down)):
        line, used = [], 0.0
        for r in rows.itertuples():
            t = ax.text(0, y, r.word, fontsize=fs, va="center", ha="left", color=ink[r.end])
            wd = t.get_window_extent(renderer).width / box.width
            if line and used + wd > 0.98:
                _centre(line, used - gap, y, gap)
                y -= step
                t.set_y(y)
                line, used = [], 0.0
            line.append((t, wd))
            used += wd + gap
        _centre(line, used - gap, y, gap)
        if part == "up":
            y -= step
            ax.plot([0.42, 0.58], [y, y], color=S.PAL["warmgrey"], lw=0.6)
            y -= step


def _centre(line: list, used: float, y: float, gap: float) -> None:
    x = (1.0 - used) / 2.0
    for t, wd in line:
        t.set_position((x, y))
        x += wd + gap


def figure(w: pd.DataFrame, layout: str) -> None:
    """Draw the words of the shared direction for each model, in 1 of the LAYOUTS."""
    import matplotlib.pyplot as plt

    S.apply_house_style()
    rows, cols, size, n_words, fs = LAYOUTS[layout]
    fig, axes = plt.subplots(rows, cols, figsize=size)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08, hspace=0.06)
    renderer = fig.canvas.get_renderer()
    for ax, (model, name) in zip(np.ravel(axes), MODELS.items()):
        draw_words(ax, w[w.model == model], name, n_words, fs, renderer)
    stem = "shared_direction_words" + ("" if layout == "wide" else f"_{layout}")
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)  # the same bytes on each run
    caption = {
        "plot-title": "Words of the shared aesthetic direction",
        "plot-caption": (
            f"The {n_words} words at each end of the shared aesthetic direction at the noun token ('place'), for 3 models. "
            "The direction is fit to the answers on 'cool', 'nice', 'fancy' and 'pretty' (F10.27). Its weights are "
            "z-scored at each layer and averaged over the layers, so no layer is chosen by hand. The order runs from "
            "the top word (first, teal) to the bottom word (last, coral); the rule separates the 2 ends. Words are lens "
            "vocabulary entries, shown as they are, including word pieces and other languages."),
        "plot-latex-label": "fig:shared-direction-words",
        "plot-tags": ["f10", "F10.28", "shared-direction", "words", layout],
        "provenance": {"script": "research/scripts/f10/hop_hierarchy.py", "version": __version__,
                       "written": datetime.now().isoformat(timespec="seconds")},
    }
    (FIG / f"{stem}.json").write_text(json.dumps(caption, indent=1))
    plt.close(fig)


PROMPT_ADJECTIVES = ("cool", "nice", "pretty", "fancy", "ordinary", "urban", "ugly")
NOUNS = ("place", "scene")
UNSURE_MODELS = ("gemma-4-31b",)  # the Qwen models give a yes or a no to almost every image
UNSURE_MIN = 0.05  # test P_unsure only where at least this share of the answers is `unable`
YES = ["lp_Yes", "lp_yes", "lp_ Yes", "lp_ yes"]
NO = ["lp_No", "lp_no", "lp_ No", "lp_ no"]
TABLE = ROOT / "research/notebooks/facct/tables/f10_prompt_ses.tex"


def answer_scores(model: str, noun: str, adjective: str) -> pd.DataFrame:
    """Give for each image E = P(yes) - P(no) and P_unsure = 1 - P(yes) - P(no).

    The log probabilities are over the full vocabulary, and each side pools the 4 yes or the 4 no
    forms of F10.26. E is the expected answer with yes = +1, unsure = 0 and no = -1. Thus an answer
    that the model does not give as yes or no sits near 0, between the yes and the no answers. D167
    leaves the margin of such an answer empty, thus the margin cannot order all of the images.
    """
    a = pd.read_parquet(HOP_LOGIT / answer_store(model, adjective, noun) / "answers.parquet")
    py, pn = np.exp(a[YES]).sum(1), np.exp(a[NO]).sum(1)
    out = pd.DataFrame({"image_idx": a.image_idx, "E": py - pn, "p_unsure": (1.0 - py - pn).clip(lower=0.0)})
    f = pd.read_parquet(DATA / store_name(model, adjective, noun) / "frame.parquet")[["image_idx", "nta2020", "outcome"]]
    out = f.merge(out, on="image_idx", validate="1:1")
    if len(out) != 591:
        raise ValueError(f"{model} {noun} {adjective}: {len(out)} images")
    return out


def prompt_ses() -> pd.DataFrame:
    """Give Spearman rho over NTAs of the NTA mean E (and, for Gemma, of P_unsure) of each prompt with each ACS feature."""
    acs = pd.read_parquet(ROOT / "data/f4_traces/nta_acs.parquet").set_index("nta2020")
    rows = []
    for model in MODELS:
        for noun in NOUNS:
            for adj in PROMPT_ADJECTIVES:
                s = answer_scores(model, noun, adj)
                nta = s.groupby("nta2020")[["E", "p_unsure"]].mean().join(acs[list(FEATURES)], how="left")
                share = float((s.outcome == "unable").mean())
                unsure = model in UNSURE_MODELS and share >= UNSURE_MIN
                for measure in ("E", "p_unsure") if unsure else ("E",):
                    for feat in FEATURES:
                        ok = nta[[measure, feat]].dropna()
                        rho, p = stats.spearmanr(ok[measure], ok[feat])
                        rows.append({"model": model, "noun": noun, "adjective": adj, "measure": measure, "feature": feat,
                                     "rho": float(rho), "p": float(p), "n_nta": len(ok),
                                     "share_unable": share})
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = (df.p * len(df)).clip(upper=1.0)
    return df


def _cell(r) -> str:
    v = f"{r.rho:+.2f}".replace("0.", ".")
    return f"${v}^{{*}}$" if r.p_bonferroni < 0.05 else f"${v}\\phantom{{^{{*}}}}$"


def prompt_table(df: pd.DataFrame) -> None:
    """Write the booktabs table: 1 row for each prompt (E), and 1 sub-row for the Gemma P_unsure."""
    feats = list(FEATURES)
    short = {"pct_white_nh": r"\%W", "pct_black_nh": r"\%B", "pct_bachelors": r"\%BA", "median_household_income": "Inc."}
    head = " & ".join(["Prompt"] + [r"\multicolumn{4}{c}{" + name + "}" for name in MODELS.values()])
    rules = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * k, 5 + 4 * k) for k in range(len(MODELS)))
    sub = " & ".join([""] + [short[f] for f in feats] * len(MODELS))

    def get(model: str, noun: str, adj: str, measure: str, feat: str):
        return df[(df.model == model) & (df.noun == noun) & (df.adjective == adj)
                  & (df.measure == measure) & (df.feature == feat)].iloc[0]

    body = []
    for noun in NOUNS:
        for adj in PROMPT_ADJECTIVES:
            body.append(" & ".join([f"{noun} {adj}"] + [_cell(get(m, noun, adj, "E", f)) for m in MODELS for f in feats]) + r" \\")
            if ((df.model == UNSURE_MODELS[0]) & (df.noun == noun) & (df.adjective == adj) & (df.measure == "p_unsure")).any():
                share = get(UNSURE_MODELS[0], noun, adj, "E", feats[0]).share_unable
                unsure = [_cell(get(m, noun, adj, "p_unsure", f)) if m in UNSURE_MODELS else "" for m in MODELS for f in feats]
                body.append(" & ".join([r"\quad\textit{unsure} (%d\%%)" % round(100 * share)] + unsure) + r" \\")
        if noun == "place":
            body.append(r"\midrule")
    n, k = len(df), int((df.p_bonferroni < 0.05).sum())
    note = (r"Rows: $E=P(\text{yes})-P(\text{no})$, the expected answer with yes $=+1$, unsure $=0$, no $=-1$, from the "
            r"probabilities of the yes and no tokens over the full vocabulary. \textit{unsure}: "
            r"$P_\text{unsure}=1-P(\text{yes})-P(\text{no})$, for Gemma-4-31B on the prompts where it answers neither yes nor no "
            r"to at least 5\% of images (the share in parentheses); the Qwen models answer yes or no to almost every image. NTA means over 197 NTAs. \%W, \%B: share of White and of Black residents; \%BA: share of adults "
            r"with a bachelor's degree; Inc.: median household income. "
            f"$^*$: Bonferroni $p<.05$ over {n} tests ({k} pass).")
    tex = "\n".join([
        f"% Generated by research/scripts/f10/hop_hierarchy.py {__version__}. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Answers and the neighborhood.} Spearman $\rho$ over NTAs between the answer to each one-word "
        r"prompt and four ACS features of the NTA, for three models.}",
        r"\label{tab:prompt-ses}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "rrrr" * len(MODELS) + "}",
        r"\toprule",
        head + r" \\",
        rules,
        sub + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\par\smallskip",
        r"{\footnotesize " + note + "}",
        r"\end{table}",
        ""])
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(tex)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = words()
    w.to_csv(OUT / "shared_direction_words.csv", index=False)
    for layout in LAYOUTS:
        figure(w, layout)
    df = ses()
    df.to_csv(OUT / "ses.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"version": __version__, "n_tests": len(df),
                                                  "n_pass": int((df.p_bonferroni < 0.05).sum())}, indent=1))
    tp = prompt_ses()
    tp.to_csv(OUT / "ses_prompts.csv", index=False)
    prompt_table(tp)
    pd.set_option("display.width", 200)
    print(tp.pivot_table(index=["noun", "adjective"], columns=["model", "measure", "feature"], values="rho").round(2).to_string())
    print("prompt tests", len(tp), "pass", int((tp.p_bonferroni < 0.05).sum()))
    print(df.pivot_table(index=["model", "measure"], columns="feature", values="rho").round(2).to_string())
    print(df.pivot_table(index=["model", "measure"], columns="feature", values="p_bonferroni").round(3).to_string())


if __name__ == "__main__":
    main()
