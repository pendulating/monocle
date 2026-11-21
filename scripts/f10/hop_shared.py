#!/usr/bin/env python
"""F10.27: 1 shared judgment direction over the aesthetic prompts. CPU only.

Input: the NTA mean readouts of F10.26 stage 1 (`data/f10/prompts/{store}/nta_{position}.npy`).
Only stores that are complete (`meta.json` "complete": true) are used. A missing store is reported.

Method, for each model, position and layer:

| Step | Rule |
|------|------|
| Judgment vector | For each word: r of the NTA readout with the NTA answer margin |
| Fit set | `cool`, `nice`, `fancy`, `pretty`, all with `place`. All have the sign + |
| Shared direction | The mean of the unit judgment vectors of the fit set |
| Folds | 5 folds of NTAs. A vector is fit on 4 folds and scores the 5th fold |
| Leave 1 prompt out | A prompt in the fit set is scored by the direction of the other 3 |
| Held-out prompts | The 3 other `place` prompts and the 7 `scene` prompts. Scored by the direction of all 4 |
| Sense removal | In the `cool` readouts only, and not at the noun token, regress each word on the NTA sense score (social words minus temperature words). At the noun the model has not read `cool`, and the 4 `place` readouts are the same |
| Polarity removal | Variant: in all readouts, regress each word on a score of fixed yes words minus no words |
| Test layer | The layer with the best mean leave-1-out r over the fit set, in the main variant. The held-out prompts do not choose it |
| Test | r over NTAs, t test of r, Bonferroni over the held-out prompts at the test layer (p times the number of tests) |

Variants: `base` (no removal), `main` (sense removal in `cool`), `main_polarity` (main, and polarity removal).
Expected signs: + for `cool`, `nice`, `fancy`, `pretty`; - for `ugly`, `ordinary`. `urban` has no expected sign.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py fit --model gemma-4-31b
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py fit --model gemma-4-31b --position place
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_shared.py figure
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from hop_prompts import ADJECTIVES, MODELS, NAMES, NOUNS, store_name
from hop_prompts import frame as build_frame  # noqa: E402
from hop_sense import SOCIAL, TEMPERATURE  # noqa: E402

__version__ = "1.2.0"

DATA = ROOT / "data/f10/prompts"
FIG = ROOT / "figures/f10/shared"
FIT = [("cool", "place"), ("nice", "place"), ("fancy", "place"), ("pretty", "place")]
EXPECTED = {"cool": 1, "nice": 1, "fancy": 1, "pretty": 1, "ugly": -1, "ordinary": -1, "urban": 0}
# A fixed list of answer words. Their score is yes words minus no words. Words not in the vocab are dropped.
YES_WORDS = ["yes", "indeed", "certainly", "surely", "definitely", "absolutely", "affirmative", "correct", "true", "oui", "ja"]
NO_WORDS = ["no", "not", "never", "neither", "nor", "nowhere", "none", "nothing", "nobody", "denied", "deny", "refusal",
            "lack", "without", "nunca", "kein", "nicht", "nada", "nadie", "nulla", "rien", "tampoco", "negative"]
N_FOLDS = 5
SEED = 0
N_WORDS = 30


def tag(adjective: str, noun: str) -> str:
    return f"{noun}-{adjective}"


def complete(model: str, adjective: str, noun: str) -> bool:
    meta = DATA / store_name(model, adjective, noun) / "meta.json"
    return meta.exists() and bool(json.loads(meta.read_text()).get("complete", False))


def partial_ok(model: str, adjective: str, noun: str, position: str) -> bool:
    """Accept a store that is not complete when its readout file exists and its noun readout equals that of `cool`.

    The summary saves the files only after the pass over all shards. Attention is causal, thus the noun readout
    of each store with the same noun is the same. A gap of 0 shows that the pass covered all images.
    """
    d, ref = DATA / store_name(model, adjective, noun), DATA / store_name(model, "cool", noun)
    if not ((d / f"nta_{position}.npy").exists() and (d / "nta_place.npy").exists() and complete(model, "cool", noun)):
        return False
    return float(np.abs(np.load(d / "nta_place.npy") - np.load(ref / "nta_place.npy")).max()) == 0.0


def load(model: str, adjective: str, noun: str, position: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    d = DATA / store_name(model, adjective, noun)
    x = np.load(d / f"nta_{position}.npy")
    if (d / "frame.parquet").exists():
        return x, pd.read_parquet(d / "frame.parquet"), json.loads((d / "meta.json").read_text())
    ref = json.loads((DATA / store_name(model, "cool", noun) / "meta.json").read_text())
    return x, build_frame(model, adjective, noun), {"layers": ref["layers"], "vocab": x.shape[-1]}


def corr_cols(m: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Give r of each column of m with y. Rows with no finite y are dropped."""
    ok = np.isfinite(y)
    mc = m[ok] - m[ok].mean(0)
    yc = y[ok] - y[ok].mean()
    return (mc * yc[:, None]).sum(0) / (np.sqrt((mc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12)


def r_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    res = stats.pearsonr(a[ok], b[ok])
    return float(res.statistic), float(res.pvalue), int(ok.sum())


def regress_out(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Remove from each column of m its least-squares fit on s. m is centred."""
    s = (s - s.mean()) / s.std()
    return m - np.outer(s, s @ m / len(s))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def fit(model: str, position: str, allow_partial: bool = False) -> None:
    prompts = [(a, n) for n in NOUNS for a in ADJECTIVES]
    have = [p for p in prompts if complete(model, *p)]
    partial = [p for p in prompts if p not in have and allow_partial and partial_ok(model, *p, position)]
    have = [p for p in prompts if p in have or p in partial]
    missing = [p for p in prompts if p not in have]
    if any(p not in have for p in FIT):
        raise SystemExit(f"fit set not complete for {model}: {[p for p in FIT if p not in have]}")
    data = {p: load(model, *p, position) for p in have}
    order = pd.factorize(data[FIT[0]][1]["nta2020"])[1].tolist()
    for p, (_, fr, meta) in data.items():
        if pd.factorize(fr["nta2020"])[1].tolist() != order:
            raise ValueError(f"NTA order differs in {p}")
    layers = data[FIT[0]][2]["layers"]
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{store_name(model, 'cool', 'place')}/vocab.parquet")["token"]
    vocab = vocab.str.strip().to_numpy().astype(str)
    if any(m["vocab"] != vocab.size for _, _, m in data.values()):
        raise ValueError("vocab size differs between stores")
    low = pd.Series(vocab).str.lower()
    hot = np.flatnonzero(low.isin(TEMPERATURE).to_numpy())
    soc = np.flatnonzero(low.isin(SOCIAL).to_numpy())
    yes = np.flatnonzero(pd.Series(vocab).isin(YES_WORDS).to_numpy())
    no = np.flatnonzero(pd.Series(vocab).isin(NO_WORDS).to_numpy())
    margins = {p: fr.groupby("nta2020")["margin"].mean().reindex(order).to_numpy(float) for p, (_, fr, _) in data.items()}
    fold = np.random.default_rng(SEED).permutation(len(order)) % N_FOLDS
    others = [p for p in have if p not in FIT]

    out = FIG / f"{model}_{position}"
    out.mkdir(parents=True, exist_ok=True)
    rows, cos_rows, sense_rows, dirs = [], [], [], {}
    for b, layer in enumerate(layers):
        base = {}
        for p in have:
            m = data[p][0][:, b, :].astype(np.float64)
            base[p] = m - m.mean(0, keepdims=True)
        sense = {p: base[p][:, soc].mean(1) - base[p][:, hot].mean(1) for p in have}
        polarity = {p: base[p][:, yes].mean(1) - base[p][:, no].mean(1) for p in have}
        variants = {"base": base,
                    "main": {p: regress_out(base[p], sense[p]) if p[0] == "cool" and position != "place" else base[p] for p in have}}
        variants["main_polarity"] = {p: regress_out(variants["main"][p], polarity[p]) for p in have}
        for name, ms in variants.items():
            score = {p: np.full(len(order), np.nan) for p in have}
            for f in range(N_FOLDS):
                tr, te = fold != f, fold == f
                g = {p: unit(corr_cols(ms[p][tr], margins[p][tr])) for p in FIT}
                for p in have:
                    use = [q for q in FIT if q != p]
                    score[p][te] = ms[p][te] @ np.mean([g[q] for q in use], 0)
            for p in have:
                rr, pv, n = r_test(score[p], margins[p])
                rows.append({"model": model, "position": position, "layer": layer, "variant": name, "prompt": tag(*p),
                             "adjective": p[0], "noun": p[1], "role": "fit (leave 1 out)" if p in FIT else "held out",
                             "expected_sign": EXPECTED[p[0]], "r": rr, "p": pv, "n_nta": n,
                             "r_with_sense": r_test(score[p], sense[p])[0], "r_with_polarity": r_test(score[p], polarity[p])[0]})
        # The full-data direction of the main variant: its words and the cosines of the judgment vectors.
        ms = variants["main"]
        g = {p: unit(corr_cols(ms[p], margins[p])) for p in have}
        d = np.mean([g[p] for p in FIT], 0)
        dirs[layer] = d.astype(np.float32)
        for p in have:
            for q in have:
                cos_rows.append({"layer": layer, "a": tag(*p), "b": tag(*q), "cosine": float(g[p] @ g[q])})
        sense_rows.append({"layer": layer, "n_yes_words": len(yes), "n_no_words": len(no),
                           **{f"r_margin_sense_{tag(*p)}": r_test(sense[p], margins[p])[0] for p in have}})
        print(f"[f10.27] {model} {position} L{layer}", flush=True)

    df = pd.DataFrame(rows)
    fitrows = df[(df.variant == "main") & (df.role != "held out")]
    test_layer = int(fitrows.groupby("layer")["r"].mean().idxmax())
    held = df[(df.layer == test_layer) & (df.role == "held out")].copy()
    for name in held.variant.unique():
        sel = held.variant == name
        held.loc[sel, "p_bonferroni"] = (held.loc[sel, "p"] * sel.sum()).clip(upper=1.0)
    df = df.merge(held[["layer", "variant", "prompt", "p_bonferroni"]], on=["layer", "variant", "prompt"], how="left")
    df.to_csv(out / "heldout.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(out / "cosine.csv", index=False)
    pd.DataFrame(sense_rows).to_csv(out / "sense_margin.csv", index=False)
    np.savez(out / "direction.npz", layers=np.array(layers), direction=np.stack([dirs[l] for l in layers]), vocab=vocab)

    words = []
    for layer in layers:
        d = dirs[layer]
        for end, idx in (("up", np.argsort(-d)[:N_WORDS]), ("down", np.argsort(d)[:N_WORDS])):
            words.append({"layer": layer, "end": end, "words": " ".join(vocab[idx])})
    pd.DataFrame(words).to_csv(out / "words.csv", index=False)
    summary = {"version": __version__, "model": model, "position": position, "test_layer": test_layer,
               "fit": [tag(*p) for p in FIT], "held_out": [tag(*p) for p in others], "missing": [tag(*p) for p in missing],
               "partial": [tag(*p) for p in partial],
               "yes_words": vocab[yes].tolist(), "no_words": vocab[no].tolist(), "n_nta": len(order)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 250)
    print(json.dumps(summary, indent=1))
    t = df[df.layer == test_layer].pivot_table(index=["role", "prompt", "expected_sign"], columns="variant", values="r").round(2)
    print(f"r at the test layer L{test_layer}\n{t.to_string()}")
    print(df[(df.layer == test_layer) & (df.variant == "main") & (df.role == "held out")][["prompt", "r", "p", "p_bonferroni", "n_nta"]].round(4).to_string())


ORDER = ["place-cool", "place-nice", "place-fancy", "place-pretty", "scene-cool", "scene-nice", "scene-fancy", "scene-pretty",
         "place-ugly", "scene-ugly", "place-ordinary", "scene-ordinary", "place-urban", "scene-urban"]
MARKS = [("place", "main_polarity", "o", "full", "noun token, yes/no score removed"),
         ("after_place", "main", "s", "none", "adjective token"),
         ("after_place", "main_polarity", "s", "full", "adjective token, yes/no score removed")]


def figure() -> None:
    """Draw a dot chart: r of each prompt at the test layer, for the noun and the adjective token, 1 panel for each model."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    runs = [m for m in MODELS if all((FIG / f"{m}_{pos}" / "heldout.csv").exists() for pos in ("place", "after_place"))]
    if not runs:
        raise SystemExit("no fit output for both positions")
    colors = {1: "#5674B3", -1: "#E57264", 0: "#9C948A"}
    height = 1.9 * len(runs) + 0.9
    fig, axes = plt.subplots(len(runs), 1, figsize=(6.8, height), sharex=True, squeeze=False)
    caption_layers = {}
    for ax, m in zip(axes[:, 0], runs):
        layers = {}
        for k, (pos, variant, marker, fill, _) in enumerate(MARKS):
            d = FIG / f"{m}_{pos}"
            test = json.loads((d / "summary.json").read_text())["test_layer"]
            layers[pos] = test
            df = pd.read_csv(d / "heldout.csv").query("variant == @variant and layer == @test").set_index("prompt")
            for i, prompt in enumerate(ORDER):
                if prompt not in df.index:
                    continue
                row = df.loc[prompt]
                c = colors[int(row["expected_sign"])]
                ax.plot(i + (k - 1) * 0.22, row["r"], marker=marker, ms=4.5, ls="none", color=c,
                        markerfacecolor=c if fill == "full" else "white", markeredgewidth=0.9, clip_on=False)
        caption_layers[m] = layers
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(3.5, color="black", lw=0.4, ls=":")
        ax.set_ylim(-1.05, 1.08)
        ax.set_ylabel("r", fontsize=8)
        ax.set_title(f"{NAMES[m]}  (noun L{layers['place']}, adjective L{layers['after_place']})", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax = axes[-1, 0]
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([p.replace("-", " ") for p in ORDER], rotation=45, ha="right", fontsize=7)
    handles = [Line2D([], [], marker=mk, ls="none", color="black", markerfacecolor="black" if f == "full" else "white", label=lab)
               for _, _, mk, f, lab in MARKS]
    handles += [Line2D([], [], marker="o", ls="none", color=colors[s], label=lab)
                for s, lab in ((1, "expected +"), (-1, "expected -"), (0, "no expected sign"))]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.42 / height))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"shared_direction.{ext}", dpi=200)
    (FIG / "shared_direction.json").write_text(json.dumps({
        "plot-title": "A shared aesthetic direction before the adjective, and an answer forecast after it",
        "plot-caption": ("Out-of-fold r over 197 NTAs between the answer margin of each prompt and the J++ Lens readout projected on the "
                         "shared direction: the mean judgment vector of cool, nice, fancy and pretty with the noun 'place' (left of the dotted "
                         "line; each is scored by the other 3). The other prompts are held out. Circles: readout at the noun token, before the "
                         "adjective. Squares: readout at the adjective token. Filled markers: a fixed yes-minus-no word score is regressed out. "
                         "The temperature-to-social score is regressed out of the cool readouts. Each test layer is chosen on the fit prompts only. "
                         "NTAs with no yes or no answer are left out (Gemma-4-31B ugly: 19 NTAs with place, 66 with scene)."),
        "plot-latex-label": "fig:shared-direction", "plot-tags": ["f10", "j++ lens", "shared direction"],
        "test_layers": caption_layers}, indent=1))
    print(f"wrote {FIG / 'shared_direction.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", choices=MODELS, required=True)
    f.add_argument("--position", default="after_place", choices=["place", "after_place", "after_cool", "final"])
    f.add_argument("--allow-partial", action="store_true", help="use a store that is not complete if its noun readout equals that of cool")
    sub.add_parser("figure")
    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.model, args.position, args.allow_partial)
    else:
        figure()


if __name__ == "__main__":
    main()
