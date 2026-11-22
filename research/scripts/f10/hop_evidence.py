#!/usr/bin/env python
"""F10.27b: a readout axis of the visual evidence, beyond the verdict. CPU only.

Question: does the J++ Lens readout of the one-word `cool` prompt hold the valence of the visual
evidence that the same model cites for the image in a separate run with thinking on, beyond the answer?

| Item | Rule |
|------|------|
| Target | For each NTA: the mean net valence of the `visual_evidence` spans of the `cool` traces of the same model. A span is good (+1), bad (-1) or neutral (0). Mean over the spans of a text, then over the texts of an image, then over the images of an NTA |
| Readouts | The NTA means of F10.26 stage 1 (`data/f10/prompts/{model}/nta_{position}.npy`), at the noun, the adjective, `?` and the last token, all layers |
| Evidence vector | For each word: r over NTAs of the readout with the target. Fit on 4 of 5 NTA folds, score the 5th fold (as `hop_shared.py`) |
| Controls | Partial r of the out-of-fold score with the target, given the one-word answer margin, and given the margin and the trace verdict together |
| Test | t test of the partial r (df = n - 2 - number of controls), Bonferroni over all cells of 1 model |
| Shared direction | Also: r of the noun-token shared aesthetic direction of `hop_shared.py` with the target |

The traces exist for Gemma-4-31B and Qwen3.5-9B only (`data/f10/hop_reason/`).

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_evidence.py --model gemma-4-31b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from hop_shared import N_FOLDS, SEED, corr_cols, unit  # noqa: E402

__version__ = "1.0.0"

DATA = ROOT / "data/f10/prompts"
REASON = ROOT / "data/f10/hop_reason"
FIG = ROOT / "research/figures/f10/shared"
POSITIONS = ["place", "after_place", "after_cool", "final"]
VALENCE = {"good": 1.0, "bad": -1.0, "neutral": 0.0}
VERDICT = {"yes": 1.0, "no": -1.0, "mixed": 0.0, "graded": 0.0}
N_WORDS = 20


def r(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def partial(a: np.ndarray, b: np.ndarray, *cs: np.ndarray) -> tuple[float, float]:
    """Give the r of a and b given the covariates cs (least squares with an intercept), and its t-test p."""
    m = np.isfinite(a) & np.isfinite(b)
    for c in cs:
        m &= np.isfinite(c)
    x = np.column_stack([np.ones(m.sum())] + [c[m] for c in cs])
    res = lambda v: v[m] - x @ np.linalg.lstsq(x, v[m], rcond=None)[0]  # noqa: E731
    rr = float(np.corrcoef(res(a), res(b))[0, 1])
    df = int(m.sum()) - 2 - len(cs)
    t = rr * np.sqrt(df / (1 - rr ** 2))
    return rr, float(2 * stats.t.sf(abs(t), df))


def nta_targets(model: str, frame: pd.DataFrame, order: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Give the NTA evidence valence, the NTA answer margin and the NTA trace verdict."""
    sp = pd.read_parquet(REASON / "spans.parquet", columns=["model", "doc_id", "image_idx", "extraction_class", "attributes_json"])
    sp = sp[(sp.model == model) & (sp.extraction_class == "visual_evidence")].copy()
    if sp.empty:
        raise SystemExit(f"no traces for {model}")
    sp["v"] = sp.attributes_json.map(lambda a: VALENCE.get(json.loads(a).get("valence")) if a else None)
    per_image = sp.dropna(subset=["v"]).groupby(["doc_id", "image_idx"]).v.mean().groupby("image_idx").mean()
    vd = pd.read_parquet(REASON / "verdicts.parquet", columns=["model", "image_idx", "verdict"])
    vd = vd[vd.model == model]
    trace = vd.assign(v=vd.verdict.map(VERDICT)).groupby("image_idx").v.mean()
    f = frame.assign(evidence=frame.image_idx.map(per_image), trace_verdict=frame.image_idx.map(trace))
    g = f.groupby("nta2020")
    y, margin, tv = (g[c].mean().reindex(order).to_numpy(float) for c in ("evidence", "margin", "trace_verdict"))
    info = {"evidence_spans": int(len(sp)), "valence_counts": sp.attributes_json.map(lambda a: json.loads(a).get("valence") if a else None).value_counts().to_dict(),
            "images_with_evidence": int(f.evidence.notna().sum()), "r_evidence_margin": r(y, margin),
            "r_trace_verdict_evidence": r(tv, y), "r_trace_verdict_margin": r(tv, margin)}
    return y, margin, tv, info


def fit(model: str) -> None:
    store = DATA / model
    frame = pd.read_parquet(store / "frame.parquet")
    order = pd.factorize(frame["nta2020"])[1].tolist()
    y, margin, tv, info = nta_targets(model, frame, order)
    meta = json.loads((store / "meta.json").read_text())
    shared = np.load(FIG / f"{model}_place" / "direction.npz")
    vocab = shared["vocab"]
    fold = np.random.default_rng(SEED).permutation(len(order)) % N_FOLDS
    rows, words = [], []
    for pos in POSITIONS:
        x = np.load(store / f"nta_{pos}.npy")
        for b, layer in enumerate(meta["layers"]):
            m = x[:, b, :].astype(np.float64)
            m -= m.mean(0, keepdims=True)
            score = np.full(len(order), np.nan)
            for f in range(N_FOLDS):
                tr, te = fold != f, fold == f
                score[te] = m[te] @ unit(corr_cols(m[tr], y[tr]))
            p1, pv1 = partial(score, y, margin)
            p2, pv2 = partial(score, y, margin, tv)
            row = {"model": model, "position": pos, "layer": layer, "r_evidence": r(score, y), "r_with_margin": r(score, margin),
                   "partial_given_margin": p1, "p_given_margin": pv1, "partial_given_both": p2, "p_given_both": pv2}
            if pos == "place":
                s = m @ shared["direction"][b].astype(np.float64)
                row["r_shared_with_evidence"] = r(s, y)
                row["partial_shared_given_margin"] = partial(s, y, margin)[0]
            rows.append(row)
            g = corr_cols(m, y)
            words.append({"position": pos, "layer": layer, "up": " ".join(vocab[np.argsort(-g)[:N_WORDS]]),
                          "down": " ".join(vocab[np.argsort(g)[:N_WORDS]])})
        print(f"[f10.27b] {model} {pos}", flush=True)
    df = pd.DataFrame(rows)
    df["p_both_bonferroni"] = (df.p_given_both * len(df)).clip(upper=1.0)
    out = FIG / f"evidence_{model}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "evidence_axis.csv", index=False)
    pd.DataFrame(words).to_csv(out / "evidence_words.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"version": __version__, "model": model, "n_nta": len(order), **info}, indent=1, default=str))
    pd.set_option("display.width", 220)
    print(json.dumps(info, indent=1, default=str))
    print(df.pivot_table(index="layer", columns="position", values="partial_given_both").round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=["gemma-4-31b", "qwen3.5-9b"])
    fit(ap.parse_args().model)


if __name__ == "__main__":
    main()
