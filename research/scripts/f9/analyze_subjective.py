#!/usr/bin/env python
"""F9: the subjective prompt against the F4 prompt, on the same images (H9, D139, D142).

H9: with a subjective prompt, the net valence of the answer changes more
along the NTA socioeconomic features than with the F4 prompt.

Both sides come from 1 schema (D142, Q4 option C): the F9 answers and the
F4 answers of the same images, under `urban_cues_single_subj_v1` (restaurant)
and `urban_cues_street_subj_v1` (street), with the retry pass merged. The F4
x2 spans of record (`data/f4_traces/x2/`) do not enter.

| Item | Rule |
|------|------|
| Text outcome | the net valence of method B (F8.1): (good - bad) / `visual_evidence` spans with a valence |
| Unit | 1 image, for 1 (model, case, field) |
| `d` | net valence under F9 minus net valence under F4, on the same image. An image enters when both texts have 1 such span or more |
| Test | OLS of `d` on 1 z-scored ACS feature, SEs clustered by NTA (CR1), t test with G - 1 df (`research/scripts/f8/method_b.py`) |
| Primary | the `response` field: 6 models x 2 cases x 5 features = 60 tests, Bonferroni |
| H9 for 1 row | the slope of `d` is significant and has the sign of the F4 gradient (model 0 on the F4 outcome, same images) |
| Secondary, no correction | the `reasoning` field (4 models, 40 rows); model 1 of method B on `d` |

The verdict of 1 text comes from its quotable `verdict` spans in text order
(`f9-extraction-design.md`, section 2.3; `text_verdict`).

Descriptions (no test): the span mix of the 2 prompts; the verdict shares by
model and by NTA; the verdict rate on the F4 texts; the NTA Spearman of the
net valence under each prompt; the response-against-reasoning verdict
table; the cut rate and the tokens; the `response_from_trace` row; the B10
coverage of each merged file.

Outputs go to `research/figures/f9/` (CSV, JSON, and 3 figures with a caption JSON).
See `research/docs/f9-subjective-prompt.md`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f9/analyze_subjective.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__version__ = "1.1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
from research.scripts.f8 import method_b as MB  # noqa: E402

X2 = REPO_ROOT / "data" / "f4_traces" / "subj" / "x2"
INPUT = REPO_ROOT / "data" / "f4_traces" / "extract_input"
FROM_TRACE = REPO_ROOT / "data" / "f4_traces" / "subj" / "extract_input" / "f4_restaurant_subj.parquet"
OUT_DIR = REPO_ROOT / "research" / "figures" / "f9"

CASES = ("restaurant", "street")
FIELDS = ("response", "reasoning")
PROMPTS = ("f4", "f9")
# case -> prompt -> the merged files (arm dir, input case) under X2
FILES: Dict[str, Dict[str, tuple]] = {
    "restaurant": {"f9": (("cues", "restaurant_subj_x2"),),
                   "f4": (("cues", "restaurant_x2"), ("cues", "restaurant_b2x2"))},
    "street": {"f9": (("cues_street", "street_subj_x2"),),
               "f4": (("cues_street", "street_x2"), ("cues_street", "street_b2x2"))},
}
SCHEMA_VERSIONS = frozenset({"single-subj-v1", "single-subj-v1r", "street-subj-v1", "street-subj-v1r"})
VERDICT_LABELS = ("yes", "no", "mixed", "graded", "not_applicable")
TEXT_LABELS = VERDICT_LABELS + ("none",)
CLASSES = ("visual_evidence", "inference", "uncertainty", "image_artifact", "person_reference",
           "photo_advice", "verdict")
MIN_NTA_IMAGES = 20
# The design checks of the rule (section 5, step 12): (model, case, image_idx, field, expected label).
DESIGN_CHECKS = (
    ("gemma-4-31b", "restaurant", 7880, "response", "mixed"),
    ("qwen3.8-27b", "restaurant", 8244, "response", "graded"),
    ("llama-3.2-11b", "restaurant", 7530, "response", "no"),
)
LABEL_COLORS = {"yes": "teal", "no": "coral", "mixed": "amber", "graded": "cream",
                "not_applicable": "slate", "none": "warmgrey"}


def _f4():
    return MB._f4()


def _f8():
    from research.notebooks.facct import _f8  # noqa: E402

    return _f8


# ---------------------------------------------------------------------------
# The verdict rule (section 2.3)
# ---------------------------------------------------------------------------


def text_verdict(labels: Sequence[Optional[str]]) -> str:
    """Give the verdict of 1 text from the labels of its verdict spans, in text order.

    | Spans | Label |
    |-------|-------|
    | no span with a declared label | `none` |
    | only `not_applicable` | `not_applicable` |
    | the last other span is `mixed` or `graded` | that label |
    | the last other span is `yes` or `no`, and an earlier span is the other one | `mixed` |
    | the last other span is `yes` or `no`, and no span is the other one | that label |

    A label outside `VERDICT_LABELS` (a vocabulary `other`) does not count.
    """
    found = [x for x in labels if x in VERDICT_LABELS]
    if not found:
        return "none"
    rest = [x for x in found if x != "not_applicable"]
    if not rest:
        return "not_applicable"
    last = rest[-1]
    if last in ("mixed", "graded"):
        return last
    other = "no" if last == "yes" else "yes"
    return "mixed" if other in rest else last


def rule_checks() -> None:
    """Stop when the rule gives a wrong label on the cases of section 2.3."""
    cases = [
        ([], "none"),
        (["other"], "none"),
        (["not_applicable"], "not_applicable"),
        (["not_applicable", "yes"], "yes"),
        (["mixed", "no", "yes"], "mixed"),             # the gemma conditional answer
        (["no", "mixed"], "mixed"),                    # gemma trace: Draft 1 no, Draft 3 depends
        (["no", "no"], "no"),                          # the qwen3.8-27b trace with 2 no drafts
        (["yes", "no", "graded", "graded"], "graded"),  # qwen3.8-27b street trace 7895
        (["mixed", "yes", "graded"], "graded"),        # qwen3.8-27b restaurant answer 8244
        (["yes", "no"], "mixed"),                      # a trace that changes its mind (Q3)
        (["graded", "yes"], "yes"),
        (["mixed", "no"], "no"),
    ]
    for labels, want in cases:
        got = text_verdict(labels)
        if got != want:
            raise AssertionError(f"text_verdict({labels}) = {got}, expected {want}")


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


def _input_ids(name: str) -> set:
    path = INPUT / f"f4_{name}.parquet"
    return set(pd.read_parquet(path, columns=["doc_id"]).doc_id) if path.exists() else set()


def load(x2: Path = X2):
    """Read the merged spans of both prompts and both cases: (`raw`, `coverage`).

    `raw` holds every row of every text that is not a copy suspect, with the
    columns `prompt` (f4 or f9) and `case` (restaurant or street).
    `coverage` gives the B10 numbers of each merged file.
    """
    frames, rows = [], []
    for case, by_prompt in FILES.items():
        for prompt, files in by_prompt.items():
            for arm, name in files:
                path = x2 / arm / f"f4_{name}_extractions.parquet"
                if not path.exists():
                    raise FileNotFoundError(f"{path}: run subj_pass2.txt (x2_merge_all.sh with F4X2_PLAN_ARMS=subj)")
                d = pd.read_parquet(path)
                versions = set(d.schema_version.dropna().unique())
                if not versions <= SCHEMA_VERSIONS:
                    raise ValueError(f"{path}: schema versions {sorted(versions)}; expected {sorted(SCHEMA_VERSIONS)}")
                want = _input_ids(name)
                have = set(d.doc_id)
                real = d[d.extraction_class.notna() & (d.extraction_class != "none")]
                per = d.drop_duplicates("doc_id")
                rows.append({
                    "case": case, "prompt": prompt, "file": name, "input_texts": len(want),
                    "texts": len(have), "missing": len(want - have), "extra": len(have - want),
                    "spans": len(real), "quotable": float(real.is_quotable.fillna(False).mean()),
                    "copy_suspects": int(per.copy_suspect.sum()),
                    "copy_suspect_share": float(per.copy_suspect.mean()),
                    "retried": int(per.retried.sum()),
                    "schema_versions": "+".join(sorted(versions)),
                })
                frames.append(d[~d.copy_suspect.astype(bool)].assign(case=case, prompt=prompt, source_file=name))
    raw = pd.concat(frames, ignore_index=True)
    return raw, pd.DataFrame(rows)


def spans_of(raw: pd.DataFrame) -> pd.DataFrame:
    """Give the spans that a count may use (`_f4.spans`: quotable, no duplicate, attributes opened)."""
    return _f4().spans(raw)


def _texts(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.drop_duplicates("doc_id")[["doc_id", "prompt", "model", "case", "field", "image_idx",
                                          "cut", "completion_tokens"]].reset_index(drop=True)


def text_valence(raw: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Give the net valence of each text: `good`, `bad`, `neutral`, `spans`, `outcome` (NaN with no span)."""
    vals = list(_f4().VALENCES)
    ve = sp[(sp.extraction_class == "visual_evidence") & sp.valence.isin(vals)]
    n = ve.groupby(["doc_id", "valence"]).size().unstack("valence", fill_value=0).reindex(columns=vals, fill_value=0)
    t = _texts(raw).merge(n, left_on="doc_id", right_index=True, how="left").fillna({v: 0 for v in vals})
    t["spans"] = t[vals].sum(axis=1)
    t["outcome"] = np.where(t.spans > 0, (t.good - t.bad) / t.spans.where(t.spans > 0), np.nan)
    return t


def text_verdicts(raw: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Give the verdict of each text (`label`), its verdict spans (`verdict_spans`), and the `other` spans."""
    v = sp[sp.extraction_class == "verdict"].sort_values(["doc_id", "char_start", "extraction_index"])
    lab = v.groupby("doc_id").label.agg(lambda s: text_verdict(list(s))).rename("label")
    n = v.groupby("doc_id").size().rename("verdict_spans")
    other = v[~v.label.isin(VERDICT_LABELS)].groupby("doc_id").size().rename("verdict_other")
    t = _texts(raw).merge(lab, left_on="doc_id", right_index=True, how="left")
    t = t.merge(n, left_on="doc_id", right_index=True, how="left").merge(
        other, left_on="doc_id", right_index=True, how="left")
    t["label"] = t.label.fillna("none")
    t[["verdict_spans", "verdict_other"]] = t[["verdict_spans", "verdict_other"]].fillna(0).astype(int)
    return t


def paired(val: pd.DataFrame) -> pd.DataFrame:
    """Join the F9 text and the F4 text of 1 (model, case, field, image): `outcome_f9`, `outcome_f4`, `d`."""
    keys = ["model", "case", "field", "image_idx"]
    cols = keys + ["doc_id", "outcome", "spans", "good", "bad"]
    a = val[val.prompt == "f9"][cols]
    b = val[val.prompt == "f4"][cols]
    m = a.merge(b, on=keys, how="inner", suffixes=("_f9", "_f4"), validate="one_to_one")
    m["d"] = m.outcome_f9 - m.outcome_f4
    return m


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def _fit(frame: pd.DataFrame, column: str, alpha: float) -> pd.DataFrame:
    o = frame[["model", "case", "field", "image_idx"]].assign(outcome=frame[column].astype(float))
    return MB.fit_all(MB.build_table(o), alpha=alpha)


def standardize(pair: pd.DataFrame) -> pd.DataFrame:
    """Z-score each outcome within (model, case, field) over the images that enter; `d` = z(F9) - z(F4).

    A scale-free form of the test (not set before the run). The F9 net
    valence of the restaurant case has about 3 times the SD of the F4 net
    valence, thus a raw slope of `d` can come from the scale alone. With
    z-scores, the slope of `d` compares the 2 gradients in SD units of
    each outcome.
    """
    enter = pair.dropna(subset=["d"]).copy()
    g = enter.groupby(["model", "case", "field"])
    for col in ("outcome_f4", "outcome_f9"):
        enter[col] = (enter[col] - g[col].transform("mean")) / g[col].transform("std")
    enter["d"] = enter.outcome_f9 - enter.outcome_f4
    return enter


def h9_table(pair: pd.DataFrame, fields: Sequence[str], alpha: float = 0.05) -> pd.DataFrame:
    """Fit `d`, the F4 outcome and the F9 outcome on the images that enter, for each (model, case, field, feature).

    Bonferroni goes over the rows of this call (60 for the `response` field
    alone). The F4 and F9 gradients are descriptions: they give the sign
    that H9 needs, and the 2 gradients that `d` subtracts.
    """
    enter = pair[pair.field.isin(fields)].dropna(subset=["d"])
    keys = ["model", "case", "field", "feature"]
    fd = _fit(enter, "d", alpha)
    f4 = _fit(enter, "outcome_f4", alpha)[keys + ["beta0", "se0", "p0", "outcome_mean"]]
    f9 = _fit(enter, "outcome_f9", alpha)[keys + ["beta0", "se0", "p0", "outcome_mean"]]
    out = fd.rename(columns={"beta0": "beta_d", "se0": "se_d", "p0": "p_d", "p0_bonf": "p_d_bonf",
                             "sig0": "sig_d", "beta1": "beta1_d", "se1": "se1_d", "p1": "p1_d",
                             "outcome_mean": "d_mean", "outcome_sd": "d_sd"})
    out = out.merge(f4.rename(columns={"beta0": "beta_f4", "se0": "se_f4", "p0": "p_f4", "outcome_mean": "mean_f4"}),
                    on=keys).merge(
        f9.rename(columns={"beta0": "beta_f9", "se0": "se_f9", "p0": "p_f9", "outcome_mean": "mean_f9"}), on=keys)
    same = np.sign(out.beta_d) == np.sign(out.beta_f4)
    out["h9"] = out.sig_d & same
    out["flatter"] = out.sig_d & ~same
    from scipy.stats import t as tdist

    # The same df as the test of `ols_cluster`: G - 1.
    q = tdist.ppf(1 - alpha / 2, out.n_nta - 1)
    out["ci_lo_d"] = out.beta_d - q * out.se_d
    out["ci_hi_d"] = out.beta_d + q * out.se_d
    return out.drop(columns=["share_kept", "p1_bonf", "sig1"], errors="ignore")


def verdict_fits(ver: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Fit the share of `yes` and of `no` (1 or 0 for each F9 response) on each feature. A description.

    The texts are the F9 responses with a label in yes, no, mixed, graded.
    """
    v = ver[(ver.prompt == "f9") & (ver.field == "response") & ver.label.isin(("yes", "no", "mixed", "graded"))]
    out = []
    for lab in ("yes", "no"):
        f = _fit(v.assign(y=(v.label == lab).astype(float)), "y", alpha)
        out.append(f.assign(outcome=f"share_{lab}"))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# The descriptions
# ---------------------------------------------------------------------------


def span_mix(raw: pd.DataFrame, sp: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Give the spans per text of each class, and the valence mix, for each (prompt, model, case, field)."""
    keys = ["prompt", "model", "case", "field"]
    t = _texts(raw)
    n = sp[sp.extraction_class.isin(CLASSES)].groupby(["doc_id", "extraction_class"]).size().unstack(
        fill_value=0).reindex(columns=list(CLASSES), fill_value=0)
    t = t.merge(n, left_on="doc_id", right_index=True, how="left").fillna({c: 0 for c in CLASSES})
    t = t.merge(val[["doc_id", "good", "bad", "neutral", "spans"]], on="doc_id", how="left")
    g = t.groupby(keys)
    out = g[list(CLASSES)].mean().add_prefix("per_text_")
    out["texts"] = g.size()
    out["share_with_ve"] = g.spans.apply(lambda s: float((s > 0).mean()))
    tot = g[["good", "bad", "neutral"]].sum()
    for v in ("good", "bad", "neutral"):
        out[f"ve_share_{v}"] = tot[v] / tot.sum(axis=1)
    out["cut_rate"] = g.cut.mean()
    out["tokens_p50"] = g.completion_tokens.median()
    return out.reset_index()


def verdict_shares(ver: pd.DataFrame) -> pd.DataFrame:
    """Give the share of each text label, and the share of texts with a verdict span, for each (prompt, model, case, field)."""
    keys = ["prompt", "model", "case", "field"]
    s = pd.crosstab([ver[k] for k in keys], ver.label, normalize="index").reindex(columns=list(TEXT_LABELS),
                                                                                   fill_value=0.0)
    g = ver.groupby(keys)
    s["texts"] = g.size()
    s["share_with_verdict_span"] = g.verdict_spans.apply(lambda x: float((x > 0).mean()))
    s["verdict_other_spans"] = g.verdict_other.sum()
    return s.reset_index()


def verdict_agreement(ver: pd.DataFrame) -> pd.DataFrame:
    """Give the response label against the reasoning label of the F9 answers of the 4 thinking models."""
    v = ver[ver.prompt == "f9"]
    keys = ["model", "case", "image_idx"]
    a = v[v.field == "response"][keys + ["label"]].merge(
        v[v.field == "reasoning"][keys + ["label"]], on=keys, suffixes=("_response", "_reasoning"))
    out = a.groupby(["model", "case", "label_response", "label_reasoning"]).size().rename("texts").reset_index()
    agree = a.assign(same=a.label_response == a.label_reasoning).groupby(["model", "case"]).same.mean()
    return out.merge(agree.rename("agreement").reset_index(), on=["model", "case"])


def _nta_join(frame: pd.DataFrame) -> pd.DataFrame:
    F = _f4()
    n = F.image_ntas()
    n = n[n.case.isin(CASES)][["case", "image_idx", "nta2020"]]
    return frame.merge(n, on=["case", "image_idx"], how="inner")


def nta_spearman(pair: pd.DataFrame, min_images: int = MIN_NTA_IMAGES) -> pd.DataFrame:
    """Give the Spearman rho of the NTA mean net valence with each feature, under F4 and under F9. A description.

    The images are those that enter the test. An NTA needs `min_images` of them.
    """
    from scipy.stats import spearmanr

    feats = MB.nta_features().set_index("nta2020")
    p = _nta_join(pair.dropna(subset=["d"]))
    rows = []
    for (model, case, field), g in p.groupby(["model", "case", "field"]):
        m = g.groupby("nta2020").agg(n=("d", "size"), f4=("outcome_f4", "mean"), f9=("outcome_f9", "mean"),
                                     d=("d", "mean"))
        m = m[(m.n >= min_images) & m.index.isin(feats.index)]
        for f in MB.FEATURES:
            x = feats.loc[m.index, f]
            ok = x.notna().to_numpy()
            row = {"model": model, "case": case, "field": field, "feature": f, "ntas": int(ok.sum())}
            for col in ("f4", "f9", "d"):
                rho, pv = spearmanr(m[col].to_numpy()[ok], x.to_numpy()[ok])
                row[f"rho_{col}"], row[f"p_{col}"] = float(rho), float(pv)
            rows.append(row)
    return pd.DataFrame(rows)


def verdict_by_nta(ver: pd.DataFrame, min_images: int = MIN_NTA_IMAGES):
    """Give the label shares of the F9 responses in each NTA, and their Spearman rho with each feature. A description."""
    from scipy.stats import spearmanr

    v = _nta_join(ver[(ver.prompt == "f9") & (ver.field == "response")])
    shares = pd.crosstab([v.model, v.case, v.nta2020], v.label, normalize="index").reindex(
        columns=list(TEXT_LABELS), fill_value=0.0)
    shares["images"] = v.groupby(["model", "case", "nta2020"]).size()
    shares = shares.reset_index()
    feats = MB.nta_features().set_index("nta2020")
    rows = []
    for (model, case), g in shares[shares.images >= min_images].groupby(["model", "case"]):
        g = g[g.nta2020.isin(feats.index)]
        for f in MB.FEATURES:
            x = feats.loc[g.nta2020, f].to_numpy()
            ok = ~np.isnan(x)
            row = {"model": model, "case": case, "feature": f, "ntas": int(ok.sum())}
            for lab in ("yes", "no", "mixed", "graded"):
                rho, pv = spearmanr(g[lab].to_numpy()[ok], x[ok])
                row[f"rho_{lab}"], row[f"p_{lab}"] = float(rho), float(pv)
            rows.append(row)
    return shares, pd.DataFrame(rows)


def from_trace_rows(ver: pd.DataFrame, pair: pd.DataFrame) -> List[dict]:
    """Report each response that the extraction input copied from its trace (D142, Q5)."""
    if not FROM_TRACE.exists():
        return []
    src = pd.read_parquet(FROM_TRACE)
    if "response_from_trace" not in src.columns:
        return []
    ids = src[src.response_from_trace.astype(bool)].doc_id.tolist()
    out = []
    for doc in ids:
        model, case, idx, field = doc.split("|")
        c = case.replace("_subj", "")
        lab = ver[ver.doc_id == doc].label.tolist()
        p = pair[(pair.model == model) & (pair.case == c) & (pair.field == field) & (pair.image_idx == int(idx))]
        out.append({"doc_id": doc, "label": lab[0] if lab else "missing",
                    "outcome_f9": None if p.empty else p.outcome_f9.iloc[0],
                    "outcome_f4": None if p.empty else p.outcome_f4.iloc[0],
                    "d": None if p.empty else p.d.iloc[0]})
    return out


def design_checks(ver: pd.DataFrame) -> List[dict]:
    out = []
    for model, case, idx, field, want in DESIGN_CHECKS:
        r = ver[(ver.prompt == "f9") & (ver.model == model) & (ver.case == case) & (ver.image_idx == idx)
                & (ver.field == field)]
        got = r.label.iloc[0] if len(r) else "missing"
        out.append({"model": model, "case": case, "image_idx": idx, "field": field, "expected": want,
                    "label": got, "ok": got == want})
    return out


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------


def plot_gradients(h9: pd.DataFrame, standardized: bool = False):
    """Plot the F9 gradient against the F4 gradient of each (model, feature), 1 panel for each case.

    With `standardized`, `h9` comes from `standardize`: each gradient is in
    SD units of its own outcome.
    """
    import matplotlib.pyplot as plt

    F8 = _f8()
    F = _f4()
    fig, axes = plt.subplots(1, len(CASES), figsize=(5.0, 2.55), squeeze=False)
    lim = float(np.nanmax(np.abs(h9[["beta_f4", "beta_f9"]].to_numpy()))) * 1.12
    for ax, case in zip(axes[0], CASES):
        g = h9[h9.case == case]
        ax.axhline(0, color="#888888", lw=0.5, zorder=1)
        ax.axvline(0, color="#888888", lw=0.5, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], color=F8.S.EDGE, lw=0.6, ls="--", zorder=1)
        for model in [m for m in F.MODELS if m in set(g.model)]:
            h = g[g.model == model]
            c = F.MODEL_COLORS[model]
            ax.scatter(h.beta_f4, h.beta_f9, s=18, marker=F.MODEL_MARKERS[model],
                       facecolors=np.where(h.sig_d, c, "white"), edgecolors=F8.S.EDGE, linewidths=F8.S.EDGE_LW,
                       zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(case)
        ax.set_xlabel("F4 gradient (outcome SD per 1 SD)" if standardized else "F4 gradient (per 1 SD)")
    axes[0][0].set_ylabel("F9 gradient (outcome SD per 1 SD)" if standardized else "F9 gradient (per 1 SD)")
    handles = F8._model_handles([m for m in F.MODELS if m in set(h9.model)])
    handles += F8._fill_handles("d significant", "not significant")
    F8._top_legend(fig, handles, ncol=4)
    if standardized:
        F8.describe(
            fig, "The NTA gradient of the net valence under the F4 and the F9 prompt, in SD units of each outcome",
            "As the raw figure, but each net valence is z-scored within (model, case) over the images that enter, "
            "thus each axis gives the change of the outcome in its own SDs per 1 SD of the feature. This removes "
            "the wider spread of the F9 net valence (about 3 times the F4 SD in the restaurant case). A marker "
            "farther from 0 than the dashed line y = x, on the same side, is a steeper gradient under F9. A filled "
            "marker has a significant slope of d = z(F9) minus z(F4) (CR1 by NTA, Bonferroni over 60 tests). "
            "This form of the test was not set before the run.",
            ["f9", "h9", "net-valence", "nta", "standardized"])
        return fig
    F8.describe(
        fig, "The NTA gradient of the net valence under the F4 and the F9 prompt",
        "Each marker is 1 model and 1 ACS feature (5 features: % white, % Black, % Asian, % bachelor's degree, "
        "log median household income, each z-scored over the NTAs). The x axis is the OLS slope of the net "
        "valence of the response on the feature under the F4 prompt; the y axis is the same slope under the "
        "F9 (subjective) prompt, on the same images. The dashed line is y = x. A marker farther from 0 than "
        "the line, on the same side, is a steeper gradient under F9. A filled marker has a significant "
        "slope of d = F9 minus F4 (CR1 by NTA, Bonferroni over 60 tests). The F9 net valence has a wider "
        "spread than the F4 net valence (restaurant: about 3 times the SD), and a raw slope grows with the "
        "spread; the standardized figure removes this.",
        ["f9", "h9", "net-valence", "nta"])
    return fig


def plot_verdicts(shares: pd.DataFrame):
    """Plot the label shares of the F9 responses, 1 bar for each model, 1 panel for each case."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    F8 = _f8()
    F = _f4()
    s = shares[(shares.prompt == "f9") & (shares.field == "response")]
    models = [m for m in F.MODELS if m in set(s.model)]
    fig, axes = plt.subplots(1, len(CASES), figsize=(5.0, 2.1), sharey=True, squeeze=False)
    for ax, case in zip(axes[0], CASES):
        g = s[s.case == case].set_index("model").reindex(models)
        left = np.zeros(len(models))
        for lab in TEXT_LABELS:
            w = g[lab].fillna(0).to_numpy()
            ax.barh(range(len(models)), w, left=left, color=F8.S.PAL[LABEL_COLORS[lab]], edgecolor=F8.S.EDGE,
                    linewidth=F8.S.EDGE_LW, height=0.75)
            left += w
        ax.set_yticks(range(len(models)), models)
        ax.set_xlim(0, 1)
        ax.set_title(case)
        ax.set_xlabel("share of responses")
    axes[0][0].invert_yaxis()  # 1 time: the panels share the y axis
    handles = [Patch(facecolor=F8.S.PAL[LABEL_COLORS[x]], edgecolor=F8.S.EDGE, linewidth=F8.S.EDGE_LW,
                     label=x.replace("_", " ")) for x in TEXT_LABELS]
    F8._top_legend(fig, handles, ncol=6)
    F8.describe(
        fig, "The verdict of the F9 responses",
        "The label of each F9 response, from the verdict spans of the extractor in text order: the last span "
        "decides; a last yes or no with the other one earlier in the text is mixed; graded is 1 middle rating; "
        "not applicable says that the photograph shows no restaurant, or that it cannot judge the spot; none "
        "is a response with no verdict span. 10,000 images for each model and case, minus the copy suspects.",
        ["f9", "verdict"])
    return fig


def plot_span_mix(mix: pd.DataFrame):
    """Plot the visual-evidence and inference spans per response under F4 (hollow) and F9 (filled)."""
    import matplotlib.pyplot as plt

    F8 = _f8()
    F = _f4()
    s = mix[mix.field == "response"]
    models = [m for m in F.MODELS if m in set(s.model)]
    cols = (("per_text_visual_evidence", "visual evidence"), ("per_text_inference", "inference"))
    fig, axes = plt.subplots(len(CASES), len(cols), figsize=(5.0, 3.6), sharey=True, squeeze=False)
    for i, case in enumerate(CASES):
        for j, (col, name) in enumerate(cols):
            ax = axes[i][j]
            for k, m in enumerate(models):
                for prompt, dy in (("f4", -0.15), ("f9", 0.15)):
                    r = s[(s.case == case) & (s.model == m) & (s.prompt == prompt)]
                    if r.empty:
                        continue
                    ax.scatter(r[col], [k + dy], s=20, marker=F.MODEL_MARKERS[m],
                               facecolors=F.MODEL_COLORS[m] if prompt == "f9" else "white",
                               edgecolors=F8.S.EDGE, linewidths=F8.S.EDGE_LW, zorder=3)
            ax.set_yticks(range(len(models)), models)
            ax.set_ylim(len(models) - 0.5, -0.5)
            ax.set_xlim(0, float(s[s.case == case][col].max()) * 1.12)
            ax.set_title(f"{case}: {name}")
            if i == len(CASES) - 1:
                ax.set_xlabel("spans per response")
    F8._top_legend(fig, F8._fill_handles("F9 prompt", "F4 prompt"), ncol=2)
    F8.describe(
        fig, "The span mix of the responses under the F4 and the F9 prompt",
        "The mean count of visual-evidence spans and of inference spans per response, for each model, under "
        "the F4 prompt (hollow) and the F9 subjective prompt (filled), on the same images and under 1 schema. "
        "The net valence counts the visual-evidence spans only.",
        ["f9", "span-mix"])
    return fig


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--x2", type=Path, default=X2, help="the merged F9 round (default data/f4_traces/subj/x2)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-nta-images", type=int, default=MIN_NTA_IMAGES)
    args = ap.parse_args(argv)
    rule_checks()
    t0 = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("read the merged spans ...", flush=True)
    raw, coverage = load(args.x2)
    sp = spans_of(raw)
    val = text_valence(raw, sp)
    ver = text_verdicts(raw, sp)
    pair = paired(val)
    print(f"{raw.doc_id.nunique()} texts, {len(sp)} spans, {len(pair)} pairs in {time.time() - t0:.0f} s", flush=True)

    primary = h9_table(pair, ["response"], args.alpha)
    secondary = h9_table(pair, ["reasoning"], args.alpha)
    scaled = h9_table(standardize(pair[pair.field == "response"]), ["response"], args.alpha)
    vfits = verdict_fits(ver, args.alpha)
    mix = span_mix(raw, sp, val)
    shares = verdict_shares(ver)
    agree = verdict_agreement(ver)
    rho = nta_spearman(pair, args.min_nta_images)
    nta_shares, nta_rho = verdict_by_nta(ver, args.min_nta_images)
    entered = (pair.assign(enter=pair.d.notna()).groupby(["model", "case", "field"])
               .agg(pairs=("enter", "size"), entered=("enter", "sum")).reset_index())

    for name, frame in (("coverage", coverage), ("h9_primary", primary), ("h9_reasoning", secondary),
                        ("h9_standardized", scaled),
                        ("verdict_fits", vfits), ("span_mix", mix), ("verdict_shares", shares),
                        ("verdict_agreement", agree), ("nta_spearman", rho), ("verdict_by_nta", nta_shares),
                        ("verdict_nta_spearman", nta_rho), ("entered", entered)):
        frame.to_csv(out / f"{name}.csv", index=False)
    ver.to_parquet(out / "text_verdicts.parquet", index=False)
    pair.to_parquet(out / "pairs.parquet", index=False)

    F8 = _f8()
    for name, fig in (("f9_gradients", plot_gradients(primary)),
                      ("f9_gradients_standardized", plot_gradients(scaled, standardized=True)),
                      ("f9_verdicts", plot_verdicts(shares)),
                      ("f9_span_mix", plot_span_mix(mix))):
        F8.save(fig, name, out)

    checks = design_checks(ver)
    meta = {
        "script": "research/scripts/f9/analyze_subjective.py", "version": __version__, "x2": str(args.x2),
        "alpha": args.alpha, "primary_tests": int(len(primary)),
        "h9_rows": int(primary.h9.sum()), "flatter_rows": int(primary.flatter.sum()),
        "sig_d_rows": int(primary.sig_d.sum()),
        "h9_by_case": primary.groupby("case").h9.sum().astype(int).to_dict(),
        "reasoning_rows_p05": int((secondary.p_d < args.alpha).sum()),
        "reasoning_h9_rows_bonf40": int(secondary.h9.sum()),
        "standardized_h9_rows": int(scaled.h9.sum()), "standardized_flatter_rows": int(scaled.flatter.sum()),
        "standardized_h9_by_case": scaled.groupby("case").h9.sum().astype(int).to_dict(),
        "model1_d_p05": int((primary.p1_d < args.alpha).sum()),
        "outcome_sd": pair[pair.field == "response"].dropna(subset=["d"]).groupby(["case", "model"])[
            ["outcome_f4", "outcome_f9"]].std().round(3).reset_index().to_dict("records"),
        "design_checks": checks, "response_from_trace": from_trace_rows(ver, pair),
        "b10": {"missing": int(coverage.missing.sum()), "quotable_min": float(coverage.quotable.min()),
                "copy_suspect_share_max": float(coverage.copy_suspect_share.max())},
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(coverage.to_string(index=False))
    show = ["model", "case", "feature", "n", "beta_f4", "beta_f9", "beta_d", "p_d_bonf", "h9", "flatter"]
    print(primary[show].to_string(index=False))
    print(json.dumps({k: meta[k] for k in ("h9_rows", "flatter_rows", "sig_d_rows", "h9_by_case",
                                           "design_checks", "b10")}, indent=1, default=str))
    print(f"wrote {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
