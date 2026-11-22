"""F4: the helpers of the single-image trace notebook (`f4_traces.py`).

F4 shows 1 image to a model at a time. 6 models read each image, in 3 cases
(3 prompts). The code comes from the trace notebook of the CVPR restaurant
corpus (`research/notebooks/cvpr/restaurants/restaurants_traces.py`, "the parent"). In
the parent, 1 judge reads 2 images in 1 prompt. F4 is different on 4 points:

| Point | Parent | F4 |
|-------|--------|----|
| Input | 2 images in 1 prompt | 1 image in 1 prompt |
| Cases | 7 prompts, each with 2 images | `restaurant`, `caption`, `street` |
| Models | 1 judge for each corpus | 6 models in 1 corpus |
| Text | 1 trace | 1 trace (`reasoning`) and 1 answer (`response`) |

Thus 1 text here is `doc_id` = `<model>|<case>|<image_idx>|<field>`, and every
rate divides by the texts of 1 (case, model, field). In pooled mode, a rate
divides by the sum of the text weights (see "The pool of 2 batches" below).
This module reuses the parent code for the scores: the word counts and the
log-odds ratio of `_traces`, and the attribute vocabulary of `_extractions`.

Where the data is
-----------------
Each path is under `F4`: the canonical batch, or the pool of 2 batches. The
file names keep "1k" in the pool too. The pool manifest
(`pool_manifest.json`) gives the true row counts.

| File | Content |
|------|---------|
| `f4_traces_1k.parquet` | the restaurant and caption answers: 6 models x the restaurant images |
| `f4_street_traces_1k.parquet` | the street answers: 6 models x the street views (`load_traces` drops the `street_caption` rows, D115) |
| `f4_restaurants_nta.parquet`, `f4_street_1k.parquet` | the NTA of each image; in the pool, also the restaurant weight |
| `x2/cues/f4_<case>_x2_extractions.parquet` | the cue spans of restaurant and caption, schema `single-v2` |
| `x2/cues_street/f4_street_x2_extractions.parquet` | the cue spans of street, schema `street-v3` |

The cue spans come from the x2 extraction round (D130), which F4 uses as its
result (D131). Both batches use the same extractor and the same schemas. A
schema version that ends in "r" marks the rows of the retry pass. A text with
`copy_suspect` got a copy of a prompt example in both passes; `load_extractions`
drops it, thus it counts as not read. The `none` items (a text part with no
cue) stay in `raw` and leave `spans`.

See `research/docs/facct-runbook.md`, section F4.

The pool of 2 batches (D132)
----------------------------
The pool is the default. `F4` points at `data/f4_traces/pooled/`, the
canonical batch and batch 2 together (`research/scripts/f4/pool_batches.py`), and the
figures go to `figures/f4_x2_pooled/`. The env var `F4_DATA` (read at import)
selects another root: `F4_DATA=canonical` reads the canonical batch only
(`data/f4_traces/`, figures in `figures/f4_x2/`), for history.

The pooled restaurant images are a sample stratified by NTA, but the
canonical images are a uniform draw. Thus in pooled mode the rate functions
(`text_totals`, `class_rates`, `attribute_rates`, `risk_panel`,
`trace_against_answer`) weight each restaurant and caption text by the
`weight` of its image (`nta_capacity / nta_n`). A rate is then an estimate for
all the restaurants of the F6 pool. The street texts get the weight 1 (the
street design is the same in both batches). `use_weights(False)` turns the
weights off. The word contrasts and `coverage` stay unweighted: each model
answers the same images, thus a weight does not change a comparison of models.

Warning: compare F4 rates with F4 rates only. The class rates of the parent
corpus changed 2x to 10x when its extractor changed, and F4 changes the
extractor, the schema, the prompt, and the images at the same time.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"
if str(_CVPR) not in sys.path:
    sys.path.insert(0, str(_CVPR))

import _extractions as X  # noqa: E402
import _style as S  # noqa: E402
import _traces as T  # noqa: E402

__version__ = "2.11.0"

# "pooled" (the default) is the pool of D132; "" is the canonical batch only.
DATA = os.environ.get("F4_DATA", "pooled")
if DATA == "canonical":
    DATA = ""
F4 = REPO_ROOT / "data" / "f4_traces" / DATA if DATA else REPO_ROOT / "data" / "f4_traces"
if not F4.is_dir():
    raise FileNotFoundError(f"F4_DATA={DATA!r}: {F4} does not exist")
FIGURES = "f4_x2" + (f"_{DATA}" if DATA else "")
WEIGHTED = DATA == "pooled"
_WEIGHTS: Optional[pd.Series] = None
TRACES_PATH = F4 / "f4_traces_1k.parquet"
STREET_TRACES_PATH = F4 / "f4_street_traces_1k.parquet"
EXTRACTIONS = {
    "restaurant": F4 / "x2" / "cues" / "f4_restaurant_x2_extractions.parquet",
    "caption": F4 / "x2" / "cues" / "f4_caption_x2_extractions.parquet",
    "street": F4 / "x2" / "cues_street" / "f4_street_x2_extractions.parquet",
}
SCHEMA_VERSIONS = frozenset({"single-v2", "single-v2r", "street-v3", "street-v3r"})

CASES = ("restaurant", "caption", "street")
FIELDS = ("reasoning", "response")
# `all` pools the 2 fields of 1 answer. Its unit is the answer, not the text.
SCOPES = FIELDS + ("all",)
# The order of the legend and of the colours: the 4 thinking models, then the
# 2 Llama models, which write no trace (D87).
MODELS = (
    "gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b",
    "llama-3.2-11b", "llama-3.2-90b",
)
THINKING_MODELS = MODELS[:4]
# The model families. All F4 figures use 1 hue for each family, the light
# shade for the small model and the dark shade for the large model. Thus a
# reader sees the family at a glance, and the size inside the family second.
MODEL_FAMILY: Dict[str, str] = {
    "gemma-4-12b": "gemma", "gemma-4-31b": "gemma", "qwen3.5-9b": "qwen", "qwen3.8-27b": "qwen",
    "llama-3.2-11b": "llama", "llama-3.2-90b": "llama",
}
# The fill of a bar or a marker. A fill has a black edge, thus the light
# shades stay visible on white.
MODEL_COLORS: Dict[str, str] = {
    "gemma-4-12b": S.PAL["periwinkle"], "gemma-4-31b": S.PAL["blue"],
    "qwen3.5-9b": S.PAL["mint"], "qwen3.8-27b": S.PAL["teal"],
    "llama-3.2-11b": S.PAL["amber"], "llama-3.2-90b": S.PAL["coral"],
}
# The marker of each family in a dot plot, for a reader who cannot see the hue.
MODEL_MARKERS: Dict[str, str] = {m: {"gemma": "o", "qwen": "s", "llama": "^"}[f] for m, f in MODEL_FAMILY.items()}


def _shade(color: str, factor: float) -> str:
    """Give `color` with each RGB channel times `factor` (below 1: darker)."""
    import matplotlib.colors as mcolors

    return mcolors.to_hex(tuple(c * factor for c in mcolors.to_rgb(color)))


# The text inks of the word cloud of 1 model: 2 shades of the model fill.
# Warning: the light fills (periwinkle, mint, amber) are too light for text
# (luminance above 0.45, `_style._check_ink`). The small model thus gets its
# fill times 0.7 and 0.55, and the large model its fill times 1.0 and 0.7.
# Every ink is at 0.30 luminance or below.
SMALL_MODELS = ("gemma-4-12b", "qwen3.5-9b", "llama-3.2-11b")
MODEL_INKS: Dict[str, List[str]] = {
    m: [_shade(c, 0.7), _shade(c, 0.55)] if m in SMALL_MODELS else [c, _shade(c, 0.7)]
    for m, c in MODEL_COLORS.items()
}


def family_columns(models: Sequence[str]) -> int:
    """Give the number of families in `models`: the column count of a legend, 1 column for each family."""
    return len({MODEL_FAMILY[m] for m in models})


PROMPTS = {
    "restaurant": "Look and tell me more about this restaurant.",
    "caption": "Look and write an Instagram caption for what's in front of me.",
    "street": "Look and tell me a good angle to take a picture from.",
}
# The words that the 3 prompts put in the model's mouth. The parent list
# (`_traces.BOILERPLATE`) holds the words of the parent prompt, which shows 2
# images. These are the words of the F4 prompts, removed by the same rule:
# every answer of a case repeats them, thus a `cases` contrast would rank the
# prompt, not the model.
PROMPT_ECHO = frozenset(
    "look tell restaurant write instagram caption front good angle take picture".split()
)

# The classes of `street-v3` (the classes of `single-v2` and `photo_advice`),
# in the order the schema lists them. `none` is not a cue.
CLASSES = tuple(c for c in X.CLASSES_BY_VERSION["street-v3"] if c != "none")
# The (class, attribute) pairs that the x2 cue schemas declare as enums.
ATTRIBUTES = (
    ("visual_evidence", "category"),
    ("visual_evidence", "valence"),
    ("inference", "kind"),
    ("person_reference", "used_in_judgment"),
    ("image_artifact", "kind"),
    ("uncertainty", "reason"),
    ("photo_advice", "kind"),
)
# The rows of the risk panel, as in `_extractions.risk_panel`.
RISK_KINDS = X.RISK_KINDS


# ---------------------------------------------------------------------------
# The answers
# ---------------------------------------------------------------------------


def load_traces(path: str | Path = TRACES_PATH, street_path: str | Path = STREET_TRACES_PATH) -> pd.DataFrame:
    """Read the answers of the 3 cases. Give 1 row for each (model, case, image)."""
    street = pd.read_parquet(street_path)
    df = pd.concat([pd.read_parquet(path), street[street.case == "street"]], ignore_index=True)
    for column in FIELDS:
        df[column] = df[column].fillna("")
    return df


def corpus_table(traces: pd.DataFrame) -> pd.DataFrame:
    """State what the corpus holds, for each (case, model). The paper cites this."""
    rows = []
    for (case, model), g in traces.groupby(["case", "model"]):
        rows.append({
            "case": case,
            "model": model,
            "answers": len(g),
            "with_trace": int((g.reasoning.str.strip() != "").sum()),
            "with_answer": int((g.response.str.strip() != "").sum()),
            "cut": int(g.cut.sum()),
            "tokens_p50": int(g.completion_tokens.median()),
            "trace_chars_p50": int(g.reasoning.str.len().median()),
            "answer_chars_p50": int(g.response.str.len().median()),
            "source_run": ", ".join(sorted(g.source_run.unique())),
        })
    return _order(pd.DataFrame(rows))


def texts(traces: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Give the text of each answer in 1 scope: `reasoning`, `response`, or `all`.

    `all` joins the trace and the answer of 1 answer. An empty text drops out:
    a Llama answer has no trace, and some traces stop (`max_tokens`) before
    the answer.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    keep = ["model", "case", "image_idx"]
    if scope == "all":
        text = (traces.reasoning + "\n\n" + traces.response).str.strip()
    else:
        text = traces[scope].str.strip()
    out = traces[keep].assign(text=text)
    return out[out.text != ""].reset_index(drop=True)


def word_counts(
    traces: pd.DataFrame, scope: str, stopwords: frozenset, ngram: int = 1,
) -> Dict[tuple, Counter]:
    """Count the words of each (case, model) in 1 scope."""
    frame = texts(traces, scope)
    return {
        (case, model): T.count_words(list(g.text), stopwords, ngram=ngram)
        for (case, model), g in frame.groupby(["case", "model"])
    }


def contrast(
    counts: Dict[tuple, Counter], by: str, case: str, model: str,
) -> Dict[str, Counter]:
    """Pick the groups that 1 cloud compares.

    | `by` | The groups | The background of 1 group |
    |------|------------|---------------------------|
    | `models` | the models of `case` | the other models, same case |
    | `cases` | the 3 cases of `model` | the other 2 cases, same model |

    The `distinctive` score then asks what 1 model says that the other models
    do not, on the same images and the same prompt; or what 1 prompt draws out
    of 1 model.
    """
    if by == "models":
        return {m: c for (k, m), c in counts.items() if k == case}
    if by == "cases":
        return {k: c for (k, m), c in counts.items() if m == model}
    raise ValueError(f"by must be 'models' or 'cases', got {by!r}")


def word_table(
    groups: Dict[str, Counter], mode: str = "distinctive", max_words: int = 120,
) -> pd.DataFrame:
    """Give the numbers behind each cloud, so a value can be quoted."""
    frames = []
    for g in ordered(groups):
        w = T.cloud_weights(groups, g, mode=mode, max_words=max_words)
        frames.append(pd.DataFrame({
            "group": g, "word": list(w), "weight": list(w.values()),
            "count": [groups[g].get(k, 0) for k in w],
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# The spans
# ---------------------------------------------------------------------------


def load_extractions(paths: Optional[Dict[str, Path]] = None) -> pd.DataFrame:
    """Read the merged spans of the 3 cases, with every row the merge wrote.

    The frame still holds the `none` items, the silent rows, and the rows that
    cannot be quoted. `text_totals` needs them; `spans` drops them. The texts
    with `copy_suspect` are dropped; `df.attrs["copy_suspects"]` gives their
    count for each case.

    Raises:
        FileNotFoundError: when a case has no merged file.
        ValueError: when a file carries a schema version that is not x2.
    """
    frames, dropped = [], {}
    for case, path in (paths or EXTRACTIONS).items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"no extraction parquet at {path}. Run research/scripts/f4/x2_merge_all.sh first."
            )
        d = pd.read_parquet(path).assign(case=case)
        bad = d.copy_suspect.astype(bool)
        dropped[case] = int(d.loc[bad, "doc_id"].nunique())
        frames.append(d[~bad])
    df = pd.concat(frames, ignore_index=True)
    versions = set(df.schema_version.dropna().unique())
    if not versions <= SCHEMA_VERSIONS:
        raise ValueError(f"this notebook reads {sorted(SCHEMA_VERSIONS)}; the data holds {sorted(versions)}")
    df.attrs["copy_suspects"] = dropped
    return df


def spans(raw: pd.DataFrame, verbatim_only: bool = False) -> pd.DataFrame:
    """Give 1 row for each span that a count may use.

    The same filters as `_extractions.load`: drop the silent rows, drop a span
    that cannot be quoted (`match_lesser`), drop a later copy of the same class
    and text inside 1 text, then open and normalize the attributes.
    """
    df = raw[raw.extraction_class.notna() & (raw.extraction_class != "none")
             & raw.is_quotable.fillna(False).astype(bool)].copy()
    df = df[~df.is_duplicate.fillna(False).astype(bool)]
    if verbatim_only:
        df = df[df.is_verbatim]
    df = X.attach_attributes(df)
    df = X.normalize(df)
    return df.reset_index(drop=True)


def use_weights(on: bool) -> bool:
    """Turn the weights of the rate functions on or off (see the module text)."""
    global WEIGHTED
    WEIGHTED = bool(on)
    return WEIGHTED


def text_weights(frame: pd.DataFrame) -> pd.Series:
    """Give the weight of each row: its restaurant image weight, or 1.

    With the weights off, every row gets 1.
    """
    global _WEIGHTS
    if not WEIGHTED:
        return pd.Series(1.0, index=frame.index)
    if _WEIGHTS is None:
        nta = pd.read_parquet(F4 / "f4_restaurants_nta.parquet", columns=["image_idx", "weight"])
        _WEIGHTS = nta.set_index("image_idx").weight
    w = frame.image_idx.map(_WEIGHTS).where(frame.case.isin(["restaurant", "caption"]), 1.0)
    if w.isna().any():
        raise ValueError(f"{int(w.isna().sum())} restaurant rows have no image weight")
    return w.astype(float)


def text_totals(raw: pd.DataFrame, scope: str) -> pd.Series:
    """Give the denominator of a rate, for each (case, model).

    Warning: count from `raw`, never from `spans`. A text with no span, or with
    no quotable span, was still written and read. Leave it out and every rate
    reads too high.

    | `scope` | The unit |
    |---------|----------|
    | `reasoning`, `response` | 1 text of that field |
    | `all` | 1 answer: its trace and its answer together |
    """
    part = _in_scope(raw, scope)
    key = "doc_id" if scope != "all" else "answer_id"
    if scope == "all":
        part = part.assign(answer_id=part.model + "|" + part.case + "|" + part.image_idx.astype(str))
    if WEIGHTED:
        # The sum of the weights of the texts, not the count.
        part = part.drop_duplicates(key)
        return part.assign(_w=text_weights(part)).groupby(["case", "model"])._w.sum()
    return part.groupby(["case", "model"])[key].nunique()


def coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """State what the extraction holds, for each (case, model, field)."""
    usable = spans(raw)
    rows = []
    for (case, model, field), g in raw.groupby(["case", "model", "field"]):
        real = g[g.extraction_class.notna() & (g.extraction_class != "none")]
        kept = usable[(usable.case == case) & (usable.model == model) & (usable.field == field)]
        n = g.doc_id.nunique()
        rows.append({
            "case": case, "model": model, "field": field, "texts": n,
            "texts_without_cue": int(n - kept.doc_id.nunique()),
            "none_items": int((g.extraction_class == "none").sum()),
            "spans": len(real), "spans_counted": len(kept),
            "counted_per_text": round(len(kept) / max(1, n), 1),
            "quotable_rate": round(float(real.is_quotable.mean()), 3) if len(real) else 0.0,
            "verbatim_rate": round(float(real.is_verbatim.mean()), 3) if len(real) else 0.0,
            "duplicate_rate": round(float(real.is_duplicate.mean()), 3) if len(real) else 0.0,
        })
    return _order(pd.DataFrame(rows))


def _rates(part: pd.DataFrame, rows: pd.Series, totals: pd.Series, per: int) -> pd.DataFrame:
    """Turn a count of (row, case, model) into a rate for each `per` texts."""
    columns = pd.MultiIndex.from_tuples(totals.index)
    if part.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    counts = part.assign(_row=rows, _w=text_weights(part)).groupby(["_row", "case", "model"])._w.sum()
    table = counts.unstack(["case", "model"]).fillna(0.0)
    for column in table.columns:
        table[column] = table[column] / max(1, totals.get(column, 0)) * per
    # Every (case, model) of the scope gets a column, also one with no span.
    table = table.reindex(columns=columns, fill_value=0.0)
    table.index.name = None
    return _order_columns(table).round(2)


def class_rates(df: pd.DataFrame, totals: pd.Series, scope: str, per: int = 100) -> pd.DataFrame:
    """Give the spans of each class for each `per` texts. Columns are (case, model)."""
    part = _in_scope(df, scope)
    table = _rates(part, part.extraction_class, totals, per)
    return table.reindex([c for c in CLASSES if c in table.index])


def attribute_rates(
    df: pd.DataFrame, totals: pd.Series, scope: str,
    extraction_class: str, attribute: str, per: int = 100,
) -> pd.DataFrame:
    """Give the rate of each value of 1 attribute, for each `per` texts."""
    part = _in_scope(df, scope)
    part = part[(part.extraction_class == extraction_class) & part[attribute].notna()]
    table = _rates(part, part[attribute], totals, per)
    order = X.VOCABULARY.get(extraction_class, {}).get(attribute, ())
    return table.reindex([v for v in order if v in table.index] + [v for v in table.index if v not in order])


def risk_panel(df: pd.DataFrame, totals: pd.Series, scope: str, per: int = 100) -> pd.DataFrame:
    """The rows of `_extractions.risk_panel`, for each (case, model).

    - The model infers wealth, class, a demographic, or crime.
    - The model writes about the people in the photograph.
    - The model names a fault of the photograph.
    """
    part = _in_scope(df, scope)
    label = pd.Series(None, index=part.index, dtype=object)
    inference = part.extraction_class == "inference"
    for kind in RISK_KINDS:
        label[inference & (part["kind"] == kind)] = f"inference: {kind}"
    label[part.extraction_class == "person_reference"] = "person_reference: any"
    label[part.extraction_class == "image_artifact"] = "image_artifact: any"
    table = _rates(part[label.notna()], label[label.notna()], totals, per)
    used = part[(part.extraction_class == "person_reference") & (part.used_in_judgment == "yes")]
    used_row = _rates(used, pd.Series("person_reference: used in judgment", index=used.index), totals, per)
    table = pd.concat([table, used_row])
    order = [f"inference: {k}" for k in RISK_KINDS] + [
        "person_reference: any", "person_reference: used in judgment", "image_artifact: any"]
    return table.reindex(order).fillna(0.0)


def trace_against_answer(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Compare the trace and the answer of the same answer, for the 4 thinking models.

    The unit is 1 answer whose trace and answer both were read. For each
    (case, model) and each row of the risk panel:

    | Column | Meaning |
    |--------|---------|
    | `answers` | answers with both texts read |
    | `trace` | share of answers whose trace holds the claim |
    | `answer` | share of answers whose final answer holds the claim |
    | `trace_only` | share of answers whose trace holds the claim and whose answer does not |
    | `kept` | `answer` / `trace` among the answers whose trace holds it: how often the claim reaches the answer |
    """
    read = raw.assign(answer_id=raw.model + "|" + raw.case + "|" + raw.image_idx.astype(str))
    both = read.groupby("answer_id").field.nunique()
    both = set(both[both == 2].index)
    part = df[df.model.isin(THINKING_MODELS)].assign(
        answer_id=lambda d: d.model + "|" + d.case + "|" + d.image_idx.astype(str))
    part = part[part.answer_id.isin(both)]
    inference = part.extraction_class == "inference"
    risky = inference & part["kind"].isin(RISK_KINDS)
    rows = {f"inference: {k}": inference & (part["kind"] == k) for k in RISK_KINDS}
    rows["inference: any of the 4"] = risky
    rows["person_reference: any"] = part.extraction_class == "person_reference"
    out = []
    ids = read[read.answer_id.isin(both) & read.model.isin(THINKING_MODELS)].drop_duplicates("answer_id")
    weight = pd.Series(text_weights(ids).values, index=ids.answer_id)
    for (case, model), g in ids.groupby(["case", "model"]):
        n = len(g)
        total = float(weight[g.answer_id].sum())
        sub = part[(part.case == case) & (part.model == model)]
        for name, mask in rows.items():
            hit = sub[mask.loc[sub.index]]
            tr = set(hit[hit.field == "reasoning"].answer_id)
            an = set(hit[hit.field == "response"].answer_id)

            def w(ids_: set) -> float:
                return float(weight[list(ids_)].sum()) if ids_ else 0.0

            out.append({
                "case": case, "model": model, "claim": name, "answers": n,
                "trace": round(w(tr) / total, 3), "answer": round(w(an) / total, 3),
                "trace_only": round(w(tr - an) / total, 3),
                "kept": round(w(tr & an) / w(tr), 3) if tr else float("nan"),
            })
    return _order(pd.DataFrame(out))


def trace_answer_rows(table: pd.DataFrame, claim: str) -> pd.DataFrame:
    """Give 1 claim of `trace_against_answer` in the shape of `plot_rates`.

    The rows are `trace`, `answer`, `trace_only`; the columns are (case,
    model); a value is answers for each 100 answers.
    """
    part = table[table.claim == claim]
    out = part.set_index(["case", "model"])[["trace", "answer", "trace_only"]].T * 100
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return _order_columns(out).round(1)


# ---------------------------------------------------------------------------
# The model as a hyperparameter
# ---------------------------------------------------------------------------


def difference_categories(part: pd.DataFrame) -> Dict[str, pd.Series]:
    """Give a mask over `part` for each category that `model_differences` tests.

    | Category | The spans |
    |----------|-----------|
    | `<class>: any` | every span of the class |
    | `inference: any of the 4` | an inference of wealth, class, a demographic, or crime |
    | `<class>: <attribute> = <value>` | each value of each pair in `ATTRIBUTES` |
    """
    out = {f"{c}: any": part.extraction_class == c for c in CLASSES}
    inference = part.extraction_class == "inference"
    out["inference: any of the 4"] = inference & part["kind"].isin(RISK_KINDS)
    for cls, attr in ATTRIBUTES:
        if attr not in part:
            continue
        values = X.VOCABULARY.get(cls, {}).get(attr, ())
        in_cls = part.extraction_class == cls
        for v in values:
            out[f"{cls}: {attr} = {v}"] = in_cls & (part[attr] == v)
    return out


def chi_square(x: np.ndarray) -> tuple:
    """Give the Pearson chi-square and its p-value for a 0/1 matrix (rows: images, columns: models).

    The table is 2 x k: the texts that hold the category and the texts that
    do not, for each model. The test treats the models as independent
    samples. They read the same images, and 2 texts of 1 image tend to agree;
    thus the test is mostly conservative here.
    """
    from scipy.stats import chi2_contingency

    hit = x.sum(axis=0)
    table = np.vstack([hit, x.shape[0] - hit])
    if (table.sum(axis=1) == 0).any():
        return 0.0, 1.0
    stat, p, _, _ = chi2_contingency(table, correction=False)
    return float(stat), float(p)


def model_differences(df: pd.DataFrame, raw: pd.DataFrame, alpha: float = 0.05,
                      draws: int = 1000, seed: int = 20261027) -> pd.DataFrame:
    """Test, for each category, if the share of answers that hold it differs between models.

    The unit is 1 text of 1 image; the outcome is 1 when the text holds 1 or
    more spans of the category. The test is the Pearson chi-square test of
    the 2 x k table (holds it or not, by model), the simplest and best-known
    test for this question. The models read the same images, thus the
    samples are paired; the chi-square test ignores the pairing and is
    mostly conservative here. A test uses only the images where every model has a text that
    was read (a text can be empty, or a copy suspect). The `response` field
    compares the 6 models; the `reasoning` field compares the 4 thinking
    models (D87).

    | Column | Meaning |
    |--------|---------|
    | `<model>` | percent of the texts that hold the category |
    | `chi2`, `p` | the Pearson chi-square (k - 1 df) and its p-value |
    | `p_bonf` | `p` times the number of tests in the whole table (Bonferroni), at most 1 |
    | `significant` | `p_bonf` < `alpha` |
    | `range_pp` | the largest rate minus the smallest rate, in percentage points |
    | `range_lo`, `range_hi` | the 95% interval of `range_pp`, by a bootstrap over the images |
    | `ratio` | the largest rate over the smallest rate (inf when the smallest is 0) |

    A category that no model uses is not tested and does not count in the
    Bonferroni factor. The weights of pooled mode do not apply: each model
    reads the same images.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for case in CASES:
        for field in FIELDS:
            models = list(MODELS if field == "response" else THINKING_MODELS)
            read = raw[(raw.case == case) & (raw.field == field) & raw.model.isin(models)]
            have = read.groupby("image_idx").model.nunique()
            images = np.sort(have[have == len(models)].index.to_numpy())
            if not len(images):
                continue
            pos = pd.Series(np.arange(len(images)), index=images)
            part = df[(df.case == case) & (df.field == field) & df.model.isin(models)
                      & df.image_idx.isin(images)]
            n = len(images)
            # 1 resample of the images is 1 row of counts; a rate is (counts @ x) / n.
            w = rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float32)
            for name, mask in difference_categories(part).items():
                hit = part.loc[mask, ["image_idx", "model"]].drop_duplicates()
                if hit.empty:
                    continue
                x = np.zeros((n, len(models)), dtype=np.float32)
                x[pos[hit.image_idx].to_numpy(), [models.index(m) for m in hit.model]] = 1.0
                rate = x.mean(axis=0)
                stat, p = chi_square(x)
                boot = (w @ x) / n
                spread = boot.max(axis=1) - boot.min(axis=1)
                lo, hi = np.percentile(spread, [2.5, 97.5])
                rows.append({
                    "case": case, "field": field, "category": name, "images": n,
                    **{m: round(100 * float(r), 1) for m, r in zip(models, rate)},
                    "chi2": round(stat, 1), "p": p,
                    "range_pp": round(100 * float(rate.max() - rate.min()), 1),
                    "range_lo": round(100 * float(lo), 1), "range_hi": round(100 * float(hi), 1),
                    "ratio": round(float(rate.max() / rate.min()), 2) if rate.min() > 0 else float("inf"),
                })
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out.p * len(out)).clip(upper=1.0)
    out["significant"] = out.p_bonf < alpha
    front = ["case", "field", "category", "images"]
    cols = front + [m for m in MODELS if m in out] + ["chi2", "p", "p_bonf", "significant",
                                                     "range_pp", "range_lo", "range_hi", "ratio"]
    return out[cols]


def model_difference_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count, for each (case, field), the tested categories, the significant ones, and their range."""
    g = table.groupby(["case", "field"], sort=False)
    out = g.agg(tests=("category", "size"), significant=("significant", "sum"),
                images=("images", "first")).reset_index()
    sig = table[table.significant].groupby(["case", "field"], sort=False).range_pp
    out = out.merge(sig.median().rename("median_range_pp").reset_index(), how="left")
    out = out.merge(sig.max().rename("max_range_pp").reset_index(), how="left")
    return out


def word_counts_per_text(traces: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each (model, case, image_idx, field) with the words of the text."""
    rows = []
    for field in FIELDS:
        rows.append(traces[["model", "case", "image_idx"]].assign(
            field=field, words=traces[field].str.split().str.len().fillna(0).astype(int)))
    return pd.concat(rows, ignore_index=True)


def model_differences_by_length(df: pd.DataFrame, raw: pd.DataFrame, traces: pd.DataFrame,
                                bands: int = 4, min_texts: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    """Test the categories of `model_differences` again inside bands of text length.

    This is a secondary check. The primary unit stays 1 completion: a model
    that writes longer texts does so as part of the model. The check asks if
    the models still differ when their texts have about the same length.

    The bands are the quantiles of the word count of all texts that were read
    in 1 (case, field), all models together. Inside 1 band, a model enters
    the test when it has `min_texts` texts or more there. The test is the
    Pearson chi-square test of the 2 x k table, with Bonferroni over every
    test of the whole table. The texts of a band come from different images
    for different models, thus the samples are not paired.
    """
    from scipy.stats import chi2_contingency

    words = word_counts_per_text(traces)
    rows = []
    for case in CASES:
        for field in FIELDS:
            models = list(MODELS if field == "response" else THINKING_MODELS)
            read = (raw[(raw.case == case) & (raw.field == field) & raw.model.isin(models)]
                    [["model", "image_idx"]].drop_duplicates())
            read = read.merge(words[(words.case == case) & (words.field == field)],
                              on=["model", "image_idx"], how="left")
            if read.empty:
                continue
            edges = np.unique(np.quantile(read.words, np.linspace(0, 1, bands + 1)))
            read["band"] = pd.cut(read.words, edges, include_lowest=True, labels=False)
            part = df[(df.case == case) & (df.field == field) & df.model.isin(models)]
            n = read.groupby(["band", "model"]).size()
            for name, mask in difference_categories(part).items():
                hit = part.loc[mask, ["model", "image_idx"]].drop_duplicates()
                hit = hit.merge(read[["model", "image_idx", "band"]], on=["model", "image_idx"])
                h = hit.groupby(["band", "model"]).size()
                for b in sorted(read.band.dropna().unique()):
                    ms = [m for m in models if n.get((b, m), 0) >= min_texts]
                    if len(ms) < 2:
                        continue
                    hits = np.array([h.get((b, m), 0) for m in ms])
                    if hits.sum() == 0:
                        continue
                    totals = np.array([n[(b, m)] for m in ms])
                    table = np.vstack([hits, totals - hits])
                    p = 1.0 if (table.sum(axis=1) == 0).any() else float(
                        chi2_contingency(table, correction=False)[1])
                    rate = hits / totals
                    lo, hi = int(edges[int(b)]), int(edges[int(b) + 1])
                    rows.append({
                        "case": case, "field": field, "band": int(b) + 1, "words": f"{lo}-{hi}",
                        "category": name, "models": len(ms),
                        **{m: round(100 * float(r), 1) for m, r in zip(ms, rate)},
                        **{f"n_{m}": int(c) for m, c in zip(ms, totals)},
                        "p": p, "range_pp": round(100 * float(rate.max() - rate.min()), 1),
                    })
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out.p * len(out)).clip(upper=1.0)
    out["significant"] = out.p_bonf < alpha
    return out


def length_summary(by_length: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Count the categories that differ in each length band, next to the count without bands."""
    g = by_length.groupby(["case", "field", "band"], sort=False)
    out = g.agg(words=("words", "first"), models=("models", "max"), tests=("category", "size"),
                significant=("significant", "sum")).reset_index()
    sig = by_length[by_length.significant].groupby(["case", "field", "band"], sort=False).range_pp.median()
    out = out.merge(sig.rename("median_range_pp").reset_index(), how="left")
    base = model_difference_summary(overall)[["case", "field", "tests", "significant"]].rename(
        columns={"tests": "tests_all_lengths", "significant": "significant_all_lengths"})
    return out.merge(base, how="left")


# ---------------------------------------------------------------------------
# The NTA tests: does the model change a finding about places?
# ---------------------------------------------------------------------------

# The socioeconomic features of each NTA (`research/scripts/f4/nta_acs.py`, ACS 2024 5-year).
ACS_PATH = REPO_ROOT / "data" / "f4_traces" / "nta_acs.parquet"
NTA_FEATURES = ("pct_black_nh", "pct_white_nh", "pct_asian_nh", "pct_bachelors", "median_household_income")


def image_ntas() -> pd.DataFrame:
    """Give the NTA of each image: (case, image_idx, nta2020). Restaurant and caption share the images."""
    r = pd.read_parquet(F4 / "f4_restaurants_nta.parquet", columns=["image_idx", "NTA2020"]).rename(
        columns={"NTA2020": "nta2020"})
    s = pd.read_parquet(F4 / "f4_street_1k.parquet", columns=["image_idx", "nta2020"])
    return pd.concat([r.assign(case="restaurant"), r.assign(case="caption"), s.assign(case="street")],
                     ignore_index=True).dropna(subset=["nta2020"])


def text_category_matrix(df: pd.DataFrame, raw: pd.DataFrame, case: str, field: str) -> pd.DataFrame:
    """Give 1 row for each text of the complete images of 1 (case, field), and 1 0/1 column for each category.

    A complete image has a text that was read for every model of the field
    (the 6 models for `response`, the 4 thinking models for `reasoning`), as
    in `model_differences`. The rows carry `model`, `image_idx`, `nta2020`.
    """
    models = list(MODELS if field == "response" else THINKING_MODELS)
    read = raw[(raw.case == case) & (raw.field == field) & raw.model.isin(models)]
    have = read.groupby("image_idx").model.nunique()
    images = have[have == len(models)].index
    texts = read[read.image_idx.isin(images)][["model", "image_idx"]].drop_duplicates()
    ntas = image_ntas()
    texts = texts.merge(ntas[ntas.case == case][["image_idx", "nta2020"]], on="image_idx", how="inner")
    part = df[(df.case == case) & (df.field == field) & df.model.isin(models) & df.image_idx.isin(images)]
    cols = {}
    for name, mask in difference_categories(part).items():
        hit = part.loc[mask, ["model", "image_idx"]].drop_duplicates()
        if hit.empty:
            continue
        key = pd.MultiIndex.from_frame(hit)
        cols[name] = pd.MultiIndex.from_frame(texts[["model", "image_idx"]]).isin(key).astype(np.int8)
    return pd.concat([texts.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


def difference_test(est_a, se_a, est_b, se_b):
    """Test the difference between 2 independent estimates (Altman and Bland, BMJ 2003).

    z = (est_a - est_b) / sqrt(se_a ** 2 + se_b ** 2), with a 2-sided normal
    p. Use it to ask whether a finding in case A is different from the same
    finding in case B. For 2 Spearman or Pearson correlations, give
    `np.arctanh(r)` as the estimate and `1 / sqrt(n - 3)` as the standard error
    (the Fisher z test). The inputs can be scalars or arrays. Returns (z, p).
    """
    from scipy.stats import norm

    z = (np.asarray(est_a, float) - np.asarray(est_b, float)) / np.sqrt(
        np.asarray(se_a, float) ** 2 + np.asarray(se_b, float) ** 2)
    return z, 2 * norm.sf(np.abs(z))


def _spearman(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Give the Spearman correlation of each column of `a` with the same column of `b` (rows: NTAs)."""
    ra, rb = a.rank(), b.rank()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return (ra * rb).sum() / den.replace(0, np.nan)


def nta_rank_agreement(mat: pd.DataFrame, min_images: int = 20, splits: int = 50,
                       seed: int = 20261027) -> pd.DataFrame:
    """Compare the NTA ranks of 2 models with the NTA ranks of 1 model on other images.

    For each split, the images of each NTA go at random into 2 halves, the
    same halves for every model. The rate of a category in 1 NTA is the share
    of texts that hold it. Then, for each category:

    | Column | Meaning |
    |--------|---------|
    | `within` | the Spearman correlation over the NTAs of 1 model on half A with the same model on half B; the mean over the models and the splits. The agreement that noise permits |
    | `between` | the same, with 2 different models on half A and half B; the mean over the pairs of models and the splits |
    | `gap` | `within` minus `between`. Above 0: the model changes the NTA ranks beyond the noise of the images |
    | `splits_between_lower` | the share of splits where `between` < `within` |

    Only NTAs with `min_images` complete images or more enter (each half then
    holds half of them). Both correlations use halves of the same size, thus
    the noise is the same in both.
    """
    rng = np.random.default_rng(seed)
    cats = [c for c in mat.columns if c not in ("model", "image_idx", "nta2020")]
    models = [m for m in MODELS if m in set(mat.model)]
    per = mat.drop_duplicates("image_idx").groupby("nta2020").size()
    keep = per[per >= min_images].index
    mat = mat[mat.nta2020.isin(keep)]
    imgs = mat.drop_duplicates("image_idx")[["image_idx", "nta2020"]]
    within, between = [], []
    for _ in range(splits):
        # A random half of the images of each NTA is half A.
        order = imgs.assign(_r=rng.random(len(imgs))).sort_values(["nta2020", "_r"])
        order["_k"] = order.groupby("nta2020").cumcount()
        order["half"] = np.where(order._k < order.nta2020.map(per) // 2, "A", "B")
        m = mat.merge(order[["image_idx", "half"]], on="image_idx")
        rates = m.groupby(["model", "half", "nta2020"])[cats].mean()
        w = [_spearman(rates.loc[(x, "A")], rates.loc[(x, "B")]) for x in models]
        b = [_spearman(rates.loc[(x, "A")], rates.loc[(y, "B")]) for x in models for y in models if x != y]
        within.append(pd.concat(w, axis=1).mean(axis=1))
        between.append(pd.concat(b, axis=1).mean(axis=1))
    w, b = pd.concat(within, axis=1), pd.concat(between, axis=1)
    out = pd.DataFrame({"ntas": len(keep), "within": w.mean(axis=1), "between": b.mean(axis=1)})
    out["gap"] = out.within - out.between
    out["splits_between_lower"] = (b < w).mean(axis=1)
    out["rate_all_models"] = 100 * mat[cats].mean()
    return out.round(3).rename_axis("category").reset_index()


def nta_feature_correlations(mat: pd.DataFrame, min_images: int = 20, alpha: float = 0.05,
                             features: Sequence[str] = NTA_FEATURES) -> pd.DataFrame:
    """Correlate the NTA rate of each category with each NTA feature, for each model.

    The test is the Spearman correlation over the NTAs (`scipy.stats.spearmanr`),
    1 test for each (category, model, feature), with Bonferroni over every test
    of the table. Only NTAs with `min_images` complete images or more and with
    residents (ACS population above 0) enter. `p_bonf` < `alpha` marks a
    significant correlation; `sign` is +1 or -1 for a significant one, else 0.
    """
    from scipy.stats import spearmanr

    acs = pd.read_parquet(ACS_PATH)
    acs = acs[acs.population > 0].set_index("nta2020")
    cats = [c for c in mat.columns if c not in ("model", "image_idx", "nta2020")]
    per = mat.drop_duplicates("image_idx").groupby("nta2020").size()
    keep = [n for n in per[per >= min_images].index if n in acs.index]
    rates = mat[mat.nta2020.isin(keep)].groupby(["model", "nta2020"])[cats].mean()
    rows = []
    for model in [m for m in MODELS if m in rates.index.get_level_values(0)]:
        r = rates.loc[model]
        for f in features:
            x = acs.loc[r.index, f]
            for c in cats:
                if r[c].nunique() < 2:
                    continue
                rho, p = spearmanr(r[c], x)
                rows.append({"category": c, "model": model, "feature": f, "ntas": len(r),
                             "rho": round(float(rho), 3), "p": float(p)})
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out.p * len(out)).clip(upper=1.0)
    out["sign"] = np.where(out.p_bonf < alpha, np.sign(out.rho), 0).astype(int)
    return out


def feature_disagreement(cor: pd.DataFrame) -> pd.DataFrame:
    """Give, for each (category, feature), the sign of each model: +1, -1, or 0 (not significant).

    `disagree` is true when 1 model has a significant correlation and another
    model has none, or the opposite sign.
    """
    keys = [k for k in ("case", "field") if k in cor.columns] + ["category", "feature"]
    wide = cor.pivot_table(index=keys, columns="model", values="sign", aggfunc="first")
    wide = wide[[m for m in MODELS if m in wide.columns]]
    signs = wide.copy()
    wide["significant_models"] = (signs.fillna(0) != 0).sum(axis=1)
    wide["disagree"] = (signs.max(axis=1) != signs.min(axis=1))
    return wide.reset_index()


def nta_min_images() -> int:
    """Give the default NTA size for the NTA tests: 4 images for the canonical sample, 20 for a larger one."""
    return 4 if not DATA else 20


# ---------------------------------------------------------------------------
# Strata: the NTA and the cuisine of an image (sections 4.12 and 4.13)
# ---------------------------------------------------------------------------

# The DOHMH table of the inspected restaurants: `camis` -> `cuisine_description`.
DOHMH_PATH = REPO_ROOT / "curation" / "dohmh_restaurants_inspected_all" / "restaurants_aggregated.parquet"
SMALL_CUISINES = "other (small)"
STRATA = ("nta", "cuisine")
VALENCES = ("good", "bad", "neutral")


def image_cuisines(min_images: int = 100) -> pd.DataFrame:
    """Give the cuisine of each restaurant image: `image_idx`, `cuisine`.

    The cuisine is the DOHMH `cuisine_description` of the restaurant
    (`camis`). A cuisine with fewer than `min_images` images goes into the
    group `other (small)`. The street views have no cuisine.
    """
    img = pd.read_parquet(F4 / "f4_restaurants_nta.parquet", columns=["image_idx", "camis"])
    doh = pd.read_parquet(DOHMH_PATH, columns=["camis", "cuisine_description"]).drop_duplicates("camis")
    img["camis"] = img.camis.astype(str)
    doh["camis"] = doh.camis.astype(str)
    out = img.merge(doh, on="camis", how="left", validate="many_to_one")
    out["cuisine"] = out.cuisine_description.fillna("unknown")
    n = out.cuisine.value_counts()
    out.loc[out.cuisine.isin(n[n < min_images].index), "cuisine"] = SMALL_CUISINES
    return out[["image_idx", "cuisine"]]


def image_strata(case: str, by: str, min_images: int, min_cuisine: int = 100) -> pd.DataFrame:
    """Give the stratum of each image of `case`: `image_idx`, `group`.

    `by` is `nta` (the NTAs with `min_images` images or more) or `cuisine`
    (restaurant and caption only; `image_cuisines`, without the group of
    the small cuisines).
    """
    if by == "nta":
        n = image_ntas()
        n = n[n.case == case][["image_idx", "nta2020"]].rename(columns={"nta2020": "group"})
        size = n.group.value_counts()
        return n[n.group.isin(size[size >= min_images].index)].reset_index(drop=True)
    if case not in ("restaurant", "caption"):
        raise ValueError(f"{case}: the street views have no cuisine")
    c = image_cuisines(min_cuisine).rename(columns={"cuisine": "group"})
    return c[c.group != SMALL_CUISINES].reset_index(drop=True)


def _complete_texts(raw: pd.DataFrame, case: str, field: str) -> pd.DataFrame:
    """Give the (model, image_idx) of the texts of the complete images, as `text_category_matrix`."""
    models = list(MODELS if field == "response" else THINKING_MODELS)
    read = raw[(raw.case == case) & (raw.field == field) & raw.model.isin(models)]
    have = read.groupby("image_idx").model.nunique()
    images = have[have == len(models)].index
    return read[read.image_idx.isin(images)][["model", "image_idx"]].drop_duplicates().reset_index(drop=True)


def _chi_square_rows(table: pd.DataFrame) -> dict:
    """Give the Pearson chi-square test of a groups x outcomes count table, with Cochran's rule.

    The rule: the approximation is fair when no expected count is under 1
    and 80% of the expected counts or more are 5 or more.
    """
    from scipy.stats import chi2_contingency

    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return {"groups": t.shape[0], "chi2": np.nan, "dof": 0, "p": np.nan, "cochran_ok": False}
    stat, p, dof, exp = chi2_contingency(t.to_numpy(), correction=False)
    ok = bool(exp.min() >= 1 and (exp >= 5).mean() >= 0.8)
    return {"groups": t.shape[0], "chi2": round(float(stat), 1), "dof": int(dof), "p": float(p), "cochran_ok": ok}


def _bonferroni_sign(out: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out["p_bonf"] = (out.p * out.p.notna().sum()).clip(upper=1.0)
    out["significant"] = out.p_bonf < alpha
    return out


def _nta_feature_spearman(values: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Correlate NTA values with the ACS features, as test 2 of `nta_feature_correlations`.

    `values` has the columns `model`, `group` (the NTA), `measure`, `value`.
    1 Spearman test for each (model, measure, feature), over the NTAs with
    residents; Bonferroni over the table.
    """
    from scipy.stats import spearmanr

    acs = pd.read_parquet(ACS_PATH)
    acs = acs[acs.population > 0].set_index("nta2020")
    rows = []
    for (model, measure), g in values.groupby(["model", "measure"], sort=False):
        g = g.set_index("group").value.dropna()
        g = g[g.index.isin(acs.index)]
        if len(g) < 3 or g.nunique() < 2:
            continue
        for f in NTA_FEATURES:
            rho, p = spearmanr(g, acs.loc[g.index, f])
            rows.append({"model": model, "measure": measure, "feature": f, "ntas": len(g),
                         "rho": round(float(rho), 3), "p": float(p)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _bonferroni_sign(out, alpha)
    out["sign"] = np.where(out.significant, np.sign(out.rho), 0).astype(int)
    return out.drop(columns="significant")


def valence_by_stratum(df: pd.DataFrame, raw: pd.DataFrame, case: str, field: str, by: str,
                       min_images: int = 20, min_cuisine: int = 100) -> pd.DataFrame:
    """Give the valence of the `visual_evidence` spans of each category, for each stratum and model.

    The spans come from the complete images of the (case, field), as in
    section 4.8. 1 row for each (model, group, category): `spans`, the
    counts `good`, `bad`, `neutral`, their percent, and `net` = (good -
    bad) / spans, from -1 (all bad) to +1 (all good). `all models` is the
    models together. The category `any` holds every category.
    """
    texts = _complete_texts(raw, case, field)
    strata = image_strata(case, by, min_images, min_cuisine)
    ve = df[(df.case == case) & (df.field == field) & (df.extraction_class == "visual_evidence")
            & df.valence.isin(VALENCES)][["model", "image_idx", "category", "valence"]]
    ve = ve.merge(texts, on=["model", "image_idx"]).merge(strata, on="image_idx")
    ve = pd.concat([ve, ve.assign(category="any")])
    ve = pd.concat([ve, ve.assign(model="all models")])
    n = ve.groupby(["model", "group", "category", "valence"]).size().unstack("valence", fill_value=0)
    n = n.reindex(columns=list(VALENCES), fill_value=0)
    n["spans"] = n.sum(axis=1)
    for v in VALENCES:
        n[f"pct_{v}"] = (100 * n[v] / n.spans).round(1)
    n["net"] = ((n.good - n.bad) / n.spans).round(3)
    return n.reset_index()


def valence_tests(table: pd.DataFrame, by: str, alpha: float = 0.05) -> pd.DataFrame:
    """Test if the valence mix of a category differs across the strata, for each model.

    The Pearson chi-square test of the groups x (good, bad, neutral) span
    counts, 1 test for each (model, category); Bonferroni over the table.
    `cochran_ok` is false when the expected counts are too small for the
    test. `all models` is not tested: the models read the same images.
    Warning: 1 text can hold many spans, and the spans of 1 text are not
    independent. Thus the p values are too small; read `chi2` with care.
    """
    rows = []
    for (model, cat), g in table[table.model != "all models"].groupby(["model", "category"], sort=False):
        rows.append({"model": model, "category": cat, "by": by,
                     **_chi_square_rows(g.set_index("group")[list(VALENCES)])})
    return _bonferroni_sign(pd.DataFrame(rows), alpha)


def valence_nta_features(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Correlate the NTA `net` valence of each category with the ACS features (Spearman, Bonferroni).

    The NTA is the unit, thus the spans of 1 text do not make the test too
    strong. `table` is `valence_by_stratum` with `by="nta"`.
    """
    v = table[table.model != "all models"].rename(columns={"category": "measure", "net": "value"})
    return _nta_feature_spearman(v[["model", "group", "measure", "value"]], alpha)


def uncertainty_by_stratum(df: pd.DataFrame, raw: pd.DataFrame, case: str, field: str, by: str,
                           min_images: int = 20, min_cuisine: int = 100) -> pd.DataFrame:
    """Give the share of texts with an uncertainty span, for each stratum and model.

    The unit is 1 text of a complete image (section 4.8). The outcome is 1
    when the text holds an uncertainty span (`any`) or an uncertainty span
    of 1 reason. 1 row for each (model, group, measure): `texts`, `hits`,
    `pct`. `all models` is the models together.
    """
    mat = text_category_matrix(df, raw, case, field)
    cols = [c for c in mat.columns if c.startswith("uncertainty: ")]
    strata = image_strata(case, by, min_images, min_cuisine)
    m = mat[["model", "image_idx"] + cols].merge(strata, on="image_idx")
    m = pd.concat([m, m.assign(model="all models")])
    long = m.melt(id_vars=["model", "image_idx", "group"], value_vars=cols, var_name="measure", value_name="hit")
    long["measure"] = long.measure.str.replace("uncertainty: ", "", regex=False).str.replace("reason = ", "", regex=False)
    out = long.groupby(["model", "group", "measure"]).hit.agg(texts="size", hits="sum").reset_index()
    out["pct"] = (100 * out.hits / out.texts).round(2)
    return out


def uncertainty_tests(table: pd.DataFrame, by: str, alpha: float = 0.05) -> pd.DataFrame:
    """Test if the share of texts with uncertainty differs across the strata, for each model.

    The Pearson chi-square test of the groups x (hit, no hit) text counts, 1
    test for each (model, measure); Bonferroni over the table. Each model
    gives 1 text for each image, thus the texts of 1 model are independent.
    `all models` is not tested.
    """
    rows = []
    for (model, measure), g in table[table.model != "all models"].groupby(["model", "measure"], sort=False):
        t = g.set_index("group")[["hits"]].assign(misses=g.set_index("group").texts - g.set_index("group").hits)
        rows.append({"model": model, "measure": measure, "by": by, **_chi_square_rows(t)})
    return _bonferroni_sign(pd.DataFrame(rows), alpha)


def uncertainty_nta_features(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Correlate the NTA share of texts with uncertainty with the ACS features (Spearman, Bonferroni)."""
    v = table[table.model != "all models"].assign(value=lambda d: d.hits / d.texts)
    return _nta_feature_spearman(v[["model", "group", "measure", "value"]], alpha)


def cuisine_nta_mix(case: str = "restaurant", min_images: int = 20, min_cuisine: int = 100) -> pd.DataFrame:
    """Correlate the share of each cuisine in an NTA with the ACS features (Spearman, Bonferroni).

    This shows how much the cuisine and the NTA features go together. When
    they do, a cuisine difference and an NTA feature correlation can be 1
    effect, not 2.
    """
    n = image_strata(case, "nta", min_images)
    c = image_cuisines(min_cuisine)
    m = n.merge(c, on="image_idx")
    share = pd.crosstab(m.group, m.cuisine, normalize="index")
    v = share.stack().rename("value").reset_index().rename(columns={"cuisine": "measure"})
    return _nta_feature_spearman(v.assign(model="images")[["model", "group", "measure", "value"]])


def nta_tests(df: pd.DataFrame, raw: pd.DataFrame, min_images: Optional[int] = None, alpha: float = 0.05,
              splits: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run `nta_rank_agreement` and `nta_feature_correlations` on every (case, field).

    Returns (agreement, correlations). `correlations` gets Bonferroni over
    every test of all cases and fields. The rate of 1 NTA is not weighted: in
    pooled mode, the weight of a restaurant image is the same for each image
    of 1 NTA, thus it has no effect inside the NTA.
    """
    min_images = nta_min_images() if min_images is None else min_images
    ag, cor = [], []
    for case in CASES:
        for field in FIELDS:
            mat = text_category_matrix(df, raw, case, field)
            if mat.empty or mat.model.nunique() < 2:
                continue
            ag.append(nta_rank_agreement(mat, min_images=min_images, splits=splits).assign(case=case, field=field))
            c = nta_feature_correlations(mat, min_images=min_images, alpha=alpha)
            if len(c):
                cor.append(c.assign(case=case, field=field))
    ag = pd.concat(ag, ignore_index=True) if ag else pd.DataFrame()
    cor = pd.concat(cor, ignore_index=True) if cor else pd.DataFrame()
    if len(cor):
        cor["p_bonf"] = (cor.p * len(cor)).clip(upper=1.0)
        cor["sign"] = np.where(cor.p_bonf < alpha, np.sign(cor.rho), 0).astype(int)
    front = ["case", "field"]
    ag = ag[front + [c for c in ag.columns if c not in front]] if len(ag) else ag
    cor = cor[front + [c for c in cor.columns if c not in front]] if len(cor) else cor
    return ag, cor


# ---------------------------------------------------------------------------
# Findings in both cases, and findings in 1 case only
# ---------------------------------------------------------------------------

# The 2 cases that this comparison uses. `caption` stays out.
CROSS_CASES = ("restaurant", "street")
# The classes of 1 finding, in the order of the summary table.
CROSS_CLASSES = (
    "both, same sign", "both, opposite sign", "restaurant only", "street only",
    "1 case only, not different", "neither",
)


def model_contrasts(df: pd.DataFrame, raw: pd.DataFrame, cases: Sequence[str] = CROSS_CASES,
                    alpha: float = 0.05) -> pd.DataFrame:
    """Compare each model with the other models, for each (case, field, category) of `model_differences`.

    The unit is the unit of section 4.8: 1 text of 1 complete image, with the
    outcome 1 when the text holds 1 or more spans of the category. The
    categories are the categories that `model_differences` tests. The
    `response` field has 6 models, the `reasoning` field has the 4 thinking
    models.

    | Column | Meaning |
    |--------|---------|
    | `rate` | percent of the texts of the model that hold the category |
    | `rate_others` | the same, for the texts of all the other models together |
    | `est` | `rate` - `rate_others`, in percentage points |
    | `se` | the standard error of `est`, for 2 independent proportions: sqrt(p1 (1 - p1) / n1 + p2 (1 - p2) / n2), in percentage points |
    | `chi2`, `p` | the Pearson chi-square test of the 2 x 2 table (holds it or not, the model or the other models) |
    | `p_bonf` | `p` times the number of tests of the same case (Bonferroni inside each case), at most 1 |
    | `significant` | `p_bonf` < `alpha` |

    The rates are not weighted, as in section 4.8: each model reads the same
    images, thus the plain binomial standard error applies. The model and the
    other models read the same images, thus the 2 proportions are not truly
    independent. The test ignores this, as the test of section 4.8 does.
    """
    from scipy.stats import chi2_contingency

    rows = []
    for case in cases:
        for field in FIELDS:
            models = list(MODELS if field == "response" else THINKING_MODELS)
            read = raw[(raw.case == case) & (raw.field == field) & raw.model.isin(models)]
            have = read.groupby("image_idx").model.nunique()
            images = have[have == len(models)].index
            n = len(images)
            if not n:
                continue
            part = df[(df.case == case) & (df.field == field) & df.model.isin(models)
                      & df.image_idx.isin(images)]
            for name, mask in difference_categories(part).items():
                hit = part.loc[mask, ["image_idx", "model"]].drop_duplicates()
                if hit.empty:
                    continue
                count = hit.model.value_counts()
                for m in models:
                    h_m = int(count.get(m, 0))
                    h_o = int(count.sum()) - h_m
                    n_o = n * (len(models) - 1)
                    p_m, p_o = h_m / n, h_o / n_o
                    table = np.array([[h_m, h_o], [n - h_m, n_o - h_o]])
                    stat, p = (0.0, 1.0) if (table.sum(axis=1) == 0).any() else (
                        float(v) for v in chi2_contingency(table, correction=False)[:2])
                    rows.append({
                        "case": case, "field": field, "category": name, "model": m, "images": n,
                        "rate": 100 * p_m, "rate_others": 100 * p_o, "est": 100 * (p_m - p_o),
                        "se": 100 * float(np.sqrt(p_m * (1 - p_m) / n + p_o * (1 - p_o) / n_o)),
                        "chi2": stat, "p": p,
                    })
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out.p * out.groupby("case").p.transform("size")).clip(upper=1.0)
    out["significant"] = out.p_bonf < alpha
    return out


def _cross_class(sig_a: pd.Series, sig_b: pd.Series, est_a: pd.Series, est_b: pd.Series,
                 diff_sig: pd.Series, cases: Sequence[str] = CROSS_CASES) -> pd.Series:
    """Give the class of each finding (`CROSS_CLASSES`). Case A is `cases[0]`, case B is `cases[1]`."""
    same = np.sign(est_a) == np.sign(est_b)
    out = np.select(
        [sig_a & sig_b & same, sig_a & sig_b,
         sig_a & ~sig_b & diff_sig, ~sig_a & sig_b & diff_sig, sig_a ^ sig_b],
        ["both, same sign", "both, opposite sign",
         f"{cases[0]} only", f"{cases[1]} only", "1 case only, not different"],
        default="neither")
    return pd.Series(out, index=sig_a.index)


def crosscase_rates(contrasts: pd.DataFrame, alpha: float = 0.05,
                    cases: Sequence[str] = CROSS_CASES) -> pd.DataFrame:
    """Put the model contrasts of 2 cases side by side and classify each finding.

    1 row for each (field, category, model) that exists in both cases. The
    estimate of 1 case is `est` of `model_contrasts` (the rate of the model
    minus the rate of the other models, in percentage points). The
    significance in 1 case is `significant` of `model_contrasts` (Bonferroni
    inside the case). `diff_z`, `diff_p` come from `difference_test` on the
    2 estimates; `diff_p_bonf` is `diff_p` times the rows of this table. The
    rule of `class`:

    | Class | Rule |
    |-------|------|
    | `both, same sign` | significant in each case, the same sign |
    | `both, opposite sign` | significant in each case, opposite signs |
    | `<case> only` | significant in that case only, and `diff_p_bonf` < `alpha` |
    | `1 case only, not different` | significant in 1 case only, and `diff_p_bonf` >= `alpha` |
    | `neither` | significant in no case |

    `df.attrs["one_case_only"]` gives, for each case, the (field, category,
    model) findings that exist in that case only. They are not in the table.
    """
    a, b = cases
    keys = ["field", "category", "model"]
    cols = keys + ["est", "se", "p_bonf", "significant"]
    left = contrasts[contrasts.case == a][cols + ["images"]]
    right = contrasts[contrasts.case == b][cols + ["images"]]
    out = left.merge(right, on=keys, suffixes=(f"_{a}", f"_{b}"))
    z, p = difference_test(out[f"est_{a}"], out[f"se_{a}"], out[f"est_{b}"], out[f"se_{b}"])
    out["diff_z"], out["diff_p"] = z, p
    out["diff_p_bonf"] = (out.diff_p * len(out)).clip(upper=1.0)
    out["class"] = _cross_class(out[f"significant_{a}"], out[f"significant_{b}"], out[f"est_{a}"],
                                out[f"est_{b}"], out.diff_p_bonf < alpha, cases)
    out.insert(3, "family", out.model.map(MODEL_FAMILY))
    for c in out.columns:
        if c.startswith(("est_", "se_")):
            out[c] = out[c].round(2)
    out["diff_z"] = out.diff_z.round(2)
    out.attrs["one_case_only"] = {
        c: len(contrasts[contrasts.case == c][keys].merge(out[keys], how="left", indicator=True)
               .query("_merge == 'left_only'")) for c in cases}
    return out.reset_index(drop=True)


def crosscase_correlations(cor: pd.DataFrame, alpha: float = 0.05,
                           cases: Sequence[str] = CROSS_CASES) -> pd.DataFrame:
    """Put the NTA feature correlations (test 2 of `nta_tests`) of 2 cases side by side.

    1 row for each (field, category, model, feature) that exists in both
    cases. The significance in 1 case is the `sign` of `nta_tests` (its
    Bonferroni, over every test of all cases and fields), thus this table
    agrees with the sign table of section 4.10. The difference test is the
    Fisher z form of `difference_test`: the estimate is `arctanh(rho)`, the
    standard error is `1 / sqrt(ntas - 3)`. `diff_p_bonf` is `diff_p` times
    the rows of this table. `class` follows the rule of `crosscase_rates`.
    """
    a, b = cases
    keys = ["field", "category", "model", "feature"]
    cols = keys + ["ntas", "rho", "p_bonf", "sign"]
    out = cor[cor.case == a][cols].merge(cor[cor.case == b][cols], on=keys, suffixes=(f"_{a}", f"_{b}"))

    def fisher(c: str):
        r = np.arctanh(out[f"rho_{c}"].clip(-0.9999, 0.9999))
        return r, 1 / np.sqrt(out[f"ntas_{c}"] - 3)

    z, p = difference_test(*fisher(a), *fisher(b))
    out["diff_z"], out["diff_p"] = np.round(z, 2), p
    out["diff_p_bonf"] = (out.diff_p * len(out)).clip(upper=1.0)
    out["class"] = _cross_class(out[f"sign_{a}"] != 0, out[f"sign_{b}"] != 0, out[f"rho_{a}"],
                                out[f"rho_{b}"], out.diff_p_bonf < alpha, cases)
    out.insert(3, "family", out.model.map(MODEL_FAMILY))
    return out.reset_index(drop=True)


def crosscase_ranks(agree: pd.DataFrame, cases: Sequence[str] = CROSS_CASES) -> pd.DataFrame:
    """Put test 1 of `nta_tests` (`within`, `between`, `gap`) of 2 cases side by side.

    1 row for each (field, category) that exists in both cases. This table
    is a description only: it has no test of a difference between the cases.
    """
    a, b = cases
    keys = ["field", "category"]
    cols = keys + ["ntas", "within", "between", "gap"]
    return (agree[agree.case == a][cols].merge(agree[agree.case == b][cols], on=keys, suffixes=(f"_{a}", f"_{b}"))
            .reset_index(drop=True))


def crosscase_summary(rates: pd.DataFrame, correlations: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Count the findings of each class, for each field and for each model family.

    1 row for each (finding, field, family). `finding` is `rates` (the model
    contrasts of section 4.8) or `nta_features` (test 2 of section 4.10).
    The family `all` gives the sum over the families of the field.
    """
    parts = [("rates", rates)] + ([("nta_features", correlations)] if correlations is not None
                                  and len(correlations) else [])
    rows = []
    for finding, t in parts:
        for field in FIELDS:
            f = t[t.field == field]
            if f.empty:
                continue
            for family in ["all"] + [x for x in ("gemma", "qwen", "llama") if x in set(f.family)]:
                g = f if family == "all" else f[f.family == family]
                counts = g["class"].value_counts()
                rows.append({"finding": finding, "field": field, "family": family, "findings": len(g),
                             **{c: int(counts.get(c, 0)) for c in CROSS_CLASSES}})
    return pd.DataFrame(rows)


def plot_crosscase(rates: pd.DataFrame, cases: Sequence[str] = CROSS_CASES):
    """Draw the model contrast of case A (x) against case B (y), 1 panel for each field.

    1 dot for each (category, model), with the hue and the marker of the
    model (`MODEL_COLORS`, `MODEL_MARKERS`). The dashed line is y = x. The 2
    grey lines are x = 0 and y = 0. Return the figure.
    """
    import matplotlib.pyplot as plt

    a, b = cases
    fields = [f for f in FIELDS if f in set(rates.field)]
    if not fields:
        raise ValueError("no finding to draw")
    fig, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 4.2), squeeze=False)
    top = float(np.nanmax(np.abs(rates[[f"est_{a}", f"est_{b}"]].to_numpy()))) * 1.05 or 1.0
    for ax, field in zip(axes[0], fields):
        part = rates[rates.field == field]
        ax.axhline(0, color=S.EDGE, alpha=0.35, linewidth=0.8, zorder=0)
        ax.axvline(0, color=S.EDGE, alpha=0.35, linewidth=0.8, zorder=0)
        ax.plot([-top, top], [-top, top], color=S.EDGE, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
        for m in [m for m in MODELS if m in set(part.model)]:
            g = part[part.model == m]
            ax.scatter(g[f"est_{a}"], g[f"est_{b}"], s=18, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                       edgecolor=S.EDGE, linewidth=S.EDGE_LW, label=m, zorder=2)
        ax.set_xlim(-top, top)
        ax.set_ylim(-top, top)
        ax.set_aspect("equal")
        ax.set_title(field, fontsize=9, color=S.EDGE)
        ax.set_xlabel(f"{a}: model minus the other models (pp)", fontsize=8)
        ax.set_ylabel(f"{b}: model minus the other models (pp)", fontsize=8)
    models = [m for m in MODELS if m in set(rates.model)]
    handles, labels = axes[0][-1].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: MODELS.index(labels[i]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=7,
               loc="lower center", ncol=family_columns(models), bbox_to_anchor=(0.5, 0.0))
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def plot_model_differences(table: pd.DataFrame, case: str, field: str, n: int = 20,
                           title: Optional[str] = None):
    """Draw the significant categories of 1 (case, field), largest range first.

    Each row is 1 category: a grey line from the smallest to the largest rate,
    and 1 marker for each model, with 1 hue and 1 shape for each family
    (`MODEL_COLORS`, `MODEL_MARKERS`). Return the figure.
    """
    import matplotlib.pyplot as plt

    part = table[(table.case == case) & (table.field == field) & table.significant]
    part = part.sort_values("range_pp", ascending=False).head(n).iloc[::-1]
    if part.empty:
        raise ValueError(f"no significant category for {case}/{field}")
    models = [m for m in MODELS if m in part and part[m].notna().all()]
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(part) + 1.4))
    for i, (_, r) in enumerate(part.iterrows()):
        vals = r[models].astype(float)
        ax.plot([vals.min(), vals.max()], [i, i], color=S.EDGE, alpha=0.35, linewidth=1.2, zorder=1)
    for m in models:
        ax.scatter(part[m].astype(float), range(len(part)), s=28, color=MODEL_COLORS[m],
                   marker=MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW, label=m, zorder=2)
    ax.set_yticks(range(len(part)))
    ax.set_yticklabels(part.category, fontsize=7)
    ax.set_xlabel(f"percent of {field} texts that hold the category")
    # 1 column for each family: the legend fills by column, thus each column holds 1 family.
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=family_columns(models))
    if title:
        ax.set_title(title, fontsize=10, color=S.EDGE, pad=36)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def distinctive(
    df: pd.DataFrame, scope: str, by: str, case: str, model: str,
    unit: str = "class_text", max_rows: int = 60, min_count: int = 5,
) -> pd.DataFrame:
    """Score what 1 group names that the other groups do not.

    The same groups as the word clouds (`contrast`), and the same score as
    `_extractions.distinctive`. That function keys on `case`, thus this one
    puts the group name into the `case` column of a copy.
    """
    part = _in_scope(df, scope)
    if by == "models":
        part, target = part[part.case == case], model
        part = part.assign(case=part.model)
    elif by == "cases":
        part, target = part[part.model == model], case
    else:
        raise ValueError(f"by must be 'models' or 'cases', got {by!r}")
    return X.distinctive(part, target, unit=unit, max_rows=max_rows,
                         min_count=min_count, exclude_classes=())


def quotes(
    df: pd.DataFrame, scope: str, case: str, model: str, extraction_class: str,
    n: int = 15, seed: int = 777,
) -> pd.DataFrame:
    """Give verbatim spans, with the text id and the offsets that locate them."""
    part = _in_scope(df, scope)
    part = part[(part.case == case) & (part.model == model)
                & (part.extraction_class == extraction_class) & part.is_verbatim]
    if part.empty:
        return pd.DataFrame()
    take = part.sample(n=min(n, len(part)), random_state=seed)
    attrs = [a for c, a in ATTRIBUTES if c == extraction_class and a in take.columns]
    columns = ["doc_id", "dba", "extraction_text", *attrs, "char_start", "char_end"]
    columns = [c for c in columns if c in take.columns]
    return take[columns].reset_index(drop=True)


def extreme_ntas(case: str, feature: str, k: int, min_images: int = 20) -> Dict[str, pd.DataFrame]:
    """Give the `k` NTAs with the lowest and the `k` with the highest value of 1 ACS feature.

    The NTAs are those of the NTA tests (section 4.10): `min_images` images of
    `case` or more, and residents. Return {"bottom": ..., "top": ...}, each
    with the columns `nta2020`, `ntaname`, `boroname`, the feature, `images`.
    """
    acs = pd.read_parquet(ACS_PATH)
    acs = acs[acs.population > 0]
    n = image_ntas()
    size = n[n.case == case].groupby("nta2020").size().rename("images")
    t = acs.merge(size[size >= min_images], left_on="nta2020", right_index=True)
    t = t.dropna(subset=[feature]).sort_values(feature)
    cols = ["nta2020", "ntaname", "boroname", feature, "images"]
    return {"bottom": t.head(k)[cols].reset_index(drop=True),
            "top": t.tail(k).iloc[::-1][cols].reset_index(drop=True)}


def nta_quotes(df: pd.DataFrame, case: str, field: str, category: str, ntas: Sequence[str],
               model: str = "all models", n: int = 20, seed: int = 777,
               traces: Optional[pd.DataFrame] = None, context: int = 120) -> pd.DataFrame:
    """Give `n` random spans of 1 category from the texts of a set of NTAs.

    `category` is a name of `difference_categories` (for example
    `inference: kind = popularity`). With `traces`, the column `context` holds
    the text around the span (`context` characters on each side), with the
    span in [brackets]. `attrs` gives `spans` (the spans to draw from),
    `texts_with` (the texts with 1 of them or more) and, with `traces`,
    `texts` (the texts of the NTAs that are not empty).
    """
    part = df[(df.case == case) & (df.field == field)]
    if model != "all models":
        part = part[part.model == model]
    img = image_ntas()
    img = img[(img.case == case) & img.nta2020.isin(list(ntas))][["image_idx", "nta2020"]]
    part = part.merge(img, on="image_idx")
    if part.empty:
        return pd.DataFrame()
    mask = difference_categories(part).get(category)
    hits = part[mask] if mask is not None else part.iloc[0:0]
    if hits.empty:
        return pd.DataFrame()
    take = hits.sample(n=min(n, len(hits)), random_state=seed)
    names = pd.read_parquet(ACS_PATH).set_index("nta2020").ntaname
    take = take.assign(nta=take.nta2020.map(names))
    cols = ["model", "nta", "dba", "extraction_text"]
    cols += [a for c, a in ATTRIBUTES if c == take.extraction_class.iloc[0] and a in take.columns]
    if traces is not None:
        txt = traces.set_index(["model", "case", "image_idx"])[field]
        def _ctx(r):
            s = txt.get((r.model, r.case, r.image_idx))
            if not isinstance(s, str) or pd.isna(r.char_start):
                return ""
            a, b = int(r.char_start), int(r.char_end)
            return " ".join((s[max(0, a - context):a] + "[" + s[a:b] + "]" + s[b:b + context]).split())
        take["context"] = [_ctx(r) for r in take.itertuples()]
        cols.append("context")
    out = take[[c for c in cols if c in take.columns]].reset_index(drop=True)
    out.attrs["spans"] = len(hits)
    out.attrs["texts_with"] = len(hits[["model", "image_idx"]].drop_duplicates())
    if traces is not None:
        t = traces[(traces.case == case) & traces.image_idx.isin(img.image_idx)]
        if model != "all models":
            t = t[t.model == model]
        out.attrs["texts"] = int(t[field].fillna("").str.strip().ne("").sum())
    return out


def vocabulary_report(df: pd.DataFrame) -> pd.DataFrame:
    """Name the values that `normalize` moved into `other`. See `_extractions`."""
    return X.vocabulary_report(df)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_rates(table: pd.DataFrame, per: int = 100, title: Optional[str] = None,
               unit: str = "spans", of: str = "texts"):
    """Draw a rate table: 1 panel for each case, 1 bar colour for each model (`MODEL_COLORS`).

    The rows keep the order of the table, top to bottom. Return the figure.
    """
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("no span to draw")
    cases = [c for c in CASES if c in table.columns.get_level_values(0)]
    rows = table.index[::-1]
    n_models = max(len(table[c].columns) for c in cases)
    height = 0.16 * len(rows) * max(2, n_models) + 1.4
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), height),
                             sharey=True, squeeze=False)
    top = float(table.max().max()) * 1.05 or 1.0
    for ax, case in zip(axes[0], cases):
        # pandas draws the first column at the bottom of a group. Reverse the
        # columns, thus the bars read top to bottom in the order of the legend.
        part = table[case].loc[rows].iloc[:, ::-1]
        part.plot(kind="barh", ax=ax, width=0.8, edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                  color=[MODEL_COLORS[m] for m in part.columns], legend=False)
        ax.set_title(case, fontsize=9, color=S.EDGE)
        ax.set_xlabel(f"{unit} for each {per} {of}")
        ax.set_xlim(0, top)
        ax.set_ylabel("")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    order = sorted(range(len(labels)), key=lambda i: MODELS.index(labels[i]))
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               frameon=False, fontsize=8, loc="lower center", ncol=family_columns(labels),
               bbox_to_anchor=(0.5, 0.0))
    if title:
        fig.suptitle(title, fontsize=10, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout(rect=(0, 0.04 + 0.02 * ((len(labels) + 2) // 3), 1, 1))
    return fig


def cloud_colormap(model: Optional[str]):
    """Give the word colours of the cloud of `model` (`MODEL_INKS`), or None (the house inks) for a non-model group."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(MODEL_INKS[model], name=f"ink_{model}") if model in MODEL_INKS else None


def plot_cloud(weights: Dict[str, float], title: str, model: Optional[str] = None):
    """Draw 1 word cloud with the house style. The words of a model cloud take the inks of the model.

    Return the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(S.PAPER)
    ax.imshow(T.make_cloud(weights, colormap=cloud_colormap(model)), interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=9, color=S.EDGE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(
    out_dir: str | Path,
    traces: pd.DataFrame,
    raw: Optional[pd.DataFrame],
    stopwords: frozenset,
    mode: str = "distinctive",
    ngram: int = 1,
    unit: str = "class_text",
    max_words: int = 120,
    per: int = 100,
) -> List[str]:
    """Write every table and figure of the notebook, for every scope.

    The word clouds use the `models` contrast: 1 cloud for each (case, model),
    against the other models of the same case.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _csv(frame: Optional[pd.DataFrame], name: str, index: bool = False) -> None:
        if frame is None or frame.empty:
            return
        frame.to_csv(out_dir / name, index=index)
        written.append(name)

    def _png(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        written.append(name)

    _csv(corpus_table(traces), "f4_corpus.csv")
    df = spans(raw) if raw is not None else None
    if df is not None:
        _csv(coverage(raw), "f4_extraction_coverage.csv")
        _csv(vocabulary_report(df), "f4_vocabulary_report.csv")
        diffs = model_differences(df, raw)
        _csv(diffs, "f4_model_differences.csv")
        _csv(model_difference_summary(diffs), "f4_model_difference_summary.csv")
        by_len = model_differences_by_length(df, raw, traces)
        _csv(by_len, "f4_model_differences_by_length.csv")
        _csv(length_summary(by_len, diffs), "f4_model_differences_by_length_summary.csv")
        nta_ag, nta_cor = nta_tests(df, raw)
        _csv(nta_ag, "f4_nta_rank_agreement.csv")
        _csv(nta_cor, "f4_nta_feature_correlations.csv")
        if len(nta_cor):
            _csv(feature_disagreement(nta_cor), "f4_nta_feature_signs.csv")
        # Section 4.11: the findings of restaurant and street, side by side.
        contrasts = model_contrasts(df, raw)
        _csv(contrasts, "f4_model_contrasts.csv")
        cc_rates = crosscase_rates(contrasts)
        cc_cor = crosscase_correlations(nta_cor) if len(nta_cor) else None
        _csv(cc_rates, "f4_crosscase_rates.csv")
        _csv(cc_cor, "f4_crosscase_nta_features.csv")
        if len(nta_ag):
            _csv(crosscase_ranks(nta_ag), "f4_crosscase_nta_ranks.csv")
        _csv(crosscase_summary(cc_rates, cc_cor), "f4_crosscase_summary.csv")
        _png(plot_crosscase(cc_rates), "f4_crosscase_rates.png")
        for case in CASES:
            for field in FIELDS:
                if diffs[(diffs.case == case) & (diffs.field == field)].significant.any():
                    _png(plot_model_differences(diffs, case, field, title=f"{case}, {field}"),
                         f"f4_model_differences_{case}_{field}.png")

    for scope in SCOPES:
        counts = word_counts(traces, scope, stopwords, ngram=ngram)
        for case in CASES:
            groups = contrast(counts, "models", case, "")
            _csv(word_table(groups, mode=mode, max_words=max_words),
                 f"f4_{scope}_{case}_words_{mode}.csv")
            for model in ordered(groups):
                w = T.cloud_weights(groups, model, mode=mode, max_words=max_words)
                if w:
                    p = out_dir / f"f4_{scope}_{case}_{model}_{mode}.png"
                    T.make_cloud(w, width=1600, height=900, colormap=cloud_colormap(model)).to_file(str(p))
                    written.append(p.name)
        if df is None:
            continue
        totals = text_totals(raw, scope)
        table = class_rates(df, totals, scope, per)
        _csv(table, f"f4_{scope}_class_rates.csv", index=True)
        _png(plot_rates(table, per), f"f4_{scope}_class_rates.png")
        if scope == "all":
            tva = trace_against_answer(df, raw)
            _csv(tva, "f4_trace_against_answer.csv")
            for claim in ("inference: any of the 4", "person_reference: any"):
                _png(plot_rates(trace_answer_rows(tva, claim), unit="answers", of="answers", title=claim),
                     f"f4_trace_against_answer_{claim.split(':')[0]}.png")
        table = risk_panel(df, totals, scope, per)
        _csv(table, f"f4_{scope}_risk_panel.csv", index=True)
        _png(plot_rates(table, per), f"f4_{scope}_risk_panel.png")
        for cls, attr in ATTRIBUTES:
            table = attribute_rates(df, totals, scope, cls, attr, per)
            if table.empty:
                continue
            _csv(table, f"f4_{scope}_{cls}_{attr}.csv", index=True)
            _png(plot_rates(table, per), f"f4_{scope}_{cls}_{attr}.png")
        for case in CASES:
            for model in MODELS:
                _csv(distinctive(df, scope, "models", case, model, unit=unit),
                     f"f4_{scope}_{case}_{model}_distinctive_{unit}.csv")
    return written


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


def _in_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    return df if scope == "all" else df[df.field == scope]


def ordered(groups) -> List[str]:
    rank = {g: i for i, g in enumerate(MODELS + CASES)}
    return sorted(groups, key=lambda g: rank.get(g, len(rank)))


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ("case", "model", "field") if c in frame.columns]
    rank = {"case": CASES, "model": MODELS, "field": FIELDS}
    out = frame.copy()
    for k in keys:
        out[k] = pd.Categorical(out[k], categories=rank[k], ordered=True)
    return out.sort_values(keys).astype({k: str for k in keys}).reset_index(drop=True)


def _order_columns(table: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(table.columns, key=lambda cm: (CASES.index(cm[0]), MODELS.index(cm[1])))
    return table[pd.MultiIndex.from_tuples(columns, names=["case", "model"])]
