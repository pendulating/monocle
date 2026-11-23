"""F4: the helpers of the descriptor notebook (`f4_descriptors.py`).

F4 asks 6 models about 1 image at a time. The descriptor runs (D109 to
D124, then the x2 round of D130) extract the words with which a text
characterizes the place, the scene, or the planned photo. 1 row is 1 span
with 6 attributes: `head` (the plain form), `vibe`, `slot` (the frame noun),
`aspect`, `negated`, and `target` (in x2: `place`, `current_photo`,
`planned_photo`; in v1: street only, `place` or `photo`). Only the
`restaurant` and the `street` cases have descriptors; `caption` has none.

Where the data is
-----------------
2 extraction rounds exist. The x2 round (D130, D131) is the main result
and the default. `use_round("v1")`, or the env var
`F4DESC_ROUND=v1`, reads the first round, for history only. The first round
exists for the canonical batch only (`F4_DATA=canonical`), not for the
pool (see below).

| Round | Case | Descriptors | Cues |
|-------|------|-------------|------|
| `x2` (D130) | `restaurant` | `x2/desc_restaurant/f4_restaurant_x2_extractions.parquet`, `descriptors-v6-restaurant` | `x2/cues/f4_restaurant_x2_extractions.parquet`, `single-v2` |
| `x2` (D130) | `street` | `x2/desc_street/f4_street_x2_extractions.parquet`, `descriptors-v6-street` | `x2/cues_street/f4_street_x2_extractions.parquet`, `street-v3` |
| `v1` | `restaurant` | `descriptors/f4_restaurant_desc_final.parquet`, `descriptors-v3` and `-v3r` (D121) | `extractions/f4_restaurant_extractions.parquet`, `single-v1` |
| `v1` | `street` | `descriptors_street/f4_street_desc_final.parquet`, `descriptors-v5` (D123) | `extractions_street/f4_street_extractions.parquet`, `street-v2` (D125 to D129) |

The paths are under the data root of `_f4` (see below). The cue spans give the Track C
filter: a descriptor whose innermost cue span is an `inference` or a
`visual_evidence` span. In `x2`, a cue span can lie inside another one
(a seen detail inside camera advice); the innermost span gives the class.

The `x2` tables differ from `v1` in 4 ways:

1. `target` has 3 values: `place`, `current_photo`, `planned_photo`. The
   `target` value `photo` of `select` keeps both photo values.
2. `aspect` has 3 more values: `evaluation`, `size`, `composition`.
3. A text part with no descriptor gives 1 `none` item. `spans` drops it.
4. The merge flags `copy_suspect` (a text that got a copy of the examples)
   and `example_source` (a text that a prompt example comes from).
   `load_raw` drops the copy suspects; they count as not read.

The descriptor input changes U+2011 into "-" and keeps the length of each
text, thus the offsets of the runs match. The street cue run reads the
descriptor input itself.

Street Track C is a conservative set: the street cue run misses some of
the place words (D129, D130). The check of D129 (`robustness`) compares it with
a second filter, `in_alt`: the descriptors with target place that are not
inside a `photo_advice` effect span or an `image_artifact` span.

The unit of a count
-------------------
A head counts 1 time for each text that holds it (its document frequency).
The extractor sometimes writes a later use of the same head again; this
unit removes those repeats, and a model that repeats a word in a long trace
does not win on that alone.

Warning: compare models inside 1 (case, field). The 2 cases use different
schema versions, prompts, and images. The case comparison (`case_signatures`,
`case_exclusive`, `case_rates`) thus compares only the contrast of a model
against the other models, inside each case.

The pool of 2 batches (D132)
----------------------------
The data root comes from `_f4`. The pool is the default: the paths of the
x2 round point at `data/f4_traces/pooled/` (10,000 images for each case, the
canonical batch and batch 2), and the export folder gets the suffix
`_pooled`. `F4_DATA=canonical` reads the canonical batch only, for
history. The tables of sections 1 to 10 of the notebook compare models on
the same images, thus they stay unweighted (see `_f4`). The rate comparison
of the 2 cases (`case_rates`, section 11) is different: it follows
`_f4.WEIGHTED`, thus in pooled mode it weights each restaurant text by the
weight of its image.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent

from research.notebooks.facct import _f4 as F  # noqa: E402
from research.notebooks.cvpr import _style as S  # noqa: E402
from research.notebooks.cvpr import _traces as T  # noqa: E402

__version__ = "1.8.0"

# The data root of `_f4`: the pool of D132 (the default), or the canonical batch.
F4 = F.F4
DATA = F.DATA
X2 = F4 / "x2"
ROUNDS = {
    "x2": {
        "FINAL": {"restaurant": X2 / "desc_restaurant" / "f4_restaurant_x2_extractions.parquet",
                  "street": X2 / "desc_street" / "f4_street_x2_extractions.parquet"},
        # The suffix "r" marks the rows of the retry pass (`x2_retry.py`).
        "SCHEMAS": {"restaurant": {"descriptors-v6-restaurant", "descriptors-v6-restaurantr"},
                    "street": {"descriptors-v6-street", "descriptors-v6-streetr"}},
        "INPUT": {"restaurant": F4 / "extract_input" / "f4_restaurant_x2.parquet",
                  "street": F4 / "extract_input" / "f4_street_x2.parquet"},
        "CUES": {"restaurant": X2 / "cues" / "f4_restaurant_x2_extractions.parquet",
                 "street": X2 / "cues_street" / "f4_street_x2_extractions.parquet"},
        "CUE_SCHEMAS": {"restaurant": {"single-v2", "single-v2r"}, "street": {"street-v3", "street-v3r"}},
        "TARGETS": ("both", "place", "photo", "current_photo", "planned_photo"),
    },
    "v1": {
        "FINAL": {"restaurant": F4 / "descriptors" / "f4_restaurant_desc_final.parquet",
                  "street": F4 / "descriptors_street" / "f4_street_desc_final.parquet"},
        "SCHEMAS": {"restaurant": {"descriptors-v3", "descriptors-v3r"}, "street": {"descriptors-v5"}},
        "INPUT": {"restaurant": F4 / "extract_input" / "f4_restaurant_desc.parquet",
                  "street": F4 / "extract_input" / "f4_street_desc.parquet"},
        "CUES": {"restaurant": F4 / "extractions" / "f4_restaurant_extractions.parquet",
                 "street": F4 / "extractions_street" / "f4_street_extractions.parquet"},
        "CUE_SCHEMAS": {"restaurant": {"single-v1"}, "street": {"street-v2"}},
        "TARGETS": ("both", "place", "photo"),
    },
}
ROUND = ""
FINAL: Dict[str, Path] = {}
SCHEMAS: Dict[str, set] = {}
INPUT: Dict[str, Path] = {}
CUES: Dict[str, Path] = {}
CUE_SCHEMAS: Dict[str, set] = {}
TARGETS: tuple = ()


def use_round(name: str) -> str:
    """Point the paths, the schema checks, and `TARGETS` at 1 extraction round."""
    global ROUND, FINAL, SCHEMAS, INPUT, CUES, CUE_SCHEMAS, TARGETS
    r = ROUNDS[name]
    ROUND = name
    FINAL, SCHEMAS, INPUT, CUES, CUE_SCHEMAS, TARGETS = (
        r["FINAL"], r["SCHEMAS"], r["INPUT"], r["CUES"], r["CUE_SCHEMAS"], r["TARGETS"])
    _TEXT_CACHE.clear()
    return name


_TEXT_CACHE: Dict[str, pd.Series] = {}
NTA = {
    "restaurant": F4 / "f4_restaurants_nta.parquet",
    "street": F4 / "f4_street_1k.parquet",
}
CASES = ("restaurant", "street")
FIELDS = ("reasoning", "response")
MODELS = F.MODELS
MODEL_COLORS = F.MODEL_COLORS
PROMPTS = {
    "restaurant": F.PROMPTS["restaurant"],
    "street": "Look and tell me a good angle to take a picture from.",
}
TRACKS = ("C", "V")
CUE_CLASSES = ("inference", "visual_evidence")
ATTRS = ("head", "vibe", "slot", "aspect", "negated", "target")

_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    """Read the 2 final tables, with every row: silent texts and unquotable spans too.

    The texts with `copy_suspect` are dropped; `raw.attrs["copy_suspects"]`
    gives their count for each case.

    Raises:
        FileNotFoundError: when a final table is missing.
        ValueError: when a table holds a schema version this notebook does not read.
    """
    frames, dropped = [], {}
    for case, path in FINAL.items():
        if not path.exists():
            raise FileNotFoundError(f"no table at {path}; run research/scripts/f4/combine_descriptor_retry.py")
        d = pd.read_parquet(path)
        versions = set(d.schema_version.dropna().unique())
        if not versions <= SCHEMAS[case]:
            raise ValueError(f"{case}: the table holds {sorted(versions)}, want {sorted(SCHEMAS[case])}")
        if "copy_suspect" in d:
            dropped[case] = int(d.loc[d.copy_suspect.astype(bool), "doc_id"].nunique())
            d = d[~d.copy_suspect.astype(bool)]
        frames.append(d.assign(case=case))
    raw = pd.concat(frames, ignore_index=True)
    raw.attrs["copy_suspects"] = dropped
    return raw


def normalize_head(head: object) -> str:
    """Give the plain form of a head: lower case, 1 space, plain hyphens, no quote marks."""
    s = str(head or "").lower().replace("‑", "-").replace("‐", "-")
    s = s.strip().strip("\"'#“”‘’").strip()
    return _SPACE.sub(" ", s)


# British spelling -> American spelling, 1 word at a time. The table comes
# from breame 0.1.2; _GB_US_EXTRA adds the British forms in the heads that
# the table does not hold.
_GB_US = json.loads((_HERE / "british_american_spellings.json").read_text())["british_to_american"]
_GB_US_EXTRA = {"storey": "story", "storeys": "stories", "storeyed": "storied", "overstorey": "overstory",
                "greyscale": "grayscale", "corbelled": "corbeled", "flavourful": "flavorful"}
_GB_US = {**_GB_US, **_GB_US_EXTRA}
_GB_US_SUFFIXES = ("ful", "ish", "ness", "ly", "ed", "es", "s", "ing")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def _american(word: str) -> str:
    """Give the American spelling of 1 word, also with a suffix (greyish -> grayish)."""
    if word in _GB_US:
        return _GB_US[word]
    for suf in _GB_US_SUFFIXES:
        if word.endswith(suf) and word[: -len(suf)] in _GB_US:
            return _GB_US[word[: -len(suf)]] + suf
    return word


def head_key(head: object) -> str:
    """Give the spelling key of a head: 2 heads with the same key are 1 word.

    The key removes the accents, puts each word in American spelling, removes
    the spaces, hyphens and other marks, and removes a plural "s" when the key
    has more than 4 letters. Thus "Closed-down", "closed down", "white-grey"
    and "white/gray", "taquería" and "taqueria" each give 1 key. The key does
    not join different words with near spellings (carved, curved).
    """
    s = unicodedata.normalize("NFKD", normalize_head(head)).encode("ascii", "ignore").decode()
    k = "".join(_american(w) for w in _NOT_ALNUM.split(s) if w)
    return k[:-1] if k.endswith("s") and len(k) > 4 else k


_US_GB: Dict[str, List[str]] = {}
for _gb, _us in _GB_US.items():
    _US_GB.setdefault(_us, []).append(_gb)


def _spelling_pattern(forms: Sequence[str]) -> "re.Pattern[str]":
    """Give a regex that finds a head in a text in each of its spellings.

    Each word matches its British and American forms. Between the words, a
    space, a hyphen, a slash or nothing matches. A plural "s" at the end
    matches too.
    """
    alts = set()
    for form in forms:
        words = [w for w in re.split(r"[\W_]+", normalize_head(form)) if w]
        if not words:
            continue
        parts = []
        for w in words:
            plain = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
            us = _american(plain)
            parts.append("(?:" + "|".join(sorted(map(re.escape, {w, plain, us, *_US_GB.get(us, [])}))) + ")")
        alts.add(r"[\s\-/]?".join(parts))
    body = "|".join(sorted(alts))
    return re.compile(r"(?<![\w-])(?:" + body + r")s?(?![\w-])", re.IGNORECASE)


def spans(raw: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each quotable span, with the attributes in columns.

    The restaurant case of `v1` has no `target` (schema v3); it gets `place`.
    The `none` items are dropped. The column `first_use` is true for the
    first span of a head in its text.
    """
    df = raw[raw.extraction_class.notna() & (raw.extraction_class != "none")
             & raw.is_quotable.fillna(False)].copy()
    attrs = df.attributes_json.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for k in ATTRS:
        df[k] = attrs.map(lambda a, k=k: a.get(k))
    df["target"] = df.target.fillna("place")
    df["head"] = df["head"].map(normalize_head)
    df["slot"] = df.slot.map(normalize_head).replace("", "none")
    df = df[df["head"] != ""]
    df = df.sort_values(["doc_id", "char_start"])
    df["first_use"] = ~df.duplicated(["doc_id", "head"])
    return attach_nta(df.reset_index(drop=True))


def attach_nta(df: pd.DataFrame) -> pd.DataFrame:
    """Add `nta2020` and `nta_name`: the restaurant point (F6), or the street recording."""
    r = pd.read_parquet(NTA["restaurant"], columns=["image_idx", "NTA2020", "NTAName"]).rename(
        columns={"NTA2020": "nta2020", "NTAName": "nta_name"}).assign(case="restaurant")
    s = pd.read_parquet(NTA["street"], columns=["image_idx", "nta2020", "nta_name"]).assign(case="street")
    return df.merge(pd.concat([r, s]), on=["case", "image_idx"], how="left")


def text_totals(raw: pd.DataFrame) -> pd.Series:
    """Give the texts of each (case, field, model), silent texts included.

    Warning: count from `raw`, never from `spans`. A text with no span was still
    read; leave it out and every rate reads too high.
    """
    return raw.groupby(["case", "field", "model"]).doc_id.nunique()


def coverage(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Give, for each (case, field, model): texts, texts with no descriptor, spans, heads, vibe share."""
    tot = text_totals(raw)
    cls = raw.extraction_class
    have = raw[cls.notna() & (cls != "none")].groupby(["case", "field", "model"]).doc_id.nunique()
    silent = tot - have.reindex(tot.index).fillna(0)
    g = df.groupby(["case", "field", "model"])
    out = pd.DataFrame({
        "texts": tot,
        "empty_texts": silent.reindex(tot.index).fillna(0).astype(int),
        "spans": g.size().reindex(tot.index).fillna(0).astype(int),
        "distinct_heads": g["head"].nunique().reindex(tot.index).fillna(0).astype(int),
        "vibe_share": g.vibe.apply(lambda v: (v == "yes").mean()).reindex(tot.index).round(3),
        "photo_share": g.target.apply(lambda v: (v != "place").mean()).reindex(tot.index).round(3),
    })
    out["spans_per_text"] = (out.spans / out.texts).round(2)
    return _order_index(out)


# ---------------------------------------------------------------------------
# The track filters
# ---------------------------------------------------------------------------


def load_cues(case: str) -> Optional[pd.DataFrame]:
    """Read the quotable cue spans of 1 case, or None when the file is missing.

    Raises:
        ValueError: when the file holds a schema version this notebook does not read.
    """
    path = CUES[case]
    if not path.exists():
        return None
    cols = ["doc_id", "extraction_class", "attributes_json", "char_start", "char_end", "is_quotable",
            "schema_version"]
    has_flag = "copy_suspect" in pq.read_schema(path).names
    cues = pd.read_parquet(path, columns=cols + (["copy_suspect"] if has_flag else []))
    versions = set(cues.schema_version.dropna().unique())
    if not versions <= CUE_SCHEMAS[case]:
        raise ValueError(f"{case} cues: the file holds {sorted(versions)}, want {sorted(CUE_SCHEMAS[case])}")
    cues = cues[cues.extraction_class.notna() & (cues.extraction_class != "none")
                & cues.is_quotable.fillna(False)].copy()
    # A text with no span still has 1 row with a null class: it was read. A
    # copy suspect was not read: its descriptors get NA, not "outside".
    every = pd.read_parquet(path, columns=["doc_id"] + (["copy_suspect"] if has_flag else []))
    if has_flag:
        every = every[~every.copy_suspect.astype(bool)]
        cues = cues[~cues.copy_suspect.astype(bool)]
    cues.attrs["docs"] = set(every.doc_id)
    cues["kind"] = cues.attributes_json.map(lambda a: json.loads(a).get("kind") if isinstance(a, str) else None)
    return cues


def _innermost(rows: pd.DataFrame, cues: pd.DataFrame) -> pd.DataFrame:
    """Give, for each row of `rows` inside a cue span, the class and kind of the shortest such span.

    The result has the index labels of `rows`. A row inside no cue span is absent.
    """
    c = cues[["doc_id", "char_start", "char_end", "extraction_class", "kind"]].rename(
        columns={"char_start": "c0", "char_end": "c1"})
    m = rows.reset_index()[["index", "doc_id", "char_start", "char_end"]].merge(c, on="doc_id")
    m = m[(m.char_start >= m.c0) & (m.char_end <= m.c1)]
    m = m.assign(width=m.c1 - m.c0).sort_values(["index", "width"])
    return m.drop_duplicates("index").set_index("index")[["extraction_class", "kind"]]


def mark_cue_spans(df: pd.DataFrame, classes: Sequence[str] = CUE_CLASSES) -> pd.DataFrame:
    """Add `in_cue` and `in_alt` to the descriptor spans.

    | Column | True when |
    |--------|-----------|
    | `in_cue` | the innermost cue span around the descriptor is of `classes` |
    | `in_alt` | street only: target place, and the innermost cue span is not a `photo_advice` effect span or an `image_artifact` span (D129) |

    `cue_class` gives the class of the innermost cue span ("" when the
    descriptor lies in no cue span), and `cue_kind` its `kind` (inference,
    image_artifact, photo_advice).

    A case with no cue file, or a text that the cue file lacks, gets NA in
    both columns.
    """
    out = df.copy()
    out["in_cue"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["in_alt"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["cue_class"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    out["cue_kind"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    for case in CASES:
        cues = load_cues(case)
        if cues is None:
            continue
        mask = (out.case == case) & out.doc_id.isin(cues.attrs["docs"])
        rows = out[mask]
        inner = _innermost(rows, cues).reindex(rows.index)
        out.loc[mask, "cue_class"] = inner.extraction_class.fillna("").to_numpy()
        out.loc[mask, "cue_kind"] = inner["kind"].to_numpy()
        out.loc[mask, "in_cue"] = inner.extraction_class.isin(classes).to_numpy()
        if case == "street":
            off = (inner.extraction_class == "image_artifact") | (
                (inner.extraction_class == "photo_advice") & (inner.kind == "effect"))
            out.loc[mask, "in_alt"] = (rows.target == "place").to_numpy() & ~off.to_numpy()
    return out


def has_cues(df: pd.DataFrame, case: str) -> bool:
    """Tell if the cue spans of `case` are present."""
    return bool(df.loc[df.case == case, "in_cue"].notna().any())


def select(df: pd.DataFrame, case: str, field: str, track: str = "C", target: str = "both",
           cue_only: bool = True) -> pd.DataFrame:
    """Give the spans of 1 (case, field) for 1 track, first uses only.

    | Track | Spans |
    |-------|-------|
    | C | every descriptor; with `cue_only`, only those inside an inference or a visual_evidence span |
    | C2 | street only: the second filter of D129 (`in_alt`) |
    | V | the descriptors with `vibe` yes, in the full text (D109) |

    `target` keeps 1 value of `target`, or `both` for all. `photo` keeps
    every target that is not `place`. When a case has no cue file, track C
    keeps every descriptor.
    """
    part = df[(df.case == case) & (df.field == field) & df.first_use]
    if track == "C" and cue_only and has_cues(df, case):
        part = part[part.in_cue.fillna(False).astype(bool)]
    elif track == "C2":
        part = part[part.in_alt.fillna(False).astype(bool)]
    elif track == "V":
        part = part[part.vibe == "yes"]
    if target == "photo":
        part = part[part.target != "place"]
    elif target != "both":
        part = part[part.target == target]
    return part


# ---------------------------------------------------------------------------
# Signature vocabulary (log-odds) and dispersion
# ---------------------------------------------------------------------------


def head_counters(part: pd.DataFrame) -> Dict[str, Counter]:
    """Give, for each model, the number of texts that hold each head."""
    return {m: Counter(g["head"]) for m, g in part.groupby("model")}


def signature(part: pd.DataFrame, totals: pd.Series, model: str, max_rows: int = 40,
              min_count: int = 5, prior_weight: float = 500.0) -> pd.DataFrame:
    """Score the heads that 1 model uses more than the other models of the same (case, field).

    The score is the log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi, and Quinn, 2008; `_traces.distinctive_scores`), over
    document frequencies. The dispersion columns tell whether a head is a
    signature of the model or of a few images:

    | Column | Meaning |
    |--------|---------|
    | `texts` | the texts of this model that hold the head |
    | `share` | `texts` over all texts of this model in the (case, field) |
    | `share_others` | the same share for the other models together |
    | `ntas` | the distinct NTAs of those texts |
    | `nta_share` | `ntas` over the NTAs that this model's texts cover |
    """
    by_model = head_counters(part)
    target = by_model.get(model, Counter())
    if not target:
        return pd.DataFrame()
    background: Counter = Counter()
    for m, c in by_model.items():
        if m != model:
            background.update(c)
    scores = T.distinctive_scores(target, background, prior_weight=prior_weight, min_count=min_count)
    top = [(h, z) for h, z in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if z > 0][:max_rows]
    if not top:
        return pd.DataFrame()
    case, field = part.case.iloc[0], part.field.iloc[0]
    n_model = int(totals.get((case, field, model), 0)) or 1
    n_others = int(sum(v for (c, f, m), v in totals.items() if c == case and f == field and m != model
                       and m in by_model)) or 1
    mine = part[part.model == model]
    ntas_all = mine.nta2020.nunique() or 1
    nta_by_head = mine.groupby("head").nta2020.nunique()
    return pd.DataFrame([{
        "head": h, "z": round(z, 2), "texts": target[h],
        "share": round(target[h] / n_model, 4), "share_others": round(background.get(h, 0) / n_others, 4),
        "ntas": int(nta_by_head.get(h, 0)), "nta_share": round(nta_by_head.get(h, 0) / ntas_all, 3),
    } for h, z in top])


def signatures(part: pd.DataFrame, totals: pd.Series, max_rows: int = 15, **kw) -> pd.DataFrame:
    """Give the signature of every model of `part`, stacked, with a `model` column."""
    rows = []
    for m in [m for m in MODELS if m in set(part.model)]:
        t = signature(part, totals, m, max_rows=max_rows, **kw)
        if not t.empty:
            rows.append(t.assign(model=m))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def exclusive_heads(part: pd.DataFrame, df: pd.DataFrame, totals: pd.Series, alpha: float = 0.05,
                    check_text: bool = True) -> pd.DataFrame:
    """Find the heads that only 1 model uses, with an exact binomial test.

    `part` is the selection of 1 (case, field), as `select` gives it; `df` is
    every descriptor. The unit is the spelling key (`head_key`): "closed-down"
    and "closed down", or "grey" and "gray", are 1 word. A candidate is a key
    of `part` that no other model holds as a descriptor in the same (case,
    field), over all descriptors, not only the selection. `head` gives the
    most frequent form of the key in the model; `forms` lists all its forms.

    The test: the models read the same images with the same prompt. If a
    head has no link to a model, each of its texts comes from model M with
    the probability `share_m` (the texts of M over all texts of the case and
    field: about 1/6 for the answers, 1/4 for the traces). The chance that
    all `k` texts come from M is `share_m ** k`. `p_bonf` multiplies `p` by
    the number of candidates (Bonferroni). A head passes when `p_bonf` < `alpha`.

    With `check_text`, `others_in_text` counts the texts of the other models
    that hold the word in any of its spellings (`_spelling_pattern`: word
    borders, no case). The extractor can miss a word; the search cannot. A
    head with `others_in_text` 0 is exclusive in the text too.
    """
    if part.empty:
        return pd.DataFrame()
    case, field = part.case.iloc[0], part.field.iloc[0]
    n_all = sum(int(v) for (c, f, m), v in totals.items() if c == case and f == field)
    pool = df[(df.case == case) & (df.field == field)]
    keys = {h: head_key(h) for h in pool["head"].unique()} | {h: head_key(h) for h in part["head"].unique()}
    users = pool.groupby(pool["head"].map(keys)).model.nunique()
    sel = part.assign(key=part["head"].map(keys))
    mine = sel.drop_duplicates(["doc_id", "key"]).groupby(["model", "key"]).doc_id.nunique().reset_index(name="k")
    forms = sel.groupby(["model", "key"])["head"].agg(lambda h: h.value_counts().index.tolist())
    mine["head"] = [forms[(m, k)][0] for m, k in zip(mine.model, mine.key)]
    mine["forms"] = ["; ".join(sorted(forms[(m, k)])) for m, k in zip(mine.model, mine.key)]
    cand = mine[mine.key.map(users).fillna(1) == 1].drop(columns="key")[["model", "head", "k", "forms"]].copy()
    if cand.empty:
        return pd.DataFrame()
    share = cand.model.map(lambda m: int(totals[(case, field, m)]) / n_all)
    cand["p"] = share ** cand.k
    cand["p_bonf"] = (cand.p * len(cand)).clip(upper=1.0)
    out = cand[cand.p_bonf < alpha].rename(columns={"k": "texts"}).copy()
    out.attrs["candidates"] = len(cand)
    if check_text and not out.empty:
        text = _texts(case)
        parts = pd.Series(text.index, index=text.index).str.split("|", expand=True)
        counts = []
        for r in out.itertuples():
            others = text[(parts[3] == field) & (parts[0] != r.model)]
            counts.append(int(others.str.contains(_spelling_pattern(r.forms.split("; "))).sum()))
        out["others_in_text"] = counts
    out["p"] = out.p.map(lambda v: float(f"{v:.2g}"))
    out["p_bonf"] = out.p_bonf.map(lambda v: float(f"{v:.2g}"))
    out = _order_rows(out.sort_values(["model", "texts"], ascending=[True, False]))
    out.attrs["candidates"] = len(cand)
    return out


def robustness(df: pd.DataFrame, totals: pd.Series, field: str, n: int = 15) -> pd.DataFrame:
    """Compare the street signatures of track C with those of the second filter (D129).

    | Column | Meaning |
    |--------|---------|
    | `heads_c`, `heads_c2` | the head uses of the model under each filter |
    | `shared_top` | the heads that both top-`n` lists hold |
    | `spearman` | the rank correlation of z over the heads that both tables score |
    | `only_c2` | the top-`n` heads of C2 that are not in the top `n` of C |
    """
    a = select(df, "street", field, "C", "place")
    b = select(df, "street", field, "C2", "place")
    sa, sb = signatures(a, totals, max_rows=200), signatures(b, totals, max_rows=200)
    have = (set(sa.model) if len(sa) else set()) | (set(sb.model) if len(sb) else set())
    rows = []
    for m in [m for m in MODELS if m in have]:
        x = sa[sa.model == m].set_index("head").z if len(sa) else pd.Series(dtype=float)
        y = sb[sb.model == m].set_index("head").z if len(sb) else pd.Series(dtype=float)
        # `signatures` gives the heads in z order, thus the first n are the top n.
        tx, ty = list(x.index[:n]), list(y.index[:n])
        both = x.index.intersection(y.index)
        rho = float(x[both].rank().corr(y[both].rank())) if len(both) > 2 else np.nan
        rows.append({"model": m, "heads_c": int((a.model == m).sum()), "heads_c2": int((b.model == m).sum()),
                     "shared_top": len(set(tx) & set(ty)), "spearman": round(rho, 2),
                     "only_c2": ", ".join(h for h in ty if h not in tx)})
    return pd.DataFrame(rows).set_index("model")


def top_heads(part: pd.DataFrame, totals: pd.Series, n: int = 15) -> pd.DataFrame:
    """Give the most frequent heads of each model, as the share of its texts."""
    rows = []
    for m, g in part.groupby("model"):
        tot = int(totals.get((g.case.iloc[0], g.field.iloc[0], m), 0)) or 1
        for h, k in Counter(g["head"]).most_common(n):
            rows.append({"model": m, "head": h, "texts": k, "share": round(k / tot, 4)})
    return _order_rows(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Concentration: how monolithic is the vocabulary of 1 model?
# ---------------------------------------------------------------------------


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def concentration(part: pd.DataFrame, k: Optional[int] = None, draws: int = 200,
                  seed: int = 20261026) -> pd.DataFrame:
    """Measure, for each model, how concentrated its head vocabulary is.

    | Column | Meaning |
    |--------|---------|
    | `tokens` | head uses (first use in each text) |
    | `distinct` | distinct heads |
    | `top10_share` | the share of uses that the 10 most frequent heads take |
    | `entropy` | the plug-in Shannon entropy of the heads, in bits |
    | `entropy_rare` | the mean entropy of `k` uses drawn without replacement, `draws` times |

    The plug-in entropy grows with the sample, and the models give very
    different numbers of spans. Compare `entropy_rare` across models. `k` is
    the smallest `tokens` of the models when it is not given.
    """
    groups = {m: g["head"].to_numpy() for m, g in part.groupby("model")}
    if not groups:
        return pd.DataFrame()
    k = k or min(len(v) for v in groups.values())
    rng = np.random.default_rng(seed)
    rows = []
    for m, heads in groups.items():
        c = Counter(heads)
        counts = np.array(list(c.values()), dtype=float)
        top10 = sum(n for _, n in c.most_common(10)) / len(heads)
        codes = pd.factorize(heads)[0]
        rare = [_entropy(np.bincount(rng.choice(codes, size=k, replace=False))) for _ in range(draws)] \
            if k <= len(heads) else [np.nan]
        rows.append({"model": m, "tokens": len(heads), "distinct": len(c), "top10_share": round(top10, 3),
                     "entropy": round(_entropy(counts), 2), "entropy_rare": round(float(np.mean(rare)), 2),
                     "k": k})
    return _order_rows(pd.DataFrame(rows)).set_index("model")


def nta_overlap(part: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Give, for each model, the mean share of NTAs in which its top heads appear.

    A model whose top heads appear in most NTAs writes the same words about
    every neighborhood. The value is the mean of `nta_share` over its `top`
    most frequent heads.
    """
    rows = []
    for m, g in part.groupby("model"):
        n = g.nta2020.nunique() or 1
        heads = [h for h, _ in Counter(g["head"]).most_common(top)]
        per = g[g["head"].isin(heads)].groupby("head").nta2020.nunique() / n
        rows.append({"model": m, "ntas": n, f"top{top}_mean_nta_share": round(float(per.mean()), 3)})
    return _order_rows(pd.DataFrame(rows)).set_index("model")


# ---------------------------------------------------------------------------
# Track V extras
# ---------------------------------------------------------------------------


# The inference kinds of the risk question: the parent's 4 (`_extractions.RISK_KINDS`)
# and safety, which a street view or a storefront invites.
SOCIAL_KINDS = ("wealth", "class", "demographic", "crime", "safety")


def cue_kind_shares(df: pd.DataFrame, case: str, field: str) -> pd.DataFrame:
    """Give, for each model, where its descriptors sit: the share of first uses in each innermost cue.

    The rows are the models. A column is a cue class, or `inference: <kind>`
    for an inference span; `none` is no cue span. The texts that the cue file
    lacks are left out.
    """
    part = df[(df.case == case) & (df.field == field) & df.first_use & df.cue_class.notna()]
    where = part.cue_class.replace("", "none")
    inf = part.cue_class == "inference"
    where = where.where(~inf, "inference: " + part.cue_kind.fillna("other").astype(str))
    out = pd.crosstab(part.model, where, normalize="index").round(3)
    order = [m for m in MODELS if m in out.index]
    return out.reindex(order)


def social_heads(df: pd.DataFrame, totals: pd.Series, case: str, field: str,
                 kinds: Sequence[str] = SOCIAL_KINDS, n: int = 12) -> pd.DataFrame:
    """Give the descriptors that each model puts inside a social inference.

    A social inference is an `inference` span whose `kind` is in `kinds`. 1
    row for each (model, head) among the top `n` heads of the model, by texts.

    | Column | Meaning |
    |--------|---------|
    | `texts` | texts of the model that hold the head inside a social inference |
    | `per_1000` | the same, for each 1,000 texts of the model |
    | `kinds` | the inference kinds of those spans, most frequent first |
    """
    part = df[(df.case == case) & (df.field == field) & df.first_use
              & (df.cue_class == "inference") & df.cue_kind.isin(kinds)]
    rows = []
    for model, g in part.groupby("model"):
        tot = int(totals.get((case, field, model), 0))
        per_text = g.drop_duplicates(["doc_id", "head"])
        for head, h in per_text.groupby("head"):
            rows.append({"model": model, "head": head, "texts": h.doc_id.nunique(),
                         "per_1000": round(1000 * h.doc_id.nunique() / max(1, tot), 2),
                         "kinds": ", ".join(h.cue_kind.value_counts().index)})
    out = pd.DataFrame(rows, columns=["model", "head", "texts", "per_1000", "kinds"])
    if out.empty:
        return out
    out = out.sort_values(["model", "texts"], ascending=[True, False]).groupby("model").head(n)
    return _order_rows(out).reset_index(drop=True)


def vibe_rates(df: pd.DataFrame, totals: pd.Series, per: int = 100) -> pd.DataFrame:
    """Give the vibe descriptors (first uses) for each `per` texts, by (case, field, model)."""
    v = df[df.first_use & (df.vibe == "yes")].groupby(["case", "field", "model"]).size()
    out = (v.reindex(totals.index).fillna(0) / totals * per).round(1).rename(f"vibe_per_{per}_texts")
    return _order_index(out.to_frame())


def aspect_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Give the share of each `aspect` among the first uses of each (case, field, model)."""
    d = df[df.first_use]
    out = pd.crosstab([d.case, d.field, d.model], d.aspect, normalize="index").round(3)
    return _order_index(out)


def frames(part: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Give the most frequent frame nouns (`slot` not none) of each model."""
    s = part[part.slot != "none"]
    rows = [{"model": m, "slot": k, "spans": c} for m, g in s.groupby("model")
            for k, c in Counter(g.slot).most_common(n)]
    return _order_rows(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Findings in both cases, and findings in 1 case only (restaurant and street)
# ---------------------------------------------------------------------------


# The 2 cases that this comparison reads. `caption` has no descriptors.
CASE_PAIR = ("restaurant", "street")
# The classes of a finding with a test in each case (signature words, rates).
CASE_CLASSES = ("both, same sign", "both, opposite sign", "restaurant only", "street only",
                "1 case only, not different", "neither")
# The classes of a signature word that has no estimate in the other case.
VOCAB_CLASSES = ("vocabulary of 1 case", "too rare in the other case")
# The classes of an exclusive head.
EXCLUSIVE_CLASSES = ("both", "1 case, another model in the other case",
                     "1 case, only this model in the other case", "1 case, absent in the other case")


def default_selection(df: pd.DataFrame, case: str, field: str, cue_only: bool = True) -> pd.DataFrame:
    """Give the default selection of section 3: track C, all aspects, target place on street."""
    return select(df, case, field, "C", "place" if case == "street" else "both", cue_only)


def _logodds(target: Counter, background: Counter, prior_weight: float = 500.0,
             min_count: int = 5) -> Dict[str, tuple]:
    """Give the log-odds estimate and its standard error for each word.

    This is the formula of `_traces.distinctive_scores` (Monroe, Colaresi,
    and Quinn, 2008). That function gives only z = estimate / se; the case
    comparison needs the estimate and the se apart.
    """
    pooled = Counter(target)
    pooled.update(background)
    pooled = Counter({w: n for w, n in pooled.items() if n >= min_count})
    total_pooled = sum(pooled.values()) or 1
    n_t = sum(target[w] for w in pooled) or 1
    n_b = sum(background[w] for w in pooled) or 1
    out = {}
    for w, n_pool in pooled.items():
        alpha = prior_weight * n_pool / total_pooled
        yt, yb = target.get(w, 0) + alpha, background.get(w, 0) + alpha
        delta = np.log(yt / max(n_t + prior_weight - yt, 1e-9)) - np.log(yb / max(n_b + prior_weight - yb, 1e-9))
        out[w] = (float(delta), float(np.sqrt(1.0 / yt + 1.0 / yb)))
    return out


def key_logodds(part: pd.DataFrame, min_count: int = 5, prior_weight: float = 500.0) -> pd.DataFrame:
    """Give the log-odds of each (model, key) of 1 selection against the other models.

    The formula and the prior are those of `signature`, but the unit is the
    spelling key (`head_key`), not the head. Thus the z here can differ a
    little from the z of `signature`. A key counts 1 time for each text that
    holds it. The table holds both signs.

    | Column | Meaning |
    |--------|---------|
    | `key`, `head` | the spelling key, and its most frequent form in the selection |
    | `texts` | the texts of the model that hold the key |
    | `estimate`, `se`, `z` | the log-odds, its standard error, and their ratio |
    """
    if part.empty:
        return pd.DataFrame(columns=["model", "key", "head", "texts", "estimate", "se", "z"])
    sel = part.assign(key=part["head"].map(head_key)).drop_duplicates(["doc_id", "key"])
    form = sel.groupby("key")["head"].agg(lambda h: h.value_counts().index[0])
    by_model = {m: Counter(g.key) for m, g in sel.groupby("model")}
    rows = []
    for m in [m for m in MODELS if m in by_model]:
        background: Counter = Counter()
        for o, c in by_model.items():
            if o != m:
                background.update(c)
        for k, (est, se) in _logodds(by_model[m], background, prior_weight, min_count).items():
            rows.append({"model": m, "key": k, "head": form[k], "texts": by_model[m].get(k, 0),
                         "estimate": est, "se": se, "z": est / se})
    return pd.DataFrame(rows)


def _p_two_sided(z) -> np.ndarray:
    from scipy.stats import norm

    return 2 * norm.sf(np.abs(np.asarray(z, float)))


def _classify(sig_r: pd.Series, sig_s: pd.Series, sign_r: pd.Series, sign_s: pd.Series,
              diff_sig: pd.Series) -> pd.Series:
    """Give the class of each finding from the 2 tests inside the cases and the difference test."""
    out = pd.Series("neither", index=sig_r.index, dtype=object)
    both = sig_r & sig_s
    out[both & (sign_r == sign_s)] = "both, same sign"
    out[both & (sign_r != sign_s)] = "both, opposite sign"
    one = sig_r ^ sig_s
    out[one & ~diff_sig] = "1 case only, not different"
    out[one & diff_sig & sig_r] = "restaurant only"
    out[one & diff_sig & sig_s] = "street only"
    return out


def _bonferroni(p: pd.Series) -> pd.Series:
    return (p * len(p)).clip(upper=1.0)


def case_signatures(df: pd.DataFrame, field: str, alpha: float = 0.05, cue_only: bool = True,
                    min_count: int = 5) -> pd.DataFrame:
    """Compare the signature words of each model in the restaurant case and in the street case.

    The selection of each case is `default_selection`. In each case, each
    (model, key) gets a 2-sided normal p from its log-odds z (`key_logodds`);
    `p_bonf_<case>` multiplies p by the number of (model, key) tests of that
    case (Bonferroni). A key is significant in a case when `p_bonf` < `alpha`.

    1 row for each (model, key) with an estimate in both cases. `z_diff` and
    `p_diff` come from `_f4.difference_test` on the 2 estimates;
    `p_diff_bonf` multiplies `p_diff` by the number of these rows. `class`
    is 1 of `CASE_CLASSES`.

    More rows: a key that is significant with a positive z in 1 case (a
    signature word) and has no estimate in the other case. The class is
    `vocabulary of 1 case` when the key is not a descriptor of the other
    case (same field, all descriptors); else `too rare in the other case`
    (fewer than `min_count` texts in the selection of the other case).
    `case` names the case of the finding.
    """
    est = {}
    for c in CASE_PAIR:
        t = key_logodds(default_selection(df, c, field, cue_only), min_count=min_count)
        t["p"] = _p_two_sided(t.z)
        t["p_bonf"] = _bonferroni(t.p)
        t["sig"] = t.p_bonf < alpha
        est[c] = t
    a, b = (est[c].set_index(["model", "key"]) for c in CASE_PAIR)
    keep = ["head", "texts", "estimate", "se", "z", "p_bonf", "sig"]
    both = a[keep].join(b[keep], how="inner", lsuffix="_restaurant", rsuffix="_street")
    if both.empty:
        return pd.DataFrame()
    z, p = F.difference_test(both.estimate_restaurant, both.se_restaurant, both.estimate_street, both.se_street)
    both["z_diff"], both["p_diff"] = z, p
    both["p_diff_bonf"] = _bonferroni(both.p_diff)
    both["class"] = _classify(both.sig_restaurant, both.sig_street, np.sign(both.z_restaurant),
                              np.sign(both.z_street), both.p_diff_bonf < alpha)
    both["head"] = both.head_restaurant
    both["case"] = np.where(both["class"].isin(["both, same sign", "both, opposite sign", "neither"]), "both",
                            np.where(both.sig_restaurant, "restaurant", "street"))
    both = both.drop(columns=["head_restaurant", "head_street", "sig_restaurant", "sig_street"]).reset_index()
    # The signature words with no estimate in the other case.
    extra = []
    for c, o in (CASE_PAIR, CASE_PAIR[::-1]):
        other_keys = set(df.loc[(df.case == o) & (df.field == field), "head"].map(head_key))
        t = est[c]
        t = t[t.sig & (t.z > 0)].set_index(["model", "key"])
        t = t[~t.index.isin(b.index if c == CASE_PAIR[0] else a.index)].reset_index()
        if t.empty:
            continue
        t = t.rename(columns={k: f"{k}_{c}" for k in ["texts", "estimate", "se", "z", "p_bonf"]})
        t["class"] = np.where(t.key.isin(other_keys), VOCAB_CLASSES[1], VOCAB_CLASSES[0])
        extra.append(t.drop(columns=["p", "sig"]).assign(case=c))
    out = pd.concat([both] + extra, ignore_index=True)
    cols = ["model", "key", "head", "class", "case"] + [f"{k}_{c}" for c in CASE_PAIR
                                                         for k in ["texts", "estimate", "se", "z", "p_bonf"]]
    out = out[cols + ["z_diff", "p_diff", "p_diff_bonf"]]
    for col in out.columns:
        if col.startswith(("estimate", "se_", "z")):
            out[col] = out[col].round(3)
        elif col.startswith("texts_"):
            out[col] = out[col].astype("Int64")
        elif col.startswith("p_"):
            out[col] = out[col].map(lambda v: float(f"{v:.2g}") if pd.notna(v) else v)
    out.attrs["alpha"] = alpha
    return _order_rows(out.sort_values(["model", "z_diff"], key=lambda s: s.abs() if s.name == "z_diff" else s,
                                       ascending=[True, False]))


def case_exclusive(df: pd.DataFrame, totals: pd.Series, field: str, alpha: float = 0.05,
                   cue_only: bool = True, check_text: bool = False) -> pd.DataFrame:
    """Compare the exclusive heads of section 3.1 in the restaurant case and in the street case.

    The selection of each case is `default_selection`; the test is that of
    `exclusive_heads`. 1 row for each (model, key) that passes in 1 case or
    in both. The other case counts all its descriptors of the same field,
    not only the selection.

    | `class` | The key passes |
    |---------|----------------|
    | `both` | in both cases, for this model |
    | `1 case, another model in the other case` | in 1 case; in the other case, another model uses it |
    | `1 case, only this model in the other case` | in 1 case; in the other case, only this model uses it, and it does not pass |
    | `1 case, absent in the other case` | in 1 case; no model uses it in the other case |

    With `check_text`, `others_in_text_<case>` comes from `exclusive_heads`
    (the case of the pass). The default is off: the search of the texts
    takes minutes on the pool, and the class does not use it.
    """
    ex, users = {}, {}
    for c in CASE_PAIR:
        e = exclusive_heads(default_selection(df, c, field, cue_only), df, totals, alpha=alpha,
                            check_text=check_text)
        ex[c] = e.assign(key=e["head"].map(head_key)) if not e.empty else pd.DataFrame(
            columns=["model", "head", "texts", "forms", "p", "p_bonf", "key"])
        pool = df[(df.case == c) & (df.field == field)]
        users[c] = pool.groupby(pool["head"].map(head_key)).model.agg(set)
    rows = []
    for c, o in (CASE_PAIR, CASE_PAIR[::-1]):
        passed_other = set(zip(ex[o].model, ex[o].key))
        for r in ex[c].to_dict("records"):
            if (r["model"], r["key"]) in passed_other:
                if c == CASE_PAIR[1]:
                    continue
                cls, case = EXCLUSIVE_CLASSES[0], "both"
            else:
                who = users[o].get(r["key"], set())
                cls = (EXCLUSIVE_CLASSES[1] if who - {r["model"]} else
                       EXCLUSIVE_CLASSES[2] if who else EXCLUSIVE_CLASSES[3])
                case = c
            rows.append({"model": r["model"], "key": r["key"], "head": r["head"], "class": cls, "case": case,
                         f"texts_{c}": r["texts"], f"p_bonf_{c}": r["p_bonf"],
                         f"others_in_text_{c}": r.get("others_in_text"),
                         "other_case_models": ", ".join(m for m in MODELS if m in users[o].get(r["key"], set()))})
    cols = ["model", "key", "head", "class", "case"] + [f"{k}_{c}" for c in CASE_PAIR
                                                         for k in ("texts", "p_bonf", "others_in_text")]
    out = pd.DataFrame(rows, columns=cols + ["other_case_models"])
    if out.empty:
        return out
    # A key that passes in both cases: add the street values to its row.
    for i in out.index[out["class"] == EXCLUSIVE_CLASSES[0]]:
        s = ex[CASE_PAIR[1]]
        s = s[(s.model == out.at[i, "model"]) & (s.key == out.at[i, "key"])].iloc[0]
        out.loc[i, ["texts_street", "p_bonf_street", "others_in_text_street"]] = [
            s["texts"], s["p_bonf"], s.get("others_in_text")]
        out.at[i, "other_case_models"] = ""
    for c in CASE_PAIR:
        out[f"texts_{c}"] = out[f"texts_{c}"].astype("Int64")
    out = out.dropna(axis=1, how="all")
    return _order_rows(out.sort_values(["model", "class", "key"]))


def _wmean_se(x: np.ndarray, w: np.ndarray) -> tuple:
    """Give the weighted mean of `x` and its standard error (the linear form).

    se = sqrt(sum(w^2 (x - mean)^2)) / sum(w). With all weights 1, this is
    the plain standard error of a mean (with n, not n - 1).
    """
    sw = w.sum()
    if sw == 0:
        return np.nan, np.nan
    m = float((w * x).sum() / sw)
    return m, float(np.sqrt((w ** 2 * (x - m) ** 2).sum()) / sw)


def case_rates(raw: pd.DataFrame, df: pd.DataFrame, field: str, alpha: float = 0.05,
               per: int = 100, weighted: bool = False) -> pd.DataFrame:
    """Compare the vibe rate and the aspect shares of each model in the restaurant and street cases.

    The measures are those of section 6, over all descriptors (first uses),
    not only the selection:

    | `measure` | Unit | Value of 1 unit |
    |-----------|------|-----------------|
    | `vibe_per_100` | 1 text (silent texts too) | its vibe descriptors, times `per` |
    | `aspect: <a>` | 1 descriptor | 1 when its aspect is `a`, else 0 |

    The estimate of a model in a case is its mean minus the mean of the
    other models of the same (case, field), with
    se = sqrt(se_model^2 + se_others^2). The units are not weighted, as in
    section 6 and in `_f4.model_contrasts`: each model reads the same images,
    thus this is a comparison of models. With `weighted` (and `_f4.WEIGHTED`),
    a restaurant unit gets the weight of its image and the se is the weighted
    form (`_wmean_se`); the street units get the weight 1. Each case gets a 2-sided normal p, with Bonferroni over
    the rows of the table; `z_diff`, `p_diff` and `p_diff_bonf` come from
    `_f4.difference_test`, with Bonferroni over the rows of the table.
    """
    rows = []
    for c in CASE_PAIR:
        texts = raw[(raw.case == c) & (raw.field == field)].drop_duplicates("doc_id")[
            ["doc_id", "case", "model", "image_idx"]].reset_index(drop=True)
        texts["w"] = F.text_weights(texts).to_numpy() if weighted else 1.0
        sp = df[(df.case == c) & (df.field == field) & df.first_use]
        vibe = sp[sp.vibe == "yes"].groupby("doc_id").size()
        texts["vibe"] = texts.doc_id.map(vibe).fillna(0).to_numpy() * per
        sp = sp.merge(texts[["doc_id", "w"]], on="doc_id", how="inner")
        units = [("vibe_per_100", texts, texts.vibe.to_numpy())]
        for a in sorted(sp.aspect.dropna().unique()):
            units.append((f"aspect: {a}", sp, (sp.aspect == a).to_numpy(float)))
        for measure, frame, x in units:
            for m in [m for m in MODELS if m in set(frame.model)]:
                mine = (frame.model == m).to_numpy()
                mm, sm = _wmean_se(x[mine], frame.w.to_numpy()[mine])
                mo, so = _wmean_se(x[~mine], frame.w.to_numpy()[~mine])
                rows.append({"measure": measure, "model": m, "case": c, "value": mm, "others": mo,
                             "estimate": mm - mo, "se": float(np.hypot(sm, so))})
    long = pd.DataFrame(rows)
    out = long.pivot_table(index=["measure", "model"], columns="case",
                           values=["value", "others", "estimate", "se"], aggfunc="first")
    out.columns = [f"{k}_{c}" for k, c in out.columns]
    out = out.dropna().reset_index()
    for c in CASE_PAIR:
        out[f"z_{c}"] = out[f"estimate_{c}"] / out[f"se_{c}"]
        out[f"p_bonf_{c}"] = _bonferroni(pd.Series(_p_two_sided(out[f"z_{c}"]), index=out.index))
    z, p = F.difference_test(out.estimate_restaurant, out.se_restaurant, out.estimate_street, out.se_street)
    out["z_diff"], out["p_diff"] = z, p
    out["p_diff_bonf"] = _bonferroni(out.p_diff)
    out["class"] = _classify(out.p_bonf_restaurant < alpha, out.p_bonf_street < alpha,
                             np.sign(out.z_restaurant), np.sign(out.z_street), out.p_diff_bonf < alpha)
    cols = ["measure", "model", "class"] + [f"{k}_{c}" for c in CASE_PAIR
                                            for k in ("value", "others", "estimate", "se", "z", "p_bonf")]
    out = out[cols + ["z_diff", "p_diff", "p_diff_bonf"]]
    for col in out.columns:
        if col.startswith("p_"):
            out[col] = out[col].map(lambda v: float(f"{v:.2g}"))
        elif out[col].dtype == float:
            out[col] = out[col].round(4)
    order = {m: i for i, m in enumerate(MODELS)}
    out = out.assign(_v=out.measure != "vibe_per_100", _m=out.model.map(order)).sort_values(
        ["_v", "measure", "_m"]).drop(columns=["_v", "_m"]).reset_index(drop=True)
    out.attrs["weighted"] = bool(weighted and F.WEIGHTED)
    return out


CASE_FINDINGS = ("signatures", "exclusive", "rates")


def case_tables(raw: pd.DataFrame, df: pd.DataFrame, totals: pd.Series, field: str,
                cue_only: bool = True) -> Dict[str, pd.DataFrame]:
    """Give the 3 tables of the case comparison for 1 field, by the names of `CASE_FINDINGS`."""
    return {"signatures": case_signatures(df, field, cue_only=cue_only),
            "exclusive": case_exclusive(df, totals, field, cue_only=cue_only),
            "rates": case_rates(raw, df, field)}


def case_summaries(tables: Dict[str, Dict[str, pd.DataFrame]], name: str) -> pd.DataFrame:
    """Stack the `case_summary` of 1 finding for each field. `tables` is {field: `case_tables`}."""
    rows = []
    for field, by_name in tables.items():
        s = case_summary(by_name[name])
        if not s.empty:
            rows.append(s.assign(field=field).reset_index().set_index(["field", "model"]))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows).fillna(0).astype(int)
    known = CASE_CLASSES + VOCAB_CLASSES + EXCLUSIVE_CLASSES
    return out[[c for c in known if c in out.columns]]


def case_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count the rows of each class, for each model, in 1 table of `case_signatures`, `case_exclusive` or `case_rates`."""
    if table.empty:
        return pd.DataFrame()
    known = CASE_CLASSES + VOCAB_CLASSES + EXCLUSIVE_CLASSES
    out = pd.crosstab(table.model, table["class"])
    out = out[[c for c in known if c in out.columns]]
    out = out.reindex([m for m in MODELS if m in out.index])
    out.loc["all"] = out.sum()
    out.index.name = "model"
    out.columns.name = None
    return out


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def _texts(case: str) -> pd.Series:
    if case not in _TEXT_CACHE:
        _TEXT_CACHE[case] = pd.read_parquet(INPUT[case], columns=["doc_id", "text"]).set_index("doc_id").text
    return _TEXT_CACHE[case]


def quotes(df: pd.DataFrame, case: str, field: str, model: str, head: str, n: int = 12,
           width: int = 90, seed: int = 777) -> pd.DataFrame:
    """Give spans of 1 head in their context, with the text id and the NTA."""
    part = df[(df.case == case) & (df.field == field) & (df.model == model) & (df["head"] == head)]
    if part.empty:
        return pd.DataFrame()
    take = part.sample(n=min(n, len(part)), random_state=seed)
    texts = _texts(case)
    rows = []
    for r in take.itertuples():
        t = texts.get(r.doc_id, "")
        a, b = int(r.char_start), int(r.char_end)
        ctx = t[max(0, a - width):a] + "[" + t[a:b] + "]" + t[b:b + width]
        rows.append({"doc_id": r.doc_id, "nta": r.nta_name, "target": r.target,
                     "context": _SPACE.sub(" ", ctx)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_signatures(table: pd.DataFrame, n: int = 12, title: Optional[str] = None):
    """Draw the top `n` signature heads of each model, 1 panel each. Return the figure."""
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("no signature to draw")
    models = [m for m in MODELS if m in set(table.model)]
    # 1 column for each family, the small model above the large model.
    families = list(dict.fromkeys(F.MODEL_FAMILY[m] for m in models))
    cols, rows = len(families), 2
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 0.26 * n * rows + 1.2 * rows), squeeze=False)
    used = set()
    for m in models:
        fam = [x for x in models if F.MODEL_FAMILY[x] == F.MODEL_FAMILY[m]]
        ax = axes[fam.index(m), families.index(F.MODEL_FAMILY[m])]
        used.add(id(ax))
        t = table[table.model == m].head(n).iloc[::-1]
        ax.barh(t["head"], t.z, color=MODEL_COLORS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW)
        ax.set_title(m, fontsize=9, color=S.EDGE)
        ax.set_xlabel("log-odds z")
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.flat:
        if id(ax) not in used:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def plot_concentration(table: pd.DataFrame, title: Optional[str] = None):
    """Draw the rarefied entropy and the top-10 share of each model. Return the figure."""
    import matplotlib.pyplot as plt

    t = table.reindex([m for m in MODELS if m in table.index])
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    colors = [MODEL_COLORS[m] for m in t.index]
    axes[0].barh(t.index[::-1], t.entropy_rare[::-1], color=colors[::-1], edgecolor=S.EDGE, linewidth=S.EDGE_LW)
    axes[0].set_xlabel(f"entropy of {int(t.k.iloc[0])} head uses (bits)")
    axes[1].barh(t.index[::-1], t.top10_share[::-1], color=colors[::-1], edgecolor=S.EDGE, linewidth=S.EDGE_LW)
    axes[1].set_xlabel("share of uses: the 10 most frequent heads")
    axes[1].set_yticklabels([])
    if title:
        fig.suptitle(title, fontsize=10, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def plot_case_signatures(table: pd.DataFrame, n_labels: int = 5, title: Optional[str] = None):
    """Draw the log-odds of each shared key in the restaurant case (x) and the street case (y).

    `table` is the output of `case_signatures`. 1 panel for each model, laid
    out as in `plot_signatures` (1 column for each family). The line is
    y = x. The labels name the `n_labels` keys with the largest absolute
    `z_diff`. A hollow marker is a key that the model writes in 0 texts of 1
    case: its log-odds in that case is the prior only, thus these keys sit on
    1 floor value. Return the figure.
    """
    import matplotlib.pyplot as plt

    t = table[table["class"].isin(CASE_CLASSES)] if not table.empty else table
    if t.empty:
        raise ValueError("no shared key to draw")
    models = [m for m in MODELS if m in set(t.model)]
    families = list(dict.fromkeys(F.MODEL_FAMILY[m] for m in models))
    cols, rows = len(families), 2
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    lo = float(min(t.estimate_restaurant.min(), t.estimate_street.min())) - 0.2
    hi = float(max(t.estimate_restaurant.max(), t.estimate_street.max())) + 0.2
    used = set()
    for m in models:
        fam = [x for x in models if F.MODEL_FAMILY[x] == F.MODEL_FAMILY[m]]
        ax = axes[fam.index(m), families.index(F.MODEL_FAMILY[m])]
        used.add(id(ax))
        g = t[t.model == m]
        ax.plot([lo, hi], [lo, hi], color=S.EDGE, linewidth=0.6, linestyle="--", zorder=1)
        ax.axhline(0, color=S.EDGE, linewidth=0.3, zorder=1)
        ax.axvline(0, color=S.EDGE, linewidth=0.3, zorder=1)
        zero = (g.texts_restaurant == 0) | (g.texts_street == 0)
        ax.scatter(g.estimate_restaurant[~zero], g.estimate_street[~zero], s=10, color=MODEL_COLORS[m],
                   edgecolor=S.EDGE, linewidth=0.2, alpha=0.7, marker=F.MODEL_MARKERS[m], zorder=2)
        ax.scatter(g.estimate_restaurant[zero], g.estimate_street[zero], s=10, facecolor="none",
                   edgecolor=MODEL_COLORS[m], linewidth=0.5, alpha=0.7, marker=F.MODEL_MARKERS[m], zorder=2)
        for r in g.reindex(g.z_diff.abs().sort_values(ascending=False).index).head(n_labels).itertuples():
            ax.annotate(r.head, (r.estimate_restaurant, r.estimate_street), fontsize=7, color=S.EDGE,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{m} ({len(g)} keys, {int(zero.sum())} hollow)", fontsize=9, color=S.EDGE)
        ax.set_xlabel("log-odds, restaurant")
        ax.set_ylabel("log-odds, street")
    for ax in axes.flat:
        if id(ax) not in used:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, color=S.EDGE)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def plot_cloud(weights: Dict[str, float], title: str, model: Optional[str] = None):
    """Draw 1 word cloud with the house style and the inks of `model` (`_f4.plot_cloud`)."""
    return F.plot_cloud(weights, title, model=model)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(out_dir: str | Path, raw: pd.DataFrame, df: pd.DataFrame, cue_only: bool = True,
           n_sig: int = 40) -> List[str]:
    """Write every table and figure of the notebook, for every (case, field, track, target)."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = text_totals(raw)
    written = []

    def _save(obj, name):
        p = out / name
        if hasattr(obj, "savefig"):
            obj.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(obj)
        else:
            obj.to_csv(p)
        written.append(str(p))

    _save(coverage(raw, df), "coverage.csv")
    _save(vibe_rates(df, totals), "vibe_rates.csv")
    _save(aspect_shares(df), "aspect_shares.csv")
    for case in CASES:
        for field in FIELDS:
            part = select(df, case, field, "C", "both", cue_only=False)
            _save(exclusive_heads(part, df, totals), f"exclusive_heads_{case}_{field}.csv")
    if "cue_class" in df:
        for case in CASES:
            for field in FIELDS:
                _save(cue_kind_shares(df, case, field), f"cue_kind_shares_{case}_{field}.csv")
                _save(social_heads(df, text_totals(raw), case, field, n=25), f"social_heads_{case}_{field}.csv")
    for case in CASES:
        for field in FIELDS:
            for track in TRACKS:
                for target in (TARGETS if case == "street" else ("both",)):
                    part = select(df, case, field, track, target, cue_only)
                    if part.empty:
                        continue
                    tag = f"{case}_{field}_track{track}_{target}"
                    sig = signatures(part, totals, max_rows=n_sig)
                    if not sig.empty:
                        _save(sig, f"signature_{tag}.csv")
                        _save(plot_signatures(sig, title=tag), f"signature_{tag}.png")
                        for m, g in sig.groupby("model"):
                            _save(plot_cloud(dict(zip(g["head"], g.z)), f"{m}: {tag}", model=m),
                                  f"cloud_{tag}_{m}.png")
                    _save(top_heads(part, totals, n=n_sig), f"top_heads_{tag}.csv")
                    conc = concentration(part)
                    if not conc.empty:
                        _save(conc, f"concentration_{tag}.csv")
                        _save(plot_concentration(conc, title=tag), f"concentration_{tag}.png")
                    _save(nta_overlap(part), f"nta_overlap_{tag}.csv")
                    if track == "V":
                        _save(frames(part), f"frames_{tag}.csv")
    if has_cues(df, "street"):
        for field in FIELDS:
            _save(robustness(df, totals, field), f"robustness_street_{field}.csv")
    # Section 11: the findings in both cases, and the findings in 1 case only.
    tables = {field: case_tables(raw, df, totals, field, cue_only=cue_only) for field in FIELDS}
    for field, by_name in tables.items():
        for name, t in by_name.items():
            _save(t, f"cases_{name}_{field}.csv")
        if not by_name["signatures"].empty:
            _save(plot_case_signatures(by_name["signatures"], title=f"restaurant against street / {field}"),
                  f"cases_signatures_{field}.png")
    for name in CASE_FINDINGS:
        _save(case_summaries(tables, name), f"cases_summary_{name}.csv")
    return written


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


def _order_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "model" not in frame.columns:
        return frame
    key = frame.model.map({m: i for i, m in enumerate(MODELS)})
    return frame.assign(_k=key).sort_values(["_k"], kind="stable").drop(columns="_k").reset_index(drop=True)


def _order_index(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.index.to_frame(index=False)
    idx["_c"] = idx.case.map({c: i for i, c in enumerate(CASES)})
    idx["_f"] = idx.field.map({f: i for i, f in enumerate(FIELDS)})
    idx["_m"] = idx.model.map({m: i for i, m in enumerate(MODELS)})
    order = idx.sort_values(["_c", "_f", "_m"]).index
    return frame.iloc[order]


# The main round is x2 (D130, D131). v1 is for history only.
use_round(os.environ.get("F4DESC_ROUND") or "x2")
