"""F11: the cue spans of the pairwise traces against the F9 single-image traces.

The pairwise battery asks `Which restaurant would you rather eat at?` over 2
images. F9 asks `Does this look like a nice restaurant to eat at?` over 1
image. If a model applies 1 standard, then it gives a cue the same valence in
both paradigms. This module measures the distance between the 2 sets of cue
spans (`facct-runbook.md`, F11, D150 and D151).

| Arm | Texts | Extractor and schema |
|-----|-------|----------------------|
| `pairwise` | the thinking traces of the canonical trace runs (10,000 pairs) | the primary extractor, `urban_cues/v2` |
| `pairwise_alt` | the same traces, 2,000 pairs (reference R1) | the other extractor, `urban_cues/v2` |
| `f9` | the F9 `reasoning` field (10,000 images) | DeepSeek-V4-Flash, the F9 schema |
| `f4` | the F4 `reasoning` field on the same images (reference R2) | DeepSeek-V4-Flash, the F9 schema |
| `repeat` | the 1,000 pairwise repeats (reference R3) | the primary extractor |

The primary extractor of a judge is the strongest model of the OTHER family
(`research/notebooks/cvpr/_extractions.ARMS`): Qwen3.8-27B reads the gemma-4-12b
traces, and Gemma-4-31B reads the qwen3.5-9b traces. The other extractor
read a sample of 2,000 traces as the control.

The text unit
-------------
1 text counts 1 key 1 time. A pairwise trace speaks of 2 images, thus 1
pairwise text is 1 (pair, image) side: the part of the trace that the
`image` attribute gives to 1 photograph. That is the single-image analog of
1 F9 answer. The bootstrap resamples the pair (pairwise) or the image (F9),
because the 2 sides of 1 pair are not independent.

The units
---------
| Unit | Distribution | Key |
|------|--------------|-----|
| U1 | P(category) | the category after the crosswalk |
| U2 | P(valence \\| category) | 1 category |
| U3 | P(valence \\| cue) | 1 content word or 2 adjacent content words of a span |
| U4 | P(word \\| valence) | `good` or `bad`; the support is the U3 unigrams |

**Warning: the extractor differs between the arms.** A raw divergence mixes
the paradigm with the extractor. Read each value against the references
(`references`), never alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"

from research.notebooks.cvpr import _cues_across as CA  # noqa: E402

__version__ = "1.3.0"

DATA = REPO_ROOT / "data"
CANON = _CVPR / "canonical_data" / "trace" / "canonical"
SUBJ = DATA / "f4_traces" / "subj"

#: The 2 models that have both a pairwise trace run and an F9 run.
MODELS = {"gemma-4-12b": "gemma12b", "qwen3.5-9b": "qwen9b"}

#: case -> (pairwise case, F9 cue folder, F9 file stem, F4 file stem)
CASES = {
    "restaurant": ("restaurants", "cues", "f4_restaurant_subj_x2", "f4_restaurant_x2"),
    "street": ("street_photography", "cues_street", "f4_street_subj_x2", "f4_street_x2"),
}

#: The pairwise extractors of each judge: arm -> folder suffix.
PAIRWISE_EXTRACTORS = {
    "gemma-4-12b": {"pairwise": "qwen27b", "pairwise_alt": "gemma31b"},
    "qwen3.5-9b": {"pairwise": "gemma31b", "pairwise_alt": "qwen27b"},
}

VALENCES = ("bad", "neutral", "good")

#: The F9 schema has 13 categories, `urban_cues/v2` has 11. Step 0 read the
#: spans: the pairwise `people` holds crowds and street life, as the F9
#: `activity` does, and the pairwise `other` holds the colours and murals
#: (D151, Q125).
CROSSWALK = {"activity": "people", "color": "other"}

#: Words that the stopword list of scikit-learn holds but that name a cue.
KEEP_WORDS = frozenset((
    "empty", "full", "front", "back", "side", "top", "bottom", "fire", "well",
    "less", "more", "few", "many", "old", "new", "behind", "inside", "outside",
))

#: Words that name the photograph or the frame, not the place.
FRAME_WORDS = frozenset((
    "image", "images", "photo", "photos", "picture", "pictures", "shot",
    "view", "scene", "frame", "crop", "visible", "seen", "appears", "looks",
    "look", "like", "seems", "feel", "feels", "make", "makes", "ha", "wa",
    "doe", "does", "did", "can",
))

#: The floor of a U3 key: texts in EACH arm of the paradigm comparison.
MIN_TEXTS = 25

SMOOTH = 0.5


def _stopwords() -> frozenset:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset((set(ENGLISH_STOP_WORDS) - KEEP_WORDS) | FRAME_WORDS)


STOP = _stopwords()


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def _spans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the quotable `visual_evidence` spans, 1 copy each, and open the attributes."""
    df = df[(df.extraction_class == "visual_evidence") & df.attributes_json.notna()]
    if "is_quotable" in df.columns:
        df = df[df.is_quotable.astype(str) == "True"]
    if "is_duplicate" in df.columns:
        df = df[df.is_duplicate.astype(str) != "True"]
    attrs = df.attributes_json.map(json.loads)
    out = df.copy()
    out["category"] = attrs.map(lambda a: a.get("category"))
    out["valence"] = attrs.map(lambda a: a.get("valence"))
    out["image"] = attrs.map(lambda a: a.get("image"))
    out = out[out.valence.isin(VALENCES) & out.category.notna()]
    out["category"] = out.category.replace(CROSSWALK)
    return out


def load_pairwise(model: str, case: str, arm: str = "pairwise") -> pd.DataFrame:
    """The spans of 1 canonical pairwise trace run, 1 row for each span.

    `arm` is `pairwise` (the primary extractor) or `pairwise_alt` (the
    other one). The 1,000 repeats have `repeat` True; the main arm drops them.
    """
    pcase = CASES[case][0]
    folder = DATA / f"trace_extractions_v2_{MODELS[model]}_by_{PAIRWISE_EXTRACTORS[model][arm]}"
    cols = ["pair_id", "judge_answered", "extraction_class", "extraction_text",
            "attributes_json", "is_quotable", "is_duplicate"]
    df = pd.read_parquet(folder / f"{pcase}_extractions.parquet", columns=cols)
    return _pairwise_frame(df, model, case, arm)


def _pairwise_frame(df: pd.DataFrame, model: str, case: str, arm: str) -> pd.DataFrame:
    """Open the spans of 1 pairwise run and join the images of each pair."""
    pcase = CASES[case][0]
    df = df[df.judge_answered.astype(str) != "False"]
    df = _spans(df)
    df = df[df.image.isin(["A", "B"])]

    pairs = pd.read_parquet(CANON / f"{pcase}__{model}" / "pairs.parquet")
    key = "unit_uid" if case == "restaurant" else "recording_id"
    pairs = pairs[["pair_id", "canonical_pair_id", "repeat_idx",
                   f"{key}_a", f"{key}_b"]]
    df = df.merge(pairs, on="pair_id", how="left", validate="many_to_one")
    df["image_key"] = np.where(df.image == "A", df[f"{key}_a"], df[f"{key}_b"]).astype(str)
    df["text_id"] = df.pair_id + "|" + df.image
    df["cluster"] = df.canonical_pair_id
    df["repeat"] = df.repeat_idx > 0
    df["model"], df["case"], df["arm"] = model, case, arm
    return df[["model", "case", "arm", "text_id", "cluster", "image_key", "repeat",
               "pair_id", "canonical_pair_id", "image", "category", "valence",
               "extraction_text"]].reset_index(drop=True)


def load_single(model: str, case: str, arm: str = "f9",
                field: str = "reasoning") -> pd.DataFrame:
    """The spans of the F9 (`f9`) or the F4 (`f4`) answers, 1 row for each span."""
    _, folder, f9_stem, f4_stem = CASES[case]
    stem = f9_stem if arm == "f9" else f4_stem
    cols = ["model", "field", "doc_id", "image_idx", "uid", "extraction_class",
            "extraction_text", "attributes_json", "is_quotable", "is_duplicate"]
    df = pd.read_parquet(SUBJ / "x2" / folder / f"{stem}_extractions.parquet", columns=cols)
    df = df[(df.model == model) & (df.field == field)]
    df = _spans(df)
    df["text_id"] = df.doc_id
    df["cluster"] = df.image_idx.astype(str)
    df["image_key"] = df.uid.astype(str)
    df["repeat"] = False
    df["case"], df["arm"] = case, arm
    return df[["model", "case", "arm", "text_id", "cluster", "image_key", "repeat",
               "category", "valence", "extraction_text"]].reset_index(drop=True)


#: The shard outputs of R4 (`research/scripts/f4/extract.sbatch`, arm `f11_pw_cues`).
R4_ROOT = REPO_ROOT / "multirun" / "f4_f11_pw_cues_dsv4"


def load_r4(model: str, case: str, root: Path = R4_ROOT) -> Optional[pd.DataFrame]:
    """R4: the DeepSeek-V4-Flash spans of the pairwise traces, or None before the job ends.

    The script reads each finished shard; a shard that still runs has no
    output parquet yet. The input holds first answers only.
    """
    import glob
    files = sorted(glob.glob(str(root / "runs" / "*" / "outputs" / "extractions" / "*.parquet")))
    if not files:
        return None
    cols = ["f11_model", "f11_case", "pair_id", "judge_answered", "extraction_class",
            "extraction_text", "attributes_json", "is_quotable", "is_duplicate"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df[(df.f11_model == model) & (df.f11_case == CASES[case][0])]
    if df.empty:
        return None
    out = _pairwise_frame(df, model, case, "pairwise_dsv4")
    return out[~out.repeat].reset_index(drop=True)


def load_arms(model: str, case: str) -> Dict[str, pd.DataFrame]:
    """All the arms of 1 (model, case). `repeat` holds the repeats and their first answers."""
    pw = load_pairwise(model, case, "pairwise")
    rep_ids = set(pw.loc[pw.repeat, "canonical_pair_id"])
    arms = {
        "pairwise": pw[~pw.repeat].reset_index(drop=True),
        "pairwise_alt": (lambda d: d[~d.repeat].reset_index(drop=True))(
            load_pairwise(model, case, "pairwise_alt")),
        "f9": load_single(model, case, "f9"),
        "f4": load_single(model, case, "f4"),
        # R3: the repeat and its first answer share the cluster, thus the
        # bootstrap pairs them.
        "repeat": pw[pw.repeat].reset_index(drop=True),
        "first": pw[~pw.repeat & pw.canonical_pair_id.isin(rep_ids)].reset_index(drop=True),
    }
    r4 = load_r4(model, case)
    if r4 is not None:
        arms["pairwise_dsv4"] = r4
    return arms


# --------------------------------------------------------------------------
# Cues
# --------------------------------------------------------------------------

#: The words that negate the next words of a span. `_tokens` turns `n't`
#: into `not`.
NEGATORS = frozenset(("not", "no", "never", "without", "nor", "lack", "lacks", "lacking"))

_NOT = __import__("re").compile(r"(\w)n[\'\u2019]t\b")
_CONTRACTION = {"can": "can", "won": "will", "isn": "is", "aren": "are", "wasn": "was",
                "weren": "were", "doesn": "does", "don": "do", "didn": "did"}


def _tokens(span) -> List[str]:
    """The rules of `_cues_across.normalise_cue` without the predicate swap.

    The swap (`X is Y` to `Y X`) moves words across a negator, thus this
    unit keeps the order of the text. `n't` becomes `not` first.
    """
    if span is None or (isinstance(span, float) and pd.isna(span)):
        return []
    text = _NOT.sub(r"\1 not", str(span).lower())
    text = text.replace("/", " or ").replace("&", " and ").replace("-", " ").replace("_", " ")
    toks = CA._NON_WORD.sub(" ", text).split()
    toks = [_CONTRACTION.get(t, t) for t in toks]
    toks = [t for t in toks if t not in CA._ARTICLES and t not in CA._DEGREE]
    return [CA._singular(t) for t in toks]

#: The number of tokens after a negator that it negates. `but` ends the scope.
NEG_SCOPE = 4


def cue_words(span) -> List[str]:
    """The content words and the pairs of adjacent content words of 1 span.

    The span goes through the rules of `_cues_across.normalise_cue` first
    (`_tokens`), thus `trees` joins `tree` and `well-maintained` joins
    `well maintained`. A pair needs 2 content words side by side.

    A word in the scope of a negator becomes `not <word>`. The F9 clauses
    often deny a standard (`the brick isn't pristine`, `it doesn't look
    dingy`). Without this rule, `pristine` gets a bad valence there and a
    good one in the pairwise phrases, and the change of sign is an artefact
    of the unit.
    """
    toks = _tokens(span)
    neg, left = [], 0
    for t in toks:
        if t in NEGATORS:
            left = NEG_SCOPE
            neg.append(False)
            continue
        if t == "but":
            left = 0
        neg.append(left > 0)
        left = max(left - 1, 0)
    ok = [len(t) > 2 and t.isalpha() and t not in STOP for t in toks]
    name = [f"not {t}" if n else t for t, n in zip(toks, neg)]
    words = [w for w, k in zip(name, ok) if k]
    pairs = [f"{'not ' if na else ''}{a} {b}"
             for a, b, ka, kb, na, nb in zip(toks, toks[1:], ok, ok[1:], neg, neg[1:])
             if ka and kb and na == nb]
    return list(dict.fromkeys(words + pairs))


def negated(span) -> bool:
    """True when the span holds a negator."""
    return any(t in NEGATORS for t in _tokens(span))


def add_cues(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each (span, cue)."""
    out = df.assign(cue=df.extraction_text.map(cue_words)).explode("cue")
    return out[out.cue.notna()].reset_index(drop=True)


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------

class Counts:
    """The count of each (key, valence) cell for each cluster, as a sparse matrix.

    A row is a cluster (a pair or an image). A column is a cell of the unit.
    `total(w)` sums the rows with the weights `w`, thus 1 matrix serves the
    point value and every bootstrap draw.
    """

    def __init__(self, matrix: sparse.csr_matrix, clusters: pd.Index, cells: pd.Index):
        self.matrix, self.clusters, self.cells = matrix, clusters, cells

    def total(self, w: Optional[np.ndarray] = None) -> np.ndarray:
        if w is None:
            return np.asarray(self.matrix.sum(axis=0)).ravel()
        return np.asarray(self.matrix.T @ w).ravel()


def unit_rows(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """The (cluster, cell) rows of 1 unit, 1 row for each text and cell.

    The cell of U1 is the category. The cell of U2 is (category, valence).
    The cell of U3 is (cue, valence). The cell of U4 is (valence, word), for
    the `good` and the `bad` spans and the unigram cues.
    """
    if unit == "U1":
        rows = df.assign(key=df.category, val="")
    elif unit == "U2":
        rows = df.assign(key=df.category, val=df.valence)
    elif unit == "U3":
        rows = df.assign(key=df.cue, val=df.valence)
    elif unit == "U4":
        rows = df[df.valence.isin(["good", "bad"]) & ~df.cue.str.contains(" ")]
        rows = rows.assign(key=rows.valence, val=rows.cue)
    else:
        raise ValueError(unit)
    rows = rows[["cluster", "text_id", "key", "val"]].drop_duplicates()
    return rows


def counts(rows: pd.DataFrame, cells: pd.MultiIndex,
           clusters: Optional[pd.Index] = None) -> Counts:
    """Build the sparse counts of 1 arm over a fixed list of cells."""
    clusters = pd.Index(sorted(rows.cluster.unique())) if clusters is None else clusters
    rows = rows[rows.set_index(["key", "val"]).index.isin(cells)]
    r = clusters.get_indexer(rows.cluster)
    c = cells.get_indexer(pd.MultiIndex.from_frame(rows[["key", "val"]]))
    keep = (r >= 0) & (c >= 0)
    m = sparse.csr_matrix((np.ones(keep.sum()), (r[keep], c[keep])),
                          shape=(len(clusters), len(cells)))
    return Counts(m, clusters, cells)


def support(unit: str, a: pd.DataFrame, b: pd.DataFrame,
            min_texts: int = MIN_TEXTS) -> pd.MultiIndex:
    """The cells of 1 unit for the comparison of arm `a` with arm `b`.

    U1 and U2 use every category of either arm. U3 keeps a cue with
    `min_texts` texts or more in EACH arm. U4 keeps the U3 unigrams.
    """
    ra, rb = unit_rows(a, unit), unit_rows(b, unit)
    if unit in ("U1", "U2"):
        keys = sorted(set(ra.key) | set(rb.key))
    else:
        if unit == "U4":
            ra3, rb3 = unit_rows(a, "U3"), unit_rows(b, "U3")
        else:
            ra3, rb3 = ra, rb
        na = ra3.groupby("key").text_id.nunique()
        nb = rb3.groupby("key").text_id.nunique()
        keys = sorted(set(na[na >= min_texts].index) & set(nb[nb >= min_texts].index))
        if unit == "U4":
            words = [k for k in keys if " " not in k]
            return pd.MultiIndex.from_product([["bad", "good"], words], names=["key", "val"])
    vals = [""] if unit == "U1" else list(VALENCES)
    return pd.MultiIndex.from_product([keys, vals], names=["key", "val"])


# --------------------------------------------------------------------------
# The measure
# --------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) in bits along the last axis."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / q), 0.0)
    return t.sum(axis=-1)


def _table(total: np.ndarray, cells: pd.MultiIndex) -> Tuple[np.ndarray, List[str]]:
    """Reshape a cell vector into (keys, values) with the smoothing added."""
    keys = list(dict.fromkeys(cells.get_level_values(0)))
    vals = list(dict.fromkeys(cells.get_level_values(1)))
    return total.reshape(len(keys), len(vals)) + SMOOTH, keys


def _rake(t: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Scale the valence columns so that the margin of `t` equals `target`."""
    margin = t.sum(axis=0) / t.sum()
    return t * (target / margin)


def measure(ta: np.ndarray, tb: np.ndarray, cells: pd.MultiIndex, unit: str,
            rake: bool = False) -> Dict[str, float]:
    """JS, the 2 KLs and, for U3, Spearman rho of the net valence.

    U1 is 1 distribution. U2 and U3 are 1 distribution for each key; the mean
    weights each key by its mean share of texts in the 2 arms. U4 is 1
    distribution over the words for each valence, with equal weights.
    `rake` sets the valence margins of both arms to their mean first.
    """
    a, keys = _table(ta, cells)
    b, _ = _table(tb, cells)
    if unit == "U1":
        a, b = a.T, b.T
    if rake and unit in ("U2", "U3"):
        target = (a.sum(axis=0) / a.sum() + b.sum(axis=0) / b.sum()) / 2
        a, b = _rake(a, target), _rake(b, target)
    pa = a / a.sum(axis=1, keepdims=True)
    pb = b / b.sum(axis=1, keepdims=True)
    m = (pa + pb) / 2
    js = 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)
    kl_ab, kl_ba = _kl(pa, pb), _kl(pb, pa)
    if unit in ("U2", "U3"):
        w = a.sum(axis=1) / a.sum() + b.sum(axis=1) / b.sum()
        w = w / w.sum()
    else:
        w = np.full(len(js), 1 / len(js))
    out = {"js": float(w @ js), "kl_ab": float(w @ kl_ab), "kl_ba": float(w @ kl_ba)}
    if unit == "U3":
        vi = {v: i for i, v in enumerate(VALENCES)}
        na = (pa[:, vi["good"]] - pa[:, vi["bad"]])
        nb = (pb[:, vi["good"]] - pb[:, vi["bad"]])
        out["rho"] = float(spearmanr(na, nb).statistic) if len(na) > 2 else np.nan
    return out


def _weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """1 bootstrap draw of the clusters, as counts."""
    return rng.multinomial(n, np.full(n, 1 / n)).astype(float)


def compare(a: pd.DataFrame, b: pd.DataFrame, unit: str, cells: pd.MultiIndex,
            n_boot: int = 1000, paired: bool = False, seed: int = 0,
            rake: bool = False, n_null: int = 200) -> Dict[str, float]:
    """The point value, the null and the 95% bootstrap interval of `measure`.

    `paired` draws the shared clusters 1 time for both arms (R1, R2, R3: the
    same texts or the same images). Else each arm draws its own clusters.

    A divergence of 2 samples is larger than 0 when the 2 distributions are
    equal, and the bias grows when the sample gets smaller. Thus each
    comparison also gets a permutation null: the arm labels are shuffled
    (unpaired: over the pooled clusters, with the same sizes; paired: a swap
    inside each cluster). `js_excess` is the JS less the mean of the null,
    and it compares across sample sizes. The bootstrap gives the spread only:
    the interval of each value is the 2.5% and 97.5% distance of the draws
    from their median, put around the point. (A bootstrap draw repeats
    clusters, thus its divergence has its own upward bias.)
    """
    ra, rb = unit_rows(a, unit), unit_rows(b, unit)
    if paired:
        shared = pd.Index(sorted(set(ra.cluster) & set(rb.cluster)))
        ca, cb = counts(ra, cells, shared), counts(rb, cells, shared)
    else:
        ca, cb = counts(ra, cells), counts(rb, cells)
    point = measure(ca.total(), cb.total(), cells, unit, rake)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        wa = _weights(len(ca.clusters), rng)
        wb = wa if paired else _weights(len(cb.clusters), rng)
        draws.append(measure(ca.total(wa), cb.total(wb), cells, unit, rake))
    boot = pd.DataFrame(draws)

    nulls = []
    if paired:
        for _ in range(n_null):
            s = (rng.random(len(ca.clusters)) < 0.5).astype(float)
            ta = ca.total(s) + cb.total(1 - s)
            tb = cb.total(s) + ca.total(1 - s)
            nulls.append(measure(ta, tb, cells, unit, rake)["js"])
    else:
        pool = sparse.vstack([ca.matrix, cb.matrix]).tocsr()
        na = len(ca.clusters)
        for _ in range(n_null):
            h = (rng.permutation(pool.shape[0]) < na).astype(float)
            ta = np.asarray(pool.T @ h).ravel()
            tb = np.asarray(pool.T @ (1 - h)).ravel()
            nulls.append(measure(ta, tb, cells, unit, rake)["js"])

    out = dict(point)
    for k in point:
        lo, mid, hi = np.nanpercentile(boot[k], [2.5, 50, 97.5])
        out[f"{k}_lo"], out[f"{k}_hi"] = float(point[k] - (mid - lo)), float(point[k] + (hi - mid))
    out["js_null"] = float(np.mean(nulls))
    out["js_null_hi"] = float(np.percentile(nulls, 95))
    out["js_excess"] = out["js"] - out["js_null"]
    out["js_excess_lo"] = out["js_lo"] - out["js_null"]
    out["js_excess_hi"] = out["js_hi"] - out["js_null"]
    out["clusters_a"], out["clusters_b"] = len(ca.clusters), len(cb.clusters)
    out["keys"] = len(dict.fromkeys(cells.get_level_values(0)))
    return out


def split_half(df: pd.DataFrame, unit: str, cells: pd.MultiIndex,
               n: int = 200, seed: int = 0, rake: bool = False) -> Dict[str, float]:
    """R0: the divergence between 2 random halves of 1 arm, by cluster.

    Each half has half the texts, thus R0 is an upper bound on the noise of
    the full arm.
    """
    rows = unit_rows(df, unit)
    c = counts(rows, cells)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        h = rng.permutation(len(c.clusters)) < len(c.clusters) // 2
        vals.append(measure(c.total(h.astype(float)), c.total((~h).astype(float)),
                            cells, unit, rake))
    boot = pd.DataFrame(vals)
    out = {k: float(boot[k].mean()) for k in boot}
    for k in boot:
        lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    out["clusters_a"] = out["clusters_b"] = len(c.clusters) // 2
    out["keys"] = len(dict.fromkeys(cells.get_level_values(0)))
    return out


#: The comparisons of 1 (model, case): name -> (arm a, arm b, paired).
COMPARISONS = {
    "paradigm": ("pairwise", "f9", False),
    "R1_extractor": ("pairwise", "pairwise_alt", True),
    "R4_extractor": ("pairwise", "pairwise_dsv4", True),
    "R2_prompt": ("f4", "f9", True),
    "R3_retest": ("first", "repeat", True),
}


def run_cell(model: str, case: str, units: Sequence[str] = ("U1", "U2", "U3", "U4"),
             n_boot: int = 1000, arms: Optional[Dict[str, pd.DataFrame]] = None
             ) -> pd.DataFrame:
    """Every comparison and every unit of 1 (model, case), as 1 long table.

    Each comparison uses the support of the paradigm comparison, thus all the
    rows of 1 unit measure the same keys.
    """
    arms = arms or {k: add_cues(v) for k, v in load_arms(model, case).items()}
    rows = []
    for unit in units:
        cells = support(unit, arms["pairwise"], arms["f9"])
        for name, (a, b, paired) in COMPARISONS.items():
            if a not in arms or b not in arms:
                continue
            for rake in ((False, True) if unit in ("U2", "U3") else (False,)):
                r = compare(arms[a], arms[b], unit, cells, n_boot, paired, rake=rake)
                rows.append({"model": model, "case": case, "unit": unit,
                             "comparison": name, "raked": rake, **r})
        for arm in ("pairwise", "f9"):
            for rake in ((False, True) if unit in ("U2", "U3") else (False,)):
                r = split_half(arms[arm], unit, cells, rake=rake)
                rows.append({"model": model, "case": case, "unit": unit,
                             "comparison": f"R0_{arm}", "raked": rake, **r})
    return pd.DataFrame(rows)


def cue_table(arms: Dict[str, pd.DataFrame], cells: pd.MultiIndex) -> pd.DataFrame:
    """The net valence and the JS of each U3 cue, pairwise against F9."""
    ta = counts(unit_rows(arms["pairwise"], "U3"), cells).total()
    tb = counts(unit_rows(arms["f9"], "U3"), cells).total()
    a, keys = _table(ta, cells)
    b, _ = _table(tb, cells)
    pa = a / a.sum(axis=1, keepdims=True)
    pb = b / b.sum(axis=1, keepdims=True)
    m = (pa + pb) / 2
    vi = {v: i for i, v in enumerate(VALENCES)}
    out = pd.DataFrame({
        "cue": keys,
        "n_pairwise": (a.sum(axis=1) - SMOOTH * 3).round().astype(int),
        "n_f9": (b.sum(axis=1) - SMOOTH * 3).round().astype(int),
        "net_pairwise": pa[:, vi["good"]] - pa[:, vi["bad"]],
        "net_f9": pb[:, vi["good"]] - pb[:, vi["bad"]],
        "js": 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m),
    })
    out["sign_flip"] = (np.sign(out.net_pairwise) * np.sign(out.net_f9) < 0) & (
        (out.net_pairwise.abs() > 0.1) & (out.net_f9.abs() > 0.1))
    return out.sort_values("js", ascending=False).reset_index(drop=True)


def describe(arms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Step 0 and step 1: the texts, the spans and the words of each arm."""
    rows = []
    for name, df in arms.items():
        spans = df.drop_duplicates(["text_id", "extraction_text"])
        rows.append({
            "arm": name, "clusters": df.cluster.nunique(), "texts": df.text_id.nunique(),
            "spans": len(spans),
            "words_per_span": spans.extraction_text.str.split().str.len().median(),
            "share_negated": spans.extraction_text.map(negated).mean(),
            "share_good": (spans.valence == "good").mean(),
            "share_bad": (spans.valence == "bad").mean(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The judgment of the same restaurant (no KL)
# --------------------------------------------------------------------------

#: The score of 1 F9 verdict. `not_applicable` drops out.
VERDICT_SCORE = {"yes": 1.0, "no": -1.0, "mixed": 0.0, "graded": 0.0}


def f9_verdicts(model: str) -> pd.DataFrame:
    """The F9 answer of each restaurant: the label of the LAST `verdict` span of `response`.

    The F9 response is prose, thus the extractor's `verdict` span is the
    answer. A response that names 2 verdicts ends on its decision.
    """
    _, folder, stem, _ = CASES["restaurant"]
    cols = ["model", "field", "doc_id", "uid", "extraction_class", "attributes_json", "char_start"]
    df = pd.read_parquet(SUBJ / "x2" / folder / f"{stem}_extractions.parquet", columns=cols)
    df = df[(df.model == model) & (df.field == "response")
            & (df.extraction_class == "verdict") & df.attributes_json.notna()].copy()
    df["label"] = df.attributes_json.map(lambda a: json.loads(a).get("label"))
    df["pos"] = pd.to_numeric(df.char_start, errors="coerce").fillna(-1)
    last = df.sort_values("pos").groupby("doc_id").tail(1)
    last = last[last.label.isin(list(VERDICT_SCORE))]
    return pd.DataFrame({"unit_uid": last.uid.astype(str).values,
                         "f9_verdict": last.label.values,
                         "f9_score": last.label.map(VERDICT_SCORE).values})


def place_table(model: str, arms: Optional[Dict[str, pd.DataFrame]] = None,
                min_comparisons: int = 5) -> pd.DataFrame:
    """1 row for each restaurant that both paradigms judge.

    | Column | Source |
    |--------|--------|
    | `trace_score` | the mean ordinal score of the restaurant in the trace run (`_provenance.score_units`) |
    | `battery_score` | the same in the 1,000,000-pair battery (no thinking) |
    | `f9_score` | yes 1, no -1, mixed or graded 0 |
    | `net_pairwise`, `net_f9` | (good - bad) / spans on the image of the restaurant (U5) |
    """
    from research.notebooks.cvpr import _provenance as prov

    run = pd.read_parquet(CANON / f"restaurants__{model}" / "results.parquet")
    run = run[run.repeat_idx == 0][["pair_id", "relative_label"]]
    # The results hold the label; the pair table holds the restaurants.
    pairs = pd.read_parquet(CANON / f"restaurants__{model}" / "pairs.parquet",
                            columns=["pair_id", "unit_uid_a", "unit_uid_b",
                                     "unit_name_a", "unit_name_b"])
    run = run.merge(pairs, on="pair_id", how="left", validate="one_to_one")
    for c in ("unit_uid_a", "unit_uid_b"):
        run[c] = run[c].astype(str)
    trace = prov.score_units(run, min_comparisons=min_comparisons)
    trace = trace.assign(unit_uid=trace.unit_uid.astype(str))[
        ["unit_uid", "mean_score", "n_comparisons"]].rename(
        columns={"mean_score": "trace_score", "n_comparisons": "trace_n"})

    battery = None
    for rec in prov.discover_runs("restaurants"):
        if rec.model.startswith(model):
            b = prov.load_run(rec)
            for c in ("unit_uid_a", "unit_uid_b"):
                b[c] = b[c].astype(str)
            battery = prov.score_units(b)
            battery = battery.assign(unit_uid=battery.unit_uid.astype(str))[
                ["unit_uid", "mean_score", "n_comparisons"]].rename(
                columns={"mean_score": "battery_score", "n_comparisons": "battery_n"})
            break

    arms = arms or load_arms(model, "restaurant")

    def net(df: pd.DataFrame, name: str) -> pd.DataFrame:
        sp = df.drop_duplicates(["text_id", "extraction_text"])
        g = sp.groupby("image_key").valence
        return pd.DataFrame({
            f"net_{name}": g.apply(lambda v: ((v == "good").sum() - (v == "bad").sum()) / len(v)),
            f"spans_{name}": g.size()}).rename_axis("unit_uid").reset_index()

    out = f9_verdicts(model)
    out = out.merge(net(arms["f9"], "f9"), on="unit_uid", how="left")
    out = out.merge(net(arms["pairwise"], "pairwise"), on="unit_uid", how="left")
    out = out.merge(trace, on="unit_uid", how="left")
    if battery is not None:
        out = out.merge(battery, on="unit_uid", how="left")
    return out.assign(model=model)


def place_rho(table: pd.DataFrame, x: str, y: str, n_boot: int = 1000,
              seed: int = 0) -> Dict[str, float]:
    """Spearman rho of 2 columns over the restaurants with both, with a bootstrap interval."""
    d = table[[x, y]].dropna()
    rho = float(spearmanr(d[x], d[y]).statistic)
    rng = np.random.default_rng(seed)
    v = d.to_numpy()
    boots = [spearmanr(*v[rng.integers(0, len(v), len(v))].T).statistic for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"x": x, "y": y, "n": len(d), "rho": rho, "rho_lo": float(lo), "rho_hi": float(hi)}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

OUT = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "f11"
RESULTS = REPO_ROOT / "data" / "f11"

MODEL_LABEL = {"gemma-4-12b": "Gemma-4-12B", "qwen3.5-9b": "Qwen3.5-9B"}
CASE_LABEL = {"restaurant": "Restaurant", "street": "Street photography"}
COMPARISON_LABEL = {
    "paradigm": "pairwise vs. F9",
    "R1_extractor": "R1 extractor",
    "R4_extractor": "R4 extractor (DSV4)",
    "R2_prompt": "R2 F4 vs. F9",
    "R3_retest": "R3 retest",
}
UNIT_LABEL = {"U1": "U1  P(category)", "U2": "U2  P(valence | category)",
              "U3": "U3  P(valence | cue)", "U4": "U4  P(word | valence)"}


def _colour(model: str) -> str:
    from research.notebooks.cvpr import _style as S
    return S.PAL["teal"] if model.startswith("gemma") else S.PAL["blue"]


def _meta(fig, title: str, caption: str) -> None:
    fig._plot_meta = {"plot-title": title, "plot-caption": caption, "plot-tags": ["F11"]}


def plot_divergence(table: pd.DataFrame, units: Sequence[str] = ("U1", "U2", "U3"),
                    raked: bool = False):
    """The excess JS of each comparison, 1 panel for each (case, unit).

    The paradigm row is the test; the R rows are the anchors. A value of 0
    is the permutation null: the 2 arms differ no more than 2 random halves.
    """
    import matplotlib.pyplot as plt

    t = table[~table.comparison.str.startswith("R0") & table.unit.isin(units)]
    t = t[(t.raked == raked) | ~t.unit.isin(["U2", "U3"])]
    comps = [c for c in COMPARISON_LABEL if c in set(t.comparison)]
    cases = [c for c in CASES if c in set(t.case)]
    fig, axes = plt.subplots(len(cases), len(units), figsize=(7.0, 1.0 + 0.55 * len(comps) * len(cases)),
                             sharey=True, squeeze=False)
    for i, case in enumerate(cases):
        for j, unit in enumerate(units):
            ax = axes[i, j]
            ax.axvline(0, color="0.6", lw=0.6)
            for k, model in enumerate(MODELS):
                part = t[(t.case == case) & (t.unit == unit) & (t.model == model)].set_index("comparison")
                for y, comp in enumerate(comps):
                    if comp not in part.index:
                        continue
                    r = part.loc[comp]
                    yy = y + (k - 0.5) * 0.3
                    ax.plot([r.js_excess_lo, r.js_excess_hi], [yy, yy], color=_colour(model), lw=1.2)
                    ax.plot(r.js_excess, yy, "o", ms=4, color=_colour(model),
                            mfc=_colour(model) if comp == "paradigm" else "white",
                            label=MODEL_LABEL[model] if (y == 0 and i == 0 and j == 0) else None)
            ax.set_yticks(range(len(comps)))
            ax.set_yticklabels([COMPARISON_LABEL[c] for c in comps])
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(UNIT_LABEL[unit], fontsize=8)
            if i == len(cases) - 1:
                ax.set_xlabel("JS above the null (bits)")
            if j == 0:
                ax.set_ylabel(CASE_LABEL[case])
    # The axes share y, thus 1 call inverts all of them. The test row is on top.
    axes[0, 0].invert_yaxis()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=7,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _meta(fig, "f11_divergence",
          "The Jensen-Shannon divergence above the permutation null, with 95% bootstrap intervals. "
          "The filled dot compares the pairwise traces with the F9 single-image traces of the same model. "
          "The hollow dots are the anchors: R1 and R4 change the extractor on the same pairwise traces, "
          "R2 changes the prompt (F4 description against F9 judgment) on the same images, "
          "and R3 repeats the same pair.")
    return fig


def plot_cues(cues: pd.DataFrame, n_label: int = 5):
    """The net valence of each U3 cue, pairwise (x) against F9 (y), 1 panel for each (model, case)."""
    import matplotlib.pyplot as plt

    from research.notebooks.cvpr import _style as S

    cells = [(m, c) for m in MODELS for c in CASES]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4), sharex=True, sharey=True)
    for ax, (model, case) in zip(axes.ravel(), cells):
        d = cues[(cues.model == model) & (cues.case == case)]
        d = d[~d.cue.str.contains(" ")]
        size = 4 + 30 * np.sqrt((d.n_pairwise + d.n_f9) / (d.n_pairwise + d.n_f9).max())
        ax.grid(False)
        ax.scatter(d.net_pairwise, d.net_f9, s=size, color=_colour(model), alpha=0.35, lw=0)
        ax.plot([-1, 1], [-1, 1], color="0.6", lw=0.6)
        ax.axhline(0, color="0.85", lw=0.5)
        ax.axvline(0, color="0.85", lw=0.5)
        lab = pd.concat([d[d.sign_flip].nlargest(n_label, "js"), d.nlargest(n_label, "js")]).drop_duplicates("cue")
        for r in lab.itertuples():
            ax.annotate(r.cue, (r.net_pairwise, r.net_f9), fontsize=6,
                        color=S.PAL["coral"] if r.sign_flip else "0.2",
                        xytext=(2, 2), textcoords="offset points")
        rho = spearmanr(d.net_pairwise, d.net_f9).statistic
        ax.set_title(f"{MODEL_LABEL[model]}, {CASE_LABEL[case].lower()} "
                     f"({len(d)} words, rho {rho:.2f})", fontsize=8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("net valence, pairwise traces")
    for ax in axes[:, 0]:
        ax.set_ylabel("net valence, F9 traces")
    fig.tight_layout()
    _meta(fig, "f11_cues",
          "The net valence, (good - bad) / texts, of each single-word cue with 25 texts or more in both "
          "paradigms. Labels: the words with the largest JS; coral marks a change of sign "
          "(|net| > 0.1 on both sides).")
    return fig


def plot_places(places: Dict[str, pd.DataFrame]):
    """The battery score of each restaurant against the F9 net valence, 1 panel for each model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(places), figsize=(7.0, 3.0), sharey=True, squeeze=False)
    for ax, (model, t) in zip(axes[0], places.items()):
        d = t[["battery_score", "net_f9", "f9_score"]].dropna()
        ax.hexbin(d.battery_score, d.net_f9, gridsize=30, mincnt=1, cmap="Greys", lw=0)
        r1 = place_rho(t, "battery_score", "net_f9", n_boot=200)
        r2 = place_rho(t, "battery_score", "f9_score", n_boot=200)
        ax.set_title(f"{MODEL_LABEL[model]}: rho {r1['rho']:.2f} (net valence), "
                     f"{r2['rho']:.2f} (verdict); n {r1['n']:,}", fontsize=8)
        ax.set_xlabel("pairwise battery score (mean ordinal label)")
    axes[0, 0].set_ylabel("F9 net valence of the spans")
    fig.tight_layout()
    _meta(fig, "f11_places",
          "1 hexagon bin holds restaurants. x: the mean score in the 1,000,000-pair battery (no thinking). "
          "y: (good - bad) / spans in the F9 reasoning of the same restaurant. The 2 paradigms show "
          "different views of the restaurant.")
    return fig


def save(fig, name: str, out: Optional[Path] = None) -> List[Path]:
    """Write 1 figure as PDF and PNG, and its caption data as JSON (as `_f8.save`)."""
    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=300)
    meta = getattr(fig, "_plot_meta", {"plot-title": name, "plot-caption": "", "plot-tags": []})
    meta = {**meta, "plot-latex-label": "fig:" + name.replace("_", "-")}
    paths.append(out / f"{name}.json")
    paths[-1].write_text(json.dumps(meta, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "f11"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables, cues, desc = [], [], []
    for model in MODELS:
        for case in CASES:
            arms = {k: add_cues(v) for k, v in load_arms(model, case).items()}
            d = describe(arms).assign(model=model, case=case)
            desc.append(d)
            print(d.to_string(index=False), flush=True)
            t = run_cell(model, case, n_boot=args.n_boot, arms=arms)
            tables.append(t)
            print(t[["unit", "comparison", "raked", "js", "js_lo", "js_hi", "keys"]
                    + (["rho"] if "rho" in t else [])].to_string(index=False), flush=True)
            cues.append(cue_table(arms, support("U3", arms["pairwise"], arms["f9"]))
                        .assign(model=model, case=case))
    pd.concat(tables).to_parquet(out / "f11_divergence.parquet", index=False)
    pd.concat(cues).to_parquet(out / "f11_cues.parquet", index=False)
    pd.concat(desc).to_parquet(out / "f11_arms.parquet", index=False)
    for model, short in MODELS.items():
        place_table(model).to_parquet(out / f"f11_places_{short}.parquet", index=False)
    print(f"wrote {out}")
