"""Word rates across the 7 cases: 1 vocabulary, or 7 vocabularies?

`_traces` draws 1 word cloud for each case, and its `distinctive` score answers
"which words does THIS case use and the others do not". This module asks the
opposite question. Which words does the model write in EVERY case, and does it
give each one the same rank each time?

That is a test of an aesthetic hierarchy. If 1 hierarchy serves all 7 prompts,
the model writes `clean`, `maintained`, and `modern` at about the same rate for
a school, a park, and a subway entrance, and it ranks them in the same order.
If each prompt has its own hierarchy, the rates and the ranks separate.

The picture
-----------
1 point is 1 word. The axes are:

| Axis | Meaning |
|------|---------|
| x | the mean rate of the word, per 10,000 trace words |
| y | the evenness of that rate over the cases, from 0 to 1 |

A word at the top right is common and even: the model writes it about as often
for a library as for a road. A word at the bottom right is common and uneven,
thus it belongs to 1 case. `library`, `menu`, and `platform` sit there.

Evenness
--------
Evenness is the Shannon entropy of the rate shares, divided by the entropy of a
flat share. A value of 1.0 means the same rate in every case. A value near 0
means 1 case holds the whole rate.

Evenness uses the RATE, not the count. A count share would reward the case with
the longest traces, and the trace lengths differ by more than 2 times.

The word set
------------
The scatter shows the union of each case's top words, by rate. Thus every point
is a word that at least 1 case writes often, and the picture stays readable.
A word that no case ranks high cannot enter, because a hierarchy is about the
top of the list.

Words, or phrases
-----------------
`run_counts(ngram=1)` counts single words, through `_traces.counts_for_run`,
thus a unigram figure matches the per-case clouds exactly. `ngram=2` and
`ngram=3` count PHRASES, through `count_phrases` in this module.

A phrase says what a word cannot. `well lit`, `less safe`, `not maintained`,
and `well maintained` are the judgment, and a word count sees none of them:
`well`, `less`, and `not` are stopwords, and `maintained` alone cannot say
which way the model leans.

Warning: `count_phrases` is NOT `_traces.count_words(ngram=2)`
--------------------------------------------------------------
`_traces.count_words` removes the stopwords BEFORE it joins the tokens. Thus:

| Text | `_traces.count_words` | `count_phrases` |
|------|----------------------|-----------------|
| `the building is clean` | `building clean` | nothing |
| `well maintained` | nothing | `well maintained` |

The first row is a phrase that nobody wrote, and the second is a phrase that
the model wrote 1,000s of times. `count_phrases` joins ADJACENT tokens, inside
1 sentence, and then it applies `phrase_ok`.

Warning: a case noun is not a defect
------------------------------------
`BOILERPLATE` in `_traces` drops the prompt echo, but it keeps `library`,
`school`, and `platform`. Those words carry the contrast: they show what an
uneven word looks like, so a reader can see that the even band is not an
artifact of the method.

Warning: the counts come from a truncated cache
-----------------------------------------------
`_traces.counts_for_run` keeps the 20,000 commonest words of a run, and the
phrase cache keeps 60,000. A word run holds under 20,000 distinct words, thus
no word is lost. A phrase run holds far more, and the tail IS cut.

The rate divides by the whole kept mass, which `run_counts` returns beside the
counts, never by the sum of the truncated counter. Without that rule the top
30,000 trigrams — under half of the mass — would move every rate.

See `vlm-narratives-docs/aesthetic-hierarchy.md`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _style as S
import _traces as T

__version__ = "1.0.0"

# A rate is words for each 10,000 words of trace.
RATE_PER = 10_000

# The floor that keeps a log finite when a case never writes the word.
RATE_FLOOR = 0.01

# The 2 label sets of the commitment score. A committed trace names a winner.
# An undecided trace calls the 2 photographs the same, or it abstains.
COMMITTED = ("MuchLess", "Less", "More", "MuchMore")
UNDECIDED = ("Same", "NotSure")

# The reasoning scaffold: the words that carry the model's VOICE, not its eye.
#
# `_traces.BOILERPLATE` drops the prompt echo alone, and that is right for a
# per-case cloud: the `distinctive` score divides out anything the 7 cases
# share. This figure asks the opposite question, so the shared scaffold does
# not divide out. It fills the even band instead: `looks`, `like`, `maybe`,
# `actually`, and `prompt` all sit above 0.95 evenness, and they push the
# aesthetic words off the top of the picture.
#
# The city name is NOT here. `CITY_NAME` and `CITY_ECHO` below hold it, because
# it needs a different rule at 1 word and at 2.
#
# The set is a switch, not a rule. Turn it off to see the scaffold band.
SCAFFOLD = frozenset("""
look looks looking looked seem seems seemed appear appears appeared
show shows showing shown see sees seen like likes

maybe perhaps probably possibly likely unlikely definitely certainly
clearly obviously actually really quite rather fairly slightly
relatively generally usually typically typical somewhat bit lot just
still

prompt asks ask asking analysis analyze analyse evaluate evaluating
interpret interpretation assume assumes assuming implies imply implied
wants want premise judge judgment judging subjective criteria criterion
terms specific based read reads check checks try tries consider
considering decide decision note notes rate rating score scores

say says said mean means meaning thing things use used using different
difference differences similar similarity closely closer possible
impossible hard easy sense guess kind sort don doesn isn aren didn wasn
won wouldn couldn shouldn

""".split())

# The place names of the prompt. The prompt names New York in every case, thus
# every trace repeats it, and `barred_words` keeps them out of every phrase.
# Without that rule `new york` and `york city` take the top of the shared band,
# where they say only that the prompt names a city.
# Warning: `island` is NOT here. `staten` alone kills `staten island`, and
# `island` on its own is a road feature: a traffic island.
CITY_NAME = frozenset("""
york nyc manhattan brooklyn queens bronx staten
""".split())

# `new` is the 1 word of the city name that also does real work. At 1 word for
# each token nothing separates `New York` from `new building`, thus `new` joins
# the stopwords there. A PHRASE separates them: `new york` dies on `york`, and
# `new building` lives.
CITY_ECHO = CITY_NAME | {"new"}


def stopwords(extra: Sequence[str] = (), drop_scaffold: bool = True,
              ngram: int = 1) -> frozenset:
    """The stopword set this figure needs.

    It is `_traces.default_stopwords` plus `SCAFFOLD`. Pass
    `drop_scaffold=False` to keep the reasoning voice in the picture.

    `CITY_ECHO` joins the set at `ngram=1` and leaves it at `ngram=2` or more.
    1 word cannot separate `New York` from `new building`, thus `new` is a
    stopword there. A phrase separates them: `barred_words` drops `york` from
    every phrase, so `new york` dies and `new building` lives.
    """
    words = list(extra)
    if drop_scaffold:
        words += sorted(SCAFFOLD)
        if int(ngram) <= 1:
            words += sorted(CITY_ECHO)
    return T.default_stopwords(words)


def barred_words(drop_scaffold: bool = True,
                 extra: Sequence[str] = ()) -> frozenset:
    """The tokens a phrase may not hold ANYWHERE, not even in the middle.

    A plain stopword glues a phrase: `cars are parked` and `near the entrance`
    are phrases, and `cars parked` and `near entrance` are not. Thus
    `phrase_ok` bars a plain stopword at the edges alone.

    The prompt echo and the reasoning voice are different. `looks`, `image`,
    and `prompt` say nothing wherever they sit, so a phrase that holds one does
    not describe the photograph. Without this rule a trigram set fills with
    `more seems appropriate` and `roadway looks better`.

    `CITY_NAME` joins the set for the same reason. `new york` and `york city`
    are the 2 most EVEN phrases of the corpus, because the prompt names the
    city in every case, and they would take the top of the shared band.

    A `MODIFIER` is exempt. `more` and `less` are label words, and they are
    also the words that carry `more modern` and `less safe`.
    """
    words = set(T.BOILERPLATE) | set(CITY_NAME)
    if drop_scaffold:
        words |= set(SCAFFOLD)
    words |= {w.strip().lower() for w in extra if w and w.strip()}
    return frozenset(words - MODIFIER)


# The degree and the polarity words. Every stopword list drops them, and for 1
# word that is right: `very` alone says nothing, and `less` is a label word.
# Inside a PHRASE they carry the meaning. `well lit`, `less safe`, and `not
# maintained` are the judgment, and each one dies without this set.
MODIFIER = frozenset("""
very well more most less least much little too not never
quite rather fairly somewhat slightly extremely really pretty
""".split())

# A phrase stops at a sentence. Without this the counter joins the last word of
# 1 sentence to the first word of the next, and it writes phrases that nobody
# said: `entrance the entrance`, `safer image less`, `safety two subway`.
_SENTENCE_RE = re.compile(r"[.!?;:\n\r]+")


def _sentence_tokens(text: str) -> List[List[str]]:
    """Split 1 trace into sentences, and each sentence into raw tokens.

    The tokens keep the stopwords. A phrase counter needs the original order,
    thus it cannot start from `_traces.tokenize`, which removes them first.
    """
    clean = T._MARKUP_RE.sub(" ", str(text).lower())
    return [T._TOKEN_RE.findall(part) for part in _SENTENCE_RE.split(clean)]


def phrase_ok(tokens: Sequence[str], stops: frozenset,
              modifiers: frozenset = MODIFIER,
              barred: frozenset = frozenset()) -> bool:
    """Test 1 candidate phrase. 3 rules, and each one drops a different fault.

    | Rule | It drops |
    |------|----------|
    | No token is barred | `roadway looks better`, `safer image less` |
    | The first and the last token carry meaning | `look the`, `the entrance` |
    | 1 token or more is a real content word | `very well`, `much more` |

    A modifier passes the second rule and fails the third, thus `less safe`
    stays and `much more` goes. A plain stopword may still sit INSIDE a phrase
    of 3 words, because `near the entrance` is a phrase and `near entrance` is
    not. `barred_words` says which tokens the middle also refuses.
    """
    for t in tokens:
        if t in barred and t not in modifiers:
            return False
    for edge in (tokens[0], tokens[-1]):
        if edge in stops and edge not in modifiers:
            return False
    return any(t not in stops and t not in modifiers for t in tokens)


def count_phrases(
    texts: Sequence[str],
    stops: frozenset,
    n: int = 2,
    modifiers: frozenset = MODIFIER,
    barred: frozenset = frozenset(),
) -> Counter:
    """Count the phrases of a set of traces.

    Warning: this is NOT `_traces.count_words(ngram=2)`. That function removes
    the stopwords BEFORE it joins the tokens, thus it reports a phrase that
    nobody wrote — `the building is clean` gives `building clean` — and it can
    never report `well maintained`, because `well` is a stopword.

    This function joins ADJACENT tokens of the raw text, inside 1 sentence, and
    then it applies `phrase_ok`.
    """
    c: Counter = Counter()
    if n <= 1:
        for text in texts:
            for toks in _sentence_tokens(text):
                c.update(t for t in toks
                         if phrase_ok((t,), stops, modifiers, barred))
        return c
    for text in texts:
        for toks in _sentence_tokens(text):
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if phrase_ok(gram, stops, modifiers, barred):
                    c[" ".join(gram)] += 1
    return c


# How many phrases the cache keeps. A bigram corpus holds about 360,000
# distinct phrases against 20,000 words, thus the word cap would cut the head.
PHRASE_CACHE_TOP = 60_000


def phrase_counts_for_run(
    run: T.TraceRun,
    stops: frozenset,
    n: int = 2,
    modifiers: frozenset = MODIFIER,
    barred: frozenset = frozenset(),
    labels: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> Tuple[Counter, int]:
    """Count the phrases of 1 run, and keep the result on disk.

    Returns:
        (counts, total). `total` is the WHOLE kept mass, before the cache cuts
        the tail. A rate must divide by that number, not by the sum of the
        truncated counter: the top 30,000 trigrams hold under half of the mass,
        and a truncated sum would then move every rate.
    """
    mtime = os.path.getmtime(run.results_path) if run.is_readable else 0
    key = T._cache_key("phrase", run.results_path, mtime, n,
                       sorted(labels or []), T._cache_key(*sorted(stops)),
                       T._cache_key(*sorted(modifiers)),
                       T._cache_key(*sorted(barred)), __version__)
    # The `phrase_` prefix is load-bearing: `.gitignore` drops these files.
    # A phrase cache is about 900 kB, against 30 kB for a word cache, and the
    # repository tracks the word caches.
    cache_file = T.CACHE_DIR / f"phrase_{key}.json"
    if use_cache and cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text())
            return Counter(blob["counts"]), int(blob["total"])
        except Exception:
            pass

    df = T.load_traces(run)
    if labels:
        df = df[df[T.LABEL_COL].isin(list(labels))]
    c = count_phrases(df[T.REASONING_COL].tolist(), stops, n=n,
                      modifiers=modifiers, barred=barred)
    total = int(sum(c.values()))

    if use_cache:
        T.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            cache_file.write_text(json.dumps({
                "total": total,
                "counts": dict(c.most_common(PHRASE_CACHE_TOP)),
            }))
        except Exception:
            pass
    return c, total


CASE_ORDER = T.BATTERY_CASES

CASE_LABEL = {
    "subway_safety": "Subway safety",
    "libraries": "Libraries",
    "schools": "Schools",
    "road_quality": "Road quality",
    "parks_plazas": "Parks and plazas",
    "restaurants": "Restaurants",
    "street_photography": "Street photography",
}

# 1 swatch for each case, from the house palette. The 7 hues separate, and each
# one is dark enough to read as a marker on the white ground.
CASE_COLOUR = {
    "subway_safety": S.PAL["blue"],
    "libraries": S.PAL["teal"],
    "schools": S.PAL["amber"],
    "road_quality": S.PAL["warmgrey"],
    "parks_plazas": S.PAL["green"],
    "restaurants": S.PAL["coral"],
    "street_photography": S.PAL["tan"],
}


def case_label(case: str) -> str:
    return CASE_LABEL.get(case, case)


def short_model(model: str) -> str:
    """`gemma-4-12b/instruct_thinking` -> `gemma-4-12b`."""
    return (model or "").split("/")[0]


# --------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------

def run_counts(
    runs: Sequence[T.TraceRun],
    stopwords: frozenset,
    ngram: int = 1,
    labels: Optional[Sequence[str]] = None,
    use_cache: bool = True,
    modifiers: frozenset = MODIFIER,
    barred: Optional[frozenset] = None,
) -> Tuple[Dict[Tuple[str, str], Counter], Dict[Tuple[str, str], int]]:
    """Count the words, or the phrases, of each run.

    The key is the case and the model, so a later step can pool the 2 raters or
    hold them apart. `_traces.counts_by_group` cannot do that: it pools on the
    case alone.

    Args:
        ngram: 1 counts single words, through `_traces.counts_for_run`, thus a
            unigram figure matches the per-case clouds exactly. 2 and 3 count
            PHRASES, through `phrase_counts_for_run`, which reads adjacent
            tokens inside 1 sentence.
        barred: The tokens a phrase refuses anywhere. `None` uses
            `barred_words()`. It does nothing at `ngram=1`.

    Returns:
        (counts, totals). `totals` holds the whole kept mass of each key, and
        `rates_frame` divides by it.
    """
    block = barred_words() if barred is None else barred
    out: Dict[Tuple[str, str], Counter] = {}
    totals: Dict[Tuple[str, str], int] = {}
    for run in runs:
        key = (run.case, short_model(run.model))
        if ngram <= 1:
            c = T.counts_for_run(run, stopwords, ngram=1, labels=labels,
                                 use_cache=use_cache)
            total = int(sum(c.values()))
        else:
            c, total = phrase_counts_for_run(
                run, stopwords, n=ngram, modifiers=modifiers, barred=block,
                labels=labels, use_cache=use_cache)
        if not c:
            continue
        out.setdefault(key, Counter()).update(c)
        totals[key] = totals.get(key, 0) + total
    return out, totals


def rates_frame(
    counts: Dict[Tuple[str, str], Counter],
    totals: Optional[Dict[Tuple[str, str], int]] = None,
) -> pd.DataFrame:
    """Turn the counters into 1 tidy table: case, model, word, count, rate.

    Args:
        totals: The whole kept mass of each key. Pass the second value of
            `run_counts`. Without it the rate divides by the sum of the counter
            it was given, and a truncated phrase counter then moves every rate.
    """
    frames = []
    for (case, model), c in counts.items():
        total = ((totals or {}).get((case, model)) or sum(c.values())) or 1
        words = list(c)
        frames.append(pd.DataFrame({
            "case": case,
            "model": model,
            "word": words,
            "count": [c[w] for w in words],
            "corpus_words": total,
            "rate": [c[w] * RATE_PER / total for w in words],
        }))
    if not frames:
        return pd.DataFrame(columns=["case", "model", "word", "count",
                                     "corpus_words", "rate"])
    return pd.concat(frames, ignore_index=True)


def corpus_sizes(rates: pd.DataFrame) -> pd.DataFrame:
    """How many words each case and model holds. Read this before a figure."""
    size = (rates.groupby(["case", "model"], as_index=False)
                 .agg(corpus_words=("corpus_words", "max"),
                      distinct=("word", "size")))
    size["case_label"] = size["case"].map(CASE_LABEL).fillna(size["case"])
    return size.sort_values(["case", "model"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# The word by case matrix
# --------------------------------------------------------------------------

def rate_matrix(
    rates: pd.DataFrame,
    models: Optional[Sequence[str]] = None,
    top_per_case: int = 60,
    min_count: int = 500,
    group_by: str = "case",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Build the word-by-group matrix of rates.

    Args:
        models: Keep these raters only. `None` keeps every rater, and a group
            then pools them.
        top_per_case: How many words each group contributes to the word set.
        min_count: Drop a word that the whole corpus writes fewer times. A
            phrase needs a lower floor than a word: the corpus holds about 20
            times more distinct bigrams, thus each one is rarer.
            `min_count_for(ngram)` gives the scaled default.
        group_by: `case` gives 7 columns. `case_model` gives 14, which asks
            whether 1 rater carries the result.

    Returns:
        (matrix, tops). `matrix` is words by groups, in rate per 10,000 words.
        `tops` names the top words of each group, which `breadth` counts.

    Warning: a group pools the 2 raters by count, not by rate. A rate mean of
    2 raters would give the shorter corpus the same weight as the longer one.
    """
    df = rates if models is None else rates[rates["model"].isin(list(models))]
    if df.empty:
        return pd.DataFrame(), {}

    if group_by == "case_model":
        df = df.assign(group=df["case"] + " / " + df["model"])
    else:
        df = df.assign(group=df["case"])

    # Pool by count, then divide by the pooled corpus size.
    totals = (df.drop_duplicates(["group", "case", "model"])
                .groupby("group")["corpus_words"].sum())
    pooled = (df.groupby(["group", "word"], as_index=False)["count"].sum())
    pooled["rate"] = pooled["count"] * RATE_PER / pooled["group"].map(totals)

    keep = pooled.groupby("word")["count"].sum()
    keep = set(keep[keep >= min_count].index)

    tops: Dict[str, List[str]] = {}
    universe: set = set()
    for group, part in pooled.groupby("group"):
        part = part[part["word"].isin(keep)]
        top = part.nlargest(top_per_case, "rate")["word"].tolist()
        tops[str(group)] = top
        universe |= set(top)

    wide = (pooled[pooled["word"].isin(universe)]
            .pivot(index="word", columns="group", values="rate")
            .fillna(0.0))
    wide.columns.name = None
    return wide[_ordered_columns(wide.columns)], tops


# The count floor of each token length. A bigram corpus is about 20 times
# sparser than a word corpus, and a trigram corpus is sparser again, so 1 fixed
# floor either buries the phrases or floods the words.
MIN_COUNT = {1: 500, 2: 200, 3: 60}


def min_count_for(ngram: int) -> int:
    """The default count floor for a token of `ngram` words."""
    return MIN_COUNT.get(int(ngram), 50)


def _ordered_columns(columns: Sequence[str]) -> List[str]:
    """The paper's case order first, then anything else, in name order."""
    cols = list(columns)
    ordered: List[str] = []
    for case in CASE_ORDER:
        ordered += sorted(c for c in cols if c == case or c.startswith(f"{case} /"))
    return ordered + sorted(set(cols) - set(ordered))


def _case_of_column(column: str) -> str:
    return column.split(" / ")[0]


# --------------------------------------------------------------------------
# The summary that the scatter draws
# --------------------------------------------------------------------------

def evenness(row: np.ndarray) -> float:
    """The entropy of the rate shares, divided by the flat entropy.

    1.0 says every case writes the word at the same rate. A value near 0 says
    1 case holds it alone.
    """
    total = float(row.sum())
    if total <= 0 or len(row) < 2:
        return float("nan")
    p = row / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(row)))


def summarise(matrix: pd.DataFrame,
              tops: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Give each word its height in the hierarchy and its evenness.

    | Column | Meaning |
    |--------|---------|
    | `mean_rate` | the mean rate over the groups |
    | `floor_rate` | the rate of the group that writes it least |
    | `evenness` | the flatness of the rate over the groups, 0 to 1 |
    | `sd_log10` | the spread of the rate, in orders of magnitude |
    | `peak` | the group with the highest rate |
    | `peak_share` | the share of the rate that the peak group holds |
    | `breadth` | how many groups rank the word in their own top list |
    | `mean_rank` | the mean rank inside a group, over this word set |
    | `worst_rank` | the rank of the group that ranks it lowest |
    """
    if matrix.empty:
        return pd.DataFrame()
    ranks = matrix.rank(ascending=False, method="min")
    values = matrix.to_numpy(dtype=float)
    logs = np.log10(values + RATE_FLOOR)
    peak_idx = values.argmax(axis=1)
    row_sum = values.sum(axis=1)

    out = pd.DataFrame({
        "word": matrix.index,
        "mean_rate": values.mean(axis=1),
        "floor_rate": values.min(axis=1),
        "evenness": [evenness(r) for r in values],
        "sd_log10": logs.std(axis=1, ddof=0),
        "peak": [matrix.columns[i] for i in peak_idx],
        "peak_share": np.divide(values.max(axis=1), row_sum,
                                out=np.zeros_like(row_sum),
                                where=row_sum > 0),
        "mean_rank": ranks.mean(axis=1).to_numpy(),
        "worst_rank": ranks.max(axis=1).to_numpy(),
    })
    out["peak_case"] = out["peak"].map(_case_of_column)
    if tops:
        member = {w: 0 for w in matrix.index}
        for words in tops.values():
            for w in words:
                if w in member:
                    member[w] += 1
        out["breadth"] = out["word"].map(member)
    else:
        out["breadth"] = len(matrix.columns)
    out["n_groups"] = len(matrix.columns)
    return out.sort_values("mean_rate", ascending=False).reset_index(drop=True)


def commitment_lift(
    runs: Sequence[T.TraceRun],
    stopwords: frozenset,
    words: Sequence[str],
    models: Optional[Sequence[str]] = None,
    ngram: int = 1,
    barred: Optional[frozenset] = None,
) -> pd.Series:
    """Score each word by the traces that named a winner.

    The score is the log-odds ratio with an informative Dirichlet prior, the
    same statistic as the trace clouds. The target is the traces that picked a
    side. The background is the traces that called the 2 photographs the same,
    or that abstained.

    A positive score says the model writes the word when it commits. A negative
    score says it writes the word when it cannot separate the 2 photographs.

    The score is computed INSIDE each case, and then averaged over the cases.
    A pooled score would follow the case mix instead: the street photography
    runs abstain on 1% of the pairs, and the schools runs on 56%.
    """
    keep = set(words)
    picked = [r for r in runs if models is None or short_model(r.model) in set(models)]
    by_case: Dict[str, List[T.TraceRun]] = {}
    for r in picked:
        by_case.setdefault(r.case, []).append(r)

    block = barred_words() if barred is None else barred

    def _count(run: T.TraceRun, labels: Sequence[str]) -> Counter:
        if ngram <= 1:
            return T.counts_for_run(run, stopwords, ngram=1, labels=labels)
        return phrase_counts_for_run(run, stopwords, n=ngram, barred=block,
                                     labels=labels)[0]

    frames = []
    for case, case_runs in by_case.items():
        yes, no = Counter(), Counter()
        for r in case_runs:
            yes.update(_count(r, COMMITTED))
            no.update(_count(r, UNDECIDED))
        if not yes or not no:
            continue
        scores = T.distinctive_scores(yes, no, min_count=1)
        frames.append(pd.Series({w: scores.get(w, 0.0) for w in keep}, name=case))
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames, axis=1).mean(axis=1)


# --------------------------------------------------------------------------
# Agreement between the cases
# --------------------------------------------------------------------------

def shared_words(summary: pd.DataFrame,
                 min_breadth: Optional[int] = None) -> List[str]:
    """The words that at least `min_breadth` groups rank in their own top list.

    The default is every group. Read the agreement over these words to ask
    whether 1 hierarchy exists. Read it over the whole word set to ask whether
    the vocabularies match, case nouns and all.
    """
    if summary.empty:
        return []
    floor = int(summary["n_groups"].iloc[0]) if min_breadth is None else min_breadth
    return summary.loc[summary["breadth"] >= floor, "word"].tolist()


def agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho between every pair of groups, over the shared word set.

    A high rho says the 2 cases put the words in the same order, which is what
    "1 hierarchy" means. The rank is over this word set only, thus rho answers
    a question about the TOP of the vocabulary.
    """
    if matrix.shape[1] < 2:
        return pd.DataFrame()
    from scipy.stats import spearmanr

    cols = list(matrix.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho, p = spearmanr(matrix[a], matrix[b])
            rows.append({"a": a, "b": b, "rho": float(rho), "p": float(p)})
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def agreement_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """The same rho values, as a square table."""
    pairs = agreement(matrix)
    cols = list(matrix.columns)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for _, r in pairs.iterrows():
        out.loc[r["a"], r["b"]] = r["rho"]
        out.loc[r["b"], r["a"]] = r["rho"]
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _to_axes(ax, x: float, y: float) -> Tuple[float, float]:
    """Data coordinates to axes fraction. Set the limits before you call it."""
    fx, fy = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))
    return float(fx), float(fy)


def _place_labels(ax, points, texts, colours, n_cols: int = 30,
                  n_rows: int = 20, fontsize: float = 6.0,
                  chars_per_cell: float = 4.6) -> None:
    """Write a label beside each point, and keep 2 labels apart.

    The axes become a grid of label-size cells. Each point takes the free cell
    nearest to it, and a hair line joins the point to its label. The caller's
    order is the priority order, thus the first label gets the best cell.

    A word cloud cannot serve this figure, because a cloud drops the position.
    Here the position IS the result, so the words must sit beside their points.
    """
    taken: set = set()
    for (px, py), text, colour in zip(points, texts, colours):
        fx, fy = _to_axes(ax, px, py)
        span = max(1, int(math.ceil(len(text) / chars_per_cell)))
        home = (int(fx * n_cols), int(fy * n_rows))
        best = None
        for radius in range(0, max(n_cols, n_rows)):
            ring = [(home[0] + dx, home[1] + dy)
                    for dx in range(-radius, radius + 1)
                    for dy in range(-radius, radius + 1)
                    if max(abs(dx), abs(dy)) == radius]
            ring.sort(key=lambda c: (c[0] - fx * n_cols) ** 2
                      + ((c[1] - fy * n_rows) * 1.6) ** 2)
            for cx, cy in ring:
                cells = {(cx + i, cy) for i in range(span)}
                if cx < 0 or cy < 0 or cx + span > n_cols or cy >= n_rows:
                    continue
                if cells & taken:
                    continue
                best = (cx, cy, cells)
                break
            if best:
                break
        if not best:
            continue
        cx, cy, cells = best
        taken |= cells
        ax.annotate(
            text,
            xy=(px, py), xycoords="data",
            xytext=((cx + span / 2) / n_cols, (cy + 0.5) / n_rows),
            textcoords=ax.transAxes,
            fontsize=fontsize, color=S.EDGE, ha="center", va="center",
            zorder=5,
            # The halo keeps a word readable where it sits over a point.
            bbox=dict(boxstyle="square,pad=0.12", facecolor=S.PAPER,
                      edgecolor="none", alpha=0.75),
            arrowprops=dict(arrowstyle="-", linewidth=0.3, color=colour,
                            shrinkA=0.0, shrinkB=1.2, alpha=0.8),
        )


def label_picks(summary: pd.DataFrame, n_core: int = 22,
                n_contrast: int = 8) -> pd.DataFrame:
    """Choose the words to name in the scatter.

    2 sets, and the figure needs both:

    | Set | Rule | What it shows |
    |-----|------|---------------|
    | core | the highest `floor_rate` | the rungs every case keeps |
    | contrast | the lowest `evenness`, among the common words | the case nouns |

    The floor is the strict test. A word takes a high floor only when the case
    that writes it LEAST still writes it often.
    """
    if summary.empty:
        return summary
    core = summary.nlargest(n_core, "floor_rate").assign(pick="core")
    rest = summary[~summary["word"].isin(core["word"])]
    common = rest[rest["mean_rate"] >= rest["mean_rate"].median()]
    contrast = common.nsmallest(n_contrast, "evenness").assign(pick="contrast")
    return pd.concat([core, contrast], ignore_index=True)


def plot_hierarchy(
    summary: pd.DataFrame,
    colour_by: str = "peak_case",
    lift: Optional[pd.Series] = None,
    n_core: int = 22,
    n_contrast: int = 8,
    figsize: Tuple[float, float] = (7.0, 4.4),
):
    """Draw the scatter: rate against evenness, 1 point for each word.

    Args:
        colour_by: `peak_case` colours a point by the case that writes the word
            most. `commitment` colours it by `commitment_lift`, and it needs
            `lift`.

    The words that sit high on both axes are the aesthetic hierarchy that every
    case shares. The words low on the y axis belong to 1 prompt.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if summary.empty:
        raise ValueError("no words to draw")

    fig, ax = plt.subplots(figsize=figsize)
    # The labels name the token they draw. A phrase set holds a space, thus
    # they follow the data and no caller has to pass the word.
    unit = "phrases" if any(" " in str(w) for w in summary["word"]) else "words"
    x = summary["mean_rate"].to_numpy()
    y = summary["evenness"].to_numpy()
    edge = np.where(summary["breadth"].to_numpy() >= summary["n_groups"].to_numpy(),
                    0.9, 0.25)

    if colour_by == "commitment":
        if lift is None:
            raise ValueError("colour_by='commitment' needs the lift series")
        z = summary["word"].map(lift).fillna(0.0).to_numpy()
        span = float(np.percentile(np.abs(z), 98)) or 1.0
        sc = ax.scatter(x, y, c=z, cmap=S.CMAP_DIV, vmin=-span, vmax=span,
                        s=26, linewidths=edge, edgecolors=S.EDGE, alpha=0.95,
                        zorder=3)
        bar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
        bar.set_label("commitment lift (z)\nundecided ←   → picks a winner",
                      fontsize=6.5)
        bar.ax.tick_params(labelsize=6)
        point_colour = [S.EDGE] * len(summary)
    else:
        point_colour = [CASE_COLOUR.get(c, S.PAL["warmgrey"])
                        for c in summary["peak_case"]]
        ax.scatter(x, y, c=point_colour, s=26, linewidths=edge,
                   edgecolors=S.EDGE, alpha=0.95, zorder=3)
        seen = [c for c in CASE_ORDER if c in set(summary["peak_case"])]
        # The legend sits UNDER the axes. Inside, it covers the low-evenness
        # corner, which is where the case nouns are, and those are the
        # contrast the figure needs.
        fig.legend(
            handles=[Line2D([], [], marker="o", linestyle="none",
                            markersize=4.0, markeredgecolor=S.EDGE,
                            markeredgewidth=0.3,
                            markerfacecolor=CASE_COLOUR[c],
                            label=case_label(c)) for c in seen],
            title=f"the case that writes the {unit[:-1]} most", title_fontsize=6.5,
            loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
            frameon=False,
        )

    ax.set_xscale("log")
    ax.set_xlabel(f"mean rate over the cases ({unit} per {RATE_PER:,})")
    ax.set_ylabel(f"evenness over the {int(summary['n_groups'].iloc[0])} cases"
                  " (1 = the same rate in each)")
    ax.set_ylim(min(0.0, float(np.nanmin(y)) - 0.03), 1.04)
    # Room on the right, so the label of the commonest word is not cut.
    ax.set_xlim(float(x.min()) * 0.75, float(x.max()) * 2.2)

    picks = label_picks(summary, n_core=n_core, n_contrast=n_contrast)
    if colour_by == "commitment":
        leader = {w: S.PAL["warmgrey"] for w in summary["word"]}
    else:
        leader = dict(zip(summary["word"], point_colour))
    _place_labels(
        ax,
        list(zip(picks["mean_rate"], picks["evenness"])),
        list(picks["word"]),
        [leader.get(w, S.EDGE) for w in picks["word"]],
    )
    if colour_by != "commitment":
        fig.tight_layout(rect=(0, 0.10, 1, 1))
    else:
        fig.tight_layout()
    return fig


def plot_case_pairs(matrix: pd.DataFrame, words: Optional[Sequence[str]] = None,
                    n_cols: int = 4,
                    figsize: Tuple[float, float] = (7.0, 4.0)):
    """1 panel for each case: its rate against the rate of the other cases.

    A tight diagonal says the case keeps the shared order. A cloud says the
    case has an order of its own. Each panel carries its Spearman rho.

    Args:
        words: Draw these words only. Pass `shared_words(summary)` to ask
            whether the SHARED vocabulary keeps 1 order. The default draws the
            whole word set, where each case also carries its own nouns, and
            those nouns sit near the floor of every other case.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    if matrix.shape[1] < 2:
        raise ValueError("2 groups or more are needed")
    if words is not None:
        matrix = matrix.loc[[w for w in words if w in matrix.index]]
    if matrix.empty:
        raise ValueError("no words to draw")
    cols = list(matrix.columns)
    n_rows = int(math.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        others = matrix[[c for c in cols if c != col]].mean(axis=1)
        rho, _ = spearmanr(matrix[col], others)
        colour = CASE_COLOUR.get(_case_of_column(col), S.PAL["warmgrey"])
        lo = min(others.min(), matrix[col].min()) + RATE_FLOOR
        hi = max(others.max(), matrix[col].max())
        ax.plot([lo, hi], [lo, hi], linewidth=0.6, color=S.EDGE, alpha=0.5,
                zorder=1)
        ax.scatter(others + RATE_FLOOR, matrix[col] + RATE_FLOOR, s=9,
                   color=colour, edgecolors=S.EDGE, linewidths=0.2,
                   alpha=0.85, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _halo = dict(facecolor=S.PAPER, edgecolor="none", alpha=0.8, pad=1.2)
        ax.text(0.04, 0.96, case_label(_case_of_column(col)),
                transform=ax.transAxes, fontsize=7, va="top", color=S.EDGE,
                bbox=_halo, zorder=4)
        ax.text(0.96, 0.05, f"rho = {rho:.2f}", transform=ax.transAxes,
                fontsize=6.5, va="bottom", ha="right", color=S.EDGE,
                bbox=_halo, zorder=4)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.supxlabel(f"rate in the other cases (per {RATE_PER:,})", fontsize=8)
    fig.supylabel(f"rate in this case (per {RATE_PER:,})", fontsize=8)
    fig.tight_layout()
    return fig


def plot_agreement(matrix: pd.DataFrame,
                   figsize: Tuple[float, float] = (4.4, 3.8)):
    """The Spearman rho of every case pair, as a square block."""
    import matplotlib.pyplot as plt

    square = agreement_matrix(matrix)
    if square.empty:
        raise ValueError("2 groups or more are needed")
    labels = [case_label(_case_of_column(c)) for c in square.columns]
    values = square.to_numpy(dtype=float)
    off = values[~np.eye(len(values), dtype=bool)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap=S.CMAP_SEQ, vmin=max(0.0, off.min() - 0.05),
                   vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.8, color=S.text_on(
                        im.cmap(im.norm(values[i, j]))))
    bar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
    bar.set_label("Spearman rho of the word rates", fontsize=6.5)
    bar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export(
    summary: pd.DataFrame,
    matrix: pd.DataFrame,
    out_dir: Path,
    tag: str = "aesthetic_hierarchy",
    lift: Optional[pd.Series] = None,
    n_core: int = 22,
    n_contrast: int = 8,
) -> List[str]:
    """Write the scatter, the 2 support figures, and the tables.

    Args:
        out_dir: Pass `master/figures`. A cross-case figure belongs to no 1
            prompt, thus it must not land in a prompt folder.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _save(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=300)
        plt.close(fig)
        written.append(name)

    _save(plot_hierarchy(summary, colour_by="peak_case", n_core=n_core,
                         n_contrast=n_contrast), f"{tag}_scatter.png")
    if lift is not None and not lift.empty:
        _save(plot_hierarchy(summary, colour_by="commitment", lift=lift,
                             n_core=n_core, n_contrast=n_contrast),
              f"{tag}_scatter_commitment.png")
    _save(plot_case_pairs(matrix), f"{tag}_case_pairs.png")
    # A second panel over the words that HALF the cases rank. It asks the
    # narrower question: where the vocabulary IS shared, is the order shared?
    core = shared_words(summary, min_breadth=max(2, len(matrix.columns) // 2))
    if len(core) >= 20:
        _save(plot_case_pairs(matrix, words=core),
              f"{tag}_case_pairs_shared.png")
    _save(plot_agreement(matrix), f"{tag}_agreement.png")

    table = summary.copy()
    if lift is not None and not lift.empty:
        table["commitment_lift"] = table["word"].map(lift)
    for col in ("mean_rate", "floor_rate", "evenness", "sd_log10",
                "peak_share", "mean_rank", "worst_rank", "commitment_lift"):
        if col in table:
            table[col] = table[col].astype(float).round(4)
    table.to_csv(out_dir / f"{tag}_words.csv", index=False)
    written.append(f"{tag}_words.csv")

    matrix.round(4).to_csv(out_dir / f"{tag}_rate_matrix.csv")
    written.append(f"{tag}_rate_matrix.csv")

    pairs = agreement(matrix)
    if not pairs.empty:
        pairs.round(4).to_csv(out_dir / f"{tag}_agreement.csv", index=False)
        written.append(f"{tag}_agreement.csv")
    return written
