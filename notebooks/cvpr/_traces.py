"""Reasoning-trace discovery and word counts for the thinking sweeps.

The consolidated battery runs greedy and emits only the structured label, so
`model_reasoning` is empty in a canonical run. A THINKING sweep turns the
thought channel on and keeps the trace. This module finds those runs and turns
their traces into word counts, one set of counts for each prompt.

Why a second module
-------------------
`_provenance.py` serves the validation-via-proxy notebooks. It keeps only the
canonical rater set, because a thinking run uses different sampling and its
labels are not comparable. This module wants the opposite set: the runs that
`_provenance` drops on purpose. It reuses the W&B discovery chain there, and
then applies its own filter.

Warning: do not walk the `multirun/` tree. It sits on NFS and a full glob costs
minutes. Discovery goes through W&B, which names each stage directory.

Two ways to score a word
------------------------
| Mode | What it shows | When to use it |
|------|---------------|----------------|
| `frequency` | The most common words | A first look at one prompt |
| `distinctive` | Words this prompt uses more than the others | A comparison |

A raw frequency cloud looks nearly the same for every prompt, because the
traces share a scaffold: the model names the 2 images, lists cues, and then
picks a label. The `distinctive` mode divides out that shared scaffold. It uses
the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi, and
Quinn, 2008), which corrects the variance of a rare word. Thus a word that
appears 3 times does not beat a word that appears 3,000 times.

Prompt echo
-----------
A trace recites the prompt, so `image`, `Image A`, and the 6 label words show
up in every trace of every case. `BOILERPLATE` drops them. Keep this list
short: each word you add is a word the cloud can no longer show you.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import _provenance as P

__version__ = "1.1.0"

REPO_ROOT = P.REPO_ROOT
CACHE_DIR = REPO_ROOT / "notebooks" / "cvpr" / ".trace_cache"

# A results parquet holds the trace in this column. An empty column means the
# run was label-only, thus this module skips it.
REASONING_COL = "model_reasoning"
LABEL_COL = "presented_label"

# The 6 labels, in scale order. `NotSure` comes last because it is an
# abstention, not a point on the scale.
LABELS = ("MuchLess", "Less", "Same", "More", "MuchMore", "NotSure")

# The 7 cases of the consolidated battery (2026-08-11). A notebook selects
# these by default. W&B also holds older cases such as `cyclomedia_sterility`,
# which run to 95,000 rows and belong to a different question. Count them only
# when you ask for them.
BATTERY_CASES = (
    "subway_safety", "libraries", "schools", "road_quality",
    "parks_plazas", "restaurants", "street_photography",
)

# A run smaller than this is a smoke test, not data. 2 subway runs of 2026-08-13
# hold 18 rows each: they are the layout probes that found the image-binding
# bug, and 1 of them ran the BROKEN `images_then_text` layout, where 15 of 18
# traces say the model saw only one image. That text is a defect, not a
# judgment, so it must not enter a cloud.
MIN_ROWS = 1000

# The word that the prompt puts in the model's mouth. Every trace repeats these,
# in every case, so they carry no signal.
BOILERPLATE = frozenset("""
image images imagea imageb photo photos picture pictures
a b user users assistant model
compare compares compared comparison comparing relative
label labels answer answers output outputs return returns response
json schema field exactly choose choice select pick
muchless much less same more muchmore notsure sure
one two both first second left right
question task instruction instructions
let lets okay yeah hmm wait
""".split())

# A token is a run of 3 letters or more. The class holds no apostrophe on
# purpose: `it's` then breaks into `it` and `s`, and the stopword list and the
# 3-letter minimum remove the 2 parts. An apostrophe inside the class instead
# keeps `it's`, `there's`, and `let's` as words, and they fill the cloud.
_TOKEN_RE = re.compile(r"[a-z]{3,}")

# Markdown that the trace uses for structure. Remove it before you tokenize.
_MARKUP_RE = re.compile(r"[*_`#>\[\]()]+")


@dataclass
class TraceRun:
    """One run whose traces this notebook can read."""

    wandb_id: str
    wandb_url: str
    created_at: str
    state: str
    case: str
    model: str
    sweep: str
    pipeline: str
    stage_dir: str
    results_path: Optional[str]
    rows: Optional[int] = None
    traced_rows: Optional[int] = None
    question: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        return bool(self.results_path and os.path.exists(self.results_path))

    @property
    def tag(self) -> str:
        """A short name for a legend or a file name."""
        return f"{self.model.split('/')[0]}_{self.case}_{self.wandb_id[:6]}"


def case_of(pipeline: str) -> str:
    """Turn a pipeline name into a case name.

    `pairwise_subway_safety_mvp` becomes `subway_safety`.
    """
    s = re.sub(r"^pairwise_", "", pipeline or "")
    s = re.sub(r"_(mvp|ordinal|large)$", "", s)
    return s or "unknown"


def case_of_parquet(name: str) -> str:
    """Read the case out of a results file name.

    `subway_safety_mvp_20260711_115302.parquet` gives `subway_safety`.
    """
    stem = Path(name).stem
    stem = re.sub(r"\.presplit$", "", stem)
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
    return case_of(stem)


def find_results_parquet(
    stage_dir: str | Path, case: Optional[str] = None,
) -> Optional[str]:
    """Find the results parquet in a stage directory.

    Args:
        case: Keep a file whose name names this case. Give this whenever you
            know the case.

    Warning: give the `case` argument. A stage directory usually holds 1
    results parquet, but 2 sweeps that start in the same second share a
    directory, and then it holds 1 file for each case. Without the `case`
    argument this function returns the newest file, which is the WRONG case for
    1 of the 2 runs. The `think10k` sweep has this shape.

    Warning: this skips `*.presplit.parquet`. A gemma-4-e2b run needs a
    post-hoc split of its trace from its answer, and the tool keeps the original
    file beside the fixed one. The original has no usable trace column.
    """
    out_dir = Path(stage_dir) / "outputs" / "pairwise"
    if not out_dir.is_dir():
        return None
    cands = [
        p for p in out_dir.glob("*.parquet")
        if p.name != "pairs.parquet" and ".presplit." not in p.name
    ]
    if case:
        matched = [p for p in cands if case_of_parquet(p.name) == case]
        if matched:
            cands = matched
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


def scan_sweep_dir(
    sweep_dir: str | Path,
    require_trace: bool = True,
    sweep_name: Optional[str] = None,
) -> List[TraceRun]:
    """Read one named sweep directory from disk, without W&B.

    Use this for a sweep that the W&B chain cannot resolve. An older stage
    directory holds no `.hydra/overrides.yaml`, so `_provenance` cannot name
    its pipeline and drops it. The `think10k` sweep of 2026-07-11 is the
    example: 3 models, 2 cases, 10k pairs each, and no Hydra record.

    The function reads the case out of each parquet file name. It takes the
    model from `.hydra/overrides.yaml` when that file exists, and from the
    directory name when it does not. A model that comes from a directory name
    carries a `dir:` prefix, so a reader can see that it is a label, not a
    config value.

    Warning: this looks 1 level down only. It never walks the tree, because
    `multirun/` sits on NFS and a full glob costs minutes.
    """
    # A relative path resolves against the repository root, not the working
    # directory. marimo can start from any directory.
    root = Path(sweep_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not root.is_dir():
        return []
    stage_dirs = [root] + [p for p in root.iterdir() if p.is_dir()]

    out: List[TraceRun] = []
    for sd in stage_dirs:
        out_dir = sd / "outputs" / "pairwise"
        if not out_dir.is_dir():
            continue
        ov = _read_overrides(sd)
        model = ov.get("model") or f"dir:{sd.name}"
        for p in sorted(out_dir.glob("*.parquet")):
            if p.name == "pairs.parquet" or ".presplit." in p.name:
                continue
            case = case_of_parquet(p.name)
            has_trace, rows = _peek_traces(str(p))
            if require_trace and not has_trace:
                continue
            out.append(TraceRun(
                wandb_id=f"disk:{sd.name}:{case}",
                wandb_url="",
                created_at=_stamp_of(p.name),
                state="finished",
                case=case,
                model=model,
                sweep=sweep_name or root.name,
                pipeline=ov.get("pipeline") or f"pairwise_{case}_mvp",
                stage_dir=str(sd),
                results_path=str(p),
                rows=rows,
                question=read_question(sd),
                notes=["found on disk, not through W&B"],
            ))
    return out


def _read_overrides(stage_dir: Path) -> Dict[str, str]:
    f = stage_dir / ".hydra" / "overrides.yaml"
    return P._parse_overrides(f) if f.exists() else {}


def read_question(stage_dir: str | Path) -> str:
    """Read the question that a run actually asked.

    The question comes from `<stage_dir>/.hydra/config.yaml`, which holds the
    RESOLVED prompt. `overrides.yaml` does not carry it, because the prompt
    arrives through the pipeline default and no override names it.

    Why this matters: a case name is NOT a question. The schools case asked
    "Which school would you rather send your child to?" until 2026-08-13, and
    "Which looks to be the better school?" after it. Both runs carry the case
    name `schools`. Pool them and the cloud mixes 2 different questions, which
    no reader can see and no label can warn about.

    Returns an empty string when the file is absent or holds no question.
    """
    f = Path(stage_dir) / ".hydra" / "config.yaml"
    if not f.exists():
        return ""
    try:
        import yaml

        cfg = yaml.safe_load(f.read_text()) or {}
        template = (cfg.get("prompt") or {}).get("user_template", "") or ""
    except Exception:
        return ""
    for line in template.splitlines():
        s = line.strip()
        if s.endswith("?"):
            return s
    return ""


def _stamp_of(name: str) -> str:
    """Read the `YYYYMMDD_HHMMSS` stamp out of a results file name."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", name)
    if not m:
        return ""
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}"


def _peek_traces(path: str, n: int = 256) -> Tuple[bool, int]:
    """Test the head of a parquet for a trace. Return (has_trace, rows).

    This reads 1 column, so it costs far less than a full read.
    """
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        rows = pf.metadata.num_rows
        if REASONING_COL not in pf.schema_arrow.names:
            return False, rows
        head = next(pf.iter_batches(batch_size=n, columns=[REASONING_COL]))
        s = head.column(0).to_pandas().fillna("")
        return bool((s.str.len() > 0).any()), rows
    except Exception:
        return False, 0


DISCOVERY_CACHE = CACHE_DIR / "discovery.json"


def discover_trace_runs(
    only_finished: bool = True,
    limit: int = 400,
    require_trace: bool = True,
    cases: Optional[Sequence[str]] = None,
    refresh: bool = False,
    extra_sweep_dirs: Sequence[str] = (),
    min_date: Optional[str] = P.CONSOLIDATION_DATE,
    min_rows: int = MIN_ROWS,
    source: Optional[str] = None,
) -> List[TraceRun]:
    """Find every run that keeps a reasoning trace, newest first.

    Args:
        only_finished: Keep only the runs that W&B marks `finished`. Set it to
            False to see a run that is still in progress. A run that is still
            in progress has no parquet, so it shows as not readable.
        require_trace: Test the head of each parquet and keep only the runs
            that hold a trace. Set it to False to see every run.
        cases: Keep only these case names. `None` keeps all of them.
        refresh: Read W&B again. The default reads the cache on disk, if the
            cache holds the same query.
        extra_sweep_dirs: Sweep directories to read from disk as well. Give a
            sweep that has no Hydra record, thus W&B cannot resolve it. See
            `scan_sweep_dir`.
        min_date: Drop a run that starts before this date. The default is the
            consolidation date, 2026-08-11. An earlier run used a persona and a
            cue list, so its traces repeat that text and describe the old
            prompt, not the model. Pass `None` to see every run.
        min_rows: Drop a run smaller than this. See `MIN_ROWS`.

    Warning: a refresh reads the network and downloads 1 metadata file for each
    run, thus a large limit costs minutes. A sweep that lands today needs a
    refresh before this function can see it.

    Note: only the W&B part uses the cache, and its key holds only the W&B
    arguments. Thus a change to `cases` or to `extra_sweep_dirs` costs nothing.
    Put those 2 arguments in the key and every change to either one forces
    another 20-minute read of the network.
    """
    if (P.RUN_SOURCE if source is None else source) == "canonical":
        # The registry holds 1 thinking run for each case and model, thus there
        # is nothing to search and no cache to go stale.
        return _discover_canonical(cases)

    out = list(_discover_wandb(only_finished, limit, require_trace, refresh))
    if cases:
        out = [r for r in out if r.case in set(cases)]

    # Add the sweeps that W&B cannot resolve. A disk record never replaces a
    # W&B record: the W&B record carries the provenance, thus it wins.
    seen = {r.results_path for r in out if r.results_path}
    for d in extra_sweep_dirs:
        for r in scan_sweep_dir(d, require_trace=require_trace):
            if r.results_path in seen:
                continue
            if cases and r.case not in set(cases):
                continue
            seen.add(r.results_path)
            out.append(r)

    if min_date:
        out = [r for r in out if (r.created_at or "") >= min_date]
    if min_rows:
        out = [r for r in out if (r.rows or 0) >= min_rows]
    out.sort(key=lambda r: r.created_at or "", reverse=True)
    return out


def counts_by_group(
    runs: Sequence[TraceRun],
    stopwords: frozenset,
    ngram: int = 1,
    labels: Optional[Sequence[str]] = None,
    split_questions: bool = True,
    use_cache: bool = True,
) -> Tuple[Dict[str, Counter], Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
    """Pool the runs into groups and count the words of each group.

    A group is a case, unless that case asked more than 1 question and
    `split_questions` is on. Then the group is the case and the question
    together, because a cloud that mixes 2 questions shows neither.

    Returns:
        (counts, pairs, run_count, mixed), each keyed by group name. `mixed`
        maps a case to its questions, and it is empty when every case is clean.
    """
    mixed = mixed_question_cases(runs)

    def group_of(run: TraceRun) -> str:
        if split_questions and run.case in mixed and run.question:
            return f"{run.case} | {run.question}"
        return run.case

    counts: Dict[str, Counter] = {}
    pairs: Dict[str, int] = {}
    run_count: Dict[str, int] = {}
    for run in runs:
        c = counts_for_run(run, stopwords, ngram=ngram, labels=labels,
                           use_cache=use_cache)
        if not c:
            continue
        key = group_of(run)
        counts.setdefault(key, Counter()).update(c)
        pairs[key] = pairs.get(key, 0) + (run.rows or 0)
        run_count[key] = run_count.get(key, 0) + 1
    return counts, pairs, run_count, mixed


def _discover_canonical(cases: Optional[Sequence[str]] = None) -> List[TraceRun]:
    """Build the trace runs from the canonical registry.

    The registry names 1 thinking run for each case and model, and
    `verify_or_raise` stops the call when a file moved. No network, no cache,
    and no chance of a run from an older prompt.
    """
    import _canonical as C

    C.verify_or_raise()
    out: List[TraceRun] = []
    for r in C.runs(kind="trace"):
        if cases and r.case not in set(cases):
            continue
        out.append(TraceRun(
            wandb_id=r.run_id,
            wandb_url="",
            created_at=r.created_at,
            state="finished",
            case=r.case,
            model=r.model_config,
            sweep=r.sweep,
            pipeline=r.pipeline,
            stage_dir=r.stage_dir,
            results_path=r.results_link,
            rows=r.rows,
            traced_rows=r.trace_rows,
            question=r.question,
            notes=list(r.problems),
        ))
    out.sort(key=lambda r: (r.case, r.model))
    return out


def _discover_wandb(
    only_finished: bool, limit: int, require_trace: bool, refresh: bool,
) -> List[TraceRun]:
    """Read the trace runs that W&B can resolve. This part uses the cache."""
    query = _cache_key(only_finished, limit, require_trace, __version__)
    if not refresh and DISCOVERY_CACHE.exists():
        try:
            blob = json.loads(DISCOVERY_CACHE.read_text())
            if blob.get("query") == query:
                return [TraceRun(**d) for d in blob["runs"]]
        except Exception:
            pass

    records = P.discover_runs(
        case="", only_canonical=False, only_finished=only_finished, limit=limit,
    )
    out: List[TraceRun] = []
    for r in records:
        if not r.stage_dir or not r.pipeline:
            continue
        case = case_of(r.pipeline)
        path = find_results_parquet(r.stage_dir, case)
        notes: List[str] = []
        has_trace, rows = (False, None)
        if path:
            has_trace, rows = _peek_traces(path)
            if not has_trace:
                notes.append("no trace in this parquet (label-only run)")
        else:
            notes.append("no parquet yet")
        if require_trace and not has_trace:
            continue
        out.append(TraceRun(
            wandb_id=r.wandb_id,
            wandb_url=r.wandb_url,
            created_at=str(r.created_at),
            state=r.state,
            case=case,
            model=r.model or "unknown",
            sweep=r.sweep or "",
            pipeline=r.pipeline,
            stage_dir=r.stage_dir,
            results_path=path,
            rows=rows,
            question=read_question(r.stage_dir),
            notes=notes,
        ))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        DISCOVERY_CACHE.write_text(json.dumps(
            {"query": query, "runs": [asdict(r) for r in out]}, indent=1,
        ))
    except Exception:
        pass
    return out


def runs_table(runs: Sequence[TraceRun]) -> pd.DataFrame:
    """Build the provenance table that the paper cites."""
    if not runs:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in runs])
    df["notes"] = df["notes"].apply(lambda x: "; ".join(x) if x else "")
    df["readable"] = [r.is_readable for r in runs]
    cols = ["case", "question", "model", "sweep", "state", "created_at", "rows",
            "readable", "wandb_id", "results_path", "wandb_url", "notes"]
    return df[[c for c in cols if c in df.columns]]


def normalize_question(q: str) -> str:
    """Reduce a question to a form that 2 runs can be compared on.

    This lowers the case and collapses the white space. Without it, the same
    question counts twice: the deprecated schools prompt carried "which school
    would you rather send your child to?" inside a sentence, and the newer file
    starts the line with "Which". They are 1 question, not 2.

    Warning: a template variable survives this on purpose. The street
    photography prompt exists in a plain form and a `{PURPOSE}` form, and those
    ARE 2 different prompts. Do not paper over that.
    """
    return " ".join((q or "").lower().split())


def safe_name(group: str) -> str:
    """Turn a group name into a file name that cannot collide.

    A group name can carry the question, for example
    `street_photography | Which block is better suited for street photography?`.
    A plain truncation of that is NOT safe: the 2 street photography groups
    agree for the first 47 characters, so both wrote the same PNG and 1
    silently replaced the other. Thus a name that needs a cut also carries a
    hash of the whole group name.
    """
    base = re.sub(r"[^A-Za-z0-9]+", "_", group).strip("_")
    if len(base) <= 60:
        return base
    return f"{base[:60]}_{hashlib.sha1(group.encode()).hexdigest()[:6]}"


def mixed_question_cases(runs: Sequence[TraceRun]) -> Dict[str, List[str]]:
    """Find each case whose runs did not all ask the same question.

    Call this before you pool runs into 1 cloud. A case with 2 questions must
    be split, or the cloud mixes them with nothing on screen to say so.

    A run with no recorded question does not enter the test. `think10k` has no
    Hydra record at all, thus its question is unknown, and an unknown cannot
    prove a difference. Read `runs_table` to see which runs those are.
    """
    by_case: Dict[str, Dict[str, str]] = {}
    for r in runs:
        if r.question:
            # Key on the normalized form, but show the run's own wording.
            by_case.setdefault(r.case, {}).setdefault(
                normalize_question(r.question), r.question
            )
    return {c: sorted(v.values()) for c, v in by_case.items() if len(v) > 1}


def load_traces(run: TraceRun) -> pd.DataFrame:
    """Read the trace column and the label column of one run."""
    if not run.is_readable:
        raise FileNotFoundError(f"run {run.wandb_id} has no parquet on disk")
    cols = [REASONING_COL, LABEL_COL]
    try:
        df = pd.read_parquet(run.results_path, columns=cols)
    except Exception:
        df = pd.read_parquet(run.results_path)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df[cols]
    df[REASONING_COL] = df[REASONING_COL].fillna("")
    df[LABEL_COL] = df[LABEL_COL].fillna("")
    return df


def tokenize(text: str, stopwords: frozenset) -> List[str]:
    """Turn 1 trace into a list of words.

    The function makes the text lower case, removes the markdown, and keeps the
    words of 3 letters or more that are not in `stopwords`.
    """
    t = _MARKUP_RE.sub(" ", text.lower())
    return [w for w in _TOKEN_RE.findall(t) if w not in stopwords]


def default_stopwords(extra: Iterable[str] = ()) -> frozenset:
    """Build the stopword set: English, plus prompt echo, plus your words."""
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        base = set(ENGLISH_STOP_WORDS)
    except Exception:
        base = set()
    base |= set(BOILERPLATE)
    base |= {w.strip().lower() for w in extra if w and w.strip()}
    return frozenset(base)


def count_words(
    texts: Sequence[str],
    stopwords: frozenset,
    ngram: int = 1,
) -> Counter:
    """Count the words, or the word pairs, of a set of traces.

    Args:
        ngram: 1 counts single words. 2 counts pairs of neighbour words, which
            reads better for a phrase such as `well maintained`.
    """
    c: Counter = Counter()
    for t in texts:
        toks = tokenize(t, stopwords)
        if ngram <= 1:
            c.update(toks)
        else:
            c.update(" ".join(toks[i:i + ngram]) for i in range(len(toks) - ngram + 1))
    return c


def _cache_key(*parts: Any) -> str:
    h = hashlib.sha1("||".join(str(p) for p in parts).encode()).hexdigest()
    return h[:20]


def counts_for_run(
    run: TraceRun,
    stopwords: frozenset,
    ngram: int = 1,
    labels: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> Counter:
    """Count the words of one run, and keep the result on disk.

    Args:
        labels: Keep only the traces that end in these labels. `None` keeps all
            of them.

    A 10k-pair run holds about 40 MB of text, thus the first count costs some
    seconds. The cache makes each later call immediate.
    """
    mtime = os.path.getmtime(run.results_path) if run.is_readable else 0
    key = _cache_key(run.results_path, mtime, ngram, sorted(labels or []),
                     _cache_key(*sorted(stopwords)), __version__)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        try:
            return Counter(json.loads(cache_file.read_text()))
        except Exception:
            pass

    df = load_traces(run)
    if labels:
        df = df[df[LABEL_COL].isin(list(labels))]
    c = count_words(df[REASONING_COL].tolist(), stopwords, ngram=ngram)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            # Keep the head only. A full count holds a long tail of words that
            # appear 1 time, and no cloud ever shows them.
            cache_file.write_text(json.dumps(dict(c.most_common(20000))))
        except Exception:
            pass
    return c


def distinctive_scores(
    target: Counter,
    background: Counter,
    prior_weight: float = 500.0,
    min_count: int = 5,
) -> Dict[str, float]:
    """Score the words that 1 prompt uses more than the other prompts.

    This is the log-odds ratio with an informative Dirichlet prior. The prior
    comes from the 2 corpora together, and the score divides by its own standard
    deviation. Thus a word that appears 3 times cannot beat a word that appears
    3,000 times, which a plain ratio always lets it do.

    Args:
        target: The counts of this prompt.
        background: The counts of the other prompts together.
        prior_weight: The size of the prior, in words. A larger value holds down
            the rare words more strongly.
        min_count: Drop a word that appears fewer times than this in the 2
            corpora together.

    Returns:
        A dict of word to z-score. A positive score means this prompt uses the
        word more. The caller usually keeps the positive scores only.
    """
    pooled = Counter(target)
    pooled.update(background)
    pooled = Counter({w: n for w, n in pooled.items() if n >= min_count})
    total_pooled = sum(pooled.values()) or 1

    n_t = sum(target[w] for w in pooled) or 1
    n_b = sum(background[w] for w in pooled) or 1

    scores: Dict[str, float] = {}
    for w, n_pool in pooled.items():
        alpha = prior_weight * n_pool / total_pooled
        yt = target.get(w, 0) + alpha
        yb = background.get(w, 0) + alpha
        # The odds of the word inside each corpus, against the rest of it.
        odds_t = yt / max(n_t + prior_weight - yt, 1e-9)
        odds_b = yb / max(n_b + prior_weight - yb, 1e-9)
        delta = math.log(odds_t) - math.log(odds_b)
        var = 1.0 / yt + 1.0 / yb
        scores[w] = delta / math.sqrt(var)
    return scores


def cloud_weights(
    counts_by_case: Dict[str, Counter],
    case: str,
    mode: str = "distinctive",
    max_words: int = 120,
    prior_weight: float = 500.0,
    min_count: int = 5,
) -> Dict[str, float]:
    """Build the word-to-weight dict that a word cloud draws.

    Args:
        mode: `frequency` or `distinctive`.

    Warning: `distinctive` needs 2 cases or more. With 1 case there is no
    background, thus the function falls back to `frequency`.
    """
    target = counts_by_case.get(case, Counter())
    if mode == "frequency" or len(counts_by_case) < 2:
        return dict(target.most_common(max_words))

    background: Counter = Counter()
    for other, c in counts_by_case.items():
        if other != case:
            background.update(c)
    scores = distinctive_scores(target, background, prior_weight, min_count)
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_words]
    return {w: s for w, s in top if s > 0}


def make_cloud(
    weights: Dict[str, float],
    width: int = 900,
    height: int = 500,
    colormap=None,
    background_color: Optional[str] = None,
    random_state: Optional[int] = None,
):
    """Draw 1 word cloud. Return the `WordCloud` object.

    The colours come from the paper's house palette in `_style.py`, so a cloud
    and a chart read as one system. Pass `colormap` and `background_color` only
    to leave that system on purpose.

    The background is TRANSPARENT since 2026-08-14. A cloud thus brings no
    coloured box onto the page, and it matches the figures, which also save
    transparent. Pass `background_color="white"` to get a solid ground back.

    Warning: a transparent cloud needs `mode="RGBA"`. Without that mode the
    library paints black behind the words, and the ink is dark, so the picture
    is unreadable.

    Warning: `random_state` is set. A word cloud lays words out in a random
    order, thus without a seed the same counts draw a different picture every
    run, and no reader can tell a real change from a reshuffle.

    Raises:
        ValueError: if `weights` is empty. A caller must test for this, because
            a case with no readable run gives an empty dict.
    """
    from wordcloud import WordCloud

    import _style as S

    if not weights:
        raise ValueError("no words to draw")
    wc = WordCloud(
        width=width, height=height,
        background_color=background_color,
        mode="RGB" if background_color else "RGBA",
        colormap=colormap if colormap is not None else S.WORDCLOUD_CMAP,
        prefer_horizontal=0.9, max_words=len(weights),
        relative_scaling=0.5,
        random_state=S.WORDCLOUD_SEED if random_state is None else random_state,
    )
    return wc.generate_from_frequencies(weights)
