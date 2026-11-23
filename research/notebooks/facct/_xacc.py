"""F9: the accuracy and the performance of the span extraction, and the floors of span precision.

The F9 round (D142) extracts spans from 400,000 texts: the F9 answers and
the F4 answers of the same images, under 1 schema for each case
(`urban_cues_single_subj_v1`, `urban_cues_street_subj_v1`). The H9 test
counts the `visual_evidence` spans and their valence. This module tells how
far a reader can trust those counts, at 3 levels:

| Level | Question | Source | Kind of number |
|-------|----------|--------|----------------|
| 1. Grounding | Is the span text in the source text? | the aligner (`is_verbatim`, `alignment_status`) | exact, for every span |
| 2. Class | Is the span of the class the extractor gave? | a person labels a random sample | a floor (a one-sided lower bound) |
| 3. Class and attribute | Is the key attribute right too (the valence of a `visual_evidence` span)? | the same labels | a floor |

A floor is the one-sided Clopper-Pearson lower bound of the precision of a
stratum: with probability 1 - alpha, the true share of correct spans is at
or above it. The floor times the count of spans in the corpus gives the
floor COUNT: "at least N of the spans of this class are correct". The floor
of a class pools its strata with the corpus weights, and the alpha of each
stratum is alpha / (the strata of that class) (Bonferroni). Thus the pooled
floor holds with probability 1 - alpha too.

Recall (the share of the true spans that the extractor found) needs a person
to mark every span of a text. The module draws that sample too
(`draw_recall_sample`), but the notebook reads it only when labels exist.

Use, from the project root (about 5 minutes; it reads all merged spans)::

    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/facct/_xacc.py build

It writes the tables to `research/figures/f9/xacc/` and the label sheets to
`data/f9_xacc/`. A filled sheet is `data/f9_xacc/precision_labels.csv`
(the same columns as `precision_sheet.csv`).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__version__ = "1.1.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
X2 = REPO_ROOT / "data" / "f4_traces" / "subj" / "x2"
INPUT = REPO_ROOT / "data" / "f4_traces" / "extract_input"
OUT = REPO_ROOT / "research" / "figures" / "f9" / "xacc"
SAMPLE_DIR = REPO_ROOT / "data" / "f9_xacc"
LOGS = REPO_ROOT / "logs" / "f4"

CASES = ("restaurant", "street")
PROMPTS = ("f4", "f9")
FIELDS = ("reasoning", "response")
# case -> prompt -> the merged files (arm dir, input case), as research/scripts/f9/analyze_subjective.py
FILES: Dict[str, Dict[str, tuple]] = {
    "restaurant": {"f9": (("cues", "restaurant_subj_x2"),),
                   "f4": (("cues", "restaurant_x2"), ("cues", "restaurant_b2x2"))},
    "street": {"f9": (("cues_street", "street_subj_x2"),),
               "f4": (("cues_street", "street_x2"), ("cues_street", "street_b2x2"))},
}
QUESTIONS = {
    ("restaurant", "f9"): "Does this look like a nice restaurant to eat at?",
    ("restaurant", "f4"): "Look and tell me more about this restaurant.",
    ("street", "f9"): "Would this street make a good spot for taking a photo?",
    ("street", "f4"): "Look and tell me a good angle to take a picture from.",
}
CLASSES = ("visual_evidence", "inference", "uncertainty", "image_artifact", "person_reference",
           "photo_advice", "verdict")
# The attribute that a label checks for each class.
KEY_ATTRIBUTE = {
    "visual_evidence": "valence", "inference": "kind", "uncertainty": "reason", "image_artifact": "kind",
    "person_reference": "used_in_judgment", "photo_advice": "kind", "verdict": "label",
}
VALENCES = ("good", "bad", "neutral")
COLUMNS = ["doc_id", "model", "field", "image_idx", "extraction_class", "extraction_text", "attributes_json",
           "char_start", "char_end", "alignment_status", "is_quotable", "is_verbatim", "alignment_rescue",
           "is_duplicate", "copy_suspect", "retried"]
SEED = 20261030
ALPHA = 0.05


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


def load_raw(x2: Path = X2) -> pd.DataFrame:
    """Read every row of the 6 merged files, with `case`, `prompt` and `source_file`."""
    frames = []
    for case, by_prompt in FILES.items():
        for prompt, files in by_prompt.items():
            for arm, name in files:
                d = pd.read_parquet(x2 / arm / f"f4_{name}_extractions.parquet", columns=COLUMNS)
                frames.append(d.assign(case=case, prompt=prompt, source_file=name))
    return pd.concat(frames, ignore_index=True)


def text_lengths() -> pd.DataFrame:
    """Give the length in characters of each input text: `doc_id`, `chars`."""
    out = []
    for by_prompt in FILES.values():
        for files in by_prompt.values():
            for _, name in files:
                t = pd.read_parquet(INPUT / f"f4_{name}.parquet", columns=["doc_id", "text"])
                out.append(pd.DataFrame({"doc_id": t.doc_id, "chars": t.text.str.len()}))
    return pd.concat(out, ignore_index=True).drop_duplicates("doc_id")


def counted(raw: pd.DataFrame) -> pd.DataFrame:
    """Give the spans that the analysis counts: a class, quotable, not a duplicate, not in a copy suspect."""
    cls = raw.extraction_class
    keep = (cls.notna() & (cls != "none") & raw.is_quotable.fillna(False).astype(bool)
            & ~raw.is_duplicate.fillna(False).astype(bool) & ~raw.copy_suspect.astype(bool))
    d = raw[keep].copy()
    attrs = d.attributes_json.fillna("{}").map(json.loads)
    d["key_attribute"] = [a.get(KEY_ATTRIBUTE.get(c, ""), None) for a, c in zip(attrs, d.extraction_class)]
    d["key_attribute"] = d.key_attribute.map(lambda v: None if v is None else str(v).strip().lower()
                                             .replace(" ", "_").replace("-", "_"))
    return d


# ---------------------------------------------------------------------------
# Level 1: the grounding, for every span
# ---------------------------------------------------------------------------


def grounding(raw: pd.DataFrame) -> pd.DataFrame:
    """Give the alignment of the spans for each (case, prompt, field, class).

    | Column | Meaning |
    |--------|---------|
    | `exact` | `match_exact`: the aligner found the span text in the source |
    | `fuzzy` | `match_fuzzy`: the aligner found a near match |
    | `lesser` | `match_lesser`: the aligner found only a part of the span |
    | `unaligned` | no alignment |
    | `quotable` | the span enters a count (exact, fuzzy, or a rescue) |
    | `verbatim` | the span text equals the source slice, character for character |

    `verbatim` is a guarantee, not an estimate: the merge compares the 2 strings.
    """
    d = raw[raw.extraction_class.notna() & (raw.extraction_class != "none")]
    st = d.alignment_status.fillna("unaligned")
    g = d.assign(exact=st == "match_exact", fuzzy=st == "match_fuzzy", lesser=st == "match_lesser",
                 unaligned=st == "unaligned", quotable=d.is_quotable.fillna(False).astype(bool),
                 verbatim=d.is_verbatim.fillna(False).astype(bool),
                 duplicate=d.is_duplicate.fillna(False).astype(bool))
    keys = ["case", "prompt", "field", "extraction_class"]
    out = g.groupby(keys)[["exact", "fuzzy", "lesser", "unaligned", "quotable", "verbatim", "duplicate"]].mean()
    out.insert(0, "spans", g.groupby(keys).size())
    out["verbatim_count"] = g.groupby(keys).verbatim.sum()
    return out.reset_index()


# ---------------------------------------------------------------------------
# Coverage: how much of each text the spans read
# ---------------------------------------------------------------------------


def _covered_chars(sp: pd.DataFrame) -> pd.Series:
    """Give the characters of each text inside 1 span or more (the union of the span intervals)."""
    s = sp[["doc_id", "char_start", "char_end"]].dropna().sort_values(["doc_id", "char_start"])
    prev = s.groupby("doc_id").char_end.cummax().groupby(s.doc_id).shift().fillna(-1.0)
    gain = (s.char_end - np.maximum(s.char_start, prev)).clip(lower=0)
    return gain.groupby(s.doc_id).sum()


def coverage(raw: pd.DataFrame, sp: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Give, for each (case, prompt, model, field): the texts, the share with a counted span, the spans per
    text, the share of the characters inside a span, and the share of the spans in the second half of the
    long texts (4,000 characters or more)."""
    t = raw.drop_duplicates("doc_id")[["doc_id", "case", "prompt", "model", "field", "copy_suspect"]]
    t = t.merge(lengths, on="doc_id", how="left")
    n = sp.groupby("doc_id").size().rename("spans")
    t = t.merge(n, left_on="doc_id", right_index=True, how="left").fillna({"spans": 0})
    t = t.merge(_covered_chars(sp).rename("covered"), left_on="doc_id", right_index=True, how="left")
    t["covered"] = t.covered.fillna(0.0)
    t["char_share"] = t.covered / t.chars.clip(lower=1)
    pos = sp.merge(lengths, on="doc_id")
    pos = pos[pos.chars >= 4000]
    half = (pos.char_start > pos.chars / 2).groupby([pos.case, pos.prompt, pos.model, pos.field]).mean()
    keys = ["case", "prompt", "model", "field"]
    g = t.groupby(keys)
    out = pd.DataFrame({
        "texts": g.size(), "copy_suspects": g.copy_suspect.sum(), "chars_p50": g.chars.median(),
        "share_with_span": g.spans.apply(lambda s: float((s > 0).mean())), "spans_per_text": g.spans.mean(),
        "char_share_mean": g.char_share.mean(), "char_share_p10": g.char_share.quantile(0.10),
    })
    out["second_half_share_long"] = half.reindex(out.index)
    return out.reset_index()


def verbatim_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Give the verbatim precision and the per-text verbatim recall of the extractor.

    A span counts when it has a class other than `none`. The 2 measures:

    | Column | Meaning |
    |--------|---------|
    | `precision` | the share of the spans whose text equals the source slice (`is_verbatim`) |
    | `recall_text` | the share of the texts with at least 1 verbatim span |
    | `class_min`, `class_min_verbatim` | the class with the lowest verbatim share, and that share |

    One row for each (case, prompt, field), and 1 row `all` that pools every text.
    """
    cls = raw.extraction_class
    d = raw[cls.notna() & (cls != "none")].copy()
    d["verbatim"] = d.is_verbatim.fillna(False).astype(bool)
    keys = ["case", "prompt", "field"]
    texts = raw.groupby(keys + ["doc_id"]).size().rename("rows").reset_index()
    hit = d.groupby(keys + ["doc_id"]).verbatim.sum().rename("n_verbatim").reset_index()
    texts = texts.merge(hit, on=keys + ["doc_id"], how="left").fillna({"n_verbatim": 0})
    rows = []
    for name, dd, tt in [(k, g, texts[(texts[keys] == pd.Series(k, index=keys)).all(axis=1)])
                         for k, g in d.groupby(keys)] + [(("all", "all", "all"), d, texts)]:
        by_class = dd.groupby("extraction_class").verbatim.mean()
        rows.append({"case": name[0], "prompt": name[1], "field": name[2], "texts": len(tt),
                     "spans": len(dd), "verbatim_spans": int(dd.verbatim.sum()),
                     "precision": dd.verbatim.mean(), "recall_text": (tt.n_verbatim > 0).mean(),
                     "class_min": by_class.idxmin(), "class_min_verbatim": by_class.min()})
    return pd.DataFrame(rows)


def position_profile(sp: pd.DataFrame, lengths: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Give the spans per 1,000 characters in each tenth of the long texts (4,000 characters or more).

    A tail that falls off says that the extractor stops reading before the end of the text.
    """
    pos = sp.merge(lengths, on="doc_id")
    pos = pos[pos.chars >= 4000]
    pos["decile"] = np.minimum((pos.char_start / pos.chars * bins).astype(int), bins - 1)
    n = pos.groupby(["case", "prompt", "field", "decile"]).size().rename("spans").reset_index()
    texts = pos.drop_duplicates("doc_id").groupby(["case", "prompt", "field"]).agg(
        texts=("doc_id", "size"), chars=("chars", "sum")).reset_index()
    n = n.merge(texts, on=["case", "prompt", "field"])
    n["per_1000_chars"] = n.spans / (n.chars / bins / 1000)
    return n


# ---------------------------------------------------------------------------
# Performance: the GPU time and the throughput of the run
# ---------------------------------------------------------------------------


_BATCH = re.compile(r"\[trace_extract\] batch \d+: (\d+) prompts in ([\d.]+)s")
_STEP = re.compile(r"^\[x2\] (start|done|FAILED) (\S+) (\S+) (\d+)/(\d+)")


def throughput(logs: Sequence[Path]) -> pd.DataFrame:
    """Read the extraction logs: for each log, the batches, the prompts, the seconds in `generate`, the steps."""
    rows = []
    for path in logs:
        text = path.read_text(errors="replace")
        b = [(int(n), float(s)) for n, s in _BATCH.findall(text)]
        steps = [m.groups() for m in map(_STEP.match, text.splitlines()) if m]
        rows.append({
            "log": path.name, "batches": len(b), "prompts": sum(n for n, _ in b),
            "generate_s": sum(s for _, s in b),
            "steps_started": sum(1 for s in steps if s[0] == "start"),
            "steps_done": sum(1 for s in steps if s[0] == "done"),
            "steps_failed": sum(1 for s in steps if s[0] == "FAILED"),
            "preempted": text.count("DUE TO PREEMPTION"),
        })
    out = pd.DataFrame(rows)
    out["prompts_per_s"] = out.prompts / out.generate_s.where(out.generate_s > 0)
    return out


def job_hours(job_ids: Sequence[str]) -> pd.DataFrame:
    """Ask `sacct` for the allocations of the jobs (every requeue is 1 row): state and elapsed hours."""
    if not job_ids:
        return pd.DataFrame()
    cmd = ["sacct", "-M", "g3", "-D", "-X", "-n", "-P", "-j", ",".join(job_ids),
           "--format=JobID,JobName,State,ElapsedRaw,NodeList"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return pd.DataFrame({"error": [str(exc)]})
    rows = [line.split("|") for line in res.stdout.strip().splitlines() if line]
    out = pd.DataFrame(rows, columns=["job", "name", "state", "elapsed_s", "node"])
    out["hours"] = out.elapsed_s.astype(float) / 3600
    return out


def run_job_ids() -> List[str]:
    """Give the job ids of the F9 extraction round from the job files of `logs/f4/`."""
    ids = ["105550"]
    p1 = LOGS / "f9x2_pass1_jobs.txt"
    if p1.exists():
        ids += [line.split()[2].split(";")[0] for line in p1.read_text().splitlines() if line.strip()]
    p2 = LOGS / "f9x2_pass2_job.txt"
    if p2.exists():
        ids += [p2.read_text().strip().split(";")[0]]
    return ids


# ---------------------------------------------------------------------------
# Levels 2 and 3: the floors
# ---------------------------------------------------------------------------


def cp_lower(k: int, n: int, alpha: float = ALPHA) -> float:
    """Give the one-sided Clopper-Pearson lower bound of a share, with `k` successes of `n`."""
    from scipy.stats import beta

    if n == 0:
        return float("nan")
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def floor_table(ns: Sequence[int] = (20, 30, 50, 100, 200), alpha: float = ALPHA) -> pd.DataFrame:
    """Give the floor for n labels with 0, 1, 2 errors, and with 5% errors: how many labels a floor needs."""
    rows = []
    for n in ns:
        row = {"labels": n}
        for e in (0, 1, 2):
            row[f"{e} errors"] = cp_lower(n - e, n, alpha)
        e5 = int(round(0.05 * n))
        row["5% errors"] = cp_lower(n - e5, n, alpha)
        rows.append(row)
    return pd.DataFrame(rows)


def strata(sp: pd.DataFrame) -> pd.Series:
    """Give the stratum of each span: case|prompt|class, and the valence for `visual_evidence`."""
    s = sp.case + "|" + sp.prompt + "|" + sp.extraction_class
    ve = sp.extraction_class == "visual_evidence"
    return s.where(~ve, s + "|" + sp.key_attribute.fillna("none").astype(str))


def populations(sp: pd.DataFrame) -> pd.DataFrame:
    """Give the count of counted spans in each stratum."""
    s = strata(sp)
    return s.value_counts().rename_axis("stratum").rename("spans").reset_index().sort_values("stratum")


def _context(text: str, start: int, end: int, width: int = 350) -> str:
    a, b = max(0, start - width), min(len(text), end + width)
    pre = ("..." if a > 0 else "") + text[a:start]
    post = text[end:b] + ("..." if b < len(text) else "")
    return f"{pre}[[{text[start:end]}]]{post}"


def _source_texts(doc_ids: Sequence[str]) -> Dict[str, str]:
    want, out = set(doc_ids), {}
    for by_prompt in FILES.values():
        for files in by_prompt.values():
            for _, name in files:
                t = pd.read_parquet(INPUT / f"f4_{name}.parquet", columns=["doc_id", "text"])
                t = t[t.doc_id.isin(want)]
                out.update(zip(t.doc_id, t.text))
    return out


def draw_precision_sample(sp: pd.DataFrame, n_per_stratum: int = 30, seed: int = SEED) -> pd.DataFrame:
    """Draw `n_per_stratum` counted spans at random from each stratum (all of them when fewer), and add the
    context of each span. The rows come in a random order, thus a labeller does not see the strata.

    Valence strata (`visual_evidence` by good, bad, neutral) are separate, because the H9 test reads the
    valence, and good and bad spans are rare in the F4 restaurant answers.
    """
    s = sp.assign(stratum=strata(sp))
    s = s[s.key_attribute.notna() | (s.extraction_class != "visual_evidence")]
    parts = [g.sample(n=min(n_per_stratum, len(g)), random_state=seed) for _, g in s.groupby("stratum")]
    smp = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    texts = _source_texts(smp.doc_id.unique())
    smp["context"] = [_context(texts[d], int(a), int(b)) if d in texts else ""
                      for d, a, b in zip(smp.doc_id, smp.char_start, smp.char_end)]
    smp["question"] = [QUESTIONS[(c, p)] for c, p in zip(smp.case, smp.prompt)]
    smp.insert(0, "sample_id", [f"p{i:04d}" for i in range(len(smp))])
    return smp


def precision_sheet(smp: pd.DataFrame) -> pd.DataFrame:
    """Give the label sheet of a precision sample. The model name and the prompt group stay out of it.

    | Label column | Value |
    |--------------|-------|
    | `class_ok` | 1 when the span is of the class given, else 0 |
    | `attribute_ok` | 1 when the key attribute is right too, else 0; blank when `class_ok` is 0 |
    | `bounds_ok` | 1 when the span holds the whole claim and no more than 1 claim, else 0 |
    | `true_class`, `true_attribute` | the right values when the given ones are wrong |
    | `note` | free text |
    """
    out = smp[["sample_id", "question", "field", "extraction_class", "key_attribute", "extraction_text",
               "context"]].rename(columns={"extraction_class": "class", "key_attribute": "attribute",
                                           "extraction_text": "span"})
    out.insert(4, "attribute_name", out["class"].map(KEY_ATTRIBUTE))
    for col in ("class_ok", "attribute_ok", "bounds_ok", "true_class", "true_attribute", "note"):
        out[col] = ""
    return out


def draw_recall_sample(raw: pd.DataFrame, lengths: pd.DataFrame, per_cell: int = 1, max_chars: int = 6000,
                       seed: int = SEED) -> pd.DataFrame:
    """Draw `per_cell` texts of each (case, prompt, model, field) with at most `max_chars` characters, for a
    person to mark every span (recall). The cap keeps the work of 1 text under about 10 minutes."""
    t = raw.drop_duplicates("doc_id")[["doc_id", "case", "prompt", "model", "field", "copy_suspect"]]
    t = t[~t.copy_suspect.astype(bool)].merge(lengths, on="doc_id")
    t = t[t.chars <= max_chars]
    parts = [g.sample(n=min(per_cell, len(g)), random_state=seed)
             for _, g in t.groupby(["case", "prompt", "model", "field"])]
    smp = pd.concat(parts).reset_index(drop=True)
    texts = _source_texts(smp.doc_id)
    smp["text"] = smp.doc_id.map(texts)
    smp["question"] = [QUESTIONS[(c, p)] for c, p in zip(smp.case, smp.prompt)]
    smp.insert(0, "sample_id", [f"r{i:03d}" for i in range(len(smp))])
    return smp


def load_labels(path: Path = SAMPLE_DIR / "precision_labels.csv") -> Optional[pd.DataFrame]:
    """Read a filled label sheet and join it to the sample. None when no labels exist yet."""
    if not path.exists():
        return None
    lab = pd.read_csv(path, dtype=str)
    lab = lab[lab.class_ok.fillna("").str.strip().isin(("0", "1"))]
    if lab.empty:
        return None
    smp = pd.read_parquet(SAMPLE_DIR / "precision_sample.parquet")
    lab = smp.merge(lab[["sample_id", "class_ok", "attribute_ok", "bounds_ok", "true_class", "true_attribute",
                         "note"]], on="sample_id", how="inner")
    lab["class_ok"] = lab.class_ok.astype(int)
    lab["both_ok"] = (lab.class_ok == 1) & (lab.attribute_ok.fillna("").str.strip() == "1")
    lab["bounds_ok"] = pd.to_numeric(lab.bounds_ok, errors="coerce")
    lab["stratum"] = strata(lab)
    return lab


def precision_floors(lab: pd.DataFrame, pops: pd.DataFrame, alpha: float = ALPHA) -> Dict[str, pd.DataFrame]:
    """Give the floors of each stratum and of each (case, prompt, class).

    | Table | Rows |
    |-------|------|
    | `strata` | 1 row for each stratum: labels, the share correct, its floor (alpha), the floor count |
    | `classes` | 1 row for each (case, prompt, class): the weighted share correct and the pooled floor. The alpha of each stratum is alpha / (the strata of the class), thus the pooled floor holds with 1 - alpha |

    Both tables give 2 measures: `class` (`class_ok`) and `class_attr` (`class_ok` and `attribute_ok`).
    """
    pop = pops.set_index("stratum").spans
    rows = []
    for st, g in lab.groupby("stratum"):
        n = len(g)
        for measure, ok in (("class", g.class_ok == 1), ("class_attr", g.both_ok)):
            k = int(ok.sum())
            rows.append({"stratum": st, "measure": measure, "labels": n, "correct": k, "share": k / n,
                         "floor": cp_lower(k, n, alpha), "spans": int(pop.get(st, 0))})
    by = pd.DataFrame(rows)
    by["floor_count"] = (by.floor * by.spans).round().astype(int)
    parts = by.stratum.str.split("|", expand=True)
    by["case"], by["prompt"], by["class"] = parts[0], parts[1], parts[2]
    out = []
    for (case, prompt, cls, measure), g in by.groupby(["case", "prompt", "class", "measure"]):
        m = len(g)
        w = g.spans / g.spans.sum()
        lows = [cp_lower(int(k), int(n), alpha / m) for k, n in zip(g.correct, g.labels)]
        out.append({"case": case, "prompt": prompt, "class": cls, "measure": measure, "strata": m,
                    "labels": int(g.labels.sum()), "spans": int(g.spans.sum()),
                    "share": float((w * g.share).sum()), "floor": float((w * np.array(lows)).sum())})
    cls_tab = pd.DataFrame(out)
    cls_tab["floor_count"] = (cls_tab.floor * cls_tab.spans).round().astype(int)
    return {"strata": by, "classes": cls_tab}


# ---------------------------------------------------------------------------
# The notebook side: the tables and the figures
# ---------------------------------------------------------------------------


TABLES = ("grounding", "verbatim_summary", "coverage", "position_profile", "populations", "throughput", "job_hours", "floor_table")


def load_tables(out: Path = OUT) -> Dict[str, pd.DataFrame]:
    """Read the tables that `build` wrote.

    Raises:
        FileNotFoundError: when `build` did not run.
    """
    missing = [t for t in TABLES if not (out / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"{out}: no {missing}. Run `python research/notebooks/facct/_xacc.py build` first.")
    return {t: pd.read_csv(out / f"{t}.csv") for t in TABLES}


def grounding_wide(g: pd.DataFrame, column: str = "verbatim") -> pd.DataFrame:
    """Give the share `column` of each class (rows) for each case and prompt (columns), over both fields."""
    w = g.assign(x=g[column] * g.spans).groupby(["extraction_class", "case", "prompt"])[["x", "spans"]].sum()
    w = (w.x / w.spans).unstack(["case", "prompt"])
    w.columns = [f"{c}, {p.upper()}" for c, p in w.columns]
    return w.reindex([c for c in CLASSES if c in w.index])


def plot_grounding(g: pd.DataFrame):
    """Plot the verbatim share of each class, 1 marker for each case and prompt."""
    import matplotlib.pyplot as plt

    from research.notebooks.facct import _f8 as F8

    w = grounding_wide(g)
    colors = {"restaurant": F8.S.PAL["blue"], "street": F8.S.PAL["teal"]}
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for j, col in enumerate(w.columns):
        case, prompt = col.split(", ")
        y = np.arange(len(w)) + (j - 1.5) * 0.15
        ax.scatter(w[col], y, s=18, marker="o" if prompt == "F4" else "s",
                   facecolors=colors[case] if prompt == "F9" else "white", edgecolors=colors[case],
                   linewidths=0.9, zorder=3, label=col)
    ax.set_yticks(range(len(w)), [c.replace("_", " ") for c in w.index])
    ax.invert_yaxis()
    ax.set_xlabel("share of spans that are verbatim")
    ax.set_xlim(0.85, 1.0)
    F8._top_legend(fig, ncol=4)
    F8.describe(
        fig, "The grounding of the extracted spans",
        "The share of the spans of each class whose text equals the source slice, character for character "
        "(`is_verbatim`), over the reasoning and response fields. Hollow: the F4 prompt; filled: the F9 prompt. "
        "The aligner checks this for every span; it is not an estimate. A verbatim span exists in the text, "
        "but its class and its attributes can still be wrong.",
        ["f9", "extraction", "grounding"])
    return fig


def plot_position(p: pd.DataFrame):
    """Plot the spans per 1,000 characters in each tenth of the long reasoning traces."""
    import matplotlib.pyplot as plt

    from research.notebooks.facct import _f8 as F8

    colors = {"restaurant": F8.S.PAL["blue"], "street": F8.S.PAL["teal"]}
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for (case, prompt), g in p[p.field == "reasoning"].groupby(["case", "prompt"]):
        g = g.sort_values("decile")
        ax.plot(g.decile * 10 + 5, g.per_1000_chars, color=colors[case], ls="-" if prompt == "f9" else "--",
                marker="o", ms=3.5, mfc=colors[case] if prompt == "f9" else "white", mec=colors[case],
                label=f"{case}, {prompt.upper()}")
    ax.set_xlabel("position in the trace (%)")
    ax.set_ylabel("spans per 1,000 characters")
    ax.set_ylim(bottom=0)
    with plt.rc_context({"legend.handlelength": 3.0}):  # long enough to show the dashes
        F8._top_legend(fig, ncol=2)
    F8.describe(
        fig, "The span density along the long reasoning traces",
        "The counted spans per 1,000 characters in each tenth of the reasoning traces of 4,000 characters or "
        "more. Dashed with hollow markers: the F4 prompt; solid with filled markers: the F9 prompt. A density that falls to 0 near the end would say "
        "that the extractor stops reading; in the last tenth the density is 47% to 72% of its peak.",
        ["f9", "extraction", "coverage"])
    return fig


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build(n_per_stratum: int = 30, out: Path = OUT, sample_dir: Path = SAMPLE_DIR, redraw: bool = False) -> None:
    """Write the tables of levels 1 and coverage and performance, and draw the 2 samples (once)."""
    out.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    print("read the merged spans ...", flush=True)
    raw = load_raw()
    sp = counted(raw)
    lengths = text_lengths()
    grounding(raw).to_csv(out / "grounding.csv", index=False)
    verbatim_summary(raw).to_csv(out / "verbatim_summary.csv", index=False)
    coverage(raw, sp, lengths).to_csv(out / "coverage.csv", index=False)
    position_profile(sp, lengths).to_csv(out / "position_profile.csv", index=False)
    pops = populations(sp)
    pops.to_csv(out / "populations.csv", index=False)
    logs = sorted(LOGS.glob("f9x2-*.out"))
    throughput(logs).to_csv(out / "throughput.csv", index=False)
    job_hours(run_job_ids()).to_csv(out / "job_hours.csv", index=False)
    floor_table().to_csv(out / "floor_table.csv", index=False)
    pfile = sample_dir / "precision_sample.parquet"
    if redraw or not pfile.exists():
        smp = draw_precision_sample(sp, n_per_stratum)
        smp.to_parquet(pfile, index=False)
        precision_sheet(smp).to_csv(sample_dir / "precision_sheet.csv", index=False)
        print(f"drew {len(smp)} spans in {smp.stratum.nunique()} strata -> {pfile}")
    rfile = sample_dir / "recall_sample.parquet"
    if redraw or not rfile.exists():
        draw_recall_sample(raw, lengths).to_parquet(rfile, index=False)
    meta = {"module": "research/notebooks/facct/_xacc.py", "version": __version__, "seed": SEED,
            "n_per_stratum": n_per_stratum, "spans_counted": int(len(sp)), "rows": int(len(raw)),
            "texts": int(raw.doc_id.nunique())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n-per-stratum", type=int, default=30)
    b.add_argument("--redraw", action="store_true", help="draw the samples again (a filled sheet stays)")
    sub.add_parser("verbatim", help="write verbatim_summary.csv only (about 4 minutes)")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.n_per_stratum, redraw=args.redraw)
    elif args.cmd == "verbatim":
        OUT.mkdir(parents=True, exist_ok=True)
        t = verbatim_summary(load_raw())
        t.to_csv(OUT / "verbatim_summary.csv", index=False)
        print(t.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
