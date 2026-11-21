import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F9 — how accurate is the span extraction?

    DeepSeek-V4-Flash (langextract) reads 400,000 texts of the F9 round:
    the F9 answers and the F4 answers of the same images, under 1 schema for
    each case (D142). The H9 test counts the `visual_evidence` spans and
    their valence. This notebook asks how many spans of each class are
    accurately extracted, and gives a **floor**: a number that the true
    count is at or above, with probability 0.95.

    | Level | Question | Source | Kind of number | Section |
    |-------|----------|--------|----------------|---------|
    | 1. Grounding | Is the span text in the source text? | the aligner, every span | exact | 1 |
    | 2. Class | Is the span of the class the extractor gave? | a person labels a random sample | a floor | 5 |
    | 3. Class and attribute | Is the key attribute right too (the valence of `visual_evidence`)? | the same labels | a floor | 5 |
    | Recall | Did the extractor find the spans that a person marks? | a person marks every span of a text | a floor | 6 |

    **The floor.** For a stratum with `k` correct spans of `n` labelled
    spans, the floor is the one-sided Clopper-Pearson lower bound (alpha
    0.05). The floor times the count of spans in the corpus is the floor
    count. The floor of a class pools its strata with the corpus weights,
    with alpha / (strata) for each stratum (Bonferroni), thus the pooled
    floor holds with probability 0.95 too.

    **Status (2026-10-30).** Levels 1, coverage and performance are done.
    The precision sample is drawn (1,020 spans, 30 in each of 34 strata),
    but no label exists yet; sections 5 and 6 show the floors when the
    labels arrive.

    The tables come from `python notebooks/facct/_xacc.py build`
    (`figures/f9/xacc/`); the label sheets are in `data/f9_xacc/`.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo
    import pandas as pd

    import _f8 as F8
    import _xacc as XA

    NOTEBOOK_VERSION = "1.0.0"
    return F8, XA, mo, pd


@app.cell
def _(XA):
    T = XA.load_tables()
    return (T,)


@app.cell(hide_code=True)
def _(F8, T, XA, mo):
    _g = T["grounding"]
    _w = XA.grounding_wide(_g).round(3)
    _n = _g.groupby(["extraction_class", "case", "prompt"]).spans.sum().unstack(["case", "prompt"])
    _n.columns = [f"{c}, {p.upper()}" for c, p in _n.columns]
    _n = _n.reindex(_w.index).fillna(0).astype(int).map(lambda v: f"{v:,}")
    _q = XA.grounding_wide(_g, "quotable").round(3)
    _fig = XA.plot_grounding(_g)
    F8.save(_fig, "xacc_grounding", XA.OUT)
    mo.vstack([
        mo.md(
            "## 1. Grounding: is the span in the text?\n\n"
            "The merge compares each span with the slice of the source text at its offsets. A **verbatim** "
            "span equals the slice, character for character. This is a check of every span, not an "
            "estimate. The analysis counts the **quotable** spans: verbatim, a near match of the aligner, "
            "or a span that the rescue pass found (markup, case).\n\n"
            "### 1.1 The share of verbatim spans\n\n" + F8.md_table(_w) +
            "\n\n### 1.2 The share of quotable spans\n\n" + F8.md_table(_q) +
            "\n\n### 1.3 The spans of each class (both fields)\n\n" + F8.md_table(_n)
        ),
        _fig,
        mo.md(
            "**Reading.** 89.5% to 99.8% of the spans of each class are verbatim, and 98.6% or more are "
            "quotable. The weakest classes are the F9 street `photo_advice` (0.895) and `verdict` "
            "(0.928) spans. About 95% of their non-verbatim spans are spans where the extractor dropped "
            "the markdown of the source (`**Watch the Lighting:**` became `Watch the Lighting:`), and the "
            "rescue pass found them again (`alignment_rescue` = `markup`). A non-verbatim span is still a "
            "statement of the text, but do not quote it in the paper (`is_verbatim`)."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(F8, T, XA, mo):
    _c = T["coverage"].copy()
    _c["model"] = _c.model.str.replace("-instruct", "")
    _show = _c[["case", "prompt", "model", "field", "texts", "copy_suspects", "chars_p50", "share_with_span",
                "spans_per_text", "char_share_mean", "char_share_p10", "second_half_share_long"]].round(3)
    _fig = XA.plot_position(T["position_profile"])
    F8.save(_fig, "xacc_position", XA.OUT)
    mo.vstack([
        mo.md(
            "## 2. Coverage: how much of each text do the spans read?\n\n"
            "| Column | Meaning |\n|--------|---------|\n"
            "| `share_with_span` | texts with 1 counted span or more |\n"
            "| `char_share_mean`, `char_share_p10` | the share of the characters of a text inside a span: mean, 10th percentile |\n"
            "| `second_half_share_long` | the share of the spans of the long texts (4,000 characters or more) that start in the second half |\n\n"
            "Coverage is not recall: a text holds words that belong to no class. But a text with a low "
            "coverage, or a long text whose spans stop in the middle, is a text the extractor did not read "
            "to the end. The spans cover 29% to 56% of a reasoning trace and 64% to 92% of a response; "
            "the long reasoning traces have 36% to 52% of their spans in the second half."
        ),
        mo.ui.table(_show, selection=None, page_size=40),
        _fig,
        mo.md(
            "**Reading.** The span density does not fall to 0 at the end of the long traces: in the last "
            "tenth it is 47% to 72% of its peak. Thus the extractor reads them to the end. The F9 street "
            "reasoning of the 2 gemma models has the most texts with no counted span (8.7% and 5.9%) and "
            "the most copy suspects (110 and 55). Section 5 does not yet sample those texts apart."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(F8, T, mo):
    _t = T["throughput"]
    _j = T["job_hours"]
    _tot = _t[["batches", "prompts", "generate_s", "steps_done", "steps_failed", "preempted"]].sum()
    _hours = _j.groupby("state").hours.agg(["size", "sum"]).round(2) if "hours" in _j else _j
    _rate = _tot.prompts / _tot.generate_s
    mo.vstack([
        mo.md(
            "## 3. Performance: what the run cost\n\n"
            "1 prompt is 1 chunk of 1 text (langextract cuts a long text into chunks). The `generate` time "
            "is the time inside the engine, from the batch lines of the logs; the job hours come from "
            "`sacct`, 1 row for each allocation (a preempted allocation and its requeue are 2 rows).\n\n"
            f"| Item | Value |\n|------|-------|\n"
            f"| Prompts | {int(_tot.prompts):,} in {int(_tot.batches):,} batches |\n"
            f"| Time in `generate` | {_tot.generate_s / 3600:.1f} GPU hours |\n"
            f"| Throughput | {_rate:.2f} prompts per second on 1 B300 |\n"
            f"| Steps done, failed | {int(_tot.steps_done)}, {int(_tot.steps_failed)} |\n"
            f"| Preemptions (QoS `g3`) | {int(_tot.preempted)} |\n"
            "| Retry pass | 964 restaurant and 2,876 street copy suspects read again (pass 2) |\n"
            "| Wall time | smoke 15 min; pass 1 about 6 h 15 min on up to 9 GPUs; pass 2 33 min |\n\n"
            "### 3.1 The allocations, by state\n\n" + F8.md_table(_hours)
        ),
        mo.ui.table(_t.round(3), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(F8, T, mo):
    _f = T["floor_table"].set_index("labels").round(3)
    mo.md(
        "## 4. The floor: how many labels a guarantee needs\n\n"
        "The one-sided 95% Clopper-Pearson lower bound of the precision, for `n` labelled spans with 0, 1 "
        "or 2 errors, or with 5% errors. Read a row as: \"with `n` labels and this many errors, the true "
        "precision is at or above this value, with probability 0.95\".\n\n" + F8.md_table(_f) +
        "\n\n- 30 labels can show at most a floor of 0.905 (0 errors). The sample uses 30 for each "
        "stratum, thus each stratum can show a floor of about 0.9.\n"
        "- A floor of 0.95 needs about 60 labels with 0 errors, about 100 with 1 error, and about 130 "
        "with 2.\n"
        "- A pooled class floor uses alpha / (strata), thus it is a little lower than the floor of a "
        "single stratum with the same labels."
    )
    return


@app.cell(hide_code=True)
def _(F8, T, XA, mo, pd):
    _pop = T["populations"].set_index("stratum").spans
    _smp = pd.read_parquet(XA.SAMPLE_DIR / "precision_sample.parquet")
    _drawn = _smp.assign(stratum=XA.strata(_smp)).groupby("stratum").size()
    _tab = pd.DataFrame({"spans in the corpus": _pop, "drawn": _drawn}).dropna(subset=["drawn"])
    _tab = _tab.astype(int).map(lambda v: f"{v:,}")
    LABELS = XA.load_labels()
    _status = ("**No label exists yet.**" if LABELS is None else
               f"**{len(LABELS):,} of {len(_smp):,} spans carry a label.**")
    mo.md(
        "## 5. Precision: the labelled sample\n\n"
        "The sample draws 30 counted spans at random from each stratum: case x prompt x class, with "
        "the `visual_evidence` spans split by valence (good, bad, neutral), because the H9 test reads the "
        "valence and good and bad spans are rare in the F4 restaurant answers. The rows of the sheet come "
        "in a random order, and the sheet does not name the model or the prompt group.\n\n"
        f"{_status}\n\n"
        "| File | Use |\n|------|-----|\n"
        "| `data/f9_xacc/precision_sheet.csv` | the sheet to fill: 1 span, its context (the span in `[[ ]]`), the question, the class and the key attribute |\n"
        "| `data/f9_xacc/precision_labels.csv` | the filled sheet (the same columns); this notebook reads it |\n"
        "| `data/f9_xacc/precision_sample.parquet` | the sample with the model, the prompt and the offsets |\n\n"
        "| Label | Value |\n|-------|-------|\n"
        "| `class_ok` | 1 when the span is of the class given, else 0 |\n"
        "| `attribute_ok` | 1 when the key attribute is right too (valence, kind, reason, label, used_in_judgment), else 0; blank when `class_ok` is 0 |\n"
        "| `bounds_ok` | 1 when the span holds the whole claim and no more than 1 claim, else 0 |\n"
        "| `true_class`, `true_attribute`, `note` | the right values when the given ones are wrong; free text |\n\n"
        + F8.md_table(_tab)
    )
    return (LABELS,)


@app.cell(hide_code=True)
def _(F8, LABELS, T, XA, mo):
    if LABELS is None:
        _out = mo.md(
            "## 6. The floors\n\n_The floors appear here when `data/f9_xacc/precision_labels.csv` holds "
            "labels. A partial sheet works: the notebook uses the rows with `class_ok` set._"
        )
    else:
        FLOORS = XA.precision_floors(LABELS, T["populations"])
        _c = FLOORS["classes"].copy()
        _c["floor count"] = _c.floor_count.map(lambda v: f"{v:,}")
        _c["spans"] = _c.spans.map(lambda v: f"{v:,}")
        _c = _c[["case", "prompt", "class", "measure", "labels", "spans", "share", "floor", "floor count"]].round(3)
        _ve = FLOORS["strata"]
        _ve = _ve[_ve["class"] == "visual_evidence"][["stratum", "measure", "labels", "correct", "share",
                                                       "floor", "spans", "floor_count"]].round(3)
        _out = mo.vstack([
            mo.md(
                "## 6. The floors\n\n"
                "`share`: the weighted share correct. `floor`: the pooled 95% floor. `floor count`: the "
                "floor times the spans of the class in the corpus, thus \"at least this many spans of the "
                "class are correct\". `class`: the class only; `class_attr`: the class and the key "
                "attribute.\n\n" + F8.md_table(_c.set_index("case")) +
                "\n\n### 6.1 The `visual_evidence` strata (the H9 outcome)\n\n"
                "A valence error that differs between F4 and F9 would bias `d`. Compare the 2 prompts in "
                "each valence stratum.\n\n" + F8.md_table(_ve.set_index("stratum"))
            ),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(XA, mo, pd):
    _r = pd.read_parquet(XA.SAMPLE_DIR / "recall_sample.parquet")
    mo.md(
        "## 7. Recall: the next step\n\n"
        "Precision says how many of the extracted spans are right. Recall says how many of the true spans "
        "the extractor found; it needs a person to mark every span of a text, as the gold set of the "
        "pairwise traces does (`langextract-trace-extraction.md`, \"The gold set\"; `_gold_alignment.py`, "
        "which measures character overlap because people mark whole clauses and nested spans).\n\n"
        f"`data/f9_xacc/recall_sample.parquet` holds {len(_r)} texts: 1 for each (case, prompt, model, "
        f"field), each 6,000 characters or fewer (median {int(_r.text.str.len().median()):,}). With "
        "Label Studio (`scripts/build_gold_set.py`, `scripts/label_studio_setup.py`) a person marks the "
        "spans of these texts with the classes of the F9 schema. Then the recall floor of a class is the "
        "Clopper-Pearson floor of the share of the human spans that a same-class extracted span covers by "
        "50% or more of its characters."
    )
    return


if __name__ == "__main__":
    app.run()
