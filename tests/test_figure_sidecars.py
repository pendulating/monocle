"""Tests for the caption sidecars of the version 3 figures.

A figure carries no caption prose. Its caption goes into a `.json` file
beside the image (`research/notebooks/cvpr/_figmeta.py`). These tests build small
frames by hand. They read no corpus and need no GPU.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "notebooks" / "cvpr"))

from research.notebooks.cvpr import _figmeta as F  # noqa: E402
from research.notebooks.cvpr import _gold_alignment as G  # noqa: E402
from research.notebooks.cvpr import _ic_metadata as M  # noqa: E402
from research.notebooks.cvpr import _ic_v3 as I  # noqa: E402
from research.notebooks.cvpr import _trace_channels as TC  # noqa: E402

ARM = "gemma_by_qwen27_v3"


def _no_footer(fig, max_label_chars: int = 80) -> None:
    """The figure holds no caption prose: no free figure text, and a
    shared x label (if any) that is a short axis label, not a sentence."""
    sup = getattr(fig, "_supxlabel", None)
    free = [t for t in fig.texts if t is not sup]
    assert not free, [t.get_text() for t in free]
    if sup is not None:
        text = sup.get_text()
        assert len(text) <= max_label_chars, text
        assert ". " not in text and "\n" not in text, text
    assert getattr(fig, "_suptitle", None) is None


def _check_meta(fig, label_start: str) -> dict:
    meta = F.meta_of(fig)
    assert meta is not None
    for key in F.REQUIRED_KEYS:
        assert key in meta
    assert meta["plot-caption"].strip()
    assert meta["plot-latex-label"].startswith(label_start)
    assert "v3" in meta["plot-tags"]
    return meta


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------

def test_save_figure_writes_the_4_keys(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    side = F.save_figure(fig, tmp_path / "a" / "line.png", title="A line",
                         caption="The line runs from 0 to 1; $x = y$.",
                         label="fig:a-line", tags=["test", "v3", "test"],
                         dpi=50, provenance={"module": "test"})
    plt.close(fig)
    assert (tmp_path / "a" / "line.png").exists()
    assert side == tmp_path / "a" / "line.json"
    assert side.parent == (tmp_path / "a" / "line.png").parent
    raw = side.read_text(encoding="utf-8")
    assert raw.startswith("{\n  \"")          # indent=2
    data = json.loads(raw)
    assert list(data)[:4] == list(F.REQUIRED_KEYS)
    assert data["plot-title"] == "A line"
    assert data["plot-caption"] == "The line runs from 0 to 1; $x = y$."
    assert data["plot-latex-label"] == "fig:a-line"
    assert data["plot-tags"] == ["test", "v3"]
    assert data["provenance"]["module"] == "test"
    assert data["provenance"]["file"] == "line.png"


def test_save_figure_takes_the_attached_caption(tmp_path):
    fig, _ = plt.subplots()
    F.attach(fig, F.figure_meta("T", "C ünïcode.", "fig:t", ["x"], arm=None))
    side = F.save_figure(fig, tmp_path / "t.pdf")
    plt.close(fig)
    data = F.read_sidecar(tmp_path / "t.pdf")
    assert side.name == "t.json"
    assert data["plot-caption"] == "C ünïcode."
    assert "ünïcode" in side.read_text(encoding="utf-8")


def test_save_figure_refuses_a_figure_with_no_caption(tmp_path):
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        F.save_figure(fig, tmp_path / "x.png")
    plt.close(fig)
    assert not (tmp_path / "x.png").exists()


@pytest.mark.parametrize("label", ["tab:x", "fig:Bad_Label", "fig:", "x"])
def test_figure_meta_refuses_a_bad_label(label):
    with pytest.raises(ValueError):
        F.figure_meta("T", "C", label, [])


def test_fig_label_is_kebab_case():
    assert F.fig_label("ic", "perspective-ends", ARM) == \
        "fig:ic-perspective-ends-gemma-by-qwen27-v3"
    assert F.fig_label("ic", "nesting", None) == "fig:ic-nesting"


def test_caption_md_names_title_and_label():
    meta = F.figure_meta("T", "The caption.", "fig:t", [])
    assert F.caption_md(meta) == "**T.** The caption. (`fig:t`)"


# --------------------------------------------------------------------------
# The plot functions draw no caption
# --------------------------------------------------------------------------

def _uncertainty_corpus():
    rows = []
    for case, n_abs in (("libraries", 30), ("schools", 0)):
        for i in range(n_abs):
            for _ in range(4):
                rows.append({"case": case, "pair_id": f"a{i}",
                             "presented_label": "NotSure",
                             "extraction_class": "uncertainty",
                             "reason": "cannot_see", "marker": "explicit_doubt"})
        for i in range(40):
            rows.append({"case": case, "pair_id": f"d{i}",
                         "presented_label": "More",
                         "extraction_class": "uncertainty",
                         "reason": "other", "marker": "qualifier"})
            rows.append({"case": case, "pair_id": f"d{i}",
                         "presented_label": "More",
                         "extraction_class": "decision",
                         "reason": None, "marker": None})
    return pd.DataFrame(rows)


def test_plot_abstention_has_no_footer():
    traces = TC.trace_frame(_uncertainty_corpus())
    stats, dose = TC.abstention_table(traces), TC.abstention_dose(traces)
    fig = TC.plot_abstention(dose, stats, arm=ARM)
    _no_footer(fig)
    meta = _check_meta(fig, "fig:abstention-dose")
    # The removed footer named the case with no abstention: the caption
    # carries it now.
    assert "never abstained" in meta["plot-caption"]
    assert ARM in meta["plot-tags"]
    plt.close(fig)


def test_plot_reasons_has_no_footer():
    fig = TC.plot_reasons(TC.reason_mix(_uncertainty_corpus()), arm=ARM)
    _no_footer(fig)
    _check_meta(fig, "fig:abstention-reasons")
    plt.close(fig)


def _axis_table():
    rng = np.random.default_rng(0)
    n = 6
    agree = np.linspace(0.1, 0.9, n)
    return pd.DataFrame({
        "axis": [f"axis {i}" for i in range(n)], "n": rng.integers(400, 900, n),
        "n_cases": 7, "agree": agree, "ci_low": agree - 0.03,
        "ci_high": agree + 0.03, "hedge": rng.uniform(0, 0.3, n),
        "agree_plain": agree + 0.02, "agree_hedged": agree - 0.05,
        "bind": np.abs(agree - 0.5) * 2,
        "polarity": np.where(agree < 0.5, "negative", "positive")})


def test_plot_axes_and_hedge_have_no_footer():
    table = _axis_table()
    for fig, label in ((TC.plot_axes(table, arm=ARM), "fig:axes-binding"),
                       (TC.plot_axis_hedge(table, {"agree_plain": 0.6,
                                                   "agree_hedged": 0.5},
                                           arm=ARM), "fig:axes-hedge")):
        _no_footer(fig)
        _check_meta(fig, label)
        plt.close(fig)


def _lift_tables():
    rows = []
    for case in ("libraries", "schools", "subway_safety"):
        for j, key in enumerate(("wealth", "class", "crime")):
            rows.append({"case": case, "key": key, "n": 50 + j, "wins": 30,
                         "lift": 0.05 * (j - 1), "base_rate": 0.5})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("name,label", [("inference", "fig:inference-kind-lift"),
                                        ("artifact", "fig:artifact-kind-lift"),
                                        ("inference_pole", "fig:inference-pole-lift")])
def test_plot_kind_ladder_has_no_footer(name, label):
    tables = _lift_tables()
    fig = TC.plot_kind_ladder(tables, TC.kind_summary(tables), name=name,
                              note="An extra note.", legend_loc="below",
                              arm=ARM)
    _no_footer(fig)
    meta = _check_meta(fig, label)
    assert "An extra note." in meta["plot-caption"]
    plt.close(fig)


def _ends():
    rows = []
    for case in ("libraries", "schools"):
        for end, sign in (("top", 1), ("bottom", -1)):
            for r in range(2):
                lift = sign * 0.05 * (r + 1)
                rows.append({"case": case, "key": f"k {end} {r}", "n": 40,
                             "lift": lift, "win_rate": 0.5 + lift,
                             "base_rate": 0.5, "ci_low": 0.4 + lift,
                             "ci_high": 0.6 + lift, "ci_lift_low": lift - 0.1,
                             "ci_lift_high": lift + 0.1, "end": end,
                             "rank": r})
    return pd.DataFrame(rows)


def test_plot_case_ends_keeps_only_an_axis_label():
    fig = I.plot_case_ends(_ends(), "perspective", arm=ARM)
    _no_footer(fig)
    meta = _check_meta(fig, "fig:ic-perspective-ends-gemma-by-qwen27-v3")
    assert "Wilson" in meta["plot-caption"]
    assert "ic_perspectives" == meta["plot-tags"][0]
    plt.close(fig)


def test_plot_consistent_carries_the_ic_caption():
    tables = _lift_tables()
    held = TC.kind_summary(tables)
    fig = I.plot_consistent(tables, held, "perspective", arm=ARM)
    _no_footer(fig)
    _check_meta(fig, "fig:ic-perspective-consistent-gemma-by-qwen27-v3")
    plt.close(fig)


def test_plot_top_targets_keeps_only_an_axis_label():
    targets = pd.DataFrame({
        "case": ["libraries"] * 3 + ["schools"] * 3,
        "target_key": ["a", "b", "c", "d", "e", "f"],
        "n": [9, 5, 3, 8, 4, 3],
        "about_mode": ["premise", "cue", "perspective"] * 2})
    fig = I.plot_top_targets(targets, arm=ARM)
    _no_footer(fig)
    assert fig.axes[-1].get_xlabel() == "dismissals"
    _check_meta(fig, "fig:ic-dismissal-targets-gemma-by-qwen27-v3")
    plt.close(fig)


def test_writer_saves_png_and_sidecar(tmp_path):
    w = I._Writer(tmp_path, "ic_test")
    w.fig(lambda: I.plot_case_ends(_ends(), "inference", arm=ARM), "ends")
    assert w.written == ["ic_test_ends.png", "ic_test_ends.json"]
    data = json.loads((tmp_path / "ic_test_ends.json").read_text())
    assert set(F.REQUIRED_KEYS) <= set(data)
    assert data["provenance"]["module"] == "_ic_v3"


def test_plot_contrasts_has_no_footer():
    table = pd.DataFrame({"feature": ["costly", "hedged_any"],
                          "group": ["income", "unit"], "diff": [0.02, -0.01],
                          "lo": [0.0, -0.03], "hi": [0.04, 0.01]})
    fig = M.plot_contrasts(table, arm=ARM)
    _no_footer(fig)
    _check_meta(fig, "fig:ic-meta-proxy-contrasts-gemma-by-qwen27-v3")
    plt.close(fig)


def test_plot_char_prf_has_no_footer():
    prf = pd.DataFrame({"task_id": [1, 1, 2, 2], "annotation_id": [1, 1, 2, 2],
                        "arm": [ARM] * 4,
                        "cls": ["dimension", "inference"] * 2,
                        "recall": [0.5, 0.4, 0.6, 0.3],
                        "precision": [0.7, 0.2, 0.5, 0.4]})
    fig = G.plot_char_prf(prf, [ARM])
    _no_footer(fig)
    meta = _check_meta(fig, "fig:gold-char-agreement")
    assert "2 traces" in meta["plot-caption"]
    plt.close(fig)
