"""Unit tests for the abstention part of `research/notebooks/cvpr/_trace_channels.py`.

These tests build small frames by hand. They read no corpus and need no GPU.
They cover the trace frame (the reason and marker counts), the AUC of each
count, the dose bins, and the shape of the long frames across arms.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "notebooks" / "cvpr"))

import _trace_channels as TC  # noqa: E402


def _span(case, pair, label, cls, reason=None, marker=None):
    return {"case": case, "pair_id": pair, "presented_label": label,
            "extraction_class": cls, "reason": reason, "marker": marker}


def _corpus(case="libraries", n_abstain=30, n_decide=40, with_marker=True):
    """A synthetic corpus: an abstaining trace has 3 `explicit_doubt` spans
    and 1 `reconsideration` span. A deciding trace has 1 `reconsideration`
    span and 0 or 1 `qualifier` span. Each trace has 1 decision."""
    rows = []
    for i in range(n_abstain):
        pair = f"a{i}"
        rows.append(_span(case, pair, "NotSure", "decision"))
        for _ in range(3):
            rows.append(_span(case, pair, "NotSure", "uncertainty",
                              "cannot_see", "explicit_doubt"))
        rows.append(_span(case, pair, "NotSure", "uncertainty", "other",
                          "reconsideration"))
    for i in range(n_decide):
        pair = f"d{i}"
        rows.append(_span(case, pair, "More", "decision"))
        rows.append(_span(case, pair, "More", "uncertainty", "looks_equal",
                          "reconsideration"))
        if i % 2:
            rows.append(_span(case, pair, "More", "uncertainty", "other",
                              "qualifier"))
    df = pd.DataFrame(rows)
    if not with_marker:
        df = df.drop(columns="marker")
    return df


def test_trace_frame_counts_markers_and_reasons():
    traces = TC.trace_frame(_corpus())
    assert len(traces) == 70
    for marker in TC.MARKER_ORDER:
        assert f"n_marker_{marker}" in traces.columns
    one = traces.set_index("tid").rename(index=lambda t: t.split("|")[1])
    assert one.loc["a0", "n_uncertain"] == 4
    assert one.loc["a0", "n_marker_explicit_doubt"] == 3
    assert one.loc["a0", "n_marker_reconsideration"] == 1
    assert one.loc["a0", "n_cannot_see"] == 3
    assert one.loc["a0", "n_core"] == 3
    assert one.loc["d1", "n_marker_qualifier"] == 1
    assert one.loc["d0", "n_marker_qualifier"] == 0
    assert one.loc["d0", "n_looks_equal"] == 1
    assert one["abstains"].sum() == 30


def test_trace_frame_without_marker_column():
    traces = TC.trace_frame(_corpus(with_marker=False))
    assert not any(c.startswith("n_marker_") for c in traces.columns)
    assert (traces["n_core"] == traces["n_uncertain"]).all()


def test_trace_id_keeps_cases_apart():
    df = pd.concat([_corpus("libraries", 2, 2), _corpus("schools", 2, 2)])
    assert len(TC.trace_frame(df)) == 8


def test_abstention_table_auc_per_marker():
    stats = TC.abstention_table(TC.trace_frame(_corpus()))
    row = stats.iloc[0]
    assert row["n_abstain"] == 30
    assert row["abstain_rate"] == pytest.approx(30 / 70)
    # An abstention always holds more spans: a perfect separation.
    assert row["auc"] == pytest.approx(1.0)
    assert row["auc_marker_explicit_doubt"] == pytest.approx(1.0)
    assert row["auc_cannot_see"] == pytest.approx(1.0)
    # Every trace holds 1 reconsideration span: a full tie reads 0.5.
    assert row["auc_marker_reconsideration"] == pytest.approx(0.5)
    # Only a deciding trace holds a qualifier: the AUC is below 0.5.
    assert row["auc_marker_qualifier"] < 0.5
    assert row["auc_looks_equal"] == pytest.approx(0.0)
    # A marker no row carries gives a tie, not an error.
    assert row["auc_marker_contrastive"] == pytest.approx(0.5)


def test_abstention_table_nan_without_markers_or_floor():
    stats = TC.abstention_table(TC.trace_frame(_corpus(with_marker=False)))
    assert np.isnan(stats.loc[0, "auc_marker_qualifier"])
    assert stats.loc[0, "auc"] == pytest.approx(1.0)
    few = TC.abstention_table(TC.trace_frame(_corpus(n_abstain=5)),
                              min_abstain=25)
    assert np.isnan(few.loc[0, "auc"])
    assert np.isnan(few.loc[0, "auc_marker_explicit_doubt"])
    assert few.loc[0, "n_abstain"] == 5


def test_auc_long_names_each_score():
    long = TC.auc_long(TC.abstention_table(TC.trace_frame(_corpus())))
    scores = set(long["score"])
    assert {"all", "core", "cannot_see", "marker_explicit_doubt"} <= scores
    assert long.loc[long["score"] == "all", "auc"].iloc[0] == pytest.approx(1.0)


def test_dose_fixed_and_quantile_bins():
    traces = TC.trace_frame(_corpus())
    fixed = TC.abstention_dose(traces, min_bin=1)
    assert list(fixed["bin"]) == list(TC.DOSE_LABELS)
    assert fixed["traces"].sum() == 70
    quant = TC.abstention_dose(traces, min_bin=1, quantiles=5)
    assert quant["bin"].str.match(r"^Q\d$").all()
    assert quant["traces"].sum() == 70
    assert quant["bin_range"].notna().all()


@pytest.fixture
def fake_arms(monkeypatch):
    """4 arms of 2 judges. `X.load` gives a synthetic corpus for each root."""
    specs = {
        "gemma_by_qwen27_v3": ("gemma-4-12b", "qwen3.8-27b", 30),
        "gemma_by_gemma31_v3": ("gemma-4-12b", "gemma-4-31b", 30),
        "qwen_by_gemma31_v3": ("qwen3.5-9b", "gemma-4-31b", 26),
        "qwen_by_qwen27_v3": ("qwen3.5-9b", "qwen3.8-27b", 26),
    }

    def arm_roots(arms=None, present_only=True):
        names = list(arms) if arms else list(specs)
        return {n: {"root": f"/fake/{n}", "judge": specs[n][0],
                    "extractor": specs[n][1], "schema_version": "v3"}
                for n in names if n in specs}

    def load(root=None, require_canonical=True, **_):
        name = Path(root).name
        n_abstain = specs[name][2]
        return pd.concat([_corpus("libraries", n_abstain, 40),
                          _corpus("schools", n_abstain, 40)],
                         ignore_index=True)

    monkeypatch.setattr(TC.X, "arm_roots", arm_roots)
    monkeypatch.setattr(TC.X, "load", load)
    return specs


def test_abstention_across_arms_long_shape(fake_arms):
    stats, dose = TC.abstention_across_arms(arms=list(fake_arms))
    assert list(stats.columns[:5]) == list(TC.ARM_COLUMNS)
    assert len(stats) == 4 * 2
    assert set(stats["arm"]) == set(fake_arms)
    assert set(stats.loc[stats["role"] == "primary", "arm"]) == {
        "gemma_by_qwen27_v3", "qwen_by_gemma31_v3"}
    assert "auc_marker_explicit_doubt" in stats.columns
    assert list(dose.columns[:5]) == list(TC.ARM_COLUMNS)
    assert dose.groupby(["arm", "case"])["traces"].sum().eq(
        pd.Series({k: 30 + 40 if k[0].startswith("gemma") else 26 + 40
                   for k in dose.groupby(["arm", "case"]).groups})).all()
    table = TC.auc_by_arm(stats)
    assert list(table.columns) == ["case_label", "gemma_by_qwen27_v3",
                                   "gemma_by_gemma31_v3", "qwen_by_gemma31_v3",
                                   "qwen_by_qwen27_v3"]


def test_abstention_across_arms_default_is_v3(fake_arms, monkeypatch):
    seen = {}

    def arm_roots(arms=None, present_only=True):
        seen["arms"] = list(arms)
        return {}

    monkeypatch.setattr(TC.X, "arm_roots", arm_roots)
    stats, dose = TC.abstention_across_arms()
    assert stats.empty and dose.empty
    assert seen["arms"] == [n for n, s in TC.X.ARMS.items()
                            if s["schema_version"] == "v3"]
    assert len(seen["arms"]) == 4


def test_plot_abstention_arms_draws(fake_arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats, dose = TC.abstention_across_arms(arms=list(fake_arms))
    fig = TC.plot_abstention_arms(dose, stats)
    assert len(fig.axes) == 2
    plt.close(fig)
