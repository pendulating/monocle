"""Tests for `notebooks/cvpr/_judge_vocab.py` on small synthetic frames."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "cvpr"))

import _judge_vocab as J  # noqa: E402
import _latex as L  # noqa: E402


def _spans(rows):
    return pd.DataFrame(rows, columns=["case", "pair_id", "extraction_class", "name"])


def test_normaliser_folds_variants():
    keys = J.normalise_keys(pd.Series(
        ["Well-Maintained", "well maintained", "WELL MAINTAINED.",
         "Trees", "tree", "upkeep", None]))
    assert keys.iloc[0] == keys.iloc[1] == keys.iloc[2] == "well maintained"
    assert keys.iloc[3] == keys.iloc[4] == "tree"
    # No synonym list: upkeep stays its own key.
    assert keys.iloc[5] == "upkeep"
    assert keys.iloc[6] is None or pd.isna(keys.iloc[6])


def test_doc_frequency_counts_a_trace_once():
    spans = J.slim_spans(_spans([
        ("parks", "p1", "perspective", "Maintenance"),
        ("parks", "p1", "perspective", "maintenance"),
        ("parks", "p1", "perspective", "maintenance"),
        ("parks", "p2", "perspective", "maintenance"),
        ("parks", "p2", "decision", "ignored"),
    ]))
    docs = J.doc_counts(spans, pd.Series({"parks": 4}))
    row = docs.set_index("key").loc["maintenance"]
    assert row["docs"] == 2
    assert row["df"] == 0.5
    assert set(docs["extraction_class"]) == {"perspective"}


def test_log_odds_sign_and_zero():
    # Equal use gives z = 0.
    _, _, z = J.log_odds_z([10, 20, 30], [10, 20, 30])
    assert np.allclose(z, 0.0)
    # Equal shares under different volumes also give z = 0.
    _, _, z = J.log_odds_z([10, 20, 30], [20, 40, 60])
    assert np.allclose(z, 0.0)
    # Group A uses key 0 more: z > 0 there, < 0 on the key B uses more.
    _, _, z = J.log_odds_z([50, 10, 30], [10, 50, 30])
    assert z[0] > 0 and z[1] < 0
    # A key that neither group uses gives NaN.
    _, _, z = J.log_odds_z([5, 0], [5, 0])
    assert np.isnan(z[1])


def _scores(z_q27, z_g31, docs=50):
    rows = []
    for ex, z in (("qwen3.8-27b", z_q27), ("gemma-4-31b", z_g31)):
        if z is None:
            continue
        rows.append({"case": "parks", "extraction_class": "perspective",
                     "extractor": ex, "key": "k", "z": z,
                     "df_gemma": 0.5 if z > 0 else 0.1,
                     "df_qwen": 0.1 if z > 0 else 0.5,
                     "docs_pooled": docs, "docs_gemma": docs // 2,
                     "docs_qwen": docs // 2})
    return pd.DataFrame(rows)


def test_robust_flag_needs_both_extractors():
    r = J.robust_table(_scores(3.0, 2.5))
    assert bool(r["robust"].iloc[0]) and r["side"].iloc[0] == "gemma-4-12b"
    assert bool(r["rate_agrees"].iloc[0])
    # Significant under 1 extractor only.
    assert not bool(J.robust_table(_scores(3.0, 1.0))["robust"].iloc[0])
    # Opposite signs.
    assert not bool(J.robust_table(_scores(3.0, -3.0))["robust"].iloc[0])
    # Seen under 1 extractor only.
    assert not bool(J.robust_table(_scores(3.0, None))["robust"].iloc[0])
    # Under the floor.
    assert not bool(J.robust_table(_scores(3.0, 2.5, docs=4),
                                   min_traces=10)["robust"].iloc[0])
    # A qwen key.
    r = J.robust_table(_scores(-4.0, -2.0))
    assert bool(r["robust"].iloc[0]) and r["side"].iloc[0] == "qwen3.5-9b"


def _docs_grid():
    """2 cases, 1 class, 2 extractors, 2 judges, 100 traces each."""
    rng = np.random.default_rng(0)
    rows = []
    for case in ("parks_plazas", "libraries"):
        for ex in J.EXTRACTORS:
            for judge in J.JUDGES:
                for i in range(30):
                    key = f"k{i}"
                    base = 40
                    if key == "k0":
                        base = 90 if judge == "gemma-4-12b" else 10
                    if key == "k1":
                        base = 10 if judge == "gemma-4-12b" else 90
                    docs = int(np.clip(base + rng.integers(-3, 4), 0, 100))
                    rows.append({"case": case, "extraction_class": "perspective",
                                 "key": key, "docs": docs, "n_traces": 100.0,
                                 "df": docs / 100, "judge": judge,
                                 "extractor": ex, "arm": f"{judge}|{ex}"})
    return pd.DataFrame(rows)


def test_bundle_tables_and_latex():
    b = J.build_bundle(_docs_grid(), min_traces=10, min_cases=2)
    r = b["robust"].set_index(["case", "key"])
    assert bool(r.loc[("libraries", "k0"), "robust"])
    assert r.loc[("libraries", "k0"), "side"] == "gemma-4-12b"
    assert r.loc[("libraries", "k1"), "side"] == "qwen3.5-9b"
    cc = b["cross_case"].set_index(["judge", "key"])
    assert cc.loc[("gemma-4-12b", "k0"), "n_cases"] == 2
    assert cc.loc[("gemma-4-12b", "k0"), "scope"] == "general"
    assert (b["agreement"]["n_both"] == 30).all()
    for tex in (J.latex_keys(b["robust"], "perspective"),
                J.latex_agreement(b["agreement"])):
        assert L.check_table(tex) == []
