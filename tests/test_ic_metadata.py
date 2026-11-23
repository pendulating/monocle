"""Unit tests for `research/notebooks/cvpr/_ic_metadata.py`.

These tests build small frames by hand. They read no corpus, no results
parquet, and no proxy. They cover the image join (the swap), the
same-image consistency count, the partner test, and the band-controlled
agreement contrast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "notebooks" / "cvpr"))

from research.notebooks.cvpr import _ic_metadata as M  # noqa: E402


def _span(case, pair, label, cls, **attrs):
    row = {"case": case, "pair_id": pair, "presented_label": label,
           "extraction_class": cls, "extraction_text": "x", "name": None,
           "image": None, "valence": None, "favors": None, "direction": None}
    row.update(attrs)
    return row


def _meta():
    """2 pairs. p1 is not swapped: the left image is image_path_a (s1). p2
    is swapped: the left image is image_path_b (s3)."""
    return pd.DataFrame({
        "case": ["c", "c"], "pair_id": ["p1", "p2"],
        "canonical_pair_id": ["p1", "p2"],
        "sample_id_a": ["s1", "s2"], "sample_id_b": ["s2", "s3"],
        "image_path_a": ["/a/s1.jpg", "/a/s2.jpg"],
        "image_path_b": ["/a/s2.jpg", "/a/s3.jpg"],
        "presented_left_path": ["/a/s1.jpg", "/a/s3.jpg"],
        "presented_right_path": ["/a/s2.jpg", "/a/s2.jpg"],
        "is_swapped": [False, True],
        "unit_uid_a": ["u1", "u2"], "unit_uid_b": ["u2", "u3"],
        "relative_score": [1.0, -1.0],
    })


# --------------------------------------------------------------------------
# The image join
# --------------------------------------------------------------------------

def test_presented_sides_follow_the_swap():
    meta = M.presented_sides(_meta())
    assert list(meta["canon_of_A"]) == ["a", "b"]
    assert list(meta["sample_A"]) == ["s1", "s3"]
    assert list(meta["sample_B"]) == ["s2", "s2"]
    assert list(meta["unit_A"]) == ["u1", "u3"]


def test_path_wins_over_a_wrong_swap_flag():
    """The path decides. A flag that disagrees with the paths is ignored."""
    raw = _meta()
    raw.loc[1, "is_swapped"] = False
    meta = M.presented_sides(raw)
    assert meta.loc[1, "sample_A"] == "s3"


def test_attach_images_maps_the_span_side():
    meta = M.presented_sides(_meta())
    df = M.prepare(pd.DataFrame([
        _span("c", "p2", "More", "dimension", name="greenery", image="A", valence="good"),
        _span("c", "p2", "More", "dimension", name="trash", image="B", valence="bad"),
        _span("c", "p2", "More", "perspective", name="modern", favors="A"),
        _span("c", "p2", "More", "dimension", name="sky", image="both", valence="neutral"),
    ]))
    out = M.attach_images(df, meta)
    # Image A of the swapped pair p2 is s3, the canonical b side.
    assert list(out["entity_sample"][:3]) == ["s3", "s2", "s3"]
    assert list(out["other_sample"][:3]) == ["s2", "s3", "s2"]
    assert out["entity_sample"].iloc[3] is None
    assert out["winner"].iloc[0] == "A"


# --------------------------------------------------------------------------
# The same-image consistency
# --------------------------------------------------------------------------

def _votes():
    """1 image, 1 name, 3 traces: good, good, bad. 1 more image, 1 name, 2
    traces: bad, bad. A third image in 1 trace only."""
    return pd.DataFrame({
        "case": ["c"] * 6,
        "entity": ["e1", "e1", "e1", "e2", "e2", "e3"],
        "key_name": ["facade"] * 6,
        "tid": ["c|t1", "c|t2", "c|t3", "c|t4", "c|t5", "c|t6"],
        "valence": ["good", "good", "bad", "bad", "bad", "good"],
    })


def test_consistency_counts_groups_and_pairs():
    out = M.consistency(_votes(), by=()).iloc[0]
    assert out["groups"] == 2            # e3 has 1 trace only
    assert out["consistent"] == 1        # e2
    assert out["pair_n"] == 4            # 3 pairs in e1, 1 in e2
    # Agreeing pairs: (t1, t2) in e1 and (t4, t5) in e2.
    assert out["pair_agree"] == pytest.approx(2 / 4)
    # The name is good in 3 of 6 traces, thus the chance level is 0.5.
    assert out["chance_agree"] == pytest.approx(0.5)


def test_trace_valence_drops_a_tie():
    meta = M.presented_sides(_meta())
    df = M.prepare(pd.DataFrame([
        _span("c", "p1", "More", "dimension", name="facade", image="A", valence="good"),
        _span("c", "p1", "More", "dimension", name="facade", image="A", valence="bad"),
        _span("c", "p2", "Less", "dimension", name="facade", image="B", valence="bad"),
    ]))
    out = M.trace_valence(M.attach_images(df, meta), "entity_sample")
    assert len(out) == 1
    assert out.iloc[0]["entity"] == "s2" and out.iloc[0]["valence"] == "bad"


def test_partner_effect_puts_good_in_the_win():
    """s2 sits in 2 pairs. It wins p2 (B wins, B is s2) with `facade` good,
    and it loses p1 (A wins) with `facade` bad."""
    meta = M.presented_sides(_meta())
    df = M.prepare(pd.DataFrame([
        _span("c", "p1", "More", "dimension", name="facade", image="B", valence="bad"),
        _span("c", "p2", "Less", "dimension", name="facade", image="B", valence="good"),
    ]))
    df = M.attach_images(df, meta)
    votes = M.trace_valence(df, "entity_sample")
    out = M.partner_effect(votes, df).iloc[0]
    assert out["win_loss_pairs"] == 1
    assert out["n"] == 1 and out["k"] == 1
    assert out["disagree_win_loss"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The band-controlled agreement
# --------------------------------------------------------------------------

def _pairs(n1_agree, n1, n0_agree, n0, band):
    rows = []
    for i in range(n1):
        rows.append({"cell": "x", "gap_band": band, "feat": True, "agrees": i < n1_agree})
    for i in range(n0):
        rows.append({"cell": "x", "gap_band": band, "feat": False, "agrees": i < n0_agree})
    return rows


def test_banded_contrast_pools_inside_the_bands():
    """Band 0: 8/10 against 5/10 (d = 0.3). Band 1: 6/20 against 3/20
    (d = 0.15). The weights are n1 n0 / (n1 + n0): 5 and 10."""
    frame = pd.DataFrame(_pairs(8, 10, 5, 10, 0) + _pairs(6, 20, 3, 20, 1))
    out = M.banded_contrast(frame, "feat")
    assert out["strata"] == 2
    assert out["diff"] == pytest.approx((5 * 0.3 + 10 * 0.15) / 15)
    assert out["n1"] == 30 and out["n0"] == 30
    assert out["lo"] < out["diff"] < out["hi"]


def test_banded_contrast_removes_a_difficulty_confound():
    """The feature sits mostly in the hard band, where every group agrees
    less. Inside each band the feature changes nothing, thus the pooled
    difference is 0 while the raw difference is negative."""
    frame = pd.DataFrame(_pairs(4, 40, 1, 10, 0) + _pairs(9, 10, 36, 40, 1))
    raw = frame[frame["feat"]]["agrees"].mean() - frame[~frame["feat"]]["agrees"].mean()
    assert raw < -0.2
    out = M.banded_contrast(frame, "feat")
    assert out["diff"] == pytest.approx(0.0, abs=1e-12)


def test_banded_contrast_skips_a_thin_stratum():
    frame = pd.DataFrame(_pairs(2, 3, 1, 3, 0) + _pairs(6, 10, 3, 10, 1))
    out = M.banded_contrast(frame, "feat", min_n=5)
    assert out["strata"] == 1
    assert out["diff"] == pytest.approx(0.3)


def test_banded_contrast_empty_gives_nan():
    frame = pd.DataFrame(_pairs(1, 2, 1, 2, 0))
    out = M.banded_contrast(frame, "feat")
    assert out["strata"] == 0 and np.isnan(out["diff"])
