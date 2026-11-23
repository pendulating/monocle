"""Tests for the 1-cell deletion statistic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))

from research.scripts import k1_separation as k1  # noqa: E402


def frame(truth, extra=None, layer=18, ids=None):
    n = len(truth)
    ids = ids or [f"img{i}" for i in range(n)]
    rows = []
    for i in range(n):
        r = {"layer": layer, "target": ids[i], "causal": list(truth[i])}
        for name, vals in (extra or {}).items():
            r[name] = list(vals[i])
        rows.append(r)
    return pd.DataFrame(rows)


def test_perfect_ranking_gives_the_ceiling():
    t = np.array([-2.0, 0.0, 1.0, 5.0])
    got = k1.sep_z(t, t)
    want = (t.max() - t.min()) / np.std(t, ddof=1)
    assert got == pytest.approx(want)


def test_reversed_ranking_flips_the_sign():
    t = np.array([-2.0, 0.0, 1.0, 5.0])
    assert k1.sep_z(-t, t) == pytest.approx(-k1.sep_z(t, t))


def test_a_flat_ranking_reads_the_first_cell_twice():
    """argmax and argmin both give index 0, so the separation is 0."""
    t = np.array([1.0, 2.0, 3.0])
    assert k1.sep_z(np.zeros(3), t) == pytest.approx(0.0)


def test_a_flat_truth_gives_nan():
    t = np.ones(5)
    assert np.isnan(k1.sep_z(np.arange(5.0), t))


def test_scale_of_the_ranking_does_not_matter():
    rng = np.random.default_rng(0)
    t = rng.normal(size=32)
    v = rng.normal(size=32)
    assert k1.sep_z(v, t) == pytest.approx(k1.sep_z(7.5 * v, t))


def test_table_names_the_instrument_columns():
    truth = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 1.0, 2.0, 3.0])]
    good = [t.copy() for t in truth]
    tab, cols = k1.table(frame(truth, {"good": good}), 18)
    assert cols == ["good"]
    assert list(tab.index) == ["img0", "img1"]
    assert np.allclose(tab["good"], tab["ceiling"])


def test_table_keeps_only_the_asked_layer():
    truth = [np.array([1.0, 2.0, 3.0, 4.0])]
    a = frame(truth, layer=18)
    b = frame(truth, layer=24, ids=["img0"])
    tab, _ = k1.table(pd.concat([a, b]), 24)
    assert len(tab) == 1


def test_a_duplicate_image_is_counted_once():
    truth = [np.array([1.0, 2.0, 3.0, 4.0])] * 2
    tab, _ = k1.table(frame(truth, ids=["img0", "img0"]), 18)
    assert len(tab) == 1


def test_the_shuffle_null_sits_at_zero():
    rng = np.random.default_rng(1)
    truth = [rng.normal(size=64) for _ in range(200)]
    s = k1.shuffle_null(frame(truth), 18, seed=3)
    assert abs(float(s.mean())) < 0.35


def test_the_position_prior_leaves_its_own_image_out():
    """One image with a unique peak must not gain from its own truth."""
    base = np.zeros(8)
    truth = [base.copy() for _ in range(5)]
    truth[0] = np.array([9.0, 0, 0, 0, 0, 0, 0, -9.0])
    for i in range(1, 5):
        truth[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0.0])
    s = k1.position_prior(frame(truth), 18)
    # The other 4 images are flat, so the prior is flat, so image 0 scores 0.
    assert s["img0"] == pytest.approx(0.0)


def test_the_position_prior_finds_a_shared_frame_bias():
    rng = np.random.default_rng(2)
    bias = np.linspace(-1.0, 1.0, 16)
    truth = [bias + 0.05 * rng.normal(size=16) for _ in range(40)]
    s = k1.position_prior(frame(truth), 18)
    assert float(s.mean()) > 2.0


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.3, 0.9])
    q = k1.bh(p)
    assert np.all(q >= p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert q.max() <= 1.0


def test_boot_ci_brackets_the_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(loc=1.0, scale=1.0, size=200)
    lo, hi = k1.boot_ci(x, 1000, np.random.default_rng(5), "mean")
    assert lo < float(x.mean()) < hi
    assert lo > 0.0


def test_the_values_layout_is_read():
    """`causal_maps` writes image_id and values; both layouts must work."""
    t = np.array([1.0, 5.0, 2.0])
    df = pd.DataFrame([{"layer": 18, "image_id": "a", "values": list(t),
                        "gxa": list(t)}])
    tab, cols = k1.table(df, 18)
    assert cols == ["gxa"]
    assert tab.loc["a", "gxa"] == pytest.approx(tab.loc["a", "ceiling"])


def test_a_wrong_length_instrument_gives_nan():
    truth = [np.array([1.0, 2.0, 3.0, 4.0])]
    df = frame(truth, {"bad": [np.array([1.0, 2.0])]})
    tab, _ = k1.table(df, 18)
    assert np.isnan(tab.loc["img0", "bad"])


def test_pooled_ranks_averages_over_layers(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))
    from research.scripts import k1_layer_profile as lp

    t = np.array([0.0, 1.0, 2.0, 3.0])
    rows = [
        {"layer": 18, "target": "a", "causal": list(t), "gxa": [3.0, 2, 1, 0]},
        {"layer": 24, "target": "a", "causal": list(t), "gxa": [0.0, 1, 2, 3]},
    ]
    f = tmp_path / "maps.parquet"
    pd.DataFrame(rows).to_parquet(f)
    got = lp.pooled_ranks(f, "gxa")
    # The 2 layers are exact opposites, so every mean rank is the same.
    assert np.allclose(got["a"][0], 2.5)
    assert np.allclose(got["a"][1], t)


def test_pooled_ranks_skips_a_missing_arm(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))
    from research.scripts import k1_layer_profile as lp

    t = [0.0, 1.0, 2.0]
    rows = [
        {"layer": 18, "target": "a", "causal": t, "gxa": [1.0, 2.0, 3.0]},
        {"layer": 24, "target": "a", "causal": t, "gxa": None},
    ]
    f = tmp_path / "maps.parquet"
    pd.DataFrame(rows).to_parquet(f)
    got = lp.pooled_ranks(f, "gxa")
    assert np.allclose(got["a"][0], [1.0, 2.0, 3.0])


def test_pooled_ranks_drops_an_image_with_no_usable_layer(tmp_path):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "scripts"))
    from research.scripts import k1_layer_profile as lp

    rows = [{"layer": 18, "target": "a", "causal": [0.0, 1.0],
             "gxa": [np.nan, 1.0]}]
    f = tmp_path / "maps.parquet"
    pd.DataFrame(rows).to_parquet(f)
    assert lp.pooled_ranks(f, "gxa") == {}
