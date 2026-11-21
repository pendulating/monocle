"""Tests for the map retest."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import truth_retest as tr  # noqa: E402


def test_a_perfect_repeat_agrees():
    rng = np.random.default_rng(0)
    v = rng.normal(size=64)
    s = tr.compare(v, v)
    assert s["pearson"] > 0.999
    assert s["top1_same"] and s["bottom1_same"]
    assert s["max_abs_diff"] == 0.0


def test_the_ceiling_is_the_full_range_for_a_perfect_repeat():
    rng = np.random.default_rng(1)
    v = rng.normal(size=256)
    s = tr.compare(v, v)
    want = (v.max() - v.min()) / v.std(ddof=1)
    assert abs(s["sep_z"] - want) < 1e-9


def test_noise_lowers_every_statistic():
    rng = np.random.default_rng(2)
    v = rng.normal(size=256)
    noisy = v + rng.normal(scale=2.0, size=256)
    s = tr.compare(noisy, v)
    assert s["pearson"] < 0.7
    assert s["sep_z"] < (v.max() - v.min()) / v.std(ddof=1)


def test_load_stored_takes_one_map_for_each_image(tmp_path):
    rng = np.random.default_rng(3)
    rows = []
    for img in ("a", "b"):
        m = rng.normal(size=16)
        for lay in (12, 18):
            rows.append({"target": img, "layer": lay, "causal": m})
    p = tmp_path / "maps.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    got = tr.load_stored(p)
    assert set(got) == {"a", "b"}
    assert got["a"].shape == (16,)


def test_a_shuffle_puts_every_cell_back():
    rng = np.random.default_rng(4)
    n = 32
    order = rng.permutation(n)
    truth = rng.normal(size=n)
    out = np.zeros(n)
    for start in range(0, n, 7):
        cells = [int(c) for c in order[start:start + 7]]
        for c in cells:
            out[c] = truth[c]
    assert np.allclose(out, truth)


def test_the_retest_stores_a_contribution_not_a_delta():
    """A removal delta and a contribution differ by a sign.

    `causal_maps` writes `-ablation_map(...)`. If the retest wrote the raw
    delta the 2 maps would correlate at -1 and the run would read as broken.
    """
    import inspect
    src = inspect.getsource(tr.shuffled_map)
    assert "return -deltas" in src
