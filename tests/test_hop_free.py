"""Tests for the word sets and the lens-free analysis."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/scripts/f10"))
from research.scripts.f10.hop_free import folds, grid_pool, kernel, monotone_path, oof_mixed, oof_predict, r2, rdm_ranks  # noqa: E402
from research.scripts.f10.hop_sets import match_controls  # noqa: E402


def test_folds_keep_each_nta_whole():
    groups = np.repeat([f"N{i}" for i in range(20)], 3)
    for tr, te in folds(groups):
        assert not set(groups[tr]) & set(groups[te])


def test_ridge_recovers_a_linear_signal_and_not_noise():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(150, 5))
    x = latent @ rng.normal(size=(5, 400)) + rng.normal(size=(150, 400))
    groups = np.repeat(np.arange(50), 3)
    splits = folds(groups)
    K = kernel(x)
    signal = latent.sum(1)
    assert r2(signal, oof_predict(K, signal, splits), splits) > 0.5
    noise = rng.normal(size=150)
    assert r2(noise, oof_predict(K, noise, splits), splits) < 0.1


def test_mixed_kernel_gains_when_the_second_block_holds_the_signal():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(150, 300)), rng.normal(size=(150, 300))
    y = b[:, :10].sum(1)
    splits = folds(np.repeat(np.arange(50), 3))
    A, B = kernel(a), kernel(b)
    assert r2(y, oof_mixed(A, B, y, splits), splits) > r2(y, oof_predict(A, y, splits), splits) + 0.2


def test_grid_pool_gives_the_mean_of_each_quadrant():
    p = np.arange(16.0).reshape(1, 16, 1, 1)
    out = grid_pool(p, grid=2)
    assert out.shape == (1, 1, 4, 1)
    assert out[0, 0, :, 0].tolist() == [2.5, 4.5, 10.5, 12.5]


def test_rdm_ranks_length_and_scale():
    x = np.random.default_rng(2).normal(size=(10, 30))
    r = rdm_ranks(x)
    assert len(r) == 45 and abs(float(r.mean())) < 1e-5


def test_monotone_path_never_falls():
    assert monotone_path(np.array([[0.0, 9.0], [9.0, 0.0], [0.0, 1.0]])) == [0, 0, 1]


def test_match_controls_are_unique_and_outside_the_set():
    f = pd.DataFrame({"freq": [1.0, 1.1, 1.2, 3.0, 3.1], "norm": [1.0, 1.0, 1.0, 2.0, 2.0], "length": [4, 4, 4, 6, 6]}, index=list("abcde"))
    got = match_controls(f, ["a", "d"], {"a", "d"})
    assert got == ["b", "e"]


def test_borough_split_and_t():
    from research.scripts.f10.hop_borough import balanced_means, held_out, split_within, two_sample_t

    borough = np.array(["A"] * 10 + ["B"] * 6)
    half = split_within(borough)
    assert half[:10].sum() == 5 and half[10:].sum() == 3
    assert balanced_means(np.array([1.0, 3.0, 5.0, 7.0]), np.array([0, 0, 1, 1])).tolist() == [2.0, 6.0]
    rng = np.random.default_rng(3)
    m = rng.normal(size=(80, 300))
    member = np.arange(80) < 30
    m[member, :60] += 1.0
    t = two_sample_t(m, member)
    assert t[:60].mean() > 2.5 > abs(t[60:].mean())
    other = rng.normal(size=(80, 300))
    other[member, :60] += 1.0
    assert held_out(other, t, member)[1] < 1e-6


def test_link_scores_find_a_shared_word_and_ignore_a_borough_word():
    from research.scripts.f10.hop_link import link_scores, presence

    rows = pd.DataFrame({"image_idx": [0, 1, 2, 3], "text": ["alpha gamma", "beta gamma", "alpha", "beta"]})
    assert presence(rows, pd.Series([0, 1, 2, 3]), np.array(["alpha", "beta", "gamma", "delta"])).shape == (4, 4)
    rng = np.random.default_rng(4)
    p = rng.random((40, 200)) < 0.1
    x = rng.normal(size=(40, 200)) + 2.0 * p
    borough = np.array(["A", "B"] * 20)
    s, within = link_scores(x.astype(np.float32), p.astype(np.float32), borough)
    assert s.mean() > 1 and within.mean() > 1
    s0, within0 = link_scores(rng.normal(size=(40, 200)).astype(np.float32), p.astype(np.float32), borough)
    assert abs(within0.mean()) < 0.1


def test_axes_recover_the_large_axis():
    from research.scripts.f10.hop_axes import axes

    rng = np.random.default_rng(5)
    load = rng.normal(size=500)
    score = rng.normal(size=60)
    m = np.outer(score, load) * 5 + rng.normal(size=(60, 500))
    c, u, v = axes(m, 3)
    assert abs(np.corrcoef(v[0], load)[0, 1]) > 0.99
    assert np.linalg.norm(c - u[:, :1] @ v[:1]) < 0.5 * np.linalg.norm(c)


def test_argument_tag_and_icc():
    from research.scripts.f10.analyze_argument import icc, tag

    assert "brick" in tag("red brick building") and "tree or greenery" in tag("large trees")
    assert tag("nothing to see") == []
    same = np.array([[1, 1, 1], [0, 0, 0]] * 20, dtype=float)
    assert icc(same) > 0.99
    rng = np.random.default_rng(0)
    assert abs(icc(rng.integers(0, 2, (2000, 3)).astype(float))) < 0.05
