"""Tests of the value functions of research/scripts/f10/hop_prompts.py (F10.26)."""

import numpy as np
import pandas as pd
import pytest

from research.scripts.f10 import hop_prompts as hp  # noqa: E402


def test_store_name_maps_each_prompt_to_one_store():
    assert hp.store_name("gemma-4-31b", "cool", "place") == "gemma-4-31b"
    assert hp.store_name("qwen3.5-9b", "ugly", "place") == "qwen3.5-9b-ugly"
    assert hp.store_name("qwen3.6-27b", "cool", "scene") == "qwen3.6-27b-scene-cool"
    names = {hp.store_name(m, a, n) for m in hp.MODELS for a, n in hp.PROMPTS}
    assert len(names) == 42
    with pytest.raises(ValueError):
        hp.store_name("gemma-4-31b", "warm", "place")


def test_answer_store_uses_the_8_form_file_for_the_old_cool_prompts():
    assert hp.answer_store("gemma-4-31b", "cool", "place") == "gemma-4-31b-forms8"
    assert hp.answer_store("qwen3.5-9b", "cool", "place") == "qwen3.5-9b-forms8"
    assert hp.answer_store("qwen3.6-27b", "cool", "place") == "qwen3.6-27b"
    assert hp.answer_store("gemma-4-31b", "nice", "place") == "gemma-4-31b-nice"
    assert hp.answer_store_f1024("gemma-4-31b", "cool", "place") == "gemma-4-31b"


def test_uncap_inverts_the_softcap():
    x = np.linspace(-40, 40, 9)
    capped = 30.0 * np.tanh(x / 30.0)
    assert np.allclose(hp.uncap(capped, 30.0), x, atol=1e-4)
    assert hp.uncap(capped, None) is capped


def test_zscore_has_zero_mean_and_unit_sd_on_the_last_axis():
    z = hp.zscore(np.random.default_rng(0).normal(size=(3, 4, 50)))
    assert np.allclose(z.mean(-1), 0, atol=1e-12)
    assert np.allclose(z.std(-1), 1, atol=1e-12)


def test_margin_of_pools_the_two_halves():
    x = np.log(np.array([[1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]]))
    assert np.allclose(hp.margin_of(x), np.log(2.0))


def test_axis1_gives_the_top_eigen_direction():
    rng = np.random.default_rng(1)
    u = rng.normal(size=20)
    v = rng.normal(size=100)
    c = np.outer(u, v) + 0.01 * rng.normal(size=(20, 100))
    share, score, load = hp.axis1(c)
    assert share > 0.99
    uc = u - u.mean()
    assert abs(np.corrcoef(score, uc)[0, 1]) > 0.999
    assert abs(np.corrcoef(load, v)[0, 1]) > 0.999
    # score x load is the rank-1 part of the centred matrix
    cc = c - c.mean(0, keepdims=True)
    assert np.linalg.norm(cc - np.outer(score, load)) < 0.1 * np.linalg.norm(cc)


def test_embedded_counts_on_a_known_ranking():
    full = np.arange(2000, 0, -1).astype(float)  # word 0 has the highest readout
    text = full.copy()
    assert hp.embedded_counts(full, text) == (100, 0, 0)
    text2 = full.copy()
    text2[:50] = -1  # the first 50 words fall past rank 1,000 with no image
    e, raised, lowered = hp.embedded_counts(full, text2)
    assert e == 50 and raised == 50 and lowered == 0


def test_first_run_takes_the_first_run_only():
    assert hp.first_run([0, 4, 8, 12, 16], [False, True, True, False, True]) == [4, 8]
    assert hp.first_run([0, 4], [False, False]) == []


def test_mid_layer_is_the_saved_layer_nearest_half_depth():
    assert hp.mid_layer("gemma-4-31b", list(range(0, 57, 4))) == 28
    assert hp.mid_layer("qwen3.5-9b", list(range(0, 29, 4))) == 16
    assert hp.mid_layer("qwen3.6-27b", list(range(0, 61, 4))) == 32


def test_jaccard_and_top_set():
    v = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    clean = np.array([True, False, True, True, True])
    assert hp.top_set(v, clean, 2) == {0, 2}
    assert hp.jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_shared_direction_recovers_a_common_direction():
    rng = np.random.default_rng(2)
    u = rng.normal(size=50)
    u /= np.linalg.norm(u)
    W = np.stack([u + 0.03 * rng.normal(size=50) for _ in range(7)], axis=1)
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    u1, share = hp.shared_direction(W, 1)
    assert abs(hp.cosine(u1[:, 0], u)) > 0.99
    assert share[0] > 0.9


def test_prompt_decoders_recover_a_linear_target_out_of_fold():
    rng = np.random.default_rng(3)
    n, d = 200, 30
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    Y = np.column_stack([X @ w + 0.1 * rng.normal(size=n), -(X @ w) + 0.1 * rng.normal(size=n)])
    Y[:5, 1] = np.nan
    groups = np.repeat(np.arange(40), 5)
    splits = hp.folds(groups, seed=1)
    dec = hp.prompt_decoders(X, Y, groups, splits)
    assert dec["r2"].min() > 0.9
    assert hp.cosine(dec["full_w"][:, 0], w) > 0.99
    assert hp.cosine(dec["full_w"][:, 0], dec["full_w"][:, 1]) < -0.99
    assert np.isfinite(dec["pred"][:, 0]).all()


def test_subset_kernel_is_centred_with_trace_n():
    rng = np.random.default_rng(4)
    K = hp.kernel(rng.normal(size=(20, 8)))
    k = hp.subset_kernel(K, np.arange(10))
    assert np.allclose(k.sum(0), 0, atol=1e-9)
    assert np.isclose(np.trace(k), 10)


def test_check_positions_rejects_the_wrong_token():
    man = {"positions": ["after_image", "first_text", "place", "after_place", "after_cool", "final"], "tail_tokens": ["<i>", "Is", " place", " cool", "?", "<c>"]}
    assert hp.check_positions(man, "cool", "place") == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        hp.check_positions(man, "nice", "place")
    with pytest.raises(ValueError):
        hp.check_positions(man, "cool", "scene")
