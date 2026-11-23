"""Tests for the picture-statistic control, the sign split and the probe.

Each test checks a rule the analysis depends on, not the numbers of a run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "research" / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats = load("k1_image_stats")
signflip = load("k1_sign_and_flip")
locality = load("k1_locality_predict")
probe = load("k1_fitted_probe")


# --- the picture statistics -------------------------------------------

def test_cell_stats_gives_one_value_for_each_cell(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(
        (np.random.default_rng(0).random((300, 400, 3)) * 255)
        .astype("uint8")).save(p)
    out = stats.cell_stats(str(p), 16)
    for k in stats.STATS:
        assert out[k].shape == (256,), k


def test_a_bright_half_raises_the_brightness_of_its_cells(tmp_path):
    a = np.zeros((896, 896, 3), dtype="uint8")
    a[:448] = 255
    p = tmp_path / "b.png"
    Image.fromarray(a).save(p)
    out = stats.cell_stats(str(p), 16)
    lum = out["lum"].reshape(16, 16)
    assert lum[:8].mean() > 0.9
    assert lum[8:].mean() < 0.1


def test_the_row_statistic_is_the_row(tmp_path):
    p = tmp_path / "c.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype="uint8")).save(p)
    out = stats.cell_stats(str(p), 16)
    assert out["row"].reshape(16, 16)[3].tolist() == [3.0] * 16
    assert out["col"].reshape(16, 16)[:, 5].tolist() == [5.0] * 16


def test_residualise_leaves_nothing_of_the_column_it_removes():
    rng = np.random.default_rng(1)
    x = rng.normal(size=256)
    t = 3.0 * x + 1.0
    left = stats.residualise(t, x[:, None])
    assert np.allclose(left, 0.0, atol=1e-8)


def test_residualise_keeps_the_part_that_is_not_explained():
    rng = np.random.default_rng(2)
    x = rng.normal(size=256)
    other = rng.normal(size=256)
    left = stats.residualise(2.0 * x + other, x[:, None])
    assert np.corrcoef(left, other)[0, 1] > 0.9


# --- the sign split and the flip ---------------------------------------

def test_the_halves_add_up_to_the_separation():
    from research.scripts.k1_separation import sep_z
    rng = np.random.default_rng(3)
    v = rng.normal(size=256)
    t = rng.normal(size=256)
    hi, lo = signflip.halves(v, t)
    assert hi + lo == pytest.approx(sep_z(v, t))


def test_a_flip_needs_the_sign_of_c_to_change():
    t = np.zeros(256)
    t[7] = 5.0
    v = np.zeros(256)
    v[7] = 1.0
    # c0 = +2 and the cell is worth +5, so the deletion takes c to -3
    assert signflip.flip(v, t, 2.0) is True
    # c0 = +9 keeps the sign, so no flip
    assert signflip.flip(v, t, 9.0) is False


def test_a_negative_c_takes_the_bottom_cell():
    t = np.zeros(256)
    t[3] = -5.0
    v = np.zeros(256)
    v[3] = -1.0
    assert signflip.flip(v, t, -2.0) is True


# --- the locality line -------------------------------------------------

def test_top1_share_of_one_hot_is_one():
    v = np.zeros(256)
    v[11] = -4.0
    assert locality.top1_share(v) == pytest.approx(1.0)


def test_top1_share_of_a_flat_map_is_small():
    assert locality.top1_share(np.ones(256)) == pytest.approx(1 / 256)


def test_top1_share_of_an_empty_map_is_not_a_number():
    assert np.isnan(locality.top1_share(np.zeros(8)))


def test_the_line_goes_through_two_points():
    a, b = locality.fit_line(np.array([0.0, 1.0]), np.array([2.0, 5.0]))
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(3.0)


# --- the fitted probe --------------------------------------------------

def test_the_centring_takes_out_the_photograph_mean():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(256, 32)) + 100.0
    assert np.allclose(probe.prep(h).mean(axis=0), 0.0, atol=1e-9)


def test_the_target_is_scaled_to_one():
    rng = np.random.default_rng(5)
    t = rng.normal(size=256) * 7.0 + 3.0
    z = probe.zscore(t)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std(ddof=1) == pytest.approx(1.0)


def test_the_ridge_finds_a_planted_direction():
    rng = np.random.default_rng(6)
    data = {}
    w = np.zeros(20)
    w[4] = 2.0
    for i in range(4):
        X = rng.normal(size=(64, 20))
        data[str(i)] = (X, (X - X.mean(0)) @ w + 0.01 * rng.normal(size=64))
    got = probe.Gram(data, list(data)).fit(1.0)
    assert int(np.argmax(np.abs(got))) == 4


def test_a_bigger_alpha_makes_a_smaller_weight():
    rng = np.random.default_rng(7)
    data = {}
    for i in range(3):
        X = rng.normal(size=(64, 10))
        data[str(i)] = (X, (X - X.mean(0))[:, 0] * 3.0)
    g = probe.Gram(data, list(data))
    assert np.linalg.norm(g.fit(1e5)) < np.linalg.norm(g.fit(1.0))


def test_the_drop_leaves_out_exactly_one_photograph():
    rng = np.random.default_rng(11)
    data = {str(i): (rng.normal(size=(64, 6)), rng.normal(size=64))
            for i in range(3)}
    full = probe.Gram(data, list(data))
    part = probe.Gram(data, ["0", "1"])
    assert np.allclose(full.fit(10.0, drop="2"), part.fit(10.0), atol=1e-8)


def test_the_prediction_uses_the_centred_features():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(32, 5)) + 50.0
    w = rng.normal(size=5)
    got = probe.predict(x, w)
    assert got.mean() == pytest.approx(0.0, abs=1e-9)
