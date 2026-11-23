"""Tests for the analysis that decides the causal-lens question.

These scripts produce the headline numbers, so each one is checked against a
world whose answer we already know.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (str(REPO), str(REPO / "research" / "scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from research.scripts.gradient_rank import spectrum  # noqa: E402
from research.scripts.linear_probe import run as probe_run  # noqa: E402
from research.scripts.null_bar import paired  # noqa: E402

# The interpreter of the test run. Before, a fixed path to .venv-3.12, which a new clone does not have.
PY_BIN = sys.executable


class TestSpectrum:
    def test_a_rank_three_set_reads_as_three(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 3)) @ rng.normal(size=(3, 200))
        s = spectrum(x)
        assert 2.0 < s["pr"] < 4.0
        assert s["r80"] <= 3

    def test_a_full_rank_set_reads_as_many(self):
        rng = np.random.default_rng(1)
        s = spectrum(rng.normal(size=(400, 200)))
        assert s["pr"] > 100

    def test_mean_share_is_one_when_every_row_is_the_mean(self):
        rng = np.random.default_rng(2)
        mu = rng.normal(size=200)
        s = spectrum(mu + 0.01 * rng.normal(size=(400, 200)))
        assert s["mean_share"] > 0.99

    def test_mean_share_is_small_when_the_mean_is_zero(self):
        rng = np.random.default_rng(3)
        s = spectrum(rng.normal(size=(400, 200)))
        assert s["mean_share"] < 0.05

    def test_a_dead_set_gives_nan_and_does_not_raise(self):
        s = spectrum(np.zeros((10, 20)))
        assert np.isnan(s["pr"])


def make_acts(n_img, n_patch, d, *, decodable, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    out = {}
    for i in range(n_img):
        h = rng.normal(size=(n_patch, d))
        y = (h @ w + 0.3 * rng.normal(size=n_patch) if decodable
             else rng.normal(size=n_patch))
        out[f"i{i}"] = {"truth": y.astype(np.float32),
                        "h12": h.astype(np.float32)}
    return out


ALPHAS = [1e0, 1e1, 1e2, 1e3]


class TestLinearProbe:
    def test_it_finds_a_real_linear_map(self):
        acts = make_acts(20, 256, 128, decodable=True, seed=0)
        s = probe_run(acts, 12, ALPHAS, 5, 777, False)
        assert s["r"] > 0.85
        assert s["top16"] > 10

    def test_the_shuffle_control_falls_to_zero(self):
        acts = make_acts(20, 256, 128, decodable=True, seed=0)
        s = probe_run(acts, 12, ALPHAS, 5, 777, True)
        assert abs(s["r"]) < 0.15

    def test_it_rejects_a_world_with_no_linear_map(self):
        acts = make_acts(20, 256, 128, decodable=False, seed=1)
        s = probe_run(acts, 12, ALPHAS, 5, 777, False)
        assert abs(s["r"]) < 0.15
        assert abs(s["top16"] - 4.0) < 1.5

    def test_a_missing_layer_is_an_error(self):
        acts = make_acts(6, 32, 16, decodable=True, seed=2)
        with pytest.raises(SystemExit):
            probe_run(acts, 99, ALPHAS, 5, 777, False)


def arm_frame(name, ids, ks, values):
    return pd.DataFrame([{"arm": name, "image_id": i, "k": k,
                          "separation_z": values(i, k)}
                         for i in ids for k in ks])


class TestPaired:
    IDS = [f"i{i}" for i in range(20)]
    KS = [8, 16]

    def test_identical_arms_give_no_gain(self):
        rng = np.random.default_rng(0)
        v = {(i, k): rng.normal() for i in self.IDS for k in self.KS}
        a = arm_frame("a", self.IDS, self.KS, lambda i, k: v[(i, k)])
        b = arm_frame("b", self.IDS, self.KS, lambda i, k: v[(i, k)])
        r = paired(a, b, 8, 2000, np.random.default_rng(1))
        assert r["gain"] == pytest.approx(0.0)
        assert r["wins"] == 0

    def test_a_real_gain_is_found(self):
        rng = np.random.default_rng(2)
        v = {(i, k): rng.normal() for i in self.IDS for k in self.KS}
        a = arm_frame("a", self.IDS, self.KS, lambda i, k: v[(i, k)] + 3.0)
        b = arm_frame("b", self.IDS, self.KS, lambda i, k: v[(i, k)])
        r = paired(a, b, 8, 2000, np.random.default_rng(3))
        assert r["gain"] == pytest.approx(3.0, abs=1e-9)
        assert r["lo"] > 0
        assert r["wins"] == 20

    def test_it_pairs_on_the_image_not_the_order(self):
        rng = np.random.default_rng(4)
        v = {(i, k): rng.normal() for i in self.IDS for k in self.KS}
        a = arm_frame("a", self.IDS, self.KS, lambda i, k: v[(i, k)] + 1.0)
        b = arm_frame("b", list(reversed(self.IDS)), self.KS,
                      lambda i, k: v[(i, k)])
        r = paired(a, b, 8, 2000, np.random.default_rng(5))
        assert r["gain"] == pytest.approx(1.0, abs=1e-9)

    def test_it_reads_only_the_asked_k(self):
        a = arm_frame("a", self.IDS, self.KS,
                      lambda i, k: 1.0 if k == 8 else 99.0)
        b = arm_frame("b", self.IDS, self.KS, lambda i, k: 0.0)
        r = paired(a, b, 8, 2000, np.random.default_rng(6))
        assert r["gain"] == pytest.approx(1.0)


class TestMakeRankingShuffle:
    def test_no_image_keeps_its_own_map(self, tmp_path):
        rng = np.random.default_rng(0)
        rows = [{"target": f"i{i}", "layer": 18,
                 "lens_own": rng.normal(size=256)} for i in range(12)]
        src = tmp_path / "maps.parquet"
        pd.DataFrame(rows).to_parquet(src, index=False)
        out = tmp_path / "r.parquet"
        subprocess.run(
            [PY_BIN, str(REPO / "research/scripts/make_ranking.py"), "--maps", str(src),
             "--arm", "lens_own", "--layer", "18", "--group", "2",
             "--shuffle", "99", "--out", str(out)], check=True,
            capture_output=True)
        got = pd.read_parquet(out).set_index("image_id")
        import sys as _s
        _s.path.insert(0, str(REPO / "research" / "scripts"))
        from research.scripts.make_ranking import pool
        want = {r["target"]: pool(r["lens_own"], 2) for r in rows}
        same = sum(np.allclose(got.loc[k, "values"], want[k]) for k in want)
        assert same == 0
        a = sorted(tuple(np.round(v, 6)) for v in got["values"])
        b = sorted(tuple(np.round(v, 6)) for v in want.values())
        assert a == b


from research.scripts.linear_probe import block_features, run_blocks  # noqa: E402


class TestBlockFeatures:
    def test_the_shape_falls_by_the_square(self):
        h = np.zeros((256, 8))
        assert block_features(h, 2).shape == (64, 8)
        assert block_features(h, 4).shape == (16, 8)

    def test_a_block_sums_its_patches(self):
        h = np.arange(256 * 3, dtype=float).reshape(256, 3)
        b = block_features(h, 2).reshape(8, 8, 3)
        g = h.reshape(16, 16, 3)
        assert np.allclose(b[0, 0], g[0:2, 0:2].sum(axis=(0, 1)))
        assert np.allclose(b[3, 5], g[6:8, 10:12].sum(axis=(0, 1)))

    def test_the_total_is_preserved(self):
        h = np.random.default_rng(0).normal(size=(256, 5))
        assert np.allclose(block_features(h, 4).sum(0), h.sum(0))

    def test_group_one_is_the_identity(self):
        h = np.random.default_rng(1).normal(size=(256, 5))
        assert np.allclose(block_features(h, 1), h)

    def test_a_grid_that_does_not_divide_is_an_error(self):
        with pytest.raises(SystemExit):
            block_features(np.zeros((256, 4)), 3)


def make_block_world(n_img, d, group, *, additive, seed):
    """A world whose block truth IS (or is not) an additive read of h."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    acts, truth = {}, {}
    for i in range(n_img):
        h = rng.normal(size=(256, d))
        b = block_features(h, group)
        y = (b @ w if additive else rng.normal(size=b.shape[0]))
        acts[f"i{i}"] = {"h12": h.astype(np.float32)}
        truth[f"i{i}"] = y
    return acts, truth


class TestRunBlocks:
    ALPHAS = [1e0, 1e1, 1e2, 1e3]

    def test_it_finds_the_best_additive_instrument(self):
        acts, truth = make_block_world(24, 64, 2, additive=True, seed=0)
        s = run_blocks(acts, truth, 12, 2, self.ALPHAS, 4, 777, False)
        assert s["r"] > 0.85

    def test_the_shuffle_control_falls_to_zero(self):
        acts, truth = make_block_world(24, 64, 2, additive=True, seed=0)
        s = run_blocks(acts, truth, 12, 2, self.ALPHAS, 4, 777, True)
        assert abs(s["r"]) < 0.25

    def test_it_rejects_a_non_additive_world(self):
        acts, truth = make_block_world(24, 64, 2, additive=False, seed=1)
        s = run_blocks(acts, truth, 12, 2, self.ALPHAS, 4, 777, False)
        assert abs(s["r"]) < 0.25

    def test_a_truth_of_the_wrong_size_is_an_error(self):
        acts, truth = make_block_world(8, 16, 2, additive=True, seed=2)
        with pytest.raises(SystemExit):
            run_blocks(acts, truth, 12, 4, self.ALPHAS, 4, 777, False)
