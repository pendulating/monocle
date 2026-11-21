"""Tests of the OLS with clustered standard errors of `scripts/f8/method_b.py` (F8.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "f8"))

import method_b as M  # noqa: E402


def _clustered_data(seed: int = 0, groups: int = 200, per: int = 30, beta: float = 0.5):
    """Make data with a cluster-level predictor and a cluster-level error, as an NTA design."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), per)
    x = rng.normal(size=groups)[g]
    y = 1.0 + beta * x + rng.normal(scale=0.5, size=groups)[g] + rng.normal(size=g.size)
    return pd.DataFrame({"y": y, "x": x, "g": g})


def test_known_beta_is_recovered():
    d = _clustered_data(beta=0.5)
    r = M.ols_cluster(d.y, M.design(d, "x"), d.g)
    assert r["beta"][1] == pytest.approx(0.5, abs=3 * r["se"][1])
    assert r["p"][1] < 1e-6
    assert r["g"] == 200


def test_null_beta_is_not_significant_too_often():
    # Under the null, the 5% test rejects about 5% of the time.
    rejects = [M.ols_cluster(d.y, M.design(d, "x"), d.g)["p"][1] < 0.05
               for d in (_clustered_data(seed=s, groups=60, per=20, beta=0.0) for s in range(200))]
    assert np.mean(rejects) < 0.10


def test_cluster_se_is_larger_than_iid_se_with_cluster_errors():
    d = _clustered_data(beta=0.0)
    X = M.design(d, "x")
    r = M.ols_cluster(d.y, X, d.g)
    u = d.y - X @ r["beta"]
    iid = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (u @ u) / (len(d) - X.shape[1]))
    assert r["se"][1] > 1.5 * iid[1]


def test_matches_statsmodels_cr1():
    sm = pytest.importorskip("statsmodels.api")
    d = _clustered_data(seed=3, groups=50, per=10)
    X = M.design(d, "x")
    r = M.ols_cluster(d.y, X, d.g)
    ref = sm.OLS(d.y.to_numpy(), X).fit(cov_type="cluster", cov_kwds={"groups": d.g.to_numpy()}, use_t=True)
    np.testing.assert_allclose(r["beta"], ref.params, rtol=1e-10)
    np.testing.assert_allclose(r["se"], ref.bse, rtol=1e-8)
    np.testing.assert_allclose(r["p"], ref.pvalues, rtol=1e-6)


def test_fixed_effect_absorbs_a_confounder():
    # The outcome depends on the type only. The type goes with x. Model 0
    # finds an x effect; model 1 with type FE finds none.
    rng = np.random.default_rng(1)
    groups, per = 300, 20
    g = np.repeat(np.arange(groups), per)
    x = rng.normal(size=groups)[g]
    kind = np.where(rng.random(g.size) < 1 / (1 + np.exp(-2 * x)), "a", "b")
    y = 2.0 * (kind == "a") + rng.normal(size=g.size)
    d = pd.DataFrame({"y": y, "x": x, "g": g, "kind": kind})
    r0 = M.ols_cluster(d.y, M.design(d, "x"), d.g)
    r1 = M.ols_cluster(d.y, M.design(d, "x", ["kind"]), d.g)
    assert r0["p"][1] < 1e-6 and r0["beta"][1] > 0.3
    assert abs(r1["beta"][1]) < 3 * r1["se"][1]
    assert abs(r1["beta"][1]) < 0.2 * r0["beta"][1]


def test_design_drops_one_level_and_rank_check():
    d = pd.DataFrame({"x": [0.0, 1, 2, 3], "fe": ["a", "a", "b", "c"]})
    X = M.design(d, "x", ["fe"])
    assert X.shape == (4, 4)
    with pytest.raises(ValueError):
        M.ols_cluster(np.zeros(4), np.column_stack([X, X[:, 1]]), [0, 0, 1, 1])


def test_fit_all_table_and_bonferroni():
    d = _clustered_data(beta=0.4).rename(columns={"y": "outcome", "x": "f1", "g": "nta2020"})
    d["f2"] = np.random.default_rng(2).normal(size=200)[d.nta2020]
    d["borough"] = np.where(d.nta2020 % 2, "BK", "MN")
    d = d.assign(model="m", case="street", field="response", image_idx=np.arange(len(d)))
    fits = M.fit_all(d, features=["f1", "f2"])
    assert list(fits.feature) == ["f1", "f2"]
    assert (fits.n_nta == 200).all() and (fits.tests == 2).all()
    np.testing.assert_allclose(fits.p1_bonf, np.minimum(1, 2 * fits.p1))
    assert fits.set_index("feature").loc["f1", "sig1"]
    assert fits.share_kept.iloc[0] == pytest.approx(fits.beta1.iloc[0] / fits.beta0.iloc[0])
