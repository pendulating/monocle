"""Tests for scripts/pairwise_vqa_regression_report.py.

Covers the statistical core (planted slope/R² recovery, null, Δx pair-level
orientation, controls + partial R², WLS weighting), the covariate value-map
resolution paths, screen-mode BH handling, registry id determinism for
regression inputs, and end-to-end CLI runs on synthetic data. No W&B /
network involved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pairwise_vqa_regression_report as reg  # noqa: E402
import pairwise_analysis_common as common  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_frame(n=300, slope=1.0, noise=0.5, seed=0, sigma=1.0):
    """Synthetic unit frame: y = slope * x + noise, constant TrueSkill σ."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = slope * x + rng.normal(0, noise, n)
    return pd.DataFrame({
        "unit_uid": [f"u{i}" for i in range(n)],
        "unit_name": [f"u{i}" for i in range(n)],
        "y": y, "mu": y, "sigma": np.full(n, sigma),
        "ts_conservative": y - 3 * sigma,
        "x": x,
    })


def _pair_df_from_units(unit_x: dict, n_pairs: int, slope: float, seed: int = 0):
    """Direct pairs whose ordinal score tracks slope * (x_a - x_b)."""
    rng = np.random.default_rng(seed)
    uids = list(unit_x)
    rows = []
    for i in range(n_pairs):
        ua, ub = rng.choice(uids, size=2, replace=False)
        dx = unit_x[ua] - unit_x[ub]
        score = int(np.clip(round(slope * dx + rng.normal(0, 0.8)), -2, 2))
        rows.append({
            "pair_id": f"p{i}", "canonical_pair_id": f"p{i}",
            "unit_uid_a": ua, "unit_uid_b": ub, "relative_score": score,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit-level fit
# ---------------------------------------------------------------------------


class TestUnitLevelFit:
    def test_planted_slope_recovered(self):
        frame = _unit_frame(n=400, slope=2.0, noise=0.5, seed=1)
        res = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        assert res["n_units"] == 400
        assert res["beta"] == pytest.approx(2.0, abs=0.15)
        assert res["r2"] > 0.85
        assert res["p"] < 1e-10
        assert res["spearman_rho"] > 0.8
        # No controls → partial R² equals R².
        assert res["partial_r2"] == pytest.approx(res["r2"])

    def test_null_no_effect(self):
        rng = np.random.default_rng(7)
        frame = _unit_frame(n=400, slope=0.0, noise=1.0, seed=7)
        res = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        assert res["r2"] < 0.02
        assert res["p"] > 0.01

    def test_standardized_beta_sign_and_scale(self):
        frame = _unit_frame(n=500, slope=-1.5, noise=0.1, seed=3)
        res = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        assert res["beta_std"] < 0
        assert abs(res["beta_std"]) <= 1.001  # |std beta| ≤ 1 in simple OLS

    def test_categorical_control_partial_r2(self):
        # y is driven entirely by a categorical group; x correlates with the
        # group but adds nothing. Controlling for the group should collapse
        # the focal partial R² to ~0.
        rng = np.random.default_rng(11)
        n = 600
        grp = rng.integers(0, 3, n)
        x = grp * 1.0 + rng.normal(0, 0.3, n)
        y = grp * 2.0 + rng.normal(0, 0.3, n)
        frame = pd.DataFrame({
            "unit_uid": [f"u{i}" for i in range(n)],
            "unit_name": "", "y": y, "mu": y, "sigma": 1.0,
            "ts_conservative": y - 3,
            "x": x, "grp": pd.Series(grp).map({0: "a", 1: "b", 2: "c"}),
        })
        naive = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        controlled = reg.fit_unit_level(frame, x="x", controls=["grp"], wls=False)
        assert naive["r2"] > 0.5                      # confounded fit looks strong
        assert controlled["partial_r2"] < 0.02        # vanishes given the group
        assert controlled["p"] > 0.001                # focal slope no longer clearly sig

    def test_wls_downweights_noisy_units(self):
        # Half the units carry a corrupted y with huge sigma; WLS should
        # recover the true slope much better than OLS.
        rng = np.random.default_rng(5)
        n = 400
        x = rng.normal(0, 1, n)
        y = 1.0 * x
        noisy = rng.random(n) < 0.5
        y = y + np.where(noisy, rng.normal(0, 8, n), rng.normal(0, 0.2, n))
        frame = pd.DataFrame({
            "unit_uid": [f"u{i}" for i in range(n)], "unit_name": "",
            "y": y, "mu": y, "sigma": np.where(noisy, 8.0, 0.2),
            "ts_conservative": y, "x": x,
        })
        ols = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        wls = reg.fit_unit_level(frame, x="x", controls=[], wls=True)
        assert abs(wls["beta"] - 1.0) < abs(ols["beta"] - 1.0) + 0.05
        assert wls["beta"] == pytest.approx(1.0, abs=0.1)

    def test_degenerate_inputs(self):
        frame = _unit_frame(n=5)
        res = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        assert np.isnan(res["r2"])  # too few rows
        frame = _unit_frame(n=100)
        frame["x"] = 1.0
        res = reg.fit_unit_level(frame, x="x", controls=[], wls=False)
        assert np.isnan(res["r2"])  # constant covariate


# ---------------------------------------------------------------------------
# Pair-level fit
# ---------------------------------------------------------------------------


class TestPairLevelFit:
    def test_planted_dx_slope(self):
        rng = np.random.default_rng(2)
        unit_x = {f"u{i}": float(rng.normal(0, 1)) for i in range(80)}
        df = _pair_df_from_units(unit_x, n_pairs=800, slope=1.2, seed=2)
        res = reg.fit_pair_level(df, pd.Series(unit_x))
        assert res["n_pairs"] == 800
        assert res["pair_slope"] > 0.3
        assert res["pair_p"] < 1e-10
        assert res["pair_spearman_rho"] > 0.3

    def test_orientation_sign(self):
        # x_a > x_b with score always +2 → positive slope; flipping the map
        # to -x flips the slope sign.
        unit_x = {"hi": 10.0, "lo": 0.0}
        rows = [{"pair_id": f"p{i}", "canonical_pair_id": f"p{i}",
                 "unit_uid_a": "hi", "unit_uid_b": "lo", "relative_score": 2}
                for i in range(10)]
        # Add a tiny bit of variation so dx isn't constant.
        rows += [{"pair_id": "q", "canonical_pair_id": "q",
                  "unit_uid_a": "lo", "unit_uid_b": "hi", "relative_score": -2}]
        df = pd.DataFrame(rows)
        res = reg.fit_pair_level(df, pd.Series(unit_x))
        assert res["pair_slope"] > 0
        res_neg = reg.fit_pair_level(df, pd.Series({k: -v for k, v in unit_x.items()}))
        assert res_neg["pair_slope"] < 0

    def test_repeat_collapse(self):
        unit_x = {"a": 1.0, "b": 0.0, "c": 2.0, "d": -1.0}
        rows = [
            {"pair_id": "p1", "canonical_pair_id": "c1", "unit_uid_a": "a",
             "unit_uid_b": "b", "relative_score": 2},
            {"pair_id": "p1_r1", "canonical_pair_id": "c1", "unit_uid_a": "a",
             "unit_uid_b": "b", "relative_score": 0},
        ] + [
            {"pair_id": f"p{i}", "canonical_pair_id": f"c{i}", "unit_uid_a": "c",
             "unit_uid_b": "d", "relative_score": 1} for i in range(2, 12)
        ]
        res = reg.fit_pair_level(pd.DataFrame(rows), pd.Series(unit_x))
        assert res["n_pairs"] == 11  # c1 collapsed

    def test_missing_x_dropped(self):
        unit_x = {"a": 1.0}  # b missing
        df = pd.DataFrame([{"pair_id": "p", "canonical_pair_id": "p",
                            "unit_uid_a": "a", "unit_uid_b": "b",
                            "relative_score": 1}])
        res = reg.fit_pair_level(df, pd.Series(unit_x))
        assert res["n_pairs"] == 0
        assert np.isnan(res["pair_slope"])


# ---------------------------------------------------------------------------
# Value-map resolution
# ---------------------------------------------------------------------------


class TestValueMap:
    def test_numeric_from_pairs_metadata(self):
        df = pd.DataFrame({
            "unit_uid_a": ["u1"], "unit_uid_b": ["u2"],
            "score_a": ["12.5"], "score_b": [None],
        })
        m = common.build_unit_value_map(
            df, "score", column_from_pairs=True,
            unit_metadata_parquet=None, unit_metadata_id_column="uid",
            numeric=True,
        )
        assert m["u1"] == 12.5
        assert "u2" not in m.index  # null dropped

    def test_numeric_external_join(self, tmp_path):
        meta = tmp_path / "meta.parquet"
        pd.DataFrame({"uid": ["u1", "u2"], "poverty": [80.5, np.nan]}).to_parquet(meta)
        df = pd.DataFrame({"unit_uid_a": ["u1"], "unit_uid_b": ["u2"]})
        m = common.build_unit_value_map(
            df, "poverty", column_from_pairs=False,
            unit_metadata_parquet=meta, unit_metadata_id_column="uid",
            numeric=True,
        )
        assert m["u1"] == 80.5
        assert "u2" not in m.index

    def test_missing_source_errors(self):
        df = pd.DataFrame({"unit_uid_a": ["u1"], "unit_uid_b": ["u2"]})
        with pytest.raises(SystemExit, match="unit-metadata"):
            common.build_unit_value_map(
                df, "poverty", column_from_pairs=False,
                unit_metadata_parquet=None, unit_metadata_id_column="uid",
                numeric=True,
            )


# ---------------------------------------------------------------------------
# Registry id determinism (regression inputs)
# ---------------------------------------------------------------------------


def test_regression_experiment_id_sensitivity():
    base = {"tool": "regression", "source_parquet": "/r.parquet", "y": "mu",
            "x": "pct_poverty", "controls": None, "wls": False}
    assert common.compute_experiment_id(base) == common.compute_experiment_id(dict(base))
    assert common.compute_experiment_id(base) != common.compute_experiment_id(
        dict(base, controls=["borough"]))
    assert common.compute_experiment_id(base) != common.compute_experiment_id(
        dict(base, wls=True))
    assert common.compute_experiment_id(base) != common.compute_experiment_id(
        dict(base, x="building_age"))


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture()
    def synthetic_run(self, tmp_path):
        """Run where unit 'quality' drives both judgments and a covariate."""
        rng = np.random.default_rng(42)
        n_units = 80
        quality = {f"u{i}": float(rng.normal(0, 1)) for i in range(n_units)}
        uids = list(quality)
        rows_pairs, rows_out = [], []
        for i in range(1500):
            ua, ub = rng.choice(uids, size=2, replace=False)
            latent = 1.5 * (quality[ua] - quality[ub])
            score = int(np.clip(round(latent + rng.normal(0, 0.8)), -2, 2))
            pid = f"pair_{i:06d}"
            rows_pairs.append({
                "pair_id": pid, "canonical_pair_id": pid,
                "unit_uid_a": ua, "unit_uid_b": ub,
                "unit_name_a": ua, "unit_name_b": ub,
            })
            rows_out.append({"pair_id": pid, "canonical_pair_id": pid,
                             "relative_score": score})
        pd.DataFrame(rows_pairs).to_parquet(tmp_path / "pairs.parquet")
        out_pq = tmp_path / "synthetic_run.parquet"
        pd.DataFrame(rows_out).to_parquet(out_pq)
        # Covariates: 'signal' tracks quality, 'noise' doesn't.
        pd.DataFrame({
            "uid": uids,
            "signal": [quality[u] * 10 + 50 + rng.normal(0, 1) for u in uids],
            "noise": rng.normal(0, 1, n_units),
            "boro": [("east" if i % 2 else "west") for i in range(n_units)],
        }).to_parquet(tmp_path / "cov.parquet")
        return tmp_path, out_pq

    def test_single_regression_end_to_end(self, synthetic_run, monkeypatch):
        tmp_path, out_pq = synthetic_run
        regfile = tmp_path / "registry.jsonl"
        argv = [
            "prog", str(out_pq), "--x", "signal",
            "--unit-metadata-parquet", str(tmp_path / "cov.parquet"),
            "--unit-metadata-id-column", "uid",
            "--registry", str(regfile), "--no-wandb",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        reg.main()

        records = common.read_registry(regfile)
        assert len(records) == 1
        rec = records[0]
        assert rec["mode"] == "regression"
        assert rec["results"]["r2"] > 0.5         # signal explains ratings
        assert rec["results"]["p"] < 1e-6
        assert rec["results"]["pair_p"] < 1e-6    # pair-level agrees
        assert rec["results"]["pair_slope"] > 0
        assert Path(rec["report_md"]).exists()
        assert Path(rec["results_parquet"]).exists()

        # Dedupe + force semantics.
        reg.main()
        assert len(common.read_registry(regfile)) == 1
        monkeypatch.setattr(sys, "argv", argv + ["--force"])
        reg.main()
        assert len(common.read_registry(regfile)) == 2

    def test_controls_change_id_and_run(self, synthetic_run, monkeypatch):
        tmp_path, out_pq = synthetic_run
        regfile = tmp_path / "registry_ctrl.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "prog", str(out_pq), "--x", "signal", "--controls", "boro",
            "--unit-metadata-parquet", str(tmp_path / "cov.parquet"),
            "--unit-metadata-id-column", "uid",
            "--registry", str(regfile), "--no-wandb",
        ])
        reg.main()
        rec = common.read_registry(regfile)[0]
        assert rec["controls"] == ["boro"]
        assert rec["results"]["partial_r2"] > 0.5  # signal survives the control

    def test_screen_end_to_end(self, synthetic_run, monkeypatch):
        tmp_path, out_pq = synthetic_run
        regfile = tmp_path / "registry_screen.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "prog", str(out_pq), "--x-list", "signal,noise",
            "--unit-metadata-parquet", str(tmp_path / "cov.parquet"),
            "--unit-metadata-id-column", "uid",
            "--registry", str(regfile), "--no-wandb",
        ])
        reg.main()
        rec = common.read_registry(regfile)[0]
        assert rec["mode"] == "screen"
        assert rec["results"]["best_x"] == "signal"
        results = pd.read_parquet(rec["results_parquet"])
        assert {"p_adj", "pair_p_adj"}.issubset(results.columns)
        by_x = results.set_index("x")
        assert by_x.loc["signal", "p_adj"] < 0.01
        assert by_x.loc["noise", "r2"] < 0.05
