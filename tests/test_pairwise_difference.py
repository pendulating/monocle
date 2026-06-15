"""Tests for scripts/pairwise_vqa_difference_report.py.

Covers the statistical core (orientation sign, repeat collapse, planted
effect vs null), the group-resolution paths (surfaced pair metadata vs
external unit-metadata join), and the experiment registry (id determinism,
dedupe, corrupt-line tolerance). No W&B / network involved.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pairwise_vqa_difference_report as diff  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair_df(rows: list[dict]) -> pd.DataFrame:
    """Minimal merged frame as produced by diff.load_run + attach_groups."""
    df = pd.DataFrame(rows)
    if "canonical_pair_id" not in df.columns:
        df["canonical_pair_id"] = [f"pair_{i:06d}" for i in range(len(df))]
    return df


def _synthetic_run(
    n_pairs: int,
    *,
    effect: float,
    seed: int = 0,
    n_units_per_group: int = 40,
) -> pd.DataFrame:
    """Random X-vs-Y pairs with a planted preference for group X.

    ``effect`` shifts the latent score difference; 0.0 → null.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        ua = f"x{rng.integers(n_units_per_group)}"
        ub = f"y{rng.integers(n_units_per_group)}"
        ga, gb = "X", "Y"
        if rng.random() < 0.5:  # randomize which group sits on canonical side A
            ua, ub, ga, gb = ub, ua, gb, ga
        latent = effect if ga == "X" else -effect
        score = int(np.clip(round(latent + rng.normal(0, 1.2)), -2, 2))
        rows.append({
            "pair_id": f"p{i}",
            "canonical_pair_id": f"p{i}",
            "unit_uid_a": ua,
            "unit_uid_b": ub,
            "__group_a__": ga,
            "__group_b__": gb,
            "relative_score": score,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orientation + collapse
# ---------------------------------------------------------------------------


class TestHeadToHead:
    def test_orientation_sign(self):
        # A=Chinese preferred in both presentations: score +2 when Chinese is
        # side A, score -2 when Chinese is side B. Both must orient to +2.
        df = _make_pair_df([
            {"pair_id": "p1", "unit_uid_a": "c1", "unit_uid_b": "i1",
             "__group_a__": "Chinese", "__group_b__": "Italian", "relative_score": 2},
            {"pair_id": "p2", "unit_uid_a": "i2", "unit_uid_b": "c2",
             "__group_a__": "Italian", "__group_b__": "Chinese", "relative_score": -2},
        ])
        res = diff.head_to_head_test(df, "Chinese", "Italian")
        assert res["n_direct_pairs"] == 2
        assert res["mean_oriented"] == pytest.approx(2.0)
        # And the mirror comparison flips the sign.
        res_rev = diff.head_to_head_test(df, "Italian", "Chinese")
        assert res_rev["mean_oriented"] == pytest.approx(-2.0)

    def test_repeat_collapse(self):
        # 3 repeats of one canonical pair + 1 singleton → n_direct_pairs == 2,
        # repeat group averaged (2, 0, 1 → 1.0).
        df = _make_pair_df([
            {"pair_id": "p1", "canonical_pair_id": "c", "unit_uid_a": "a", "unit_uid_b": "b",
             "__group_a__": "X", "__group_b__": "Y", "relative_score": 2},
            {"pair_id": "p1_r1", "canonical_pair_id": "c", "unit_uid_a": "a", "unit_uid_b": "b",
             "__group_a__": "X", "__group_b__": "Y", "relative_score": 0},
            {"pair_id": "p1_r2", "canonical_pair_id": "c", "unit_uid_a": "a", "unit_uid_b": "b",
             "__group_a__": "X", "__group_b__": "Y", "relative_score": 1},
            {"pair_id": "p2", "canonical_pair_id": "d", "unit_uid_a": "a2", "unit_uid_b": "b2",
             "__group_a__": "X", "__group_b__": "Y", "relative_score": -1},
        ])
        res = diff.head_to_head_test(df, "X", "Y")
        assert res["n_direct_obs"] == 4
        assert res["n_direct_pairs"] == 2
        assert res["mean_oriented"] == pytest.approx((1.0 + (-1.0)) / 2)

    def test_no_direct_pairs(self):
        df = _make_pair_df([
            {"pair_id": "p1", "unit_uid_a": "a", "unit_uid_b": "b",
             "__group_a__": "X", "__group_b__": "X", "relative_score": 1},
        ])
        res = diff.head_to_head_test(df, "X", "Y")
        assert res["n_direct_pairs"] == 0
        assert np.isnan(res["h2h_p"])

    def test_planted_effect_significant(self):
        df = _synthetic_run(800, effect=0.8, seed=42)
        res = diff.head_to_head_test(df, "X", "Y")
        assert res["mean_oriented"] > 0
        assert res["h2h_p"] < 1e-6
        assert res["win_rate"] > 0.5

    def test_null_not_significant(self):
        df = _synthetic_run(800, effect=0.0, seed=7)
        res = diff.head_to_head_test(df, "X", "Y")
        assert res["h2h_p"] > 0.01  # generous: a null should not be wildly significant


class TestRatingLevel:
    def test_planted_effect(self):
        df = _synthetic_run(2000, effect=0.8, seed=3)
        ratings = diff._compute_trueskill(df, draw_prob=0.05)
        unit_groups = diff.build_unit_group_map(df)
        res = diff.rating_level_test(ratings, unit_groups, "X", "Y")
        assert res["n_units_a"] > 0 and res["n_units_b"] > 0
        assert res["delta_mu"] > 0
        assert res["rating_p"] < 0.01
        assert res["cliffs_delta"] > 0

    def test_symmetry(self):
        df = _synthetic_run(500, effect=0.5, seed=11)
        ratings = diff._compute_trueskill(df, draw_prob=0.05)
        unit_groups = diff.build_unit_group_map(df)
        ab = diff.rating_level_test(ratings, unit_groups, "X", "Y")
        ba = diff.rating_level_test(ratings, unit_groups, "Y", "X")
        assert ab["delta_mu"] == pytest.approx(-ba["delta_mu"])
        assert ab["rating_p"] == pytest.approx(ba["rating_p"])


# ---------------------------------------------------------------------------
# Group resolution
# ---------------------------------------------------------------------------


class TestGroupResolution:
    def test_groups_from_pairs_metadata(self):
        df = pd.DataFrame({
            "pair_id": ["p1"],
            "unit_uid_a": ["a"], "unit_uid_b": ["b"],
            "cuisine_a": [" Chinese "], "cuisine_b": ["nan"],
            "relative_score": [1],
        })
        out = diff.attach_groups(
            df, group_column="cuisine", group_from_pairs=True,
            unit_metadata_parquet=None, unit_metadata_id_column="unit_uid",
        )
        assert out["__group_a__"].iloc[0] == "Chinese"  # stripped
        assert pd.isna(out["__group_b__"].iloc[0])      # 'nan' → missing

    def test_external_join(self, tmp_path):
        meta_pq = tmp_path / "meta.parquet"
        pd.DataFrame({
            "camis": [101, 202],
            "cuisine_description": ["Chinese", "Italian"],
        }).to_parquet(meta_pq)
        df = pd.DataFrame({
            "pair_id": ["p1"],
            "unit_uid_a": ["101"], "unit_uid_b": ["202"],
            "relative_score": [1],
        })
        out = diff.attach_groups(
            df, group_column="cuisine_description", group_from_pairs=False,
            unit_metadata_parquet=meta_pq, unit_metadata_id_column="camis",
        )
        assert out["__group_a__"].iloc[0] == "Chinese"
        assert out["__group_b__"].iloc[0] == "Italian"

    def test_missing_join_errors(self):
        df = pd.DataFrame({"pair_id": ["p1"], "unit_uid_a": ["a"],
                           "unit_uid_b": ["b"], "relative_score": [1]})
        with pytest.raises(SystemExit, match="unit-metadata"):
            diff.attach_groups(
                df, group_column="cuisine", group_from_pairs=False,
                unit_metadata_parquet=None, unit_metadata_id_column="unit_uid",
            )

    def test_resolve_group_name_case_insensitive(self):
        observed = pd.Series(["Chinese", "Chinese", "Italian"])
        assert diff.resolve_group_name("chinese", observed) == "Chinese"
        with pytest.raises(SystemExit, match="not found"):
            diff.resolve_group_name("Klingon", observed)

    def test_unit_group_map_conflict_keeps_first(self, capsys):
        df = pd.DataFrame({
            "unit_uid_a": ["u1", "u1"], "unit_uid_b": ["u2", "u3"],
            "__group_a__": ["X", "Y"], "__group_b__": ["X", "X"],
        })
        m = diff.build_unit_group_map(df)
        assert m["u1"] == "X"
        assert ">1 group" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Multiple-comparison adjustment
# ---------------------------------------------------------------------------


def test_adjust_pvalues_handles_nans():
    res = pd.DataFrame({"h2h_p": [0.001, np.nan, 0.04], "rating_p": [0.5, 0.6, np.nan]})
    out = diff.adjust_pvalues(res, ["h2h_p", "rating_p"])
    assert np.isnan(out["h2h_p_adj"].iloc[1])
    assert out["h2h_p_adj"].iloc[0] <= out["h2h_p_adj"].iloc[2]
    assert np.isfinite(out["rating_p_adj"].iloc[0])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_experiment_id_deterministic_and_order_invariant(self):
        a = {"source_parquet": "/x.parquet", "groups": ["Chinese", "Italian"], "mode": "pair"}
        b = {"mode": "pair", "groups": ["Chinese", "Italian"], "source_parquet": "/x.parquet"}
        assert diff.compute_experiment_id(a) == diff.compute_experiment_id(b)
        c = dict(a, groups=["Chinese", "Japanese"])
        assert diff.compute_experiment_id(a) != diff.compute_experiment_id(c)

    def test_append_find_roundtrip(self, tmp_path):
        reg = tmp_path / "registry.jsonl"
        rec = {"experiment_id": "abc123", "mode": "pair", "results": {"h2h_p": 0.01}}
        diff.append_registry(reg, rec)
        diff.append_registry(reg, {"experiment_id": "def456", "mode": "matrix"})
        records = diff.read_registry(reg)
        assert len(records) == 2
        assert diff.find_in_registry(records, "abc123")["results"]["h2h_p"] == 0.01
        assert diff.find_in_registry(records, "zzz") is None

    def test_latest_record_wins(self, tmp_path):
        reg = tmp_path / "registry.jsonl"
        diff.append_registry(reg, {"experiment_id": "abc", "v": 1})
        diff.append_registry(reg, {"experiment_id": "abc", "v": 2})
        assert diff.find_in_registry(diff.read_registry(reg), "abc")["v"] == 2

    def test_corrupt_lines_tolerated(self, tmp_path, capsys):
        reg = tmp_path / "registry.jsonl"
        reg.write_text(
            json.dumps({"experiment_id": "ok1"}) + "\n"
            + "{not json}\n"
            + json.dumps({"experiment_id": "ok2"}) + "\n"
        )
        records = diff.read_registry(reg)
        assert [r["experiment_id"] for r in records] == ["ok1", "ok2"]
        assert "unparseable" in capsys.readouterr().out

    def test_missing_registry_is_empty(self, tmp_path):
        assert diff.read_registry(tmp_path / "nope.jsonl") == []

    def test_registry_path_resolution(self, tmp_path, monkeypatch):
        cli = tmp_path / "cli.jsonl"
        assert diff.registry_path(cli) == cli
        monkeypatch.setenv("MLLMSCI_DIFFTEST_REGISTRY", str(tmp_path / "env.jsonl"))
        assert diff.registry_path(None) == tmp_path / "env.jsonl"
        monkeypatch.delenv("MLLMSCI_DIFFTEST_REGISTRY")
        assert diff.registry_path(None) == diff.DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# End-to-end CLI (synthetic data, no W&B, isolated registry)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture()
    def synthetic_run_dir(self, tmp_path):
        rng = np.random.default_rng(99)
        n = 600
        units = {f"u{i}": ("Chinese" if i % 2 == 0 else "Italian") for i in range(60)}
        uids = list(units)
        rows_pairs, rows_out = [], []
        for i in range(n):
            ua, ub = rng.choice(uids, size=2, replace=False)
            ga, gb = units[ua], units[ub]
            latent = 0.9 if ga == "Chinese" and gb == "Italian" else (
                -0.9 if ga == "Italian" and gb == "Chinese" else 0.0)
            score = int(np.clip(round(latent + rng.normal(0, 1.0)), -2, 2))
            pid = f"unit_{i:08d}"
            rows_pairs.append({
                "pair_id": pid, "canonical_pair_id": pid, "repeat_idx": 0,
                "unit_uid_a": ua, "unit_uid_b": ub,
                "unit_name_a": ua, "unit_name_b": ub,
                "image_path_a": "/dev/null", "image_path_b": "/dev/null",
                "is_swapped": False, "presented_order": "A_then_B",
            })
            rows_out.append({
                "pair_id": pid, "canonical_pair_id": pid,
                "relative_score": score, "relative_label": "n/a",
                "model_reasoning": "",
            })
        pd.DataFrame(rows_pairs).to_parquet(tmp_path / "pairs.parquet")
        out_pq = tmp_path / "synthetic_run.parquet"
        pd.DataFrame(rows_out).to_parquet(out_pq)
        meta = pd.DataFrame({
            "camis": uids, "cuisine_description": [units[u] for u in uids],
        })
        meta.to_parquet(tmp_path / "meta.parquet")
        return tmp_path, out_pq

    def test_single_mode_end_to_end(self, synthetic_run_dir, monkeypatch):
        tmp_path, out_pq = synthetic_run_dir
        reg = tmp_path / "registry.jsonl"
        argv = [
            "prog", str(out_pq),
            "--group-column", "cuisine_description",
            "--unit-metadata-parquet", str(tmp_path / "meta.parquet"),
            "--unit-metadata-id-column", "camis",
            "--group-a", "chinese", "--group-b", "Italian",
            "--registry", str(reg), "--no-wandb",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        diff.main()

        records = diff.read_registry(reg)
        assert len(records) == 1
        rec = records[0]
        assert rec["mode"] == "pair"
        assert rec["group_a"] == "Chinese"  # case-insensitive resolution
        assert rec["results"]["h2h_p"] < 0.01
        assert rec["results"]["mean_oriented"] > 0
        assert Path(rec["report_md"]).exists()
        assert Path(rec["results_parquet"]).exists()

        # Dedupe: a second invocation must not append.
        diff.main()
        assert len(diff.read_registry(reg)) == 1

        # --force reruns and appends.
        monkeypatch.setattr(sys, "argv", argv + ["--force"])
        diff.main()
        assert len(diff.read_registry(reg)) == 2

    def test_multi_model_end_to_end(self, tmp_path, monkeypatch):
        """Two fake model runs in a layout-1 aggregation dir: one with a
        planted Chinese-preference, one with the opposite. Replication
        summary must reflect the 1/2 split in direction."""
        rng = np.random.default_rng(5)
        units = {f"u{i}": ("Chinese" if i % 2 == 0 else "Italian") for i in range(40)}
        uids = list(units)
        pd.DataFrame({
            "camis": uids, "cuisine_description": [units[u] for u in uids],
        }).to_parquet(tmp_path / "meta.parquet")

        agg_dir = tmp_path / "agg"
        for label, effect in (("model_pro_chinese", 1.0), ("model_pro_italian", -1.0)):
            base = agg_dir / label / "0"
            (base / ".hydra").mkdir(parents=True)
            (base / ".hydra" / "config.yaml").write_text(
                f"model:\n  model_source: /models/{label}\n"
            )
            pairwise = base / "outputs" / "pairwise"
            pairwise.mkdir(parents=True)
            rows_pairs, rows_out = [], []
            for i in range(400):
                ua, ub = rng.choice(uids, size=2, replace=False)
                ga, gb = units[ua], units[ub]
                latent = effect if (ga, gb) == ("Chinese", "Italian") else (
                    -effect if (ga, gb) == ("Italian", "Chinese") else 0.0)
                score = int(np.clip(round(latent + rng.normal(0, 0.8)), -2, 2))
                pid = f"unit_{i:08d}"
                rows_pairs.append({
                    "pair_id": pid, "canonical_pair_id": pid,
                    "unit_uid_a": ua, "unit_uid_b": ub,
                })
                rows_out.append({
                    "pair_id": pid, "canonical_pair_id": pid, "relative_score": score,
                })
            pd.DataFrame(rows_pairs).to_parquet(pairwise / "pairs.parquet")
            pd.DataFrame(rows_out).to_parquet(pairwise / f"{label}_out.parquet")

        reg = tmp_path / "registry_multi.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "prog", "--aggregation-dir", str(agg_dir),
            "--group-column", "cuisine_description",
            "--unit-metadata-parquet", str(tmp_path / "meta.parquet"),
            "--unit-metadata-id-column", "camis",
            "--group-a", "Chinese", "--group-b", "Italian",
            "--registry", str(reg), "--no-wandb",
        ])
        diff.main()

        records = diff.read_registry(reg)
        assert len(records) == 1
        rec = records[0]
        assert rec["n_models"] == 2
        assert sorted(rec["models"]) == ["model_pro_chinese", "model_pro_italian"]
        assert rec["results"]["n_toward_a_h2h"] == 1  # split directions
        assert rec["results"]["n_sig_h2h"] == 2       # both strongly significant

        results = pd.read_parquet(rec["results_parquet"])
        assert set(results["model_label"]) == {"model_pro_chinese", "model_pro_italian"}
        by_model = results.set_index("model_label")["mean_oriented"]
        assert by_model["model_pro_chinese"] > 0 > by_model["model_pro_italian"]

        report = Path(rec["report_md"]).read_text()
        assert "Replication summary" in report

    def test_matrix_mode_end_to_end(self, synthetic_run_dir, monkeypatch):
        tmp_path, out_pq = synthetic_run_dir
        reg = tmp_path / "registry_matrix.jsonl"
        monkeypatch.setattr(sys, "argv", [
            "prog", str(out_pq),
            "--group-column", "cuisine_description",
            "--unit-metadata-parquet", str(tmp_path / "meta.parquet"),
            "--unit-metadata-id-column", "camis",
            "--all-pairs", "--min-group-units", "5",
            "--registry", str(reg), "--no-wandb",
        ])
        diff.main()
        records = diff.read_registry(reg)
        assert len(records) == 1
        rec = records[0]
        assert rec["mode"] == "matrix"
        assert rec["results"]["n_pairs_tested"] == 1  # only 2 groups → 1 pair
        results = pd.read_parquet(rec["results_parquet"])
        assert {"h2h_p_adj", "rating_p_adj"}.issubset(results.columns)
