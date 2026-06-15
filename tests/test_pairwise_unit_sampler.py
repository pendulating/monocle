"""Unit tests for dagspaces.urbanpairvqa.samplers.cyclomedia_pairs.build_unit_random_pairs."""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.urbanpairvqa.samplers.cyclomedia_pairs import (
    build_unit_random_pairs,
    build_global_random_pairs,
)


def _synth_manifest(n_units: int = 5, imgs_per_unit: int = 4) -> pd.DataFrame:
    rows = []
    for u in range(n_units):
        uid = f"U{u:04d}"
        for i in range(imgs_per_unit):
            rows.append({
                "sample_id": f"{uid}_img{i}",
                "image_path": f"/tmp/{uid}/img{i}.jpg",
                "unit_uid": uid,
                "unit_name": f"UNIT {u}",
                "face": "FBLR"[i % 4],
            })
    return pd.DataFrame(rows)


class TestUnitSampler:

    def test_happy_path(self):
        df = _synth_manifest(n_units=5, imgs_per_unit=4)
        out = build_unit_random_pairs(df, max_pairs=2, seed=1)
        assert len(out) == 2
        for _, r in out.iterrows():
            # Each pair is across two distinct units
            assert r["unit_uid_a"] != r["unit_uid_b"]
            # Chosen image must be from the claimed unit
            a_img_u = df[df["sample_id"] == r["sample_id_a"]]["unit_uid"].iloc[0]
            b_img_u = df[df["sample_id"] == r["sample_id_b"]]["unit_uid"].iloc[0]
            assert a_img_u == r["unit_uid_a"]
            assert b_img_u == r["unit_uid_b"]
            # Names carried through
            assert r["unit_name_a"] == f"UNIT {int(r['unit_uid_a'][1:])}"

    def test_seed_reproducible(self):
        df = _synth_manifest(n_units=10, imgs_per_unit=3)
        a = build_unit_random_pairs(df, max_pairs=4, seed=42)
        b = build_unit_random_pairs(df, max_pairs=4, seed=42)
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True),
        )

    def test_requires_two_units(self):
        df = _synth_manifest(n_units=1, imgs_per_unit=4)
        with pytest.raises(ValueError, match="≥ 2 distinct units"):
            build_unit_random_pairs(df, max_pairs=1, seed=0)

    def test_missing_unit_column(self):
        df = _synth_manifest(n_units=4).drop(columns=["unit_uid"])
        with pytest.raises(ValueError, match="unit_column"):
            build_unit_random_pairs(df, max_pairs=1, seed=0)

    def test_no_replacement_caps_at_canonical_pair_count(self):
        df = _synth_manifest(n_units=6, imgs_per_unit=2)
        out = build_unit_random_pairs(df, max_pairs=99, seed=3, allow_replacement=False)
        # max_pairs is clipped to C(6, 2) = 15 distinct canonical pairs.
        assert len(out) == 15
        # Every canonical pair is distinct.
        keys = {
            tuple(sorted([r["unit_uid_a"], r["unit_uid_b"]]))
            for _, r in out.iterrows()
        }
        assert len(keys) == 15

    def test_no_replacement_max_pairs_none_returns_perfect_matching(self):
        df = _synth_manifest(n_units=6, imgs_per_unit=2)
        out = build_unit_random_pairs(df, max_pairs=None, seed=3, allow_replacement=False)
        # max_pairs=None preserves the legacy perfect-matching default: 6 // 2 = 3.
        assert len(out) == 3
        # Each unit appears at most once across all pairs.
        units_used = list(out["unit_uid_a"]) + list(out["unit_uid_b"])
        assert len(units_used) == len(set(units_used))

    def test_no_replacement_honors_max_pairs_above_half_units(self):
        # Regression: previously max_pairs > n_units // 2 silently capped at
        # n_units // 2. Now it should yield up to C(n_units, 2) distinct pairs.
        df = _synth_manifest(n_units=10, imgs_per_unit=2)
        out = build_unit_random_pairs(df, max_pairs=20, seed=0, allow_replacement=False)
        assert len(out) == 20
        keys = {
            tuple(sorted([r["unit_uid_a"], r["unit_uid_b"]]))
            for _, r in out.iterrows()
        }
        assert len(keys) == 20  # all canonical pairs distinct

    def test_with_replacement_hits_target(self):
        df = _synth_manifest(n_units=4, imgs_per_unit=2)
        out = build_unit_random_pairs(df, max_pairs=20, seed=0, allow_replacement=True)
        assert len(out) == 20

    def test_repeat_observations_resample_within_unit(self):
        df = _synth_manifest(n_units=3, imgs_per_unit=5)
        out = build_unit_random_pairs(
            df, max_pairs=1, seed=7, repeat_count=4,
        )
        assert len(out) == 5
        # All observations share the same canonical unit pair
        assert out["canonical_pair_id"].nunique() == 1
        # …but the sampled images can differ across repeats.
        assert len(set(zip(out["sample_id_a"], out["sample_id_b"]))) >= 2

    def test_counterbalance_balanced_alternates(self):
        df = _synth_manifest(n_units=4, imgs_per_unit=2)
        out = build_unit_random_pairs(
            df, max_pairs=2, seed=0, counterbalance_mode="balanced",
        )
        # With 2 observations, balanced means one swapped, one not.
        assert set(out["is_swapped"].tolist()) == {False, True}

    def test_presented_order_flips_on_swap(self):
        df = _synth_manifest(n_units=4, imgs_per_unit=2)
        out = build_unit_random_pairs(
            df, max_pairs=4, seed=0, counterbalance_mode="balanced",
        )
        for _, r in out.iterrows():
            if r["is_swapped"]:
                assert r["presented_left_path"] == r["image_path_b"]
                assert r["presented_right_path"] == r["image_path_a"]
                assert r["presented_order"] == "B_then_A"
            else:
                assert r["presented_left_path"] == r["image_path_a"]
                assert r["presented_right_path"] == r["image_path_b"]
                assert r["presented_order"] == "A_then_B"


class TestWeightedWithinUnit:
    """weight_column biases within-unit image selection without disturbing
    which units get paired."""

    @staticmethod
    def _weighted_manifest(n_units: int = 3, imgs_per_unit: int = 4) -> pd.DataFrame:
        # Each unit has one "hot" image with weight ~1.0 and the rest at 0.01.
        rows = []
        for u in range(n_units):
            uid = f"U{u:04d}"
            for i in range(imgs_per_unit):
                rows.append({
                    "sample_id": f"{uid}_img{i}",
                    "image_path": f"/tmp/{uid}/img{i}.jpg",
                    "unit_uid": uid,
                    "unit_name": f"UNIT {u}",
                    "attribution_confidence": 1.0 if i == 0 else 0.01,
                })
        return pd.DataFrame(rows)

    def test_high_weight_image_dominates_within_unit(self):
        df = self._weighted_manifest()
        out = build_unit_random_pairs(
            df, max_pairs=600, seed=1, allow_replacement=True,
            weight_column="attribution_confidence",
        )
        # Count how often the unit's "hot" image (suffix _img0) is chosen,
        # pooling left- and right-side draws.
        draws = pd.concat([
            out[["unit_uid_a", "image_path_a"]].rename(columns={"unit_uid_a": "u", "image_path_a": "p"}),
            out[["unit_uid_b", "image_path_b"]].rename(columns={"unit_uid_b": "u", "image_path_b": "p"}),
        ])
        hot_fraction = draws["p"].str.endswith("img0.jpg").mean()
        # Ideal fraction with weights (1.0, 0.01, 0.01, 0.01) = 1/1.03 ≈ 0.97.
        # Uniform would be 0.25. Assert we're clearly in the weighted regime.
        assert hot_fraction > 0.85

    def test_missing_column_falls_back_to_uniform(self):
        df = self._weighted_manifest()
        out = build_unit_random_pairs(
            df, max_pairs=10, seed=0, allow_replacement=True,
            weight_column="does_not_exist",
        )
        assert len(out) == 10

    def test_per_unit_all_zero_weights_falls_back(self):
        df = self._weighted_manifest()
        # Zero out all weights for one unit.
        df.loc[df["unit_uid"] == "U0000", "attribution_confidence"] = 0.0
        out = build_unit_random_pairs(
            df, max_pairs=300, seed=0, allow_replacement=True,
            weight_column="attribution_confidence",
        )
        # The zeroed unit must still produce draws (uniform fallback).
        u0_draws = pd.concat([
            out.loc[out["unit_uid_a"] == "U0000", "image_path_a"],
            out.loc[out["unit_uid_b"] == "U0000", "image_path_b"],
        ])
        # All four images of U0000 should appear (uniform over a tiny pool).
        assert u0_draws.nunique() == 4

    def test_negative_and_nan_weights_clipped(self):
        df = self._weighted_manifest()
        df.loc[df["sample_id"] == "U0000_img1", "attribution_confidence"] = -5.0
        df.loc[df["sample_id"] == "U0000_img2", "attribution_confidence"] = float("nan")
        out = build_unit_random_pairs(
            df, max_pairs=300, seed=2, allow_replacement=True,
            weight_column="attribution_confidence",
        )
        u0_draws = pd.concat([
            out.loc[out["unit_uid_a"] == "U0000", "image_path_a"],
            out.loc[out["unit_uid_b"] == "U0000", "image_path_b"],
        ])
        # img1 and img2 were clipped to 0 → must never appear.
        assert not u0_draws.str.endswith("img1.jpg").any()
        assert not u0_draws.str.endswith("img2.jpg").any()


class TestImageModeStillWorks:
    """Smoke-check the existing image-mode sampler after surrounding edits."""

    def test_image_mode_happy(self):
        df = _synth_manifest(n_units=5, imgs_per_unit=4)
        out = build_global_random_pairs(df, max_pairs=5, seed=0)
        assert len(out) == 5
        assert "pair_id" in out.columns


class TestPersistPairs:
    """Pairs should land on disk alongside the stage output."""

    def test_persist_writes_parquet_and_meta(self, tmp_path):
        from dagspaces.urbanpairvqa.orchestrator import _persist_pairs
        import json

        df = _synth_manifest(n_units=4, imgs_per_unit=2)
        pairs = build_unit_random_pairs(df, max_pairs=2, seed=0)

        class _FakeCtx:
            pass
        ctx = _FakeCtx()
        results_path = tmp_path / "outputs" / "run.parquet"
        ctx.output_paths = {"results": str(results_path)}

        path = _persist_pairs(pairs, ctx, mode="unit")
        assert path == str(tmp_path / "outputs" / "pairs.parquet")
        assert (tmp_path / "outputs" / "pairs.parquet").is_file()
        meta = json.loads((tmp_path / "outputs" / "pairs.meta.json").read_text())
        assert meta["mode"] == "unit"
        assert meta["rows"] == 2
        assert "pair_id" in meta["columns"]
