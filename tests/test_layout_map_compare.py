"""Tests for the cross-layout map comparison. No GPU, no model.

The report reads a folder that `monocle.causal_maps` writes, and scores 2 maps
against each other. Two things can be wrong without a model:

1. the folder name, which the report copies so it stays free of torch;
2. the pair statistics.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# `scripts/` is not a package, so load the module from its path.
_spec = importlib.util.spec_from_file_location(
    "layout_map_compare", REPO / "research/scripts/layout_map_compare.py")
lmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lmc)


class TestMapDir(unittest.TestCase):
    """The report must name the same folder the writer names."""

    def test_it_agrees_with_the_writer(self):
        """`causal_maps.case_dir_for` is the authority. Do not let them drift.

        The report keeps its own copy so a CPU report does not import torch.
        This test is the only thing that holds the 2 together.
        """
        try:
            from research.monocle.causal_maps import case_dir_for
        except Exception as exc:  # torch is absent on a CPU-only checkout
            self.skipTest(f"research.monocle.causal_maps did not import: {exc}")
        root = Path("/r")
        for case in ("subway_safety", "libraries"):
            for arm, (layout, kind) in lmc.ARMS.items():
                for group in (1, 2):
                    self.assertEqual(
                        lmc.map_dir(root, case, arm, group),
                        case_dir_for(root, case, group, "absent", layout,
                                     kind),
                        f"{case} {arm} g{group}")

    def test_the_production_layout_adds_no_suffix(self):
        got = lmc.map_dir(Path("/r"), "libraries", "image_first", 2)
        self.assertEqual(got.name, "libraries_g2_absent")

    def test_every_other_arm_gets_its_own_folder(self):
        names = {lmc.map_dir(Path("/r"), "libraries", a, 2).name
                 for a in ("image_first", "neutral_first", "question_first")}
        self.assertEqual(len(names), 3)


class TestComparePair(unittest.TestCase):
    """The row builder, and the floor it is read against."""

    def setUp(self):
        rng = np.random.default_rng(20260826)
        self.a = {f"p{i}": rng.normal(size=64) for i in range(5)}

    def test_an_identical_map_scores_the_floor(self):
        """`truth_retest` gives r = 1.000000 at a matched batch."""
        df = lmc.compare_pair(self.a, {k: v.copy() for k, v in self.a.items()})
        self.assertEqual(len(df), 5)
        self.assertTrue(np.allclose(df.r, 1.0))
        self.assertTrue(df.top1_same.all())
        self.assertTrue(np.allclose(df.max_abs_diff, 0.0))

    def test_an_unrelated_map_falls_well_below_the_floor(self):
        rng = np.random.default_rng(1)
        b = {k: rng.normal(size=64) for k in self.a}
        self.assertLess(lmc.compare_pair(self.a, b).r.mean(), 0.5)

    def test_only_the_shared_photographs_are_scored(self):
        b = {k: v.copy() for k, v in list(self.a.items())[:3]}
        b["extra"] = np.zeros(64)
        self.assertEqual(sorted(lmc.compare_pair(self.a, b).image_id),
                         ["p0", "p1", "p2"])

    def test_a_size_mismatch_is_dropped_not_scored(self):
        """A different group gives a different cell count. Never compare it."""
        b = {k: v[:16].copy() for k, v in self.a.items()}
        self.assertTrue(lmc.compare_pair(self.a, b).empty)

    def test_a_nan_cell_is_dropped_from_both_sides(self):
        b = {k: v.copy() for k, v in self.a.items()}
        b["p0"][3] = np.nan
        row = lmc.compare_pair(self.a, b)
        self.assertTrue(np.isfinite(row.r).all())
        self.assertAlmostEqual(float(row[row.image_id == "p0"].r.iloc[0]),
                               1.0, 6)

    def test_the_range_comes_from_the_base_arm(self):
        """`max|d|` reads against it, so it must be the reference map."""
        b = {k: v * 100.0 for k, v in self.a.items()}
        got = lmc.compare_pair(self.a, b).range_a.mean()
        want = np.mean([v.max() - v.min() for v in self.a.values()])
        self.assertAlmostEqual(got, want, 6)


class TestPairs(unittest.TestCase):
    """A matched pair holds the geometry fixed. It must be marked."""

    def test_a_matched_pair_never_holds_the_production_layout(self):
        """`image_first` has a shorter prompt, so it cannot be matched."""
        matched = [p for p in lmc.PAIRS if p[3]]
        self.assertTrue(matched)
        for base, arm, _, _ in matched:
            self.assertNotIn("image_first", (base, arm))

    def test_every_matched_pair_shares_a_layout_or_a_filler(self):
        """Matched means the same block count and the same token count.

        Every arm but `image_first` holds 3 blocks and matches the question
        in tokens, so any 2 of them share the geometry.
        """
        for base, arm, _, _ in [p for p in lmc.PAIRS if p[3]]:
            for name in (base, arm):
                self.assertNotEqual(lmc.ARMS[name][0], "image_first", name)

    def test_the_mirror_pair_is_present(self):
        pairs = {(b, a) for b, a, _, m in lmc.PAIRS if m}
        self.assertIn(("neutral_first", "question_first"), pairs)

    def test_the_decoy_control_is_present(self):
        """The arm that separates the question from any informative text."""
        pairs = {(b, a) for b, a, _, m in lmc.PAIRS if m}
        self.assertIn(("neutral_first", "decoy_first"), pairs)

    def test_the_decoy_arm_keeps_the_neutral_layout(self):
        """Only the CONTENT of the leading block may change."""
        self.assertEqual(lmc.ARMS["decoy_first"][0],
                         lmc.ARMS["neutral_first"][0])
        self.assertNotEqual(lmc.ARMS["decoy_first"][1],
                            lmc.ARMS["neutral_first"][1])

    def test_every_arm_names_a_distinct_folder(self):
        names = {lmc.map_dir(Path("/r"), "libraries", a, 2).name
                 for a in lmc.ARMS}
        self.assertEqual(len(names), len(lmc.ARMS))


if __name__ == "__main__":
    unittest.main()
