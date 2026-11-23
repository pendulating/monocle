"""The folder name must separate a decoy run from a neutral run.

A run must never overwrite the run it is compared against. The decoy arm keeps
the layout `neutral_first`, so only the filler kind tells the 2 apart.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from research.monocle.causal_maps import case_dir_for  # noqa: E402


class TestFillerKindInTheName(unittest.TestCase):

    def test_the_decoy_gets_its_own_folder(self):
        a = case_dir_for(Path("/r"), "libraries", 2, "absent",
                         "neutral_first", "neutral")
        b = case_dir_for(Path("/r"), "libraries", 2, "absent",
                         "neutral_first", "decoy")
        self.assertNotEqual(a, b)
        self.assertEqual(b.name, a.name + "_decoy")

    def test_the_default_keeps_every_older_path(self):
        """Runs written before the decoy arm must keep their folder."""
        for layout in ("image_first", "neutral_first", "question_first"):
            self.assertEqual(
                case_dir_for(Path("/r"), "libraries", 2, "absent", layout),
                case_dir_for(Path("/r"), "libraries", 2, "absent", layout,
                             "neutral"))

    def test_all_three_arms_of_the_control_differ(self):
        arms = {case_dir_for(Path("/r"), "libraries", 2, "absent", ly, k).name
                for ly, k in (("neutral_first", "neutral"),
                              ("neutral_first", "decoy"),
                              ("question_first", "neutral"))}
        self.assertEqual(len(arms), 3)


if __name__ == "__main__":
    unittest.main()
