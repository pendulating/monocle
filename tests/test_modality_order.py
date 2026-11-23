"""Tests for the modality-order probe. No GPU, no model.

The probe answers one question: does the question reach the image patches when
the text comes first? These tests cover the 2 parts that can be wrong without
a model — the prompt order, and the 3 statistics.
"""
from __future__ import annotations

import unittest

import numpy as np

from research.monocle import modality_order as mo  # noqa: E402
from research.monocle import single as sg  # noqa: E402


class TestLayout(unittest.TestCase):
    """The order of the 2 content blocks, and the default."""

    def test_default_is_the_production_order(self):
        content = sg.build_messages("subway_safety")[0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[1]["type"], "text")

    def test_image_first_matches_the_default(self):
        a = sg.build_messages("subway_safety")
        b = sg.build_messages("subway_safety", layout="image_first")
        self.assertEqual(a, b)

    def test_text_first_reverses_the_order(self):
        content = sg.build_messages(
            "subway_safety", layout="text_first")[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image")

    def test_the_text_is_the_same_under_both_layouts(self):
        """Only the ORDER may move. A different question voids the probe."""
        a = sg.build_messages("subway_safety")[0]["content"]
        b = sg.build_messages(
            "subway_safety", layout="text_first")[0]["content"]
        self.assertEqual(a[1]["text"], b[0]["text"])

    def test_an_unknown_layout_raises(self):
        with self.assertRaises(ValueError):
            sg.build_messages("subway_safety", layout="images_then_labels")

    def test_the_module_names_every_layout(self):
        self.assertEqual(
            set(sg.LAYOUTS),
            {"image_first", "text_first", "neutral_first", "question_first"})
        self.assertEqual(set(sg.FILLER_LAYOUTS),
                         {"neutral_first", "question_first"})


class TestStatistics(unittest.TestCase):
    """The 3 numbers, and the trap the centered one avoids."""

    def setUp(self):
        rng = np.random.default_rng(777)
        self.a = rng.normal(size=(16, 64))

    def test_the_gap_is_zero_for_an_identical_pair(self):
        self.assertEqual(mo.relative_gap(self.a, self.a), 0.0)

    def test_the_centered_cosine_is_one_for_an_identical_pair(self):
        self.assertAlmostEqual(mo.centered_cosine(self.a, self.a), 1.0, 6)

    def test_the_gap_grows_with_the_change(self):
        rng = np.random.default_rng(1)
        small = self.a + 0.01 * rng.normal(size=self.a.shape)
        large = self.a + 1.00 * rng.normal(size=self.a.shape)
        self.assertLess(mo.relative_gap(self.a, small),
                        mo.relative_gap(self.a, large))

    def test_a_massive_activation_fools_the_raw_cosine(self):
        """The reason the report reads the gap and the centered cosine.

        Add one huge shared channel to 2 UNRELATED sets of patches. The raw
        cosine goes to about 1. The centered cosine does not move.
        """
        rng = np.random.default_rng(2)
        b = rng.normal(size=self.a.shape)
        big = np.zeros_like(self.a)
        big[:, 0] = 500.0
        raw_before = mo.raw_cosine(self.a, b)
        raw_after = mo.raw_cosine(self.a + big, b + big)
        cen_after = mo.centered_cosine(self.a + big, b + big)
        self.assertGreater(raw_after, 0.99)
        self.assertLess(raw_before, 0.5)
        self.assertLess(cen_after, 0.5)

    def test_the_gap_survives_a_massive_activation(self):
        """A shared huge channel must not drive the gap to 0 either.

        It does shrink the gap, because the denominator grows. The test is
        that an unrelated pair still scores far above an identical one.
        """
        rng = np.random.default_rng(3)
        b = rng.normal(size=self.a.shape)
        big = np.zeros_like(self.a)
        big[:, 0] = 500.0
        self.assertGreater(mo.relative_gap(self.a + big, b + big), 0.0)
        self.assertEqual(mo.relative_gap(self.a + big, self.a + big), 0.0)

    def test_a_zero_reference_gives_nan_and_does_not_raise(self):
        z = np.zeros((4, 8))
        self.assertTrue(np.isnan(mo.relative_gap(z, z + 1)))
        self.assertTrue(np.isnan(mo.centered_cosine(z, z)))


class TestCompare(unittest.TestCase):
    """The row builder: 1 row per layer, both halves scored."""

    def test_one_row_for_each_layer_with_both_halves(self):
        rng = np.random.default_rng(4)
        layers = (12, 18)
        own = {ly: {"h": rng.normal(size=(8, 32)),
                    "g": rng.normal(size=(8, 32))} for ly in layers}
        other = {ly: {"h": own[ly]["h"].copy(),
                      "g": rng.normal(size=(8, 32))} for ly in layers}
        rows = mo.compare(own, other, layers)
        self.assertEqual([r["layer"] for r in rows], list(layers))
        for r in rows:
            # `h` is identical here, so its gap must be 0 and `g` must not be.
            self.assertEqual(r["h_gap"], 0.0)
            self.assertGreater(r["g_gap"], 0.0)
            for key in ("h_cos", "h_cos_raw", "g_cos", "g_cos_raw"):
                self.assertIn(key, r)


class TestFourArms(unittest.TestCase):
    """The 4 layouts, and the position match the decisive pair needs."""

    def test_every_layout_holds_the_question_once(self):
        for layout in sg.LAYOUTS:
            kw = {"filler": "F"} if layout in sg.FILLER_LAYOUTS else {}
            content = sg.build_messages(
                "libraries", layout=layout, **kw)[0]["content"]
            texts = [b["text"] for b in content if b["type"] == "text"]
            question = sg.prompt_text("libraries")
            self.assertEqual(texts.count(question), 1, layout)
            self.assertEqual(
                sum(b["type"] == "image" for b in content), 1, layout)

    def test_the_filler_layouts_put_the_image_in_the_middle(self):
        for layout in sg.FILLER_LAYOUTS:
            content = sg.build_messages(
                "libraries", layout=layout, filler="F")[0]["content"]
            self.assertEqual([b["type"] for b in content],
                             ["text", "image", "text"], layout)

    def test_the_decisive_pair_differs_only_by_which_block_leads(self):
        """`neutral_first` and `question_first` are mirror images.

        Same blocks, same count, reversed. So the patch positions and the
        distance to the answer match, and only the content order moves.
        """
        a = sg.build_messages(
            "libraries", layout="neutral_first", filler="F")[0]["content"]
        b = sg.build_messages(
            "libraries", layout="question_first", filler="F")[0]["content"]
        self.assertEqual(a[0]["text"], b[2]["text"])
        self.assertEqual(a[2]["text"], b[0]["text"])
        self.assertEqual(a[1], b[1])

    def test_a_filler_layout_without_a_filler_raises(self):
        for layout in sg.FILLER_LAYOUTS:
            with self.assertRaises(ValueError):
                sg.build_messages("libraries", layout=layout)

    def test_the_plain_layouts_need_no_filler(self):
        for layout in ("image_first", "text_first"):
            sg.build_messages("libraries", layout=layout)


class TestMatchedFiller(unittest.TestCase):
    """The filler must match the question in TOKENS, not in characters."""

    class FakeTok:
        """A whitespace tokenizer. Enough to exercise the correction loop."""

        def __call__(self, s, add_special_tokens=True):
            return {"input_ids": s.split()}

        def decode(self, ids):
            return " ".join(ids)

    def test_the_filler_matches_the_token_count(self):
        tok = self.FakeTok()
        target = "one two three four five six seven"
        got = mo.matched_filler(tok, target, unit="pad pad ")
        self.assertEqual(len(tok(got)["input_ids"]),
                         len(tok(target)["input_ids"]))

    def test_a_short_unit_still_reaches_a_long_target(self):
        tok = self.FakeTok()
        target = " ".join(["w"] * 60)
        got = mo.matched_filler(tok, target, unit="p ")
        self.assertEqual(len(tok(got)["input_ids"]), 60)


class TestContrasts(unittest.TestCase):
    """The contrast list must decompose the reversal without a gap."""

    def test_the_three_parts_chain_into_the_whole(self):
        parts = [(a, b) for a, b, _ in mo.CONTRASTS[:3]]
        self.assertEqual(parts[0][0], "image_first")
        self.assertEqual(parts[-1][1], "text_first")
        for (_, end), (start, _) in zip(parts, parts[1:]):
            self.assertEqual(end, start)

    def test_the_last_contrast_is_the_whole_reversal(self):
        base, arm, _ = mo.CONTRASTS[-1]
        self.assertEqual((base, arm), ("image_first", "text_first"))


class TestDecoy(unittest.TestCase):
    """The decoy closes the limit the repeated filler leaves.

    The filler is 1 sentence, repeated, so it holds little information. The
    decoy is varied prose about an unrelated subject. It thus separates the
    QUESTION from any informative text that leads the image.
    """

    class FakeTok:
        """A whitespace tokenizer. Enough to exercise the correction loop."""

        def __call__(self, s, add_special_tokens=True):
            return {"input_ids": s.split()}

        def decode(self, ids):
            return " ".join(ids)

    def test_the_decoy_matches_the_token_count(self):
        tok = self.FakeTok()
        target = "one two three four five six seven"
        got = mo.matched_decoy(tok, target)
        self.assertEqual(len(tok(got)["input_ids"]),
                         len(tok(target)["input_ids"]))

    def test_a_target_longer_than_the_source_still_matches(self):
        """The source repeats until it is long enough. It must not fail."""
        tok = self.FakeTok()
        target = " ".join(["w"] * 400)
        self.assertEqual(len(tok(mo.matched_decoy(tok, target))["input_ids"]),
                         400)

    def test_the_decoy_is_not_the_filler(self):
        tok = self.FakeTok()
        target = " ".join(["w"] * 30)
        self.assertNotEqual(mo.matched_decoy(tok, target),
                            mo.matched_filler(tok, target, unit="pad pad "))

    def test_the_decoy_is_varied_and_the_filler_is_not(self):
        """The point of the arm. Count the distinct words in each block."""
        tok = self.FakeTok()
        target = " ".join(["w"] * 40)
        decoy = set(mo.matched_decoy(tok, target).split())
        filler = set(mo.matched_filler(tok, target).split())
        self.assertGreater(len(decoy), 4 * len(filler))

    def test_the_decoy_holds_no_word_of_the_rating_scale(self):
        """A rating word would make the decoy a weak question, not a control."""
        banned = {"safe", "safety", "unsafe", "dangerous", "clean", "dirty",
                  "busy", "quiet", "quality", "good", "bad", "better",
                  "worse", "street", "building", "city", "person", "people",
                  "photograph", "image", "picture"}
        words = {w.strip(".,").lower() for w in mo.DECOY_TEXT.split()}
        self.assertEqual(words & banned, set())

    def test_build_filler_dispatches_on_the_kind(self):
        tok = self.FakeTok()
        target = " ".join(["w"] * 20)
        self.assertEqual(mo.build_filler(tok, target, "neutral"),
                         mo.matched_filler(tok, target))
        self.assertEqual(mo.build_filler(tok, target, "decoy"),
                         mo.matched_decoy(tok, target))

    def test_an_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            mo.build_filler(self.FakeTok(), "a b c", "prose")

    def test_every_kind_gives_the_same_token_count(self):
        """The arms must share the geometry, or the comparison is void."""
        tok = self.FakeTok()
        target = " ".join(["w"] * 33)
        counts = {len(tok(mo.build_filler(tok, target, k))["input_ids"])
                  for k in mo.FILLER_KINDS}
        self.assertEqual(counts, {33})


if __name__ == "__main__":
    unittest.main()
