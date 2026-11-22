"""The position-by-layer judgment trace. See research/monocle/judgment_trace.py."""
import numpy as np
import pandas as pd
import pytest
import torch

from research.monocle import judgment_trace as jt
from research.monocle.attn_route import BOI_TOKEN_ID, EOI_TOKEN_ID

IMG = 258880


def ids(n_patches=4, n_before=2, n_after=3):
    """text, <boi>, patches, <eoi>, question, answer."""
    seq = ([7] * n_before + [BOI_TOKEN_ID] + [IMG] * n_patches
           + [EOI_TOKEN_ID] + [8] * n_after + [9])
    return torch.tensor(seq)


class TestPositionGroups:
    def block(self, n_before=2, n_patches=4):
        return list(range(n_before + 1, n_before + 1 + n_patches))

    def test_every_position_lands_in_one_group(self):
        x = ids()
        g = jt.position_groups(x, self.block(), int(x.shape[0]) - 1)
        seen = [p for v in g.values() for p in v]
        assert sorted(seen) == list(range(int(x.shape[0])))

    def test_no_position_lands_in_two_groups(self):
        x = ids()
        g = jt.position_groups(x, self.block(), int(x.shape[0]) - 1)
        seen = [p for v in g.values() for p in v]
        assert len(seen) == len(set(seen))

    def test_the_image_group_is_the_patch_block(self):
        x = ids()
        g = jt.position_groups(x, self.block(), int(x.shape[0]) - 1)
        assert g["image"] == [3, 4, 5, 6]

    def test_the_answer_is_the_last_token_alone(self):
        x = ids()
        last = int(x.shape[0]) - 1
        g = jt.position_groups(x, self.block(), last)
        assert g["answer"] == [last]

    def test_the_question_sits_after_the_image(self):
        x = ids()
        g = jt.position_groups(x, self.block(), int(x.shape[0]) - 1)
        assert g["text_after"] == [8, 9, 10]

    def test_the_delimiters_are_their_own_groups(self):
        x = ids()
        g = jt.position_groups(x, self.block(), int(x.shape[0]) - 1)
        assert g["boi"] == [2] and g["eoi"] == [7]

    def test_a_short_block_is_refused(self):
        """The dropped patches would land in `text_after` and read as words."""
        x = ids()
        with pytest.raises(ValueError, match="image tokens"):
            jt.position_groups(x, [3, 4], int(x.shape[0]) - 1, IMG)

    def test_a_block_over_a_text_position_is_refused(self):
        x = ids()
        with pytest.raises(ValueError, match="image tokens"):
            jt.position_groups(x, [3, 4, 5, 6, 8], int(x.shape[0]) - 1, IMG)

    def test_coverage_alone_would_have_missed_it(self):
        """Why the token check exists: the short block still covers."""
        x = ids()
        g = dict(jt.psw.stream_groups(x, [3, 4], int(x.shape[0]) - 1))
        g["image"] = [3, 4]
        g["answer"] = [int(x.shape[0]) - 1]
        seen = sorted(p for v in g.values() for p in v)
        assert seen == list(range(int(x.shape[0])))


class TestLensAgreement:
    def frame(self, a, b):
        rows = []
        for lens, vals in (("mm", a), ("wikitext", b)):
            for p, v in enumerate(vals):
                rows.append({"lens": lens, "layer": 24, "pos": p,
                             "s_centered": v})
        return pd.DataFrame(rows)

    def test_two_identical_lenses_agree(self):
        v = list(np.random.default_rng(0).normal(size=32))
        out = jt.lens_agreement(self.frame(v, v))
        assert out.pearson.iloc[0] == pytest.approx(1.0)
        assert bool(out.trustworthy.iloc[0])

    def test_two_unrelated_lenses_are_marked_corpus(self):
        rng = np.random.default_rng(0)
        out = jt.lens_agreement(
            self.frame(list(rng.normal(size=64)), list(rng.normal(size=64))))
        assert not bool(out.trustworthy.iloc[0])

    def test_one_lens_gives_no_agreement_frame(self):
        df = pd.DataFrame([{"lens": "mm", "layer": 24, "pos": 0,
                            "s_centered": 1.0}])
        assert jt.lens_agreement(df).empty

    def test_the_floor_is_strict_enough_to_be_useful(self):
        assert jt.AGREE_FLOOR >= 0.8


class TestByGroup:
    def test_it_reports_the_largest_swing_in_a_group(self):
        df = pd.DataFrame([
            {"lens": "mm", "layer": 30, "group": "image", "pos": 0,
             "s": 1.0, "s_centered": -4.0},
            {"lens": "mm", "layer": 30, "group": "image", "pos": 1,
             "s": 2.0, "s_centered": 1.0}])
        out = jt.by_group(df)
        assert out.centered_absmax.iloc[0] == pytest.approx(4.0)

    def test_it_keeps_the_group_size(self):
        df = pd.DataFrame([
            {"lens": "mm", "layer": 30, "group": "image", "pos": p,
             "s": 0.0, "s_centered": 0.0} for p in range(5)])
        assert jt.by_group(df).n.iloc[0] == 5


class TestContract:
    def test_the_final_layer_needs_no_transport(self):
        """47 feeds the head directly, so it is the end-point check."""
        assert jt.FINAL_LAYER == 47

    def test_the_end_tolerance_beats_the_bf16_grid(self):
        assert jt.END_TOL < 0.125

    def test_two_lenses_are_configured(self):
        assert len(jt.LENSES) >= 2
