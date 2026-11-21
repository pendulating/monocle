"""Cutting the answer's view of the image. See monocle/attn_knockout.py."""
import pytest
import torch

from monocle import attn_knockout as ak


class TestMaskPatcher:
    def kwargs(self, dtype=torch.float32, seq=8):
        m = torch.zeros(1, 1, seq, seq, dtype=dtype)
        return {"attention_mask": m}, m

    def test_blocks_the_named_cells(self):
        kw, _ = self.kwargs()
        _, out = ak.MaskPatcher([7], [2, 3])(None, (), kw)
        m = out["attention_mask"]
        assert m[0, 0, 7, 2] == torch.finfo(torch.float32).min
        assert m[0, 0, 7, 3] == torch.finfo(torch.float32).min

    def test_leaves_every_other_cell_alone(self):
        kw, _ = self.kwargs()
        _, out = ak.MaskPatcher([7], [2])(None, (), kw)
        m = out["attention_mask"]
        assert m[0, 0, 7, 4] == 0
        assert m[0, 0, 6, 2] == 0

    def test_never_writes_into_the_shared_mask(self):
        """The loop hands ONE mask object to every block of a layer type."""
        kw, original = self.kwargs()
        ak.MaskPatcher([7], [2])(None, (), kw)
        assert torch.equal(original, torch.zeros_like(original))

    def test_handles_a_boolean_mask(self):
        kw, _ = self.kwargs(dtype=torch.bool)
        kw["attention_mask"] = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        _, out = ak.MaskPatcher([7], [2])(None, (), kw)
        assert out["attention_mask"][0, 0, 7, 2] == False   # noqa: E712
        assert out["attention_mask"][0, 0, 7, 3] == True    # noqa: E712

    def test_blocks_many_queries_at_once(self):
        kw, _ = self.kwargs()
        _, out = ak.MaskPatcher([5, 6], [1])(None, (), kw)
        m = out["attention_mask"]
        assert m[0, 0, 5, 1] == m[0, 0, 6, 1] == torch.finfo(torch.float32).min

    def test_raises_when_there_is_no_mask(self):
        with pytest.raises(RuntimeError, match="no attention_mask"):
            ak.MaskPatcher([0], [1])(None, (), {"attention_mask": None})

    def test_raises_when_the_mask_is_not_a_tensor(self):
        with pytest.raises(RuntimeError, match="not a tensor"):
            ak.MaskPatcher([0], [1])(None, (), {"attention_mask": {"a": 1}})


class TestArms:
    def table(self):
        block = list(range(5, 261))
        stream = {"text_before": [0, 1, 2, 3], "text_after": [262, 263, 264],
                  "boi": [4], "eoi": [261]}
        return ak.arms(block, stream, 265, 266), block

    def test_has_a_positive_control_on_the_question(self):
        t, _ = self.table()
        q, k = t["answer_question"]
        assert q == [265] and k == [262, 263, 264]

    def test_answer_patches_hides_every_patch(self):
        t, block = self.table()
        assert t["answer_patches"][1] == block

    def test_answer_last_hides_only_the_final_patch(self):
        t, block = self.table()
        assert t["answer_last"][1] == [block[-1]]

    def test_stream_patches_blinds_every_non_patch_row(self):
        t, block = self.table()
        q, k = t["stream_patches"]
        assert not set(q) & set(block)
        assert 265 in q and k == block

    def test_none_hides_nothing(self):
        t, _ = self.table()
        assert t["none"][1] == []

    def test_control_floor_is_meaningful(self):
        assert ak.CONTROL_FLOOR >= 1.0

    def test_ranges_split_at_the_handover(self):
        """The swap puts the handover between L22 and L30."""
        assert "0-21" in ak.DEFAULT_RANGES and "30-47" in ak.DEFAULT_RANGES

    def test_the_control_range_covers_every_block(self):
        assert "0-47" in ak.DEFAULT_RANGES


class TestParseRange:
    def test_expands_a_span(self):
        assert ak.parse_range("30-33") == [30, 31, 32, 33]

    def test_accepts_a_single_block(self):
        assert ak.parse_range("47") == [47]

    def test_covers_all_48_blocks(self):
        assert len(ak.parse_range("0-47")) == 48
