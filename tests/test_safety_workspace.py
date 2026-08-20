"""CPU-only unit tests for monocle.safety_workspace (phase-4 rung B).

No model, no GPU, no jlens at import: monocle.safety_workspace defers every
jlens / transformers / model import into its GPU code paths, so the module
imports in the plain `.venv` that lacks jlens. These tests exercise the pure
geometry, label-collapse, metric, and checkpoint logic on tiny tensors and
fake tokenizers.

Run: /share/pierson/matt/mllmsci/.venv/bin/python -m pytest \
        tests/test_safety_workspace.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf
import torch

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import safety_workspace as sw  # noqa: E402


# ---------------------------------------------------------------------------
# Fake tokenizers
# ---------------------------------------------------------------------------
class LabelTokenizer:
    """Fake tokenizer for label_first_tokens: `.encode` returns a scripted
    id list per label. MuchLess/MuchMore share their FIRST token ("Much" = 100)
    to reproduce the real collision."""

    ENCODE = {
        "MuchLess": [100, 7],
        "MuchMore": [100, 9],
        "Less": [7],
        "Same": [8],
        "More": [9],
    }
    TOKENS = {100: "Much", 7: "Less", 8: "Same", 9: "More"}

    def encode(self, text, add_special_tokens=False):
        return list(self.ENCODE[text])

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.TOKENS.get(i) for i in ids]
        return self.TOKENS.get(ids)


class VocabTokenizer:
    """Fake tokenizer over a fixed vocab list for vocab_ids_exact / mass. Mirrors
    tests/test_monocle.FakeTokenizer (convert_ids_to_tokens over range(len))."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.vocab[i] for i in ids]
        return self.vocab[ids]


# ---------------------------------------------------------------------------
# contiguous_runs / image_token_blocks
# ---------------------------------------------------------------------------
class TestImageTokenBlocks:
    def test_two_equal_blocks_ok(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 99, 99, 3, 99, 99, 4]])}
        blocks = sw.image_token_blocks(inputs, image_token_id=99)
        assert len(blocks) == 2
        assert blocks[0].tolist() == [2, 3]
        assert blocks[1].tolist() == [5, 6]

    def test_one_block_raises(self):
        inputs = {"input_ids": torch.tensor([[1, 99, 99, 99, 2]])}
        with pytest.raises(RuntimeError):
            sw.image_token_blocks(inputs, image_token_id=99)

    def test_ragged_blocks_raise(self):
        # runs [0,1] and [3] -> two runs but unequal length
        inputs = {"input_ids": torch.tensor([[99, 99, 5, 99, 6]])}
        with pytest.raises(RuntimeError):
            sw.image_token_blocks(inputs, image_token_id=99)

    def test_three_blocks_raise(self):
        inputs = {"input_ids": torch.tensor([[99, 1, 99, 1, 99]])}
        with pytest.raises(RuntimeError):
            sw.image_token_blocks(inputs, image_token_id=99)

    def test_zero_tokens_raise(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        with pytest.raises(RuntimeError):
            sw.image_token_blocks(inputs, image_token_id=99)

    def test_contiguous_runs_interleaved_singleton(self):
        # pure splitter: an interleaved singleton splits correctly (no assert)
        runs = sw.contiguous_runs(torch.tensor([0, 2, 3, 4]))
        assert [r.tolist() for r in runs] == [[0], [2, 3, 4]]

    def test_contiguous_runs_empty(self):
        assert sw.contiguous_runs(torch.tensor([], dtype=torch.long)) == []

    def test_contiguous_runs_single_run(self):
        runs = sw.contiguous_runs(torch.tensor([5, 6, 7]))
        assert [r.tolist() for r in runs] == [[5, 6, 7]]


# ---------------------------------------------------------------------------
# square_grid
# ---------------------------------------------------------------------------
class TestSquareGrid:
    def test_perfect_square(self):
        assert sw.square_grid(256) == (16, 16)
        assert sw.square_grid(64) == (8, 8)

    def test_non_square_needs_inputs(self):
        with pytest.raises(ValueError):
            sw.square_grid(130)

    def test_non_square_falls_back_to_aspect(self):
        # 12 patches, aspect 400/300 -> 3x4 (extract.infer_grid aspect strategy)
        rows, cols = sw.square_grid(12, inputs={}, orig_size=(400, 300))
        assert (rows, cols) == (3, 4)


# ---------------------------------------------------------------------------
# label first-tokens + Much* collapse
# ---------------------------------------------------------------------------
class TestLabelCollapse:
    # The registered runs all enable abstention, so the class set is 5-way.
    # `cfg` now drives which labels exist — see monocle/canonical.py.
    CFG_ABSTAIN = OmegaConf.create({"prompt": {"structured_output": {
        "enabled": True, "allow_not_sure": True, "not_sure_label": "NotSure"}}})
    CFG_NO_ABSTAIN = OmegaConf.create({"prompt": {"structured_output": {
        "enabled": True, "allow_not_sure": False}}})

    def test_collision_detected(self):
        label_first, class_ids, collision = sw.label_first_tokens(
            LabelTokenizer(), self.CFG_NO_ABSTAIN)
        assert collision is True
        assert label_first["MuchLess"] == label_first["MuchMore"] == 100
        assert class_ids == {"Much*": 100, "Less": 7, "Same": 8, "More": 9}

    def test_no_collision_when_distinct(self):
        tok = LabelTokenizer()
        tok.ENCODE = {**tok.ENCODE, "MuchMore": [55, 9]}  # distinct first token
        _, _, collision = sw.label_first_tokens(tok, self.CFG_NO_ABSTAIN)
        assert collision is False

    def test_abstention_joins_the_class_set(self):
        """The battery keeps abstention ON; the read must be able to see it."""
        tok = LabelTokenizer()
        tok.ENCODE = {**tok.ENCODE, "NotSure": [77, 3]}
        label_first, class_ids, _ = sw.label_first_tokens(
            tok, self.CFG_ABSTAIN)
        assert label_first["NotSure"] == 77
        assert class_ids["NotSure"] == 77
        assert set(class_ids) == {"Much*", "Less", "Same", "More", "NotSure"}

    def test_abstention_is_off_the_ordinal_scale(self):
        """An abstention is not a "Same" judgment — it scores NaN."""
        import math
        assert math.isnan(sw.collapsed_ordinal_agreement("NotSure", "More"))
        assert math.isnan(sw.collapsed_ordinal_agreement("More", "NotSure"))
        assert sw.collapsed_ordinal_agreement("More", "More") == 1.0

    def test_collapse_label(self):
        assert sw.collapse_label("MuchLess") == "Much*"
        assert sw.collapse_label("MuchMore") == "Much*"
        assert sw.collapse_label("Less") == "Less"
        assert sw.collapse_label("Same") == "Same"
        assert sw.collapse_label("More") == "More"

    def test_collapsed_index(self):
        assert sw.collapsed_index("Much*") == 0
        assert sw.collapsed_index("More") == 3

    def test_ordinal_agreement(self):
        # exact match -> 1.0
        assert sw.collapsed_ordinal_agreement("More", "More") == 1.0
        # Much* predicted, prod MuchLess collapses to Much* -> 1.0
        assert sw.collapsed_ordinal_agreement("Much*", "MuchMore") == 1.0
        # predict More(3) but prod Same(2) -> 1 - 1/4
        assert sw.collapsed_ordinal_agreement("More", "Same") == pytest.approx(0.75)
        # predict More(3) but prod MuchLess->Much*(0) -> 1 - 3/4
        assert sw.collapsed_ordinal_agreement("More", "MuchLess") == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# restricted argmax + rank + mass + position_metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    CLASS_IDS = {"Much*": 0, "Less": 1, "Same": 2, "More": 3}
    LABEL_FIRST = {"MuchLess": 0, "Less": 1, "Same": 2, "More": 3, "MuchMore": 0}

    def test_restricted_argmax_ignores_non_class_tokens(self):
        # token 5 is the global argmax but is NOT a class id -> restricted argmax
        # must pick among {0,1,2,3}; here class token 2 (Same) wins.
        logits = torch.tensor([0.1, 0.2, 4.0, 0.3, 0.0, 9.0])
        assert sw.restricted_argmax(logits, self.CLASS_IDS) == "Same"

    def test_token_rank(self):
        logits = torch.tensor([1.0, 3.0, 2.0, 5.0])
        assert sw.token_rank(logits, 3) == 0   # highest
        assert sw.token_rank(logits, 1) == 1
        assert sw.token_rank(logits, 0) == 3   # lowest

    def test_mass(self):
        probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
        ids = torch.tensor([1, 3])
        assert sw.mass(probs, ids) == pytest.approx(0.6)
        assert sw.mass(probs, torch.tensor([], dtype=torch.long)) == 0.0

    def test_position_metrics_prod_more(self):
        # 6-vocab; class ids 0..3, safety={4}, brightness={5}
        logits = torch.tensor([0.0, 0.0, 0.0, 6.0, -1.0, -1.0])
        m = sw.position_metrics(
            logits, prod_label="More", class_ids=self.CLASS_IDS,
            label_first=self.LABEL_FIRST,
            safety_ids=torch.tensor([4]), brightness_ids=torch.tensor([5]))
        assert m["argmax_class"] == "More"
        assert m["argmax_correct"] is True
        assert m["rank_prod_label"] == 0
        probs = torch.softmax(logits, dim=-1)
        assert m["p_prod_label"] == pytest.approx(float(probs[3]))
        assert m["safety_mass"] == pytest.approx(float(probs[4]))
        assert m["brightness_mass"] == pytest.approx(float(probs[5]))

    def test_position_metrics_muchless_collapses(self):
        # prod MuchLess -> first token id 0 (== Much*); a Much* argmax is correct
        logits = torch.tensor([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = sw.position_metrics(
            logits, prod_label="MuchLess", class_ids=self.CLASS_IDS,
            label_first=self.LABEL_FIRST,
            safety_ids=torch.tensor([4]), brightness_ids=torch.tensor([5]))
        assert m["argmax_class"] == "Much*"
        assert m["argmax_correct"] is True


# ---------------------------------------------------------------------------
# vocab-mass id resolution (exact display-form matching)
# ---------------------------------------------------------------------------
class TestVocabMassIds:
    def test_exact_match_union_and_no_substring(self):
        vocab = [
            "▁safe",     # 0 : display 'safe'  -> match
            "safe",           # 1 : display 'safe'  -> match
            "▁required", # 2 : contains 'safe'? no; guards substring trap anyway
            "▁safety",   # 3 : display 'safety' -> match
            "unsafe",         # 4 : display 'unsafe' -> NOT 'safe' exact
            "▁dog",      # 5 : unrelated
        ]
        tok = VocabTokenizer(vocab)
        ids = sw.vocab_mass_ids(tok, ["safe", "safety"])
        assert ids == [0, 1, 3]  # 'unsafe' excluded (exact, not substring)

    def test_empty_when_absent(self):
        tok = VocabTokenizer(["▁cat", "▁dog"])
        assert sw.vocab_mass_ids(tok, ["safe"]) == []


# ---------------------------------------------------------------------------
# checkpoint / resume round-trip
# ---------------------------------------------------------------------------
class TestCheckpoint:
    def _rows(self, pair_ids):
        return pd.DataFrame([
            {"pair_id": pid, "cond": "prod", "lens": "wikitext",
             "layer": 24, "pos": "label", "p_prod_label": 0.5}
            for pid in pair_ids])

    def test_completed_absent_is_empty(self, tmp_path):
        assert sw.completed_pair_ids(tmp_path / "nope.parquet") == set()

    def test_write_and_resume_roundtrip(self, tmp_path):
        path = tmp_path / "answer_depth.parquet"
        sw.write_long_frame(self._rows(["a", "b"]), path)
        assert sw.completed_pair_ids(path) == {"a", "b"}

        # append a third pair on top of the prior frame (the run's flush pattern)
        prior = pd.read_parquet(path)
        merged = pd.concat([prior, self._rows(["c"])], ignore_index=True)
        sw.write_long_frame(merged, path)
        assert sw.completed_pair_ids(path) == {"a", "b", "c"}
        # rows preserved, not clobbered
        assert len(pd.read_parquet(path)) == 3

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        path = tmp_path / "answer_depth.parquet"
        sw.write_long_frame(self._rows(["a"]), path)
        assert list(tmp_path.glob("*.tmp*")) == []


# ---------------------------------------------------------------------------
# CLI argparse (no execution)
# ---------------------------------------------------------------------------
class TestCliArgparse:
    def test_defaults(self):
        args = sw.build_parser().parse_args([])
        assert args.n_pairs == 300
        assert args.n_map_pairs == 8
        # None means "let the case decide" — prod + neutral, plus axis when
        # the case has a recovered phrase.
        assert args.conditions is None
        assert args.case == "subway_safety"
        assert args.kind == "proxy"
        assert args.layers == sw.FITTED_LAYERS
        assert args.map_layers == sw.MAP_LAYERS
        assert args.smoke is False
        assert len(args.lenses) == 3

    def test_overrides(self):
        args = sw.build_parser().parse_args(
            ["--n-pairs", "10", "--conditions", "prod", "neutral",
             "--layers", "24", "36", "--smoke"])
        assert args.n_pairs == 10
        assert args.conditions == ["prod", "neutral"]
        assert args.layers == [24, 36]
        assert args.smoke is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
