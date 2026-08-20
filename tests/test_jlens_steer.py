"""CPU-only unit tests for the J-vector steering rung (monocle.jlens_steer).

No GPU, no gemma load, no real jlens/model at module level. A tiny per-position
linear toy model + a fake JacobianLens with KNOWN, non-symmetric jacobians pin:

  * pull-back ORIENTATION — the result equals the analytic ``J^T @ d`` and the
    wrong orientation (``J @ d``) would fail (the whole point of the rung),
  * the injection HOOK — adding ``alpha*v`` at a position shifts a toy forward
    by exactly the linear prediction, and hook removal restores the baseline,
  * scoring: collapsed-label mapping / restricted argmax with a fake tokenizer,
  * calibration/eval split disjointness, and checkpoint parquet round-trip.

jlens is imported only inside the functions under test (two-venv fallback), so
this runs in the plain .venv where jlens is not pip-installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import jlens_steer as js  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes: a lens with known jacobians and a per-position linear toy block.
# ---------------------------------------------------------------------------
class FakeLens:
    """Stand-in for jlens.JacobianLens.

    ``transport(h, l) = h @ J_l.T`` — byte-identical to the real API — so any
    orientation check written against this fake is valid against the real lens.
    """

    def __init__(self, jacobians: dict[int, torch.Tensor], d_model: int) -> None:
        self.jacobians = {l: J.float() for l, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.d_model = d_model

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        return residual @ self.jacobians[layer].T


class FakeTokenizer:
    """encode(word)[0] returns a fixed first-token id per label. MuchLess and
    MuchMore deliberately share their first token (the "Much" collapse)."""

    FIRST = {"MuchLess": 10, "MuchMore": 10, "Much": 10,
             "Less": 11, "Same": 12, "More": 13}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # first token from the table, plus a throwaway continuation token.
        return [self.FIRST[text], 99]


# ---------------------------------------------------------------------------
# 1. Pull-back orientation (the load-bearing rung)
# ---------------------------------------------------------------------------
class TestPullBackOrientation:
    def _lens(self, d: int = 5, seed: int = 0):
        torch.manual_seed(seed)
        J = torch.randn(d, d)  # non-symmetric on purpose
        return FakeLens({7: J}, d_model=d), J

    def test_matches_analytic_transpose(self):
        lens, J = self._lens()
        d = torch.randn(5)
        v = js.pull_back_raw(J, d)
        assert torch.allclose(v, J.T @ d, atol=1e-5)

    def test_wrong_orientation_differs(self):
        # J @ d (the WRONG pull-back) is a genuinely different vector, so a
        # test pinned to J^T @ d catches the mistake.
        lens, J = self._lens()
        d = torch.randn(5)
        assert not torch.allclose(js.pull_back_raw(J, d), J @ d, atol=1e-4)

    def test_invariant_zero_for_correct_orientation(self):
        lens, J = self._lens()
        d = torch.randn(5)
        resid = js.orientation_residual(lens, J, d, layer=7)
        assert resid < 1e-5, resid

    def test_invariant_fails_for_wrong_orientation(self):
        # Emulate the bug: feed transport(d) (= J @ d) as if it were the
        # pull-back and confirm the invariant <d, transport(v)> == ||v||^2
        # is violated -> the runtime guard would fire.
        lens, J = self._lens()
        d = torch.randn(5)
        v_wrong = lens.transport(d, 7)              # J @ d, the wrong direction
        lhs = float(torch.dot(d, lens.transport(v_wrong, 7)))
        rhs = float(torch.dot(v_wrong, v_wrong))
        assert abs(lhs - rhs) / abs(rhs) > 1e-2

    def test_build_layer_vectors_unit_and_guarded(self):
        lens, J = self._lens()
        d_sig = torch.randn(5)
        d_sig = d_sig / d_sig.norm()
        d_ctl = torch.randn(5)
        d_ctl = d_ctl / d_ctl.norm()
        vecs = js.build_layer_vectors(lens, d_sig, d_ctl, [7])
        for name in ("signal", "control"):
            assert torch.allclose(vecs[name][7].norm(), torch.tensor(1.0), atol=1e-5)

    def test_build_layer_vectors_rejects_bad_lens(self):
        # A lens whose transport is J @ h but jacobians reports J.T would break
        # the invariant; simulate by handing a transpose-mismatched lens.
        d = 5
        torch.manual_seed(1)
        J = torch.randn(d, d)

        class BrokenLens(FakeLens):
            def transport(self, residual, layer):
                return residual @ self.jacobians[layer]  # missing the .T

        broken = BrokenLens({7: J}, d_model=d)
        dv = torch.randn(d)
        dv = dv / dv.norm()
        with pytest.raises(RuntimeError):
            js.build_layer_vectors(broken, dv, dv, [7])


# ---------------------------------------------------------------------------
# 2. Injection hook: exact linear shift + clean removal
# ---------------------------------------------------------------------------
class ToyBlock(nn.Module):
    """A per-position linear block; the model IS this single block so a forward
    hook on it sees the block output directly."""

    def __init__(self, d: int, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.lin = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h)


class TestInjectionHook:
    def test_adds_exact_vector_at_position(self):
        d, seq = 4, 6
        block = ToyBlock(d)
        h = torch.randn(1, seq, d)
        base = block(h)

        add = torch.arange(d).float() + 1.0
        pos = [seq - 1]
        handle = block.register_forward_hook(js.make_injection_hook(add, pos))
        try:
            out = block(h)
        finally:
            handle.remove()

        # answer position shifted by exactly `add`; all others untouched.
        assert torch.allclose(out[0, seq - 1], base[0, seq - 1] + add, atol=1e-6)
        for p in range(seq - 1):
            assert torch.allclose(out[0, p], base[0, p], atol=1e-6)

    def test_multiple_positions(self):
        d, seq = 4, 6
        block = ToyBlock(d)
        h = torch.randn(1, seq, d)
        base = block(h)
        add = torch.ones(d)
        pos = [1, 2, 3]
        handle = block.register_forward_hook(js.make_injection_hook(add, pos))
        try:
            out = block(h)
        finally:
            handle.remove()
        for p in pos:
            assert torch.allclose(out[0, p], base[0, p] + add, atol=1e-6)
        assert torch.allclose(out[0, 0], base[0, 0], atol=1e-6)

    def test_tuple_output_block(self):
        # HF blocks often return a tuple; the hook must patch element 0 only.
        d, seq = 4, 5
        block = ToyBlock(d)

        class TupleBlock(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, h):
                return (self.inner(h), "kv-cache-sentinel")

        tb = TupleBlock(block)
        h = torch.randn(1, seq, d)
        base = tb(h)[0]
        add = torch.ones(d) * 2.0
        handle = tb.register_forward_hook(js.make_injection_hook(add, [seq - 1]))
        try:
            out = tb(h)
        finally:
            handle.remove()
        assert out[1] == "kv-cache-sentinel"
        assert torch.allclose(out[0][0, seq - 1], base[0, seq - 1] + add, atol=1e-6)

    def test_removal_restores_baseline_exactly(self):
        d, seq = 4, 5
        block = ToyBlock(d)
        h = torch.randn(1, seq, d)
        a = block(h)
        handle = block.register_forward_hook(
            js.make_injection_hook(torch.ones(d) * 3.0, [seq - 1]))
        _ = block(h)
        handle.remove()
        b = block(h)
        assert torch.equal(a, b)

    def test_does_not_mutate_input_tensor(self):
        # the hook clones before adding; the block's own output object upstream
        # must be untouched (no in-place surprise for other consumers).
        d, seq = 4, 5
        block = ToyBlock(d)
        h = torch.randn(1, seq, d)
        base = block(h).clone()
        handle = block.register_forward_hook(
            js.make_injection_hook(torch.ones(d), [seq - 1]))
        try:
            block(h)
        finally:
            handle.remove()
        assert torch.allclose(block(h), base, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Scoring: collapsed labels, restricted argmax, More/Less probabilities
# ---------------------------------------------------------------------------
class TestScoring:
    def _ids(self):
        return js.label_first_tokens(FakeTokenizer())

    def test_label_first_tokens_collapse_and_distinct(self):
        ids = self._ids()
        assert set(ids) == set(js.COLLAPSED)
        assert ids["Much"] == 10 and ids["Less"] == 11
        assert ids["Same"] == 12 and ids["More"] == 13
        assert len(set(ids.values())) == 4

    def test_label_first_tokens_rejects_non_shared_much(self):
        class BadTok(FakeTokenizer):
            FIRST = {**FakeTokenizer.FIRST, "MuchMore": 14}

        with pytest.raises(ValueError):
            js.label_first_tokens(BadTok())

    def test_collapse_label(self):
        assert js.collapse_label("MuchLess") == "Much"
        assert js.collapse_label("MuchMore") == "Much"
        assert js.collapse_label("More") == "More"

    def test_collapsed_ordinal_index(self):
        assert js.collapsed_ordinal_index("Less", 0.1, 0.2) == 1
        assert js.collapsed_ordinal_index("Same", 0.1, 0.2) == 2
        assert js.collapsed_ordinal_index("More", 0.1, 0.2) == 3
        # Much resolves toward the leaning extreme.
        assert js.collapsed_ordinal_index("Much", 0.7, 0.1) == 4
        assert js.collapsed_ordinal_index("Much", 0.1, 0.7) == 0

    def test_score_answer_logits_argmax_and_agreement(self):
        ids = self._ids()  # Much10 Less11 Same12 More13
        vocab = 20
        logits = torch.full((vocab,), -5.0)
        logits[ids["More"]] = 8.0   # restricted argmax -> More
        logits[ids["Less"]] = 2.0
        m = js.score_answer_logits(logits, ids, ids["More"], ids["Less"],
                                   prod_label="More")
        assert m["steered_class"] == "More"
        assert m["steered_idx"] == 3
        assert m["prod_class"] == "More" and m["prod_idx"] == 3
        assert m["exact_collapsed"] == 1.0
        assert m["ordinal_collapsed"] == 1.0
        assert m["p_more"] > m["p_less"]
        assert m["p_more_minus_less"] == pytest.approx(m["p_more"] - m["p_less"])

    def test_score_much_collapses_to_prod_extreme(self):
        ids = self._ids()
        vocab = 20
        logits = torch.full((vocab,), -5.0)
        logits[ids["Much"]] = 8.0     # argmax -> Much
        logits[ids["More"]] = 3.0     # leans More -> resolves to MuchMore (4)
        logits[ids["Less"]] = 1.0
        m = js.score_answer_logits(logits, ids, ids["More"], ids["Less"],
                                   prod_label="MuchMore")
        assert m["steered_class"] == "Much"
        assert m["steered_idx"] == 4
        assert m["exact_collapsed"] == 1.0     # Much == collapse(MuchMore)
        assert m["ordinal_collapsed"] == 1.0

    def test_ordinal_partial_credit(self):
        ids = self._ids()
        vocab = 20
        logits = torch.full((vocab,), -5.0)
        logits[ids["Same"]] = 8.0     # predict Same (idx 2)
        m = js.score_answer_logits(logits, ids, ids["More"], ids["Less"],
                                   prod_label="MuchMore")  # idx 4
        assert m["steered_idx"] == 2 and m["prod_idx"] == 4
        assert m["exact_collapsed"] == 0.0
        assert m["ordinal_collapsed"] == pytest.approx(1.0 - 2 / 4)


# ---------------------------------------------------------------------------
# 4. contiguous_runs / image-token positions
# ---------------------------------------------------------------------------
class TestPositions:
    def test_two_runs(self):
        runs = js.contiguous_runs([3, 4, 5, 9, 10, 11])
        assert runs == [(3, 5), (9, 11)]

    def test_single_run(self):
        assert js.contiguous_runs([2, 3, 4]) == [(2, 4)]

    def test_empty(self):
        assert js.contiguous_runs([]) == []

    def test_all_image_token_positions_two_images(self):
        ids = torch.tensor([[1, 7, 7, 2, 7, 7, 3]])
        pos = js.all_image_token_positions({"input_ids": ids}, image_token_id=7)
        assert pos == [1, 2, 4, 5]
        assert js.contiguous_runs(pos) == [(1, 2), (4, 5)]


# ---------------------------------------------------------------------------
# 5. Calibration/eval split disjointness
# ---------------------------------------------------------------------------
class TestSplit:
    def test_disjoint_head_then_block(self):
        rows = [{"pair_id": i} for i in range(30)]
        calib, ev = js.split_calibration_eval(rows, n_calib=4, n_eval=10)
        assert [r["pair_id"] for r in calib] == [0, 1, 2, 3]
        assert [r["pair_id"] for r in ev] == list(range(4, 14))
        cset = {r["pair_id"] for r in calib}
        eset = {r["pair_id"] for r in ev}
        assert cset.isdisjoint(eset)

    def test_too_short_raises(self):
        rows = [{"pair_id": i} for i in range(5)]
        with pytest.raises(ValueError):
            js.split_calibration_eval(rows, n_calib=4, n_eval=10)


# ---------------------------------------------------------------------------
# 6. Parse helpers + checkpoint round-trip
# ---------------------------------------------------------------------------
class TestParseAndCheckpoint:
    def test_parse_int_list(self):
        assert js.parse_int_list("24,36,42") == [24, 36, 42]
        assert js.parse_int_list(" 24 , 36 ") == [24, 36]

    def test_parse_float_list(self):
        assert js.parse_float_list("-8,-4,0,4,8") == [-8, -4, 0, 4, 8]

    def test_checkpoint_round_trip(self, tmp_path):
        rows = [
            {"pair_id": "p0", "direction": "signal", "layer": 24, "alpha": 0.0,
             "steered_class": "More", "exact_collapsed": 1.0,
             "ordinal_collapsed": 1.0, "p_more_minus_less": 0.2},
            {"pair_id": "p0", "direction": "control", "layer": 24, "alpha": 4.0,
             "steered_class": "Same", "exact_collapsed": 0.0,
             "ordinal_collapsed": 0.75, "p_more_minus_less": -0.1},
        ]
        path = tmp_path / "steer_results.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        back = pd.read_parquet(path)
        assert set(back["pair_id"].unique()) == {"p0"}
        assert len(back) == 2
        # the resume path reconstructs done-pair ids and row dicts.
        assert back.to_dict("records")[0]["steered_class"] == "More"


# ---------------------------------------------------------------------------
# 7. summarize() aggregation
# ---------------------------------------------------------------------------
class TestSummarize:
    def test_grouping_and_means(self):
        rows = []
        for alpha, ex in ((0.0, 0.0), (0.0, 1.0), (4.0, 1.0), (4.0, 1.0)):
            rows.append({"direction": "signal", "layer": 24, "alpha": alpha,
                         "steered_class": "More" if ex else "Same",
                         "exact_collapsed": ex, "ordinal_collapsed": ex,
                         "p_more_minus_less": 0.1})
        df = pd.DataFrame(rows)
        summary = js.summarize(df)
        assert summary["signal/L24/a+0"]["exact_collapsed"] == 0.5
        assert summary["signal/L24/a+4"]["exact_collapsed"] == 1.0
        assert summary["signal/L24/a+4"]["n"] == 2
        assert summary["signal/L24/a+4"]["label_dist"] == {"More": 2}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
