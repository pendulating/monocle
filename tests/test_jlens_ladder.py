"""Tests for the corpus-faithfulness ladder (research/monocle/jlens_fit_ladder.py).

These run on CPU and need no gemma weights. They check the EXPERIMENTAL
DESIGN, not the fit: one rung must move exactly one axis, and the held-out
corpus must never touch the analysis set.
"""
from __future__ import annotations

import pytest
import torch

from research.monocle import jlens_fit_ladder as ld  # noqa: E402

AXES = ("prompt", "target", "images")


class TestRungDesign:
    """The ladder is only interpretable if one rung moves one axis."""

    def test_rungs_are_ordered(self):
        assert list(ld.RUNGS) == ["R0", "R1", "R2", "R3"]

    def test_each_rung_moves_exactly_one_axis(self):
        names = list(ld.RUNGS)
        for above, rung in zip(names, names[1:], strict=False):
            moved = [a for a in AXES
                     if ld.RUNGS[above][a] != ld.RUNGS[rung][a]]
            assert len(moved) == 1, (
                f"{above} -> {rung} moves {moved}; the ladder needs exactly 1")

    def test_r0_repeats_the_mm_lens(self):
        """R0 is the control. It must match the existing mm fit exactly."""
        assert ld.RUNGS["R0"] == {
            "prompt": "describe", "target": "span", "images": "cyclomedia",
            "isolates": "baseline (repeats jlens/mm)"}

    def test_r3_is_the_faithful_rung(self):
        assert ld.RUNGS["R3"]["prompt"] == "rating"
        assert ld.RUNGS["R3"]["target"] == "last"
        assert ld.RUNGS["R3"]["images"] == "heldout"

    def test_every_axis_moves_once_across_the_ladder(self):
        """No axis is left untested, and none is tested twice."""
        names = list(ld.RUNGS)
        moved: list[str] = []
        for above, rung in zip(names, names[1:], strict=False):
            moved += [a for a in AXES
                      if ld.RUNGS[above][a] != ld.RUNGS[rung][a]]
        assert sorted(moved) == sorted(AXES)


class TestTargetMask:
    def test_last_selects_only_the_final_position(self):
        m = ld.target_mask_for("last", 10, torch.tensor([3]))
        assert m.dtype is torch.bool
        assert int(m.sum()) == 1
        assert bool(m[9])

    def test_span_starts_at_the_first_patch(self):
        m = ld.target_mask_for("span", 10, torch.tensor([3, 4, 5]))
        assert not bool(m[2])
        assert bool(m[3])

    def test_span_excludes_the_final_position(self):
        m = ld.target_mask_for("span", 10, torch.tensor([3]))
        assert not bool(m[9]), "the last position has no next-token target"
        assert int(m.sum()) == 6  # positions 3..8

    def test_span_and_last_are_disjoint_in_intent(self):
        span = ld.target_mask_for("span", 10, torch.tensor([3]))
        last = ld.target_mask_for("last", 10, torch.tensor([3]))
        assert int((span & last).sum()) == 0

    def test_masks_are_never_empty(self):
        for kind in ("span", "last"):
            m = ld.target_mask_for(kind, 8, torch.tensor([2]))
            assert int(m.sum()) > 0

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown target kind"):
            ld.target_mask_for("middle", 10, torch.tensor([3]))


class TestHoldOut:
    """R3 must not fit on an image the gate reads."""

    def test_heldout_skips_the_reserve(self, monkeypatch):
        pool = [{"path": f"/img/{i}.jpg", "image_id": str(i)}
                for i in range(100)]
        # Patch the function on the real module (see the next test): a patch
        # of sys.modules has no effect once research.monocle.single is imported.
        monkeypatch.setattr("research.monocle.single.load_single_rows",
                            lambda n, case, **kw: pool[:n])

        rows = ld.heldout_rows("subway_safety", 20, reserve=32)
        assert len(rows) == 20
        analysis = {r["path"] for r in pool[:32]}
        assert not ({r["path"] for r in rows} & analysis)

    def test_reserve_covers_the_gate_default(self):
        """patch_map_causal defaults to 12 and has run at 16."""
        assert ld.ANALYSIS_RESERVE >= 16

    def test_raises_when_the_pool_is_too_small(self, monkeypatch):
        # `from monocle import single` reads the package ATTRIBUTE, so a patch
        # of sys.modules is ignored once monocle.single is imported. Patch the
        # function on the real module, which every import path then sees.
        monkeypatch.setattr("research.monocle.single.load_single_rows",
                            lambda n, case, **kw: [{"path": "a"}] * 5)
        with pytest.raises(RuntimeError, match="no held-out image"):
            ld.heldout_rows("subway_safety", 20, reserve=32)


class TestParser:
    def test_rung_is_optional_only_under_smoke(self):
        a = ld.build_parser().parse_args([])
        assert a.rung is None and a.smoke is False
        b = ld.build_parser().parse_args(["--smoke"])
        assert b.smoke is True

    def test_defaults_are_the_gate_case(self):
        a = ld.build_parser().parse_args(["--rung", "R1"])
        assert a.case == "subway_safety"
        assert a.rules == "jacobian"

    def test_rejects_an_unknown_rung(self):
        with pytest.raises(SystemExit):
            ld.build_parser().parse_args(["--rung", "R9"])
