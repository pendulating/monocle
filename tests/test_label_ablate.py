"""CPU tests for the pure helpers of monocle.label_ablate.

No GPU, no model, no jlens import. The GPU glue stays untested here; the
--smoke run covers it on klara.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle.label_ablate import (  # noqa: E402
    answer_metrics,
    bh_adjust,
    draw_matched_ids,
    orthonormal_basis,
    paired_stats,
    project_out,
    random_basis,
    removed_fraction,
    summarize_frame,
)


# ---------------------------------------------------------------------------
# orthonormal_basis
# ---------------------------------------------------------------------------
def test_basis_is_orthonormal():
    torch.manual_seed(0)
    vecs = torch.randn(5, 64)
    B = orthonormal_basis(vecs)
    assert B.shape == (64, 5)
    gram = B.T @ B
    assert torch.allclose(gram, torch.eye(5), atol=1e-5)


def test_basis_drops_dependent_rows():
    torch.manual_seed(0)
    v = torch.randn(64)
    vecs = torch.stack([v, 2.0 * v, torch.randn(64)])
    B = orthonormal_basis(vecs)
    assert B.shape[1] == 2


def test_basis_spans_input():
    torch.manual_seed(1)
    vecs = torch.randn(3, 32)
    B = orthonormal_basis(vecs)
    for v in vecs:
        recon = B @ (B.T @ v)
        assert torch.allclose(recon, v, atol=1e-4)


# ---------------------------------------------------------------------------
# project_out / removed_fraction
# ---------------------------------------------------------------------------
def test_projection_removes_span_and_is_idempotent():
    torch.manual_seed(2)
    B = orthonormal_basis(torch.randn(4, 48))
    h = torch.randn(48)
    out = project_out(h, B)
    assert float((B.T @ out).abs().max()) < 1e-5
    again = project_out(out, B)
    assert torch.allclose(out, again, atol=1e-5)
    assert float(out.norm()) <= float(h.norm()) + 1e-6


def test_removed_fraction_bounds():
    torch.manual_seed(3)
    B = orthonormal_basis(torch.randn(4, 48))
    h = torch.randn(48)
    f = removed_fraction(h, B)
    assert 0.0 <= f <= 1.0
    inside = B @ torch.randn(B.shape[1])
    assert removed_fraction(inside, B) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# draw_matched_ids / random_basis
# ---------------------------------------------------------------------------
def test_matched_ids_exclude_and_deterministic():
    exclude = {1, 2, 3}
    a = draw_matched_ids(1000, exclude, 5, seed=777)
    b = draw_matched_ids(1000, exclude, 5, seed=777)
    assert a == b
    assert len(a) == len(set(a)) == 5
    assert not (set(a) & exclude)
    c = draw_matched_ids(1000, exclude, 5, seed=778)
    assert a != c


def test_random_basis_orthonormal_and_deterministic():
    B1 = random_basis(64, 5, seed=777)
    B2 = random_basis(64, 5, seed=777)
    assert torch.equal(B1, B2)
    assert torch.allclose(B1.T @ B1, torch.eye(5), atol=1e-5)


# ---------------------------------------------------------------------------
# answer_metrics
# ---------------------------------------------------------------------------
def test_answer_metrics_reads_crafted_logits():
    logits = torch.full((100,), -10.0)
    class_ids = {"Much*": 10, "Less": 11, "Same": 12, "More": 13,
                 "NotSure": 14}
    logits[13] = 5.0   # More wins
    logits[11] = 2.0   # Less second
    m = answer_metrics(logits, class_ids, prod_first=13,
                       tok_more=13, tok_less=11)
    assert m["argmax_class"] == "More"
    assert m["c"] == pytest.approx(3.0)
    assert m["p_prod"] > 0.9


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def test_bh_adjust_known_values():
    q = bh_adjust([0.01, 0.02, 0.04])
    assert q[0] == pytest.approx(0.03)
    assert q[1] == pytest.approx(0.03)
    assert q[2] == pytest.approx(0.04)
    assert all(qi >= pi for qi, pi in zip(q, [0.01, 0.02, 0.04]))


def test_paired_stats_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.5, size=60)
    b = rng.normal(0.0, 0.5, size=60)
    st = paired_stats(a, b, n_boot=2000)
    assert st["mean"] == pytest.approx(1.0, abs=0.3)
    assert st["lo"] > 0.0
    assert st["p"] < 0.01


def test_paired_stats_null_covers_zero():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=60)
    b = rng.normal(0.0, 1.0, size=60)
    st = paired_stats(a, b, n_boot=2000)
    assert st["lo"] < 0.0 < st["hi"]


# ---------------------------------------------------------------------------
# summarize_frame
# ---------------------------------------------------------------------------
def _fake_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for pid in range(30):
        c0 = float(rng.normal(0, 1))
        rows.append({"pair_id": f"p{pid}", "prod_label": "More",
                     "layer": -1, "arm": "none", "removed_frac": 0.0,
                     "argmax_class": "More", "p_prod": 0.7, "c": c0})
        for layer in (24, 42):
            # the label arm moves c at L42 only; controls never move it
            bump = 2.0 if layer == 42 else 0.0
            rows.append({"pair_id": f"p{pid}", "prod_label": "More",
                         "layer": layer, "arm": "label",
                         "removed_frac": 0.3,
                         "argmax_class": "Less" if bump else "More",
                         "p_prod": 0.5 if bump else 0.7,
                         "c": c0 - bump})
            for arm in ("ctrl_matched", "ctrl_random"):
                rows.append({"pair_id": f"p{pid}", "prod_label": "More",
                             "layer": layer, "arm": arm,
                             "removed_frac": 0.05,
                             "argmax_class": "More", "p_prod": 0.7,
                             "c": c0 + float(rng.normal(0, 0.01))})
    return pd.DataFrame(rows)


def test_summarize_frame_finds_the_planted_effect():
    s = summarize_frame(_fake_frame())
    by = {(r["layer"], r["arm"]): r for r in s["by_arm"]}
    assert by[(42, "label")]["flip_rate"] == 1.0
    assert by[(24, "label")]["flip_rate"] == 0.0
    assert by[(42, "label")]["mean_abs_dc"] == pytest.approx(2.0)
    tests = {t["layer"]: t for t in s["paired_abs_dc"]["label_vs_ctrl_matched"]}
    assert tests[42]["mean"] == pytest.approx(2.0, abs=0.05)
    assert tests[42]["survives_bh"]
    # At L24 the label arm moves nothing; the paired difference against the
    # jittering control is near 0 (the sign of that tiny difference is not
    # stable, so the assertion is on the size, not on the BH flag).
    assert abs(tests[24]["mean"]) < 0.02


def test_summarize_frame_dc_is_paired_not_global():
    df = _fake_frame()
    s = summarize_frame(df)
    by = {(r["layer"], r["arm"]): r for r in s["by_arm"]}
    # the control moves c by ~0.01 per pair even though baseline c varies ~1
    assert by[(42, "ctrl_matched")]["mean_abs_dc"] < 0.05
