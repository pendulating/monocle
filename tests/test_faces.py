"""Unit tests for the camera-face module, `research/notebooks/cvpr/_faces.py`.

These tests build a small frame by hand. They read no parquet and need no GPU.
They cover the places where a quiet error changes a number in the paper:

- The side order. A `dyad` fixed effect on an unoriented frame reads the wrong
  answer, and the estimate looks fine.
- The recovery of a planted face effect, under all 3 designs.
- The centring of the 4 coefficients.
- The antisymmetry of the head-to-head matrix.
- The noise correction of the place spread.
"""

import numpy as np
import pandas as pd
import pytest

from research.notebooks.cvpr import _faces as FX  # noqa: E402


def _synth(n_places=40, n_draws=6000, beta=None, theta_sd=1.0, noise=0.5,
           seed=0, mix_theta_corr=0.8):
    """A run where the truth is known.

    Each comparison draws 2 places and 2 faces. The label is the difference of
    the place strengths, plus the difference of the face effects, plus noise.

    `mix_theta_corr` reproduces the structure that makes the side order matter.
    A place does NOT draw the 4 faces evenly: it has its own face mix, and a
    place with a strong `F` share also tends to be a strong place. That is true
    of the real data, and it is the channel that lets a between-place
    correlation leak into a within-dyad estimate. Set it to 0 and the faces
    become independent of the place.
    """
    rng = np.random.default_rng(seed)
    beta = beta if beta is not None else {"F": 0.30, "R": 0.0,
                                          "B": -0.30, "L": 0.0}
    places = np.array([f"p{i:03d}" for i in range(n_places)])
    theta = pd.Series(rng.normal(0, theta_sd, n_places), index=places)

    # The face mix of each place. The `F` share rises with the strength.
    z = (theta.to_numpy() - theta.to_numpy().mean()) / (theta.to_numpy().std() or 1)
    f_share = np.clip(0.25 + 0.18 * mix_theta_corr * z, 0.05, 0.85)
    mix = np.column_stack([f_share] + [(1 - f_share) / 3] * 3)

    # A small set of dyads, so each one repeats many times.
    ia = rng.integers(0, n_places, n_draws)
    ib = (ia + 1 + rng.integers(0, 4, n_draws)) % n_places
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    n = len(ia)

    def draw_face(idx):
        u = rng.random(len(idx))
        cum = np.cumsum(mix[idx], axis=1)
        return np.array(FX.FACES)[(u[:, None] > cum).sum(axis=1)]

    fa = draw_face(ia)
    fb = draw_face(ib)
    y = (theta.to_numpy()[ia] - theta.to_numpy()[ib]
         + np.array([beta[f] for f in fa]) - np.array([beta[f] for f in fb])
         + rng.normal(0, noise, n))

    # Side A holds each place about half the time, the way a real run does.
    flip = rng.random(n) < 0.5
    pa = np.where(flip, places[ib], places[ia])
    pb = np.where(flip, places[ia], places[ib])
    fa2 = np.where(flip, fb, fa)
    fb2 = np.where(flip, fa, fb)
    y = np.where(flip, -y, y)

    frame = pd.DataFrame({
        "pair_id": [f"x{i}" for i in range(n)],
        "place_a": pa, "place_b": pb,
        "face_a": fa2, "face_b": fb2,
        "y": y, "abstain": False, "is_swapped": flip,
        "hour_a": 12.0, "hour_b": 12.0, "month_a": 6, "month_b": 6,
    })
    frame["dyad"] = FX.dyad_key(pa, pb)
    frame.attrs.update(case="synthetic", model="test", arm="canonical",
                       image_mode=False)
    return frame, theta, beta


# --------------------------------------------------------------------------
# The side order
# --------------------------------------------------------------------------
def test_orient_fixes_the_side_order():
    frame, _, _ = _synth()
    out = FX.orient(frame)
    assert (out["place_a"] <= out["place_b"]).all()


def test_orient_keeps_the_comparison():
    """A flip must move `y`, the 2 faces and the 2 places TOGETHER."""
    frame, _, _ = _synth(n_draws=200)
    out = FX.orient(frame)
    flip = frame["place_a"].to_numpy() > frame["place_b"].to_numpy()
    assert flip.any()
    np.testing.assert_allclose(out.loc[flip, "y"], -frame.loc[flip, "y"])
    assert (out.loc[flip, "face_a"].to_numpy()
            == frame.loc[flip, "face_b"].to_numpy()).all()
    assert (out.loc[~flip, "y"].to_numpy()
            == frame.loc[~flip, "y"].to_numpy()).all()


def test_dyad_fe_without_orientation_reads_the_wrong_answer():
    """This is the bug the module guards against.

    A `dyad` group holds both side orders, and `y` changes sign with the order.
    A plain group mean therefore removes nothing, and the strength of the dyad
    stays in the error term with a random sign.

    That term is not orthogonal to the face design. A strong place also carries
    a strong `F` share, thus the leftover strength and the face contrast move
    together, and the between-place correlation leaks into the within-dyad
    estimate. `fit` orients the sides first. A fit on a frame with the sign
    left in place must NOT recover the planted effect.

    On the real libraries run this reads 0.053 rather than 0.083, and it puts
    `B` on the wrong side of 0.
    """
    frame, _, beta = _synth(seed=3)
    good = FX.fit(frame, fe="dyad")

    # Reproduce the unoriented estimate by hand.
    work = frame[frame["y"].notna()]
    free = [f for f in FX.FACES if f != FX.REFERENCE_FACE]
    X = np.column_stack([
        (work["face_a"].to_numpy() == f).astype(float)
        - (work["face_b"].to_numpy() == f).astype(float) for f in free])
    code = pd.factorize(work["dyad"])[0]
    cnt = np.bincount(code).astype(float)

    def demean(v):
        return v - (np.bincount(code, weights=v) / cnt)[code]

    Xd = np.column_stack([demean(X[:, j]) for j in range(X.shape[1])])
    bad = np.linalg.lstsq(Xd, demean(work["y"].to_numpy()), rcond=None)[0]
    bad_full = np.zeros(4)
    for j, f in enumerate(free):
        bad_full[FX.FACES.index(f)] = bad[j]
    bad_full -= bad_full.mean()

    truth = beta["F"] - beta["B"]
    assert good.swing == pytest.approx(truth, abs=0.05)
    assert abs((bad_full.max() - bad_full.min()) - truth) > 0.10

    # The `place` design needs no orientation: its signed incidence already
    # carries the side order, thus it recovers the same answer.
    assert FX.fit(frame, fe="place").swing == pytest.approx(truth, abs=0.05)


def test_orientation_does_not_matter_without_a_place_face_mix():
    """The leak needs the face mix to depend on the place.

    With `mix_theta_corr=0` the face is independent of the place, thus the
    leftover strength is orthogonal to the design and both fits agree. This
    test names the condition, so a reader knows the orientation step is not
    cosmetic.
    """
    frame, _, beta = _synth(seed=3, mix_theta_corr=0.0)
    assert FX.fit(frame, fe="dyad").swing == pytest.approx(
        beta["F"] - beta["B"], abs=0.05)


# --------------------------------------------------------------------------
# Recovery of a planted effect
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fe", ["dyad", "place"])
def test_fit_recovers_the_planted_effect(fe):
    frame, _, beta = _synth(seed=7)
    eff = FX.fit(frame, fe=fe)
    centred = {f: beta[f] - np.mean(list(beta.values())) for f in FX.FACES}
    for f in FX.FACES:
        assert eff.beta[f] == pytest.approx(centred[f], abs=0.04)
    assert eff.best_face == "F"
    assert eff.worst_face == "B"


def test_the_2_designs_agree():
    """The `dyad` and the `place` design must give the same point estimate."""
    frame, _, _ = _synth(seed=11)
    a = FX.fit(frame, fe="dyad")
    b = FX.fit(frame, fe="place")
    for f in FX.FACES:
        assert a.beta[f] == pytest.approx(b.beta[f], abs=0.02)


def test_no_effect_gives_no_effect():
    flat = {f: 0.0 for f in FX.FACES}
    frame, _, _ = _synth(beta=flat, seed=5)
    eff = FX.fit(frame, fe="place")
    for f in FX.FACES:
        assert abs(eff.beta[f]) < 0.04


def test_the_coefficients_are_centred():
    frame, _, _ = _synth(seed=13)
    eff = FX.fit(frame, fe="place")
    assert sum(eff.beta.values()) == pytest.approx(0.0, abs=1e-9)


def test_a_thin_cell_returns_none():
    frame, _, _ = _synth(n_draws=100)
    assert FX.fit(frame, fe="place") is None


# --------------------------------------------------------------------------
# The descriptive matrix
# --------------------------------------------------------------------------
def test_head_to_head_is_antisymmetric():
    frame, _, _ = _synth(seed=17)
    tab = FX.head_to_head(frame, fe="place")
    m = tab.to_numpy(dtype=float)
    np.testing.assert_allclose(m, -m.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-9)


def test_head_to_head_orders_the_faces():
    frame, _, _ = _synth(seed=19)
    tab = FX.head_to_head(frame, fe="place")
    # F against B must be the largest cell of the matrix.
    assert tab.loc["F", "B"] == pytest.approx(tab.to_numpy().max(), abs=1e-9)


# --------------------------------------------------------------------------
# The place-level numbers
# --------------------------------------------------------------------------
def test_score_spread_removes_the_sampling_noise():
    """A place with few comparisons carries noise. The correction must cut it."""
    frame, theta, _ = _synth(n_places=40, n_draws=4000, noise=1.5, seed=23)
    observed, signal = FX.score_spread(frame)
    assert observed > signal
    # The comparison is a difference of 2 places, thus the score of a place
    # regresses toward 0. The corrected spread must still be far below the
    # observed one and above 0.
    assert 0.0 < signal < observed


def test_composition_share_is_zero_without_an_effect():
    flat = {f: 0.0 for f in FX.FACES}
    frame, _, _ = _synth(beta=flat, n_draws=8000, seed=29)
    eff = FX.fit(frame, fe="place")
    assert eff.composition_share < 0.01


def test_corrected_scores_subtract_the_component():
    frame, _, _ = _synth(seed=31)
    eff = FX.fit(frame, fe="place")
    out = FX.corrected_place_scores(frame, eff)
    np.testing.assert_allclose(
        out["corrected_score"], out["mean_score"] - out["face_component"])
    assert out["rank_shift"].abs().max() <= 1.0


def test_place_face_mix_sums_to_one():
    frame, _, _ = _synth(seed=37)
    mix = FX.place_face_mix(frame)
    np.testing.assert_allclose(mix[list(FX.FACES)].sum(axis=1), 1.0, atol=1e-9)


# --------------------------------------------------------------------------
# The matched subset
# --------------------------------------------------------------------------
def test_matched_subset_keeps_the_identifying_rows():
    """A dyad with 1 face pair adds no within-group variation, thus the subset
    must give the SAME estimate as the whole frame under a dyad FE."""
    frame, _, _ = _synth(seed=41)
    sub = FX.matched_subset(frame)
    assert len(sub) <= len(frame)
    a = FX.fit(frame, fe="dyad")
    b = FX.fit(sub, fe="dyad")
    for f in FX.FACES:
        assert a.beta[f] == pytest.approx(b.beta[f], abs=1e-6)


def test_face_combo_is_unordered():
    frame = pd.DataFrame({"face_a": ["F", "B"], "face_b": ["B", "F"]})
    combo = FX._face_combo(frame)
    assert combo[0] == combo[1]


# --------------------------------------------------------------------------
# The constants
# --------------------------------------------------------------------------
def test_the_ordinal_map_matches_provenance():
    from research.notebooks.cvpr import _provenance as prov

    assert FX.ORDINAL_SCORE == prov.ORDINAL_SCORE
    assert FX.NOT_SURE == prov.NOT_SURE


def test_the_face_bearings_match_the_street_graph():
    from research.dagspaces.urbanroamvqa.graph.street_graph import FACE_BEARING_DEG

    assert FX.FACE_BEARING_DEG == FACE_BEARING_DEG


def test_the_libraries_case_is_the_only_dyad_design():
    assert FX.default_fe("libraries") == "dyad"
    for case in FX.CASES:
        if case != "libraries":
            assert FX.default_fe(case) == "place"


# --------------------------------------------------------------------------
# The abstention outcome
# --------------------------------------------------------------------------
def test_the_abstention_design_is_symmetric():
    frame, _, _ = _synth(seed=43)
    frame = frame.copy()
    rng = np.random.default_rng(0)
    # Plant an abstention that rises when EITHER side carries B.
    p = 0.1 + 0.2 * ((frame["face_a"] == "B") | (frame["face_b"] == "B"))
    frame["abstain"] = rng.random(len(frame)) < p
    eff = FX.fit(frame, fe="place", outcome="abstain")
    assert eff.beta["B"] == max(eff.beta.values())
