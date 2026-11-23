"""The effect of the camera FACE on a pairwise judgment.

The question
------------
Every photograph in the battery is 1 face of a Cyclomedia cube. The catalog
holds 4 horizontal faces, and the face is **north-referenced**, not
vehicle-referenced:

| Face | The camera looks toward | The sun sits |
|------|-------------------------|--------------|
| `F` | north (0 deg) | behind the camera, all day |
| `R` | east (90 deg) | in front in the morning, behind in the afternoon |
| `B` | south (180 deg) | in front, near noon |
| `L` | west (270 deg) | behind in the morning, in front in the afternoon |

`research/dagspaces/urbanroamvqa/graph/street_graph.py` holds the same table as
`FACE_BEARING_DEG`, and `pairs.parquet` carries the absolute bearing in
`bearing_a` / `bearing_b`. Those columns are exactly 0, 90, 180 and 270.

Thus the face is a **camera decision, not a property of the place**. A model
that reads the place must ignore it. This module measures how much the model
does not.

The design
----------
The battery draws the image of each side at random from the pool of the place,
and the pool spans several faces. The face is thus assigned at random inside a
place, and a fixed effect on the place removes every property of the place.

| `fe` | What it holds fixed | Which cases carry it |
|------|---------------------|----------------------|
| `dyad` | the SAME 2 places, compared again | libraries alone |
| `place` | the place of each side | the 5 facility cases |
| `none` | nothing; the draw is random | the 2 image cases |

`dyad` is the strongest: the same 2 libraries meet about 36 times, with a new
pair of faces each time. Nothing but the face changes. The other cases drew
about 1,000,000 DISTINCT place pairs, thus they hold no dyad again and they
need the weaker `place` design.

The 2 image cases pair random street photographs, thus they carry no place. A
recording stands in for one: 39,000 recordings gave 2 or more faces to the
pool, and `fe="place"` reads those alone. It holds the position AND the minute
of the photograph fixed, so only the direction moves.

The model
---------
The outcome `y` is the ordinal label of side A against side B, from -2 to +2.
An abstention gives NaN and drops out, the same as in `_provenance.score_units`.

    y = theta_a - theta_b + beta[face_a] - beta[face_b] + e

`theta` is the strength of the place and `beta` is the face effect. The 4
`beta` values are identified up to a constant, thus this module reports them
with a mean of 0. A positive `beta[f]` says the model prefers a photograph that
looks toward `f`.

The standard error
------------------
A comparison touches 2 places, thus 2 comparisons that share a place carry
correlated errors. `cluster="dyadic"` uses the dyadic-robust variance of
Aronow, Samii and Assenova (2015), which the code below writes as

    meat = (sum over places) g_p g_p' - (sum over dyads) g_d g_d'

`cluster="dyad"` groups on the place pair alone, which is right when a dyad
repeats many times and a place effect is already absorbed. `cluster="robust"`
is HC0, which is right only when the pairs are drawn at random with no FE.

**Warning: the 2 clusters answer different questions.** `dyad` measures the
face effect for THESE places and gives a t near 17 on the libraries case.
`dyadic` generalises to new places, of which the libraries case holds 236, and
gives a t near 2. Report the conservative one.

What the numbers mean
---------------------
`beta` is in ordinal units, which no reader can size. `FaceEffect` therefore
also carries:

- `spread` — the standard deviation of `theta`, the real spread of the places.
- `swing_in_sd` — the best face minus the worst face, in units of `spread`.
- `swing_pctile` — how far up the ranking that swing carries a median place.
- `composition_share` — the share of the between-place score variance that the
  face mix of the places explains. This is the number the paper needs: it says
  how much of a published place score is the camera.

See `research/docs/camera-face-effect.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _ablations as A
from research.notebooks.cvpr import _canonical as C
from research.notebooks.cvpr import _results_table as R

__version__ = "1.0.0"

#: The 4 horizontal faces, in compass order.
FACES: Tuple[str, ...] = ("F", "R", "B", "L")

#: The absolute bearing of each face. It repeats `FACE_BEARING_DEG` of
#: `research/dagspaces/urbanroamvqa/graph/street_graph.py`. The cube is fixed to the
#: compass, thus `F` is true north on every recording.
FACE_BEARING_DEG: Dict[str, float] = {"F": 0.0, "R": 90.0, "B": 180.0, "L": 270.0}

#: A word for each face, for an axis label.
FACE_LABEL: Dict[str, str] = {
    "F": "F (north)", "R": "R (east)", "B": "B (south)", "L": "L (west)",
}

#: The ordinal score of each label. It repeats `_provenance.ORDINAL_SCORE`, and
#: a test proves the 2 stay equal.
ORDINAL_SCORE: Dict[str, int] = {
    "MuchLess": -2, "Less": -1, "Same": 0, "More": 1, "MuchMore": 2,
}
NOT_SURE = "NotSure"

#: The 7 cases, in the order of the paper.
CASES: Sequence[str] = tuple(R.RAW_CASE_LABEL)
CASE_LABEL: Dict[str, str] = dict(R.RAW_CASE_LABEL)
MODELS: Sequence[str] = A.MODELS

#: The 2 cases that pair random street photographs. They carry no `unit_uid`,
#: thus the recording stands in for the place.
IMAGE_CASES: Tuple[str, ...] = ("road_quality", "street_photography")

#: A cell with fewer usable comparisons than this gives no readable effect.
MIN_ROWS = 5_000

#: The face of the reference level. `beta` is reported with a mean of 0, thus
#: the choice changes nothing a reader sees. It only fixes the 3 free columns.
REFERENCE_FACE = "F"


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------
def is_image_case(case: str) -> bool:
    """True when the case pairs photographs rather than facilities."""
    return case in IMAGE_CASES


def _run(case: str, model: str, arm: str, kind: str):
    hits = [r for r in C.runs(kind=kind, case=case, model=model, arm=arm)
            if r.case == case]
    return hits[0] if hits else None


@lru_cache(maxsize=32)
def load_face_frame(case: str, model: str, arm: str = "canonical",
                    kind: str = "proxy") -> Optional[pd.DataFrame]:
    """Read 1 registered run and give each comparison its 2 faces.

    Returns None when the registry holds no such cell.

    The frame is the same shape for both kinds of case, so every estimator
    below takes 1 input:

    | Column | Meaning |
    |--------|---------|
    | `pair_id` | the key of the comparison |
    | `place_a`, `place_b` | the facility uid, or the recording id |
    | `face_a`, `face_b` | 1 of F, R, B, L |
    | `y` | the ordinal label of A against B; NaN on an abstention |
    | `abstain` | True when the model wrote `NotSure` |
    | `is_swapped` | True when the run showed B first |
    | `hour_a`, `hour_b` | the local hour of the drive; NaN when absent |
    | `dyad` | the unordered place pair |

    Warning: read `relative_label`, never `presented_label`. The orchestrator
    already undoes the presentation order in `relative_label`, and the 2
    columns differ on 75% of the swapped rows.
    """
    run = _run(case, model, arm, kind)
    if run is None:
        return None

    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(run.pairs_link).schema_arrow.names)
    image_mode = "unit_uid_a" not in have
    place_a, place_b = (("recording_id_a", "recording_id_b") if image_mode
                        else ("unit_uid_a", "unit_uid_b"))
    keep = ["pair_id", place_a, place_b, "face_a", "face_b", "is_swapped"]
    keep += [c for c in ("recordedAt_a", "recordedAt_b") if c in have]

    res = pd.read_parquet(run.results_link,
                          columns=["pair_id", "relative_label"])
    pairs = pd.read_parquet(run.pairs_link, columns=keep)
    df = res.merge(pairs, on="pair_id", how="inner", validate="one_to_one")

    lab = df["relative_label"].astype(str)
    out = pd.DataFrame({
        "pair_id": df["pair_id"].to_numpy(),
        "place_a": df[place_a].to_numpy(),
        "place_b": df[place_b].to_numpy(),
        "face_a": df["face_a"].to_numpy(),
        "face_b": df["face_b"].to_numpy(),
        "y": lab.map(ORDINAL_SCORE).to_numpy(dtype=float),
        "abstain": lab.eq(NOT_SURE).to_numpy(),
        "is_swapped": df["is_swapped"].to_numpy(),
    })
    for side in ("a", "b"):
        col = f"recordedAt_{side}"
        if col in df.columns:
            t = pd.to_datetime(df[col], errors="coerce", utc=True)
            t = t.dt.tz_convert("America/New_York")
            out[f"hour_{side}"] = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
            out[f"month_{side}"] = t.dt.month.to_numpy()
        else:
            out[f"hour_{side}"] = np.nan
            out[f"month_{side}"] = np.nan

    out = out[out["face_a"].isin(FACES) & out["face_b"].isin(FACES)]
    out = out.dropna(subset=["place_a", "place_b"]).reset_index(drop=True)
    out["dyad"] = dyad_key(out["place_a"].to_numpy(),
                           out["place_b"].to_numpy())
    out.attrs["case"] = case
    out.attrs["model"] = model
    out.attrs["arm"] = arm
    out.attrs["image_mode"] = image_mode
    out.attrs["run_id"] = run.run_id
    return out


def dyad_key(place_a: np.ndarray, place_b: np.ndarray) -> np.ndarray:
    """The unordered place pair of each comparison, as one string key.

    Warning: do not reach for `np.minimum` here. It raises on a fixed-width
    string array, which is what a synthetic frame carries, and it works only by
    accident on the object array that pandas gives. `np.where` handles both.
    """
    lo = np.where(place_a < place_b, place_a, place_b)
    hi = np.where(place_a < place_b, place_b, place_a)
    return (pd.Index(lo).astype(str) + "|" + pd.Index(hi).astype(str)).to_numpy()


def orient(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the 2 sides of every comparison into one fixed order.

    A run shows the same place pair many times, and side A holds the 2 places
    about equally often. The label states A against B, thus `y` changes sign
    with that order.

    **Warning: a `dyad` fixed effect is WRONG without this step.** The group
    then holds both orders, and the group mean is not the strength of the
    dyad. The libraries effect reads 0.053 without the step and 0.083 with it.

    A `place` fixed effect does not need the step, because the signed incidence
    already carries the order. This function is safe there as well: it flips
    `y`, the 2 faces, and the 2 places together, thus every estimate below is
    unchanged by it.
    """
    flip = (frame["place_a"].to_numpy() > frame["place_b"].to_numpy())
    if not flip.any():
        return frame
    out = frame.copy()
    for a, b in (("place_a", "place_b"), ("face_a", "face_b"),
                 ("hour_a", "hour_b"), ("month_a", "month_b")):
        va, vb = out[a].to_numpy().copy(), out[b].to_numpy().copy()
        out[a] = np.where(flip, vb, va)
        out[b] = np.where(flip, va, vb)
    out["y"] = np.where(flip, -out["y"].to_numpy(), out["y"].to_numpy())
    out["is_swapped"] = np.where(flip, ~out["is_swapped"].to_numpy(),
                                 out["is_swapped"].to_numpy())
    out.attrs.update(frame.attrs)
    return out


def _face_combo(frame: pd.DataFrame) -> np.ndarray:
    """The unordered face pair of each comparison, as one key.

    The pair is unordered because side A and side B are arbitrary. `F|B` and
    `B|F` are the same photograph pair shown the other way round.
    """
    fa = frame["face_a"].to_numpy().astype(str)
    fb = frame["face_b"].to_numpy().astype(str)
    lo = np.where(fa <= fb, fa, fb)
    hi = np.where(fa <= fb, fb, fa)
    return np.char.add(np.char.add(lo, "|"), hi)


def dyad_replication(frame: pd.DataFrame) -> pd.Series:
    """How many times each place pair came up, and with how many face pairs.

    The design the question asks for needs the SAME 2 places compared again
    with a DIFFERENT pair of faces. This reports whether the case carries one.

    Only the libraries case does. It holds 236 places, thus its 1,000,000 draws
    fall on 27,730 place pairs and each pair repeats about 39 times. The other
    facility cases hold 2,000 to 18,000 places, thus nearly every draw is a new
    place pair and no pair repeats.
    """
    work = pd.DataFrame({"dyad": frame["dyad"].to_numpy(),
                         "combo": _face_combo(frame)})
    g = work.groupby("dyad")["combo"].agg(n="size", n_face_combos="nunique")
    return pd.Series({
        "n_dyads": int(len(g)),
        "median_repeats": float(g["n"].median()),
        "dyads_with_2plus_face_combos": int((g["n_face_combos"] >= 2).sum()),
        "rows_in_those_dyads": int(g.loc[g["n_face_combos"] >= 2, "n"].sum()),
        "share_of_rows": float(g.loc[g["n_face_combos"] >= 2, "n"].sum()
                               / max(len(frame), 1)),
    })


def matched_subset(frame: pd.DataFrame, min_face_combos: int = 2,
                   min_repeats: int = 2) -> pd.DataFrame:
    """Keep only the place pairs that came up again with a NEW face pair.

    This is the exact comparison the question asks for. Inside such a pair the
    2 places never change, thus the only thing that moves is which direction
    the camera pointed.

    A `dyad` fixed effect on the whole frame gives the SAME estimate, because a
    dyad with 1 face pair adds no within-group variation and drops out of its
    own accord. Use this function to COUNT the comparisons that carry the
    result, and to show a reader that they exist.
    """
    work = pd.DataFrame({"dyad": frame["dyad"].to_numpy(),
                         "combo": _face_combo(frame)})
    g = work.groupby("dyad")["combo"].agg(n="size", k="nunique")
    keep = g.index[(g["k"] >= min_face_combos) & (g["n"] >= min_repeats)]
    out = frame[frame["dyad"].isin(keep)].copy()
    out.attrs.update(frame.attrs)
    return out


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
@dataclass
class FaceEffect:
    """The face effect of 1 (case, model, arm) cell, and how to read it."""

    case: str
    model: str
    arm: str
    fe: str
    cluster: str
    outcome: str
    n: int
    n_places: int
    n_dyads: int
    beta: Dict[str, float]           # mean 0, in outcome units
    se: Dict[str, float]
    t: Dict[str, float]
    p: Dict[str, float]
    best_face: str
    worst_face: str
    swing: float                     # best minus worst
    swing_se: float
    swing_t: float
    swing_p: float
    spread: float = float("nan")            # sd of the published place scores
    spread_signal: float = float("nan")     # the same, with the sampling noise removed
    swing_in_sd: float = float("nan")
    swing_pctile: float = float("nan")
    composition_share: float = float("nan")
    composition_share_signal: float = float("nan")
    theta: Optional[pd.Series] = None
    notes: List[str] = field(default_factory=list)

    def row(self) -> Dict[str, float]:
        """1 flat row, for a table over the cells."""
        out = {"case": self.case, "model": self.model, "arm": self.arm,
               "fe": self.fe, "cluster": self.cluster, "outcome": self.outcome,
               "n": self.n, "n_places": self.n_places}
        for f in FACES:
            out[f"beta_{f}"] = self.beta[f]
            out[f"se_{f}"] = self.se[f]
        out.update(best=self.best_face, worst=self.worst_face,
                   swing=self.swing, swing_se=self.swing_se,
                   swing_t=self.swing_t, swing_p=self.swing_p,
                   spread=self.spread, spread_signal=self.spread_signal,
                   swing_in_sd=self.swing_in_sd,
                   swing_pctile=self.swing_pctile,
                   composition_share=self.composition_share,
                   composition_share_signal=self.composition_share_signal,
                   notes="; ".join(self.notes))
        return out


def _design(frame: pd.DataFrame, symmetric: bool) -> np.ndarray:
    """The face columns, with `REFERENCE_FACE` dropped.

    A directional outcome takes the DIFFERENCE of the 2 indicators, because the
    label states A against B. A symmetric outcome, such as an abstention, takes
    the SUM: the model abstains on the comparison, not on 1 side of it.
    """
    fa = frame["face_a"].to_numpy()
    fb = frame["face_b"].to_numpy()
    cols = [f for f in FACES if f != REFERENCE_FACE]
    if symmetric:
        return np.column_stack([(fa == f).astype(float) + (fb == f).astype(float)
                                for f in cols])
    return np.column_stack([(fa == f).astype(float) - (fb == f).astype(float)
                            for f in cols])


def _absorb(frame: pd.DataFrame, fe: str, symmetric: bool):
    """Return a function that takes a vector to its within-FE residual.

    `place` builds the sparse incidence of the 2 places and partials it out
    with LSQR. That is the Frisch-Waugh-Lovell step, and it costs 1 solve for
    each column rather than a dense inversion of thousands of dummies.

    `dyad` demeans inside the place pair, which is exact and needs no solve.
    """
    n = len(frame)
    if fe == "none":
        return (lambda v: v - v.mean()), None, None

    if fe == "dyad":
        code = pd.factorize(frame["dyad"])[0]
        count = np.bincount(code, minlength=code.max() + 1).astype(float)

        def resid(v: np.ndarray) -> np.ndarray:
            m = np.bincount(code, weights=v, minlength=len(count)) / count
            return v - m[code]

        return resid, code, None

    if fe != "place":
        raise ValueError(f"unknown fe {fe!r}; use place, dyad or none")

    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr

    both = pd.concat([frame["place_a"], frame["place_b"]], ignore_index=True)
    code, keys = pd.factorize(both)
    ia, ib = code[:n], code[n:]
    sign_b = 1.0 if symmetric else -1.0
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[ia, ib]
    vals = np.r_[np.ones(n), np.full(n, sign_b)]
    D = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(keys)))

    # LSQR reaches the least-squares solution in 15 to 160 steps on every case
    # of the battery, thus the cap never binds. It stays because a truncated
    # solve leaves a piece of the place effect in the residual, and the
    # estimate would then drift toward the no-FE answer with no error.
    ITER_LIM = 2_000

    def _solve(v: np.ndarray) -> np.ndarray:
        out = lsqr(D, v, atol=1e-10, btol=1e-10, iter_lim=ITER_LIM)
        if out[2] >= ITER_LIM:
            raise RuntimeError(
                f"the place fixed effect did not converge in {ITER_LIM} steps "
                f"({len(keys):,} places). Raise ITER_LIM in `_faces._absorb`; "
                f"do NOT read the estimate as it is.")
        return out[0]

    def resid(v: np.ndarray) -> np.ndarray:
        return v - D @ _solve(v)

    def strengths(v: np.ndarray) -> pd.Series:
        return pd.Series(_solve(v), index=keys)

    return resid, (ia, ib, len(keys)), strengths


def _meat(h: np.ndarray, cluster: str, groups, notes: List[str]) -> np.ndarray:
    """The middle of the sandwich, under the named clustering."""
    if cluster == "robust":
        return h.T @ h

    if cluster == "dyad":
        code = groups["dyad"]
        k = code.max() + 1
        g = np.zeros((k, h.shape[1]))
        np.add.at(g, code, h)
        return g.T @ g

    if cluster != "dyadic":
        raise ValueError(f"unknown cluster {cluster!r}")

    # Aronow, Samii and Assenova (2015). A comparison enters the sum of BOTH
    # of its places, thus a dyad is counted twice and the dyad sum removes the
    # double count.
    ia, ib, n_places = groups["place"]
    g = np.zeros((n_places, h.shape[1]))
    np.add.at(g, ia, h)
    np.add.at(g, ib, h)
    code = groups["dyad"]
    gd = np.zeros((code.max() + 1, h.shape[1]))
    np.add.at(gd, code, h)
    meat = g.T @ g - gd.T @ gd
    if np.any(np.diag(meat) <= 0):
        notes.append("the dyadic meat was not positive; the place term alone "
                     "gives the standard error")
        return g.T @ g
    return meat


def fit(frame: pd.DataFrame, fe: str = "place", cluster: Optional[str] = None,
        outcome: str = "ordinal") -> Optional[FaceEffect]:
    """Estimate the face effect of 1 cell.

    Args:
        frame: The output of `load_face_frame`, or a subset of it.
        fe: `place`, `dyad`, or `none`. See the module docstring.
        cluster: `dyadic`, `dyad`, or `robust`. The default follows `fe`:
            `place` takes `dyadic`, `dyad` takes `dyad`, `none` takes `robust`.
        outcome: `ordinal` (-2 to +2), `sign` (-1, 0, +1), or `abstain`
            (1 when the model wrote `NotSure`).

    Returns None when the frame holds fewer than `MIN_ROWS` usable rows.

    **Warning: `beta` carries no meaning on its own.** Read `swing_in_sd` and
    `composition_share`, which put it beside the real spread of the places.
    """
    if cluster is None:
        cluster = {"place": "dyadic", "dyad": "dyad", "none": "robust"}[fe]
    symmetric = outcome == "abstain"

    if outcome == "abstain":
        work = frame
        y = frame["abstain"].to_numpy(dtype=float)
    else:
        # The side order must be fixed BEFORE a dyad group is demeaned. See
        # `orient`. It costs nothing under the other 2 designs.
        work = orient(frame[frame["y"].notna()])
        y = work["y"].to_numpy(dtype=float)
        if outcome == "sign":
            y = np.sign(y)
        elif outcome != "ordinal":
            raise ValueError(f"unknown outcome {outcome!r}")
    if len(work) < MIN_ROWS:
        return None

    notes: List[str] = []
    X = _design(work, symmetric)
    resid, group_info, strengths = _absorb(work, fe, symmetric)
    Xt = np.column_stack([resid(X[:, j]) for j in range(X.shape[1])])
    yt = resid(y)

    XX = np.linalg.inv(Xt.T @ Xt)
    beta = XX @ (Xt.T @ yt)
    e = yt - Xt @ beta
    h = Xt * e[:, None]

    dyad_code = pd.factorize(work["dyad"])[0]
    groups = {"dyad": dyad_code}
    if fe == "place":
        groups["place"] = group_info
    elif cluster == "dyadic":
        notes.append("dyadic clustering needs the place index; it fell back "
                     "to the dyad")
        cluster = "dyad"
    V = XX @ _meat(h, cluster, groups, notes) @ XX

    # Put the 4 values on a mean of 0. The contrast design fixes the reference
    # at 0, and a reader must not read that 0 as "no effect".
    free = [f for f in FACES if f != REFERENCE_FACE]
    full = np.zeros(4)
    for j, f in enumerate(free):
        full[FACES.index(f)] = beta[j]
    centre = full.mean()
    beta_full = {f: float(full[i] - centre) for i, f in enumerate(FACES)}

    # The standard error of a centred value. Write it as c'b with
    # c = e_f - mean over the 4 levels, where the reference contributes 0.
    se_full, t_full, p_full = {}, {}, {}
    for f in FACES:
        c = np.zeros(len(free))
        if f in free:
            c[free.index(f)] = 1.0
        c -= np.array([1.0 / 4.0] * len(free))
        var = float(c @ V @ c)
        s = float(np.sqrt(max(var, 0.0)))
        se_full[f] = s
        t_full[f] = beta_full[f] / s if s > 0 else float("nan")
        p_full[f] = _two_sided_p(t_full[f])

    best = max(FACES, key=lambda f: beta_full[f])
    worst = min(FACES, key=lambda f: beta_full[f])
    c = np.zeros(len(free))
    if best in free:
        c[free.index(best)] += 1.0
    if worst in free:
        c[free.index(worst)] -= 1.0
    swing = float(beta_full[best] - beta_full[worst])
    swing_se = float(np.sqrt(max(float(c @ V @ c), 0.0)))
    swing_t = swing / swing_se if swing_se > 0 else float("nan")

    eff = FaceEffect(
        case=str(frame.attrs.get("case", "")),
        model=str(frame.attrs.get("model", "")),
        arm=str(frame.attrs.get("arm", "")),
        fe=fe, cluster=cluster, outcome=outcome,
        n=int(len(work)),
        n_places=int(pd.unique(np.r_[work["place_a"].to_numpy(),
                                     work["place_b"].to_numpy()]).size),
        n_dyads=int(dyad_code.max() + 1),
        beta=beta_full, se=se_full, t=t_full, p=p_full,
        best_face=best, worst_face=worst,
        swing=swing, swing_se=swing_se, swing_t=swing_t,
        swing_p=_two_sided_p(swing_t),
        notes=notes,
    )

    if outcome != "abstain" and frame.attrs.get("image_mode"):
        # An image case pairs random street photographs, thus its "place" is a
        # recording that sits in about 4 comparisons. A score built on 4
        # comparisons is nearly all sampling noise, so a spread, a percentile
        # and a variance share over those scores would describe the noise. The
        # paper aggregates such a case to a census tract instead, and this
        # module does not do that step.
        eff.notes.append("an image case carries no place score; the spread, "
                         "the percentile and the variance share are absent")
    elif outcome != "abstain":
        if strengths is not None:
            theta = strengths(y - X @ beta)
            eff.theta = theta - theta.mean()
        obs, signal = score_spread(work)
        eff.spread = obs
        eff.spread_signal = signal
        if signal > 0:
            eff.swing_in_sd = swing / signal
        scores = place_scores(work)
        if len(scores) >= 10:
            eff.swing_pctile = _pctile_of_swing(scores.to_numpy(), swing)
        eff.composition_share = composition_share(work, eff, denominator="observed")
        eff.composition_share_signal = composition_share(
            work, eff, denominator="signal")
    return eff


def _two_sided_p(t: float) -> float:
    """The 2-sided normal p value of a t statistic."""
    if not np.isfinite(t):
        return float("nan")
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def _pctile_of_swing(theta: np.ndarray, swing: float) -> float:
    """How far up the ranking the swing carries a place at the median.

    A swing in ordinal units means nothing on its own. This says: put a median
    place on the worst face, then move it to the best face. What share of the
    places does it pass?
    """
    if len(theta) < 3 or not np.isfinite(swing):
        return float("nan")
    med = float(np.median(theta))
    lo = float((theta <= med).mean())
    hi = float((theta <= med + abs(swing)).mean())
    return hi - lo


# --------------------------------------------------------------------------
# What the effect does to a published place score
# --------------------------------------------------------------------------
def place_face_mix(frame: pd.DataFrame) -> pd.DataFrame:
    """The share of each place's own photographs that carried each face.

    A place does not draw the 4 faces evenly. A library on the north side of an
    east-west street is only visible from a camera that looks north, thus its
    pool is nearly all `F`. That is the channel by which a face effect reaches
    a PLACE score and not only a single comparison.
    """
    work = frame[frame["y"].notna()]
    long = pd.concat([
        pd.DataFrame({"place": work["place_a"].to_numpy(),
                      "face": work["face_a"].to_numpy()}),
        pd.DataFrame({"place": work["place_b"].to_numpy(),
                      "face": work["face_b"].to_numpy()}),
    ], ignore_index=True)
    mix = pd.crosstab(long["place"], long["face"], normalize="index")
    for f in FACES:
        if f not in mix.columns:
            mix[f] = 0.0
    mix = mix[list(FACES)]
    mix["n_comparisons"] = long.groupby("place").size()
    return mix


def induced_component(frame: pd.DataFrame, effect: FaceEffect) -> pd.Series:
    """The part of each place score that its face mix alone explains.

    `c_p = sum over f of (share of face f in the pool of p) * beta[f]`.
    """
    mix = place_face_mix(frame)
    b = np.array([effect.beta[f] for f in FACES])
    return pd.Series(mix[list(FACES)].to_numpy() @ b, index=mix.index,
                     name="face_component")


def composition_share(frame: pd.DataFrame, effect: FaceEffect,
                      min_comparisons: int = 5,
                      denominator: str = "observed") -> float:
    """The share of the between-place score variance that the face mix carries.

    This is the number the paper needs. It answers: of the spread that a
    published map shows between 2 places, how much is the direction the camera
    pointed?

    Args:
        denominator: `observed` divides by the variance of the published
            scores, which holds the sampling noise. `signal` divides by the
            noise-corrected variance, which is the real spread of the places.

    Warning: this is a lower bound on the whole camera effect. It counts the
    face alone. The hour of the drive, the season, and the weather move a
    photograph as well, and none of them are in this term.
    """
    scores = place_scores(frame, min_comparisons=min_comparisons)
    comp = induced_component(frame, effect).reindex(scores.index)
    ok = comp.notna() & scores.notna()
    if ok.sum() < 10:
        return float("nan")
    if denominator == "observed":
        var_score = float(np.var(scores[ok], ddof=1))
    elif denominator == "signal":
        var_score = float(score_spread(frame, min_comparisons)[1] ** 2)
    else:
        raise ValueError(f"unknown denominator {denominator!r}")
    if var_score <= 0:
        return float("nan")
    return float(np.var(comp[ok], ddof=1) / var_score)


def score_spread(frame: pd.DataFrame,
                 min_comparisons: int = 5) -> Tuple[float, float]:
    """The spread of the place scores, before and after the noise correction.

    A place score is a mean over a finite number of comparisons, thus its
    spread across the places holds real spread AND sampling noise:

        var(observed) = var(true) + mean(sampling variance of a place)

    The second term is small for a library, which sits in about 7,500
    comparisons, and it is large for a restaurant, which sits in about 100.
    Read the corrected value when you size the face effect against the real
    spread of the places. Read the observed value when you ask what share of a
    PUBLISHED map the face carries, because a published map holds the noise.

    Returns `(observed sd, corrected sd)`. The corrected value is 0 when the
    noise term is larger than the observed variance.
    """
    long = _long_scores(frame)
    g = long.groupby("place")["score"]
    n = g.size()
    keep = n[n >= min_comparisons].index
    mean = g.mean().loc[keep]
    var_within = g.var(ddof=1).loc[keep]
    n = n.loc[keep]
    observed = float(np.var(mean, ddof=1)) if len(mean) > 1 else float("nan")
    noise = float(np.nanmean((var_within / n).to_numpy()))
    signal = max(observed - noise, 0.0)
    return float(np.sqrt(observed)), float(np.sqrt(signal))


def _long_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """1 row for each place of each usable comparison, with its signed score."""
    work = frame[frame["y"].notna()]
    return pd.concat([
        pd.DataFrame({"place": work["place_a"].to_numpy(),
                      "score": work["y"].to_numpy()}),
        pd.DataFrame({"place": work["place_b"].to_numpy(),
                      "score": -work["y"].to_numpy()}),
    ], ignore_index=True)


def place_scores(frame: pd.DataFrame, min_comparisons: int = 5) -> pd.Series:
    """The mean score of each place, the way `_provenance.score_units` builds it."""
    g = _long_scores(frame).groupby("place")["score"]
    out = g.mean()
    return out[g.size() >= min_comparisons]


def corrected_place_scores(frame: pd.DataFrame, effect: FaceEffect,
                           min_comparisons: int = 5) -> pd.DataFrame:
    """The place score before and after the face correction.

    The correction subtracts the induced component, thus it gives the score the
    place would carry if its pool held the 4 faces evenly.
    """
    raw = place_scores(frame, min_comparisons=min_comparisons)
    comp = induced_component(frame, effect).reindex(raw.index)
    out = pd.DataFrame({"mean_score": raw, "face_component": comp})
    out["corrected_score"] = out["mean_score"] - out["face_component"]
    out["rank_raw"] = out["mean_score"].rank(pct=True)
    out["rank_corrected"] = out["corrected_score"].rank(pct=True)
    out["rank_shift"] = out["rank_corrected"] - out["rank_raw"]
    out.index.name = "unit_uid"
    return out


# --------------------------------------------------------------------------
# The descriptive table and the mechanism test
# --------------------------------------------------------------------------
def head_to_head(frame: pd.DataFrame, fe: str = "place") -> pd.DataFrame:
    """The mean outcome of each ordered face pair, with the FE removed.

    The result is a 4 x 4 matrix. Cell (f, g) says how far above 0 the label
    sits when side A carried face `f` and side B carried face `g`. The matrix
    is antisymmetric up to sampling noise, and the diagonal is 0.

    This is the descriptive view of the same thing `fit` estimates. It assumes
    no model, thus it also shows an interaction that an additive `beta` hides.
    """
    work = orient(frame[frame["y"].notna()])
    resid, _, _ = _absorb(work, fe, symmetric=False)
    yt = resid(work["y"].to_numpy(dtype=float))
    # Stack the comparison and its mirror. The side order is arbitrary, thus
    # the matrix must be antisymmetric, and this makes it exactly so.
    both = pd.DataFrame({
        "face_a": np.r_[work["face_a"].to_numpy(), work["face_b"].to_numpy()],
        "face_b": np.r_[work["face_b"].to_numpy(), work["face_a"].to_numpy()],
        "yt": np.r_[yt, -yt],
    })
    tab = both.pivot_table(index="face_a", columns="face_b", values="yt",
                           aggfunc="mean")
    return tab.reindex(index=list(FACES), columns=list(FACES))


#: The 3 windows of the drive day, in local time. The sun sits in the east
#: before 11, in the south near noon, and in the west after 14.
HOUR_WINDOWS: Tuple[Tuple[str, float, float], ...] = (
    ("morning", 0.0, 11.0),
    ("midday", 11.0, 14.0),
    ("afternoon", 14.0, 24.0),
)


def by_hour(frame: pd.DataFrame, fe: str = "none",
            windows: Sequence[Tuple[str, float, float]] = HOUR_WINDOWS
            ) -> pd.DataFrame:
    """The face effect inside each window of the drive day.

    This is the MECHANISM test, and it makes a falsifiable prediction. The cube
    is fixed to the compass, thus the sun moves against it during the day:

    | Window | The sun sits | Backlit face | Front-lit face |
    |--------|--------------|--------------|----------------|
    | morning | east | `R` | `L` |
    | midday | south | `B` | `F` |
    | afternoon | west | `L` | `R` |

    If the model reads the light, then `R - L` must be negative in the morning,
    near 0 at midday, and positive in the afternoon. `F - B` must stay positive
    all day, because a north-looking camera never faces the sun.

    If the model reads the CITY instead, then no coefficient moves with the
    hour: a street does not change its direction at lunchtime.

    Only the 2 image cases carry `recordedAt`, thus only they can run this.
    Each window keeps the pairs whose 2 photographs both fall inside it.
    """
    if frame["hour_a"].isna().all():
        return pd.DataFrame()
    rows = []
    for name, lo, hi in windows:
        m = (frame["hour_a"] >= lo) & (frame["hour_a"] < hi) & \
            (frame["hour_b"] >= lo) & (frame["hour_b"] < hi)
        sub = frame[m]
        if len(sub) < MIN_ROWS:
            continue
        sub = sub.copy()
        sub.attrs.update(frame.attrs)
        eff = fit(sub, fe=fe)
        if eff is None:
            continue
        row = {"window": name, "n": eff.n}
        row.update({f"beta_{f}": eff.beta[f] for f in FACES})
        row.update({f"se_{f}": eff.se[f] for f in FACES})
        row["R_minus_L"] = eff.beta["R"] - eff.beta["L"]
        row["F_minus_B"] = eff.beta["F"] - eff.beta["B"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------
def default_fe(case: str) -> str:
    """The design each case can carry.

    The libraries case is the ONLY one whose place pairs repeat, thus it is the
    only one that can hold the 2 places fixed.
    """
    return "dyad" if case == "libraries" else "place"


def build(cases: Sequence[str] = CASES, models: Sequence[str] = MODELS,
          arm: str = "canonical", fe: Optional[str] = None,
          outcome: str = "ordinal") -> pd.DataFrame:
    """Estimate the face effect over the whole grid, 1 row for each cell.

    `fe=None` gives each case the strongest design it carries: `dyad` for the
    libraries case and `place` for the rest.
    """
    rows = []
    for case in cases:
        for model in models:
            frame = load_face_frame(case, model, arm)
            if frame is None:
                continue
            eff = fit(frame, fe=fe or default_fe(case), outcome=outcome)
            if eff is None:
                rows.append({"case": case, "model": model, "arm": arm,
                             "n": int(frame["y"].notna().sum()),
                             "notes": "too few usable comparisons"})
                continue
            rows.append(eff.row())
    out = pd.DataFrame(rows)
    if not out.empty:
        out["case_label"] = out["case"].map(CASE_LABEL)
    return out


def sign_test(grid: pd.DataFrame) -> Dict[str, float]:
    """Does 1 face win in every cell of the grid?

    A single cell carries a wide interval, but the grid holds 14 of them. If
    the same face wins in all 14, then chance alone is a poor account.
    """
    ok = grid.dropna(subset=["swing"])
    if ok.empty:
        return {}
    from math import comb
    counts = ok["best"].value_counts()
    top, k = counts.index[0], int(counts.iloc[0])
    n = int(len(ok))
    # P(a NAMED face wins k or more of n) under a uniform draw over 4 faces.
    p = sum(comb(n, i) * (0.25 ** i) * (0.75 ** (n - i)) for i in range(k, n + 1))
    return {"n_cells": n, "top_face": top, "n_cells_won": k,
            "p_binomial": min(1.0, 4.0 * p)}


# --------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------
#: 1 colour for each face. The pair that the sun separates most, F and B, takes
#: the 2 ends of the diverging ramp, and the 2 that trade places during the
#: day, R and L, take the 2 warm swatches.
FACE_COLOUR: Dict[str, str] = {
    "F": "#498573",   # teal, the front-lit face
    "R": "#f3b14f",   # amber
    "B": "#e57264",   # coral, the back-lit face
    "L": "#5674b3",   # blue
}


def plot_grid(grid: pd.DataFrame, figsize=(6.8, 4.6)):
    """The 4 face coefficients of every cell, with their intervals.

    The reader looks for 1 thing: does the same face sit at the top of every
    row? A face effect that changes sign from case to case is noise. A face
    effect that keeps its order across 7 questions and 2 raters is the camera.
    """
    import matplotlib.pyplot as plt

    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    rows = grid.dropna(subset=["swing"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("no cell to plot")
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, lw=0.7, zorder=1)
    offset = {f: (i - 1.5) * 0.16 for i, f in enumerate(FACES)}
    for f in FACES:
        b = rows[f"beta_{f}"].to_numpy(dtype=float)
        e = rows[f"se_{f}"].to_numpy(dtype=float)
        yy = y + offset[f]
        ax.errorbar(b, yy, xerr=1.96 * e, fmt="none", ecolor=FACE_COLOUR[f],
                    elinewidth=1.0, capsize=0, zorder=2)
        ax.scatter(b, yy, s=22, color=FACE_COLOUR[f], edgecolors=S.EDGE,
                   linewidths=S.EDGE_LW, zorder=3, label=FACE_LABEL[f])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CASE_LABEL.get(r['case'], r['case'])} — {r['model']}"
                        for _, r in rows.iterrows()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("face effect on the label (ordinal units, mean 0)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    return fig


def plot_head_to_head(tab: pd.DataFrame, title_value: str = "",
                      figsize=(3.6, 3.2)):
    """The 4 x 4 matrix as a heat map, with the value written in each cell."""
    import matplotlib.pyplot as plt

    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    m = tab.to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(m))) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, cmap=S.CMAP_DIV, vmin=-lim, vmax=lim)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center",
                        fontsize=7, color=S.text_on(im.cmap(im.norm(m[i, j]))))
    ax.set_xticks(range(4), [FACE_LABEL[f] for f in FACES], fontsize=7,
                  rotation=30, ha="right")
    ax.set_yticks(range(4), [FACE_LABEL[f] for f in FACES], fontsize=7)
    ax.set_xlabel("face of side B")
    ax.set_ylabel("face of side A")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"mean label{title_value}", fontsize=7)
    fig.tight_layout()
    return fig


def plot_by_hour(hours: pd.DataFrame, figsize=(5.4, 3.6)):
    """The face effect against the window of the drive day.

    The sun test. `R` and `L` must cross, and `F` must stay on top.
    """
    import matplotlib.pyplot as plt

    from research.notebooks.cvpr import _style as S

    S.apply_house_style()
    if hours.empty:
        raise ValueError("no window to plot")
    x = np.arange(len(hours))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0.0, color=S.EDGE, lw=0.7, zorder=1)
    for f in FACES:
        b = hours[f"beta_{f}"].to_numpy(dtype=float)
        e = hours[f"se_{f}"].to_numpy(dtype=float)
        ax.plot(x, b, "-o", ms=4, lw=1.4, color=FACE_COLOUR[f],
                markeredgecolor=S.EDGE, markeredgewidth=S.EDGE_LW,
                label=FACE_LABEL[f], zorder=3)
        ax.fill_between(x, b - 1.96 * e, b + 1.96 * e, color=FACE_COLOUR[f],
                        alpha=0.18, lw=0, zorder=2)
    ax.set_xticks(x, [f"{w}\nn={int(n):,}" for w, n in
                      zip(hours["window"], hours["n"])], fontsize=7)
    ax.set_ylabel("face effect on the label")
    ax.set_xlabel("window of the drive day, local time")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# The camera-ready table
# --------------------------------------------------------------------------
def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def _or_dash(value: float, fmt: str, scale: float = 1.0) -> str:
    """Format a value, or an em dash when the case cannot carry it."""
    if value is None or not np.isfinite(value):
        return "—"
    return fmt.format(scale * value)


def to_display(grid: pd.DataFrame) -> pd.DataFrame:
    """The grid as the paper prints it: 1 row for each cell, 4 face columns."""
    columns = (["Case", "Rater", "Design", "n"]
               + [FACE_LABEL[f] for f in FACES]
               + ["swing", "swing / sd", "pctile", "place var."])
    rows = []
    for _, r in grid.iterrows():
        if not np.isfinite(r.get("swing", np.nan)):
            # A cell with no estimate keeps its row. Drop it and a reader
            # counts 13 cells and never learns which one is missing.
            blank = {c: "—" for c in columns}
            blank["Case"] = CASE_LABEL.get(r["case"], r["case"])
            blank["Rater"] = r["model"]
            rows.append(blank)
            continue
        out = {"Case": CASE_LABEL.get(r["case"], r["case"]),
               "Rater": r["model"],
               "Design": {"dyad": "same 2 places", "place": "place",
                          "none": "none"}[r["fe"]],
               "n": f"{int(r['n']):,}"}
        for f in FACES:
            out[FACE_LABEL[f]] = f"{r[f'beta_{f}']:+.3f}"
        out["swing"] = f"{r['swing']:.3f}{_stars(r['swing_p'])}"
        # An image case has no place score, thus the last 3 columns are absent
        # rather than 0. A printed 0 would read as "the camera does nothing".
        out["swing / sd"] = _or_dash(r["swing_in_sd"], "{:.2f}")
        out["pctile"] = _or_dash(r["swing_pctile"], "{:.0f}%", scale=100)
        out["place var."] = _or_dash(r["composition_share"], "{:.2f}%", scale=100)
        rows.append(out)
    return pd.DataFrame(rows, columns=columns)


def to_latex(grid: pd.DataFrame, caption: Optional[str] = None) -> str:
    """The display table as a LaTeX `tabular`."""
    disp = to_display(grid)
    head = " & ".join(disp.columns) + r" \\"
    body = "\n".join(" & ".join(str(v) for v in row) + r" \\"
                     for row in disp.itertuples(index=False))
    cap = caption or (
        "The effect of the camera face on a pairwise judgment. A value is the "
        "shift of the ordinal label, with the 4 faces centred on 0. "
        "\\emph{swing} is the best face minus the worst, "
        "\\emph{swing / sd} states it against the spread of the place scores, "
        "\\emph{pctile} is how far it carries a median place up the ranking, "
        "and \\emph{place var.} is the share of the between-place score "
        "variance that the face mix of the places explains.")
    return "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{" + "l" * 3 + "r" * (len(disp.columns) - 3) + "}",
        r"\toprule", head, r"\midrule", body, r"\bottomrule",
        r"\end{tabular}", rf"\caption{{{cap}}}",
        r"\label{tab:face-effect}", r"\end{table}",
    ])


def summary(grid: pd.DataFrame) -> str:
    """1 paragraph a caption can carry."""
    ok = grid.dropna(subset=["swing"])
    if ok.empty:
        return "no cell gave a face effect"
    st = sign_test(grid)
    return (
        f"The camera face moves a judgment in {len(ok)} of "
        f"{len(grid)} cells. Face {st.get('top_face', '?')} wins in "
        f"{st.get('n_cells_won', 0)} of {st.get('n_cells', 0)} "
        f"(p = {st.get('p_binomial', float('nan')):.1e}). The median swing is "
        f"{ok['swing'].median():.3f} ordinal units, which is "
        f"{ok['swing_in_sd'].median():.2f} of the spread of the place scores, "
        f"and it carries a median place past "
        f"{100 * ok['swing_pctile'].median():.1f}% of the other places. "
        f"The face mix of the places explains "
        f"{100 * ok['composition_share'].median():.2f}% of the between-place "
        f"score variance, and at most "
        f"{100 * ok['composition_share'].max():.2f}%. The battery draws the "
        f"face at random, thus most of the effect cancels inside a place "
        f"score.")
