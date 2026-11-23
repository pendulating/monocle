"""The attributes of the version 3 spans: hierarchies, contradictions, and the proxy.

The version 3 extraction corpora give each span a class and a set of
attributes (`image`, `valence`, `kind`, `favors`, `source`, `direction`,
`hedged`, `about`, ...). The IC notebooks (`_ic_v3`) read the class, the
free `name`, and the vote. This module reads the other attributes. It holds
the arithmetic of 2 notebooks:

| Notebook | Questions |
|----------|-----------|
| `master/aesthetic_consistency.py` | 1. What does the model prize and penalise? 2. Where does it contradict itself? |
| `master/reasoning_vs_proxy.py` | 3. Which spans mark a trace that agrees with the outside measurement? |

The unit
--------
A trace is `(case, pair_id)`, and `tid` holds the 2 together. A span names a
side of the PRESENTED pair: Image A is the left image, `presented_left_path`.
The pair table stores the canonical pair (`sample_id_a`, `sample_id_b`), and
a swapped pair shows `image_path_b` on the left. Thus `pair_meta` maps each
presented side to a `sample_id` through the paths, never through the letter.

`presented_label` reads "Image A is <label> than Image B". More and MuchMore
mean that A wins, Less and MuchLess mean that B wins, Same and NotSure are
undecided. `winner` holds `A`, `B`, or `none`.

Warning: a perspective, an inference, and an uncertainty rate are facts
about the extractor. The 2 extractors of 1 judge differ about 2x on these
classes. Read every finding on both extractors of the judge.

Warning: this is an association, not a cause. The model narrates while it
decides, thus a span can follow the judgment.

Warning: the version 3 sample holds 1,000 traces of each case out of 11,000
pairs. Thus few images appear in 2 traces of 1 case (17 to 238 images on the
gemma arm). The same-image tables are thin. The unit tables (1 facility, many
images) and the 2 image cases (road quality and street photography read the
SAME pairs under 2 prompts) give more repeats.

See `research/docs/ic-metadata-findings.md`.
"""

from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _cues_across as CA
from research.notebooks.cvpr import _extractions as X
from research.notebooks.cvpr import _figmeta as F
from research.notebooks.cvpr import _ic_v3 as V
# Re-exported, thus a notebook can call `M.caption_md(fig)`.
from research.notebooks.cvpr._figmeta import caption_md, captioned, save_figure  # noqa: F401

__version__ = "1.2.0"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WIN_FOR_A = CA.WIN_FOR_A
WIN_FOR_B = CA.WIN_FOR_B
UNDECIDED = CA.UNDECIDED

CASES = ("subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography")

#: The attributes the IC slim frame drops and this module reads.
EXTRA_COLUMNS = ("direction", "dimension", "hedged", "mechanism", "of", "role",
                 "used_in_judgment")

#: For each class: the column that names the side (A or B) the span is about.
SIDE_COLUMN = {
    "dimension": "image", "inference": "image", "person_reference": "image",
    "image_artifact": "image", "perspective": "favors", "weighing": "favors",
    "comparison": "direction",
}

#: A dimension name with 1 of these words names an AXIS ("facade condition"),
#: not a property. An axis takes good or bad by the state it sees, thus a
#: mixed valence on an axis is not a contradiction of taste.
AXIS_WORDS = frozenset(("condition", "quality", "state", "level", "maintenance",
                        "upkeep", "cleanliness", "visibility", "density",
                        "amount", "presence", "size", "width", "type",
                        "appearance", "aesthetic", "overall"))

#: The headline unit proxy of each case, and its folder in `_unit_validation`.
UNIT_PROXY = {
    "subway_safety": ("subway", "transit_crime_per_rider"),
    "libraries": ("libraries", "building_violations_total"),
    "schools": ("schools", "school_attendance"),
    "parks_plazas": ("parks", "pip_acceptable_rate"),
    "restaurants": ("restaurants", "inspection_score_history"),
    "road_quality": ("road", "dot_pavement_rating"),
}

#: The area proxies the polygon join reads (`_ic_link.proxy_by_key`).
AREA_PROXIES = ("median_household_income", "crime_density")

#: The inference kinds that carry the risk result of the paper.
RISK_KINDS = ("wealth", "class")


# --------------------------------------------------------------------------
# Arms and the load
# --------------------------------------------------------------------------

def judge_of(arm: str) -> str:
    """The short judge name of an arm: `gemma-4-12b` or `qwen3.5-9b`."""
    return X.ARMS[arm]["judge"]


def winner_of(labels: pd.Series) -> pd.Series:
    """`A`, `B`, or `none` for each presented label."""
    lab = labels.astype(str)
    return pd.Series(np.where(lab.isin(WIN_FOR_A), "A",
                              np.where(lab.isin(WIN_FOR_B), "B", "none")),
                     index=labels.index)


def load_arm(arm: str, require_canonical: bool = True) -> pd.DataFrame:
    """Read 1 v3 arm with the paper filters, and keep the attributes.

    The frame is `_ic_v3.slim` plus `EXTRA_COLUMNS`, `key_dim` (the
    normalised `comparison.dimension`), `winner`, and `side` (the A or B
    that `SIDE_COLUMN` names, else missing).
    """
    if arm not in V.v3_arms():
        raise KeyError(f"{arm!r} is not a v3 arm")
    spec = X.arm_roots([arm], present_only=False)[arm]
    df = X.load(root=spec["root"], require_canonical=require_canonical)
    out = V.slim(df, arm)
    for col in EXTRA_COLUMNS:
        out[col] = df[col].to_numpy() if col in df.columns else None
    del df
    return prepare(out)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add `key_dim`, `winner`, and `side` to a slim frame.

    A test frame passes through here too, thus it needs only `case`,
    `pair_id`, `presented_label`, `extraction_class`, and the attributes.
    """
    out = df.copy()
    for col in ("tid",):
        if col not in out.columns:
            out["tid"] = V.trace_id(out)
    for col in ("key_name",):
        if col not in out.columns:
            out["key_name"] = V.normalise_names(out["name"]) if "name" in out.columns else None
    if "dimension" in out.columns:
        out["key_dim"] = V.normalise_names(out["dimension"])
    out["case"] = out["case"].astype(str)
    out["extraction_class"] = out["extraction_class"].astype(str)
    out["presented_label"] = out["presented_label"].astype(str)
    out["winner"] = winner_of(out["presented_label"]).to_numpy()
    side = pd.Series(None, index=out.index, dtype="object")
    for cls, col in SIDE_COLUMN.items():
        if col not in out.columns:
            continue
        mask = out["extraction_class"].eq(cls)
        side[mask] = out.loc[mask, col].astype("object")
    out["side"] = side.where(side.isin(["A", "B"]), None)
    return out


def traces(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each trace: `tid`, `case`, `pair_id`, `presented_label`,
    `winner`."""
    return (df[["tid", "case", "pair_id", "presented_label", "winner"]]
            .drop_duplicates("tid").reset_index(drop=True))


def wilson(k: int, n: int) -> Tuple[float, float]:
    """The Wilson interval of k of n (`_extractions._wilson`)."""
    return X._wilson(int(k), int(n))


def _rate_row(k: int, n: int) -> Dict[str, float]:
    lo, hi = wilson(k, n)
    return {"n": int(n), "k": int(k), "rate": (k / n) if n else float("nan"),
            "lo": lo, "hi": hi}


# --------------------------------------------------------------------------
# The image join: presented side -> sample_id, unit_uid
# --------------------------------------------------------------------------

def results_path(judge: str, case: str) -> str:
    """The judgment results parquet of the registered run of 1 judge and case.

    The map is `research.traces.submit_trace_shards.GEMMA_RUNS` and
    `QWEN_RUNS`, which the extraction stage read.
    """
    from research.traces import submit_trace_shards as S

    runs = S.GEMMA_RUNS if judge.startswith("gemma") else S.QWEN_RUNS
    path = Path(runs[case])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


RESULT_COLUMNS = ("pair_id", "canonical_pair_id", "repeat_idx", "sample_id_a",
                  "sample_id_b", "image_path_a", "image_path_b",
                  "presented_left_path", "presented_right_path", "is_swapped",
                  "relative_score", "relative_label", "presented_label")
PAIR_COLUMNS = ("pair_id", "unit_uid_a", "unit_uid_b", "latitude_a",
                "latitude_b", "longitude_a", "longitude_b")


@lru_cache(maxsize=16)
def _raw_meta(judge: str, case: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = results_path(judge, case)
    have = set(pq.read_schema(path).names)
    res = pd.read_parquet(path, columns=[c for c in RESULT_COLUMNS if c in have])
    pairs_path = os.path.join(os.path.dirname(path), "pairs.parquet")
    if os.path.exists(pairs_path):
        have_p = set(pq.read_schema(pairs_path).names)
        cols = [c for c in PAIR_COLUMNS if c in have_p]
        pairs = pd.read_parquet(pairs_path, columns=cols)
        res = res.merge(pairs, on="pair_id", how="left", validate="one_to_one")
    res["case"] = case
    return res


def presented_sides(meta: pd.DataFrame) -> pd.DataFrame:
    """Map the presented sides of each pair to the canonical sides.

    Image A of the trace is `presented_left_path`. When it equals
    `image_path_a`, presented A is canonical `a`; when it equals
    `image_path_b`, presented A is canonical `b`. A row with no path match
    falls back to `is_swapped`.

    Adds `canon_of_A` (`a` or `b`), `sample_A`, `sample_B`, and, when the
    pair table holds them, `unit_A` and `unit_B`.

    Warning: never read Image A as `sample_id_a`. Half of the pairs are
    swapped.
    """
    out = meta.copy()
    left = out.get("presented_left_path")
    swapped = out["is_swapped"].fillna(False).astype(bool)
    if left is not None and "image_path_a" in out.columns:
        is_a = left.eq(out["image_path_a"])
        is_b = left.eq(out["image_path_b"])
        canon_a = np.where(is_a, "a", np.where(is_b, "b",
                                                np.where(swapped, "b", "a")))
    else:
        canon_a = np.where(swapped, "b", "a")
    out["canon_of_A"] = canon_a
    first = out["canon_of_A"].eq("a")
    out["sample_A"] = np.where(first, out["sample_id_a"], out["sample_id_b"])
    out["sample_B"] = np.where(first, out["sample_id_b"], out["sample_id_a"])
    if "unit_uid_a" in out.columns:
        out["unit_A"] = np.where(first, out["unit_uid_a"], out["unit_uid_b"])
        out["unit_B"] = np.where(first, out["unit_uid_b"], out["unit_uid_a"])
    return out


def pair_meta(judge: str, cases: Sequence[str] = CASES) -> pd.DataFrame:
    """The pair table of 1 judge over the cases, with the presented sides.

    1 row for each (case, pair_id). See `presented_sides`.
    """
    frames = [presented_sides(_raw_meta(judge, c)) for c in cases]
    out = pd.concat(frames, ignore_index=True)
    out["pair_id"] = out["pair_id"].astype(str)
    return out


def attach_images(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Give each span the `sample_id` and the `unit_uid` of its side.

    `entity_sample` is the sample of the side the span names (`side`), and
    `other_sample` is the sample of the other side. The same for
    `entity_unit` and `other_unit`. A span with no A or B side gets missing
    values.
    """
    keep = ["case", "pair_id", "sample_A", "sample_B", "canonical_pair_id"]
    if "unit_A" in meta.columns:
        keep += ["unit_A", "unit_B"]
    m = meta[[c for c in keep if c in meta.columns]].copy()
    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    out = out.merge(m, on=["case", "pair_id"], how="left")
    is_a = out["side"].eq("A")
    is_b = out["side"].eq("B")
    for kind, a, b in (("sample", "sample_A", "sample_B"),
                       ("unit", "unit_A", "unit_B")):
        if a not in out.columns:
            out[f"entity_{kind}"] = None
            out[f"other_{kind}"] = None
            continue
        out[f"entity_{kind}"] = np.where(is_a, out[a], np.where(is_b, out[b], None))
        out[f"other_{kind}"] = np.where(is_a, out[b], np.where(is_b, out[a], None))
    return out


# --------------------------------------------------------------------------
# Question 1: what the model prizes and penalises
# --------------------------------------------------------------------------

def is_axis(name) -> bool:
    """True when a dimension name holds an `AXIS_WORDS` word."""
    if not isinstance(name, str):
        return False
    return bool(set(name.split()) & AXIS_WORDS)


def valence_table(df: pd.DataFrame, min_count: int = 25) -> pd.DataFrame:
    """The valence of each dimension name, for each case.

    1 row for each (case, name) with `min_count` or more good or bad rows.
    `net` = (good - bad) / (good + bad), from -1 (always bad) to +1 (always
    good). `good_share` with its Wilson interval; `neutral_share` of all the
    rows of the name; `axis` from `is_axis`.

    Warning: a row is a span, not a trace. A trace that names `greenery` 3
    times counts 3 times. `valence_by_trace` gives 1 vote for each trace.
    """
    dim = df[df["extraction_class"].eq("dimension") & df["key_name"].notna()]
    allrows = dim.groupby(["case", "key_name"]).size().rename("n_all")
    signed = dim[dim["valence"].isin(["good", "bad"])]
    g = signed.groupby(["case", "key_name"])["valence"]
    out = pd.DataFrame({"n": g.size(),
                        "good": g.apply(lambda s: int((s == "good").sum()))})
    out = out.join(allrows).reset_index()
    out = out[out["n"] >= min_count].copy()
    if out.empty:
        return out
    out["bad"] = out["n"] - out["good"]
    out["good_share"] = out["good"] / out["n"]
    lo, hi = zip(*(wilson(k, n) for k, n in zip(out["good"], out["n"])))
    out["lo"], out["hi"] = lo, hi
    out["net"] = (out["good"] - out["bad"]) / out["n"]
    out["neutral_share"] = 1 - out["n"] / out["n_all"]
    out["axis"] = out["key_name"].map(is_axis)
    out["settled"] = (out["lo"] > 0.5) | (out["hi"] < 0.5)
    out["mixed"] = (out["good_share"] > 0.3) & (out["good_share"] < 0.7)
    return out.sort_values(["case", "n"], ascending=[True, False]).reset_index(drop=True)


def valence_summary(table: pd.DataFrame, min_cases: int = 3) -> pd.DataFrame:
    """Each dimension name over the cases: does its valence sign hold?

    `n_good_cases` and `n_bad_cases` count the cases where the interval
    clears 0.5 on that side. `flips` is True when 1 case is settled good and
    another settled bad. `holds` is True when the name is in `min_cases`
    cases or more and every case has the same sign of `net`.
    """
    if table.empty:
        return pd.DataFrame()
    g = table.groupby("key_name")
    out = pd.DataFrame({
        "n_cases": g["case"].nunique(),
        "n": g["n"].sum(),
        "net_min": g["net"].min(),
        "net_max": g["net"].max(),
        "n_pos": g["net"].apply(lambda s: int((s > 0).sum())),
        "n_neg": g["net"].apply(lambda s: int((s < 0).sum())),
        "n_good_cases": g.apply(lambda t: int((t["lo"] > 0.5).sum()), include_groups=False),
        "n_bad_cases": g.apply(lambda t: int((t["hi"] < 0.5).sum()), include_groups=False),
        "good_cases": g.apply(lambda t: ", ".join(t.loc[t["lo"] > 0.5, "case"]), include_groups=False),
        "bad_cases": g.apply(lambda t: ", ".join(t.loc[t["hi"] < 0.5, "case"]), include_groups=False),
    }).reset_index()
    out["axis"] = out["key_name"].map(is_axis)
    out["flips"] = (out["n_good_cases"] > 0) & (out["n_bad_cases"] > 0)
    same = (out["n_pos"] == out["n_cases"]) | (out["n_neg"] == out["n_cases"])
    out["holds"] = same & (out["n_cases"] >= min_cases)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def valence_vote(df: pd.DataFrame) -> pd.DataFrame:
    """Does a good dimension sit on the winner, and a bad one on the loser?

    For each case and valence: of the dimension spans on A or B in a decided
    trace, the share on the winner, with the Wilson interval. 1 vote for each
    (trace, name, side, valence).
    """
    dim = df[df["extraction_class"].eq("dimension") & df["side"].notna()
             & df["winner"].ne("none")]
    dim = dim.drop_duplicates(["tid", "key_name", "side", "valence"])
    rows = []
    for (case, val), part in dim.groupby(["case", "valence"]):
        k = int(part["side"].eq(part["winner"]).sum())
        rows.append({"case": case, "valence": val, **_rate_row(k, len(part))})
    return pd.DataFrame(rows)


def good_margin(df: pd.DataFrame) -> pd.DataFrame:
    """For each decided trace: good and bad dimension names on the winner
    and on the loser.

    Columns: `tid`, `good_w`, `good_l`, `bad_w`, `bad_l`, `margin`
    (good_w - good_l - bad_w + bad_l). 1 count for each distinct name.
    """
    dim = df[df["extraction_class"].eq("dimension") & df["side"].notna()
             & df["valence"].isin(["good", "bad"])]
    dim = dim.drop_duplicates(["tid", "key_name", "side", "valence"])
    on_w = dim["side"].eq(dim["winner"])
    tab = pd.DataFrame({
        "tid": dim["tid"],
        "good_w": (dim["valence"].eq("good") & on_w).astype(int),
        "good_l": (dim["valence"].eq("good") & ~on_w).astype(int),
        "bad_w": (dim["valence"].eq("bad") & on_w).astype(int),
        "bad_l": (dim["valence"].eq("bad") & ~on_w).astype(int),
    }).groupby("tid").sum()
    tab["margin"] = tab["good_w"] - tab["good_l"] - tab["bad_w"] + tab["bad_l"]
    return tab.reset_index()


def keyed_win(df: pd.DataFrame, extraction_class: str, key: str,
              min_count: int = 25) -> pd.DataFrame:
    """The directional lift of any attribute of a class, by case.

    `key` names the column that becomes the key (for example `kind` of the
    inference, `source` of the perspective). The arithmetic is
    `_cues_across.keyed_lift`: 1 vote for each (trace, key, side); a key on
    both sides of 1 trace is dropped; the rate is shrunk toward the base.
    """
    part = df[df["extraction_class"].eq(extraction_class) & df["side"].notna()
              & df[key].notna()]
    frame = pd.DataFrame({"case": part["case"], "pair_id": part["pair_id"],
                          "presented_label": part["presented_label"],
                          "image": part["side"], "key": part[key].astype(str)})
    frames = [CA.keyed_lift(frame, c, min_count=min_count)
              for c in sorted(frame["case"].unique())]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def source_table(df: pd.DataFrame) -> pd.DataFrame:
    """How `source` splits the perspectives, and how each source wins.

    For each case and source: the perspective rows, the share of all the
    perspective rows of the case, and, over the rows that favor A or B in a
    decided trace, the share whose favored image won (Wilson interval).
    """
    p = df[df["extraction_class"].eq("perspective") & df["source"].notna()]
    total = p.groupby("case").size()
    rows = []
    for (case, src), part in p.groupby(["case", "source"]):
        dec = part[part["side"].notna() & part["winner"].ne("none")]
        k = int(dec["side"].eq(dec["winner"]).sum())
        rows.append({"case": case, "source": src, "rows": len(part),
                     "share": len(part) / total[case],
                     "neither_share": float(part["favors"].eq("neither").mean()),
                     **_rate_row(k, len(dec))})
    return pd.DataFrame(rows)


def source_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """The traces where a world-knowledge and a task-convention perspective
    favor opposite images: which one does the decision follow?

    1 row for each case (and 1 pooled row): `n` conflict traces (decided),
    `k` traces whose winner is the side the task convention favors.
    A trace counts when its world-knowledge rows favor only 1 side and its
    task-convention rows favor only the other side.
    """
    p = df[df["extraction_class"].eq("perspective") & df["side"].notna()
           & df["source"].isin(["world_knowledge", "task_convention"])
           & df["winner"].ne("none")]
    sides = p.groupby(["tid", "source"])["side"].agg(lambda s: "".join(sorted(set(s))))
    wide = sides.unstack()
    if "task_convention" not in wide.columns or "world_knowledge" not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna()
    wide = wide[wide["task_convention"].isin(["A", "B"])
                & wide["world_knowledge"].isin(["A", "B"])
                & (wide["task_convention"] != wide["world_knowledge"])]
    tr = traces(df).set_index("tid")
    wide = wide.join(tr[["case", "winner"]])
    wide["follows_tc"] = wide["winner"].eq(wide["task_convention"])
    rows = [{"case": c, **_rate_row(int(g["follows_tc"].sum()), len(g))}
            for c, g in wide.groupby("case")]
    rows.append({"case": "pooled", **_rate_row(int(wide["follows_tc"].sum()), len(wide))})
    return pd.DataFrame(rows)


def hedge_table(df: pd.DataFrame) -> pd.DataFrame:
    """The comparisons: does the direction match the vote, hedged or not?

    For each case and `hedged`: of the comparisons with direction A or B in
    a decided trace, the share whose direction is the winner. `direction`
    names the image with MORE of the quality, thus a comparison on a bad
    quality ("more litter") points at the loser. The table also gives
    `neither_share`, the share of comparisons with direction `neither`.
    """
    c = df[df["extraction_class"].eq("comparison") & df["hedged"].notna()]
    rows = []
    for (case, h), part in c.groupby(["case", "hedged"]):
        dec = part[part["side"].notna() & part["winner"].ne("none")]
        k = int(dec["side"].eq(dec["winner"]).sum())
        rows.append({"case": case, "hedged": h, "rows": len(part),
                     "neither_share": float(part["direction"].eq("neither").mean()),
                     **_rate_row(k, len(dec))})
    return pd.DataFrame(rows)


def hedge_by_strength(df: pd.DataFrame) -> pd.DataFrame:
    """The share of hedged comparisons in a trace, by the strength of the
    label: `strong` (MuchMore, MuchLess), `plain` (More, Less), `undecided`
    (Same, NotSure). 1 row for each case and strength, over the traces with
    at least 1 comparison."""
    c = df[df["extraction_class"].eq("comparison") & df["hedged"].notna()]
    t = c.groupby("tid").agg(hedged=("hedged", lambda s: float((s == "yes").mean())),
                             label=("presented_label", "first"),
                             case=("case", "first"))
    t["strength"] = np.where(t["label"].isin(["MuchMore", "MuchLess"]), "strong",
                             np.where(t["label"].isin(["More", "Less"]), "plain",
                                      "undecided"))
    return (t.groupby(["case", "strength"])["hedged"]
            .agg(traces="size", hedged_share="mean").reset_index())


def position_table(df: pd.DataFrame) -> pd.DataFrame:
    """The position effect: does Image A differ from Image B?

    For each case, the share that falls on A of:

    | Measure | Rows |
    |---------|------|
    | `decision_A` | decided traces: the winner is A |
    | `dimension_A` | dimension spans on A or B |
    | `good_A` | good dimension spans on A or B |
    | `bad_A` | bad dimension spans on A or B |
    | `favors_A` | perspectives that favor A or B |
    | `direction_A` | comparisons with direction A or B |
    | `inference_A` | inference spans on A or B |

    0.5 is no position effect. The interval is Wilson. Warning: the order
    of the pair is random (`is_swapped` is 0.5), thus a share far from 0.5
    is a fact about the model and not about the images.
    """
    rows = []
    tr = traces(df)
    for case in sorted(df["case"].unique()):
        sub = df[df["case"].eq(case)]
        t = tr[tr["case"].eq(case) & tr["winner"].ne("none")]
        rows.append({"case": case, "measure": "decision_A",
                     **_rate_row(int(t["winner"].eq("A").sum()), len(t))})
        specs = (("dimension_A", "dimension", None),
                 ("good_A", "dimension", "good"), ("bad_A", "dimension", "bad"),
                 ("favors_A", "perspective", None),
                 ("direction_A", "comparison", None),
                 ("inference_A", "inference", None))
        for name, cls, val in specs:
            part = sub[sub["extraction_class"].eq(cls) & sub["side"].notna()]
            if val is not None:
                part = part[part["valence"].eq(val)]
            rows.append({"case": case, "measure": name,
                         **_rate_row(int(part["side"].eq("A").sum()), len(part))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Question 2: contradictions
# --------------------------------------------------------------------------

def perspective_flips(df: pd.DataFrame, min_count: int = 25,
                      min_cases: int = 2) -> pd.DataFrame:
    """The perspectives whose lift is significant with 1 sign in 1 case and
    significant with the other sign in another case.

    The lift is `_ic_v3.lift_tables`. `pays_cases` and `costs_cases` list
    the cases where the Wilson interval clears the base rate.
    """
    t = V.lift_tables(df, "perspective", min_count=min_count)
    if t.empty:
        return t
    g = t.groupby("key")
    out = pd.DataFrame({
        "n_cases": g["case"].nunique(),
        "pays_cases": g.apply(lambda s: ", ".join(s.loc[s["pays"], "case"]), include_groups=False),
        "costs_cases": g.apply(lambda s: ", ".join(s.loc[s["costs"], "case"]), include_groups=False),
        "lift_min": g["lift"].min(), "lift_max": g["lift"].max(),
        "n": g["n"].sum(),
    }).reset_index()
    out["flips"] = out["pays_cases"].ne("") & out["costs_cases"].ne("")
    return out[out["n_cases"] >= min_cases].sort_values("n", ascending=False).reset_index(drop=True)


def trace_valence(df: pd.DataFrame, entity: str = "entity_sample") -> pd.DataFrame:
    """The valence each trace gives each dimension name on each image.

    1 row for each (case, entity, key_name, tid). `valence` is `good` or
    `bad` when the trace's good and bad rows for that name on that image do
    not tie; a tie, or neutral rows only, give no row.
    """
    dim = df[df["extraction_class"].eq("dimension") & df[entity].notna()
             & df["valence"].isin(["good", "bad"]) & df["key_name"].notna()]
    g = (dim.assign(_g=dim["valence"].eq("good").astype(int),
                    _b=dim["valence"].eq("bad").astype(int))
         .groupby(["case", entity, "key_name", "tid"])[["_g", "_b"]].sum())
    g = g[g["_g"] != g["_b"]]
    g["valence"] = np.where(g["_g"] > g["_b"], "good", "bad")
    return g.drop(columns=["_g", "_b"]).reset_index().rename(columns={entity: "entity"})


def trace_favor(df: pd.DataFrame, entity: str = "entity_sample") -> pd.DataFrame:
    """Whether each trace's perspective of a name favors each image.

    A perspective that favors 1 side gives the image of that side the value
    1 and the image of the other side the value 0. 1 row for each
    (case, entity, key_name, tid) where the trace's rows do not tie.
    """
    other = entity.replace("entity_", "other_")
    p = df[df["extraction_class"].eq("perspective") & df[entity].notna()
           & df["key_name"].notna()]
    pos = p[["case", entity, "key_name", "tid"]].assign(v=1)
    neg = (p[["case", other, "key_name", "tid"]]
           .rename(columns={other: entity}).assign(v=-1))
    both = pd.concat([pos, neg], ignore_index=True).dropna(subset=[entity])
    g = both.groupby(["case", entity, "key_name", "tid"])["v"].sum()
    g = g[g != 0]
    out = g.reset_index()
    out["valence"] = np.where(out["v"] > 0, "favors", "against")
    return out.drop(columns="v").rename(columns={entity: "entity"})


def consistency(votes: pd.DataFrame, by: Sequence[str] = ("case",)
                ) -> pd.DataFrame:
    """The within-entity consistency of a valence, against the chance level.

    `votes` holds `case`, `entity`, `key_name`, `tid`, `valence` (from
    `trace_valence` or `trace_favor`). A group is (case, entity, key_name)
    with 2 or more traces.

    | Column | Meaning |
    |--------|---------|
    | `groups` | groups with 2 or more traces |
    | `consistent` | groups where every trace gives the same valence |
    | `pair_agree` | over all the pairs of traces inside a group, the share that agree |
    | `pair_n` | those pairs of traces |
    | `chance_agree` | the agreement of 2 traces on DIFFERENT images, from the share of the name in the case: p^2 + (1-p)^2, weighted like `pair_agree` |
    | `excess` | `pair_agree - chance_agree` |

    Warning: `chance_agree` reads the marginal of the name in the case. A
    name that is almost always 1 valence gives a high chance level, thus
    read `excess`, not `pair_agree`.
    """
    if votes.empty:
        return pd.DataFrame()
    v = votes.copy()
    first = v["valence"].isin(["good", "favors"])
    v["_pos"] = first.astype(int)
    marg = v.groupby(["case", "key_name"])["_pos"].mean().rename("_p")
    grp = v.groupby(["case", "entity", "key_name"]).agg(t=("tid", "nunique"),
                                                        pos=("_pos", "sum"))
    grp = grp[grp["t"] >= 2].join(marg, on=["case", "key_name"])
    if grp.empty:
        return pd.DataFrame()
    neg = grp["t"] - grp["pos"]
    grp["pairs"] = grp["t"] * (grp["t"] - 1) / 2
    grp["agree"] = (grp["pos"] * (grp["pos"] - 1) + neg * (neg - 1)) / 2
    grp["chance"] = grp["pairs"] * (grp["_p"] ** 2 + (1 - grp["_p"]) ** 2)
    grp["consistent"] = (grp["pos"] == 0) | (neg == 0)
    grp = grp.reset_index()
    keys = list(by)
    rows = []
    for key, g in grp.groupby(keys) if keys else [((), grp)]:
        key = key if isinstance(key, tuple) else (key,)
        pa = g["agree"].sum() / g["pairs"].sum()
        ch = g["chance"].sum() / g["pairs"].sum()
        lo, hi = wilson(int(round(g["agree"].sum())), int(round(g["pairs"].sum())))
        rows.append({**dict(zip(keys, key)), "groups": len(g),
                     "consistent": int(g["consistent"].sum()),
                     "consistent_share": float(g["consistent"].mean()),
                     "pair_n": int(g["pairs"].sum()), "pair_agree": pa,
                     "lo": lo, "hi": hi, "chance_agree": ch, "excess": pa - ch})
    return pd.DataFrame(rows)


def partner_effect(votes: pd.DataFrame, df: pd.DataFrame,
                   by: Sequence[str] = ()) -> pd.DataFrame:
    """Does the valence of 1 image follow its outcome in the pair?

    Take 2 traces that name the same image with the same name, where the
    image wins in 1 trace and loses in the other. `k` counts the pairs of
    traces where the valence differs AND the good (or `favors`) valence sits
    in the win. `n` counts the pairs where the valence differs. A share near
    1 says that the image did not change but its description did, in the
    direction of the vote: the partner sets the description.

    Also gives `disagree_win_loss`: of the pairs with 1 win and 1 loss, the
    share that disagree, and `disagree_same_outcome`: the same share over the
    pairs with 2 wins or 2 losses.
    """
    if votes.empty:
        return pd.DataFrame()
    tr = traces(df).set_index("tid")
    meta = df[["tid", "sample_A", "sample_B"] + (["unit_A", "unit_B"] if "unit_A" in df.columns else [])
              ].drop_duplicates("tid").set_index("tid")
    v = votes.join(tr["winner"], on="tid").join(meta, on="tid")
    v = v[v["winner"].ne("none")]
    if v.empty:
        return pd.DataFrame()
    on_a = v["entity"].eq(v["sample_A"])
    on_b = v["entity"].eq(v["sample_B"])
    if "unit_A" in v.columns:
        on_a = on_a | v["entity"].eq(v["unit_A"])
        on_b = on_b | v["entity"].eq(v["unit_B"])
    v = v[on_a ^ on_b].copy()
    side = np.where(on_a[on_a ^ on_b], "A", "B")
    v["won"] = (side == v["winner"].to_numpy())
    v["pos"] = v["valence"].isin(["good", "favors"])
    m = v.merge(v, on=["case", "entity", "key_name"], suffixes=("_1", "_2"))
    m = m[m["tid_1"] < m["tid_2"]]
    keys = list(by)
    rows = []
    for key, g in (m.groupby(keys) if keys else [((), m)]):
        key = key if isinstance(key, tuple) else (key,)
        wl = g[g["won_1"] != g["won_2"]]
        same = g[g["won_1"] == g["won_2"]]
        dis = wl[wl["pos_1"] != wl["pos_2"]]
        k = int(((dis["pos_1"] & dis["won_1"]) | (dis["pos_2"] & dis["won_2"])).sum())
        r = _rate_row(k, len(dis))
        rows.append({**dict(zip(keys, key)), **r,
                     "win_loss_pairs": len(wl),
                     "disagree_win_loss": float((wl["pos_1"] != wl["pos_2"]).mean()) if len(wl) else float("nan"),
                     "same_outcome_pairs": len(same),
                     "disagree_same_outcome": float((same["pos_1"] != same["pos_2"]).mean()) if len(same) else float("nan")})
    return pd.DataFrame(rows)


def contradictions(votes: pd.DataFrame, df: pd.DataFrame, n: int = 20,
                   entity: str = "entity_sample") -> pd.DataFrame:
    """The clearest same-image contradictions: 1 image, 1 name, 2 valences.

    For each (case, entity, name) group whose traces disagree, 1 row with
    the pair ids, whether the image won each pair, and 1 span text of each
    valence (a span of that name, on that image, with that valence). The
    groups with the most traces come first.
    """
    if votes.empty:
        return pd.DataFrame()
    g = votes.groupby(["case", "entity", "key_name"])["valence"].nunique()
    bad = g[g > 1].reset_index()[["case", "entity", "key_name"]]
    if bad.empty:
        return pd.DataFrame()
    v = votes.merge(bad, on=["case", "entity", "key_name"])
    size = v.groupby(["case", "entity", "key_name"])["tid"].nunique().rename("traces")
    rows = []
    texts = df[df["extraction_class"].eq("dimension")][
        ["tid", "key_name", "extraction_text", "valence", entity, "side", "winner"]]
    for (case, ent, name), part in v.groupby(["case", "entity", "key_name"]):
        row = {"case": case, "entity": ent, "key_name": name,
               "traces": int(size[(case, ent, name)])}
        for val, sub in part.groupby("valence"):
            tid = sub["tid"].iloc[0]
            hit = texts[(texts["tid"] == tid) & (texts["key_name"] == name)
                        & (texts[entity] == ent) & (texts["valence"] == val)]
            row[f"{val}_pair"] = tid.split("|", 1)[1]
            if not hit.empty:
                row[f"{val}_won"] = (hit["winner"].iloc[0] == hit["side"].iloc[0])
                row[f"{val}_text"] = hit["extraction_text"].iloc[0][:160]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("traces", ascending=False).head(n).reset_index(drop=True)


def outcome_flips(df: pd.DataFrame, entity: str = "entity_sample"
                  ) -> pd.DataFrame:
    """The images that win 1 pair and lose another, with the same reason.

    For each (case, entity) in 2 or more decided traces: `wins`, `losses`,
    `flip` (at least 1 of each), and `shared_reason`: a dimension name with
    good valence on the image, or a perspective that favors it, in at least
    1 win AND at least 1 loss. `reason` names 1 such shared name.
    """
    other = entity.replace("entity_", "other_")
    tr = traces(df).set_index("tid")
    dec = tr[tr["winner"].ne("none")]
    if entity == "entity_sample":
        a_col, b_col = "sample_A", "sample_B"
    else:
        a_col, b_col = "unit_A", "unit_B"
    sides = df[["tid", a_col, b_col]].drop_duplicates("tid").set_index("tid")
    dec = dec.join(sides, how="inner")
    rows = []
    for side, col in (("A", a_col), ("B", b_col)):
        rows.append(pd.DataFrame({"case": dec["case"], "entity": dec[col],
                                  "tid": dec.index,
                                  "won": dec["winner"].eq(side).to_numpy()}
                                 ).reset_index(drop=True))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["entity"])
    # The same unit on both sides of 1 pair says nothing about its outcome.
    out = out.drop_duplicates(["tid", "entity"], keep=False)
    reasons = df[df[entity].notna() & df["key_name"].notna() & (
        (df["extraction_class"].eq("dimension") & df["valence"].eq("good"))
        | df["extraction_class"].eq("perspective"))][[entity, "tid", "key_name",
                                                     "extraction_class"]]
    reasons = reasons.rename(columns={entity: "entity"})
    reasons["reason"] = reasons["extraction_class"].str[:4] + ":" + reasons["key_name"]
    r = reasons.merge(out[["entity", "tid", "won"]], on=["entity", "tid"])
    shared = (r.groupby(["entity", "reason"])["won"].agg(["min", "max"]))
    shared = shared[(shared["min"] == 0) & (shared["max"] == 1)].reset_index()
    first_reason = shared.groupby("entity")["reason"].first()
    g = out.groupby(["case", "entity"]).agg(traces=("tid", "nunique"),
                                            wins=("won", "sum"))
    g = g[g["traces"] >= 2].reset_index()
    g["losses"] = g["traces"] - g["wins"]
    g["flip"] = (g["wins"] > 0) & (g["losses"] > 0)
    g["reason"] = g["entity"].map(first_reason)
    g["shared_reason"] = g["flip"] & g["reason"].notna()
    return g


def flip_summary(flips: pd.DataFrame) -> pd.DataFrame:
    """`outcome_flips` by case: images in 2 or more decided traces, the
    share that flip, and the share that flip with a shared reason."""
    if flips.empty:
        return pd.DataFrame()
    rows = []
    for case, g in flips.groupby("case"):
        rows.append({"case": case, "entities": len(g),
                     "flip_share": float(g["flip"].mean()),
                     "flip_with_reason": int(g["shared_reason"].sum()),
                     "flip_with_reason_share": float(g["shared_reason"].mean())})
    return pd.DataFrame(rows)


def repeat_pairs(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """The same canonical pair judged 2 times inside the sample.

    The pair table re-presents about 9% of the pairs (`repeat_idx` > 0),
    with the order of the sides drawn again. For each such pair with 2 or
    more traces: `same_winner` (the canonical winner agrees) and the Jaccard
    overlap of the dimension names of the 2 traces.
    """
    tr = traces(df).merge(meta[["case", "pair_id", "canonical_pair_id",
                                "canon_of_A"]], on=["case", "pair_id"])
    tr = tr[tr["winner"].ne("none")].copy()
    tr["canon_winner"] = np.where(tr["winner"].eq("A"), tr["canon_of_A"],
                                  np.where(tr["canon_of_A"].eq("a"), "b", "a"))
    names = (df[df["extraction_class"].eq("dimension") & df["key_name"].notna()]
             .groupby("tid")["key_name"].agg(lambda s: frozenset(s)))
    rows = []
    for (case, cp), g in tr.groupby(["case", "canonical_pair_id"]):
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        na, nb = names.get(a["tid"], frozenset()), names.get(b["tid"], frozenset())
        jac = len(na & nb) / len(na | nb) if (na | nb) else float("nan")
        rows.append({"case": case, "canonical_pair_id": cp,
                     "same_winner": a["canon_winner"] == b["canon_winner"],
                     "same_order": a["canon_of_A"] == b["canon_of_A"],
                     "jaccard": jac})
    return pd.DataFrame(rows)


def cross_prompt_valence(df: pd.DataFrame, case_x: str = "road_quality",
                         case_y: str = "street_photography") -> pd.DataFrame:
    """The same image and the same dimension name under 2 prompts.

    Road quality and street photography read the SAME pairs (the same
    `pair_id`, the same images, the same order). For each (pair, side,
    name) that both traces give a signed valence: does the valence agree?
    1 row for each name with the count and the agreement, and 1 row `all`.
    """
    v = trace_valence(df, "entity_sample")
    v["pair_id"] = v["tid"].str.split("|", n=1).str[1]
    x = v[v["case"].eq(case_x)]
    y = v[v["case"].eq(case_y)]
    m = x.merge(y, on=["pair_id", "entity", "key_name"], suffixes=("_x", "_y"))
    if m.empty:
        return pd.DataFrame()
    m["agree"] = m["valence_x"].eq(m["valence_y"])
    rows = [{"key_name": "all", **_rate_row(int(m["agree"].sum()), len(m)),
             "good_x": float(m["valence_x"].eq("good").mean()),
             "good_y": float(m["valence_y"].eq("good").mean())}]
    for name, g in m.groupby("key_name"):
        rows.append({"key_name": name, **_rate_row(int(g["agree"].sum()), len(g)),
                     "good_x": float(g["valence_x"].eq("good").mean()),
                     "good_y": float(g["valence_y"].eq("good").mean())})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def cross_prompt_winner(df: pd.DataFrame, case_x: str = "road_quality",
                        case_y: str = "street_photography") -> Dict[str, float]:
    """The same pair under 2 prompts: how often does the same image win?

    Over the pairs with a decided label in both cases. Both cases show the
    pair in the same order, thus the presented winner is comparable.
    `expected_same` is the agreement of 2 independent votes with the A
    shares of the 2 cases (the position effect), and `phi` is the
    correlation of the 2 votes (A = 1).
    """
    tr = traces(df)
    x = tr[tr["case"].eq(case_x)].set_index("pair_id")["winner"]
    y = tr[tr["case"].eq(case_y)].set_index("pair_id")["winner"]
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner")
    both = j[(j["x"] != "none") & (j["y"] != "none")]
    px, py = float(both["x"].eq("A").mean()), float(both["y"].eq("A").mean())
    phi = float(np.corrcoef(both["x"].eq("A"), both["y"].eq("A"))[0, 1]) if len(both) > 2 else float("nan")
    return {"pairs": int(len(j)), "both_decided": int(len(both)),
            "expected_same": px * py + (1 - px) * (1 - py), "phi": phi,
            **{f"same_{k}": v for k, v in
               _rate_row(int((both["x"] == both["y"]).sum()), len(both)).items()}}


def within_trace(df: pd.DataFrame) -> pd.DataFrame:
    """2 contradictions inside 1 trace, by case.

    | Column | Rule |
    |--------|------|
    | `both_valence` | share of traces with a dimension name that takes good AND bad on the SAME image |
    | `both_valence_nonaxis` | the same, over the names that `is_axis` does not mark |
    | `favors_loser` | share of decided traces with at least 1 perspective that favors the image that loses |
    | `majority_loser` | share of decided traces whose perspectives favor the loser more often than the winner |
    """
    rows = []
    tr = traces(df)
    for case in sorted(df["case"].unique()):
        sub = df[df["case"].eq(case)]
        n_tr = int(tr["case"].eq(case).sum())
        dim = sub[sub["extraction_class"].eq("dimension") & sub["side"].notna()
                  & sub["valence"].isin(["good", "bad"]) & sub["key_name"].notna()]
        vn = dim.groupby(["tid", "side", "key_name"])["valence"].nunique()
        both = vn[vn > 1].reset_index()
        both_nonaxis = both[~both["key_name"].map(is_axis)]
        p = sub[sub["extraction_class"].eq("perspective") & sub["side"].notna()
                & sub["winner"].ne("none")]
        pw = p.assign(_w=p["side"].eq(p["winner"]).astype(int),
                      _l=p["side"].ne(p["winner"]).astype(int)
                      ).groupby("tid")[["_w", "_l"]].sum()
        n_dec = int((tr["case"].eq(case) & tr["winner"].ne("none")).sum())
        rows.append({
            "case": case, "traces": n_tr,
            "both_valence": both["tid"].nunique() / n_tr,
            "both_valence_nonaxis": both_nonaxis["tid"].nunique() / n_tr,
            "decided": n_dec,
            "favors_loser": int((pw["_l"] > 0).sum()) / max(1, n_dec),
            "majority_loser": int((pw["_l"] > pw["_w"]).sum()) / max(1, n_dec),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Question 3: the proxy
# --------------------------------------------------------------------------

def _unit_values(case: str, meta: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The unit proxy of each canonical side of each pair: `value_a`,
    `value_b`. None when the case has no unit proxy."""
    from research.notebooks.cvpr import _unit_validation as U

    if case not in UNIT_PROXY:
        return None
    folder, proxy = UNIT_PROXY[case]
    m = meta[meta["case"].eq(case)].copy()
    if folder == "road":
        import geopandas as gpd
        from research.notebooks.cvpr import _proxies as px

        segments = px.pavement_segments(since_year=U.CASE_SPEC["road"]["pavement_since_year"])
        pts = []
        for s in ("a", "b"):
            pts.append(pd.DataFrame({"sample_id": m[f"sample_id_{s}"],
                                     "latitude": m[f"latitude_{s}"],
                                     "longitude": m[f"longitude_{s}"]}))
        pts = pd.concat(pts).dropna().drop_duplicates("sample_id")
        gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["longitude"], pts["latitude"]),
                               crs="EPSG:4326")
        hit = px.attach_nearest_segment(gdf, segments,
                                        max_distance_ft=U.CASE_SPEC["road"]["max_distance_ft"])
        val = hit.set_index("sample_id")["proxy_value"]
        m["value_a"] = m["sample_id_a"].map(val)
        m["value_b"] = m["sample_id_b"].map(val)
    else:
        prx = U.unit_proxy(folder, proxy).set_index("proxy_key")["proxy_value"]
        if folder == "subway":
            from research.notebooks.cvpr import _proxies_subway as sub

            key = sub.entrance_station_map().set_index("unit_uid")["station_key"]
        elif folder == "parks":
            key = U.unit_key("parks", pd.DataFrame()).set_index("unit_uid")["proxy_key"]
        else:
            key = None
        for s in ("a", "b"):
            k = m[f"unit_uid_{s}"] if key is None else m[f"unit_uid_{s}"].map(key)
            m[f"key_{s}"] = k
            m[f"value_{s}"] = k.map(prx)
        if "key_a" in m.columns:
            # 2 entrances of 1 station, or 1 park twice, carry no gap.
            same = m["key_a"].eq(m["key_b"])
            m.loc[same, ["value_a", "value_b"]] = np.nan
    m["proxy"] = proxy
    m["mode"] = "unit"
    return m


def _area_values(case: str, meta: pd.DataFrame, proxy: str,
                 layer: str = "community_district") -> Optional[pd.DataFrame]:
    """The polygon proxy of each canonical side (`_ic_link`'s join)."""
    from research.notebooks.cvpr import _ic_link as L

    if case not in L.linkable_cases() or proxy not in L.proxies_for(case):
        return None
    m = meta[meta["case"].eq(case)].copy()
    prx = L.proxy_by_key(case, layer, proxy).set_index("poly_key")["proxy_mean"]
    if case in L.IMAGE_CASES:
        m = L.image_polygons(m, layer)
    else:
        place = L.unit_polygons(case, layer).set_index("unit_uid")["poly_key"]
        m["poly_a"] = m["unit_uid_a"].map(place)
        m["poly_b"] = m["unit_uid_b"].map(place)
    m["value_a"] = m["poly_a"].map(prx)
    m["value_b"] = m["poly_b"].map(prx)
    same = m["poly_a"].eq(m["poly_b"])
    m.loc[same, ["value_a", "value_b"]] = np.nan
    m["proxy"] = proxy
    m["mode"] = "area"
    return m


def proxy_pairs(meta: pd.DataFrame, case: str, proxy: Optional[str] = None,
                mode: str = "unit", quantiles: int = 4) -> pd.DataFrame:
    """1 row for each usable pair of 1 case: the gap, the vote, the band.

    `mode = "unit"` reads the headline unit proxy of `UNIT_PROXY`
    (`_unit_validation`): the value of the unit itself (the station for the
    subway case, the DPR property for parks, the nearest rated segment for
    road quality). `mode = "area"` reads `_ic_link`: the value of the
    polygon that holds the unit (or the camera).

        proxy_gap = value(canonical a) - value(canonical b)
        agrees    = sign(proxy_gap) == sign(relative_score)

    The proxy values arrive "higher is better". A pair drops out when a
    side has no value, the 2 sides share a unit or a polygon, the gap is 0,
    or the label is Same, NotSure, or missing. `gap_band` cuts `abs_gap`
    into `quantiles` bands inside the case (0 = the smallest gaps, the
    hardest pairs).

    Warning: `meta` must carry the judge's own `relative_score`.
    """
    if mode == "unit":
        m = _unit_values(case, meta)
    else:
        m = _area_values(case, meta, proxy)
    if m is None or m.empty:
        return pd.DataFrame()
    m["proxy_gap"] = m["value_a"] - m["value_b"]
    m["model_dir"] = np.sign(pd.to_numeric(m["relative_score"], errors="coerce"))
    keep = (m["proxy_gap"].notna() & (m["proxy_gap"] != 0)
            & m["model_dir"].notna() & (m["model_dir"] != 0))
    m = m[keep].copy()
    if m.empty:
        return m
    m["agrees"] = np.sign(m["proxy_gap"]).eq(m["model_dir"])
    m["abs_gap"] = m["proxy_gap"].abs()
    m["gap_band"] = pd.qcut(m["abs_gap"].rank(method="first"), quantiles,
                            labels=False)
    cols = ["case", "pair_id", "proxy", "mode", "proxy_gap", "abs_gap",
            "gap_band", "model_dir", "agrees", "canon_of_A"]
    return m[cols].reset_index(drop=True)


def proxy_plan(cases: Sequence[str] = CASES) -> List[Tuple[str, str, Optional[str]]]:
    """The (case, mode, proxy) cells the notebook reads: the unit headline
    of each case, then the 2 area proxies where they exist."""
    from research.notebooks.cvpr import _ic_link as L

    plan: List[Tuple[str, str, Optional[str]]] = []
    for case in cases:
        if case in UNIT_PROXY:
            plan.append((case, "unit", UNIT_PROXY[case][1]))
        names = L.proxies_for(case) if case in L.linkable_cases() else []
        for prx in AREA_PROXIES:
            if prx in names:
                plan.append((case, "area", prx))
    return plan


def all_proxy_pairs(meta: pd.DataFrame, tids: Optional[Iterable[str]] = None,
                    cases: Sequence[str] = CASES, quantiles: int = 4
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """`proxy_pairs` for every cell of `proxy_plan`, cut to the sampled traces.

    The bands are cut over the pairs of the sample, inside each cell.

    Returns (pairs, coverage). `coverage` gives, for each cell, the sampled
    traces, the usable pairs, and the agreement rate with its interval, or
    a `note` that says why a cell is empty.
    """
    rows, frames = [], []
    keep = set(tids) if tids is not None else None
    for case, mode, prx in proxy_plan(cases):
        sample = meta[meta["case"].eq(case)]
        if keep is not None:
            sample = sample[(sample["case"] + "|" + sample["pair_id"]).isin(keep)]
        try:
            pairs = proxy_pairs(sample, case, prx, mode, quantiles)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            rows.append({"case": case, "mode": mode, "proxy": prx, "traces": len(sample),
                         "pairs": 0, "note": f"{type(exc).__name__}: {exc}"})
            continue
        if pairs.empty:
            rows.append({"case": case, "mode": mode, "proxy": prx, "traces": len(sample),
                         "pairs": 0, "note": "no usable pair"})
            continue
        pairs["tid"] = pairs["case"] + "|" + pairs["pair_id"].astype(str)
        pairs["cell"] = pairs["case"] + "|" + mode + "|" + str(prx)
        frames.append(pairs)
        r = _rate_row(int(pairs["agrees"].sum()), len(pairs))
        rows.append({"case": case, "mode": mode, "proxy": prx, "traces": len(sample),
                     "pairs": len(pairs), "agreement": r["rate"], "lo": r["lo"],
                     "hi": r["hi"], "note": ""})
    rows.append({"case": "street_photography", "mode": "none", "proxy": None,
                 "pairs": 0, "note": "no outside measurement for this case"})
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out, pd.DataFrame(rows)


def costly_perspectives(df: pd.DataFrame, min_count: int = 25) -> pd.DataFrame:
    """The perspectives the model names and then overrules, by case.

    A perspective is costly in a case when the Wilson interval of its win
    rate sits under the base rate (`_ic_v3.lift_tables`, `costs`).
    """
    t = V.lift_tables(df, "perspective", min_count=min_count)
    if t.empty:
        return pd.DataFrame(columns=["case", "key"])
    return t[t["costs"]][["case", "key", "n", "lift"]].reset_index(drop=True)


def trace_features(df: pd.DataFrame, costly: Optional[pd.DataFrame] = None
                   ) -> pd.DataFrame:
    """1 row for each trace, with the boolean and count features of question 3.

    | Feature | Rule |
    |---------|------|
    | `wk_for_winner` | a world-knowledge perspective favors the winner |
    | `tc_for_winner` | a task-convention perspective favors the winner |
    | `has_tc` | at least 1 task-convention perspective |
    | `tc_beats_wk` | a task-convention perspective favors the winner and a world-knowledge one favors the loser |
    | `wk_beats_tc` | the reverse |
    | `premise_dismissal` | at least 1 dismissal with `about = premise` |
    | `cue_dismissal` | at least 1 dismissal with `about = cue` |
    | `good_margin_pos` | `good_margin` > 0: more good (and fewer bad) names on the winner |
    | `good_margin_hi` | `good_margin` >= 2 |
    | `costly` | the trace names a perspective that is costly in its case |
    | `risk_inference` | an inference with kind wealth or class |
    | `wealth_inference` | an inference with kind wealth |
    | `favors_loser` | a perspective favors the loser |
    | `hedged_any` | a hedged comparison |
    | `reconsiders` | an uncertainty with marker `reconsideration` |
    | `person_used` | a person reference with `used_in_judgment = yes` |
    | `strong_label` | MuchMore or MuchLess |
    | `n_spans` | spans of the trace |
    """
    tr = traces(df).set_index("tid")
    p = df[df["extraction_class"].eq("perspective") & df["side"].notna()]
    on_w = p["side"].eq(p["winner"])

    def flag(mask: pd.Series) -> pd.Series:
        return pd.Series(True, index=pd.Index(p.loc[mask, "tid"].unique()))

    feats = pd.DataFrame(index=tr.index)
    wk_w = flag(p["source"].eq("world_knowledge") & on_w)
    wk_l = flag(p["source"].eq("world_knowledge") & ~on_w)
    tc_w = flag(p["source"].eq("task_convention") & on_w)
    tc_l = flag(p["source"].eq("task_convention") & ~on_w)
    feats["wk_for_winner"] = feats.index.isin(wk_w.index)
    feats["tc_for_winner"] = feats.index.isin(tc_w.index)
    feats["has_tc"] = feats.index.isin(
        df.loc[df["extraction_class"].eq("perspective")
               & df["source"].eq("task_convention"), "tid"].unique())
    feats["tc_beats_wk"] = feats.index.isin(tc_w.index) & feats.index.isin(wk_l.index)
    feats["wk_beats_tc"] = feats.index.isin(wk_w.index) & feats.index.isin(tc_l.index)
    feats["favors_loser"] = feats.index.isin(flag(~on_w).index)
    dm = df[df["extraction_class"].eq("dismissal")]
    feats["premise_dismissal"] = feats.index.isin(dm.loc[dm["about"].eq("premise"), "tid"].unique())
    feats["cue_dismissal"] = feats.index.isin(dm.loc[dm["about"].eq("cue"), "tid"].unique())
    gm = good_margin(df).set_index("tid")["margin"]
    feats["good_margin"] = gm.reindex(feats.index).fillna(0)
    feats["good_margin_pos"] = feats["good_margin"] > 0
    feats["good_margin_hi"] = feats["good_margin"] >= 2
    inf = df[df["extraction_class"].eq("inference")]
    feats["risk_inference"] = feats.index.isin(inf.loc[inf["kind"].isin(RISK_KINDS), "tid"].unique())
    feats["wealth_inference"] = feats.index.isin(inf.loc[inf["kind"].eq("wealth"), "tid"].unique())
    cmp_ = df[df["extraction_class"].eq("comparison")]
    feats["hedged_any"] = feats.index.isin(cmp_.loc[cmp_["hedged"].eq("yes"), "tid"].unique())
    un = df[df["extraction_class"].eq("uncertainty")]
    feats["reconsiders"] = feats.index.isin(un.loc[un["marker"].eq("reconsideration"), "tid"].unique())
    pr = df[df["extraction_class"].eq("person_reference")]
    feats["person_used"] = feats.index.isin(pr.loc[pr["used_in_judgment"].eq("yes"), "tid"].unique())
    feats["strong_label"] = tr["presented_label"].isin(["MuchMore", "MuchLess"])
    feats["n_spans"] = df.groupby("tid").size().reindex(feats.index).fillna(0).astype(int)
    if costly is not None and not costly.empty:
        pp = df[df["extraction_class"].eq("perspective")][["tid", "case", "key_name"]]
        hit = pp.merge(costly[["case", "key"]], left_on=["case", "key_name"],
                       right_on=["case", "key"])
        feats["costly"] = feats.index.isin(hit["tid"].unique())
    else:
        feats["costly"] = False
    return feats.reset_index()


BOOL_FEATURES = ("wk_for_winner", "tc_for_winner", "has_tc", "tc_beats_wk",
                 "wk_beats_tc", "premise_dismissal", "cue_dismissal",
                 "good_margin_pos", "good_margin_hi", "costly", "risk_inference",
                 "wealth_inference", "favors_loser", "hedged_any", "reconsiders",
                 "person_used", "strong_label")


def banded_contrast(frame: pd.DataFrame, feature: str,
                    strata: Sequence[str] = ("cell", "gap_band"),
                    min_n: int = 5) -> Dict[str, float]:
    """The agreement difference (feature present - absent) inside strata.

    Each stratum (by default 1 proxy cell and 1 gap band) gives
    d = p1 - p0. The pooled value is the Mantel-Haenszel risk difference:
    the weight of a stratum is n1 * n0 / (n1 + n0), and the variance sums
    w^2 (p1 q1 / n1 + p0 q0 / n0). A stratum with fewer than `min_n` pairs
    in a group drops out. The interval is the normal 95% interval of that
    pooled difference.

    Returns `diff`, `lo`, `hi`, `n1`, `n0` (pairs in the kept strata),
    `rate1`, `rate0` (pooled raw rates of those pairs), and `strata`.
    """
    f = frame.dropna(subset=[feature])
    num = den = var = 0.0
    n1 = n0 = k1 = k0 = 0
    kept = 0
    for _, g in f.groupby(list(strata)):
        x = g[feature].astype(bool)
        a, b = g.loc[x, "agrees"], g.loc[~x, "agrees"]
        if len(a) < min_n or len(b) < min_n:
            continue
        p1, p0 = a.mean(), b.mean()
        w = len(a) * len(b) / (len(a) + len(b))
        num += w * (p1 - p0)
        den += w
        var += w * w * (p1 * (1 - p1) / len(a) + p0 * (1 - p0) / len(b))
        n1 += len(a); n0 += len(b); k1 += int(a.sum()); k0 += int(b.sum())
        kept += 1
    if den == 0:
        return {"feature": feature, "diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n1": n1, "n0": n0, "rate1": float("nan"),
                "rate0": float("nan"), "strata": 0}
    d = num / den
    se = math.sqrt(var) / den
    return {"feature": feature, "diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se,
            "n1": n1, "n0": n0, "rate1": k1 / n1, "rate0": k0 / n0, "strata": kept}


def contrast_table(frame: pd.DataFrame, features: Sequence[str] = BOOL_FEATURES,
                   by: Optional[str] = None, **kw) -> pd.DataFrame:
    """`banded_contrast` for each feature, pooled or by `by` (for example
    `mode` or `case`)."""
    rows = []
    groups = [("all", frame)] if by is None else list(frame.groupby(by))
    for key, g in groups:
        for feat in features:
            if feat not in g.columns:
                continue
            rows.append({"group": key, **banded_contrast(g, feat, **kw)})
    return pd.DataFrame(rows)


def name_markers(frame: pd.DataFrame, df: pd.DataFrame,
                 extraction_class: str = "perspective", min_traces: int = 25,
                 with_valence: bool = False, **kw) -> pd.DataFrame:
    """Which names mark the traces that agree with the proxy?

    For each name of the class (in `min_traces` or more traces of the proxy
    frame): `banded_contrast` of the name present against absent, inside the
    strata. `with_valence` keys a dimension by `name | valence`.

    Warning: this tests many names. Read a name only when its interval
    clears 0 and it holds under the other extractor.
    """
    part = df[df["extraction_class"].eq(extraction_class) & df["key_name"].notna()]
    key = part["key_name"]
    if with_valence:
        key = key + " | " + part["valence"].astype(str)
    names = pd.DataFrame({"tid": part["tid"], "key": key}).drop_duplicates()
    names = names[names["tid"].isin(set(frame["tid"]))]
    counts = names.groupby("key")["tid"].nunique()
    rows = []
    for name in counts[counts >= min_traces].index:
        tids = set(names.loc[names["key"].eq(name), "tid"])
        f = frame.assign(_has=frame["tid"].isin(tids))
        r = banded_contrast(f, "_has", **kw)
        r["feature"] = name
        r["traces"] = int(counts[name])
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values("diff", ascending=False).reset_index(drop=True) if not out.empty else out


def proxy_frame(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join the proxy pairs to the trace features by `tid`."""
    return pairs.merge(features, on="tid", how="inner")


# --------------------------------------------------------------------------
# Headlines: the key numbers of 1 arm, to read beside the other extractor
# --------------------------------------------------------------------------

def _pooled(table: pd.DataFrame) -> Dict[str, float]:
    k, n = int(table["k"].sum()), int(table["n"].sum())
    return _rate_row(k, n)


def headline_q12(df: pd.DataFrame) -> pd.DataFrame:
    """The key numbers of questions 1 and 2 for 1 arm, 1 row for each number.

    `df` must carry the image columns (`attach_images`). Columns: `measure`,
    `value`, `lo`, `hi`, `n`. Read 2 arms of 1 judge side by side: a number
    that holds under both extractors comes from the traces.
    """
    rows = []

    def add(measure, value, lo=float("nan"), hi=float("nan"), n=0):
        rows.append({"measure": measure, "value": value, "lo": lo, "hi": hi, "n": n})

    vv = valence_vote(df)
    for val in ("good", "bad"):
        r = _pooled(vv[vv["valence"].eq(val)])
        add(f"{val} dimension on the winner", r["rate"], r["lo"], r["hi"], r["n"])
    kl = keyed_win(df, "inference", "kind", 25)
    if not kl.empty:
        kl = kl.assign(pays=kl["ci_low"] > kl["base_rate"], costs=kl["ci_high"] < kl["base_rate"])
        for kind in ("wealth", "class"):
            part = kl[kl["key"].eq(kind)]
            add(f"inference {kind}: cases that pay / cases", int(part["pays"].sum()),
                n=len(part))
            add(f"inference {kind}: cases that cost / cases", int(part["costs"].sum()),
                n=len(part))
            add(f"inference {kind}: mean lift", float(part["lift"].mean()), n=int(part["n"].sum()))
    st = source_table(df)
    for src in ("world_knowledge", "task_convention"):
        part = st[st["source"].eq(src)]
        r = _pooled(part)
        add(f"perspective {src}: favored image wins", r["rate"], r["lo"], r["hi"], r["n"])
    pt = position_table(df)
    for meas in ("decision_A", "favors_A", "good_A", "bad_A"):
        r = _pooled(pt[pt["measure"].eq(meas)])
        add(f"position: {meas}", r["rate"], r["lo"], r["hi"], r["n"])
    for ent in ("entity_sample", "entity_unit"):
        v = trace_valence(df, ent)
        c = consistency(v, by=())
        if not c.empty:
            c = c.iloc[0]
            add(f"{ent}: same name, same valence (pairs of traces)", c["pair_agree"],
                c["lo"], c["hi"], int(c["pair_n"]))
            add(f"{ent}: chance agreement", c["chance_agree"], n=int(c["pair_n"]))
        pe = partner_effect(v, df)
        if not pe.empty:
            pe = pe.iloc[0]
            add(f"{ent}: good valence sits in the win, of the discordant win-loss pairs",
                pe["rate"], pe["lo"], pe["hi"], int(pe["n"]))
            add(f"{ent}: valence differs, 1 win and 1 loss", pe["disagree_win_loss"],
                n=int(pe["win_loss_pairs"]))
            add(f"{ent}: valence differs, same outcome", pe["disagree_same_outcome"],
                n=int(pe["same_outcome_pairs"]))
    cw = cross_prompt_winner(df)
    add("road vs street photography: same winner", cw["same_rate"], cw["same_lo"],
        cw["same_hi"], cw["same_n"])
    add("road vs street photography: expected under independence", cw["expected_same"])
    cv = cross_prompt_valence(df)
    if not cv.empty:
        r = cv.iloc[0]
        add("road vs street photography: same name, same image, same valence", r["rate"],
            r["lo"], r["hi"], int(r["n"]))
    return pd.DataFrame(rows)


def headline_q3(frame: pd.DataFrame, features: Sequence[str] = (
        "wealth_inference", "risk_inference", "good_margin_hi", "premise_dismissal",
        "costly", "favors_loser", "tc_for_winner")) -> pd.DataFrame:
    """The band-controlled contrasts of question 3, by proxy family.

    `family` is `income` (median household income of the polygon),
    `crime_area` (crime density of the polygon, negated), or `unit` (the
    headline unit proxy of the case).
    """
    f = frame.copy()
    f["family"] = proxy_family(f["proxy"])
    return contrast_table(f, features=features, by="family")


def proxy_family(proxy: pd.Series) -> pd.Series:
    """`income`, `crime_area`, or `unit` for each proxy name."""
    return pd.Series(np.where(proxy.eq("median_household_income"), "income",
                              np.where(proxy.eq("crime_density"), "crime_area", "unit")),
                     index=proxy.index)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
#
# A figure carries no caption prose. Each `caption_<figure>` function builds
# the caption from the frame that the plot function draws, and the plot
# function attaches it (`_figmeta.attach`). The notebook shows it under the
# figure, and `_figmeta.save_figure` writes it beside the PNG.

def _meta(notebook: str, label: str, title: str, caption: str, measure: str,
          cases: Sequence[str] = (), arm: Optional[str] = None) -> Dict:
    """The caption content of 1 figure of this module."""
    spec = X.ARMS.get(arm or "")
    return F.figure_meta(
        title, caption, F.fig_label("ic-meta", label, arm),
        [notebook, *cases, *([arm] if arm else []), "v3", measure],
        module=_MODULE, version=__version__, arm=arm,
        corpus_root=spec["root"] if spec else None)


def _arm_text(arm: Optional[str]) -> str:
    spec = X.ARMS.get(arm or "")
    if not spec:
        return "Corpus: the version 3 extraction corpus of the notebook."
    return (f"Corpus: the {spec['judge']} traces, read by the "
            f"{spec['extractor']} extractor (arm `{arm}`, the "
            f"{V.arm_role(arm)} arm of the judge, schema "
            f"{spec['schema_version']}).")


def _grid_names(table: pd.DataFrame, names: Optional[Sequence[str]], n: int
                ) -> List[str]:
    if names is not None:
        return list(names)
    summ = valence_summary(table)
    return summ[summ["n_cases"] >= 3].head(n)["key_name"].tolist()


def caption_valence_grid(table: pd.DataFrame,
                         names: Optional[Sequence[str]] = None, n: int = 20,
                         arm: Optional[str] = None) -> Dict:
    """The caption of `plot_valence_grid`."""
    names = _grid_names(table, names, n)
    cases = [c for c in CASES if c in set(table["case"])]
    caption = (
        f"The valence of each dimension name, over the cases: the {len(names)} "
        "names with the most rows in 3 cases or more (rows), against the "
        f"cases (columns, {len(cases)}). The colour is the net valence, "
        "(good - bad) / (good + bad) of the signed rows, on the diverging "
        "ramp: -1 (coral) is always bad and +1 (teal) is always good. An "
        "empty cell is a name under the count floor in that case (25 good "
        "or bad rows by default). A row is a span, not a trace. A name that "
        "changes colour along its row flips its valence between the cases. "
        + _arm_text(arm))
    return _meta("aesthetic_consistency", "valence-grid",
                 "The valence of each dimension name", caption, "net_valence",
                 cases, arm)


def caption_partner_effect(rows: pd.DataFrame, arm: Optional[str] = None
                           ) -> Dict:
    """The caption of `plot_partner_effect`."""
    caption = (
        "The partner test. Take 2 traces that name the same image (or the "
        "same facility) with the same name. 1 pair of bars for each class "
        "and entity: the share of those trace pairs whose valence differs, "
        "when the image wins in 1 trace and loses in the other (coral), and "
        "when the outcome is the same (slate). A coral bar much longer than "
        "the slate bar says that the image did not change but its "
        "description did, with the vote: the partner image sets the "
        "description. For a perspective, the valence is `favors` or "
        "`against`. " + _arm_text(arm))
    return _meta("aesthetic_consistency", "partner-effect",
                 "The partner effect on the valence", caption,
                 "valence_disagreement", [], arm)


def caption_contrasts(table: pd.DataFrame, title: str = "",
                      arm: Optional[str] = None) -> Dict:
    """The caption of `plot_contrasts`."""
    t = table.dropna(subset=["diff"])
    groups = sorted(set(map(str, t["group"]))) if "group" in t else []
    caption = (
        "Does a trace feature go with agreement with the proxy? A forest "
        "plot: 1 line for each feature and proxy family (in brackets). The "
        "dot is the agreement of the traces with the feature less the "
        "agreement of the traces without it (the share whose vote has the "
        "sign of the proxy gap), pooled inside strata of 1 proxy cell and 1 "
        "gap band (a Mantel-Haenszel risk difference). The line is its "
        "normal 95% interval, and the dotted line is 0. The colour is the "
        "family: amber `income` (median household income of the polygon), "
        "coral `crime_area` (crime density of the polygon, negated), teal "
        "`unit` (the headline unit proxy), blue `all`."
        + (f" Families in the figure: {', '.join(groups)}." if groups else "")
        + " Warning: a contrast is an association inside the bands, not a "
        "cause. " + _arm_text(arm))
    return _meta("reasoning_vs_proxy", "proxy-contrasts",
                 "Trace features against the proxy", caption,
                 "proxy_agreement_diff", [], arm)

def plot_valence_grid(table: pd.DataFrame, names: Optional[Sequence[str]] = None,
                      n: int = 20, arm: Optional[str] = None):
    """A heat map of `net` valence: dimension names against the cases.

    `names` defaults to the `n` names with the most rows over 3 cases or
    more. The colour runs from -1 (always bad) to +1 (always good). An empty
    cell is a name under the count floor in that case.
    """
    import matplotlib.pyplot as plt
    from research.notebooks.cvpr import _style as S

    names = _grid_names(table, names, n)
    grid = (table[table["key_name"].isin(names)]
            .pivot(index="key_name", columns="case", values="net")
            .reindex(index=names, columns=[c for c in CASES if c in set(table["case"])]))
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(names) + 1.4))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap=S.CMAP_DIV, vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([c.replace("_", " ") for c in grid.columns], rotation=35,
                       ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04, label="net valence (good - bad) / signed")
    fig.tight_layout()
    return F.attach(fig, caption_valence_grid(table, names, n, arm))


def plot_partner_effect(rows: pd.DataFrame, arm: Optional[str] = None):
    """2 bars for each entity and class: the share of pairs of traces whose
    valence differs, when the outcome differs and when it is the same.

    `rows` holds `label`, `disagree_win_loss`, `disagree_same_outcome`.
    """
    import matplotlib.pyplot as plt
    from research.notebooks.cvpr import _style as S

    fig, ax = plt.subplots(figsize=(4.8, 0.45 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y - 0.18, rows["disagree_win_loss"], height=0.35, color=S.PAL["coral"],
            edgecolor=S.EDGE, linewidth=S.EDGE_LW, label="1 win, 1 loss")
    ax.barh(y + 0.18, rows["disagree_same_outcome"], height=0.35, color=S.PAL["slate"],
            edgecolor=S.EDGE, linewidth=S.EDGE_LW, label="same outcome")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=7)
    ax.set_xlabel("share of trace pairs whose valence differs")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    return F.attach(fig, caption_partner_effect(rows, arm))


def plot_contrasts(table: pd.DataFrame, title: str = "",
                   arm: Optional[str] = None):
    """A forest plot of `banded_contrast` rows: 1 line for each feature and
    group, the pooled difference with its 95% interval."""
    import matplotlib.pyplot as plt
    from research.notebooks.cvpr import _style as S

    t = table.dropna(subset=["diff"]).reset_index(drop=True)
    colours = {"income": S.PAL["amber"], "crime_area": S.PAL["coral"],
               "unit": S.PAL["teal"], "all": S.PAL["blue"]}
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(t) + 1.2))
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        c = colours.get(str(r["group"]), S.PAL["slate"])
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=c, linewidth=1.2)
        ax.plot(r["diff"], yi, "o", color=c, markersize=3.5)
    ax.axvline(0, color=S.EDGE, linewidth=0.6, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['feature']} ({r['group']})" for _, r in t.iterrows()],
                       fontsize=7)
    ax.set_xlabel("agreement with the proxy: present - absent, inside gap bands")
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return F.attach(fig, caption_contrasts(table, title, arm))


# --------------------------------------------------------------------------
# Paper tables (LaTeX)
# --------------------------------------------------------------------------
#
# Each function takes `{arm: frame}`, with the arms in column order. The
# notebooks pass the arm they read; `scripts` or a driver can pass the 2
# primary arms. The frames are the frames that the notebooks compute.

_MODULE = "_ic_metadata"

#: The rows of the partner and the consistency tables: (class, entity, label).
_FLIP_ROWS = (
    ("dimension", "image", "Dimension, same image"),
    ("dimension", "facility", "Dimension, same facility"),
    ("perspective", "image", "Perspective, same image"),
    ("perspective", "facility", "Perspective, same facility"),
)

#: The trace features of the proxy table, with their printed names.
FEATURE_TEX = {
    "wealth_inference": "Wealth inference",
    "risk_inference": "Wealth or class inference",
    "good_margin_hi": "Good-name margin $\\geq 2$",
    "premise_dismissal": "Premise dismissal",
    "costly": "Costly perspective",
    "favors_loser": "Perspective favors loser",
    "tc_for_winner": "Task convention for winner",
    "hedged_any": "Hedged comparison",
}

#: The proxy families, in column order, with their printed names.
FAMILY_TEX = {"income": "Income (area)", "crime_area": "Low crime (area)",
              "unit": "Unit proxy"}


def _arm_heads(arms: Sequence[str], width: int, lead: int = 1) -> List[str]:
    from research.notebooks.cvpr import _latex as L
    return L.group_header([(L.arm_label(a), width) for a in arms], lead=lead)


def _pick(frame: Optional[pd.DataFrame], cls: str, entity: str) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None
    hit = frame[frame["extraction_class"].eq(cls) & frame["entity"].eq(entity)]
    return hit if not hit.empty else None


VALENCE_FLIP_CAPTION = (
    "\\textbf{Does the description of a fixed image follow the vote?} Take 2 "
    "traces that name the same image (or facility) with the same name, where "
    "the image wins 1 pair and loses the other. \\emph{win} is the share of the "
    "pairs with 2 different valences where the good valence (for a perspective, "
    "\\emph{favors}) sits in the win; $n$ counts those pairs. \\emph{W/L} and "
    "\\emph{same} are the shares of the trace pairs whose valence differs, when "
    "the outcome differs and when it is the same."
)


def latex_valence_flip(partner: Dict[str, pd.DataFrame],
                       caption: str = VALENCE_FLIP_CAPTION,
                       label: str = "tab:ic-meta-valence-flip") -> str:
    """The win-loss valence flip: `partner_effect` rows, 4 columns per arm."""
    from research.notebooks.cvpr import _latex as L
    arms = list(partner)
    head = _arm_heads(arms, 4) + [L.row(["Entity"] + ["win", "$n$", "W/L", "same"] * len(arms))]
    body = []
    for cls, ent, text in _FLIP_ROWS:
        cells = [text]
        for a in arms:
            r = _pick(partner[a], cls, ent)
            if r is None:
                cells += [L.DASH] * 4
                continue
            r = r.iloc[0]
            cells += [L.fmt(r["rate"]), L.fmt_int(r["n"]),
                      L.fmt(r["disagree_win_loss"]), L.fmt(r["disagree_same_outcome"])]
        body.append(L.row(cells))
    return L.table(_MODULE, __version__, caption, label, "l" + "rrrr" * len(arms),
                   head, body, resize=True)


SAME_IMAGE_CAPTION = (
    "\\textbf{Does the same image keep its valence in 2 pairs?} Of the pairs of "
    "traces that name 1 image (or 1 facility) with 1 name, the share that give "
    "the same valence (\\emph{agree}, with the 95\\% Wilson interval), against "
    "\\emph{chance}: 2 traces on different images, from the share of the name in "
    "the case. $n$ counts the trace pairs, pooled over the cases."
)


def _pooled_consistency(rows: pd.DataFrame) -> Dict[str, float]:
    n = float(rows["pair_n"].sum())
    if n <= 0:
        return {"agree": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "chance": float("nan"), "n": 0}
    agree = float((rows["pair_agree"] * rows["pair_n"]).sum() / n)
    chance = float((rows["chance_agree"] * rows["pair_n"]).sum() / n)
    lo, hi = wilson(int(round(agree * n)), int(n))
    return {"agree": agree, "lo": lo, "hi": hi, "chance": chance, "n": int(n)}


def latex_same_image(same_image: Dict[str, pd.DataFrame],
                     caption: str = SAME_IMAGE_CAPTION,
                     label: str = "tab:ic-meta-same-image") -> str:
    """The same-image consistency: the `consistency` rows by case, pooled."""
    from research.notebooks.cvpr import _latex as L
    arms = list(same_image)
    head = _arm_heads(arms, 4) + [L.row(["Entity"] + ["agree", "95\\% CI", "chance", "$n$"] * len(arms))]
    body = []
    for cls, ent, text in _FLIP_ROWS:
        cells = [text]
        for a in arms:
            r = _pick(same_image[a], cls, ent)
            if r is None:
                cells += [L.DASH] * 4
                continue
            p = _pooled_consistency(r)
            cells += [L.fmt(p["agree"]), L.fmt_ci(p["lo"], p["hi"]),
                      L.fmt(p["chance"]), L.fmt_int(p["n"])]
        body.append(L.row(cells))
    return L.table(_MODULE, __version__, caption, label, "l" + "rrrr" * len(arms),
                   head, body, resize=True)


CONTENT_CAPTION = (
    "\\textbf{Does one content hierarchy serve every prompt?} Top: the property "
    "names (not axes) in 3 cases or more. \\emph{+/--} counts the cases where "
    "the good share of the name clears 0.5 above or below (Wilson interval), and "
    "\\emph{range} is the lowest and the highest net valence over the cases. "
    "Bottom: the perspectives that pay in 1 case and cost in another; "
    "\\emph{+/--} counts the cases where the win rate clears the base rate above "
    "or below, and \\emph{range} is the range of the lift. $\\dagger$ marks a "
    "name that flips on that arm."
)


def _count_cases(text) -> int:
    return len([c for c in str(text).split(", ") if c]) if isinstance(text, str) else 0


def hierarchy_names(summaries: Dict[str, pd.DataFrame], n: int = 10,
                    min_cases: int = 3) -> List[str]:
    """The property names to print: in `min_cases` cases or more on at least
    1 arm, ranked by the rows over the arms."""
    total: Dict[str, float] = {}
    for s in summaries.values():
        if s is None or s.empty:
            continue
        keep = s[(~s["axis"].astype(bool)) & (s["n_cases"] >= min_cases)]
        for name, k in zip(keep["key_name"], keep["n"]):
            total[name] = total.get(name, 0) + float(k)
    return [k for k, _ in sorted(total.items(), key=lambda kv: -kv[1])][:n]


def flip_names(flips: Dict[str, pd.DataFrame], n: int = 6) -> List[str]:
    """The perspectives that flip on at least 1 arm, ranked by the rows."""
    total: Dict[str, float] = {}
    for f in flips.values():
        if f is None or f.empty:
            continue
        for name, k in zip(f.loc[f["flips"].astype(bool), "key"],
                           f.loc[f["flips"].astype(bool), "n"]):
            total[name] = total.get(name, 0) + float(k)
    return [k for k, _ in sorted(total.items(), key=lambda kv: -kv[1])][:n]


def latex_content_hierarchy(summaries: Dict[str, pd.DataFrame],
                            flips: Dict[str, pd.DataFrame],
                            n_names: int = 10, n_flips: int = 6,
                            caption: str = CONTENT_CAPTION,
                            label: str = "tab:ic-meta-content-hierarchy") -> str:
    """The content hierarchy (`valence_summary`) and the perspective flips
    (`perspective_flips`), 2 columns per arm."""
    from research.notebooks.cvpr import _latex as L
    arms = list(summaries)
    width = 1 + 2 * len(arms)
    head = _arm_heads(arms, 2) + [L.row(["Name"] + ["+/--", "range"] * len(arms))]
    body = [L.row([f"\\multicolumn{{{width}}}{{l}}{{\\emph{{Dimension valence}}}}"])]
    for name in hierarchy_names(summaries, n_names):
        cells = [f"\\quad {L.escape(name)}"]
        for a in arms:
            s = summaries[a]
            hit = s[s["key_name"].eq(name)] if s is not None and not s.empty else s
            if hit is None or hit.empty:
                cells += [L.DASH, L.DASH]
                continue
            r = hit.iloc[0]
            mark = "$^\\dagger$" if bool(r["flips"]) else ""
            cells += [f"{int(r['n_good_cases'])}/{int(r['n_bad_cases'])}{mark}",
                      L.fmt_ci(r["net_min"], r["net_max"])]
        body.append(L.row(cells))
    names = flip_names(flips, n_flips)
    if names:
        body += ["\\midrule",
                 L.row([f"\\multicolumn{{{width}}}{{l}}{{\\emph{{Perspective lift}}}}"])]
        for name in names:
            cells = [f"\\quad {L.escape(name)}"]
            for a in arms:
                f = flips[a]
                hit = f[f["key"].eq(name)] if f is not None and not f.empty else f
                if hit is None or hit.empty:
                    cells += [L.DASH, L.DASH]
                    continue
                r = hit.iloc[0]
                mark = "$^\\dagger$" if bool(r["flips"]) else ""
                cells += [f"{_count_cases(r['pays_cases'])}/{_count_cases(r['costs_cases'])}{mark}",
                          L.fmt_ci(r["lift_min"], r["lift_max"])]
            body.append(L.row(cells))
    return L.table(_MODULE, __version__, caption, label, "l" + "rr" * len(arms),
                   head, body, resize=True)


PROXY_CAPTION = (
    "\\textbf{Which trace features mark agreement with the proxy?} The agreement "
    "of the traces with the feature less the agreement of the traces without it, "
    "inside (cell, gap band) strata: the Mantel-Haenszel risk difference with "
    "its normal 95\\% interval. Bold marks an interval that clears 0. The area "
    "proxies read the community district of the unit; crime density is negated."
)

PROXY_FEATURES = ("wealth_inference", "risk_inference", "good_margin_hi",
                  "premise_dismissal", "costly", "favors_loser", "tc_for_winner")


def _effect(r) -> str:
    from research.notebooks.cvpr import _latex as L
    if r is None or pd.isna(r["diff"]):
        return L.DASH
    text = f"{L.fmt(r['diff'], signed=True)} {L.fmt_ci(r['lo'], r['hi'])}"
    clears = (r["lo"] > 0) or (r["hi"] < 0)
    return f"\\textbf{{{text}}}" if clears else text


def latex_proxy_contrasts(contrasts: Dict[str, pd.DataFrame],
                          features: Sequence[str] = PROXY_FEATURES,
                          caption: str = PROXY_CAPTION,
                          label: str = "tab:ic-meta-proxy-contrasts") -> str:
    """The band-controlled contrasts (`contrast_table(..., by="family")`).

    1 block of rows per arm, 1 column per proxy family.
    """
    from research.notebooks.cvpr import _latex as L
    fams = list(FAMILY_TEX)
    width = 1 + len(fams)
    head = [L.row(["Feature"] + [FAMILY_TEX[f] for f in fams])]
    body = []
    for i, a in enumerate(contrasts):
        c = contrasts[a]
        if i:
            body.append("\\midrule")
        body.append(L.row([f"\\multicolumn{{{width}}}{{l}}{{\\textbf{{{L.arm_label(a)}}}}}"]))
        for feat in features:
            cells = [f"\\quad {FEATURE_TEX.get(feat, L.escape(feat))}"]
            for fam in fams:
                hit = (c[c["group"].eq(fam) & c["feature"].eq(feat)]
                       if c is not None and not c.empty else None)
                cells.append(_effect(hit.iloc[0] if hit is not None and not hit.empty else None))
            body.append(L.row(cells))
    return L.table(_MODULE, __version__, caption, label, "l" + "r" * len(fams),
                   head, body, resize=True)


MARKERS_CAPTION = (
    "\\textbf{Which names mark agreement with the proxy on both extractors?} The "
    "names whose band-controlled contrast (as in the proxy table) clears 0 with "
    "the same sign under the 2 extractors of 1 judge. \\emph{primary} and "
    "\\emph{control} give the difference under each extractor. A dimension "
    "carries its valence."
)


def latex_name_markers(names_both: Dict[str, pd.DataFrame], n: int = 10,
                       caption: str = MARKERS_CAPTION,
                       label: str = "tab:ic-meta-name-markers") -> str:
    """The names that clear 0 on both extractors (the `names_both` frame of
    the notebook replication), at most `n` per arm by the primary |diff|."""
    from research.notebooks.cvpr import _latex as L
    head = [L.row(["Name", "class", "proxy", "primary", "control"])]
    body = []
    for i, a in enumerate(names_both):
        t = names_both[a]
        if i:
            body.append("\\midrule")
        body.append(L.row([f"\\multicolumn{{5}}{{l}}{{\\textbf{{{L.model_label(judge_of(a))} traces}}}}"]))
        if t is None or t.empty:
            body.append(L.row([L.DASH] * 5))
            continue
        t = t.assign(_abs=t["diff"].abs()).sort_values("_abs", ascending=False).head(n)
        t = t.sort_values(["family", "diff"], ascending=[True, False])
        for _, r in t.iterrows():
            name = str(r["feature"]).replace(" | ", ", ")
            body.append(L.row([f"\\quad {L.escape(name)}", L.escape(r["extraction_class"]),
                               FAMILY_TEX.get(r["family"], L.escape(r["family"])),
                               L.fmt(r["diff"], signed=True), L.fmt(r["diff_sib"], signed=True)]))
    return L.table(_MODULE, __version__, caption, label, "lllrr", head, body,
                   resize=True)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export(tables: Dict[str, pd.DataFrame], out_dir: Path, prefix: str) -> List[str]:
    """Write each table as `<prefix>_<name>.csv` under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, t in tables.items():
        if t is None or (isinstance(t, pd.DataFrame) and t.empty):
            continue
        if isinstance(t, dict):
            t = pd.DataFrame([t])
        path = out_dir / f"{prefix}_{name}.csv"
        t.to_csv(path, index=False)
        written.append(path.name)
    return written


def write_latex(tables: Dict[str, str], out_dir: Path, tag: str,
                version: str) -> List[str]:
    """Write each table as `<tag>_<name>_v<version>.tex` under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, tex in tables.items():
        if not tex:
            continue
        path = out_dir / f"{tag}_{name}_v{version}.tex"
        path.write_text(tex)
        written.append(path.name)
    return written
