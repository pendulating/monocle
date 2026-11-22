"""Loaders and figures for the lens-refusal probe notebooks.

The probe asks gemma-4-12b one question about one street image under a
label pair, and it gives the model an abstain label. The plan is
`research/docs/lens-refusal-probe-plan.md` and the results are in
`research/docs/lens-refusal-probe-results.md`.

The probe has 2 rounds. `set_round()` points every path constant at 1 of
them, and every loader reads the constants at call time.

| Round | Root | Draw | E3 statistic | Probes |
|-------|------|------|--------------|--------|
| `round1` | `outputs/_monocle/probe/` | tract deciles | a Spearman (`divergence/`) | 5 |
| `round2` | `outputs/_monocle/probe_nta/` | the top-10 and bottom-10 NTAs | an AUC (`group_test/`) | 10 |

| Folder | What it holds |
|--------|---------------|
| `sample/images.parquet` | 1 row for each image: the path, the area key, the draw flags, the proxies |
| `scores/<probe>/` | the census: `c`, the label logits, the top 5 tokens, the top class |
| `vectors/<probe>/reads.parquet` | the census rows plus the lens reads at blocks 41 to 45 |
| `divergence/` | the round-1 E3 split statistic, for the own draw and for the union |
| `group_test/` | the round-2 E3 statistic: an AUC of the top group against the bottom group |
| `steer/<probe>/<run>/` | the E4 dose grid |
| `scrape/<probe>/` | the E1b and E5c word scrapes |
| `loading/` | the E5 cosines and the per-image correlations |
| `ranks/` | the label ranks along the blocks |

Every loader gives `None` when its file is not on disk. The notebook then
prints a message. The probe steps do not all finish at the same time.

Read the reads with the house style of the paper:

    .venv-mllmsci-vllm025cu129/bin/python -c "import _probe"
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent

# `monocle.probe_frame` holds the label pairs and the proxy of each probe.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# `_style` holds the palette of the paper. It sits with the CVPR notebooks.
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"
if str(_CVPR) not in sys.path:
    sys.path.insert(0, str(_CVPR))

import _style as style  # noqa: E402
from research.monocle.probe_frame import ABSTAIN, PROBE_ORDER, PROBES  # noqa: E402

__version__ = "1.1.0"

#: The rounds of the probe. The value is the folder under `outputs/_monocle/`.
ROUNDS: Dict[str, str] = {"round1": "probe", "round2": "probe_nta"}

#: What each round holds. `e3_kind` names the E3 folder and statistic.
#: `draw_kind` names the column that says where an image sits in the draw.
ROUND_KINDS: Dict[str, Dict[str, str]] = {
    "round1": {"e3_kind": "divergence", "draw_kind": "decile"},
    "round2": {"e3_kind": "group_test", "draw_kind": "group"},
}

DEFAULT_ROUND = "round1"

#: The round the constants below point at. `set_round()` changes it.
_ACTIVE_ROUND = DEFAULT_ROUND

PROBE_ROOT = REPO_ROOT / "outputs" / "_monocle" / ROUNDS[DEFAULT_ROUND]
SAMPLE_PATH = PROBE_ROOT / "sample" / "images.parquet"
SCORES_DIR = PROBE_ROOT / "scores"
VECTORS_DIR = PROBE_ROOT / "vectors"
DIVERGENCE_DIR = PROBE_ROOT / "divergence"
GROUP_TEST_DIR = PROBE_ROOT / "group_test"
STEER_DIR = PROBE_ROOT / "steer"
SCRAPE_DIR = PROBE_ROOT / "scrape"
LOADING_DIR = PROBE_ROOT / "loading"
RANKS_DIR = PROBE_ROOT / "ranks"


def set_round(name: str = DEFAULT_ROUND) -> str:
    """Point every path constant at 1 round. Give the round name back.

    A loader reads the constants at call time, thus a cell that calls this
    function before the loaders reads the new round.
    """
    if name not in ROUNDS:
        raise KeyError(f"unknown round {name!r}; the rounds are {list(ROUNDS)}")
    root = REPO_ROOT / "outputs" / "_monocle" / ROUNDS[name]
    g = globals()
    g["_ACTIVE_ROUND"] = name
    g["PROBE_ROOT"] = root
    g["SAMPLE_PATH"] = root / "sample" / "images.parquet"
    g["SCORES_DIR"] = root / "scores"
    g["VECTORS_DIR"] = root / "vectors"
    g["DIVERGENCE_DIR"] = root / "divergence"
    g["GROUP_TEST_DIR"] = root / "group_test"
    g["STEER_DIR"] = root / "steer"
    g["SCRAPE_DIR"] = root / "scrape"
    g["LOADING_DIR"] = root / "loading"
    g["RANKS_DIR"] = root / "ranks"
    return name


def active_round() -> str:
    """The name of the round the path constants point at."""
    return _ACTIVE_ROUND


def round_meta() -> Dict[str, str]:
    """The round name, its root, its E3 kind, and its draw column kind.

    | Key | Value |
    |-----|-------|
    | `round` | `round1` or `round2` |
    | `folder` | the folder under `outputs/_monocle/` |
    | `root` | the full path of the root |
    | `e3_kind` | `divergence` (round 1) or `group_test` (round 2) |
    | `draw_kind` | `decile` (round 1) or `group` (round 2) |
    """
    kinds = ROUND_KINDS.get(_ACTIVE_ROUND, ROUND_KINDS[DEFAULT_ROUND])
    return {"round": _ACTIVE_ROUND,
            "folder": ROUNDS.get(_ACTIVE_ROUND, ROUNDS[DEFAULT_ROUND]),
            "root": str(PROBE_ROOT), **kinds}


def probes_with_reads() -> List[str]:
    """The probes of `PROBE_ORDER` that hold a reads parquet in this round."""
    return [p for p in PROBE_ORDER
            if (VECTORS_DIR / p / "reads.parquet").exists()]


#: The read the plan asks for. The notebook falls back to it when the
#: E3 summary of the round is not on disk.
DEFAULT_READ = "read_C_L42"

#: The 4 values of `top_class`. `pos` and `neg` are the 2 poles, `abstain`
#: is the `unclear` label, and `other` is a refusal in free text.
CLASS_ORDER: Tuple[str, ...] = ("pos", "neg", "abstain", "other")

CLASS_COLOR: Dict[str, str] = {
    "pos": style.PAL["teal"],
    "neg": style.PAL["coral"],
    "abstain": style.PAL["warmgrey"],
    "other": style.PAL["blue"],
}

ARM_COLOR: Dict[str, str] = {
    "signal": style.PAL["teal"],
    "control": style.PAL["coral"],
}


# ---------------------------------------------------------------- the tables

def load_sample() -> Optional[pd.DataFrame]:
    """1 row for each drawn image, with the tract key and the tract proxies."""
    if not SAMPLE_PATH.exists():
        return None
    return pd.read_parquet(SAMPLE_PATH)


def load_scores(probe: str) -> Optional[pd.DataFrame]:
    """The census of 1 probe. The shards concatenate."""
    shards = sorted((SCORES_DIR / probe).glob("scores_shard*.parquet"))
    if not shards:
        return None
    return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)


def load_reads(probe: str) -> Optional[pd.DataFrame]:
    """The census rows of 1 probe plus the lens reads."""
    path = VECTORS_DIR / probe / "reads.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_vector_summary(probe: str) -> Optional[dict]:
    """The E2 summary: the agreement of each read with `c`."""
    path = VECTORS_DIR / probe / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_columns(df: pd.DataFrame) -> List[str]:
    """The lens read columns of a reads frame, in block order."""
    cols = [c for c in df.columns if c.startswith("read_")]
    return sorted(cols, key=lambda c: (c.split("_")[1], c.split("_L")[-1]))


def census_table() -> pd.DataFrame:
    """1 row for each probe: the share of each top class and the gap `c`.

    A missing probe gives no row.
    """
    rows = []
    for name in PROBE_ORDER:
        sc = load_scores(name)
        if sc is None:
            continue
        p = PROBES[name]
        row = {"probe": name, "pos": p.pos, "neg": p.neg,
               "abstain": ABSTAIN, "n": len(sc)}
        for k in CLASS_ORDER:
            row[f"share_{k}"] = float((sc["top_class"] == k).mean())
        row["mean_c"] = float(sc["c"].mean())
        row["sd_c"] = float(sc["c"].std())
        row["answered"] = int(sc["answered"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def proxy_for(probe: str, sample: Optional[pd.DataFrame] = None
              ) -> Tuple[str, int, str]:
    """(column, sign, label) of the world proxy of 1 probe.

    The rule copies `research/scripts/probe_divergence.py`: the probe takes its own
    proxy when the column holds 100 values or more, and the income fallback
    if it does not. The sign of the fallback is the sign of the class probe.
    """
    p = PROBES[probe]
    if sample is None:
        sample = load_sample()
    col = p.proxy
    if sample is None or col not in sample.columns or \
            int(sample[col].notna().sum()) < 100:
        col = p.fallback_proxy
    sign = p.proxy_sign if col == p.proxy else PROBES["class"].proxy_sign
    label = p.proxy_label if col == p.proxy else f"{col} (stand-in)"
    return col, int(sign), label


#: The sample columns a notebook wants beside the reads. A column that the
#: round does not hold drops out. Round 2 adds the NTA keys and 4 proxies.
SAMPLE_KEEP = ["sample_id", "geoid", "boroct2020", "nta2020", "ntaname",
               "boroname", "face", "recording_id",
               "median_income", "crime_density", "crime_per_capita",
               "homeless_density", "black_share", "white_share",
               "foreign_born_share", "bachelors_share", "median_home_value",
               "prostitution_density", "violent_per_capita",
               "population", "area_km2", "pop_density"]


def reads_with_sample(probe: str) -> Optional[pd.DataFrame]:
    """The reads of 1 probe joined to the sample, with the oriented proxy.

    The frame gets 4 more columns:

    | Column | Meaning |
    |--------|---------|
    | `in_draw` | the image belongs to the probe's own draw |
    | `decile` | the proxy decile of the round-1 draw, -1 when the round has none |
    | `group` | `top` or `bottom` in the round-2 draw, empty when the round has none |
    | `proxy_oriented` | the proxy, with a high value on the pos pole |
    """
    reads = load_reads(probe)
    if reads is None:
        return None
    sample = load_sample()
    if sample is None:
        return reads
    keep = ["image_path"] + [c for c in SAMPLE_KEEP if c in sample.columns]
    for extra in (f"draw_{probe}", f"decile_{probe}", f"group_{probe}"):
        if extra in sample.columns:
            keep.append(extra)
    df = reads.merge(sample[keep], on="image_path", how="left")
    df["in_draw"] = df.get(f"draw_{probe}", pd.Series(False, index=df.index))
    df["in_draw"] = df["in_draw"].fillna(False).astype(bool)
    df["decile"] = df.get(f"decile_{probe}", pd.Series(-1, index=df.index))
    df["group"] = df.get(f"group_{probe}", pd.Series("", index=df.index))
    df["group"] = df["group"].fillna("").astype(str)
    col, sign, _ = proxy_for(probe, sample)
    df["proxy_column"] = col
    df["proxy_oriented"] = sign * df[col].astype(float)
    return df


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho of 2 vectors, on the rows where both are finite."""
    from scipy.stats import spearmanr

    a = pd.Series(np.asarray(x, dtype=float))
    b = pd.Series(np.asarray(y, dtype=float))
    ok = a.notna() & b.notna()
    if int(ok.sum()) < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


# ------------------------------------------------------------ the divergence

DIV_COLS = ["probe", "proxy", "primary_read", "agree_with_c_answered",
            "n_answered", "n_refused", "rho_answered", "rho_refused",
            "ci_refused", "ratio", "verdict", "refusal_share"]


def divergence_summary(subset: str = "union") -> Optional[pd.DataFrame]:
    """The E3 verdict of every probe, for `own` or for `union`."""
    path = DIVERGENCE_DIR / f"summary_{subset}.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    if "ci_refused" in df.columns:
        df["ci_lo"] = df["ci_refused"].map(lambda v: v[0])
        df["ci_hi"] = df["ci_refused"].map(lambda v: v[1])
    if "c" in df.columns:
        df["c_answered"] = df["c"].map(lambda v: v.get("answered"))
        df["c_refused"] = df["c"].map(lambda v: v.get("refused"))
    cols = [c for c in DIV_COLS if c in df.columns and c != "ci_refused"]
    cols += [c for c in ("ci_lo", "ci_hi", "c_answered", "c_refused")
             if c in df.columns]
    return df[cols]


def divergence_rows(probe: str, subset: str = "union") -> Optional[pd.DataFrame]:
    """Every read of 1 probe, on both splits, with the bootstrap interval."""
    path = DIVERGENCE_DIR / f"{probe}_{subset}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------- the group test

#: The column order of `group_test_summary()`.
GT_COLS = ["probe", "proxy", "sign", "pos_group", "primary_read", "verdict",
           "n_answered", "n_refused", "auc_answered", "auc_refused",
           "ci_lo", "ci_hi", "auc_refused_matched", "ci_lo_m", "ci_hi_m",
           "log_density_gap", "d_refused", "p_refused", "refusal_share",
           "agree_with_c", "c_answered", "c_refused"]


def group_test_summary() -> Optional[pd.DataFrame]:
    """The round-2 E3 verdict of every probe, 1 row for each probe.

    The statistic is an AUC: P(a read of the pos group > a read of the neg
    group). 0.5 means no separation. `auc_refused_matched` holds the value
    inside NTA pairs of equal density. The interval is an NTA cluster
    bootstrap.
    """
    path = GROUP_TEST_DIR / "summary.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    flat = []
    for r in rows:
        n = r.get("n") or {}
        ci = r.get("ci_refused") or [None, None]
        cim = r.get("ci_refused_matched") or [None, None]
        cvals = r.get("c") or {}
        row = {k: r.get(k) for k in
               ("probe", "proxy", "sign", "pos_group", "primary_read",
                "verdict", "auc_answered", "auc_refused",
                "auc_refused_matched", "log_density_gap", "d_refused",
                "p_refused", "refusal_share", "agree_with_c")}
        row["n_answered"] = int(sum(n.get("answered") or []))
        row["n_refused"] = int(sum(n.get("refused") or []))
        row["ci_lo"], row["ci_hi"] = ci[0], ci[1]
        row["ci_lo_m"], row["ci_hi_m"] = cim[0], cim[1]
        row["c_answered"] = cvals.get("answered")
        row["c_refused"] = cvals.get("refused")
        flat.append(row)
    df = pd.DataFrame(flat)
    head = [k for k in GT_COLS if k in df.columns]
    return df[head + [k for k in df.columns if k not in head]]


def group_test_rows(probe: str) -> Optional[pd.DataFrame]:
    """Every read of 1 probe on the 4 splits, with the bootstrap interval."""
    path = GROUP_TEST_DIR / f"{probe}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def group_test_md() -> Optional[str]:
    """The round-2 E3 table as the writer wrote it."""
    path = GROUP_TEST_DIR / "group_test.md"
    if not path.exists():
        return None
    return path.read_text()


def census_by_group() -> Optional[pd.DataFrame]:
    """The round-2 census of the own draw, 1 row for each probe.

    The columns split the answer shares by the top group and the bottom
    group of the proxy.
    """
    path = SCORES_DIR / "census_own_draw.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def primary_read(probe: str, subset: str = "union") -> str:
    """The read of record of 1 probe. `DEFAULT_READ` when E3 is not on disk.

    Round 1 takes the read from `divergence/`, round 2 from `group_test/`.
    """
    summ = divergence_summary(subset)
    if summ is not None and "primary_read" in summ.columns:
        hit = summ[summ["probe"] == probe]
        if not hit.empty:
            return str(hit["primary_read"].iloc[0])
    gt = group_test_summary()
    if gt is not None and "primary_read" in gt.columns:
        hit = gt[gt["probe"] == probe]
        if not hit.empty and isinstance(hit["primary_read"].iloc[0], str):
            return str(hit["primary_read"].iloc[0])
    return DEFAULT_READ


# ----------------------------------------------------------------- the ranks

def ranks_trajectory() -> Optional[pd.DataFrame]:
    """The median label rank of each probe along the blocks, both splits."""
    path = RANKS_DIR / "trajectory.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def ranks_md() -> Optional[str]:
    """The rank trajectory table as the writer wrote it."""
    path = RANKS_DIR / "trajectory.md"
    if not path.exists():
        return None
    return path.read_text()


# --------------------------------------------------------------- the loading

def loading_cosines() -> Optional[pd.DataFrame]:
    """The E5 cosine of each probe vector with each case vector, as a matrix."""
    path = LOADING_DIR / "cosines.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.pivot(index="probe", columns="case", values="cos")


def loading_per_image() -> Optional[pd.DataFrame]:
    """The E5 per-image correlation of a probe read with a case read."""
    path = LOADING_DIR / "per_image.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def loading_md() -> Optional[str]:
    """The E5 report as the writer wrote it."""
    path = LOADING_DIR / "loading.md"
    if not path.exists():
        return None
    return path.read_text()


# ----------------------------------------------------------------- the steer

def steer_runs() -> pd.DataFrame:
    """1 row for each `steer/<probe>/<run>/` folder on disk.

    The E4 step writes the folder before the parquet, so a row with
    `has_parquet` False means the run is still on the queue.
    """
    rows = []
    if STEER_DIR.is_dir():
        for probe_dir in sorted(STEER_DIR.iterdir()):
            if not probe_dir.is_dir():
                continue
            for run_dir in sorted(probe_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                rows.append({
                    "probe": probe_dir.name,
                    "run": run_dir.name,
                    "has_parquet": (run_dir / "steer.parquet").exists(),
                    "has_summary": (run_dir / "summary.json").exists(),
                    "path": str(run_dir),
                })
    return pd.DataFrame(rows, columns=["probe", "run", "has_parquet",
                                       "has_summary", "path"])


def load_steer(run_path: str | Path) -> Optional[pd.DataFrame]:
    """The dose grid of 1 steering run."""
    path = Path(run_path) / "steer.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_steer_summary(run_path: str | Path) -> Optional[dict]:
    """The summary of 1 steering run, with the `go_doses` list."""
    path = Path(run_path) / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def dose_table(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each (arm, alpha): the share of each answer class."""
    rows = []
    for (arm, alpha), g in df.groupby(["arm", "alpha"]):
        row = {"arm": str(arm), "alpha": float(alpha), "n": int(len(g))}
        if "answered" in g.columns:
            row["share_answered"] = float(g["answered"].mean())
        for k in CLASS_ORDER:
            row[f"share_{k}"] = float((g["top_class"] == k).mean())
        if "c" in g.columns:
            row["mean_c"] = float(g["c"].mean())
        if "logit_abstain" in g.columns:
            row["mean_logit_abstain"] = float(g["logit_abstain"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["arm", "alpha"]).reset_index(drop=True)


def plot_dose(tab: pd.DataFrame):
    """2 panels: the answered share, and the 3 class shares, against alpha."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    for arm, g in tab.groupby("arm"):
        ax.plot(g["alpha"], g.get("share_answered", g["share_pos"] + g["share_neg"]),
                marker="o", ms=3, lw=1.2,
                color=ARM_COLOR.get(str(arm), style.PAL["blue"]), label=str(arm))
    ax.set_xlabel("alpha")
    ax.set_ylabel("share that answers a pole")
    ax.legend()

    ax = axes[1]
    dash = {"signal": "-", "control": "--"}
    for arm, g in tab.groupby("arm"):
        for k in ("pos", "neg", "abstain"):
            ax.plot(g["alpha"], g[f"share_{k}"], dash.get(str(arm), ":"),
                    marker="o", ms=2.5, lw=1.0, color=CLASS_COLOR[k],
                    label=f"{arm} {k}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("share of the images")
    ax.legend(ncol=2, fontsize=5.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- the scrape

def scrape_dirs(probe: str) -> List[Path]:
    """Every scrape folder of 1 probe.

    The exact folder comes first. A run under a name such as
    `class_v1_nomask` follows it.
    """
    out: List[Path] = []
    if not SCRAPE_DIR.is_dir():
        return out
    exact = SCRAPE_DIR / probe
    if exact.is_dir():
        out.append(exact)
    for d in sorted(SCRAPE_DIR.glob(f"{probe}_*")):
        # `housing_type` is its own probe, thus it is not a run of `housing`.
        if d.is_dir() and d.name not in PROBES:
            out.append(d)
    return out


def load_tokens(scrape_path: str | Path) -> Optional[pd.DataFrame]:
    """The top tokens of every scraped position."""
    path = Path(scrape_path) / "tokens.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_label_ranks(scrape_path: str | Path) -> Optional[pd.DataFrame]:
    """The rank of the 3 label tokens at every scraped position."""
    path = Path(scrape_path) / "label_ranks.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_scrape_summary(scrape_path: str | Path) -> Optional[dict]:
    """The scrape summary. `scrape_summary.json` wins over `summary.json`."""
    for name in ("scrape_summary.json", "summary.json"):
        path = Path(scrape_path) / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def layers_of(tokens: pd.DataFrame, group: str, lens: str) -> List[int]:
    """The blocks the scrape holds for 1 group and 1 lens."""
    d = tokens[(tokens["group"] == group) & (tokens["lens"] == lens)]
    return sorted(int(v) for v in d["layer"].unique())


def nearest_layer(layers: Sequence[int], want: int) -> Optional[int]:
    """The block on disk that sits nearest to the block the plan asks for."""
    if not len(layers):
        return None
    return int(min(layers, key=lambda v: (abs(v - want), v)))


def normalize_token(tok: str) -> str:
    """The counted form of a token: no outer space, lower case."""
    return str(tok).strip().lower()


def is_special(tok: str) -> bool:
    """True for an empty token and for a control token such as `<pad>`."""
    t = normalize_token(tok)
    return (not t) or (t.startswith("<") and t.endswith(">"))


def word_counts(tokens: pd.DataFrame, group: str, lens: str, layer: int,
                split: str, top_k: int = 5, hide_special: bool = True,
                limit: int = 25) -> pd.DataFrame:
    """The most frequent words of 1 group, lens, block, and split.

    `count` counts the positions that put the word in the top `top_k`.
    `images` counts the images that hold it at least 1 time.
    """
    d = tokens[(tokens["group"] == group) & (tokens["lens"] == lens)
               & (tokens["layer"] == layer) & (tokens["split"] == split)
               & (tokens["rank"] < top_k)].copy()
    if d.empty:
        return pd.DataFrame(columns=["token", "count", "images"])
    d["word"] = d["token"].map(normalize_token)
    if hide_special:
        d = d[~d["token"].map(is_special)]
    if d.empty:
        return pd.DataFrame(columns=["token", "count", "images"])
    tab = d.groupby("word").agg(count=("word", "size"),
                                images=("image_path", "nunique"))
    tab = tab.sort_values("count", ascending=False).head(limit).reset_index()
    return tab.rename(columns={"word": "token"})


def top_words_set(tokens: pd.DataFrame, group: str, lens: str, layer: int,
                  split: str, top_k: int = 5) -> set:
    """The word set of 1 group, lens, block, and split, above rank `top_k`."""
    d = tokens[(tokens["group"] == group) & (tokens["lens"] == lens)
               & (tokens["layer"] == layer) & (tokens["split"] == split)
               & (tokens["rank"] < top_k)]
    return {normalize_token(t) for t in d["token"]}


def question_words(tokens: pd.DataFrame, layer_c: int, split: str,
                   top_k: int = 5, hide_special: bool = True,
                   limit: int = 25) -> Tuple[pd.DataFrame, Optional[int]]:
    """The question-token words under lens C, with the lens B mark.

    The mark says that lens B also puts the word in its top `top_k` at the
    B block. Lens B does not hold every block, thus the function gives the
    B block it used. The caller prints it.
    """
    tab = word_counts(tokens, "text_after", "C", layer_c, split,
                      top_k=top_k, hide_special=hide_special, limit=limit)
    b_layers = layers_of(tokens, "text_after", "B")
    b_layer = layer_c if layer_c in b_layers else nearest_layer(b_layers, layer_c)
    if b_layer is None or tab.empty:
        tab["in_B_top5"] = ""
        return tab, b_layer
    b_words = top_words_set(tokens, "text_after", "B", b_layer, split, top_k)
    # The column names the B block, because B does not hold every block and a
    # mark from another block is a weaker claim.
    tab[f"in_B_top{top_k}_L{int(b_layer)}"] = [
        "yes" if w in b_words else "" for w in tab["token"]]
    return tab, int(b_layer)


def label_rank_table(label_ranks: pd.DataFrame, probe: str,
                     group: str = "answer", lens: str = "C") -> pd.DataFrame:
    """The median rank of the 3 label tokens, by block and split."""
    d = label_ranks[(label_ranks["group"] == group)
                    & (label_ranks["lens"] == lens)]
    if d.empty:
        return pd.DataFrame()
    cols = ["rank_pos", "rank_neg", "rank_abstain"]
    tab = d.groupby(["split", "layer"])[cols].median().reset_index()
    p = PROBES[probe]
    names = {"rank_pos": f"rank {p.pos}", "rank_neg": f"rank {p.neg}",
             "rank_abstain": f"rank {ABSTAIN}"}
    return tab.rename(columns=names).sort_values(["split", "layer"])


def plot_label_ranks(tab: pd.DataFrame, probe: str):
    """The median label rank against the block, 1 panel for each split."""
    import matplotlib.pyplot as plt

    p = PROBES[probe]
    cols = [(f"rank {p.pos}", CLASS_COLOR["pos"]),
            (f"rank {p.neg}", CLASS_COLOR["neg"]),
            (f"rank {ABSTAIN}", CLASS_COLOR["abstain"])]
    splits = sorted(tab["split"].unique())
    fig, axes = plt.subplots(1, max(1, len(splits)),
                             figsize=(3.3 * max(1, len(splits)), 2.5),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sp in zip(axes, splits):
        g = tab[tab["split"] == sp].sort_values("layer")
        for name, color in cols:
            if name not in g.columns:
                continue
            ax.plot(g["layer"], g[name].clip(lower=0.5), marker="o", ms=3,
                    lw=1.2, color=color, label=name)
        ax.set_yscale("log")
        ax.set_xlabel(f"block ({sp})")
    axes[0].set_ylabel("median rank of the label token")
    axes[0].legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- the images

def thumbnail(path: str | Path, width: int = 256):
    """The image at `path`, resized to `width` px. `None` when it is missing."""
    from PIL import Image

    try:
        im = Image.open(str(path))
        im.load()
    except Exception:
        return None
    im = im.convert("RGB")
    if im.width > width:
        height = max(1, int(round(im.height * width / im.width)))
        im = im.resize((width, height), Image.BILINEAR)
    return im


def rank_rows(df: pd.DataFrame, read_col: str, n: int = 12, pole: str = "neg",
              refused_only: bool = False) -> pd.DataFrame:
    """The `n` images with the highest read toward 1 pole.

    A high read stands for the pos pole, thus the neg pole takes the lowest
    values. `refused_only` keeps the images where the model does not answer
    a pole.
    """
    d = df.dropna(subset=[read_col])
    if refused_only and "answered" in d.columns:
        d = d[~d["answered"].astype(bool)]
    d = d.sort_values(read_col, ascending=(pole == "neg"))
    return d.head(int(n)).copy()


def caption_area(row) -> str:
    """The area line of a thumbnail: the NTA and the group, or the tract.

    Round 2 holds `ntaname` and `group`, thus the caption names the
    neighborhood and the pole of the draw. Round 1 holds the tract alone.
    """
    name = str(row.get("ntaname", "") or "")
    group = str(row.get("group", "") or "")
    if not name:
        return str(row.get("geoid", "") or "")
    if len(name) > 26:
        name = name[:25] + "\u2026"
    return f"{name} ({group})" if group else name


def contact_sheet(rows: pd.DataFrame, read_col: str, ncols: int = 4,
                  width: int = 256):
    """A grid of thumbnails, with the read, `c`, the top class, and the area.

    Warning: this sheet shows people and blocks. Keep it inside the team.
    """
    import matplotlib.pyplot as plt

    n = int(len(rows))
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no image", ha="center", va="center")
        return fig
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.75 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_visible(False)
    for ax, (_, r) in zip(axes, rows.iterrows()):
        ax.set_visible(True)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        im = thumbnail(r["image_path"], width=width)
        if im is None:
            ax.text(0.5, 0.5, "no file", ha="center", va="center", fontsize=6)
        else:
            ax.imshow(np.asarray(im))
        label = (f"{read_col.replace('read_', '')} {float(r[read_col]):+.2f}   "
                 f"c {float(r['c']):+.1f}\n{r.get('top_class', '')}   "
                 f"{str(r.get('geoid', ''))}\n{caption_area(r)}")
        ax.set_xlabel(label, fontsize=5.5, labelpad=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- the figures

def hist_by_class(df: pd.DataFrame, col: str, bins: int = 40):
    """The distribution of 1 column, 1 filled step for each top class."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    drawn = 0
    for k in CLASS_ORDER:
        sub = df.loc[df["top_class"] == k, col].dropna()
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=bins, alpha=0.55, color=CLASS_COLOR[k],
                edgecolor="none", label=f"{k} (n={len(sub)})")
        drawn += 1
    ax.set_xlabel(col)
    ax.set_ylabel("images")
    if drawn:
        ax.legend()
    fig.tight_layout()
    return fig


def scatter_read_proxy(df: pd.DataFrame, read_col: str, probe: str):
    """The read against the oriented proxy, 1 colour for each top class."""
    import matplotlib.pyplot as plt

    col, sign, label = proxy_for(probe)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for k in CLASS_ORDER:
        sub = df[df["top_class"] == k]
        if sub.empty:
            continue
        ax.scatter(sub[read_col], sub["proxy_oriented"], s=5, alpha=0.5,
                   color=CLASS_COLOR[k], edgecolor="none", label=k)
    ax.set_xlabel(read_col)
    turn = "" if sign > 0 else " x -1"
    ax.set_ylabel(f"{col}{turn}\n(a high value goes with {PROBES[probe].pos})")
    ax.legend()
    fig.tight_layout()
    return fig


def strip_read_by_group(df: pd.DataFrame, read_col: str, probe: str,
                        seed: int = 0):
    """The read of each drawn image, 1 column for each group of round 2.

    The draw holds the top-10 and the bottom-10 NTAs of the proxy. The
    colour is the top class and the black bar is the mean. The sign of the
    probe says which group carries the pos pole.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    if "group" not in df.columns:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "the round holds no group column",
                ha="center", va="center")
        return fig
    d = df[df["group"].astype(str).isin(("top", "bottom"))]
    if d.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no image of the own draw", ha="center", va="center")
        return fig
    rng = np.random.default_rng(seed)
    order = ["bottom", "top"]
    for x, g_name in enumerate(order):
        g = d[d["group"] == g_name]
        if g.empty:
            continue
        for k in CLASS_ORDER:
            sub_g = g[g["top_class"] == k]
            if sub_g.empty:
                continue
            jitter = rng.uniform(-0.22, 0.22, len(sub_g))
            ax.scatter(x + jitter, sub_g[read_col], s=5, alpha=0.5,
                       color=CLASS_COLOR[k], edgecolor="none",
                       label=k if x == 0 else None)
        mean = float(g[read_col].mean())
        ax.plot([x - 0.3, x + 0.3], [mean, mean], color="black", lw=1.6)
    ax.set_xticks(range(len(order)))
    col, sign, _ = proxy_for(probe)
    pole = PROBES[probe].pos if sign > 0 else PROBES[probe].neg
    ax.set_xticklabels([f"{g} of {col}" for g in order])
    ax.set_xlabel(f"the NTA group of the draw (the top group goes with "
                  f"{pole})", fontsize=6)
    ax.set_ylabel(read_col)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def split_spearman(df: pd.DataFrame, read_col: str) -> pd.DataFrame:
    """Spearman of the read and of `c` with the proxy, on 3 splits."""
    rows = []
    parts = {"all": df,
             "answered": df[df["answered"].astype(bool)],
             "refused": df[~df["answered"].astype(bool)]}
    for name, g in parts.items():
        rows.append({
            "split": name, "n": int(len(g)),
            "rho_read": spearman(g[read_col], g["proxy_oriented"]),
            "rho_c": spearman(g["c"], g["proxy_oriented"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"_probe {__version__}")
    for _round in ROUNDS:
        set_round(_round)
        print(f"\n=== {_round}: {PROBE_ROOT} ===")
        print(round_meta())
        print(census_table().to_string(index=False))
        print(steer_runs().to_string(index=False))
        for _p in probes_with_reads():
            print(_p, primary_read(_p), [d.name for d in scrape_dirs(_p)])
    set_round(DEFAULT_ROUND)
