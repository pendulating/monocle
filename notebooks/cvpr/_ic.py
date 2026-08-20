"""The Integrative Complexity report: load the corpus, code it, draw it.

`dagspaces/common/ic_codes.py` holds the derivation, because it is a property
of the schema and it belongs beside it. This module is the paper side: it finds
the merged corpus, tests it against the canonical registry, and builds the
tables and the figures.

The corpus comes from `scripts/merge_trace_extractions.py --schema ic`, which
writes `data/ic_ingredients/<case>_ic_ingredients.parquet`.

Warning: read the scale as 1 to 6. Code 7 needs an organizing principle above
the integrations, and the schema holds no ingredient for it. See
`vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import _provenance as P
import _style as S

REPO_ROOT = P.REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagspaces.common import ic_codes as IC  # noqa: E402

__version__ = "1.0.0"

DEFAULT_ROOT = REPO_ROOT / "data" / "ic_ingredients"

# The order the paper uses. A case that is absent drops out on its own.
CASE_ORDER = (
    "subway_safety", "libraries", "schools", "road_quality",
    "parks_plazas", "restaurants", "street_photography",
)

CASE_LABEL = {
    "subway_safety": "Subway safety",
    "libraries": "Libraries",
    "schools": "Schools",
    "road_quality": "Road quality",
    "parks_plazas": "Parks and plazas",
    "restaurants": "Restaurants",
    "street_photography": "Street photography",
}

# The rung names, for a legend. They are the codebook's own words.
CODE_LABEL = {
    1: "1 no differentiation",
    2: "2 transitional",
    3: "3 differentiation",
    4: "4 unjustified link",
    5: "5 integration",
    6: "6 integration + context",
}

LABEL_ORDER = ["MuchLess", "Less", "Same", "More", "MuchMore", "NotSure"]


# ---------------------------------------------------------------- the corpus


def find_files(root: Optional[str | Path] = None) -> List[Path]:
    """Find the merged corpus, 1 parquet for each case."""
    base = Path(root) if root else DEFAULT_ROOT
    if not base.is_absolute():
        base = REPO_ROOT / base
    return sorted(base.glob("*_ic_ingredients.parquet"))


def registry_mismatch(root: Optional[str | Path] = None) -> List[str]:
    """Name every corpus file that does NOT come from a registered trace run.

    The same test as `_extractions.registry_mismatch`, for the IC schema. An
    ingredient corpus is 1 step downstream of a thinking run, thus it goes
    stale the moment the battery runs again.
    """
    import _canonical as C

    reg = {(r.case, r.model_config): os.path.realpath(r.results_path)
           for r in C.runs(kind="trace")}
    problems: List[str] = []
    for f in find_files(root):
        cols = ["case", "judge_model", "sweep", "source_results_path"]
        try:
            rows = pd.read_parquet(f, columns=cols).drop_duplicates()
        except Exception as exc:
            problems.append(f"{f.name}: cannot read its provenance ({exc})")
            continue
        for r in rows.itertuples():
            want = reg.get((r.case, r.judge_model))
            if want is None:
                problems.append(f"{r.case}: no registered trace run for {r.judge_model}")
            elif os.path.realpath(str(r.source_results_path)) != want:
                problems.append(
                    f"{r.case}: the corpus comes from sweep {r.sweep} "
                    f"({Path(str(r.source_results_path)).name}), not from the "
                    f"registered run ({Path(want).name})")
    return problems


def load(cases: Optional[Sequence[str]] = None,
         root: Optional[str | Path] = None,
         require_canonical: bool = True) -> pd.DataFrame:
    """Read the ingredient corpus.

    Args:
        require_canonical: Stop when the corpus does not come from the
            registered trace runs. Keep this True for every paper figure.
    """
    files = find_files(root)
    if not files:
        raise FileNotFoundError(
            f"no IC corpus under {root or DEFAULT_ROOT}.\n"
            "Run the ic_extract sweep, then:\n"
            "  python scripts/merge_trace_extractions.py <sweep_dir> --schema ic")
    if require_canonical:
        stale = registry_mismatch(root)
        if stale:
            raise RuntimeError(
                "the IC corpus does not come from the canonical trace runs:\n  "
                + "\n  ".join(stale)
                + "\nRun the extraction again on the registered runs. Pass "
                  "require_canonical=False only to look at an old corpus.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if cases:
        df = df[df["case"].isin(set(cases))]
    return df


def codes(df: pd.DataFrame, thresholds: IC.Thresholds = IC.DEFAULT) -> pd.DataFrame:
    """Give every trace its components and its code."""
    return IC.code_table(df, thresholds)


def _ordered(values: Sequence[str]) -> List[str]:
    known = [c for c in CASE_ORDER if c in set(values)]
    return known + sorted(set(values) - set(known))


# ---------------------------------------------------------------- the tables


def case_table(code_rows: pd.DataFrame, drop_truncated: bool = True) -> pd.DataFrame:
    """1 row for each case: the mean code, the shares, and the components."""
    table = IC.summarize(code_rows, drop_truncated=drop_truncated)
    table["case_label"] = table["case"].map(CASE_LABEL).fillna(table["case"])
    order = _ordered(table["case"])
    table["order"] = table["case"].apply(order.index)
    return table.sort_values("order").drop(columns="order").reset_index(drop=True)


def label_table(code_rows: pd.DataFrame, drop_truncated: bool = True) -> pd.DataFrame:
    """The code against the judgment the model gave.

    A `Same` or a `NotSure` is the hard pair. If the code measures reasoning,
    those rows carry more of it than a `MuchMore`.
    """
    return IC.by_label(code_rows, drop_truncated=drop_truncated)


def quality_table(df: pd.DataFrame, code_rows: pd.DataFrame) -> pd.DataFrame:
    """What a reader must see before any number above: is the corpus sound?"""
    rows = []
    for case, part in df.groupby("case"):
        real = part[part["ingredient_type"].notna()]
        cr = code_rows[code_rows["case"] == case]
        rows.append({
            "case": CASE_LABEL.get(case, case),
            "traces": int(part["doc_id"].nunique()),
            "ingredients": int(len(real)),
            "per_trace": round(len(real) / max(1, part["doc_id"].nunique()), 1),
            "quote_found": round(float(real["quote_found"].mean()), 4) if len(real) else 0.0,
            "sub_quotes_found": (
                round(float(real["n_sub_quotes_found"].sum() / real["n_sub_quotes"].sum()), 4)
                if len(real) and real["n_sub_quotes"].sum() else float("nan")),
            "answers_cut": round(float(cr["truncated"].mean()), 4) if len(cr) else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- the figures


def plot_code_mix(code_rows: pd.DataFrame, drop_truncated: bool = True):
    """A stacked bar for each case: the share of traces at each code."""
    import matplotlib.pyplot as plt

    rows = code_rows[~code_rows["truncated"]] if drop_truncated else code_rows
    cases = _ordered(rows["case"].unique())
    shares = np.zeros((len(cases), IC.MAX_CODE))
    for i, case in enumerate(cases):
        part = rows[rows["case"] == case]
        for code in range(1, IC.MAX_CODE + 1):
            shares[i, code - 1] = float((part["ic_code"] == code).mean())

    height = 0.42 * len(cases) + 1.7
    fig, ax = plt.subplots(figsize=(5.6, height))
    colours = [S.CMAP_SEQ(x) for x in np.linspace(0.18, 0.95, IC.MAX_CODE)]
    left = np.zeros(len(cases))
    for code in range(IC.MAX_CODE):
        ax.barh(range(len(cases)), shares[:, code], left=left,
                color=colours[code], label=CODE_LABEL[code + 1])
        left += shares[:, code]
    ax.set_yticks(range(len(cases)))
    ax.set_yticklabels([CASE_LABEL.get(c, c) for c in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of traces")
    ax.grid(axis="y", visible=False)
    # The legend goes under the whole figure, not under the axes: the axes
    # height follows the number of cases, thus an offset in axes units lands on
    # the x label when 1 case is drawn and far below it when 7 are.
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.95 / height)
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    return fig


def plot_code_by_label(code_rows: pd.DataFrame, drop_truncated: bool = True,
                       min_traces: int = 20):
    """The mean code for each judgment, 1 line for each case.

    A cell with fewer than `min_traces` traces is left out: a mean over 2 traces
    is noise, and `MuchLess` is rare in every case.
    """
    import matplotlib.pyplot as plt

    rows = code_rows[~code_rows["truncated"]] if drop_truncated else code_rows
    cases = _ordered(rows["case"].unique())
    labels = [x for x in LABEL_ORDER if x in set(rows["relative_label"])]

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    colours = [S.CMAP(x) for x in np.linspace(0.05, 0.95, max(len(cases), 2))]
    for i, case in enumerate(cases):
        part = rows[rows["case"] == case]
        means, keep = [], []
        for lab in labels:
            sub = part[part["relative_label"] == lab]
            if len(sub) >= min_traces:
                means.append(float(sub["ic_code"].mean()))
                keep.append(lab)
        if not means:
            continue
        ax.plot([labels.index(k) for k in keep], means, marker="o", markersize=3,
                linewidth=1.2, color=colours[i], label=CASE_LABEL.get(case, case))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean IC code")
    ax.legend(ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_components(code_rows: pd.DataFrame, drop_truncated: bool = True):
    """The rate of each component, so a reader sees WHICH rung moved."""
    import matplotlib.pyplot as plt

    rows = code_rows[~code_rows["truncated"]] if drop_truncated else code_rows
    cases = _ordered(rows["case"].unique())
    parts = [("differentiated", "differentiated"),
             ("integrated", "integrated"),
             ("context_sensitive", "context"),
             ("verdict_revised", "verdict revised"),
             ("pseudo_differentiation", "pseudo-differentiation")]

    fig_height = 0.42 * len(cases) + 1.7
    fig, ax = plt.subplots(figsize=(5.6, fig_height))
    height = 0.8 / len(parts)
    colours = [S.PAL[c] for c in ("blue", "green", "amber", "slate", "coral")]
    for j, (col, name) in enumerate(parts):
        values = [float(rows[rows["case"] == c][col].mean()) for c in cases]
        ax.barh([i + j * height for i in range(len(cases))], values,
                height=height, color=colours[j], label=name)
    ax.set_yticks([i + 0.4 - height / 2 for i in range(len(cases))])
    ax.set_yticklabels([CASE_LABEL.get(c, c) for c in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of traces")
    ax.grid(axis="y", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.95 / fig_height)
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    return fig


# ---------------------------------------------------------------- the export


def export(df: pd.DataFrame, code_rows: pd.DataFrame, out_dir: Path,
           thresholds: IC.Thresholds = IC.DEFAULT) -> List[str]:
    """Write every table and figure of the IC report into 1 folder."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _csv(frame: pd.DataFrame, name: str) -> None:
        if frame is None or frame.empty:
            return
        frame.to_csv(out_dir / name, index=False)
        written.append(name)

    _csv(case_table(code_rows), "ic_by_case.csv")
    _csv(label_table(code_rows), "ic_by_label.csv")
    _csv(quality_table(df, code_rows), "ic_quality.csv")
    _csv(pd.DataFrame([IC.thresholds_used(thresholds)]), "ic_thresholds.csv")
    # The per-trace codes travel with the tables: a claim about a case must be
    # traceable to the traces under it.
    code_rows.to_parquet(out_dir / "ic_codes.parquet", index=False)
    written.append("ic_codes.parquet")

    for name, builder in (("ic_code_mix.png", lambda: plot_code_mix(code_rows)),
                          ("ic_code_by_label.png", lambda: plot_code_by_label(code_rows)),
                          ("ic_components.png", lambda: plot_components(code_rows))):
        fig = builder()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        written.append(name)
    return written
