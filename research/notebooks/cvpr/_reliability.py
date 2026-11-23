"""Positional bias and test-retest reliability of the pairwise battery.

A pairwise judgment carries 2 sources of error that a proxy correlation
cannot see. The first is the side: a model can favour the first image no
matter what it shows. The second is the repeat: the same 2 places, judged
again, can get a different label. This module measures both on the
registered canonical runs, and writes the 2 camera-ready tables.

How the battery controls the side
---------------------------------
The sampler swaps the presentation order on every second row
(`counterbalance_mode: balanced`), and the stage undoes the swap when it
writes `relative_label`. Thus the raw bias cancels in the aggregate, and the
de-swapped mean is the number that says whether it did.

How the battery measures the repeat
-----------------------------------
`repeat_fraction: 0.10` draws 100,000 more observations of canonical pairs
the battery already holds. A repeat gets its own swap state by parity, thus
about half the repeats come in the same order and half in the opposite
order. The 2 halves answer 2 questions:

| Half | Facility case | Image case |
|------|---------------|------------|
| same order | fresh photographs, same side | same photographs, same side: decoding stability |
| opposite order | fresh photographs, other side | same photographs, other side: order sensitivity |

A facility case draws fresh photographs of both units on each observation,
thus its repeat mixes image variation with model noise. An image case
(road quality, street photography) repeats the identical 2 images.

Warning: the read goes through the canonical registry. Every number here
names the same run as the paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _canonical as C
from research.notebooks.cvpr._results_table import MODEL_LABEL, RAW_CASE_LABEL, cohen_kappa

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "master" / "outputs"
CACHE_DIR = OUT_DIR / "reliability_cache"

#: The short model names, in the column order of the table.
MODELS: Tuple[str, ...] = tuple(MODEL_LABEL.values())
#: The row order of the table.
CASES: List[str] = list(RAW_CASE_LABEL)
#: The cases where a repeat shows the identical 2 images.
IMAGE_CASES = {"road_quality", "street_photography"}
#: The model names as the paper prints them.
DISPLAY = {"gemma-4-12b": "Gemma-4-12B", "qwen3.5-9b": "Qwen3.5-9B"}

#: A bias cell needs this many decided judgments, or it prints a dash.
MIN_DECIDED = 10_000
#: A reliability cell needs this many decided repeat pairs, or it prints a dash.
MIN_BOTH = 1_000

#: The columns the read needs. Nothing else leaves the parquet.
COLS = [
    "pair_id", "canonical_pair_id", "repeat_idx", "is_swapped",
    "presented_label", "presented_score", "relative_label", "relative_score",
    "image_path_a", "image_path_b",
]

NOT_SURE = "NotSure"
SUBSETS = ("all", "same_order", "opposite_order")


# --------------------------------------------------------------------------
# The measures
# --------------------------------------------------------------------------

def _ci_mean(x: pd.Series) -> Tuple[float, float, int]:
    """The mean and its 95% half-width, under the normal approximation."""
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if len(v) < 2:
        return float("nan"), float("nan"), len(v)
    return float(v.mean()), float(1.96 * v.std(ddof=1) / math.sqrt(len(v))), len(v)


def _ci_prop(k: int, n: int) -> Tuple[float, float]:
    """A share and its 95% half-width, under the normal approximation."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def bias_of(df: pd.DataFrame) -> Dict[str, float]:
    """Measure the side bias of 1 run, before and after the de-swap.

    `first_wins` is the share of decided judgments that favour the first
    image. A decided judgment is any label other than Same or NotSure. Under
    no bias it is 0.5. `presented_mean` is the raw score, where a positive
    value favours the first image. `relative_mean` is the same score after
    the stage undid the swap, thus it is the residual the paper reports.
    """
    ps = pd.to_numeric(df["presented_score"], errors="coerce")
    rs = pd.to_numeric(df["relative_score"], errors="coerce")
    decided = ps.notna() & (ps != 0)
    n_dec = int(decided.sum())
    first, first_hw = _ci_prop(int((ps[decided] > 0).sum()), n_dec)
    pm, pm_hw, _ = _ci_mean(ps)
    rm, rm_hw, _ = _ci_mean(rs)
    lab = df["presented_label"].astype(str)
    return {
        "n_rows": int(len(df)),
        "swapped_share": float(df["is_swapped"].astype(bool).mean()),
        "not_sure_rate": float(lab.eq(NOT_SURE).mean()),
        "same_rate": float(lab.eq("Same").mean()),
        "n_decided": n_dec,
        "decided_share": n_dec / len(df) if len(df) else float("nan"),
        "first_wins": first,
        "first_wins_hw": first_hw,
        "presented_mean": pm,
        "presented_hw": pm_hw,
        "relative_mean": rm,
        "relative_hw": rm_hw,
    }


def _retest_subset(g: pd.DataFrame) -> Dict[str, float]:
    """Compare each repeat against its base row, on 1 subset of the repeats."""
    a = g["relative_label"].astype(str)
    b = g["relative_label_base"].astype(str)
    sa = pd.to_numeric(g["relative_score"], errors="coerce")
    sb = pd.to_numeric(g["relative_score_base"], errors="coerce")
    both = sa.notna() & sb.notna()
    decided = both & (sa != 0) & (sb != 0)
    ns_a, ns_b = a.eq(NOT_SURE), b.eq(NOT_SURE)
    any_ns = ns_a | ns_b
    return {
        "n_pairs": int(len(g)),
        "same_images_share": float(g["same_images"].mean()) if len(g) else float("nan"),
        "exact_agree": float((a == b).mean()) if len(g) else float("nan"),
        "kappa": cohen_kappa(a.tolist(), b.tolist()) if len(g) else float("nan"),
        "n_answered": int(both.sum()),
        "side_agree": (float((np.sign(sa[both]) == np.sign(sb[both])).mean())
                       if both.sum() else float("nan")),
        "n_decided": int(decided.sum()),
        "sign_agree": (float((np.sign(sa[decided]) == np.sign(sb[decided])).mean())
                       if decided.sum() else float("nan")),
        "spearman": (float(sa[both].corr(sb[both], method="spearman"))
                     if both.sum() > 2 else float("nan")),
        "not_sure_agree": (float((ns_a & ns_b).sum() / any_ns.sum())
                           if any_ns.sum() else float("nan")),
    }


def retest_of(df: pd.DataFrame) -> List[Dict[str, float]]:
    """Measure the test-retest agreement of 1 run, on the 3 repeat subsets.

    Each repeat row joins its base row (`repeat_idx == 0`) on the canonical
    pair id. Both labels are already de-swapped, thus an agreement means the
    same verdict about the same 2 places, whatever side each one sat on.

    `sign_agree` keeps the repeats where both judgments are decided and counts
    the share that name the same winner. `side_agree` keeps Same as a third
    side and drops only an abstention. `exact_agree` and `kappa` compare the
    6 labels, NotSure included.
    """
    base = df[df["repeat_idx"] == 0].set_index("canonical_pair_id")
    rep = df[df["repeat_idx"] > 0]
    keep = ["is_swapped", "relative_label", "relative_score",
            "image_path_a", "image_path_b"]
    j = rep.join(base[keep], on="canonical_pair_id", rsuffix="_base", how="inner")
    j["same_order"] = j["is_swapped"].astype(bool) == j["is_swapped_base"].astype(bool)
    j["same_images"] = ((j["image_path_a"] == j["image_path_a_base"])
                        & (j["image_path_b"] == j["image_path_b_base"]))
    parts = {
        "all": j,
        "same_order": j[j["same_order"]],
        "opposite_order": j[~j["same_order"]],
    }
    return [{"subset": k, **_retest_subset(v)} for k, v in parts.items()]


# --------------------------------------------------------------------------
# The read, with a cache
# --------------------------------------------------------------------------

def _cache_path(run: C.CanonicalRun) -> Path:
    sha = (run.results_sha256 or "nosha")[:12]
    return CACHE_DIR / f"{run.arm}__{run.case}__{run.model}__{sha}.json"


def measure_run(run: C.CanonicalRun, use_cache: bool = True) -> Dict[str, object]:
    """Read 1 run and return its bias and its retest rows.

    The read of 1,100,000 rows takes about 10 seconds, and the battery holds
    14 runs. The result goes to a JSON file named by the run hash, thus a
    second call costs nothing and a new run never reads a stale file.
    """
    path = _cache_path(run)
    if use_cache and path.exists():
        return json.loads(path.read_text())
    df = pd.read_parquet(run.results_link, columns=COLS)
    out = {"bias": bias_of(df), "retest": retest_of(df)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def build(models: Sequence[str] = MODELS, arm: str = "canonical",
          use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Measure every registered proxy run of 1 arm.

    Returns 2 long frames. `bias` holds 1 row for each case and model.
    `retest` holds 3 rows for each case and model, 1 for each repeat subset.
    A cell the registry does not hold is absent, not dashed.
    """
    bias_rows, retest_rows = [], []
    for run in C.runs(kind="proxy", arm=arm):
        if run.model not in models:
            continue
        rec = measure_run(run, use_cache=use_cache)
        head = {"case": run.case, "case_label": RAW_CASE_LABEL.get(run.case, run.case),
                "model": run.model, "arm": run.arm,
                "image_case": run.case in IMAGE_CASES}
        bias_rows.append({**head, **rec["bias"]})
        for r in rec["retest"]:
            retest_rows.append({**head, **r})
    order = {c: i for i, c in enumerate(CASES)}
    morder = {m: i for i, m in enumerate(models)}

    def _sort(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return (df.assign(_c=df["case"].map(order), _m=df["model"].map(morder))
                  .sort_values(["_c", "_m"]).drop(columns=["_c", "_m"])
                  .reset_index(drop=True))

    return _sort(pd.DataFrame(bias_rows)), _sort(pd.DataFrame(retest_rows))


# --------------------------------------------------------------------------
# The wide views
# --------------------------------------------------------------------------

def bias_wide(bias: pd.DataFrame) -> pd.DataFrame:
    """1 row for each case, the 2 models side by side."""
    if bias.empty:
        return bias
    w = bias.pivot(index="case_label", columns="model",
                   values=["decided_share", "first_wins", "presented_mean",
                           "relative_mean", "relative_hw"])
    w = w.swaplevel(axis=1).sort_index(axis=1, level=0,
                                       key=lambda idx: idx.map(
                                           {m: i for i, m in enumerate(MODELS)}))
    labels = [RAW_CASE_LABEL[c] for c in CASES if RAW_CASE_LABEL[c] in w.index]
    return w.reindex(labels).round(4)


def retest_wide(retest: pd.DataFrame) -> pd.DataFrame:
    """1 row for each case, the sign agreement of both halves, both models."""
    if retest.empty:
        return retest
    sub = retest[retest["subset"] != "all"]
    w = sub.pivot(index="case_label", columns=["model", "subset"],
                  values="sign_agree")
    cols = [(m, s) for m in MODELS for s in ("same_order", "opposite_order")
            if (m, s) in w.columns]
    labels = [RAW_CASE_LABEL[c] for c in CASES if RAW_CASE_LABEL[c] in w.index]
    return w.reindex(index=labels, columns=pd.MultiIndex.from_tuples(cols)).round(3)


# --------------------------------------------------------------------------
# Camera-ready output
# --------------------------------------------------------------------------

def _num(v: float, digits: int = 2, signed: bool = False) -> str:
    """Format 1 number for the table. NaN prints a dash."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    # A value that rounds to 0 carries no sign, or it prints as "-.000".
    if round(v, digits) == 0:
        v = 0.0
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    s = s.replace("+0.", "0.", 1) if v == 0 else s
    # A leading zero costs a character the column does not have.
    s = s.replace("0.", ".", 1) if s.lstrip("+-").startswith("0.") else s
    return f"${s}$" if signed else s


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _row_label(case: str, dagger: bool) -> str:
    text = RAW_CASE_LABEL.get(case, case)
    return f"{text}$^\\dagger$" if dagger else text


def _cell(df: pd.DataFrame, case: str, model: str, **where) -> Optional[pd.Series]:
    sub = df[(df["case"] == case) & (df["model"] == model)]
    for k, v in where.items():
        sub = sub[sub[k] == v]
    return None if sub.empty else sub.iloc[0]


BIAS_LABEL = "tab:positional-bias"
BIAS_CAPTION = (
    "Positional bias over the {rows} comparisons of each case. "
    "\\emph{{first}} is the share of decided judgments (neither Same nor Not Sure) "
    "that favour the first image; 0.5 is no bias. "
    "$\\bar{{s}}$ is the mean score on the $-2$ to $+2$ scale after the "
    "presentation order is undone; its 95\\% interval is within "
    "$\\pm{hw}$ of the value in every printed cell. "
    "Half of all presentations are swapped, thus the raw bias cancels in the "
    "aggregate. A dash marks fewer than {min_decided} decided judgments."
)

RETEST_LABEL = "tab:test-retest"
RETEST_CAPTION = (
    "Test-retest sign agreement over the {repeats} repeated pairs of each case. "
    "A repeat shows the same pair again, in the same order or with "
    "the sides swapped; the table keeps the repeats where both judgments are "
    "decided and counts the share that name the same winner. "
    "Cases marked $\\dagger$ repeat the identical two images, thus "
    "\\emph{{same}} measures decoding stability alone and \\emph{{swap}} measures "
    "order sensitivity alone; the other cases draw fresh photographs of the "
    "same two units. 0.5 is chance, and a value below it means the side beats "
    "the content. A dash marks fewer than {min_both} decided repeats."
)

COMBINED_LABEL = "tab:order-effects"
COMBINED_CAPTION = (
    "Order effects in the pairwise battery, {rows} comparisons and {repeats} "
    "repeats for each case. "
    "\\emph{{first}} is the share of decided judgments that favour the first "
    "image (0.5 is no bias); after the swap is undone, the mean score sits "
    "within $\\pm{hw}$ of zero in every printed cell. "
    "\\emph{{same}} and \\emph{{swap}} are the shares of decided repeats that name "
    "the same winner, when the repeat kept or swapped the sides (0.5 is chance). "
    "Cases marked $\\dagger$ repeat the identical two images; the others draw "
    "fresh photographs of the same two units. "
    "A dash marks fewer than {min_decided} decided judgments or {min_both} "
    "decided repeats."
)


def _common_counts(bias: pd.DataFrame, retest: pd.DataFrame) -> Dict[str, str]:
    """The counts the captions state. Each is 1 number, or a range."""
    def _one(values: Sequence[int]) -> str:
        vals = sorted(set(int(v) for v in values if pd.notna(v)))
        if not vals:
            return "0"
        if len(vals) == 1:
            return _thousands(vals[0])
        return f"{_thousands(vals[0])}--{_thousands(vals[-1])}"
    rows = _one(bias["n_rows"]) if not bias.empty else "0"
    reps = (_one(retest.loc[retest["subset"] == "all", "n_pairs"])
            if not retest.empty else "0")
    printed = bias[bias["n_decided"] >= MIN_DECIDED] if not bias.empty else bias
    hw = float(printed["relative_hw"].max()) if not printed.empty else float("nan")
    hw_txt = f"{math.ceil(hw * 1000) / 1000:.3f}" if np.isfinite(hw) else "?"
    return {"rows": rows, "repeats": reps, "hw": hw_txt,
            "min_decided": _thousands(MIN_DECIDED), "min_both": _thousands(MIN_BOTH)}


def _open(env: str, caption: str, label: str, size: str, colspec: str,
          resize: bool) -> List[str]:
    lines = [
        "% Generated by _reliability.py — do not edit by hand.",
        "% Needs \\usepackage{booktabs}" + (" and \\usepackage{graphicx}." if resize else "."),
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
        size,
    ]
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    return lines


def _close(env: str, resize: bool) -> List[str]:
    lines = ["\\bottomrule", "\\end{tabular}" + ("%" if resize else "")]
    if resize:
        lines.append("}")
    lines.append(f"\\end{{{env}}}")
    return lines


def bias_to_latex(bias: pd.DataFrame, retest: Optional[pd.DataFrame] = None,
                  caption: Optional[str] = None, label: str = BIAS_LABEL,
                  models: Sequence[str] = MODELS, full_width: bool = False) -> str:
    """The positional-bias table: 1 column wide, 5 columns.

    Each model gets 2 columns: the raw share that favours the first image,
    and the de-swapped mean. The confidence interval of the mean goes to the
    caption, because it is the same width on every printed cell.
    """
    if bias.empty:
        return "% no positional-bias results"
    counts = _common_counts(bias, retest if retest is not None else pd.DataFrame())
    text = caption if caption is not None else BIAS_CAPTION.format(**counts)
    env = "table*" if full_width else "table"
    lines = _open(env, text, label, "\\small", "l" + "rr" * len(models), resize=False)
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DISPLAY.get(m, m)}}}" for m in models)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
    lines += [f" & {heads} \\\\", rules,
              "Case & " + " & ".join("first & $\\bar{s}$" for _ in models) + " \\\\",
              "\\midrule"]
    for case in CASES:
        cells = []
        for m in models:
            r = _cell(bias, case, m)
            if r is None or int(r["n_decided"]) < MIN_DECIDED:
                cells += ["--", "--"]
            else:
                cells += [_num(r["first_wins"], 2), _num(r["relative_mean"], 3, signed=True)]
        if all(c == "--" for c in cells):
            continue
        lines.append(" & ".join([RAW_CASE_LABEL.get(case, case), *cells]) + " \\\\")
    lines += _close(env, resize=False)
    return "\n".join(lines)


def retest_to_latex(retest: pd.DataFrame, bias: Optional[pd.DataFrame] = None,
                    caption: Optional[str] = None, label: str = RETEST_LABEL,
                    models: Sequence[str] = MODELS, full_width: bool = False) -> str:
    """The test-retest table: 1 column wide, 5 columns.

    Each model gets 2 columns: the sign agreement when the repeat kept the
    order, and when it swapped the sides. The image cases carry a dagger,
    because there the repeat shows the identical 2 images.
    """
    if retest.empty:
        return "% no test-retest results"
    counts = _common_counts(bias if bias is not None else pd.DataFrame(), retest)
    text = caption if caption is not None else RETEST_CAPTION.format(**counts)
    env = "table*" if full_width else "table"
    lines = _open(env, text, label, "\\small", "l" + "rr" * len(models), resize=False)
    heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{DISPLAY.get(m, m)}}}" for m in models)
    rules = "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(models)))
    lines += [f" & {heads} \\\\", rules,
              "Case & " + " & ".join("same & swap" for _ in models) + " \\\\",
              "\\midrule"]
    for case in CASES:
        cells = []
        for m in models:
            for subset in ("same_order", "opposite_order"):
                r = _cell(retest, case, m, subset=subset)
                void = r is None or int(r["n_decided"]) < MIN_BOTH
                cells.append("--" if void else _num(r["sign_agree"], 2))
        if all(c == "--" for c in cells):
            continue
        lines.append(" & ".join([_row_label(case, case in IMAGE_CASES), *cells]) + " \\\\")
    lines += _close(env, resize=False)
    return "\n".join(lines)


def combined_to_latex(bias: pd.DataFrame, retest: pd.DataFrame,
                      caption: Optional[str] = None, label: str = COMBINED_LABEL,
                      models: Sequence[str] = MODELS, full_width: bool = False) -> str:
    """Both measures in 1 table: 1 column wide, 7 columns.

    Each model gets 3 columns: first, same, swap. The de-swapped mean leaves
    the table and goes to the caption as a bound, because it is near 0 on
    every cell. `\\resizebox` fits the 7 columns to the column width.
    """
    if bias.empty or retest.empty:
        return "% no order-effect results"
    counts = _common_counts(bias, retest)
    text = caption if caption is not None else COMBINED_CAPTION.format(**counts)
    env = "table*" if full_width else "table"
    lines = _open(env, text, label, "\\small", "l" + "rrr" * len(models), resize=True)
    heads = " & ".join(f"\\multicolumn{{3}}{{c}}{{{m}}}" for m in models)
    rules = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models)))
    lines += [f" & {heads} \\\\", rules,
              "Case & " + " & ".join("first & same & swap" for _ in models) + " \\\\",
              "\\midrule"]
    for case in CASES:
        cells = []
        for m in models:
            b = _cell(bias, case, m)
            cells.append("--" if b is None or int(b["n_decided"]) < MIN_DECIDED
                         else _num(b["first_wins"], 2))
            for subset in ("same_order", "opposite_order"):
                r = _cell(retest, case, m, subset=subset)
                void = r is None or int(r["n_decided"]) < MIN_BOTH
                cells.append("--" if void else _num(r["sign_agree"], 2))
        if all(c == "--" for c in cells):
            continue
        lines.append(" & ".join([_row_label(case, case in IMAGE_CASES), *cells]) + " \\\\")
    lines += _close(env, resize=True)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The export, without marimo
# --------------------------------------------------------------------------

VERSION = "1.0.0"


def export(out_dir: Path = OUT_DIR, arm: str = "canonical",
           use_cache: bool = True, version: str = VERSION,
           cases: Optional[Sequence[str]] = None) -> List[str]:
    """Write the 3 LaTeX tables and the 2 long CSV files.

    Args:
        cases: Keep only these cases in the tables and the CSV files. None keeps all.
    """
    bias, retest = build(arm=arm, use_cache=use_cache)
    if cases is not None:
        bias = bias[bias["case"].isin(cases)].reset_index(drop=True)
        retest = retest[retest["case"].isin(cases)].reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if arm == "canonical" else f"_{arm}"
    files = {
        f"positional_bias{suffix}_v{version}.tex": bias_to_latex(bias, retest),
        f"test_retest{suffix}_v{version}.tex": retest_to_latex(retest, bias),
        f"order_effects{suffix}_v{version}.tex": combined_to_latex(bias, retest),
    }
    for name, text in files.items():
        (out_dir / name).write_text(text + "\n")
    bias.to_csv(out_dir / f"positional_bias{suffix}_v{version}.csv", index=False)
    retest.to_csv(out_dir / f"test_retest{suffix}_v{version}.csv", index=False)
    return [*files, f"positional_bias{suffix}_v{version}.csv",
            f"test_retest{suffix}_v{version}.csv"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="canonical")
    ap.add_argument("--no-cache", action="store_true", help="Read every run again.")
    ap.add_argument("--cases", nargs="+", choices=CASES, default=None,
                    help="Keep only these cases in the tables.")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="Skip the registry gate. For a look, never for the paper.")
    args = ap.parse_args()
    if not args.allow_unverified:
        C.verify_or_raise()
    for f in export(arm=args.arm, use_cache=not args.no_cache, cases=args.cases):
        print(OUT_DIR / f)
