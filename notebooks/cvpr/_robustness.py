"""Robustness of the subway-safety judgments to prompt and presentation arms.

The battery re-runs the July 2026 reviewer-2 perturbations on the CANONICAL
setup: the minimal prompt, greedy decoding, each rater's production layout,
and 1,000 canonical pairs cut out of the 1,000,000-pair table. Plan and
arms: `vlm-narratives-docs/robustness-battery-rerun-plan.md`.
Launch: `scripts/launch_robustness_battery.sh`.

What this module reads
----------------------
A battery directory holds `<rater>/<arm>/runs/subway_safety__0000/`. Each run
has a result parquet and a `.hydra/config.yaml`. The config is the PROOF of
the arm: the shard submitter records only the base overrides, thus this
module reads the prompt text, the layout, the temperature, and the image
budget from the composed config and refuses a run that does not carry the
signal its arm name promises.

The 2 anchors
-------------
| Anchor | What it is | Who has it |
|--------|------------|------------|
| `retest` | The R0 arm: the base config, run again in the battery | every rater |
| `canonical` | The registered 1,000,000-pair run, on the same `pair_id` rows | gemma-4-12b, qwen3.5-9b |

Decoding is greedy, thus R0 against `canonical` reads the batch-effect floor:
the same prompt on the same images in another batch. Every other arm reads
against R0, the anchor that shares its job shape.

The measures (rows with `repeat_idx == 0`, joined on `pair_id`)
---------------------------------------------------------------
| Column | Meaning |
|--------|---------|
| `kappa_lw` | linear-weighted Cohen's kappa on the 5 ordinal labels, NotSure rows dropped on either side |
| `agree` | share of joined rows with the same label, NotSure included as a label |
| `flip_rate` | share of rows where both scores are non-zero and their signs differ |
| `spearman` | Spearman of `relative_score` over the rows with a score on both sides |
| `notsure_arm`, `notsure_anchor` | NotSure share of each side |
| `same_arm` | Same share of the arm |
| `much_arm` | MuchLess + MuchMore share of the arm (the one-image signature is Same high and this near 0) |
| `parse_fallback` | share of rows whose raw answer names no label (free decoding only) |
| `repeat_agree` | within the arm: share of repeat rows with the same label as their first draw |
| `first_share` | share of the arm's decided judgments (neither Same nor NotSure) that favour the FIRST image; 0.5 is no side bias |

Warning: a kappa against a same-order retest cannot see a side bias. A rater
that always favours the first image agrees with itself on every pair. Read
`first_share` beside every kappa; a value near 1 makes the kappa vacuous.

The `flipped` arm asks which entrance looks LESS safe, thus its labels and
scores are mirrored before the join.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import _canonical as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = HERE / "master" / "outputs"

ORDINAL = ["MuchLess", "Less", "Same", "More", "MuchMore"]
NOT_SURE = "NotSure"
MIRROR = {"MuchLess": "MuchMore", "Less": "More", "Same": "Same",
          "More": "Less", "MuchMore": "MuchLess", NOT_SURE: NOT_SURE}

#: Rater tag of the launch script -> short model name of the registry.
RATERS: Dict[str, str] = {
    "gemma12b": "gemma-4-12b",
    "qwen9b": "qwen3.5-9b",
    "qwen2b": "qwen3.5-2b",
    "e2b": "gemma-4-e2b",
}
#: The column labels of the table.
RATER_LABEL = {"gemma12b": "12B", "qwen9b": "9B", "qwen2b": "2B", "e2b": "E2B"}
#: The production layout of each rater; None is the stage default.
BASE_LAYOUT: Dict[str, Optional[str]] = {
    "gemma12b": "interleaved_labels", "qwen9b": None, "qwen2b": None,
    "e2b": "interleaved_labels",
}
L1_LAYOUT = {"gemma12b": "images_then_labels", "qwen9b": "interleaved_labels",
             "qwen2b": "interleaved_labels", "e2b": "images_then_labels"}
L2_LAYOUT = {"gemma12b": "text_first_interleaved", "qwen9b": "text_first",
             "qwen2b": "text_first", "e2b": "text_first_interleaved"}

#: Arm id -> table row label, in table order. The blank marks the block break.
ARM_LABEL: List[Tuple[str, str]] = [
    ("R0", "Retest (greedy floor)"),
    ("W1", "Paraphrase"),
    ("W2", "Persona added"),
    ("W3", "Option-order rev."),
    ("W5", "City name removed"),
    ("F1", "Halved image budget"),
    ("F2", "Free decoding"),
    ("F3", "Temperature 0.6"),
    ("L1", "Other binding layout"),
    ("", ""),
    ("W4", "Negated question"),
    ("L2", "Text-first layout"),
]
ARMS = [a for a, _ in ARM_LABEL if a]

COLS = ["pair_id", "canonical_pair_id", "repeat_idx", "is_swapped",
        "answer", "presented_score", "relative_label", "relative_score"]


# --------------------------------------------------------------------------
# Discovery and proof
# --------------------------------------------------------------------------

def newest_battery() -> Path:
    dirs = sorted(glob.glob(str(ROOT / "multirun" / "robustness_a_*")))
    if not dirs:
        raise SystemExit("no multirun/robustness_a_* directory")
    return Path(dirs[-1])


def run_dir(battery: Path, rater: str, arm: str) -> Optional[Path]:
    d = battery / rater / arm / "runs" / "subway_safety__0000"
    return d if d.is_dir() else None


def result_path(d: Path) -> Optional[Path]:
    files = sorted(glob.glob(str(d / "outputs" / "pairwise" / "subway_safety_mvp_*.parquet")))
    return Path(files[-1]) if files else None


def signals(d: Path) -> Dict[str, object]:
    """What the composed config says the run did."""
    cfg = yaml.safe_load(open(d / ".hydra" / "config.yaml"))
    p = cfg.get("prompt") or {}
    so = p.get("structured_output") or {}
    mm = (((cfg.get("model") or {}).get("engine_kwargs") or {})
          .get("mm_processor_kwargs") or {})
    user = str(p.get("user_template") or "")
    labels = re.findall(r"^\s*-\s*(MuchLess|Less|Same|More|MuchMore)\s*$", user, re.M)
    return {
        "temperature": float((cfg.get("sampling_params_vqa") or {}).get("temperature", -1)),
        "layout": p.get("image_layout"),
        "allow_not_sure": bool(so.get("allow_not_sure")),
        "structured": bool(so.get("enabled")),
        "system": p.get("system") is not None,
        "first_label": labels[0] if labels else "",
        "less_safe": "less safe" in user,
        "city": "New York City" in user,
        "user_head": user.strip().split("\n")[0][:40],
        "max_soft_tokens": mm.get("max_soft_tokens"),
        "max_pixels": mm.get("max_pixels"),
    }


def expected(rater: str, arm: str) -> Dict[str, object]:
    """The signals an arm must carry. Everything not named must equal the base."""
    base = {"temperature": 0.0, "layout": BASE_LAYOUT[rater], "allow_not_sure": True,
            "structured": True, "system": False, "first_label": "MuchLess",
            "less_safe": False, "city": True,
            "user_head": "Image A is the first image and Image B", "max_soft_tokens": None}
    if rater.startswith("qwen"):
        base["max_pixels"] = 1048576
    else:
        base["max_pixels"] = None
    change = {
        "R0": {},
        "W1": {"user_head": "The first image is Image A and the sec"},
        "W2": {"system": True},
        "W3": {"first_label": "MuchMore"},
        "W4": {"less_safe": True},
        "W5": {"city": False},
        "F1": ({"max_pixels": 262144} if rater.startswith("qwen")
               else {"max_soft_tokens": 140}),
        "F2": {"structured": False},
        "F3": {"temperature": 0.6},
        "L1": {"layout": L1_LAYOUT[rater]},
        "L2": {"layout": L2_LAYOUT[rater]},
    }[arm]
    return {**base, **change}


def prove(rater: str, arm: str, d: Path) -> List[str]:
    """The signals that do not match. Empty means the run is what it claims."""
    have, want = signals(d), expected(rater, arm)
    bad = []
    for k, v in want.items():
        h = have.get(k)
        if k == "user_head":
            ok = str(h).startswith(str(v))
        else:
            ok = h == v
        if not ok:
            bad.append(f"{k}: want {v!r}, have {h!r}")
    return bad


# --------------------------------------------------------------------------
# The measures
# --------------------------------------------------------------------------

def weighted_kappa(a: pd.Series, b: pd.Series) -> float:
    """Linear-weighted Cohen's kappa over the 5-point ordinal scale."""
    k = len(ORDINAL)
    idx = {lab: i for i, lab in enumerate(ORDINAL)}
    ai = a.map(idx).to_numpy()
    bi = b.map(idx).to_numpy()
    n = len(ai)
    if n == 0:
        return float("nan")
    obs = np.zeros((k, k))
    np.add.at(obs, (ai, bi), 1.0)
    obs /= n
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    denom = float((w * exp).sum())
    return float("nan") if denom == 0 else float(1.0 - (w * obs).sum() / denom)


_TOKEN = re.compile(r"much\s*less|much\s*more|not\s*sure|less|more|same", re.I)


def parse_fallback(raw: object) -> bool:
    """True when the raw answer names no label, thus the stage fell back."""
    return _TOKEN.search(str(raw)) is None


def read_run(path: Path, flip: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=COLS)
    if flip:
        df["relative_label"] = df["relative_label"].map(MIRROR).fillna(df["relative_label"])
        df["relative_score"] = -df["relative_score"]
    return df


def first_share(df: pd.DataFrame) -> float:
    """Share of decided judgments that favour the first image, as presented."""
    s = df.loc[df.repeat_idx == 0, "presented_score"].astype(float)
    s = s[s.notna() & (s != 0)]
    return float((s > 0).mean()) if len(s) else float("nan")


def repeat_agreement(df: pd.DataFrame) -> Tuple[float, int]:
    first = df[df.repeat_idx == 0].drop_duplicates("canonical_pair_id")
    rep = df[df.repeat_idx > 0]
    if rep.empty:
        return float("nan"), 0
    m = rep.merge(first[["canonical_pair_id", "relative_label"]],
                  on="canonical_pair_id", suffixes=("", "_0"))
    ok = (m.relative_label != NOT_SURE) & (m.relative_label_0 != NOT_SURE)
    if ok.sum() == 0:
        return float("nan"), 0
    return float((m.relative_label[ok] == m.relative_label_0[ok]).mean()), int(ok.sum())


def compare(arm: pd.DataFrame, anchor: pd.DataFrame) -> Dict[str, float]:
    a = arm[arm.repeat_idx == 0].set_index("pair_id")
    b = anchor[anchor.repeat_idx == 0].set_index("pair_id")
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    la, lb = j.relative_label_a, j.relative_label_b
    sa, sb = j.relative_score_a.astype(float), j.relative_score_b.astype(float)
    ok = la.isin(ORDINAL) & lb.isin(ORDINAL)
    nz = sa.notna() & sb.notna() & (sa != 0) & (sb != 0)
    both = sa.notna() & sb.notna()
    return {
        "n": int(len(j)),
        "coverage": float(len(j) / max(len(a), 1)),
        "kappa_lw": weighted_kappa(la[ok], lb[ok]),
        "n_kappa": int(ok.sum()),
        "agree": float((la == lb).mean()) if len(j) else float("nan"),
        "flip_rate": float((np.sign(sa[nz]) != np.sign(sb[nz])).mean()) if nz.any() else float("nan"),
        "spearman": float(sa[both].corr(sb[both], method="spearman")) if both.sum() > 2 else float("nan"),
        "notsure_arm": float((la == NOT_SURE).mean()),
        "notsure_anchor": float((lb == NOT_SURE).mean()),
        "same_arm": float((la == "Same").mean()),
        "much_arm": float(la.isin(["MuchLess", "MuchMore"]).mean()),
    }


# --------------------------------------------------------------------------
# The build
# --------------------------------------------------------------------------

def canonical_frame(rater: str, pair_ids: Sequence[str]) -> Optional[pd.DataFrame]:
    """The registered canonical run, on the battery's rows."""
    model = RATERS[rater]
    runs = C.runs(kind="proxy", case="subway", model=model)
    if not runs:
        return None
    df = pd.read_parquet(runs[0].results_link, columns=COLS)
    return df[df.pair_id.isin(set(pair_ids))].reset_index(drop=True)


def build(battery: Path, raters: Sequence[str] = tuple(RATERS),
          strict: bool = True) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rater in raters:
        frames: Dict[str, pd.DataFrame] = {}
        proofs: Dict[str, List[str]] = {}
        for arm in ARMS:
            d = run_dir(battery, rater, arm)
            p = result_path(d) if d else None
            if not p:
                continue
            bad = prove(rater, arm, d)
            if bad and strict:
                raise SystemExit(f"{rater}/{arm} is not what it claims: {bad}")
            proofs[arm] = bad
            frames[arm] = read_run(p, flip=(arm == "W4"))
        if "R0" not in frames:
            print(f"[robustness] {rater}: no R0 run, skipped")
            continue
        retest = frames["R0"]
        canon = canonical_frame(rater, retest.pair_id)
        anchors = {"retest": retest}
        if canon is not None and len(canon):
            anchors["canonical"] = canon
        for arm, df in frames.items():
            rep, rep_n = repeat_agreement(df)
            first = first_share(df)
            fallback = float(df[df.repeat_idx == 0]["answer"].map(parse_fallback).mean())
            for name, anc in anchors.items():
                if name == "retest" and arm == "R0":
                    continue
                rows.append({"rater": rater, "model": RATERS[rater], "arm": arm,
                             "anchor": name, "rows": int(len(df)),
                             "repeat_agree": rep, "repeat_n": rep_n,
                             "first_share": first,
                             "parse_fallback": fallback,
                             "proof": "; ".join(proofs[arm]) or "ok",
                             **compare(df, anc)})
    order = {a: i for i, a in enumerate(ARMS)}
    out = pd.DataFrame(rows)
    if len(out):
        out["_o"] = out.arm.map(order)
        out = out.sort_values(["rater", "_o", "anchor"]).drop(columns="_o").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

def _num(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    s = f"{v:.2f}"
    return f"${s}$" if v < 0 else s


def _cell(df: pd.DataFrame, rater: str, arm: str, anchor: str, col: str) -> float:
    m = df[(df.rater == rater) & (df.arm == arm) & (df.anchor == anchor)]
    return float(m[col].iloc[0]) if len(m) else float("nan")


def to_latex(df: pd.DataFrame, raters: Sequence[str] = tuple(RATERS),
             col: str = "kappa_lw") -> str:
    have = [r for r in raters if (df.rater == r).any()]
    head = " & ".join(RATER_LABEL[r] for r in have)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{l" + "r" * len(have) + "}", r"\toprule",
        r"& \multicolumn{" + str(len(have)) + r"}{c}{Agreement $\kappa$ vs.\ retest} \\",
        r"\cmidrule(lr){2-" + str(len(have) + 1) + "}",
        "Perturbation & " + head + r" \\", r"\midrule",
    ]
    for arm, label in ARM_LABEL:
        if not arm:
            lines.append(r"\midrule")
            continue
        if arm == "R0":
            cells = [_num(_cell(df, r, "R0", "canonical", col)) for r in have]
            lines.append(r"\textit{" + label + r"} & " + " & ".join(
                r"\textit{" + c + "}" for c in cells) + r" \\")
            lines.append(r"\midrule")
            continue
        cells = [_num(_cell(df, r, arm, "retest", col)) for r in have]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{\textbf{Robustness of subway-safety judgments to prompt and "
              r"presentation perturbations.} Linear-weighted $\kappa$ between each "
              r"perturbed arm and a same-job retest of the canonical configuration "
              r"($1{,}000$ canonical pairs cut from the production draw; greedy "
              r"decoding; abstentions excluded). The first row is the retest against "
              r"the registered production run on the same pairs, which is the "
              r"batch-effect floor of greedy decoding. ``---'': not available.}",
              r"\label{tab:robustness}", r"\end{table}"]
    return "\n".join(lines)


def rates_to_latex(df: pd.DataFrame, raters: Sequence[str] = tuple(RATERS),
                   col: str = "notsure_arm", caption: str = "",
                   tab_label: str = "tab:robustness-abstention") -> str:
    """A share of each arm as a percentage, the companion tables."""
    have = [r for r in raters if (df.rater == r).any()]
    head = " & ".join(RATER_LABEL[r] for r in have)
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * len(have) + "}", r"\toprule",
             "Perturbation & " + head + r" \\", r"\midrule"]
    for arm, label in ARM_LABEL:
        if not arm:
            lines.append(r"\midrule")
            continue
        vals = []
        for r in have:
            m = df[(df.rater == r) & (df.arm == arm)]
            vals.append(f"{100 * float(m[col].iloc[0]):.0f}\\%" if len(m) else "---")
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{" + (caption or "Abstention (``not sure'') share of each perturbation arm.") + "}",
              r"\label{" + tab_label + "}", r"\end{table}"]
    return "\n".join(lines)


def export(battery: Path, out_dir: Path = OUT_DIR, version: str = "1.0.0",
           strict: bool = True) -> List[Path]:
    C.verify_or_raise(quick=True)
    df = build(battery, strict=strict)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    csv = out_dir / f"robustness_v{version}.csv"
    df.to_csv(csv, index=False)
    written.append(csv)
    side = ("Side bias of each perturbation arm: the share of decided judgments "
            "(neither Same nor NotSure) that favour the first image as presented. "
            "0.5 is no bias. A kappa against a same-order retest cannot see this, "
            "thus a value near 0 or 1 makes the kappa in \\cref{tab:robustness} vacuous.")
    for name, text in [(f"robustness_v{version}.tex", to_latex(df)),
                       (f"robustness_abstention_v{version}.tex", rates_to_latex(df)),
                       (f"robustness_side_v{version}.tex",
                        rates_to_latex(df, col="first_share", caption=side,
                                       tab_label="tab:robustness-side"))]:
        p = out_dir / name
        p.write_text("% Generated by _robustness.py — do not edit by hand.\n"
                     f"% Battery: {battery}\n% Needs \\usepackage{{booktabs}}.\n"
                     + text + "\n")
        written.append(p)
    (out_dir / f"robustness_v{version}.json").write_text(json.dumps(
        {"battery": str(battery), "registry": C.load().get("generated_at")}, indent=1))
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--battery", type=Path, default=None,
                    help="the multirun/robustness_a_* dir; default: the newest")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--no-strict", action="store_true",
                    help="report a run whose config does not prove its arm instead of stopping")
    args = ap.parse_args(argv)
    battery = args.battery or newest_battery()
    for f in export(battery, args.out_dir, args.version, strict=not args.no_strict):
        print(f"[robustness] wrote {f}")
    df = pd.read_csv(args.out_dir / f"robustness_v{args.version}.csv")
    show = df[df.anchor.eq("retest") | df.arm.eq("R0")]
    print(show[["rater", "arm", "anchor", "n_kappa", "kappa_lw", "agree", "flip_rate",
                "notsure_arm", "same_arm", "much_arm", "first_share", "repeat_agree",
                "parse_fallback", "proof"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
