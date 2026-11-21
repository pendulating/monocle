"""Step 1 report: what the model writes under each injection.

Per case, mode and condition: the share of generations that name the
positive word first, the negative word first, or neither, and the share
of free generations that form the JSON answer object.

Usage:
    python scripts/steer_generate_report.py --cases all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
ROOT = REPO / "outputs/_monocle/steer_generate"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]


def classify_strict(text: str, pos: str, neg: str) -> str:
    """``pos``, ``neg``, ``both`` (a hedge names both words) or ``other``."""
    t = text.lower()
    has_p, has_n = pos in t, neg in t
    if has_p and has_n:
        return "both"
    if has_p:
        return "pos"
    if has_n:
        return "neg"
    return "other"


DEGENERATE = re.compile(r"(\b\w{2,}\b)(?:\W{0,3}\1){3,}")


def degenerate(text: str) -> bool:
    """A word repeated 4 or more times in a row is a broken continuation."""
    return bool(DEGENERATE.search(text.lower()))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--cases", default="all")
    args = ap.parse_args(argv)
    root = Path(args.root)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        f = root / case / "generations.parquet"
        if not f.exists():
            print(f"[steer-generate-report] {case}: no run yet")
            continue
        s = json.loads((root / case / "summary.json").read_text())
        pos_w, neg_w = s["words"]
        g = pd.read_parquet(f)
        g["cls"] = [classify_strict(t, pos_w, neg_w) for t in g["text"]]
        g["degenerate"] = [degenerate(t) for t in g["text"]]
        for (mode, arm, dose), d in g.groupby(["mode", "arm", "dose"]):
            rows.append({
                "case": case, "mode": mode, "arm": arm, "dose": float(dose),
                "n": int(len(d)),
                "pos": float((d["cls"] == "pos").mean()),
                "neg": float((d["cls"] == "neg").mean()),
                "both": float((d["cls"] == "both").mean()),
                "other": float((d["cls"] == "other").mean()),
                "degenerate": float(d["degenerate"].mean()),
                "json_ok": float(d["json_ok"].mean()),
                "example": str(d["text"].iloc[0])[:50].replace("\n", " "),
            })
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    order = {"baseline": 0, "axis": 1, "signal": 2, "control": 3,
             "control_raw": 4}
    df["k"] = df["arm"].map(order)
    df = df.sort_values(["case", "mode", "k", "dose"]).drop(columns="k")
    for mode in ("prefix", "free"):
        d = df[df["mode"] == mode]
        if not len(d):
            continue
        print(f"\nGENERATION, mode={mode}  (share of images)")
        print(d[["case", "arm", "dose", "pos", "neg", "both", "other",
                 "degenerate", "json_ok", "example"]].round(3)
              .to_string(index=False))
    df.to_parquet(root / "generate_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
