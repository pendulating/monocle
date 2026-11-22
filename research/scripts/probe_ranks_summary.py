"""Summarize the label-rank trajectory of the whole census (CPU).

Reads `outputs/_monocle/probe/ranks/<probe>/ranks_shard*.parquet` and
the census (`scores/<probe>/`), and writes `ranks/trajectory.csv` and
`ranks/trajectory.md`: for each probe, block, and split (answered or
refused), the median rank of the 3 labels, the share of images where the
stigmatized pole leads the plain one, and the share where the abstain
token leads both.

    .venv-3.12/bin/python research/scripts/probe_ranks_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_frame import ABSTAIN, PROBE_ORDER, PROBES  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe"


def main() -> int:
    global ROOT
    if len(sys.argv) > 2 and sys.argv[1] == "--root":
        ROOT = Path(sys.argv[2])
    rows = []
    for probe in PROBE_ORDER:
        parts = sorted((ROOT / "ranks" / probe).glob("ranks_shard*.parquet"))
        if not parts:
            print(f"[ranks-summary] {probe}: no shards; skipped")
            continue
        rk = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        cen = pd.concat([pd.read_parquet(p) for p in
                         sorted((ROOT / "scores" / probe).glob("scores_shard*of*.parquet"))])
        rk = rk.merge(cen[["image_path", "answered", "top_class"]], on="image_path", how="left")
        rk["split"] = rk["answered"].map({True: "answered", False: "refused"})
        g = rk.groupby(["split", "layer"]).agg(
            n=("image_path", "nunique"), rank_pos=("rank_pos", "median"),
            rank_neg=("rank_neg", "median"), rank_abstain=("rank_abstain", "median"),
            neg_leads=("neg_leads", "mean"), abstain_leads=("abstain_leads", "mean"),
            s_mean=("s", "mean")).reset_index()
        g.insert(0, "probe", probe)
        rows.append(g)
    if not rows:
        return 1
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(ROOT / "ranks" / "trajectory.csv", index=False)
    md = ["| Probe | Split | Block | n | rank pos | rank neg | rank abstain | neg above pos | abstain above both |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        p = PROBES[r.probe]
        md.append(f"| {r.probe} ({p.pos}/{p.neg}) | {r.split} | {int(r.layer)} | {int(r.n)} | "
                  f"{r.rank_pos:.0f} | {r.rank_neg:.0f} | {r.rank_abstain:.0f} | "
                  f"{r.neg_leads:.0%} | {r.abstain_leads:.0%} |")
    (ROOT / "ranks" / "trajectory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
