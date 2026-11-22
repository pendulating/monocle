"""Turn a lens-free rank census into the reads table of the group test.

The base-model arm of the lens-refusal probe (2026-09-07) scores 3 arms
with `monocle.probe_ranks --lens none`: the pretrained checkpoint under
the raw frame, the instruct checkpoint under the raw frame, and the
instruct checkpoint under its chat frame. Each arm gives, per image and
block, the plain logit-lens gap `s = logit(pos) - logit(neg)` and the
top token. This script writes, for one arm root:

- `<root>/vectors/<probe>/reads.parquet` with `c` (the gap at the final
  block), `answered`, `top_class`, `calib` (False), and one
  `read_none_L<block>` column for each block, so
  `research/scripts/probe_group_test.py --root <root>` runs as it is;
- `<root>/sample` as a link to the round-2 sample when it is absent;
- `<root>/census_own_draw.csv`, the top-class shares on the own draw.

    .venv-vllm025cu129/bin/python research/scripts/probe_base_reads.py --root outputs/_monocle/probe_base/base_raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_frame import PROBE_ORDER, PROBES  # noqa: E402

SAMPLE = REPO / "outputs/_monocle/probe_nta/sample"


def log(m: str) -> None:
    print(f"[probe-base-reads] {m}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--probes", nargs="+", default=list(PROBE_ORDER))
    args = ap.parse_args(argv)
    root = Path(args.root)
    if not (root / "sample").exists():
        (root / "sample").symlink_to(SAMPLE)
    images = pd.read_parquet(SAMPLE / "images.parquet")
    census = []
    for p in args.probes:
        parts = sorted((root / "ranks" / p).glob("ranks_shard*.parquet"))
        if not parts:
            log(f"{p}: no shards; skipped")
            continue
        rk = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
        final = int(rk["layer"].max())
        wide = rk.pivot_table(index="image_path", columns="layer", values="s")
        wide.columns = [f"read_none_L{int(c)}" for c in wide.columns]
        last = rk[rk["layer"] == final].set_index("image_path")
        df = wide.join(last[["s", "top1", "top5", "top_class", "rank_pos", "rank_neg",
                             "rank_abstain"]])
        df = df.rename(columns={"s": "c"})
        df["answered"] = df["top_class"].isin(["pos", "neg"])
        df["calib"] = False
        df["probe"] = p
        df = df.reset_index()
        out = root / "vectors" / p
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "reads.parquet", index=False)
        own = df.merge(images[["image_path", f"draw_{p}", f"group_{p}"]], on="image_path")
        own = own[own[f"draw_{p}"]]
        vc = own["top_class"].value_counts(normalize=True)
        oth = own[own["top_class"] == "other"]["top1"].value_counts().head(5).to_dict()
        by = own.groupby(f"group_{p}")["top_class"].value_counts(normalize=True).unstack(fill_value=0)
        census.append({"probe": p, "pos": PROBES[p].pos, "neg": PROBES[p].neg, "n": len(own),
                       "pos_pct": round(100 * vc.get("pos", 0), 1),
                       "neg_pct": round(100 * vc.get("neg", 0), 1),
                       "abstain_pct": round(100 * vc.get("abstain", 0), 1),
                       "other_pct": round(100 * vc.get("other", 0), 1), "other_top": oth,
                       "neg_pct_top": round(100 * by.get("neg", pd.Series()).get("top", 0), 1),
                       "neg_pct_bottom": round(100 * by.get("neg", pd.Series()).get("bottom", 0), 1),
                       "pos_pct_top": round(100 * by.get("pos", pd.Series()).get("top", 0), 1),
                       "pos_pct_bottom": round(100 * by.get("pos", pd.Series()).get("bottom", 0), 1),
                       "final_block": final, "images": len(df)})
        log(f"{p}: {len(df)} images, final block {final}, own draw {len(own)}: "
            f"pos {census[-1]['pos_pct']}% neg {census[-1]['neg_pct']}% "
            f"abstain {census[-1]['abstain_pct']}% other {census[-1]['other_pct']}%")
    if census:
        pd.DataFrame(census).to_csv(root / "census_own_draw.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
