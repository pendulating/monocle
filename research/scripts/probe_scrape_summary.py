"""Summarize the scrape of the lens-refusal probe (E1b, E5c) on the CPU.

Reads `outputs/_monocle/probe/scrape/<probe>/{tokens,label_ranks}.parquet`
and writes `scrape_summary.json` and `scrape.md` beside them:

| Table | Content |
|-------|---------|
| patch words | the top-3 words at the patches, blocks 18, 24, 36, and 42 under `mm`, refused against answered, and the words that differ most between the 2 splits |
| question words | the top-5 words at the question tokens under C at blocks 30, 38, and 42; at 30 and 42 the share of them also in the top-5 under B |
| label ranks | at the answer position, the median rank of the pos, neg, and abstain tokens by block and split, and the share of images where the NEG pole ranks above the POS pole |

The rank trajectory is the legible form of H1: a pole that leads at the
answer position in the blocks before the read, and an abstain token that
overtakes it in the last blocks, is a judgment the output withholds.

    .venv-3.12/bin/python research/scripts/probe_scrape_summary.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
sys.path.insert(0, str(REPO))
from research.monocle.probe_frame import ABSTAIN, PROBE_ORDER, PROBES  # noqa: E402

ROOT = REPO / "outputs/_monocle/probe/scrape"
SKIP = {"<pad>", "<|image>", "<|channel>", "<channel|>", "<turn|>", "<|turn>", ""}


def words(df: pd.DataFrame, k: int) -> Counter:
    d = df[df["rank"] < k]
    return Counter(t for t in d["token"].str.strip().str.lower() if t not in SKIP)


def top(c: Counter, n: int = 20) -> list[tuple[str, int]]:
    return c.most_common(n)


def diff(a: Counter, b: Counter, n: int = 15) -> list[tuple[str, int]]:
    """Words that the first split holds more often than the second, per
    image count (the 2 splits hold the same number of images)."""
    keys = set(a) | set(b)
    d = {k: a.get(k, 0) - b.get(k, 0) for k in keys}
    return sorted(d.items(), key=lambda kv: -kv[1])[:n]


def summarize(probe: str) -> tuple[dict, str]:
    d = ROOT / probe
    toks = pd.read_parquet(d / "tokens.parquet")
    rk = pd.read_parquet(d / "label_ranks.parquet")
    p = PROBES[probe]
    n_img = {s: int(toks[toks.split == s]["image_path"].nunique()) for s in ("refused", "answered")}
    out: dict = {"probe": probe, "images": n_img}
    md = [f"# Scrape: {probe} ({p.pos}/{p.neg}/{ABSTAIN})", "",
          f"Images: {n_img['refused']} refused, {n_img['answered']} answered.", ""]

    out["patch_words"] = {}
    for L in (18, 24, 36, 42):
        pat = toks[(toks.group == "image") & (toks.layer == L) & (toks.lens == "mm")]
        if pat.empty:
            continue
        cr, ca = words(pat[pat.split == "refused"], 3), words(pat[pat.split == "answered"], 3)
        out["patch_words"][str(L)] = {"refused": top(cr), "answered": top(ca),
                                      "more_in_refused": diff(cr, ca), "more_in_answered": diff(ca, cr)}
        md += [f"## Patch words, block {L} (lens mm, top 3 per patch)", "",
               "| Split | Words |", "|---|---|",
               f"| refused | {', '.join(f'{t} ({n})' for t, n in top(cr))} |",
               f"| answered | {', '.join(f'{t} ({n})' for t, n in top(ca))} |",
               f"| more in refused | {', '.join(f'{t} (+{n})' for t, n in diff(cr, ca))} |",
               f"| more in answered | {', '.join(f'{t} (+{n})' for t, n in diff(ca, cr))} |", ""]

    out["question_words"] = {}
    for L in (30, 38, 42):
        q = toks[(toks.group == "text_after") & (toks.layer == L)]
        if q.empty:
            continue
        rec = {}
        lines = [f"## Question-token words, block {L} (lens C, top 5 per position)", "",
                 "| Split | Words | Also top-5 under B |", "|---|---|---|"]
        for s in ("refused", "answered"):
            qc = q[(q.split == s) & (q.lens == "C")]
            qb = q[(q.split == s) & (q.lens == "B")]
            cw = words(qc, 5)
            bw = set(words(qb, 5)) if not qb.empty else None
            shared = [t for t, _ in top(cw, 30) if bw is not None and t in bw]
            rec[s] = {"top_C": top(cw, 30), "shared_with_B": shared,
                      "share_shared": (len(shared) / 30) if bw is not None else None}
            lines.append(f"| {s} | {', '.join(f'{t} ({n})' for t, n in top(cw, 20))} | "
                         f"{', '.join(shared) if bw is not None else 'B holds no block ' + str(L)} |")
        out["question_words"][str(L)] = rec
        md += lines + [""]

    ar = rk[(rk.group == "answer") & (rk.lens == "C")].copy()
    ar["neg_leads"] = ar["rank_neg"] < ar["rank_pos"]
    ar["abstain_leads"] = (ar["rank_abstain"] < ar["rank_pos"]) & (ar["rank_abstain"] < ar["rank_neg"])
    g = ar.groupby(["split", "layer"]).agg(
        rank_pos=("rank_pos", "median"), rank_neg=("rank_neg", "median"),
        rank_abstain=("rank_abstain", "median"), neg_leads=("neg_leads", "mean"),
        abstain_leads=("abstain_leads", "mean"), n=("image_path", "nunique")).reset_index()
    out["answer_ranks"] = g.round(3).to_dict("records")
    md += ["## Label ranks at the answer position (lens C)", "",
           f"| Split | Block | rank {p.pos} | rank {p.neg} | rank {ABSTAIN} | {p.neg} above {p.pos} | {ABSTAIN} above both |",
           "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        md.append(f"| {r.split} | {int(r.layer)} | {r.rank_pos:.0f} | {r.rank_neg:.0f} | "
                  f"{r.rank_abstain:.0f} | {r.neg_leads:.0%} | {r.abstain_leads:.0%} |")
    md.append("")
    text = "\n".join(md)
    (d / "scrape_summary.json").write_text(json.dumps(out, indent=2))
    (d / "scrape.md").write_text(text)
    return out, text


def main(argv=None) -> int:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", nargs="+", default=list(PROBE_ORDER))
    ap.add_argument("--root", default=str(ROOT), help="the scrape directory")
    args = ap.parse_args(argv)
    ROOT = Path(args.root)
    for probe in args.probes:
        if not (ROOT / probe / "tokens.parquet").exists():
            print(f"[scrape-summary] {probe}: no tokens.parquet; skipped")
            continue
        _, text = summarize(probe)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
