"""PREVIEW the word figures of a thinking corpus that is not registered yet.

Why this exists
---------------
`research/scripts/export_cvpr_trace_figures.py` gates on the canonical registry, and it
must: a paper figure comes from a registered run. `register` REPLACES the whole
registry, thus it cannot record the new trace half while the proxy half still
runs.

This script reads a merged sweep directory straight from disk and writes the
word figures somewhere else. Use it to LOOK at the words of a new corpus. The
figures it writes are NOT paper figures — run the real exporter after
registration, which writes into `research/notebooks/cvpr/<folder>/figures/`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "notebooks" / "cvpr"))

import _gen_trace_notebooks as G  # noqa: E402
import _style as S  # noqa: E402
import _trace_notebook as TN  # noqa: E402
import _traces as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", required=True,
                    help="The `merged` directory that merge_pairwise_shards wrote.")
    ap.add_argument("--out-root", required=True, help="Where the preview goes.")
    ap.add_argument("--cases", nargs="*", default=list(G.CASES))
    ap.add_argument("--mode", default="distinctive",
                    choices=["distinctive", "frequency"])
    args = ap.parse_args()

    S.apply_house_style()
    runs = [r for r in T.scan_sweep_dir(args.merged_dir, require_trace=True)
            if r.is_readable]
    print(f"[preview] {len(runs)} trace runs from {args.merged_dir}")
    for r in sorted(runs, key=lambda x: x.case):
        print(f"    {r.case:<20} rows={r.rows} traced={r.traced_rows}")
    if not runs:
        print("[preview] nothing to read")
        return 1

    print("[preview] word counts (the first pass costs some minutes)")
    t0 = time.time()
    stopwords = T.default_stopwords()
    counts, _pairs, _n_runs, mixed = T.counts_by_group(runs, stopwords, ngram=1)
    print(f"[preview] counted in {time.time() - t0:.0f}s; {len(counts)} groups")
    if mixed:
        print(f"[preview] WARNING a case asked more than 1 question: {mixed}")

    out_root = Path(args.out_root)
    total = 0
    for case in args.cases:
        groups = TN.groups_for_case(counts, case)
        if not groups:
            print(f"[preview] {case}: no trace run, skipped")
            continue
        out_dir = out_root / case
        out_dir.mkdir(parents=True, exist_ok=True)
        written = TN.export(case, counts, groups, runs, out_dir, mode=args.mode)
        total += len(written)
        print(f"[preview] {case}: {len(written)} files -> {out_dir}")
    print(f"[preview] {total} files. These are NOT paper figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
