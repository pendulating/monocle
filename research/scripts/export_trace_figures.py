#!/usr/bin/env python
"""Write the figures and tables of the 2 trace notebooks without the button.

`research/notebooks/cvpr/master/cue_hierarchy.py` and `master/trace_channels.py` write
their PNG and CSV files when a reader presses a `run_button`. `marimo export`
runs every cell but presses no button, thus a headless rerun writes nothing.
This script calls the same export functions with the notebooks' default
control values, so that a corpus change reaches `master/figures/` in 1 command.

The defaults mirror the notebooks on 2026-09-22:

| Control | Value |
|---------|-------|
| unit | `category_valence` |
| class | `visual_evidence` |
| fewest decided comparisons for a key | 25 |
| rungs in the ladder | 30 |
| fewest decided comparisons an axis needs | 400 |

Use:
    python research/scripts/export_trace_figures.py                 # both notebooks, the default root
    python research/scripts/export_trace_figures.py --only channels
    python research/scripts/export_trace_figures.py --root data/trace_extractions_v2_qwen9b_by_gemma31b --tag-suffix _qwen
    python research/scripts/export_trace_figures.py --only channels --latex

`--latex` writes only the LaTeX tables of `trace_channels` (the abstention
AUC of the 2 x 2) into `master/outputs/`. It writes no figure, and nothing
into `master/figures/`. The file names carry the `NOTEBOOK_VERSION` of
`master/trace_channels.py`.

A corpus that fails the registry gate writes every name with a `_stale`
suffix, as the notebooks do.

See `research/docs/langextract-trace-extraction.md`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent.parent / "research" / "notebooks" / "cvpr"

__version__ = "1.2.0"


def notebook_version(name: str) -> str:
    """Read `NOTEBOOK_VERSION` from a notebook in `master/`."""
    import re
    text = (NB / "master" / name).read_text()
    m = re.search(r'NOTEBOOK_VERSION = "([^"]+)"', text)
    return m.group(1) if m else "headless"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["cues", "channels"], default=None)
    ap.add_argument("--root", default=None, help="a corpus root; default `_extractions.DEFAULT_ROOT`")
    ap.add_argument("--fig-dir", default=str(NB / "master" / "figures"))
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--max-keys", type=int, default=30)
    ap.add_argument("--axis-floor", type=int, default=400)
    ap.add_argument("--latex", action="store_true",
                    help="write only the LaTeX tables of trace_channels into --tex-dir")
    ap.add_argument("--tex-dir", default=str(NB / "master" / "outputs"))
    ap.add_argument("--allow-stale", action="store_true",
                    help="read a corpus the registry does not name; the files take `_stale`")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _cues_across as CA
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _trace_channels as TC
    from research.notebooks.cvpr import _words_across as WA

    C.verify_or_raise()
    style.apply_house_style()

    if args.latex:
        # The tables read the 4 version 3 arms only. Nothing goes to figures.
        if args.only == "cues":
            raise SystemExit("--latex has tables for --only channels only")
        gate_ok = not args.allow_stale
        stats, _ = TC.abstention_across_arms(require_canonical=gate_ok)
        written = TC.export_latex(stats, Path(args.tex_dir),
                                  version=notebook_version("trace_channels.py"),
                                  gate_ok=gate_ok)
        for name in written:
            print("   ", name)
        return 0

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        corpus = CA.load(require_canonical=not args.allow_stale, root=args.root)
        gate_ok = not args.allow_stale
    except RuntimeError as exc:
        raise SystemExit(f"the corpus fails the registry gate:\n{exc}")
    print(f"[export] corpus: {len(corpus):,} rows, {corpus.case.nunique()} cases, "
          f"root {args.root or 'default'}, gate_ok={gate_ok}", flush=True)

    written = []
    if args.only in (None, "cues"):
        tables = CA.lift_tables(corpus, unit="category_valence",
                                classes=("visual_evidence",), min_count=args.min_count)
        matrix = CA.lift_matrix(tables)
        summary = CA.lift_summary(matrix, tables)
        rates = CA.rates_frame(corpus, unit="category_valence", classes=("visual_evidence",))
        rate_matrix, tops = WA.rate_matrix(rates, top_per_case=60, min_count=1)
        rate_summary = WA.summarise(rate_matrix, tops)
        w = CA.export(matrix, summary, tables, rate_summary, fig_dir,
                      max_keys=args.max_keys, gate_ok=gate_ok,
                      provenance=CA.provenance_table(corpus, gate_ok))
        w += CA.export_words(corpus, fig_dir, gate_ok=gate_ok)
        print(f"[export] cue_hierarchy: {len(w)} files", flush=True)
        written += w
    if args.only in (None, "channels"):
        # Each PNG gets a `.json` caption sidecar (`_figmeta.save_figure`).
        w = TC.export(corpus, fig_dir, axis_min_count=args.axis_floor,
                      version="headless", gate_ok=gate_ok,
                      arm=TC.arm_of_root(args.root))
        print(f"[export] trace_channels: {len(w)} files", flush=True)
        written += w
    for name in written:
        print("   ", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
