#!/usr/bin/env python
"""Write every figure and table of the CVPR trace notebooks, without marimo.

A trace notebook writes its files when you press "Write the PNG and CSV files".
That is fine for one case, but it cannot regenerate the whole set after a
change to `_traces.py`, `_extractions.py`, or `_style.py`. This script runs the
same 2 export calls the button runs, for each case, and puts the files in that
prompt's own `figures/` folder.

Usage:
    python scripts/export_cvpr_trace_figures.py [--cases schools libraries]
                                                [--mode distinctive]
                                                [--unit class_text]
                                                [--min-count 25]
                                                [--unit-photos 6]

Warning: run it from the canonical venv. `.venv-3.12` has no marimo, no
wordcloud, and no geopandas.

See `notebooks/cvpr/README.md`.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR = REPO_ROOT / "notebooks" / "cvpr"
sys.path.insert(0, str(CVPR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import _canonical as C  # noqa: E402
import _extractions as X  # noqa: E402
import _gen_trace_notebooks as G  # noqa: E402
import _provenance as P  # noqa: E402
import _style as S  # noqa: E402
import _trace_notebook as TN  # noqa: E402
import _traces as T  # noqa: E402

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="*", default=list(G.CASES))
    ap.add_argument("--mode", default="distinctive", choices=["distinctive", "frequency"])
    ap.add_argument("--unit", default="class_text", choices=list(X.UNITS))
    ap.add_argument("--min-count", type=int, default=25)
    ap.add_argument("--unit-photos", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="read W&B again")
    ap.add_argument("--allow-stale-extractions", action="store_true",
                    help="use an extraction corpus from runs that are not "
                         "registered. For a look only, never for the paper.")
    args = ap.parse_args()

    S.apply_house_style()

    # The gate. A figure must come from the registered runs, thus stop here
    # when the registry does not match the disk.
    print("[export] gate: the canonical registry")
    C.verify_or_raise()
    print(f"[export] {C.summary()}")

    print("[export] discovery")
    runs = TN.discover(min_date=P.CONSOLIDATION_DATE, refresh=args.refresh)
    print(f"[export] {len(runs)} trace runs, cases: "
          f"{sorted({r.case for r in runs})}")

    print("[export] word counts (the first pass costs some minutes)")
    t0 = time.time()
    stopwords = T.default_stopwords()
    counts, _pairs, _n_runs, mixed = T.counts_by_group(runs, stopwords, ngram=1)
    print(f"[export] counted in {time.time() - t0:.0f}s; "
          f"{len(counts)} groups")
    if mixed:
        print(f"[export] WARNING: a case asked more than 1 question: {mixed}")

    # The extraction corpus is optional: a case can have traces and no
    # extraction yet. It is also 1 step DOWNSTREAM of a trace run, thus it goes
    # stale when the battery runs again. `X.load` refuses a corpus that does
    # not come from the registered runs, and the figures then hold the words
    # alone. That is the correct outcome: a class-rate panel from an older
    # prompt beside a word cloud from the new one is unreadable.
    try:
        ext = X.load(require_canonical=not args.allow_stale_extractions)
        print(f"[export] {len(ext):,} quotable extractions over "
              f"{int(X.trace_totals().sum()):,} traces")
    except FileNotFoundError as exc:
        ext = None
        print(f"[export] no extraction data ({exc}); the word figures only")
    except RuntimeError as exc:
        ext = None
        print(f"[export] STALE EXTRACTIONS, the panels are skipped:\n{exc}")

    written_total = 0
    for case in args.cases:
        title, folder = G.CASES.get(case, (case, case))
        out_dir = CVPR / folder / "figures"
        groups = TN.groups_for_case(counts, case)
        if not groups:
            print(f"[export] {case}: no trace run, skipped")
            continue

        written = TN.export(case, counts, groups, runs, out_dir, mode=args.mode)
        if ext is not None and (ext.case == case).any():
            written += X.export(
                case, ext, out_dir, unit=args.unit,
                min_count=args.min_count, unit_photos=args.unit_photos,
            )
        written_total += len(written)
        print(f"[export] {case}: {len(written)} files -> {out_dir.relative_to(REPO_ROOT)}")
        for name in written:
            print(f"           {name}")

    print(f"[export] {written_total} files in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
