#!/usr/bin/env python
"""Write the Integrative Complexity tables and figures, without marimo.

`notebooks/cvpr/master/ic_complexity.py` shows the same numbers in a browser.
This script writes them from a command, so a figure in the paper can be made
again without a person in the loop.

It holds the same 2 gates as the other exports:

1. The canonical registry matches the disk.
2. The ingredient corpus comes from the REGISTERED thinking runs. A corpus of
   an older battery describes older prompts, and no figure may carry it.

Usage:
    python scripts/export_cvpr_ic_figures.py
    python scripts/export_cvpr_ic_figures.py --cases schools subway_safety
    python scripts/export_cvpr_ic_figures.py --supporting-quotes 3

Warning: run it from the canonical venv, `.venv-mllmsci-vllm025cu129`.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR = REPO_ROOT / "notebooks" / "cvpr"
sys.path.insert(0, str(CVPR))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import _canonical as C  # noqa: E402
import _ic  # noqa: E402
import _ic_link as LK  # noqa: E402
import _ic_words as W  # noqa: E402
import _style as S  # noqa: E402

from dagspaces.common import ic_codes as IC  # noqa: E402

__version__ = "1.0.0"

OUT_DIR = CVPR / "master" / "figures"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--supporting-quotes", type=int, default=IC.DEFAULT.supporting_quotes,
                    help="located spans that make a perspective 'developed'")
    ap.add_argument("--distinct-dimensions", type=int,
                    default=IC.DEFAULT.distinct_dimensions)
    ap.add_argument("--no-words", action="store_true",
                    help="skip the word blocks over the spans")
    ap.add_argument("--words-mode", default="distinctive",
                    choices=["distinctive", "frequency"])
    ap.add_argument("--words-type", default=W.DEFAULT_CASE_TYPE,
                    choices=list(W.TYPE_ORDER),
                    help="the ingredient type the per-case blocks draw")
    ap.add_argument("--max-words", type=int, default=120)
    ap.add_argument("--min-count", type=int, default=10,
                    help="a word under this count cannot enter a block")
    ap.add_argument("--layer", default="community_district",
                    choices=["community_district", "nta", "census_tract"],
                    help="the polygon layer the proxy link joins on")
    ap.add_argument("--split-code", type=int, default=5,
                    help="a trace counts as complex at this code or above")
    ap.add_argument("--bands", type=int, default=4,
                    help="difficulty bands, by the size of the proxy gap")
    ap.add_argument("--no-link", action="store_true",
                    help="skip the link to the proxy")
    ap.add_argument("--keep-truncated", action="store_true",
                    help="keep a trace whose answer the token cap cut")
    args = ap.parse_args()

    S.apply_house_style()

    print("[ic] gate 1: the canonical registry")
    C.verify_or_raise()
    print(f"[ic] {C.summary()}")

    print("[ic] gate 2: the corpus comes from the registered trace runs")
    try:
        raw = _ic.load(cases=args.cases)
    except FileNotFoundError as exc:
        print(f"[ic] {exc}")
        return 1
    except RuntimeError as exc:
        print(f"[ic] REFUSED:\n{exc}")
        return 1
    print(f"[ic] {len(raw):,} ingredient rows over {raw.doc_id.nunique():,} traces, "
          f"cases: {', '.join(sorted(raw.case.unique()))}")

    thresholds = IC.Thresholds(
        supporting_quotes=args.supporting_quotes,
        distinct_dimensions=args.distinct_dimensions,
    )
    codes = _ic.codes(raw, thresholds)

    print("\n" + _ic.quality_table(raw, codes).to_string(index=False))
    print("\n" + _ic.case_table(
        codes, drop_truncated=not args.keep_truncated).to_string(index=False))

    written = _ic.export(raw, codes, Path(args.out), thresholds)

    # The word blocks over the located spans. 1 block for each ingredient type
    # over the whole battery, and 1 block for each case inside the dimensions.
    if not args.no_words:
        print("\n[ic] the word blocks")
        written += W.export(raw, Path(args.out), mode=args.words_mode,
                            max_words=args.max_words, min_count=args.min_count,
                            ingredient_type=args.words_type)

    # The link to the proxy. It runs only for the 5 unit-mode cases: a pair of
    # random images holds no unit to place in a polygon.
    if not args.no_link:
        print("\n[ic] the link to the proxy, at the "
              f"{args.layer} layer")
        summary, tables = LK.link_all(codes, layer=args.layer,
                                      split_code=args.split_code,
                                      quantiles=args.bands)
        if summary.empty:
            print("[ic] no case reached the test; is the proxy exported?")
        else:
            print(summary.to_string(index=False))
            written += LK.export(summary, tables, Path(args.out),
                                 split_code=args.split_code, quantiles=args.bands)

    print(f"\n[ic] {len(written)} files -> {Path(args.out).relative_to(REPO_ROOT)}")
    for name in written:
        print(f"       {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
