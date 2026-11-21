#!/usr/bin/env python
"""Write every figure and table of the 3 IC version 3 notebooks, without marimo.

The notebooks `master/ic_perspectives.py`, `master/ic_grounding.py`, and
`master/ic_differentiation.py` write their files when you press their export
button, 1 arm at a time. This script reads the 4 version 3 arms 1 at a time,
and it writes:

| Tag | Arms | What |
|-----|------|------|
| `ic_perspectives_<arm>_` | the 2 primary arms | question 1, with the agreement of the 2 extractors of the judge |
| `ic_grounding_<arm>_` | the 2 primary arms | question 2 |
| `ic_differentiation_<arm>_` | the 2 primary arms | questions 3 and 4 |
| `ic_differentiation_arms_` | all 4 arms | questions 3 and 4, side by side |
| `ic_grounding_arms_` | all 4 arms | the nesting shares, side by side |

With `--latex`, the script writes only the LaTeX tables, into
`master/outputs/` (or `--tex-dir`), and no figure or CSV file:

| File (`<tag>_<table>_v<notebook version>.tex`) | Arms |
|------------------------------------------------|------|
| `ic_perspectives_<arm>_ends`, `_signs` | the 2 primary arms |
| `ic_grounding_<arm>_nesting`, `_strongest_nests` | the 2 primary arms |
| `ic_differentiation_<arm>_pseudo_by_case`, `_dismissals` | the 2 primary arms |
| `ic_differentiation_arms_pseudo`, `ic_grounding_arms_nesting` | all 4 arms |

The version in each name is the `NOTEBOOK_VERSION` of the notebook that
shows the table.

Usage:
    python scripts/export_ic_v3_figures.py [--fig-dir DIR] [--min-count 25]
                                           [--allow-stale-extractions]
    python scripts/export_ic_v3_figures.py --latex [--tex-dir DIR]

Warning: the script reads about 3.4 GB of corpus in all, 1 arm at a time.
It takes about 2 minutes. Run it from the canonical venv.

See `vlm-narratives-docs/ic-v3-analysis.md`.
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
import _ic_v3 as I  # noqa: E402
import _style as S  # noqa: E402

__version__ = "1.1.0"

NOTEBOOKS = {"perspectives": "ic_perspectives", "grounding": "ic_grounding",
             "differentiation": "ic_differentiation"}


def notebook_version(name: str) -> str:
    """Read `NOTEBOOK_VERSION` from `master/<name>.py`, thus a file name of
    the script equals the file name of the notebook export."""
    text = (CVPR / "master" / f"{name}.py").read_text()
    m = re.search(r'NOTEBOOK_VERSION = "([^"]+)"', text)
    if not m:
        raise RuntimeError(f"no NOTEBOOK_VERSION in master/{name}.py")
    return m.group(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig-dir", type=Path,
                    default=CVPR / "master" / "figures",
                    help="the folder for the files (default: master/figures)")
    ap.add_argument("--min-count", type=int, default=25,
                    help="the count floor of a key or a pair")
    ap.add_argument("--n-ends", type=int, default=5,
                    help="the keys at each end of a case")
    ap.add_argument("--min-cases", type=int, default=3,
                    help="the fewest cases for a key that holds its sign")
    ap.add_argument("--latex", action="store_true",
                    help="write only the LaTeX tables, into --tex-dir")
    ap.add_argument("--tex-dir", type=Path,
                    default=CVPR / "master" / "outputs",
                    help="the folder for the LaTeX tables (default: master/outputs)")
    ap.add_argument("--allow-stale-extractions", action="store_true",
                    help="read a corpus the registry does not name. For a "
                         "look only; every file takes a `_stale` suffix.")
    args = ap.parse_args()

    gate_ok = not args.allow_stale_extractions
    if gate_ok:
        C.verify_or_raise()
    S.apply_house_style()

    arms = I.present_arms()
    primaries = [a for a in I.primary_arms() if a in arms]
    print(f"arms on disk: {arms}; primary: {primaries}", flush=True)

    bundles = {}
    for arm in arms:
        start = time.time()
        df = I.load_arm(arm, require_canonical=gate_ok)
        # A control arm needs question 1 for the agreement table, and
        # questions 2 to 4 for the tables side by side.
        bundles[arm] = I.arm_bundle(df, arm, min_count=args.min_count)
        del df
        print(f"{arm}: {time.time() - start:.0f} s", flush=True)

    if args.latex:
        if "figures" in args.tex_dir.resolve().parts:
            raise SystemExit("--tex-dir must not be a figures folder")
        ver = {k: notebook_version(v) for k, v in NOTEBOOKS.items()}
        written = []
        for arm in primaries:
            written += I.export_perspectives_latex(
                bundles[arm], args.tex_dir, arm, ver["perspectives"],
                gate_ok=gate_ok, min_cases=args.min_cases)
            written += I.export_grounding_latex(
                bundles[arm], args.tex_dir, arm, ver["grounding"], gate_ok=gate_ok)
            written += I.export_differentiation_latex(
                bundles[arm], args.tex_dir, arm, ver["differentiation"],
                gate_ok=gate_ok)
        written += I.export_across_arms_latex(
            bundles, args.tex_dir, ver["differentiation"], ver["grounding"],
            gate_ok=gate_ok)
        print(f"wrote {len(written)} files to {args.tex_dir}")
        for name in written:
            print(f"  {name}")
        return 0

    written = []
    for arm in primaries:
        sib = I.sibling_arm(arm)
        written += I.export_perspectives(
            bundles[arm], args.fig_dir, arm, gate_ok=gate_ok,
            n_ends=args.n_ends, min_cases=args.min_cases,
            sibling=bundles.get(sib) if sib else None)
        written += I.export_grounding(bundles[arm], args.fig_dir, arm,
                                      gate_ok=gate_ok)
        written += I.export_differentiation(bundles[arm], args.fig_dir, arm,
                                            gate_ok=gate_ok)
    written += I.export_across_arms(bundles, args.fig_dir, gate_ok=gate_ok)
    print(f"wrote {len(written)} files to {args.fig_dir}")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
