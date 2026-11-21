#!/usr/bin/env python
"""Write the prompt-ablation tables and figures, without marimo.

`notebooks/cvpr/master/ablations.py` writes its files when you press a
button. A paper table must be reproducible from a command, thus this script
runs the same calls and writes the same files to `master/outputs/`.

It holds the same gate as the notebook: the canonical registry must match the
disk (`_canonical.verify_or_raise`). Section 4 of the notebook also reads the
proxy parquets that the case notebooks exported, thus run those first.

Usage:
    python scripts/export_cvpr_ablations.py
    python scripts/export_cvpr_ablations.py --layers community_district nta
    python scripts/export_cvpr_ablations.py --skip proxy_arms

Warning: run it from the canonical venv, `.venv-mllmsci-vllm025cu129`. The
proxy re-run reads the 1,100,000-row parquets of 4 arms and takes minutes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR = REPO_ROOT / "notebooks" / "cvpr"
sys.path.insert(0, str(CVPR))

import _ablations as A  # noqa: E402
import _canonical as C  # noqa: E402
import _proxy_arms as PA  # noqa: E402
import _quantile_shift as Q  # noqa: E402
import _results_table as R  # noqa: E402
import _style as style  # noqa: E402

__version__ = "1.0.0"

OUT_DIR = CVPR / "master" / "outputs"
NOTEBOOK_VERSION = "1.0.0"

SECTIONS = ("ablations", "quantile_shift", "proxy_arms")


def _stamp(t0: float) -> str:
    return f"{time.time() - t0:6.0f}s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", nargs="*", default=[R.DEFAULT_LAYER],
                    help="the geography layers of the proxy re-run")
    ap.add_argument("--skip", nargs="*", default=[], choices=SECTIONS,
                    help="sections to leave out")
    args = ap.parse_args()
    t0 = time.time()

    print("[ablations] gate: the canonical registry")
    C.verify_or_raise()
    print(f"[ablations] {C.summary()}")
    style.apply_house_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    # Section 1 and 2: the ablation table and the abstention flow. Both read
    # the comparisons, thus they carry no layer.
    if "ablations" not in args.skip:
        ablations = A.all_arms()
        name = f"ablations_v{NOTEBOOK_VERSION}.csv"
        ablations.to_csv(OUT_DIR / name, index=False)
        written.append(name)
        flow = A.abstention_flow_table()
        name = f"ablation_abstention_flow_v{NOTEBOOK_VERSION}.csv"
        flow.to_csv(OUT_DIR / name, index=False)
        written.append(name)
        print(f"[ablations] {_stamp(t0)} sections 1-2: {len(ablations)} arm rows, "
              f"{len(flow)} flow rows")

    # Section 3: the quantile shift of the place scores. No layer either.
    if "quantile_shift" not in args.skip:
        qshift = Q.shift_table()
        qcurves = Q.delta_curves()
        q = Q.export(qshift, qcurves, OUT_DIR, version=NOTEBOOK_VERSION,
                     skipped_rows=Q.skipped())
        written += q
        print(f"[ablations] {_stamp(t0)} section 3: {len(qshift)} quantile rows, "
              f"{len(q)} files")

    # Section 4: the proxy re-run on every arm, for each layer.
    if "proxy_arms" not in args.skip:
        for layer in args.layers:
            arms_long = PA.build(layer=layer)
            arms_drift = PA.drift(arms_long)
            a = PA.export(arms_long, arms_drift, OUT_DIR,
                          version=NOTEBOOK_VERSION, layer=layer)
            written += a
            weighted = PA.weighted_tau_table(layer=layer)
            name = f"proxy_weighted_tau_{layer}_v{NOTEBOOK_VERSION}.csv"
            weighted.to_csv(OUT_DIR / name, index=False)
            written.append(name)
            held = PA.held_table(layer=layer)
            name = f"held_pairs_{layer}_v{NOTEBOOK_VERSION}.csv"
            held.to_csv(OUT_DIR / name, index=False)
            written.append(name)
            print(f"[ablations] {_stamp(t0)} section 4, {layer}: "
                  f"{len(arms_drift)} drift rows, {len(a) + 2} files")

    print(f"[ablations] {len(written)} files -> {OUT_DIR.relative_to(REPO_ROOT)}")
    for w in written:
        print(f"          {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
