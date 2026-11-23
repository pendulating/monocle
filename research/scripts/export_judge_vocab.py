"""Write the judge-vocabulary figures, tables, and CSVs without marimo.

The script reads the 4 v3 arms (2 judges x 2 extractors) with the paper
filters, runs `_judge_vocab.build_bundle`, and writes the files of
`research/notebooks/cvpr/master/judge_vocabulary.py` into
`research/notebooks/cvpr/master/outputs/judge_vocab/`.

    .venv-mllmsci-vllm025cu129/bin/python research/scripts/export_judge_vocab.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from research.notebooks.cvpr import _canonical as C  # noqa: E402
from research.notebooks.cvpr import _judge_vocab as J  # noqa: E402
from research.notebooks.cvpr import _style as style  # noqa: E402

#: Keep this equal to `NOTEBOOK_VERSION` of the notebook.
NOTEBOOK_VERSION = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-traces", type=int, default=10)
    ap.add_argument("--min-cases", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=J.OUT_DIR)
    args = ap.parse_args()

    C.verify_or_raise()
    style.apply_house_style()
    docs = J.load_grid(require_canonical=True)
    bundle = J.build_bundle(docs, min_traces=args.min_traces,
                            min_cases=args.min_cases)
    written = J.export(bundle, args.out_dir, version=NOTEBOOK_VERSION,
                       gate_ok=True, k=args.top_k, n_tex=5,
                       min_traces=args.min_traces)
    print(f"Wrote {len(written)} files to {args.out_dir}:")
    for name in written:
        print(f"  {name}")
    print(bundle["summary"].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
