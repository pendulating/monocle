"""Check that a merged J-lens file is usable.

It reports the layer count, the image count, and any layer that holds a
value which is not finite. It also reports the diagonal mass by depth: a
block near the output changes the residual less, so its Jacobian must look
more like the identity than an early block does.

Usage:

    python research/scripts/jlens_verify_lens.py outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt
"""
from __future__ import annotations

import sys

import torch

from jlens import JacobianLens                       # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: jlens_verify_lens.py LENS.pt")
        return 2
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 47
    lens = JacobianLens.load(path)
    layers = sorted(lens.source_layers)
    bad = [l for l in layers if not torch.isfinite(lens.jacobians[l]).all()]
    zero = [l for l in layers if float(lens.jacobians[l].abs().max()) == 0.0]
    print(f"layers   {len(layers)} (want {want})")
    print(f"images   {lens.n_prompts}")
    print(f"d_model  {lens.d_model}")
    print(f"non-finite layers {bad or 'none'}")
    print(f"all-zero layers   {zero or 'none'}")
    print("\ndiagonal mass by depth")
    for l in layers[::8] + [layers[-1]]:
        J = lens.jacobians[l].float()
        print(f"  L{l:<3} {float(torch.diagonal(J).abs().mean()):.4f}")
    ok = len(layers) == want and not bad and not zero
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
