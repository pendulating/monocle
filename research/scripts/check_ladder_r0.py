"""Harness check: R0 must reproduce the lens it copies.

R0 uses the same 128 images, the same describe-contexts, and the same target
span as `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt`. So the merged R0 lens
must match it. A mismatch means the ladder harness changed something, and
every rung after it would be uninterpretable.

bfloat16 forwards are not bit-reproducible, so we report a relative
difference and judge it, rather than test equality.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

import torch  # noqa: E402
from jlens import JacobianLens  # noqa: E402

#: Above this relative difference the harness is not a copy of the mm fit.
TOLERANCE = 0.02


def main() -> int:
    a_path = REPO / "outputs/_monocle/jlens/ladder/R0/gemma4_12b_lens.pt"
    b_path = REPO / "outputs/_monocle/jlens/mm/gemma4_12b_lens.pt"
    for p in (a_path, b_path):
        if not p.exists():
            print(f"[check-r0] missing {p}")
            return 1
    a, b = JacobianLens.load(str(a_path)), JacobianLens.load(str(b_path))
    print(f"[check-r0] R0 n_prompts={a.n_prompts}  mm n_prompts={b.n_prompts}")
    if a.n_prompts != b.n_prompts:
        print("[check-r0] WARNING: the 2 fits saw a different image count")

    worst = 0.0
    print(f"[check-r0] {'layer':>5} {'rel diff':>10} {'cos':>8}")
    for layer in sorted(set(a.jacobians) & set(b.jacobians)):
        # float64: a float32 reduction over 14.7M elements loses enough
        # precision here to report a cosine above 1.0, which is impossible.
        # Gemma-4's massive activations widen the value range and make it
        # worse. The relative difference is a norm ratio and stays sound
        # either way; the cosine is the fragile one.
        x = a.jacobians[layer].double()
        y = b.jacobians[layer].double()
        rel = float((x - y).norm() / y.norm())
        cos = float(torch.nn.functional.cosine_similarity(
            x.flatten(), y.flatten(), dim=0))
        worst = max(worst, rel)
        print(f"[check-r0] {layer:>5} {rel:>10.4f} {cos:>8.4f}")
    ok = worst <= TOLERANCE
    print(f"[check-r0] worst rel diff {worst:.4f} -- "
          f"{'harness ok' if ok else 'HARNESS MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
