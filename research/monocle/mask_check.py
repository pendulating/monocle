"""Prove that an image-mask mode is applied, on 1 image, with a substitution.

A substitution of 1 cell changes no position. So the block-l residual of
another cell can change only through attention to the substituted cell:

| mode | cells before the swap | cells after the swap |
|---|---|---|
| bidirectional | change | change |
| causal | bit-exact | change |
| self | bit-exact | bit-exact |

The check prints the max abs change on each side, and the contrast `c` of
the unedited image under each mode. Run the modes in the given order: a mode
installs hooks that stay on the model.

    sbatch research/monocle/monocle.sub monocle.mask_check --case restaurants
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402
from research.monocle import patch_ablate as pab  # noqa: E402
from research.monocle import patch_causal as pc  # noqa: E402
from research.monocle import single_steer as ss  # noqa: E402
from research.monocle.jlens_steer import (  # noqa: E402
    load_model_and_lens,
    split_calibration_eval,
)


def log(m: str) -> None:
    print(f"[mask-check] {m}", flush=True)


def cell_resid(model, inputs: dict, block: int, slots: list[int]) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs, use_cache=False, logits_to_keep=1,
                    output_hidden_states=True)
    hs = out.hidden_states[block]
    return hs[0, slots, :].float().cpu()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=ss.CASE_DEFAULT)
    ap.add_argument("--kind", default=ss.KIND_DEFAULT)
    ap.add_argument("--modes", default="bidirectional,causal,self")
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--swap", type=int, default=128)
    ap.add_argument("--model-dir", default=str(MODEL_DIR_DEFAULT))
    ap.add_argument("--out", default="outputs/_monocle/mask_check.json")
    args = ap.parse_args(argv)

    proc, model, tmpl, lens_model, image_token_id = load_model_and_lens(
        args.model_dir)
    impl = model.config.get_text_config()._attn_implementation
    log(f"attention implementation: {impl}")
    tok_pos, tok_neg = ss.polar_token_ids(proc.tokenizer, args.case)
    paths = ss.load_images(args.case, args.kind)
    _, ev = split_calibration_eval([{"p": p} for p in paths], 16, 2)
    target, donor = ev[0]["p"], ev[1]["p"]
    build = ss.make_build_inputs(proc, tmpl, args.case)
    donor_pixels = build(donor)["pixel_values"][0]

    report = {}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        ss.install_image_mask(model, image_token_id, mode)
        inputs = build(target)
        slots = pab.slots_of(inputs, image_token_id)
        base = cell_resid(model, inputs, args.block, slots)
        swapped = pab.substitute_row(inputs, donor_pixels, [args.swap])
        edit = cell_resid(model, swapped, args.block, slots)
        diff = (edit - base).abs().max(dim=1).values.numpy()
        before = float(diff[:args.swap].max())
        after = float(diff[args.swap + 1:].max())
        own = float(diff[args.swap])
        c = float(pab.contrast(model, lens_model, inputs, tok_pos, tok_neg)[0])
        n_changed = int((diff > 0).sum())
        report[mode] = {"max_before": before, "max_after": after,
                        "own": own, "n_cells_changed": n_changed,
                        "contrast": c, "block": args.block}
        log(f"{mode:14s} block {args.block}: before={before:.4g} "
            f"after={after:.4g} own={own:.4g} changed={n_changed}/{len(slots)} "
            f"c={c:+.2f}")

    expect = {"bidirectional": (True, True), "causal": (False, True),
              "self": (False, False)}
    bad = []
    for mode, r in report.items():
        want = expect[mode]
        got = (r["max_before"] > 0, r["max_after"] > 0)
        if got != want:
            bad.append(f"{mode}: before/after change {got}, expected {want}")
    if bad:
        for b in bad:
            log(f"FAIL {b}")
    else:
        log("PASS: every mode moves exactly the cells its mask permits")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"case": args.case, "attn": impl, "swap": args.swap,
         "target": target, "donor": donor, "modes": report}, indent=2))
    log(f"wrote {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
