"""Compare two fitted Jacobian lenses on the same image(s).

Quantifies how corpus-dependent the transport is — the question the workspace
paper doesn't examine. For each fitted layer, on the same forward pass
(activations are recorded once; only the J_l transport differs):

  - topk_jaccard: per-patch top-k token overlap between the two lenses'
    readouts (Jaccard, averaged over patches). 1.0 = identical top-k.
  - js_div: Jensen-Shannon divergence between the per-patch readout
    distributions, averaged over patches (natural log; max ln 2 ~ 0.693).
  - dog/red quadrant profiles per lens on the synthetic quadrant image.

The final layer is lens-independent by construction (out.logits verbatim) and
is included as a zero-line sanity row.

Usage:
    sbatch monocle/monocle.sub monocle.jlens_compare \
        --lens-a outputs/_monocle/jlens/gemma4_12b_lens.pt \
        --lens-b outputs/_monocle/jlens/urban/gemma4_12b_lens.pt \
        [--image PATH] [--k 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402
from monocle import extract, jlens_read, validate  # noqa: E402
# Single source of truth for the per-patch overlap/divergence primitives; the
# emergence maps and this comparator must not drift. These are the mean-over-
# patch scalars (thin wrappers over emergence.*_per_patch).
from monocle.emergence import js_div, topk_jaccard  # noqa: E402,F401


def log(msg: str) -> None:
    print(f"[jlens-cmp] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens-a", required=True, help="Reference lens .pt")
    ap.add_argument("--lens-b", required=True, help="Comparison lens .pt")
    ap.add_argument("--image", default=validate.DEFAULT_IMAGE)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out", default=str(REPO / "outputs/_monocle/jlens/compare_report.json"))
    args = ap.parse_args()

    lens_a = jlens_read.load_lens(args.lens_a)
    lens_b = jlens_read.load_lens(args.lens_b)
    shared = sorted(set(lens_a.source_layers) & set(lens_b.source_layers))
    if not shared:
        log("FATAL: lenses share no fitted layers")
        return 1
    log(f"A={args.lens_a} ({lens_a.n_prompts} prompts)")
    log(f"B={args.lens_b} ({lens_b.n_prompts} prompts)")
    log(f"shared layers: {shared}")

    proc, model, tmpl = extract.load_model(args.model_dir)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    tokenizer = proc.tokenizer
    report: dict = {"lens_a": args.lens_a, "lens_b": args.lens_b,
                    "image": args.image, "k": args.k, "layers": {}}

    dog_ids = validate.vocab_ids_exact(tokenizer, "dog")
    red_ids = validate.vocab_ids_exact(tokenizer, "red")

    for tag, image in (
        ("real", extract.open_image(args.image)),
        ("synthetic", validate.synthetic_quadrant_image()),
    ):
        inputs = extract.build_inputs(proc, tmpl, image)
        activations, positions, final_logits = jlens_read.record_patch_activations(
            model, lens_model, inputs, shared)
        grid = extract.infer_grid(inputs, positions.numel(), image.size)
        per_a = jlens_read.lens_patch_logits(
            lens_a, lens_model, activations, positions, final_logits, shared)
        per_b = jlens_read.lens_patch_logits(
            lens_b, lens_model, activations, positions, final_logits, shared)

        log(f"--- {tag} ({grid.n_rows}x{grid.n_cols}) ---")
        log(f"  {'layer':>5} {'jacc@'+str(args.k):>8} {'JS':>7}"
            + ("  dogTL(A/B)  redR(A/B)" if tag == "synthetic" else ""))
        for layer in shared:
            row: dict = {
                "jaccard": topk_jaccard(per_a[layer], per_b[layer], args.k),
                "js": js_div(per_a[layer], per_b[layer]),
            }
            extra = ""
            if tag == "synthetic":
                for name, lens_logits in (("a", per_a), ("b", per_b)):
                    p = torch.softmax(lens_logits[layer], dim=-1)
                    q_dog = validate.quadrant_mass(p, grid, dog_ids)
                    q_red = validate.quadrant_mass(p, grid, red_ids)
                    row[f"dog_tl_{name}"] = q_dog["TL"]
                    row[f"red_right_{name}"] = q_red["TR"] + q_red["BR"]
                extra = (f"  {row['dog_tl_a']:.2f}/{row['dog_tl_b']:.2f}"
                         f"   {row['red_right_a']:.2f}/{row['red_right_b']:.2f}")
            log(f"  L{layer:>4} {row['jaccard']:8.3f} {row['js']:7.4f}{extra}")
            report["layers"].setdefault(str(layer), {})[tag] = row

    Path(args.out).write_text(json.dumps(report, indent=2))
    log(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
