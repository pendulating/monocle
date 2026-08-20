"""Validation for the Jacobian-lens readout path (monocle.jlens_read).

Two stages, gated on what exists:

  A. CONSISTENCY (no lens needed) — unembed(h_final) at patch positions must
     reproduce the model's own out.logits (max abs diff ~ dtype noise).
     Catches a wrong layer convention (hidden_states off-by-one) or a broken
     unembed path before any lens results are trusted.

  B. LENS (needs --lens CKPT) —
     1. Text sanity: on a factual-recall prompt, the J-lens top-5 at mid
        layers should surface the answer earlier/stronger than the plain
        logit lens (use_jacobian=False baseline), qualitatively replicating
        the paper's headline behaviour.
     2. Depth-resolved quadrant test: per fitted layer, exact-'dog' mass in
        TL and exact-'red' right-half mass on the synthetic quadrant image
        (same criteria as monocle.validate, per layer). Also reports the
        emergence profile — the first layer at which each criterion holds —
        which is itself the first science output.

Usage:
    sbatch monocle/monocle.sub monocle.jlens_validate                 # stage A
    sbatch monocle/monocle.sub monocle.jlens_validate --lens PATH     # A + B
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
from monocle import extract, jlens_read, scoring, validate  # noqa: E402

FACT_PROMPT = ("Fact: the capital of the state containing Dallas is")


def log(msg: str) -> None:
    print(f"[jlens-val] {msg}", flush=True)


def stage_a(proc, model, tmpl, lens_model, report: dict) -> bool:
    log("=" * 60)
    log("STAGE A: final-layer unembed consistency at patch positions")
    im = extract.open_image(validate.DEFAULT_IMAGE)
    inputs = extract.build_inputs(proc, tmpl, im)
    diff = jlens_read.final_layer_consistency(model, lens_model, inputs)
    # bf16 residual -> fp32 unembed vs the model's internal path; allow loose
    # numeric slack, catch structural errors (wrong layer/norm), which give O(1+)
    ok = diff < 0.1
    log(f"  max |direct - via_unembed| = {diff:.2e} -> {'OK' if ok else 'FAIL'}")
    report["stage_a"] = {"max_abs_diff": diff, "pass": ok}
    return ok


def stage_b_text(lens, lens_model, report: dict) -> None:
    log("=" * 60)
    log("STAGE B1: factual-recall text sanity (J-lens vs logit lens)")
    layers = lens.source_layers
    # wrap_for_unembed leaves add_bos_token=False (chat template supplies
    # <bos>); raw-text prompts need the attention-sink BOS spelled out.
    prompt = (lens_model.tokenizer.bos_token or "") + FACT_PROMPT
    for use_j, name in ((True, "jlens"), (False, "logit-lens")):
        lens_logits, model_logits, _ = lens.apply(
            lens_model, prompt, positions=[-1], use_jacobian=use_j,
            layers=layers if use_j else layers)
        tops = {}
        for layer, logits in sorted(lens_logits.items()):
            toks = [lens_model.tokenizer.decode([t])
                    for t in logits[0].topk(5).indices]
            tops[layer] = toks
            log(f"  {name} L{layer:02d}: {toks}")
        report.setdefault("stage_b_text", {})[name] = {
            str(k): v for k, v in tops.items()}
    final_top = lens_model.tokenizer.decode([int(model_logits[0].argmax())])
    log(f"  model final top-1: {final_top!r}")
    report["stage_b_text"]["model_top1"] = final_top


def stage_b_quadrant(proc, model, tmpl, lens, lens_model, report: dict) -> bool:
    log("=" * 60)
    log("STAGE B2: depth-resolved quadrant localization")
    tokenizer = proc.tokenizer
    dog_ids = validate.vocab_ids_exact(tokenizer, "dog")
    red_ids = validate.vocab_ids_exact(tokenizer, "red")
    syn = validate.synthetic_quadrant_image()
    per_layer, grid = jlens_read.lens_image_layers(
        proc, model, tmpl, lens, lens_model, syn)

    profile: dict[str, dict] = {}
    dog_first = red_first = None
    for layer in sorted(per_layer):
        p = torch.softmax(per_layer[layer], dim=-1)
        q_dog = validate.quadrant_mass(p, grid, dog_ids)
        q_red = validate.quadrant_mass(p, grid, red_ids)
        dog_ok = max(q_dog, key=lambda k: q_dog[k]) == "TL"
        red_right = q_red["TR"] + q_red["BR"]
        log(f"  L{layer:02d}: dog TL={q_dog['TL']:.2f} "
            f"{'OK  ' if dog_ok else 'no  '} red right={red_right:.2f}")
        profile[str(layer)] = {
            "dog": q_dog, "red": q_red, "dog_tl": dog_ok,
            "red_right": red_right}
        if dog_ok and dog_first is None:
            dog_first = layer
        if red_right >= 0.85 and red_first is None:
            red_first = layer
    final = max(per_layer)
    ok = profile[str(final)]["dog_tl"] and profile[str(final)]["red_right"] >= 0.85
    log(f"  emergence: dog TL first OK at L{dog_first}, "
        f"red right-half first >=0.85 at L{red_first}")
    log(f"  final layer criteria: {'OK' if ok else 'FAIL'}")
    report["stage_b_quadrant"] = {
        "profile": profile, "dog_first_layer": dog_first,
        "red_first_layer": red_first, "final_pass": ok}
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default=None, help="Fitted lens .pt (stage B).")
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/_monocle/jlens"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"lens": args.lens, "model_dir": args.model_dir}

    proc, model, tmpl = extract.load_model(args.model_dir)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)

    ok = stage_a(proc, model, tmpl, lens_model, report)
    if args.lens and ok:
        lens = jlens_read.load_lens(args.lens)
        log(f"lens: {lens!r}")
        stage_b_text(lens, lens_model, report)
        ok = stage_b_quadrant(proc, model, tmpl, lens, lens_model, report) and ok

    (out_dir / "validate_report.json").write_text(json.dumps(report, indent=2))
    log(f"report -> {out_dir / 'validate_report.json'}")
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
