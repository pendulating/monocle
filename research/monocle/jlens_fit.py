"""Fit a Jacobian lens (sub/jacobian-lens) for gemma-4-12B's text decoder.

The lens transports a residual at layer l into the final-layer basis
(lens_l(h) = unembed(J_l @ h), J_l = E[dh_final/dh_l]) so monocle can read
what each image patch is "disposed to make the model say" at ANY depth, not
just the final layer. Fitting is text-only (the estimator backprops through
the bare language decoder); whether a text-fitted J transports image-patch
residuals faithfully is checked downstream by monocle.jlens_read's
final-layer-consistency and per-layer quadrant tests.

Cost model: ceil(d_model / dim_batch) backward passes per prompt — corpus
size and dim_batch dominate, NOT the number of source layers. Run --smoke
first to calibrate dim_batch on the actual GPU, then shard the real fit
across jobs and merge (JacobianLens.merge is n_prompts-weighted).

Usage (klara, .venv-nightly + LD_PRELOAD — see research/monocle/monocle.sub):
    sbatch research/monocle/monocle.sub monocle.jlens_fit --smoke
    sbatch research/monocle/monocle.sub monocle.jlens_fit --shard 0/4
    ... (shards 1/4..3/4) then:
    python -m research.monocle.jlens_fit --merge outputs/_monocle/jlens/shard*.pt
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import glob as globlib
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

from research.monocle import MODEL_DIR_DEFAULT  # noqa: E402

DEFAULT_CORPUS = str(REPO / "outputs/_monocle/jlens/corpus_wikitext_1k.json")
DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/jlens")
# Sources must be < target (default final layer, 47). start_graph_at=min
# bounds the retained autograd graph, so nothing below layer 6 is retained.
DEFAULT_SOURCE_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]

# The original grid is stride 6 from layer 6. The GLOBAL blocks sit at
# 5, 11, 17, 23, 29, 35, 41, 47 — also stride 6. Thus 7 of the 8 fitted
# layers land immediately AFTER a global block by accident, and the one that
# does not (L46) is the one whose readout is degenerate.
#
# Readability may therefore track the block type, not the depth. This band
# breaks the confound: it fits a global block's own output (41), several
# mid-sliding layers (37-40, 43-45), and keeps L42 and L46 for comparison.
#
# Extra source layers are nearly free — the cost is
# ceil(d_model / dim_batch) backward passes per PROMPT, not per layer.
DENSE_SOURCE_LAYERS = [30, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

GLOBAL_BLOCKS = [5, 11, 17, 23, 29, 35, 41, 47]

#: `jacobian` averages the raw gradient. `lrp` averages a relevance
#: coefficient instead, which makes an R-lens. See research/monocle/rlens_rules.py and
#: research/docs/r-lens-proposal.md.
RULE_CHOICES = ("jacobian", "lrp")


def log(msg: str) -> None:
    print(f"[jlens-fit] {msg}", flush=True)


def fit_date() -> str:
    """Today, as an ISO date. It goes into the lens meta."""
    return _datetime.date.today().isoformat()


def load_lens_model(model_dir: str, *, eager: bool = False):
    """Load gemma-4 via the verified monocle recipe and wrap it for jlens.

    from_hf auto-detects the model.language_model layout (verified on the
    meta device 2026-07-20: 48 layers, d_model 3840, softcap 30.0).

    `eager` is needed for an R fit only. SDPA hides the softmax inside a
    fused kernel, and the softmax rule cannot reach it there.

    Returns ``(lens_model, hf_model)``. The rules go on the HF model.
    """
    import jlens

    if eager:
        from research.monocle import attn_route as ar
        proc, model, _tmpl = ar.load_eager(model_dir, "cuda:0")
    else:
        from research.monocle import extract
        proc, model, _tmpl = extract.load_model(model_dir)
    return jlens.from_hf(model, proc.tokenizer), model


def rule_context(hf_model, rules: str):
    """The LRP rules for an R fit, and nothing at all for a J fit."""
    if rules == "lrp":
        from research.monocle import rlens_rules as rr
        return rr.lrp_rules(hf_model)
    return contextlib.nullcontext()


def parse_shard(spec: str) -> tuple[int, int]:
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise ValueError(f"bad shard spec {spec!r}")
    return i, n


def cmd_smoke(args: argparse.Namespace) -> int:
    """Calibrate dim_batch: time one prompt at each setting, report peak mem.

    The smoke uses the layer set the fit will use (`args.source_layers`, after
    `--dense` resolves). The retained graph roots at min(source_layers), so a
    different layer set gives a different memory peak.
    """
    from jlens.fitting import jacobian_for_prompt

    prompts = json.load(open(args.corpus))
    m, hf = load_lens_model(args.model_dir, eager=args.rules == "lrp")
    log(f"smoke sources={args.source_layers}, "
        f"estimate for {args.n_prompts} prompts")
    for dim_batch in (8, 16, 32):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with rule_context(hf, args.rules):
                _J, seq_len, n_valid = jacobian_for_prompt(
                    m, prompts[0], args.source_layers, dim_batch=dim_batch)
        except torch.cuda.OutOfMemoryError:
            log(f"dim_batch={dim_batch}: OOM")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"dim_batch={dim_batch}: {dt:.1f}s/prompt "
            f"(seq={seq_len}, valid={n_valid}), peak {peak:.1f} GiB, "
            f"est. {dt * args.n_prompts / 3600:.1f} h "
            f"per {args.n_prompts} prompts")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import jlens

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    prompts = json.load(open(args.corpus))[: args.n_prompts]
    shard_i, shard_n = parse_shard(args.shard)
    shard_prompts = prompts[shard_i::shard_n]
    log(f"shard {shard_i}/{shard_n}: {len(shard_prompts)} prompts, "
        f"dim_batch={args.dim_batch}, sources={args.source_layers}, "
        f"rules={args.rules}")
    for l in args.source_layers:
        kind = ("GLOBAL block" if l in GLOBAL_BLOCKS else
                "post-global" if (l - 1) in GLOBAL_BLOCKS else "mid-sliding")
        log(f"    L{l:<3} {kind}")

    m, hf = load_lens_model(args.model_dir, eager=args.rules == "lrp")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"fit_ckpt_shard{shard_i}of{shard_n}.pt"
    with rule_context(hf, args.rules) as report:
        if report is not None:
            log(f"LRP rules installed: {report}")
        lens = jlens.fit(
            m, shard_prompts,
            source_layers=args.source_layers,
            dim_batch=args.dim_batch,
            checkpoint_path=str(ckpt),
            checkpoint_every=5,
        )
        if report is not None:
            log(f"the softmax rule ran {report['softmax_calls']} times")
    out = out_dir / f"shard{shard_i}of{shard_n}.pt"
    # fp32, not the fp16 default: a shard is an input to the merge, and the
    # fp16 error does not cancel in the weighted mean. The R0 check measured a
    # 4e-4 residual that came from the fp16 shard saves alone.
    lens.meta.update({
        "rules": args.rules,
        "corpus": str(args.corpus),
        "n_prompts_requested": int(args.n_prompts),
        "shard": args.shard,
        "source_layers": [int(l) for l in args.source_layers],
        "dim_batch": int(args.dim_batch),
        "model_dir": str(args.model_dir),
        # The source is every valid position, and the target is the span.
        "estimator": "text_span",
        "skip_first": 16,
        "fit_date": fit_date(),
    })
    lens.save(str(out), dtype=torch.float32)
    log(f"saved {out} ({lens!r})")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from jlens import JacobianLens

    paths = sorted(p for pat in args.merge for p in globlib.glob(pat))
    lenses = [JacobianLens.load(p) for p in paths]
    log(f"merging {len(paths)} shards: {[Path(p).name for p in paths]}")
    merged = JacobianLens.merge(lenses)
    out_dir = Path(args.out_dir)
    if args.rescale_against:
        from research.monocle import rlens_rules as rr
        ref = JacobianLens.load(args.rescale_against)
        merged.jacobians, scales = rr.match_scale(
            merged.jacobians, ref.jacobians)
        # transport() caches a device copy of each J. The rescale replaces the
        # jacobians dict, so the cache must go.
        merged._device_cache.clear()
        merged.meta["rescaled_against"] = str(args.rescale_against)
        log(f"rescaled against {args.rescale_against}")
        for layer in sorted(scales):
            log(f"    L{layer:<3} x{scales[layer]:.1f}")
        (out_dir / "scale.json").write_text(json.dumps(
            {"reference": args.rescale_against,
             "scales": {str(k): v for k, v in scales.items()}}, indent=1))
    out = out_dir / "gemma4_12b_lens.pt"
    # fp32 again: the final lens is what every read uses, and fp16 costs
    # precision for no benefit here.
    merged.save(str(out), dtype=torch.float32)
    log(f"saved {out} ({merged!r})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Calibrate dim_batch on one prompt; no fitting.")
    ap.add_argument("--merge", nargs="+", metavar="GLOB",
                    help="Merge shard lens files into the final lens.")
    ap.add_argument("--rescale-against", metavar="LENS",
                    help=("Merge only. Give each layer the Frobenius norm of "
                          "the same layer in LENS. An R lens needs this "
                          "before any softmax-based score is read from it. "
                          "No ranking moves. See rlens_rules.match_scale."))
    ap.add_argument("--shard", default="0/1", help="i/n prompt-slice shard.")
    ap.add_argument("--n-prompts", type=int, default=128,
                    help="Prompts from the corpus (paper: saturates ~100).")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--source-layers", nargs="+", type=int,
                    default=DEFAULT_SOURCE_LAYERS,
                    help=("Layers to fit a transport for. Nearly free to add "
                          "— the cost is per prompt, not per layer. Pass the "
                          "dense band to break the post-global confound."))
    ap.add_argument("--dense", action="store_true",
                    help="Shorthand for --source-layers DENSE_SOURCE_LAYERS.")
    ap.add_argument("--rules", default="jacobian", choices=RULE_CHOICES,
                    help=("`lrp` fits an R-lens: the same estimator with "
                          "relevance rules in the backward pass. It forces "
                          "an eager load and its own output directory."))
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    if args.dense:
        args.source_layers = DENSE_SOURCE_LAYERS
    # An R fit must never overwrite a J shard, so it gets its own directory.
    if args.rules == "lrp" and args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = str(Path(DEFAULT_OUT_DIR) / "lrp")
        log(f"rules=lrp, so the output goes to {args.out_dir}")

    if args.smoke:
        return cmd_smoke(args)
    if args.merge:
        return cmd_merge(args)
    return cmd_fit(args)


if __name__ == "__main__":
    sys.exit(main())
