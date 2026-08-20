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

Usage (klara, .venv-nightly + LD_PRELOAD — see monocle/monocle.sub):
    sbatch monocle/monocle.sub monocle.jlens_fit --smoke
    sbatch monocle/monocle.sub monocle.jlens_fit --shard 0/4
    ... (shards 1/4..3/4) then:
    python -m monocle.jlens_fit --merge outputs/_monocle/jlens/shard*.pt
"""
from __future__ import annotations

import argparse
import glob as globlib
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO))

from monocle import MODEL_DIR_DEFAULT  # noqa: E402

DEFAULT_CORPUS = str(REPO / "outputs/_monocle/jlens/corpus_wikitext_1k.json")
DEFAULT_OUT_DIR = str(REPO / "outputs/_monocle/jlens")
# Sources must be < target (default final layer, 47). start_graph_at=min
# bounds the retained autograd graph, so nothing below layer 6 is retained.
DEFAULT_SOURCE_LAYERS = [6, 12, 18, 24, 30, 36, 42, 46]


def log(msg: str) -> None:
    print(f"[jlens-fit] {msg}", flush=True)


def load_lens_model(model_dir: str):
    """Load gemma-4 via the verified monocle recipe and wrap it for jlens.

    from_hf auto-detects the model.language_model layout (verified on the
    meta device 2026-07-20: 48 layers, d_model 3840, softcap 30.0)."""
    import jlens

    from monocle import extract

    proc, model, _tmpl = extract.load_model(model_dir)
    return jlens.from_hf(model, proc.tokenizer)


def parse_shard(spec: str) -> tuple[int, int]:
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise ValueError(f"bad shard spec {spec!r}")
    return i, n


def cmd_smoke(args: argparse.Namespace) -> int:
    """Calibrate dim_batch: time one prompt at each setting, report peak mem."""
    from jlens.fitting import jacobian_for_prompt

    prompts = json.load(open(args.corpus))
    m = load_lens_model(args.model_dir)
    for dim_batch in (8, 16, 32):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            _J, seq_len, n_valid = jacobian_for_prompt(
                m, prompts[0], DEFAULT_SOURCE_LAYERS, dim_batch=dim_batch)
        except torch.cuda.OutOfMemoryError:
            log(f"dim_batch={dim_batch}: OOM")
            torch.cuda.empty_cache()
            continue
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"dim_batch={dim_batch}: {dt:.0f}s/prompt "
            f"(seq={seq_len}, valid={n_valid}), peak {peak:.1f} GiB, "
            f"est. {dt * 250 / 3600:.1f} h per 250-prompt shard")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    import jlens

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    prompts = json.load(open(args.corpus))[: args.n_prompts]
    shard_i, shard_n = parse_shard(args.shard)
    shard_prompts = prompts[shard_i::shard_n]
    log(f"shard {shard_i}/{shard_n}: {len(shard_prompts)} prompts, "
        f"dim_batch={args.dim_batch}, sources={DEFAULT_SOURCE_LAYERS}")

    m = load_lens_model(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"fit_ckpt_shard{shard_i}of{shard_n}.pt"
    lens = jlens.fit(
        m, shard_prompts,
        source_layers=DEFAULT_SOURCE_LAYERS,
        dim_batch=args.dim_batch,
        checkpoint_path=str(ckpt),
        checkpoint_every=5,
    )
    out = out_dir / f"shard{shard_i}of{shard_n}.pt"
    lens.save(str(out))
    log(f"saved {out} ({lens!r})")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from jlens import JacobianLens

    paths = sorted(p for pat in args.merge for p in globlib.glob(pat))
    lenses = [JacobianLens.load(p) for p in paths]
    log(f"merging {len(paths)} shards: {[Path(p).name for p in paths]}")
    merged = JacobianLens.merge(lenses)
    out = Path(args.out_dir) / "gemma4_12b_lens.pt"
    merged.save(str(out))
    log(f"saved {out} ({merged!r})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Calibrate dim_batch on one prompt; no fitting.")
    ap.add_argument("--merge", nargs="+", metavar="GLOB",
                    help="Merge shard lens files into the final lens.")
    ap.add_argument("--shard", default="0/1", help="i/n prompt-slice shard.")
    ap.add_argument("--n-prompts", type=int, default=128,
                    help="Prompts from the corpus (paper: saturates ~100).")
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-dir", default=MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if args.smoke:
        return cmd_smoke(args)
    if args.merge:
        return cmd_merge(args)
    return cmd_fit(args)


if __name__ == "__main__":
    sys.exit(main())
