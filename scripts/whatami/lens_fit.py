#!/usr/bin/env python3
"""Fit the Jacobian lenses of one whatami model: a text lens and an image lens.

| Lens | Corpus | Sources | Target |
|---|---|---|---|
| `text` | the first 128 WikiText prompts (corpus_wikitext_1k.json), raw text, 128 tokens max | every position after the first 16, minus the last | the same positions |
| `mm` | the first 128 images of mm/fit_images.json, a describe prompt in rotation | the image patches | every position from the first image token, minus the last |
| `mm_textafter` | the same images | the text positions after the image | the same span as `mm` |

These are the corpora and estimands of the gemma-4-12B lenses `wikitext`,
`mm` and `mm_textafter` (vlm-narratives-docs/lens-registry.md). None of the
fit images is in the whatami sample (no recording in common).

The image corpus gives `mm` and `mm_textafter` from the SAME backward passes:
the passes give the gradient at every position, and the two lenses only
average it over different source positions. mllama (Llama 3.2 Vision) has
no image patches in its decoder sequence, thus it gets `mm_textafter` only.

The mllama text lens leaves out the cross-attention layers: with no image,
HF skips them, and their hooks never fire.

Each lens is fitted at every block below the last (target: the last block).
A shard writes the merge format of monocle.jlens_merge_mm: a checkpoint with
the SUM (fit_ckpt_shard{i}of{n}.pt) while it runs, and a lens with the MEAN
(shard{i}of{n}.pt) at the end.

Usage:
    python scripts/whatami/lens_fit.py --model gemma-4-12b --corpus image --smoke
    python scripts/whatami/lens_fit.py --model gemma-4-12b --corpus image --shard 0/2
    python -m monocle.jlens_merge_mm --in-dir outputs/_monocle/jlens/whatami/gemma-4-12b/mm \
        --out outputs/_monocle/jlens/whatami/gemma-4-12b/mm/lens.pt
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

import lens_common as lc
from monocle.jlens_fit_mm import DESCRIBE_CONTEXTS, _replicate_inputs, _resolve_layers


def jacobians_multi_source(model, lens_model, inputs, sources: dict[str, torch.Tensor],
                           target_mask, *, source_layers, target_layer, dim_batch):
    """monocle.jlens_fit_mm.jacobian_for_inputs, with several source sets.

    The mechanics are the same: one forward on the input replicated dim_batch
    times, a retained graph, and ceil(d_model / dim_batch) backward passes with
    one-hot cotangents at the target positions. Each source set gets its own
    mean over its positions. Returns {name: {layer: [d, d] fp32 CPU}}.
    """
    from jlens.hooks import ActivationRecorder

    d_model = lens_model.d_model
    out = {name: {l: torch.zeros(d_model, d_model) for l in source_layers} for name in sources}
    n_passes = math.ceil(d_model / dim_batch)
    with ActivationRecorder(lens_model.layers, at=[*source_layers, target_layer],
                            start_graph_at=min(source_layers)) as rec, torch.enable_grad():
        try:
            model(**_replicate_inputs(inputs, dim_batch, use_repeat=False))
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError):
            rec.activations.clear()
            model(**_replicate_inputs(inputs, dim_batch, use_repeat=True))
        target_act = rec.activations[target_layer]
        src_acts = [rec.activations[l] for l in source_layers]
        dev = target_act.device
        tpos = target_mask.nonzero(as_tuple=True)[0].to(dev)
        b = torch.arange(dim_batch, device=dev)
        spos = {k: v.to(dev) for k, v in sources.items()}
        cot = torch.zeros_like(target_act)
        for i, start in enumerate(range(0, d_model, dim_batch)):
            n = min(dim_batch, d_model - start)
            cot.zero_()
            cot[b[:n, None], tpos[None, :], start + b[:n, None]] = 1.0
            grads = torch.autograd.grad(target_act, src_acts, grad_outputs=cot,
                                        retain_graph=i < n_passes - 1)
            for l, g in zip(source_layers, grads, strict=True):
                for name, p in spos.items():
                    out[name][l][start:start + n] = g[:n, p, :].float().mean(1).cpu()
            del grads
    return out


def image_item(L: lc.Loaded, entry: dict, global_idx: int):
    """Inputs, source sets and target mask for one fit image."""
    context = DESCRIBE_CONTEXTS[global_idx % len(DESCRIBE_CONTEXTS)]
    inputs = lc.build_prompt(L, entry["path"], context)
    img = lc.image_positions(L, inputs)
    seq = int(inputs["input_ids"].shape[1])
    target = torch.arange(seq) >= int(img[0])
    target[seq - 1] = False  # the last position has no next-token target
    sources = {"mm_textafter": torch.arange(int(img[-1]) + 1, seq - 1)}
    if L.spec.family != "mllama":
        sources["mm"] = img
    inputs["use_cache"] = False
    return inputs, sources, target, seq


def text_item(L: lc.Loaded, prompt: str):
    """Raw text as jlens.fitting.jacobian_for_prompt builds it."""
    from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask

    ids = L.lens_model.encode(prompt, max_length=128)
    seq = int(ids.shape[1])
    mask = valid_position_mask(seq, skip_first=SKIP_FIRST_N_POSITIONS)
    return ids, {"text": mask.nonzero(as_tuple=True)[0]}, mask, seq


def source_layers_for(L: lc.Loaded, corpus: str) -> tuple[list[int], int]:
    n = L.lens_model.n_layers
    want = list(range(n - 1))
    if corpus == "text" and L.spec.family == "mllama":
        xattn = set(L.model.config.text_config.cross_attention_layers)
        want = [l for l in want if l not in xattn]
    return _resolve_layers(want, None, n)


class TextForward:
    """model(**inputs) for a raw-text item: jlens runs the bare text module."""

    def __init__(self, lens_model):
        self.lens_model = lens_model

    def __call__(self, input_ids, **_):
        return self.lens_model.forward(input_ids)


def run_item(L, corpus, item, dim_batch, source_layers, target_layer):
    if corpus == "text":
        ids, sources, mask, _ = item
        return jacobians_multi_source(
            TextForward(L.lens_model), L.lens_model, {"input_ids": ids}, sources, mask,
            source_layers=source_layers, target_layer=target_layer, dim_batch=dim_batch)
    inputs, sources, target, _ = item
    return jacobians_multi_source(
        L.model, L.lens_model, inputs, sources, target,
        source_layers=source_layers, target_layer=target_layer, dim_batch=dim_batch)


def corpus_items(corpus: str, n: int) -> list:
    if corpus == "text":
        return lc.load_json(lc.TEXT_CORPUS)[:n]
    return lc.load_json(lc.IMAGE_CORPUS)[:n]


def build_item(L, corpus, global_idx, entry):
    return text_item(L, entry) if corpus == "text" else image_item(L, entry, global_idx)


def rule_context(L: lc.Loaded, rules: str):
    """The LRP rules for an R-lens fit; nothing for a J-lens fit."""
    import contextlib

    if rules == "lrp":
        from monocle import rlens_rules as rr
        return rr.lrp_rules(L.model)
    return contextlib.nullcontext()


def lens_name(name: str, rules: str) -> str:
    return lc.RLENS_PREFIX + name if rules == "lrp" else name


def cmd_smoke(args) -> int:
    L = lc.load(args.model, force_bos=args.corpus == "text", eager=args.rules == "lrp")
    source_layers, target = source_layers_for(L, args.corpus)
    entry = corpus_items(args.corpus, 1)[0]
    item = build_item(L, args.corpus, 0, entry)
    lc.log(f"smoke {args.model}/{args.corpus}: seq_len={item[3]}, sources="
           f"{ {k: int(v.numel()) for k, v in item[1].items()} }, layers={len(source_layers)}, target={target}")
    for db in args.smoke_dims:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with rule_context(L, args.rules):
                J = run_item(L, args.corpus, item, db, source_layers, target)
        except torch.cuda.OutOfMemoryError:
            lc.log(f"dim_batch={db}: OOM")
            break
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        bad = sum(int(not torch.isfinite(m).all()) for d in J.values() for m in d.values())
        lc.log(f"dim_batch={db}: {dt:.1f} s/item, peak {peak:.1f} GiB, non-finite layers {bad}; "
               f"{args.n} items -> {dt * args.n / 3600:.2f} h on 1 GPU")
    return 0


def cmd_fit(args) -> int:
    from jlens import JacobianLens

    entries = corpus_items(args.corpus, args.n)
    si, sn = (int(x) for x in args.shard.split("/"))
    indexed = list(enumerate(entries))[si::sn]
    L = lc.load(args.model, force_bos=args.corpus == "text", eager=args.rules == "lrp")
    source_layers, target = source_layers_for(L, args.corpus)
    base = ["text"] if args.corpus == "text" else lc.LENSES[L.spec.family][1:]
    names = [lens_name(nm, args.rules) for nm in base]
    d = L.lens_model.d_model
    lc.log(f"fit {args.model}/{args.corpus} shard {si}/{sn}: {len(indexed)} items, lenses {names}, "
           f"{len(source_layers)} source layers, target {target}, dim_batch={args.dim_batch}")

    dirs = {nm: lc.lens_dir(args.model, nm) for nm in names}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    ckpts = {nm: dirs[nm] / f"fit_ckpt_shard{si}of{sn}.pt" for nm in names}
    meta_key = {"source_layers": source_layers, "target_layer": target,
                "model": args.model, "corpus": args.corpus, "rules": args.rules}

    sums, n_done, next_idx = {}, 0, 0
    if all(p.exists() for p in ckpts.values()):
        states = {nm: torch.load(p, map_location="cpu", weights_only=True) for nm, p in ckpts.items()}
        for nm, st in states.items():
            for k, v in meta_key.items():
                if st.get(k) != v:
                    raise ValueError(f"{ckpts[nm]} has {k}={st.get(k)!r}, not {v!r}; delete it to refit")
        if len({(st["n_done"], st["next_idx"]) for st in states.values()}) != 1:
            raise ValueError("the checkpoints of one shard disagree; delete them to refit")
        sums = {nm: st["jacobian_sum"] for nm, st in states.items()}
        n_done, next_idx = states[names[0]]["n_done"], states[names[0]]["next_idx"]
        lc.log(f"resume: {next_idx}/{len(indexed)} items done")
    else:
        sums = {nm: {l: torch.zeros(d, d) for l in source_layers} for nm in names}

    def write_ckpt():
        for nm in names:
            tmp = f"{ckpts[nm]}.tmp.{os.getpid()}"
            torch.save({"jacobian_sum": sums[nm], "n_done": n_done, "next_idx": next_idx,
                        "mode": "whatami", **meta_key}, tmp)
            os.replace(tmp, ckpts[nm])

    t_start = time.time()
    for local, (gi, entry) in enumerate(indexed):
        if local < next_idx:
            continue
        t0 = time.time()
        item = build_item(L, args.corpus, gi, entry)
        with rule_context(L, args.rules):
            J = run_item(L, args.corpus, item, args.dim_batch, source_layers, target)
        for bn, nm in zip(base, names, strict=True):
            for l in source_layers:
                sums[nm][l] += J[bn][l]
        n_done += 1
        next_idx = local + 1
        lc.log(f"  item {local + 1}/{len(indexed)} (global {gi}) seq_len={item[3]} {time.time() - t0:.0f}s")
        if next_idx % args.ckpt_every == 0:
            write_ckpt()
    write_ckpt()
    lc.log(f"shard done in {(time.time() - t_start) / 3600:.2f} h")

    for nm in names:
        lens = JacobianLens(jacobians={l: sums[nm][l] / n_done for l in source_layers},
                            n_prompts=n_done, d_model=d)
        lens.meta.update({
            "model": args.model, "model_dir": L.spec.path, "lens": nm, "corpus": args.corpus,
            "corpus_path": str(lc.TEXT_CORPUS if args.corpus == "text" else lc.IMAGE_CORPUS),
            "n_requested": args.n, "shard": args.shard, "dim_batch": args.dim_batch,
            "source_layers": source_layers, "target_layer": target,
            "estimator": {"text": "text_valid_positions", "mm": "mm_patch_source",
                          "mm_textafter": "mm_text_after_source"}[nm.removeprefix(lc.RLENS_PREFIX)],
            "rules": args.rules,
            "fit_date": time.strftime("%Y-%m-%d"),
        })
        out = dirs[nm] / f"shard{si}of{sn}.pt"
        lens.save(str(out), dtype=torch.float32)
        lc.log(f"saved {out}")
        if sn == 1:
            # One shard holds the whole mean; no merge is needed. A symlink,
            # not a copy: a 90B lens is about 27 GB.
            link = dirs[nm] / "lens.pt"
            link.unlink(missing_ok=True)
            link.symlink_to(out.name)
            lc.log(f"linked {link} -> {out.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(lc.MODELS))
    ap.add_argument("--corpus", required=True, choices=["text", "image"])
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dim-batch", type=int, default=32)
    ap.add_argument("--ckpt-every", type=int, default=4)
    ap.add_argument("--rules", default="jacobian", choices=["jacobian", "lrp"],
                    help="lrp fits an R-lens (monocle/rlens_rules.py); its lenses get the prefix r_")
    ap.add_argument("--smoke", action="store_true", help="time one item at each --smoke-dims; no fit")
    ap.add_argument("--smoke-dims", type=int, nargs="+", default=[16, 32, 64, 128])
    args = ap.parse_args()
    return cmd_smoke(args) if args.smoke else cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
