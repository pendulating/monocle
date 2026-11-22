"""Corpus-faithfulness ladder for the multimodal Jacobian lens.

Every rung fits the SAME estimator on a DIFFERENT corpus. One rung changes one
axis. Thus a difference between two rungs names its cause.

The ladder targets the single-image causal gate (`research/monocle/patch_map_causal.py`),
because that gate holds the result the ladder must move: a fitted transport
scores +0.10 where the local Jacobian scores +0.49. See
`research/docs/lens-corpus-faithfulness.md` for the preregistration.

| Rung | Images | Prompt | Target positions |
|---|---|---|---|
| R0 | 256 cyclomedia faces | "Describe this scene." | first patch onward |
| R1 | 256 cyclomedia faces | "Describe this scene." | the last position |
| R2 | 256 cyclomedia faces | `rating_2point`, per case | the last position |
| R3 | held-out analysis pool | `rating_2point`, per case | the last position |

R0 repeats the current `jlens/mm/` lens, so the ladder carries its own control.
R3 is the faithful lens.

The hold-out rule
-----------------
`single.load_single_rows` returns a deterministic prefix under seed 777. The
gate reads the first `ANALYSIS_RESERVE` rows. R3 fits on the rows AFTER them,
so the fit corpus and the analysis set stay disjoint by construction.

Warning: R2 and R3 make the lens case-specific. A lens fitted on
`subway_safety` has no shared basis with a lens fitted on another case.

Usage:
    python -m research.monocle.jlens_fit_ladder --rung R1 --shard 0/4 --n-images 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import jlens_fit_mm as mm  # noqa: E402

#: Rows of the analysis pool the gate may read. R3 never fits on these.
#: The stage-4 runs used 12 and 16; 32 leaves room for a wider re-run.
ANALYSIS_RESERVE = 32

DEFAULT_OUT_ROOT = str(REPO / "outputs/_monocle/jlens/ladder")

#: Which axis each rung moves, against the rung above it.
RUNGS: dict[str, dict[str, Any]] = {
    "R0": {"prompt": "describe", "target": "span",
           "images": "cyclomedia", "isolates": "baseline (repeats jlens/mm)"},
    "R1": {"prompt": "describe", "target": "last",
           "images": "cyclomedia", "isolates": "the target span"},
    "R2": {"prompt": "rating", "target": "last",
           "images": "cyclomedia", "isolates": "the prompt text"},
    "R3": {"prompt": "rating", "target": "last",
           "images": "heldout", "isolates": "the image content"},
}


def log(msg: str) -> None:
    print(f"[jlens-ladder] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def heldout_rows(case: str, n: int, *, reserve: int = ANALYSIS_RESERVE
                 ) -> list[dict]:
    """Analysis-pool images the gate never reads.

    Takes the deterministic prefix of length ``reserve + n`` and drops the
    first ``reserve``. Thus the fit corpus and the analysis set are disjoint.
    """
    from research.monocle import single as sg

    rows = sg.load_single_rows(reserve + n, case)
    if len(rows) <= reserve:
        raise RuntimeError(
            f"{case}: pool holds {len(rows)} rows, at or below the reserve "
            f"of {reserve}; no held-out image is available")
    out = rows[reserve:]
    log(f"held-out corpus: {len(out)} images, after a reserve of {reserve}")
    return out


def rung_images(rung: str, args: argparse.Namespace) -> list[dict]:
    """The image list for one rung."""
    if RUNGS[rung]["images"] == "heldout":
        return heldout_rows(args.case, args.n_images)
    return mm.load_images(args.images)[: args.n_images]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def target_mask_for(
    kind: str, seq_len: int, source_positions: torch.Tensor
) -> torch.Tensor:
    """The cotangent-injection mask.

    ``span``: every position from the first patch onward, minus the final one
    (the reference fitter's next-token exclusion). ``last``: the final
    position alone, which is the position that emits the label.
    """
    if kind == "last":
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[seq_len - 1] = True
        return mask
    if kind == "span":
        mask = torch.arange(seq_len) >= int(source_positions[0])
        mask[seq_len - 1] = False
        return mask
    raise ValueError(f"unknown target kind {kind!r}")


def build_rung_inputs(
    rung: str, proc: Any, tmpl: str, entry: dict, global_idx: int, *,
    case: str, image_token_id: int, device: str = "cuda:0",
    logits_to_keep: int | None = 1,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """``(inputs, source_positions, target_mask)`` for one image on one rung.

    Both prompt forms put the image first, then the text. That is the
    single-image form of ``images_then_labels``, which every gemma-4 read in
    this project must use.
    """
    from research.monocle import extract

    spec = RUNGS[rung]
    image = extract.open_image(entry["path"])

    if spec["prompt"] == "describe":
        context = mm.DESCRIBE_CONTEXTS[global_idx % len(mm.DESCRIBE_CONTEXTS)]
        inputs = extract.build_inputs(
            proc, tmpl, image, user_text=context, device=device)
    elif spec["prompt"] == "rating":
        from research.monocle import single as sg
        # The forced prefix is mandatory. Without it the last position emits
        # `{`, and a contrast read there is a disposition, not the answer.
        inputs = sg.build_single_inputs(
            proc, tmpl, image, case, device=device)
    else:
        raise ValueError(f"unknown prompt kind {spec['prompt']!r}")

    source_positions = extract.image_token_positions(inputs, image_token_id)
    seq_len = int(inputs["input_ids"].shape[1])
    target_mask = target_mask_for(spec["target"], seq_len, source_positions)

    inputs = dict(inputs)
    inputs["use_cache"] = False  # match the verified readout path
    if logits_to_keep is not None:
        # The LM head sits AFTER the last decoder layer, so it is off both the
        # recording path and the gradient path from target_layer to the source
        # layers. Slicing it to 1 position frees the [batch, seq, vocab] logits
        # and their softcap temporaries, which the retained graph would other-
        # wise hold for all 480 backward passes. `cmd_check_head` proves the
        # fitted Jacobian does not move.
        inputs["logits_to_keep"] = logits_to_keep
    return inputs, source_positions, target_mask


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def cmd_fit(args: argparse.Namespace) -> int:
    from jlens.lens import JacobianLens

    rung = args.rung
    spec = RUNGS[rung]
    if spec["prompt"] == "rating" and not args.case:
        raise SystemExit(f"rung {rung} needs --case")

    entries = rung_images(rung, args)
    shard_i, shard_n = mm.parse_shard(args.shard)
    indexed = list(enumerate(entries))[shard_i::shard_n]
    log(f"rung {rung} ({spec['isolates']}): shard {shard_i}/{shard_n}, "
        f"{len(indexed)} of {len(entries)} images, case={args.case}, "
        f"rules={args.rules}")

    out_dir = Path(args.out_dir) / rung
    if args.rules == "lrp":
        out_dir = out_dir / "lrp"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc, model, tmpl, lens_model, image_token_id = mm.load_lens_model(
        args.model_dir, eager=(args.rules == "lrp"))
    source_layers = args.layers or mm.DEFAULT_SOURCE_LAYERS
    d_model = lens_model.d_model

    ckpt = out_dir / f"fit_ckpt_shard{shard_i}of{shard_n}.pt"
    state_keys = {"rung": rung, "case": args.case, "rules": args.rules,
                  "source_layers": list(source_layers)}
    if ckpt.exists():
        # weights_only=True: the state holds tensors, ints, strings, and a
        # list of ints. Nothing in it needs the unsafe unpickler.
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        for key, want in state_keys.items():
            if state.get(key) != want:
                raise SystemExit(
                    f"checkpoint {ckpt} carries {key}={state.get(key)!r}, "
                    f"not {want!r}; move it aside or fix the flags")
        jacobian_sum = state["jacobian_sum"]
        n_done, next_idx = state["n_done"], state["next_idx"]
        log(f"  resume: {next_idx}/{len(indexed)} images processed")
    else:
        jacobian_sum = {
            l: torch.zeros(d_model, d_model, dtype=torch.float32)
            for l in source_layers}
        n_done = next_idx = 0

    def write_checkpoint() -> None:
        tmp = f"{ckpt}.tmp.{os.getpid()}"
        torch.save({**state_keys, "jacobian_sum": jacobian_sum,
                    "n_done": n_done, "next_idx": next_idx}, tmp)
        os.replace(tmp, ckpt)

    n_skipped = 0
    t0 = time.time()
    for local_idx in range(next_idx, len(indexed)):
        global_idx, entry = indexed[local_idx]
        image_id = (entry.get("image_id") or entry.get("recording_id")
                    or Path(entry["path"]).name)
        try:
            inputs, source_positions, target_mask = build_rung_inputs(
                rung, proc, tmpl, entry, global_idx, case=args.case,
                image_token_id=image_token_id, device=args.device)
        except Exception as exc:  # noqa: BLE001 -- unreadable image / build fail
            log(f"  skip {local_idx} ({image_id}): {exc}")
            n_skipped += 1
            next_idx = local_idx + 1
            continue

        with mm.rule_context(model, args.rules):
            per_image_J = mm.jacobian_for_inputs(
                model, lens_model, inputs, source_positions, target_mask,
                source_layers=source_layers, dim_batch=args.dim_batch)

        mean_rel_change = float("nan")
        if n_done > 0:
            mean_rel_change = max(
                float((per_image_J[l] - jacobian_sum[l] / n_done).norm()
                      / ((n_done + 1) * (jacobian_sum[l] / n_done).norm()))
                for l in source_layers)
        for layer in source_layers:
            jacobian_sum[layer] += per_image_J[layer]
        n_done += 1
        next_idx = local_idx + 1
        log(f"  {local_idx + 1}/{len(indexed)} {image_id} "
            f"n_target={int(target_mask.sum())} "
            f"{time.time() - t0:.0f}s  max_d_mean={mean_rel_change:.2e}")
        if (local_idx + 1) % args.ckpt_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        log("no image fitted")
        return 1
    jacobian_mean = {l: jacobian_sum[l] / n_done for l in source_layers}
    lens = JacobianLens(
        jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
    out = out_dir / f"shard{shard_i}of{shard_n}.pt"
    # fp32, not the fp16 default: a shard is an input to the merge, and the
    # fp16 error does not cancel in the weighted mean. The R0 check measured a
    # 4e-4 residual that came from the fp16 shard saves alone.
    lens.meta.update({
        "rung": rung,
        "case": args.case,
        "rules": args.rules,
        "target_kind": spec["target"],
        "prompt_kind": spec["prompt"],
        "images_source": spec["images"],
        "source_layers": [int(l) for l in source_layers],
        "dim_batch": int(args.dim_batch),
        "model_dir": str(args.model_dir),
        "analysis_reserve": ANALYSIS_RESERVE,
        "fit_date": mm.fit_date(),
    })
    lens.save(str(out), dtype=torch.float32)
    (out_dir / f"provenance_shard{shard_i}of{shard_n}.json").write_text(
        json.dumps({**state_keys, "n_fitted": n_done, "n_skipped": n_skipped,
                    "n_images_requested": args.n_images,
                    "images_source": spec["images"],
                    "analysis_reserve": ANALYSIS_RESERVE,
                    "target_kind": spec["target"],
                    "prompt_kind": spec["prompt"]}, indent=1))
    log(f"saved {out}; {n_done} fitted, {n_skipped} skipped")
    return 0


def cmd_check_head(args: argparse.Namespace) -> int:
    """Prove that the LM-head slice does not move the fitted Jacobian.

    `logits_to_keep=1` frees several GiB of retained graph. It is only safe if
    the head is truly off the gradient path. This fits ONE image both ways and
    compares. A non-zero difference means the flag is not inert, and every
    rung must then use the same setting.
    """
    proc, model, tmpl, lens_model, image_token_id = mm.load_lens_model(
        args.model_dir)
    entry = mm.load_images(args.images)[0]
    rung = args.rung or "R2"
    layers = args.layers or mm.DEFAULT_SOURCE_LAYERS
    log(f"rung {rung}, image {Path(entry['path']).name}, dim_batch "
        f"{args.dim_batch}")

    # --repeat runs the SAME setting twice. That is the control: if two
    # identical runs differ as much as the two settings do, the difference
    # measures nondeterministic reduction order, not the flag.
    settings = (1, 1) if args.repeat else (None, 1)
    out = {}
    for arm, keep in enumerate(settings):
        inputs, src, mask = build_rung_inputs(
            rung, proc, tmpl, entry, 0, case=args.case,
            image_token_id=image_token_id, device=args.device,
            logits_to_keep=keep)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out[arm] = mm.jacobian_for_inputs(
            model, lens_model, inputs, src, mask,
            source_layers=layers, dim_batch=args.dim_batch)
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"  arm {arm} logits_to_keep={keep}: {time.time() - t0:.0f}s, "
            f"peak {peak:.1f} GiB")

    worst = 0.0
    log(f"{'layer':>6} {'max abs diff':>14} {'rel diff':>12}")
    for layer in layers:
        a, b = out[0][layer], out[1][layer]
        diff = float((a - b).abs().max())
        rel = float((a - b).norm() / a.norm()) if a.norm() > 0 else 0.0
        worst = max(worst, diff)
        log(f"{layer:>6} {diff:>14.3e} {rel:>12.3e}")
    ok = worst == 0.0
    what = "REPEAT of one setting" if args.repeat else "logits_to_keep"
    if args.repeat:
        verdict = ("DETERMINISTIC at this config" if ok else
                   "NONDETERMINISTIC -- this is the noise floor, not the flag")
    else:
        verdict = ("INERT, safe to mix" if ok else
                   "differs -- compare against the --repeat control")
    log(f"[{what}] worst abs diff {worst:.3e} -- {verdict}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
def cmd_smoke(args: argparse.Namespace) -> int:
    """Build inputs for every rung on one image, and prove the read position.

    The one fault that would void the whole ladder is a wrong target position.
    R1 to R3 read the LAST position, which must be the position that emits the
    label. This decodes the tail of each prompt so a human can see it.
    """
    proc, model, tmpl, lens_model, image_token_id = mm.load_lens_model(
        args.model_dir)
    tok = proc.tokenizer

    entry = mm.load_images(args.images)[0]
    log(f"image: {entry['path']}")
    for rung in sorted(RUNGS):
        spec = RUNGS[rung]
        try:
            src_entry = (heldout_rows(args.case, 1)[0]
                         if spec["images"] == "heldout" else entry)
            inputs, src, mask = build_rung_inputs(
                rung, proc, tmpl, src_entry, 0, case=args.case,
                image_token_id=image_token_id, device=args.device)
        except Exception as exc:  # noqa: BLE001
            log(f"{rung}: BUILD FAILED -- {type(exc).__name__}: {exc}")
            return 1
        ids = inputs["input_ids"][0]
        seq_len = int(ids.shape[0])
        tail = tok.decode(ids[-12:])
        log(f"{rung} ({spec['isolates']}): seq_len={seq_len} "
            f"n_source={src.numel()} n_target={int(mask.sum())} "
            f"target={spec['target']}")
        log(f"     tail: {tail!r}")
        if spec["target"] == "last":
            nxt = int(mask.nonzero()[0])
            if nxt != seq_len - 1:
                log(f"     FAIL: target at {nxt}, not {seq_len - 1}")
                return 1
    log("smoke ok -- every rung builds, and every `last` mask hits the tail")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeat", action="store_true",
                    help="Control: run ONE setting twice, to size the noise.")
    ap.add_argument("--check-head", action="store_true",
                    help="Prove logits_to_keep=1 does not move the Jacobian.")
    ap.add_argument("--smoke", action="store_true",
                    help="Build every rung on one image; fit nothing.")
    ap.add_argument("--rung", choices=sorted(RUNGS))
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--shard", default="0/1", help="i/n image-slice shard.")
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--rules", default="jacobian", choices=mm.RULE_CHOICES)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--images", default=mm.DEFAULT_IMAGES)
    ap.add_argument("--model-dir", default=mm.MODEL_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt-every", type=int, default=4)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.check_head:
        return cmd_check_head(args)
    if args.smoke:
        return cmd_smoke(args)
    if not args.rung:
        raise SystemExit("--rung is required unless --smoke is given")
    return cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
